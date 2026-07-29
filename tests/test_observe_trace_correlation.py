"""TDD RED for neograph-s65y2 — correlating the structlog ``run_id`` with the
Langfuse trace id.

With ``observe=`` on, neograph produced TWO independent 32-hex identities for one
execution and never related them: the structlog ``run_id`` (``uuid4().hex``, minted
by ``_mint_run_id``) and the Langfuse trace id (minted separately inside the
handler). Both look interchangeable and are not — a ``run_id`` taken from a real
eval log 404s against ``GET /api/public/traces/{id}``. The logs hold the mechanics
(loops, tool_calls, tokens, durations, ERRORS) and the traces hold the content
(rendered prompt, reasoning, tool payloads); answering "why did this run fail" or
"what is the error rate per experiment arm" needs both, joined per run.

The fix DERIVES the trace id from the run id via ``Langfuse.create_trace_id(seed=)``
(deterministic) and hands it to the handler as ``trace_context``. Deriving in this
direction — rather than the ticket's original "run_id from the trace id" — leaves
``_mint_run_id`` and everything keyed on ``run_id`` (``_run_cache``/``evict_run``)
untouched, and makes the join computable offline from a bare log line.

WHAT THESE PIN (the in-process sub-claims):
  (a) trace_id == create_trace_id(seed=run_id), and it reaches config;
  (b) the handler is constructed with that id as ``trace_context``;
  (c) every node_start/node_complete line carries ``trace_id`` beside ``run_id``;
  (d) the two honest-silence branches (no keys, user-supplied handler) emit NO
      trace_id — ours would not match their trace, and logging it would lie.

WHAT THEY DO NOT PIN: that Langfuse actually records the trace under that id.
That is an EXTERNAL effect requiring live LANGFUSE_* keys; see the closure
constraint on neograph-s65y2. Green here proves the sub-claims, not the acceptance.

Three-surface parity is N/A — a driver/runtime config concern, not an IR-shape
change (same exemption as tests/test_observe_langfuse.py).
"""

from __future__ import annotations

import types as _types

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from structlog.testing import capture_logs

import neograph
from neograph import compile, construct_from_module, node
from neograph._state_keys import StateKeys
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText


class _RecordingHandler(BaseCallbackHandler):
    """Records the kwargs each construction received, so the test can assert the
    trace_context actually handed to langfuse (not merely that one was computed)."""

    instances: list[_RecordingHandler] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.init_kwargs = kwargs
        _RecordingHandler.instances.append(self)


class _FakeClient:
    def __init__(self) -> None:
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


@pytest.fixture
def fake_langfuse(monkeypatch):
    import langfuse
    import langfuse.langchain

    _RecordingHandler.instances = []
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", _RecordingHandler)
    monkeypatch.setattr(langfuse, "get_client", lambda *a, **k: _FakeClient())
    return _types.SimpleNamespace(handler_cls=_RecordingHandler)


@pytest.fixture
def langfuse_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-public")


def _pipeline():
    mod = _types.ModuleType("test_observe_trace_correlation_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="scripted", outputs=Claims)
    def gen(fetch: RawText) -> Claims:
        return Claims(items=[fetch.text])

    mod.fetch = fetch
    mod.gen = gen
    return compile(
        construct_from_module(mod, name="observe-trace-corr"),
        **build_test_compile_kwargs(),
    )


def _spy_config(graph, captured):
    inner = graph.graph
    orig = inner.invoke

    def spy(engine_input, config=None, **kw):
        captured["config"] = config
        return orig(engine_input, config=config, **kw)

    inner.invoke = spy


def _expected_trace_id(run_id: str) -> str:
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=run_id)


class TestTraceIdDerivedFromRunId:
    def test_trace_id_is_the_seeded_derivation_of_run_id(self, fake_langfuse, langfuse_env):
        graph = _pipeline()
        captured: dict = {}
        _spy_config(graph, captured)

        neograph.run(graph, input={"node_id": "x"}, observe=True)

        configurable = captured["config"]["configurable"]
        run_id = configurable[StateKeys.RUN_ID]
        trace_id = configurable[StateKeys.TRACE_ID]

        assert trace_id == _expected_trace_id(run_id), (
            "the trace id must be create_trace_id(seed=run_id) so the join is "
            "computable from a log line alone"
        )
        assert trace_id != run_id, "they are related by derivation, not identical"

    def test_handler_receives_the_derived_id_as_trace_context(self, fake_langfuse, langfuse_env):
        graph = _pipeline()
        captured: dict = {}
        _spy_config(graph, captured)

        neograph.run(graph, input={"node_id": "x"}, observe=True)

        assert len(fake_langfuse.handler_cls.instances) == 1
        kwargs = fake_langfuse.handler_cls.instances[0].init_kwargs
        expected = captured["config"]["configurable"][StateKeys.TRACE_ID]
        assert kwargs.get("trace_context") == {"trace_id": expected}, (
            "the derived id must be HANDED to langfuse, not merely computed"
        )

    def test_derivation_is_stable_for_a_given_run_id(self):
        assert _expected_trace_id("abc") == _expected_trace_id("abc")
        assert _expected_trace_id("abc") != _expected_trace_id("xyz")


class TestTraceIdOnNodeLogLines:
    def test_every_node_lifecycle_line_carries_run_id_and_trace_id(self, fake_langfuse, langfuse_env):
        graph = _pipeline()
        captured: dict = {}
        _spy_config(graph, captured)

        with capture_logs() as logs:
            neograph.run(graph, input={"node_id": "x"}, observe=True)

        expected = captured["config"]["configurable"][StateKeys.TRACE_ID]
        lifecycle = [e for e in logs if e.get("event") in {"node_start", "node_complete"}]
        assert lifecycle, "expected node lifecycle log lines"
        for entry in lifecycle:
            assert entry.get("trace_id") == expected, (
                f"{entry.get('event')} for node {entry.get('node')!r} is missing the trace id — "
                "the logs and the traces stay unjoinable"
            )
            assert entry.get("run_id"), "run_id must remain on the line, not be replaced"


class TestHonestSilenceWhenWeDidNotAttach:
    def test_no_trace_id_when_langfuse_keys_are_absent(self, fake_langfuse, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        graph = _pipeline()
        captured: dict = {}
        _spy_config(graph, captured)

        with capture_logs() as logs:
            neograph.run(graph, input={"node_id": "x"}, observe=True)

        assert StateKeys.TRACE_ID not in captured["config"]["configurable"], (
            "no trace exists, so no trace id may be advertised"
        )
        assert all("trace_id" not in e for e in logs if e.get("event") == "node_start")

    def test_no_trace_id_when_user_supplied_their_own_handler(self, fake_langfuse, langfuse_env):
        """Dedupe branch: their handler owns a trace id we did not derive. Emitting
        ours would be a silent lie — strictly worse than the honest disconnect."""
        graph = _pipeline()
        captured: dict = {}
        _spy_config(graph, captured)
        user_handler = fake_langfuse.handler_cls()

        neograph.run(graph, input={"node_id": "x"}, observe=True, config={"callbacks": [user_handler]})

        assert StateKeys.TRACE_ID not in captured["config"]["configurable"]
