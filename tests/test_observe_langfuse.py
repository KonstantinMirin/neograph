"""TDD RED for nva4 — ``observe=`` opt-in Langfuse auto-attach (merge / no-clobber
/ env-gate no-op / flush-in-finalize / double-attach dedupe / sync+async twin).

These tests pin the AMENDED nva4 plan (design amendment on atom neograph-938w.29):

  * MERGE (never clobber) a Langfuse ``CallbackHandler`` into
    ``config["callbacks"]`` as a per-run config merge in the pre-engine brain
    (``_prepare``/``_aprepare``), building a FRESH config + FRESH callbacks list.
  * ENV-GATE on BOTH ``LANGFUSE_SECRET_KEY`` AND ``LANGFUSE_PUBLIC_KEY`` present
    (R-M2): absent -> clean no-op (no handler, no crash); SECRET-only -> no attach.
  * FLUSH on completion (R-plan step 5): run/arun via try/finally; stream/astream
    after generator exhaustion AND on early close (GeneratorExit).
  * DEDUPE (R-L1): observe=True + a user-supplied Langfuse handler already in
    config.callbacks does NOT add a second Langfuse handler.
  * TWIN: ``observe=`` threaded through run/stream (sync) AND arun/astream (async);
    the sync and async merge paths behave identically.

WHY THESE FAIL NOW: ``observe=`` does not exist yet on any verb, so every call
raises ``TypeError: ... unexpected keyword argument 'observe'``. That is the
expected TDD-red reason (a driver-feature parameter gap), same class of red as a
missing param. Implementation lands in nva4 (neograph-938w.23).

Three-surface parity is N/A: this is a driver/runtime feature (config merge in the
verbs), not an IR-shape change — same exemption documented in
``tests/test_async_observability.py``. langfuse is a REAL installed dep (>=3.0);
we monkeypatch env keys and patch ``langfuse.langchain.CallbackHandler`` /
``langfuse.get_client`` with recording fakes rather than opening a real trace,
so the tests stay offline/CI-safe and define the CONTRACT, not a live integration.
"""

from __future__ import annotations

import asyncio
import types as _types

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler

import neograph
from neograph import compile, construct_from_module, node
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText


# --------------------------------------------------------------------------- #
# Recording fakes (stand in for langfuse's CallbackHandler + client.flush)     #
# --------------------------------------------------------------------------- #
class _FakeLangfuseHandler(BaseCallbackHandler):
    """Stands in for ``langfuse.langchain.CallbackHandler``.

    A real ``BaseCallbackHandler`` subclass so it is a valid entry in
    ``config["callbacks"]`` (LangGraph will accept and drive it). Records every
    construction so tests can assert "exactly one Langfuse handler was attached"
    (dedupe) and "a Langfuse handler was added" (merge). Zero-arg constructible,
    matching langfuse v3/v4's ``CallbackHandler()``.
    """

    instances: list[_FakeLangfuseHandler] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        _FakeLangfuseHandler.instances.append(self)


class _FakeLangfuseClient:
    """Stands in for the process singleton returned by ``langfuse.get_client()``.

    Records ``flush()`` calls so flush-in-finalize can be asserted deterministically.
    """

    def __init__(self) -> None:
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1


@pytest.fixture
def fake_langfuse(monkeypatch):
    """Patch ``langfuse.langchain.CallbackHandler`` and ``langfuse.get_client``.

    The nva4 implementation imports these FUNCTION-LOCALLY inside the observe
    path (``from langfuse.langchain import CallbackHandler`` /
    ``from langfuse import get_client``), so patching the attributes on the
    ``langfuse`` package makes the observe helpers pick up our recording fakes
    at call time. Returns a namespace exposing the handler class and the client.
    """
    import langfuse
    import langfuse.langchain

    _FakeLangfuseHandler.instances = []
    client = _FakeLangfuseClient()
    monkeypatch.setattr(langfuse.langchain, "CallbackHandler", _FakeLangfuseHandler)
    monkeypatch.setattr(langfuse, "get_client", lambda *a, **k: client)

    ns = _types.SimpleNamespace(
        handler_cls=_FakeLangfuseHandler,
        client=client,
    )
    return ns


@pytest.fixture
def langfuse_env(monkeypatch):
    """Both Langfuse keys present — the ONLY state in which observe= attaches."""
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-public")


# --------------------------------------------------------------------------- #
# Pipeline harness (two scripted nodes — no LLM, offline-safe)                 #
# --------------------------------------------------------------------------- #
def _scripted_pipeline():
    """fetch (scripted) -> gen (scripted). Emits multiple ``values`` chunks so the
    stream early-close (GeneratorExit) flush can be exercised mid-stream."""
    mod = _types.ModuleType("test_observe_langfuse_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="scripted", outputs=Claims)
    def gen(fetch: RawText) -> Claims:
        return Claims(items=[fetch.text])

    mod.fetch = fetch
    mod.gen = gen
    return compile(
        construct_from_module(mod, name="observe-langfuse"),
        **build_test_compile_kwargs(),
    )


def _spy_invoke(graph, captured):
    """Capture the config that reaches the SYNC engine verb (post-merge)."""
    inner = graph.graph
    orig = inner.invoke

    def spy(engine_input, config=None, **kw):
        captured["config"] = config
        return orig(engine_input, config=config, **kw)

    inner.invoke = spy


def _spy_ainvoke(graph, captured):
    """Capture the config that reaches the ASYNC engine verb (post-merge)."""
    inner = graph.graph
    orig = inner.ainvoke

    async def spy(engine_input, config=None, **kw):
        captured["config"] = config
        return await orig(engine_input, config=config, **kw)

    inner.ainvoke = spy


def _langfuse_handlers(config, handler_cls):
    """The Langfuse handlers present in a captured config's callbacks list."""
    callbacks = (config or {}).get("callbacks") or []
    return [h for h in callbacks if isinstance(h, handler_cls)]


# =========================================================================== #
# 6. TWIN — observe= accepted by all four verbs                               #
# =========================================================================== #
class TestObserveParamAcceptedByAllVerbs:
    """``observe=`` is threaded through BOTH sync (run/stream) and async
    (arun/astream) verbs. FAILS NOW: none of the four accept the param."""

    def test_run_accepts_observe(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()
        result = neograph.run(graph, input={"node_id": "x"}, observe=True)
        assert result["gen"] == Claims(items=["hello"])

    def test_stream_accepts_observe(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()
        chunks = list(
            neograph.stream(graph, input={"node_id": "x"}, observe=True, stream_mode="values")
        )
        assert chunks  # stream ran to completion with observe=

    def test_arun_accepts_observe(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()

        async def _drive():
            return await neograph.arun(graph, input={"node_id": "x"}, observe=True)

        result = asyncio.run(_drive())
        assert result["gen"] == Claims(items=["hello"])

    def test_astream_accepts_observe(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()

        async def _drive():
            out = []
            async for chunk in neograph.astream(
                graph, input={"node_id": "x"}, observe=True, stream_mode="values"
            ):
                out.append(chunk)
            return out

        chunks = asyncio.run(_drive())
        assert chunks


# =========================================================================== #
# 2. MERGE semantics (never clobber a user-supplied handler)                  #
# =========================================================================== #
class TestObserveMergesWithoutClobber:
    """observe=True MERGES a Langfuse handler into config.callbacks alongside the
    user's own handler — the user handler is STILL present and a FRESH list is
    built (the caller's callbacks list is not mutated). Twin: sync == async."""

    def test_run_merges_langfuse_and_keeps_user_handler(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()
        user_handler = BaseCallbackHandler()
        user_callbacks = [user_handler]
        captured: dict[str, object] = {}
        _spy_invoke(graph, captured)

        neograph.run(
            graph,
            input={"node_id": "x"},
            config={"callbacks": user_callbacks, "configurable": {}},
            observe=True,
        )

        merged = captured["config"]["callbacks"]
        # User handler survived (no clobber) AND a Langfuse handler was added.
        assert user_handler in merged
        assert len(_langfuse_handlers(captured["config"], fake_langfuse.handler_cls)) == 1
        # No-mutation: the caller's original list is untouched (fresh list built).
        assert user_callbacks == [user_handler]
        assert merged is not user_callbacks

    def test_arun_merge_path_matches_sync(self, fake_langfuse, langfuse_env):
        """Async twin — identical merge/no-clobber semantics under arun."""
        graph = _scripted_pipeline()
        user_handler = BaseCallbackHandler()
        user_callbacks = [user_handler]
        captured: dict[str, object] = {}
        _spy_ainvoke(graph, captured)

        async def _drive():
            return await neograph.arun(
                graph,
                input={"node_id": "x"},
                config={"callbacks": user_callbacks, "configurable": {}},
                observe=True,
            )

        asyncio.run(_drive())

        merged = captured["config"]["callbacks"]
        assert user_handler in merged
        assert len(_langfuse_handlers(captured["config"], fake_langfuse.handler_cls)) == 1
        assert user_callbacks == [user_handler]
        assert merged is not user_callbacks


# =========================================================================== #
# 3. ENV-GATE on BOTH keys (clean no-op when absent / partial)                #
# =========================================================================== #
class TestObserveEnvGate:
    """observe= gates on BOTH LANGFUSE_SECRET_KEY AND LANGFUSE_PUBLIC_KEY. With
    neither (or only one) present it is a clean no-op — no handler attached, no
    crash — so offline/CI stays green."""

    def test_noop_when_neither_key_present(self, fake_langfuse, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        graph = _scripted_pipeline()
        captured: dict[str, object] = {}
        _spy_invoke(graph, captured)

        # Must NOT raise offline, and must NOT attach a Langfuse handler.
        result = neograph.run(graph, input={"node_id": "x"}, observe=True)
        assert result["gen"] == Claims(items=["hello"])
        assert _langfuse_handlers(captured["config"], fake_langfuse.handler_cls) == []
        assert _FakeLangfuseHandler.instances == []

    def test_does_not_attach_when_only_secret_key_present(self, fake_langfuse, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test-secret")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        graph = _scripted_pipeline()
        captured: dict[str, object] = {}
        _spy_invoke(graph, captured)

        result = neograph.run(graph, input={"node_id": "x"}, observe=True)
        assert result["gen"] == Claims(items=["hello"])
        # Gate requires BOTH keys — a half-configured handler must NOT attach.
        assert _langfuse_handlers(captured["config"], fake_langfuse.handler_cls) == []

    def test_arun_noop_when_keys_absent(self, fake_langfuse, monkeypatch):
        """Async twin of the clean-no-op boundary."""
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        graph = _scripted_pipeline()
        captured: dict[str, object] = {}
        _spy_ainvoke(graph, captured)

        async def _drive():
            return await neograph.arun(graph, input={"node_id": "x"}, observe=True)

        result = asyncio.run(_drive())
        assert result["gen"] == Claims(items=["hello"])
        assert _langfuse_handlers(captured["config"], fake_langfuse.handler_cls) == []


# =========================================================================== #
# 4. FLUSH in finalize (symmetric env-gate: flush iff attached)               #
# =========================================================================== #
class TestObserveFlush:
    """observe= flushes the Langfuse client on completion. run/arun flush in a
    finally after invoke; stream/astream flush AFTER generator exhaustion (and on
    early close). Flush fires iff the same env-gate that attached passed."""

    def test_run_flushes_on_completion(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()
        neograph.run(graph, input={"node_id": "x"}, observe=True)
        assert fake_langfuse.client.flush_calls == 1

    def test_arun_flushes_on_completion(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()

        async def _drive():
            return await neograph.arun(graph, input={"node_id": "x"}, observe=True)

        asyncio.run(_drive())
        assert fake_langfuse.client.flush_calls == 1

    def test_run_does_not_flush_when_keys_absent(self, fake_langfuse, monkeypatch):
        """Symmetric gate: no attach -> no flush (never flush a mis-configured client)."""
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        graph = _scripted_pipeline()
        neograph.run(graph, input={"node_id": "x"}, observe=True)
        assert fake_langfuse.client.flush_calls == 0

    def test_stream_flushes_after_exhaustion(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()
        gen = neograph.stream(
            graph, input={"node_id": "x"}, observe=True, stream_mode="values"
        )
        # Not flushed before the generator is driven / exhausted.
        assert fake_langfuse.client.flush_calls == 0
        list(gen)  # exhaust
        assert fake_langfuse.client.flush_calls == 1

    def test_astream_flushes_after_exhaustion(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()

        async def _drive():
            async for _ in neograph.astream(
                graph, input={"node_id": "x"}, observe=True, stream_mode="values"
            ):
                pass

        asyncio.run(_drive())
        assert fake_langfuse.client.flush_calls == 1

    def test_stream_flushes_on_early_close(self, fake_langfuse, langfuse_env):
        """Consumer closes the generator mid-stream (GeneratorExit) -> flush still
        fires from the verb's finally, so no trace batch is stranded."""
        graph = _scripted_pipeline()
        gen = neograph.stream(
            graph, input={"node_id": "x"}, observe=True, stream_mode="values"
        )
        next(gen)  # consume one chunk, then abandon
        gen.close()
        assert fake_langfuse.client.flush_calls == 1


# =========================================================================== #
# 5. DOUBLE-ATTACH dedupe (R-L1)                                              #
# =========================================================================== #
class TestObserveDedupe:
    """observe=True + a user-supplied Langfuse ``CallbackHandler`` already in
    config.callbacks must NOT add a second Langfuse handler (no duplicate traces).
    """

    def test_does_not_double_attach_existing_langfuse_handler(self, fake_langfuse, langfuse_env):
        graph = _scripted_pipeline()
        # User already wired their own Langfuse handler manually (escape hatch).
        user_langfuse = fake_langfuse.handler_cls()
        captured: dict[str, object] = {}
        _spy_invoke(graph, captured)

        neograph.run(
            graph,
            input={"node_id": "x"},
            config={"callbacks": [user_langfuse], "configurable": {}},
            observe=True,
        )

        # Exactly ONE Langfuse handler in the merged config — the user's own,
        # not a second observe-added one.
        lf = _langfuse_handlers(captured["config"], fake_langfuse.handler_cls)
        assert len(lf) == 1
        assert lf[0] is user_langfuse
