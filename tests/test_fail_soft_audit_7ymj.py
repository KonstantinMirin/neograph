"""neograph-7ymj / PAT-02 — fail-soft audit: the bounded queue of sites where a
fail-SOFT catch silently hands back a WRONG result under a fail-LOUD
durability/type-resolution banner.

TDD: each test asserts the *silent-wrong* path is now surfaced (raise / warn),
not swallowed. Written before the fix so they fail against the pre-audit code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from neograph.errors import ConfigurationError


@pytest.fixture
def structlog_to_stdlib():
    """Route structlog through stdlib logging so ``caplog`` captures events.

    Mirrors the pattern in tests/test_inline_prompts.py — structlog's default
    console renderer writes to stdout, which caplog does not see.
    """
    import structlog

    structlog.configure(
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    yield
    structlog.reset_defaults()


# ═══════════════════════════════════════════════════════════════════════════
# Site 1 — runner._has_existing_checkpoint / _ahas_existing_checkpoint
# ═══════════════════════════════════════════════════════════════════════════


class _CorruptSaver:
    """A checkpointer whose read RAISES (corrupt/unreadable) — NOT 'absent'.

    An absent checkpoint returns None from get_tuple; a raise is genuine
    corruption. The pre-audit code swallowed the raise and returned False,
    silently starting a fresh run that ignores durable state.
    """

    def get_tuple(self, config):
        raise TypeError("stored checkpoint payload is malformed")

    async def aget_tuple(self, config):
        raise TypeError("stored checkpoint payload is malformed")


def test_corrupt_checkpoint_read_raises_not_silent_fresh_run():
    from neograph.runner import _has_existing_checkpoint

    graph = SimpleNamespace(checkpointer=_CorruptSaver())
    with pytest.raises(ConfigurationError) as exc:
        _has_existing_checkpoint(graph, {"configurable": {"thread_id": "t1"}})
    # The probe error must be chained so the corruption is traceable.
    assert isinstance(exc.value.__cause__, TypeError)


@pytest.mark.asyncio
async def test_corrupt_checkpoint_aread_raises_not_silent_fresh_run():
    from neograph.runner import _ahas_existing_checkpoint

    graph = SimpleNamespace(checkpointer=_CorruptSaver())
    with pytest.raises(ConfigurationError) as exc:
        await _ahas_existing_checkpoint(graph, {"configurable": {"thread_id": "t1"}})
    assert isinstance(exc.value.__cause__, TypeError)


# ═══════════════════════════════════════════════════════════════════════════
# Site 2 — runner._required_checkpointer_driver introspection failure
# ═══════════════════════════════════════════════════════════════════════════


def test_driver_introspection_failure_warns_that_guard_is_bypassed(caplog, structlog_to_stdlib):
    from neograph.runner import _required_checkpointer_driver

    class _OpaqueSaver:
        # A builtin as aget_tuple — inspect.getsource raises TypeError, so the
        # sync/async classifier cannot read the source. Pre-audit: silent None
        # (guard bypassed with no trace).
        aget_tuple = len

    with caplog.at_level("WARNING"):
        result = _required_checkpointer_driver(_OpaqueSaver())
    assert result is None  # still unknown — third-party savers are legitimate
    joined = " ".join(r.message for r in caplog.records) + " ".join(
        str(getattr(r, "event", "")) for r in caplog.records
    )
    assert "introspection" in joined.lower() or "bypass" in joined.lower(), (
        f"expected a warning naming the bypassed guard; got {caplog.records}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Site 3 — shared _hints.resolve_hints (the six get_type_hints copies)
# ═══════════════════════════════════════════════════════════════════════════


def test_resolve_hints_isolates_one_bad_annotation_from_the_rest():
    """THE core PAT-02 fix: one unresolvable annotation must NOT discard the
    others (the pre-audit all-or-nothing get_type_hints swallow)."""
    from neograph._hints import resolve_hints

    def f(good: int, bad: TotallyUndefinedType) -> str:  # noqa: F821 - intentional
        return ""

    hints = resolve_hints(f, owner="f")
    # The resolvable siblings survive; only the offender is omitted.
    assert hints["good"] is int
    assert hints["return"] is str
    assert "bad" not in hints


def test_resolve_hints_logs_the_omitted_annotation(caplog, structlog_to_stdlib):
    from neograph._hints import resolve_hints

    def f(bad: TotallyUndefinedType) -> int:  # noqa: F821 - intentional bad ref
        return 1

    with caplog.at_level("DEBUG"):
        resolve_hints(f, owner="f")
    joined = " ".join(r.message for r in caplog.records)
    assert "partially_unresolved" in joined or "bad" in joined


def test_resolve_hints_resolves_valid_annotations():
    from neograph._hints import resolve_hints

    def f(x: int, y: str) -> bool:
        return True

    hints = resolve_hints(f, owner="f")
    assert hints["x"] is int
    assert hints["y"] is str
    assert hints["return"] is bool


# ═══════════════════════════════════════════════════════════════════════════
# Site 3 (decorators) — a resolvable sibling keeps its type when another param
# is an unresolvable non-DI annotation (must NOT raise — legitimate under
# `from __future__ import annotations` with locally-scoped markers).
# ═══════════════════════════════════════════════════════════════════════════


def test_node_with_non_di_annotated_marker_does_not_raise():
    """A non-DI Annotated marker whose type is unresolvable in the classifier's
    namespace must be skipped, not raise — a blanket fail-loud broke this."""
    from typing import Annotated

    from neograph import node

    class CustomMarker:  # locally-scoped, not importable from _di_classify's ns
        pass

    @node(outputs=int)
    def consumer(count: Annotated[int, CustomMarker()]) -> int:
        return count

    # Decoration succeeded; the resolvable inner type is kept for the input.
    assert consumer.inputs == {"count": int}


# ═══════════════════════════════════════════════════════════════════════════
# Site 5 — _tool_loop malformed args + empty-AIMessage recovery
# ═══════════════════════════════════════════════════════════════════════════


def test_malformed_tool_args_warns_not_silently_blanked(caplog, structlog_to_stdlib):
    from neograph._tool_loop import _coerce_string_args_result

    tc = {"name": "search", "args": "this is not json", "id": "call_1"}
    msg = SimpleNamespace(tool_calls=[tc])
    gen = SimpleNamespace(message=msg)
    raw = SimpleNamespace(generations=[gen])

    with caplog.at_level("WARNING"):
        _coerce_string_args_result(raw)

    joined = " ".join(r.message for r in caplog.records) + " ".join(
        str(getattr(r, "event", "")) for r in caplog.records
    )
    assert "search" in joined or "malformed" in joined.lower() or "args" in joined.lower(), (
        f"expected a warning naming the tool / malformed args; got {caplog.records}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Site 5a follow-up (neograph-arus) — unparseable tool args become a RETRIABLE
# ToolMessage error to the LLM instead of silently running the tool with {}.
# ═══════════════════════════════════════════════════════════════════════════


def test_unparseable_args_stamped_with_marker_not_blanked_to_empty(caplog, structlog_to_stdlib):
    """RED before arus: the coercion path blanked an unparseable args string to
    ``{}`` (tool ran with empty args). Now it stamps the marker + preserves the
    raw string so the tool-execution seam can surface a retriable error."""
    from neograph._tool_loop import UNPARSEABLE_ARGS_MARKER, _coerce_string_args_result

    tc = {"name": "search", "args": "this is not json", "id": "call_1"}
    msg = SimpleNamespace(tool_calls=[tc])
    gen = SimpleNamespace(message=msg)
    raw = SimpleNamespace(generations=[gen])

    with caplog.at_level("WARNING"):
        _coerce_string_args_result(raw)

    # Not blanked to {} — instead marker-stamped with the raw string preserved.
    assert tc["args"] != {}
    assert isinstance(tc["args"], dict)
    assert tc["args"].get(UNPARSEABLE_ARGS_MARKER) == "this is not json"


def test_unparseable_args_precheck_emits_retriable_error_without_running_tool():
    """RED before arus: an unparseable-args tool_call must short-circuit at the
    tool-execution seam with a ToolMessage ERROR (so the LLM can re-emit valid
    args), NOT invoke the tool. Budget must not be consumed."""
    from langchain_core.messages import ToolMessage

    from neograph._agent_cycle import _tool_call_precheck
    from neograph._tool_loop import UNPARSEABLE_ARGS_MARKER
    from neograph.tool import Tool, ToolBudgetTracker

    ran: list = []

    class _RecordingTool:
        def invoke(self, args, config=None):
            ran.append(args)
            return "ok"

    tracker = ToolBudgetTracker([Tool(name="search", budget=3)])
    tc = {"name": "search", "args": {UNPARSEABLE_ARGS_MARKER: "not json"}, "id": "call_9"}

    kind, payload = _tool_call_precheck(tc, tracker, {"search": _RecordingTool()})

    assert kind == "msg", "unparseable args must short-circuit, not run the tool"
    assert isinstance(payload, ToolMessage)
    assert payload.content.startswith("error: could not parse tool args")
    assert payload.tool_call_id == "call_9"
    assert ran == [], "the tool must NOT be invoked for unparseable args"
    # No budget consumed — precheck's 'msg' kind never records a call.
    assert tracker.can_call("search")


def test_empty_recovery_message_warns_when_coercion_yields_nothing(caplog, structlog_to_stdlib):
    from pydantic import BaseModel, ValidationError

    from neograph._tool_loop import _CoercingToolWrapper

    class _M(BaseModel):
        tool_calls: dict

    try:
        _M(tool_calls="notadict")
        raise AssertionError("expected ValidationError")
    except ValidationError as e:
        verr = e

    class _Bound:
        def invoke(self, messages, **kwargs):
            raise verr

        def _generate(self, messages, run_manager=None):
            return SimpleNamespace(generations=[])  # coercion yields None

    wrapper = _CoercingToolWrapper(_Bound())
    with caplog.at_level("WARNING"):
        result = wrapper.invoke([{"role": "user", "content": "hi"}])

    assert getattr(result, "content", None) == ""
    joined = " ".join(r.message for r in caplog.records) + " ".join(
        str(getattr(r, "event", "")) for r in caplog.records
    )
    assert "empty" in joined.lower() or "recover" in joined.lower() or "coercion" in joined.lower(), (
        f"expected a warning on the empty-AIMessage fallback; got {caplog.records}"
    )
