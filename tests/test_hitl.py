"""Integration + E2E tests for ``ask_human()`` — the typed mid-loop HITL sugar
(neograph-p8wz). TDD RED.

``ask_human`` is a pure Layer-2 passthrough over ``langgraph.types.interrupt()``:
it calls ``interrupt(payload.model_dump())`` and, when a ``resume_model`` is
given, returns ``resume_model.model_validate(returned)`` — otherwise it returns
the raw resume dict UNCHANGED (byte-identical to the raw keystone path in
``tests/test_agent_subgraph_keystone.py``).

These tests drive the behavior through the PUBLIC ``run()``/``arun()`` drivers,
invoking a real agent/act node whose tool bodies call ``ask_human`` inside the
inline ReAct cycle — exactly how an external consumer (agent-stark/piarch) would.
The pause/resume path, checkpointing, and interrupt mechanics are all the real
LangGraph machinery; the ONLY thing under test is the typed sugar contract.

RED TODAY: ``neograph.hitl`` does not exist, so the tool's local
``from neograph.hitl import ask_human`` raises ``ModuleNotFoundError`` when the
cycle invokes the tool, and the drive fails before an ``__interrupt__`` ever
surfaces. GREEN AFTER: the passthrough exists and the typed/raw contracts hold.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ValidationError

from neograph import Tool, arun, compile, construct_from_functions, node, run
from tests.fakes import (
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)

# ── Typed payloads ─────────────────────────────────────────────────────────


class AskPayload(BaseModel, frozen=True):
    """What the tool hands to ask_human — surfaces via __interrupt__.value."""

    question: str


class ResumeDecision(BaseModel, frozen=True):
    """The typed resume model ask_human validates the resumed dict into."""

    decision: str
    priority: int


class HResult(BaseModel, frozen=True):
    """The agent node's typed final output."""

    items: list[str]


# The exact raw payload the keystone's _AskTool passes to interrupt(). Reusing
# it verbatim lets the E2E assert a BYTE-IDENTICAL __interrupt__ surface vs the
# raw path (tests/test_agent_subgraph_keystone.py:142,197).
_KEYSTONE_QUESTION = "found person left the company — decide?"


# ── History-driven, stateless LLM fakes (mirror the keystone shape) ─────────


class _AskOnceFake:
    """First turn: call the ask tool (which interrupts mid-loop). After the tool
    result is present, emit the final parseable answer."""

    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _AskOnceFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _AskOnceFake:
        return self

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        if self._structured:
            assert self._model is not None
            return self._model(items=["done"])
        n_results = sum(isinstance(m, ToolMessage) for m in messages)
        if n_results == 0:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "ask_operator", "args": {"q": "decide"}, "id": "a1"}]
            return msg
        return AIMessage(content='{"items": ["done"]}')

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _AskOnceFake:
        clone = _AskOnceFake()
        clone._model = model
        clone._structured = True
        return clone


class _TwoToolFake:
    """record → ask_operator → final. Mirrors the keystone's history-driven fake
    so replay reproduces the same tool sequence a real model would."""

    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _TwoToolFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _TwoToolFake:
        return self

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        if self._structured:
            assert self._model is not None
            return self._model(items=["done"])
        n_results = sum(isinstance(m, ToolMessage) for m in messages)
        if n_results == 0:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "record", "args": {"fact": "x"}, "id": "r1"}]
            return msg
        if n_results == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "ask_operator", "args": {"q": "decide"}, "id": "a1"}]
            return msg
        return AIMessage(content='{"items": ["done"]}')

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _TwoToolFake:
        clone = _TwoToolFake()
        clone._model = model
        clone._structured = True
        return clone


# ── Tools that route their interrupt through ask_human ──────────────────────


class _TypedAskTool:
    """Calls ask_human with a resume_model — the resumed dict must arrive as a
    VALIDATED ResumeDecision instance."""

    name = "ask_operator"

    def __init__(self, received: list) -> None:
        self._received = received

    def invoke(self, args: dict) -> str:
        from neograph.hitl import ask_human

        answer = ask_human(AskPayload(question=_KEYSTONE_QUESTION), resume_model=ResumeDecision)
        self._received.append(answer)
        return f"operator decided: {answer}"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


class _RawAskTool:
    """Calls ask_human WITHOUT a resume_model — the raw resume dict must be
    returned unchanged (byte-identical to raw interrupt())."""

    name = "ask_operator"

    def __init__(self, received: list) -> None:
        self._received = received

    def invoke(self, args: dict) -> str:
        from neograph.hitl import ask_human

        answer = ask_human(AskPayload(question=_KEYSTONE_QUESTION))
        self._received.append(answer)
        return f"operator decided: {answer}"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


class _RecordTool:
    """Side-effecting tool: increments a shared counter each time it runs."""

    name = "record"

    def __init__(self, counter: list[int]) -> None:
        self._counter = counter

    def invoke(self, args: dict) -> str:
        self._counter[0] += 1
        return "recorded"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


def _drive(graph: Any, *, input: dict | None, resume: dict | None, config: dict, is_async: bool) -> Any:
    if is_async:
        return asyncio.run(arun(graph, input=input, resume=resume, config=config))
    return run(graph, input=input, resume=resume, config=config)


def _single_ask_graph(ask_tool_factory, fake_factory=lambda tier: _AskOnceFake()) -> Any:
    register_tool_factory("ask_operator", ask_tool_factory)

    @node(
        mode="agent",
        outputs=HResult,
        model="reason",
        prompt="test/explore",
        tools=[Tool(name="ask_operator", budget=3)],
    )
    def research() -> HResult: ...

    return compile(
        construct_from_functions("hitl", [research]),
        checkpointer=MemorySaver(),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(fake_factory),
    )


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION — typed ask_human contract
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
class TestAskHumanTypedContract:
    """ask_human pauses with the payload dict and, on resume, delivers a typed
    ResumeModel instance (or the raw dict when no resume_model)."""

    def test_resume_delivers_validated_resume_model_instance(self, is_async: bool) -> None:
        received: list = []
        graph = _single_ask_graph(lambda config, tc: _TypedAskTool(received))
        config = {"configurable": {"thread_id": f"hitl-typed-{is_async}"}}

        result1 = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)

        # Pause surfaces the payload DICT (payload.model_dump()), identical to raw.
        assert "__interrupt__" in result1, (
            "ask_human did not pause mid-loop — interrupt() did not surface as __interrupt__"
        )
        assert result1["__interrupt__"][0].value == {"question": _KEYSTONE_QUESTION}, (
            f"ask_human surfaced the wrong payload: {result1['__interrupt__'][0].value!r}"
        )

        # Resume with a raw dict that is VALID for ResumeDecision.
        result2 = _drive(
            graph,
            input=None,
            resume={"decision": "add-and-research", "priority": 5},
            config=config,
            is_async=is_async,
        )

        assert result2.get("research") == HResult(items=["done"]), (
            f"node did not finalize after resume: {result2!r}"
        )
        # The tool received a VALIDATED ResumeDecision INSTANCE — not the raw dict.
        assert len(received) == 1, f"tool did not receive exactly one resume value: {received!r}"
        answer = received[0]
        assert isinstance(answer, ResumeDecision), (
            f"ask_human(resume_model=ResumeDecision) did not deliver a validated instance; "
            f"got {type(answer).__name__}: {answer!r}"
        )
        assert answer.decision == "add-and-research" and answer.priority == 5, (
            f"validated resume model carried wrong field values: {answer!r}"
        )

    def test_malformed_resume_raises_validation_error_at_ask_human_boundary(self, is_async: bool) -> None:
        received: list = []
        graph = _single_ask_graph(lambda config, tc: _TypedAskTool(received))
        config = {"configurable": {"thread_id": f"hitl-bad-{is_async}"}}

        _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)

        # Resume with a dict that FAILS ResumeDecision validation (missing/mistyped
        # fields). ask_human must raise pydantic ValidationError AT its boundary.
        with pytest.raises(ValidationError):
            _drive(
                graph,
                input=None,
                resume={"decision": "ok", "priority": "not-an-int"},
                config=config,
                is_async=is_async,
            )

        # The error surfaced AT the ask_human call, not deep in tool code: the tool
        # body after ask_human (append + return) never ran, so nothing was recorded.
        assert received == [], (
            f"malformed resume was swallowed inside tool code instead of raising at the "
            f"ask_human boundary — tool recorded a value: {received!r}"
        )

    def test_resume_model_none_returns_raw_dict_unchanged(self, is_async: bool) -> None:
        received: list = []
        graph = _single_ask_graph(lambda config, tc: _RawAskTool(received))
        config = {"configurable": {"thread_id": f"hitl-raw-{is_async}"}}

        result1 = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)
        assert result1["__interrupt__"][0].value == {"question": _KEYSTONE_QUESTION}

        resume_dict = {"decision": "add", "priority": 9, "extra": "kept"}
        result2 = _drive(graph, input=None, resume=resume_dict, config=config, is_async=is_async)

        assert result2.get("research") == HResult(items=["done"])
        # resume_model=None → ask_human returns interrupt()'s raw dict UNCHANGED,
        # byte-identical to the raw keystone path (no wrapping, no coercion).
        assert received == [resume_dict], (
            f"ask_human(resume_model=None) did not return the raw resume dict unchanged: {received!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# E2E — exactly-once idempotency THROUGH ask_human
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
class TestAskHumanMidLoopIdempotency:
    """Mirror the keystone two-tool exactly-once pattern, but route the interrupt
    through ask_human instead of raw interrupt(). The pre-interrupt side-effecting
    tool must run EXACTLY ONCE across interrupt+resume, and the __interrupt__
    surface must be byte-identical to the raw path."""

    def test_pre_interrupt_tool_runs_exactly_once_through_ask_human(self, is_async: bool) -> None:
        counter = [0]
        received: list = []
        register_tool_factory("record", lambda config, tc: _RecordTool(counter))
        register_tool_factory("ask_operator", lambda config, tc: _RawAskTool(received))

        @node(
            mode="agent",
            outputs=HResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="record", budget=3), Tool(name="ask_operator", budget=3)],
        )
        def research() -> HResult: ...

        graph = compile(
            construct_from_functions("hitl-e2e", [research]),
            checkpointer=MemorySaver(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _TwoToolFake()),
        )
        config = {"configurable": {"thread_id": f"hitl-e2e-{is_async}"}}

        result1 = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)

        # __interrupt__ surface identical to the raw keystone path.
        assert "__interrupt__" in result1, "ask_human did not pause mid-loop"
        assert result1["__interrupt__"][0].value == {"question": _KEYSTONE_QUESTION}, (
            f"ask_human's __interrupt__ surface diverged from the raw path: "
            f"{result1['__interrupt__'][0].value!r}"
        )
        assert counter[0] == 1, f"record should have run once before the interrupt, ran {counter[0]}x"

        resume_dict = {"decision": "add-and-research"}
        result2 = _drive(graph, input=None, resume=resume_dict, config=config, is_async=is_async)

        assert result2.get("research") == HResult(items=["done"]), (
            f"node did not finalize after resume: {result2!r}"
        )
        assert received == [resume_dict], (
            f"ask_human did not deliver the raw resume value to the tool: {received!r}"
        )
        # KEYSTONE PARITY: the pre-interrupt side-effecting tool must NOT re-run on
        # resume — exactly-once must hold THROUGH ask_human, not just raw interrupt().
        assert counter[0] == 1, (
            f"MID-LOOP RE-EXECUTION through ask_human: pre-interrupt tool 'record' ran "
            f"{counter[0]}x across interrupt+resume (expected 1)"
        )
