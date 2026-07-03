"""Keystone acceptance test for neograph-m6d3 (agent-as-subgraph) — TDD RED.

This is the decision-agnostic behavioral test the migration must make pass. It
asserts BEHAVIOR through run()/arun() (the public drivers), NOT the compiled-graph
mechanism, so it is valid under either candidate compile strategy (inline-into-parent
or nested-subgraph). It does not touch _wiring.py/state.py or any production code.

THE FOOTGUN IT PINS. Today an agent/act node compiles to ONE LangGraph node whose
body runs a ``while True`` ReAct loop internally (``_tool_loop.py``). When a tool
raises ``interrupt()`` mid-loop, LangGraph checkpoints at the NODE boundary — the
whole loop is a single superstep — so on resume the node re-executes FROM THE TOP:
every tool called before the interrupt runs AGAIN. ``interrupt()`` itself is
position-cached (it won't re-pause), but the surrounding loop is not memoized.

The test drives a two-tool agent node:
  1. ``record`` — a side-effecting tool (increments a counter) called first.
  2. ``ask_operator`` — a tool that calls ``interrupt(payload)`` mid-loop.

First run pauses at the interrupt (payload surfaces via ``__interrupt__``); resume
supplies a structured answer; the node finalizes. The KEYSTONE assertion is that
``record`` ran EXACTLY ONCE across the interrupt+resume.

RED TODAY: the monolith re-executes the node top-to-bottom on resume, so ``record``
runs a SECOND time (counter == 2).

GREEN AFTER MIGRATION: with the ReAct turn compiled to supersteps and message
history / tool_log / budget as state channels, ``record`` ran in a checkpointed
tool-node superstep that is NOT re-executed on resume (counter stays 1) —
turn-boundary interrupt becomes correct BY CONSTRUCTION.

The LLM fake here is deliberately STATELESS (it decides its next tool call from the
message history it receives, like a real model), so replay reproduces the same
tool-call sequence — which is exactly what exposes the double-execution.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from pydantic import BaseModel

from neograph import Construct, Tool, arun, compile, construct_from_functions, node, run
from tests.fakes import (
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)


class KAnswer(BaseModel, frozen=True):
    """Typed resume payload the operator supplies on resume."""

    decision: str


class KResult(BaseModel, frozen=True):
    """The agent node's typed final output."""

    items: list[str]


# ── Stateless, history-driven LLM fake ────────────────────────────────────
#
# A real LLM is stateless: it emits the next tool call based on the message
# history it is handed. Modeling that (rather than an internal call counter) is
# essential — on interrupt+resume the node rebuilds ``messages`` from scratch, so
# a history-driven fake replays the SAME sequence a real model would, which is
# what surfaces the re-execution of ``record``.


class _HistoryDrivenFake:
    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _HistoryDrivenFake:
        return self  # return self so no state is lost across rebinds

    def abind_tools(self, *a: Any, **k: Any) -> _HistoryDrivenFake:
        return self

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        if self._structured:
            assert self._model is not None
            return self._model(items=["done"])

        n_results = sum(isinstance(m, ToolMessage) for m in messages)
        if n_results == 0:
            # First turn: call the side-effecting tool.
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "record", "args": {"fact": "x"}, "id": "r1"}]
            return msg
        if n_results == 1:
            # Second turn: ask the operator (this tool interrupts mid-loop).
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "ask_operator", "args": {"q": "decide"}, "id": "a1"}]
            return msg
        # Both tool results present: emit the final parseable answer.
        return AIMessage(content='{"items": ["done"]}')

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _HistoryDrivenFake:
        clone = _HistoryDrivenFake()
        clone._model = model
        clone._structured = True
        return clone


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


class _AskTool:
    """Tool that raises a typed human-in-the-loop interrupt mid-loop and returns
    the operator's structured answer once resumed."""

    name = "ask_operator"

    def __init__(self, received: list) -> None:
        self._received = received

    def invoke(self, args: dict) -> str:
        answer = interrupt({"question": "found person left the company — decide?"})
        self._received.append(answer)
        return f"operator decided: {answer}"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


def _build_graph(counter: list[int], received: list) -> Any:
    register_tool_factory("record", lambda config, tool_config: _RecordTool(counter))
    register_tool_factory("ask_operator", lambda config, tool_config: _AskTool(received))

    @node(
        mode="agent",
        outputs=KResult,
        model="reason",
        prompt="test/explore",
        tools=[Tool(name="record", budget=3), Tool(name="ask_operator", budget=3)],
    )
    def research() -> KResult: ...

    return compile(
        construct_from_functions("keystone", [research]),
        checkpointer=MemorySaver(),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: _HistoryDrivenFake()),
    )


def _drive(graph: Any, *, input: dict | None, resume: dict | None, config: dict, is_async: bool) -> Any:
    if is_async:
        return asyncio.run(arun(graph, input=input, resume=resume, config=config))
    return run(graph, input=input, resume=resume, config=config)


@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
class TestAgentSubgraphMidLoopInterruptIdempotency:
    """The migration acceptance test: a mid-loop interrupt pauses at a turn
    boundary and a pre-interrupt side-effecting tool runs exactly once across
    resume. RED today (the monolith re-runs the whole node on resume)."""

    def test_pre_interrupt_tool_runs_exactly_once_across_resume(self, is_async: bool) -> None:
        counter = [0]
        received: list = []
        graph = _build_graph(counter, received)
        config = {"configurable": {"thread_id": f"keystone-{is_async}"}}

        # First run — the ask_operator tool interrupts mid-loop.
        result1 = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)

        assert "__interrupt__" in result1, (
            "agent node did not pause at the mid-loop interrupt — the ask_operator "
            "tool's interrupt() did not surface as __interrupt__ through run()/arun()"
        )
        payload = result1["__interrupt__"][0].value
        assert payload == {"question": "found person left the company — decide?"}, (
            f"interrupt surfaced the wrong payload: {payload!r}"
        )
        # Pre-interrupt tool ran exactly once on the first pass.
        assert counter[0] == 1, f"record should have run once before the interrupt, ran {counter[0]}x"

        # Resume with the operator's structured answer (typed payload → dict,
        # mirroring the human_feedback resume contract).
        answer = KAnswer(decision="add-and-research")
        result2 = _drive(graph, input=None, resume=answer.model_dump(), config=config, is_async=is_async)

        # The node finalized with its typed output.
        assert result2.get("research") == KResult(items=["done"]), (
            f"node did not finalize after resume: {result2!r}"
        )
        # The ask_operator tool received the resume value. LangGraph delivers the
        # whole Command(resume=) dict as interrupt()'s return — the typed
        # payload/resume contract mirrors the existing human_feedback dict shape.
        assert received == [answer.model_dump()], (
            f"tool did not receive the resume value: {received!r}"
        )

        # KEYSTONE (RED today): the side-effecting tool must NOT re-execute on
        # resume. The monolith re-runs the whole node from the top, so record
        # runs a second time (counter == 2). Agent-as-subgraph checkpoints the
        # tool superstep, so it stays at 1.
        assert counter[0] == 1, (
            f"MID-LOOP RE-EXECUTION FOOTGUN: pre-interrupt tool 'record' ran "
            f"{counter[0]}x across the interrupt+resume (expected 1). The agent "
            f"node's ReAct loop re-executed from the top on resume instead of "
            f"resuming at the interrupted turn boundary."
        )

        # No-leak guard (binding condition 2): the agent node's ReAct internals —
        # message history, tool_log, budget/iteration counters — must NOT leak into
        # the returned state. Under agent-as-subgraph these become neo_-prefixed
        # channels (StateKeys.agent_*) that _strip_internals removes and the schema
        # fingerprint excludes, exactly like neo_oracle_/neo_eachoracle_. Passes
        # trivially today (the monolith holds them in locals); catches a plain-named
        # ({node}__messages) channel leak once the inline expander lands.
        for res in (result1, result2):
            assert not any(k.startswith("neo_") for k in res), (
                f"neo_ framework channel leaked into returned state: {sorted(res)}"
            )
            assert not any(("messages" in k or "tool_log" in k or "budget" in k) for k in res), (
                f"agent ReAct internal channel leaked into returned state (binding "
                f"condition 2 — name agent channels under the neo_ prefix): {sorted(res)}"
            )


# ── Budget-exhaust → forced-parse E2E (binding condition 1) ───────────────
#
# The architect flagged the exhaustion → forced-finalize edge as "where a bug
# will hide": when the LLM keeps requesting tools past the iteration budget, the
# cycle must inject a "final answer now" nudge, take ONE unbound turn, and route
# to parse — NOT loop forever. A stateless history-driven fake models a model
# that keeps calling the tool until it sees the limit nudge, then answers.


class _GreedyFake:
    """Always requests the loop tool until it sees a 'limit reached' nudge, then
    emits a final parseable answer — models a model that would loop forever if
    the iteration guard did not force a finalize."""

    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _GreedyFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _GreedyFake:
        return self

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        if self._structured:
            assert self._model is not None
            return self._model(items=["forced"])
        hit_limit = any(
            isinstance(m, ToolMessage) and "limit reached" in str(m.content).lower()
            for m in messages
        )
        if hit_limit:
            return AIMessage(content='{"items": ["forced"]}')
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": "loop_tool", "args": {}, "id": "l1"}]
        return msg

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _GreedyFake:
        clone = _GreedyFake()
        clone._model = model
        clone._structured = True
        return clone


class _CountingTool:
    name = "loop_tool"

    def __init__(self, counter: list[int]) -> None:
        self._counter = counter

    def invoke(self, args: dict) -> str:
        self._counter[0] += 1
        return "looped"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
def test_iteration_budget_forces_finalize_instead_of_looping_forever(is_async: bool) -> None:
    counter = [0]
    register_tool_factory("loop_tool", lambda config, tool_config: _CountingTool(counter))

    @node(
        mode="agent",
        outputs=KResult,
        model="reason",
        prompt="test/explore",
        tools=[Tool(name="loop_tool", budget=0)],  # unlimited per-tool budget
        llm_config={"max_iterations": 3},           # the iteration guard must bound it
    )
    def greedy() -> KResult: ...

    graph = compile(
        construct_from_functions("exhaust", [greedy]),
        checkpointer=MemorySaver(),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: _GreedyFake()),
    )
    config = {"configurable": {"thread_id": f"exhaust-{is_async}"}}

    result = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)

    # The node finalized (forced parse) rather than looping forever.
    assert result.get("greedy") == KResult(items=["forced"]), (
        f"greedy agent node did not force-finalize on iteration-budget exhaustion: {result!r}"
    )
    # The loop tool ran a bounded number of times (the iteration guard fired) —
    # not zero (it did run) and not unbounded.
    assert 0 < counter[0] <= 3, (
        f"loop_tool ran {counter[0]}x — the iteration guard did not bound the "
        f"forced-finalize edge (expected 1..3)"
    )


class _Port(BaseModel, frozen=True):
    node_id: str


@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
def test_forced_finalize_reachable_for_nested_agent_at_default_recursion_limit(is_async: bool) -> None:
    """A NESTED agent (inside a sub-construct) must reach its graceful
    budget-exhaust → forced-final edge at DEFAULT config, not crash on LangGraph's
    recursion_limit.

    Why nested: the inline cycle spends ~2 supersteps per ReAct turn (agent +
    tools), so a loop at the default max_iterations=20 costs ~42 supersteps.
    LangGraph applies its default recursion_limit=25 to SUB-CONSTRUCT invokes
    (example 13's shipped shape), so a nested agent trips it at ~turn 12 — BEFORE
    the forced-final can fire. The runner raises recursion_limit to a floor
    covering every agent node's worst case, and that raised limit propagates into
    the sub-construct invoke, so forced-final becomes reachable and the node
    produces typed output instead of raising GraphRecursionError.

    (A flat top-level agent does NOT need this — LangGraph's top-level invoke
    default is higher — but a nested one does; the fix covers both.)

    This test FAILS without the runner floor (GraphRecursionError at ~turn 12)
    and passes with it — it exercises the edge end-to-end, not the config value.
    """
    counter = [0]
    register_tool_factory("loop_tool", lambda config, tool_config: _CountingTool(counter))

    @node(
        mode="agent",
        outputs=KResult,
        model="reason",
        prompt="test/explore",
        tools=[Tool(name="loop_tool", budget=0)],  # unlimited per-tool budget
        # No max_iterations override → the default (20).
    )
    def inner() -> KResult: ...

    sub = construct_from_functions("subc", [inner], input=_Port, output=KResult)
    graph = compile(
        Construct("outer", nodes=[sub]),
        checkpointer=MemorySaver(),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: _GreedyFake()),
    )
    # DEFAULT config — no explicit recursion_limit. The sub-construct invoke would
    # otherwise default to 25 and crash before forced-final.
    config = {"configurable": {"thread_id": f"nested-exhaust-{is_async}"}}

    result = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)

    assert result.get("subc") == KResult(items=["forced"]), (
        f"nested agent at default max_iterations did not reach the forced-final "
        f"edge (recursion_limit floor missing / not propagated to the child): {result!r}"
    )
    # Bounded by max_iterations=20 — the loop terminated, not ran away.
    assert 0 < counter[0] <= 20, (
        f"loop_tool ran {counter[0]}x — expected the default max_iterations=20 to bound it"
    )
