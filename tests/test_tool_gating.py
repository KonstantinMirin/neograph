"""Tool-gating HITL + payoff verification for neograph-m6d3.4 — TDD RED.

This file pins the PAYOFF of the agent-as-subgraph migration: because an
agent/act node's ReAct turns are now REAL parent supersteps
(``{node}__agent`` / ``{node}__tools`` / ``{node}__parse`` from
``_agent_cycle.py``), we can finally pause a run BEFORE the ``{node}__tools``
body executes — i.e. get human approval before a tool's side effects run
(the promise ``human-in-the-loop.mdx`` line 6 already makes).

Two RED item-(a) tests define the NEW ``gate_tools_when=`` @node kwarg (a Node
field mirroring ``skip_when``), the feature that does not exist yet:

  * ``TestToolGatingKeystone`` — the keystone: a side-effecting tool must NOT
    run until the human approves, then runs exactly once. RED today because
    ``gate_tools_when=`` is not a recognized kwarg/field (unknown-kwarg error),
    so no gate is inserted, the tool runs before any approval, and the run
    never pauses.
  * ``TestGateToolsWhenValidator`` — ``gate_tools_when`` on a NON-agent node
    (scripted/think) must raise ``ConstructError`` at assembly. RED today
    because no such validator rule exists.

Two verification tests ride on behavior that already exists post-migration and
are marked GUARD/CHARACTERIZATION (they may PASS today — they are NOT the red):

  * ``TestAgentTurnStreaming`` — item (b): ``astream`` surfaces distinct
    ``{node}__agent`` and ``{node}__tools`` superstep updates.
  * ``TestBudgetAcrossCheckpoint`` — item (c): per-turn budget does not reset
    across an interrupt+resume, forced-finalize still fires, and no
    ``neo_``-internal agent channel leaks into the returned state.

All tests drive the REAL ``compile``/``run``/``arun``/``astream`` surface
against a REAL MemorySaver — no mocking of the engine. LLM behavior comes from
history-driven fakes (a real model is stateless; it decides its next tool call
from the message history it is handed).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Node,
    Tool,
    arun,
    astream,
    compile,
    construct_from_functions,
    node,
    run,
)
from tests.fakes import (
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)


class KResult(BaseModel, frozen=True):
    """The agent node's typed final output."""

    items: list[str]


# ── One side-effecting tool + a two-turn history-driven fake ──────────────
#
# Turn 1 (no tool results yet): call the side-effecting ``record`` tool.
# Turn 2 (one tool result present): emit the final parseable answer.
# A real model is stateless, so this replays identically on resume.


class _TwoTurnFake:
    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _TwoTurnFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _TwoTurnFake:
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
        return AIMessage(content='{"items": ["done"]}')

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _TwoTurnFake:
        clone = _TwoTurnFake()
        clone._model = model
        clone._structured = True
        return clone


class _RecordTool:
    """Side-effecting tool: increments a shared counter each time it runs."""

    name = "record"

    def __init__(self, counter: list[int]) -> None:
        self._counter = counter

    def invoke(self, args: dict, config: Any = None, **kwargs: Any) -> str:
        self._counter[0] += 1
        return "recorded"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


def _gate_condition(state: Any) -> dict:
    """Always-fire gate condition (mirrors an Operator ``when`` predicate,
    receiving the full state). Returns a truthy payload → the interrupt value."""
    return {"pending_tool": "record", "reason": "approve tool call?"}


def _build_gated_graph(counter: list[int], *, surface: str) -> Any:
    """Build a single-agent-node pipeline that gates its tools on the
    always-fire condition. ``surface`` selects the API surface under test."""
    register_tool_factory("record", lambda config, tool_config: _RecordTool(counter))

    if surface == "node":

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="record", budget=3)],
            gate_tools_when=_gate_condition,
        )
        def research() -> KResult: ...

        construct = construct_from_functions("gating", [research])
    elif surface == "programmatic":
        construct = Construct(
            "gating",
            nodes=[
                Node(
                    name="research",
                    mode="agent",
                    outputs=KResult,
                    model="reason",
                    prompt="test/explore",
                    tools=[Tool(name="record", budget=3)],
                    gate_tools_when=_gate_condition,
                )
            ],
        )
    else:  # pragma: no cover - guard
        raise AssertionError(f"unknown surface {surface!r}")

    return compile(
        construct,
        checkpointer=MemorySaver(),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: _TwoTurnFake()),
    )


def _drive(graph: Any, *, input: dict | None, resume: Any, config: dict, is_async: bool) -> Any:
    if is_async:
        return asyncio.run(arun(graph, input=input, resume=resume, config=config))
    return run(graph, input=input, resume=resume, config=config)


# ── Item (a) PRIMARY RED — tool-gating keystone ───────────────────────────


@pytest.mark.parametrize("surface", ["node", "programmatic"])
@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
class TestToolGatingKeystone:
    """RED: ``gate_tools_when=`` pauses the agent BEFORE its ``{node}__tools``
    body runs, so a side-effecting tool does not execute until the human
    approves — then runs exactly once. Three-surface parity: @node +
    programmatic ``Node``."""

    def test_side_effecting_tool_gated_until_approval_then_runs_once(self, is_async: bool, surface: str) -> None:
        counter = [0]
        graph = _build_gated_graph(counter, surface=surface)
        config = {"configurable": {"thread_id": f"gate-{surface}-{is_async}"}}

        # First run — the gate must pause BEFORE the tool superstep executes.
        result1 = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)

        assert "__interrupt__" in result1, (
            "agent node did not pause at the tool gate — gate_tools_when did not "
            "insert a pre-tools interrupt, so the run finished (or ran the tool) "
            "without asking for approval"
        )
        payload = result1["__interrupt__"][0].value
        assert payload == {"pending_tool": "record", "reason": "approve tool call?"}, (
            f"tool gate surfaced the wrong payload: {payload!r}"
        )
        # THE KEYSTONE: the side-effecting tool MUST NOT have run pre-approval.
        assert counter[0] == 0, (
            f"TOOL-GATE VIOLATION: side-effecting tool 'record' ran {counter[0]}x "
            f"BEFORE the human approved — gate_tools_when must pause before the "
            f"{{node}}__tools body executes, not after"
        )

        # Resume with an approval — the gate lets the tool run.
        result2 = _drive(graph, input=None, resume={"approved": True}, config=config, is_async=is_async)

        # Tool ran exactly once after approval and the node finalized.
        assert counter[0] == 1, f"after approval the gated tool 'record' ran {counter[0]}x (expected exactly 1)"
        assert result2.get("research") == KResult(items=["done"]), (
            f"agent node did not finalize after gate approval: {result2!r}"
        )


# ── Item (a) PRIMARY RED — assembly-time validator rule ───────────────────


class TestGateToolsWhenValidator:
    """RED: ``gate_tools_when`` only makes sense where a ``{node}__tools`` node
    exists (agent/act mode). Setting it on a scripted/think node must raise
    ``ConstructError`` at assembly time. RED today — no such rule exists."""

    def test_gate_tools_when_on_scripted_node_raises_construct_error(self) -> None:
        with pytest.raises(ConstructError):

            @node(
                mode="scripted",
                outputs=KResult,
                gate_tools_when=_gate_condition,
            )
            def scripted_gate() -> KResult:
                return KResult(items=["x"])

            # Assembly of the construct is where the rule must fire, if not at
            # decoration time.
            construct_from_functions("bad-gate", [scripted_gate])

    def test_gate_tools_when_on_programmatic_scripted_node_raises_construct_error(self) -> None:
        with pytest.raises(ConstructError):
            Construct(
                "bad-gate-prog",
                nodes=[
                    Node(
                        name="scripted_gate",
                        mode="scripted",
                        outputs=KResult,
                        gate_tools_when=_gate_condition,
                    )
                ],
            )


# ── Item (b) VERIFICATION (GUARD/CHARACTERIZATION — may pass today) ───────


class TestAgentTurnStreaming:
    """GUARD/CHARACTERIZATION (item b, may pass today): because agent/tools/parse
    are real parent supersteps, ``astream`` with ``stream_mode="updates"``
    surfaces distinct ``{node}__agent`` and ``{node}__tools`` updates. Not the
    red — pins that agent turns are visible per-superstep post-migration."""

    async def test_astream_surfaces_distinct_agent_and_tools_supersteps(self) -> None:
        counter = [0]
        register_tool_factory("record", lambda config, tool_config: _RecordTool(counter))

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="record", budget=3)],
        )
        def research() -> KResult: ...

        graph = compile(
            construct_from_functions("streaming", [research]),
            checkpointer=MemorySaver(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _TwoTurnFake()),
        )
        config = {"configurable": {"thread_id": "stream-turns"}}

        seen_nodes: set[str] = set()
        async for chunk in astream(graph, input={"node_id": "REQ-1"}, config=config, stream_mode="updates"):
            if isinstance(chunk, dict):
                seen_nodes.update(chunk.keys())

        assert "research__agent" in seen_nodes, (
            f"agent turn superstep 'research__agent' not visible via astream updates; saw {sorted(seen_nodes)}"
        )
        assert "research__tools" in seen_nodes, (
            f"tools turn superstep 'research__tools' not visible via astream updates; saw {sorted(seen_nodes)}"
        )
        assert counter[0] == 1, f"record tool ran {counter[0]}x (expected 1)"


# ── Item (c) VERIFICATION (GUARD/CHARACTERIZATION — may pass today) ───────
#
# A greedy model that keeps requesting a loop tool until it sees the
# iteration-limit nudge, but first calls an interrupting ``ask`` tool (no side
# effect) so the run checkpoints mid-cycle. On resume the per-turn budget must
# NOT reset, so forced-finalize still fires honestly and the loop stays bounded.


class _GreedyGatedFake:
    def __init__(self) -> None:
        self._model: type[BaseModel] | None = None
        self._structured = False

    def bind_tools(self, tools: list) -> _GreedyGatedFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _GreedyGatedFake:
        return self

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        if self._structured:
            assert self._model is not None
            return self._model(items=["forced"])
        hit_limit = any(isinstance(m, ToolMessage) and "limit reached" in str(m.content).lower() for m in messages)
        if hit_limit:
            return AIMessage(content='{"items": ["forced"]}')
        n_results = sum(isinstance(m, ToolMessage) for m in messages)
        if n_results == 0:
            # First turn: pause for approval (no side effect in this tool).
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "ask", "args": {"q": "ok?"}, "id": "a1"}]
            return msg
        # Post-resume turns: greedily loop the counting tool.
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": "loop_tool", "args": {}, "id": f"l{n_results}"}]
        return msg

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _GreedyGatedFake:
        clone = _GreedyGatedFake()
        clone._model = model
        clone._structured = True
        return clone


class _AskTool:
    """Interrupts for approval; no side effect (safe to re-enter on resume)."""

    name = "ask"

    def invoke(self, args: dict, config: Any = None, **kwargs: Any) -> str:
        answer = interrupt({"question": "approve mid-cycle?"})
        return f"answered: {answer}"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


class _CountingTool:
    name = "loop_tool"

    def __init__(self, counter: list[int]) -> None:
        self._counter = counter

    def invoke(self, args: dict, config: Any = None, **kwargs: Any) -> str:
        self._counter[0] += 1
        return "looped"

    async def ainvoke(self, *a: Any, **k: Any) -> str:
        return self.invoke(*a, **k)


@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
class TestBudgetAcrossCheckpoint:
    """GUARD/CHARACTERIZATION (item c, may pass today): per-turn budget survives
    an interrupt+resume — forced-finalize still fires honestly and the loop stays
    bounded — and no neo_-internal agent channel leaks into the returned state
    (binding condition 2). Not the red."""

    def test_budget_honest_and_no_leak_across_resume(self, is_async: bool) -> None:
        counter = [0]
        register_tool_factory("ask", lambda config, tool_config: _AskTool())
        register_tool_factory("loop_tool", lambda config, tool_config: _CountingTool(counter))

        @node(
            mode="agent",
            outputs=KResult,
            model="reason",
            prompt="test/explore",
            tools=[Tool(name="ask", budget=3), Tool(name="loop_tool", budget=0)],
            llm_config={"max_iterations": 3},
        )
        def greedy() -> KResult: ...

        graph = compile(
            construct_from_functions("budget-ckpt", [greedy]),
            checkpointer=MemorySaver(),
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _GreedyGatedFake()),
        )
        config = {"configurable": {"thread_id": f"budget-ckpt-{is_async}"}}

        # Pause mid-cycle at the ask interrupt (checkpoint written).
        result1 = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)
        assert "__interrupt__" in result1, f"agent node did not pause at the mid-cycle ask interrupt: {result1!r}"

        # Resume — the per-turn budget must continue from where it paused.
        result2 = _drive(graph, input=None, resume={"approved": True}, config=config, is_async=is_async)

        # Forced-finalize still fires honestly post-resume.
        assert result2.get("greedy") == KResult(items=["forced"]), (
            f"agent did not force-finalize post-resume — budget likely reset across the checkpoint: {result2!r}"
        )
        # Budget did not reset: the loop stayed bounded by max_iterations=3
        # across the resume (a reset would re-arm a full iteration allowance).
        assert 0 < counter[0] <= 3, (
            f"loop_tool ran {counter[0]}x across the checkpoint — the per-turn "
            f"budget reset on resume instead of surviving it (expected 1..3)"
        )
        # No-leak guard (binding condition 2): neo_ agent-cycle channels must
        # not surface in the returned state.
        for res in (result1, result2):
            assert not any(k.startswith("neo_") for k in res), (
                f"neo_ framework channel leaked into returned state: {sorted(res)}"
            )
            assert not any(("messages" in k or "tool_log" in k or "budget" in k) for k in res), (
                f"agent ReAct internal channel leaked into returned state (binding condition 2): {sorted(res)}"
            )


# ── Item (a) deny keystone — RED for neograph-whq0 ────────────────────────
#
# The approve path (TestToolGatingKeystone) proves the gate PAUSES before the
# tool superstep. neograph-whq0: the resume DECISION is written to
# HUMAN_FEEDBACK (_wiring.py) but never read — an unconditional gate->tools edge
# runs the tool regardless. A deny therefore executes the tool anyway, which is
# worse than no gate (false sense of a safety control). The gate must honor the
# decision: deny must NOT run the pending tool and must feed a denial back to
# the agent so the loop continues to a final answer.


@pytest.mark.parametrize("surface", ["node", "programmatic"])
@pytest.mark.parametrize("is_async", [False, True], ids=["run", "arun"])
class TestToolGatingDenyKeystone:
    """RED (neograph-whq0): resuming a tool gate with a deny decision must NOT
    execute the pending tool, and must feed the denial back to the agent so it
    produces a final answer. Today the deny is silently ignored — the
    unconditional gate->tools edge runs the tool anyway."""

    def test_deny_does_not_run_tool_and_agent_finalizes(self, is_async: bool, surface: str) -> None:
        counter = [0]
        graph = _build_gated_graph(counter, surface=surface)
        config = {"configurable": {"thread_id": f"deny-{surface}-{is_async}"}}

        # First run — the gate pauses before the tool superstep.
        result1 = _drive(graph, input={"node_id": "REQ-1"}, resume=None, config=config, is_async=is_async)
        assert "__interrupt__" in result1, "agent node did not pause at the tool gate before deny"
        assert counter[0] == 0, "tool ran before the human decided"

        # Resume with a DENY — the gate must reject the pending tool call.
        result2 = _drive(graph, input=None, resume={"approved": False}, config=config, is_async=is_async)

        # THE DENY KEYSTONE: the side-effecting tool MUST NOT have run.
        assert counter[0] == 0, (
            f"DENY VIOLATION: gated tool 'record' ran {counter[0]}x after a DENY "
            f"— gate_tools_when ignored the deny decision and executed the tool "
            f"anyway (neograph-whq0)"
        )
        # The agent received the denial and still produced a final answer
        # (the loop continues rather than crashing or hanging).
        assert result2.get("research") == KResult(items=["done"]), (
            f"agent did not finalize after the tool call was denied: {result2!r}"
        )

        # M7: the denial must actually reach the agent as a ToolMessage answering
        # the pending tool_call (one per denied call) — 'agent sees why', pinned
        # directly rather than inferred from finalization.
        snapshot = graph.get_state(config)
        msgs = snapshot.values.get("neo_agent_messages_research", [])
        denial_tool_msgs = [
            m for m in msgs if isinstance(m, ToolMessage) and "denied by a human reviewer" in str(m.content)
        ]
        assert denial_tool_msgs, (
            f"no denial ToolMessage was fed back to the agent after deny; the LLM "
            f"could not see why the tool was rejected. messages={msgs!r}"
        )
        assert denial_tool_msgs[0].tool_call_id == "r1", (
            f"denial ToolMessage did not answer the pending tool_call id 'r1': {denial_tool_msgs[0]!r}"
        )
