"""Async ReAct turn: parallel tool calls run concurrently (neograph-dyy7).

When one ReAct turn emits multiple tool calls, ``atools_body`` (the async twin
of the ``{node}__tools`` superstep) must ``asyncio.gather`` the per-call
coroutines instead of awaiting them one-at-a-time. The sync twin stays
sequential (no threads) — this is an async-only change.

Three contracts are pinned here:

1. **Concurrency** (RED before the fix) — two tools in ONE turn that each block
   on a shared ``asyncio.Barrier(2)`` only release if BOTH are in flight at once.
   Sequential awaiting deadlocks (the first blocks forever waiting for a second
   party that never starts), so the run times out. Concurrent gather releases.
2. **Ordering** — a fast second call completing before a slow first call must NOT
   reorder the ToolInteraction / ToolMessage history: results are assembled in the
   ORIGINAL tool_call order (the message-history contract).
3. **Budget under concurrency** — two parallel calls to a ``budget=1`` tool: the
   budget is pre-reserved in tool_call order BEFORE gathering, so the second call
   short-circuits exactly as in the sync twin (it must not race the first's
   can_call check and both slip through).
"""

from __future__ import annotations

import asyncio

from neograph import Tool, ToolInteraction, compile, construct_from_functions, node, run
from tests.fakes import (
    ReActFake,
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)
from tests.schemas import Claims

_CFG = {"configurable": {}}
_INPUT = {"node_id": "t"}


class _RecordingTool:
    """Tool stand-in mirroring the ``.invoke``/``.ainvoke`` surface the agent
    cycle drives. Its async body runs ``async_body`` (a coroutine fn) so tests
    can inject sleeps / barriers, and it counts invocations."""

    def __init__(self, name: str, async_body, response: str):
        self.name = name
        self._async_body = async_body
        self.response = response
        self.calls = 0

    def invoke(self, args: dict, config=None) -> str:
        self.calls += 1
        return self.response

    async def ainvoke(self, args: dict, config=None) -> str:
        self.calls += 1
        await self._async_body()
        return self.response


def _agent_node(tools: list[Tool]):
    @node(
        mode="agent",
        outputs={"result": Claims, "tool_log": list[ToolInteraction]},
        model="reason",
        prompt="test/explore",
        tools=tools,
    )
    def explore() -> Claims: ...

    return explore


def _compile(tools: list[Tool], turn: list[dict]):
    fake = ReActFake(
        tool_calls=[turn, []],  # one tool-using turn, then the final structured turn
        final=lambda m: m(items=["done"]),
    )
    return compile(
        construct_from_functions("p", [_agent_node(tools)]),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: fake),
    )


class TestAsyncParallelToolCalls:
    def test_parallel_calls_run_concurrently(self):
        """RED before the fix: sequential awaiting deadlocks on the barrier and
        the run times out; concurrent gather lets both parties arrive."""
        barrier = asyncio.Barrier(2)

        async def wait_barrier():
            await barrier.wait()

        a = _RecordingTool("alpha", wait_barrier, "a-done")
        b = _RecordingTool("beta", wait_barrier, "b-done")
        register_tool_factory("alpha", lambda config, tool_config: a)
        register_tool_factory("beta", lambda config, tool_config: b)

        graph = _compile(
            [Tool(name="alpha", budget=1), Tool(name="beta", budget=1)],
            [
                {"name": "alpha", "args": {}, "id": "a1"},
                {"name": "beta", "args": {}, "id": "b1"},
            ],
        )

        async def drive():
            return await asyncio.wait_for(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)), timeout=3.0)

        result = asyncio.run(drive())
        assert result["explore_result"] == Claims(items=["done"])
        assert a.calls == 1 and b.calls == 1

    def test_result_order_follows_tool_call_order_not_completion_order(self):
        """A fast second call finishing before a slow first call must not reorder
        the tool_log — assembly is in original tool_call order."""

        async def slow():
            await asyncio.sleep(0.06)

        async def fast():
            await asyncio.sleep(0.01)

        slow_tool = _RecordingTool("slow", slow, "slow-done")
        fast_tool = _RecordingTool("fast", fast, "fast-done")
        register_tool_factory("slow", lambda config, tool_config: slow_tool)
        register_tool_factory("fast", lambda config, tool_config: fast_tool)

        graph = _compile(
            [Tool(name="slow", budget=1), Tool(name="fast", budget=1)],
            [
                {"name": "slow", "args": {}, "id": "s1"},
                {"name": "fast", "args": {}, "id": "f1"},
            ],
        )

        result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))
        tool_log = result["explore_tool_log"]
        assert [t.tool_name for t in tool_log] == ["slow", "fast"], (
            "tool_log must follow tool_call order, not completion order"
        )

    def test_budget_enforced_across_parallel_calls_to_same_tool(self):
        """Two parallel calls to a budget=1 tool: the budget is pre-reserved in
        order before gathering, so only ONE call actually runs. A naive
        gather-then-record would let both through (can_call races)."""

        async def noop():
            return None

        dup = _RecordingTool("dup", noop, "dup-done")
        register_tool_factory("dup", lambda config, tool_config: dup)

        graph = _compile(
            [Tool(name="dup", budget=1)],
            [
                {"name": "dup", "args": {}, "id": "d1"},
                {"name": "dup", "args": {}, "id": "d2"},
            ],
        )

        result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))
        assert dup.calls == 1, "budget=1 must permit exactly one parallel call to the same tool"
        tool_log = result["explore_tool_log"]
        assert len(tool_log) == 1, "only the budgeted call produces a ToolInteraction"

    def test_sync_twin_still_sequential_and_correct(self):
        """The sync path is unchanged: two tools in one turn still execute and
        the budget still applies. (No concurrency claim for sync.)"""

        async def noop():
            return None

        a = _RecordingTool("alpha", noop, "a-done")
        b = _RecordingTool("beta", noop, "b-done")
        register_tool_factory("alpha", lambda config, tool_config: a)
        register_tool_factory("beta", lambda config, tool_config: b)

        graph = _compile(
            [Tool(name="alpha", budget=1), Tool(name="beta", budget=1)],
            [
                {"name": "alpha", "args": {}, "id": "a1"},
                {"name": "beta", "args": {}, "id": "b1"},
            ],
        )
        result = run(graph, input=dict(_INPUT), config=dict(_CFG))
        assert result["explore_result"] == Claims(items=["done"])
        assert a.calls == 1 and b.calls == 1
