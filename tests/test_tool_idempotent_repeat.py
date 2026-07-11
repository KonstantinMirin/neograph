"""Idempotent repeat-call guard (ox-troubleshooting-demo 8ko.34): a REPEATED identical call to an
``idempotent=True`` tool within one node is served from the cycle's own history — it does NOT
re-invoke the tool and does NOT consume budget.

The motivating incident (v7-efficiency P4): a json_mode parse-failure retry re-issued a node's
whole tool batch with FRESH tool_call ids after valid results had already arrived. Each repeat
burned per-tool budget, so a budget=2 dimension sweep was spent as FORMAT×2 and DEVICE/COUNTRY
were NEVER swept — a silent coverage crowd-out, not just wasted cost.

Contract pinned here:
  1. CROWD-OUT KILLED (the RED test): an identical repeat (same tool, same args) of an
     ``idempotent=True`` tool is answered with the cached result — the tool body runs ONCE for
     those args, the repeat consumes NO budget, and a later DIFFERENT-args call still runs.
  2. The cached reply is byte-identical to the original rendered ToolMessage.
  3. DIFFERENT args are never deduped (a sweep's FORMAT then DEVICE both run).
  4. A non-idempotent tool is never deduped (repeat semantics are opt-in via Tool.idempotent —
     the same flag that already gates transport replay).
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


class _CountingTool:
    """Counts real invocations and answers per-args, mirroring the .invoke/.ainvoke surface."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []

    def _answer(self, args: dict) -> str:
        return f"{self.name}:{args.get('groupByColumn', '?')}"

    def invoke(self, args: dict, config=None) -> str:
        self.calls.append(dict(args))
        return self._answer(args)

    async def ainvoke(self, args: dict, config=None) -> str:
        self.calls.append(dict(args))
        return self._answer(args)


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


def _compile(tools: list[Tool], turns: list[list[dict]]):
    fake = ReActFake(
        tool_calls=[*turns, []],  # scripted tool-using turns, then the final structured turn
        final=lambda m: m(items=["done"]),
    )
    return compile(
        construct_from_functions("p", [_agent_node(tools)]),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: fake),
    )


def _fmt(call_id: str) -> dict:
    return {"name": "sweep", "args": {"groupByColumn": "FORMAT"}, "id": call_id}


def _dev(call_id: str) -> dict:
    return {"name": "sweep", "args": {"groupByColumn": "DEVICE"}, "id": call_id}


class TestIdempotentRepeatGuard:
    def test_retry_reissue_cannot_crowd_out_unswept_axes(self):
        """THE 8ko.34 pin: budget=2, turn 1 sweeps FORMAT, turn 2 (the retry) re-issues the SAME
        FORMAT call with a fresh id, turn 3 asks for DEVICE. Without the guard the repeat burns
        the budget as FORMAT×2 and DEVICE short-circuits 'budget exhausted' — the crowd-out.
        With it: FORMAT runs once, the repeat is served from cache (no budget), DEVICE runs."""
        sweep = _CountingTool("sweep")
        register_tool_factory("sweep", lambda config, tool_config: sweep)

        graph = _compile(
            [Tool(name="sweep", budget=2, idempotent=True)],
            [[_fmt("c1")], [_fmt("c2-retry")], [_dev("c3")]],
        )
        result = run(graph, input=dict(_INPUT), config=dict(_CFG))

        swept = [c.get("groupByColumn") for c in sweep.calls]
        assert swept == ["FORMAT", "DEVICE"], (
            f"the repeat must not re-invoke nor consume budget: expected one FORMAT + one DEVICE "
            f"real invocation, got {swept} (FORMAT×2 == the crowd-out bug)"
        )

    def test_cached_reply_is_byte_identical_to_the_original(self):
        """Unit leg (the neo_ message channel is internal): the cache key is canonical over arg
        ORDER, and _seed_repeat_cache maps a prior interaction's key to its EXACT rendered result
        — what the guard serves back verbatim."""
        from neograph._agent_cycle import _idempotent_repeat_key, _seed_repeat_cache

        flags = {"sweep": True}
        k1 = _idempotent_repeat_key({"name": "sweep", "args": {"a": 1, "b": 2}}, flags)
        k2 = _idempotent_repeat_key({"name": "sweep", "args": {"b": 2, "a": 1}}, flags)
        assert k1 == k2 and k1 is not None, "arg order must not defeat the dedup key"

        prior = ToolInteraction(tool_name="sweep", args={"a": 1, "b": 2}, result="sweep:RENDERED")
        cache = _seed_repeat_cache([prior], flags)
        assert cache == {k1: "sweep:RENDERED"}, "the cache must serve the ORIGINAL render verbatim"

        assert _idempotent_repeat_key({"name": "sweep", "args": {}}, {"sweep": False}) is None

    def test_different_args_are_never_deduped(self):
        sweep = _CountingTool("sweep")
        register_tool_factory("sweep", lambda config, tool_config: sweep)

        graph = _compile(
            [Tool(name="sweep", budget=3, idempotent=True)],
            [[_fmt("c1")], [_dev("c2")]],
        )
        run(graph, input=dict(_INPUT), config=dict(_CFG))

        assert [c.get("groupByColumn") for c in sweep.calls] == ["FORMAT", "DEVICE"]

    def test_non_idempotent_tool_is_never_deduped(self):
        """Repeat-dedup is OPT-IN via Tool.idempotent — a mutating tool must re-run (and re-spend
        budget) on every call, even with identical args."""
        sweep = _CountingTool("sweep")
        register_tool_factory("sweep", lambda config, tool_config: sweep)

        graph = _compile(
            [Tool(name="sweep", budget=3, idempotent=False)],
            [[_fmt("c1")], [_fmt("c2")]],
        )
        run(graph, input=dict(_INPUT), config=dict(_CFG))

        assert len(sweep.calls) == 2, "a non-idempotent tool must never be served from cache"

    def test_async_twin_applies_the_same_guard(self):
        """The async path (pre-reserve + gather) must apply the SAME dedup: an in-batch repeat
        and a cross-turn repeat are both served from cache."""
        sweep = _CountingTool("sweep")
        register_tool_factory("sweep", lambda config, tool_config: sweep)

        graph = _compile(
            [Tool(name="sweep", budget=2, idempotent=True)],
            [[_fmt("c1")], [_fmt("c2-retry")], [_dev("c3")]],
        )

        result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))
        swept = [c.get("groupByColumn") for c in sweep.calls]
        assert swept == ["FORMAT", "DEVICE"], f"async twin must dedupe identically, got {swept}"
