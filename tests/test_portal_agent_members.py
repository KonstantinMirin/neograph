"""Runtime routing tests for agent/act Portal mesh members (neograph-nnds9).

Proves the Core Invariant end-to-end: an agent/act mesh member's Portal-
visible identity is its entry (``{member}__agent``) and reconverging exit
(``{member}__parse``) — a peer's ``Command(goto=...)`` lands on the real
entry node, and the mesh's static entry edge resolves correctly when the
ENTRY itself is agent/act (the case the architect review flagged as
structurally dodged by a scripted-entry-only fixture).

Uses a single-turn fake LLM (no tool calls) so the agent cycle reaches its
parse node in one superstep: the router sends straight to parse when the
LLM response carries no ``tool_calls``, and ``parse_body`` parses the
response's JSON content directly as the node's structured output.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from neograph import HANDOFF_END, Construct, Node, Portal, compile, run
from neograph.runner import (
    _AGENT_CYCLE_OVERHEAD,
    _LANGGRAPH_DEFAULT_RECURSION_LIMIT,
    _SUPERSTEPS_PER_AGENT_TURN,
    _ensure_agent_recursion_limit,
)
from tests.fakes import build_fake_llm_kwargs, build_test_compile_kwargs, register_scripted


class RouteHop(BaseModel, frozen=True):
    goto: str
    hops: int = 0


class _SingleTurnFake:
    """Returns valid JSON content on the first turn -- no tool_calls, so the
    router sends the cycle straight to parse (one superstep)."""

    def __init__(self, response_json: str) -> None:
        self._json = response_json

    def bind_tools(self, tools: list) -> _SingleTurnFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _SingleTurnFake:
        return self

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        return AIMessage(content=self._json)

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)

    def with_structured_output(self, model: type[BaseModel], **kwargs: Any) -> _SingleTurnFake:
        return self


class TestAgentPeerMember:
    """An agent-mode member as a PEER (scripted entry routes to it)."""

    def test_agent_peer_routes_to_its_real_entry_node(self):
        def triage_fn(input_data, config):
            incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
            if incoming is None:
                return RouteHop(goto="researcher", hops=1)
            return RouteHop(goto=HANDOFF_END, hops=incoming.hops + 1)

        register_scripted("nnds9_triage", triage_fn)

        triage = Node(
            name="triage",
            mode="scripted",
            inputs={"handoff": RouteHop},
            outputs=RouteHop,
            scripted_fn="nnds9_triage",
        ) | Portal(to=["researcher"], max_hops=6)
        researcher = Node(
            name="researcher",
            mode="agent",
            inputs={"handoff": RouteHop},
            outputs=RouteHop,
            model="reason",
            prompt="test/explore",
            tools=[],
        ) | Portal(to=[])

        mesh = Construct("nnds9-peer-mesh", nodes=[triage, researcher])

        graph = compile(
            mesh,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _SingleTurnFake('{"goto": "__end__", "hops": 2}')),
        )
        result = run(graph, input={})

        assert result.get("triage") == RouteHop(goto="researcher", hops=1)
        assert result.get("researcher") == RouteHop(goto=HANDOFF_END, hops=2)


class TestAgentEntryMember:
    """An agent-mode member as the mesh ENTRY -- the case the architect
    review flagged as structurally dodged by a scripted-entry-only fixture:
    the static `prev -> entry` edge must resolve to `{entry}__agent`, not
    the bare `entry.name` (which does not exist as a LangGraph node)."""

    def test_agent_entry_static_edge_resolves_to_its_agent_node(self):
        def billing_fn(input_data, config):
            incoming = input_data["handoff"]
            return RouteHop(goto=HANDOFF_END, hops=incoming.hops + 1)

        register_scripted("nnds9_billing", billing_fn)

        triage = Node(
            name="triage",
            mode="agent",
            inputs={"handoff": RouteHop},
            outputs=RouteHop,
            model="reason",
            prompt="test/explore",
            tools=[],
        ) | Portal(to=["billing"], max_hops=6)
        billing = Node(
            name="billing",
            mode="scripted",
            inputs={"handoff": RouteHop},
            outputs=RouteHop,
            scripted_fn="nnds9_billing",
        ) | Portal(to=[])

        mesh = Construct("nnds9-entry-mesh", nodes=[triage, billing])

        graph = compile(
            mesh,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _SingleTurnFake('{"goto": "billing", "hops": 1}')),
        )
        result = run(graph, input={})

        assert result.get("triage") == RouteHop(goto="billing", hops=1)
        assert result.get("billing") == RouteHop(goto=HANDOFF_END, hops=2)


class TestRecursionFloorAccountsForAgentMeshMemberCost:
    """The recursion-limit floor must budget an agent/act mesh member's own
    ReAct-cycle cost PER HOP (max_hops * per_hop_cost), not a flat
    1-superstep-per-hop assumption -- the architect review's recursion-floor
    finding."""

    def test_floor_multiplies_max_hops_by_agent_member_hop_cost(self):
        register_scripted("f", lambda i, c: RouteHop(goto=HANDOFF_END))
        billing = Node(
            name="billing", mode="scripted", inputs={"handoff": RouteHop}, outputs=RouteHop, scripted_fn="f"
        ) | Portal(to=[])
        triage = Node(
            name="triage",
            mode="agent",
            inputs={"handoff": RouteHop},
            outputs=RouteHop,
            model="reason",
            prompt="test/explore",
            tools=[],
        ) | Portal(to=["billing"], max_hops=6)
        mesh = Construct("nnds9-recursion-mesh", nodes=[triage, billing])

        graph = compile(
            mesh,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _SingleTurnFake('{"goto": "__end__"}')),
        )

        result = _ensure_agent_recursion_limit(graph, None)

        from neograph._llm_config import _coerce_llm_config

        max_iters = _coerce_llm_config(triage.llm_config).max_iterations
        per_hop_cost = max_iters * _SUPERSTEPS_PER_AGENT_TURN + _AGENT_CYCLE_OVERHEAD
        expected_floor = _LANGGRAPH_DEFAULT_RECURSION_LIMIT + 6 * per_hop_cost

        assert result is not None
        assert result["recursion_limit"] == expected_floor, (
            f"expected max_hops(6) * per_hop_cost({per_hop_cost}) = {6 * per_hop_cost} added to the default, "
            f"not a flat max_hops(6) — got {result['recursion_limit']}"
        )

    def test_agent_mesh_member_not_double_counted_against_flat_agent_cost(self):
        """An agent/act node that is ALSO a Portal mesh member must be excluded
        from the flat per-node agent_cost sum -- its cost is captured entirely
        by the mesh-aware calculation."""
        register_scripted("f", lambda i, c: RouteHop(goto=HANDOFF_END))
        billing = Node(
            name="billing", mode="scripted", inputs={"handoff": RouteHop}, outputs=RouteHop, scripted_fn="f"
        ) | Portal(to=[])
        triage = Node(
            name="triage",
            mode="agent",
            inputs={"handoff": RouteHop},
            outputs=RouteHop,
            model="reason",
            prompt="test/explore",
            tools=[],
        ) | Portal(to=["billing"], max_hops=1)
        mesh = Construct("nnds9-no-double-count-mesh", nodes=[triage, billing])

        graph = compile(
            mesh,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: _SingleTurnFake('{"goto": "__end__"}')),
        )

        result = _ensure_agent_recursion_limit(graph, None)

        from neograph._llm_config import _coerce_llm_config

        max_iters = _coerce_llm_config(triage.llm_config).max_iterations
        per_hop_cost = max_iters * _SUPERSTEPS_PER_AGENT_TURN + _AGENT_CYCLE_OVERHEAD
        # max_hops=1 -- if the agent node's cost were ALSO added flat (double
        # counted), the floor would be default + per_hop_cost (mesh) +
        # per_hop_cost (flat) = default + 2*per_hop_cost.
        expected_floor = _LANGGRAPH_DEFAULT_RECURSION_LIMIT + 1 * per_hop_cost

        assert result is not None
        assert result["recursion_limit"] == expected_floor, (
            "agent/act mesh member cost must not be double-counted "
            f"(flat + mesh-aware) — got {result['recursion_limit']}, expected {expected_floor}"
        )
