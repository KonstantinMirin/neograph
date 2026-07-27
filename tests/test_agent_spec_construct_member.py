"""C1/C2 (neograph-s7zt3.12): Construct-as-Portal-mesh-member Agent Spec
export/import + dispatch-mode Portal fail-loud.

C1 export: a Construct mesh member lowers to its recursively-exported sub-Flow,
which IS a pyagentspec AgenticComponent, so it drops directly into
Swarm.first_agent/relationships beside the Agent members (no wrapper).

C1 import: a Flow Swarm member reconstructs to a Construct mesh member (boundary
I/O forced to the synthesized uniform payload); the Swarm's HandoffMode maps onto
the native Portal trigger sub-mode (OPTIONAL/ALWAYS -> 'tool', NEVER -> 'output').

C2: a dispatch-mode Portal (route='decide') is genuinely unrepresentable in Agent
Spec (no runtime-flow-synthesis primitive), so export fails LOUD.

pyagentspec-gated by module-level importorskip (the same safe pattern the sibling
round-trip suites use), testing the export/import layer directly.
"""

from __future__ import annotations

import warnings

import pytest
from pydantic import BaseModel

pytest.importorskip("pyagentspec")

from neograph import HANDOFF_END, Construct, Node, Operator, Portal  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from neograph.errors import ConfigurationError  # noqa: E402
from neograph.loader import from_agent_spec  # noqa: E402
from tests.fakes import register_condition, register_scripted  # noqa: E402


class Handoff(BaseModel, frozen=True):
    goto: str


def _construct_member_mesh(name: str = "parent_mesh", *, gated: bool = False) -> Construct:
    """A three-member mesh: scripted entry -> Construct peer (resolver_sub) ->
    scripted closer. Optionally gate the closer with an Operator (atomic, legal)."""
    register_scripted("_cm_resolve", lambda i, c: Handoff(goto="closer"))
    register_scripted("_cm_entry", lambda i, c: Handoff(goto="resolver_sub"))
    register_scripted("_cm_close", lambda i, c: Handoff(goto=HANDOFF_END))
    register_condition("_cm_gate", lambda d: None)

    resolver_sub = Construct(
        "resolver_sub",
        input=Handoff,
        output=Handoff,
        nodes=[Node.scripted("resolve", fn="_cm_resolve", outputs=Handoff)],
    )
    closer = Node.scripted("closer", fn="_cm_close", inputs={"handoff": Handoff}, outputs=Handoff) | Portal(to=[])
    if gated:
        closer = closer | Operator(when="_cm_gate")
    return Construct(
        name,
        nodes=[
            Node.scripted("entry", fn="_cm_entry", outputs=Handoff) | Portal(to=["resolver_sub"], max_hops=6),
            resolver_sub | Portal(to=["closer"]),
            closer,
        ],
    )


class TestConstructMemberExport:
    """C1 export: a Construct mesh member becomes a Flow inside the Swarm."""

    def test_construct_member_exports_as_flow_agenticcomponent(self):
        swarm = to_agent_spec(_construct_member_mesh())
        assert type(swarm).__name__ == "Swarm"
        members = [swarm.first_agent, *(dst for _src, dst in swarm.relationships)]
        by_name = {m.name: type(m).__name__ for m in members}
        # The Construct member is a Flow (recursively exported); the Node members
        # are Agents (the -agent suffix is _make_agent's).
        assert by_name["resolver_sub"] == "Flow"
        assert by_name["entry-agent"] == "Agent"
        assert by_name["closer-agent"] == "Agent"

    def test_construct_member_relationships_preserved(self):
        swarm = to_agent_spec(_construct_member_mesh())
        edges = {(s.name, d.name) for s, d in swarm.relationships}
        assert ("entry-agent", "resolver_sub") in edges
        assert ("resolver_sub", "closer-agent") in edges


class TestConstructMemberRoundTrip:
    """C1 round-trip: export -> import reconstructs the Construct mesh member and
    the mesh reassembles (validation passes)."""

    def test_construct_member_round_trips_as_construct(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = from_agent_spec(to_agent_spec(_construct_member_mesh()))
        by_kind = {n.name: type(n).__name__ for n in back.nodes}
        assert by_kind["resolver_sub"] == "Construct"
        # The mesh reassembled cleanly (Construct.__init__ ran _check_portal_mesh
        # with no raise) and the Construct member kept its Portal peer.
        sub = next(n for n in back.nodes if n.name == "resolver_sub")
        assert sub.modifier_set.portal is not None
        assert sub.modifier_set.portal.to == ["closer-agent"]

    def test_construct_member_composes_with_operator_gate(self):
        """A gated mesh containing a Construct member exports as the Operator
        mesh-exit composite AND round-trips both the Construct member and the
        Operator gate on the atomic member (must not break s7zt3.2)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flow = to_agent_spec(_construct_member_mesh("gated_parent", gated=True))
            assert type(flow).__name__ == "Flow"  # the operator-gate composite
            back = from_agent_spec(flow)
        gates = {n.name: (n.modifier_set.operator.when if n.modifier_set.operator else None) for n in back.nodes}
        kinds = {n.name: type(n).__name__ for n in back.nodes}
        assert kinds["resolver_sub"] == "Construct"  # Construct member survives
        assert gates["closer-agent"] == "_cm_gate"  # Operator gate survives
        assert gates["resolver_sub"] is None  # Construct members can't be gated


class TestHandoffModeTriggerMapping:
    """C1 import: Swarm.handoff maps onto the native Portal trigger sub-mode."""

    def _foreign_swarm(self, handoff_mode):
        from pyagentspec.agent import Agent
        from pyagentspec.llms import VllmConfig
        from pyagentspec.swarm import Swarm

        llm = VllmConfig(name="llm", model_id="m", url="http://x")
        a = Agent(name="a", system_prompt="x", llm_config=llm)
        b = Agent(name="b", system_prompt="y", llm_config=llm)
        return Swarm(name="s", first_agent=a, relationships=[(a, b), (b, a)], handoff=handoff_mode)

    @pytest.mark.parametrize("mode_name", ["OPTIONAL", "ALWAYS"])
    def test_optional_and_always_map_to_tool(self, mode_name):
        from pyagentspec.swarm import HandoffMode

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = from_agent_spec(self._foreign_swarm(getattr(HandoffMode, mode_name)))
        assert all(n.modifier_set.portal.trigger == "tool" for n in back.nodes)

    def test_never_maps_to_output(self):
        from pyagentspec.swarm import HandoffMode

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = from_agent_spec(self._foreign_swarm(HandoffMode.NEVER))
        assert all(n.modifier_set.portal.trigger == "output" for n in back.nodes)

    def test_native_tool_triggered_mesh_round_trips_trigger(self):
        """A native agent mesh with Portal(trigger='tool') sets handoff=OPTIONAL on
        export and reimports as trigger='tool' (round-trip fidelity)."""
        triage = Node(
            name="triage", mode="agent", model="fast", prompt="p", inputs={"handoff": Handoff}, outputs=Handoff, tools=[]
        ) | Portal(to=["worker"], trigger="tool", max_hops=4)
        worker = Node(
            name="worker", mode="agent", model="fast", prompt="p", inputs={"handoff": Handoff}, outputs=Handoff, tools=[]
        ) | Portal(to=["triage"], trigger="tool")
        swarm = to_agent_spec(Construct("native_tool_mesh", nodes=[triage, worker]))
        assert swarm.handoff.value in ("optional", "always")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = from_agent_spec(swarm)
        assert all(n.modifier_set.portal.trigger == "tool" for n in back.nodes)


class TestDispatchModePortalFailsLoud:
    """C2: dispatch-mode Portal export is a permanent, evidence-backed scope
    boundary — fail LOUD (no runtime-flow-synthesis primitive in Agent Spec)."""

    def test_dispatch_mode_portal_export_raises(self):
        class Emitted(BaseModel, frozen=True):
            spec: dict
            dispatch_input: dict

        class Summary(BaseModel, frozen=True):
            text: str

        register_scripted("_c2_planner", lambda i, c: Emitted(spec={}, dispatch_input={}))
        pipeline = Construct(
            "dispatch-pipe",
            nodes=[
                Node.scripted("planner", fn="_c2_planner", outputs=Emitted)
                | Portal(route="decide", spec_field="spec", input_field="dispatch_input", output=Summary, max_depth=3)
            ],
        )
        with pytest.raises(ConfigurationError, match="dispatch-mode Portal"):
            to_agent_spec(pipeline)
