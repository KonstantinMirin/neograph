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
            name="triage",
            mode="agent",
            model="fast",
            prompt="p",
            inputs={"handoff": Handoff},
            outputs=Handoff,
            tools=[],
        ) | Portal(to=["worker"], trigger="tool", max_hops=4)
        worker = Node(
            name="worker",
            mode="agent",
            model="fast",
            prompt="p",
            inputs={"handoff": Handoff},
            outputs=Handoff,
            tools=[],
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

    def test_dispatch_mode_fail_loud_is_consistent_with_the_swarm_encoding_table(self):
        """neograph-dgbqv.5 step 5: the dispatch-mode fail-loud stays AS-IS at
        _agent_spec.py:295-303 (not rewired) -- this pins the EXISTING raise as
        DATA-consistent with SWARM_ENCODING[PortalMemberClass.DISPATCH], rather
        than the table dictating a code change."""
        from neograph._agent_spec_swarm_encoding import SWARM_ENCODING
        from neograph._portal_member import PortalMemberClass

        row = SWARM_ENCODING[PortalMemberClass.DISPATCH]
        assert row.exportable is False
        assert row.reason, "DISPATCH's non-exportability must carry a reason string"
        assert row.spec_class is None
        assert row.export_trigger is None


class TestSwarmEncodingTable:
    """neograph-dgbqv.5 step 9: the SWARM_ENCODING / HANDOFF_MODE_TRIGGER tables
    and their derived helpers -- totality, fail-loud inverses, mesh-level
    aggregation, and the documented-lossy mixed-mesh cell. Written against the
    not-yet-created table module (TDD red: ImportError until neograph-jn555.39
    lands _agent_spec_swarm_encoding.py)."""

    def test_swarm_encoding_is_total_over_portal_member_class(self):
        from neograph._agent_spec_swarm_encoding import SWARM_ENCODING
        from neograph._portal_member import PortalMemberClass

        missing = [cls for cls in PortalMemberClass if cls not in SWARM_ENCODING]
        assert not missing, f"SWARM_ENCODING has no row for: {missing}"

    def test_handoff_mode_trigger_is_total_over_handoff_mode_values(self):
        from neograph._agent_spec_swarm_encoding import HANDOFF_MODE_TRIGGER

        mode_values = {row.mode_value for row in HANDOFF_MODE_TRIGGER}
        assert mode_values == {"never", "optional", "always"}

    def test_sub_construct_forces_output_trigger_even_under_optional_handoff(self):
        """The SUB_CONSTRUCT row's import_forced_trigger overrides the mesh
        trigger -- a Construct member cannot be trigger='tool' (s7zt3.14: tool-
        trigger requires an agent/act member with a ReAct turn)."""
        from neograph._agent_spec_swarm_encoding import SWARM_ENCODING
        from neograph._portal_member import PortalMemberClass

        assert SWARM_ENCODING[PortalMemberClass.SUB_CONSTRUCT].import_forced_trigger == "output"

    def test_mode_inverse_resolves_tool_to_the_canonical_optional_not_always(self):
        from neograph._agent_spec_swarm_encoding import handoff_mode_for_class
        from neograph._portal_member import PortalMemberClass

        # AGENT_CYCLE_TOOL's export_trigger is 'tool'; the canonical mode for
        # 'tool' is 'optional' (ALWAYS is byte-identical in the reference
        # LangGraph adapter and is therefore non-canonical, per the table).
        assert handoff_mode_for_class(PortalMemberClass.AGENT_CYCLE_TOOL) == "optional"

    def test_spec_class_inverse_raises_for_the_non_invertible_agent_direction(self):
        """Four PortalMemberClass rows produce spec_class='Agent'; asking which
        ONE of them a foreign 'Agent' import maps back to is a question the
        table cannot answer (the import path never recovers a PortalMemberClass
        from 'Agent' -- it builds a Node and applies the mesh trigger instead).
        This must raise loud, never silently pick one."""
        from neograph._agent_spec_swarm_encoding import spec_class_to_member_class
        from neograph.errors import ConfigurationError as _ConfigurationError

        with pytest.raises(_ConfigurationError):
            spec_class_to_member_class("Agent")

    def test_mesh_handoff_mode_any_tool_wins_over_output(self):
        from neograph._agent_spec_swarm_encoding import mesh_handoff_mode
        from neograph._portal_member import PortalMemberClass

        mode = mesh_handoff_mode([PortalMemberClass.ATOMIC, PortalMemberClass.AGENT_CYCLE_TOOL])
        assert mode == "optional"

    def test_mesh_handoff_mode_all_output_is_never(self):
        from neograph._agent_spec_swarm_encoding import mesh_handoff_mode
        from neograph._portal_member import PortalMemberClass

        mode = mesh_handoff_mode([PortalMemberClass.ATOMIC, PortalMemberClass.SUB_CONSTRUCT])
        assert mode == "never"

    def test_mesh_handoff_mode_raises_on_dispatch_member(self):
        """A dispatch-mode Portal is not a mesh member (_portal_member.py's own
        docstring) -- DISPATCH must never reach mesh_handoff_mode."""
        from neograph._agent_spec_swarm_encoding import mesh_handoff_mode
        from neograph._portal_member import PortalMemberClass
        from neograph.errors import ConfigurationError as _ConfigurationError

        with pytest.raises(_ConfigurationError):
            mesh_handoff_mode([PortalMemberClass.ATOMIC, PortalMemberClass.DISPATCH])

    def test_mixed_agent_tool_and_think_mesh_round_trips_lossily_documented(self):
        """neograph-dgbqv.8 (filed): a legal mesh of one agent(trigger='tool')
        member + one think member exports handoff=OPTIONAL; re-import's
        _swarm_trigger forces trigger='tool' onto EVERY member, which
        _validation_portal.py would normally reject on a think-mode member --
        except _agent_spec_node_import.py's markerless-agent mode='agent'
        hard-code prevents a ConstructError. The two defects CANCEL. This test
        PINS that lossy round trip as documented behavior, not a silent one --
        see neograph-dgbqv.8 for the fix (do not fix either side without the
        other)."""
        tool_member = Node(
            name="tool_member",
            mode="agent",
            model="fast",
            prompt="p",
            inputs={"handoff": Handoff},
            outputs=Handoff,
            tools=[],
        ) | Portal(to=["think_member"], trigger="tool")
        think_member = Node(
            name="think_member", mode="think", model="fast", prompt="p", inputs={"handoff": Handoff}, outputs=Handoff
        ) | Portal(to=["tool_member"])
        swarm = to_agent_spec(Construct("mixed_mesh", nodes=[tool_member, think_member]))
        assert swarm.handoff.value == "optional"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = from_agent_spec(swarm)
        # DOCUMENTED LOSSY: the think member silently becomes an agent member.
        assert all(n.modifier_set.portal.trigger == "tool" for n in back.nodes)

    def test_gated_mesh_round_trips_through_flow_from_dict(self):
        """Step 9(f): the ATOMIC_OPERATOR (gated) mesh survives a full
        to_dict/from_dict serialization round trip, not just an in-memory one."""
        flow = to_agent_spec(_construct_member_mesh("gated_dict_rt", gated=True))
        rehydrated_dict = flow.to_dict()
        rehydrated = type(flow).from_dict(rehydrated_dict)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            back = from_agent_spec(rehydrated)
        gates = {n.name: (n.modifier_set.operator.when if n.modifier_set.operator else None) for n in back.nodes}
        assert gates["closer-agent"] == "_cm_gate"
