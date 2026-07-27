"""Phase 6 / neograph-s7zt3.2 -- PORTAL_OPERATOR HITL-gate preservation on
Agent Spec Swarm export.

CORE INVARIANT under test: a Portal mesh member's Operator (HITL human-approval
gate) MUST NOT silently vanish when the mesh is exported to Agent Spec. Before
the fix, ``_lower_portal_mesh_to_swarm`` returned a bare ``Swarm`` unconditionally
-- the ``Operator(when=...)`` gate on a member was dropped with ZERO error, a
safety-critical silent control-flow seam.

The fix (design: docs/design/architecture-audit-phase6-design-2026-07-27.md):
when ANY mesh member carries an Operator, wrap the ``Swarm`` in a mesh-exit
pause composite -- ``AgentNode(agent=Swarm)`` -> ``BranchingNode`` ->
``InputMessageNode`` -- carrying every gated member's ``when`` condition in a new
``_MARK_PORTAL_OPERATOR_SPEC`` dict-valued marker ({member_name: when}) on the
check node, generalizing to any number of independently-gated members as ONE
shared composite. The loader recognizes the composite structurally and
re-attaches ``Operator(when=...)`` per gated member.

Design verified live against pyagentspec 26.1.2 (Swarm has no interior per-member
pause primitive, so mesh-exit is the faithful approximation). This is a
round-trip-lossless serialization, NOT a behaviorally-faithful foreign runtime
(see design §2's fidelity boundary).
"""

from __future__ import annotations

import warnings

from pydantic import BaseModel

from neograph import Construct, Node, Operator, Portal
from neograph._agent_spec import (
    _MARK_MODIFIER,
    _MARK_PORTAL_OPERATOR_SPEC,
    to_agent_spec,
)
from neograph.loader import from_agent_spec
from tests.fakes import register_condition, register_scripted


class _Payload(BaseModel):
    goto: str


def _register() -> None:
    register_scripted("_po_a", lambda i, c: _Payload(goto="b"))
    register_scripted("_po_b", lambda i, c: _Payload(goto="c"))
    register_scripted("_po_c", lambda i, c: _Payload(goto="__end__"))
    register_condition("_po_gate_a", lambda d: True)
    register_condition("_po_gate_b", lambda d: False)


def _one_gated_mesh() -> Construct:
    """Two-member mesh, entry gated by an Operator."""
    _register()
    return Construct(
        "po_mesh1",
        nodes=[
            Node.scripted("a", fn="_po_a", inputs={"handoff": _Payload}, outputs=_Payload)
            | Portal(to=["b"])
            | Operator(when="_po_gate_a"),
            Node.scripted("b", fn="_po_b", inputs={"handoff": _Payload}, outputs=_Payload) | Portal(to=[]),
        ],
    )


def _two_gated_mesh() -> Construct:
    """Three-member mesh, two members (``a`` and ``b``) INDEPENDENTLY gated by
    two DIFFERENT Operator conditions."""
    _register()
    return Construct(
        "po_mesh2",
        nodes=[
            Node.scripted("a", fn="_po_a", inputs={"handoff": _Payload}, outputs=_Payload)
            | Portal(to=["b"])
            | Operator(when="_po_gate_a"),
            Node.scripted("b", fn="_po_b", inputs={"handoff": _Payload}, outputs=_Payload)
            | Portal(to=["c"])
            | Operator(when="_po_gate_b"),
            Node.scripted("c", fn="_po_c", inputs={"handoff": _Payload}, outputs=_Payload) | Portal(to=[]),
        ],
    )


def _import(flow):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_agent_spec(flow)


class TestGatePreservedOnExport:
    """The bug: the Operator gate silently vanishes on export. The exported
    artifact must NOT be a bare Swarm; it must carry the gate marker."""

    def test_export_is_not_a_bare_swarm_when_a_member_is_gated(self):
        flow = to_agent_spec(_one_gated_mesh())
        assert type(flow).__name__ == "Flow", (
            "a gated mesh must export to a Flow mesh-exit composite, not a bare Swarm "
            "-- the Operator gate would otherwise silently vanish"
        )

    def test_export_still_a_bare_swarm_when_no_member_is_gated(self):
        _register()
        ungated = Construct(
            "po_ungated",
            nodes=[
                Node.scripted("a", fn="_po_a", inputs={"handoff": _Payload}, outputs=_Payload) | Portal(to=["b"]),
                Node.scripted("b", fn="_po_b", inputs={"handoff": _Payload}, outputs=_Payload) | Portal(to=[]),
            ],
        )
        flow = to_agent_spec(ungated)
        assert type(flow).__name__ == "Swarm", "an ungated mesh must keep today's bare-Swarm export (zero behavior change)"

    def test_check_node_carries_the_gated_member_and_its_condition(self):
        # Marker is keyed by the AGENT name (`{member}-agent`, set by
        # _make_agent) -- the identity that survives export->import, so the
        # loader's `member.name in gated` matches the reconstructed member.
        flow = to_agent_spec(_one_gated_mesh())
        check = next(n for n in flow.nodes if (n.metadata or {}).get(_MARK_MODIFIER) == "portal_operator")
        marker = check.metadata[_MARK_PORTAL_OPERATOR_SPEC]
        assert marker == {"a-agent": "_po_gate_a"}

    def test_check_node_uses_a_distinct_modifier_value(self):
        """The mesh-exit check must NOT reuse the plain single-node "operator"
        _MARK_MODIFIER value, or import lookahead would conflate the two shapes."""
        flow = to_agent_spec(_one_gated_mesh())
        modifiers = {(n.metadata or {}).get(_MARK_MODIFIER) for n in flow.nodes}
        assert "portal_operator" in modifiers
        assert "operator" not in modifiers


class TestMultipleIndependentlyGatedMembers:
    """Design §2: any number of independently-gated members generalize to ONE
    shared mesh-exit composite carrying a multi-entry marker dict."""

    def test_two_members_both_conditions_in_one_shared_marker(self):
        flow = to_agent_spec(_two_gated_mesh())
        checks = [n for n in flow.nodes if (n.metadata or {}).get(_MARK_MODIFIER) == "portal_operator"]
        assert len(checks) == 1, "all gated members must share ONE mesh-exit composite"
        assert checks[0].metadata[_MARK_PORTAL_OPERATOR_SPEC] == {"a-agent": "_po_gate_a", "b-agent": "_po_gate_b"}


class TestRoundTrip:
    """Export -> import recovers the Operator gate on exactly the gated members,
    with their exact condition strings.

    NOTE: the pre-existing Swarm importer names each reconstructed member by its
    Agent's name (``{member}-agent``, set by ``_make_agent``), so imported
    members are ``a-agent``/``b-agent``/... -- a lossy-rename characteristic of
    the Swarm round-trip that predates this task, not introduced by it."""

    def test_single_gated_member_round_trips(self):
        back = _import(to_agent_spec(_one_gated_mesh()))
        by_name = {n.name: n for n in back.nodes}
        assert by_name["a-agent"].modifier_set.operator is not None
        assert by_name["a-agent"].modifier_set.operator.when == "_po_gate_a"
        assert by_name["b-agent"].modifier_set.operator is None

    def test_two_gated_members_round_trip_with_distinct_conditions(self):
        back = _import(to_agent_spec(_two_gated_mesh()))
        by_name = {n.name: n for n in back.nodes}
        assert by_name["a-agent"].modifier_set.operator is not None
        assert by_name["a-agent"].modifier_set.operator.when == "_po_gate_a"
        assert by_name["b-agent"].modifier_set.operator is not None
        assert by_name["b-agent"].modifier_set.operator.when == "_po_gate_b"
        assert by_name["c-agent"].modifier_set.operator is None

    def test_portal_topology_and_entry_knobs_survive_alongside_gate(self):
        back = _import(to_agent_spec(_one_gated_mesh()))
        by_name = {n.name: n for n in back.nodes}
        entry_portal = by_name["a-agent"].modifier_set.portal
        assert entry_portal is not None
        assert entry_portal.to == ["b-agent"]

    def test_flow_from_dict_to_dict_round_trip_then_reimport(self):
        """The design verified Flow.from_dict(to_dict()) live -- a stronger check
        than key inspection. The gate must survive full serialization."""
        flow = to_agent_spec(_two_gated_mesh())
        rebuilt = type(flow).from_dict(flow.to_dict())
        back = _import(rebuilt)
        by_name = {n.name: n for n in back.nodes}
        assert by_name["a-agent"].modifier_set.operator.when == "_po_gate_a"
        assert by_name["b-agent"].modifier_set.operator.when == "_po_gate_b"
