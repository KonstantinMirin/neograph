"""Portal peer-mesh -> Agent Spec Swarm lowering.

Extracted from ``_agent_spec.py`` (neograph-3ffdg.3) as a pure file split — the
functions below are unchanged, only their home moved. This is the structurally
separate mesh-detection path; the linear lowering stays in ``_agent_spec.py``.

Mesh lowering recursively exports each member construct, which would close an
import cycle with the parent exporter. Rather than break that with a deferred
import — which would have required growing
``FUNCTION_LOCAL_IMPORT_ALLOWLIST`` in
``tests/test_guards_sidecar_imports.py``, and this project's ratchet allowlists
may only SHRINK — ``_lower_portal_mesh_to_swarm`` takes the exporter as an
``export_flow`` parameter. The single call site in ``_agent_spec.py`` passes
``to_agent_spec``. No cycle exists to document.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from neograph._agent_spec_markers import (
    _MARK_MODIFIER,
    _MARK_PORTAL_MEMBER_SPEC,
    _MARK_PORTAL_OPERATOR_SPEC,
    _MARK_PORTAL_SPEC,
    _MARK_PROMPT_SPEC,
    Branch,
    _import_agent_spec_flow_classes,
)
from neograph._agent_spec_node_lowering import _make_agent
from neograph._agent_spec_placeholders import (
    _prompt_spec_marker,
    _properties_for,
    _translate_placeholders,
)
from neograph._agent_spec_swarm_encoding import SWARM_ENCODING, mesh_handoff_mode
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph.construct import Construct
from neograph.node import Node


def _lower_portal_mesh_to_swarm(
    construct: Construct,
    members: list[Node | Construct],
    tools_mod: Any,
    export_flow: Callable[[Construct], Any],
) -> Any:
    """Export a Portal mode-(a) peer mesh to a top-level pyagentspec ``Swarm``
    -- the export-direction mirror of ``loader.py``'s ``_reconstruct_swarm_mesh``
    Swarm import.

    Swarm.first_agent/relationships are typed ``AgenticComponent`` (pyagentspec
    swarm.py/agent.py), so each member lowers to a real ``Agent`` (via the
    SAME ``_make_agent`` helper agent/act-mode Flow nodes use), never an
    ``LlmNode``. The entry-only knobs (``max_hops``/``on_exhaust``/``route``)
    have no native ``Swarm`` field -- they ride a ``neograph/portal_spec``
    metadata marker (mirrors the Oracle/Each/Loop per-group marker
    convention), so the information is not lost even though the current
    Swarm importer does not read it back yet.

    ``construct.nodes`` is trusted here: ``_check_portal_mesh`` (construct-
    assembly validation) has ALREADY enforced contiguity/entry-first/uniform-
    payload/reachability for any Construct reaching export, so every ``to``
    peer reference is guaranteed to name a real member of this same mesh.
    """
    entry = members[0]
    entry_portal = entry.modifier_set.portal
    assert entry_portal is not None  # collected as Portal-modified

    # pyagentspec's Agent ties inputs Properties to {{placeholder}} names in its
    # own system_prompt (ComponentWithIO._validate_no_extra_property), so a mesh
    # member's prompt is Option-F-translated exactly like every other _make_agent
    # caller (neograph-s7zt3.1): the Agent declares ONLY the referenced flat
    # Properties, which match the rewritten {{ flat }} names by construction. A
    # member may reference the reserved 'handoff' input (${handoff.field}) --
    # shipping it raw would hand a foreign Swarm runtime a placeholder it can
    # neither fill nor flag. Outputs stay [] -- the payload/routing shape rides
    # the neograph/portal_spec marker, and the untranslated ${var} text rides a
    # per-member neograph/prompt_spec marker on the Agent itself so the Swarm
    # import recovers the original prompt grammar.
    agents_by_name: dict[str, Any] = {}
    for member in members:
        # C1 (do0d9): a Construct mesh member lowers to its recursively-exported
        # sub-Flow. A pyagentspec Flow IS an AgenticComponent (Flow.__mro__
        # includes AgenticComponent), so it drops straight into
        # Swarm.first_agent/relationships with NO Agent wrapper — the same
        # recursive-Flow-production pattern _lower_item_body uses for a bare
        # Construct item's FlowNode subflow. A Construct has no .prompt (its
        # interior prompts are Option-F-translated inside the recursive
        # to_agent_spec call) and cannot carry an Operator gate (rejected at
        # assembly), so it never needs the prompt marker or enters `gated`.
        if portal_member_class(member) is PortalMemberClass.SUB_CONSTRUCT:
            assert isinstance(member, Construct)  # classifier decided; this narrows the type only
            agents_by_name[member.name] = export_flow(member)
            continue
        assert isinstance(member, Node)  # the only other classified shape; narrows the type only
        rewritten, ref_props, flat_to_original = _translate_placeholders(
            member.prompt or "", _properties_for(member.inputs), member.name
        )
        agent = _make_agent(member, tools_mod, ref_props, [], rewritten)
        member_portal = member.modifier_set.portal
        assert member_portal is not None  # collected as Portal-modified
        agent.metadata = {
            **(agent.metadata or {}),
            _MARK_PROMPT_SPEC: _prompt_spec_marker(member, flat_to_original),
            # Swarm.handoff is MESH-level and Agent carries no mode, so without
            # this the member's own mode and trigger never reach the wire and the
            # importer can only guess (neograph-dgbqv.8). Foreign Swarms have no
            # marker and keep the mesh-level inference, which is right for them:
            # every foreign member really is an Agent.
            _MARK_PORTAL_MEMBER_SPEC: {
                "mode": member.mode,
                "trigger": member_portal.trigger,
            },
        }
        agents_by_name[member.name] = agent

    relationships = [
        (agents_by_name[member.name], agents_by_name[peer])
        for member in members
        for peer in (member.modifier_set.portal.to or [])  # type: ignore[union-attr]
    ]

    from pyagentspec.swarm import HandoffMode, Swarm

    # C1/s7zt3.14 round-trip: carry the mesh's trigger sub-mode on Swarm.handoff so
    # _reconstruct_swarm_mesh's _swarm_trigger reads it back. mesh_handoff_mode
    # (neograph-dgbqv.5) is the ONE named aggregation rule -- 'optional' wins
    # over 'never' if ANY member's export_trigger is 'tool' (a Construct member
    # is never tool-triggered by validation, so only Node members can tip it).
    member_classes = [portal_member_class(m) for m in members]
    assert all(cls is not None for cls in member_classes)  # _check_portal_mesh already validated every member
    swarm = Swarm(
        name=construct.name,
        first_agent=agents_by_name[entry.name],
        relationships=relationships,
        handoff=HandoffMode(mesh_handoff_mode(cast("list[PortalMemberClass]", member_classes))),
        metadata={
            _MARK_PORTAL_SPEC: {
                "max_hops": entry_portal.max_hops,
                "on_exhaust": entry_portal.on_exhaust,
                "route": entry_portal.route,
            }
        },
    )

    # Phase 6 (neograph-s7zt3.2): a mesh member's Operator (HITL approval gate)
    # must NOT silently vanish on export. Collect every gated member's `when` in
    # member order (deterministic, matches every other _MARK_*_SPEC builder).
    #
    # Keyed by the AGENT name (`{member.name}-agent`, set by _make_agent), NOT
    # the raw member name: _reconstruct_swarm_mesh names each imported member by
    # its Swarm agent's `.name`, so keying by the agent name is what makes the
    # loader's `member.name in gated` match on round-trip. (The design doc's §4
    # said "member name"; its repro used ad-hoc agents and so never exercised
    # _make_agent's `-agent` suffix -- the agent name is the identity that
    # actually survives export->import.)
    def _is_gated(item: Node | Construct) -> bool:
        cls = portal_member_class(item)
        return cls is not None and SWARM_ENCODING[cls].gated

    gated: dict[str, str] = {
        agents_by_name[member.name].name: member.modifier_set.operator.when
        for member in members
        if member.modifier_set.operator is not None and _is_gated(member)
    }
    if not gated:
        return swarm  # unchanged today's-behavior path: pure PORTAL cell untouched

    # Mesh-exit pause composite. Swarm has no interior per-member pause primitive
    # (verified live, pyagentspec 26.1.2 -- Swarm.relationships is a flat
    # AgenticComponent adjacency list, no Node graph to splice a check into), so
    # the gate is approximated at the point control returns to the enclosing
    # Flow, mirroring _lower_operator's existing BranchingNode + InputMessageNode
    # shape one-for-one, with the Swarm wrapped in an AgentNode (legal:
    # AgentNode.agent: SerializeAsAny[AgenticComponent], and Swarm IS an
    # AgenticComponent). ALL gated members ride one shared marker dict on the
    # single check node -- a mesh is one connected component (_check_portal_mesh),
    # so "one shared composite" is always well-defined. This is a round-trip-
    # lossless serialization, NOT a behaviorally-faithful foreign runtime: a
    # foreign engine sees one exit-point BranchingNode, not neograph's own
    # per-member interior gate (factory.make_portal_approval_fn).
    nodes_mod, flow_mod, edges_mod, property_mod, _tools_mod = _import_agent_spec_flow_classes()

    agent_node = nodes_mod.AgentNode(name=f"{construct.name}__mesh", agent=swarm)
    check = nodes_mod.BranchingNode(
        name=f"{construct.name}__portal_operator_check",
        mapping={Branch.TRUE: Branch.PAUSE, Branch.FALSE: Branch.DEFAULT},
        metadata={
            _MARK_MODIFIER: "portal_operator",
            _MARK_PORTAL_OPERATOR_SPEC: gated,
        },
    )
    input_message = nodes_mod.InputMessageNode(
        name=f"{construct.name}__portal_operator_pause",
        outputs=[property_mod.StringProperty(title="user_input")],
    )
    start = nodes_mod.StartNode(name=f"{construct.name}__start")
    end_default = nodes_mod.EndNode(name=f"{construct.name}__end_default")
    end_paused = nodes_mod.EndNode(name=f"{construct.name}__end_paused")

    return flow_mod.Flow(
        name=construct.name,
        start_node=start,
        nodes=[start, agent_node, check, input_message, end_default, end_paused],
        control_flow_connections=[
            edges_mod.ControlFlowEdge(name=f"{construct.name}__start_to_mesh", from_node=start, to_node=agent_node),
            edges_mod.ControlFlowEdge(name=f"{construct.name}__mesh_to_check", from_node=agent_node, to_node=check),
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__check_to_pause",
                from_node=check,
                from_branch=Branch.PAUSE,
                to_node=input_message,
            ),
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__check_to_default",
                from_node=check,
                from_branch=Branch.DEFAULT,
                to_node=end_default,
            ),
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__pause_to_end", from_node=input_message, to_node=end_paused
            ),
        ],
    )


def _is_peer_mesh_member(item: Any) -> bool:
    """True iff ``item`` carries a PEER-mode (non-dispatch) Portal — i.e. it is
    a Portal mesh member — using the SAME structural, modifier-agnostic
    detection ``portal_member_class`` uses, never an ``isinstance(Node)`` gate
    (A1).

    ``portal_member_class`` reads ``.modifier_set`` on Node AND Construct, so a
    Construct mesh member (do0d9) is correctly detected as a member rather than
    misclassified as a "non-mesh node" and false-rejecting the whole mesh.
    """
    return portal_member_class(item) not in (None, PortalMemberClass.DISPATCH)
