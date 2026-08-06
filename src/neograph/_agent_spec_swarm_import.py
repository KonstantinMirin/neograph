"""Agent Spec IMPORT: the Swarm-import cluster.

Extracted from ``loader.py`` (neograph-jtawq.10). This module reconstructs a
foreign pyagentspec ``Swarm`` -- a top-level ``AgenticComponent``, not a
``Flow`` node -- onto a native Portal peer mesh, including the Phase 6
mesh-exit pause composite (Operator gates re-attached per member).

``from_agent_spec`` is INJECTED as a callable rather than imported. That is
what keeps the layering acyclic: ``loader`` imports this module, so this
module must not import ``loader``. It is also why the seam lives here and not
in ``loader`` -- if it stayed there, every reconstructor below would call
upward and the one-way property would break. Reuses the SAME
``_construct_from_subflow(subflow, name, from_spec)`` convention
``_agent_spec_group_import.py`` already proved, rather than inventing a
second injection mechanism.

Layer order, verified from the AST: ``_agent_spec_node_import`` <-
``_agent_spec_group_import`` <- this module <- ``loader``. No back-edges.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, create_model

from neograph._agent_spec_group_import import _construct_from_subflow
from neograph._agent_spec_markers import (
    _MARK_MODIFIER,
    _MARK_PORTAL_OPERATOR_SPEC,
    _MARK_PORTAL_SPEC,
    _MARK_PROMPT_SPEC,
)
from neograph._agent_spec_node_import import _agent_spec_props_to_type, _node_from_spec_agent
from neograph._agent_spec_swarm_encoding import SWARM_ENCODING, mode_to_trigger
from neograph._normalize import _with_declared_io
from neograph._portal_member import PortalMemberClass
from neograph.construct import Construct
from neograph.modifiers import Operator, Portal
from neograph.node import Node


def _swarm_agents_ordered(swarm: Any) -> list[Any]:
    """The Swarm's agents in mesh order: ``first_agent`` FIRST (it becomes the
    entry member, so ``max_hops``/``on_exhaust`` ride its Portal), then every
    other agent in order of first appearance across the directed
    ``relationships`` tuples. Deduplicated by identity."""
    ordered: list[Any] = []
    seen: set[int] = set()

    def _add(agent: Any) -> None:
        if id(agent) not in seen:
            seen.add(id(agent))
            ordered.append(agent)

    _add(swarm.first_agent)
    for src, dst in swarm.relationships:
        _add(src)
        _add(dst)
    return ordered


def _synthesize_swarm_payload(swarm: Any, agents: list[Any]) -> type[BaseModel]:
    """Synthesize the SINGLE uniform mesh payload model reused (by identity)
    as every member's ``outputs`` AND ``inputs['handoff']`` -- ``_check_portal_mesh``
    checks uniform-payload and the reserved handoff key with ``is`` identity.

    Always carries the Portal ``route`` field (``goto: str``). The Swarm's
    declared Properties (its own ``inputs``/``outputs``) and each agent's output
    schema are FOLDED IN as optional data fields where present, so a
    data-carrying Swarm does not silently lose its data channel (refinement
    addendum: no-silent-downgrade)."""
    fields: dict[str, Any] = {"goto": (str, ...)}

    prop_sources = [getattr(swarm, "outputs", None), getattr(swarm, "inputs", None)]
    prop_sources.extend(getattr(agent, "outputs", None) for agent in agents)
    for props in prop_sources:
        model = _agent_spec_props_to_type(props)
        if model is None:
            continue
        for fname, finfo in model.model_fields.items():
            if fname == "goto" or fname in fields:
                continue
            # Optional (default None) so a member that does not fill every folded
            # field can still emit the uniform payload.
            fields[fname] = (finfo.annotation | None, None)

    return create_model(f"{swarm.name}_SwarmHandoff", __base__=BaseModel, **fields)


def _swarm_trigger(swarm: Any) -> Literal["output", "tool"]:
    """Map a ``Swarm.handoff`` mode onto a native Portal ``trigger`` (C1 import,
    grounded in the swarm-langgraph-compilation-spike).

    ``HandoffMode.OPTIONAL``/``ALWAYS`` (or the bool ``True``) -> ``"tool"``: the
    reference LangGraph adapter compiles BOTH to one bound ``transfer_to_<peer>``
    tool per relationship (byte-identical — do NOT try to split OPTIONAL vs
    ALWAYS, that distinction does not materially exist in the backend), which
    neograph now represents natively as ``Portal(trigger="tool")`` (s7zt3.14).
    ``HandoffMode.NEVER`` (or ``False``/absent) -> ``"output"``: the member never
    initiates a transfer, so it keeps the existing typed-``goto`` routing.
    """
    handoff = getattr(swarm, "handoff", None)
    if handoff is True:
        return "tool"
    if handoff is False or handoff is None:
        return "output"
    value = getattr(handoff, "value", handoff)  # HandoffMode enum -> its str value
    return mode_to_trigger(value)


def _flow_member_to_construct(agent: Any, payload: type[BaseModel], from_spec: Callable[[Any], Construct]) -> Construct:
    """Reconstruct a Flow Swarm member (C1) onto a Construct mesh member whose
    boundary I/O is the synthesized uniform mesh ``payload`` (by identity).

    Calls ``_construct_from_subflow`` -- the SAME seam the bare-FlowNode item
    path uses, now genuinely shared rather than re-implemented here (it used to
    claim this reuse while holding its own copy, which is how the boundary-drop
    bug lived in one copy and not the other). It takes a ``Flow``, not a
    ``FlowNode``, and must not register into any ``output_types`` map, so it
    calls the inner seam directly rather than ``_reconstruct_item_body``.

    On top of the seam it forces BOTH the Construct boundary
    (``input``/``output``) AND its terminal interior producer's ``outputs`` to
    ``payload`` — the sub-construct output-boundary validator requires an internal
    node to actually produce the declared ``output`` type, and a round-tripped
    interior re-synthesizes its own per-node types (agent-spec export does not
    preserve type identity), so the boundary alone would not satisfy it. This is
    the Construct-member analog of forcing an Agent member's ``outputs=payload``
    (a foreign Swarm member never produced a ``goto`` value either) — a best-
    effort structural import, flagged by the caller's mesh-level warning, not a
    behaviorally-faithful foreign runtime. A non-Node terminal item (e.g. a
    nested Construct) is left untouched: it will fail LOUD at the boundary check
    rather than being silently mis-coerced, per the maintainer's fail-loud-over-
    silent default for the interior-already-fixed case.
    """
    sub = _construct_from_subflow(agent, agent.name, from_spec)
    nodes = list(sub.nodes)
    if nodes and isinstance(nodes[-1], Node):
        nodes[-1] = _with_declared_io(nodes[-1], outputs=payload)
    return sub.model_copy(update={"input": payload, "output": payload, "nodes": nodes})


def _reconstruct_swarm_mesh(swarm: Any, from_spec: Callable[[Any], Construct]) -> Construct:
    """Import a foreign pyagentspec ``Swarm`` onto a native Portal peer mesh
    (gap 2, ratification §3a).

    A ``Swarm`` is a top-level ``AgenticComponent`` (NOT a ``Flow`` node), so it
    is dispatched here at the ``from_agent_spec`` entry, not through
    ``_reconstruct_primitive_node``. Each Swarm agent becomes an agent-mode
    member ``Node(inputs={'handoff': Payload}, outputs=Payload) | Portal(to=[peers])``;
    ``first_agent`` is the entry (``nodes[0]``). ``handoff_param``/``handoff_channel``
    are NOT set here -- ``normalize_ir`` is their sole writer (fires in
    ``Construct.__init__``) -- and the assembled mesh is validated by the same
    ``_check_portal_mesh`` a hand-written mesh gets.

    This is a best-effort/warning arm (Core Invariant no-silent-downgrade): the
    synthesized payload is route-centric and the members are name-bound live-LLM
    agents needing a factory at compile.
    """
    agents = _swarm_agents_ordered(swarm)
    payload = _synthesize_swarm_payload(swarm, agents)

    # B3: the entry-only Portal knobs (max_hops/on_exhaust/route) ride the
    # neograph/portal_spec marker on export (_lower_portal_mesh_to_swarm) but
    # were silently dropped on re-import. Read them back the SAME way the sibling
    # reconstructors read their _MARK_*_SPEC markers (e.g. _reconstruct_oracle_group,
    # _reconstruct_loop_item) and apply them to the ENTRY member's Portal only --
    # max_hops/on_exhaust are entry-only per _check_portal_mesh, and route is
    # taken from the entry. Foreign Swarms have no marker -> plain Portal(to=peers).
    portal_spec = (getattr(swarm, "metadata", None) or {}).get(_MARK_PORTAL_SPEC)
    mesh_trigger = _swarm_trigger(swarm)

    members: list[Any] = []
    for idx, agent in enumerate(agents):
        peers = [dst.name for (src, dst) in swarm.relationships if src is agent]
        member: Node | Construct
        member_trigger: Literal["output", "tool"]
        if type(agent).__name__ == SWARM_ENCODING[PortalMemberClass.SUB_CONSTRUCT].spec_class:
            # C1 import: a Flow Swarm member (a neograph Construct exported via
            # _lower_portal_mesh_to_swarm, or any foreign sub-Flow agent)
            # reconstructs to a Construct mesh member, reusing the SAME
            # FlowNode->Construct recursion the bare-FlowNode item path uses.
            # Its boundary I/O is forced to the synthesized uniform payload (by
            # identity) so _check_portal_mesh's uniform-payload rule holds. A
            # Construct member cannot be trigger="tool" (s7zt3.14 validation:
            # tool-trigger requires an agent/act member with a ReAct turn), so it
            # always routes via typed output regardless of the Swarm's HandoffMode.
            member = _flow_member_to_construct(agent, payload, from_spec)
            forced_trigger = SWARM_ENCODING[PortalMemberClass.SUB_CONSTRUCT].import_forced_trigger
            assert forced_trigger is not None  # SUB_CONSTRUCT's row always forces one; table invariant
            member_trigger = forced_trigger
        else:
            # The reserved mesh-channel input key is the literal "handoff" (design
            # §3.3, mirrored in example 28's declarative form and _ir_normalize's
            # sole-writer check); normalize_ir derives handoff_param/handoff_channel.
            member = _node_from_spec_agent(agent.name, agent, None, {"handoff": payload}, payload)
            # Option F (neograph-s7zt3.1): a neograph-exported mesh Agent carries
            # the untranslated ${var} prompt in a neograph/prompt_spec marker (its
            # system_prompt is the translated {{ flat }} wire form) -- prefer it so
            # the round trip recovers the original grammar. Foreign Swarms have no
            # marker and keep the system_prompt as-is.
            prompt_marker = (getattr(agent, "metadata", None) or {}).get(_MARK_PROMPT_SPEC)
            if prompt_marker is not None:
                member = member.model_copy(update={"prompt": prompt_marker["original_text"] or None})
            member_trigger = mesh_trigger
        if idx == 0 and portal_spec is not None:
            portal = Portal(
                to=peers,
                trigger=member_trigger,
                max_hops=portal_spec["max_hops"],
                on_exhaust=portal_spec["on_exhaust"],
                route=portal_spec["route"],
            )
        else:
            portal = Portal(to=peers, trigger=member_trigger)
        members.append(member | portal)

    # Trigger-aware best-effort warning (ACTION ITEM, retargeted now that
    # s7zt3.14's Portal(trigger="tool") exists). pyagentspec Swarms route via
    # handoff/send_message TOOLS called mid-conversation (HandoffMode
    # NEVER/OPTIONAL/ALWAYS). The precise remaining mismatch depends on the mode:
    #   - OPTIONAL/ALWAYS -> Portal(trigger="tool"): neograph NOW represents the
    #     mid-loop optional handoff faithfully (a synthesized transfer_to_<peer>
    #     tool), so the old "OPTIONAL collapses into a must-always-emit-a-goto
    #     contract" caveat NO LONGER applies. The residual downgrade is only that
    #     members are name-bound live-LLM agents needing an LLM factory at compile.
    #   - NEVER -> Portal(trigger="output"): the member routes via a typed 'goto'
    #     final output, so it IS forced to always decide a goto as structured
    #     output (the genuine tool-call-vs-typed-output mismatch that remains).
    if mesh_trigger == "tool":
        detail = (
            "its OPTIONAL/ALWAYS handoff maps to Portal(trigger='tool') (a synthesized "
            "transfer_to_<peer> tool per relationship), so the optional mid-loop handoff is "
            "represented faithfully; the residual downgrade is only that members are name-bound "
            "live-LLM agents requiring an LLM factory at compile"
        )
    else:
        detail = (
            "pyagentspec Swarms route via handoff/send_message TOOLS called mid-conversation, but a "
            "HandoffMode.NEVER member maps to Portal(trigger='output') typed routing — so it is "
            "forced to ALWAYS decide a 'goto' as structured final output (a tool-call-vs-typed-output "
            "mismatch), and the payload is a route-only synthesis (a 'goto' field plus any folded "
            "Swarm/agent data properties)"
        )
    warnings.warn(
        f"Swarm {swarm.name!r} imported onto a native Portal mesh (best-effort): {detail}. This is a "
        "structural downgrade of the Swarm's own runtime, not a lossless import.",
        stacklevel=2,
    )
    return Construct(name=swarm.name, nodes=members)


def _reconstruct_swarm_mesh_with_operator_gates(flow: Any, from_spec: Callable[[Any], Construct]) -> Construct | None:
    """Recognize the Phase 6 mesh-exit pause composite
    (``AgentNode(Swarm)`` -> ``BranchingNode['portal_operator']`` ->
    ``InputMessageNode``) and reconstruct the underlying Portal mesh with
    per-member Operator gates re-attached (neograph-s7zt3.2, inverse of
    ``_agent_spec._lower_portal_mesh_to_swarm``'s gated arm).

    Returns ``None`` if the structure does not match -- the caller falls back to
    treating ``flow`` as a plain (possibly foreign) Flow, never trusting the
    marker blindly (same "confirm the structure, don't trust the marker"
    discipline ``_group_flow_items`` applies to Loop/Operator lookahead).
    """
    agent_nodes = [n for n in flow.nodes if type(n).__name__ == "AgentNode"]
    if len(agent_nodes) != 1 or type(agent_nodes[0].agent).__name__ != "Swarm":
        return None
    agent_node = agent_nodes[0]
    swarm = agent_node.agent

    check = next(
        (n for n in flow.nodes if (n.metadata or {}).get(_MARK_MODIFIER) == "portal_operator"),
        None,
    )
    if check is None or _MARK_PORTAL_OPERATOR_SPEC not in (check.metadata or {}):
        return None

    # Structural confirmation, not marker trust: the AgentNode really leads into
    # this specific check via a real ControlFlowEdge.
    edge_ok = any(
        e.from_node.name == agent_node.name and e.to_node.name == check.name for e in flow.control_flow_connections
    )
    if not edge_ok:
        return None

    base = _reconstruct_swarm_mesh(swarm, from_spec)  # existing helper, unchanged
    gated: dict[str, str] = check.metadata[_MARK_PORTAL_OPERATOR_SPEC]
    # Operator.when is always a plain str (never a callable, unlike Loop.when),
    # so the marker's string passes straight through -- exactly as
    # _reconstruct_operator_item does for the single-node case, no parse_condition.
    # _reconstruct_swarm_mesh only ever yields Node members (Agent -> Node |
    # Portal), but base.nodes is typed list[ConstructItem]; the isinstance guard
    # narrows to Node for the `| Operator` compose (a _BranchNode has no `|`).
    updated_nodes = [
        (member | Operator(when=gated[member.name])) if isinstance(member, Node) and member.name in gated else member
        for member in base.nodes
    ]
    return base.model_copy(update={"nodes": updated_nodes})
