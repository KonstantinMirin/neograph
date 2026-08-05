"""Spec loader -- YAML/JSON pipeline spec -> Construct IR.

    from neograph.loader import load_spec
    construct = load_spec("pipeline.yaml")
    graph = compile(construct)
    result = run(graph, input={...})

The spec is parsed into a typed ``Spec`` Pydantic model from
``_spec_schema``; typos and unknown fields raise ``ConfigurationError``
at load time. Types are resolved from a project surface or via
pre-registered entries.
"""

from __future__ import annotations

# --- names loader.py imported and RE-EXPORTED before the split; the moved
# --- spec-loader half were their only local consumers here.
import json  # noqa: E402,F401
import warnings
from pathlib import Path  # noqa: E402,F401
from typing import Any, Literal

import structlog
import yaml  # type: ignore[import-untyped]  # noqa: E402,F401
from pydantic import (
    BaseModel,
    ValidationError,  # noqa: E402,F401
    create_model,
)

from neograph._agent_spec_markers import (
    _MARK_GROUP_ID,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    _MARK_PORTAL_OPERATOR_SPEC,
    _MARK_PORTAL_SPEC,
    _MARK_PROMPT_SPEC,
    import_pyagentspec,
)
from neograph._normalize import (
    _with_declared_io,
    primary_output_field,  # noqa: E402,F401
)

# --- extracted cluster (neograph-3ffdg.4), re-exported so existing
# --- `from neograph.loader import ...` call sites keep resolving unchanged.
from neograph._spec_loader import (  # noqa: E402,F401
    MAX_SPEC_SIZE,
    _apply_modifiers,
    _build_construct,
    _build_node,
    _build_sub_construct,
    _parse_input,
    _resolve_tool,
    _validate_spec,
    load_spec,
)
from neograph._spec_schema import (  # noqa: E402,F401
    ConstructSpec,
    NodeSpec,
    Spec,
    ToolSpec,
)
from neograph._state_keys import StateKeys  # noqa: E402,F401
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import (
    COMBO_DECOMPOSITION,
    Operator,
    Portal,
    PrimaryShape,
    combo_for_modifier_names,
    is_each_oracle_fused,
)
from neograph.naming import field_name_for  # noqa: E402,F401
from neograph.node import Node
from neograph.spec_types import (
    load_project_types,  # noqa: E402,F401
)

log = structlog.get_logger()


def _import_agent_spec_import_classes() -> Any:
    """Function-local import of pyagentspec's Flow/node classes for import.

    Thin wrapper over ``import_pyagentspec`` -- only calling
    ``from_agent_spec()`` pulls in the optional ``[agent-spec]`` extra.
    """
    return import_pyagentspec("pyagentspec.flows.nodes", found="ImportError on pyagentspec.flows.nodes")


# Per-family endpoint attribute names for the client-initiated remote-agent


def _trailing_operator(nodes: list[Any], j: int, primary_spec: Any, flow: Any) -> Any | None:
    """Variable-length lookahead for the Operator pause composite that may
    follow ANY primary group (bare / Each / Oracle / Loop). Returns the check
    ``BranchingNode`` on a match -- the caller then consumes TWO nodes (check +
    pause) -- or None.

    ONE helper, called from every branch of the recognition walk, because
    ``_lower_operator`` emits the SAME three-part composite regardless of what
    it wraps. Structural confirmation, never marker trust: the marker must be
    backed by a real ``primary -> check`` edge AND a real ``check --pause-->``
    edge, or the shape is not an Operator and the nodes import as primitives.
    """
    if j + 1 >= len(nodes):
        return None
    check = nodes[j]
    if (check.metadata or {}).get(_MARK_MODIFIER) != "operator":
        return None
    if not any(
        e.from_node.name == primary_spec.name and e.to_node.name == check.name for e in flow.control_flow_connections
    ):
        return None
    pause = nodes[j + 1]
    if not any(
        e.from_node.name == check.name and e.from_branch == "pause" and e.to_node.name == pause.name
        for e in flow.control_flow_connections
    ):
        return None
    return check


def _group_flow_items(flow: Any) -> list[tuple[frozenset[str], dict[str, Any]]]:
    """Walk ``Flow.nodes`` in order, skipping Start/End sentinels, and group
    contiguous nodes into the shapes ``to_agent_spec`` emits.

    RECOGNIZE, then CLASSIFY (the import-side mirror of the export dispatch):
    each group yields the frozenset of modifier NAMES its structure encodes,
    which ``from_agent_spec`` hands to ``combo_for_modifier_names`` and
    dispatches on ``COMBO_DECOMPOSITION``. Emitting a fixed ``kind`` string per
    shape does not survive composition -- there are 12 combos and only five
    primary shapes, so a string enumeration would have to re-derive the
    decomposition the table already owns.

    The payload dict carries the recognized spec nodes: ``primary`` (bare),
    ``group`` (Oracle run), ``map_node`` (Each), ``body``/``check`` (Loop), and
    ``operator_check`` whenever the trailing pause composite was recognized.
    """
    nodes = flow.nodes
    n = len(nodes)
    items: list[tuple[frozenset[str], dict[str, Any]]] = []
    i = 0
    while i < n:
        node = nodes[i]
        cls_name = type(node).__name__
        if cls_name in ("StartNode", "EndNode"):
            i += 1
            continue

        metadata = node.metadata or {}
        modifier = metadata.get(_MARK_MODIFIER)

        names: set[str] = set()
        payload: dict[str, Any] = {}

        if modifier == "oracle":
            group_id = metadata[_MARK_GROUP_ID]
            group = [node]
            j = i + 1
            while j < n and ((nodes[j].metadata or {}).get(_MARK_GROUP_ID) == group_id):
                group.append(nodes[j])
                j += 1
            names.add("oracle")
            payload["group"] = group
            # The merge node (last in the run) is the group's control-flow
            # identity -- the node a trailing Operator check attaches to.
            primary_spec = group[-1]

        elif modifier == "each":
            names.add("each")
            payload["map_node"] = node
            if _subflow_oracle_group(node) is not None:
                names.add("oracle")  # Each x Oracle fusion
            primary_spec = node
            j = i + 1

        elif modifier in ("loop", "operator"):
            # A floating check node with no preceding body (the lookahead
            # below always consumes body+check together) means the marker
            # doesn't match the actual structure -- fall back to primitive.
            payload["primary"] = node
            primary_spec = node
            j = i + 1

        else:
            # A bare node MAY be the body of a following Loop check -- peek ahead
            # and confirm the control-flow edge actually connects them (per the
            # Core Invariant: never trust a marker without checking the structure
            # it claims to describe). A trailing OPERATOR is not special-cased
            # here: it is the shared postlude below, exactly as on the export side.
            payload["primary"] = node
            primary_spec = node
            j = i + 1
            nxt = nodes[i + 1] if i + 1 < n else None
            if nxt is not None and (nxt.metadata or {}).get(_MARK_MODIFIER) == "loop":
                edge_to_nxt = any(
                    e.from_node.name == node.name and e.to_node.name == nxt.name for e in flow.control_flow_connections
                )
                back_edge = any(
                    e.from_node.name == nxt.name and e.from_branch == "continue" and e.to_node.name == node.name
                    for e in flow.control_flow_connections
                )
                if edge_to_nxt and back_edge:
                    names.add("loop")
                    payload.pop("primary")
                    payload["body"] = node
                    payload["check"] = nxt
                    primary_spec = nxt
                    j = i + 2

        # ONE shared trailing-Operator lookahead for every primary group above.
        check = _trailing_operator(nodes, j, primary_spec, flow)
        if check is not None:
            names.add("operator")
            payload["operator_check"] = check
            j += 2

        items.append((frozenset(names), payload))
        i = j

    return items


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
    return "tool" if value in ("optional", "always") else "output"


def _flow_member_to_construct(agent: Any, payload: type[BaseModel]) -> Construct:
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
    sub = _construct_from_subflow(agent, agent.name, from_agent_spec)
    nodes = list(sub.nodes)
    if nodes and isinstance(nodes[-1], Node):
        nodes[-1] = _with_declared_io(nodes[-1], outputs=payload)
    return sub.model_copy(update={"input": payload, "output": payload, "nodes": nodes})


def _reconstruct_swarm_mesh(swarm: Any) -> Construct:
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
        if type(agent).__name__ == "Flow":
            # C1 import: a Flow Swarm member (a neograph Construct exported via
            # _lower_portal_mesh_to_swarm, or any foreign sub-Flow agent)
            # reconstructs to a Construct mesh member, reusing the SAME
            # FlowNode->Construct recursion the bare-FlowNode item path uses.
            # Its boundary I/O is forced to the synthesized uniform payload (by
            # identity) so _check_portal_mesh's uniform-payload rule holds. A
            # Construct member cannot be trigger="tool" (s7zt3.14 validation:
            # tool-trigger requires an agent/act member with a ReAct turn), so it
            # always routes via typed output regardless of the Swarm's HandoffMode.
            member = _flow_member_to_construct(agent, payload)
            member_trigger = "output"
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


def _reconstruct_swarm_mesh_with_operator_gates(flow: Any) -> Construct | None:
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

    base = _reconstruct_swarm_mesh(swarm)  # existing helper, unchanged
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


def from_agent_spec(flow: Any) -> Construct:
    """Import an Open Agent Spec ``Flow`` into a neograph ``Construct`` --
    the inverse of ``to_agent_spec()``.

    Sibling of ``load_spec()``: import-guarded (mirrors
    ``_agent_spec._import_agent_spec_flow_classes()``) so ``src/neograph``
    core stays Agent-Spec-free by default.

    Per-group ``neograph/*_spec`` metadata markers (emitted by
    ``to_agent_spec``) are read and STRUCTURALLY VALIDATED against the
    actual primitives around them (never blindly trusted) to losslessly
    reconstruct Oracle/Each/Loop/Operator. A Flow with no markers (a
    foreign/third-party Agent Spec) imports as plain primitives. There is
    no whole-pipeline ``Flow.metadata['neograph/source']`` blob to read --
    fidelity rides only on the per-group markers, matching what
    ``to_agent_spec`` actually emits.

    Agent/act (``AgentNode``) reconstruction is implemented per neograph-aa5gq:
    a ``neograph/agent_spec`` marker reconstructs the exact agent/act Node
    losslessly; a foreign ``Agent`` imports best-effort as agent-mode; a
    client-initiated ``RemoteAgent``/``A2AAgent``/``OciAgent`` imports
    best-effort as a name-bound scripted Node WITH a warning. A top-level
    ``Swarm`` imports onto a native Portal mesh (also best-effort, with a
    warning). Orchestrator-side surfaces (e.g. a ServerTool-as-agent) still
    fail loud rather than silently downgrade.
    """
    _import_agent_spec_import_classes()

    # A Swarm is a top-level AgenticComponent (NOT a Flow node), so it never
    # reaches the Flow.nodes walk below -- dispatch it onto a Portal mesh here.
    if type(flow).__name__ == "Swarm":
        return _reconstruct_swarm_mesh(flow)

    # A PORTAL_OPERATOR mesh export is a Flow wrapping a Swarm (the mesh-exit
    # pause composite, neograph-s7zt3.2), not a bare Swarm -- recognize that
    # whole-Flow shape here, before the per-item _group_flow_items walk (a
    # top-level dispatch decision like the bare-Swarm check above, not a
    # per-item concern). Returns None if the structure does not match, so a
    # foreign Flow falls through to the generic walk.
    gated_mesh = _reconstruct_swarm_mesh_with_operator_gates(flow)
    if gated_mesh is not None:
        return gated_mesh

    output_types: dict[str, Any] = {}
    pipeline_items: list[Any] = []

    # RECOGNIZE -> CLASSIFY -> dispatch on the DECOMPOSED shape: the exact mirror
    # of _agent_spec._lower_construct_item's export dispatch. The walk recognizes
    # which modifier NAMES a node grouping encodes; combo_for_modifier_names maps
    # that to a ModifierCombo via the single _COMBO_MAP; COMBO_DECOMPOSITION says
    # what the combo means. No local re-derivation of combo semantics lives here.
    for names, payload in _group_flow_items(flow):
        combo = combo_for_modifier_names(names, context=flow.name)
        decomp = COMBO_DECOMPOSITION[combo]

        item: Any
        # The fusion split runs BEFORE the shape match, exactly as on the export
        # side, and asks the ONE shared presence predicate rather than open-coding
        # it -- see neograph-c265k. ``names`` is the recognized modifier-NAME set (the
        # loader recognizes structure, so it has no Modifier instances to hand
        # over); ``dict.fromkeys(names, True)`` presents it in the {name: <present>}
        # mapping shape ``classify_modifiers`` returns. The sentinel must be
        # non-None: the predicate tests ``mods.get(k) is not None``, so a
        # None-valued key would read as ABSENT and silently un-fuse the import.
        if is_each_oracle_fused(dict.fromkeys(names, True)):
            # Fused Each x Oracle -- the MapNode's sub-flow IS an Oracle group.
            item = _reconstruct_fused_each_oracle_node(payload["map_node"], output_types, from_agent_spec)
        else:
            match decomp.primary:
                case PrimaryShape.ORACLE:
                    reconstructed = _reconstruct_oracle_group(payload["group"], flow, output_types, from_agent_spec)
                    if reconstructed is None:
                        # Stale marker -- fall back to importing every node in the
                        # group as a bare primitive (per the Core Invariant: never
                        # silently reconstruct a modifier that diverges from the
                        # actual structure). A trailing Operator is meaningless
                        # against a group that did not reconstruct, so skip it too.
                        for spec_node in payload["group"]:
                            pipeline_items.append(_reconstruct_primitive_node(spec_node, flow, output_types))
                        continue
                    item = reconstructed

                case PrimaryShape.EACH:
                    item = _reconstruct_each_node(payload["map_node"], flow, output_types, from_agent_spec)

                case PrimaryShape.LOOP:
                    item = _reconstruct_loop_item(
                        payload["body"], payload["check"], flow, output_types, from_agent_spec
                    )

                case PrimaryShape.BARE:
                    spec_node = payload["primary"]
                    if type(spec_node).__name__ == "FlowNode":
                        item = _reconstruct_item_body(spec_node, flow, output_types, from_agent_spec)
                    elif decomp.has_operator:
                        # External inputs land on the PRIMARY node, not the
                        # property-less check BranchingNode -- mirror
                        # to_agent_spec's input_targets routing for BARE+Operator.
                        item = _reconstruct_operator_primary(spec_node, flow, output_types, from_agent_spec)
                    else:
                        item = _reconstruct_primitive_node(spec_node, flow, output_types)

                case PrimaryShape.PORTAL:  # pragma: no cover - a Portal mesh never reaches this walk
                    raise ConfigurationError.build(
                        f"Flow {flow.name!r} encodes a Portal at item level — no primitive import",
                        expected="a peer-mode Portal mesh, imported from a Swarm before this walk",
                        found=combo.name,
                    )

        # ONE shared Operator postlude, mirroring the export side's.
        if decomp.has_operator:
            item = item | Operator(when=payload["operator_check"].metadata[_MARK_OPERATOR_SPEC]["when"])

        pipeline_items.append(item)

    return Construct(name=flow.name, nodes=pipeline_items)


# -- Parsing -----------------------------------------------------------------


# -- Builder -----------------------------------------------------------------


# Re-exports: the Agent Spec import cluster moved out (neograph-s7zt3.11) so
# loader.py could come back under its ratchet ceiling. Kept here so existing
# `from neograph.loader import ...` call sites resolve unchanged.
# noqa F401 is REQUIRED -- without it ruff --fix strips these as unused.
from neograph._agent_spec_group_import import (  # noqa: E402,F401
    _construct_from_subflow,
    _oracle_kwargs,
    _reconstruct_each_node,
    _reconstruct_fused_each_oracle_node,
    _reconstruct_item_body,
    _reconstruct_loop_item,
    _reconstruct_operator_primary,
    _reconstruct_oracle_group,
    _subflow_inner_nodes,
    _subflow_oracle_group,
)

# --- Names loader.py imported and RE-EXPORTED before the neograph-s7zt3.11 split.
# --- The moved cluster was their only local consumer, so `ruff --fix` strips them
# --- as unused unless they carry F401. Verified against the pre-split surface
# --- (defined names UNION imported names), not by eye.
from neograph._agent_spec_markers import (  # noqa: E402,F401
    _MARK_AGENT_SPEC,
    _MARK_EACH_SPEC,
    _MARK_LOOP_SPEC,
    _MARK_ORACLE_SPEC,
    _MARK_TOOL_SPEC,
)
from neograph._agent_spec_node_import import (  # noqa: E402,F401  # noqa: E402,F401
    _REMOTE_AGENT_ENDPOINT_ATTRS,
    _agent_spec_props_to_type,
    _augment_inputs_from_prompt_marker,
    _dict_form_inputs_from_props,
    _inputs_from_data_edges,
    _node_from_spec_agent,
    _reconstruct_agent_node,
    _reconstruct_primitive_node,
    _tools_from_foreign_agent,
    _tools_from_marker,
)
from neograph._normalize import normalize_outputs  # noqa: E402,F401
from neograph.conditions import parse_condition  # noqa: E402,F401
from neograph.modifiers import Each, Loop, Oracle  # noqa: E402,F401
from neograph.spec_types import (  # noqa: E402,F401
    _import_agent_spec_property_classes,
    _structural_type_name,
    agent_spec_properties_to_types,
    lookup_type,
)
from neograph.tool import Tool  # noqa: E402,F401
