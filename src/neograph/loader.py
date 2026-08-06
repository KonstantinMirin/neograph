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
import warnings  # noqa: E402,F401
from pathlib import Path  # noqa: E402,F401
from typing import Any, Literal  # noqa: E402,F401

import structlog
import yaml  # type: ignore[import-untyped]  # noqa: E402,F401
from pydantic import (
    BaseModel,  # noqa: E402,F401
    ValidationError,  # noqa: E402,F401
    create_model,  # noqa: E402,F401
)

from neograph._agent_spec_markers import (
    _MARK_GROUP_ID,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    _MARK_PORTAL_OPERATOR_SPEC,  # noqa: E402,F401
    _MARK_PORTAL_SPEC,  # noqa: E402,F401
    _MARK_PROMPT_SPEC,  # noqa: E402,F401
    import_pyagentspec,
)
from neograph._normalize import (
    _with_declared_io,  # noqa: E402,F401
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
    Portal,  # noqa: E402,F401
    PrimaryShape,
    combo_for_modifier_names,
)
from neograph.naming import field_name_for  # noqa: E402,F401
from neograph.node import Node  # noqa: E402,F401
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
        return _reconstruct_swarm_mesh(flow, from_agent_spec)

    # A PORTAL_OPERATOR mesh export is a Flow wrapping a Swarm (the mesh-exit
    # pause composite, neograph-s7zt3.2), not a bare Swarm -- recognize that
    # whole-Flow shape here, before the per-item _group_flow_items walk (a
    # top-level dispatch decision like the bare-Swarm check above, not a
    # per-item concern). Returns None if the structure does not match, so a
    # foreign Flow falls through to the generic walk.
    gated_mesh = _reconstruct_swarm_mesh_with_operator_gates(flow, from_agent_spec)
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
        # side, reading decomp.fused -- the table's own answer, needing no
        # modifier instances to ask the question of (this loader recognizes
        # structure, not instances, so it never had any to hand over).
        if decomp.fused:
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

# Re-exports: the Swarm-import cluster moved out (neograph-jtawq.10) so
# loader.py could come back under the plain file-size cap. Kept here so
# existing `from neograph.loader import ...` call sites resolve unchanged.
# noqa F401 is REQUIRED -- without it ruff --fix strips these as unused. Also
# the site `from_agent_spec` (defined above) calls _reconstruct_swarm_mesh and
# _reconstruct_swarm_mesh_with_operator_gates by name -- this works because
# Python resolves module-global names at CALL time, not at function-definition
# time, the same pattern the neograph-s7zt3.11 block above already relies on.
from neograph._agent_spec_swarm_import import (  # noqa: E402,F401
    _flow_member_to_construct,
    _reconstruct_swarm_mesh,
    _reconstruct_swarm_mesh_with_operator_gates,
    _swarm_agents_ordered,
    _swarm_trigger,
    _synthesize_swarm_payload,
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
