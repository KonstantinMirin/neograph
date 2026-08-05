"""State bus compiler — auto-generates Pydantic state from Node I/O union.

Each Construct gets its own state model with exactly the fields its Nodes need.
No monolithic state that grows with every derivation type.
"""

from __future__ import annotations

import warnings
from typing import Annotated, Any

import structlog
from pydantic import BaseModel, create_model

from neograph._ir_branch import _BranchNode
from neograph._ir_protocols import ConstructItem
from neograph.construct import Construct
from neograph.errors import CompileError
from neograph.naming import field_name_for, output_field_name
from neograph.spec_types import lookup_type

log = structlog.get_logger()
# --- typing names state.py imported and RE-EXPORTED before the split; the
# --- fingerprint cluster was their only local consumer here.
from typing import assert_never, get_args, get_origin  # noqa: E402,F401

from neograph._normalize import _declared_output, normalize_outputs

# --- extracted clusters (neograph-3ffdg.14), re-exported so existing
# --- `from neograph.state import ...` call sites keep resolving unchanged.
from neograph._schema_fingerprint import (  # noqa: E402,F401
    _type_signature,
    compute_node_fingerprints,
    compute_schema_fingerprint,
)
from neograph._state_keys import StateKeys
from neograph._state_reducers import (  # noqa: E402,F401
    _append_loop_result,
    _concat_reducer,
    _last_write_wins,
    _merge_dicts,
)
from neograph.modifiers import (
    COMBO_DECOMPOSITION,
    SUB_CONSTRUCT_UNSUPPORTED_COMBOS,
    EachFailure,
    PrimaryShape,
    _group_portal_members,
    classify_modifiers,
    primary_shape,
)
from neograph.node import Node


def compile_state_model(
    construct: Construct,
    *,
    context_types: dict[str, type] | None = None,
) -> type[BaseModel]:
    """Generate a Pydantic state model from the union of Node I/O fields.

    Each Node's output becomes a state field. Fan-out nodes get dict reducers.
    The resulting model is used as the LangGraph StateGraph schema.

    Args:
        context_types: When compiling a subconstruct, the parent passes concrete
            types for context fields (instead of Any). Keys are field_name_for'd
            context names, values are the parent's output types.
    """
    fields: dict[str, Any] = {}

    nodes_only = [n for n in construct.nodes if isinstance(n, Node)]
    sub_constructs = [n for n in construct.nodes if isinstance(n, Construct)]
    branch_nodes = [n for n in construct.nodes if isinstance(n, _BranchNode)]

    # Detect field-name collisions from hyphen/underscore normalization.
    # Two nodes "my-node" and "my_node" both map to state field "my_node",
    # which would silently share loop counters, reducers, etc.
    seen_fields: dict[str, str] = {}  # field_name → original node name
    for item in nodes_only + sub_constructs:
        field_name = field_name_for(item.name)
        if field_name in seen_fields:
            raise CompileError.build(
                "node name collision",
                expected="unique state field names",
                found=f"'{item.name}' and '{seen_fields[field_name]}' both map to state field '{field_name}'",
                hint="rename one of them so the normalized field names differ",
            )
        seen_fields[field_name] = item.name

    for node in nodes_only:
        _add_output_field(node, fields)
        _add_agent_channels(node, fields)

    # Branch arm nodes: add state fields for nodes inside branch arms.
    # Arms can contain both Nodes and Constructs (e.g., self.loop() in
    # ForwardConstruct produces a Construct in the branch arm).
    for branch in branch_nodes:
        meta = branch._neo_branch_meta
        for arm_item in meta.true_arm_nodes + meta.false_arm_nodes:
            if isinstance(arm_item, Construct):
                # Construct in branch arm — same handling as sub-constructs
                if arm_item.output is None:
                    continue
                field_name = field_name_for(arm_item.name)
                arm_combo, _ = classify_modifiers(arm_item)
                if COMBO_DECOMPOSITION[arm_combo].primary is PrimaryShape.LOOP:
                    fields[field_name] = (
                        Annotated[list[arm_item.output], _append_loop_result],  # type: ignore[name-defined]
                        [],
                    )
                    fields[StateKeys.loop_count(field_name)] = (int, 0)
                else:
                    fields[field_name] = (arm_item.output | None, None)
            else:
                _add_output_field(arm_item, fields)

    # Sub-constructs: handle modifiers same as nodes
    for sub in sub_constructs:
        if sub.output is None:
            raise CompileError.build(
                "sub-construct has no output type",
                hint="declare output=SomeModel on the sub-construct",
                construct=sub.name,
            )
        field_name = field_name_for(sub.name)

        combo, mods = classify_modifiers(sub)
        if combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS:
            # Each x Oracle on a Construct. Gate checked FIRST, mirroring
            # _add_subgraph's order — but deliberately NON-raising: state building
            # runs at compiler.py:240, BEFORE _add_subgraph at :308, so raising
            # here would pre-empt and replace the user-visible CompileError that
            # _add_subgraph owns. Build a plain field and let the compiler raise.
            # Pinned by tests/check_fixtures/should_fail/subconstruct_each_oracle_fusion.py.
            fields[field_name] = (sub.output | None, None)
            continue
        match COMBO_DECOMPOSITION[combo].primary:
            case PrimaryShape.ORACLE:
                # Oracle on Construct: collector + consumer field
                collector_field = StateKeys.oracle_collector(field_name)
                fields[collector_field] = (
                    Annotated[list[sub.output], _concat_reducer],  # type: ignore[name-defined]
                    [],
                )
                fields[field_name] = (sub.output | None, None)  # type: ignore[name-defined]
            case PrimaryShape.EACH:
                # Each on Construct: dict field. Under on_error='collect' the
                # barrier may hold a typed EachFailure per thrown item, so the
                # value type widens to accept it (default 'raise' unchanged).
                # The Each x Oracle fusion never reaches here — it left via the
                # SUB_CONSTRUCT_UNSUPPORTED_COMBOS gate above — so mods["each"]
                # is the plain-Each modifier.
                each_mod = mods["each"]
                value_type: Any = (
                    sub.output | EachFailure  # type: ignore[name-defined]
                    if each_mod.on_error == "collect"
                    else sub.output
                )
                field_type = dict[str, value_type] | None  # type: ignore[name-defined,valid-type]
                fields[field_name] = (
                    Annotated[field_type, _merge_dicts],
                    None,
                )
            case PrimaryShape.LOOP:
                # Loop on Construct: append-list + iteration counter
                fields[field_name] = (
                    Annotated[list[sub.output], _append_loop_result],  # type: ignore[name-defined]
                    [],
                )
                fields[StateKeys.loop_count(field_name)] = (int, 0)
            case PrimaryShape.BARE:
                fields[field_name] = (sub.output | None, None)
            case PrimaryShape.PORTAL:
                # LIVE, not defensive: a Portal-carrying Construct is a first-class
                # mesh member and may even be the mesh ENTRY (neograph-s7zt3.5;
                # compiler.py admits a Construct at the mesh-entry detection site).
                # A mesh member writes its own output plainly, mirroring
                # _add_single_output_field's BARE/PORTAL arm. Pinned by
                # tests/test_portal_construct_entry.py. Do NOT fold this into a
                # "defensively unreachable" arm — the field it builds is load-bearing.
                fields[field_name] = (sub.output | None, None)
            case _ as unreachable:
                assert_never(unreachable)

    # Oracle support: generator ID + optional model override passed via state
    all_items = nodes_only + sub_constructs
    has_any_oracle = False
    has_any_each = False
    for item in all_items:
        # Modifier PRESENCE, deliberately NOT a decomposition dispatch: an
        # Each x Oracle node needs BOTH the Oracle fields and the Each item slot,
        # but EACH_ORACLE decomposes to primary=EACH — so asking the table for a
        # shape here would silently drop ORACLE_GEN_ID/ORACLE_MODEL for every
        # fused node. classify_modifiers inserts a key only when the slot is set.
        _item_combo, item_mods = classify_modifiers(item)
        if "oracle" in item_mods:
            has_any_oracle = True
        if "each" in item_mods:
            has_any_each = True
    if has_any_oracle:
        fields[StateKeys.ORACLE_GEN_ID] = (str | None, None)
        fields[StateKeys.ORACLE_MODEL] = (str | None, None)
    if has_any_each:
        fields[StateKeys.EACH_ITEM] = (Any, None)

    # Loop support: iteration counter per looped node
    for n in nodes_only:
        n_combo, n_mods = classify_modifiers(n)
        if COMBO_DECOMPOSITION[n_combo].primary is PrimaryShape.LOOP:
            field_name = field_name_for(n.name)
            fields[StateKeys.loop_count(field_name)] = (int, 0)
            loop = n_mods["loop"]

    # Portal support: per-mesh hop counter + shared payload channel, keyed off
    # EACH NAMED GROUP's own ENTRY (the first member of that group in node
    # order — neograph-fefar extends design §3.1 from one mesh per level to
    # one mesh per (level, name) pair; assembly validation guarantees one
    # contiguous mesh per group). Both are neo_-prefixed → excluded from the
    # schema fingerprint (member OUTPUT fields carry the fingerprint). The
    # channel/counter are runtime-inert until T2 lowering reads them. Grouped
    # via the SAME shared helper (_group_portal_members) the validator and IR
    # normalizer use — never a re-derived inline grouping.
    def _is_dispatch(n: ConstructItem) -> bool:
        km = n.modifier_set.portal
        return km is not None and km.is_dispatch

    # PEER-mode members only: a dispatch node (route="decide") is NOT a mesh member
    # — it has no hop counter / mesh channel; it gets a {field}_dispatch field below.
    #
    # Filter over construct.nodes DIRECTLY, preserving position (neograph-s7zt3.5):
    # a Portal member may be a sub-Construct — including the mesh ENTRY. Do NOT
    # source from `nodes_only` (excludes a Construct entry) nor from
    # `nodes_only + sub_constructs` (concatenation reorders relative to
    # construct.nodes, so _group_portal_members — which treats group_members[0] as
    # the entry — would pick a Node peer as the entry and mis-key the channel).
    portal_members: list[ConstructItem] = [
        m for m in construct.nodes if primary_shape(m) is PrimaryShape.PORTAL and not _is_dispatch(m)
    ]
    for _group_name, group_members in _group_portal_members(portal_members).items():
        entry = group_members[0]
        entry_field = field_name_for(entry.name)
        # Single-type by assembly validation (dict-form members rejected); typed
        # Any so the `| None` field spec matches the sibling arms' pattern.
        payload: Any = _declared_output(entry)
        fields[StateKeys.handoff_hops(entry_field)] = (int, 0)
        fields[StateKeys.handoff_payload(entry_field)] = (payload | None, None)

    # Portal+Operator approval gate: each Operator-guarded
    # member gets its OWN proposed-target field (unlike the mesh-entry-keyed
    # hop counter/channel above) -- the approval node reads it to know which
    # peer to route to on approval.
    for m in portal_members:
        if m.modifier_set.operator is not None:
            member_field = field_name_for(m.name)
            fields[StateKeys.portal_proposed_target(member_field)] = (str | None, None)

    # Portal DISPATCH support (design §4.2): a route="decide" node writes the
    # dispatched flow's typed result to a regular (fingerprinted, NON-neo_-prefixed)
    # field `{field_name}_dispatch` — an output-contract change correctly
    # invalidates checkpoints. The node's OWN output (the emitted spec/input model)
    # is written to its plain output field by the PORTAL arm in
    # `_add_single_output_field`; this is the SEPARATE dispatch-result field.
    for n in nodes_only:
        if _is_dispatch(n):
            km = n.modifier_set.portal
            assert km is not None  # _is_dispatch guarantees it
            out_spec = km.output
            assert out_spec is not None  # dispatch-mode invariant (T1 validation)
            dispatch_field = output_field_name(field_name_for(n.name), "dispatch")
            resolved = lookup_type(out_spec) if isinstance(out_spec, str) else out_spec
            fields[dispatch_field] = (resolved | None, None)
            if km.on_invalid == "route_to_error":
                fields[StateKeys.dispatch_error(field_name_for(n.name))] = (str | None, None)

    # Subgraph input port — when this Construct declares an input type
    if construct.input is not None:
        fields[StateKeys.SUBGRAPH_INPUT] = (construct.input | None, None)

    # Context fields — forwarded from parent state for nodes that declare context=
    # When context_types is provided (subconstruct compilation), use the concrete
    # type from the parent instead of Any. This ensures the msgpack allowlist
    # includes the context field types for checkpoint serialization.
    _ctx_types = context_types or {}
    for n in nodes_only:
        if n.context:
            for ctx_name in n.context:
                ctx_field = field_name_for(ctx_name)
                if ctx_field not in fields:
                    ctx_type = _ctx_types.get(ctx_field, Any)
                    fields[ctx_field] = (ctx_type if ctx_type is not Any else Any, None)
    # Also check branch arm nodes (skip Constructs — they handle context internally)
    for branch in branch_nodes:
        meta = branch._neo_branch_meta
        for arm_node in meta.true_arm_nodes + meta.false_arm_nodes:
            if isinstance(arm_node, Construct):
                continue
            if arm_node.context:
                for ctx_name in arm_node.context:
                    ctx_field = field_name_for(ctx_name)
                    if ctx_field not in fields:
                        ctx_type = _ctx_types.get(ctx_field, Any)
                        fields[ctx_field] = (ctx_type if ctx_type is not Any else Any, None)

    # Framework fields — always present
    # node_id and project_root have defaults so consumers can omit them
    # in run(input=...); they're still accessible via config["configurable"]
    # for node functions that need pipeline metadata.
    fields[StateKeys.NODE_ID] = (str, "")
    fields[StateKeys.PROJECT_ROOT] = (str, "")
    fields[StateKeys.HUMAN_FEEDBACK] = (dict[str, Any] | None, None)
    fields[StateKeys.SCHEMA_FINGERPRINT] = (str, "")
    fields[StateKeys.NODE_FINGERPRINTS] = (dict[str, str], {})

    return create_model(f"{construct.name}State", **fields)


def build_output_schema_model(state_model: type[BaseModel]) -> type[BaseModel]:
    """Build the StateGraph ``output_schema``: every state field NOT ``neo_``-prefixed.

    Declared at compile time (``StateGraph(state_model, output_schema=...)``) so the
    ENGINE itself filters framework plumbing out of ``invoke``/``ainvoke`` results —
    replacing the hand-rolled ``_strip_internals`` wrap the runner and sub-construct
    exits used to carry (neograph-pjqe: declare, don't wrap). The filter is the
    ``neo_`` prefix, mirroring ``_strip_internals`` EXACTLY: the three non-``neo_``
    framework-injected fields (``node_id``/``project_root``/``human_feedback``) still
    surface, so the user-visible contract is unchanged; only enforcement moves from a
    runtime wrapper we own to a compile-time declaration the engine honours.

    Field annotations (including reducer ``Annotated`` metadata) are preserved via
    ``rebuild_annotation`` so the output channels match the state channels exactly.
    See docs/design/langgraph-output-schema-research-2026-07-03.md (R1/R3).
    """
    fields: dict[str, Any] = {
        name: (finfo.rebuild_annotation(), finfo)
        for name, finfo in state_model.model_fields.items()
        if not name.startswith(StateKeys.FRAMEWORK_PREFIX)
    }
    # A user node named e.g. ``validate`` produces a state field that shadows a
    # BaseModel attribute; Pydantic already warned once when ``state_model`` was
    # built. This synthesized Output model mirrors the same fields, so it would
    # re-emit the identical warning — a duplicate the user can do nothing about.
    # Suppress ONLY the framework copy here; the user-facing original still fires
    # on their own state model. See neograph-tj53.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Field name .* shadows an attribute in parent .*",
            category=UserWarning,
        )
        return create_model(f"{state_model.__name__}Output", **fields)


def _add_agent_channels(node: Node, fields: dict[str, Any]) -> None:
    """Add the agent-cycle state channels for an agent/act node.

    These carry the ReAct loop's per-turn state — message history, tool_log,
    resource manifest, and budget/iteration counters — so the inline agent-cycle
    expander (``_wiring._add_agent_cycle``) can make every turn a checkpointed
    superstep. All are ``neo_``-prefixed, so ``_strip_internals`` removes them
    from returned state and ``compute_schema_fingerprint`` excludes them.

    Only agent/act nodes get channels; think/scripted/raw nodes never enter a
    ReAct loop.
    """
    from langgraph.graph.message import add_messages

    if node.mode not in ("agent", "act"):
        return

    field_name = field_name_for(node.name)
    fields[StateKeys.agent_messages(field_name)] = (Annotated[list, add_messages], [])
    fields[StateKeys.agent_tool_log(field_name)] = (Annotated[list, _concat_reducer], [])
    fields[StateKeys.resource_manifest(field_name)] = (Annotated[list, _concat_reducer], [])
    fields[StateKeys.agent_budget(field_name)] = (dict | None, None)


def _add_output_field(node: Node, fields: dict[str, Any]) -> None:
    """Add a node's output type(s) as field(s) on the state model.

    When outputs is a dict (multi-output), creates one field per key:
    ``{node_name}_{output_key}``. Each/Oracle modifiers apply per key.
    When outputs is a single type (backward compat), creates ``{node_name}``.
    """
    if node.outputs is None:
        raise CompileError.build(
            "node has no output type",
            hint="every node must declare outputs=SomeModel",
            node=node.name,
        )

    field_name = field_name_for(node.name)
    no = normalize_outputs(node.outputs)

    # Dict-form outputs: one state field per key (neograph-1bp.2).
    if no.is_dict_form:
        combo, _ = classify_modifiers(node)
        match COMBO_DECOMPOSITION[combo].primary:
            case PrimaryShape.EACH if COMBO_DECOMPOSITION[combo].fused:
                # Each×Oracle fusion + dict-form: tagged collector + dict output
                # per key. Same as single-type fusion but per-key.
                #
                # This body must NOT be folded into the plain-Each delegation
                # below: _add_single_output_field re-classifies and would emit a
                # PER-KEY eachoracle_collector(key_field), not the node-level
                # collector the fusion's redirect_fn reads.
                collector_field = StateKeys.eachoracle_collector(field_name)
                fields[collector_field] = (
                    Annotated[list, _concat_reducer],
                    [],
                )
                for output_key, output_type in no.all_keys.items():
                    key_field = output_field_name(field_name, output_key)
                    field_type = dict[str, output_type] | None  # type: ignore[valid-type]
                    fields[key_field] = (
                        Annotated[field_type, _merge_dicts],
                        None,
                    )
            case PrimaryShape.ORACLE:
                # Oracle + dict-form: single collector for the whole result dict,
                # per-key consumer fields without per-key collectors.
                collector_field = StateKeys.oracle_collector(field_name)
                fields[collector_field] = (
                    Annotated[list[dict], _concat_reducer],
                    [],
                )
                for output_key, output_type in no.all_keys.items():
                    key_field = output_field_name(field_name, output_key)
                    fields[key_field] = (output_type | None, None)
            case PrimaryShape.BARE | PrimaryShape.EACH | PrimaryShape.LOOP | PrimaryShape.PORTAL:
                # EACH reaches here only UNFUSED — the fused case is the guarded
                # arm above; a plain Each dict-form defers per key like the rest.
                # PORTAL dict-form is rejected at assembly (D-DICT-OUTPUTS);
                # the arm is defensively-unreachable and defers to the per-key
                # single-output builder (which treats PORTAL as bare).
                for output_key, output_type in no.all_keys.items():
                    key_field = output_field_name(field_name, output_key)
                    _add_single_output_field(node, key_field, output_type, fields)
            case _ as unreachable:
                assert_never(unreachable)
        return

    # Single-type outputs (backward compat): one field named after the node.
    _add_single_output_field(node, field_name, no.primary, fields)


def _add_single_output_field(
    node: Node,
    field_name: str,
    output_type: Any,
    fields: dict[str, Any],
) -> None:
    """Add one output field to the state model, applying modifier wrapping."""
    combo, _ = classify_modifiers(node)
    match COMBO_DECOMPOSITION[combo].primary:
        case PrimaryShape.EACH:
            # The Each×Oracle fusion adds a tagged collector; its FINAL output
            # field is otherwise identical to plain Each's (dict[str, merged]),
            # so the two paths share the field build rather than duplicating it.
            # Split by COMBO_DECOMPOSITION[combo].fused, the table's own answer.
            if COMBO_DECOMPOSITION[combo].fused:
                collector_field = StateKeys.eachoracle_collector(field_name)
                fields[collector_field] = (
                    Annotated[list, _concat_reducer],
                    [],
                )
            field_type = dict[str, output_type] | None  # type: ignore[valid-type]
            fields[field_name] = (
                Annotated[field_type, _merge_dicts],
                None,
            )
        case PrimaryShape.ORACLE:
            collector_field = StateKeys.oracle_collector(field_name)
            # When oracle_gen_type is set, the collector holds per-variant types
            # (list[gen_type]), not the post-merge type. The consumer-facing field
            # keeps node.outputs (the post-merge type).
            collector_type = node.oracle_gen_type if node.oracle_gen_type is not None else output_type
            fields[collector_field] = (
                Annotated[list[collector_type], _concat_reducer],  # type: ignore[valid-type]
                [],
            )
            fields[field_name] = (output_type | None, None)
        case PrimaryShape.LOOP:
            # Loop: append-list reducer. Each iteration pushes to the list.
            # _extract_input unwraps [-1] for the node on re-entry.
            # Downstream nodes after loop exit see the final value (unwrapped).
            fields[field_name] = (
                Annotated[list[output_type], _append_loop_result],
                [],
            )
        case PrimaryShape.BARE | PrimaryShape.PORTAL:
            # A Portal mesh member (with or without an Operator approval gate)
            # writes its OWN output field as a plain value (like a bare node);
            # the mesh channel + hop counter are separate neo_-prefixed fields
            # added per mesh entry below.
            fields[field_name] = (output_type | None, None)
        case _ as unreachable:
            assert_never(unreachable)
