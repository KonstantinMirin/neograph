"""State bus compiler — auto-generates Pydantic state from Node I/O union.

Each Construct gets its own state model with exactly the fields its Nodes need.
No monolithic state that grows with every derivation type.
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from pydantic import BaseModel, create_model

from neograph.construct import Construct
from neograph.errors import CompileError
from neograph.forward import _BranchNode
from neograph.naming import field_name_for

log = structlog.get_logger()
from typing import assert_never

from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.node import Node


def _last_write_wins(existing: Any, new: Any) -> Any:
    """Reducer: last write wins (default for sequential nodes)."""
    return new


def _append_loop_result(existing: Any, new: Any) -> list:
    """Reducer: append each loop iteration's result to a list."""
    if existing is None:
        existing = []
    return [*existing, new]


def _collect_oracle_results(existing: Any, new: Any) -> list:
    """Reducer: collect oracle fan-out results into a list."""
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    return [*existing, new]


def _append_tagged(existing: Any, new: Any) -> list:
    """Reducer: append tagged (key, result) tuples for Each×Oracle fusion."""
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    return [*existing, new]


def _merge_dicts(existing: Any, new: dict) -> dict:
    """Reducer: merge dicts additively (for fan-out results).

    On duplicate keys, keeps the existing (first) value. Logs a single
    summary instead of per-key warnings (neograph-o0tv: noisy on resume).
    """
    if existing is None:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(new, dict):
        return existing
    merged = {**existing}
    dupes = []
    for key, val in new.items():
        if key in merged:
            dupes.append(key)
            continue
        merged[key] = val
    if dupes:
        log.debug("each_duplicate_keys", count=len(dupes), keys=dupes[:5],
                  action="kept_existing", truncated=len(dupes) > 5)
    return merged


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
                if arm_combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR):
                    fields[field_name] = (
                        Annotated[list[arm_item.output], _append_loop_result],  # type: ignore[name-defined]
                        [],
                    )
                    fields[f'neo_loop_count_{field_name}'] = (int, 0)
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
        match combo:
            case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
                # Oracle on Construct: collector + consumer field
                collector_field = f"neo_oracle_{field_name}"
                fields[collector_field] = (
                    Annotated[list[sub.output], _collect_oracle_results],  # type: ignore[name-defined]
                    [],
                )
                fields[field_name] = (sub.output | None, None)  # type: ignore[name-defined]
            case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
                # Each on Construct: dict field
                field_type = dict[str, sub.output] | None  # type: ignore[name-defined]
                fields[field_name] = (
                    Annotated[field_type, _merge_dicts],
                    None,
                )
            case ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
                # Loop on Construct: append-list + iteration counter
                fields[field_name] = (
                    Annotated[list[sub.output], _append_loop_result],  # type: ignore[name-defined]
                    [],
                )
                fields[f'neo_loop_count_{field_name}'] = (int, 0)
            case ModifierCombo.BARE | ModifierCombo.OPERATOR:
                fields[field_name] = (sub.output | None, None)
            case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
                # Each×Oracle on Constructs not supported
                # Compiler raises earlier; this is a defensive fallback.
                fields[field_name] = (sub.output | None, None)
            case _ as unreachable:
                assert_never(unreachable)

    # Oracle support: generator ID + optional model override passed via state
    all_items = nodes_only + sub_constructs
    has_any_oracle = False
    has_any_each = False
    for item in all_items:
        item_combo, _ = classify_modifiers(item)
        if item_combo in (
            ModifierCombo.ORACLE, ModifierCombo.ORACLE_OPERATOR,
            ModifierCombo.EACH_ORACLE, ModifierCombo.EACH_ORACLE_OPERATOR,
        ):
            has_any_oracle = True
        if item_combo in (
            ModifierCombo.EACH, ModifierCombo.EACH_OPERATOR,
            ModifierCombo.EACH_ORACLE, ModifierCombo.EACH_ORACLE_OPERATOR,
        ):
            has_any_each = True
    if has_any_oracle:
        fields["neo_oracle_gen_id"] = (str | None, None)
        fields["neo_oracle_model"] = (str | None, None)
    if has_any_each:
        fields["neo_each_item"] = (Any, None)

    # Loop support: iteration counter per looped node
    for n in nodes_only:
        n_combo, n_mods = classify_modifiers(n)
        if n_combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR):
            field_name = field_name_for(n.name)
            fields[f'neo_loop_count_{field_name}'] = (int, 0)
            loop = n_mods["loop"]
            if loop.history:
                fields[f'neo_loop_history_{field_name}'] = (
                    Annotated[list, _collect_oracle_results], []
                )

    # Subgraph input port — when this Construct declares an input type
    if construct.input is not None:
        fields["neo_subgraph_input"] = (construct.input | None, None)

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
    fields["node_id"] = (str, "")
    fields["project_root"] = (str, "")
    fields["human_feedback"] = (dict[str, Any] | None, None)
    fields["neo_schema_fingerprint"] = (str, "")
    fields["neo_node_fingerprints"] = (dict[str, str], {})

    return create_model(f"{construct.name}State", **fields)


def compute_node_fingerprints(construct: Any) -> dict[str, str]:
    """Compute per-node output type fingerprints for checkpoint invalidation.

    Returns {field_name: sha256_prefix} for each node in the construct.
    Used to identify which specific nodes changed between runs.
    """
    import hashlib
    from neograph.naming import field_name_for

    result = {}
    for item in construct.nodes:
        if hasattr(item, "outputs") and item.outputs is not None:
            fname = field_name_for(item.name)
            if isinstance(item.outputs, dict):
                # Dict-form outputs: fingerprint each key
                for key, typ in item.outputs.items():
                    full_name = f"{fname}_{key}"
                    result[full_name] = hashlib.sha256(
                        f"{full_name}:{type(typ).__name__ if isinstance(typ, type) else str(typ)}".encode()
                    ).hexdigest()[:12]
            else:
                typ = item.outputs
                result[fname] = hashlib.sha256(
                    f"{fname}:{typ.__qualname__ if isinstance(typ, type) else str(typ)}".encode()
                ).hexdigest()[:12]
        elif hasattr(item, "nodes"):
            # Sub-construct: fingerprint its output
            fname = field_name_for(item.name)
            if hasattr(item, "output") and item.output is not None:
                typ = item.output
                result[fname] = hashlib.sha256(
                    f"{fname}:{typ.__qualname__ if isinstance(typ, type) else str(typ)}".encode()
                ).hexdigest()[:12]
    return result


def compute_schema_fingerprint(state_model: type) -> str:
    """Compute a stable fingerprint from the state model's non-framework fields.

    The fingerprint changes when node output types change (field added/removed,
    type changed, class renamed). Framework fields (neo_*, node_id, project_root,
    human_feedback) are excluded — they change with modifier config, not schema.
    """
    import hashlib
    _FRAMEWORK_PREFIXES = ("neo_", "node_id", "project_root", "human_feedback")
    items = []
    for fname, finfo in state_model.model_fields.items():
        if any(fname.startswith(p) or fname == p for p in _FRAMEWORK_PREFIXES):
            continue
        items.append((fname, str(finfo.annotation)))
    items.sort()
    raw = repr(items).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


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

    # Dict-form outputs: one state field per key (neograph-1bp.2).
    if isinstance(node.outputs, dict):
        combo, mods = classify_modifiers(node)
        match combo:
            case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
                # Each×Oracle fusion + dict-form: tagged collector + dict output
                # per key. Same as single-type fusion but per-key.
                collector_field = f"neo_eachoracle_{field_name}"
                fields[collector_field] = (
                    Annotated[list, _append_tagged],
                    [],
                )
                for output_key, output_type in node.outputs.items():
                    key_field = f"{field_name}_{output_key}"
                    field_type = dict[str, output_type] | None  # type: ignore[valid-type]
                    fields[key_field] = (
                        Annotated[field_type, _merge_dicts],
                        None,
                    )
            case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
                # Oracle + dict-form: single collector for the whole result dict,
                # per-key consumer fields without per-key collectors.
                collector_field = f"neo_oracle_{field_name}"
                fields[collector_field] = (
                    Annotated[list[dict], _collect_oracle_results],
                    [],
                )
                for output_key, output_type in node.outputs.items():
                    key_field = f"{field_name}_{output_key}"
                    fields[key_field] = (output_type | None, None)
            case (
                ModifierCombo.BARE | ModifierCombo.OPERATOR
                | ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR
                | ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR
            ):
                for output_key, output_type in node.outputs.items():
                    key_field = f"{field_name}_{output_key}"
                    _add_single_output_field(node, key_field, output_type, fields)
            case _ as unreachable:
                assert_never(unreachable)
        return

    # Single-type outputs (backward compat): one field named after the node.
    _add_single_output_field(node, field_name, node.outputs, fields)


def _add_single_output_field(
    node: Node,
    field_name: str,
    output_type: Any,
    fields: dict[str, Any],
) -> None:
    """Add one output field to the state model, applying modifier wrapping."""
    combo, mods = classify_modifiers(node)
    match combo:
        case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
            # Each×Oracle fusion: tagged collector + dict output
            collector_field = f"neo_eachoracle_{field_name}"
            fields[collector_field] = (
                Annotated[list, _append_tagged],
                [],
            )
            # Final output: same shape as Each alone (dict[str, merged_type])
            field_type = dict[str, output_type] | None  # type: ignore[valid-type]
            fields[field_name] = (
                Annotated[field_type, _merge_dicts],
                None,
            )
        case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
            field_type = dict[str, output_type] | None  # type: ignore[valid-type, misc, assignment]
            fields[field_name] = (
                Annotated[field_type, _merge_dicts],
                None,
            )
        case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
            collector_field = f"neo_oracle_{field_name}"
            # When oracle_gen_type is set, the collector holds per-variant types
            # (list[gen_type]), not the post-merge type. The consumer-facing field
            # keeps node.outputs (the post-merge type).
            collector_type = node.oracle_gen_type if node.oracle_gen_type is not None else output_type
            fields[collector_field] = (
                Annotated[list[collector_type], _collect_oracle_results],  # type: ignore[valid-type]
                [],
            )
            fields[field_name] = (output_type | None, None)
        case ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
            # Loop: append-list reducer. Each iteration pushes to the list.
            # _extract_input unwraps [-1] for the node on re-entry.
            # Downstream nodes after loop exit see the final value (unwrapped).
            fields[field_name] = (
                Annotated[list[output_type], _append_loop_result],
                [],
            )
        case ModifierCombo.BARE | ModifierCombo.OPERATOR:
            fields[field_name] = (output_type | None, None)
        case _ as unreachable:
            assert_never(unreachable)
