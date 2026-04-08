"""State bus compiler — auto-generates Pydantic state from Node I/O union.

Each Construct gets its own state model with exactly the fields its Nodes need.
No monolithic state that grows with every derivation type.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, create_model

import structlog

from neograph.construct import Construct
from neograph.errors import CompileError
from neograph.forward import _BranchNode

log = structlog.get_logger()
from neograph.modifiers import Oracle, Each, Loop
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


def _merge_dicts(existing: Any, new: dict) -> dict:
    """Reducer: merge dicts additively (for fan-out results).

    On duplicate keys, keeps the existing (first) value and logs a warning.
    """
    if existing is None:
        existing = {}
    merged = {**existing}
    for key, val in new.items():
        if key in merged:
            log.warning("each_duplicate_key", key=key, action="kept_existing")
            continue
        merged[key] = val
    return merged


def compile_state_model(construct: Construct) -> type[BaseModel]:
    """Generate a Pydantic state model from the union of Node I/O fields.

    Each Node's output becomes a state field. Fan-out nodes get dict reducers.
    The resulting model is used as the LangGraph StateGraph schema.
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
        field_name = item.name.replace("-", "_")
        if field_name in seen_fields:
            raise CompileError(
                f"Node name collision: '{item.name}' and '{seen_fields[field_name]}' "
                f"both map to state field '{field_name}'. Rename one of them."
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
                field_name = arm_item.name.replace("-", "_")
                if arm_item.has_modifier(Loop):
                    fields[field_name] = (
                        Annotated[list[arm_item.output], _append_loop_result],
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
            msg = f"Sub-construct '{sub.name}' has no output type. Declare output=SomeModel."
            raise CompileError(msg)
        field_name = sub.name.replace("-", "_")

        if sub.has_modifier(Oracle):
            # Oracle on Construct: collector + consumer field
            collector_field = f"neo_oracle_{field_name}"
            fields[collector_field] = (
                Annotated[list[sub.output], _collect_oracle_results],
                [],
            )
            fields[field_name] = (sub.output | None, None)
        elif sub.has_modifier(Each):
            # Each on Construct: dict field
            field_type = dict[str, sub.output] | None
            fields[field_name] = (
                Annotated[field_type, _merge_dicts],
                None,
            )
        elif sub.has_modifier(Loop):
            # Loop on Construct: append-list + iteration counter
            fields[field_name] = (
                Annotated[list[sub.output], _append_loop_result],
                [],
            )
            fields[f'neo_loop_count_{field_name}'] = (int, 0)
        else:
            fields[field_name] = (sub.output | None, None)

    # Oracle support: generator ID + optional model override passed via state
    all_items = nodes_only + sub_constructs
    if any(item.has_modifier(Oracle) for item in all_items):
        fields["neo_oracle_gen_id"] = (str | None, None)
        fields["neo_oracle_model"] = (str | None, None)

    # Each support: current item passed via state
    if any(item.has_modifier(Each) for item in all_items):
        fields["neo_each_item"] = (Any, None)

    # Loop support: iteration counter per looped node
    for n in nodes_only:
        if n.has_modifier(Loop):
            field_name = n.name.replace('-', '_')
            fields[f'neo_loop_count_{field_name}'] = (int, 0)
            loop = n.get_modifier(Loop)
            if loop.history:
                fields[f'neo_loop_history_{field_name}'] = (
                    Annotated[list, _collect_oracle_results], []
                )

    # Subgraph input port — when this Construct declares an input type
    if construct.input is not None:
        fields["neo_subgraph_input"] = (construct.input | None, None)

    # Context fields — forwarded from parent state for nodes that declare context=
    for n in nodes_only:
        if n.context:
            for ctx_name in n.context:
                ctx_field = ctx_name.replace("-", "_")
                if ctx_field not in fields:
                    fields[ctx_field] = (Any, None)
    # Also check branch arm nodes (skip Constructs — they handle context internally)
    for branch in branch_nodes:
        meta = branch._neo_branch_meta
        for arm_node in meta.true_arm_nodes + meta.false_arm_nodes:
            if isinstance(arm_node, Construct):
                continue
            if arm_node.context:
                for ctx_name in arm_node.context:
                    ctx_field = ctx_name.replace("-", "_")
                    if ctx_field not in fields:
                        fields[ctx_field] = (Any, None)

    # Framework fields — always present
    # node_id and project_root have defaults so consumers can omit them
    # in run(input=...); they're still accessible via config["configurable"]
    # for node functions that need pipeline metadata.
    fields["node_id"] = (str, "")
    fields["project_root"] = (str, "")
    fields["human_feedback"] = (dict[str, Any] | None, None)

    return create_model(f"{construct.name}State", **fields)


def _add_output_field(node: Node, fields: dict[str, Any]) -> None:
    """Add a node's output type(s) as field(s) on the state model.

    When outputs is a dict (multi-output), creates one field per key:
    ``{node_name}_{output_key}``. Each/Oracle modifiers apply per key.
    When outputs is a single type (backward compat), creates ``{node_name}``.
    """
    if node.outputs is None:
        msg = f"Node '{node.name}' has no output type. Every node must declare outputs=SomeModel."
        raise CompileError(msg)

    field_name = node.name.replace("-", "_")

    # Dict-form outputs: one state field per key (neograph-1bp.2).
    if isinstance(node.outputs, dict):
        if node.has_modifier(Oracle):
            # Oracle + dict-form: single collector for the whole result dict,
            # per-key consumer fields without per-key collectors (neograph-7ft).
            collector_field = f"neo_oracle_{field_name}"
            fields[collector_field] = (
                Annotated[list[dict], _collect_oracle_results],
                [],
            )
            for output_key, output_type in node.outputs.items():
                key_field = f"{field_name}_{output_key}"
                fields[key_field] = (output_type | None, None)
        else:
            for output_key, output_type in node.outputs.items():
                key_field = f"{field_name}_{output_key}"
                _add_single_output_field(node, key_field, output_type, fields)
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
    if node.has_modifier(Each):
        field_type = dict[str, output_type] | None
        fields[field_name] = (
            Annotated[field_type, _merge_dicts],
            None,
        )
    elif node.has_modifier(Oracle):
        collector_field = f"neo_oracle_{field_name}"
        # When oracle_gen_type is set, the collector holds per-variant types
        # (list[gen_type]), not the post-merge type. The consumer-facing field
        # keeps node.outputs (the post-merge type).
        collector_type = node.oracle_gen_type if node.oracle_gen_type is not None else output_type
        fields[collector_field] = (
            Annotated[list[collector_type], _collect_oracle_results],
            [],
        )
        fields[field_name] = (output_type | None, None)
    elif node.has_modifier(Loop):
        # Loop: append-list reducer. Each iteration pushes to the list.
        # _extract_input unwraps [-1] for the node on re-entry.
        # Downstream nodes after loop exit see the final value (unwrapped).
        fields[field_name] = (
            Annotated[list[output_type], _append_loop_result],
            [],
        )
    else:
        fields[field_name] = (output_type | None, None)
