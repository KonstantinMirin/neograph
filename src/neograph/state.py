"""State bus compiler — auto-generates Pydantic state from Node I/O union.

Each Construct gets its own state model with exactly the fields its Nodes need.
No monolithic state that grows with every derivation type.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, create_model

from neograph.construct import Construct
from neograph.errors import CompileError, ExecutionError
from neograph.forward import _BranchNode
from neograph.modifiers import Oracle, Each
from neograph.node import Node



def _last_write_wins(existing: Any, new: Any) -> Any:
    """Reducer: last write wins (default for sequential nodes)."""
    return new


def _collect_oracle_results(existing: Any, new: Any) -> list:
    """Reducer: collect oracle fan-out results into a list."""
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    return [*existing, new]


def _merge_dicts(existing: Any, new: dict) -> dict:
    """Reducer: merge dicts additively (for fan-out results)."""
    if existing is None:
        existing = {}
    merged = {**existing}
    for key, val in new.items():
        if key in merged:
            msg = f"Each fan-out: duplicate key '{key}'. Two items produced the same dispatch key."
            raise ExecutionError(msg)
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

    for node in nodes_only:
        _add_output_field(node, fields)

    # Branch arm nodes: add state fields for nodes inside branch arms
    for branch in branch_nodes:
        meta = branch._neo_branch_meta
        for arm_node in meta.true_arm_nodes + meta.false_arm_nodes:
            _add_output_field(arm_node, fields)

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
        else:
            fields[field_name] = (sub.output | None, None)

    # Oracle support: generator ID passed via state
    all_items = nodes_only + sub_constructs
    if any(item.has_modifier(Oracle) for item in all_items):
        fields["neo_oracle_gen_id"] = (str | None, None)

    # Each support: current item passed via state
    if any(item.has_modifier(Each) for item in all_items):
        fields["neo_each_item"] = (Any, None)

    # Subgraph input port — when this Construct declares an input type
    if construct.input is not None:
        fields["neo_subgraph_input"] = (construct.input | None, None)

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
        fields[collector_field] = (
            Annotated[list[output_type], _collect_oracle_results],
            [],
        )
        fields[field_name] = (output_type | None, None)
    else:
        fields[field_name] = (output_type | None, None)
