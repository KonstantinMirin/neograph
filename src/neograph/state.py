"""State bus compiler — auto-generates Pydantic state from Node I/O union.

Each Construct gets its own state model with exactly the fields its Nodes need.
No monolithic state that grows with every derivation type.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, create_model

from neograph.construct import Construct
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
            raise ValueError(msg)
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
            raise ValueError(msg)
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
    """Add a node's output type as a field on the state model."""
    if node.output is None:
        msg = f"Node '{node.name}' has no output type. Every node must declare output=SomeModel."
        raise ValueError(msg)

    field_name = node.name.replace("-", "_")
    output_type = node.output

    # Fan-out nodes (Each) produce dict[key, output_type]
    if node.has_modifier(Each):
        field_type = dict[str, output_type] | None
        fields[field_name] = (
            Annotated[field_type, _merge_dicts],
            None,
        )
    # Ensemble nodes (Oracle): internal collector + consumer-facing merge result
    elif node.has_modifier(Oracle):
        # Internal: collects N generator outputs
        collector_field = f"neo_oracle_{field_name}"
        fields[collector_field] = (
            Annotated[list[output_type], _collect_oracle_results],
            [],
        )
        # Consumer-facing: the merged result, named after the node
        fields[field_name] = (output_type | None, None)
    # Sequential nodes — simple last-write-wins
    else:
        fields[field_name] = (output_type | None, None)
