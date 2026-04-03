"""State bus compiler — auto-generates Pydantic state from Node I/O union.

Each Construct gets its own state model with exactly the fields its Nodes need.
No monolithic state that grows with every derivation type.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, get_args, get_origin

from pydantic import BaseModel, create_model

from neograph.construct import Construct
from neograph.modifiers import Oracle, Replicate
from neograph.node import Node


class AtomResult(BaseModel, frozen=True):
    """Record of a completed atom execution."""

    node_name: str
    success: bool
    error: str | None = None


def _last_write_wins(existing: Any, new: Any) -> Any:
    """Reducer: last write wins (default for sequential nodes)."""
    return new


def _merge_dicts(existing: dict, new: dict) -> dict:
    """Reducer: merge dicts additively (for fan-out results)."""
    merged = {**existing}
    merged.update(new)
    return merged


def compile_state_model(construct: Construct) -> type[BaseModel]:
    """Generate a Pydantic state model from the union of Node I/O fields.

    Each Node's output becomes a state field. Fan-out nodes get dict reducers.
    The resulting model is used as the LangGraph StateGraph schema.
    """
    fields: dict[str, Any] = {}

    for node in construct.nodes:
        _add_output_field(node, fields)

    # Framework fields — always present
    fields["node_id"] = (str, ...)
    fields["project_root"] = (str, "")
    fields["total_cost"] = (Annotated[float, operator.add], 0.0)
    fields["completed_atoms"] = (Annotated[list[AtomResult], operator.add], [])
    fields["human_feedback"] = (dict[str, Any] | None, None)

    return create_model(f"{construct.name}State", **fields)


def _add_output_field(node: Node, fields: dict[str, Any]) -> None:
    """Add a node's output type as a field on the state model."""
    if node.output is None:
        return

    field_name = node.name.replace("-", "_")
    output_type = node.output

    # Fan-out nodes (Replicate) produce dict[key, output_type]
    if node.has_modifier(Replicate):
        field_type = dict[str, output_type] | None
        fields[field_name] = (
            Annotated[field_type, _merge_dicts],
            None,
        )
    # Ensemble nodes (Oracle) use last-write-wins (merge node produces single output)
    elif node.has_modifier(Oracle):
        fields[field_name] = (
            Annotated[output_type | None, _last_write_wins],
            None,
        )
    # Sequential nodes — simple last-write-wins
    else:
        fields[field_name] = (output_type | None, None)
