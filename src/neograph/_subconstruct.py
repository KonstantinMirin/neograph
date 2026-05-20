"""Builds the per-sub-construct Callable that LangGraph adds to the parent StateGraph; encodes the sub-construct's input/output boundary semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._oracle import _inject_oracle_config
from neograph._state_bus import StateBus, adapt_state
from neograph.construct import Construct
from neograph.errors import ExecutionError
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.naming import field_name_for

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

log = structlog.get_logger()


def _scan_subgraph_input(state: StateBus, sub_input_type: type) -> Any:
    """Scan parent state in reverse order for first value matching sub.input.

    Reverse iteration so later pipeline nodes take precedence (e.g., loop
    output over seed). Unwraps append-lists by checking val[-1] against the
    declared input type.
    """
    for attr_name in reversed(state.keys()):
        val = state.get(attr_name)
        check_val = val
        if isinstance(val, list) and val:
            check_val = val[-1]
        if check_val is not None and isinstance(check_val, sub_input_type):
            return check_val
    return None


def _scan_subgraph_output(sub_result: dict[str, Any], sub_output_type: type) -> Any:
    """Scan sub-graph result in reverse order for first value matching sub.output.

    Reverse iteration so later pipeline nodes take precedence. Unwraps loop
    append-lists (list[T] from reducer) by checking val[-1] against the
    declared output type.
    """
    for val in reversed(list(sub_result.values())):
        check_val = val
        if isinstance(val, list) and val:
            check_val = val[-1]
        if isinstance(check_val, sub_output_type):
            return check_val
    return None


def make_subgraph_fn(sub: Construct, sub_graph: CompiledStateGraph) -> Callable:
    """Create a function that runs a sub-Construct in isolation.

    Extracts input from parent state by type, runs sub_graph,
    extracts output by type, returns {field_name: output}.
    """
    from neograph.runner import _strip_internals

    sub_log = log.bind(subgraph=sub.name)
    field_name = field_name_for(sub.name)

    sub_combo, _ = classify_modifiers(sub)
    has_loop = sub_combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR)

    def subgraph_node(state: BaseModel | dict[str, Any], config: RunnableConfig) -> dict:
        sub_log.info("subgraph_start")
        bus = adapt_state(state)

        # Loop re-entry: on iteration 2+, read from own append-list.
        # When output type matches input type (classic refine pattern),
        # feed the output back as input.  When output differs from input
        # (produce+validate pattern), skip the shortcut and re-read
        # original inputs from parent state.
        input_data = None
        if has_loop:
            own_val = bus.get(field_name)
            if isinstance(own_val, list) and own_val:
                latest = own_val[-1]
                if sub.input is None or isinstance(latest, sub.input):
                    input_data = latest

        # First iteration, non-loop, or input!=output loop re-entry:
        # extract input by type from parent state.
        if input_data is None and sub.input is not None:
            input_data = _scan_subgraph_input(bus, sub.input)

        # Run sub-graph with isolated state
        sub_input: dict[str, Any] = {"node_id": bus.get("node_id", "")}
        if input_data is not None:
            sub_input["neo_subgraph_input"] = input_data

        # Forward context fields from parent state into sub-construct
        for n in sub.nodes:
            if hasattr(n, "context") and n.context:
                for ctx_name in n.context:
                    ctx_field = field_name_for(ctx_name)
                    val = bus.get(ctx_field)
                    if val is not None:
                        sub_input[ctx_field] = val

        # Forward Oracle gen_id + model override from parent state into config
        config = _inject_oracle_config(bus, config)

        sub_result = _strip_internals(sub_graph.invoke(sub_input, config=config))

        # Extract the declared output type from sub result.
        output_val = None
        if sub.output is not None:
            output_val = _scan_subgraph_output(sub_result, sub.output)

        # Runtime defense: if no internal node produced a compatible output,
        # fail loud instead of writing None silently.
        if output_val is None and sub.output is not None:  # pragma: no cover — defensive
            raise ExecutionError.build(
                "No internal node produced a compatible output value",
                expected=sub.output.__name__,
                hint="Check that at least one node writes the declared output type",
                construct=sub.name,
            )

        sub_log.info("subgraph_complete")
        update: dict[str, Any] = {field_name: output_val}
        if has_loop:
            count_field = f"neo_loop_count_{field_name}"
            current = bus.get(count_field) or 0
            update[count_field] = current + 1
        return update

    subgraph_node.__name__ = field_name
    return subgraph_node
