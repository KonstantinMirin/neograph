"""Builds the per-sub-construct Callable that LangGraph adds to the parent StateGraph; encodes the sub-construct's input/output boundary semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel

from neograph._ir_branch import iter_with_arms
from neograph._oracle import _inject_oracle_config
from neograph._state_bus import StateBus, adapt_state
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.di import _unwrap_loop_value
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
        # REQUIRED: iterating state.keys() — every key is by definition present.
        val = state.get_required(attr_name)
        check_val = _unwrap_loop_value(val, object)
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
        check_val = _unwrap_loop_value(val, object)
        if isinstance(check_val, sub_output_type):
            return check_val
    return None


def make_subgraph_fn(sub: Construct, sub_graph: CompiledStateGraph) -> RunnableLambda:
    """Create a Runnable that runs a sub-Construct in isolation.

    Extracts input from parent state by type, runs sub_graph, extracts output
    by type, returns {field_name: output}.

    Dual-path (driver-selected, neograph-expi): returns
    ``RunnableLambda(subgraph_node, afunc=asubgraph_node)`` so the DRIVER picks
    the path — ``graph.invoke`` runs the sync twin (``sub_graph.invoke``),
    ``graph.ainvoke`` runs the async twin (``await sub_graph.ainvoke``). Without
    the afunc twin, LangGraph threadpools the sync closure under ``ainvoke`` and
    the ENTIRE child runs synchronously, blocking the loop and silently
    downgrading any async-only leaf inside the child (Phase-1 H2 invariant: async
    must propagate through every nesting level). The two twins share the same
    input-extraction (``_build_sub_input``) and update-shaping
    (``_build_update``) helpers so the sync/async paths cannot drift.
    """
    sub_log = log.bind(subgraph=sub.name)
    field_name = field_name_for(sub.name)

    sub_combo, _ = classify_modifiers(sub)
    has_loop = sub_combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR)

    def _build_sub_input(state: BaseModel | dict[str, Any], config: RunnableConfig) -> tuple[dict[str, Any], StateBus, RunnableConfig]:
        """Shared pre-invoke logic: extract input, forward context, inject config.

        Returns ``(sub_input, bus, config)``. Identical for both twins — the only
        difference between sync and async is the ``invoke`` vs ``ainvoke`` call.
        """
        bus = adapt_state(state)

        # Loop re-entry: on iteration 2+, read from own append-list.
        # When output type matches input type (classic refine pattern),
        # feed the output back as input.  When output differs from input
        # (produce+validate pattern), skip the shortcut and re-read
        # original inputs from parent state.
        input_data = None
        if has_loop:
            # StateBus.get optional: loop-bootstrap — sub-construct's own field
            # is unbound on iteration 0.
            own_val = bus.get(field_name)
            if isinstance(own_val, list):
                # Latest-of-append-list unwrap delegates to the di monopoly
                # per neograph-ovx1: None for the unbound/empty first iteration,
                # own_val[-1] otherwise.
                latest = _unwrap_loop_value(own_val, object)
                if latest is not None and (sub.input is None or isinstance(latest, sub.input)):
                    input_data = latest

        # First iteration, non-loop, or input!=output loop re-entry:
        # extract input by type from parent state.
        if input_data is None and sub.input is not None:
            input_data = _scan_subgraph_input(bus, sub.input)

        # Run sub-graph with isolated state.
        # StateBus.get optional: framework — node_id is a DI-style context key
        # that may not be present; empty-string default propagates to sub-graph.
        sub_input: dict[str, Any] = {StateKeys.NODE_ID: bus.get(StateKeys.NODE_ID, "")}
        if input_data is not None:
            sub_input[StateKeys.SUBGRAPH_INPUT] = input_data

        # Forward context fields from parent state into sub-construct.
        # iter_with_arms so a context node living inside a branch arm of the
        # sub-construct gets its context field forwarded, not resolved to None.
        # See neograph-vn5f (site 11).
        for n in iter_with_arms(sub):
            if hasattr(n, "context") and n.context:
                for ctx_name in n.context:
                    ctx_field = field_name_for(ctx_name)
                    # StateBus.get optional: context forwarding is best-effort;
                    # missing context propagates as None to sub-node, whose own
                    # _extract_context read enforces required-ness (see §7 Q3).
                    val = bus.get(ctx_field)
                    if val is not None:
                        sub_input[ctx_field] = val

        # Forward Oracle gen_id + model override from parent state into config
        config = _inject_oracle_config(bus, config)
        return sub_input, bus, config

    def _build_update(sub_result: dict[str, Any], bus: StateBus) -> dict[str, Any]:
        """Shared post-invoke logic: extract declared output, shape state update.

        Identical for both twins.
        """
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
            count_field = StateKeys.loop_count(field_name)
            # Counter bootstrap (absent/None -> 0) lives in StateBus.get_counter.
            current = bus.get_counter(count_field)
            update[count_field] = current + 1
        return update

    def subgraph_node(state: BaseModel | dict[str, Any], config: RunnableConfig) -> dict:
        sub_log.info("subgraph_start")
        sub_input, bus, config = _build_sub_input(state, config)
        # No strip: the child compile declared output_schema=non-neo_ fields, so
        # sub_graph.invoke() already returns neo_-free results. See neograph-pjqe.
        # The reverse-scan in _build_update sees the same dict _strip_internals made.
        sub_result = sub_graph.invoke(sub_input, config=config)
        return _build_update(sub_result, bus)

    async def asubgraph_node(state: BaseModel | dict[str, Any], config: RunnableConfig) -> dict:
        sub_log.info("subgraph_start")
        sub_input, bus, config = _build_sub_input(state, config)
        # Async twin: await the child's ainvoke so a sub-construct under the async
        # driver propagates async selection into the child graph, instead of
        # blocking the loop on sub_graph.invoke. See neograph-expi.
        # No strip: child output_schema filters ainvoke() results. See neograph-pjqe.
        sub_result = await sub_graph.ainvoke(sub_input, config=config)
        return _build_update(sub_result, bus)

    # Driver-selected dual path. __name__ stays informational; routing is the
    # graph.add_node(name, fn) argument (always sub.name/item.name). See
    # neograph-y20i.
    return RunnableLambda(subgraph_node, afunc=asubgraph_node)
