"""Node-function construction — turns Node definitions into LangGraph callables."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._dispatch import (  # noqa: F401 — re-exported for tests/backward compat
    ModeDispatch,
    NodeInput,
    NodeOutput,
    ScriptedDispatch,
    ThinkDispatch,
    ToolDispatch,
    _dispatch_for_mode,
    _render_input,
)
from neograph._modifier_io import (  # noqa: F401 — internal helpers re-exported for tests
    InputShape,
    _apply_skip_when,
    _classify_input_shape,
    _extract_each_item,
    _extract_fan_in_dict,
    _extract_input,
    _extract_loop_reentry,
    _extract_single_type,
)
from neograph._normalize import normalize_inputs
from neograph._observability import _extract_context  # noqa: F401 — re-exported
from neograph._oracle import (  # noqa: F401 — re-exported so compiler.py imports stay stable
    _build_oracle_merge_result,
    _unwrap_oracle_results,
    make_each_redirect_fn,
    make_eachoracle_redirect_fn,
    make_oracle_merge_fn,
    make_oracle_redirect_fn,
)
from neograph._registry import registry
from neograph._runtime_registry import (  # noqa: F401 — re-exported for public API
    lookup_condition,
    lookup_scripted,
    register_condition,
    register_scripted,
    register_tool_factory,
)
from neograph._state_bus import adapt_state
from neograph._state_io import (  # noqa: F401 — re-exported for tests
    _build_state_update,
    _inject_oracle_config,
)
from neograph.construct import Construct

# Backward-compat re-exports for tests that imported these helpers from
# factory.py before the §4 split. Each is `noqa`'d individually because ruff
# strips items inside a parenthesized import group even with a line-level noqa.
from neograph.di import _isinstance_safe as _is_instance_safe  # noqa: F401
from neograph.di import (
    _unwrap_each_dict,  # noqa: F401
    _unwrap_loop_value,  # noqa: F401
)
from neograph.errors import ConfigurationError, ExecutionError
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.naming import field_name_for
from neograph.node import Node, TypeSpecStatic

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

log = structlog.get_logger()


def _type_name(t: TypeSpecStatic) -> str | None:
    """Get a readable name from a type, or None."""
    if t is None:
        return None
    if isinstance(t, dict):
        parts = ", ".join(
            f"{k}: {getattr(v, '__name__', str(v))}" for k, v in t.items()
        )
        return "{" + parts + "}"
    return getattr(t, '__name__', str(t))


def _execute_node(
    node: Node,
    state: BaseModel,
    config: RunnableConfig,
    dispatch: ModeDispatch,
) -> dict[str, Any]:
    """Single execution path for all non-raw node modes.

    Preamble: log, Oracle config, input extraction, skip_when check.
    Dispatch: mode-specific logic via ModeDispatch protocol.
    Postamble: state update, log complete.

    _apply_skip_when has state-writing side effects — if it returns
    non-None, we return immediately and do NOT call _build_state_update.
    """
    field_name = field_name_for(node.name)
    node_log = log.bind(node=node.name, mode=node.mode)
    node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))

    t0 = time.monotonic()

    bus = adapt_state(state)
    config = _inject_oracle_config(bus, config)
    raw_input = _extract_input(bus, node)

    skip_result = _apply_skip_when(node, raw_input, field_name, t0, node_log, bus)
    if skip_result is not None:
        return skip_result

    context_data = _extract_context(bus, node) if node.mode != "scripted" else None

    if isinstance(raw_input, dict) and normalize_inputs(node.inputs).is_dict_form:
        node_input = NodeInput(fan_in=raw_input)
    else:
        node_input = NodeInput(single=raw_input)

    output = dispatch.execute(node, node_input, config, context_data)
    update = _build_state_update(node, field_name, output.value, bus)

    elapsed = time.monotonic() - t0
    node_log.info("node_complete", duration_s=round(elapsed, 3))
    return update


def make_node_fn(node: Node) -> Callable:
    """Create a LangGraph node function from a Node definition.

    This is the core of NeoGraph — the generic factory that eliminates
    the 70% boilerplate from every hand-coded node.

    Raw nodes get a minimal observability wrapper. All other modes
    (scripted, think, agent, act) go through _execute_node with a
    mode-specific ModeDispatch.
    """
    # Raw node — wrap with observability so node_start/node_complete fire
    if node.raw_fn is not None:
        return _make_raw_wrapper(node)

    # Validate scripted registration early
    if node.mode == "scripted" and node.scripted_fn not in registry.scripted:
        raise ConfigurationError.build(
            f"Scripted function '{node.scripted_fn}' not registered",
            hint="Use register_scripted() to register it before compilation",
            node=node.name,
        )

    dispatch = _dispatch_for_mode(node)
    field_name = field_name_for(node.name)

    def node_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        return _execute_node(node, state, config, dispatch)

    node_wrapper.__name__ = field_name
    return node_wrapper


def _make_raw_wrapper(node: Node) -> Callable:
    """Wrap a raw_fn dispatch with observability (node_start/node_complete).

    Only used for explicit ``mode='raw'`` escape-hatch nodes. Raw nodes
    bypass the unified _execute_node path — no DI/input/output wrapping,
    only logging.
    """
    assert node.raw_fn is not None, f"node '{node.name}' has mode='raw' but no raw_fn"
    raw_fn = node.raw_fn
    field_name = field_name_for(node.name)

    def raw_node_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        node_log = log.bind(node=node.name, mode="raw")
        node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))
        t0 = time.monotonic()

        result = raw_fn(state, config)

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return result

    raw_node_wrapper.__name__ = field_name
    return raw_node_wrapper


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
        # Iterate in reverse so later pipeline nodes take precedence
        # (e.g., loop output over seed). Unwrap append-lists.
        if input_data is None and sub.input is not None:
            sub_input_type = sub.input
            for attr_name in reversed(bus.keys()):
                val = bus.get(attr_name)
                check_val = val
                if isinstance(val, list) and val:
                    check_val = val[-1]
                if check_val is not None and isinstance(check_val, sub_input_type):
                    input_data = check_val
                    break

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
        # Iterate in reverse so later pipeline nodes take precedence.
        # Unwrap loop append-lists: Loop nodes have list[T] from the
        # append-list reducer; check val[-1] against the output type.
        output_val = None
        if sub.output is not None:
            sub_output_type = sub.output
            for val in reversed(list(sub_result.values())):
                check_val = val
                if isinstance(val, list) and val:
                    check_val = val[-1]
                if isinstance(check_val, sub_output_type):
                    output_val = check_val
                    break

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
