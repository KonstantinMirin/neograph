"""Generic node factory — creates LangGraph node functions from Node definitions.

All non-raw modes (scripted, think, agent, act) go through a single
``_execute_node`` path with mode-specific behavior injected via the
``ModeDispatch`` protocol:

    ScriptedDispatch — deterministic Python function
    ThinkDispatch    — single LLM call, structured JSON output
    ToolDispatch     — ReAct tool loop with tools (read-only or mutation)

Raw nodes (``mode='raw'``) get a minimal observability wrapper via
``_make_raw_wrapper`` — no DI/input/output wrapping.

Also provides higher-order factory functions for modifier wiring:
    make_subgraph_fn          — creates function to run a sub-Construct

Oracle/Each modifier helpers (make_oracle_redirect_fn, make_oracle_merge_fn,
make_each_redirect_fn, etc.) live in ``_oracle.py`` and are re-exported here
so that ``compiler.py`` imports remain unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from typing import get_origin as _get_origin

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
from neograph.di import _unwrap_each_dict, _unwrap_loop_value
from neograph.errors import ConfigurationError, ExecutionError
from neograph.modifiers import Each, ModifierCombo, classify_modifiers
from neograph.naming import field_name_for
from neograph.node import Node

log = structlog.get_logger()




def _state_get(state: Any, key: str) -> Any:
    """Read a key from state, handling both dict and Pydantic model forms."""
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


def _inject_oracle_config(state: Any, config: RunnableConfig) -> RunnableConfig:
    """Inject Oracle generator ID and model override into config if present.

    Reads neo_oracle_gen_id and neo_oracle_model from state, merges them
    into config['configurable']. Returns the original config unchanged
    when no oracle fields are present.
    """
    oracle_gen_id = _state_get(state, "neo_oracle_gen_id")
    if oracle_gen_id is None:
        return config
    configurable = config.get("configurable", {})
    extra = {"_generator_id": oracle_gen_id}
    oracle_model = _state_get(state, "neo_oracle_model")
    if oracle_model is not None:
        extra["_oracle_model"] = oracle_model
    return {**config, "configurable": {**configurable, **extra}}


def _extract_context(state: Any, node: Node) -> dict[str, Any] | None:
    """Extract verbatim context fields from state for LLM nodes.

    Returns a dict of {context_name: state_value} if the node declares
    context fields, or None if no context is configured.
    """
    if not node.context:
        return None
    return {
        name: _state_get(state, field_name_for(name))
        for name in node.context
    }


# Singleton registry instance — replaces former module-level dicts.
from neograph._registry import registry


def register_scripted(name: str, fn: Callable) -> None:
    """Register a deterministic function for Node.scripted."""
    registry.scripted[name] = fn


def register_condition(name: str, fn: Callable) -> None:
    """Register a condition function for Operator(when=...)."""
    registry.condition[name] = fn


def register_tool_factory(name: str, fn: Callable) -> None:
    """Register a tool factory that creates LangChain @tool functions."""
    registry.tool_factory[name] = fn


def lookup_condition(name: str) -> Callable:
    """Look up a registered condition function by name. Raises ConfigurationError if missing."""
    fn = registry.condition.get(name)
    if fn is None:
        raise ConfigurationError.build(
            f"Condition '{name}' not registered",
            hint="Use register_condition() to register it before compilation",
        )
    return fn


def lookup_scripted(name: str) -> Callable:
    """Look up a registered scripted function by name. Raises ConfigurationError if missing."""
    fn = registry.scripted.get(name)
    if fn is None:
        raise ConfigurationError.build(
            f"Scripted function '{name}' not registered",
            hint="Use register_scripted() to register it before compilation",
        )
    return fn


def _type_name(t: Any) -> str | None:
    """Get a readable name from a type, or None."""
    if t is None:
        return None
    if isinstance(t, dict):
        parts = ", ".join(
            f"{k}: {getattr(v, '__name__', str(v))}" for k, v in t.items()
        )
        return "{" + parts + "}"
    return getattr(t, '__name__', str(t))


def _apply_skip_when(
    node: Node,
    input_data: Any,
    field_name: str,
    t0: float,
    node_log: Any,
    state: Any = None,
) -> dict[str, Any] | None:
    """Check skip_when predicate and return early state update if skipped.

    Returns a state-update dict if the node should be skipped, or None if
    execution should continue.  Unwraps single-key dicts so skip_when
    receives a typed value for single-upstream nodes (consistent across
    @node and Node() surfaces).

    When the node has an Each modifier, the skip_value result is routed
    through ``_build_state_update`` so it gets wrapped in the dispatch key
    dict (``{key: value}``) that the ``_merge_dicts`` reducer expects.
    """
    if node.skip_when is None:
        return None
    skip_input = input_data
    if isinstance(input_data, dict) and len(input_data) == 1:
        skip_input = next(iter(input_data.values()))
    try:
        should_skip = node.skip_when(skip_input)
    except (AttributeError, TypeError, KeyError) as exc:
        raise ExecutionError.build(
            f"skip_when raised {type(exc).__name__}: {exc}",
            hint="Check that the lambda accesses valid fields on the input type",
            node=node.name,
        ) from exc
    if not should_skip:
        return None
    elapsed = time.monotonic() - t0
    node_log.info("node_skipped", reason="skip_when", duration_s=round(elapsed, 3))
    if node.skip_value is not None:
        result = node.skip_value(skip_input)
        return _build_state_update(node, field_name, result, state)
    # No skip_value — still need to increment the loop counter so the
    # loop_router eventually exits.
    update: dict[str, Any] = {}
    _, skip_mods = classify_modifiers(node)
    loop_mod = skip_mods.get("loop")
    if loop_mod is not None:
        count_field = f"neo_loop_count_{field_name}"
        current_count = _state_get(state, count_field) or 0
        update[count_field] = current_count + 1
    return update




def _build_state_update(
    node: Node,
    field_name: str,
    result: Any,
    state: Any,
) -> dict[str, Any]:
    """Build a state update dict, handling dict-form and single-type outputs.

    For dict-form outputs (``outputs={'a': A, 'b': B}``):
      - result must be a dict with matching keys
      - each key writes to ``{field_name}_{key}``
      - Each modifier wraps per-key

    For single-type outputs: writes to ``{field_name}`` as before.
    """
    from typing import assert_never

    if result is None or node.outputs is None:
        return {}

    combo, mods = classify_modifiers(node)

    # Determine Each wrapping behavior.
    # Skip Each wrapping when Oracle is also present — the Each×Oracle
    # fusion handles tagging in the redirect_fn.
    each_mod: Each | None = None
    match combo:
        case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
            each_mod = mods["each"]
        case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
            each_mod = None  # fusion handles tagging
        case ModifierCombo.BARE | ModifierCombo.OPERATOR | ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR | ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
            each_mod = None
        case _ as unreachable:
            assert_never(unreachable)

    each_item = _state_get(state, "neo_each_item")

    # Dict-form outputs: per-key state fields (neograph-1bp.3).
    if isinstance(node.outputs, dict) and isinstance(result, dict):
        update: dict[str, Any] = {}
        for key in node.outputs:
            val = result.get(key)
            if val is None:
                continue
            key_field = f"{field_name}_{key}"
            if each_mod and each_item is not None:
                key_val = getattr(each_item, each_mod.key, str(each_item))
                update[key_field] = {key_val: val}
            else:
                update[key_field] = val
    else:
        # Single-type outputs (backward compat).
        if each_mod and each_item is not None:
            key_val = getattr(each_item, each_mod.key, str(each_item))
            update = {field_name: {key_val: result}}
        else:
            update = {field_name: result}

    # Loop modifier: increment iteration counter and optionally collect history.
    loop_mod = mods.get("loop")
    if loop_mod is not None:
        count_field = f"neo_loop_count_{field_name}"
        current_count = _state_get(state, count_field) or 0
        update[count_field] = current_count + 1
        if loop_mod.history:
            history_field = f"neo_loop_history_{field_name}"
            update[history_field] = result

    return update




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

    config = _inject_oracle_config(state, config)
    raw_input = _extract_input(state, node)

    skip_result = _apply_skip_when(node, raw_input, field_name, t0, node_log, state)
    if skip_result is not None:
        return skip_result

    context_data = _extract_context(state, node) if node.mode != "scripted" else None

    if isinstance(raw_input, dict) and isinstance(node.inputs, dict):
        node_input = NodeInput(fan_in=raw_input)
    else:
        node_input = NodeInput(single=raw_input)

    output = dispatch.execute(node, node_input, config, context_data)
    update = _build_state_update(node, field_name, output.value, state)

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


from neograph.di import _isinstance_safe as _is_instance_safe  # noqa: E402


# ── Input shape dispatch ───────────────────────────────────────────────────
# _extract_input classifies the input shape and dispatches to a named helper.
# Same pattern as classify_modifiers: enum + match + assert_never.

from enum import Enum
from typing import assert_never


class InputShape(Enum):
    """Classification of how a node reads its input from state."""

    NONE = "none"
    LOOP_REENTRY = "loop_reentry"
    EACH_ITEM = "each_item"
    FAN_IN_DICT = "fan_in_dict"
    SINGLE_TYPE = "single_type"


def _classify_input_shape(state: Any, node: Node) -> InputShape:
    """Determine which extraction strategy applies. Priority order matters."""
    if node.inputs is None:
        return InputShape.NONE

    combo, _ = classify_modifiers(node)
    if combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR):
        own_field = field_name_for(node.name)
        if isinstance(node.outputs, dict):
            primary_key = next(iter(node.outputs))
            own_field = f"{own_field}_{primary_key}"
        own_val = _state_get(state, own_field)
        if isinstance(own_val, list) and own_val:
            return InputShape.LOOP_REENTRY

    replicate_item = _state_get(state, "neo_each_item")
    if replicate_item is not None and _is_instance_safe(replicate_item, node.inputs):
        return InputShape.EACH_ITEM

    if isinstance(node.inputs, dict):
        return InputShape.FAN_IN_DICT

    return InputShape.SINGLE_TYPE


def _extract_loop_reentry(state: Any, node: Node) -> Any:
    """Read from the node's own append-list on loop iteration 1+."""
    own_field = field_name_for(node.name)
    if isinstance(node.outputs, dict):
        primary_key = next(iter(node.outputs))
        own_field = f"{own_field}_{primary_key}"
    own_val = _state_get(state, own_field)
    latest = own_val[-1]  # type: ignore[index]

    if not isinstance(node.inputs, dict):
        return latest

    # Single-key dict: always self-reference
    if len(node.inputs) == 1:
        first_key = next(iter(node.inputs))
        return {first_key: latest}

    # Multi-key dict: self-reference key gets latest, others read from state.
    result = {}
    node_own_field = field_name_for(node.name)
    placed_latest = False
    for key, expected_type in node.inputs.items():
        state_key = field_name_for(key)
        upstream_val = _state_get(state, state_key)
        if upstream_val is not None and state_key != node_own_field:
            value = upstream_val
            if isinstance(value, list) and value and _get_origin(expected_type) is not list:
                value = value[-1]
            result[key] = value
        else:
            result[key] = latest
            placed_latest = True
    if not placed_latest:
        first_key = next(iter(node.inputs))
        result[first_key] = latest
    return result


def _extract_each_item(state: Any, node: Node) -> Any:
    """Read the fan-out item from neo_each_item."""
    return _state_get(state, "neo_each_item")


def _extract_fan_in_dict(state: Any, node: Node) -> dict[str, Any]:
    """Read each named upstream from state by key."""
    assert isinstance(node.inputs, dict)
    result: dict[str, Any] = {}
    for input_name, expected_type in node.inputs.items():
        if input_name == node.fan_out_param:
            value = _state_get(state, "neo_each_item")
        else:
            state_key = field_name_for(input_name)
            value = _state_get(state, state_key)
            value = _unwrap_loop_value(value, expected_type)
            if (
                value is not None
                and _get_origin(expected_type) is list
                and isinstance(value, dict)
            ):
                value = list(value.values())
        result[input_name] = value
    return result


def _extract_single_type(state: Any, node: Node) -> Any:
    """Scan state fields for first value matching the node's input type."""
    if isinstance(state, dict):
        fields = list(state.keys())
    else:
        fields = list(state.__class__.model_fields.keys())

    for attr_name in fields:
        val = _state_get(state, attr_name)
        val = _unwrap_loop_value(val, node.inputs)
        val = _unwrap_each_dict(val, node.inputs)
        if val is not None and _is_instance_safe(val, node.inputs):
            return val
    return None


def _extract_input(state: Any, node: Node) -> Any:
    """Extract typed input from state — pure dispatch to shape helpers."""
    shape = _classify_input_shape(state, node)
    match shape:
        case InputShape.NONE:
            return None
        case InputShape.LOOP_REENTRY:
            return _extract_loop_reentry(state, node)
        case InputShape.EACH_ITEM:
            return _extract_each_item(state, node)
        case InputShape.FAN_IN_DICT:
            return _extract_fan_in_dict(state, node)
        case InputShape.SINGLE_TYPE:
            return _extract_single_type(state, node)
    assert_never(shape)


# ── Oracle & Each modifier wiring (extracted to _oracle.py) ──────────────
# Re-exported so compiler.py imports remain unchanged.
from neograph._oracle import (  # noqa: E402, F401
    _build_oracle_merge_result,
    _unwrap_oracle_results,
    make_each_redirect_fn,
    make_eachoracle_redirect_fn,
    make_oracle_merge_fn,
    make_oracle_redirect_fn,
)


def make_subgraph_fn(sub: Any, sub_graph: Any) -> Callable:
    """Create a function that runs a sub-Construct in isolation.

    Extracts input from parent state by type, runs sub_graph,
    extracts output by type, returns {field_name: output}.
    """
    from neograph.runner import _strip_internals

    sub_log = log.bind(subgraph=sub.name)
    field_name = field_name_for(sub.name)

    sub_combo, _ = classify_modifiers(sub)
    has_loop = sub_combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR)

    def subgraph_node(state: Any, config: RunnableConfig) -> dict:
        sub_log.info("subgraph_start")

        # Loop re-entry: on iteration 2+, read from own append-list.
        # When output type matches input type (classic refine pattern),
        # feed the output back as input.  When output differs from input
        # (produce+validate pattern), skip the shortcut and re-read
        # original inputs from parent state.
        input_data = None
        if has_loop:
            own_val = _state_get(state, field_name)
            if isinstance(own_val, list) and own_val:
                latest = own_val[-1]
                if sub.input is None or isinstance(latest, sub.input):
                    input_data = latest

        # First iteration, non-loop, or input!=output loop re-entry:
        # extract input by type from parent state.
        # Iterate in reverse so later pipeline nodes take precedence
        # (e.g., loop output over seed). Unwrap append-lists.
        if input_data is None:
            if isinstance(state, dict):  # pragma: no cover — state is always a Pydantic model
                for val in reversed(list(state.values())):
                    check_val = val
                    if isinstance(val, list) and val:
                        check_val = val[-1]
                    if check_val is not None and isinstance(check_val, sub.input):
                        input_data = check_val
                        break
            else:
                for attr_name in reversed(list(state.__class__.model_fields)):
                    val = getattr(state, attr_name, None)
                    check_val = val
                    if isinstance(val, list) and val:
                        check_val = val[-1]
                    if check_val is not None and isinstance(check_val, sub.input):
                        input_data = check_val
                        break

        # Run sub-graph with isolated state
        sub_input: dict[str, Any] = {"node_id": state.get("node_id", "") if isinstance(state, dict) else getattr(state, "node_id", "")}
        if input_data is not None:
            sub_input["neo_subgraph_input"] = input_data

        # Forward context fields from parent state into sub-construct
        for n in sub.nodes:
            if hasattr(n, "context") and n.context:
                for ctx_name in n.context:
                    ctx_field = field_name_for(ctx_name)
                    val = _state_get(state, ctx_field)
                    if val is not None:
                        sub_input[ctx_field] = val

        # Forward Oracle gen_id + model override from parent state into config
        config = _inject_oracle_config(state, config)

        sub_result = _strip_internals(sub_graph.invoke(sub_input, config=config))

        # Extract the declared output type from sub result.
        # Iterate in reverse so later pipeline nodes take precedence.
        # Unwrap loop append-lists: Loop nodes have list[T] from the
        # append-list reducer; check val[-1] against the output type.
        output_val = None
        for val in reversed(list(sub_result.values())):
            check_val = val
            if isinstance(val, list) and val:
                check_val = val[-1]
            if isinstance(check_val, sub.output):
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
            current = _state_get(state, count_field) or 0
            update[count_field] = current + 1
        return update

    subgraph_node.__name__ = field_name
    return subgraph_node
