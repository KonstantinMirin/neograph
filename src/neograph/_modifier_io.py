"""Modifier-aware input extraction and skip-when handling."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, assert_never
from typing import get_origin as _get_origin

from structlog.stdlib import BoundLogger

from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._state_bus import StateBus
from neograph._state_io import _build_state_update
from neograph.di import _isinstance_safe as _is_instance_safe
from neograph.di import _unwrap_each_dict, _unwrap_loop_value
from neograph.errors import ExecutionError
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.naming import field_name_for
from neograph.node import Node


def _apply_skip_when(
    node: Node,
    input_data: Any,
    field_name: str,
    t0: float,
    node_log: BoundLogger,
    state: StateBus | None = None,
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
        current_count = (state.get(count_field) if state is not None else None) or 0
        update[count_field] = current_count + 1
    return update


class InputShape(Enum):
    """Classification of how a node reads its input from state."""

    NONE = "none"
    LOOP_REENTRY = "loop_reentry"
    EACH_ITEM = "each_item"
    FAN_IN_DICT = "fan_in_dict"
    SINGLE_TYPE = "single_type"


def _classify_input_shape(state: StateBus, node: Node) -> InputShape:
    """Determine which extraction strategy applies. Priority order matters."""
    if node.inputs is None:
        return InputShape.NONE

    combo, _ = classify_modifiers(node)
    if combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR):
        own_field = field_name_for(node.name)
        no = normalize_outputs(node.outputs)
        if no.is_dict_form:
            own_field = f"{own_field}_{no.primary_key}"
        own_val = state.get(own_field)
        if isinstance(own_val, list) and own_val:
            return InputShape.LOOP_REENTRY

    replicate_item = state.get("neo_each_item")
    if replicate_item is not None and _is_instance_safe(replicate_item, node.inputs):
        return InputShape.EACH_ITEM

    if normalize_inputs(node.inputs).is_dict_form:
        return InputShape.FAN_IN_DICT

    return InputShape.SINGLE_TYPE


def _extract_loop_reentry(state: StateBus, node: Node) -> Any:
    """Read from the node's own append-list on loop iteration 1+."""
    own_field = field_name_for(node.name)
    no_out = normalize_outputs(node.outputs)
    if no_out.is_dict_form:
        own_field = f"{own_field}_{no_out.primary_key}"
    own_val = state.get(own_field)
    latest = own_val[-1]  # type: ignore[index]

    ni = normalize_inputs(node.inputs)
    if not ni.is_dict_form:
        return latest

    by_name = ni.by_name
    # Single-key dict: always self-reference
    if len(by_name) == 1:
        first_key = next(iter(by_name))
        return {first_key: latest}

    # Multi-key dict: self-reference key gets latest, others read from state.
    result = {}
    node_own_field = field_name_for(node.name)
    placed_latest = False
    for key, expected_type in by_name.items():
        state_key = field_name_for(key)
        upstream_val = state.get(state_key)
        if upstream_val is not None and state_key != node_own_field:
            value = upstream_val
            if isinstance(value, list) and value and _get_origin(expected_type) is not list:
                value = value[-1]
            result[key] = value
        else:
            result[key] = latest
            placed_latest = True
    if not placed_latest:
        first_key = next(iter(by_name))
        result[first_key] = latest
    return result


def _extract_each_item(state: StateBus, node: Node) -> Any:
    """Read the fan-out item from neo_each_item."""
    return state.get("neo_each_item")


def _extract_fan_in_dict(state: StateBus, node: Node) -> dict[str, Any]:
    """Read each named upstream from state by key."""
    ni = normalize_inputs(node.inputs)
    assert ni.is_dict_form
    result: dict[str, Any] = {}
    for input_name, expected_type in ni.by_name.items():
        if input_name == node.fan_out_param:
            value = state.get("neo_each_item")
        else:
            state_key = field_name_for(input_name)
            value = state.get(state_key)
            value = _unwrap_loop_value(value, expected_type)
            if (
                value is not None
                and _get_origin(expected_type) is list
                and isinstance(value, dict)
            ):
                value = list(value.values())
        result[input_name] = value
    return result


def _extract_single_type(state: StateBus, node: Node) -> Any:
    """Scan state fields for first value matching the node's input type."""
    for attr_name in state.keys():
        val = state.get(attr_name)
        val = _unwrap_loop_value(val, node.inputs)
        val = _unwrap_each_dict(val, node.inputs)
        if val is not None and _is_instance_safe(val, node.inputs):
            return val
    return None


def _extract_input(state: StateBus, node: Node) -> Any:
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
