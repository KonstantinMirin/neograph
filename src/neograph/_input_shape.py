"""Read-side: classifies a node's input shape against state and extracts the typed input."""

from __future__ import annotations

from enum import Enum
from typing import Any, assert_never
from typing import get_origin as _get_origin

from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._state_bus import StateBus
from neograph._state_keys import StateKeys
from neograph.di import _isinstance_safe, _unwrap_each_dict, _unwrap_loop_value
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node


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
            assert no.primary_key is not None  # dict-form always has a primary key
            own_field = output_field_name(own_field, no.primary_key)
        # StateBus.get optional: loop-bootstrap — first router pass may have no
        # self-output yet; absence signals "iteration 0" and falls through.
        own_val = state.get(own_field)
        if isinstance(own_val, list) and own_val:
            return InputShape.LOOP_REENTRY

    # StateBus.get optional: framework — neo_each_item is absent for non-fan-out
    # nodes; absence is the documented signal.
    replicate_item = state.get(StateKeys.EACH_ITEM)
    if replicate_item is not None and _isinstance_safe(replicate_item, node.inputs):
        return InputShape.EACH_ITEM

    if normalize_inputs(node.inputs).is_dict_form:
        return InputShape.FAN_IN_DICT

    return InputShape.SINGLE_TYPE


def _extract_loop_reentry(state: StateBus, node: Node) -> Any:
    """Read from the node's own append-list on loop iteration 1+."""
    own_field = field_name_for(node.name)
    no_out = normalize_outputs(node.outputs)
    if no_out.is_dict_form:
        assert no_out.primary_key is not None  # dict-form always has a primary key
        own_field = output_field_name(own_field, no_out.primary_key)
    # REQUIRED: _classify_input_shape already confirmed own_val is non-empty list.
    own_val = state.get_required(own_field, node_label=node.name)
    latest = own_val[-1]

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
        # StateBus.get optional: loop-bootstrap — sibling keys may not have
        # been re-produced this iteration; documented sentinel for "use latest".
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
    # REQUIRED: dispatched only after classification confirmed EACH_ITEM presence.
    return state.get_required(StateKeys.EACH_ITEM, node_label=node.name)


def _extract_fan_in_dict(state: StateBus, node: Node) -> dict[str, Any]:
    """Read each named upstream from state by key.

    ``node.fan_out_param`` is set once at Construct construction (see
    ``neograph._ir_normalize.normalize_ir``) so all three API surfaces —
    declarative, ``@node``, programmatic/YAML — produce identical IR by
    the time the runtime sees the node.
    """
    ni = normalize_inputs(node.inputs)
    assert ni.is_dict_form
    result: dict[str, Any] = {}
    for input_name, expected_type in ni.by_name.items():
        if input_name == node.fan_out_param:
            # REQUIRED: node IS the fan-out target; EACH_ITEM is the dispatched value.
            value = state.get_required(StateKeys.EACH_ITEM, node_label=node.name)
        else:
            state_key = field_name_for(input_name)
            # REQUIRED: fan-in upstreams guaranteed by _validate_node_chain.
            value = state.get_required(state_key, node_label=node.name)
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
        # REQUIRED: iterating state.keys() — every key is by definition present.
        val = state.get_required(attr_name, node_label=node.name)
        val = _unwrap_loop_value(val, node.inputs)
        val = _unwrap_each_dict(val, node.inputs)
        if val is not None and _isinstance_safe(val, node.inputs):
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
