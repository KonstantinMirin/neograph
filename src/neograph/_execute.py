"""Runs one node invocation through the standard preamble → dispatch → postamble lifecycle."""

from __future__ import annotations

import time
from typing import Any, cast

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._dispatch import ModeDispatch, NodeInput
from neograph._input_shape import _extract_input
from neograph._normalize import normalize_inputs
from neograph._oracle import _inject_oracle_config
from neograph._state_bus import StateBus, adapt_state
from neograph._state_write import _apply_skip_when, _build_state_update
from neograph.describe_type import type_display_name
from neograph.naming import field_name_for
from neograph.node import Node, TypeSpecStatic

log = structlog.get_logger()


def _type_name(t: TypeSpecStatic) -> str | None:
    """Get a readable name from a type, or None (logging contract).

    Delegates rendering to ``type_display_name`` (the single source of truth);
    only the ``None -> None`` adaptation lives here, for structlog callers that
    omit the field when the type is absent.
    """
    if t is None:
        return None
    return type_display_name(t)


def _extract_context(state: StateBus, node: Node) -> dict[str, str] | None:
    """Extract verbatim context fields from state for LLM nodes.

    Returns a dict of {context_name: state_value} if the node declares
    context fields, or None if no context is configured. Context values
    are user-declared string fields rendered verbatim into prompts.
    """
    if not node.context:
        return None
    # REQUIRED: context fields are validator-guaranteed (see
    # _construct_validation.py); missing → wiring bug, fail loud rather than
    # render the literal string "None" into the LLM prompt.
    return {
        name: cast(str, state.get_required(field_name_for(name), node_label=node.name))
        for name in node.context
    }


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
