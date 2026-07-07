"""Runs one node invocation through the standard preamble → dispatch → postamble lifecycle."""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._dispatch import ModeDispatch, NodeInput
from neograph._input_shape import _extract_context, _extract_input
from neograph._normalize import normalize_inputs
from neograph._oracle import _inject_oracle_config
from neograph._sidecar import _get_param_res
from neograph._state_bus import adapt_state
from neograph._state_keys import StateKeys
from neograph._state_write import _apply_skip_when, _build_state_update
from neograph.describe_type import type_display_name
from neograph.di import DIKind
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


def _run_id_binds(config: RunnableConfig) -> dict[str, str]:
    """structlog bind kwargs carrying the per-run id, or ``{}`` when absent.

    The framework-minted ``RUN_ID`` (config['configurable']) is a natural
    trace-correlation key. Surfaced here on the node lifecycle spans so every
    node_start/node_complete line for one run shares the same id. Omitted (not
    bound as ``None``) when the id is absent — e.g. a node invoked outside a
    run()/arun() driver — so a direct-invoke log line stays clean."""
    run_id = (config or {}).get("configurable", {}).get(StateKeys.RUN_ID)
    return {"run_id": run_id} if run_id else {}


def _inject_resource_manifest(
    state: BaseModel, node: Node, config: RunnableConfig
) -> RunnableConfig:
    """Stash the checkpointed resource manifest onto config for a ref-hydrating node.

    A ``FromResource(ref=<kind>)`` binding hydrates a ``ResourceRef`` looked up
    from the manifest, which lives in the CHECKPOINTED state channel(s) — not in
    config. The scripted-shim / di_inputs resolution seams see only
    ``(input_data, config)``, not full state, so this collects every
    ``neo_resource_manifest_*`` channel's refs off ``state`` and injects the merged
    list under ``StateKeys.RESOURCE_MANIFEST_INJECT`` (copy-not-mutate, mirroring
    ``_inject_oracle_config`` / ``_inject_di_inputs``). Only fires for a node that
    actually declares a ref binding — returns config unchanged otherwise, so it is
    zero-overhead for the common case (async path only; the sync driver fails loud
    on any FROM_RESOURCE binding before resolution). See neograph-a5nh.
    """
    param_res = _get_param_res(node)
    if not param_res:
        return config
    if not any(
        b.kind is DIKind.FROM_RESOURCE and b.ref_kind is not None
        for b in param_res.values()
    ):
        return config
    refs: list[Any] = []
    for fname in type(state).model_fields:
        if fname.startswith(StateKeys.RESOURCE_MANIFEST_PREFIX):
            val = getattr(state, fname, None)
            if val:
                refs.extend(val)
    configurable = config.get("configurable", {})
    return {
        **config,
        "configurable": {**configurable, StateKeys.RESOURCE_MANIFEST_INJECT: refs},
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
    node_log = log.bind(node=node.name, mode=node.mode, **_run_id_binds(config))
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


async def _aexecute_node(
    node: Node,
    state: BaseModel,
    config: RunnableConfig,
    dispatch: ModeDispatch,
) -> dict[str, Any]:
    """Async twin of :func:`_execute_node` (Phase 1a — driver-selected async).

    Structurally identical to the sync path: every pure preamble/postamble
    helper (adapt_state, _inject_oracle_config, _extract_input, _apply_skip_when,
    _extract_context, NodeInput shaping, _build_state_update, logging, timing) is
    reused VERBATIM. The ONLY divergence is the terminal call —
    ``await dispatch.aexecute(...)`` instead of ``dispatch.execute(...)`` — so the
    sync and async node paths cannot silently drift (Core Invariant).
    """
    field_name = field_name_for(node.name)
    node_log = log.bind(node=node.name, mode=node.mode, **_run_id_binds(config))
    node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))

    t0 = time.monotonic()

    bus = adapt_state(state)
    config = _inject_oracle_config(bus, config)
    config = _inject_resource_manifest(state, node, config)
    raw_input = _extract_input(bus, node)

    skip_result = _apply_skip_when(node, raw_input, field_name, t0, node_log, bus)
    if skip_result is not None:
        return skip_result

    context_data = _extract_context(bus, node) if node.mode != "scripted" else None

    if isinstance(raw_input, dict) and normalize_inputs(node.inputs).is_dict_form:
        node_input = NodeInput(fan_in=raw_input)
    else:
        node_input = NodeInput(single=raw_input)

    output = await dispatch.aexecute(node, node_input, config, context_data)
    update = _build_state_update(node, field_name, output.value, bus)

    elapsed = time.monotonic() - t0
    node_log.info("node_complete", duration_s=round(elapsed, 3))
    return update
