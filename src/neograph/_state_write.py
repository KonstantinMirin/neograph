"""Rules for writing state updates from node outputs — modifier-aware key wrapping (Each, Loop, Oracle-fusion), dict-form output projection, skip-when state writes."""

from __future__ import annotations

import time
from typing import Any, assert_never

from structlog.stdlib import BoundLogger

from neograph._normalize import normalize_outputs
from neograph._state_bus import StateBus
from neograph._state_keys import StateKeys
from neograph.describe_type import type_display_name
from neograph.errors import ExecutionError, NodeOutputError
from neograph.modifiers import Each, ModifierCombo, classify_modifiers
from neograph.naming import output_field_name
from neograph.node import Node


def _raise_none_output(
    node: Node, declared: object, field: str, *, key: str | None = None,
) -> None:
    """Fail loud when a node RAN and produced None against its declared output.

    Backstop for the single write boundary. ``declared`` is the
    declared output type (single-type or a dict-form key's type); ``field`` is
    the state field that stayed empty; ``key`` names the dict-form output key
    when applicable.
    """
    where = f" output key '{key}'" if key is not None else ""
    raise NodeOutputError.build(
        f"declared{where} output but the node body produced None",
        expected=type_display_name(declared),
        found="None",
        node=node.name,
        location=f"state field '{field}'",
        hint="A node that runs must return a value of its declared outputs= type. "
             "For a node that may legitimately produce nothing, use skip_when / an "
             "untaken branch arm (never-ran fields stay absent) rather than returning None.",
    )


def _build_state_update(
    node: Node,
    field_name: str,
    result: Any,
    state: StateBus | None,
) -> dict[str, Any]:
    """Build a state update dict, handling dict-form and single-type outputs.

    For dict-form outputs (``outputs={'a': A, 'b': B}``):
      - result must be a dict with matching keys
      - each key writes to ``{field_name}_{key}``
      - Each modifier wraps per-key

    For single-type outputs: writes to ``{field_name}`` as before.
    """
    if node.outputs is None:
        # No declared output contract — nothing to write, nothing to enforce.
        return {}
    if result is None:
        # RAN-AND-VIOLATED-CONTRACT: node executed but produced None against a
        # declared outputs= type. Fail loud. Never-ran / legitimately-absent
        # fields (untaken branches, skip_when without skip_value) never reach
        # here with a ran result.
        _raise_none_output(node, normalize_outputs(node.outputs).primary, field_name)

    combo, mods = classify_modifiers(node)

    # MODIFIER_RULE_TOUCHPOINT: Each / Each×Oracle fusion key-wrapping rule.
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

    # StateBus.get optional: framework — EACH_ITEM is absent for non-fan-out
    # dispatches; absence drives the "not inside fan-out" branches below.
    each_item = state.get(StateKeys.EACH_ITEM) if state is not None else None

    # MODIFIER_RULE_TOUCHPOINT: dict-form output projection + Each per-key wrap.
    no = normalize_outputs(node.outputs)
    if no.is_dict_form and isinstance(result, dict):
        update: dict[str, Any] = {}
        for key in no.all_keys:
            val = result.get(key)
            if val is None:
                if key == no.primary_key:
                    # Primary key None is the dict-form equivalent of the
                    # single-type contract violation — fail loud.
                    _raise_none_output(
                        node, no.primary, output_field_name(field_name, key), key=key,
                    )
                # Secondary keys (framework-collected, e.g. tool_log) are
                # demand-driven and legitimately absent — stay tolerant.
                continue
            key_field = output_field_name(field_name, key)
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

    # MODIFIER_RULE_TOUCHPOINT: Loop iteration counter + optional history collection.
    loop_mod = mods.get("loop")
    if loop_mod is not None:
        count_field = StateKeys.loop_count(field_name)
        # Counter bootstrap (absent/None -> 0) lives in StateBus.get_counter.
        current_count = state.get_counter(count_field) if state is not None else 0
        update[count_field] = current_count + 1
        if loop_mod.history:
            history_field = StateKeys.loop_history(field_name)
            update[history_field] = result

    return update


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
    # MODIFIER_RULE_TOUCHPOINT: Loop counter increment on no-skip-value path
    # so loop_router eventually exits when skip_when keeps firing.
    update: dict[str, Any] = {}
    _, skip_mods = classify_modifiers(node)
    loop_mod = skip_mods.get("loop")
    if loop_mod is not None:
        count_field = StateKeys.loop_count(field_name)
        # Counter bootstrap (absent/None -> 0) lives in StateBus.get_counter.
        current_count = state.get_counter(count_field) if state is not None else 0
        update[count_field] = current_count + 1
    return update
