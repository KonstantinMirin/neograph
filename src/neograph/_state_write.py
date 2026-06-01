"""Rules for writing state updates from node outputs — modifier-aware key wrapping (Each, Loop, Oracle-fusion), dict-form output projection, skip-when state writes."""

from __future__ import annotations

import time
from typing import Any, assert_never

from structlog.stdlib import BoundLogger

from neograph._normalize import normalize_outputs
from neograph._state_bus import StateBus
from neograph._state_keys import StateKeys
from neograph.errors import ExecutionError
from neograph.modifiers import Each, ModifierCombo, classify_modifiers
from neograph.node import Node


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
    if result is None or node.outputs is None:
        return {}

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

    # MODIFIER_RULE_TOUCHPOINT: Loop iteration counter + optional history collection.
    loop_mod = mods.get("loop")
    if loop_mod is not None:
        count_field = StateKeys.loop_count(field_name)
        # StateBus.get optional: loop-counter — absent before first iteration;
        # `or 0` is the documented bootstrap value.
        current_count = (state.get(count_field) if state is not None else None) or 0
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
        # StateBus.get optional: loop-counter — same justification as above.
        current_count = (state.get(count_field) if state is not None else None) or 0
        update[count_field] = current_count + 1
    return update
