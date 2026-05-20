"""State I/O helpers — building state-update dicts and injecting Oracle config."""

from __future__ import annotations

from typing import Any, assert_never

from langchain_core.runnables import RunnableConfig

from neograph._normalize import normalize_outputs
from neograph._state_bus import StateBus
from neograph.modifiers import Each, ModifierCombo, classify_modifiers
from neograph.node import Node


def _inject_oracle_config(state: StateBus, config: RunnableConfig) -> RunnableConfig:
    """Inject Oracle generator ID and model override into config if present.

    Reads neo_oracle_gen_id and neo_oracle_model from state, merges them
    into config['configurable']. Returns the original config unchanged
    when no oracle fields are present.
    """
    oracle_gen_id = state.get("neo_oracle_gen_id")
    if oracle_gen_id is None:
        return config
    configurable = config.get("configurable", {})
    extra = {"_generator_id": oracle_gen_id}
    oracle_model = state.get("neo_oracle_model")
    if oracle_model is not None:
        extra["_oracle_model"] = oracle_model
    return {**config, "configurable": {**configurable, **extra}}


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

    each_item = state.get("neo_each_item") if state is not None else None

    # Dict-form outputs: per-key state fields (neograph-1bp.3).
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

    # Loop modifier: increment iteration counter and optionally collect history.
    loop_mod = mods.get("loop")
    if loop_mod is not None:
        count_field = f"neo_loop_count_{field_name}"
        current_count = (state.get(count_field) if state is not None else None) or 0
        update[count_field] = current_count + 1
        if loop_mod.history:
            history_field = f"neo_loop_history_{field_name}"
            update[history_field] = result

    return update
