"""Oracle modifier factory helpers — redirect, merge, and Each wiring.

Extracted from factory.py. These functions create the
LangGraph node functions that implement Oracle and Each×Oracle modifier
wiring: redirect generators to collector fields, merge barrier functions,
and Each-keyed redirect wrappers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._llm_config import LlmConfig
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._state_bus import StateBus, adapt_state
from neograph._state_keys import StateKeys
from neograph.errors import ConfigurationError, ExecutionError
from neograph.modifiers import Each, Oracle
from neograph.naming import field_name_for
from neograph.node import HasName, TypeSpecStatic


def _inject_oracle_config(state: StateBus, config: RunnableConfig) -> RunnableConfig:
    """Inject Oracle generator ID and model override into config if present.

    Reads neo_oracle_gen_id and neo_oracle_model from state, merges them
    into config['configurable']. Returns the original config unchanged
    when no oracle fields are present.
    """
    # StateBus.get optional: framework — neo_oracle_gen_id only set inside
    # an Oracle fan-out dispatch; absence is the "not in Oracle" signal.
    oracle_gen_id = state.get(StateKeys.ORACLE_GEN_ID)
    if oracle_gen_id is None:
        return config
    configurable = config.get("configurable", {})
    extra = {"_generator_id": oracle_gen_id}
    # StateBus.get optional: framework — neo_oracle_model only present when
    # Oracle was configured with models=; legitimately absent otherwise.
    oracle_model = state.get(StateKeys.ORACLE_MODEL)
    if oracle_model is not None:
        extra["_oracle_model"] = oracle_model
    return {**config, "configurable": {**configurable, **extra}}


def make_oracle_redirect_fn(
    raw_fn: Callable, field_name: str, collector_field: str,
    item: HasName,
) -> Callable:
    """Wrap a node function to redirect output from field_name to collector_field.

    Used by Oracle generators: the node writes to the collector (list reducer)
    instead of the consumer-facing field.

    Handles both single-type outputs (result has field_name key) and dict-form
    outputs (result has {field_name}_{key} keys). For dict-form, collects the
    full result dict into the collector so the merge fn can process per-key.

    ``item`` carries the IR node/construct; ``item.name`` is the user-facing
    label. This factory does not currently raise StateMissingError, so it does
    not read ``item.name`` — the parameter is kept for signature parity with
    the other two redirect factories (a future get_required here asks
    ``item.name`` directly, never a threaded string nor ``raw_fn.__name__``).
    """
    prefix = f"{field_name}_"

    def oracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        result = raw_fn(state, config)
        val = result.get(field_name)
        if val is not None:
            return {collector_field: val}
        # Dict-form outputs: per-key fields like {field_name}_{key}
        if any(k.startswith(prefix) for k in result):
            return {collector_field: result}
        return result

    return oracle_redirect_fn


def make_eachoracle_redirect_fn(
    raw_fn: Callable, field_name: str, collector_field: str, each_key: str,
    item: HasName,
) -> Callable:
    """Wrap a node function for Each x Oracle fusion.

    Like make_oracle_redirect_fn, but tags each result with the each_key
    extracted from neo_each_item. The collector accumulates (key, result) tuples.

    ``item`` is the captured IR node/construct; ``item.name`` is the
    user-declared label surfaced in StateMissingError messages. The closure
    asks the IR object directly (Information Expert) rather than receiving a
    pre-extracted string.
    """
    prefix = f"{field_name}_"

    def eachoracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        result = raw_fn(state, config)
        # REQUIRED: flat Each×Oracle router always populates EACH_ITEM in the
        # Send payload. Absence = wiring bug.
        each_item = adapt_state(state).get_required(
            StateKeys.EACH_ITEM, node_label=item.name,
        )
        key = getattr(each_item, each_key, str(each_item))
        # Single-type outputs: result has {field_name: val}
        val = result.get(field_name)
        if val is not None:
            return {collector_field: [(key, val)]}
        # Dict-form outputs: result has per-key fields.
        # Collect the full per-key dict as the tagged value.
        if any(k.startswith(prefix) for k in result):
            per_key = {k[len(prefix):]: v for k, v in result.items() if k.startswith(prefix)}
            return {collector_field: [(key, per_key)]}
        return result

    return eachoracle_redirect_fn


def _unwrap_oracle_results(
    results: list,
    field_name: str,
    output_model: TypeSpecStatic,
) -> tuple[list, dict[str, list] | None]:
    """Unwrap collected Oracle results for the merge function.

    For single-type outputs: results is [val1, val2, ...] -- return as-is.
    For dict-form outputs: results is [{field_result: v1, field_meta: m1}, ...].
    Extract primary values for the merge fn, collect secondary values.

    Returns (primary_values, secondary_by_key) where secondary_by_key is None
    for single-type outputs.
    """
    if not results or not isinstance(results[0], dict):
        return results, None

    # Dict-form: extract primary (first key) for merge, collect secondaries
    prefix = f"{field_name}_"
    primary_key = None
    secondary_keys: list[str] = []

    if isinstance(output_model, dict):
        keys = list(output_model)
        primary_key = f"{prefix}{keys[0]}"
        secondary_keys = [f"{prefix}{k}" for k in keys[1:]]
    else:
        # Fallback: find keys by prefix
        for k in results[0]:
            if k.startswith(prefix):
                if primary_key is None:
                    primary_key = k
                else:
                    secondary_keys.append(k)

    if primary_key is None:
        return results, None

    primary_values = [r.get(primary_key) for r in results if r.get(primary_key) is not None]
    secondaries = {k: [r.get(k) for r in results if r.get(k) is not None] for k in secondary_keys}
    return primary_values, secondaries


def _build_oracle_merge_result(
    merged: Any,
    field_name: str,
    output_model: TypeSpecStatic,
    secondaries: dict[str, list] | None,
) -> dict:
    """Build the state update dict after Oracle merge.

    For single-type: {field_name: merged}.
    For dict-form: {field_name_primary: merged, field_name_secondary: last_value, ...}.

    Validates the merge result against the expected output type. Raises
    ExecutionError if the merge_fn returns the wrong type -- catches silent
    garbage before it propagates through the pipeline.
    """
    if isinstance(output_model, dict):
        expected_type = cast(type, next(iter(output_model.values())))
    else:
        expected_type = cast(type, output_model)

    if not isinstance(merged, expected_type):
        raise ExecutionError.build(
            "Oracle merge_fn returned wrong type",
            expected=expected_type.__name__,
            found=type(merged).__name__,
            hint="The merge function must return an instance of the node's output type",
            node=field_name,
        )

    if secondaries is None:
        return {field_name: merged}

    prefix = f"{field_name}_"
    if isinstance(output_model, dict):
        primary_field = f"{prefix}{next(iter(output_model))}"
    else:
        primary_field = field_name

    update = {primary_field: merged}
    for key, values in secondaries.items():
        if values:
            update[key] = values[-1]  # take last variant's secondary value
    return update


def make_oracle_merge_fn(
    oracle: Oracle,
    field_name: str,
    collector_field: str,
    output_model: TypeSpecStatic,
    node_inputs: dict[str, TypeSpecStatic] | None = None,
    llm_config: LlmConfig | None = None,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
) -> Callable:
    """Create the merge barrier function for Oracle.

    If oracle.merge_prompt, calls invoke_structured (LLM judge).
    If oracle.merge_fn, calls the registered scripted function -- with
    FromInput/FromConfig DI if it was declared via ``@merge_fn``.
    Reads from collector_field, writes to field_name (or per-key fields
    for dict-form outputs).

    When *node_inputs* is provided (dict-form inputs from the node),
    upstream values are extracted from state and passed alongside the
    variant list as a dict: ``{"variants": [...], upstream_key: val, ...}``.

    Merge hooks (merge_prompt path only):
    - ``merge_pre_process(variants) -> dict``: replaces default input_data.
      When set, upstream context auto-injection is skipped.
    - ``merge_post_process(result, variants) -> result``: transforms the
      parsed LLM result. Only runs on LLM success (skipped when fallback fires).
    - ``merge_fallback(variants, error) -> result``: catches invoke_structured
      errors. Returns a deterministic result instead of raising.
    """
    _node_inputs = node_inputs if isinstance(node_inputs, dict) else None
    _llm_config = llm_config
    _runtime = runtime

    if oracle.merge_prompt:
        assert oracle.merge_prompt is not None  # narrowing for mypy
        _merge_prompt: str = oracle.merge_prompt
        _pre_process = oracle.merge_pre_process
        _post_process = oracle.merge_post_process
        _fallback = oracle.merge_fallback

        def merge_fn(state: Any, config: RunnableConfig) -> dict:
            from neograph._llm import invoke_structured

            results = getattr(state, collector_field, [])
            primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)

            # Pre-process hook replaces default input_data construction.
            # invoke_structured accepts BaseModel | dict[str, Any] | str.
            input_data: BaseModel | dict[str, Any] | str
            if _pre_process is not None:
                input_data = _pre_process(primary)
            else:
                input_data = {"variants": primary}
                if _node_inputs:
                    for key in _node_inputs:
                        val = getattr(state, field_name_for(key), None)
                        if val is not None:
                            input_data[key] = val

            used_fallback = False
            try:
                if isinstance(output_model, dict):
                    primary_output_model = cast(type[BaseModel], next(iter(output_model.values())))
                else:
                    primary_output_model = cast(type[BaseModel], output_model)
                merged = invoke_structured(
                    _runtime,
                    model_tier=oracle.merge_model,
                    prompt_template=_merge_prompt,
                    input_data=input_data,
                    output_model=primary_output_model,
                    config=config,
                    llm_config=_llm_config,
                )
            except Exception as exc:
                if _fallback is not None:
                    merged = _fallback(primary, exc)
                    used_fallback = True
                else:
                    raise

            if _post_process is not None and not used_fallback:
                merged = _post_process(merged, primary)

            return _build_oracle_merge_result(merged, field_name, output_model, secondaries)
    else:
        from neograph.decorators import _resolve_merge_args, get_merge_fn_metadata

        assert oracle.merge_fn is not None
        _merge_fn_name: str = oracle.merge_fn
        metadata = get_merge_fn_metadata(_merge_fn_name)
        if metadata is not None:
            user_fn, param_res = metadata

            def merge_fn(state: Any, config: RunnableConfig) -> dict:
                results = getattr(state, collector_field, [])
                primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)
                merged = user_fn(primary, *_resolve_merge_args(param_res, config, state))
                return _build_oracle_merge_result(merged, field_name, output_model, secondaries)
        else:
            per_compile = scripted_lookup or {}
            scripted_merge = per_compile.get(_merge_fn_name)
            if scripted_merge is None:
                raise ConfigurationError.build(
                    f"Scripted function '{_merge_fn_name}' not registered",
                    hint="Pass scripted={'" + _merge_fn_name + "': fn} to compile().",
                )

            def merge_fn(state: Any, config: RunnableConfig) -> dict:
                results = getattr(state, collector_field, [])
                primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)
                merged = scripted_merge(primary, config)
                return _build_oracle_merge_result(merged, field_name, output_model, secondaries)

    return merge_fn


def make_each_redirect_fn(
    raw_fn: Callable, field_name: str, each: Each,
    item: HasName,
) -> Callable:
    """Wrap a node function to key the result by the Each item's key field.

    Reads neo_each_item from state, uses each.key to extract the dispatch key.
    ``item`` is the captured IR node/construct; ``item.name`` is the
    user-declared label surfaced in StateMissingError messages — asked of the
    IR object directly, never threaded as a string.
    """

    def each_redirect_fn(state: Any, config: RunnableConfig = None) -> dict:  # type: ignore[assignment]
        # REQUIRED: Each router always populates EACH_ITEM in the Send payload.
        each_item = adapt_state(state).get_required(
            StateKeys.EACH_ITEM, node_label=item.name,
        )

        result = raw_fn(state, config) if config else raw_fn(state)
        val = result.get(field_name)

        if val is not None:
            key_val = getattr(each_item, each.key, str(each_item))
            return {field_name: {key_val: val}}
        return result

    return each_redirect_fn
