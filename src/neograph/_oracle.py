"""Oracle modifier factory helpers — redirect, merge, and Each wiring.

Extracted from factory.py. These functions create the
LangGraph node functions that implement Oracle and Each×Oracle modifier
wiring: redirect generators to collector fields, merge barrier functions,
and Each-keyed redirect wrappers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from neograph.naming import field_name_for

from neograph.errors import ExecutionError
from neograph.modifiers import Each, Oracle


def _state_get(state: Any, key: str) -> Any:
    """Read a key from state — imported from factory at call time to avoid cycles."""
    from neograph.factory import _state_get as _fg
    return _fg(state, key)


def _lookup_scripted(name: str) -> Callable:
    from neograph.factory import lookup_scripted
    return lookup_scripted(name)


def make_oracle_redirect_fn(raw_fn: Callable, field_name: str, collector_field: str) -> Callable:
    """Wrap a node function to redirect output from field_name to collector_field.

    Used by Oracle generators: the node writes to the collector (list reducer)
    instead of the consumer-facing field.

    Handles both single-type outputs (result has field_name key) and dict-form
    outputs (result has {field_name}_{key} keys). For dict-form, collects the
    full result dict into the collector so the merge fn can process per-key.
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

    oracle_redirect_fn.__name__ = raw_fn.__name__
    return oracle_redirect_fn


def make_eachoracle_redirect_fn(
    raw_fn: Callable, field_name: str, collector_field: str, each_key: str,
) -> Callable:
    """Wrap a node function for Each x Oracle fusion.

    Like make_oracle_redirect_fn, but tags each result with the each_key
    extracted from neo_each_item. The collector accumulates (key, result) tuples.
    """
    prefix = f"{field_name}_"

    def eachoracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        result = raw_fn(state, config)
        # Extract the each_key from the item
        item = _state_get(state, "neo_each_item")
        key = getattr(item, each_key, str(item)) if item is not None else "unknown"
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

    eachoracle_redirect_fn.__name__ = raw_fn.__name__
    return eachoracle_redirect_fn


def _unwrap_oracle_results(
    results: list,
    field_name: str,
    output_model: Any,
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
    output_model: Any,
    secondaries: dict[str, list] | None,
) -> dict:
    """Build the state update dict after Oracle merge.

    For single-type: {field_name: merged}.
    For dict-form: {field_name_primary: merged, field_name_secondary: last_value, ...}.

    Validates the merge result against the expected output type. Raises
    ExecutionError if the merge_fn returns the wrong type -- catches silent
    garbage before it propagates through the pipeline.
    """
    expected_type = output_model
    if isinstance(output_model, dict):
        expected_type = next(iter(output_model.values()))

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
    output_model: Any,
    node_inputs: dict[str, Any] | None = None,
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
    """
    _node_inputs = node_inputs if isinstance(node_inputs, dict) else None

    if oracle.merge_prompt:
        assert oracle.merge_prompt is not None  # narrowing for mypy
        _merge_prompt: str = oracle.merge_prompt

        def merge_fn(state: Any, config: RunnableConfig) -> dict:
            from neograph._llm import invoke_structured

            results = getattr(state, collector_field, [])
            primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)

            # Build input dict: variants + upstream context from state
            input_data: dict[str, Any] = {"variants": primary}
            if _node_inputs:
                for key in _node_inputs:
                    val = getattr(state, field_name_for(key), None)
                    if val is not None:
                        input_data[key] = val

            merged = invoke_structured(
                model_tier=oracle.merge_model,
                prompt_template=_merge_prompt,
                input_data=input_data,
                output_model=output_model if not isinstance(output_model, dict) else next(iter(output_model.values())),
                config=config,
            )
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
            scripted_merge = _lookup_scripted(_merge_fn_name)

            def merge_fn(state: Any, config: RunnableConfig) -> dict:
                results = getattr(state, collector_field, [])
                primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)
                merged = scripted_merge(primary, config)
                return _build_oracle_merge_result(merged, field_name, output_model, secondaries)

    return merge_fn


def make_each_redirect_fn(raw_fn: Callable, field_name: str, each: Each) -> Callable:
    """Wrap a node function to key the result by the Each item's key field.

    Reads neo_each_item from state, uses each.key to extract the dispatch key.
    """

    def each_redirect_fn(state: Any, config: RunnableConfig = None) -> dict:  # type: ignore[assignment]
        # Get the item being processed
        each_item = _state_get(state, "neo_each_item")

        result = raw_fn(state, config) if config else raw_fn(state)
        val = result.get(field_name)

        if val is not None and each_item is not None:
            key_val = getattr(each_item, each.key, str(each_item))
            return {field_name: {key_val: val}}
        return result

    each_redirect_fn.__name__ = raw_fn.__name__ if hasattr(raw_fn, '__name__') else field_name
    return each_redirect_fn
