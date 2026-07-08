"""Oracle modifier factory helpers — redirect, merge, and Each wiring.

Extracted from factory.py. These functions create the
LangGraph node functions that implement Oracle and Each×Oracle modifier
wiring: redirect generators to collector fields, merge barrier functions,
and Each-keyed redirect wrappers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel

from neograph._config_carrier import _with_configurable
from neograph._di_classify import _resolve_merge_args
from neograph._llm_config import LlmConfig
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import normalize_outputs, primary_output_field
from neograph._sidecar import get_merge_fn_metadata
from neograph._state_bus import StateBus, adapt_state
from neograph._state_keys import StateKeys
from neograph.errors import ConfigurationError, ExecutionError
from neograph.modifiers import Each, EachFailure, Oracle
from neograph.naming import field_name_for, output_field_name, split_output_field
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
    extra = {"_generator_id": oracle_gen_id}
    # StateBus.get optional: framework — neo_oracle_model only present when
    # Oracle was configured with models=; legitimately absent otherwise.
    oracle_model = state.get(StateKeys.ORACLE_MODEL)
    if oracle_model is not None:
        extra[StateKeys.ORACLE_MODEL_OVERRIDE] = oracle_model
    return _with_configurable(config, **extra)


def make_oracle_redirect_fn(
    raw_fn: Runnable,
    field_name: str,
    collector_field: str,
    item: HasName,
) -> Runnable:
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

    def _project(result: dict) -> dict:
        val = result.get(field_name)
        if val is not None:
            return {collector_field: val}
        # Dict-form outputs: per-key fields like {field_name}_{key}
        if any(split_output_field(k, field_name) is not None for k in result):
            return {collector_field: result}
        return result

    def oracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        return _project(raw_fn.invoke(state, config))

    async def aoracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        return _project(await raw_fn.ainvoke(state, config))

    # Dual-path: the gen node the redirect wraps has its own sync/async twins;
    # under ainvoke we MUST await raw_fn.ainvoke() or an Oracle-wrapped LLM node
    # would be threadpooled and block the loop (review H2). Shared _project()
    # keeps sync/async post-processing from drifting.
    return RunnableLambda(oracle_redirect_fn, afunc=aoracle_redirect_fn)


def make_eachoracle_redirect_fn(
    raw_fn: Runnable,
    field_name: str,
    collector_field: str,
    each_key: str,
    item: HasName,
) -> Runnable:
    """Wrap a node function for Each x Oracle fusion.

    Like make_oracle_redirect_fn, but tags each result with the each_key
    extracted from neo_each_item. The collector accumulates (key, result) tuples.

    ``item`` is the captured IR node/construct; ``item.name`` is the
    user-declared label surfaced in StateMissingError messages. The closure
    asks the IR object directly (Information Expert) rather than receiving a
    pre-extracted string.
    """

    def _project(state: Any, result: dict) -> dict:
        # REQUIRED: flat Each×Oracle router always populates EACH_ITEM in the
        # Send payload. Absence = wiring bug.
        each_item = adapt_state(state).get_required(
            StateKeys.EACH_ITEM,
            node_label=item.name,
        )
        key = getattr(each_item, each_key, str(each_item))
        # Single-type outputs: result has {field_name: val}
        val = result.get(field_name)
        if val is not None:
            return {collector_field: [(key, val)]}
        # Dict-form outputs: result has per-key fields.
        # Collect the full per-key dict as the tagged value.
        per_key = {ok: v for k, v in result.items() if (ok := split_output_field(k, field_name)) is not None}
        if per_key:
            return {collector_field: [(key, per_key)]}
        return result

    def eachoracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        return _project(state, raw_fn.invoke(state, config))

    async def aeachoracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        return _project(state, await raw_fn.ainvoke(state, config))

    return RunnableLambda(eachoracle_redirect_fn, afunc=aeachoracle_redirect_fn)


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
    primary_key = None
    secondary_keys: list[str] = []

    if isinstance(output_model, dict):
        keys = list(output_model)
        primary_key = output_field_name(field_name, keys[0])
        secondary_keys = [output_field_name(field_name, k) for k in keys[1:]]
    else:
        # Fallback: find per-key fields by the {field_name}_ convention
        for k in results[0]:
            if split_output_field(k, field_name) is not None:
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
    expected_type = cast(type, normalize_outputs(output_model).primary)

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

    primary_field = primary_output_field(field_name, output_model)

    update = {primary_field: merged}
    for key, values in secondaries.items():
        if values:
            update[key] = values[-1]  # take last variant's secondary value
    return update


def _build_upstream_context(bus: StateBus, node_inputs: dict[str, TypeSpecStatic] | None) -> dict[str, Any]:
    """Extract upstream node-input values from state for merge-prompt injection.

    Returns ``{key: value}`` for each ``node_inputs`` key whose state field
    (``field_name_for(key)``) is present and non-None. This is the SINGLE source
    of the upstream-context rule (``field_name_for`` + None-skip), shared by the
    standard Oracle merge (``make_oracle_merge_fn``) and the Each×Oracle fused
    merge (``_wiring.group_merge_barrier``) so the two paths cannot diverge.

    Reads go through the StateBus; callers ``adapt_state(state)`` first.
    """
    ctx: dict[str, Any] = {}
    if isinstance(node_inputs, dict):
        for key in node_inputs:
            # StateBus.get optional: upstream-context is best-effort — an
            # absent/None upstream field is skipped, not required.
            val = bus.get(field_name_for(key))
            if val is not None:
                ctx[key] = val
    return ctx


def _run_merge_prompt(
    oracle: Oracle,
    variants: list,
    output_model: TypeSpecStatic,
    config: RunnableConfig,
    *,
    upstream_context: dict[str, Any] | None = None,
    llm_config: LlmConfig | None = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
) -> Any:
    """Run the merge_prompt (LLM-judge) merge step on a variants list.

    Owns the pre_process-vs-default input_data branch: a ``merge_pre_process``
    hook fully replaces input_data (and upstream context is intentionally NOT
    injected); otherwise input_data is ``{"variants": variants, **upstream}``.
    Applies merge_fallback (on invoke_structured error) and merge_post_process
    (on success). Returns the merged value.

    Sync twin of :func:`_arun_merge_prompt`. Both share the pure input build
    (``_merge_prompt_input``) and post-process (``_merge_prompt_post``); only the
    LLM seam differs (invoke_structured vs await ainvoke_structured), mirroring
    _llm.invoke_structured / ainvoke_structured. See neograph-p3c7.
    """
    from neograph._llm import invoke_structured

    input_data, primary_output_model = _merge_prompt_input(
        oracle,
        variants,
        output_model,
        upstream_context,
    )
    used_fallback = False
    try:
        merged = invoke_structured(
            runtime,
            model_tier=oracle.merge_model,
            prompt_template=oracle.merge_prompt,
            input_data=input_data,
            output_model=primary_output_model,
            config=config,
            llm_config=llm_config,
        )
    except Exception as exc:  # noqa: BLE001 — _merge_fallback_or_reraise re-raises bubble-ups/no-fallback
        merged = _merge_fallback_or_reraise(oracle, variants, exc)
        used_fallback = True
    return _merge_prompt_post(oracle, variants, merged, used_fallback)


def _merge_prompt_input(
    oracle: Oracle,
    variants: list,
    output_model: TypeSpecStatic,
    upstream_context: dict[str, Any] | None,
) -> tuple[Any, type[BaseModel]]:
    """Pure pre-seam: build the merge LLM input_data + primary output model.

    Shared by the sync/async merge_prompt twins so the pre_process-vs-default
    input branch cannot drift.
    """
    assert oracle.merge_prompt is not None  # narrowing — caller guards on it
    input_data: BaseModel | dict[str, Any] | str
    if oracle.merge_pre_process is not None:
        input_data = oracle.merge_pre_process(variants)
    else:
        input_data = {"variants": variants}
        if upstream_context:
            input_data.update(upstream_context)
    primary_output_model = cast(type[BaseModel], normalize_outputs(output_model).primary)
    return input_data, primary_output_model


def _merge_prompt_post(oracle: Oracle, variants: list, merged: Any, used_fallback: bool) -> Any:
    """Pure post-seam: apply merge_post_process on success. Shared by the twins."""
    if oracle.merge_post_process is not None and not used_fallback:
        merged = oracle.merge_post_process(merged, variants)
    return merged


def _merge_fallback_or_reraise(oracle: Oracle, variants: list, exc: Exception) -> Any:
    """Except-branch shared by the sync/async merge_prompt twins.

    Re-raises LangGraph control-flow signals (``GraphBubbleUp``: HITL interrupt /
    ``Command`` routing / cancellation) so they are NEVER captured into
    ``merge_fallback`` — mirrors the guard ``make_each_redirect_fn`` already has.
    Otherwise applies ``merge_fallback`` if set, or re-raises the original error.
    Callers set ``used_fallback=True`` only when this returns (it raises when it
    does not fall back).
    """
    if isinstance(exc, GraphBubbleUp):
        raise exc
    if oracle.merge_fallback is None:
        raise exc
    return oracle.merge_fallback(variants, exc)


async def _arun_merge_prompt(
    oracle: Oracle,
    variants: list,
    output_model: TypeSpecStatic,
    config: RunnableConfig,
    *,
    upstream_context: dict[str, Any] | None = None,
    llm_config: LlmConfig | None = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
) -> Any:
    """Async twin of :func:`_run_merge_prompt` per neograph-p3c7.

    Awaits ``_llm.ainvoke_structured`` so an LLM-judge merge under the async
    driver runs on the loop instead of blocking it. Shares the pure input build
    and post-process with the sync twin; only the seam differs.
    """
    from neograph._llm import ainvoke_structured

    input_data, primary_output_model = _merge_prompt_input(
        oracle,
        variants,
        output_model,
        upstream_context,
    )
    used_fallback = False
    try:
        merged = await ainvoke_structured(
            runtime,
            model_tier=oracle.merge_model,
            prompt_template=oracle.merge_prompt,
            input_data=input_data,
            output_model=primary_output_model,
            config=config,
            llm_config=llm_config,
        )
    except Exception as exc:  # noqa: BLE001 — _merge_fallback_or_reraise re-raises bubble-ups/no-fallback
        merged = _merge_fallback_or_reraise(oracle, variants, exc)
        used_fallback = True
    return _merge_prompt_post(oracle, variants, merged, used_fallback)


def _run_merge_fn(
    oracle: Oracle,
    variants: list,
    config: RunnableConfig,
    *,
    scripted_lookup: dict[str, Callable] | None = None,
    state_for_di: Any = None,
) -> Any:
    """Run the scripted/``@merge_fn`` merge step on a variants list.

    For ``@merge_fn``-declared functions, resolves DI + from_state params via
    ``_resolve_merge_args`` (``state_for_di`` supplies from_state values).
    Otherwise calls the scripted function from ``scripted_lookup``. Returns the
    merged value.
    """
    assert oracle.merge_fn is not None
    metadata = get_merge_fn_metadata(oracle.merge_fn)
    if metadata is not None:
        user_fn, param_res = metadata
        return user_fn(variants, *_resolve_merge_args(param_res, config, state_for_di))

    per_compile = scripted_lookup or {}
    scripted_merge = per_compile.get(oracle.merge_fn)
    if scripted_merge is None:
        raise ConfigurationError.build(
            f"Scripted function '{oracle.merge_fn}' not registered",
            hint="Pass scripted={'" + oracle.merge_fn + "': fn} to compile().",
        )
    return scripted_merge(variants, config)


def _assert_merge_fn_registered(oracle: Oracle, scripted_lookup: dict[str, Callable] | None) -> None:
    """Compile-time validation: a scripted merge_fn must be resolvable.

    Raises ConfigurationError if ``oracle.merge_fn`` is set but is neither a
    ``@merge_fn``-declared function nor present in ``scripted_lookup``. This is
    a registration check (not a merge step), called at barrier-build time so
    the error fires during compile() rather than at run time.
    """
    if not oracle.merge_fn or oracle.merge_prompt:
        return
    if get_merge_fn_metadata(oracle.merge_fn) is not None:
        return
    if (scripted_lookup or {}).get(oracle.merge_fn) is not None:
        return
    raise ConfigurationError.build(
        f"Scripted function '{oracle.merge_fn}' not registered",
        hint="Pass scripted={'" + oracle.merge_fn + "': fn} to compile().",
    )


def _merge_variants(
    oracle: Oracle,
    variants: list,
    output_model: TypeSpecStatic,
    config: RunnableConfig,
    *,
    upstream_context: dict[str, Any] | None = None,
    llm_config: LlmConfig | None = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    state_for_di: Any = None,
) -> Any:
    """Canonical Oracle merge: run the configured merge MODE on a variants list
    and return the merged value.

    This is the ONE site for the merge algorithm. Both orchestrators delegate
    here: ``make_oracle_merge_fn`` (single-group, reads state collector) and
    ``_wiring._merge_one_group`` (per-group, Each×Oracle fusion). The internal
    branch is merge_prompt-vs-merge_fn (the Oracle's two merge modes) — NOT
    per-group-vs-single-group, which stays the callers' concern.
    """
    if oracle.merge_prompt:
        return _run_merge_prompt(
            oracle,
            variants,
            output_model,
            config,
            upstream_context=upstream_context,
            llm_config=llm_config,
            runtime=runtime,
        )
    return _run_merge_fn(
        oracle,
        variants,
        config,
        scripted_lookup=scripted_lookup,
        state_for_di=state_for_di,
    )


async def _amerge_variants(
    oracle: Oracle,
    variants: list,
    output_model: TypeSpecStatic,
    config: RunnableConfig,
    *,
    upstream_context: dict[str, Any] | None = None,
    llm_config: LlmConfig | None = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    state_for_di: Any = None,
) -> Any:
    """Async twin of :func:`_merge_variants` per neograph-p3c7.

    Same ONE merge algorithm; only the merge_prompt (LLM-judge) branch differs —
    it awaits ``_arun_merge_prompt`` so the merge runs on the loop under the async
    driver. The merge_fn (scripted) branch is pure Python (no I/O) and is reused
    verbatim from the sync path.
    """
    if oracle.merge_prompt:
        return await _arun_merge_prompt(
            oracle,
            variants,
            output_model,
            config,
            upstream_context=upstream_context,
            llm_config=llm_config,
            runtime=runtime,
        )
    return _run_merge_fn(
        oracle,
        variants,
        config,
        scripted_lookup=scripted_lookup,
        state_for_di=state_for_di,
    )


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
) -> RunnableLambda:
    """Create the merge barrier function for Oracle (single-group orchestrator).

    Reads variants from ``collector_field`` on state, runs the canonical merge
    step (``_merge_variants``), and shapes the state update via
    ``_build_oracle_merge_result``. When *node_inputs* is provided (dict-form
    inputs), upstream values are injected into a merge_prompt's input_data via
    ``_build_upstream_context`` (parity with the Each×Oracle fused path).

    Merge hooks (merge_prompt path only) live in ``_run_merge_prompt``.
    """
    _node_inputs = node_inputs if isinstance(node_inputs, dict) else None
    # Fail-fast at compile time (closure-build) when a scripted merge_fn is not
    # registered — the merge barrier is built during compile(), so an
    # unregistered name surfaces as a ConfigurationError at compile, not at run.
    _assert_merge_fn_registered(oracle, scripted_lookup)

    def _prep(state: Any) -> tuple[list, Any, Any]:
        """Shared pre-merge read: collector variants + upstream context. Keeps the
        sync/async twins from drifting on state reads and result shaping."""
        bus = adapt_state(state)
        # StateBus.get optional: collector field is unbound until the first
        # generator writes a variant; empty-list default is the correct zero.
        results = bus.get(collector_field, [])
        primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)
        upstream_context = _build_upstream_context(bus, _node_inputs)
        return primary, secondaries, upstream_context

    def merge_fn(state: Any, config: RunnableConfig) -> dict:
        primary, secondaries, upstream_context = _prep(state)
        merged = _merge_variants(
            oracle,
            primary,
            output_model,
            config,
            upstream_context=upstream_context,
            llm_config=llm_config,
            runtime=runtime,
            scripted_lookup=scripted_lookup,
            state_for_di=state,
        )
        return _build_oracle_merge_result(merged, field_name, output_model, secondaries)

    async def amerge_fn(state: Any, config: RunnableConfig) -> dict:
        primary, secondaries, upstream_context = _prep(state)
        # Async twin: await the merge so an LLM-judge merge_prompt runs on the
        # loop instead of blocking it under graph.ainvoke per neograph-p3c7.
        merged = await _amerge_variants(
            oracle,
            primary,
            output_model,
            config,
            upstream_context=upstream_context,
            llm_config=llm_config,
            runtime=runtime,
            scripted_lookup=scripted_lookup,
            state_for_di=state,
        )
        return _build_oracle_merge_result(merged, field_name, output_model, secondaries)

    # Driver-selected dual path: graph.invoke -> merge_fn (sync invoke_structured),
    # graph.ainvoke -> amerge_fn (await ainvoke_structured). See neograph-p3c7.
    return RunnableLambda(merge_fn, afunc=amerge_fn)


def make_each_redirect_fn(
    raw_fn: Runnable,
    field_name: str,
    each: Each,
    item: HasName,
) -> Runnable:
    """Wrap a node function to key the result by the Each item's key field.

    Reads neo_each_item from state, uses each.key to extract the dispatch key.
    ``item`` is the captured IR node/construct; ``item.name`` is the
    user-declared label surfaced in StateMissingError messages — asked of the
    IR object directly, never threaded as a string.
    """

    def _key_val(state: Any) -> Any:
        # REQUIRED: Each router always populates EACH_ITEM in the Send payload.
        each_item = adapt_state(state).get_required(
            StateKeys.EACH_ITEM,
            node_label=item.name,
        )
        return getattr(each_item, each.key, str(each_item))

    def _project(state: Any, result: dict) -> dict:
        val = result.get(field_name)
        if val is not None:
            return {field_name: {_key_val(state): val}}
        return result

    def _project_failure(state: Any, exc: Exception) -> dict:
        # on_error='collect': key a typed EachFailure into the barrier so the
        # barrier always completes with one entry per planned key.
        key_val = _key_val(state)
        failure = EachFailure(
            key=str(key_val),
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return {field_name: {key_val: failure}}

    # raw_fn is a RunnableLambda (make_node_fn or a wrapped subgraph_fn);
    # .invoke(state, None) is safe (langchain synthesizes a config).
    def each_redirect_fn(state: Any, config: RunnableConfig = None) -> dict:  # type: ignore[assignment]
        if each.on_error == "collect":
            try:
                return _project(state, raw_fn.invoke(state, config))
            except GraphBubbleUp:
                # HITL interrupt / Command routing / cancellation must propagate,
                # never be collected into an EachFailure.
                raise
            except Exception as exc:  # noqa: BLE001 — collect any per-item fault
                return _project_failure(state, exc)
        return _project(state, raw_fn.invoke(state, config))

    async def aeach_redirect_fn(state: Any, config: RunnableConfig = None) -> dict:  # type: ignore[assignment]
        if each.on_error == "collect":
            try:
                return _project(state, await raw_fn.ainvoke(state, config))
            except GraphBubbleUp:
                # HITL interrupt / Command routing / cancellation must propagate,
                # never be collected into an EachFailure.
                raise
            except Exception as exc:  # noqa: BLE001 — collect any per-item fault
                return _project_failure(state, exc)
        return _project(state, await raw_fn.ainvoke(state, config))

    return RunnableLambda(each_redirect_fn, afunc=aeach_redirect_fn)
