"""Generic node factory — creates LangGraph node functions from Node definitions.

Dispatches by mode:
    produce       — single LLM call, structured JSON output
    gather/execute — ReAct tool loop with tools (read-only or mutation)
    scripted      — deterministic Python function

Also provides higher-order factory functions for modifier wiring:
    make_oracle_redirect_fn   — redirects node output to collector field
    make_oracle_merge_fn      — creates the merge barrier function
    make_subgraph_fn          — creates function to run a sub-Construct
    make_each_redirect_fn     — wraps node output keyed by Each item
"""

from __future__ import annotations

import time
from typing import Any, Callable, get_origin as _get_origin

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph.errors import ConfigurationError
from neograph.modifiers import Each, Loop, Oracle
from neograph.node import Node
from neograph.tool import ToolBudgetTracker

log = structlog.get_logger()


def _state_get(state: Any, key: str) -> Any:
    """Read a key from state, handling both dict and Pydantic model forms."""
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


# Registry for scripted functions and condition checks
_scripted_registry: dict[str, Callable] = {}
_condition_registry: dict[str, Callable] = {}
_tool_factory_registry: dict[str, Callable] = {}


def register_scripted(name: str, fn: Callable) -> None:
    """Register a deterministic function for Node.scripted."""
    _scripted_registry[name] = fn


def register_condition(name: str, fn: Callable) -> None:
    """Register a condition function for Operator(when=...)."""
    _condition_registry[name] = fn


def register_tool_factory(name: str, fn: Callable) -> None:
    """Register a tool factory that creates LangChain @tool functions."""
    _tool_factory_registry[name] = fn


def lookup_condition(name: str) -> Callable:
    """Look up a registered condition function by name. Raises ConfigurationError if missing."""
    fn = _condition_registry.get(name)
    if fn is None:
        msg = f"Condition '{name}' not registered. Use register_condition()."
        raise ConfigurationError(msg)
    return fn


def lookup_scripted(name: str) -> Callable:
    """Look up a registered scripted function by name. Raises ConfigurationError if missing."""
    fn = _scripted_registry.get(name)
    if fn is None:
        msg = f"Scripted function '{name}' not registered. Use register_scripted()."
        raise ConfigurationError(msg)
    return fn


def _type_name(t: Any) -> str | None:
    """Get a readable name from a type, or None."""
    if t is None:
        return None
    if isinstance(t, dict):
        parts = ", ".join(
            f"{k}: {getattr(v, '__name__', str(v))}" for k, v in t.items()
        )
        return "{" + parts + "}"
    return getattr(t, '__name__', str(t))


def _apply_skip_when(
    node: Node,
    input_data: Any,
    field_name: str,
    t0: float,
    node_log: Any,
    state: Any = None,
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
    if not node.skip_when(skip_input):
        return None
    elapsed = time.monotonic() - t0
    node_log.info("node_skipped", reason="skip_when", duration_s=round(elapsed, 3))
    if node.skip_value is not None:
        result = node.skip_value(skip_input)
        return _build_state_update(node, field_name, result, state)
    return {}


def _render_input(node: Node, input_data: Any) -> Any:
    """Apply renderer dispatch chain: node renderer > global renderer.

    Returns rendered input_data, or the original if no renderer is active.
    """
    from neograph.renderers import render_input
    try:
        from neograph._llm import _get_global_renderer
        effective_renderer = node.renderer or _get_global_renderer()
    except ImportError:
        effective_renderer = node.renderer
    if effective_renderer is not None:
        return render_input(input_data, renderer=effective_renderer)
    return input_data


def _resolve_primary_output(node: Node) -> tuple[Any, str | None]:
    """Resolve the LLM output model and primary key for dict-form outputs.

    For dict-form outputs, the LLM produces the primary type (first key).
    Secondary outputs (e.g. tool_log) are framework-collected.

    When ``node.oracle_gen_type`` is set (Oracle with type-transforming merge_fn),
    the generator type overrides ``node.outputs`` — the LLM should produce the
    per-variant type, not the post-merge type.

    Returns (output_model, primary_key) where primary_key is None for
    single-type outputs.
    """
    # Oracle generator type override: merge_fn transforms A → B, generators produce A.
    if node.oracle_gen_type is not None:
        return node.oracle_gen_type, None

    if isinstance(node.outputs, dict):
        primary_key = next(iter(node.outputs))
        return node.outputs[primary_key], primary_key
    return node.outputs, None


def _build_state_update(
    node: Node,
    field_name: str,
    result: Any,
    state: Any,
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

    each_mod = node.get_modifier(Each)
    each_item = _state_get(state, "neo_each_item")

    # Dict-form outputs: per-key state fields (neograph-1bp.3).
    if isinstance(node.outputs, dict) and isinstance(result, dict):
        update: dict[str, Any] = {}
        for key in node.outputs:
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
    loop_mod = node.get_modifier(Loop)
    if loop_mod is not None:
        count_field = f"neo_loop_count_{field_name}"
        current_count = _state_get(state, count_field) or 0
        update[count_field] = current_count + 1
        if loop_mod.history:
            history_field = f"neo_loop_history_{field_name}"
            update[history_field] = result

    return update


def make_node_fn(node: Node) -> Callable:
    """Create a LangGraph node function from a Node definition.

    This is the core of NeoGraph — the generic factory that eliminates
    the 70% boilerplate from every hand-coded node.
    """
    # Raw node — wrap with observability so node_start/node_complete fire
    if node.raw_fn is not None:
        return _make_raw_wrapper(node)

    # Scripted node — look up registered function
    if node.mode == "scripted":
        if node.scripted_fn not in _scripted_registry:
            msg = f"Scripted function '{node.scripted_fn}' not registered. Use register_scripted()."
            raise ConfigurationError(msg)
        return _make_scripted_wrapper(node)

    # LLM nodes — dispatch by mode
    if node.mode == "think":
        return _make_produce_fn(node)
    if node.mode in ("agent", "act"):
        return _make_tool_fn(node)


def _make_raw_wrapper(node: Node) -> Callable:
    """Wrap a raw_fn dispatch with observability (node_start/node_complete).

    Only used for explicit ``mode='raw'`` escape-hatch nodes. Scripted
    ``@node`` functions route through ``_make_scripted_wrapper`` via
    ``register_scripted`` since neograph-kqd.8.
    """
    raw_fn = node.raw_fn
    field_name = node.name.replace("-", "_")

    def raw_node_wrapper(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        node_log = log.bind(node=node.name, mode="raw")
        node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))
        t0 = time.monotonic()

        result = raw_fn(state, config)

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return result

    raw_node_wrapper.__name__ = field_name
    return raw_node_wrapper


def _make_scripted_wrapper(node: Node) -> Callable:
    """Wrap a scripted function with state extraction and output wiring."""
    fn = _scripted_registry[node.scripted_fn]
    field_name = node.name.replace("-", "_")

    def scripted_node(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        node_log = log.bind(node=node.name, mode="scripted", fn=node.scripted_fn)
        node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))

        t0 = time.monotonic()

        # Inject oracle generator ID into config if present in state
        oracle_gen_id = _state_get(state, "neo_oracle_gen_id")
        if oracle_gen_id is not None:
            configurable = config.get("configurable", {})
            config = {**config, "configurable": {**configurable, "_generator_id": oracle_gen_id}}

        # Extract input from state if specified
        input_data = _extract_input(state, node)
        result = fn(input_data, config)

        update = _build_state_update(node, field_name, result, state)

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return update

    scripted_node.__name__ = node.name.replace("-", "_")
    return scripted_node


def _make_produce_fn(node: Node) -> Callable:
    """Single LLM call with structured JSON output. No tools."""
    field_name = node.name.replace("-", "_")

    def produce_node(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        from neograph._llm import invoke_structured

        node_log = log.bind(node=node.name, mode="think", model=node.model, prompt=node.prompt)
        node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))

        t0 = time.monotonic()
        input_data = _extract_input(state, node)

        skip_result = _apply_skip_when(node, input_data, field_name, t0, node_log, state)
        if skip_result is not None:
            return skip_result

        input_data = _render_input(node, input_data)
        output_model, primary_key = _resolve_primary_output(node)

        # Extract verbatim context fields from state
        context_data = None
        if node.context:
            context_data = {
                name: _state_get(state, name.replace("-", "_"))
                for name in node.context
            }

        result = invoke_structured(
            model_tier=node.model,
            prompt_template=node.prompt,
            input_data=input_data,
            output_model=output_model,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
            context=context_data,
        )

        if primary_key is not None and result is not None:
            # Wrap single LLM result as a dict so _build_state_update routes per-key
            result = {primary_key: result}

        update = _build_state_update(node, field_name, result, state)

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return update

    produce_node.__name__ = node.name.replace("-", "_")
    return produce_node


def _make_tool_fn(node: Node) -> Callable:
    """ReAct tool loop — used for both gather and execute modes."""
    field_name = node.name.replace("-", "_")

    def tool_node(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        from neograph._llm import invoke_with_tools

        node_log = log.bind(node=node.name, mode=node.mode, model=node.model, prompt=node.prompt)
        node_log.info("node_start",
                      input_type=_type_name(node.inputs), output_type=_type_name(node.outputs),
                      tools=[t.name for t in node.tools],
                      budgets={t.name: t.budget for t in node.tools})

        t0 = time.monotonic()
        input_data = _extract_input(state, node)

        skip_result = _apply_skip_when(node, input_data, field_name, t0, node_log, state)
        if skip_result is not None:
            return skip_result

        input_data = _render_input(node, input_data)
        output_model, primary_key = _resolve_primary_output(node)

        budget_tracker = ToolBudgetTracker(node.tools)

        # Resolve renderer for tool result rendering in ToolMessage
        try:
            from neograph._llm import _get_global_renderer
            effective_renderer = node.renderer or _get_global_renderer()
        except (ImportError, AttributeError):
            effective_renderer = node.renderer

        # Extract verbatim context fields from state
        context_data = None
        if node.context:
            context_data = {
                name: _state_get(state, name.replace("-", "_"))
                for name in node.context
            }

        result, tool_interactions = invoke_with_tools(
            model_tier=node.model,
            prompt_template=node.prompt,
            input_data=input_data,
            output_model=output_model,
            tools=node.tools,
            budget_tracker=budget_tracker,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
            renderer=effective_renderer,
            context=context_data,
        )

        if primary_key is not None and result is not None:
            # Build a dict with primary LLM result and tool_log if present
            result_dict: dict[str, Any] = {primary_key: result}
            # Write tool_log if the node declares it as an output key
            if isinstance(node.outputs, dict) and "tool_log" in node.outputs:
                result_dict["tool_log"] = tool_interactions
            result = result_dict
        elif isinstance(node.outputs, dict):
            result = {next(iter(node.outputs)): result} if result is not None else None
            if result is not None and "tool_log" in node.outputs:
                result["tool_log"] = tool_interactions

        update = _build_state_update(node, field_name, result, state)

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return update

    tool_node.__name__ = node.name.replace("-", "_")
    return tool_node


def _is_instance_safe(val: Any, type_spec: Any) -> bool:
    """isinstance() that handles parameterized generics like dict[str, X]."""
    from typing import get_origin, get_args

    origin = get_origin(type_spec)
    if origin is not None:
        # Parameterized generic: check base type (dict, list, etc.)
        return isinstance(val, origin)
    try:
        return isinstance(val, type_spec)
    except TypeError:
        return False


def _extract_input(state: Any, node: Node) -> Any:
    """Extract typed input from state based on node's inputs spec."""
    if node.inputs is None:
        return None

    def _fields() -> list[str]:
        if isinstance(state, dict):
            return list(state.keys())
        return list(state.__class__.model_fields.keys())

    # Loop re-entry: on iteration 1+, read from the node's OWN output field
    # (an append-list) instead of the upstream. The append-list reducer
    # accumulates each iteration's result; we unwrap [-1] for the latest.
    if node.has_modifier(Loop):
        own_field = node.name.replace("-", "_")
        own_val = _state_get(state, own_field)
        if isinstance(own_val, list) and own_val:
            latest = own_val[-1]
            # For dict-form inputs, wrap in a dict matching the input spec
            if isinstance(node.inputs, dict):
                # Use the first input key as the wrapper key
                first_key = next(iter(node.inputs))
                return {first_key: latest}
            return latest

    # Each fan-out: item is passed via neo_each_item
    replicate_item = _state_get(state, "neo_each_item")
    if replicate_item is not None and _is_instance_safe(replicate_item, node.inputs):
        return replicate_item

    # Fan-in dict: inputs={'upstream_name': expected_type, ...}. Read each
    # named state field by key. Special cases:
    #   - fan_out_param key → read from state["neo_each_item"] (neograph-kqd.8)
    #   - list[X] consumer over dict state → unwrap via list(values()) (kqd.3)
    if isinstance(node.inputs, dict):
        result = {}
        for field_name, expected_type in node.inputs.items():
            if field_name == node.fan_out_param:
                value = _state_get(state, "neo_each_item")
            else:
                state_key = field_name.replace("-", "_")
                value = _state_get(state, state_key)
                # Unwrap loop-list fields: upstream was a Loop node, field is
                # a list from the append reducer. Read [-1] for latest.
                if isinstance(value, list) and value and not _get_origin(expected_type) is list:
                    value = value[-1]
                if (
                    value is not None
                    and _get_origin(expected_type) is list
                    and isinstance(value, dict)
                ):
                    value = list(value.values())
            result[field_name] = value
        return result

    # Single type — find matching field in state by type or name
    for attr_name in _fields():
        val = _state_get(state, attr_name)
        # Unwrap loop-list fields for downstream consumers
        if isinstance(val, list) and val and not _get_origin(node.inputs) is list:
            val = val[-1]
        if val is not None and _is_instance_safe(val, node.inputs):
            return val

    return None


# ── Factory functions for modifier wiring ──────────────────────────────


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


def _unwrap_oracle_results(
    results: list,
    field_name: str,
    output_model: Any,
) -> tuple[list, dict[str, list] | None]:
    """Unwrap collected Oracle results for the merge function.

    For single-type outputs: results is [val1, val2, ...] — return as-is.
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
    """
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
) -> Callable:
    """Create the merge barrier function for Oracle.

    If oracle.merge_prompt, calls invoke_structured (LLM judge).
    If oracle.merge_fn, calls the registered scripted function — with
    FromInput/FromConfig DI if it was declared via ``@merge_fn``.
    Reads from collector_field, writes to field_name (or per-key fields
    for dict-form outputs).
    """
    if oracle.merge_prompt:
        def merge_fn(state: Any, config: RunnableConfig) -> dict:
            from neograph._llm import invoke_structured

            results = getattr(state, collector_field, [])
            primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)
            merged = invoke_structured(
                model_tier=oracle.merge_model,
                prompt_template=oracle.merge_prompt,
                input_data=primary,
                output_model=output_model if not isinstance(output_model, dict) else next(iter(output_model.values())),
                config=config,
            )
            return _build_oracle_merge_result(merged, field_name, output_model, secondaries)
    else:
        from neograph.decorators import get_merge_fn_metadata, _resolve_di_args

        metadata = get_merge_fn_metadata(oracle.merge_fn)
        if metadata is not None:
            user_fn, param_res = metadata

            def merge_fn(state: Any, config: RunnableConfig) -> dict:
                results = getattr(state, collector_field, [])
                primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)
                merged = user_fn(primary, *_resolve_di_args(param_res, config))
                return _build_oracle_merge_result(merged, field_name, output_model, secondaries)
        else:
            scripted_merge = lookup_scripted(oracle.merge_fn)

            def merge_fn(state: Any, config: RunnableConfig) -> dict:
                results = getattr(state, collector_field, [])
                primary, secondaries = _unwrap_oracle_results(results, field_name, output_model)
                merged = scripted_merge(primary, config)
                return _build_oracle_merge_result(merged, field_name, output_model, secondaries)

    return merge_fn


def make_subgraph_fn(sub: Any, sub_graph: Any) -> Callable:
    """Create a function that runs a sub-Construct in isolation.

    Extracts input from parent state by type, runs sub_graph,
    extracts output by type, returns {field_name: output}.
    """
    from neograph.runner import _strip_internals

    sub_log = log.bind(subgraph=sub.name)
    field_name = sub.name.replace("-", "_")

    def subgraph_node(state: Any, config: RunnableConfig) -> dict:
        sub_log.info("subgraph_start")

        # Extract input from parent state by type
        input_data = None
        if isinstance(state, dict):
            for val in state.values():
                if val is not None and isinstance(val, sub.input):
                    input_data = val
                    break
        else:
            for attr_name in state.__class__.model_fields:
                val = getattr(state, attr_name, None)
                if val is not None and isinstance(val, sub.input):
                    input_data = val
                    break

        # Run sub-graph with isolated state
        sub_input: dict[str, Any] = {"node_id": state.get("node_id", "") if isinstance(state, dict) else getattr(state, "node_id", "")}
        if input_data is not None:
            sub_input["neo_subgraph_input"] = input_data

        # Forward context fields from parent state into sub-construct
        for n in sub.nodes:
            if hasattr(n, "context") and n.context:
                for ctx_name in n.context:
                    ctx_field = ctx_name.replace("-", "_")
                    val = _state_get(state, ctx_field)
                    if val is not None:
                        sub_input[ctx_field] = val

        sub_result = _strip_internals(sub_graph.invoke(sub_input, config=config))

        # Extract the declared output type from sub result.
        # Iterate in reverse so later pipeline nodes take precedence.
        # Unwrap loop append-lists: Loop nodes have list[T] from the
        # append-list reducer; check val[-1] against the output type.
        output_val = None
        for val in reversed(list(sub_result.values())):
            check_val = val
            if isinstance(val, list) and val:
                check_val = val[-1]
            if isinstance(check_val, sub.output):
                output_val = check_val
                break

        sub_log.info("subgraph_complete")
        return {field_name: output_val}

    subgraph_node.__name__ = field_name
    return subgraph_node


def make_each_redirect_fn(raw_fn: Callable, field_name: str, each: Each) -> Callable:
    """Wrap a node function to key the result by the Each item's key field.

    Reads neo_each_item from state, uses each.key to extract the dispatch key.
    """

    def each_redirect_fn(state: Any, config: RunnableConfig = None) -> dict:
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
