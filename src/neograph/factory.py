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

from neograph.modifiers import Each, Oracle
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
    """Look up a registered condition function by name. Raises ValueError if missing."""
    fn = _condition_registry.get(name)
    if fn is None:
        msg = f"Condition '{name}' not registered. Use register_condition()."
        raise ValueError(msg)
    return fn


def lookup_scripted(name: str) -> Callable:
    """Look up a registered scripted function by name. Raises ValueError if missing."""
    fn = _scripted_registry.get(name)
    if fn is None:
        msg = f"Scripted function '{name}' not registered. Use register_scripted()."
        raise ValueError(msg)
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
        return update

    # Single-type outputs (backward compat).
    if each_mod and each_item is not None:
        key_val = getattr(each_item, each_mod.key, str(each_item))
        return {field_name: {key_val: result}}
    return {field_name: result}


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
            raise ValueError(msg)
        return _make_scripted_wrapper(node)

    # LLM nodes — dispatch by mode
    if node.mode == "produce":
        return _make_produce_fn(node)
    if node.mode in ("gather", "execute"):
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

        node_log = log.bind(node=node.name, mode="produce", model=node.model, prompt=node.prompt)
        node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))

        t0 = time.monotonic()
        input_data = _extract_input(state, node)

        # Conditional produce: skip LLM when predicate is true (neograph-s14).
        # Unwrap single-key dicts so skip_when receives a typed value for
        # single-upstream nodes (consistent across @node and Node() surfaces).
        if node.skip_when is not None:
            skip_input = input_data
            if isinstance(input_data, dict) and len(input_data) == 1:
                skip_input = next(iter(input_data.values()))
            if node.skip_when(skip_input):
                elapsed = time.monotonic() - t0
                node_log.info("node_skipped", reason="skip_when", duration_s=round(elapsed, 3))
                if node.skip_value is not None:
                    return {field_name: node.skip_value(skip_input)}
                return {}

        # Apply renderer dispatch chain (neograph-ni6)
        from neograph.renderers import render_input
        try:
            from neograph._llm import _get_global_renderer
            effective_renderer = node.renderer or _get_global_renderer()
        except ImportError:
            effective_renderer = node.renderer
        if effective_renderer is not None:
            input_data = render_input(input_data, renderer=effective_renderer)

        # For dict-form outputs, the LLM produces the primary type (first key).
        # Secondary outputs (e.g. tool_log) are framework-collected (1bp.6).
        output_model = node.outputs
        primary_key: str | None = None
        if isinstance(node.outputs, dict):
            primary_key = next(iter(node.outputs))
            output_model = node.outputs[primary_key]

        result = invoke_structured(
            model_tier=node.model,
            prompt_template=node.prompt,
            input_data=input_data,
            output_model=output_model,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
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

        # Conditional produce: skip LLM when predicate is true (neograph-s14).
        if node.skip_when is not None:
            skip_input = input_data
            if isinstance(input_data, dict) and len(input_data) == 1:
                skip_input = next(iter(input_data.values()))
            if node.skip_when(skip_input):
                elapsed = time.monotonic() - t0
                node_log.info("node_skipped", reason="skip_when", duration_s=round(elapsed, 3))
                if node.skip_value is not None:
                    return {field_name: node.skip_value(skip_input)}
                return {}

        # Apply renderer dispatch chain (neograph-ni6)
        from neograph.renderers import render_input
        try:
            from neograph._llm import _get_global_renderer
            effective_renderer = node.renderer or _get_global_renderer()
        except ImportError:
            effective_renderer = node.renderer
        if effective_renderer is not None:
            input_data = render_input(input_data, renderer=effective_renderer)

        # For dict-form outputs, the LLM produces the primary type (first key).
        output_model = node.outputs
        primary_key: str | None = None
        if isinstance(node.outputs, dict):
            primary_key = next(iter(node.outputs))
            output_model = node.outputs[primary_key]

        budget_tracker = ToolBudgetTracker(node.tools)

        result = invoke_with_tools(
            model_tier=node.model,
            prompt_template=node.prompt,
            input_data=input_data,
            output_model=output_model,
            tools=node.tools,
            budget_tracker=budget_tracker,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
        )

        if primary_key is not None and result is not None:
            result = {primary_key: result}

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
        if val is not None and _is_instance_safe(val, node.inputs):
            return val

    return None


# ── Factory functions for modifier wiring ──────────────────────────────


def make_oracle_redirect_fn(raw_fn: Callable, field_name: str, collector_field: str) -> Callable:
    """Wrap a node function to redirect output from field_name to collector_field.

    Used by Oracle generators: the node writes to the collector (list reducer)
    instead of the consumer-facing field.
    """

    def oracle_redirect_fn(state: Any, config: RunnableConfig) -> dict:
        result = raw_fn(state, config)
        val = result.get(field_name)
        if val is not None:
            return {collector_field: val}
        return result

    oracle_redirect_fn.__name__ = raw_fn.__name__
    return oracle_redirect_fn


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
    Reads from collector_field, writes to field_name.
    """
    if oracle.merge_prompt:
        def merge_fn(state: Any, config: RunnableConfig) -> dict:
            from neograph._llm import invoke_structured

            results = getattr(state, collector_field, [])
            return {field_name: invoke_structured(
                model_tier=oracle.merge_model,
                prompt_template=oracle.merge_prompt,
                input_data=results,
                output_model=output_model,
                config=config,
            )}
    else:
        # Check for @merge_fn DI metadata first. If present, call the
        # original user function with resolved DI parameters. Otherwise
        # fall back to the legacy (variants, config) scripted signature.
        from neograph.decorators import get_merge_fn_metadata, _resolve_di_args

        metadata = get_merge_fn_metadata(oracle.merge_fn)
        if metadata is not None:
            user_fn, param_res = metadata

            def merge_fn(state: Any, config: RunnableConfig) -> dict:
                results = getattr(state, collector_field, [])
                return {field_name: user_fn(results, *_resolve_di_args(param_res, config))}
        else:
            scripted_merge = lookup_scripted(oracle.merge_fn)

            def merge_fn(state: Any, config: RunnableConfig) -> dict:
                results = getattr(state, collector_field, [])
                return {field_name: scripted_merge(results, config)}

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

        sub_result = _strip_internals(sub_graph.invoke(sub_input, config=config))

        # Extract the declared output type from sub result
        output_val = None
        for val in sub_result.values():
            if isinstance(val, sub.output):
                output_val = val
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
