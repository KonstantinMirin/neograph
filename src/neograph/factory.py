"""Generic node factory — creates LangGraph node functions from Node definitions.

Dispatches by mode:
    produce       — single LLM call, structured JSON output
    gather        — ReAct tool loop with per-tool budgets
    execute       — ReAct tool loop with mutation tools
    scripted      — deterministic Python function

Also provides higher-order factory functions for modifier wiring:
    make_oracle_redirect_fn   — redirects node output to collector field
    make_oracle_merge_fn      — creates the merge barrier function
    make_subgraph_fn          — creates function to run a sub-Construct
    make_each_redirect_fn     — wraps node output keyed by Each item
"""

from __future__ import annotations

import time
from typing import Any, Callable

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph.modifiers import Each, Oracle
from neograph.node import Node
from neograph.tool import ToolBudgetTracker

log = structlog.get_logger()


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
        msg = f"Merge function '{name}' not registered. Use register_scripted()."
        raise ValueError(msg)
    return fn


def make_node_fn(node: Node) -> Callable:
    """Create a LangGraph node function from a Node definition.

    This is the core of NeoGraph — the generic factory that eliminates
    the 70% boilerplate from every hand-coded node.
    """
    # Raw node — use the function directly
    if node.raw_fn is not None:
        return node.raw_fn

    # Scripted node — look up registered function
    if node.mode == "scripted":
        if node.scripted_fn not in _scripted_registry:
            msg = f"Scripted function '{node.scripted_fn}' not registered. Use register_scripted()."
            raise ValueError(msg)
        return _make_scripted_wrapper(node)

    # LLM nodes — dispatch by mode
    if node.mode == "produce":
        return _make_produce_fn(node)
    if node.mode == "gather":
        return _make_gather_fn(node)
    if node.mode == "execute":
        return _make_execute_fn(node)


def _make_scripted_wrapper(node: Node) -> Callable:
    """Wrap a scripted function with state extraction and output wiring."""
    fn = _scripted_registry[node.scripted_fn]
    field_name = node.name.replace("-", "_")

    def scripted_node(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        node_log = log.bind(node=node.name, mode="scripted", fn=node.scripted_fn)
        node_log.info("node_start",
                      input_type=node.input.__name__ if node.input and hasattr(node.input, '__name__') else None,
                      output_type=node.output.__name__ if node.output and hasattr(node.output, '__name__') else None)

        t0 = time.monotonic()

        # Helper for dict/model state access
        def _state_get(key: str) -> Any:
            if isinstance(state, dict):
                return state.get(key)
            return getattr(state, key, None)

        # Inject oracle generator ID into config if present in state
        oracle_gen_id = _state_get("neo_oracle_gen_id")
        if oracle_gen_id is not None:
            configurable = config.get("configurable", {})
            config = {**config, "configurable": {**configurable, "_generator_id": oracle_gen_id}}

        # Extract input from state if specified
        input_data = _extract_input(state, node)
        result = fn(input_data, config)

        update: dict[str, Any] = {}
        if node.output is not None and result is not None:
            # Each fan-out: wrap result in dict keyed by item's key field
            each_mod = node.get_modifier(Each)
            each_item = _state_get("neo_each_item")
            if each_mod and each_item is not None:
                key_val = getattr(each_item, each_mod.key, str(each_item))
                update[field_name] = {key_val: result}
            else:
                # Oracle redirection handled by compiler wrapper, not here
                update[field_name] = result

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
        node_log.info("node_start", output_type=node.output.__name__ if node.output else None)

        t0 = time.monotonic()
        input_data = _extract_input(state, node)
        result = invoke_structured(
            model_tier=node.model,
            prompt_template=node.prompt,
            input_data=input_data,
            output_model=node.output,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
        )

        update: dict[str, Any] = {}
        if result is not None:
            update[field_name] = result

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return update

    produce_node.__name__ = node.name.replace("-", "_")
    return produce_node


def _make_gather_fn(node: Node) -> Callable:
    """ReAct tool loop with per-tool budgets."""
    field_name = node.name.replace("-", "_")

    def gather_node(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        from neograph._llm import invoke_with_tools

        node_log = log.bind(node=node.name, mode="gather", model=node.model, prompt=node.prompt)
        node_log.info("node_start",
                      tools=[t.name for t in node.tools],
                      budgets={t.name: t.budget for t in node.tools})

        t0 = time.monotonic()
        input_data = _extract_input(state, node)
        budget_tracker = ToolBudgetTracker(node.tools)

        result = invoke_with_tools(
            model_tier=node.model,
            prompt_template=node.prompt,
            input_data=input_data,
            output_model=node.output,
            tools=node.tools,
            budget_tracker=budget_tracker,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
        )

        update: dict[str, Any] = {}
        if result is not None:
            update[field_name] = result

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return update

    gather_node.__name__ = node.name.replace("-", "_")
    return gather_node


def _make_execute_fn(node: Node) -> Callable:
    """ReAct tool loop with mutation tools."""
    field_name = node.name.replace("-", "_")

    def execute_node(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        from neograph._llm import invoke_with_tools

        node_log = log.bind(node=node.name, mode="execute", model=node.model, prompt=node.prompt)
        node_log.info("node_start",
                      tools=[t.name for t in node.tools],
                      budgets={t.name: t.budget for t in node.tools})

        t0 = time.monotonic()
        input_data = _extract_input(state, node)
        budget_tracker = ToolBudgetTracker(node.tools)

        result = invoke_with_tools(
            model_tier=node.model,
            prompt_template=node.prompt,
            input_data=input_data,
            output_model=node.output,
            tools=node.tools,
            budget_tracker=budget_tracker,
            config=config,
            node_name=node.name,
            llm_config=node.llm_config,
        )

        update: dict[str, Any] = {}
        if result is not None:
            update[field_name] = result

        elapsed = time.monotonic() - t0
        node_log.info("node_complete", duration_s=round(elapsed, 3))
        return update

    execute_node.__name__ = node.name.replace("-", "_")
    return execute_node


def _extract_input(state: Any, node: Node) -> Any:
    """Extract typed input from state based on node's input spec."""
    if node.input is None:
        return None

    # Handle both Pydantic model and dict states (Send passes dicts)
    def _get(key: str) -> Any:
        if isinstance(state, dict):
            return state.get(key)
        return getattr(state, key, None)

    def _fields() -> list[str]:
        if isinstance(state, dict):
            return list(state.keys())
        return list(state.__class__.model_fields.keys())

    # Each fan-out: item is passed via neo_each_item
    replicate_item = _get("neo_each_item")
    if replicate_item is not None and isinstance(replicate_item, node.input):
        return replicate_item

    # dict[str, type] — multiple fields from state
    if isinstance(node.input, dict):
        result = {}
        for field_name, _field_type in node.input.items():
            state_key = field_name.replace("-", "_")
            result[field_name] = _get(state_key)
        return result

    # Single type — find matching field in state by type or name
    for attr_name in _fields():
        val = _get(attr_name)
        if val is not None and isinstance(val, node.input):
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
    If oracle.merge_fn, calls the registered scripted function.
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
        each_item = state.get("neo_each_item") if isinstance(state, dict) else getattr(state, "neo_each_item", None)

        result = raw_fn(state, config) if config else raw_fn(state)
        val = result.get(field_name)

        if val is not None and each_item is not None:
            key_val = getattr(each_item, each.key, str(each_item))
            return {field_name: {key_val: val}}
        return result

    each_redirect_fn.__name__ = raw_fn.__name__ if hasattr(raw_fn, '__name__') else field_name
    return each_redirect_fn
