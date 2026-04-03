"""Generic node factory — creates LangGraph node functions from Node definitions.

Dispatches by mode:
    produce       — single LLM call, structured JSON output
    gather        — ReAct tool loop with per-tool budgets
    execute       — ReAct tool loop with mutation tools
    scripted      — deterministic Python function
"""

from __future__ import annotations

from typing import Any, Callable

import structlog
from pydantic import BaseModel

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

    msg = f"Unknown mode: {node.mode}"
    raise ValueError(msg)


def _make_scripted_wrapper(node: Node) -> Callable:
    """Wrap a scripted function with state extraction and output wiring."""
    fn = _scripted_registry[node.scripted_fn]
    field_name = node.name.replace("-", "_")

    def scripted_node(state: BaseModel, config: dict) -> dict[str, Any]:
        node_log = log.bind(node=node.name, mode="scripted")
        node_log.info("node_start")

        # Extract input from state if specified
        input_data = _extract_input(state, node)
        result = fn(input_data, config) if input_data is not None else fn(config)

        update: dict[str, Any] = {}
        if node.output is not None and result is not None:
            update[field_name] = result

        node_log.info("node_complete")
        return update

    scripted_node.__name__ = node.name.replace("-", "_")
    return scripted_node


def _make_produce_fn(node: Node) -> Callable:
    """Single LLM call with structured JSON output. No tools."""
    field_name = node.name.replace("-", "_")

    def produce_node(state: BaseModel, config: dict) -> dict[str, Any]:
        # Import here to avoid circular deps and allow optional LangChain
        from neograph._llm import invoke_structured

        node_log = log.bind(node=node.name, mode="produce")
        node_log.info("node_start")

        input_data = _extract_input(state, node)
        result = invoke_structured(
            model_tier=node.model,
            prompt_template=node.prompt,
            input_data=input_data,
            output_model=node.output,
            config=config,
        )

        update: dict[str, Any] = {}
        if result is not None:
            update[field_name] = result
        node_log.info("node_complete", output_field=field_name)
        return update

    produce_node.__name__ = node.name.replace("-", "_")
    return produce_node


def _make_gather_fn(node: Node) -> Callable:
    """ReAct tool loop with per-tool budgets."""
    field_name = node.name.replace("-", "_")

    def gather_node(state: BaseModel, config: dict) -> dict[str, Any]:
        from neograph._llm import invoke_with_tools

        node_log = log.bind(node=node.name, mode="gather")
        node_log.info("node_start", tools=[t.name for t in node.tools])

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
        )

        update: dict[str, Any] = {}
        if result is not None:
            update[field_name] = result
        node_log.info("node_complete", output_field=field_name)
        return update

    gather_node.__name__ = node.name.replace("-", "_")
    return gather_node


def _make_execute_fn(node: Node) -> Callable:
    """ReAct tool loop with mutation tools."""
    field_name = node.name.replace("-", "_")

    def execute_node(state: BaseModel, config: dict) -> dict[str, Any]:
        from neograph._llm import invoke_with_tools

        node_log = log.bind(node=node.name, mode="execute")
        node_log.info("node_start", tools=[t.name for t in node.tools])

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
        )

        update: dict[str, Any] = {}
        if result is not None:
            update[field_name] = result
        node_log.info("node_complete", output_field=field_name)
        return update

    execute_node.__name__ = node.name.replace("-", "_")
    return execute_node


def _extract_input(state: BaseModel, node: Node) -> Any:
    """Extract typed input from state based on node's input spec."""
    if node.input is None:
        return None

    # dict[str, type] — multiple fields from state
    if isinstance(node.input, dict):
        result = {}
        for field_name, _field_type in node.input.items():
            state_key = field_name.replace("-", "_")
            result[field_name] = getattr(state, state_key, None)
        return result

    # Single type — find matching field in state by type or name
    field_name = node.name.replace("-", "_")
    for attr_name in state.model_fields:
        val = getattr(state, attr_name, None)
        if val is not None and isinstance(val, node.input):
            return val

    return getattr(state, field_name, None)
