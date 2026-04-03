"""LLM invocation layer — thin abstraction over LangChain chat models.

This module is intentionally minimal. NeoGraph does NOT own the LLM client
configuration — consumers register their model routing. This module provides
the dispatch mechanics (structured output, ReAct loop, tool budget enforcement).
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from neograph.tool import Tool, ToolBudgetTracker

# Consumer-provided LLM factory
_llm_factory: Callable[[str], Any] | None = None

# Consumer-provided prompt compiler
_prompt_compiler: Callable[[str, Any], list] | None = None


def configure_llm(
    llm_factory: Callable[[str], Any],
    prompt_compiler: Callable[[str, Any], list],
) -> None:
    """Configure NeoGraph's LLM layer.

    Args:
        llm_factory: Callable that takes a model tier ("fast", "reason", "large")
                     and returns a LangChain BaseChatModel.
        prompt_compiler: Callable that takes (template_name, input_data) and
                        returns list[BaseMessage].

    Usage:
        from neograph import configure_llm

        def my_llm_factory(tier: str) -> BaseChatModel:
            if tier == "fast":
                return ChatOpenAI(model="deepseek-chat")
            elif tier == "reason":
                return ChatOpenAI(model="deepseek-reasoner")
            ...

        configure_llm(llm_factory=my_llm_factory, prompt_compiler=my_prompt_compiler)
    """
    global _llm_factory, _prompt_compiler  # noqa: PLW0603
    _llm_factory = llm_factory
    _prompt_compiler = prompt_compiler


def _get_llm(tier: str) -> Any:
    if _llm_factory is None:
        msg = "LLM not configured. Call neograph.configure_llm() first."
        raise RuntimeError(msg)
    return _llm_factory(tier)


def _compile_prompt(template: str, input_data: Any) -> list:
    if _prompt_compiler is None:
        msg = "Prompt compiler not configured. Call neograph.configure_llm() first."
        raise RuntimeError(msg)
    return _prompt_compiler(template, input_data)


def invoke_structured(
    model_tier: str,
    prompt_template: str,
    input_data: Any,
    output_model: type[BaseModel],
    config: dict,
) -> BaseModel:
    """Single LLM call with structured JSON output. Mode: produce."""
    llm = _get_llm(model_tier)
    messages = _compile_prompt(prompt_template, input_data)
    structured_llm = llm.with_structured_output(output_model)
    return structured_llm.invoke(messages, config=config)


def invoke_with_tools(
    model_tier: str,
    prompt_template: str,
    input_data: Any,
    output_model: type[BaseModel] | None,
    tools: list[Tool],
    budget_tracker: ToolBudgetTracker,
    config: dict,
) -> BaseModel | None:
    """ReAct tool loop with per-tool budget enforcement. Mode: gather/execute."""
    from langchain_core.messages import ToolMessage

    from neograph.factory import _tool_factory_registry

    llm = _get_llm(model_tier)
    messages = _compile_prompt(prompt_template, input_data)

    # Create tool instances from registered factories
    tool_instances = {}
    for tool_spec in tools:
        if tool_spec.name not in _tool_factory_registry:
            msg = f"Tool '{tool_spec.name}' not registered. Use register_tool_factory()."
            raise ValueError(msg)
        factory = _tool_factory_registry[tool_spec.name]
        tool_instances[tool_spec.name] = factory(config, tool_spec.config)

    active_tools = list(tool_instances.values())
    llm_with_tools = llm.bind_tools(active_tools)

    while True:
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]

            if not budget_tracker.can_call(tool_name):
                messages.append(ToolMessage(
                    content=f"Tool '{tool_name}' budget exhausted ({budget_tracker._budgets[tool_name]} calls used). Use remaining tools or respond.",
                    tool_call_id=tool_call["id"],
                ))
                continue

            tool_fn = tool_instances.get(tool_name)
            if tool_fn is None:
                messages.append(ToolMessage(
                    content=f"Unknown tool: {tool_name}",
                    tool_call_id=tool_call["id"],
                ))
                continue

            result = tool_fn.invoke(tool_call["args"])
            budget_tracker.record_call(tool_name)
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ))

        # Rebind with only tools that still have budget
        active_tools = [
            tool_instances[t.name]
            for t in tools
            if budget_tracker.can_call(t.name)
        ]
        if not active_tools:
            llm_with_tools = llm  # force final response
        else:
            llm_with_tools = llm.bind_tools(active_tools)

    # Parse final response as output model if specified
    if output_model is not None:
        final_llm = llm.with_structured_output(output_model)
        return final_llm.invoke(messages, config=config)

    return None
