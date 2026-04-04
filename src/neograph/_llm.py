"""LLM invocation layer — thin abstraction over LangChain chat models.

This module is intentionally minimal. NeoGraph does NOT own the LLM client
configuration — consumers register their model routing. This module provides
the dispatch mechanics (structured output, ReAct loop, tool budget enforcement).

Observability: callbacks (e.g. Langfuse CallbackHandler) flow through via
RunnableConfig. If the consumer wires them at the run() call site, every
LLM call here inherits them automatically. If not, nothing happens.
Structlog captures the framework-level view regardless.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import structlog
from pydantic import BaseModel

from neograph.tool import Tool, ToolBudgetTracker

log = structlog.get_logger()

# Consumer-provided LLM factory
_llm_factory: Callable[[str], Any] | None = None

# Consumer-provided prompt compiler
_prompt_compiler: Callable[[str, Any], list] | None = None


def configure_llm(
    llm_factory: Callable,
    prompt_compiler: Callable[[str, Any], list],
) -> None:
    """Configure NeoGraph's LLM layer.

    Args:
        llm_factory: Callable that creates LLM instances. Signature:
                     (tier: str, node_name: str, llm_config: dict) → BaseChatModel
                     Or the simpler (tier: str) → BaseChatModel for basic usage.
        prompt_compiler: Callable that takes (template_name, input_data) and
                        returns list[BaseMessage].

    Usage:
        # Simple: just tier routing
        def my_factory(tier, **kwargs):
            return ChatOpenAI(model={"fast": "gpt-4o-mini", "reason": "o1"}[tier])

        # Advanced: per-node configuration
        def my_factory(tier, node_name=None, llm_config=None):
            llm_config = llm_config or {}
            return ChatOpenAI(
                model={"fast": "gpt-4o-mini", "reason": "o1"}[tier],
                temperature=llm_config.get("temperature", 0),
                max_tokens=llm_config.get("max_tokens"),
                timeout=llm_config.get("timeout", 30),
            )

        configure_llm(llm_factory=my_factory, prompt_compiler=my_prompt_compiler)
    """
    global _llm_factory, _prompt_compiler  # noqa: PLW0603
    _llm_factory = llm_factory
    _prompt_compiler = prompt_compiler


def _get_llm(tier: str, node_name: str = "", llm_config: dict | None = None) -> Any:
    if _llm_factory is None:
        msg = "LLM not configured. Call neograph.configure_llm() first."
        raise RuntimeError(msg)
    try:
        return _llm_factory(tier, node_name=node_name, llm_config=llm_config or {})
    except TypeError:
        # Backward compatible: factory only accepts (tier,)
        return _llm_factory(tier)


def _compile_prompt(template: str, input_data: Any) -> list:
    return _prompt_compiler(template, input_data)


def invoke_structured(
    model_tier: str,
    prompt_template: str,
    input_data: Any,
    output_model: type[BaseModel],
    config: dict,
    node_name: str = "",
    llm_config: dict | None = None,
) -> BaseModel:
    """Single LLM call with structured JSON output. Mode: produce."""
    llm_log = log.bind(tier=model_tier, prompt=prompt_template, output=output_model.__name__)

    llm = _get_llm(model_tier, node_name=node_name, llm_config=llm_config)
    messages = _compile_prompt(prompt_template, input_data)
    try:
        structured_llm = llm.with_structured_output(output_model, include_raw=True)
        include_raw = True
    except TypeError:
        structured_llm = llm.with_structured_output(output_model)
        include_raw = False

    t0 = time.monotonic()
    raw_result = structured_llm.invoke(messages, config=config)
    elapsed = time.monotonic() - t0

    # Extract parsed model and usage metadata
    if include_raw and isinstance(raw_result, dict) and "parsed" in raw_result:
        result = raw_result["parsed"]
        raw_msg = raw_result.get("raw")
        usage = getattr(raw_msg, "usage_metadata", None) if raw_msg else None
    else:
        result = raw_result
        usage = None

    usage_info = {}
    if usage:
        usage_info = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    llm_log.info("llm_call", mode="produce", duration_s=round(elapsed, 3), **usage_info)
    return result


def invoke_with_tools(
    model_tier: str,
    prompt_template: str,
    input_data: Any,
    output_model: type[BaseModel],
    tools: list[Tool],
    budget_tracker: ToolBudgetTracker,
    config: dict,
    node_name: str = "",
    llm_config: dict | None = None,
) -> BaseModel | None:
    """ReAct tool loop with per-tool budget enforcement. Mode: gather/execute."""
    from langchain_core.messages import ToolMessage

    from neograph.factory import _tool_factory_registry

    llm_log = log.bind(
        tier=model_tier,
        prompt=prompt_template,
        tools=[t.name for t in tools],
        budgets={t.name: t.budget for t in tools},
    )

    llm = _get_llm(model_tier, node_name=node_name, llm_config=llm_config)
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

    loop_count = 0
    total_tool_calls = 0
    t0 = time.monotonic()

    while True:
        loop_count += 1
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)

        if not response.tool_calls:
            llm_log.debug("react_final_response", loop=loop_count)
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]

            if not budget_tracker.can_call(tool_name):
                llm_log.info("tool_budget_exhausted", tool=tool_name)
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

            tool_t0 = time.monotonic()
            result = tool_fn.invoke(tool_call["args"])
            tool_elapsed = time.monotonic() - tool_t0

            budget_tracker.record_call(tool_name)
            total_tool_calls += 1
            llm_log.debug(
                "tool_call",
                tool=tool_name,
                call_num=total_tool_calls,
                duration_s=round(tool_elapsed, 3),
            )

            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ))

        # Check if all budgeted tools are spent
        if budget_tracker.all_exhausted():
            llm_log.info("all_tools_exhausted",
                         exhausted=budget_tracker.exhausted_tools(),
                         forcing_response=True)
            llm_with_tools = llm  # unbind tools, force final response
        else:
            # Rebind with only tools that still have budget
            active_tools = [
                tool_instances[t.name]
                for t in tools
                if budget_tracker.can_call(t.name)
            ]
            llm_with_tools = llm.bind_tools(active_tools)

    elapsed = time.monotonic() - t0

    # Parse final response as structured output
    try:
        final_llm = llm.with_structured_output(output_model, include_raw=True)
        final_include_raw = True
    except TypeError:
        final_llm = llm.with_structured_output(output_model)
        final_include_raw = False

    raw_result = final_llm.invoke(messages, config=config)

    if final_include_raw and isinstance(raw_result, dict) and "parsed" in raw_result:
        parse_result = raw_result["parsed"]
        raw_msg = raw_result.get("raw")
        usage = getattr(raw_msg, "usage_metadata", None) if raw_msg else None
    else:
        parse_result = raw_result
        usage = None

    # Collect total usage from all LLM calls in the loop
    total_input_tokens = 0
    total_output_tokens = 0
    for msg in messages:
        msg_usage = getattr(msg, "usage_metadata", None)
        if msg_usage:
            total_input_tokens += msg_usage.get("input_tokens", 0)
            total_output_tokens += msg_usage.get("output_tokens", 0)
    # Add final structured parse call
    if usage:
        total_input_tokens += usage.get("input_tokens", 0)
        total_output_tokens += usage.get("output_tokens", 0)

    usage_info = {}
    if total_input_tokens or total_output_tokens:
        usage_info = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        }

    llm_log.info(
        "llm_call",
        mode="react",
        loops=loop_count,
        tool_calls=total_tool_calls,
        duration_s=round(elapsed, 3),
        output=output_model.__name__,
        **usage_info,
    )
    return parse_result
