"""ReAct tool loop — agent/act mode LLM invocation with tool calling.

Extracted from _llm.py to keep the LLM module lean. This module owns
the tool loop, tool-call coercion, and tool-result rendering.

Import graph: _tool_loop → _llm (one-way, no cycles).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._llm import (
    _call_structured,
    _compile_prompt,
    _get_llm,
    _invoke_json_with_retry,
    _notify_cost,
    _parse_json_response,
)
from neograph._registry import registry
from neograph.describe_type import describe_value
from neograph.errors import ConfigurationError, ExecutionError
from neograph.tool import Tool, ToolBudgetTracker, ToolInteraction

log = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════
# Tool result rendering
# ═══════════════════════════════════════════════════════════════════════════


def _render_tool_result_for_llm(result: Any, renderer: Any = None) -> str:
    """Render a typed tool result for the LLM's ToolMessage content.

    When ``renderer`` is provided, it is used for Pydantic models and lists
    of models.  Otherwise falls back to ``describe_value`` (BAML-style
    notation with field descriptions as inline ``//`` comments).
    Falls back to str() for non-Pydantic returns.
    """
    if isinstance(result, BaseModel) or (isinstance(result, list) and result and isinstance(result[0], BaseModel)):
        if renderer is not None:
            return renderer.render(result)
        return describe_value(result, prefix="Tool result:")

    return str(result)


# ═══════════════════════════════════════════════════════════════════════════
# Provider resilience: string tool_calls.args coercion
# ═══════════════════════════════════════════════════════════════════════════


class _CoercingToolWrapper:
    """Wraps a tool-bound LLM to coerce string tool_calls.args to dicts.

    Some providers (DeepSeek R1 via OpenRouter) emit tool_calls with
    ``args`` as a JSON string. LangChain AIMessage Pydantic validation
    rejects this. This wrapper catches the ValidationError and
    reconstructs the AIMessage via the ``additional_kwargs`` path which
    handles string arguments correctly (``default_tool_parser`` calls
    ``json.loads`` on them).

    Usage (automatic — applied by ``invoke_with_tools``)::

        wrapped = _CoercingToolWrapper(llm.bind_tools(tools))
        response = wrapped.invoke(messages)  # never raises for string args
    """

    def __init__(self, bound_llm: Any):
        self._bound = bound_llm

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        import json as _json

        from langchain_core.messages import AIMessage
        from pydantic import ValidationError

        try:
            return self._bound.invoke(messages, **kwargs)
        except ValidationError as exc:
            errors = exc.errors()
            tool_call_errors = [
                e for e in errors
                if "tool_calls" in str(e.get("loc", "")) and e.get("type") == "dict_type"
            ]
            if not tool_call_errors:
                raise

            log.warning("tool_calls_args_coercion",
                         error_count=len(tool_call_errors),
                         hint="provider returned tool_calls.args as JSON string; "
                              "reconstructing via additional_kwargs path")

            try:
                from langchain_core.messages import HumanMessage

                lc_messages = []
                for m in messages:
                    if isinstance(m, dict):
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if role == "user":
                            lc_messages.append(HumanMessage(content=content))
                        else:
                            lc_messages.append(AIMessage(content=content) if role == "assistant" else m)
                    else:
                        lc_messages.append(m)

                raw_result = self._bound._generate(lc_messages, run_manager=None)
                if raw_result.generations:
                    gen = raw_result.generations[0]
                    raw_msg = gen.message if hasattr(gen, 'message') else gen
                    if hasattr(raw_msg, 'tool_calls'):
                        for tc in raw_msg.tool_calls:
                            if isinstance(tc.get("args"), str):
                                try:
                                    tc["args"] = _json.loads(tc["args"])
                                except (_json.JSONDecodeError, TypeError):
                                    tc["args"] = {}
                    return raw_msg
            except Exception as inner:
                log.warning("tool_calls_coercion_generate_failed",
                             error=str(inner),
                             hint="falling back to empty response")

            return AIMessage(content="")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bound, name)


# ═══════════════════════════════════════════════════════════════════════════
# Tool loop
# ═══════════════════════════════════════════════════════════════════════════


def invoke_with_tools(
    model_tier: str,
    prompt_template: str,
    input_data: Any,
    output_model: type[BaseModel],
    tools: list[Tool],
    budget_tracker: ToolBudgetTracker,
    config: RunnableConfig,
    node_name: str = "",
    llm_config: dict | None = None,
    renderer: Any = None,
    context: dict[str, Any] | None = None,
) -> tuple[BaseModel | None, list]:
    """ReAct tool loop with per-tool budget enforcement. Mode: agent/act.

    Returns (parsed_result, tool_interactions) where tool_interactions is a
    list of ToolInteraction records from the ReAct loop.

    Loop guards (configured via llm_config):

    - ``max_iterations`` (int, default 20): maximum ReAct iterations before
      forcing a final response. When hit, tool calls are skipped and the LLM
      is asked to respond immediately.
    - ``token_budget`` (int or None, default None): if set, abort the loop
      when cumulative input tokens exceed this threshold.
    """
    from langchain_core.messages import ToolMessage

    llm_log = log.bind(
        tier=model_tier,
        prompt=prompt_template,
        tools=[t.name for t in tools],
        budgets={t.name: t.budget for t in tools},
    )

    llm = _get_llm(model_tier, node_name=node_name, llm_config=llm_config)
    messages = list(
        _compile_prompt(
            prompt_template,
            input_data,
            node_name=node_name,
            config=config,
            output_model=output_model,
            llm_config=llm_config,
            context=context,
        )
    )

    # Create tool instances from registered factories
    tool_instances = {}
    for tool_spec in tools:
        if tool_spec.name not in registry.tool_factory:
            raise ConfigurationError.build(
                f"Tool '{tool_spec.name}' not registered",
                hint="Call register_tool_factory() to register the tool before using it.",
            )
        factory = registry.tool_factory[tool_spec.name]
        tool_instances[tool_spec.name] = factory(config, tool_spec.config)

    active_tools = list(tool_instances.values())
    llm_with_tools = _CoercingToolWrapper(llm.bind_tools(active_tools))

    llm_config = llm_config or {}
    max_iterations = llm_config.get("max_iterations", 20)
    token_budget = llm_config.get("token_budget")
    cumulative_input_tokens = 0

    loop_count = 0
    total_tool_calls = 0
    tool_interactions: list[ToolInteraction] = []
    _guard_fired = False
    t0 = time.monotonic()

    while True:
        loop_count += 1
        response = llm_with_tools.invoke(messages, config=config)
        messages.append(response)

        # Track token usage across iterations
        response_usage = getattr(response, "usage_metadata", None) or {}
        cumulative_input_tokens += response_usage.get("input_tokens", 0)

        if not response.tool_calls:
            llm_log.debug("react_final_response", loop=loop_count)
            break

        # Safety break: if the guard already fired (tools unbound) but the LLM
        # still returns tool_calls, force-break instead of looping forever.
        if _guard_fired:
            llm_log.warning("react_guard_forced_break", loops=loop_count, tool_calls=total_tool_calls)
            break

        # Check iteration and token budget guards
        max_iter_hit = loop_count >= max_iterations
        budget_hit = token_budget is not None and cumulative_input_tokens > token_budget

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]

            if max_iter_hit or budget_hit:
                messages.append(
                    ToolMessage(
                        content=(
                            f"React loop limit reached"
                            f" ({'max iterations' if max_iter_hit else 'token budget'})."
                            f" Provide your final answer now."
                        ),
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            if not budget_tracker.can_call(tool_name):
                llm_log.info("tool_budget_exhausted", tool=tool_name)
                messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' budget exhausted. Use remaining tools or respond.",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            tool_fn = tool_instances.get(tool_name)
            if tool_fn is None:
                messages.append(
                    ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_call["id"],
                    )
                )
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

            rendered = _render_tool_result_for_llm(result, renderer)

            tool_interactions.append(
                ToolInteraction(
                    tool_name=tool_name,
                    args=tool_call.get("args", {}),
                    result=rendered,
                    typed_result=result,
                    duration_ms=int(tool_elapsed * 1000),
                )
            )

            messages.append(
                ToolMessage(
                    content=rendered,
                    tool_call_id=tool_call["id"],
                )
            )

        # Unbind tools if a loop guard was hit
        if max_iter_hit or budget_hit:
            reason = (
                "max_iterations+token_budget"
                if max_iter_hit and budget_hit
                else ("max_iterations" if max_iter_hit else "token_budget")
            )
            llm_log.warning(
                f"react_{reason}_exceeded",
                max_iterations=max_iterations,
                token_budget=token_budget,
                cumulative_input_tokens=cumulative_input_tokens,
                loops=loop_count,
                tool_calls=total_tool_calls,
            )
            llm_with_tools = llm
            _guard_fired = True
            continue

        # Check if all budgeted tools are spent
        if budget_tracker.all_exhausted():
            llm_log.info("all_tools_exhausted", exhausted=budget_tracker.exhausted_tools(), forcing_response=True)
            llm_with_tools = llm
        else:
            active_tools = [tool_instances[t.name] for t in tools if budget_tracker.can_call(t.name)]
            llm_with_tools = _CoercingToolWrapper(llm.bind_tools(active_tools))

    elapsed = time.monotonic() - t0

    # Parse final response as structured output — strategy-aware
    strategy = llm_config.get("output_strategy", "structured")
    max_retries = llm_config.get("max_retries", 1)

    if strategy in ("json_mode", "text"):
        last_msg = messages[-1]
        raw_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        try:
            parse_result = _parse_json_response(raw_text, output_model)
        except ExecutionError:
            parse_result, _ = _invoke_json_with_retry(llm, messages, output_model, config, max_retries=max_retries)
        usage = getattr(last_msg, "usage_metadata", None)
    else:
        parse_result, usage = _call_structured(llm, messages, output_model, strategy, config, max_retries=max_retries)

    # Collect total usage
    total_input_tokens = 0
    total_output_tokens = 0
    for msg in messages:
        msg_usage = getattr(msg, "usage_metadata", None)
        if msg_usage:
            total_input_tokens += msg_usage.get("input_tokens", 0)
            total_output_tokens += msg_usage.get("output_tokens", 0)
    if usage and strategy not in ("json_mode", "text"):
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
    _notify_cost(model_tier, usage_info if usage_info else None,
                 node_name=node_name, mode="react", duration_s=round(elapsed, 3))
    return parse_result, tool_interactions
