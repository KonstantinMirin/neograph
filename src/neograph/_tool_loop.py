"""ReAct tool loop — agent/act mode LLM invocation with tool calling.

Extracted from _llm.py to keep the LLM module lean. This module owns
the tool loop, tool-call coercion, and tool-result rendering.

Import graph: _tool_loop → _llm (one-way, no cycles).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._agent_output_schema_preamble import render_output_schema_preamble
from neograph._dsml import message_text
from neograph._llm import _get_llm, _notify_cost
from neograph._llm_config import _coerce_llm_config
from neograph._llm_dispatch import _acall_structured, _call_structured
from neograph._llm_render import _compile_prompt
from neograph._llm_retry import (
    _ainvoke_json_with_retry,
    _invoke_json_with_retry,
    _parse_json_response,
    arecover_dsml,
    recover_dsml,
)
from neograph._llm_runtime import LlmRuntime
from neograph._tool_budget_preamble import render_tool_budget_preamble
from neograph.describe_type import describe_value
from neograph.errors import ConfigurationError, ExecutionError
from neograph.renderers import Renderer

log = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════
# Tool result rendering
# ═══════════════════════════════════════════════════════════════════════════


def _render_tool_result_for_llm(result: Any, renderer: Renderer | None = None) -> str:
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
        from pydantic import ValidationError

        try:
            return self._bound.invoke(messages, **kwargs)
        except ValidationError as exc:
            if not _string_args_tool_errors(exc):
                raise
            try:
                raw_result = self._bound._generate(_to_lc_messages(messages), run_manager=None)
                coerced = _coerce_string_args_result(raw_result)
                if coerced is not None:
                    return coerced
            except Exception as inner:
                _log_coercion_generate_failed(inner)
            from langchain_core.messages import AIMessage
            return AIMessage(content="")

    async def ainvoke(self, messages: list, **kwargs: Any) -> Any:
        # MUST be an explicit override — __getattr__ would forward `ainvoke` to
        # self._bound and silently bypass string-args coercion under arun.
        # **kwargs forwards config through (config rides in kwargs).
        from pydantic import ValidationError

        try:
            return await self._bound.ainvoke(messages, **kwargs)
        except ValidationError as exc:
            if not _string_args_tool_errors(exc):
                raise
            try:
                raw_result = await self._bound._agenerate(_to_lc_messages(messages), run_manager=None)
                coerced = _coerce_string_args_result(raw_result)
                if coerced is not None:
                    return coerced
            except Exception as inner:
                _log_coercion_generate_failed(inner)
            from langchain_core.messages import AIMessage
            return AIMessage(content="")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bound, name)


def _string_args_tool_errors(exc: Any) -> list:
    """Pure: the tool_calls string-args ValidationError entries (empty if none).

    Shared by _CoercingToolWrapper.invoke and .ainvoke; a non-empty list logs
    the coercion warning and triggers the _generate/_agenerate recovery.
    """
    tool_call_errors = [
        e for e in exc.errors()
        if "tool_calls" in str(e.get("loc", "")) and e.get("type") == "dict_type"
    ]
    if tool_call_errors:
        log.warning("tool_calls_args_coercion",
                     error_count=len(tool_call_errors),
                     hint="provider returned tool_calls.args as JSON string; "
                          "reconstructing via additional_kwargs path")
    return tool_call_errors


def _to_lc_messages(messages: list) -> list:
    """Pure: convert dict messages to LangChain message objects for _generate."""
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    lc_messages: list[BaseMessage | dict] = []
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
    return lc_messages


def _coerce_string_args_result(raw_result: Any) -> Any | None:
    """Pure: extract the message from a _generate/_agenerate result and json-load
    any string tool_call args. Returns the coerced message, or None if empty."""
    import json as _json

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
    return None


def _log_coercion_generate_failed(inner: Exception) -> None:
    log.warning("tool_calls_coercion_generate_failed",
                 error=str(inner),
                 hint="falling back to empty response")


# ═══════════════════════════════════════════════════════════════════════════
# Tool loop
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _ToolLoopPrep:
    """Pre-loop state shared by invoke_with_tools and ainvoke_with_tools."""

    runtime: LlmRuntime
    llm_log: Any
    cfg: Any
    llm: Any
    messages: list
    tool_instances: dict
    llm_with_tools: Any
    max_iterations: int
    token_budget: int | None


def _prepare_tool_loop(
    *,
    runtime: LlmRuntime,
    model_tier: str | None,
    prompt_template: str | None,
    input_data: Any,
    output_model: Any,
    tools: list,
    config: RunnableConfig | None,
    node_name: str,
    llm_config: Any,
    context: dict[str, Any] | None,
    tool_factory_lookup: dict[str, Any] | None,
) -> _ToolLoopPrep:
    """Pure preamble: build llm, messages (+schema/budget preambles), tool
    instances, and the tool-bound wrapper. No network I/O (bind_tools is local).
    Shared verbatim by the sync and async tool loops."""
    llm_log = log.bind(
        tier=model_tier,
        prompt=prompt_template,
        tools=[t.name for t in tools],
        budgets={t.name: t.budget for t in tools},
    )

    cfg = _coerce_llm_config(llm_config)
    llm = _get_llm(runtime, model_tier, node_name=node_name, llm_config=cfg)
    messages = list(
        _compile_prompt(
            runtime,
            prompt_template,
            input_data,
            node_name=node_name,
            config=config,
            output_model=output_model,
            llm_config=cfg,
            context=context,
        )
    )

    messages.insert(
        0,
        {
            "role": "system",
            "content": render_output_schema_preamble(output_model),
        },
    )

    if cfg.announce_tool_budget:
        messages.insert(
            0,
            {
                "role": "system",
                "content": render_tool_budget_preamble(tools, cfg.max_iterations),
            },
        )

    per_compile_tools = tool_factory_lookup or {}
    tool_instances = {}
    for tool_spec in tools:
        factory = per_compile_tools.get(tool_spec.name)
        if factory is None:
            raise ConfigurationError.build(
                f"Tool '{tool_spec.name}' not registered",
                hint="Pass tool_factories={'" + tool_spec.name + "': factory_fn} to compile().",
            )
        tool_instances[tool_spec.name] = factory(config, tool_spec.config)

    active_tools = list(tool_instances.values())
    llm_with_tools = _CoercingToolWrapper(llm.bind_tools(active_tools))

    return _ToolLoopPrep(
        runtime=runtime,
        llm_log=llm_log,
        cfg=cfg,
        llm=llm,
        messages=messages,
        tool_instances=tool_instances,
        llm_with_tools=llm_with_tools,
        max_iterations=cfg.max_iterations,
        token_budget=cfg.token_budget,
    )


def _finish_tool_loop(
    *,
    messages: list,
    fallback_usage: Any,
    parse_result: Any,
    tool_interactions: list,
    loop_count: int,
    total_tool_calls: int,
    t0: float,
    llm_log: Any,
    runtime: LlmRuntime,
    model_tier: str | None,
    node_name: str,
    output_model: Any,
) -> tuple:
    """Pure postamble: sum usage across messages (+ any fallback), log, notify
    cost. Shared verbatim by the sync and async tool loops."""
    elapsed = time.monotonic() - t0

    total_input_tokens = 0
    total_output_tokens = 0
    for msg in messages:
        msg_usage = getattr(msg, "usage_metadata", None)
        if msg_usage:
            total_input_tokens += msg_usage.get("input_tokens", 0)
            total_output_tokens += msg_usage.get("output_tokens", 0)
    if fallback_usage:
        total_input_tokens += fallback_usage.get("input_tokens", 0)
        total_output_tokens += fallback_usage.get("output_tokens", 0)

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
    _notify_cost(runtime, model_tier, usage_info if usage_info else None,
                 node_name=node_name, mode="react", duration_s=round(elapsed, 3))
    return parse_result, tool_interactions


def _parse_final_turn(
    *,
    messages: list,
    output_model: Any,
    cfg: Any,
    config: RunnableConfig,
    llm: Any,
) -> tuple[Any, Any]:
    """Parse the ReAct final turn as the node's structured output.

    Agent/act parses ``messages[-1]`` directly (the schema was seeded into the
    loop, so the final turn is the answer — 0 extra calls on the happy path).
    Only ON PARSE FAILURE does it fall back, with ``cfg.output_strategy``
    selecting the fallback mechanism (not the primary path):
      'structured'         -> constrained decoding via _call_structured
      'json_mode' / 'text' -> recover_dsml + _invoke_json_with_retry
    Never returns None — an unrecoverable fallback raises. Returns
    ``(parse_result, fallback_usage)``.

    Single source of truth for the hard cluster — shared by the monolithic tool
    loop and the inline agent-cycle parse node so they cannot drift.
    """
    strategy = cfg.output_strategy
    max_retries = cfg.max_retries

    last_msg = messages[-1]
    raw_text = message_text(last_msg)
    fallback_usage = None
    try:
        parse_result = _parse_json_response(raw_text, output_model)
    except ExecutionError:
        if strategy == "structured":
            parse_result, fallback_usage = _call_structured(
                llm, messages, output_model, strategy, config,
                cfg=cfg, max_retries=max_retries,
            )
            if parse_result is None:
                raise ExecutionError.build(
                    "agent structured fallback produced no parseable output",
                    expected=f"valid {output_model.__name__}",
                    found="model returned no structured content and no recoverable markup",
                    hint="The ReAct final turn was unparseable and the structured "
                         "fallback returned nothing. Check the model/prompt or set "
                         "output_strategy='json_mode'.",
                ) from None
        else:
            recovered = recover_dsml(
                raw_text, output_model, llm, messages, config, cfg,
                strategy=strategy,
            )
            if recovered is not None:
                parse_result = recovered
            else:
                parse_result, fallback_usage = _invoke_json_with_retry(
                    llm, messages, output_model, config, max_retries=max_retries,
                )
    return parse_result, fallback_usage


async def _aparse_final_turn(
    *,
    messages: list,
    output_model: Any,
    cfg: Any,
    config: RunnableConfig,
    llm: Any,
) -> tuple[Any, Any]:
    """Async twin of :func:`_parse_final_turn` — awaits the async fallback seams."""
    strategy = cfg.output_strategy
    max_retries = cfg.max_retries

    last_msg = messages[-1]
    raw_text = message_text(last_msg)
    fallback_usage = None
    try:
        parse_result = _parse_json_response(raw_text, output_model)
    except ExecutionError:
        if strategy == "structured":
            parse_result, fallback_usage = await _acall_structured(
                llm, messages, output_model, strategy, config,
                cfg=cfg, max_retries=max_retries,
            )
            if parse_result is None:
                raise ExecutionError.build(
                    "agent structured fallback produced no parseable output",
                    expected=f"valid {output_model.__name__}",
                    found="model returned no structured content and no recoverable markup",
                    hint="The ReAct final turn was unparseable and the structured "
                         "fallback returned nothing. Check the model/prompt or set "
                         "output_strategy='json_mode'.",
                ) from None
        else:
            recovered = await arecover_dsml(
                raw_text, output_model, llm, messages, config, cfg,
                strategy=strategy,
            )
            if recovered is not None:
                parse_result = recovered
            else:
                parse_result, fallback_usage = await _ainvoke_json_with_retry(
                    llm, messages, output_model, config, max_retries=max_retries,
                )
    return parse_result, fallback_usage

