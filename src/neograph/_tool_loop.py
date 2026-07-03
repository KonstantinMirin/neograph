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
from neograph._llm_config import LlmConfig, _coerce_llm_config
from neograph._llm_dispatch import _acall_structured, _call_structured
from neograph._llm_render import _compile_prompt
from neograph._llm_retry import (
    _ainvoke_json_with_retry,
    _invoke_json_with_retry,
    _parse_json_response,
    arecover_dsml,
    recover_dsml,
)
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._tool_budget_preamble import render_tool_budget_preamble
from neograph.describe_type import describe_value
from neograph.errors import ConfigurationError, ExecutionError
from neograph.renderers import Renderer
from neograph.tool import Tool, ToolBudgetTracker, ToolInteraction

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


def invoke_with_tools(
    *args: Any,
    model_tier: str | None = None,
    prompt_template: str | None = None,
    input_data: Any = None,
    output_model: Any = None,
    tools: list[Tool] | None = None,
    budget_tracker: ToolBudgetTracker | None = None,
    config: RunnableConfig | None = None,
    node_name: str = "",
    llm_config: LlmConfig | dict | None = None,
    renderer: Renderer | None = None,
    context: dict[str, Any] | None = None,
    runtime: LlmRuntime | None = None,
    tool_factory_lookup: dict[str, Any] | None = None,
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

    # Resolve runtime: positional first-arg or fall back to compat
    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
    elif runtime is None:
        runtime = EMPTY_RUNTIME

    assert tools is not None  # mypy narrowing
    assert budget_tracker is not None
    prep = _prepare_tool_loop(
        runtime=runtime, model_tier=model_tier, prompt_template=prompt_template,
        input_data=input_data, output_model=output_model, tools=tools,
        config=config, node_name=node_name, llm_config=llm_config,
        context=context, tool_factory_lookup=tool_factory_lookup,
    )
    llm_log, cfg, llm, messages = prep.llm_log, prep.cfg, prep.llm, prep.messages
    tool_instances, llm_with_tools = prep.tool_instances, prep.llm_with_tools
    max_iterations, token_budget = prep.max_iterations, prep.token_budget
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
            try:
                result = tool_fn.invoke(tool_call["args"])
            except NotImplementedError as exc:
                # An async-only tool (e.g. an MCP tool loaded via
                # langchain-mcp-adapters) has no sync implementation. The sync
                # run() driver cannot execute it — the async driver arun() must
                # be used so the async tool loop (ainvoke) runs instead. Surface
                # a clear neograph error instead of leaking NotImplementedError.
                raise ConfigurationError.build(
                    f"Tool '{tool_name}' does not support synchronous invocation",
                    expected="an async driver (arun())",
                    found="sync run() driving an async-only tool",
                    hint=(
                        "This tool is async-only (e.g. an MCP tool). Drive the "
                        "graph with arun() instead of run() so the async tool "
                        "loop (ainvoke) is used."
                    ),
                    node=node_name or None,
                ) from exc
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

    # Parse the ReAct final turn as the node's structured output.
    # Agent/act mode parses messages[-1] directly — the schema was injected into
    # the loop above, so the final turn is the answer (0 extra calls on the happy
    # path). Only ON PARSE FAILURE do we fall back, and cfg.output_strategy
    # selects the fallback mechanism (not the primary path):
    #   'structured'         -> constrained decoding via _call_structured
    #                           (weak-model recourse; its own DSML recovery)
    #   'json_mode' / 'text' -> recover_dsml + _invoke_json_with_retry
    # The agent tail never returns None — an unrecoverable fallback raises.
    assert config is not None
    strategy = cfg.output_strategy
    max_retries = cfg.max_retries

    last_msg = messages[-1]
    raw_text = message_text(last_msg)
    fallback_usage = None
    try:
        parse_result = _parse_json_response(raw_text, output_model)
    except ExecutionError:
        if strategy == "structured":
            # Constrained-decoding fallback. _call_structured handles DSML
            # recovery internally (Raw(dsml=True)/Failed arms). It can surface a
            # silent None (parsed=None, no markup) — agent/act nodes must produce
            # a typed object or error, so raise rather than leak None downstream.
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
            # Layer C: DSML/XML tool-call markup in final response — strategy-
            # orthogonal recovery via shared helper. See neograph-0tid.
            recovered = recover_dsml(
                raw_text, output_model, llm, messages, config, cfg,
                strategy=strategy,
            )
            if recovered is not None:
                parse_result = recovered
            else:
                # Layer B: no DSML markup detected — fall through to generic retry.
                # NOTE (pre-existing, out of scope): recover_dsml discards its
                # retry usage, so a successful DSML recovery here under-counts
                # tokens. Not introduced by this change.
                parse_result, fallback_usage = _invoke_json_with_retry(
                    llm, messages, output_model, config, max_retries=max_retries,
                )

    return _finish_tool_loop(
        messages=messages, fallback_usage=fallback_usage, parse_result=parse_result,
        tool_interactions=tool_interactions, loop_count=loop_count,
        total_tool_calls=total_tool_calls, t0=t0, llm_log=llm_log, runtime=runtime,
        model_tier=model_tier, node_name=node_name, output_model=output_model,
    )


async def ainvoke_with_tools(
    *args: Any,
    model_tier: str | None = None,
    prompt_template: str | None = None,
    input_data: Any = None,
    output_model: Any = None,
    tools: list[Tool] | None = None,
    budget_tracker: ToolBudgetTracker | None = None,
    config: RunnableConfig | None = None,
    node_name: str = "",
    llm_config: LlmConfig | dict | None = None,
    renderer: Renderer | None = None,
    context: dict[str, Any] | None = None,
    runtime: LlmRuntime | None = None,
    tool_factory_lookup: dict[str, Any] | None = None,
) -> tuple[BaseModel | None, list]:
    """Async twin of :func:`invoke_with_tools` (Phase 1c).

    Shares the pure preamble (_prepare_tool_loop) and postamble
    (_finish_tool_loop) verbatim. The ReAct while-loop and the parse-fallback
    tail are duplicated because their control flow interleaves awaits; the only
    per-line divergence from the sync loop is awaiting the network seams
    (llm_with_tools.ainvoke, tool_fn.ainvoke, and the _acall_structured /
    arecover_dsml / _ainvoke_json_with_retry fallbacks). config threads into every
    awaited hop (M6a).
    """
    from langchain_core.messages import ToolMessage

    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
    elif runtime is None:
        runtime = EMPTY_RUNTIME

    assert tools is not None
    assert budget_tracker is not None
    prep = _prepare_tool_loop(
        runtime=runtime, model_tier=model_tier, prompt_template=prompt_template,
        input_data=input_data, output_model=output_model, tools=tools,
        config=config, node_name=node_name, llm_config=llm_config,
        context=context, tool_factory_lookup=tool_factory_lookup,
    )
    llm_log, cfg, llm, messages = prep.llm_log, prep.cfg, prep.llm, prep.messages
    tool_instances, llm_with_tools = prep.tool_instances, prep.llm_with_tools
    max_iterations, token_budget = prep.max_iterations, prep.token_budget
    cumulative_input_tokens = 0

    loop_count = 0
    total_tool_calls = 0
    tool_interactions: list[ToolInteraction] = []
    _guard_fired = False
    t0 = time.monotonic()

    while True:
        loop_count += 1
        response = await llm_with_tools.ainvoke(messages, config=config)
        messages.append(response)

        response_usage = getattr(response, "usage_metadata", None) or {}
        cumulative_input_tokens += response_usage.get("input_tokens", 0)

        if not response.tool_calls:
            llm_log.debug("react_final_response", loop=loop_count)
            break

        if _guard_fired:
            llm_log.warning("react_guard_forced_break", loops=loop_count, tool_calls=total_tool_calls)
            break

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
            result = await tool_fn.ainvoke(tool_call["args"])
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

        if budget_tracker.all_exhausted():
            llm_log.info("all_tools_exhausted", exhausted=budget_tracker.exhausted_tools(), forcing_response=True)
            llm_with_tools = llm
        else:
            active_tools = [tool_instances[t.name] for t in tools if budget_tracker.can_call(t.name)]
            llm_with_tools = _CoercingToolWrapper(llm.bind_tools(active_tools))

    assert config is not None
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

    return _finish_tool_loop(
        messages=messages, fallback_usage=fallback_usage, parse_result=parse_result,
        tool_interactions=tool_interactions, loop_count=loop_count,
        total_tool_calls=total_tool_calls, t0=t0, llm_log=llm_log, runtime=runtime,
        model_tier=model_tier, node_name=node_name, output_model=output_model,
    )
