"""ReAct tool loop — agent/act mode LLM invocation with tool calling.

Extracted from _llm.py to keep the LLM module lean. This module owns
the tool loop, tool-call coercion, and tool-result rendering.

Import graph: _tool_loop → _llm (one-way, no cycles).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Any, NoReturn

import structlog
from langchain_core.runnables import RunnableConfig

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
from neograph._run_cache import aget_or_build, get_or_build
from neograph._tool_budget_preamble import render_tool_budget_preamble

# --- extracted cluster (neograph-3ffdg.13), re-exported so existing
# --- `from neograph._tool_loop import ...` call sites keep resolving unchanged.
from neograph._tool_call_coercion import (  # noqa: E402,F401
    UNPARSEABLE_ARGS_MARKER,
    _coerce_string_args_result,
    _CoercingToolWrapper,
    _empty_recovery_message,
    _string_args_tool_errors,
    _to_lc_messages,
    _unparseable_args_raw,
)
from neograph._usage import _usage_dict
from neograph.errors import ConfigurationError, ExecutionError
from neograph.renderers import Renderer, to_rendered

log = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════
# Tool result rendering
# ═══════════════════════════════════════════════════════════════════════════


def _render_tool_result_for_llm(result: Any, renderer: Renderer | None = None) -> str:
    """Render a typed tool result for the LLM's ToolMessage content.

    Delegates to ``to_rendered`` -- the ONE rendering ladder -- with this
    channel's only genuine difference, the ``"Tool result:"`` header. Before
    neograph-l2a7w this carried a third partial copy of the rule that ignored
    ``render_for_prompt()``, emitted the literal ``"None"`` for a None result,
    and shipped a Python repr for a ``dict[str, BaseModel]``.
    """
    return to_rendered(result, renderer, prefix="Tool result:")


# ═══════════════════════════════════════════════════════════════════════════
# Provider resilience: string tool_calls.args coercion
# ═══════════════════════════════════════════════════════════════════════════


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


def _raise_async_factory_error(tool_name: str, node_name: str) -> NoReturn:
    """Fail-loud when a sync run() drives an async (coroutine / awaitable-returning)
    tool factory. Mirrors the async-only-tool error at _agent_cycle.py:352 — a
    driver/config mismatch, not a runtime execution fault."""
    raise ConfigurationError.build(
        f"Tool '{tool_name}' has an async tool factory",
        expected="an async driver (arun())",
        found="sync run() driving an async tool factory",
        hint=(
            "This tool's factory is async (e.g. it awaits a per-run token "
            "provider or builds an MCP client). Drive the graph with arun() "
            "instead of run() so the async tool loop awaits the factory."
        ),
        node=node_name or None,
    )


def _lookup_factory(tool_name: str, per_compile_tools: dict[str, Any]) -> Any:
    factory = per_compile_tools.get(tool_name)
    if factory is None:
        raise ConfigurationError.build(
            f"Tool '{tool_name}' not registered",
            hint="Pass tool_factories={'" + tool_name + "': factory_fn} to compile().",
        )
    return factory


def _factory_tool_config(tool_spec: Any) -> dict[str, Any]:
    """The dict a tool factory receives as its second argument: the spec's
    ``config`` plus the spec-declared replay-safety flag.

    ``idempotent`` rides the factory-call channel so a factory-built wrapper
    (e.g. the MCP battery's transport retry) reads the SAME authority as the
    hydration replay gate — the node's ``Tool(idempotent=)`` — instead of a
    second hand-set flag that could drift. The spec flag wins over a same-named
    ``config`` key: ``Tool.idempotent`` is the declared contract."""
    return {**tool_spec.config, "idempotent": tool_spec.idempotent}


def _instantiate_tools(
    tools: list,
    tool_factory_lookup: dict[str, Any] | None,
    config: RunnableConfig | None,
    node_name: str,
) -> dict[str, Any]:
    """Sync tool instantiation. Fails loud on an async factory: a coroutine
    function is caught BEFORE it is called (so no un-awaited coroutine is
    created); a sync factory that returns an awaitable is caught after."""
    per_compile_tools = tool_factory_lookup or {}
    tool_instances: dict[str, Any] = {}
    for tool_spec in tools:
        factory = _lookup_factory(tool_spec.name, per_compile_tools)
        if asyncio.iscoroutinefunction(factory):
            _raise_async_factory_error(tool_spec.name, node_name)
        result = factory(config, _factory_tool_config(tool_spec))
        if inspect.isawaitable(result):
            # Close a coroutine to avoid the 'coroutine was never awaited' warning
            # (a non-coroutine awaitable has no close()).
            close = getattr(result, "close", None)
            if close is not None:
                close()
            _raise_async_factory_error(tool_spec.name, node_name)
        tool_instances[tool_spec.name] = result
    return tool_instances


async def _ainstantiate_tools(
    tools: list,
    tool_factory_lookup: dict[str, Any] | None,
    config: RunnableConfig | None,
    node_name: str,
) -> dict[str, Any]:
    """Async twin of _instantiate_tools: awaits coroutine / awaitable-returning
    factories (per-run token mint / MCP client build), passes sync factories
    through unchanged."""
    per_compile_tools = tool_factory_lookup or {}
    tool_instances: dict[str, Any] = {}
    for tool_spec in tools:
        factory = _lookup_factory(tool_spec.name, per_compile_tools)
        result = factory(config, _factory_tool_config(tool_spec))
        if inspect.isawaitable(result):
            result = await result
        tool_instances[tool_spec.name] = result
    return tool_instances


def _build_loop_preamble(
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
) -> tuple[Any, Any, Any, list]:
    """Shared preamble for both tool-loop twins: build llm + messages (schema and
    budget preambles). No tool instantiation, no network I/O. Returns
    (llm_log, cfg, llm, messages)."""
    llm_log = log.bind(
        tier=model_tier,
        prompt=prompt_template,
        tools=[t.name for t in tools],
        budgets={t.name: t.budget for t in tools},
    )

    cfg = _coerce_llm_config(llm_config)
    # Per-run live-handle reuse: within one run the agent cycle
    # rebuilds this preamble every superstep, so cache the LLM client on the
    # framework-minted RUN_ID (config-only, re-minted on resume -> rebuild). No
    # RUN_ID (graph invoked directly) -> _get_llm runs uncached every call.
    llm = get_or_build(
        config,
        f"llm:{node_name}:{model_tier}",
        lambda: _get_llm(runtime, model_tier, node_name=node_name, llm_config=cfg),
    )
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

    return llm_log, cfg, llm, messages


def _assemble_tool_loop_prep(
    *,
    runtime: LlmRuntime,
    llm_log: Any,
    cfg: Any,
    llm: Any,
    messages: list,
    tool_instances: dict[str, Any],
) -> _ToolLoopPrep:
    """Bind the instantiated tools and assemble the prep. Shared by both twins."""
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
    Sync driver path: an async tool factory fails loud (drive with arun())."""
    llm_log, cfg, llm, messages = _build_loop_preamble(
        runtime=runtime,
        model_tier=model_tier,
        prompt_template=prompt_template,
        input_data=input_data,
        output_model=output_model,
        tools=tools,
        config=config,
        node_name=node_name,
        llm_config=llm_config,
        context=context,
    )
    # Per-run tool-handle reuse: same RUN_ID keying as the LLM
    # handle — build the tool instances once per run, reuse across supersteps.
    tool_instances = get_or_build(
        config,
        f"tools:{node_name}",
        lambda: _instantiate_tools(tools, tool_factory_lookup, config, node_name),
    )
    return _assemble_tool_loop_prep(
        runtime=runtime,
        llm_log=llm_log,
        cfg=cfg,
        llm=llm,
        messages=messages,
        tool_instances=tool_instances,
    )


async def _aprepare_tool_loop(
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
    """Async twin of _prepare_tool_loop: awaits coroutine / awaitable-returning
    tool factories so per-run identity (token mint / MCP client build) is native
    on the arun() path. All other preamble work is identical to the sync twin."""
    llm_log, cfg, llm, messages = _build_loop_preamble(
        runtime=runtime,
        model_tier=model_tier,
        prompt_template=prompt_template,
        input_data=input_data,
        output_model=output_model,
        tools=tools,
        config=config,
        node_name=node_name,
        llm_config=llm_config,
        context=context,
    )
    # Per-run tool-handle reuse, async twin: await the factory
    # once per run (per-run token mint / MCP client build), reuse across supersteps.
    tool_instances = await aget_or_build(
        config,
        f"tools:{node_name}",
        lambda: _ainstantiate_tools(tools, tool_factory_lookup, config, node_name),
    )
    return _assemble_tool_loop_prep(
        runtime=runtime,
        llm_log=llm_log,
        cfg=cfg,
        llm=llm,
        messages=messages,
        tool_instances=tool_instances,
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

    usage_info = _usage_dict(total_input_tokens, total_output_tokens, empty={})

    llm_log.info(
        "llm_call",
        mode="react",
        loops=loop_count,
        tool_calls=total_tool_calls,
        duration_s=round(elapsed, 3),
        output=output_model.__name__,
        **usage_info,
    )
    _notify_cost(
        runtime,
        model_tier,
        usage_info if usage_info else None,
        node_name=node_name,
        mode="react",
        duration_s=round(elapsed, 3),
    )
    return parse_result, tool_interactions


def _raise_no_structured_output(output_model: Any) -> NoReturn:
    """Single-site fail-loud for the "structured fallback produced nothing" case,
    shared by the sync and async final-parse twins. ``from None`` suppresses the
    handled-ExecutionError context exactly as the inline raise did."""
    raise ExecutionError.build(
        "agent structured fallback produced no parseable output",
        expected=f"valid {output_model.__name__}",
        found="model returned no structured content and no recoverable markup",
        hint="The ReAct final turn was unparseable and the structured "
        "fallback returned nothing. Check the model/prompt or set "
        "output_strategy='json_mode'.",
    ) from None


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
                llm,
                messages,
                output_model,
                strategy,
                config,
                cfg=cfg,
                max_retries=max_retries,
            )
            if parse_result is None:
                _raise_no_structured_output(output_model)
        else:
            recovered = recover_dsml(
                raw_text,
                output_model,
                llm,
                messages,
                config,
                cfg,
                strategy=strategy,
            )
            if recovered is not None:
                parse_result = recovered
            else:
                parse_result, fallback_usage = _invoke_json_with_retry(
                    llm,
                    messages,
                    output_model,
                    config,
                    max_retries=max_retries,
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
                llm,
                messages,
                output_model,
                strategy,
                config,
                cfg=cfg,
                max_retries=max_retries,
            )
            if parse_result is None:
                _raise_no_structured_output(output_model)
        else:
            recovered = await arecover_dsml(
                raw_text,
                output_model,
                llm,
                messages,
                config,
                cfg,
                strategy=strategy,
            )
            if recovered is not None:
                parse_result = recovered
            else:
                parse_result, fallback_usage = await _ainvoke_json_with_retry(
                    llm,
                    messages,
                    output_model,
                    config,
                    max_retries=max_retries,
                )
    return parse_result, fallback_usage
