"""Produce-mode orchestration — the `invoke_structured` entry plus the two
runtime adapters (LLM resolution and cost callback dispatch) it owns.

Single-responsibility: orchestrate a single produce-mode call against a
user-supplied LLM, threading the runtime bundle and emitting telemetry.
Change axis: how a produce-mode call is staged (resolve LLM → compile
prompt → dispatch strategy → log/emit cost).

Prompt rendering lives in `_llm_render`; JSON parsing / retry lives in
`_llm_retry`; output-strategy dispatch lives in `_llm_dispatch`. The
Protocols + LlmRuntime + LlmConfig types are in `_llm_protocols`,
`_llm_runtime`, and `_llm_config` respectively.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._llm_config import LlmConfig, _coerce_llm_config
from neograph._llm_dispatch import _acall_structured, _call_structured
from neograph._llm_protocols import CostCallback, LlmFactory, PromptCompiler  # noqa: F401 — re-exported
from neograph._llm_render import _compile_prompt, render_prompt  # noqa: F401 — re-exported
from neograph._llm_runtime import (
    _ACCEPT_ALL,
    EMPTY_RUNTIME,
    LlmRuntime,
    _accepted_params,  # noqa: F401 — re-exported
)
from neograph.describe_type import describe_type
from neograph.errors import ConfigurationError

log = structlog.get_logger()


def _notify_cost(
    *args: Any,
    runtime: LlmRuntime | None = None,
    node_name: str | None = None,
    mode: str | None = None,
    duration_s: float | None = None,
) -> None:
    """Dispatch cost callback if configured.

    Extra kwargs (node_name, mode, duration_s) are passed through to the
    callback. Old-style callbacks that only accept (tier, input_tokens,
    output_tokens) are supported via TypeError fallback.

    Accepts both call shapes:
        _notify_cost(runtime, tier, usage, ...)  # post-§2
        _notify_cost(tier, usage, ...)           # legacy — uses compat runtime
    """
    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
        tier = args[1] if len(args) > 1 else ""
        usage = args[2] if len(args) > 2 else None
    else:
        tier = args[0] if args else ""
        usage = args[1] if len(args) > 1 else None
        if runtime is None:
            runtime = EMPTY_RUNTIME
    cb = runtime.cost_callback
    if cb is None or usage is None:
        return
    kwargs: dict = {
        "tier": tier,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    if node_name is not None:
        kwargs["node_name"] = node_name
    if mode is not None:
        kwargs["mode"] = mode
    if duration_s is not None:
        kwargs["duration_s"] = duration_s
    try:
        cb(**kwargs)
    except TypeError:
        try:
            cb(
                tier=tier,
                input_tokens=kwargs["input_tokens"],
                output_tokens=kwargs["output_tokens"],
            )
        except TypeError as exc:
            log.warning("cost_callback_failed", error=str(exc))


def _get_llm(
    *args: Any,
    runtime: LlmRuntime | None = None,
    node_name: str = "",
    llm_config: LlmConfig | dict | None = None,
) -> Any:
    """Resolve an LLM client for *tier* from the runtime's factory.

    Backward-compatible argument shapes:
        _get_llm(runtime, tier, ...)   # post-§2 (positional)
        _get_llm(tier, ...)            # legacy — runtime defaults to compat
        _get_llm(tier, runtime=...)    # explicit keyword
    """
    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
        tier = args[1] if len(args) > 1 else None
    else:
        tier = args[0] if args else None
        if runtime is None:
            runtime = EMPTY_RUNTIME
    if runtime.llm_factory is None:
        raise ConfigurationError.build(
            "LLM not configured",
            hint="Pass llm_factory= to compile().",
        )
    cfg = _coerce_llm_config(llm_config)
    all_kwargs = {"node_name": node_name, "llm_config": cfg.as_factory_kwargs()}
    if runtime.llm_factory_params is _ACCEPT_ALL:
        kwargs = all_kwargs
    else:
        kwargs = {k: v for k, v in all_kwargs.items() if k in runtime.llm_factory_params}
    assert tier is not None and isinstance(tier, str)
    return runtime.llm_factory(tier, **kwargs)


def invoke_structured(
    *args: Any,
    model_tier: str | None = None,
    prompt_template: str | None = None,
    input_data: Any = None,
    output_model: Any = None,
    config: RunnableConfig | None = None,
    node_name: str = "",
    llm_config: LlmConfig | dict | None = None,
    context: dict[str, Any] | None = None,
    runtime: LlmRuntime | None = None,
) -> BaseModel:
    """Single LLM call with structured JSON output. Mode: produce.

    Output strategy (from llm_config.output_strategy):
        "structured" — llm.with_structured_output(model) (default, widest LangChain support)
        "json_mode"  — inject schema into prompt, LLM returns raw JSON, framework parses
        "text"       — LLM returns plain text, framework extracts and parses JSON from it

    Accepts both call shapes:
        invoke_structured(runtime, model_tier=..., ...)   # post-§2
        invoke_structured(model_tier=..., ...)            # legacy — uses compat runtime
    """
    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
    elif runtime is None:
        runtime = EMPTY_RUNTIME
    cfg = _coerce_llm_config(llm_config)
    llm, messages, strategy, llm_log = _prepare_structured_call(
        runtime, model_tier, prompt_template, input_data,
        output_model, config, node_name, cfg, context,
    )
    max_retries = cfg.max_retries
    t0 = time.monotonic()
    assert config is not None
    result, usage = _call_structured(llm, messages, output_model, strategy, config, max_retries=max_retries)
    return _finish_structured_call(result, usage, t0, llm_log, runtime, model_tier, node_name)


def _prepare_structured_call(
    runtime: LlmRuntime,
    model_tier: str | None,
    prompt_template: str | None,
    input_data: Any,
    output_model: Any,
    config: RunnableConfig | None,
    node_name: str,
    cfg: Any,
    context: dict[str, Any] | None,
) -> tuple[Any, list, str, Any]:
    """Pure preamble shared by invoke_structured and ainvoke_structured.

    No network I/O (``_get_llm`` is a factory call, ``_compile_prompt`` renders
    the prompt). Returns (llm, messages, strategy, llm_log) so the sync and async
    orchestrators diverge only at the awaited ``_call_structured`` seam.
    """
    strategy = cfg.output_strategy
    llm_log = log.bind(tier=model_tier, prompt=prompt_template, output=output_model.__name__, strategy=strategy)

    output_schema = None
    if strategy == "json_mode":
        output_schema = describe_type(output_model)

    llm = _get_llm(runtime, model_tier, node_name=node_name, llm_config=cfg)
    messages = _compile_prompt(
        runtime,
        prompt_template,
        input_data,
        node_name=node_name,
        config=config,
        output_model=output_model,
        llm_config=cfg,
        output_schema=output_schema,
        context=context,
    )
    return llm, messages, strategy, llm_log


def _finish_structured_call(
    result: Any,
    usage: Any,
    t0: float,
    llm_log: Any,
    runtime: LlmRuntime,
    model_tier: str | None,
    node_name: str,
) -> BaseModel:
    """Pure postamble shared by invoke_structured and ainvoke_structured."""
    elapsed = time.monotonic() - t0

    usage_info = {}
    if usage:
        usage_info = {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    llm_log.info("llm_call", mode="think", duration_s=round(elapsed, 3), **usage_info)
    _notify_cost(runtime, model_tier, usage, node_name=node_name, mode="think", duration_s=round(elapsed, 3))
    return result


async def ainvoke_structured(
    *args: Any,
    model_tier: str | None = None,
    prompt_template: str | None = None,
    input_data: Any = None,
    output_model: Any = None,
    config: RunnableConfig | None = None,
    node_name: str = "",
    llm_config: LlmConfig | dict | None = None,
    context: dict[str, Any] | None = None,
    runtime: LlmRuntime | None = None,
) -> BaseModel:
    """Async twin of :func:`invoke_structured` (Phase 1c).

    Shares the pure preamble (_prepare_structured_call) and postamble
    (_finish_structured_call) verbatim; the only divergence is awaiting the
    network seam ``_acall_structured``.
    """
    if args and isinstance(args[0], LlmRuntime):
        runtime = args[0]
    elif runtime is None:
        runtime = EMPTY_RUNTIME
    cfg = _coerce_llm_config(llm_config)
    llm, messages, strategy, llm_log = _prepare_structured_call(
        runtime, model_tier, prompt_template, input_data,
        output_model, config, node_name, cfg, context,
    )
    max_retries = cfg.max_retries
    t0 = time.monotonic()
    assert config is not None
    result, usage = await _acall_structured(llm, messages, output_model, strategy, config, max_retries=max_retries)
    return _finish_structured_call(result, usage, t0, llm_log, runtime, model_tier, node_name)
