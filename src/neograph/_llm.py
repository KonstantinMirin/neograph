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

# Provider-native JSON-object mode payload. Bound onto the model for the
# ``json_mode`` strategy so the provider constrains decoding server-side; the
# schema-in-prompt + parse path (``_llm_retry``) stays the universal fallback.
_JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}


def _is_response_format_rejection(exc: BaseException) -> bool:
    """True when a provider rejected the native ``response_format`` kwarg.

    A provider without JSON-object mode either raises ``TypeError`` (the kwarg is
    an unexpected keyword) or a provider ``BadRequest`` (the kwarg reached an API
    that does not accept it). We identify the rejection by the kwarg name in the
    message — NEVER by provider class name (that drifts). Any other error is a
    genuine failure and must propagate untouched.
    """
    if "response_format" not in str(exc).lower():
        return False
    if isinstance(exc, TypeError):
        return True
    return "badrequest" in type(exc).__name__.lower()


def _ensure_json_instruction(messages: list) -> list:
    """Guarantee the literal word 'json' appears somewhere in the messages.

    OpenAI-compatible ``response_format={'type':'json_object'}`` silently 400s
    unless a message mentions 'json'. The ``describe_type`` schema block is
    TypeScript-style and does not guarantee the literal word, and the prompt is
    app-``prompt_compiler``-built, so the framework appends one instruction line
    when the word is absent. Returns a new list; never mutates the compiled prompt.
    """
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if isinstance(content, str) and "json" in content.lower():
            return messages
    return [*messages, {"role": "user", "content": "Respond with a single valid JSON object and nothing else."}]


class _NativeJsonModeLLM:
    """Attempt-bind-and-fall-back wrapper for json_mode native ``response_format``.

    Sends the provider-native ``response_format={'type':'json_object'}`` on the
    model call. If the provider rejects the kwarg (a provider without JSON-object
    support, e.g. ``ChatAnthropic`` forwarding to an API that 400s), it logs ONCE
    (``json_mode_native_unsupported``) and falls back to the UNBOUND client for
    this and every subsequent call. The schema-in-prompt + ``_parse_json_response``
    machinery stays the universal safety net either way — mirroring the
    ``_llm_structured_compat`` philosophy (a quirk is handled at the boundary, not
    branched into the dispatch).

    Bound ONCE at the shared ``_prepare_structured_call`` seam so BOTH the sync
    (``invoke`` -> ``_call_structured`` -> ``_invoke_json_with_retry``) and async
    (``ainvoke`` -> ``_acall_structured`` -> ``_ainvoke_json_with_retry``) twins
    inherit the SAME wrapper and reuse it across error-feedback retries. Each of
    the two entrypoints does attempt-bind-and-fall-back independently; the
    ``_fell_back`` flag is shared, so the first rejection (on either surface)
    switches every subsequent call to the unbound client.
    """

    def __init__(self, llm: Any) -> None:
        self._unbound = llm
        self._bound = llm.bind(response_format=_JSON_OBJECT_RESPONSE_FORMAT)
        self._fell_back = False

    def _record_fallback(self, exc: BaseException) -> None:
        log.warning(
            "json_mode_native_unsupported",
            provider=type(self._unbound).__name__,
            error=str(exc),
        )
        self._fell_back = True

    def invoke(self, messages: list, **kwargs: Any) -> Any:
        if self._fell_back:
            return self._unbound.invoke(messages, **kwargs)
        try:
            return self._bound.invoke(messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 — re-raised unless it is the known rejection
            if not _is_response_format_rejection(exc):
                raise
            self._record_fallback(exc)
            return self._unbound.invoke(messages, **kwargs)

    async def ainvoke(self, messages: list, **kwargs: Any) -> Any:
        if self._fell_back:
            return await self._unbound.ainvoke(messages, **kwargs)
        try:
            return await self._bound.ainvoke(messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 — re-raised unless it is the known rejection
            if not _is_response_format_rejection(exc):
                raise
            self._record_fallback(exc)
            return await self._unbound.ainvoke(messages, **kwargs)


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
        "json_mode"  — send the provider-native response_format={'type':'json_object'}
                       (attempt-bind-and-fall-back: providers without it fall back
                       unbound) AND inject the schema into the prompt; the LLM
                       returns raw JSON that the framework parses. Native json mode
                       constrains the emitted TEXT to a JSON object; it does NOT
                       suppress a reasoning model's internal reasoning
                       (gemini-2.5-pro / deepseek-reasoner still think first), which
                       is exactly why the schema-in-prompt + reason-then-coerce
                       raw-text parse remains the universal safety net.
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
        runtime,
        model_tier,
        prompt_template,
        input_data,
        output_model,
        config,
        node_name,
        cfg,
        context,
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

    if strategy == "json_mode":
        # ONE site, both twins inherit: this preamble is shared by
        # invoke_structured (sync) and ainvoke_structured (async). Guarantee the
        # json-word (OpenAI-compat silent-400 trap), then bind the provider-native
        # response_format. The wrapper flows into _call_structured /
        # _acall_structured -> _(a)invoke_json_with_retry, so the initial call AND
        # its error-feedback retries reuse the bound client (one bind). An LLM
        # double without .bind can't do native json mode -> stay on the
        # schema-in-prompt fallback unchanged.
        messages = _ensure_json_instruction(messages)
        if hasattr(llm, "bind"):
            llm = _NativeJsonModeLLM(llm)

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
        runtime,
        model_tier,
        prompt_template,
        input_data,
        output_model,
        config,
        node_name,
        cfg,
        context,
    )
    max_retries = cfg.max_retries
    t0 = time.monotonic()
    assert config is not None
    result, usage = await _acall_structured(llm, messages, output_model, strategy, config, max_retries=max_retries)
    return _finish_structured_call(result, usage, t0, llm_log, runtime, model_tier, node_name)
