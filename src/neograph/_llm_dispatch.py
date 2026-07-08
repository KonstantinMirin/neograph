"""Output-strategy dispatch — pick the structured-output mechanism per request.

Single-responsibility: given a strategy name (`structured`, `json_mode`,
`text`), invoke the LLM with the right adapter and return (parsed_model,
usage). Adding a new output_strategy means editing exactly this file (and,
when it surfaces as a config option, the `Literal` in `_llm_config.py`).

Provider quirks of the `structured` path (include_raw rejection, parsed=None,
DSML markup) are NOT handled here. They are normalized by the compat shim
(`_llm_structured_compat`) into a `StructuredResult` tagged union; this module
just switches on the variant. Adding a new quirk is a new decorator there, not
a new branch here.
"""

from __future__ import annotations

from typing import Any, NoReturn

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

from neograph._dsml import contains_dsml, message_text
from neograph._llm_retry import (
    _ainvoke_json_with_retry,
    _invoke_json_with_retry,
    arecover_dsml,
    recover_dsml,
    structured_retry_messages,
)
from neograph._llm_structured_compat import (
    Failed,
    Parsed,
    Raw,
    build_default_adapter,
)
from neograph.errors import ExecutionError


def _raise_decoded_none(output_model: type[BaseModel], raw_text: str | None) -> NoReturn:
    """Fail loud on the structured path's parsed=None silent variant (undecodable
    response), not surface None which the write boundary swallows. Sync+async."""
    raise ExecutionError.build(
        "structured output decoded to None",
        expected=f"valid {output_model.__name__}",
        found=f"unparseable response (first 200 chars): {(raw_text or '')[:200]!r}",
        hint="Response undecodable into the output model with no tool-call markup to recover — usually a transient decode flake.",
    )


# Single-site fail-loud helpers shared by the sync/async dispatch twins so a
# message edit lands once; the twins differ only at the await seam. NoReturn
# (like _raise_decoded_none) so the ExecutionError content lives in one place.
def _raise_markup_unrecoverable(output_model: type[BaseModel], raw_text: str) -> NoReturn:
    raise ExecutionError.build(
        "structured output contained unrecoverable tool-call markup",
        expected=f"valid {output_model.__name__}",
        found=f"DSML markup (first 200 chars): {raw_text[:200]!r}",
        hint="Model emitted tool-call markup instead of structured output.",
    )


def _raise_dispatch_failed(output_model: type[BaseModel], err: Any) -> NoReturn:
    raise ExecutionError.build(
        "structured output dispatch failed",
        expected=f"valid {output_model.__name__}",
        found=repr(err),
        hint="Provider rejected include_raw=True and the compat retry also failed.",
    ) from err


def _raise_structured_retry_exhausted(
    output_model: type[BaseModel],
    err: ValidationError,
    attempts: int,
) -> NoReturn:
    """The structured re-prompt loop exhausted ``max_retries`` on a validation
    failure. Single-site so the sync/async twins share the message. neograph-zcxd."""
    raise ExecutionError.build(
        f"structured output failed validation after {attempts} re-prompt(s)",
        expected=f"valid {output_model.__name__}",
        found=repr(err),
        hint="The provider's constrained decode kept emitting output that fails the "
        "output model's validation; raise max_retries or switch this node to "
        "output_strategy='json_mode'.",
    ) from err


def _raise_unknown_strategy(strategy: str) -> NoReturn:
    raise ExecutionError.build(
        "Unknown output_strategy",
        expected="'structured', 'json_mode', or 'text'",
        found=repr(strategy),
        hint="Set output_strategy in llm_config to one of the supported values.",
    )


def _raise_unexpected_variant(result: Any) -> NoReturn:
    raise ExecutionError.build(
        "unexpected structured-output result variant",
        expected="Parsed | Raw | Failed",
        found=repr(result),
        hint="A new StructuredResult variant needs a matching case in _call_structured.",
    )


def _call_structured(
    llm: Any,
    messages: list,
    output_model: type[BaseModel],
    strategy: str,
    config: RunnableConfig,
    *,
    cfg: Any = None,
    max_retries: int = 1,
) -> tuple[BaseModel, Any]:
    """Dispatch structured output by strategy. Returns (result, usage_metadata).

    For the ``structured`` strategy, the compat shim classifies the provider's
    response into ``Parsed | Raw | Failed`` and this function switches on it:

    - ``Parsed`` — return the model and its usage.
    - ``Raw(dsml=True)`` — content-level DSML recovery via ``recover_dsml`` (a
      retry concern); usage from the original response is preserved.
    - ``Raw`` (no DSML, the ``parsed=None`` silent variant) — preserve the
      legacy passthrough: return ``(None, usage)`` without re-prompting.
    - ``Failed(ValidationError)`` — the provider's constrained decode emitted
      structurally-valid JSON that failed the output model's validation (a
      weakly-enforced decode). Re-prompt the SAME structured adapter with the
      validation error as a repair hint, bounded by ``max_retries`` (parity with
      json_mode); exhaustion raises. Constrained decode stays engaged on the
      retries.
    - ``Failed`` (other) — ``include_raw`` was rejected and the compat retry also
      failed. As a last resort fetch the provider's real output here (the
      orchestration boundary owns this IO, not the compat decorators) and, if
      it is DSML, recover; otherwise surface the dispatch error.
    """
    if strategy == "structured":
        adapter = build_default_adapter()
        current_messages = messages
        attempts = 0
        while True:
            result = adapter.invoke(llm, output_model, current_messages, config)
            match result:
                case Parsed(model=parsed, usage=usage):
                    return parsed, usage
                case Raw(dsml=True, raw_text=raw_text, usage=usage):
                    if cfg is not None:
                        recovered = recover_dsml(
                            raw_text,
                            output_model,
                            llm,
                            messages,
                            config,
                            cfg,
                            strategy="structured",
                        )
                        if recovered is not None:
                            return recovered, usage
                    _raise_markup_unrecoverable(output_model, raw_text)
                case Raw(raw_text=raw_text, usage=_usage):
                    _raise_decoded_none(output_model, raw_text)
                case Failed(error=err, raw_text=raw_text) if isinstance(err, ValidationError):
                    if attempts >= max_retries:
                        _raise_structured_retry_exhausted(output_model, err, attempts)
                    attempts += 1
                    current_messages = structured_retry_messages(
                        current_messages,
                        raw_text,
                        err,
                        output_model,
                    )
                    continue
                case Failed(error=err, raw_text=raw_text):
                    if cfg is not None:
                        text = raw_text or message_text(llm.invoke(messages, config=config))
                        if contains_dsml(text):
                            recovered = recover_dsml(
                                text,
                                output_model,
                                llm,
                                messages,
                                config,
                                cfg,
                                strategy="structured",
                            )
                            if recovered is not None:
                                return recovered, None
                    _raise_dispatch_failed(output_model, err)
                case _:  # defensive: a future StructuredResult variant added without a dispatch arm
                    _raise_unexpected_variant(result)

    if strategy in ("json_mode", "text"):
        return _invoke_json_with_retry(llm, messages, output_model, config, max_retries=max_retries)

    _raise_unknown_strategy(strategy)


async def _acall_structured(
    llm: Any,
    messages: list,
    output_model: type[BaseModel],
    strategy: str,
    config: RunnableConfig,
    *,
    cfg: Any = None,
    max_retries: int = 1,
) -> tuple[BaseModel, Any]:
    """Async twin of :func:`_call_structured`.

    Identical strategy dispatch and Parsed/Raw/Failed match arms (same
    ExecutionError.build messages); the only divergence is awaiting the network
    seams — the adapter's ``.ainvoke``, ``arecover_dsml``, ``llm.ainvoke``, and
    ``_ainvoke_json_with_retry``. config threads into every awaited hop (M6a).
    """
    if strategy == "structured":
        adapter = build_default_adapter()
        current_messages = messages
        attempts = 0
        while True:
            result = await adapter.ainvoke(llm, output_model, current_messages, config)
            match result:
                case Parsed(model=parsed, usage=usage):
                    return parsed, usage
                case Raw(dsml=True, raw_text=raw_text, usage=usage):
                    if cfg is not None:
                        recovered = await arecover_dsml(
                            raw_text,
                            output_model,
                            llm,
                            messages,
                            config,
                            cfg,
                            strategy="structured",
                        )
                        if recovered is not None:
                            return recovered, usage
                    _raise_markup_unrecoverable(output_model, raw_text)
                case Raw(raw_text=raw_text, usage=_usage):
                    _raise_decoded_none(output_model, raw_text)  # async twin
                case Failed(error=err, raw_text=raw_text) if isinstance(err, ValidationError):
                    if attempts >= max_retries:
                        _raise_structured_retry_exhausted(output_model, err, attempts)
                    attempts += 1
                    current_messages = structured_retry_messages(
                        current_messages,
                        raw_text,
                        err,
                        output_model,
                    )
                    continue
                case Failed(error=err, raw_text=raw_text):
                    if cfg is not None:
                        text = raw_text or message_text(await llm.ainvoke(messages, config=config))
                        if contains_dsml(text):
                            recovered = await arecover_dsml(
                                text,
                                output_model,
                                llm,
                                messages,
                                config,
                                cfg,
                                strategy="structured",
                            )
                            if recovered is not None:
                                return recovered, None
                    _raise_dispatch_failed(output_model, err)
                case _:
                    _raise_unexpected_variant(result)

    if strategy in ("json_mode", "text"):
        return await _ainvoke_json_with_retry(llm, messages, output_model, config, max_retries=max_retries)

    _raise_unknown_strategy(strategy)
