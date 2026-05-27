"""Output-strategy dispatch — pick the structured-output mechanism per request.

Single-responsibility: given a strategy name (`structured`, `json_mode`,
`text`), invoke the LLM with the right adapter and return (parsed_model,
usage). Adding a new output_strategy means editing exactly this file (and,
when it surfaces as a config option, the `Literal` in `_llm_config.py`).
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from neograph._llm_retry import (
    _DSML_PATTERN,
    _attempt_dsml_recovery,
    _invoke_json_with_retry,
)
from neograph.errors import ExecutionError


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

    For the ``structured`` strategy, two DSML recovery paths fire when ``cfg``
    is provided (see neograph-0tid):

    1. **TypeError variant** — ``with_structured_output(..., include_raw=True)``
       raises TypeError (often LangChain-compat: provider rejected
       ``include_raw=True``). The compat retry (without ``include_raw``) is
       attempted; if it also raises TypeError, raw ``llm.invoke`` is consulted
       and the helper inspects the text for DSML markup.
    2. **Silent variant** — provider returns ``{"parsed": None, "raw": <AIMessage
       with DSML>, "parsing_error": ...}``. Detection is content-based; the
       helper re-parses via a targeted retry.
    """
    if strategy == "structured":
        usage = None
        try:
            structured_llm = llm.with_structured_output(output_model, include_raw=True)
            raw_result = structured_llm.invoke(messages, config=config)
            if isinstance(raw_result, dict) and "parsed" in raw_result:
                result = raw_result["parsed"]
                raw_msg = raw_result.get("raw")
                usage = getattr(raw_msg, "usage_metadata", None) if raw_msg else None

                # Silent variant: parsed=None + raw contains DSML markup.
                # See neograph-0tid.
                if result is None and raw_msg is not None and cfg is not None:
                    raw_text = (
                        raw_msg.content if hasattr(raw_msg, "content")
                        else str(raw_msg)
                    )
                    if _DSML_PATTERN.search(raw_text):
                        recovered = _attempt_dsml_recovery(
                            raw_text, output_model, llm, messages, config, cfg,
                            strategy="structured",
                        )
                        if recovered is not None:
                            result = recovered
            else:
                result = raw_result
        except TypeError as initial_exc:
            # LangChain compat: provider rejected include_raw=True. Retry without it.
            structured_llm = llm.with_structured_output(output_model)
            try:
                result = structured_llm.invoke(messages, config=config)
            except TypeError as exc:
                # Compat path also threw TypeError — likely DSML markup in the
                # response. Peek the last message in the conversation (from the
                # ReAct loop's final iteration) for DSML markers; if absent,
                # invoke the raw llm directly. See neograph-0tid.
                if cfg is None:
                    raise
                last_msg = messages[-1] if messages else None
                raw_text = ""
                if last_msg is not None:
                    raw_text = (
                        last_msg.content if hasattr(last_msg, "content")
                        else str(last_msg)
                    )
                if not _DSML_PATTERN.search(raw_text):
                    # No DSML in the existing trailing message — fall back to
                    # a fresh raw invoke to surface the provider's actual output.
                    raw_response = llm.invoke(messages, config=config)
                    raw_text = (
                        raw_response.content if hasattr(raw_response, "content")
                        else str(raw_response)
                    )
                recovered = _attempt_dsml_recovery(
                    raw_text, output_model, llm, messages, config, cfg,
                    strategy="structured",
                )
                if recovered is not None:
                    result = recovered
                else:
                    raise exc from initial_exc
        return result, usage

    if strategy in ("json_mode", "text"):
        return _invoke_json_with_retry(llm, messages, output_model, config, max_retries=max_retries)

    raise ExecutionError.build(
        "Unknown output_strategy",
        expected="'structured', 'json_mode', or 'text'",
        found=repr(strategy),
        hint="Set output_strategy in llm_config to one of the supported values.",
    )
