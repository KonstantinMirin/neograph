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

from neograph._llm_retry import _invoke_json_with_retry
from neograph.errors import ExecutionError


def _call_structured(
    llm: Any,
    messages: list,
    output_model: type[BaseModel],
    strategy: str,
    config: RunnableConfig,
    max_retries: int = 1,
) -> tuple[BaseModel, Any]:
    """Dispatch structured output by strategy. Returns (result, usage_metadata)."""
    if strategy == "structured":
        usage = None
        try:
            structured_llm = llm.with_structured_output(output_model, include_raw=True)
            raw_result = structured_llm.invoke(messages, config=config)
            if isinstance(raw_result, dict) and "parsed" in raw_result:
                result = raw_result["parsed"]
                raw_msg = raw_result.get("raw")
                usage = getattr(raw_msg, "usage_metadata", None) if raw_msg else None
            else:
                result = raw_result
        except TypeError:
            structured_llm = llm.with_structured_output(output_model)
            result = structured_llm.invoke(messages, config=config)
        return result, usage

    if strategy in ("json_mode", "text"):
        return _invoke_json_with_retry(llm, messages, output_model, config, max_retries=max_retries)

    raise ExecutionError.build(
        "Unknown output_strategy",
        expected="'structured', 'json_mode', or 'text'",
        found=repr(strategy),
        hint="Set output_strategy in llm_config to one of the supported values.",
    )
