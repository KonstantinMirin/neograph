"""Usage-metadata dict assembly — the single site for the token-usage shape.

Pure leaf (no neograph imports). The rule "sum input/output, emit
``{input_tokens, output_tokens, total_tokens}`` when either is non-zero, else
the empty sentinel" recurred across the LLM/tool vertical (``_llm_retry`` ×4,
``_tool_loop``) and drifted subtly (``or 0`` vs ``.get`` defaults, ``None`` vs
``{}`` when empty). Consolidating it here makes the shape single-site so a
change to the usage contract is applied once. See neograph-ykun (DRY-06).
"""

from __future__ import annotations

from typing import Any


def _usage_dict(input_tokens: int, output_tokens: int, *, empty: Any = None) -> Any:
    """Build the token-usage dict, or ``empty`` when both counts are zero.

    ``total_tokens`` is always ``input_tokens + output_tokens`` (the sum rule).
    Callers pass ``empty={}`` for the tool-loop postamble, ``empty=usage`` to
    preserve a provider-reported dict on the in-loop path, or the default
    ``None`` when no usage should be reported.
    """
    if input_tokens or output_tokens:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    return empty
