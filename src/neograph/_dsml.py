"""DSML/XML tool-call markup detection — the single source of truth.

Some providers (notably DeepSeek R1 after budget exhaustion) emit XML-style
tool-call markup instead of the requested structured JSON. Detecting that
markup is a strategy-orthogonal concern: it can surface in the `structured`
compat path, in the `json_mode`/`text` retry path, and in the ReAct tool
loop's final response. This module owns the detection regex and the trivial
text-extraction helper so every site shares ONE definition.

Pure leaf: no `neograph._llm*` imports. Recovery (re-prompting the LLM and
re-parsing) is a *retry* concern and lives in `_llm_retry.recover_dsml`;
this module only answers "does this text contain DSML markup?".
"""

from __future__ import annotations

import re
from typing import Any

# Single source of truth for DSML/XML tool-call markup detection. Used by
# _llm_retry (parse + recovery), _llm_structured_compat (classification), and
# _tool_loop (json_mode/text final response). See neograph-0tid / neograph-ble3.
DSML_PATTERN = re.compile(
    r"<[^>]*(?:function_call|invoke|DSML)[^>]*>", re.IGNORECASE,
)


def contains_dsml(text: str) -> bool:
    """True if *text* contains DSML/XML tool-call markup."""
    return bool(DSML_PATTERN.search(text))


def message_text(msg: Any) -> str:
    """Extract text from an LLM message-like object.

    Returns ``msg.content`` when present, else ``str(msg)``. Centralizes the
    ``.content if hasattr else str`` idiom that was copy-pasted across the
    dispatch, retry, and tool-loop layers.
    """
    if msg is None:
        return ""
    return msg.content if hasattr(msg, "content") else str(msg)
