"""JSON extraction from a model response that wrapped it in prose or fences.

Extracted from ``_llm_retry.py`` (neograph-3ffdg.15) as a pure file split — the
two functions below are unchanged, only their home moved. ``_llm_retry.py``
re-exports them, so existing imports keep resolving.

``_extract_balanced`` scans for a balanced brace/bracket run rather than
regex-matching, so a JSON object containing braces inside strings survives.
"""

from __future__ import annotations


def _extract_balanced(text: str, start: int, open_ch: str, close_ch: str) -> str | None:
    """Extract a balanced JSON value starting at *start* with given delimiters."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(text: str) -> str:
    """Extract the first balanced JSON value (object or array) from LLM text.

    Finds the first '{' or '[' and tracks depth to find the matching closer.
    Handles markdown fences, thinking tags, prose before/after JSON.
    When both '{' and '[' are present, picks whichever comes first.
    """
    brace_pos = text.find("{")
    bracket_pos = text.find("[")

    if bracket_pos != -1 and (brace_pos == -1 or bracket_pos < brace_pos):
        result = _extract_balanced(text, bracket_pos, "[", "]")
        if result is not None:
            return result
        # Array was truncated (no closing ]). Do NOT fall through to object
        # extraction — that would return the first inner dict, causing silent
        # empty results for list-field target models. Return the raw text and
        # let json_repair handle truncation recovery.
        return text.strip()

    pos = 0
    while True:
        start = text.find("{", pos)
        if start == -1:
            break

        # Quick check: JSON objects start with {"  or {digit or {[  or { "
        # Prose braces like {a + b} start with a letter after {.
        after_brace = text[start + 1 : start + 2].lstrip()
        if after_brace and after_brace not in ('"', "'", "[", "]", "{", "}", ""):
            pos = start + 1
            continue

        result = _extract_balanced(text, start, "{", "}")
        if result is not None:
            return result

        pos = start + 1

    first_open = brace_pos if brace_pos != -1 else bracket_pos
    if first_open != -1:
        close_char = "}" if first_open == brace_pos else "]"
        last = text.rfind(close_char)
        if last > first_open:
            return text[first_open : last + 1]
    return text.strip()
