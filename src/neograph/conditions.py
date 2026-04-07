"""Safe condition expression evaluator for spec-driven pipelines.

Parses simple expressions like ``'score < 0.8'`` into callables that
evaluate against Pydantic models or dicts.  No ``eval``/``exec`` — only
a whitelisted grammar of ``field op literal``.
"""

from __future__ import annotations

import operator
import re
from typing import Any, Callable

# ---- grammar ----------------------------------------------------------

_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

# dotted_field  op  literal
# literal: float, int, bool (true/false), quoted string ("..." or '...')
_EXPR_RE = re.compile(
    r"^"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_.]*)"  # dotted field
    r"\s+"
    r"(?P<op>[<>!=]=?)"  # operator
    r"\s+"
    r"(?P<literal>.+)"  # literal (parsed below)
    r"$"
)


def _parse_literal(raw: str) -> int | float | bool | str:
    """Convert a literal token to a Python value."""
    # booleans
    if raw == "true":
        return True
    if raw == "false":
        return False

    # quoted string
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]

    # numeric — try int first, then float
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass

    raise ValueError(
        f"Cannot parse literal: {raw!r}. "
        "Expected a number, boolean (true/false), or quoted string."
    )


def _resolve_field(obj: Any, dotted: str) -> Any:
    """Walk a dotted path on a Pydantic model or dict."""
    current = obj
    for part in dotted.split("."):
        if isinstance(current, dict):
            try:
                current = current[part]
            except KeyError:
                raise AttributeError(
                    f"Field {part!r} not found in dict "
                    f"while resolving {dotted!r}"
                ) from None
        else:
            try:
                current = getattr(current, part)
            except AttributeError:
                raise AttributeError(
                    f"Field {part!r} not found on {type(current).__name__} "
                    f"while resolving {dotted!r}"
                ) from None
    return current


# ---- public API -------------------------------------------------------


def parse_condition(expr: str) -> Callable[[Any], bool]:
    """Parse a condition expression and return a callable evaluator.

    Supports: ``field op literal`` where *op* is one of
    ``< > <= >= == !=`` and *literal* is a number, boolean
    (``true``/``false``), or a quoted string (``"..."``).

    Dotted field access is supported: ``result.score < 0.8``.

    Raises :class:`ValueError` for any expression that does not match
    the grammar.
    """
    expr = expr.strip()
    m = _EXPR_RE.match(expr)
    if m is None:
        raise ValueError(
            f"Invalid condition expression: {expr!r}. "
            "Expected: field op literal  "
            "(e.g. 'score < 0.8', 'passed == true', 'name != \"draft\"')"
        )

    field = m.group("field")
    op_str = m.group("op")
    literal_raw = m.group("literal").strip()

    if op_str not in _OPS:
        raise ValueError(
            f"Unsupported operator {op_str!r} in expression {expr!r}. "
            f"Allowed: {', '.join(sorted(_OPS))}"
        )

    op_fn = _OPS[op_str]
    literal = _parse_literal(literal_raw)

    def _evaluate(value: Any) -> bool:
        resolved = _resolve_field(value, field)
        return op_fn(resolved, literal)

    return _evaluate
