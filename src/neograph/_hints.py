"""Shared type-hint resolution — the ONE home for the ``get_type_hints()``
try/except that used to be copy-pasted across ``decorators.py``,
``_di_classify.py``, ``_sidecar.py``, and ``_validation_types.py``.

All six copies did ``except (NameError, AttributeError, TypeError): resolved =
{}`` around ``typing.get_type_hints()``. ``get_type_hints`` is ALL-OR-NOTHING —
a SINGLE unresolvable annotation makes the whole call raise — so those catches
discarded hint resolution for EVERY parameter of the function. One bad
forward-ref then mis-typed every OTHER parameter's ``Node.inputs`` / DI binding,
surfacing later as a confusing downstream mismatch instead of at
decoration/assembly time. PAT-02 / neograph-7ymj.

:func:`resolve_hints` fixes that by falling back to PER-ANNOTATION resolution
when the batch call fails: each annotation is resolved independently, the
resolvable ones survive, and only the genuinely-unresolvable ones are omitted
(with a ``debug`` breadcrumb naming them, so the drop is never fully silent).

Why per-annotation isolation and NOT fail-loud-raise: a blanket raise on any
unresolvable annotation breaks legitimate code. Under
``from __future__ import annotations`` every annotation is a string, and a
non-DI marker or a locally-scoped type (e.g. ``Annotated[str, CustomMarker()]``
with a test-local ``CustomMarker``) is unresolvable from this module's namespace
yet perfectly valid — the parameter is simply wired by name as an upstream. The
audit's actual defect is the all-or-nothing DISCARD, and per-annotation
isolation removes it without penalising valid decorations.

Layer: leaf helper. Imports only ``typing`` + ``structlog`` — no neograph
imports — so it is safe to import from every layer without an import cycle.
"""

from __future__ import annotations

import sys
import typing
from typing import Any

import structlog

log = structlog.get_logger()

# The exceptions ``typing.get_type_hints`` raises on an unresolvable annotation:
# NameError (undefined forward ref), plus AttributeError/TypeError for the
# malformed-annotation edge cases the six copies already guarded.
_HINT_ERRORS = (NameError, AttributeError, TypeError)


def resolve_hints(
    obj: object,
    *,
    localns: dict[str, Any] | None = None,
    include_extras: bool = False,
    owner: str | None = None,
) -> dict[str, Any]:
    """Resolve *obj*'s type hints, isolating unresolvable annotations.

    Returns ``{name: type}`` for every annotation that resolves. On the fast
    path this is exactly ``typing.get_type_hints``. When that raises (one bad
    annotation), each annotation is retried individually so the resolvable ones
    are NOT discarded along with the offender; the unresolvable ones are omitted
    and logged at ``debug`` (naming *owner* + the offending annotations).
    """
    try:
        return typing.get_type_hints(obj, localns=localns, include_extras=include_extras)
    except _HINT_ERRORS as exc:
        resolved, unresolved = _resolve_per_annotation(obj, localns, include_extras)
        if unresolved:
            log.debug(
                "type_hints_partially_unresolved",
                obj=owner or getattr(obj, "__qualname__", None) or repr(obj),
                unresolved=sorted(unresolved),
                error=str(exc),
            )
        return resolved


def _resolve_per_annotation(
    obj: object,
    localns: dict[str, Any] | None,
    include_extras: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve each of *obj*'s annotations independently.

    Returns ``(resolved, unresolved_names)``. ``get_type_hints`` is
    all-or-nothing; this walks ``__annotations__`` so a single bad forward-ref
    no longer poisons its siblings.
    """
    raw = getattr(obj, "__annotations__", None) or {}
    globalns = getattr(obj, "__globals__", None)
    if globalns is None:
        module = getattr(obj, "__module__", None)
        globalns = getattr(sys.modules.get(module), "__dict__", {}) if module else {}
    resolved: dict[str, Any] = {}
    unresolved: list[str] = []
    for field, ann in raw.items():
        try:
            target = typing.ForwardRef(ann) if isinstance(ann, str) else ann
            value = typing._eval_type(target, globalns, localns)  # type: ignore[attr-defined]
        except _HINT_ERRORS:
            # One bad annotation must not sink the rest — record + skip it.
            unresolved.append(field)
            continue
        if not include_extras:
            value = _strip_extras(value)
        resolved[field] = value
    return resolved, unresolved


def _strip_extras(value: Any) -> Any:
    """Unwrap ``Annotated[T, ...]`` to ``T`` (mirrors ``include_extras=False``)."""
    if typing.get_origin(value) is typing.Annotated:
        return typing.get_args(value)[0]
    return value
