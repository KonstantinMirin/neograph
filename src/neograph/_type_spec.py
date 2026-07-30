"""``TypeSpec`` — the validated shape of a user-declared type value.

Extracted from ``node.py`` (neograph-3ffdg.18) as a pure file split — the
validator, the predicate and the two aliases below are unchanged, only their
home moved. ``node.py`` re-exports them, so existing imports keep resolving.

Python has no TypeForm (PEP 747), so the static annotation is ``Any`` and
``PlainValidator(_validate_type_spec)`` is the real enforcement point.
"""

from __future__ import annotations

import types as _types_mod
from typing import Annotated, Any

from pydantic import PlainValidator


def _validate_type_spec(v: Any) -> Any:
    """Accept type objects, generic aliases, and dict-form type specs.

    Rejects ints and other non-type garbage that would silently pass through
    to compile() and produce confusing errors downstream.

    Valid forms: None, concrete type, generic alias (list[X], dict[str,X],
    Optional[X], X|None), dict[str, type|str|GenericAlias].  Dict values may
    be strings (loader path uses type names before resolution, and the decorator
    fallback path produces string annotations when get_type_hints fails).
    """
    if v is None:
        return None
    if isinstance(v, dict):
        for key, val in v.items():
            if not isinstance(key, str):
                raise TypeError(f"dict-form type spec keys must be strings, got {type(key).__name__}")
            if not (isinstance(val, (type, str)) or _is_type_like(val)):
                raise TypeError(
                    f"dict-form type spec value for '{key}' must be a type or type name, got {type(val).__name__}: {val!r}"
                )
        return v
    if isinstance(v, type) or _is_type_like(v):
        return v
    raise TypeError(f"inputs/outputs must be a type, dict[str, type], or None — got {type(v).__name__}: {v!r}")


def _is_type_like(v: Any) -> bool:
    """Check if v is a generic alias (list[X], dict[str, X], Optional[X], X | None)."""
    import types as _types
    import typing

    return (
        hasattr(v, "__origin__")
        or isinstance(v, (typing._GenericAlias, typing._SpecialForm))  # type: ignore[attr-defined]
        or isinstance(v, _types.UnionType)
    )


# Valid forms: None | type | GenericAlias (list[X], dict[str,X], X|None) |
# dict[str, type|str|GenericAlias].  Static annotation is Any because Python
# has no TypeForm (PEP 747). PlainValidator is the real enforcement point.
TypeSpec = Annotated[Any, PlainValidator(_validate_type_spec)]

# Static-annotation alias for user-declared type values flowing through the
# framework. Distinct from the Pydantic-validator-bearing TypeSpec field type,
# which carries _validate_type_spec on top of the same union. Use this on
# parameter and return annotations of helpers that introspect user-declared
# types (closes the PEP 747 gap until TypeForm lands). See
# docs/design/architecture-decisions.md §5 and §8.
TypeSpecStatic = type | dict[str, type] | _types_mod.GenericAlias | _types_mod.UnionType | None
