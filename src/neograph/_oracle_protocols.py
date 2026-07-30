"""Typing Protocols for the Oracle merge hooks.

Extracted from ``modifiers.py`` (neograph-3ffdg.5) as a pure file split — the
Protocols below are unchanged, only their home moved. ``modifiers.py``
re-exports them.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel
from typing_extensions import TypeVar

_Variant = TypeVar("_Variant", default=Any)
_FallbackResult = TypeVar("_FallbackResult", covariant=True, default=Any)
_PostResult = TypeVar("_PostResult", default=Any)


@runtime_checkable
class MergePreProcess(Protocol[_Variant]):
    """Replaces the default ``{variants: ..., **upstream}`` input_data
    construction for the ``merge_prompt`` path. Returns the data passed
    verbatim to ``invoke_structured`` -- which accepts ``BaseModel | dict | str``.
    """

    def __call__(self, variants: list[_Variant]) -> BaseModel | dict[str, Any] | str: ...


@runtime_checkable
class MergePostProcess(Protocol[_PostResult, _Variant]):
    """Transforms the parsed LLM merge result before it is written to state."""

    def __call__(self, result: _PostResult, variants: list[_Variant]) -> _PostResult: ...


@runtime_checkable
class MergeFallback(Protocol[_Variant, _FallbackResult]):
    """Catches errors from ``invoke_structured`` during merge. Returns a
    deterministic fallback result instead of propagating the exception.
    """

    def __call__(self, variants: list[_Variant], error: Exception) -> _FallbackResult: ...
