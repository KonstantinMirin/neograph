"""Structural Protocols for node-facing callables.

Extracted from ``node.py`` (neograph-3ffdg.18) as a pure file split — the
Protocols below are unchanged, only their home moved. ``node.py`` re-exports
them, so existing imports keep resolving.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from typing_extensions import TypeVar

_SkipIn = TypeVar("_SkipIn", contravariant=True, default=Any)
_SkipOut = TypeVar("_SkipOut", covariant=True, default=Any)


@runtime_checkable
class SkipPredicate(Protocol[_SkipIn]):
    """Returns True to bypass the LLM call. Receives extracted ``input_data``
    (after ``_extract_input``, before renderer dispatch).
    """

    def __call__(self, input_data: _SkipIn) -> bool: ...


@runtime_checkable
class SkipValueFactory(Protocol[_SkipIn, _SkipOut]):
    """Produces the output value when ``skip_when`` fires. Receives the same
    ``input_data`` shape as ``skip_when``. If absent, the node returns an
    empty state update.
    """

    def __call__(self, input_data: _SkipIn) -> _SkipOut: ...


@runtime_checkable
class RawNodeFn(Protocol):
    """Raw escape hatch for ``mode='raw'``. Direct LangGraph node signature."""

    def __call__(self, state: BaseModel, config: RunnableConfig) -> dict[str, Any]: ...


@runtime_checkable
class HasName(Protocol):
    """Anything that carries a user-facing declaration name.

    Both ``Node`` and ``Construct`` satisfy this structurally (each declares
    ``name: str``). Redirect-closure factories (`_oracle.py`) capture a
    ``HasName`` and read ``.name`` for error/observability labels — the label
    concern is sourced from the IR object (Information Expert), never threaded
    as a string kwarg nor scraped from a wrapper's ``__name__``.
    """

    name: str
