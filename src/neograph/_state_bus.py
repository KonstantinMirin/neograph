"""State-bus protocol — uniform read interface across BaseModel/dict state.

The compiled graph passes a Pydantic state model to most node wrappers, but
subgraph dispatch and isolated execution can pass a ``dict``. Helpers that
read from state used to take ``state: Any`` and branch internally; this
module resolves that polymorphism into a single Protocol with two concrete
implementations.

Use ``adapt_state(state)`` at the dispatch entry point and pass the resulting
``StateBus`` down to helpers. Helpers never see the raw union.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class StateBus(Protocol):
    """Read-only state accessor for node helpers.

    Mirrors ``dict.get`` semantics: ``get(key, default)`` returns the bound
    value (including ``None`` when explicitly stored) or ``default`` when the
    key is absent. ``keys()`` enumerates all bound field names in their
    declared/insertion order.
    """

    def get(self, key: str, default: Any = None) -> Any: ...

    def keys(self) -> list[str]: ...


class _DictStateBus:
    __slots__ = ("_state",)

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def keys(self) -> list[str]:
        return list(self._state.keys())


class _ModelStateBus:
    __slots__ = ("_state",)

    def __init__(self, state: BaseModel) -> None:
        self._state = state

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self._state, key, default)

    def keys(self) -> list[str]:
        return list(self._state.__class__.model_fields.keys())


def adapt_state(state: BaseModel | dict[str, Any]) -> StateBus:
    """Adapt a BaseModel or dict state into a uniform StateBus.

    Resolves state-bus polymorphism ONCE at the dispatch entry. Downstream
    helpers take ``state: StateBus`` and never branch on shape.
    """
    if isinstance(state, dict):
        return _DictStateBus(state)
    return _ModelStateBus(state)
