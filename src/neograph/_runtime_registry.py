"""Decoration-time registry — leaf module owning the import-time shim stores.

``@node`` (inline body-merge / ``interrupt_when``), ``@merge_fn``, and ``@tool``
run at *import* time, before any ``compile()`` call exists, so their shims need
a landing spot keyed by name. That store used to be three module-level mutable
dicts in ``decorators.py`` (cleared between tests by a manual conftest block) —
which kept ``decorators.py`` (the DX layer) owning runtime registration state.

This leaf module owns that state instead, as a single ``Registry`` instance:

* ``register_scripted`` / ``register_condition`` / ``register_tool_factory``
  write into it at decoration time.
* ``compile()`` seeds its fresh per-compile lookups from this one.
* ``reset()`` (and ``session()`` on the instance) give tests isolation without
  a hand-maintained ``.clear()`` block reaching into ``decorators.py`` internals.

``decorators.py`` re-exports the three ``register_*`` helpers for back-compat,
but carries ZERO module-level mutable dicts of its own (neograph-v3xx HIGH-01).
"""

from __future__ import annotations

from collections.abc import Callable

from neograph._registry import Registry  # noqa: F401 — re-exported for internal use

# The single import-time registry for decorator-side shims. Populated as
# modules import (@node/@merge_fn/@tool decoration); read by compile().
_decoration_registry = Registry()


def register_scripted(name: str, fn: Callable) -> None:
    """Register a decoration-time scripted shim (inline body-merge / @merge_fn)."""
    _decoration_registry.scripted[name] = fn


def register_condition(name: str, fn: Callable) -> None:
    """Register a decoration-time condition shim (inline interrupt_when)."""
    _decoration_registry.condition[name] = fn


def register_tool_factory(name: str, fn: Callable) -> None:
    """Register a decoration-time tool factory (@tool)."""
    _decoration_registry.tool_factory[name] = fn


def reset() -> None:
    """Clear the decoration-time registry — used by test isolation fixtures."""
    _decoration_registry.reset()
