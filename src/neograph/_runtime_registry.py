"""Runtime registry accessors — leaf module shared by `factory.py` and `_oracle.py`.

These thin wrappers around the singleton ``Registry`` were historically defined
in ``factory.py``. Because ``factory.py`` imports from ``_oracle.py``,
``_oracle.py`` could not import them at module scope — it used a function-local
``from neograph.factory import lookup_scripted`` to defer the import past the
circular edge.

Pulling the wrappers down to this leaf module (which depends only on
``_registry`` and ``errors``) lets both ``factory.py`` and ``_oracle.py``
import at module scope. ``factory.py`` re-exports the names so the public
``from neograph.factory import register_scripted`` surface is unchanged.

Recipe template: ``_sidecar.py`` (extracted from ``decorators.py`` to break
the ``decorators <-> _construct_builder`` cycle).
"""

from __future__ import annotations

from collections.abc import Callable

from neograph._registry import registry
from neograph.errors import ConfigurationError


def register_scripted(name: str, fn: Callable) -> None:
    """Register a deterministic function for Node.scripted."""
    registry.scripted[name] = fn


def register_condition(name: str, fn: Callable) -> None:
    """Register a condition function for Operator(when=...)."""
    registry.condition[name] = fn


def register_tool_factory(name: str, fn: Callable) -> None:
    """Register a tool factory that creates LangChain @tool functions."""
    registry.tool_factory[name] = fn


def lookup_condition(name: str) -> Callable:
    """Look up a registered condition function by name. Raises ConfigurationError if missing."""
    fn = registry.condition.get(name)
    if fn is None:
        raise ConfigurationError.build(
            f"Condition '{name}' not registered",
            hint="Use register_condition() to register it before compilation",
        )
    return fn


def lookup_scripted(name: str) -> Callable:
    """Look up a registered scripted function by name. Raises ConfigurationError if missing."""
    fn = registry.scripted.get(name)
    if fn is None:
        raise ConfigurationError.build(
            f"Scripted function '{name}' not registered",
            hint="Use register_scripted() to register it before compilation",
        )
    return fn
