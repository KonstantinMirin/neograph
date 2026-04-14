"""Centralized registry for scripted functions, conditions, and tool factories.

Replaces the three module-level dicts that previously lived in factory.py.
Provides ``session()`` context manager for test isolation.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any


class Registry:
    """Thread-aware registry for runtime-registered callables.

    All three registries (scripted, condition, tool_factory) live here
    instead of as module-level dicts in factory.py.
    """

    def __init__(self) -> None:
        self.scripted: dict[str, Callable[..., Any]] = {}
        self.condition: dict[str, Callable[..., Any]] = {}
        self.tool_factory: dict[str, Callable[..., Any]] = {}

    def reset(self) -> None:
        """Clear all registries. Used by test fixtures."""
        self.scripted.clear()
        self.condition.clear()
        self.tool_factory.clear()

    @contextmanager
    def session(self) -> Generator[Registry, None, None]:
        """Snapshot-and-restore context manager for test isolation.

        Saves the current state, yields a clean registry, and restores
        the original state on exit.
        """
        saved = (
            dict(self.scripted),
            dict(self.condition),
            dict(self.tool_factory),
        )
        self.reset()
        try:
            yield self
        finally:
            self.scripted, self.condition, self.tool_factory = saved


#: Module-level singleton — the one registry instance used by the entire runtime.
registry = Registry()
