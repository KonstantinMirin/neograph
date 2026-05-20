"""Per-compile registry container.

Post-ticket-bbov: this module no longer instantiates a process-global
`Registry()`. `compile()` builds a fresh `Registry()` per call, walks the
construct to populate the `scripted` dict from each node's
`_scripted_shim`, and seeds `condition`/`tool_factory` from the deprecated
fallback maintained in `_runtime_registry.py`. Factory closures close
over the per-compile registry instance.

Legacy consumers that still import `registry` get the fallback bridge
maintained by `_runtime_registry.py` — same semantics as before for
manual `register_scripted()`/etc., until ticket 5 removes that API.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any


class Registry:
    """Per-compile container for runtime-registered callables.

    Holds three dicts: scripted shims (@node-derived), conditions (Operator/Loop),
    and tool factories. `compile()` builds a fresh instance and threads it
    through factory closures so two `compile()` calls cannot collide.
    """

    def __init__(self) -> None:
        self.scripted: dict[str, Callable[..., Any]] = {}
        self.condition: dict[str, Callable[..., Any]] = {}
        self.tool_factory: dict[str, Callable[..., Any]] = {}

    def reset(self) -> None:
        """Clear all maps. Used by test fixtures that need a blank slate."""
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
