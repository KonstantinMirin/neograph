"""Per-compile registry container — leaf module.

This module re-exports the `Registry` class from `_registry.py` for
internal callers. Post-§2 (ticket ezqz), the legacy module-level
`register_*` helpers and the `_FALLBACK` instance are GONE. compile()
builds a fresh `Registry` per call and threads it via closures.

Test-side `register_scripted`/`register_condition`/`register_tool_factory`
helpers (for tests/conftest fixtures and test sites that build up
registrations across helpers) now live in `tests/fakes.py`.
"""

from __future__ import annotations

from neograph._registry import Registry  # noqa: F401 — re-exported for internal use
