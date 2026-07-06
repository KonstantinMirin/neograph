"""Public testing utilities for neograph.

Two submodules, one supported surface:

- ``scaffold`` — the CLI test-scaffold code generator (``scaffold_tests``).
- ``fakes`` — battle-hardened LLM doubles + the canned ``FakeLLM`` /
  ``install_fake_llm`` entry points, promoted from tests/fakes.py so external
  consumers stop hand-rolling AIMessage duck-types that break on loop refactors.

The scaffold names (``scaffold_tests`` and the ``_collect_*`` helpers) are
re-exported here so ``neograph.testing.scaffold_tests`` and the existing
``from neograph.testing import _collect_edges, _collect_items`` sites keep working
after the module became a package.
"""

from __future__ import annotations

from neograph.testing.fakes import (
    FakeLLM,
    FakeTool,
    GatedAsyncFake,
    GuardFake,
    ReActFake,
    StringArgsFake,
    StructuredFake,
    StructuredFakeWithRaw,
    StubbornFake,
    TextFake,
    event_loop_lag_watchdog,
    install_fake_llm,
)
from neograph.testing.scaffold import (
    _collect_edges,
    _collect_items,
    scaffold_tests,
)

__all__ = [
    # Fakes — public testing doubles + canned entry points
    "FakeLLM",
    "install_fake_llm",
    "StructuredFake",
    "StructuredFakeWithRaw",
    "ReActFake",
    "StringArgsFake",
    "TextFake",
    "FakeTool",
    "GuardFake",
    "StubbornFake",
    "GatedAsyncFake",
    "event_loop_lag_watchdog",
    # Scaffold codegen
    "scaffold_tests",
    "_collect_edges",
    "_collect_items",
]
