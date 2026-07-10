"""Structural guard: the MCP ``structuredContent`` -> ``output_model`` rehydration
lives at exactly ONE site — ``neograph_mcp._typed.rehydrate`` (neograph-32qtx).

Before neograph-32qtx the rehydration block (the missing-``structuredContent``
``ValueError`` + the ``parse(structured) ... else output_model.model_validate(...)``
return) was hand-cloned at TWO sites (``_session.py``, ``_client.py``); the test
fakes would have made a third. Factoring it into ``_typed.py`` made real==fake
parity STRUCTURAL. This guard pins that: any new copy of the block outside
``_typed.py`` (a future surface re-inlining it, drifting the fake into an echo
chamber) fails here — pointing the author at the shared helper.

Scans the ``src/neograph_mcp`` package source directly (text-shaped disease, one
exact literal), mirroring the whole-tree scan the neograph-32qtx codebase-scan atom
recorded.
"""

from __future__ import annotations

from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent / "src" / "neograph_mcp"
_HELPER = _PKG / "_typed.py"

# The two literals that uniquely identify the rehydration block. Either one showing
# up outside _typed.py means a surface re-inlined the rehydration instead of calling
# rehydrate() — the exact drift the helper exists to prevent.
_REHYDRATION_MARKERS = (
    "parse(structured) if parse is not None else",
    "server returned no structuredContent",
)


def test_mcp_rehydration_block_only_lives_in_typed_helper():
    """No ``neograph_mcp`` module other than ``_typed.py`` may contain the
    rehydration block — every consumer surface calls ``rehydrate()`` instead."""
    offenders: list[str] = []
    for path in sorted(_PKG.rglob("*.py")):
        if path == _HELPER:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in _REHYDRATION_MARKERS:
            if marker in text:
                offenders.append(f"{path.relative_to(_PKG.parent.parent)} contains {marker!r}")

    assert not offenders, (
        "MCP structuredContent rehydration was re-inlined instead of calling "
        "neograph_mcp._typed.rehydrate(); this reopens the real-vs-fake echo chamber "
        "(neograph-32qtx). Offending sites:\n  " + "\n  ".join(offenders)
    )


def test_rehydration_helper_still_exists():
    """A canary: the guard above is only meaningful while the shared helper exists.
    If ``rehydrate`` is renamed/removed, this fails loudly so the guard is revisited
    rather than silently passing on an empty package."""
    assert _HELPER.is_file(), "src/neograph_mcp/_typed.py (the shared rehydrate helper) is missing"
    assert "def rehydrate(" in _HELPER.read_text(encoding="utf-8"), (
        "neograph_mcp._typed.rehydrate was renamed — update the rehydration guard + its callers"
    )
