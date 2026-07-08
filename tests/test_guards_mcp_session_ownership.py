"""Structural guard: neograph owns NO MCP session lifecycle (w74k.3.1, guard c).

## Core Invariant
neograph CARRIES per-run MCP identity and never DECIDES it: it does not create,
hold, or dispose MCP client sessions. Per-run tools are built by consumer-owned
*async tool factories* (awaited on the arun() path) that return tools whose
per-request auth is minted by the ecosystem (httpx.Auth). The MCP session, if
any, lives inside the consumer's factory / adapter — never in neograph src.

This guard LOCKS the architect verdict that killed the session-ownership design
(nmb2; docs/design/mcp-session-ownership-review-2026-07-05.md). Without it, a
future PR could quietly reintroduce ``MultiServerMCPClient``/``ClientSession``
ownership into src/neograph and re-open the exact latency + cancellation-cleanup
surface the re-scope removed.

Non-vacuity is proven by ``test_scanner_flags_injected_session_ownership``, which
feeds synthetic source (not the real tree) through the SAME scanner and confirms
it FLAGS a planted violation.
"""

from __future__ import annotations

import pathlib
import re

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# Patterns that indicate neograph OWNS an MCP session / client lifecycle. The
# consumer's factory may reference these; neograph src must not.
_BANNED = re.compile(
    r"\bClientSession\b"
    r"|\bMultiServerMCPClient\b"
    r"|\blangchain_mcp_adapters\b"
    r"|\.session\s*\("  # `client.session(` context-manager ownership
    r"|^\s*from\s+mcp\b"
    r"|^\s*import\s+mcp\b",
    re.MULTILINE,
)


def _scan(text: str) -> list[str]:
    return [m.group(0).strip() for m in _BANNED.finditer(text)]


class TestNoMcpSessionOwnershipInSrc:
    def test_src_contains_no_mcp_session_lifecycle_code(self):
        offenders: dict[str, list[str]] = {}
        for path in SRC_DIR.rglob("*.py"):
            hits = _scan(path.read_text())
            if hits:
                offenders[str(path.relative_to(SRC_DIR))] = hits
        assert offenders == {}, (
            "neograph src must own NO MCP session lifecycle (guard c, w74k.3.1). "
            "Per-run MCP identity is carried via consumer-owned async tool "
            "factories + httpx.Auth, NOT by neograph creating/holding/disposing "
            f"sessions. Offending files: {offenders}"
        )

    def test_slip_banned_flags_injected_session_ownership(self, tmp_path: pathlib.Path):
        """Slip meta-test for the ``_BANNED`` regex (non-vacuity): the scanner must
        FLAG a planted violation."""
        bad = (
            "from mcp import ClientSession\n"
            "async def build(config, tc):\n"
            "    async with client.session() as s:\n"
            "        return s\n"
        )
        assert _scan(bad), "scanner failed to flag planted MCP session ownership"

    def test_slip_banned_flags_bare_session_call_without_imports(self, tmp_path: pathlib.Path):
        """Slip meta-test: the ``.session(`` clause must fire ON ITS OWN — an
        ownership form that dodges the import clauses (e.g. a client object
        threaded in from elsewhere) is still session ownership. Verification
        2026-07-06 found the original lookbehind ``(?<![\\w.])`` could never
        match the idiomatic ``client.session(`` (the char before the dot is a
        word char), so the clause was dead weight carried by the import lines."""
        bad = "async def build(config, tc, client):\n    async with client.session() as s:\n        return s\n"
        assert _scan(bad), "scanner failed to flag bare client.session( ownership"

    def test_scanner_accepts_clean_factory_source(self):
        """A consumer-neutral async factory (no session ownership) is NOT flagged."""
        good = (
            "async def factory(config, tool_config):\n"
            "    token = config['configurable']['mcp_auth']['server']\n"
            "    return build_tools(auth=token)\n"
        )
        assert _scan(good) == []
