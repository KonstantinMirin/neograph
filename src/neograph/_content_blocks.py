"""MCP tool-result content-block scanning — a neutral leaf.

Extracted from ``_agent_cycle.py`` neograph-a5nh so BOTH the ref-lift point
(``_agent_cycle._lift_resource_refs``) AND the hydration replay path
(``_resource_hydrate``) can scan ``resource_link`` blocks without importing the
heavy ``_agent_cycle`` module. Stdlib-only; imports nothing from ``neograph``.
"""

from __future__ import annotations

from typing import Any


def _iter_content_blocks(result: Any) -> list:
    """Yield candidate content blocks from a tool result.

    MCP tools loaded via langchain-mcp-adapters return content as a list of
    blocks; a ``content_and_artifact`` tool returns ``(content, artifact)``; some
    return ``{"content": [...]}``. A plain string / Pydantic model (the common
    case) carries no blocks. Kept permissive so a resource-emitting server's
    ``resource_link`` blocks are found wherever they land, without coupling to a
    single result shape."""
    if isinstance(result, list):
        return result
    if isinstance(result, tuple):
        return [b for part in result if isinstance(part, list) for b in part]
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        return result["content"]
    return []


def _block_field(block: Any, key: str) -> Any:
    """Read a content-block field from a dict block or an object block."""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _first_resource_link_uri(result: Any) -> str | None:
    """The uri of the first ``resource_link`` block in a tool result, or None.

    Used on the hydration replay path: after re-invoking a producing call, the
    fresh ``resource_link`` it emits carries the re-derived (unexpired) uri to
    read from."""
    for block in _iter_content_blocks(result):
        if _block_field(block, "type") == "resource_link":
            uri = _block_field(block, "uri")
            if uri:
                return uri
    return None
