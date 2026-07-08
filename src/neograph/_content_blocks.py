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

    Used on the hydration replay path as the LAST-RESORT re-derivation when a
    producing call emits a single ``resource_link`` (or none matches the ref's
    kind). For a MULTI-link producer prefer ``_resource_link_uri_for_kind``."""
    for block in _iter_content_blocks(result):
        if _block_field(block, "type") == "resource_link":
            uri = _block_field(block, "uri")
            if uri:
                return uri
    return None


def _resource_link_kind(block: Any) -> str:
    """The domain KIND of a ``resource_link`` block.

    The SINGLE derivation shared by the ref-lift point
    (``_agent_cycle._lift_resource_refs``, which stamps ``ResourceRef.kind``) and
    the hydration replay matcher (``_resource_link_uri_for_kind``): the block's
    ``name``, else the uri scheme, else ``"resource"``. Kept in one place so the
    kind a ref is STAMPED with at lift time and the kind the replay path MATCHES
    against can never drift. neograph-m9sj"""
    # str(): the REPLAY path scans raw mcp ``ResourceLink`` objects whose ``.uri``
    # is a pydantic ``AnyUrl`` (``"://" in AnyUrl`` raises TypeError), not the str
    # uris the lift path sees on dict blocks. neograph-m9sj
    uri = str(_block_field(block, "uri") or "")
    scheme = uri.split("://", 1)[0] if "://" in uri else ""
    return _block_field(block, "name") or scheme or "resource"


def _resource_link_uri_for_kind(result: Any, kind: str) -> str | None:
    """The uri of the first ``resource_link`` block whose KIND matches ``kind``.

    The kind-aware re-derivation for the hydration replay path. neograph-m9sj
    When a producing call emits MORE THAN ONE ``resource_link`` — e.g. a CRM
    ``get_deal`` returning both an activity-history and an email-history link —
    taking the FIRST link regardless of kind (``_first_resource_link_uri``) would
    heal a ref to a DIFFERENT-kind blob and silently parse it into the wrong
    model. Matching on the ref's ``kind`` via the same derivation the lift used
    re-reads the correct resource. Returns ``None`` when no block matches so the
    caller can fall back to first-link then ``ref.uri``."""
    for block in _iter_content_blocks(result):
        if _block_field(block, "type") != "resource_link":
            continue
        if _resource_link_kind(block) == kind:
            uri = _block_field(block, "uri")
            if uri:
                return str(uri)
    return None
