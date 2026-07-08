"""Shared FastMCP demo server for the MCP examples (23 and 24) — real protocol, offline.

NOT a numbered example and NOT part of the neograph public API (underscore-prefixed,
like ``_shared.py``). This is the server side that examples 23 and 24 spawn as a
stdio subprocess so they exercise the ACTUAL Model Context Protocol end-to-end —
no fakes at the protocol layer, no network, no API keys. The examples are the
CLIENT; only their LLMs stay fakes.

Why a real server instead of fake MCP-shaped tools: a fake ``StructuredTool`` demos
neograph, not MCP. Against a real server the per-operator identity beat (example 23)
carries a token through a real round-trip, and the self-healing hydration beat
(example 24) gets a REAL ``-32002`` resource-not-found from a real server.

The demo domain (shared by both examples): a tiny CRM.

  TOOLS
    crm_search(query, token)   read-only, idempotent — search deals
    kb_lookup(topic, token)    read-only, idempotent — knowledge-base article
    get_deal(deal_id, token)   returns the deal AS structured text PLUS two
                               resource_link content blocks pointing at the
                               activity-history and email-history resources
    update_deal(deal_id, ...)  MUTATING — the gated-mutation beat (example 23)
    arm_email_expiry()         control tool — arms the one-shot email-history
                               expiry so example 24 gets a real -32002 on resume

  RESOURCES (served by a custom low-level read_resource handler — see below)
    mcp://crm/deals/{id}/activity         activity history (JSON)
    mcp://crm/deals/{id}/emails{?from,to} email history, RFC-6570 query fraction

  AUTH ECHO
    stdio has no HTTP headers, so per-operator identity travels as a ``token``
    tool ARGUMENT that every tool stamps into its result. Example 23 runs the
    pipeline as operator-A then operator-B and observes which token each run's
    tools carried. (The streamable-http variant would use httpx.Auth / bearer
    headers instead — documented in comments only; see run() at the bottom.)

TRANSPORT: stdio only. The client spawns this file as
``StdioConnection(command=sys.executable, args=[<this file>], env={...})`` via
``MultiServerMCPClient``. No ports, no network — CI-safe.

────────────────────────────────────────────────────────────────────────────
TWO VERIFIED SDK GAPS (mcp 1.28.x) and how this server works around them
────────────────────────────────────────────────────────────────────────────

1. FastMCP's ``@mcp.resource`` decorator CANNOT express RFC-6570 query templates.
   It extracts URI params with ``re.findall(r"{(\\w+)}", uri)`` and matches reads
   with ``uri_template.replace("{","(?P<").replace("}", ">[^/]+)")`` — so
   ``{?from,to}`` never parses and a ``?from=..&to=..`` query string on a read URI
   breaks the ``$``-anchored match. WORKAROUND: we do not use ``@mcp.resource`` for
   reads at all. A custom low-level ``read_resource`` handler parses the full URI —
   query string included — with ``urllib.parse``, so the ``emails{?from,to}``
   fraction query works exactly as written.

2. FastMCP's ``@mcp.tool`` / ``@mcp.resource`` wrappers SWALLOW JSON-RPC error
   codes: a resource handler's exception is re-raised as ``ValueError`` then
   ``ResourceError``, reaching the client as a generic ``code: 0``. So neither
   decorator can emit a real ``-32002``. WORKAROUND: the custom low-level
   ``read_resource`` handler (registered directly on ``mcp._mcp_server``, where the
   request loop preserves a handler-raised ``McpError`` verbatim) raises
   ``McpError(ErrorData(code=-32002, ...))`` and the client receives a real -32002.

The custom low-level ``read_resource`` handler registers into the same
``request_handlers[ReadResourceRequest]`` slot FastMCP uses, and last registration
wins — so it must be defined AFTER ``FastMCP(...)`` construction (it is).

Run standalone (for debugging):
    uv run --extra mcp-examples python examples/_mcp_demo_server.py
"""

from __future__ import annotations

import json
import os
import pathlib
from urllib.parse import parse_qs, urlparse

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, ResourceLink, ResourceTemplate, TextContent
from pydantic import AnyUrl

# JSON-RPC "resource not found / no longer available". The MCP spec assigns
# -32002 to this; mcp.types does NOT export a named constant, so we spell it out.
RESOURCE_NOT_FOUND = -32002

# ── In-memory CRM domain ─────────────────────────────────────────────────────

_DEALS: dict[str, dict[str, object]] = {
    "D1": {"id": "D1", "name": "Acme renewal", "stage": "negotiation", "amount": 50_000, "owner": "alice"},
    "D2": {"id": "D2", "name": "Globex expansion", "stage": "discovery", "amount": 120_000, "owner": "bob"},
}

_KB: dict[str, str] = {
    "renewal-playbook": "Anchor on realized value, propose a 2-year term for a discount.",
    "discovery-checklist": "Confirm budget owner, timeline, and the metric they are measured on.",
}

_ACTIVITY: dict[str, list[dict[str, str]]] = {
    "D1": [
        {"ts": "2024-03-01", "kind": "call", "note": "Kickoff with procurement"},
        {"ts": "2024-05-14", "kind": "email", "note": "Sent renewal quote"},
    ],
    "D2": [{"ts": "2024-06-02", "kind": "meeting", "note": "Discovery call"}],
}

# Email corpus per deal — the "large corpus consumed selectively" that example 24
# queries a FRACTION of via the {?from,to} date-range template.
_EMAILS: dict[str, list[dict[str, str]]] = {
    "D1": [
        {"ts": "2024-02-10", "subject": "Intro", "from": "alice@acme"},
        {"ts": "2024-04-22", "subject": "Pricing questions", "from": "cfo@acme"},
        {"ts": "2024-06-30", "subject": "Renewal terms", "from": "alice@acme"},
    ],
    "D2": [{"ts": "2024-06-05", "subject": "Follow-up", "from": "bob@globex"}],
}


def _state_file() -> pathlib.Path:
    """Marker file backing the one-shot email-history expiry.

    File-backed (not a module global) ON PURPOSE: each stdio session spawns a
    FRESH server subprocess with fresh module state, so an in-memory flag would
    not survive the pause/resume boundary example 24 needs. The client passes a
    stable path via ``env={"NEOGRAPH_MCP_DEMO_STATE": ...}`` so every spawn agrees.
    """
    return pathlib.Path(os.environ.get("NEOGRAPH_MCP_DEMO_STATE", "/tmp/neograph_mcp_demo_state.marker"))


def _email_history_expired_once() -> bool:
    """True on the FIRST email read after expiry is armed; self-clears afterward.

    The one-shot shape is what makes example 24's self-heal deterministic: the
    resume read hits -32002 exactly once, the pipeline replays the (idempotent)
    producing call, and the retry succeeds.
    """
    marker = _state_file()
    if marker.exists() and marker.read_text().strip() == "armed":
        marker.write_text("fired")
        return True
    return False


mcp = FastMCP("neograph-crm-demo")


# ── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool()
def crm_search(query: str, token: str = "anon") -> dict:
    """Read-only, idempotent. Returns deals whose name matches `query`.

    `token` is echoed under `acting_as` — the per-operator identity beat."""
    q = query.lower()
    hits = [
        {"id": d["id"], "name": d["name"], "stage": d["stage"]} for d in _DEALS.values() if q in str(d["name"]).lower()
    ]
    return {"query": query, "hits": hits, "acting_as": token}


@mcp.tool()
def kb_lookup(topic: str, token: str = "anon") -> dict:
    """Read-only, idempotent. Returns a knowledge-base article for `topic`."""
    return {"topic": topic, "article": _KB.get(topic, "No article found."), "acting_as": token}


@mcp.tool()
def get_deal(deal_id: str, token: str = "anon") -> list:
    """Return the deal as structured text PLUS resource_link blocks.

    The two resource_links are the manifest example 24 lifts onto the bus: refs
    to the activity-history and email-history resources — blobs stay on the
    server, only the links travel. The email-history link carries the RFC-6570
    query fraction (`?from=..&to=..`) example 24 reads selectively."""
    deal = _DEALS.get(deal_id, {"id": deal_id, "error": "unknown deal"})
    return [
        TextContent(type="text", text=json.dumps({**deal, "acting_as": token})),
        ResourceLink(
            type="resource_link",
            uri=AnyUrl(f"mcp://crm/deals/{deal_id}/activity"),
            name="activity-history",
            mimeType="application/json",
        ),
        ResourceLink(
            type="resource_link",
            uri=AnyUrl(f"mcp://crm/deals/{deal_id}/emails?from=2024-01-01&to=2024-12-31"),
            name="email-history",
            mimeType="application/json",
        ),
    ]


@mcp.tool()
def update_deal(deal_id: str, stage: str, token: str = "anon") -> dict:
    """MUTATING. Advances a deal's stage — the gated-mutation beat (example 23).

    (Mutation is in-process only; it does not persist across stdio subprocess
    spawns. The demo's point is the neograph gate that PAUSES before this fires,
    not durable CRM state.)"""
    deal = _DEALS.get(deal_id)
    if deal is None:
        return {"ok": False, "error": "unknown deal", "acting_as": token}
    prior = deal["stage"]
    deal["stage"] = stage
    return {"ok": True, "id": deal_id, "prior_stage": prior, "new_stage": stage, "acting_as": token}


@mcp.tool()
def arm_email_expiry() -> str:
    """Control tool: arm the one-shot email-history expiry.

    Example 24 calls this during the Operator pause; the resume read then gets a
    real -32002 from the email-history resource exactly once."""
    _state_file().write_text("armed")
    return "email-history expiry armed (one-shot)"


# ── Resources: custom low-level handlers (see module docstring for why) ───────


@mcp._mcp_server.read_resource()
async def _read_resource(uri) -> list[ReadResourceContents]:
    """Serve activity + email-history resources; emit a real -32002 on expiry.

    Parses the full URI including the query string, so the emails{?from,to}
    fraction query works despite FastMCP's decorator not supporting query
    templates."""
    parsed = urlparse(str(uri))
    segments = parsed.path.strip("/").split("/")  # deals/<id>/<kind>
    deal_id = segments[1] if len(segments) > 1 else ""
    kind = segments[-1] if segments else ""

    if kind == "emails":
        if _email_history_expired_once():
            raise McpError(ErrorData(code=RESOURCE_NOT_FOUND, message=f"email-history no longer available: {uri}"))
        qs = parse_qs(parsed.query)
        lo = qs.get("from", [""])[0]
        hi = qs.get("to", [""])[0]
        emails = [e for e in _EMAILS.get(deal_id, []) if (not lo or e["ts"] >= lo) and (not hi or e["ts"] <= hi)]
        body = {"deal_id": deal_id, "from": lo, "to": hi, "emails": emails}
        return [ReadResourceContents(content=json.dumps(body), mime_type="application/json")]

    if kind == "activity":
        body = {"deal_id": deal_id, "events": _ACTIVITY.get(deal_id, [])}
        return [ReadResourceContents(content=json.dumps(body), mime_type="application/json")]

    raise McpError(ErrorData(code=RESOURCE_NOT_FOUND, message=f"unknown resource: {uri}"))


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> list[ResourceTemplate]:
    """Advertise both resource templates — including the RFC-6570 query form for
    email-history, so a client can discover the typed reader's shape."""
    return [
        ResourceTemplate(
            uriTemplate="mcp://crm/deals/{deal_id}/activity",
            name="activity-history",
            mimeType="application/json",
        ),
        ResourceTemplate(
            uriTemplate="mcp://crm/deals/{deal_id}/emails{?from,to}",
            name="email-history",
            mimeType="application/json",
        ),
    ]


if __name__ == "__main__":
    # stdio transport: no ports, no network. The client spawns this process.
    #
    # Streamable-HTTP variant (documented, not used by the offline examples):
    #     mcp.run(transport="streamable-http")   # then the client connects over
    # HTTP and passes per-operator identity as a bearer header via httpx.Auth,
    # instead of the stdio `token` argument echoed above.
    mcp.run(transport="stdio")
