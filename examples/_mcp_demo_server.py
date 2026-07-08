"""Shared FastMCP demo server for the MCP examples (23 and 24) — real protocol, offline.

NOT a numbered example and NOT part of the neograph public API (underscore-prefixed,
like ``_shared.py``). This is the server side that examples 23 and 24 spawn as a
stdio subprocess so they exercise the ACTUAL Model Context Protocol end-to-end —
no fakes at the protocol layer, no network, no API keys. The examples are the
CLIENT; only their LLMs stay fakes.

Why a real server instead of fake MCP-shaped tools: a fake ``StructuredTool`` demos
neograph, not MCP. Against a real server the per-operator identity beat (example 23)
carries a token through a real round-trip, and the self-healing hydration beat
(example 24) gets a REAL server-side error on resume that neograph heals from.

This server is PLAIN FastMCP on purpose — only ``@mcp.tool`` / ``@mcp.resource``,
no low-level handlers. A demo's job is to be copied, and every workaround line gets
copied too; the majority of MCP servers people write are plain FastMCP, so the demo
stays plain FastMCP. Two SDK gaps that a low-level handler could paper over are
handled by DESIGN instead (path-segment templates; code-agnostic expiry) and
documented in the APPENDIX at the bottom of this docstring.

The demo domain (shared by both examples): a tiny CRM.

  TOOLS
    crm_search(query, token)   read-only, idempotent — search deals
    kb_lookup(topic, token)    read-only, idempotent — knowledge-base article
    get_deal(deal_id, token)   returns the deal AS structured text PLUS two
                               resource_link content blocks pointing at the
                               activity-history and email-history resources
    update_deal(deal_id, ...)  MUTATING — the gated-mutation beat (example 23)
    arm_email_expiry()         control tool — arms the one-shot email-history
                               expiry so example 24 gets a real server error on
                               resume

  RESOURCES (plain ``@mcp.resource`` — parameterized, auto-listed as templates)
    mcp://crm/deals/{deal_id}/activity              activity history (JSON)
    mcp://crm/deals/{deal_id}/emails/{start}/{end}  email history, a date-range
                                                    FRACTION via PATH SEGMENTS

  AUTH ECHO
    stdio has no HTTP headers, so per-operator identity travels as a ``token``
    tool ARGUMENT that every tool stamps into its result. Example 23 runs the
    pipeline as operator-A then operator-B and observes which token each run's
    tools carried. (The streamable-http variant would use httpx.Auth / bearer
    headers instead — documented in comments only; see run() at the bottom.)

TRANSPORT: stdio only. The client spawns this file as
``StdioConnection(command=sys.executable, args=[<this file>], env={...})`` via
``MultiServerMCPClient``. No ports, no network — CI-safe.

Run standalone (for debugging):
    uv run --extra mcp-examples python examples/_mcp_demo_server.py

────────────────────────────────────────────────────────────────────────────
APPENDIX — two verified FastMCP SDK gaps (mcp 1.28.x), and why this plain
server does not need to work around them
────────────────────────────────────────────────────────────────────────────

1. Query templates are inexpressible in ``@mcp.resource``. FastMCP extracts URI
   params with ``re.findall(r"{(\\w+)}", uri)`` and matches reads with
   ``uri_template.replace("{","(?P<").replace("}", ">[^/]+)")``, so an RFC-6570
   query fraction like ``emails{?from,to}`` never parses and a ``?from=..&to=..``
   query string on a read URI breaks the ``$``-anchored match. Rather than a
   low-level ``read_resource`` handler that parses the query string by hand, we
   express the date-range fraction as PATH SEGMENTS —
   ``emails/{start}/{end}`` — which ``@mcp.resource`` supports natively and which
   neograph's client-side RFC-6570 level-1 expansion (``_uri_template._expand_uri``)
   interpolates without any special-casing.

2. The ``@tool`` / ``@resource`` wrappers swallow JSON-RPC error codes. A resource
   handler's exception is re-raised through ``ResourceError`` and reaches the
   client as a generic ``code: 0`` — so a decorator-based resource CANNOT emit a
   real ``-32002``. We do NOT forge one with a low-level handler. neograph's
   self-heal is code-agnostic: ``di.hydrate_resource_ref`` replays the (idempotent)
   producing call on ANY fetch failure, so a plain ``raise`` from the expired
   ``email_history`` resource — surfacing as ``code: 0`` — triggers the exact same
   replay-and-retry a ``-32002`` would. The robustness is the feature; the demo
   proves it with the error real FastMCP actually produces.
"""

from __future__ import annotations

import json
import os
import pathlib

from mcp.server.fastmcp import FastMCP
from mcp.types import ResourceLink, TextContent
from pydantic import AnyUrl

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
# queries a FRACTION of via the emails/{start}/{end} date-range path template.
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
    resume read fails exactly once, the pipeline replays the (idempotent)
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
    server, only the links travel. The email-history link carries a date-range
    FRACTION as path segments (`/emails/<start>/<end>`) example 24 reads
    selectively."""
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
            uri=AnyUrl(f"mcp://crm/deals/{deal_id}/emails/2024-01-01/2024-12-31"),
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

    Example 24 calls this during the Operator pause; the resume read then fails
    once from the email-history resource, and neograph self-heals by replaying
    the idempotent producing call."""
    _state_file().write_text("armed")
    return "email-history expiry armed (one-shot)"


# ── Resources: plain @mcp.resource (parameterized -> auto-listed templates) ───


@mcp.resource("mcp://crm/deals/{deal_id}/activity", mime_type="application/json")
def activity_history(deal_id: str) -> str:
    """Activity history for a deal (static path template)."""
    return json.dumps({"deal_id": deal_id, "events": _ACTIVITY.get(deal_id, [])})


@mcp.resource("mcp://crm/deals/{deal_id}/emails/{start}/{end}", mime_type="application/json")
def email_history(deal_id: str, start: str, end: str) -> str:
    """Email-history FRACTION between ``start`` and ``end`` (inclusive), by path.

    On the one-shot expiry a plain ``raise`` is enough: FastMCP wraps it as a
    code-0 error and neograph's replay-on-any-fetch-failure heals regardless
    (see APPENDIX gap 2 in the module docstring)."""
    if _email_history_expired_once():
        raise ValueError(f"email-history no longer available: deals/{deal_id}/emails/{start}/{end}")
    emails = [e for e in _EMAILS.get(deal_id, []) if (not start or e["ts"] >= start) and (not end or e["ts"] <= end)]
    return json.dumps({"deal_id": deal_id, "start": start, "end": end, "emails": emails})


if __name__ == "__main__":
    # stdio transport: no ports, no network. The client spawns this process.
    #
    # Streamable-HTTP variant (documented, not used by the offline examples):
    #     mcp.run(transport="streamable-http")   # then the client connects over
    # HTTP and passes per-operator identity as a bearer header via httpx.Auth,
    # instead of the stdio `token` argument echoed above.
    mcp.run(transport="stdio")
