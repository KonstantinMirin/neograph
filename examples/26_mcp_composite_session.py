"""Example 26: composing over federated MCP primitives — ``mcp_session``.

Examples 23/25 BIND one federated tool 1:1 (``mcp_tool_factories`` /
``mcp_tool_factory``): the agent decides when to call it, and each binding is one
tool. But some work is a COMPOSITE — a scripted step that must ISSUE several
primitive calls and assemble the result deterministically (resolve a company, then
fetch its notes/emails/calls/meetings, then flatten). Teaching an agent to
orchestrate that would be non-deterministic and burn tool-calls; you want plain
Python control flow that calls the primitives itself.

``mcp_session`` is that seam: connect ONCE, call N tools by name over the single
connection, get typed/structured results, close. It reuses the exact transport +
token machinery the factories have internally, and — like them — is consumer-owned
and offline-at-build (the connect defers to ``async with`` entry).

The scenario: a ``deal_review`` composite talks to the REAL demo MCP server
(``examples/_mcp_demo_server.py``, a stdio subprocess). It calls TWO primitives
over ONE session — ``crm_search`` (typed: rehydrated into a client model) then
``get_deal`` (content + a resource_link manifest) — and assembles a typed
``DealReview``. No LLM, no API keys: a composite is deterministic Python.

Three beats:

  1. ONE SESSION, N CALLS — ``async with mcp_session(...) as s`` then
     ``s.call("crm_search", ...)`` and ``s.call("get_deal", ...)`` over the SAME
     connection (not two connects, not two materialized factories).

  2. TYPED + STRUCTURED RESULTS — ``s.call("crm_search", ..., output_model=Model)``
     rehydrates the server's structuredContent into a client model;
     ``s.call("get_deal", ...)`` returns an ``McpCallResult`` whose ``.text`` is the
     deal JSON and whose ``.content`` carries the resource_link manifest as blocks.

  3. PER-RUN IDENTITY, MINTED ONCE — a ``token_provider`` reads the operator from
     ``config['configurable']`` at ``__aenter__``; the SAME identity rides every
     call over the session (the server echoes it under ``acting_as``).

Driven through neograph's normal node execution: the composite is a ``raw``-mode
node, compiled and run with ``arun()`` — the blessed "scripted composite over
federated primitives" pattern.

Run (needs the mcp-examples extra; no API keys, no network beyond the local
subprocess):
    uv run --extra dev --extra mcp-examples python examples/26_mcp_composite_session.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

import neograph
from neograph import compile, construct_from_functions, node
from neograph_mcp import StdioServer, mcp_session

DEMO_SERVER = str((Path(__file__).parent / "_mcp_demo_server.py").resolve())

CRM = StdioServer(command=sys.executable, args=[DEMO_SERVER])


# ── Per-run identity: the token_provider (beat 3) ────────────────────────────
# mcp_session calls this ONCE at __aenter__ with config['configurable']. Over
# stdio the returned token rides as the `token` tool argument (the server echoes
# it under `acting_as`); over streamable-http it would ride as a bearer header.


def token_provider(configurable: dict[str, Any]) -> str:
    return configurable.get("mcp_auth", "anon")


# ── Client-side result models ────────────────────────────────────────────────
# The typed channel: declare output_model= on a call and mcp_session rehydrates
# the server's structuredContent into OUR model (the fields we care about).


class DealHit(BaseModel, frozen=True):
    id: str
    name: str
    stage: str


class CrmSearchResult(BaseModel, frozen=True):
    query: str
    hits: list[DealHit]
    acting_as: str  # the per-run identity the server echoed


class DealReview(BaseModel, frozen=True):
    """The assembled composite output — built from TWO primitive calls."""

    deal_id: str
    deal_name: str
    stage: str
    acting_as: str
    manifest_refs: int  # resource_link blocks get_deal returned (the manifest)


# ── The composite: ONE session, two primitives, assembled result ─────────────


@node(mode="raw", outputs=DealReview)
async def deal_review(state, config):  # raw mode: (state, config) -> state-update dict
    query = config["configurable"]["query"]

    async with mcp_session("crm", CRM, token_provider=token_provider, config=config) as s:
        # Beat 1+2: primitive #1 — typed search (rehydrated into CrmSearchResult).
        search = await s.call("crm_search", {"query": query}, output_model=CrmSearchResult)
        top = search.hits[0]

        # Beat 1: primitive #2 — get_deal over the SAME session. Content-only
        # (text + a resource_link manifest), so we read McpCallResult, not a model.
        deal = await s.call("get_deal", {"deal_id": top.id})

    detail = json.loads(deal.text or "{}")
    manifest_refs = sum(1 for b in deal.content if b.get("type") == "file")

    return {
        "deal_review": DealReview(
            deal_id=top.id,
            deal_name=top.name,
            stage=detail.get("stage", top.stage),
            acting_as=search.acting_as,
            manifest_refs=manifest_refs,
        )
    }


pipeline = construct_from_functions("deal-review", [deal_review])


async def main() -> None:
    print("=" * 66)
    print("Composite over federated MCP primitives — one session, two calls")
    print("=" * 66)

    graph = compile(pipeline)

    for operator in ("operator-A", "operator-B"):
        result = await neograph.arun(
            graph,
            input={},
            config={"configurable": {"query": "Acme", "mcp_auth": operator}},
        )
        review: DealReview = result["deal_review"]
        print(f"\nRun as {operator}:")
        print(f"  deal        : {review.deal_id} — {review.deal_name} [{review.stage}]")
        print(f"  acting_as   : {review.acting_as}  (identity minted once, rode both calls)")
        print(f"  manifest    : {review.manifest_refs} resource_link refs from get_deal")
        assert review.deal_id == "D1"
        assert review.acting_as == operator  # same identity on crm_search AND get_deal
        assert review.manifest_refs == 2

    print("\n" + "=" * 66)
    print("One connection, two federated primitives, a typed assembled result —")
    print("deterministic Python control flow, zero hand-rolled MCP transport.")
    print("=" * 66)


if __name__ == "__main__":
    asyncio.run(main())
