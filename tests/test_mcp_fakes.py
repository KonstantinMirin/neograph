"""TDD RED (neograph-5jrd0.6) for the MCP test-fakes feature (neograph-32qtx).

Pins the CORE INVARIANT: a downstream consumer can unit-test an MCP composite (a
raw-mode node issuing federated-primitive calls via ``mcp_session``) with NO
subprocess and NO network, using ``neograph_mcp.testing.FakeMcpSession`` — and the
fake honours the SAME output_model / typed_result rehydration and the SAME
identity-minting semantics as the real ``McpSession`` (no echo chamber).

These tests are RED today: ``neograph_mcp.testing`` (``FakeMcpSession``) and the
shared ``_rehydrate`` helper the plan factors out do NOT exist yet — every test
fails at its in-body ``from neograph_mcp.testing import FakeMcpSession`` import.
Imports live INSIDE the test bodies (mirroring ``tests/test_mcp_battery.py``) so
collection succeeds and each test reports FAILED rather than a collection ERROR.

Three acceptance behaviours (per the refined 32qtx plan):
  1. A raw-mode composite node driven by ``FakeMcpSession`` runs with zero
     subprocess/network; ``.calls`` records identity+args; ``output_model``
     rehydration produces the client model.
  2. PARITY (gated, real demo server): capture REAL ``structuredContent`` from
     ``mcp_session`` against ``examples/_mcp_demo_server.py``, script the fake from
     that CAPTURED payload, assert the fake's rehydrated model == the real one.
  3. Fake missing-``structuredContent`` raises the identical ``ValueError``; fake
     ``isError`` raises ``McpToolCallError``.

Gated with ``pytest.importorskip('mcp')`` + ``('langchain_mcp_adapters')`` because
importing ``neograph_mcp`` (any submodule, incl. ``testing``) fail-loud-requires
the ``mcp`` extra; the fakes remove SUBPROCESS+NETWORK, not the dependency.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("langchain_mcp_adapters")

from pydantic import BaseModel  # noqa: E402 — after the extra gate by design

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SERVER = REPO_ROOT / "examples" / "_mcp_demo_server.py"

_HAS_MCP = bool(importlib.util.find_spec("mcp")) and bool(importlib.util.find_spec("langchain_mcp_adapters"))
requires_mcp = pytest.mark.skipif(not _HAS_MCP, reason="requires the mcp extra (mcp + langchain-mcp-adapters)")


def _demo_stdio_server():
    """A StdioServer pointed at the shared FastMCP demo server (offline, keyless)."""
    from neograph_mcp import StdioServer

    return StdioServer(command=sys.executable, args=[str(DEMO_SERVER)])


# ── Client-side result models (defined independently of the demo server's) ────


class ClientDealHit(BaseModel):
    id: str
    name: str
    stage: str


class ClientCrmResult(BaseModel):
    query: str
    hits: list[ClientDealHit]
    acting_as: str  # the per-run identity the server echoed


class DealReview(BaseModel):
    """The assembled composite output — built from TWO primitive calls."""

    deal_id: str
    deal_name: str
    stage: str
    acting_as: str
    manifest_refs: int


def _token_provider(configurable):
    return configurable.get("mcp_auth", "anon")


# ── Behaviour 1: composite node driven by the fake, zero subprocess/network ────


class TestFakeSessionComposite:
    """A raw-mode composite node runs against ``FakeMcpSession`` — zero subprocess,
    zero network — ``.calls`` records identity+args, and ``output_model``
    rehydration produces the CLIENT model."""

    async def test_raw_composite_runs_on_fake_records_identity_and_rehydrates(self, monkeypatch):
        """Drive a real neograph raw-mode node through ``compile()`` + ``arun()``
        with the session obtained from a config-injected builder that returns a
        ``FakeMcpSession``. Assert: the pipeline produces the assembled ``DealReview``
        (proving ``output_model=ClientCrmResult`` rehydration happened on the fake);
        the fake's ``.calls`` recorded BOTH primitive calls with the minted identity
        and the verbatim caller args; and the REAL transport was never touched
        (``_client_for`` is monkeypatched to explode)."""
        import neograph

        # Zero-network proof: if the fake ever reached the real client stitching this
        # blows up. FakeMcpSession must connect NOWHERE.
        import neograph_mcp._client as _client_mod
        from neograph import compile, construct_from_functions, node
        from neograph_mcp import McpCallResult
        from neograph_mcp.testing import FakeMcpSession

        def _boom(*a, **k):
            raise AssertionError("FakeMcpSession touched the real MCP transport (_client_for)")

        monkeypatch.setattr(_client_mod, "_client_for", _boom)

        # Scripted server payloads: crm_search emits structuredContent; get_deal is
        # content-only (a text mirror + two file/resource_link blocks).
        crm_structured = {
            "query": "acme",
            "hits": [{"id": "D1", "name": "Acme renewal", "stage": "negotiation"}],
            "acting_as": "operator-A",
            "bearer_identity": "operator-A",  # extra key a narrower client model drops
        }
        get_deal_content = [
            {"type": "text", "text": json.dumps({"id": "D1", "stage": "won", "acting_as": "operator-A"})},
            {"type": "file", "url": "mcp://crm/notes/D1"},
            {"type": "file", "url": "mcp://crm/emails/D1"},
        ]

        made: list = []

        def build_session(config):
            s = FakeMcpSession(
                results={
                    "crm_search": McpCallResult(content=[], structured=crm_structured),
                    "get_deal": McpCallResult(content=get_deal_content, structured=None),
                },
                token_provider=_token_provider,
                config=config,
            )
            made.append(s)
            return s

        @node(mode="raw", outputs=DealReview)
        async def deal_review(state, config):
            cfg = config["configurable"]
            async with cfg["session_builder"](config) as s:
                search = await s.call("crm_search", {"query": cfg["query"]}, output_model=ClientCrmResult)
                top = search.hits[0]
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
        graph = compile(pipeline)

        result = await neograph.arun(
            graph,
            input={},
            config={"configurable": {"query": "acme", "mcp_auth": "operator-A", "session_builder": build_session}},
        )

        # output_model rehydration produced the CLIENT model, flowing through the node.
        review: DealReview = result["deal_review"]
        assert review == DealReview(
            deal_id="D1",
            deal_name="Acme renewal",
            stage="won",
            acting_as="operator-A",
            manifest_refs=2,
        )

        # The fake recorded identity + verbatim args for BOTH primitive calls.
        assert len(made) == 1, "the composite opened exactly one session"
        calls = made[0].calls
        assert [c.tool for c in calls] == ["crm_search", "get_deal"]
        assert [c.args for c in calls] == [{"query": "acme"}, {"deal_id": "D1"}]
        # Identity minted ONCE via the SAME _resolve_token path the real session uses.
        assert [c.identity for c in calls] == ["operator-A", "operator-A"]

    async def test_tool_names_returns_scripted_keys(self):
        """``tool_names()`` reports the scripted tool set — the fake's stand-in for
        the real session's paginated ``tools/list``."""
        from neograph_mcp import McpCallResult
        from neograph_mcp.testing import FakeMcpSession

        async with FakeMcpSession(
            results={
                "crm_search": McpCallResult(content=[], structured={"query": "x", "hits": [], "acting_as": "anon"}),
                "get_deal": McpCallResult(content=[], structured=None),
            }
        ) as s:
            names = await s.tool_names()
        assert set(names) == {"crm_search", "get_deal"}


# ── Behaviour 2: PARITY against the REAL demo server (anti-echo-chamber) ───────


@requires_mcp
class TestFakeSessionParity:
    """The acceptance-critical parity test: script the fake from the demo server's
    CAPTURED ``structuredContent`` and prove fake-rehydrated == real-rehydrated. A
    demo-server ``structuredContent`` SHAPE change breaks this — that is the point
    (a hand-authored literal fed to both sides would be tautological)."""

    async def test_fake_rehydration_equals_real_captured_model(self):
        """(a) Call REAL ``mcp_session`` against ``_mcp_demo_server.py``, capturing
        the raw ``structuredContent`` dict AND the real rehydrated model. (b) Script
        ``FakeMcpSession`` from that CAPTURED structuredContent. (c) Assert the fake's
        rehydrated model equals the real captured model."""
        from neograph_mcp import McpCallResult, mcp_session
        from neograph_mcp.testing import FakeMcpSession

        # (a) Real capture over the demo server (one session, two calls).
        async with mcp_session("crm", _demo_stdio_server(), token_provider=_token_provider) as s:
            real_raw = await s.call("crm_search", {"query": "acme"})  # McpCallResult
            real_model = await s.call("crm_search", {"query": "acme"}, output_model=ClientCrmResult)

        assert isinstance(real_model, ClientCrmResult)
        captured_structured = real_raw.structured
        assert captured_structured is not None, "crm_search must emit structuredContent to script the fake"

        # (b) Script the fake from the CAPTURED payload (not a hand-authored literal).
        async with FakeMcpSession(
            results={"crm_search": McpCallResult(content=[], structured=captured_structured)}
        ) as fs:
            fake_model = await fs.call("crm_search", {"query": "acme"}, output_model=ClientCrmResult)

        # (c) fake-rehydrated == real-rehydrated: pins fake output == real server output.
        assert isinstance(fake_model, ClientCrmResult)
        assert fake_model == real_model


# ── Behaviour 3: error-path parity (identical ValueError; McpToolCallError) ────


class TestFakeSessionErrorParity:
    """The fake raises the SAME typed errors as the real session — the missing
    ``structuredContent`` ``ValueError`` (via the shared ``_rehydrate``) and
    ``McpToolCallError`` on a scripted ``isError``."""

    async def test_missing_structured_content_with_output_model_raises_typed_value_error(self):
        """A content-only scripted result (``structured=None``) with ``output_model=``
        set raises the SAME typed ``ValueError`` the real session raises — naming the
        tool, mentioning ``structuredContent``, and pointing at the server-annotation
        fix (NOT a bare AttributeError/KeyError, NOT ``McpToolCallError``)."""
        from neograph_mcp import McpCallResult
        from neograph_mcp.testing import FakeMcpSession

        async with FakeMcpSession(
            results={"get_deal": McpCallResult(content=[{"type": "text", "text": "[]"}], structured=None)}
        ) as s:
            with pytest.raises(ValueError) as excinfo:
                await s.call("get_deal", {"deal_id": "D1"}, output_model=ClientCrmResult)

        msg = str(excinfo.value)
        assert "get_deal" in msg, f"typed error must name the tool: {msg!r}"
        assert "structuredContent" in msg, f"typed error must mention structuredContent: {msg!r}"
        assert "annotation" in msg.lower() or "-> dict" in msg or "Pydantic" in msg, (
            f"typed error must point at the server-annotation fix: {msg!r}"
        )

    async def test_scripted_iserror_raises_mcp_tool_call_error(self):
        """A tool scripted as ``isError`` raises ``McpToolCallError`` (the negative
        path composite tests need), carrying the tool name — and the call is still
        recorded with its minted identity before the raise."""
        from neograph_mcp import McpToolCallError
        from neograph_mcp.testing import FakeMcpSession

        async with FakeMcpSession(
            errors={"boom"},
            token_provider=_token_provider,
            config={"configurable": {"mcp_auth": "operator-A"}},
        ) as s:
            with pytest.raises(McpToolCallError) as excinfo:
                await s.call("boom", {"deal_id": "D1"})

        assert excinfo.value.tool_name == "boom"
        # The call is recorded (identity + args) even though it raised.
        assert s.calls[-1].tool == "boom"
        assert s.calls[-1].args == {"deal_id": "D1"}
        assert s.calls[-1].identity == "operator-A"


# ── Surface 2: the bound-tool factory (agent/act path), driven keyless ────────


class TestFakeToolFactory:
    """``fake_mcp_tool_factory`` is a real ``ToolFactory``: it drops into
    ``compile(tool_factories={...})`` and drives an agent node with NO subprocess /
    NO protocol, recording the per-run identity minted via the same ``_resolve_token``
    the real factory uses, and returning the REAL rehydrated model as the tool
    result (shared ``rehydrate`` — no echo chamber)."""

    def test_fake_tool_factory_drives_agent_node_and_records_identity(self):
        import asyncio

        import neograph
        from neograph import compile
        from neograph_mcp.testing import fake_mcp_tool_factory
        from tests.fakes import build_test_compile_kwargs, configure_fake_llm
        from tests.schemas import Claims
        from tests.test_mcp_battery import _build_agent_construct, _react_fake

        # The factory returns the REAL rehydrated ClientCrmResult (built from scripted
        # structuredContent via the shared helper), and records identity from config.
        class _CrmSearchArgs(BaseModel):
            query: str

        factory = fake_mcp_tool_factory(
            "crm_search",
            result=None,
            output_model=ClientCrmResult,
            structured={
                "query": "acme",
                "hits": [{"id": "D1", "name": "Acme renewal", "stage": "won"}],
                "acting_as": "operator-A",
            },
            token_provider=_token_provider,
            args_schema=_CrmSearchArgs,
        )

        construct = _build_agent_construct("crm_search")
        llm_kw = configure_fake_llm(lambda tier: _react_fake("crm_search"))
        graph = compile(
            construct,
            tool_factories={"crm_search": factory},
            **build_test_compile_kwargs(),
            **llm_kw,
        )
        result = asyncio.run(
            neograph.arun(graph, input={"query": "acme"}, config={"configurable": {"mcp_auth": "operator-A"}})
        )

        assert result["scan_result"] == Claims(items=["done"])
        # The fake tool was driven keyless, recording the minted identity + verbatim args.
        assert factory.recorded_calls, "fake tool factory was never driven"
        rec = factory.recorded_calls[0]
        assert rec.tool == "crm_search"
        assert rec.args == {"query": "acme"}
        assert rec.identity == "operator-A"


# ── Surface 3: the resource fetcher tuple (FromResource path), keyless ─────────


class TestFakeResourceFetcher:
    """``fake_mcp_resource_fetcher`` returns the SAME ``(fetcher, replayer)`` tuple
    shape as the real ``mcp_resource_fetcher`` — so it drops into the same
    ``config['configurable']`` slot — with scripted content and recorded calls, no
    subprocess/network."""

    async def test_fetcher_and_replayer_return_scripted_and_record(self):
        from neograph_mcp.testing import fake_mcp_resource_fetcher

        seen: list = []
        fetcher, replayer = fake_mcp_resource_fetcher(
            resources={"mcp://crm/notes/D1": ("deal notes for D1", "text/plain")},
            replays={"get_deal": [{"type": "text", "text": "D1 dossier"}]},
            calls=seen,
        )

        content, mime = await fetcher("mcp://crm/notes/D1")
        assert (content, mime) == ("deal notes for D1", "text/plain")

        blocks = await replayer("get_deal", {"deal_id": "D1"})
        assert blocks == [{"type": "text", "text": "D1 dossier"}]

        # Both interactions recorded in order.
        assert seen == [("mcp://crm/notes/D1", {}), ("get_deal", {"deal_id": "D1"})]

    async def test_fetcher_raises_on_unscripted_uri(self):
        from neograph_mcp.testing import fake_mcp_resource_fetcher

        fetcher, _replayer = fake_mcp_resource_fetcher(resources={"mcp://crm/notes/D1": ("x", None)})
        with pytest.raises(KeyError, match="no scripted resource"):
            await fetcher("mcp://crm/unknown/Z9")
