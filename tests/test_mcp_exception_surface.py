"""MCP boundary exception surfacing — the bare-leaf invariant (neograph-2itlh).

Every ``neograph_mcp`` exit-boundary surfaces a single-leaf transport
``ExceptionGroup`` as its BARE leaf, so a consumer catching a specific type
(``RuntimeError`` / ``ValueError`` / their own guard) around an MCP tool call or
factory build gets that type — never an ``ExceptionGroup`` wrapper.

This file pins the FACTORY boundary (``_discover_tool_names`` build-time +
``_factory`` per-call ``get_tools()``): a consumer ``token_provider`` that
raises surfaces as its bare type through the REAL streamable-http transport. The
qslrx per-request ``httpx.Auth`` moved provider invocation INSIDE the
transport's anyio TaskGroup (``_TokenProviderAuth.async_auth_flow``), which
wraps the raise in an ``ExceptionGroup``; the factory's ``get_tools()``
boundary must unwrap on exit (like ``_session`` / ``_prompt`` / ``_run_context``
already do), not re-raise the group. This is the regression that broke the
ox-troubleshooting-demo ``test_mcp_factory`` token-provider tests.

The ``_resilient`` tool-call wrapper boundary — same invariant, different
boundary — has its own grouped-exception twin in
``tests/test_mcp_transport_resilience.py``.

Real transports, no protocol mocking — the shared FastMCP demo server echoes the
``Authorization`` header. Run with::

    uv run --extra dev --extra mcp-examples pytest tests/test_mcp_exception_surface.py
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_mcp_battery import _demo_http_server, requires_mcp


@requires_mcp
class TestTokenProviderErrorSurfacesBareAtFactory:
    """A consumer ``token_provider`` that raises surfaces as its BARE type at the
    factory's ``get_tools()`` boundary — never an ``ExceptionGroup`` wrapper.

    The qslrx regression: provider invocation moved into the per-request
    anyio-driven ``httpx.Auth``, so its raise is wrapped by the transport's
    TaskGroup. The factory must unwrap on exit so a consumer's
    ``except RuntimeError`` (or ``ValueError``, or their own guard) still catches
    it — the same bare-leaf contract ``_session`` / ``_prompt`` / ``_run_context``
    already uphold. A ``ValueError`` twin proves it is not RuntimeError-specific."""

    @pytest.mark.parametrize("leaf_type", [RuntimeError, ValueError], ids=["runtime_error", "value_error"])
    async def test_singular_factory_invocation_surfaces_bare_leaf(self, tmp_path, leaf_type: type):
        """The lazy ``mcp_tool_factory`` (no build-time connect) raises the bare
        leaf when the factory is INVOKED — the per-call ``get_tools()`` connect
        drives the per-request ``httpx.Auth`` → the provider raises inside the
        transport's anyio TaskGroup → the factory unwraps and re-raises bare."""
        from neograph_mcp import HttpServer, mcp_tool_factory

        def provider(configurable=None) -> str:
            raise leaf_type("no token provider wired")

        with _demo_http_server(tmp_path / "state.marker") as url:
            factory = mcp_tool_factory(
                "crm", HttpServer(url=url), tool_name="crm_search", token_provider=provider
            )
            with pytest.raises(leaf_type):
                await factory({"configurable": {}}, {})

    @pytest.mark.parametrize("leaf_type", [RuntimeError, ValueError], ids=["runtime_error", "value_error"])
    def test_plural_factory_build_surfaces_bare_leaf(self, tmp_path, leaf_type: type):
        """The eager ``mcp_tool_factories`` (build-time enumerate via
        ``_discover_tool_names``) raises the bare leaf at BUILD time — the
        enumeration connect drives the same per-request ``httpx.Auth`` and the
        same TaskGroup wrap; the build boundary must unwrap on exit."""
        from neograph_mcp import HttpServer, mcp_tool_factories

        def provider(configurable=None) -> str:
            raise leaf_type("no token provider wired")

        with _demo_http_server(tmp_path / "state.marker") as url:
            with pytest.raises(leaf_type):
                mcp_tool_factories({"crm": HttpServer(url=url)}, token_provider=provider, namespace=False)


# ── post-enter grouped-raise doubles (neograph-lcrwd) ─────────────────────────
#
# The four LATENT twins of the _resilient bug: boundaries reached only AFTER a
# successful connect (mcp_session.call / _ensure_listing) or inside a
# consumer-owned session (fetcher / replayer). A real-transport repro would need
# the provider to succeed at enter and fail mid-call; instead these doubles
# inject the exact grouped form the transport's anyio TaskGroup produces — the
# sanctioned post-enter harness from the neograph-lcrwd TDD note, mirroring the
# ``_grouped_tool`` double in test_mcp_transport_resilience.py.


class _GroupedRaiseSession:
    """A protocol-session double whose transport methods raise a single-leaf
    ``ExceptionGroup`` wrapping ``leaf_factory()`` — what the stdio /
    streamable-http anyio TaskGroup does to a failure raised inside the request."""

    def __init__(self, leaf_factory: Any) -> None:
        self._leaf_factory = leaf_factory

    def _raise(self) -> Any:
        raise ExceptionGroup("taskgroup", [self._leaf_factory()])

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        self._raise()

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        self._raise()

    async def read_resource(self, *args: Any, **kwargs: Any) -> Any:
        self._raise()


class _FakeSessionCM:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> Any:
        return self._session

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeClient:
    """Stands in for ``_client_for``'s MultiServerMCPClient: connect (enter)
    SUCCEEDS; only the transport methods fail — the post-enter scenario."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def session(self, server_key: str) -> _FakeSessionCM:
        return _FakeSessionCM(self._session)


@requires_mcp
class TestHeldSessionSurfacesBareLeafPostEnter:
    """``McpSession.call`` and ``tool_names()`` (via ``_ensure_listing``) surface
    a single-leaf grouped transport failure as its BARE leaf. Enter succeeds —
    the grouped raise happens on the in-session ``tools/call`` / ``tools/list``,
    the two ``_session.py`` boundaries the neograph-2itlh fix did not reach."""

    @pytest.mark.parametrize("leaf_type", [RuntimeError, ValueError], ids=["runtime_error", "value_error"])
    async def test_call_surfaces_bare_leaf(self, monkeypatch, leaf_type: type):
        import neograph_mcp._session as session_mod
        from neograph_mcp import HttpServer, mcp_session

        fake = _FakeClient(_GroupedRaiseSession(lambda: leaf_type("boom")))
        monkeypatch.setattr(session_mod, "_client_for", lambda *a, **k: fake)

        async with mcp_session("crm", HttpServer(url="http://unused.invalid/mcp")) as s:
            with pytest.raises(leaf_type):
                await s.call("crm_search", {"query": "x"})

    @pytest.mark.parametrize("leaf_type", [RuntimeError, ValueError], ids=["runtime_error", "value_error"])
    async def test_tool_names_surfaces_bare_leaf(self, monkeypatch, leaf_type: type):
        import neograph_mcp._session as session_mod
        from neograph_mcp import HttpServer, mcp_session

        fake = _FakeClient(_GroupedRaiseSession(lambda: leaf_type("boom")))
        monkeypatch.setattr(session_mod, "_client_for", lambda *a, **k: fake)

        async with mcp_session("crm", HttpServer(url="http://unused.invalid/mcp")) as s:
            with pytest.raises(leaf_type):
                await s.tool_names()

    async def test_call_grouped_cancelled_error_stays_grouped(self, monkeypatch):
        """CancelledError is EXEMPT from the bare-leaf unwrap (the neograph-2itlh
        decision ``_resilient`` already pins): cooperative cancellation propagates
        in its original grouped form, never re-raised ``from None``."""
        import asyncio

        import neograph_mcp._session as session_mod
        from neograph_mcp import HttpServer, mcp_session

        class _CancelledGroupSession(_GroupedRaiseSession):
            def _raise(self) -> Any:
                raise BaseExceptionGroup("taskgroup", [asyncio.CancelledError()])

        fake = _FakeClient(_CancelledGroupSession(lambda: None))
        monkeypatch.setattr(session_mod, "_client_for", lambda *a, **k: fake)

        async with mcp_session("crm", HttpServer(url="http://unused.invalid/mcp")) as s:
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await s.call("crm_search", {})
        assert isinstance(exc_info.value.exceptions[0], asyncio.CancelledError)


@requires_mcp
class TestResourceFetcherAndReplayerSurfaceBareLeaf:
    """The ``mcp_resource_fetcher`` pair: a GROUPED ``McpError`` from
    ``read_resource`` misses the fetcher's bare ``except McpError`` (so the
    -32002 expiry -> replay path never engages), and the replayer's
    ``call_tool`` has no unwrap at all."""

    async def test_fetcher_surfaces_grouped_mcp_error_as_bare(self, monkeypatch):
        from mcp.shared.exceptions import McpError
        from mcp.types import ErrorData

        import neograph_mcp._client as client_mod
        from neograph_mcp import HttpServer, mcp_resource_fetcher

        def _expired() -> McpError:
            return McpError(ErrorData(code=-32002, message="resource expired"))

        fake = _FakeClient(_GroupedRaiseSession(_expired))
        monkeypatch.setattr(client_mod, "_client_for", lambda *a, **k: fake)

        fetcher, _replayer = mcp_resource_fetcher({"crm": HttpServer(url="http://unused.invalid/mcp")})
        with pytest.raises(McpError):
            await fetcher("mcp://crm/deals/42")

    @pytest.mark.parametrize("leaf_type", [RuntimeError, ValueError], ids=["runtime_error", "value_error"])
    async def test_replayer_surfaces_bare_leaf(self, monkeypatch, leaf_type: type):
        import neograph_mcp._client as client_mod
        from neograph_mcp import HttpServer, mcp_resource_fetcher

        fake = _FakeClient(_GroupedRaiseSession(lambda: leaf_type("boom")))
        monkeypatch.setattr(client_mod, "_client_for", lambda *a, **k: fake)

        _fetcher, replayer = mcp_resource_fetcher({"crm": HttpServer(url="http://unused.invalid/mcp")})
        with pytest.raises(leaf_type):
            await replayer("crm_search", {"query": "x"})
