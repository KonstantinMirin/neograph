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
