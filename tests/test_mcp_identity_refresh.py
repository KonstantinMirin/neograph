"""Mid-run identity refresh under the RUN_ID tool cache (neograph-qslrx) — TDD RED.

The bug: the tool instance built per (RUN_ID, node) bakes in the bearer minted
from ``token_provider`` at BUILD time (first superstep) and reuses it for every
subsequent ReAct turn of the run — a run whose tool phase outlives the IdP
access-token lifespan sends a STALE token and the server denies (the
ox-troubleshooting-demo Keycloak incident). Option D unifies the static
``token_provider`` path onto the same per-request ``httpx.Auth`` mechanism the
OAuth ``HttpServer.auth`` path already uses, so the provider is re-invoked per
request and a refreshing provider takes effect mid-run.

Four pins, mirroring the qslrx design manifest (item 12):

1. http twin (RED today): a counter ``token_provider`` across two tool-calling
   supersteps under the RUN_ID cache — turn 2 must carry a FRESH token.
2. stdio twin (RED today): same freeze on the stdio tool-argument channel.
3. static-string regression (GREEN before and after): a constant provider keeps
   sending the same value — unifying the mechanism changes nothing observable.
4. OAuth-across-supersteps pin (GREEN today — the load-bearing assumption):
   an ``httpx.Auth`` on ``HttpServer.auth`` runs per request THROUGH the cached
   tool, proving httpx drives auth_flow per tools/call against the adapter's
   client under the agent loop, not just via mcp_session.

Real transports, no protocol mocking — the shared FastMCP demo server echoes the
``Authorization`` header under ``bearer_identity`` (http) and the ``token`` tool
argument under ``acting_as`` (stdio). Run with::

    uv run --extra dev --extra mcp-examples pytest tests/test_mcp_identity_refresh.py
"""

from __future__ import annotations

import re

from tests.test_mcp_battery import (
    _build_agent_construct,
    _demo_http_server,
    _demo_stdio_server,
    requires_mcp,
)


def _two_round_react_fake(tool_name: str):
    """ReActFake scripting TWO tool-calling turns then a final parse — the
    minimal shape where the second superstep hits the RUN_ID tool cache."""
    from tests.fakes import ReActFake
    from tests.schemas import Claims

    return ReActFake(
        tool_calls=[
            [{"name": tool_name, "args": {"query": "acme"}, "id": "c1"}],
            [{"name": tool_name, "args": {"query": "globex"}, "id": "c2"}],
            [],
        ],
        final=lambda m: m(items=["done"]),
        output_model=Claims,
    )


def _counting_provider(prefix: str):
    """A token_provider whose every invocation mints a NEW token — the
    deterministic stand-in for 'the IdP token expired and was re-fetched'."""
    minted: list[str] = []

    def provider(configurable) -> str:
        minted.append(f"{prefix}-{len(minted) + 1}")
        return minted[-1]

    return provider, minted


async def _run_two_superstep_agent(tool_name: str, factories: dict) -> list:
    """Compile a one-agent-node construct bound to ``tool_name`` and drive it
    through ``neograph.arun`` (mints RUN_ID -> the per-run tool cache engages).
    Returns the run's tool_log (one entry per superstep tool call)."""
    import neograph
    from neograph import compile
    from tests.fakes import build_test_compile_kwargs, configure_fake_llm
    from tests.schemas import Claims

    construct = _build_agent_construct(tool_name)
    llm_kw = configure_fake_llm(lambda tier: _two_round_react_fake(tool_name))
    graph = compile(construct, tool_factories=factories, **build_test_compile_kwargs(), **llm_kw)
    result = await neograph.arun(graph, input={"query": "acme"})

    assert result["scan_result"] == Claims(items=["done"])
    tool_log = result["scan_tool_log"]
    assert len(tool_log) == 2, f"expected two tool-calling supersteps, got {len(tool_log)}"
    return tool_log


def _token_of(tool_log_entry, pattern: str) -> str:
    """Extract the identity the SERVER echoed for one tool call."""
    match = re.search(pattern, tool_log_entry.result)
    assert match, f"no identity matching {pattern!r} in tool result: {tool_log_entry.result[:400]}"
    return match.group(0)


@requires_mcp
class TestTokenProviderMidRunRefresh:
    """The static token_provider identity must be re-resolvable MID-RUN."""

    async def test_http_bearer_is_fresh_on_second_superstep_under_run_cache(self, tmp_path):
        """(1, RED) http twin: with a counter token_provider, the SECOND ReAct
        superstep's tool call must reach the server with a FRESH bearer — not
        the one frozen into the cached tool at first-superstep build time."""
        from neograph_mcp import HttpServer, mcp_tool_factories

        provider, minted = _counting_provider("http-tok")
        with _demo_http_server(tmp_path / "state.marker") as url:
            factories = mcp_tool_factories(
                {"crm": HttpServer(url=url)},
                token_provider=provider,
                namespace=False,
            )
            tool_log = await _run_two_superstep_agent(
                "crm_search", {"crm_search": factories["crm_search"]}
            )

        first = _token_of(tool_log[0], r"http-tok-\d+")
        second = _token_of(tool_log[1], r"http-tok-\d+")
        assert len(minted) >= 2, (
            f"token_provider was invoked {len(minted)} time(s) across two supersteps — "
            "the run-cached tool froze the build-time mint"
        )
        assert second != first, (
            f"second superstep reused the frozen bearer {first!r} — a run outliving "
            "the IdP token lifespan would send a stale token (neograph-qslrx)"
        )

    async def test_stdio_token_arg_is_fresh_on_second_superstep_under_run_cache(self):
        """(2, RED) stdio twin: identity rides as a tool ARGUMENT; the cached
        tool's injection wrapper must re-resolve the provider per call rather
        than closing over the build-time value."""
        from neograph_mcp import mcp_tool_factories

        provider, minted = _counting_provider("stdio-tok")
        factories = mcp_tool_factories(
            {"crm": _demo_stdio_server()},
            token_provider=provider,
            namespace=False,
        )
        tool_log = await _run_two_superstep_agent(
            "crm_search", {"crm_search": factories["crm_search"]}
        )

        first = _token_of(tool_log[0], r"stdio-tok-\d+")
        second = _token_of(tool_log[1], r"stdio-tok-\d+")
        assert len(minted) >= 2, (
            f"token_provider was invoked {len(minted)} time(s) across two supersteps — "
            "the stdio injection wrapper froze the build-time token"
        )
        assert second != first, (
            f"second superstep reused the frozen stdio token {first!r} (neograph-qslrx)"
        )

    async def test_static_string_provider_keeps_sending_the_same_value(self, tmp_path):
        """(3, regression — green before and after) a constant provider still
        sends the SAME identity on every superstep: per-request re-resolution
        makes refresh POSSIBLE; whether the token changes stays the provider's
        business."""
        from neograph_mcp import HttpServer, mcp_tool_factories

        with _demo_http_server(tmp_path / "state.marker") as url:
            factories = mcp_tool_factories(
                {"crm": HttpServer(url=url)},
                token_provider=lambda configurable: "operator-static",
                namespace=False,
            )
            tool_log = await _run_two_superstep_agent(
                "crm_search", {"crm_search": factories["crm_search"]}
            )

        assert _token_of(tool_log[0], r"operator-static") == "operator-static"
        assert _token_of(tool_log[1], r"operator-static") == "operator-static"

    async def test_http_session_reresolves_token_provider_per_call(self, tmp_path):
        """(5, session path) within ONE held ``mcp_session`` connection — the
        long-lived shape where a connect-time bearer previously froze for the
        session's whole lifetime — each call carries a freshly-resolved token:
        the provider rides the connection as a per-request httpx.Auth."""
        from neograph_mcp import HttpServer, mcp_session

        provider, minted = _counting_provider("sess-tok")
        with _demo_http_server(tmp_path / "state.marker") as url:
            async with mcp_session("crm", HttpServer(url=url), token_provider=provider) as session:
                first = (await session.call("crm_search", {"query": "acme"})).structured["bearer_identity"]
                second = (await session.call("crm_search", {"query": "globex"})).structured["bearer_identity"]

        assert first and first.startswith("sess-tok-"), first
        assert second and second.startswith("sess-tok-"), second
        assert second != first, (
            f"second call on the same session reused the frozen bearer {first!r} (neograph-qslrx)"
        )
        assert len(minted) >= 2

    async def test_stdio_session_reresolves_token_provider_per_call(self):
        """(6, session path — stdio twin, neograph-hs3mr) within ONE held stdio
        ``mcp_session`` connection, each ``call()`` must carry a freshly-resolved
        token as the ``stdio_token_arg`` argument — not the value minted once at
        ``__aenter__``. The http side of the SAME surface already refreshes
        per request; a transport-dependent freeze here is the last mint-once
        fork left by neograph-qslrx."""
        from neograph_mcp import mcp_session

        provider, minted = _counting_provider("sess-stdio")
        async with mcp_session("crm", _demo_stdio_server(), token_provider=provider) as session:
            first = (await session.call("crm_search", {"query": "acme"})).structured["acting_as"]
            second = (await session.call("crm_search", {"query": "globex"})).structured["acting_as"]

        assert first and first.startswith("sess-stdio-"), first
        assert second and second.startswith("sess-stdio-"), second
        assert second != first, (
            f"second stdio call on the same session reused the frozen token {first!r} — "
            "identity must be per-call fresh on EVERY surface and transport (neograph-hs3mr)"
        )
        assert len(minted) >= 2

    async def test_stdio_session_static_provider_keeps_sending_the_same_value(self):
        """(7, regression) a constant token_provider over a stdio session still
        sends the SAME identity on every ``call()`` — per-call re-resolution
        makes refresh POSSIBLE; a constant provider is how a consumer PINS one
        principal for the whole session (the framework never freezes for you)."""
        from neograph_mcp import mcp_session

        async with mcp_session(
            "crm", _demo_stdio_server(), token_provider=lambda configurable: "operator-static"
        ) as session:
            first = (await session.call("crm_search", {"query": "acme"})).structured["acting_as"]
            second = (await session.call("crm_search", {"query": "globex"})).structured["acting_as"]

        assert first == "operator-static"
        assert second == "operator-static"

    async def test_httpx_auth_runs_per_request_through_the_cached_tool(self, tmp_path):
        """(4, load-bearing pin — green today) an ``HttpServer.auth`` httpx.Auth
        is driven PER REQUEST through the RUN_ID-cached tool across supersteps:
        each superstep's tools/call carries a token minted by that request's
        auth_flow, so the OAuth path never freezes mid-run. This is the
        mechanism Option D collapses token_provider onto — if this ever fails,
        per-request identity through the cache is broken at the transport layer."""
        import httpx

        from neograph_mcp import HttpServer, mcp_tool_factories

        class CounterAuth(httpx.Auth):
            def __init__(self) -> None:
                self.minted: list[str] = []

            def auth_flow(self, request):
                self.minted.append(f"auth-tok-{len(self.minted) + 1}")
                request.headers["Authorization"] = f"Bearer {self.minted[-1]}"
                yield request

        auth = CounterAuth()
        with _demo_http_server(tmp_path / "state.marker") as url:
            factories = mcp_tool_factories(
                {"crm": HttpServer(url=url, auth=auth)},
                namespace=False,
            )
            tool_log = await _run_two_superstep_agent(
                "crm_search", {"crm_search": factories["crm_search"]}
            )

        first = _token_of(tool_log[0], r"auth-tok-\d+")
        second = _token_of(tool_log[1], r"auth-tok-\d+")
        assert second != first, (
            "httpx.Auth was not re-driven for the second superstep's tools/call — "
            "the per-request identity mechanism itself is frozen under the run cache"
        )
