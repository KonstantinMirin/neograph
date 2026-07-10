"""First-class OAuth on the HTTP transport (neograph-27h3s) — TDD RED.

Core invariant under test (per the 27h3s MAINTAINER DECISION): a production
consumer gets refresh/expiry-handling OAuth identity by attaching an SDK-built
``httpx.Auth`` to the transport via ``HttpServer(auth=client_credentials_auth(...))``;
neograph_mcp only CARRIES that Auth from config to the adapter — it never parses,
stores, or refreshes the token itself, so the token stays out of state, the
checkpoint, and the fingerprint.

Every test below is RED today: ``client_credentials_auth`` is not exported from
``neograph_mcp`` and the frozen ``HttpServer`` dataclass has no ``auth`` field.
Imports of the missing surface live INSIDE test bodies (the TestMcpSession /
TestMcpProgressStreaming precedent in tests/test_mcp_battery.py), so collection
stays green and each test fails at runtime with the ImportError that is the
correct red for a not-yet-implemented public surface.

The fixture (tests/_mcp_oauth_demo_server.py) is REAL end-to-end: a FastMCP
streamable-http server whose /mcp endpoint enforces bearer tokens, plus the
OAuth authorization-server half the SDK provider actually exercises — RFC-9728
protected-resource metadata, RFC-8414 authorization-server metadata, and a
client_credentials /token endpoint. No protocol mocking anywhere; the fixture
was validated green against the raw SDK ``ClientCredentialsOAuthProvider``
before these tests were written, so the ONLY thing failing here is the missing
neograph_mcp surface. Refresh is forced deterministically (a control route
revokes the current token -> exactly one 401), never by wall-clock expiry.

Run with::

    uv run --extra dev --extra mcp-examples pytest tests/test_mcp_oauth.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OAUTH_SERVER = REPO_ROOT / "tests" / "_mcp_oauth_demo_server.py"

_HAS_MCP = bool(importlib.util.find_spec("mcp")) and bool(importlib.util.find_spec("langchain_mcp_adapters"))
requires_mcp = pytest.mark.skipif(not _HAS_MCP, reason="requires the mcp extra (mcp + langchain-mcp-adapters)")

# Must match tests/_mcp_oauth_demo_server.py (not imported: that module reads
# NEOGRAPH_MCP_OAUTH_PORT from the environment at import time — subprocess-only).
CLIENT_ID = "neo-demo-client"
CLIENT_SECRET = "neo-demo-secret-XyZzY"
TOKEN_PREFIX = "neo-oauth-access-"

LOSING_STATIC_BEARER = "static-bearer-should-lose"


# ── local OAuth-guarded server harness (the _demo_http_server pattern) ────────


def _free_port() -> int:
    """Reserve an ephemeral localhost port (bind-then-close; standard harness race)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_listening(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f"oauth demo server never listened on {host}:{port}") from last_error


@contextlib.contextmanager
def _oauth_server():
    """Launch the OAuth-guarded FastMCP server subprocess; yield its base URL."""
    port = _free_port()
    env = {**os.environ, "NEOGRAPH_MCP_OAUTH_PORT": str(port)}
    proc = subprocess.Popen(
        [sys.executable, str(OAUTH_SERVER)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
    )
    try:
        _wait_for_listening("127.0.0.1", port)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _oauth_http_spec(base: str):
    """HttpServer with the battery's OAuth httpx.Auth attached — the surface
    under test. RED: neither ``client_credentials_auth`` nor ``HttpServer.auth``
    exists yet."""
    from neograph_mcp import HttpServer, client_credentials_auth  # RED: not yet implemented

    return HttpServer(
        url=f"{base}/mcp",
        auth=client_credentials_auth(
            server_url=f"{base}/mcp",
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        ),
    )


async def _minted_tokens(base: str) -> dict:
    import httpx

    async with httpx.AsyncClient() as client:
        return (await client.get(f"{base}/control/tokens")).json()


async def _revoke_current(base: str) -> str:
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base}/control/revoke-current")
    assert response.status_code == 200, f"revoke-current failed: {response.text}"
    return response.json()["revoked"]


def _kb_agent_construct():
    """One agent node binding the fixture's ``kb_lookup`` (which never echoes the
    bearer), so any token bytes found in state/checkpoint are a genuine leak."""
    from neograph import Construct, Node
    from neograph.tool import Tool, ToolInteraction
    from tests.schemas import Claims

    return Construct(
        "kb_pipeline",
        nodes=[
            Node(
                "scan",
                mode="agent",
                outputs={"result": Claims, "tool_log": list[ToolInteraction]},
                model="fast",
                prompt="test/scan",
                tools=[Tool("kb_lookup", budget=0)],
            ),
        ],
    )


def _kb_react_fake():
    from tests.fakes import ReActFake
    from tests.schemas import Claims

    return ReActFake(
        tool_calls=[
            [{"name": "kb_lookup", "args": {"topic": "pricing"}, "id": "c1"}],
            [],
        ],
        final=lambda m: m(items=["done"]),
        output_model=Claims,
    )


@requires_mcp
class TestHttpServerOAuth:
    """First-class OAuth (client_credentials) on HttpServer — neograph-27h3s.

    Mirrors TestHttpServerSmoke's real-server discipline: real FastMCP server,
    real SDK OAuth flow, real ``await`` — no MCP or OAuth mocking. All RED today
    (missing ``client_credentials_auth`` export / ``HttpServer.auth`` field).
    """

    async def test_oauth_guarded_tool_call_succeeds_via_client_credentials_auth(self):
        """(1) An ``HttpServer(auth=client_credentials_auth(...))`` bound through
        ``mcp_tool_factories`` authenticates against the OAuth-GUARDED endpoint:
        the tool call succeeds and the server observably received a token IT
        minted through the client-credentials flow (echoed by ``whoami``).
        Without the Auth this endpoint 401s everything, so success is proof."""
        from neograph_mcp import mcp_tool_factories

        with _oauth_server() as base:
            factories = mcp_tool_factories({"crm": _oauth_http_spec(base)}, namespace=False)
            tool = await factories["whoami"]({"configurable": {}}, None)
            result = json.loads((await tool.ainvoke({}))[0]["text"])

            identity = result["bearer_identity"]
            assert identity is not None, "guarded endpoint saw NO bearer token — the Auth never engaged"
            assert identity.startswith(TOKEN_PREFIX), f"unexpected identity: {identity!r}"
            tokens = await _minted_tokens(base)
            assert identity in tokens["issued"], (
                f"identity {identity!r} was not minted by the fixture's token endpoint: {tokens}"
            )

    async def test_token_refresh_on_expiry_succeeds_on_same_session_without_rebuild(self):
        """(2) Refresh-on-expiry WITHOUT rebuilding the client/session: inside a
        single ``mcp_session`` connection, call once (token 1), revoke that token
        server-side (the deterministic 'expiry' -> the next request 401s), call
        again on the SAME session — the httpx.Auth re-runs the flow in place and
        the second call succeeds with a NEW minted token. Exactly two tokens are
        minted overall: one initial + one refresh, no per-call or reconnect churn."""
        from neograph_mcp import mcp_session

        with _oauth_server() as base:
            async with mcp_session("crm", _oauth_http_spec(base)) as session:
                first = json.loads((await session.call("whoami", {})).text)["bearer_identity"]
                assert first is not None and first.startswith(TOKEN_PREFIX), first

                revoked = await _revoke_current(base)
                assert revoked == first, "the fixture must revoke exactly the in-use token"

                # SAME session object, no re-entry, no rebuild: the next call must
                # survive the forced 401 via the Auth's in-client refresh.
                second = json.loads((await session.call("whoami", {})).text)["bearer_identity"]

            assert second is not None and second.startswith(TOKEN_PREFIX), second
            assert second != first, "second call reused the revoked token — no refresh happened"
            tokens = await _minted_tokens(base)
            assert tokens["issued"] == [first, second], (
                f"expected exactly [initial, refreshed] mints, got {tokens['issued']}"
            )

    async def test_client_secret_and_minted_token_never_enter_state_or_checkpoint(self, tmp_path):
        """(3) The identity NEVER lands on the state bus: a full agent run over the
        OAuth-guarded server with a real sqlite checkpointer leaves no trace of the
        client_secret OR any minted access token in the final state, any checkpoint
        snapshot, or the raw checkpoint DB bytes. The bound tool (``kb_lookup``)
        never echoes the bearer, so any hit is a genuine leak. Non-vacuity: the
        run must have minted a token and produced a real tool result."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        import neograph
        from neograph import compile
        from neograph_mcp import mcp_tool_factory
        from tests.fakes import build_test_compile_kwargs, configure_fake_llm
        from tests.schemas import Claims

        db = str(tmp_path / "oauth_checkpoints.sqlite")
        with _oauth_server() as base:
            spec = _oauth_http_spec(base)
            async with AsyncSqliteSaver.from_conn_string(db) as saver:
                factory = mcp_tool_factory("crm", spec, tool_name="kb_lookup")
                llm_kw = configure_fake_llm(lambda tier: _kb_react_fake())
                graph = compile(
                    _kb_agent_construct(),
                    tool_factories={"kb_lookup": factory},
                    checkpointer=saver,
                    **build_test_compile_kwargs(),
                    **llm_kw,
                )
                config = {"configurable": {"thread_id": "oauth-iso"}}
                final = await neograph.arun(graph, input={"query": "pricing"}, config=config)

                # Non-vacuity: the OAuth flow ran and the guarded tool answered.
                tokens = await _minted_tokens(base)
                assert tokens["issued"], "no token was minted — the isolation assertions would be vacuous"
                assert final["scan_result"] == Claims(items=["done"])
                assert final["scan_tool_log"], "the guarded tool never ran"
                assert "pricing" in final["scan_tool_log"][0].result

                # Final state carries neither the secret nor any minted token.
                state_dump = repr(final)
                assert CLIENT_SECRET not in state_dump, "client_secret leaked into run state"
                assert TOKEN_PREFIX not in state_dump, "a minted access token leaked into run state"

                # Every checkpoint snapshot, observed through the public
                # state-history surface over the REAL sqlite saver, is clean.
                snapshots = [s async for s in graph.aget_state_history(config)]
                assert snapshots, "expected checkpoints from the sqlite saver"
                for snap in snapshots:
                    dump = repr(snap.values)
                    assert CLIENT_SECRET not in dump, f"client_secret leaked into a checkpoint: {dump[:400]}"
                    assert TOKEN_PREFIX not in dump, f"an access token leaked into a checkpoint: {dump[:400]}"

        # Belt-and-braces: the raw checkpoint DB bytes never carry either secret.
        raw = Path(db).read_bytes()
        assert CLIENT_SECRET.encode() not in raw, "client_secret leaked into the raw checkpoint DB"
        assert TOKEN_PREFIX.encode() not in raw, "a minted access token leaked into the raw checkpoint DB"

    async def test_httpx_auth_wins_when_bearer_token_provider_also_set(self):
        """(4) Precedence (MAINTAINER DECISION): when BOTH ``HttpServer.auth`` and a
        bearer ``token_provider`` are configured, the httpx.Auth wins — the
        identity that reaches the guarded server is the OAuth-minted token, never
        the provider's static bearer (and there is no double-Authorization
        confusion: the server sees exactly one usable identity)."""
        from neograph_mcp import mcp_tool_factories

        with _oauth_server() as base:
            factories = mcp_tool_factories(
                {"crm": _oauth_http_spec(base)},
                token_provider=lambda configurable: LOSING_STATIC_BEARER,
                namespace=False,
            )
            tool = await factories["whoami"]({"configurable": {}}, None)
            result = json.loads((await tool.ainvoke({}))[0]["text"])

            identity = result["bearer_identity"]
            assert identity != LOSING_STATIC_BEARER, (
                "the static bearer from token_provider reached the server — httpx.Auth must win"
            )
            assert identity is not None and identity.startswith(TOKEN_PREFIX), (
                f"expected the OAuth-minted identity to win, got {identity!r}"
            )
            tokens = await _minted_tokens(base)
            assert identity in tokens["issued"]
