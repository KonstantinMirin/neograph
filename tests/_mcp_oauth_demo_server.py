"""OAuth-guarded FastMCP streamable-http demo server (test infrastructure).

The fixture for the neograph-27h3s regression tests (tests/test_mcp_oauth.py):
a REAL client-credentials OAuth 2.1 authorization server + a bearer-guarded MCP
endpoint in one process, so the SDK's ``ClientCredentialsOAuthProvider`` can run
its genuine flow — no protocol mocking anywhere:

  401 on /mcp
    -> RFC-9728 protected-resource metadata discovery
         (/.well-known/oauth-protected-resource[/mcp])
    -> RFC-8414 authorization-server metadata discovery
         (/.well-known/oauth-authorization-server)
    -> client_credentials token exchange at /token (client_secret_basic or _post)
    -> retried request with the minted Bearer token succeeds.

Per the architect review of 27h3s, serving ONLY a token endpoint is not enough:
the SDK provider discovers the token endpoint via the metadata documents, so the
fixture serves BOTH discovery documents AND the token endpoint.

Determinism (no timing games): minted tokens are ``neo-oauth-access-<n>`` with
``expires_in`` far in the future; the refresh-on-expiry leg is forced by the
unauthenticated control route ``POST /control/revoke-current``, which invalidates
the newest issued token so the NEXT /mcp request 401s exactly once and the
httpx.Auth must re-run the flow inside the SAME httpx client / MCP session.

Tools:
  - ``whoami()``            -> echoes the received bearer token (identity probe)
  - ``kb_lookup(topic)``    -> plain payload, NO bearer echo (state-isolation leg)

Control routes (unauthenticated, test-only):
  - ``POST /control/revoke-current`` -> revoke the newest issued token
  - ``GET  /control/tokens``         -> {"issued": [...], "revoked": [...]}

Env:
  NEOGRAPH_MCP_OAUTH_PORT  (required) — 127.0.0.1 port to bind
"""

from __future__ import annotations

import base64
import contextlib
import os
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

CLIENT_ID = "neo-demo-client"
CLIENT_SECRET = "neo-demo-secret-XyZzY"  # distinctive: byte-scanned by the tests
TOKEN_PREFIX = "neo-oauth-access-"  # distinctive: byte-scanned by the tests

_PORT = int(os.environ["NEOGRAPH_MCP_OAUTH_PORT"])
_BASE = f"http://127.0.0.1:{_PORT}"

_KB = {"pricing": "Volume discounts apply above 100 seats."}

# Token bookkeeping (single-process, module-level — the fixture IS the AS).
_issued: list[str] = []
_revoked: set[str] = set()


mcp = FastMCP("neograph-oauth-demo")


def _bearer_identity() -> str | None:
    """The incoming ``Authorization`` bearer value, or None (mirrors the shared
    demo server's auth-echo helper)."""
    request = mcp.get_context().request_context.request
    if request is None:
        return None
    auth = request.headers.get("authorization")
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth[len("bearer ") :]
    return auth


@mcp.tool()
def whoami() -> dict[str, Any]:
    """Identity probe: echoes the bearer token the guarded endpoint received."""
    return {"bearer_identity": _bearer_identity()}


@mcp.tool()
def kb_lookup(topic: str) -> dict[str, Any]:
    """Plain payload with NO bearer echo — used by the token-never-in-state leg,
    so any token bytes found in state/checkpoint are a genuine leak, not an echo."""
    return {"topic": topic, "article": _KB.get(topic, "No article found.")}


# ── OAuth authorization-server half ───────────────────────────────────────────


def _check_client_auth(request: Request, form: dict[str, str]) -> bool:
    """client_secret_basic (Authorization: Basic b64(id:secret)) or client_secret_post."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode()
        except Exception:
            return False
        return decoded == f"{CLIENT_ID}:{CLIENT_SECRET}"
    return form.get("client_id") == CLIENT_ID and form.get("client_secret") == CLIENT_SECRET


async def token_endpoint(request: Request) -> JSONResponse:
    form = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    if form.get("grant_type") != "client_credentials":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
    if not _check_client_auth(request, form):
        return JSONResponse({"error": "invalid_client"}, status_code=401)
    token = f"{TOKEN_PREFIX}{len(_issued) + 1}"
    _issued.append(token)
    return JSONResponse({"access_token": token, "token_type": "Bearer", "expires_in": 3600})


async def protected_resource_metadata(request: Request) -> JSONResponse:
    return JSONResponse({"resource": f"{_BASE}/mcp", "authorization_servers": [_BASE]})


async def authorization_server_metadata(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "issuer": _BASE,
            "authorization_endpoint": f"{_BASE}/authorize",  # required by the model; unused
            "token_endpoint": f"{_BASE}/token",
            "grant_types_supported": ["client_credentials"],
            "response_types_supported": ["code"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        }
    )


# ── Control routes (test-only, unauthenticated) ───────────────────────────────


async def revoke_current(request: Request) -> JSONResponse:
    if not _issued:
        return JSONResponse({"revoked": None}, status_code=409)
    token = _issued[-1]
    _revoked.add(token)
    return JSONResponse({"revoked": token})


async def list_tokens(request: Request) -> JSONResponse:
    return JSONResponse({"issued": list(_issued), "revoked": sorted(_revoked)})


# ── Bearer guard around the MCP endpoint ──────────────────────────────────────


def _guarded(app: Any) -> Any:
    """Pure ASGI wrapper: /mcp requires a currently-valid issued bearer token.

    401 carries a WWW-Authenticate pointing at the protected-resource metadata
    (RFC 9728), which the SDK provider uses to seed discovery.
    """

    async def guard(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            auth = headers.get("authorization", "")
            token = auth[7:] if auth.lower().startswith("bearer ") else None
            if token not in _issued or token in _revoked:
                www = f'Bearer resource_metadata="{_BASE}/.well-known/oauth-protected-resource/mcp"'
                response = JSONResponse(
                    {"error": "invalid_token"}, status_code=401, headers={"WWW-Authenticate": www}
                )
                await response(scope, receive, send)
                return
        await app(scope, receive, send)

    return guard


def main() -> None:
    mcp_app = mcp.streamable_http_app()  # serves /mcp

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with mcp.session_manager.run():
            yield

    outer = Starlette(
        routes=[
            Route("/.well-known/oauth-protected-resource/mcp", protected_resource_metadata),
            Route("/.well-known/oauth-protected-resource", protected_resource_metadata),
            Route("/.well-known/oauth-authorization-server", authorization_server_metadata),
            Route("/token", token_endpoint, methods=["POST"]),
            Route("/control/revoke-current", revoke_current, methods=["POST"]),
            Route("/control/tokens", list_tokens),
            Mount("/", app=_guarded(mcp_app)),
        ],
        lifespan=lifespan,
    )
    uvicorn.run(outer, host="127.0.0.1", port=_PORT, log_level="error")


if __name__ == "__main__":
    main()
