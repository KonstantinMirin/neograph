"""First-class production OAuth for ``HttpServer`` — SDK-wrapping battery helpers.

``token_provider`` returns a bare bearer string minted once per superstep and
stamped into a static ``Authorization`` header — it structurally cannot do
401-triggered refresh-without-reconnect. Real OAuth identity lives on the
TRANSPORT: an ``httpx.Auth`` attached to ``HttpServer(auth=...)`` rides the
adapter's persistent httpx client (the ``_connection`` choke point wires it for
all three surfaces — tool factories, resource fetcher, ``mcp_session``) and
handles token exchange, expiry, and refresh on its own.

These helpers WRAP the MCP SDK's shipped providers — token exchange/refresh is
never re-implemented here. The SDK providers live in
``mcp.client.auth.extensions.client_credentials`` (NOT re-exported at
``mcp.client.auth`` level in mcp 1.28.x — import from the submodule).

neograph_mcp only CARRIES the Auth from config to the adapter: it never parses,
stores, or refreshes the token, and neither the client secret nor a minted
access token ever enters run state, a checkpoint, or the schema fingerprint
(the Auth object lives inside the transport, outside the state bus entirely).

The three flows:

- **client-credentials** (machine-to-machine): :func:`client_credentials_auth`
  — the production default for a service consumer.
- **private-key JWT / RFC 7523** (on-behalf-of, key-based client auth): wrap
  ``PrivateKeyJWTOAuthProvider`` / ``RFC7523OAuthClientProvider`` from the same
  SDK module with an :class:`_InMemoryTokenStorage`, exactly as
  :func:`client_credentials_auth` does — the constructor takes the signing-key
  parameters instead of a client secret.
- **interactive authorization-code**: the SDK's base ``OAuthClientProvider``;
  needs a redirect handler, so it stays a direct-SDK affair (no battery sugar).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only; httpx rides in with the mcp extra
    import httpx


class _InMemoryTokenStorage:
    """The SDK's ``TokenStorage`` protocol, held in memory for one Auth's lifetime.

    Storage only — token EXCHANGE and REFRESH stay in the SDK provider. In-memory
    is deliberate: the token's lifetime is the Auth object's (== the consumer's
    server spec), it is never persisted, and so can never leak into a checkpoint.
    """

    def __init__(self) -> None:
        self._tokens: Any = None
        self._client_info: Any = None

    async def get_tokens(self) -> Any:
        return self._tokens

    async def set_tokens(self, tokens: Any) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> Any:
        return self._client_info

    async def set_client_info(self, client_info: Any) -> None:
        self._client_info = client_info


def client_credentials_auth(
    server_url: str,
    client_id: str,
    client_secret: str,
    *,
    scopes: str | None = None,
    auth_method: Literal["client_secret_basic", "client_secret_post"] = "client_secret_basic",
) -> httpx.Auth:
    """An ``httpx.Auth`` doing the OAuth 2.1 client-credentials flow for
    ``HttpServer(auth=...)`` — token exchange, expiry, and refresh handled by
    the SDK provider, free of any hand-rolled OAuth.

    ``server_url`` is the MCP server the token is minted FOR (the provider
    discovers the authorization server via RFC 8414 / protected-resource
    metadata from there). ``scopes`` is a space-separated scope string;
    ``auth_method`` picks how the client authenticates at the token endpoint.

    The returned Auth attaches to the transport's persistent httpx client, so a
    401 mid-run triggers a token refresh WITHOUT rebuilding the client or the
    MCP session. Build it once next to the ``HttpServer`` spec and share it —
    each call builds a fresh in-memory token store, so two Auths do not share
    tokens.
    """
    from mcp.client.auth.extensions.client_credentials import ClientCredentialsOAuthProvider

    return ClientCredentialsOAuthProvider(
        server_url=server_url,
        storage=_InMemoryTokenStorage(),
        client_id=client_id,
        client_secret=client_secret,
        token_endpoint_auth_method=auth_method,
        scopes=scopes,
    )
