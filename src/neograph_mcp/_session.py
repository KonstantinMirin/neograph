"""Consumer-side MCP SESSION — call N federated tools over ONE connection.

The shipped, overridable battery for a COMPOSITE: a scripted node/tool that must
ISSUE several federated-primitive calls (e.g. ``search_crm_objects`` then
``get_crm_objects`` xM) and assemble the result. ``mcp_tool_factory`` /
``mcp_tool_factories`` BIND one federated tool 1:1 as a callable; they cannot make
N calls over a single connection, and materialising N factories opens 2 sessions
per call. ``mcp_session`` fills that gap by exposing the adapter's persistent
session mode (which ``mcp_resource_fetcher`` already uses internally) as a small,
typed, consumer-owned surface.

## Ownership + lifetime (the nmb2 invariant — do NOT reopen)

The ``MultiServerMCPClient`` / ``ClientSession`` is created and OWNED strictly
INSIDE ``McpSession``, which the CONSUMER holds via ``async with``; neograph core
NEVER creates, holds, or disposes it. This module lives in the ``neograph_mcp``
battery (never ``src/neograph``), so the session-ownership guard stays green. The
session's lifetime is ``<= one node/tool invocation``: open it INSIDE the body that
uses it, enter and exit in the SAME task (an anyio cancel-scope rule — crossing
tasks fails loudly with a ``RuntimeError``), and never store it in state or a
checkpoint (``McpSession`` is not serialisable, so a checkpoint attempt fails
loudly). A resumed run re-opens a session and re-mints identity because the
consumer's body re-executes.

## Offline-at-build + timeout

Constructing ``mcp_session(...)`` performs ZERO network I/O; the connect
(``initialize`` handshake) fires at ``async with`` entry, bounded by ``timeout``.
The default is 30s, NOT ``None``: a stdio server that never answers ``initialize``
hangs ``__aenter__`` indefinitely, and a silent-accepting HTTP server stalls on the
adapter's 300s read timeout — both empirically confirmed. ``timeout`` bounds the
connect (``asyncio.timeout``) AND each ``call()`` (``read_timeout_seconds``).

## Typed results align with the factory path

``call(..., output_model=Model)`` mirrors ``mcp_tool_factory(output_model=)`` /
``_wrap_output_model``: it rehydrates the server's ``structuredContent`` into the
consumer's model (via ``parse`` if given), with the SAME missing-structuredContent
``ValueError`` message. Without ``output_model`` a ``call()`` returns the full
``McpCallResult`` (converted content blocks + the raw ``structuredContent`` dict).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TypeVar, overload

from pydantic import BaseModel

_Model = TypeVar("_Model", bound=BaseModel)

from neograph_mcp._client import (
    HttpServer,
    ParseFn,
    StdioServer,
    TokenProvider,
    _client_for,
    _declares_arg,
    _resolve_token,
    _resolve_token_no_config,
)
from neograph_mcp._typed import rehydrate


class McpToolCallError(Exception):
    """A federated ``tools/call`` returned ``isError=True``.

    Unlike the factory path (where the adapter's ``_convert_call_tool_result``
    raises), the RAW session's ``call_tool`` returns ``CallToolResult(isError=True)``
    WITHOUT raising — so the session inspects ``.isError`` itself and raises this,
    carrying the already-converted error content and any ``structuredContent``."""

    def __init__(
        self,
        server_key: str,
        tool_name: str,
        content: list[dict[str, Any]],
        structured: dict[str, Any] | None = None,
    ) -> None:
        self.server_key = server_key
        self.tool_name = tool_name
        self.content = content
        self.structured = structured
        text = _first_text(content) or ""
        super().__init__(f"MCP tool '{tool_name}' on '{server_key}' returned an error: {text}".rstrip(": "))


@dataclass(frozen=True)
class McpCallResult:
    """One ``tools/call`` result, converted once, no lossy post-processing.

    ``content`` is ALWAYS a list of langchain-style content-block dicts (the same
    shape the bound-tool path carries); ``structured`` is the server's
    ``CallToolResult.structuredContent`` verbatim (``None`` when the tool is
    content-only)."""

    content: list[dict[str, Any]]
    structured: dict[str, Any] | None

    @property
    def text(self) -> str | None:
        """The first text block's text, else ``None`` — a convenience for the common
        single-text-block result."""
        return _first_text(self.content)


def _first_text(content: list[dict[str, Any]]) -> str | None:
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text") is not None:
            return str(block["text"])
    return None


# ── content-block conversion ──────────────────────────────────────────────────
# Replicated from langchain-mcp-adapters 0.3.0 ``tools.py:175-223``
# (``_convert_mcp_content_to_lc_block``) against the PUBLIC block constructors the
# adapter itself uses — we do NOT import its private ``_convert_call_tool_result``.
# Pinned to that revision; ``test_session_content_table_matches_adapter`` asserts
# parity so a future adapter bump prompts a re-diff.


def _convert_content_block(block: Any) -> dict[str, Any]:
    from langchain_core.messages.content import (
        create_file_block,
        create_image_block,
        create_text_block,
    )
    from mcp.types import (
        AudioContent,
        BlobResourceContents,
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
        TextResourceContents,
    )

    # The langchain constructors return typed content-block TypedDicts; coerce each
    # to a plain ``dict`` for a uniform ``McpCallResult.content`` surface.
    if isinstance(block, TextContent):
        return dict(create_text_block(text=block.text))
    if isinstance(block, ImageContent):
        return dict(create_image_block(base64=block.data, mime_type=block.mimeType))
    if isinstance(block, AudioContent):
        raise NotImplementedError(
            f"AudioContent conversion is not supported (mime type: {block.mimeType})"
        )
    if isinstance(block, ResourceLink):
        mime_type = block.mimeType or None
        if mime_type and mime_type.startswith("image/"):
            return dict(create_image_block(url=str(block.uri), mime_type=mime_type))
        return dict(create_file_block(url=str(block.uri), mime_type=mime_type))
    if isinstance(block, EmbeddedResource):
        resource = block.resource
        if isinstance(resource, TextResourceContents):
            return dict(create_text_block(text=resource.text))
        if isinstance(resource, BlobResourceContents):
            mime_type = resource.mimeType or None
            if mime_type and mime_type.startswith("image/"):
                return dict(create_image_block(base64=resource.blob, mime_type=mime_type))
            return dict(create_file_block(base64=resource.blob, mime_type=mime_type))
        raise ValueError(f"Unknown embedded resource type: {type(resource).__name__}")
    raise ValueError(f"Unknown MCP content type: {type(block).__name__}")


def _convert_content(blocks: Any) -> list[dict[str, Any]]:
    return [_convert_content_block(b) for b in (blocks or [])]


# ── ExceptionGroup unwrap ─────────────────────────────────────────────────────


def _unwrap_single(exc: BaseException) -> BaseException:
    """Recursively descend a single-leaf ``BaseExceptionGroup`` to the bare error.

    The anyio task groups inside the stdio / streamable-http transports wrap
    failures in ``ExceptionGroup``s — including at CONNECT time, sometimes nested
    two deep (a refused port surfaced as ``ExceptionGroup(ConnectError)``; a dead
    stdio server as ``ExceptionGroup(ExceptionGroup(McpError))``). Descend while the
    group has exactly one leaf so a transport failure reaches the consumer as its
    own type. A genuine multi-leaf group (a real double failure) is returned as-is —
    no information is discarded. ``asyncio.TimeoutError`` is raised OUTSIDE the anyio
    stack, so a connect timeout is a bare ``TimeoutError`` this never touches."""
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


# ── the session ───────────────────────────────────────────────────────────────


class McpSession:
    """One MCP connection, N tool calls. Async context manager; consumer-owned.

    Build via :func:`mcp_session`. Zero network at construction; the connect fires
    at ``__aenter__`` (bounded by ``timeout``). Identity is minted ONCE at entry
    (from ``config['configurable']`` via ``token_provider`` when ``config`` is
    given, else the no-config provider shape). Open it INSIDE the node/tool body
    that uses it, enter and exit in the same task, and never store it.
    """

    def __init__(
        self,
        server_key: str,
        spec: StdioServer | HttpServer,
        *,
        token_provider: TokenProvider | None,
        config: Any | None,
        stdio_token_arg: str,
        timeout: float | None,
    ) -> None:
        self._server_key = server_key
        self._spec = spec
        self._token_provider = token_provider
        self._config = config
        self._stdio_token_arg = stdio_token_arg
        self._timeout = timeout
        # Populated at __aenter__.
        self._token: str | None = None
        self._session: Any = None
        self._cm: Any = None
        # Lazily cached tool-arg declarations (stdio identity injection + tool_names).
        self._declared_args: dict[str, set[str]] | None = None
        self._names: list[str] | None = None

    async def __aenter__(self) -> McpSession:
        # Mint identity once (config path when config given, else no-config path).
        if self._config is not None:
            self._token = await _resolve_token(self._token_provider, self._config)
        else:
            self._token = await _resolve_token_no_config(self._token_provider)
        try:
            async with asyncio.timeout(self._timeout):
                client = _client_for(self._server_key, self._spec, self._token)
                self._cm = client.session(self._server_key)
                self._session = await self._cm.__aenter__()
        except BaseException as exc:  # noqa: BLE001 - normalise the transport's group wrapping
            raise _unwrap_single(exc) from None
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._cm is None:
            return
        try:
            await self._cm.__aexit__(*exc)
        except BaseException as err:  # noqa: BLE001 - normalise the transport's group wrapping
            raise _unwrap_single(err) from None
        finally:
            self._cm = None
            self._session = None

    async def _ensure_listing(self) -> None:
        """Connect-once tools/list over the SAME session, paginated, cached. Backs
        both stdio identity injection and ``tool_names()``."""
        if self._names is not None:
            return
        names: list[str] = []
        declared: dict[str, set[str]] = {}
        cursor: str | None = None
        while True:
            result = await self._session.list_tools(cursor)
            for tool in result.tools:
                names.append(tool.name)
                schema = getattr(tool, "inputSchema", None) or {}
                declared[tool.name] = set((schema.get("properties") or {}).keys())
            cursor = getattr(result, "nextCursor", None)
            if not cursor:
                break
        self._names = names
        self._declared_args = declared

    async def tool_names(self) -> list[str]:
        """The server's tool names (paginated ``tools/list`` over this session, cached)."""
        await self._ensure_listing()
        assert self._names is not None
        return list(self._names)

    @overload
    async def call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = ...,
    ) -> McpCallResult: ...

    @overload
    async def call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = ...,
        *,
        output_model: type[_Model],
        parse: ParseFn | None = ...,
    ) -> _Model: ...

    async def call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        output_model: type[BaseModel] | None = None,
        parse: ParseFn | None = None,
    ) -> McpCallResult | BaseModel:
        """Call ``tool_name`` with ``args`` over this session; return the result.

        Over stdio, per-run identity is injected as the ``stdio_token_arg`` argument
        when the tool declares it (minted token OVERRIDES a caller-supplied value —
        identity is framework-carried); over http it rides the bearer header set at
        connect, so nothing is injected here. ``isError=True`` raises
        :class:`McpToolCallError`. With ``output_model`` the server's
        ``structuredContent`` is rehydrated into that model (via ``parse`` if given)
        and returned bare — a content-only tool raises the same typed ``ValueError``
        as the factory path; a ``ValidationError`` propagates. Without
        ``output_model`` the full :class:`McpCallResult` is returned.
        """
        call_args = dict(args or {})
        if isinstance(self._spec, StdioServer) and self._token is not None:
            await self._ensure_listing()
            assert self._declared_args is not None
            if _declares_arg(self._declared_args.get(tool_name, set()), self._stdio_token_arg):
                call_args[self._stdio_token_arg] = self._token

        read_timeout = timedelta(seconds=self._timeout) if self._timeout is not None else None
        result = await self._session.call_tool(tool_name, call_args, read_timeout_seconds=read_timeout)

        content = _convert_content(getattr(result, "content", None))
        structured = getattr(result, "structuredContent", None)

        # isError BEATS output_model: the session owns the raise (no adapter wrapper
        # on the raw path). Convert content first (mirrors the adapter), then raise.
        if getattr(result, "isError", False):
            raise McpToolCallError(self._server_key, tool_name, content, structured)

        if output_model is not None:
            return rehydrate(output_model, parse, structured, tool_name)

        return McpCallResult(content=content, structured=structured)


def mcp_session(
    server_key: str,
    spec: StdioServer | HttpServer,
    *,
    token_provider: TokenProvider | None = None,
    config: Any | None = None,
    stdio_token_arg: str = "token",
    timeout: float | None = 30.0,
) -> McpSession:
    """Build an :class:`McpSession` for calling N federated tools over ONE connection.

    Construction is ZERO network I/O; the connect fires at ``async with`` entry. Use
    this for a scripted COMPOSITE — a node/tool that issues several primitive calls
    and assembles the result — where binding one tool 1:1 (``mcp_tool_factory``)
    cannot. The caller passes the tool's name as the server exposes it (a gateway
    re-exposes federated tools namespaced, e.g. ``<peer>-<tool>``); there is no
    rename on this path — discover names via :meth:`McpSession.tool_names`.

    ``token_provider`` mints per-run identity (from ``config['configurable']`` when
    ``config`` is given, else called with no arguments); ``stdio_token_arg`` is the
    tool argument identity rides on for stdio (http uses a bearer header);
    ``timeout`` (seconds, default 30) bounds the connect and each call — pass
    ``None`` to opt out (NOT recommended: a hung server otherwise stalls up to 300s).

    The ``MultiServerMCPClient`` / session is owned strictly inside the returned
    object (the nmb2 invariant); open it in the same task and never store it.
    """
    return McpSession(
        server_key,
        spec,
        token_provider=token_provider,
        config=config,
        stdio_token_arg=stdio_token_arg,
        timeout=timeout,
    )
