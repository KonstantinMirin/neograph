"""Consumer-side MCP client stitching — the shipped, overridable battery.

This module graduates the ``MultiServerMCPClient`` + transport-config +
token-provider glue the g4q9 MCP examples hand-rolled into a supported helper, in
the shape of the ``DefaultPromptCompiler`` seam-plus-battery precedent: the seam is
``compile(tool_factories=...)`` / the resource-fetcher config keys (already in
neograph core, unchanged); this is only an OPTIONAL default on top of it. A
consumer who wants house rules hand-rolls their own factory exactly as before.

## The nmb2 invariant (do NOT reopen)

The ``MultiServerMCPClient`` / ``ClientSession`` is created and OWNED **inside** the
returned factories/callables, which the CONSUMER holds via ``config`` — neograph
core NEVER creates, holds, or disposes it. The factories are re-invoked per
superstep (reading ``config`` fresh), so per-run identity is minted per run by the
``token_provider``. This module imports the adapter but NOTHING from the langgraph
engine or neograph run-layer internals — it produces plain factories/callables the
existing seams already accept (enforced by
``tests/test_guards_mcp_session_ownership.py``).

## Transport-aware per-run identity

stdio has no HTTP headers, so per-run identity rides as a tool ARGUMENT (default
name ``token``; the demo server echoes it under ``acting_as``). streamable-http
uses a bearer ``Authorization`` header. ``mcp_tool_factories`` handles both.

## namespace / collision policy

``namespace=True`` (default) keys each factory ``"{server}::{tool}"`` and RENAMES
the returned tool to match (so the LLM-facing name, the ``Tool`` spec name, and the
factory-lookup key all agree). ``namespace=False`` keys by the bare tool name and
raises ``ValueError`` if two servers expose the same tool name. The return is a
DICT so a consumer can slice it per node for least-privilege selective binding.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import socket
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel

from neograph_mcp._typed import rehydrate

# A consumer-owned tool factory: (config, tool_config) -> tool (or awaitable of one).
# The MCP-style factories this module builds are async (they await a token
# provider / build an MCP client), so they must be driven by arun().
ToolFactory = Callable[[Any, Any], Any]

# token_provider: mints per-run identity from config['configurable'] (tool path) or
# with no arguments (resource path — the fetch seam carries no config). Sync or async.
TokenProvider = Callable[..., Awaitable[str] | str]

# parse: rehydrate the tool's structuredContent dict into the declared output model.
# No mime argument (unlike resource_reader's parse) — the structured payload is
# already a JSON object, not opaque bytes.
ParseFn = Callable[[dict[str, Any]], BaseModel]

# Battery-private, CONFIG-ONLY key: mcp_run_context stashes its held
# {server_key: live_session} map under config['configurable'][RUN_SESSIONS_KEY]
# so the tool factory can bind to a held session instead of reconnecting per
# call. Never enters state or a checkpoint (sessions are not serialisable);
# defined HERE (not _run_context.py) to keep the import graph one-way
# (_run_context -> _client).
RUN_SESSIONS_KEY = "_neograph_mcp_run_sessions"


def _held_sessions(config: Any) -> dict[str, Any]:
    """The run-context's held ``{server_key: session}`` map from ``config``, or
    ``{}`` when the consumer did not open (or thread) an ``mcp_run_context`` —
    the factory then falls back to the stateless per-call path."""
    if isinstance(config, dict):
        configurable = config.get("configurable", {}) or {}
    else:
        configurable = getattr(config, "configurable", {}) or {}
    return configurable.get(RUN_SESSIONS_KEY) or {}


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


# ── transport resilience (per-call timeout + bounded retry) ──────────────────


def _is_transport_error(leaf: BaseException) -> bool:
    """NARROW transport allowlist: only failures of the WIRE, never of the result.

    ``ConnectionError`` (refused/reset/aborted), ``TimeoutError`` (per-call bound
    or read timeout), DNS resolution (``socket.gaierror``), the SDK's ``McpError``
    (protocol-level transport failures), and httpx's ``TransportError`` family
    (streamable-http connect/read/write). Everything else — ``ToolException``
    (isError is a RESULT), ``ValidationError`` (a schema mismatch), arbitrary tool
    bugs — surfaces immediately."""
    if isinstance(leaf, (ConnectionError, TimeoutError, socket.gaierror)):
        return True
    # Deferred protocol imports (match _client_for's function-local adapter import).
    from mcp.shared.exceptions import McpError

    if isinstance(leaf, McpError):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx rides in with the mcp extra
        return False
    return isinstance(leaf, httpx.TransportError)


def _is_pre_send_safe(leaf: BaseException) -> bool:
    """True for CONNECT-phase failures where the request never reached the server —
    always safe to retry, regardless of idempotency: the call cannot have executed.
    A refused connection or failed DNS lookup happens before anything is sent.
    Everything else transport-shaped (read timeout, reset mid-flight, protocol
    error) is AMBIGUOUS — the tool may have partially executed."""
    if isinstance(leaf, (ConnectionRefusedError, socket.gaierror)):
        return True
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx rides in with the mcp extra
        return False
    return isinstance(leaf, httpx.ConnectError)


def _with_transport_resilience(
    tool: Any,
    *,
    timeout: float | None = 30.0,
    max_attempts: int = 3,
    backoff: float = 0.25,
    tool_config: Any = None,
) -> Any:
    """Wrap ``tool`` so each call gets a per-call timeout and TRANSPORT-only retry.

    The tool-path arm of the single-responsibility retry split: ``_llm_retry``
    owns model-output recovery; this owns the wire. Each attempt runs inside
    ``asyncio.timeout(timeout)`` (parity with ``mcp_session``'s ``timeout=30`` —
    the factory path otherwise inherits the adapter's 300s read timeout), and a
    failure is retried with bounded exponential backoff ONLY when it is:

    - a transport error (the narrow :func:`_is_transport_error` allowlist —
      ``ToolException``/isError is a RESULT the model must see and is NEVER
      retried; classification trumps idempotency), AND
    - safe to replay: CONNECT-phase failures (refused, DNS) retry regardless —
      nothing was sent; AMBIGUOUS post-send failures (read timeout, reset,
      protocol error) retry ONLY when the tool is declared idempotent.

    Idempotency arrives via the FACTORY-CALL CHANNEL: ``tool_config`` (the dict
    core passes as ``factory(config, tool_config)``) carrying ``idempotent=True``
    — single-sourced from the node's ``Tool(idempotent=)`` spec, never a second
    hand-set flag. Missing/None/unknown means NON-idempotent (never replay).

    A retry is the SAME logical call: it happens inside the wrapped coroutine,
    invisible to ``ToolBudgetTracker`` — budget counts the logical call once.
    Composed OUTERMOST in the factory (after identity injection, rename, and the
    output-model wrap) so a retry re-runs the whole transport call while domain
    errors from any inner layer propagate untouched.
    """
    from langchain_core.tools import ToolException

    idempotent = bool((tool_config or {}).get("idempotent") or False) if isinstance(tool_config, dict) else False
    orig = tool.coroutine

    async def _resilient(**kwargs: Any) -> Any:
        attempt = 0
        while True:
            attempt += 1
            try:
                async with asyncio.timeout(timeout):
                    return await orig(**kwargs)
            except (ToolException, asyncio.CancelledError):
                raise  # a RESULT (isError) or a cancellation — never transport
            except BaseException as exc:  # noqa: BLE001 - classify, then re-raise or retry
                leaf = _unwrap_single(exc)
                if isinstance(leaf, (ToolException, asyncio.CancelledError)):
                    raise
                if not _is_transport_error(leaf):
                    raise
                if attempt >= max_attempts:
                    raise
                if not idempotent and not _is_pre_send_safe(leaf):
                    raise  # ambiguous post-send failure on a non-idempotent tool
                await asyncio.sleep(backoff * (2 ** (attempt - 1)))

    return tool.model_copy(update={"coroutine": _resilient})


@dataclass(frozen=True)
class StdioServer:
    """A stdio MCP server spawned as a subprocess (no ports, no network).

    Per-run identity rides as a tool argument (stdio has no HTTP headers) — see
    ``mcp_tool_factories(stdio_token_arg=...)``.
    """

    command: str
    args: list[str]
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class HttpServer:
    """A streamable-http MCP server. Per-run identity rides as a bearer
    ``Authorization`` header minted by the ``token_provider``."""

    url: str
    headers: dict[str, str] | None = None


# ── transport wiring ──────────────────────────────────────────────────────────


def _connection(spec: StdioServer | HttpServer, token: str | None) -> dict[str, Any]:
    """Build the langchain-mcp-adapters connection dict for one server + identity."""
    if isinstance(spec, StdioServer):
        conn: dict[str, Any] = {"transport": "stdio", "command": spec.command, "args": list(spec.args)}
        if spec.env is not None:
            conn["env"] = dict(spec.env)
        return conn
    headers = dict(spec.headers or {})
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    conn = {"transport": "streamable_http", "url": spec.url}
    if headers:
        conn["headers"] = headers
    return conn


def _client_for(
    server_key: str,
    spec: StdioServer | HttpServer,
    token: str | None,
    *,
    callbacks: Any | None = None,
) -> Any:
    """Create a consumer-held MultiServerMCPClient for ONE server (function-local
    import keeps the module import light and the fail-loud check in ``__init__``).

    ``callbacks`` (an adapter ``Callbacks``) rides the client constructor —
    ``get_tools()`` forwards it into ``load_mcp_tools`` verbatim, so the built
    tools carry e.g. an ``on_progress`` notification handler with no change to
    the build path. Default ``None`` keeps discovery / resource paths
    callback-free."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    # The connection dict matches the adapter's Stdio/StreamableHttp TypedDicts at
    # runtime; cast past the structural mismatch the plain dict builder produces.
    return MultiServerMCPClient(cast(Any, {server_key: _connection(spec, token)}), callbacks=callbacks)


async def _resolve_token(token_provider: TokenProvider | None, config: Any) -> str | None:
    """Mint per-run identity from ``config['configurable']`` (tool path). Sync or async."""
    if token_provider is None:
        return None
    if isinstance(config, dict):
        configurable = config.get("configurable", {}) or {}
    else:
        configurable = getattr(config, "configurable", {}) or {}
    result = token_provider(configurable)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _resolve_token_no_config(token_provider: TokenProvider | None) -> str | None:
    """Mint identity on the resource path, which carries no config: the provider is
    called with no arguments (a per-run closure), falling back to ``{}`` for a
    provider written against the configurable-dict tool-path shape."""
    if token_provider is None:
        return None
    try:
        result = token_provider()
    except TypeError:
        result = token_provider({})
    if inspect.isawaitable(result):
        result = await result
    return result


def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Drive ``coro`` to completion from a sync call site — even when a loop is
    already running (an async test / notebook), by using a worker thread. Build-time
    tool discovery is sync (``mcp_tool_factories`` returns a plain dict), but the
    adapter's ``get_tools`` is async; this bridges the two without forcing the
    consumer to await the builder."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


# ── tool factories ────────────────────────────────────────────────────────────


def _declares_arg(declared_names: Any, arg_name: str) -> bool:
    """Pure membership: is ``arg_name`` among a tool's declared argument names?

    The CALLER supplies the name-set so this one check serves both surfaces: the
    factory path passes langchain ``tool.args`` (a name->schema dict); the raw
    session path passes ``mcp.types.Tool.inputSchema['properties']`` (langchain's
    ``tool.args`` does not exist on a raw MCP tool). Defensive against a
    non-container ``declared_names`` (a tool without an introspectable schema →
    ``False``)."""
    try:
        return arg_name in (declared_names or {})
    except Exception:  # noqa: BLE001 - a tool without an introspectable schema
        return False


def _inject_stdio_token(tool: Any, token: str, arg_name: str) -> Any:
    """Wrap a stdio tool so every call carries per-run identity as ``arg_name``.

    stdio has no headers, so identity travels as a tool argument. Only injected
    when the tool actually declares ``arg_name`` (else the server would reject an
    unknown kwarg); our value overrides any model-supplied one so identity is
    framework-carried, never LLM-chosen."""
    # Access tool.args inside the guard: a tool without an introspectable schema
    # may raise here (the factory-path fallback must stay declares=False).
    try:
        declared = tool.args
    except Exception:  # noqa: BLE001 - a tool without an introspectable schema
        declared = None
    if not _declares_arg(declared, arg_name):
        return tool

    original = tool.coroutine

    async def _with_identity(**kwargs: Any) -> Any:
        kwargs[arg_name] = token
        return await original(**kwargs)

    return tool.model_copy(update={"coroutine": _with_identity})


async def _discover_tool_names(server_key: str, spec: StdioServer | HttpServer) -> list[str]:
    """Connect once (identity-agnostic) to enumerate the server's tool names."""
    client = _client_for(server_key, spec, None)
    tools = await client.get_tools(server_name=server_key)
    return [t.name for t in tools]


def _wrap_output_model(
    tool: Any,
    tool_name: str,
    output_model: type[BaseModel],
    parse: ParseFn | None,
) -> Any:
    """Wrap ``tool`` so its RESULT is the rehydrated ``output_model`` instance.

    The discovered adapter tool has ``response_format="content_and_artifact"``: its
    coroutine returns ``(content, artifact)`` where ``artifact`` is an
    ``MCPToolArtifact`` TypedDict carrying ``structured_content`` (a plain dict at
    runtime — read it with ``.get``, not ``getattr``). We take that dict and
    validate it into the consumer's declared model, so the typed channel
    (``ToolInteraction.typed_result`` + the BAML ToolMessage render) carries a model
    instead of raw content blocks. Reset to ``response_format="content"`` so the
    tool interface returns the bare model.

    isError invariant (do NOT reopen): a server ``isError=True`` RAISES
    ``_MCPToolExecutionError`` from inside ``orig`` (adapter ``tools.py`` — the
    conversion raises before returning), so it propagates through this wrapper
    UNTOUCHED. We must NOT try/except-convert it: ``model_copy`` preserves the
    tool's ``handle_tool_errors`` (True), and langchain's ``arun`` catches the
    ToolException there and surfaces it as a self-correcting error ToolMessage — the
    adapter-native path. Consequently the ONLY ``None`` artifact this wrapper ever
    sees is a genuine missing ``structuredContent`` (a bare ``-> dict``/``-> list``
    server annotation); the adapter's ``(ToolMessage|Command, None)`` return is
    interceptor-only and the battery wires no interceptors."""
    orig = tool.coroutine

    async def _typed(**kwargs: Any) -> BaseModel:
        _content, artifact = await orig(**kwargs)  # isError raises here → propagates
        structured = artifact.get("structured_content") if artifact else None
        return rehydrate(output_model, parse, structured, tool_name)

    return tool.model_copy(update={"coroutine": _typed, "response_format": "content"})


def _make_tool_factory(
    server_key: str,
    spec: StdioServer | HttpServer,
    tool_name: str,
    rename_to: str | None,
    token_provider: TokenProvider | None,
    stdio_token_arg: str,
    output_model: type[BaseModel] | None = None,
    parse: ParseFn | None = None,
    timeout: float | None = 30.0,
    max_attempts: int = 3,
    backoff: float = 0.25,
) -> ToolFactory:
    """Build ONE consumer-owned async factory for ``tool_name`` on ``server_key``.

    Per superstep: mint identity from config, build a fresh client (owned here,
    disposed by the adapter's stateless-per-call session model), fetch the tool,
    bind per-run identity, (when namespaced) rename it so the LLM-facing name
    matches the ``Tool`` spec / factory key, and (when ``output_model`` is declared)
    wrap it so the result is the rehydrated model.

    The ``output_model`` wrap happens after identity injection and rename — so it
    wraps the already-injected coroutine (token injection preserved) and the
    renamed tool. The transport-resilience wrap is OUTERMOST (per-call ``timeout``
    + transport-only bounded retry; idempotency read from the factory-call
    ``tool_config`` channel) so a retry re-runs the whole transport call while
    isError/domain errors from any inner layer propagate untouched."""

    async def _factory(config: Any, tool_config: Any) -> Any:
        from neograph_mcp._progress import _progress_callbacks

        token = await _resolve_token(token_provider, config)
        callbacks = _progress_callbacks(server_key)
        held = _held_sessions(config).get(server_key)
        if held is not None:
            # Consumer-held run-scoped session (mcp_run_context): bind the tool
            # to the ONE live connection instead of the stateless per-call path.
            # Import inside the factory like _client_for's own adapter import —
            # keeps module import light behind the fail-loud extra check.
            from langchain_mcp_adapters.tools import load_mcp_tools

            tools = await load_mcp_tools(held, callbacks=callbacks, server_name=server_key)
        else:
            client = _client_for(server_key, spec, token, callbacks=callbacks)
            tools = await client.get_tools(server_name=server_key)
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool is None:
            raise ValueError(
                f"MCP server '{server_key}' no longer exposes tool '{tool_name}' "
                f"(available: {sorted(t.name for t in tools)})"
            )
        if isinstance(spec, StdioServer) and token is not None:
            tool = _inject_stdio_token(tool, token, stdio_token_arg)
        if rename_to is not None:
            tool = tool.model_copy(update={"name": rename_to})
        if output_model is not None:
            tool = _wrap_output_model(tool, tool_name, output_model, parse)
        return _with_transport_resilience(
            tool,
            timeout=timeout,
            max_attempts=max_attempts,
            backoff=backoff,
            tool_config=tool_config,
        )

    return _factory


def mcp_tool_factories(
    servers: dict[str, StdioServer | HttpServer],
    *,
    token_provider: TokenProvider | None = None,
    namespace: bool = True,
    stdio_token_arg: str = "token",
    output_models: dict[str, type[BaseModel]] | None = None,
    parses: dict[str, ParseFn] | None = None,
    timeout: float | None = 30.0,
    max_attempts: int = 3,
    backoff: float = 0.25,
) -> dict[str, ToolFactory]:
    """Build the ``{name: async factory}`` dict for ``compile(tool_factories=...)``.

    Connects once per server at call time to enumerate tools, then returns a
    consumer-owned async factory per tool — the ``MultiServerMCPClient`` is created
    and owned INSIDE each factory (per the nmb2 invariant; neograph core never
    holds it). Slice the returned dict per node for least-privilege binding.

    ``token_provider`` mints per-run identity from ``config['configurable']`` on the
    ``arun()`` path; ``namespace=True`` keys ``"{server}::{tool}"`` and renames the
    tool to match; ``stdio_token_arg`` is the tool-argument name identity rides on
    for stdio servers (streamable-http uses a bearer header instead).

    ``output_models`` / ``parses`` opt each named tool into TYPED results: when a
    tool's returned-dict key is present in ``output_models``, the factory wraps that
    tool so its result is the rehydrated model (validated from the server's
    ``structuredContent``, via the matching ``parses`` entry if given). The keys
    match the returned factory-dict keys (``"{server}::{tool}"`` when namespaced,
    the bare tool name otherwise). Tools not listed keep the raw content-block
    result — the type channel is opt-in by declaration.

    Note (deviation from the pinned sketch): ``stdio_token_arg`` is an added
    keyword — stdio has no headers, so the transport-aware token path needs to know
    which tool argument carries identity (default ``"token"``, matching the demo
    server's echoed arg).

    ``timeout`` / ``max_attempts`` / ``backoff`` tune the per-call transport
    resilience every built tool carries (see :func:`_with_transport_resilience`):
    each call is bounded by ``timeout`` seconds (parity with ``mcp_session``) and a
    TRANSPORT failure retries with bounded backoff — an ``isError`` result never
    retries, and an ambiguous post-send failure replays only when the node's
    ``Tool(idempotent=True)`` spec says so (threaded via the factory-call channel).
    """
    factories: dict[str, ToolFactory] = {}
    for server_key, spec in servers.items():
        names = _run_sync(_discover_tool_names(server_key, spec))
        for name in names:
            key = f"{server_key}::{name}" if namespace else name
            if key in factories:
                raise ValueError(
                    f"tool name collision on '{key}': two servers expose '{name}'. "
                    "Pass namespace=True to key factories as 'server::tool'."
                )
            factories[key] = _make_tool_factory(
                server_key,
                spec,
                name,
                key if namespace else None,
                token_provider,
                stdio_token_arg,
                output_model=(output_models or {}).get(key),
                parse=(parses or {}).get(key),
                timeout=timeout,
                max_attempts=max_attempts,
                backoff=backoff,
            )
    return factories


def mcp_tool_factory(
    server_key: str,
    spec: StdioServer | HttpServer,
    *,
    tool_name: str,
    rename_to: str | None = None,
    token_provider: TokenProvider | None = None,
    stdio_token_arg: str = "token",
    output_model: type[BaseModel] | None = None,
    parse: ParseFn | None = None,
    timeout: float | None = 30.0,
    max_attempts: int = 3,
    backoff: float = 0.25,
) -> ToolFactory:
    """Build ONE lazy async factory for a SINGLE known tool — no build-time connect.

    Use this (over the plural ``mcp_tool_factories``) when you already know the
    ``tool_name`` and want to bind exactly one tool with ZERO network I/O at
    construction: the ``MultiServerMCPClient`` connect is deferred into the returned
    factory body and fires only on the first ``await`` (per superstep). This suits a
    consumer whose factory-build path runs offline — e.g. at ``compile()`` time or in
    a deterministic test suite — where the plural builder's per-server enumeration
    connect (``get_tools`` at build) would be an unwanted live call.

    ``rename_to`` maps the discovered name back to a fixed bare ``Tool(name)``
    binding. A gateway (e.g. IBM ContextForge) re-exposes a federated tool NAMESPACED
    as ``<peer>-<tool>``; passing ``tool_name="<peer>-<tool>", rename_to="<tool>"``
    makes both the factory and the bound ``tool.name`` the bare name the node's
    ``Tool`` spec references.

    ``token_provider`` mints per-run identity from ``config['configurable']``;
    ``stdio_token_arg`` is the tool-argument name identity rides on for stdio servers
    (streamable-http uses a bearer header instead). Identity injection happens BEFORE
    the rename, so it introspects the server's real declared arguments.

    ``output_model`` opts into TYPED results: when declared, the factory wraps the
    tool so its result is the rehydrated model — validated from the server's
    ``structuredContent`` (via ``parse`` if given, else ``model_validate``) — so the
    typed channel (``ToolInteraction.typed_result`` + the BAML ToolMessage render)
    carries a model, not raw content blocks. Mirrors ``resource_reader(output_model=,
    parse=)``. Omit it to keep the raw content-block result (opt-in by declaration).
    A tool returning no ``structuredContent`` (a bare ``-> dict``/``-> list`` server
    annotation) raises a typed ``ValueError``; a server ``isError`` follows the
    adapter-native error path unchanged.

    ``timeout`` / ``max_attempts`` / ``backoff`` tune the per-call transport
    resilience (see :func:`_with_transport_resilience`): each call is bounded by
    ``timeout`` seconds and a TRANSPORT failure retries with bounded backoff —
    an ``isError`` result never retries; an ambiguous post-send failure replays
    only when the node's ``Tool(idempotent=True)`` spec says so.

    A thin public alias over the same builder the plural form delegates to — the
    client stays owned strictly inside the returned factory (the nmb2 invariant).
    """
    return _make_tool_factory(
        server_key,
        spec,
        tool_name,
        rename_to,
        token_provider,
        stdio_token_arg,
        output_model=output_model,
        parse=parse,
        timeout=timeout,
        max_attempts=max_attempts,
        backoff=backoff,
    )


# ── resource fetcher / replayer ───────────────────────────────────────────────


def _route_uri(servers: dict[str, StdioServer | HttpServer], uri: str) -> str:
    """Pick the server for a resource ``uri``: single server -> it; else match the
    uri host (``mcp://<host>/...``) against a server key."""
    if len(servers) == 1:
        return next(iter(servers))
    host = urlparse(uri).netloc
    if host in servers:
        return host
    raise ValueError(f"cannot route resource uri '{uri}' to any of {sorted(servers)} (host '{host}' unmatched)")


def _route_tool(servers: dict[str, StdioServer | HttpServer], tool_name: str) -> tuple[str, str]:
    """Pick (server_key, real_tool_name) for a replayed producing call. A namespaced
    ``server::tool`` name routes explicitly; otherwise a single server is assumed."""
    if "::" in tool_name:
        server_key, real = tool_name.split("::", 1)
        if server_key not in servers:
            raise ValueError(f"replay tool '{tool_name}' names unknown server '{server_key}'")
        return server_key, real
    if len(servers) == 1:
        return next(iter(servers)), tool_name
    raise ValueError(f"ambiguous replay tool '{tool_name}' across {sorted(servers)}; use a 'server::tool' name")


def _read_resource_content(result: Any) -> tuple[Any, str | None]:
    """Unpack the first content block of a ``read_resource`` result into (content, mime)."""
    contents = getattr(result, "contents", None) or []
    if not contents:
        return "", None
    block = contents[0]
    text = getattr(block, "text", None)
    mime = getattr(block, "mimeType", None)
    if text is not None:
        return text, mime
    return getattr(block, "blob", None), mime


def mcp_resource_fetcher(
    servers: dict[str, StdioServer | HttpServer],
    *,
    token_provider: TokenProvider | None = None,
) -> tuple[Callable[[str], Awaitable[tuple[Any, str | None]]], Callable[[str, dict], Awaitable[Any]]]:
    """Return ``(fetcher, replayer)`` for ``config['configurable']`` so FromResource
    hydration + layered-expiry replay work out of the box.

    - ``fetcher(uri) -> (content, mime)``: reads a resource over a consumer-owned
      session. An ``McpError`` (e.g. a ``-32002`` expiry) is caught INSIDE the
      session context and re-raised AFTER the ``async with`` closes — so it escapes
      as a bare ``McpError`` rather than an anyio-teardown ``ExceptionGroup``, which
      is exactly what ``hydrate_resource_ref`` treats as candidate expiry (-> replay).
    - ``replayer(tool_name, args) -> raw_result``: re-invokes the producing call
      through the same client so an expired ``resource_link`` can be re-derived. The
      raw session's ``resource_link`` blocks are preserved (unlike the langchain
      adapter, which surfaces them as ``file`` blocks), so neograph's
      ``_first_resource_link_uri`` finds the fresh uri. The idempotency GATE is
      enforced upstream in ``hydrate_resource_ref`` (a non-idempotent producer never
      reaches the replayer), so no idempotency logic lives here.

    Since the fetch seam carries no ``config``, ``token_provider`` is invoked with no
    arguments on this path (bind per-run identity by constructing the fetcher inside
    the run scope, or return a static token).
    """
    from mcp.shared.exceptions import McpError
    from pydantic import AnyUrl

    async def fetcher(uri: str) -> tuple[Any, str | None]:
        server_key = _route_uri(servers, uri)
        token = await _resolve_token_no_config(token_provider)
        client = _client_for(server_key, servers[server_key], token)
        error: McpError | None = None
        result: Any = None
        async with client.session(server_key) as session:
            try:
                result = await session.read_resource(AnyUrl(uri))
            except McpError as exc:
                # Catch in-scope: escaping the `async with` wraps it in an anyio
                # ExceptionGroup; re-raise the bare McpError after the block closes.
                error = exc
        if error is not None:
            raise error
        return _read_resource_content(result)

    async def replayer(tool_name: str, args: dict) -> Any:
        server_key, real = _route_tool(servers, tool_name)
        token = await _resolve_token_no_config(token_provider)
        client = _client_for(server_key, servers[server_key], token)
        async with client.session(server_key) as session:
            result = await session.call_tool(real, args or {})
        # Return the raw content list so neograph's _first_resource_link_uri scans
        # the fresh resource_link blocks (raw session preserves the 'resource_link'
        # block type; the langchain adapter would rewrite it to 'file').
        return list(getattr(result, "content", None) or [])

    return fetcher, replayer
