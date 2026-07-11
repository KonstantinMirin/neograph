"""Consumer-held, run-scoped MCP connection reuse — ``mcp_run_context``.

An N-turn agent/act run pays one MCP connect per tool CALL today: the bound-tool
factory path uses the adapter's documented stateless per-call session model.
That default is mainstream and stays the fallback — but a multi-turn run against
a slow-spawn stdio server (or a stateful server) wants ONE live connection per
server for the whole run.

The anyio cancel scopes inside the MCP transports are task-affine: a session may
be CALLED from any asyncio task (spike-verified), but it must be opened and
closed in the SAME task or disposal raises ``RuntimeError`` ("Attempted to exit
cancel scope in a different task"). LangGraph supersteps each run as a distinct
task, so no superstep can own the session. The one task that spans every
superstep is the CONSUMER'S — the task calling ``arun()``/``astream()``. This is
the ecosystem's hold-open pattern (langchain-mcp-adapters ``client.session()``,
pydantic-ai's "entered first and exited last in the same place"); see
``docs/design/mcp-connection-reuse-2026-07-10.md`` for the verified survey.

Usage — the consumer holds the context around the run and threads it via config::

    async with mcp_run_context({"crm": spec}, token_provider=..., config=base) as ctx:
        result = await neograph.arun(graph, input=..., config=ctx.config(base))

- ``__aenter__`` wires http identity as a PER-REQUEST httpx.Auth (the same
  ``_http_identity`` builder as every other surface — a refreshing
  ``token_provider`` survives the run's whole superstep span) and opens one
  persistent adapter session per server — all in the consumer's task. ``__aexit__`` closes them in reverse (LIFO) order in that
  same task: safe disposal by construction.
- ``ctx.config(base)`` returns a config whose ``configurable`` carries the held
  sessions under a battery-private key. The tool factory binds to a held session
  when one is present for its server; otherwise it falls back UNCHANGED to the
  stateless per-call path — so a run without the CM, a resumed run in a fresh
  process, or an absent entry all transparently reconnect. Reuse is an
  optimization, never an assumption a connection survives.
- The nmb2 ownership invariant is strengthened, not bent: the consumer literally
  holds the client/session lifecycle; nothing here touches ``src/neograph``.
- The held-session map is CONFIG-ONLY (like the resource fetcher): it never
  enters state or a checkpoint (sessions are not serialisable — that fails
  loudly, by design).
"""

from __future__ import annotations

import asyncio
from typing import Any

from neograph_mcp._client import (
    RUN_SESSIONS_KEY,
    HttpServer,
    StdioServer,
    TokenProvider,
    _client_for,
    _http_identity,
    _unwrap_single,
)


class McpRunContext:
    """Run-scoped holder of one persistent MCP session per server.

    Build via :func:`mcp_run_context`. Zero network at construction; connects
    fire at ``__aenter__`` (each bounded by ``timeout``). Enter and exit in the
    SAME task — the task that calls ``arun()``/``astream()`` — and never store
    the context (or its sessions) in state or a checkpoint.
    """

    def __init__(
        self,
        servers: dict[str, StdioServer | HttpServer],
        *,
        token_provider: TokenProvider | None,
        config: Any | None,
        timeout: float | None,
    ) -> None:
        self._servers = dict(servers)
        self._token_provider = token_provider
        self._config = config
        self._timeout = timeout
        # Populated at __aenter__; closed (reverse order) at __aexit__.
        self._cms: list[Any] = []
        self._sessions: dict[str, Any] = {}

    async def __aenter__(self) -> McpRunContext:
        try:
            for server_key, spec in self._servers.items():
                # http identity rides each held connection as a PER-REQUEST
                # httpx.Auth — the held sessions span every superstep of the
                # run, so a connect-time bearer would freeze for the whole run
                # (neograph-qslrx). stdio identity is injected per call by the
                # tool factory, not here.
                auth = _http_identity(
                    spec, self._token_provider, self._config, use_config=self._config is not None
                )
                async with asyncio.timeout(self._timeout):
                    client = _client_for(server_key, spec, auth)
                    cm = client.session(server_key)
                    session = await cm.__aenter__()
                self._cms.append(cm)
                self._sessions[server_key] = session
        except BaseException as exc:  # noqa: BLE001 - normalise + don't leak partial opens
            await self._aclose()
            raise _unwrap_single(exc) from None
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._aclose(*exc)

    async def _aclose(self, *exc: Any) -> None:
        """Close every opened session in reverse (LIFO) order, in THIS task."""
        errors: list[BaseException] = []
        exit_args = exc if len(exc) == 3 else (None, None, None)
        while self._cms:
            cm = self._cms.pop()
            try:
                await cm.__aexit__(*exit_args)
            except BaseException as err:  # noqa: BLE001 - keep closing the rest
                errors.append(_unwrap_single(err))
        self._sessions.clear()
        if errors:
            raise errors[0]

    def config(self, base: Any | None = None) -> dict[str, Any]:
        """Return a copy of ``base`` whose ``configurable`` carries the held
        sessions (under a battery-private, config-only key) for the tool factory
        to bind against. ``base`` is not mutated."""
        merged = dict(base or {})
        configurable = dict(merged.get("configurable") or {})
        configurable[RUN_SESSIONS_KEY] = dict(self._sessions)
        merged["configurable"] = configurable
        return merged


def mcp_run_context(
    servers: dict[str, StdioServer | HttpServer],
    *,
    token_provider: TokenProvider | None = None,
    config: Any | None = None,
    timeout: float | None = 30.0,
) -> McpRunContext:
    """Build an :class:`McpRunContext` holding ONE live MCP connection per server
    for a whole run (opt-in reuse over the stateless per-call default).

    Open it ``async with`` AROUND the ``arun()``/``astream()`` call — in that
    same task — and thread ``ctx.config(base_config)`` into the run so the tool
    factories bind to the held sessions instead of reconnecting per call. A run
    driven WITHOUT this context keeps today's per-call reconnect behavior
    (the adapter's documented default for stateless tools).

    ``token_provider`` supplies http identity as a PER-REQUEST httpx.Auth
    (re-invoked on every request, from ``config['configurable']`` when
    ``config`` is given, else called bare) — the same mechanism every other
    battery surface uses, so a refreshing provider takes effect across the held
    sessions; ``timeout`` (seconds, default 30) bounds each server's connect.
    Construction is ZERO network I/O.
    """
    return McpRunContext(
        servers,
        token_provider=token_provider,
        config=config,
        timeout=timeout,
    )
