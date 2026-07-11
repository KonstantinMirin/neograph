"""Keyless, serverless test doubles for the MCP consumer surfaces.

Graduates the hand-rolled MCP fakes (``tests/test_mcp_battery.py``) into a
supported ``neograph_mcp`` surface — mirroring how ``neograph.testing`` promoted
the LLM fakes. The three doubles cover the three consumer surfaces a downstream
composite is built from, and each runs with **zero subprocess and zero network**:

- :class:`FakeMcpSession` — a drop-in for :class:`~neograph_mcp.McpSession` (the
  ``mcp_session`` composite path): scripted results in, recorded ``(tool, args,
  identity)`` out.
- :func:`fake_mcp_tool_factory` — a :data:`~neograph_mcp.ToolFactory` for
  ``compile(tool_factories={...})`` (the agent/act bound-tool path).
- :func:`fake_mcp_resource_fetcher` — the ``(fetcher, replayer)`` tuple
  ``mcp_resource_fetcher`` returns (the ``FromResource`` hydration path).

## The anti-echo-chamber invariant (do NOT reopen)

Every double that carries an ``output_model`` rehydrates the scripted
``structuredContent`` through the SAME :func:`neograph_mcp._typed.rehydrate` the
real ``McpSession`` / ``_wrap_output_model`` call — never a pre-built model
instance. So a schema mismatch that would break on the real server also breaks on
the fake (identical ``ValueError`` on a missing ``structuredContent``, identical
``ValidationError`` on a shape mismatch). A test green on the fake is green on the
real demo server; the parity test (``tests/test_mcp_fakes.py``) pins it by scripting
the fake from the demo server's CAPTURED payload.

## Identity, not transport

The fake mints per-run identity through the SAME ``_resolve_token`` /
``_resolve_token_no_config`` the real session uses at ``__aenter__``, and records it
separately on each call (``.calls[].identity``). It does NOT simulate transport-level
``stdio_token_arg`` injection-into-args (that needs a live server tools-list the fake
has no access to) — identity is surfaced via ``.calls[].identity``, NEVER folded into
``.calls[].args``, so consumer assertions stay stable regardless of transport.

Importing this module fail-loud-requires the ``mcp`` extra (like any
``neograph_mcp`` submodule): the fakes remove SUBPROCESS + NETWORK, not the
dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from neograph_mcp._client import (
    ParseFn,
    TokenProvider,
    _resolve_token,
    _resolve_token_no_config,
)
from neograph_mcp._session import McpCallResult, McpToolCallError
from neograph_mcp._typed import rehydrate

__all__ = [
    "FakeMcpSession",
    "RecordedCall",
    "fake_mcp_tool_factory",
    "fake_mcp_resource_fetcher",
]


@dataclass(frozen=True)
class RecordedCall:
    """One recorded ``.call()`` — the tool, the VERBATIM caller args (no injected
    identity), and the per-run identity minted at session entry."""

    tool: str
    args: dict[str, Any]
    identity: str | None


class FakeMcpSession:
    """A serverless drop-in for :class:`~neograph_mcp.McpSession`.

    Script results keyed by tool name (``results=``) and/or mark tools that must
    surface an ``isError`` (``errors=``). ``async with`` mints per-run identity via
    the same resolver the real session uses; each :meth:`call` records a
    :class:`RecordedCall` and honours the real ``output_model`` rehydration + typed
    errors. No subprocess, no network.
    """

    def __init__(
        self,
        results: dict[str, McpCallResult | list[McpCallResult] | Callable[[dict[str, Any]], McpCallResult]]
        | None = None,
        *,
        errors: set[str] | None = None,
        server_key: str = "fake",
        token_provider: TokenProvider | None = None,
        config: Any | None = None,
    ) -> None:
        # Internal storage also holds an Iterator[McpCallResult] once a list value is consumed
        # (converted in place on first use — see _resolve_scripted); the public param stays
        # constant | list | callable.
        self._results: dict[
            str, McpCallResult | list[McpCallResult] | Callable[[dict[str, Any]], McpCallResult] | Iterator[McpCallResult]
        ] = dict(results or {})
        self._errors = set(errors or set())
        self._server_key = server_key
        self._token_provider = token_provider
        self._config = config
        self._token: str | None = None
        self._entered = False
        self.calls: list[RecordedCall] = []

    async def __aenter__(self) -> FakeMcpSession:
        # Mint identity once, exactly as McpSession.__aenter__ does (config path when
        # config is given, else the no-config provider shape).
        if self._config is not None:
            self._token = await _resolve_token(self._token_provider, self._config)
        else:
            self._token = await _resolve_token_no_config(self._token_provider)
        self._entered = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._entered = False

    async def tool_names(self) -> list[str]:
        """The scripted tool set — the fake's stand-in for the real session's
        paginated ``tools/list`` (scripted results + error-marked tools)."""
        return list(self._results.keys() | self._errors)

    def _resolve_scripted(self, tool_name: str, args: dict[str, Any] | None) -> McpCallResult | None:
        """Resolve a tool's tri-modal scripted VALUE to a concrete :class:`McpCallResult`
        (neograph-4o7yu): a bare ``McpCallResult`` is a CONSTANT (backward-compat); a
        ``list[McpCallResult]`` is a FIFO SEQUENCE (converted to an iterator IN PLACE under the
        same key so ``tool_names()`` / the not-found list keep working); a ``Callable`` is a
        per-args RESOLVER (must return an ``McpCallResult``). ``None`` -> not scripted. Called
        AFTER the ``RecordedCall`` append, so a sequence-exhaustion raise still records the call."""
        value = self._results.get(tool_name)
        if value is None:
            return None
        if callable(value):
            result = value(dict(args or {}))
            if not isinstance(result, McpCallResult):
                raise TypeError(
                    f"FakeMcpSession resolver for '{tool_name}' must return an McpCallResult, "
                    f"got {type(result).__name__}."
                )
            return result
        if isinstance(value, list):
            value = self._results[tool_name] = iter(value)  # FIFO: iterator in place on first use
        if isinstance(value, Iterator):
            try:
                return next(value)
            except StopIteration:
                raise RuntimeError(
                    f"FakeMcpSession sequence for '{tool_name}' is exhausted "
                    f"(more calls than scripted results)."
                ) from None
        return value  # a constant McpCallResult

    async def call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        output_model: type[BaseModel] | None = None,
        parse: ParseFn | None = None,
    ) -> McpCallResult | BaseModel:
        """Look up the scripted result, record the call (with identity), and return
        it — honouring the real ``isError`` -> :class:`McpToolCallError`,
        ``output_model`` rehydration, and missing-``structuredContent`` ``ValueError``
        semantics. The per-tool scripted value is tri-modal (neograph-4o7yu): a constant, a
        FIFO ``list``, or a per-args ``Callable`` (see :meth:`_resolve_scripted`).
        """
        # Record BEFORE any raise: even an isError call is a recorded interaction
        # (identity is surfaced separately, caller args stay verbatim).
        recorded = RecordedCall(tool=tool_name, args=dict(args or {}), identity=self._token)
        self.calls.append(recorded)

        scripted = self._resolve_scripted(tool_name, args)

        if tool_name in self._errors:
            content = scripted.content if scripted is not None else []
            structured = scripted.structured if scripted is not None else None
            raise McpToolCallError(self._server_key, tool_name, content, structured)

        if scripted is None:
            available = sorted(self._results.keys() | self._errors)
            raise KeyError(
                f"FakeMcpSession has no scripted result for tool '{tool_name}'. "
                f"Scripted tools: {available}. Pass it in results= (or errors=)."
            )

        if output_model is not None:
            return rehydrate(output_model, parse, scripted.structured, tool_name)

        return scripted


def fake_mcp_tool_factory(
    tool_name: str,
    result: Any,
    *,
    output_model: type[BaseModel] | None = None,
    structured: dict[str, Any] | None = None,
    parse: ParseFn | None = None,
    token_provider: TokenProvider | None = None,
    args_schema: type[BaseModel] | None = None,
    description: str = "fake MCP tool",
    calls: list[RecordedCall] | None = None,
) -> Any:
    """Build a serverless :data:`~neograph_mcp.ToolFactory` for one bound tool.

    Returns an async ``(config, tool_config) -> StructuredTool`` suitable for
    ``compile(tool_factories={tool_name: factory})``. The built tool records each
    invocation onto ``calls`` (with the per-run identity minted via the same
    ``_resolve_token`` the real factory uses) and returns ``result`` — or, when
    ``output_model`` is declared, the REAL rehydrated model built from ``structured``
    via the shared :func:`~neograph_mcp._typed.rehydrate` (so
    ``ToolInteraction.typed_result`` parity holds). No subprocess, no network.

    ``args_schema`` declares the tool's argument model so the LLM-supplied call args
    reach ``_run`` (and get recorded) — without it ``StructuredTool`` infers an empty
    schema from ``**kwargs`` and langchain filters every arg out. Omit only for a
    genuinely no-argument tool.

    ``calls`` (if supplied) is the recording sink the test asserts on; a fresh list
    is created and returned via the closure otherwise (access it as
    ``factory.recorded_calls``).
    """
    from langchain_core.tools import StructuredTool

    recorded: list[RecordedCall] = calls if calls is not None else []

    async def factory(config: Any, tool_config: Any) -> Any:
        identity = await _resolve_token(token_provider, config)

        async def _run(**kwargs: Any) -> Any:
            recorded.append(RecordedCall(tool=tool_name, args=dict(kwargs), identity=identity))
            if output_model is not None:
                return rehydrate(output_model, parse, structured, tool_name)
            return result

        return StructuredTool.from_function(
            coroutine=_run, name=tool_name, description=description, args_schema=args_schema
        )

    factory.recorded_calls = recorded  # type: ignore[attr-defined]
    return factory


def fake_mcp_resource_fetcher(
    resources: dict[str, tuple[Any, str | None]] | None = None,
    *,
    replays: dict[str, list[dict[str, Any]]] | None = None,
    token_provider: TokenProvider | None = None,
    calls: list[tuple[str, dict[str, Any]]] | None = None,
) -> tuple[Any, Any]:
    """Build the serverless ``(fetcher, replayer)`` tuple ``mcp_resource_fetcher``
    returns, so ``FromResource`` hydration runs keyless.

    ``fetcher(uri) -> (content, mime)`` returns the scripted ``resources[uri]``
    tuple; ``replayer(tool_name, args) -> list[block]`` returns the scripted
    ``replays[tool_name]`` content-list. Both record onto ``calls`` (if supplied).
    Signature matches the real ``mcp_resource_fetcher`` return so it drops into the
    same ``config['configurable']`` slot. No subprocess, no network.
    """
    scripted = dict(resources or {})
    scripted_replays = dict(replays or {})
    recorded: list[tuple[str, dict[str, Any]]] = calls if calls is not None else []

    async def fetcher(uri: str) -> tuple[Any, str | None]:
        recorded.append((uri, {}))
        if uri not in scripted:
            raise KeyError(
                f"fake_mcp_resource_fetcher has no scripted resource for '{uri}'. "
                f"Scripted URIs: {sorted(scripted)}."
            )
        return scripted[uri]

    async def replayer(tool_name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        recorded.append((tool_name, dict(args)))
        if tool_name not in scripted_replays:
            raise KeyError(
                f"fake_mcp_resource_fetcher has no scripted replay for '{tool_name}'. "
                f"Scripted replays: {sorted(scripted_replays)}."
            )
        return scripted_replays[tool_name]

    return fetcher, replayer
