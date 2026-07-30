"""MCP resource hydration — turning a resource ref into content.

Extracted from ``di.py`` (neograph-3ffdg.19) as a pure file split — the
functions and constants below are unchanged, only their home moved. ``di.py``
re-exports them, so existing imports keep resolving.

The fetcher and the replayer are CONSUMER-owned callables passed through
config['configurable'], exactly like tool factories: neograph never owns an MCP
session. An expired ref is re-derived by replaying its producing tool call, and
only ever for an idempotent producer -- the hard gate. Absent a replayer, an
expired ref fails loud rather than silently returning stale content.

``_get_configurable`` moved here too and ``di.py`` imports it back: both halves
use it, and the dependency has to point one way.
"""

from __future__ import annotations

from typing import Any

from neograph._content_blocks import _first_resource_link_uri, _resource_link_uri_for_kind
from neograph.errors import ConfigurationError as _ConfigurationError
from neograph.errors import NonIdempotentReplayError as _NonIdempotentReplayError
from neograph.errors import ResourceExpiredError as _ResourceExpiredError

# Config['configurable'] key holding the consumer-supplied resource fetcher —
# an ``async def fetch(uri) -> (content, mime)`` callable. Consumer-owned, exactly
# like tool factories and the per-run token provider (no session ownership). Shared
# by ``resource_reader`` (tool.py) and ``FromResource`` DI (DIBinding.aresolve).
RESOURCE_FETCHER_KEY = "mcp_resource_fetcher"

# Config['configurable'] key holding the consumer-supplied resource REPLAYER — an
# ``async def replay(tool_name, args) -> raw_tool_result`` callable that re-invokes
# a producing tool call so an EXPIRED ``resource_link`` can be re-derived (layered
# expiry step 2, neograph-a5nh). Consumer-owned (no session ownership), exactly
# like the fetcher. Optional: absent -> an expired ref fails loud rather than
# replaying. Only ever consulted for an IDEMPOTENT producer (the hard gate).
RESOURCE_REPLAYER_KEY = "mcp_resource_replayer"


# The single hint literal for the consumer-owned resource fetcher. Routed
# exclusively through ``_require_fetcher`` so the fail-loud policy has one home
# — neograph-hnbsq. Prose mentions elsewhere phrase it differently on purpose.
_FETCHER_HINT = "provide config['configurable']['{key}'] = an async 'fetch(uri) -> (content, mime)' callable"


def _get_configurable(config: Any, key: str) -> Any:
    """Read a key from config['configurable'], handling dict and attr forms."""
    cfg = config or {}
    if isinstance(cfg, dict):
        return cfg.get("configurable", {}).get(key)
    return getattr(cfg, "configurable", {}).get(key)


def _require_fetcher(config: Any, *, subject: str, required: bool = True) -> Any:
    """Read the consumer-owned resource fetcher from ``config['configurable']`` and
    fail loud when absent. The single monopoly point for the fetcher read + the
    signature hint — neograph-hnbsq. ``subject`` names the failing consumer for the
    lead message (e.g. "resource ref ...", "resource tool ...", "resource DI
    parameter ...") — it is per-site information, not part of the shared invariant.
    When ``required`` is False and the fetcher is absent, returns ``None`` instead
    of raising (the optional-resource short-circuit).
    """
    fetcher = _get_configurable(config, RESOURCE_FETCHER_KEY)
    if fetcher is None:
        if not required:
            return None
        raise _ConfigurationError.build(
            subject,
            hint=_FETCHER_HINT.format(key=RESOURCE_FETCHER_KEY),
        )
    return fetcher


def parse_resource_content(
    content: Any,
    mime: str | None,
    output_model: Any,
    parse: Any = None,
    *,
    marker_mime: str | None = None,
) -> Any:
    """Turn a fetched resource blob into the declared type. v1 rules:

    - explicit ``parse(content, mime)`` callable wins (the general escape hatch);
    - ``str`` target -> raw text passthrough (decode bytes);
    - ``application/json`` (or ``*+json``, or an absent mime) into a BaseModel ->
      ``model_validate_json``;
    - any other mime into a BaseModel with no parser -> FAIL LOUD. neograph NEVER
      runs a silent LLM parse inside DI/resource resolution (banned hidden
      cognition + cost). The caller supplies an explicit ``parse=`` for text/*.

    Shared by ``resource_reader`` (tool.py) and ``FromResource`` (aresolve).
    """
    if parse is not None:
        return parse(content, mime)

    if output_model is str:
        return content.decode() if isinstance(content, bytes) else content

    base = (mime or marker_mime or "").split(";")[0].strip().lower()
    if base == "" or base == "application/json" or base.endswith("+json"):
        return output_model.model_validate_json(content)

    model_name = getattr(output_model, "__name__", str(output_model))
    raise _ConfigurationError.build(
        f"resource mime '{mime or marker_mime or '?'}' cannot be parsed into {model_name} without an explicit parser",
        hint="pass parse=(content, mime)->model (or type the param as str for raw "
        "text passthrough); neograph never runs a silent LLM parse inside "
        "DI/resource resolution",
    )


def _enforce_max_bytes(content: Any, max_bytes: int | None, *, name: str, uri: str) -> None:
    """Fail loud when a fetched resource exceeds ``max_bytes`` (Risk-2 mitigation).

    Called AFTER the fetch but BEFORE parse/return, so an oversized blob turns
    into a clean fail-loud-at-node-entry naming the param + size + limit rather
    than a confusing downstream provider 400 once the text hits the prompt. Only
    sizes bytes/str (the fetch content); a non-sized object is left alone."""
    if max_bytes is None:
        return
    size = len(content) if isinstance(content, (bytes, bytearray, str)) else None
    if size is not None and size > max_bytes:
        raise _ConfigurationError.build(
            f"resource '{name}' ({uri}) is {size} bytes, exceeds max_bytes {max_bytes}",
            hint="use a templated resource_reader / FromResource(uri) to slice the "
            "resource (e.g. ?range=1-5) instead of hydrating the whole blob "
            "into a prompt",
        )


async def hydrate_resource_ref(
    ref: Any,
    config: Any,
    output_model: Any,
    *,
    parse: Any = None,
    marker_mime: str | None = None,
    max_bytes: int | None = None,
    node: str | None = None,
) -> Any:
    """Hydrate a ``ResourceRef`` with LAYERED EXPIRY neograph-a5nh.

    1. **read** ``ref.uri`` via the consumer's ``RESOURCE_FETCHER_KEY`` fetcher.
       On success, size-check + parse into ``output_model`` and return. A PARSE
       failure here propagates unchanged (it is NOT expiry — never masked by a
       replay).
    2. **replay** on a fetch failure (``-32002`` / any fetch error): the ONLY
       protocol-reliable re-derivation path for a lifetime-free MCP resource_link
       is to re-invoke the producing tool call. This is gated on the HARD
       IDEMPOTENCY GATE — a non-idempotent producer (an ``act``-mode mutation)
       refuses replay with ``NonIdempotentReplayError`` rather than double-apply
       the side effect (a read may replay, a mutation may not). The replayer
       (``RESOURCE_REPLAYER_KEY``) re-invokes ``ref.producing_call`` and the fresh
       ``resource_link`` uri it emits is read.
    3. **fail loud** with ``ResourceExpiredError`` if replay is impossible (no
       replayer configured) or the replay itself fails. Silent staleness is worse
       than a loud failure.
    """
    fetcher = _require_fetcher(
        config,
        subject=f"resource ref '{getattr(ref, 'uri', '?')}' has no fetcher to read from",
    )

    # 1. read
    read_error: Exception | None = None
    try:
        content, mime = await fetcher(ref.uri)
    except Exception as exc:  # noqa: BLE001 - any fetch failure = candidate expiry
        read_error = exc
    if read_error is None:
        _enforce_max_bytes(content, max_bytes, name=getattr(ref, "kind", "?"), uri=ref.uri)
        return parse_resource_content(content, mime, output_model, parse, marker_mime=marker_mime)

    # 2. replay — gated on the producer's idempotency (the hard gate)
    producing = ref.producing_call
    if not getattr(producing, "producer_idempotent", False):
        raise _NonIdempotentReplayError.of(producing.tool_name, node=node)

    replayer = _get_configurable(config, RESOURCE_REPLAYER_KEY)
    if replayer is None:
        raise _ResourceExpiredError.of(
            ref,
            node=node,
            detail="no resource replayer configured to re-derive the expired ref",
            cause=read_error,
        )
    try:
        replay_result = await replayer(producing.tool_name, producing.args)
        # Re-derive by the ref's KIND first: a multi-link producer (e.g. get_deal
        # emitting both activity- and email-history links) must heal THIS ref to
        # its own kind, not blindly the first link. Fall back to first-link
        # (single-link producers) then the original uri. neograph-m9sj
        fresh_uri = (
            _resource_link_uri_for_kind(replay_result, getattr(ref, "kind", ""))
            or _first_resource_link_uri(replay_result)
            or ref.uri
        )
        content, mime = await fetcher(fresh_uri)
    except Exception as exc:  # noqa: BLE001 - replay failure = confirmed expiry
        raise _ResourceExpiredError.of(
            ref,
            node=node,
            detail="replaying the producing call failed to re-derive the resource",
            cause=exc,
        ) from exc
    _enforce_max_bytes(content, max_bytes, name=getattr(ref, "kind", "?"), uri=fresh_uri)
    return parse_resource_content(content, mime, output_model, parse, marker_mime=marker_mime)
