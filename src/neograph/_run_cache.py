"""Per-run handle/resource cache keyed on ``StateKeys.RUN_ID``.

The inline agent cycle rebuilds its tool-loop preamble ONCE PER SUPERSTEP — each
ReAct turn (agent/tools/parse) is a separate checkpointed node body, so a live
LLM handle, tool instances, and an awaited FROM_RESOURCE fetch are otherwise
re-created every turn. A stateless rebuild is *correct on resume* but wasteful
within a single run. This cache lets those be reused ACROSS the supersteps of one
run while staying two-lifetime-correct (§5):

- **Sound key.** Entries are keyed on ``StateKeys.RUN_ID`` — a framework-minted,
  config-only per-run id minted fresh in ``runner._prepare``/``_aprepare`` and
  NEVER persisted into a checkpoint. Within one run the id is stable across every
  superstep (cache HIT -> reuse); a resume re-runs the pre-engine brain and mints
  a NEW id (cache MISS -> a fresh rebuild/refetch). So a cached entry can never
  outlive the run/process it was built for — the invalidate-on-resume guarantee
  is structural, not a heuristic. ``thread_id`` is deliberately NOT used: it is
  absent without a checkpointer and REUSED across resume, which would serve a
  stale handle into a fresh-process lifetime.
- **No key -> no cache.** When the compiled graph is invoked directly (bypassing
  the runner, so no RUN_ID is minted) there is no sound per-run key, so nothing
  is cached and every call rebuilds. Correct, just unoptimized.
- **Bounded LRU.** Evicting an old run's entry only costs a rebuild on that run's
  next superstep (a perf miss, never a correctness fault), so a long-lived
  process cannot leak unboundedly.

Within one run the cycle's supersteps are sequential (one turn at a time on a
thread), so there is no concurrent build on a single key; the build callback runs
OUTSIDE the lock (it may do I/O) and only the small map mutation is locked.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Any

from neograph._state_keys import StateKeys

# Bound on distinct (run_id, subkey) entries held at once. Generous — an evicted
# still-live run just rebuilds on its next superstep (perf miss only).
_MAX_ENTRIES = 1024

_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
_lock = Lock()
_MISSING = object()


def _run_id(config: Any) -> str | None:
    """The framework-minted per-run id from ``config['configurable']``, or None
    when absent (graph invoked directly, bypassing the runner's mint)."""
    if not config:
        return None
    configurable = config.get("configurable") or {}
    return configurable.get(StateKeys.RUN_ID)


def _lookup(key: tuple[str, str]) -> Any:
    with _lock:
        hit = _cache.get(key, _MISSING)
        if hit is not _MISSING:
            _cache.move_to_end(key)
        return hit


def _store(key: tuple[str, str], value: Any) -> None:
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)


def get_or_build(config: Any, subkey: str, build: Callable[[], Any]) -> Any:
    """Return the cached value for ``(run_id, subkey)``, building + caching on a
    miss. No run id -> always build (never cache): there is no sound per-run key.

    ``build`` runs OUTSIDE the lock; within one run the cycle's supersteps are
    sequential, so a single key is never built concurrently."""
    run_id = _run_id(config)
    if run_id is None:
        return build()
    key = (run_id, subkey)
    hit = _lookup(key)
    if hit is not _MISSING:
        return hit
    value = build()
    _store(key, value)
    return value


async def aget_or_build(
    config: Any, subkey: str, build: Callable[[], Awaitable[Any]]
) -> Any:
    """Async twin of :func:`get_or_build`; ``build`` is awaited on a miss."""
    run_id = _run_id(config)
    if run_id is None:
        return await build()
    key = (run_id, subkey)
    hit = _lookup(key)
    if hit is not _MISSING:
        return hit
    value = await build()
    _store(key, value)
    return value


def clear() -> None:
    """Drop all cached entries. Test hook — keeps cross-test isolation cheap."""
    with _lock:
        _cache.clear()
