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
- **Run-end eviction.** ``evict_run`` drops a run's entries the moment the run's
  driver verb returns (wired into the runner verbs' ``finally``), so a loop-bound
  MCP/LLM handle does not linger past its run holding an event-loop-affine object
  until LRU pressure. The bounded LRU below is the backstop, not the primary
  lifecycle.
- **Per-key single-flight.** Two concurrent misses on ONE key (fan-over-agent
  branches under ``arun``) would otherwise both build — double LLM-handle build
  (waste) / double resource fetch (cost; a correctness edge for non-idempotent
  fetchers). A per-key latch minted under the map lock serializes the build:
  the loser blocks, then double-checks the map and reuses the winner's value.
  The async latch is LOOP-AFFINE (keyed by running-loop id) so an
  ``asyncio.Lock`` is never awaited on a loop other than the one it bound to.

Cross-run isolation is structural: distinct runs have distinct RUN_IDs, so their
keys never collide — single-flight dedups WITHIN a key, never across runs. There
is deliberately no mid-run expiry recheck: a cached entry is served for the whole
run; a fresh RUN_ID (resume/replay) is the only thing that forces a refetch.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import Any

from neograph._config_carrier import run_id_of

# Bound on distinct (run_id, subkey) entries held at once. Generous — an evicted
# still-live run just rebuilds on its next superstep (perf miss only).
_MAX_ENTRIES = 1024

_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
_lock = Lock()
_MISSING = object()

# Per-key single-flight latches, all minted UNDER ``_lock``. The sync map keys on
# (run_id, subkey); the async map ALSO keys on the running-loop id so a lock is
# never shared across event loops (awaiting an asyncio.Lock on a foreign loop
# raises). Both are dropped for a run by ``evict_run`` alongside its cache entries.
_latches: dict[tuple[str, str], Lock] = {}
_alatches: dict[tuple[int, str, str], asyncio.Lock] = {}


def _run_id(config: Any) -> str | None:
    """The framework-minted per-run id from ``config['configurable']``, or None
    when absent (graph invoked directly, bypassing the runner's mint).

    Thin alias for the canonical ``_config_carrier.run_id_of`` (DRY-L1 pair)."""
    return run_id_of(config)


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


def _latch(key: tuple[str, str]) -> Lock:
    """Mint-or-fetch the sync single-flight latch for ``key``, under ``_lock``."""
    with _lock:
        latch = _latches.get(key)
        if latch is None:
            latch = Lock()
            _latches[key] = latch
        return latch


def _alatch(key: tuple[str, str]) -> asyncio.Lock:
    """Mint-or-fetch the async single-flight latch for ``key`` ON THE RUNNING
    LOOP, under ``_lock``. Keyed by loop id so the returned ``asyncio.Lock`` is
    only ever awaited on the loop it binds to (loop-affine)."""
    loop_key = (id(asyncio.get_running_loop()), key[0], key[1])
    with _lock:
        latch = _alatches.get(loop_key)
        if latch is None:
            latch = asyncio.Lock()
            _alatches[loop_key] = latch
        return latch


def get_or_build(config: Any, subkey: str, build: Callable[[], Any]) -> Any:
    """Return the cached value for ``(run_id, subkey)``, building + caching on a
    miss. No run id -> always build (never cache): there is no sound per-run key.

    ``build`` runs OUTSIDE the map lock. A per-key latch gives single-flight: on a
    miss the caller takes the latch, DOUBLE-CHECKS the map (a peer may have built
    while it waited), and only then builds — so one key is built exactly once even
    under concurrent misses."""
    run_id = _run_id(config)
    if run_id is None:
        return build()
    key = (run_id, subkey)
    hit = _lookup(key)
    if hit is not _MISSING:
        return hit
    with _latch(key):
        hit = _lookup(key)
        if hit is not _MISSING:
            return hit
        value = build()
        _store(key, value)
        return value


async def aget_or_build(config: Any, subkey: str, build: Callable[[], Awaitable[Any]]) -> Any:
    """Async twin of :func:`get_or_build`; ``build`` is awaited on a miss. The
    single-flight latch is the loop-affine ``asyncio.Lock`` from
    :func:`_alatch`."""
    run_id = _run_id(config)
    if run_id is None:
        return await build()
    key = (run_id, subkey)
    hit = _lookup(key)
    if hit is not _MISSING:
        return hit
    async with _alatch(key):
        hit = _lookup(key)
        if hit is not _MISSING:
            return hit
        value = await build()
        _store(key, value)
        return value


def evict_run(run_id: str) -> None:
    """Drop every cached entry and single-flight latch for ``run_id``.

    Called from each runner verb's ``finally`` the instant the run's driver
    returns, so loop-bound handles do not linger past their run. Idempotent — a
    run with no cached entries (e.g. graph invoked directly, or a scripted-only
    run) is a no-op."""
    with _lock:
        for key in [k for k in _cache if k[0] == run_id]:
            del _cache[key]
        for key in [k for k in _latches if k[0] == run_id]:
            del _latches[key]
        for akey in [k for k in _alatches if k[1] == run_id]:
            del _alatches[akey]


def clear() -> None:
    """Drop all cached entries and latches. Test hook — keeps cross-test
    isolation cheap."""
    with _lock:
        _cache.clear()
        _latches.clear()
        _alatches.clear()
