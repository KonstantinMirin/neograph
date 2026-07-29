"""Checkpointer <-> driver compatibility guard.

Extracted from ``runner.py`` (neograph-3ffdg.9) as a pure file split — the
functions below are unchanged, only their home moved.

A sync checkpointer under an async driver (or vice versa) fails loud here rather
than corrupting a thread's checkpoint history silently.
"""

from __future__ import annotations

import structlog

from neograph._compiled import CompiledNeograph
from neograph.errors import ConfigurationError

log = structlog.get_logger()


def _required_checkpointer_driver(checkpointer: object) -> str | None:
    """Classify which driver a saver can ONLY be used with, or ``None`` if it is
    dual-capable / unknown.

    Returns ``"async"`` for async-only savers, ``"sync"`` for sync-only savers,
    ``None`` otherwise. The two mismatch cases each fail badly at the checkpoint
    probe, so the guard needs one authoritative classifier:

    * ``"async"`` — async-only savers (``AsyncSqliteSaver``, ``AsyncPostgresSaver``)
      capture an asyncio event loop at construction and bridge their *sync*
      ``get_tuple`` onto it via ``run_coroutine_threadsafe``. Driven from a sync
      ``run()`` that loop is not running in a background thread, so ``get_tuple``
      BLOCKS forever (or, when called from within the loop's own thread, raises
      ``InvalidStateError``). The bound event loop is the mechanism-level marker.
    * ``"sync"`` — sync-only savers (``SqliteSaver``, ``PostgresSaver``) follow
      LangGraph's convention of stubbing the async API: ``aget_tuple`` raises
      ``NotImplementedError``, which surfaces raw deep in ``arun()``'s checkpoint
      probe. We detect the stub without invoking it (invoking needs an event
      loop) by reading the method source for the ``NotImplementedError`` raise —
      the signal LangGraph's base ``BaseCheckpointSaver.aget_tuple`` and the
      sqlite/postgres sync savers share. A dual-capable saver (``MemorySaver``)
      implements ``aget_tuple`` for real, so its source carries no such raise.

    The event-loop check runs first, so a saver is never classified as both.
    """
    import asyncio

    if isinstance(getattr(checkpointer, "loop", None), asyncio.AbstractEventLoop):
        return "async"
    aget = getattr(type(checkpointer), "aget_tuple", None)
    if aget is None:
        return None
    import inspect

    try:
        source = inspect.getsource(aget)
    except (OSError, TypeError) as exc:
        # Introspection failed — we can't classify this saver's driver. Do NOT
        # raise (third-party savers are legitimate), but do NOT stay silent
        # either: the returned None is the mismatch guard's SOLE input, so a
        # None here BYPASSES the sync/async protection for this saver. Warn so
        # the bypass is traceable (7ymj).
        #
        # Capability-probing (does the saver own aget_tuple/get_tuple natively?)
        # was evaluated as a replacement for this source-sniffing and REJECTED:
        # LangGraph's sync SqliteSaver/PostgresSaver define their OWN aget_tuple
        # stub (qualname 'SqliteSaver.aget_tuple') that raises NotImplementedError,
        # so a method-owner probe cannot distinguish a sync-only saver from a
        # dual-capable one (InMemorySaver also owns aget_tuple). Reading the
        # method source for the NotImplementedError raise is the only signal
        # that separates the stub from a real async implementation.
        log.warning(
            "checkpointer_driver_introspection_failed",
            saver=type(checkpointer).__name__,
            error=str(exc),
            hint=(
                "could not read aget_tuple source to classify sync/async driver; "
                "the sync/async mismatch guard is BYPASSED for this saver"
            ),
        )
        return None
    return "sync" if "NotImplementedError" in source else None


def _assert_checkpointer_matches_driver(
    graph: CompiledNeograph,
    *,
    is_async: bool,
) -> None:
    """Fail loud at run/arun ENTRY when the checkpointer's sync/async capability
    does not match the driver.

    The wrong-driver mismatch otherwise fails badly: a sync ``run()`` with an
    async-only saver BLOCKS on the bridged ``get_tuple`` (and the swallow in
    ``_has_existing_checkpoint`` can even discard the failure and SILENTLY start
    a fresh run that ignores an existing checkpoint), while an ``arun()`` with a
    sync-only saver raises a raw ``NotImplementedError``. Detecting the mismatch
    here — before any checkpoint I/O — replaces both with a clear
    ``ConfigurationError`` that names the right driver. Called from the shared
    ``_prepare`` / ``_aprepare`` brains, so it covers every driver verb.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return
    required = _required_checkpointer_driver(checkpointer)
    saver_name = type(checkpointer).__name__
    if not is_async and required == "async":
        raise ConfigurationError.build(
            "Async-only checkpointer passed to a synchronous driver (run/stream).",
            found=f"{saver_name} (async-only)",
            expected="a synchronous checkpointer (e.g. SqliteSaver, PostgresSaver, MemorySaver)",
            hint=(
                "Drive this saver with arun()/astream(), or pass a sync saver to "
                "run()/stream(). An async-only saver bridges get_tuple() onto its "
                "own event loop, which blocks a synchronous run()."
            ),
        )
    if is_async and required == "sync":
        raise ConfigurationError.build(
            "Sync-only checkpointer passed to an async driver (arun/astream).",
            found=f"{saver_name} (sync-only)",
            expected="an async checkpointer (e.g. AsyncSqliteSaver, AsyncPostgresSaver, MemorySaver)",
            hint=(
                "Drive this saver with run()/stream(), or pass an async saver to "
                "arun()/astream(). A sync-only saver's aget_tuple() raises "
                "NotImplementedError under the async driver."
            ),
        )
