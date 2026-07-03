"""``emit_progress`` — typed custom progress/domain events from nodes.

A neograph-sanctioned seam over LangGraph's ``stream_mode='custom'`` writer
(``get_stream_writer``). A node body (scripted / think / agent / tool) calls
``emit_progress(model)`` with a typed Pydantic event; when the graph is driven
by ``stream``/``astream`` consuming ``custom``, the event surfaces in the stream
with a stable envelope.

Event envelope (documented, stable shape)::

    {"neograph_event": "progress", "event_type": "<ClassName>", "data": <model.model_dump(mode="json")>}

The envelope key is deliberately NOT ``neo_*``-prefixed: a custom event is a
USER payload, not framework state, so ``runner._finalize_chunk`` passes it
through untouched, and it never enters state or the schema fingerprint.

Review L1 — must NOT vanish silently. Under a non-streaming driver
(``run``/``arun``, or ``stream``/``astream`` without ``custom``) the no-op
writer would discard the event. ``emit_progress`` detects this via the
``STREAM_CUSTOM`` config flag (set by the streaming verbs) and warns ONCE per
process instead of vanishing. It still calls the writer whenever one is present,
so an event is never dropped when a real consumer exists (e.g. a raw
``graph.astream`` reach-around) — the warning is a best-effort "you probably
forgot to consume the stream" hint keyed on the neograph flag.
"""

from __future__ import annotations

import warnings

from pydantic import BaseModel

from neograph._state_keys import StateKeys
from neograph.errors import ConfigurationError

# Warn-once dedup. Per-process (review L1 accepts once per process/run). Reset
# by tests via ``_reset_warned`` so each case observes the first-emit warning.
_warned = False


def _reset_warned() -> None:
    """Test hook: clear the warn-once latch so the next drop warns again."""
    global _warned
    _warned = False


def _warn_progress_dropped() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    warnings.warn(
        "emit_progress() called under a non-streaming driver — the progress "
        "event was not consumed. Drive the graph with neograph.stream()/"
        "astream(stream_mode='custom') (or include 'custom' in stream_mode) to "
        "receive progress events. This warns once per process.",
        UserWarning,
        stacklevel=3,
    )


def _custom_stream_active() -> bool:
    """True if the current run is driven by a neograph streaming verb consuming
    ``stream_mode='custom'`` (the STREAM_CUSTOM flag is set in config)."""
    from langgraph.config import get_config

    try:
        config = get_config()
    except RuntimeError:
        return False
    if not isinstance(config, dict):
        return False
    configurable = config.get("configurable") or {}
    return bool(configurable.get(StateKeys.STREAM_CUSTOM))


def emit_progress(event: BaseModel) -> None:
    """Emit a typed progress/domain event from inside a node body or tool.

    ``event`` must be a Pydantic ``BaseModel`` (the typed contract). The event
    is wrapped in the documented envelope and written to LangGraph's custom
    stream. Callable from any node type and from sync or async bodies.

    Under a non-streaming driver the event cannot be delivered; ``emit_progress``
    warns once (review L1) rather than vanishing silently. It never raises for a
    missing consumer.
    """
    if not isinstance(event, BaseModel):
        raise ConfigurationError.build(
            f"emit_progress() requires a Pydantic BaseModel event, got "
            f"{type(event).__name__}",
            hint="Define a typed event model (class MyEvent(BaseModel): ...) and "
            "pass an instance so the event has a stable, documented shape.",
        )

    payload = {
        "neograph_event": "progress",
        "event_type": type(event).__name__,
        "data": event.model_dump(mode="json"),
    }

    from langgraph.config import get_stream_writer

    try:
        writer = get_stream_writer()
    except RuntimeError:
        # Called outside any runnable context (not inside a node at all).
        writer = None

    if writer is None or not _custom_stream_active():
        _warn_progress_dropped()

    if writer is not None:
        writer(payload)
