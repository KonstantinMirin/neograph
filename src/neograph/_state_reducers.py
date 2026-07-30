"""LangGraph channel reducers.

Extracted from ``state.py`` (neograph-3ffdg.14) as a pure file split — the four
reducers below are unchanged, only their home moved. ``state.py`` re-exports
them, so existing imports keep resolving.

Each one is the merge rule for a state channel: last-write-wins for ordinary
outputs, append for loop results, concat for fan-out barriers, dict-merge for
per-key Each results.
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger()


def _last_write_wins(existing: Any, new: Any) -> Any:
    """Reducer: last write wins (default for sequential nodes)."""
    return new


def _append_loop_result(existing: Any, new: Any) -> list:
    """Reducer: append each loop iteration's result to a list."""
    if existing is None:
        existing = []
    return [*existing, new]


def _concat_reducer(existing: Any, new: Any) -> list:
    """Reducer: concatenate list-valued writes onto an accumulator.

    The single list-append reducer shared by every additive channel:
      - oracle fan-out results (``list[sub.output]``)
      - Each×Oracle tagged (key, result) tuples
      - agent-cycle ToolInteraction records (``tool_log``, per-turn concat)
      - agent-cycle ResourceRef records (``resource_manifest``, per-turn concat)

    A per-turn write is a ``list`` (extend); a single value is appended. These
    four channels were byte-identical functions (neograph-yrph item 4). LangGraph
    keys channels by FIELD NAME, not reducer identity, so one shared operator is
    safe — the same pattern ``_last_write_wins``/``_merge_dicts`` already use
    across many distinct channels. A structural guard bans re-planting a
    byte-identical concat twin.
    """
    if existing is None:
        existing = []
    if isinstance(new, list):
        return existing + new
    return [*existing, new]


def _merge_dicts(existing: Any, new: dict) -> dict:
    """Reducer: merge dicts additively (for fan-out results).

    On duplicate keys, keeps the existing (first) value. Logs a single
    summary instead of per-key warnings (neograph-o0tv: noisy on resume).
    """
    if existing is None:
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(new, dict):
        return existing
    merged = {**existing}
    dupes = []
    for key, val in new.items():
        if key in merged:
            dupes.append(key)
            continue
        merged[key] = val
    if dupes:
        log.debug(
            "each_duplicate_keys", count=len(dupes), keys=dupes[:5], action="kept_existing", truncated=len(dupes) > 5
        )
    return merged
