"""Schema-aware checkpoint divergence detection and auto-rewind.

Extracted from ``runner.py`` (neograph-3ffdg.9) as a pure file split — the
functions below are unchanged, only their home moved. The sync and async twins
are deliberately reunited here so a change to one is visibly adjacent to the
other; their pairing is pinned by the async-dispatch twin guards.

Fail-loud contract preserved verbatim: when a schema change invalidates nodes but
NO checkpoint has an invalidated node still pending, ``_raise_no_rewind_point``
raises rather than silently resuming from the tip and handing back stale results.
"""

from __future__ import annotations

from typing import Any, NoReturn

import structlog
from pydantic import ValidationError

from neograph._compiled import CompiledNeograph
from neograph._ir_branch import iter_with_arms
from neograph._state_keys import StateKeys
from neograph.errors import CheckpointSchemaError
from neograph.naming import field_name_for

log = structlog.get_logger()


def _decide_checkpoint_schema(
    graph: CompiledNeograph,
    saved: Any,
    *,
    auto_resume: bool,
) -> set[str] | None:
    """Pure checkpoint-schema decision shared by both verify twins.

    Given the fetched checkpoint tuple ``saved`` (sync ``get_tuple`` / async
    ``aget_tuple`` — the only twin difference), decide whether the stored schema
    fingerprint diverged from the current graph:

    - returns ``None`` when there is nothing to do (no fingerprint, no
      checkpoint, pre-fingerprint checkpoint, or a clean match);
    - returns the invalidated node set (possibly empty) to rewind when the
      fingerprints differ and ``auto_resume`` is True;
    - raises :class:`CheckpointSchemaError` when they differ and ``auto_resume``
      is False.

    Single-sites the mismatch error message and the auto-resume log so a change
    to either lands once (neograph-ykun / DRY-09).
    """
    current_fp = graph.schema_fingerprint
    if current_fp is None or saved is None:
        return None

    channel_values = saved.checkpoint.get("channel_values", {})
    stored_fp = None
    if isinstance(channel_values, dict):
        stored_fp = channel_values.get(StateKeys.SCHEMA_FINGERPRINT)
    elif hasattr(channel_values, "get"):
        stored_fp = channel_values.get(StateKeys.SCHEMA_FINGERPRINT)

    if stored_fp is None or stored_fp == "" or stored_fp == current_fp:
        return None

    invalidated = _compute_invalidated_nodes(graph, channel_values)

    if not auto_resume:
        raise CheckpointSchemaError(
            f"Checkpoint schema fingerprint mismatch: "
            f"stored={stored_fp!r}, current={current_fp!r}. "
            f"Invalidated nodes: {sorted(invalidated) if invalidated else 'all'}. "
            f"Invalidate the checkpoint or migrate the state.",
            invalidated_nodes=invalidated,
        )

    log.info(
        "auto_resume_schema_change",
        invalidated=sorted(invalidated),
        stored_fp=stored_fp,
        current_fp=current_fp,
    )
    return invalidated


def _raise_no_rewind_point(invalidated: set[str]) -> NoReturn:
    """Fail loud when auto_resume finds invalidated nodes but no rewind point.

    Single-sited between the sync and async rewind twins. The schema changed and
    ``invalidated`` is non-empty, yet no checkpoint in the thread's history has
    any invalidated node pending in ``.next`` — because the history was pruned,
    or every invalidated node already ran to completion. Resuming from the tip
    would silently re-hand the caller stale results (the durability pitch's one
    false spot). We refuse: this is a resume-time precondition failure, surfaced
    BEFORE any node re-executes, with ``invalidated_nodes`` as the actionable
    signal — the same contract as ``auto_resume=False``. See neograph-v63o.
    """
    raise CheckpointSchemaError(
        "auto_resume could not find a rewind point. The checkpoint schema changed "
        f"and these nodes are invalidated: {sorted(invalidated)}, but no checkpoint "
        "in this thread's history has any of them pending in `.next` (history "
        "pruned, or every invalidated node already ran to completion). Resuming "
        "from the tip would silently skip them and return stale results. Start a "
        "new thread_id, or invalidate this checkpoint to re-run from scratch.",
        invalidated_nodes=invalidated,
    )


def _raise_incompatible_schema(invalidated: set[str], exc: ValidationError) -> NoReturn:
    """Fail clean when a NON-COERCIBLE schema change makes the checkpoint history
    un-materializable into the current state schema.

    Single-sited between the sync and async rewind twins. LangGraph's
    ``get_state_history`` re-validates every historical snapshot against the
    CURRENT state model to compute its ``.next``. A coercible widening (int ->
    float) validates cleanly, so the walk succeeds and the rewind proceeds. A
    non-coercible change (int -> str) makes pydantic reject the stored value, and
    the raw ``ValidationError`` bubbles from INSIDE the walk before any rewind
    decision runs. We translate it into the same schema-divergence signal the
    rest of the seam speaks — ``CheckpointSchemaError`` carrying
    ``invalidated_nodes`` — surfaced BEFORE any node re-executes, the same
    contract as ``auto_resume=False`` and ``_raise_no_rewind_point``. See
    neograph-1gdw.
    """
    raise CheckpointSchemaError(
        "auto_resume cannot rewind across an INCOMPATIBLE schema change. The "
        f"checkpoint schema changed and these nodes are invalidated: {sorted(invalidated)}, "
        "but the stored checkpoint's values cannot be coerced into the new state "
        f"schema ({type(exc).__name__} while re-materializing the thread's history). "
        "A coercible widening (e.g. int -> float) resumes cleanly; a non-coercible "
        "change (e.g. int -> str) cannot. Start a new thread_id, or invalidate this "
        "checkpoint to re-run from scratch.",
        invalidated_nodes=invalidated,
    ) from exc


def _compute_invalidated_nodes(graph: CompiledNeograph, channel_values: Any) -> set[str]:
    """Compute which nodes changed + their transitive descendants.

    Compares per-node fingerprints stored in the checkpoint against the
    current graph's per-node fingerprints, then walks the construct's
    producer→consumer adjacency (keyed by state-field name) to return the
    full transitive closure.
    """
    current_nfp = graph.node_fingerprints
    if current_nfp is None:
        return set()

    stored_nfp = None
    if isinstance(channel_values, dict):
        stored_nfp = channel_values.get(StateKeys.NODE_FINGERPRINTS)
    elif hasattr(channel_values, "get"):
        stored_nfp = channel_values.get(StateKeys.NODE_FINGERPRINTS)

    if not stored_nfp or not isinstance(stored_nfp, dict):
        return set()

    # Find directly changed nodes (by state-field name).
    changed: set[str] = set()
    for node_field, current_fp in current_nfp.items():
        stored_fp = stored_nfp.get(node_field)
        if stored_fp is not None and stored_fp != current_fp:
            changed.add(node_field)

    if not changed:
        return set()

    construct = graph.construct
    if construct is None:
        return changed

    adjacency = _build_producer_consumer_adjacency(construct)
    return _transitive_closure(changed, adjacency)


def _build_producer_consumer_adjacency(construct: Any) -> dict[str, set[str]]:
    """Map upstream-producer field-name → set of consumer field-names.

    Modifier-bearing nodes participate via their state-field names — the
    same key the per-node fingerprint store uses. Dict-form outputs are
    registered under both their composite key (``{field}_{output_key}``)
    and their base field name, so consumers that read either form are
    captured.
    """
    adjacency: dict[str, set[str]] = {}

    def add_edge(producer_key: str, consumer_field: str) -> None:
        adjacency.setdefault(producer_key, set()).add(consumer_field)

    # iter_with_arms expands _BranchNode sentinels so a bare arm consumer
    # contributes its producer->consumer edges — otherwise a change to an
    # upstream field would not mark the arm node for checkpoint re-execution.
    # See neograph-vn5f (site 5).
    for item in iter_with_arms(construct):
        consumer_name = getattr(item, "name", None)
        if consumer_name is None:
            continue
        consumer_field = field_name_for(consumer_name)

        inputs = getattr(item, "inputs", None)
        if isinstance(inputs, dict):
            for upstream_name in inputs:
                add_edge(upstream_name, consumer_field)

        # Each.over names a producer field (root may contain dotted path).
        ms = getattr(item, "modifier_set", None)
        each = getattr(ms, "each", None) if ms is not None else None
        if each is not None:
            over = getattr(each, "over", None)
            if isinstance(over, str) and over:
                root = over.split(".", 1)[0]
                add_edge(root, consumer_field)

        # context= references upstream fields by name.
        for ctx_name in getattr(item, "context", None) or ():
            add_edge(field_name_for(ctx_name), consumer_field)

    return adjacency


def _transitive_closure(seeds: set[str], adjacency: dict[str, set[str]]) -> set[str]:
    """BFS through producer→consumer adjacency from ``seeds``."""
    closure: set[str] = set(seeds)
    frontier: list[str] = list(seeds)
    while frontier:
        producer = frontier.pop()
        for consumer in adjacency.get(producer, ()):
            if consumer not in closure:
                closure.add(consumer)
                frontier.append(consumer)
    return closure
