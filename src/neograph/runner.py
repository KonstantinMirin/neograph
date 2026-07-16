"""Runner — execute compiled graphs with checkpointing.

result = run(graph, input={"node_id": "BR-RW-042", "project_root": "/path"})

# With shared resources in config:
result = run(graph,
    input={"node_id": "BR-RW-042"},
    config={"configurable": {"project_root": "/path", "rate_limiter": my_limiter}})

# Resume after Operator interrupt:
result = run(graph, resume={"approved": True}, config=config)
"""

from __future__ import annotations

import os
from typing import Any, NoReturn
from uuid import uuid4

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import ValidationError

from neograph._compiled import CompiledNeograph
from neograph._ir_branch import iter_with_arms
from neograph._llm_config import _coerce_llm_config
from neograph._run_cache import evict_run
from neograph._state_keys import StateKeys, _strip_internals
from neograph.construct import Construct, iter_nodes
from neograph.errors import CheckpointSchemaError, ConfigurationError, ExecutionError
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.naming import field_name_for
from neograph.node import Node

log = structlog.get_logger()

# LangGraph's default per-invoke superstep ceiling. Agent/act nodes compile to an
# inline ReAct cycle (2 supersteps/turn: agent + tools), so a loop near its
# max_iterations bound can exceed this ceiling BEFORE the graceful budget-exhaust
# forced-final fires. This bites most in a NESTED agent (a sub-construct invoke
# defaults to 25); the raised top-level limit propagates into the child invoke.
# _ensure_agent_recursion_limit raises the ceiling so the forced-final edge is
# reachable at any max_iterations.
_LANGGRAPH_DEFAULT_RECURSION_LIMIT = 25
# Supersteps a single agent turn costs (agent node + tools node).
_SUPERSTEPS_PER_AGENT_TURN = 2
# Per-agent overhead beyond the turns: the parse node + the forced-final turn.
_AGENT_CYCLE_OVERHEAD = 3


def _member_hop_cost(node: Node) -> int:
    """Superstep cost of ONE hop through a Portal mesh member.

    Atomic members (scripted/think/raw) cost exactly 1 superstep per hop —
    the pre-nnds9 assumption. An agent/act member's hop re-enters its own
    ReAct cycle (agent<->tools loop, up to ``max_iterations`` turns) before
    its parse node ever emits the mesh ``Command`` — so ONE hop through an
    agent/act member can itself cost ``max_iterations * 2 + overhead``
    supersteps. Mirrors the flat per-node agent/act cost computed in
    ``_ensure_agent_recursion_limit``.
    """
    if node.mode in ("agent", "act"):
        max_iters = _coerce_llm_config(node.llm_config).max_iterations
        return max_iters * _SUPERSTEPS_PER_AGENT_TURN + _AGENT_CYCLE_OVERHEAD
    return 1


def _mesh_hop_cost(construct: Construct) -> int:
    """Sum the worst-case superstep cost of every Portal mesh, recursing
    sub-constructs.

    A K-hop mesh consumes up to K hops (one ``Command(goto)`` per hop), each
    landing on WHICHEVER member the route target names — so the worst case is
    every hop landing on the mesh's MOST EXPENSIVE member (per
    ``_member_hop_cost``), not a flat 1-superstep-per-hop assumption (which
    undercounts an agent/act member's own internal ReAct supersteps). A
    contiguous run of Portal-modified sibling Nodes is ONE mesh (design
    §3.1 r2) and only its ENTRY (first member) carries the real ``max_hops``
    (entry-only, T1) — non-entry members default to 10. ``iter_nodes``
    leaf-flattens and cannot identify the entry (nor a mesh whose entry left
    ``max_hops`` at the default), so this uses the level-preserving
    ``iter_with_arms`` walk that mirrors the compiler's mesh detection.
    """
    total = 0
    current_run: list[Node] = []

    def _flush() -> None:
        nonlocal total
        if not current_run:
            return
        entry_portal = current_run[0].modifier_set.portal
        if entry_portal is not None:
            per_hop = max(_member_hop_cost(m) for m in current_run)
            total += entry_portal.max_hops * per_hop
        current_run.clear()

    for item in iter_with_arms(construct):
        if isinstance(item, Construct):
            _flush()
            total += _mesh_hop_cost(item)
            continue
        if isinstance(item, Node) and classify_modifiers(item)[0] == ModifierCombo.PORTAL:
            current_run.append(item)
        else:
            _flush()
    _flush()
    return total


def _portal_mesh_member_ids(construct: Construct) -> set[int]:
    """``id()`` of every Node that is a Portal mesh member, recursing
    sub-constructs.

    An agent/act mesh member's cost is captured ENTIRELY by ``_mesh_hop_cost``
    (its per-hop ReAct-cycle cost, times the mesh's ``max_hops``) — so the flat
    per-node ``agent_cost`` loop in ``_ensure_agent_recursion_limit`` must
    EXCLUDE mesh members, or an agent/act mesh member's cost is double-counted
    (once flat, once mesh-aware).
    """
    ids: set[int] = set()
    for item in iter_with_arms(construct):
        if isinstance(item, Construct):
            ids |= _portal_mesh_member_ids(item)
        elif isinstance(item, Node) and classify_modifiers(item)[0] == ModifierCombo.PORTAL:
            ids.add(id(item))
    return ids


def _ensure_agent_recursion_limit(
    graph: CompiledNeograph,
    config: RunnableConfig | None,
) -> RunnableConfig | None:
    """Raise ``recursion_limit`` so an agent/act cycle OR a Portal mesh can
    reach its graceful budget-exhaust edge instead of hitting LangGraph's default
    superstep ceiling first.

    Each STANDALONE agent/act node's cycle can cost ``max_iterations * 2 +
    overhead`` supersteps; each Portal mesh can cost up to
    ``entry.max_hops * worst_member_hop_cost`` supersteps (an agent/act mesh
    member's own hop cost, per ``_mesh_hop_cost``/``_member_hop_cost``). Both
    run in distinct supersteps, so their costs ADD across the run. The floor
    sums every STANDALONE agent/act node's worst case AND every mesh's
    worst-case hop budget on top of the default (which already covers the
    surrounding non-agent nodes) — an agent/act node that is ALSO a Portal
    mesh member is excluded from the flat per-node sum
    (``_portal_mesh_member_ids``) so its cost is not double-counted (once
    flat, once mesh-aware). Only RAISES to the floor — a larger
    user-supplied ``recursion_limit`` is kept. Pure config mutation (no
    engine verb); shared verbatim by ``_prepare`` and ``_aprepare``.
    """
    construct = getattr(graph, "construct", None)
    if construct is None:
        return config

    mesh_member_ids = _portal_mesh_member_ids(construct)
    agent_cost = 0
    for node in iter_nodes(construct):
        if isinstance(node, Node) and node.mode in ("agent", "act") and id(node) not in mesh_member_ids:
            max_iters = _coerce_llm_config(node.llm_config).max_iterations
            agent_cost += max_iters * _SUPERSTEPS_PER_AGENT_TURN + _AGENT_CYCLE_OVERHEAD

    mesh_cost = _mesh_hop_cost(construct)

    if agent_cost == 0 and mesh_cost == 0:
        return config  # no agent/act nodes and no mesh — leave the default untouched

    floor = _LANGGRAPH_DEFAULT_RECURSION_LIMIT + agent_cost + mesh_cost
    current = (config or {}).get("recursion_limit", _LANGGRAPH_DEFAULT_RECURSION_LIMIT)
    if current >= floor:
        return config  # user asked for at least what agents/mesh need — keep theirs

    new_config: RunnableConfig = {**(config or {})}
    new_config["recursion_limit"] = floor
    return new_config


def _preflight_di_check(graph: CompiledNeograph, config: RunnableConfig) -> None:
    """Validate that all required DI params are present before starting any node.

    ``compile()`` records ``required_di`` on the CompiledNeograph — a dict with
    "input" and "config" sets of required param names. This check runs before
    graph.invoke() so missing params fail at the gate, not mid-pipeline.
    """
    required = graph.required_di
    if required is None:  # pragma: no cover — required_di is always populated
        return

    configurable = config.get("configurable", {})
    missing_input = required.get("input", set()) - set(configurable.keys())
    missing_config = required.get("config", set()) - set(configurable.keys())

    if missing_input or missing_config:
        parts = []
        if missing_input:
            parts.append(f"missing from run(input=): {sorted(missing_input)}")
        if missing_config:
            parts.append(f"missing from config['configurable']: {sorted(missing_config)}")
        raise ExecutionError.build(
            f"Required DI parameters not provided: {'; '.join(parts)}",
            hint="Add the missing keys to run(input={{...}}) or config={{'configurable': {{...}}}}",
        )


def _inject_input_to_config(
    input: dict[str, Any],
    config: RunnableConfig | None,
) -> RunnableConfig:
    """Merge initial input fields into config["configurable"].

    Every node function receives config — this ensures pipeline metadata
    (node_id, project_root, etc.) is accessible via config["configurable"]
    without reaching into state.
    """
    config = config or {}
    configurable = config.get("configurable", {})
    # Input fields become configurable (input takes precedence)
    merged = {**configurable, **input}
    return {**config, "configurable": merged}


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


def _verify_checkpoint_schema(graph: CompiledNeograph, config: RunnableConfig, *, auto_resume: bool = True) -> None:
    """Verify checkpoint state schema matches the current graph.

    Compares the neo_schema_fingerprint stored in the checkpoint against
    the fingerprint computed at compile time. When auto_resume is True,
    rewinds to the checkpoint before the earliest changed node and re-invokes.
    When False, raises CheckpointSchemaError. Fetch (sync ``get_tuple``) is the
    only divergence from the async twin; the decision is shared via
    ``_decide_checkpoint_schema``.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None or graph.schema_fingerprint is None:
        return

    saved = checkpointer.get_tuple(config)
    invalidated = _decide_checkpoint_schema(graph, saved, auto_resume=auto_resume)
    if invalidated is not None:
        _auto_resume_from_divergence(graph, config, invalidated)


def _auto_resume_from_divergence(
    graph: CompiledNeograph,
    config: RunnableConfig,
    invalidated: set[str],
) -> None:
    """Rewind checkpoint to before the earliest invalidated node.

    Uses LangGraph time-travel: walks state_history to find the checkpoint
    where the earliest invalidated node was about to execute (in ``next``).
    Overwrites the main config's checkpoint_id to point to that checkpoint,
    so the subsequent ``invoke(None, config)`` resumes from the rewind point.

    Fail-loud: if ``invalidated`` is non-empty but no snapshot has an
    invalidated node pending in ``.next``, raises ``CheckpointSchemaError``
    (via ``_raise_no_rewind_point``) rather than silently resuming from the
    tip with stale results. See neograph-v63o.

    Fail-clean on a NON-COERCIBLE change: ``get_state_history`` re-materializes
    every historical snapshot into the CURRENT state schema (to compute each
    ``.next``). A coercible widening (int -> float) validates and the walk
    proceeds; a non-coercible change (int -> str) makes pydantic reject the
    stored value and raises a raw ``ValidationError`` from INSIDE the walk. We
    translate that into the same schema-divergence signal the rest of the seam
    speaks — ``CheckpointSchemaError(invalidated_nodes=...)`` via
    ``_raise_incompatible_schema`` — surfaced BEFORE any node re-executes. See
    neograph-1gdw.
    """
    if not invalidated:
        return

    # ``get_state_history`` yields newest-first. We want the OLDEST checkpoint
    # whose ``next`` intersects the invalidated set — that's the rewind point
    # that re-executes every invalidated node, not just the latest one.
    rewind_checkpoint_id = None
    try:
        for state_snapshot in graph.get_state_history(config):
            next_nodes = set(state_snapshot.next)
            if next_nodes & invalidated:
                candidate = state_snapshot.config.get("configurable", {}).get("checkpoint_id")
                if candidate is not None:
                    rewind_checkpoint_id = candidate
    except ValidationError as exc:
        _raise_incompatible_schema(invalidated, exc)
    if rewind_checkpoint_id is None:
        _raise_no_rewind_point(invalidated)
    config.setdefault("configurable", {})["checkpoint_id"] = rewind_checkpoint_id


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


def _has_existing_checkpoint(graph: CompiledNeograph, config: RunnableConfig) -> bool:
    """Check if a checkpoint exists for this thread_id.

    Returns True if the graph has a checkpointer and it contains saved state
    for the thread specified in config. Used to decide whether to resume from
    checkpoint or start a new execution.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return False
    try:
        saved = checkpointer.get_tuple(config)
    except (AttributeError, TypeError, KeyError) as exc:
        # yc38: a checkpoint READ that raises means the stored state is corrupt
        # or the saver is misconfigured — NOT "no checkpoint". Absent state
        # returns None from get_tuple; a raise is genuine corruption. Silently
        # returning False here would start a FRESH run that ignores durable
        # state (the durability pitch's worst false spot), so surface it (7ymj).
        raise ConfigurationError.build(
            "checkpoint read failed",
            found=f"{type(checkpointer).__name__}.get_tuple raised {type(exc).__name__}: {exc}",
            hint=(
                "the stored checkpoint appears corrupt or the saver is "
                "misconfigured; inspect the checkpoint backend for this thread_id"
            ),
        ) from exc
    return saved is not None and bool(saved.checkpoint.get("channel_versions"))


def _prepare_resume_config(config: RunnableConfig | None) -> RunnableConfig | None:
    """Re-inject stashed input into config on resume so FromInput DI resolves
    for post-interrupt nodes. Pure (no I/O); shared by run() and arun()."""
    if config is not None:
        neo_input = config.get("configurable", {}).get(StateKeys.CONFIG_INPUT)
        if neo_input is not None:
            config = _inject_input_to_config(neo_input, config)
    return config


def _prepare_new_input(
    graph: CompiledNeograph,
    input: dict[str, Any],
    config: RunnableConfig | None,
) -> tuple[dict[str, Any], RunnableConfig]:
    """Prep for a new execution: stash CONFIG_INPUT in the caller's config,
    inject input into config for DI, defensively copy input, and inject the
    schema/node fingerprints for checkpoint storage. Pure (no I/O); shared by
    run() and arun() so the two drivers cannot fork on input handling."""
    if config is None:
        config = {}
    configurable = config.setdefault("configurable", {})
    configurable[StateKeys.CONFIG_INPUT] = input
    config = _inject_input_to_config(input, config)

    # Defensive copy: framework keys must not leak into the caller's dict.
    input = {**input}

    fp = graph.schema_fingerprint
    if fp is not None:
        input[StateKeys.SCHEMA_FINGERPRINT] = fp
    node_fps = graph.node_fingerprints
    if node_fps is not None:
        input[StateKeys.NODE_FINGERPRINTS] = node_fps
    return input, config


def _mark_stream_custom(config: RunnableConfig | None) -> RunnableConfig:
    """Return a config whose ``configurable`` carries the STREAM_CUSTOM flag.

    Set by the streaming verbs when the driver consumes ``stream_mode='custom'``
    so ``emit_progress`` can distinguish a live progress consumer from a
    non-streaming driver (review L1). Builds a fresh dict — never mutates the
    caller's config in place. The flag is a config['configurable'] entry, so it
    never enters state and cannot touch the schema fingerprint."""
    config = config or {}
    configurable = {**config.get("configurable", {}), StateKeys.STREAM_CUSTOM: True}
    return {**config, "configurable": configurable}


def _mint_run_id(config: RunnableConfig | None) -> RunnableConfig:
    """Return a config whose ``configurable`` carries a FRESH per-run id.

    Minted by the pre-engine brains (``_prepare`` / ``_aprepare``) once per
    execution attempt, so it is stable across every superstep of the run and
    re-minted on resume (resume re-enters ``_prepare`` -> a new id). Mirrors
    ``_mark_stream_custom`` EXACTLY: builds a fresh dict — never mutates the
    caller's config in place — so two ``arun`` calls sharing one config dict each
    get their own id. ``RUN_ID`` is a config['configurable'] entry, so it never
    enters state and cannot touch the schema fingerprint or persist in a
    checkpoint. NOT accepted from the caller — always framework-minted via
    ``uuid4().hex`` — which is what keeps it fresh-per-attempt."""
    config = config or {}
    configurable = {**config.get("configurable", {}), StateKeys.RUN_ID: uuid4().hex}
    return {**config, "configurable": configurable}


def _wants_custom(stream_mode: str | list[str]) -> bool:
    """True if ``stream_mode`` requests LangGraph's ``custom`` channel."""
    if isinstance(stream_mode, str):
        return stream_mode == "custom"
    return "custom" in stream_mode


# ── observe= : opt-in Langfuse auto-attach (per-run config merge + finalize flush)
#
# A THIN-VERB concern (docs/design/three-layer-principle): observe= merges a
# CallbackHandler into config BEFORE the engine verb and flushes AFTER it — it
# NEVER wraps the engine. langfuse is the OPTIONAL [langfuse] extra, so every
# langfuse import here is FUNCTION-LOCAL inside the observe path (enforced by
# TestNoModuleLevelLangfuseImports); the core import graph stays langfuse-free.


def _observe_wants_langfuse(observe: bool | str | None) -> bool:
    """Normalize the ``observe`` argument to 'is Langfuse requested?'.

    ``None`` / ``False`` -> off; ``True`` / ``'langfuse'`` -> on; any other string
    is an explicit misconfiguration and raises (fail-loud on a typo'd backend)."""
    if observe is None or observe is False:
        return False
    if observe is True or observe == "langfuse":
        return True
    raise ConfigurationError.build(
        f"unknown observe backend {observe!r}",
        hint="use observe=True or observe='langfuse' (the only backend today).",
    )


def _langfuse_keys_present() -> bool:
    """Both Langfuse keys must be set before we attach or flush.

    Gating on BOTH LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY is the clean-no-op
    boundary: a half-configured handler warns and silently drops traces, and
    ``get_client().flush()`` would flush a mis-configured client. Absent/partial
    keys -> observe is a no-op, so offline/CI stays green."""
    return bool(os.environ.get("LANGFUSE_SECRET_KEY")) and bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))


def _merge_observe_callbacks(config: RunnableConfig | None, observe: bool | str | None) -> RunnableConfig | None:
    """Return a config with a Langfuse ``CallbackHandler`` MERGED into
    ``callbacks`` — fresh dict, fresh list, never mutating the caller's, never
    clobbering a user-supplied handler.

    No-op (returns the config unchanged) when observe is off or the env gate
    fails. Deduplicates: if a Langfuse handler is already present (the user wired
    one manually), no second one is added."""
    if not _observe_wants_langfuse(observe) or not _langfuse_keys_present():
        return config

    from langfuse.langchain import CallbackHandler  # function-local: optional extra

    config = config or {}
    existing = config.get("callbacks")
    if existing is None:
        existing_list: list[Any] = []
    elif isinstance(existing, list):
        existing_list = existing
    else:
        # A BaseCallbackManager (or other non-list) — attaching to it would mean
        # mutating a shared object. Fail loud with the documented escape hatch.
        raise ConfigurationError.build(
            "observe= requires config['callbacks'] to be a list (or absent)",
            found=type(existing).__name__,
            hint="pass callbacks as a list, or attach the Langfuse handler manually.",
        )

    if any(isinstance(h, CallbackHandler) for h in existing_list):
        return config  # dedupe: user already wired a Langfuse handler

    return {**config, "callbacks": [*existing_list, CallbackHandler()]}


def _flush_observe(observe: bool | str | None) -> None:
    """Flush the Langfuse client on completion — symmetric with the attach gate
    (flush iff we would have attached), so a mis-configured client is never
    flushed. Safe to call unconditionally from a verb's ``finally``."""
    if not _observe_wants_langfuse(observe) or not _langfuse_keys_present():
        return
    from langfuse import get_client  # function-local: optional extra

    get_client().flush()


def _evict_run_cache(config: RunnableConfig | None) -> None:
    """Drop this run's per-run handle/resource cache entries — the SAME finalize
    seam as ``_flush_observe``, wired into every verb's ``finally``. ``_prepare``
    minted the RUN_ID into ``config['configurable']``; reading it back here (not
    threading it through) keeps one hook per verb. No RUN_ID (graph invoked
    directly) -> nothing was cached -> no-op."""
    run_id = (config or {}).get("configurable", {}).get(StateKeys.RUN_ID)
    if run_id is not None:
        evict_run(run_id)


def _finalize_prepare_config(
    graph: CompiledNeograph,
    config: RunnableConfig | None,
    *,
    stream_custom: bool,
    observe: bool | str | None,
) -> RunnableConfig | None:
    """The shared pre-engine config TAIL, run verbatim by both ``_prepare`` and
    ``_aprepare`` after their (awaited-vs-not) mode dispatch and run-id mint.

    Applies the last, driver-agnostic config normalizations — stream-custom flag,
    Langfuse callback merge, agent recursion-limit floor. Extracted per the
    extract-then-thin convention so the sync/async divergence stays confined to
    the awaited checkpoint-I/O seam. See neograph-yrph. Nothing here awaits, so
    one function serves both twins; the run-id mint stays inline in each twin so
    the mint-symmetry guard pins it per-driver."""
    if stream_custom:
        config = _mark_stream_custom(config)
    config = _merge_observe_callbacks(config, observe)
    config = _ensure_agent_recursion_limit(graph, config)
    return config


def _prepare(
    graph: CompiledNeograph,
    *,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    stream_custom: bool = False,
    observe: bool | str | None = None,
) -> tuple[Any, RunnableConfig | None]:
    """Compute ``(engine_input, config)`` for ONE execution — the single
    pre-engine brain shared by every driver verb (run/stream and, via
    ``_aprepare``, arun/astream).

    ``engine_input`` is exactly what the engine verb receives:
        * ``Command(resume=...)`` — resume after an Operator interrupt;
        * ``None`` — resume from an existing checkpoint (post-input resume or
          crash recovery); LangGraph's ``invoke(None, config)`` continues the
          thread;
        * the fingerprint-injected input dict — a fresh new execution.

    All pre-engine responsibilities live here so no verb re-implements them:
    CONFIG_INPUT stash / re-inject, input→config injection, defensive input
    copy, fingerprint injection, preflight-DI, checkpoint-exists probe, and the
    auto-resume rewind. The rewind is pure config mutation and runs HERE (not
    lazily inside a stream generator) so the first streamed chunk fires against
    the already-rewound checkpoint.
    """
    _assert_checkpointer_matches_driver(graph, is_async=False)
    if resume is not None:
        config = _prepare_resume_config(config)
        engine_input: Any = Command(resume=resume)
    elif input is not None:
        input, config = _prepare_new_input(graph, input, config)
        _preflight_di_check(graph, config)
        # A checkpoint for this thread means "resume, don't restart": pass None
        # so LangGraph continues from it; input is already stashed for DI.
        if _has_existing_checkpoint(graph, config):
            _verify_checkpoint_schema(graph, config, auto_resume=auto_resume)
            engine_input = None
        else:
            engine_input = input
    else:
        # Crash recovery. DI contract: the caller re-provides DI in config
        # because checkpoints do not persist config (same rule as FromConfig).
        if config is not None:
            _preflight_di_check(graph, config)
        if _has_existing_checkpoint(graph, config or {}):
            _verify_checkpoint_schema(graph, config or {}, auto_resume=auto_resume)
        engine_input = None

    # Mint the per-run id LAST, after all config normalization — a fresh id per
    # execution attempt (re-minted on resume because _prepare re-runs), stable
    # across every superstep of this run. Config-only, so it never enters state.
    config = _mint_run_id(config)
    config = _finalize_prepare_config(graph, config, stream_custom=stream_custom, observe=observe)
    return engine_input, config


def _finalize_by_mode(payload: Any, mode: str) -> Any:
    """Strip framework plumbing from ONE chunk according to its stream mode.

    * ``values`` — a full state dict; strip top-level ``neo_*`` keys.
    * ``updates`` — a ``{node: delta}`` dict; strip ``neo_*`` inside each per-node
      delta (a delta can carry fingerprints), but leave non-dict values (e.g.
      the ``__interrupt__`` tuple) untouched.
    * anything else (``custom`` / ``messages`` / ``debug``) — a user payload or
      token tuple; return it UNTOUCHED (identity), never stripped.
    """
    # ENGINE-GAP RESIDUE, see neograph-pjqe: compile declares output_schema so the
    # engine strips neo_* from invoke/ainvoke results, and run()/arun()/sub-construct
    # exits no longer wrap. But langgraph 1.2.4's output_schema does NOT filter
    # streamed chunks — stream_mode=values/updates emit raw channel writes, so a
    # synthesized barrier writing neo_oracle_* would leak here. These two arms are
    # the only surviving _strip_internals sites, kept by cited necessity, not habit.
    # See docs/design/langgraph-output-schema-research-2026-07-03.md (R1).
    if mode == "values":
        return _strip_internals(payload)
    if mode == "updates":
        if isinstance(payload, dict):
            return {
                node: (_strip_internals(delta) if isinstance(delta, dict) else delta) for node, delta in payload.items()
            }
        return payload
    return payload


def _finalize_chunk(chunk: Any, stream_mode: str | list[str]) -> Any:
    """Finalize one streamed chunk. The ONLY place that can leak ``neo_*`` or
    corrupt a user payload, so its stripping is mode-exact.

    A ``str`` ``stream_mode`` yields bare chunks finalized by that mode. A
    ``list`` ``stream_mode`` makes LangGraph yield ``(mode, chunk)`` tuples;
    each is finalized by ITS OWN mode and re-wrapped as a tuple.
    """
    if isinstance(stream_mode, str):
        return _finalize_by_mode(chunk, stream_mode)
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        mode, payload = chunk
        return (mode, _finalize_by_mode(payload, mode))
    return chunk


def run(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    observe: bool | str | None = None,
) -> Any:
    """Execute a compiled neograph graph (batch). Thin verb over ``_prepare``.

    Three modes:
        run(graph, input={...})              -- new execution
        run(graph, resume={...}, config=...) -- resume after Operator interrupt
        run(graph, config=...)               -- resume from checkpoint (crash recovery)

    Args:
        graph: Compiled LangGraph StateGraph (from compile()).
        input: Initial state values (for first run). All fields are also
               injected into config["configurable"] so node functions can
               access pipeline metadata (node_id, project_root, etc.)
               without reaching into state.
        resume: Human feedback (for resuming after Operator interrupt).
        config: LangGraph RunnableConfig (thread_id, callbacks, etc.).
               Put shared resources in config["configurable"].
        auto_resume: When True (default), automatically rewind to the
               checkpoint before the earliest changed node and re-execute
               from there. When False, raise CheckpointSchemaError on
               schema mismatch. Based on the Prefect cache-miss model.

    Crash recovery:
        When both input and resume are None, the graph resumes from its
        last checkpoint. Requires config with thread_id and a persistent
        checkpointer (SqliteSaver, PostgresSaver). LangGraph skips completed
        supersteps and continues from the failure point::

            run(graph, config={"configurable": {"thread_id": "same-id"}})
    """
    engine_input, config = _prepare(
        graph,
        input=input,
        resume=resume,
        config=config,
        auto_resume=auto_resume,
        observe=observe,
    )
    # No strip: compile declares output_schema=non-neo_ fields, so the engine
    # filters framework channels out of invoke() results. See neograph-pjqe.
    try:
        return graph.invoke(engine_input, config=config)
    finally:
        _flush_observe(observe)
        _evict_run_cache(config)


def stream(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    stream_mode: str | list[str] = "values",
    observe: bool | str | None = None,
) -> Any:
    """Stream a compiled neograph graph (sync). Thin verb over ``_prepare`` +
    ``_finalize_chunk``.

    Mirrors ``run``'s three modes (new input / resume / crash recovery) and adds
    ``stream_mode`` (str or list — see LangGraph). Each yielded chunk is passed
    through ``_finalize_chunk`` so ``values``/``updates`` chunks are ``neo_*``-
    free while ``custom``/``messages``/``debug`` payloads pass through untouched.
    """
    engine_input, config = _prepare(
        graph,
        input=input,
        resume=resume,
        config=config,
        auto_resume=auto_resume,
        stream_custom=_wants_custom(stream_mode),
        observe=observe,
    )
    # flush in finally so it fires after exhaustion AND on early GeneratorExit
    # (consumer .close()/GC) — no trace batch is stranded on a partial stream.
    try:
        for chunk in graph.stream(engine_input, config=config, stream_mode=stream_mode):
            yield _finalize_chunk(chunk, stream_mode)
    finally:
        _flush_observe(observe)
        _evict_run_cache(config)


async def _ahas_existing_checkpoint(graph: CompiledNeograph, config: RunnableConfig) -> bool:
    """Async twin of :func:`_has_existing_checkpoint` (awaits aget_tuple)."""
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return False
    try:
        saved = await checkpointer.aget_tuple(config)
    except (AttributeError, TypeError, KeyError) as exc:
        # yc38 (async twin): mirror the sync corrupt-read policy — a raising
        # read is corruption/misconfig, not "no checkpoint"; fail loud rather
        # than silently starting a fresh run that ignores durable state (7ymj).
        raise ConfigurationError.build(
            "checkpoint read failed",
            found=f"{type(checkpointer).__name__}.aget_tuple raised {type(exc).__name__}: {exc}",
            hint=(
                "the stored checkpoint appears corrupt or the saver is "
                "misconfigured; inspect the checkpoint backend for this thread_id"
            ),
        ) from exc
    return saved is not None and bool(saved.checkpoint.get("channel_versions"))


async def _averify_checkpoint_schema(
    graph: CompiledNeograph, config: RunnableConfig, *, auto_resume: bool = True
) -> None:
    """Async twin of :func:`_verify_checkpoint_schema`.

    Identical fingerprint-compare/invalidation logic; awaits ``aget_tuple`` and,
    on mismatch, ``_aauto_resume_from_divergence``. The decision (incl. the error
    message and auto-resume log) is shared via ``_decide_checkpoint_schema``.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None or graph.schema_fingerprint is None:
        return

    saved = await checkpointer.aget_tuple(config)
    invalidated = _decide_checkpoint_schema(graph, saved, auto_resume=auto_resume)
    if invalidated is not None:
        await _aauto_resume_from_divergence(graph, config, invalidated)


async def _aauto_resume_from_divergence(
    graph: CompiledNeograph,
    config: RunnableConfig,
    invalidated: set[str],
) -> None:
    """Async twin of :func:`_auto_resume_from_divergence`.

    ``aget_state_history`` is an async generator — consumed via ``async for``,
    never awaited. Identical rewind-checkpoint-id selection + config mutation,
    including the fail-loud ``_raise_no_rewind_point`` raise when no snapshot has
    an invalidated node pending in ``.next`` and the fail-clean
    ``_raise_incompatible_schema`` translation of a non-coercible-change
    ``ValidationError`` bubbling from the history re-materialization. See
    neograph-v63o and neograph-1gdw.
    """
    if not invalidated:
        return

    rewind_checkpoint_id = None
    try:
        async for state_snapshot in graph.aget_state_history(config):
            next_nodes = set(state_snapshot.next)
            if next_nodes & invalidated:
                candidate = state_snapshot.config.get("configurable", {}).get("checkpoint_id")
                if candidate is not None:
                    rewind_checkpoint_id = candidate
    except ValidationError as exc:
        _raise_incompatible_schema(invalidated, exc)
    if rewind_checkpoint_id is None:
        _raise_no_rewind_point(invalidated)
    config.setdefault("configurable", {})["checkpoint_id"] = rewind_checkpoint_id


async def _aprepare(
    graph: CompiledNeograph,
    *,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    stream_custom: bool = False,
    observe: bool | str | None = None,
) -> tuple[Any, RunnableConfig | None]:
    """Async twin of :func:`_prepare`.

    Identical mode dispatch and pre-engine responsibilities; diverges ONLY where
    the sync helpers touch the checkpointer I/O — ``await
    _ahas_existing_checkpoint`` / ``await _averify_checkpoint_schema`` (which
    ``async for`` the async state history). The pure helpers
    (_prepare_resume_config / _prepare_new_input / _preflight_di_check) and the
    ``_finalize_prepare_config`` tail are shared verbatim with the sync path, so
    divergence is confined to the awaited checkpoint-I/O seam. See neograph-yrph.
    """
    _assert_checkpointer_matches_driver(graph, is_async=True)
    if resume is not None:
        config = _prepare_resume_config(config)
        engine_input: Any = Command(resume=resume)
    elif input is not None:
        input, config = _prepare_new_input(graph, input, config)
        _preflight_di_check(graph, config)
        if await _ahas_existing_checkpoint(graph, config):
            await _averify_checkpoint_schema(graph, config, auto_resume=auto_resume)
            engine_input = None
        else:
            engine_input = input
    else:
        if config is not None:
            _preflight_di_check(graph, config)
        if await _ahas_existing_checkpoint(graph, config or {}):
            await _averify_checkpoint_schema(graph, config or {}, auto_resume=auto_resume)
        engine_input = None

    # Mint the per-run id LAST, after all config normalization — a fresh id per
    # execution attempt (re-minted on resume because _aprepare re-runs), stable
    # across every superstep of this run. Config-only, so it never enters state.
    config = _mint_run_id(config)
    config = _finalize_prepare_config(graph, config, stream_custom=stream_custom, observe=observe)
    return engine_input, config


async def arun(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    observe: bool | str | None = None,
) -> Any:
    """Async twin of :func:`run` (batch). Thin verb over ``_aprepare``.

    Driver-parallel to run(): shares the entire pre-engine brain via
    ``_aprepare`` and diverges ONLY at the engine I/O — ``await graph.ainvoke``.
    """
    engine_input, config = await _aprepare(
        graph,
        input=input,
        resume=resume,
        config=config,
        auto_resume=auto_resume,
        observe=observe,
    )
    # No strip: output_schema (declared at compile) filters ainvoke() results too.
    # See neograph-pjqe. Symmetric with the sync run() exit above.
    try:
        return await graph.ainvoke(engine_input, config=config)
    finally:
        _flush_observe(observe)
        _evict_run_cache(config)


async def astream(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    stream_mode: str | list[str] = "values",
    observe: bool | str | None = None,
) -> Any:
    """Async twin of :func:`stream`. Thin verb over ``_aprepare`` +
    ``_finalize_chunk``.

    The production streaming surface for AG-UI/SSE consumers: yields the same
    finalized chunks as ``stream`` while running the LLM/tool vertical on the
    event loop. The auto-resume rewind runs inside ``_aprepare`` BEFORE the first
    ``astream`` chunk, so the stream never fires against an un-rewound checkpoint.
    """
    engine_input, config = await _aprepare(
        graph,
        input=input,
        resume=resume,
        config=config,
        auto_resume=auto_resume,
        stream_custom=_wants_custom(stream_mode),
        observe=observe,
    )
    # flush in finally: after exhaustion AND on early GeneratorExit/cancellation.
    try:
        async for chunk in graph.astream(engine_input, config=config, stream_mode=stream_mode):
            yield _finalize_chunk(chunk, stream_mode)
    finally:
        _flush_observe(observe)
        _evict_run_cache(config)
