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

from typing import Any
from uuid import uuid4

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import ValidationError

from neograph._compiled import CompiledNeograph
from neograph._config_carrier import _with_configurable
from neograph._state_keys import StateKeys, _strip_internals
from neograph.errors import ConfigurationError, ExecutionError

log = structlog.get_logger()

# --- extracted clusters (neograph-3ffdg.9), re-exported so every existing
# --- `from neograph.runner import ...` call site keeps resolving unchanged.
from neograph._checkpoint_driver import (  # noqa: E402,F401
    _assert_checkpointer_matches_driver,
    _required_checkpointer_driver,
)
from neograph._checkpoint_rewind import (  # noqa: E402,F401
    _build_producer_consumer_adjacency,
    _compute_invalidated_nodes,
    _decide_checkpoint_schema,
    _raise_incompatible_schema,
    _raise_no_rewind_point,
    _transitive_closure,
)
from neograph._observe import (  # noqa: E402,F401
    _evict_run_cache,
    _flush_observe,
    _langfuse_keys_present,
    _merge_observe_callbacks,
    _observe_wants_langfuse,
)
from neograph._recursion_budget import (  # noqa: E402,F401
    _AGENT_CYCLE_OVERHEAD,
    _LANGGRAPH_DEFAULT_RECURSION_LIMIT,
    _SUPERSTEPS_PER_AGENT_TURN,
    _ensure_agent_recursion_limit,
    _member_hop_cost,
    _mesh_hop_cost,
    _portal_mesh_member_ids,
)


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
    # Input fields become configurable (input takes precedence)
    return _with_configurable(config or {}, **input)


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
    return _with_configurable(config or {}, **{StateKeys.STREAM_CUSTOM: True})


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
    return _with_configurable(config or {}, **{StateKeys.RUN_ID: uuid4().hex})


def _wants_custom(stream_mode: str | list[str]) -> bool:
    """True if ``stream_mode`` requests LangGraph's ``custom`` channel."""
    if isinstance(stream_mode, str):
        return stream_mode == "custom"
    return "custom" in stream_mode


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


# ── auto-rewind: these two STAY in runner.py deliberately ────────────────────
# They call graph.get_state_history()/aget_state_history(), which the
# three-layer guard confines to the typed facade (_compiled.py) and the
# runner verbs. Moving them into _checkpoint_rewind.py with the rest of the
# rewind logic would have required widening ALLOWED_GRAPH_ONLY_MODULES --
# a smaller extraction beats a wider architectural allowlist
# (neograph-3ffdg.9). The pure schema-diff logic they call lives there.


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
