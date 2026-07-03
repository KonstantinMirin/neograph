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

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from neograph._compiled import CompiledNeograph
from neograph._ir_branch import iter_with_arms
from neograph._llm_config import _coerce_llm_config
from neograph._state_keys import StateKeys, _strip_internals
from neograph.construct import iter_nodes
from neograph.errors import CheckpointSchemaError, ExecutionError
from neograph.naming import field_name_for
from neograph.node import Node

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


def _ensure_agent_recursion_limit(
    graph: CompiledNeograph, config: RunnableConfig | None,
) -> RunnableConfig | None:
    """Raise ``recursion_limit`` so an agent/act cycle can reach its graceful
    budget-exhaust → forced-final edge instead of hitting LangGraph's default
    superstep ceiling first.

    Each agent/act node's cycle can cost ``max_iterations * 2 + overhead``
    supersteps; sequential agent nodes run in distinct supersteps, so their costs
    ADD across the run. The floor sums every agent/act node's worst case on top of
    the default (which already covers the surrounding non-agent nodes). Only
    RAISES to the floor — a larger user-supplied ``recursion_limit`` is kept.
    Pure config mutation (no engine verb); shared verbatim by ``_prepare`` and
    ``_aprepare``.
    """
    construct = getattr(graph, "construct", None)
    if construct is None:
        return config

    agent_cost = 0
    for node in iter_nodes(construct):
        if isinstance(node, Node) and node.mode in ("agent", "act"):
            max_iters = _coerce_llm_config(node.llm_config).max_iterations
            agent_cost += max_iters * _SUPERSTEPS_PER_AGENT_TURN + _AGENT_CYCLE_OVERHEAD

    if agent_cost == 0:
        return config  # no agent/act nodes — leave the default untouched

    floor = _LANGGRAPH_DEFAULT_RECURSION_LIMIT + agent_cost
    current = (config or {}).get("recursion_limit", _LANGGRAPH_DEFAULT_RECURSION_LIMIT)
    if current >= floor:
        return config  # user asked for at least what agents need — keep theirs

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


def _verify_checkpoint_schema(graph: CompiledNeograph, config: RunnableConfig, *, auto_resume: bool = True) -> None:
    """Verify checkpoint state schema matches the current graph.

    Compares the neo_schema_fingerprint stored in the checkpoint against
    the fingerprint computed at compile time. When auto_resume is True,
    rewinds to the checkpoint before the earliest changed node and re-invokes.
    When False, raises CheckpointSchemaError.
    """
    current_fp = graph.schema_fingerprint
    if current_fp is None:
        return  # no fingerprint on graph (pre-feature compile)

    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return

    saved = checkpointer.get_tuple(config)
    if saved is None:
        return

    # Extract fingerprint from checkpoint's channel values
    channel_values = saved.checkpoint.get("channel_values", {})
    stored_fp = None
    if isinstance(channel_values, dict):
        stored_fp = channel_values.get(StateKeys.SCHEMA_FINGERPRINT)
    elif hasattr(channel_values, "get"):
        stored_fp = channel_values.get(StateKeys.SCHEMA_FINGERPRINT)

    if stored_fp is None or stored_fp == "":
        return  # checkpoint from before fingerprinting was added

    if stored_fp != current_fp:
        invalidated = _compute_invalidated_nodes(graph, channel_values)

        if not auto_resume:
            raise CheckpointSchemaError(
                f"Checkpoint schema fingerprint mismatch: "
                f"stored={stored_fp!r}, current={current_fp!r}. "
                f"Invalidated nodes: {sorted(invalidated) if invalidated else 'all'}. "
                f"Invalidate the checkpoint or migrate the state.",
                invalidated_nodes=invalidated,
            )

        # Auto-resume: rewind to before the earliest changed node
        import structlog
        log = structlog.get_logger()
        log.info(
            "auto_resume_schema_change",
            invalidated=sorted(invalidated),
            stored_fp=stored_fp,
            current_fp=current_fp,
        )
        _auto_resume_from_divergence(graph, config, invalidated)


def _auto_resume_from_divergence(
    graph: CompiledNeograph, config: RunnableConfig, invalidated: set[str],
) -> None:
    """Rewind checkpoint to before the earliest invalidated node.

    Uses LangGraph time-travel: walks state_history to find the checkpoint
    where the earliest invalidated node was about to execute (in ``next``).
    Overwrites the main config's checkpoint_id to point to that checkpoint,
    so the subsequent ``invoke(None, config)`` resumes from the rewind point.
    """
    if not invalidated:
        return

    # ``get_state_history`` yields newest-first. We want the OLDEST checkpoint
    # whose ``next`` intersects the invalidated set — that's the rewind point
    # that re-executes every invalidated node, not just the latest one.
    rewind_checkpoint_id = None
    for state_snapshot in graph.get_state_history(config):
        next_nodes = set(state_snapshot.next)
        if next_nodes & invalidated:
            candidate = state_snapshot.config.get("configurable", {}).get("checkpoint_id")
            if candidate is not None:
                rewind_checkpoint_id = candidate
    if rewind_checkpoint_id is not None:
        config.setdefault("configurable", {})["checkpoint_id"] = rewind_checkpoint_id


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
        return saved is not None and bool(saved.checkpoint.get("channel_versions"))
    except (AttributeError, TypeError, KeyError):
        return False


def _prepare_resume_config(config: RunnableConfig | None) -> RunnableConfig | None:
    """Re-inject stashed input into config on resume so FromInput DI resolves
    for post-interrupt nodes. Pure (no I/O); shared by run() and arun()."""
    if config is not None:
        neo_input = config.get("configurable", {}).get(StateKeys.CONFIG_INPUT)
        if neo_input is not None:
            config = _inject_input_to_config(neo_input, config)
    return config


def _prepare_new_input(
    graph: CompiledNeograph, input: dict[str, Any], config: RunnableConfig | None,
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


def _wants_custom(stream_mode: str | list[str]) -> bool:
    """True if ``stream_mode`` requests LangGraph's ``custom`` channel."""
    if isinstance(stream_mode, str):
        return stream_mode == "custom"
    return "custom" in stream_mode


def _prepare(
    graph: CompiledNeograph,
    *,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    stream_custom: bool = False,
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

    if stream_custom:
        config = _mark_stream_custom(config)
    config = _ensure_agent_recursion_limit(graph, config)
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
    if mode == "values":
        return _strip_internals(payload)
    if mode == "updates":
        if isinstance(payload, dict):
            return {
                node: (_strip_internals(delta) if isinstance(delta, dict) else delta)
                for node, delta in payload.items()
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
        graph, input=input, resume=resume, config=config, auto_resume=auto_resume
    )
    return _strip_internals(graph.invoke(engine_input, config=config))


def stream(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    stream_mode: str | list[str] = "values",
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
    )
    for chunk in graph.stream(engine_input, config=config, stream_mode=stream_mode):
        yield _finalize_chunk(chunk, stream_mode)


async def _ahas_existing_checkpoint(graph: CompiledNeograph, config: RunnableConfig) -> bool:
    """Async twin of :func:`_has_existing_checkpoint` (awaits aget_tuple)."""
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return False
    try:
        saved = await checkpointer.aget_tuple(config)
        return saved is not None and bool(saved.checkpoint.get("channel_versions"))
    except (AttributeError, TypeError, KeyError):
        return False


async def _averify_checkpoint_schema(graph: CompiledNeograph, config: RunnableConfig, *, auto_resume: bool = True) -> None:
    """Async twin of :func:`_verify_checkpoint_schema`.

    Identical fingerprint-compare/invalidation logic; awaits ``aget_tuple`` and,
    on mismatch, ``_aauto_resume_from_divergence``. Reuses _compute_invalidated_nodes.
    """
    current_fp = graph.schema_fingerprint
    if current_fp is None:
        return

    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        return

    saved = await checkpointer.aget_tuple(config)
    if saved is None:
        return

    channel_values = saved.checkpoint.get("channel_values", {})
    stored_fp = None
    if isinstance(channel_values, dict):
        stored_fp = channel_values.get(StateKeys.SCHEMA_FINGERPRINT)
    elif hasattr(channel_values, "get"):
        stored_fp = channel_values.get(StateKeys.SCHEMA_FINGERPRINT)

    if stored_fp is None or stored_fp == "":
        return

    if stored_fp != current_fp:
        invalidated = _compute_invalidated_nodes(graph, channel_values)

        if not auto_resume:
            raise CheckpointSchemaError(
                f"Checkpoint schema fingerprint mismatch: "
                f"stored={stored_fp!r}, current={current_fp!r}. "
                f"Invalidated nodes: {sorted(invalidated) if invalidated else 'all'}. "
                f"Invalidate the checkpoint or migrate the state.",
                invalidated_nodes=invalidated,
            )

        import structlog
        log = structlog.get_logger()
        log.info(
            "auto_resume_schema_change",
            invalidated=sorted(invalidated),
            stored_fp=stored_fp,
            current_fp=current_fp,
        )
        await _aauto_resume_from_divergence(graph, config, invalidated)


async def _aauto_resume_from_divergence(
    graph: CompiledNeograph, config: RunnableConfig, invalidated: set[str],
) -> None:
    """Async twin of :func:`_auto_resume_from_divergence`.

    ``aget_state_history`` is an async generator — consumed via ``async for``,
    never awaited. Identical rewind-checkpoint-id selection + config mutation.
    """
    if not invalidated:
        return

    rewind_checkpoint_id = None
    async for state_snapshot in graph.aget_state_history(config):
        next_nodes = set(state_snapshot.next)
        if next_nodes & invalidated:
            candidate = state_snapshot.config.get("configurable", {}).get("checkpoint_id")
            if candidate is not None:
                rewind_checkpoint_id = candidate
    if rewind_checkpoint_id is not None:
        config.setdefault("configurable", {})["checkpoint_id"] = rewind_checkpoint_id


async def _aprepare(
    graph: CompiledNeograph,
    *,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    stream_custom: bool = False,
) -> tuple[Any, RunnableConfig | None]:
    """Async twin of :func:`_prepare`.

    Identical mode dispatch and pre-engine responsibilities; diverges ONLY where
    the sync helpers touch the checkpointer I/O — ``await
    _ahas_existing_checkpoint`` / ``await _averify_checkpoint_schema`` (which
    ``async for`` the async state history). The pure helpers
    (_prepare_resume_config / _prepare_new_input / _preflight_di_check /
    _mark_stream_custom) are shared verbatim with the sync path.
    """
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

    if stream_custom:
        config = _mark_stream_custom(config)
    config = _ensure_agent_recursion_limit(graph, config)
    return engine_input, config


async def arun(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
) -> Any:
    """Async twin of :func:`run` (batch). Thin verb over ``_aprepare``.

    Driver-parallel to run(): shares the entire pre-engine brain via
    ``_aprepare`` and diverges ONLY at the engine I/O — ``await graph.ainvoke``.
    """
    engine_input, config = await _aprepare(
        graph, input=input, resume=resume, config=config, auto_resume=auto_resume
    )
    return _strip_internals(await graph.ainvoke(engine_input, config=config))


async def astream(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
    stream_mode: str | list[str] = "values",
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
    )
    async for chunk in graph.astream(engine_input, config=config, stream_mode=stream_mode):
        yield _finalize_chunk(chunk, stream_mode)
