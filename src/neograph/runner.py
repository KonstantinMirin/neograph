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
from neograph._state_keys import StateKeys, _strip_internals
from neograph.errors import CheckpointSchemaError, ExecutionError
from neograph.naming import field_name_for


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

    for item in getattr(construct, "nodes", []):
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


def run(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
) -> Any:
    """Execute a compiled neograph graph.

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
    if resume is not None:
        config = _prepare_resume_config(config)
        return _strip_internals(graph.invoke(Command(resume=resume), config=config))

    if input is not None:
        input, config = _prepare_new_input(graph, input, config)

        # Pre-flight: check all required DI params are present
        _preflight_di_check(graph, config)

        # Check if a checkpoint already exists for this thread.
        # If yes: this is a resume — inject input into config for DI but
        # pass None to graph.invoke() so LangGraph resumes from checkpoint.
        # If no: this is a new execution — pass input normally.
        if _has_existing_checkpoint(graph, config):
            _verify_checkpoint_schema(graph, config, auto_resume=auto_resume)
            return _strip_internals(graph.invoke(None, config=config))

        return _strip_internals(graph.invoke(input, config=config))

    # No input, no resume: resume from checkpoint.
    # LangGraph's _first() treats invoke(None, config) as "resume from last
    # checkpoint" when the thread already has saved state. This is the crash
    # recovery path — skips completed supersteps, continues from failure point.
    #
    # DI contract: the caller must re-provide DI values in config['configurable']
    # because checkpoints do not persist config. This is the same contract as
    # FromConfig (rate limiters, shared resources are never checkpointed).
    if config is not None:
        _preflight_di_check(graph, config)
    if _has_existing_checkpoint(graph, config or {}):
        _verify_checkpoint_schema(graph, config or {}, auto_resume=auto_resume)
    return _strip_internals(graph.invoke(None, config=config))


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


async def arun(
    graph: CompiledNeograph,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: RunnableConfig | None = None,
    auto_resume: bool = True,
) -> Any:
    """Async twin of :func:`run` (Phase 1d).

    Driver-parallel to run(): mirrors the exact three-mode branch structure
    (resume / input+nested-checkpoint / crash-recovery), reuses the shared pure
    prep helpers (_prepare_resume_config / _prepare_new_input / _preflight_di_check
    / _strip_internals), and diverges ONLY at the graph/checkpointer I/O —
    ``await graph.ainvoke`` and the async checkpoint twins.
    """
    if resume is not None:
        config = _prepare_resume_config(config)
        return _strip_internals(await graph.ainvoke(Command(resume=resume), config=config))

    if input is not None:
        input, config = _prepare_new_input(graph, input, config)

        _preflight_di_check(graph, config)

        if await _ahas_existing_checkpoint(graph, config):
            await _averify_checkpoint_schema(graph, config, auto_resume=auto_resume)
            return _strip_internals(await graph.ainvoke(None, config=config))

        return _strip_internals(await graph.ainvoke(input, config=config))

    if config is not None:
        _preflight_di_check(graph, config)
    if await _ahas_existing_checkpoint(graph, config or {}):
        await _averify_checkpoint_schema(graph, config or {}, auto_resume=auto_resume)
    return _strip_internals(await graph.ainvoke(None, config=config))
