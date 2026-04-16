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

from langgraph.types import Command

from neograph.errors import CheckpointSchemaError, ExecutionError


def _strip_internals(result: Any) -> Any:
    """Remove neo_* framework plumbing from the result dict."""
    if not isinstance(result, dict):
        return result
    return {k: v for k, v in result.items() if not k.startswith("neo_")}


def _preflight_di_check(graph: Any, config: dict[str, Any]) -> None:
    """Validate that all required DI params are present before starting any node.

    compile() stashes _neo_required_di on the graph — a dict with "input"
    and "config" sets of required param names. This check runs before
    graph.invoke() so missing params fail at the gate, not mid-pipeline.
    """
    required = getattr(graph, "_neo_required_di", None)
    if required is None:
        return  # graph compiled without DI metadata (e.g., raw LangGraph)

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
    config: dict[str, Any] | None,
) -> dict[str, Any]:
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


def _verify_checkpoint_schema(graph: Any, config: dict[str, Any], *, auto_resume: bool = True) -> None:
    """Verify checkpoint state schema matches the current graph.

    Compares the neo_schema_fingerprint stored in the checkpoint against
    the fingerprint computed at compile time. When auto_resume is True,
    rewinds to the checkpoint before the earliest changed node and re-invokes.
    When False, raises CheckpointSchemaError.
    """
    current_fp = getattr(graph, "_neo_schema_fingerprint", None)
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
        stored_fp = channel_values.get("neo_schema_fingerprint")
    elif hasattr(channel_values, "get"):
        stored_fp = channel_values.get("neo_schema_fingerprint")

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
    graph: Any, config: dict[str, Any], invalidated: set[str],
) -> None:
    """Rewind checkpoint to before the earliest invalidated node.

    Uses LangGraph time-travel: walks state_history to find the checkpoint
    where the earliest invalidated node was about to execute (in ``next``).
    Overwrites the main config's checkpoint_id to point to that checkpoint,
    so the subsequent ``invoke(None, config)`` resumes from the rewind point.
    """
    if not invalidated:
        return

    # Walk state history to find the checkpoint where an invalidated node
    # was about to execute — this is the rewind point
    for state_snapshot in graph.get_state_history(config):
        next_nodes = set(state_snapshot.next)
        if next_nodes & invalidated:
            # Found it. Extract the checkpoint_id and inject into the caller's config
            # so invoke(None, config) resumes from here.
            rewind_checkpoint_id = state_snapshot.config.get("configurable", {}).get("checkpoint_id")
            if rewind_checkpoint_id is not None:
                config.setdefault("configurable", {})["checkpoint_id"] = rewind_checkpoint_id
            return

    # Can't find rewind point — fall through to normal resume


def _compute_invalidated_nodes(graph: Any, channel_values: Any) -> set[str]:
    """Compute which nodes changed + their transitive descendants.

    Compares per-node fingerprints stored in the checkpoint against the
    current graph's per-node fingerprints. Returns the union of changed
    nodes and all nodes that transitively depend on them.
    """
    current_nfp = getattr(graph, "_neo_node_fingerprints", None)
    if current_nfp is None:
        return set()

    stored_nfp = None
    if isinstance(channel_values, dict):
        stored_nfp = channel_values.get("neo_node_fingerprints")
    elif hasattr(channel_values, "get"):
        stored_nfp = channel_values.get("neo_node_fingerprints")

    if not stored_nfp or not isinstance(stored_nfp, dict):
        return set()

    # Find directly changed nodes
    changed = set()
    for node_field, current_fp in current_nfp.items():
        stored_fp = stored_nfp.get(node_field)
        if stored_fp is not None and stored_fp != current_fp:
            changed.add(node_field)

    if not changed:
        return set()

    # Compute transitive descendants via the graph's node adjacency
    # The graph nodes have a mapping we can derive from the channel specs
    # For simplicity, return changed nodes — DAG walking requires construct access
    # which isn't available on the compiled graph. The changed set is sufficient
    # for the user to identify what needs invalidation.
    return changed


def _has_existing_checkpoint(graph: Any, config: dict[str, Any]) -> bool:
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


def run(
    graph: Any,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    auto_resume: bool = True,
) -> Any:
    """Execute a compiled NeoGraph graph.

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
        # Re-inject input fields on resume.
        # The initial run stashes input in the caller's config dict.
        # On resume, re-inject so FromInput DI resolves for post-interrupt nodes.
        if config is not None:
            neo_input = config.get("configurable", {}).get("_neo_input")
            if neo_input is not None:
                config = _inject_input_to_config(neo_input, config)
        return _strip_internals(graph.invoke(Command(resume=resume), config=config))

    if input is not None:
        # Stash input in the CALLER'S config so resume can re-inject.
        if config is None:
            config = {}
        configurable = config.setdefault("configurable", {})
        configurable["_neo_input"] = input
        config = _inject_input_to_config(input, config)

        # Inject schema fingerprint into initial state for checkpoint storage
        fp = getattr(graph, "_neo_schema_fingerprint", None)
        if fp is not None:
            input["neo_schema_fingerprint"] = fp
        node_fps = getattr(graph, "_neo_node_fingerprints", None)
        if node_fps is not None:
            input["neo_node_fingerprints"] = node_fps

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
