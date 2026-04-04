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


def _strip_internals(result: Any) -> Any:
    """Remove neo_* framework plumbing from the result dict."""
    if not isinstance(result, dict):
        return result
    return {k: v for k, v in result.items() if not k.startswith("neo_")}


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


def run(
    graph: Any,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Execute a compiled NeoGraph graph.

    Args:
        graph: Compiled LangGraph StateGraph (from compile()).
        input: Initial state values (for first run). All fields are also
               injected into config["configurable"] so node functions can
               access pipeline metadata (node_id, project_root, etc.)
               without reaching into state.
        resume: Human feedback (for resuming after Operator interrupt).
        config: LangGraph RunnableConfig (thread_id, callbacks, etc.).
               Put shared resources in config["configurable"].
    """
    if resume is not None:
        return _strip_internals(graph.invoke(Command(resume=resume), config=config))

    if input is not None:
        config = _inject_input_to_config(input, config)
        return _strip_internals(graph.invoke(input, config=config))

    msg = "Either input or resume must be provided."
    raise ValueError(msg)
