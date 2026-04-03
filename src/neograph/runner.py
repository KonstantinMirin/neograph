"""Runner — execute compiled graphs with checkpointing.

    result = run(graph, input={"node_id": "BR-RW-042", "project_root": "/path"})

    # Resume after Operator interrupt:
    result = run(graph, resume={"approved": True}, config=config)
"""

from __future__ import annotations

from typing import Any

from langgraph.types import Command


def run(
    graph: Any,
    input: dict[str, Any] | None = None,
    resume: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Execute a compiled NeoGraph graph.

    Args:
        graph: Compiled LangGraph StateGraph (from compile()).
        input: Initial state values (for first run).
        resume: Human feedback (for resuming after Operator interrupt).
        config: LangGraph RunnableConfig (thread_id, checkpointer, etc.).
    """
    if resume is not None:
        # Resume after interrupt
        return graph.invoke(Command(resume=resume), config=config)

    if input is not None:
        return graph.invoke(input, config=config)

    msg = "Either input or resume must be provided."
    raise ValueError(msg)
