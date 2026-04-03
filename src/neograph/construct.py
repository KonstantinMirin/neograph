"""Construct — ordered composition of Nodes. The pipeline blueprint.

    rw_pipeline = Construct(
        "rw-ingestion",
        description="Process raw input nodes into graph dispositions",
        nodes=[read_node, decompose, classify, ...],
    )
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph.node import Node


class Construct(BaseModel):
    """An ordered composition of Nodes that compiles to a LangGraph StateGraph.

    Nodes execute in sequence. Modifiers (Oracle, Replicate, Operator) on
    individual nodes modify the topology — the Construct itself is a flat list.
    The compiler handles fan-out, barriers, and interrupts.
    """

    name: str
    description: str = ""
    nodes: list[Node] = []

    model_config = {"arbitrary_types_allowed": True}
