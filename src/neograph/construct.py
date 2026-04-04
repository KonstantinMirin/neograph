"""Construct — ordered composition of Nodes. The pipeline blueprint.

    # Top-level pipeline
    rw_pipeline = Construct(
        "rw-ingestion",
        nodes=[read_node, decompose, classify, ...],
    )

    # Sub-construct with declared I/O boundary
    enrich = Construct(
        "enrich",
        input=Claims,
        output=ScoredClaims,
        nodes=[lookup, verify, score],
    )

    # Compose: sub-construct in a parent pipeline
    main = Construct("main", nodes=[decompose, enrich, report])
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from neograph.modifiers import Modifiable, Modifier
from neograph.node import Node


class Construct(Modifiable, BaseModel):
    """An ordered composition of Nodes that compiles to a LangGraph StateGraph.

    Nodes execute in sequence. Modifiers (Oracle, Each, Operator) on
    individual nodes modify the topology — the Construct itself is a flat list.
    The compiler handles fan-out, barriers, and interrupts.

    When used as a sub-construct inside another Construct, declare input/output
    to define the state boundary. The sub-construct gets its own isolated state.

    Modifiers can be applied to Constructs via pipe:
        sub | Oracle(n=3, merge_fn="merge")   — ensemble the entire sub-pipeline
        sub | Each(over="items", key="label") — run sub-pipeline per item
        sub | Operator(when="check")          — interrupt after sub-pipeline
    """

    name: str
    description: str = ""
    nodes: list[Any] = []  # list[Node | Construct] — Any avoids circular ref issues

    # I/O boundary — required when used as a sub-construct
    input: Any = None   # type[BaseModel] | None
    output: Any = None  # type[BaseModel] | None

    # Modifiers applied via | operator
    modifiers: list[Modifier] = []

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name_: str | None = None, /, **kwargs):
        if name_ is not None:
            kwargs["name"] = name_
        super().__init__(**kwargs)

    # has_modifier, get_modifier, __or__ inherited from Modifiable
