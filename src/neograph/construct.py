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

from pydantic import BaseModel, Field

from neograph._construct_validation import ConstructError, _validate_node_chain
from neograph.modifiers import Modifiable, Modifier

__all__ = ["Construct", "ConstructError"]


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

    Type safety: input/output compatibility across the node chain is validated
    at assembly time (when `Construct(nodes=[...])` is called), not at
    `compile()` time. Mismatches raise `ConstructError` with a pointer to the
    user source line and a suggestion (e.g. "did you forget .map()?"). See
    `neograph._construct_validation` for the walker.

    Subclassing note: subclasses that override `__init__` must chain to this
    constructor so `_validate_node_chain` still runs.
    """

    name: str
    description: str = ""
    # list[Node | Construct] — Any avoids a circular type reference.
    nodes: list[Any] = Field(default_factory=list)

    # I/O boundary — required when used as a sub-construct
    input: type[BaseModel] | None = None
    output: type[BaseModel] | None = None

    # Default LLM config inherited by every node. Per-node llm_config merges
    # over this (node wins on conflicts). Common use: setting
    # output_strategy="json_mode" once for a whole pipeline instead of on
    # every produce node.
    llm_config: dict[str, Any] = Field(default_factory=dict)

    # Modifiers applied via | operator
    modifiers: list[Modifier] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name_: str | None = None, /, **kwargs: Any) -> None:
        if name_ is not None:
            kwargs["name"] = name_
        super().__init__(**kwargs)
        # Propagate default llm_config to child nodes BEFORE validation so
        # downstream compile() sees the merged config. Per-node llm_config
        # merges over the Construct default (node wins on conflicts).
        if self.llm_config:
            for item in self.nodes:
                if hasattr(item, "llm_config"):
                    merged = {**self.llm_config, **item.llm_config}
                    # Node is a frozen-ish pydantic model — use model_copy
                    # to produce the merged version, then replace in-place.
                    try:
                        item.llm_config = merged
                    except (TypeError, ValueError):
                        # Frozen model — skip. Sub-constructs inherit via
                        # their own __init__ merging.
                        pass
        # Validate after pydantic finishes so ConstructError escapes cleanly
        # rather than being wrapped in a pydantic ValidationError. Nested
        # constructs self-validate during their own __init__.
        _validate_node_chain(self)

    # has_modifier, get_modifier, __or__, map inherited from Modifiable
