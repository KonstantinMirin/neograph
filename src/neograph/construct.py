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

from collections.abc import Iterator
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field

from neograph._construct_validation import ConstructError, _validate_node_chain
from neograph._ir_normalize import normalize_ir
from neograph._ir_protocols import ConstructItem
from neograph._llm_config import LlmConfig
from neograph.modifiers import Modifiable, ModifierSet
from neograph.node import Node
from neograph.renderers import Renderer


def _validate_node_list(v: Any) -> list[ConstructItem]:
    """Validate that nodes list contains Node, Construct, _BranchNode, or compatible BaseModel.

    Accepts any BaseModel with a ``name`` attribute (covers Node, Construct,
    _BranchNode via Modifiable, and test fakes). Rejects plain dicts, strings,
    and other non-model types that would silently break downstream.
    """
    if not isinstance(v, list):
        raise TypeError(f"nodes must be a list, got {type(v).__name__}")
    for i, item in enumerate(v):
        if not (isinstance(item, (BaseModel, Modifiable)) and hasattr(item, "name")):
            raise TypeError(
                f"nodes[{i}] must be a Node, Construct, or compatible model — "
                f"got {type(item).__name__}: {item!r}"
            )
    return v

__all__ = ["Construct", "ConstructError", "iter_nodes"]


def iter_nodes(construct: Construct) -> Iterator[Node]:
    """Yield every leaf ``Node`` in ``construct``, recursing into sub-constructs.

    Single source of truth for the IR node-tree walk. Replaces the hand-rolled
    ``isinstance(item, Construct) -> recurse; isinstance(item, Node) -> body``
    skeleton that was duplicated across the compiler, the LLM runtime, and lint.
    ``_BranchNode`` sentinels (neither Node nor Construct) are skipped, matching
    the prior walks' ``isinstance(Node)`` gate.
    """
    for item in construct.nodes:
        if isinstance(item, Construct):
            yield from iter_nodes(item)
        elif isinstance(item, Node):
            yield item


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
    # ConstructItem is a runtime_checkable Protocol over name/modifier_set —
    # covers Node, Construct, and the non-Pydantic _BranchNode sentinel uniformly.
    # The BeforeValidator still enforces actual acceptance rules at runtime;
    # see neograph._ir_protocols for the Protocol definition.
    nodes: Annotated[list[ConstructItem], BeforeValidator(_validate_node_list)] = Field(default_factory=list)

    # I/O boundary — required when used as a sub-construct
    input: type[BaseModel] | None = None
    output: type[BaseModel] | None = None

    # Default LLM config inherited by every node. Per-node llm_config merges
    # over this via LlmConfig.merged_with (node wins on conflicts). Common
    # use: setting output_strategy="json_mode" once for a whole pipeline
    # instead of on every produce node.
    llm_config: LlmConfig = Field(default_factory=LlmConfig)

    # Default renderer for all child nodes. Propagated to children that don't
    # have their own renderer set. Dispatch hierarchy:
    # model.render_for_prompt() > node.renderer > construct.renderer > global > None.
    renderer: Renderer | None = None

    # Modifiers applied via | operator (typed slots, not a list)
    modifier_set: ModifierSet = Field(default_factory=ModifierSet)

    # arbitrary_types_allowed: required for ``nodes`` (holds the ``_BranchNode``
    # sentinel from ForwardConstruct, which is not a Pydantic model) and
    # ``renderer`` (a runtime_checkable Protocol, not a BaseModel).
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name_: str | None = None, /, **kwargs: Any) -> None:
        if name_ is not None:
            kwargs["name"] = name_
        # Reject legacy modifiers=[...] constructor form.
        if "modifiers" in kwargs:
            raise ConstructError.build(
                "Construct(modifiers=[...]) is no longer supported",
                hint="use the pipe syntax instead: construct | Oracle(...) | Each(...)",
            )
        super().__init__(**kwargs)
        if not self.nodes:
            raise ConstructError.build(
                "Construct has no nodes",
                construct=self.name,
                hint="add at least one Node or sub-Construct",
            )
        # Propagate default llm_config to child nodes BEFORE validation so
        # downstream compile() sees the merged config. Per-node llm_config
        # merges over the Construct default (node wins on conflicts).
        # Propagate parent defaults only when the parent has something to
        # contribute -- either an explicit framework field or any provider
        # knob. Skipping the no-op case preserves `is`-identity for
        # child nodes under a bare Construct (matches pre-pej0 behavior).
        parent_has_overrides = bool(
            self.llm_config.model_fields_set or self.llm_config.provider_kwargs
        )
        for i, item in enumerate(self.nodes):
            updates: dict[str, Any] = {}
            if parent_has_overrides and hasattr(item, "llm_config"):
                updates["llm_config"] = self.llm_config.merged_with(item.llm_config)
            if self.renderer is not None and hasattr(item, "renderer") and getattr(item, "renderer", None) is None:
                updates["renderer"] = self.renderer
            if updates:
                # All paths that populate `updates` first gate on hasattr for
                # the field — both gates only fire on BaseModel children.
                assert isinstance(item, BaseModel)
                self.nodes[i] = item.model_copy(update=updates)
        # IR-parity: inferred IR-level fields (fan_out_param, oracle_gen_type)
        # must be identical regardless of which API surface built this
        # Construct. normalize_ir is the single inference site — run once,
        # before validation, so all three surfaces converge. The @node
        # builder may pre-populate fan_out_param from richer signature data;
        # normalize_ir is idempotent and leaves an already-set field alone.
        normalize_ir(self)
        # Validate after pydantic finishes so ConstructError escapes cleanly
        # rather than being wrapped in a pydantic ValidationError. Nested
        # constructs self-validate during their own __init__.
        _validate_node_chain(self)

    # has_modifier, get_modifier, __or__, map inherited from Modifiable
