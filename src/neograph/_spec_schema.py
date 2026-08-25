"""Pydantic schema for YAML/JSON pipeline specs.

Replaces the hand-written JSON Schema (which silently allowed typos and
was permissive about ``llm_config`` extras) with typed Pydantic models.
``extra='forbid'`` rejects unknown fields at load time; nested models
provide field-path errors out of the box; ``Spec.model_json_schema()``
generates the JSON schema for non-Python consumers.

Structural shape preserved from the legacy loader:
- ``Spec.nodes`` is a flat list of ``NodeSpec`` definitions.
- ``Spec.constructs`` is a flat list of ``ConstructSpec`` sub-pipelines,
  whose ``nodes`` field holds *string references* into the top-level
  node pool (not recursive ``NodeSpec`` containment).
- ``Spec.pipeline.nodes`` is the ordered list of references that build
  the final ``Construct``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neograph._llm_config import LlmConfig


class OracleSpec(BaseModel):
    """Oracle modifier in a node or sub-construct spec."""

    model_config = ConfigDict(extra="forbid")

    n: int | None = None
    models: list[str] | None = None
    merge_fn: str | None = None
    merge_prompt: str | None = None
    merge_model: str = "reason"


class EachSpec(BaseModel):
    """Each (fan-out) modifier in a node or sub-construct spec."""

    model_config = ConfigDict(extra="forbid")

    over: str
    key: str


class LoopSpec(BaseModel):
    """Loop modifier with parsed ``when`` condition."""

    model_config = ConfigDict(extra="forbid")

    when: str
    max_iterations: int = 10
    on_exhaust: Literal["error", "last"] = "error"


class OperatorSpec(BaseModel):
    """Operator modifier with registered condition name."""

    model_config = ConfigDict(extra="forbid")

    when: str


class PortalSpec(BaseModel):
    """Portal (dynamic handoff) modifier in a node or sub-construct spec.

    Two modes, mirroring ``neograph._portal.Portal`` field-for-field (identical
    names, identical defaults where one exists) so ``_spec_loader``'s forward-
    every-explicitly-written-field pass-through gives YAML parity with the
    programmatic form BY CONSTRUCTION: peer routing (``to=``) and dynamic
    flow dispatch (``route="decide"``). ``scripted``/``conditions`` are
    Python-only CALLABLE registries (``dict[str, Callable]``) and deliberately
    omitted -- not YAML-expressible. See the spec-format docs for the exact
    narrowing this omission means for a dispatch-mode Portal's emitted flow:
    for a spec-emitted flow, the condition half is reachable only through
    ``operator.when`` (``loop.when`` is expression-parsed at LOAD time, so it
    never consults the emitted flow's condition registry).
    """

    model_config = ConfigDict(extra="forbid")

    # -- peer routing --
    to: list[str] | None = None
    route: str = "goto"
    # None = unset, mirroring Portal's value sentinels (neograph-dgbqv.6). Eager
    # defaults here would make every YAML NON-entry member carry max_hops=10, which
    # the entry-only rule now rejects on VALUE -- breaking YAML Portal support.
    trigger: Literal["output", "tool"] | None = None
    max_hops: int | None = None
    on_exhaust: Literal["error", "exit"] | None = None
    name: str | None = None

    # -- dynamic flow dispatch (route="decide") --
    spec_field: str | None = None
    input_field: str | None = None
    output: str | None = None
    on_invalid: Literal["raise", "route_to_error"] = "raise"
    error_handler: str | None = None
    max_depth: int | None = None


class ToolSpec(BaseModel):
    """Per-tool budget and config (forward-compatible alternative to bare strings)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    budget: int = 0
    config: dict[str, Any] = Field(default_factory=dict)


class NodeSpec(BaseModel):
    """A single node definition in a spec."""

    model_config = ConfigDict(extra="forbid")

    name: str
    mode: Literal["think", "agent", "act", "scripted", "raw"] = "scripted"
    inputs: str | dict[str, str] | None = None
    # Dict form mirrors ``inputs``: the canonical tool-binding node declares
    # ``{"result": Model, "tool_log": list[Entry]}``, which had no spec slot
    # (GH #9).
    outputs: str | dict[str, str]
    prompt: str | None = None
    model: str | None = None
    scripted_fn: str | None = None
    context: list[str] | None = None
    llm_config: LlmConfig = Field(default_factory=LlmConfig)
    tools: list[str | ToolSpec] = Field(default_factory=list)
    oracle: OracleSpec | None = None
    each: EachSpec | None = None
    loop: LoopSpec | None = None
    operator: OperatorSpec | None = None
    portal: PortalSpec | None = None


class ConstructSpec(BaseModel):
    """A sub-Construct definition.

    ``nodes`` references top-level :class:`NodeSpec` names by string;
    sub-constructs share the global node pool by reference, not by
    containment. Recursive ``NodeSpec`` nesting is not part of the spec
    format.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    input: str
    output: str
    #: GH #17: the NODE NAME whose output is the boundary. Distinct from
    #: ``output`` above, which is a TYPE NAME resolved via ``lookup_type`` --
    #: the two must not share a spelling. Optional: the default rule already
    #: prefers the last declared member producing ``output``.
    output_from: str | None = None
    nodes: list[str]
    oracle: OracleSpec | None = None
    each: EachSpec | None = None
    loop: LoopSpec | None = None
    operator: OperatorSpec | None = None
    portal: PortalSpec | None = None


class PipelineRef(BaseModel):
    """Ordered reference list that defines the final pipeline composition."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[str]


class Spec(BaseModel):
    """Top-level pipeline spec."""

    model_config = ConfigDict(extra="forbid")

    # Forward-compat versioning gate. Future format-breaking changes bump
    # this to '2' and add new Literal entries; specs with an unknown
    # version raise ValidationError at load time.
    version: Literal["1"] = "1"
    name: str
    description: str = ""
    types: dict[str, dict[str, Any]] = Field(default_factory=dict)
    nodes: list[NodeSpec] = Field(default_factory=list)
    constructs: list[ConstructSpec] = Field(default_factory=list)
    pipeline: PipelineRef
