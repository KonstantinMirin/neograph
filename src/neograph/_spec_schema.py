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
    outputs: str
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
    nodes: list[str]
    oracle: OracleSpec | None = None
    each: EachSpec | None = None
    loop: LoopSpec | None = None
    operator: OperatorSpec | None = None


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
