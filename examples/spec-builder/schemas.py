"""Schemas for the spec-builder pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class NodeSpec(BaseModel, frozen=True):
    """A single node extracted from the workflow description."""
    name: str
    purpose: str
    mode: str = "think"
    model: str = "reason"
    inputs_description: str = ""
    outputs_description: str = ""


class TypeDefinition(BaseModel, frozen=True):
    """A Pydantic model definition extracted from the workflow."""
    name: str
    fields: list[str]
    description: str = ""


class WorkflowRequest(BaseModel, frozen=True):
    """The natural language workflow description to build a spec from."""
    description: str


class AnalysisResult(BaseModel, frozen=True):
    """Output of the analyze_request node."""
    summary: str
    node_specs: list[NodeSpec]
    type_definitions: list[TypeDefinition]
    flow_description: str


class GeneratedTypes(BaseModel, frozen=True):
    """Output of the generate_types node: Python code for Pydantic models."""
    python_code: str
    type_names: list[str]


class GeneratedSpec(BaseModel, frozen=True):
    """Output of the generate_spec node: a YAML spec string."""
    yaml_spec: str
    node_count: int
    pipeline_name: str


class ValidationResult(BaseModel, frozen=True):
    """Output of the validate_spec node."""
    valid: bool
    errors: list[str]
    warnings: list[str]
    parsed_node_names: list[str]
