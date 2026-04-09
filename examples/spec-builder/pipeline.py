"""Spec Builder -- neograph flagship demo.

Describe a workflow in natural language, get a running neograph YAML spec.

    OPENROUTER_API_KEY=sk-... python examples/spec-builder/pipeline.py

Four-stage pipeline:
  1. analyze_request  (LLM) -- understand the workflow, extract nodes/types/flow
  2. generate_types   (LLM) -- produce Pydantic model definitions as Python code
  3. generate_spec    (LLM) -- produce a neograph YAML spec
  4. validate_spec    (scripted) -- parse YAML and validate structure
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from neograph import FromInput, compile, configure_llm, construct_from_functions, node, run

from schemas import (
    AnalysisResult,
    GeneratedSpec,
    GeneratedTypes,
    ValidationResult,
    WorkflowRequest,
)


# =============================================================================
# LLM setup
# =============================================================================

MODELS = {
    "reason": "anthropic/claude-sonnet-4",
    "fast": "google/gemini-2.0-flash-001",
}


def _llm_factory(tier: str, *, node_name: str = "", llm_config: dict | None = None):
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY to run this example.")
    return ChatOpenAI(
        model=MODELS.get(tier, MODELS["fast"]),
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=(llm_config or {}).get("temperature", 0.4),
        max_tokens=(llm_config or {}).get("max_tokens", 4000),
    )


def _prompt(name: str) -> str:
    return (_HERE / "prompts" / f"{name}.md").read_text()


configure_llm(
    llm_factory=_llm_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": template}],
)


# =============================================================================
# Pipeline nodes
# =============================================================================


@node(
    outputs=AnalysisResult,
    model="reason",
    prompt=_prompt("analyze"),
)
def analyze_request(request: Annotated[WorkflowRequest, FromInput]) -> AnalysisResult: ...


@node(
    outputs=GeneratedTypes,
    model="reason",
    prompt=_prompt("generate_types"),
)
def generate_types(analyze_request: AnalysisResult) -> GeneratedTypes: ...


@node(
    outputs=GeneratedSpec,
    model="reason",
    prompt=_prompt("generate_spec"),
)
def generate_spec(generate_types: GeneratedTypes) -> GeneratedSpec: ...


@node(outputs=ValidationResult)
def validate_spec(generate_spec: GeneratedSpec) -> ValidationResult:
    """Parse the YAML spec and validate its structure."""
    errors: list[str] = []
    warnings: list[str] = []
    parsed_node_names: list[str] = []

    # Parse YAML
    try:
        parsed = yaml.safe_load(generate_spec.yaml_spec)
    except yaml.YAMLError as exc:
        return ValidationResult(
            valid=False,
            errors=[f"YAML parse error: {exc}"],
            warnings=[],
            parsed_node_names=[],
        )

    if not isinstance(parsed, dict):
        return ValidationResult(
            valid=False,
            errors=["Spec must be a YAML mapping"],
            warnings=[],
            parsed_node_names=[],
        )

    # Check top-level keys
    if "name" not in parsed:
        errors.append("Missing top-level 'name' field")

    if "nodes" not in parsed:
        errors.append("Missing top-level 'nodes' field")
        return ValidationResult(
            valid=False,
            errors=errors,
            warnings=warnings,
            parsed_node_names=[],
        )

    nodes = parsed["nodes"]
    if not isinstance(nodes, list):
        errors.append("'nodes' must be a list")
        return ValidationResult(
            valid=False,
            errors=errors,
            warnings=warnings,
            parsed_node_names=[],
        )

    # Validate each node
    seen_names: set[str] = set()
    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            errors.append(f"Node {i} is not a mapping")
            continue

        name = n.get("name")
        if not name:
            errors.append(f"Node {i} missing 'name'")
            continue

        if name in seen_names:
            errors.append(f"Duplicate node name: {name}")
        seen_names.add(name)
        parsed_node_names.append(name)

        if "mode" not in n:
            warnings.append(f"Node '{name}' missing 'mode', will default to think")

        if "outputs" not in n:
            errors.append(f"Node '{name}' missing 'outputs'")

        mode = n.get("mode", "think")
        if mode == "think" and "prompt" not in n:
            warnings.append(f"Node '{name}' is think-mode but has no prompt path")

        if mode == "scripted" and "scripted_fn" not in n:
            warnings.append(
                f"Node '{name}' is scripted but has no scripted_fn"
            )

    # Check pipeline section
    if "pipeline" in parsed:
        pipeline_nodes = parsed["pipeline"].get("nodes", [])
        for pn in pipeline_nodes:
            if pn not in seen_names:
                errors.append(
                    f"Pipeline references unknown node: {pn}"
                )
    else:
        warnings.append("No 'pipeline' section -- node order is implicit")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        parsed_node_names=parsed_node_names,
    )


# =============================================================================
# Construct
# =============================================================================

pipeline = construct_from_functions(
    "spec-builder",
    [analyze_request, generate_types, generate_spec, validate_spec],
)


# =============================================================================
# Run
# =============================================================================


def _load_sample(index: int = 0) -> WorkflowRequest:
    with open(_HERE / "data" / "sample_requests.json") as f:
        samples = json.load(f)
    return WorkflowRequest(**samples[index])


def main():
    request = _load_sample(0)
    print(f"Workflow request:\n  {request.description}\n")

    graph = compile(pipeline)
    result = run(
        graph,
        input={
            "node_id": "spec-builder-demo",
            "request": request,
        },
    )

    # Show analysis
    analysis = result["analyze_request"]
    print(f"[analyze] {analysis.summary}")
    print(f"[analyze] Nodes: {[n.name for n in analysis.node_specs]}")
    print(f"[analyze] Types: {[t.name for t in analysis.type_definitions]}")

    # Show generated types
    types = result["generate_types"]
    print(f"\n[types] Generated {len(types.type_names)} models: {types.type_names}")
    print(f"[types] Python code:\n{types.python_code}")

    # Show spec
    spec = result["generate_spec"]
    print(f"\n[spec] Pipeline: {spec.pipeline_name} ({spec.node_count} nodes)")
    print(f"[spec] YAML:\n{spec.yaml_spec}")

    # Show validation
    validation = result["validate_spec"]
    status = "VALID" if validation.valid else "INVALID"
    print(f"\n[validate] {status}")
    if validation.errors:
        print(f"[validate] Errors:")
        for e in validation.errors:
            print(f"    {e}")
    if validation.warnings:
        print(f"[validate] Warnings:")
        for w in validation.warnings:
            print(f"    {w}")
    print(f"[validate] Parsed nodes: {validation.parsed_node_names}")


if __name__ == "__main__":
    main()
