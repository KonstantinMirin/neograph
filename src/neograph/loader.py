"""Spec loader -- YAML/JSON pipeline spec -> Construct IR.

    from neograph.loader import load_spec
    construct = load_spec("pipeline.yaml")
    graph = compile(construct)
    result = run(graph, input={...})

The spec is parsed into a typed ``Spec`` Pydantic model from
``_spec_schema``; typos and unknown fields raise ``ConfigurationError``
at load time. Types are resolved from a project surface or via
pre-registered entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from neograph._spec_schema import (
    ConstructSpec,
    NodeSpec,
    Spec,
    ToolSpec,
)
from neograph._state_keys import StateKeys
from neograph.conditions import parse_condition
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import Each, Loop, Operator, Oracle
from neograph.naming import field_name_for
from neograph.node import Node
from neograph.spec_types import load_project_types, lookup_type
from neograph.tool import Tool

log = structlog.get_logger()


def load_spec(
    spec: str | dict[str, Any],
    project: str | dict[str, Any] | None = None,
) -> Construct:
    """Load a pipeline spec and return a compilable Construct.

    Args:
        spec: Pipeline spec as a YAML/JSON string, a file path, or a
              pre-parsed dict.
        project: Project surface (types/tools/models) as a YAML/JSON
                 string, file path, or pre-parsed dict. Optional --
                 types can also be pre-registered via ``register_type``.

    Returns:
        A ``Construct`` ready for ``compile()``.
    """
    spec_dict = _parse_input(spec)

    # Project types must be registered before _validate_spec converts
    # spec into the typed Spec model, so lookup_type can resolve them
    # during the build phase. (Pydantic itself does not call lookup_type
    # -- it only validates the spec shape.)
    if project is not None:
        project_dict = _parse_input(project)
        load_project_types(project_dict)

    typed_spec = _validate_spec(spec_dict)
    return _build_construct(typed_spec)


# -- Parsing -----------------------------------------------------------------


MAX_SPEC_SIZE = 1_048_576  # 1 MB


def _parse_input(source: str | dict[str, Any]) -> dict[str, Any]:
    """Parse a spec source into a dict."""
    if isinstance(source, dict):
        return source

    text = source
    if len(source) <= 4096 and "\n" not in source:
        p = Path(source)
        if p.exists() and p.is_file():
            text = p.read_text()

    if len(text) > MAX_SPEC_SIZE:
        raise ConfigurationError.build(
            f"Spec exceeds maximum size ({MAX_SPEC_SIZE} bytes)",
            hint="Refusing to parse; reduce spec size or split into multiple specs.",
        )

    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    return yaml.safe_load(text)


def _validate_spec(raw: dict[str, Any]) -> Spec:
    """Parse the raw dict into a typed ``Spec`` model.

    Pydantic raises on unknown fields, type mismatches, and missing
    required fields. The resulting typed model removes every
    ``.get(key, default)`` site from the build phase.
    """
    try:
        return Spec.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError.build(
            "invalid spec",
            hint=str(exc),
        ) from exc


# -- Builder -----------------------------------------------------------------


def _build_construct(spec: Spec) -> Construct:
    """Build a Construct from a validated Spec."""
    node_defs: dict[str, Node] = {}
    explicit_inputs: set[str] = set()
    for node_spec in spec.nodes:
        node = _build_node(node_spec)
        field_name = field_name_for(node.name)
        node_defs[field_name] = node
        if node_spec.inputs:
            explicit_inputs.add(field_name)

    construct_defs: dict[str, Construct] = {}
    for construct_spec in spec.constructs:
        sub = _build_sub_construct(construct_spec, node_defs, explicit_inputs)
        field_name = field_name_for(sub.name)
        construct_defs[field_name] = sub

    pipeline_nodes: list[Any] = []
    for ref in spec.pipeline.nodes:
        field_ref = field_name_for(ref)
        if field_ref in construct_defs:
            pipeline_nodes.append(construct_defs[field_ref])
        elif field_ref in node_defs:
            pipeline_nodes.append(node_defs[field_ref])
        else:
            raise ConfigurationError.build(
                f"pipeline references unknown node or construct '{ref}'",
                hint="check that the name matches a defined node or construct in the spec",
            )

    return Construct(
        name=spec.name,
        description=spec.description,
        nodes=pipeline_nodes,
    )


def _resolve_tool(t: str | ToolSpec) -> Tool:
    """Normalize bare-string and dict-form tool entries into a typed Tool.

    Bare strings: ``Tool(name=t)``.
    Dict form: ``Tool(name, budget, config)``.

    Unknown tool names (no registered factory) raise ``ConfigurationError``
    with a hint pointing to ``register_tool_factory``.
    """
    spec = ToolSpec(name=t) if isinstance(t, str) else t
    # No factory check here — compile() validates tool factories via
    # the per-compile tool_factories= kwarg.
    return Tool(name=spec.name, budget=spec.budget, config=spec.config)


def _build_node(node_spec: NodeSpec) -> Node:
    """Build a Node from a typed NodeSpec."""
    outputs = lookup_type(node_spec.outputs)

    inputs: Any
    spec_inputs = node_spec.inputs
    if isinstance(spec_inputs, dict):
        inputs = {k: lookup_type(v) for k, v in spec_inputs.items()}
    elif isinstance(spec_inputs, str):
        inputs = lookup_type(spec_inputs)
    else:
        inputs = outputs  # single-type fallback for type-scan extraction

    node = Node(
        name=node_spec.name,
        mode=node_spec.mode,
        inputs=inputs,
        outputs=outputs,
        prompt=node_spec.prompt,
        model=node_spec.model,
        scripted_fn=node_spec.scripted_fn,
        context=node_spec.context,
        llm_config=node_spec.llm_config,
        tools=[_resolve_tool(t) for t in node_spec.tools],
    )

    return _apply_modifiers(node, node_spec)


def _build_sub_construct(
    construct_spec: ConstructSpec,
    all_nodes: dict[str, Node],
    explicit_inputs: set[str] | None = None,
) -> Construct:
    """Build a sub-Construct from a ConstructSpec."""
    name = construct_spec.name
    input_type = lookup_type(construct_spec.input)
    output_type = lookup_type(construct_spec.output)

    nodes: list[Node] = []
    for i, ref in enumerate(construct_spec.nodes):
        field_ref = field_name_for(ref)
        if field_ref not in all_nodes:
            raise ConfigurationError.build(
                f"construct references unknown node '{ref}'",
                hint="check that the node name matches a defined node in the spec",
                construct=name,
            )
        node = all_nodes[field_ref]
        if field_ref not in (explicit_inputs or set()):
            if i == 0:
                node = node.model_copy(update={"inputs": input_type})
            else:
                inputs_dict: dict[str, Any] = {StateKeys.SUBGRAPH_INPUT: input_type}
                for prev_ref in construct_spec.nodes[:i]:
                    prev_field = field_name_for(prev_ref)
                    prev_node = all_nodes[prev_field]
                    inputs_dict[prev_field] = prev_node.outputs
                node = node.model_copy(update={"inputs": inputs_dict})
        nodes.append(node)

    sub = Construct(
        name=name,
        input=input_type,
        output=output_type,
        nodes=nodes,
    )

    return _apply_modifiers(sub, construct_spec)


def _apply_modifiers(item: Any, spec: NodeSpec | ConstructSpec) -> Any:
    """Apply oracle/each/loop/operator modifiers from the typed spec."""
    if spec.oracle is not None:
        kwargs: dict[str, Any] = {}
        if spec.oracle.n is not None:
            kwargs["n"] = spec.oracle.n
        if spec.oracle.models is not None:
            kwargs["models"] = spec.oracle.models
        if spec.oracle.merge_fn is not None:
            kwargs["merge_fn"] = spec.oracle.merge_fn
        if spec.oracle.merge_prompt is not None:
            kwargs["merge_prompt"] = spec.oracle.merge_prompt
        # merge_model has a Pydantic default; only forward when set explicitly
        if "merge_model" in spec.oracle.model_fields_set:
            kwargs["merge_model"] = spec.oracle.merge_model
        item = item | Oracle(**kwargs)

    if spec.each is not None:
        item = item | Each(over=spec.each.over, key=spec.each.key)

    if spec.loop is not None:
        condition = parse_condition(spec.loop.when)
        item = item | Loop(
            when=condition,
            max_iterations=spec.loop.max_iterations,
            on_exhaust=spec.loop.on_exhaust,
        )

    if spec.operator is not None:
        item = item | Operator(when=spec.operator.when)

    return item
