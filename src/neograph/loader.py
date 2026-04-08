"""Spec loader — YAML/JSON pipeline spec -> Construct IR.

    from neograph.loader import load_spec
    construct = load_spec("pipeline.yaml")
    graph = compile(construct)
    result = run(graph, input={...})

The spec format is validated against neograph-pipeline.schema.json.
Types, tools, and models are resolved from a project surface file
or from pre-registered entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger()

from neograph.conditions import parse_condition
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import Each, Loop, Operator, Oracle
from neograph.node import Node
from neograph.spec_types import load_project_types, lookup_type


def load_spec(
    spec: str | dict[str, Any],
    project: str | dict[str, Any] | None = None,
) -> Construct:
    """Load a pipeline spec and return a compilable Construct.

    Args:
        spec: Pipeline spec as a YAML/JSON string, a file path, or a
              pre-parsed dict.
        project: Project surface (types/tools/models) as a YAML/JSON
                 string, file path, or pre-parsed dict. Optional —
                 types can also be pre-registered via ``register_type``.

    Returns:
        A ``Construct`` ready for ``compile()``.
    """
    spec_dict = _parse_input(spec)
    _validate_spec(spec_dict)

    if project is not None:
        project_dict = _parse_input(project)
        load_project_types(project_dict)

    return _build_construct(spec_dict)


# -- Parsing -----------------------------------------------------------------


def _parse_input(source: str | dict[str, Any]) -> dict[str, Any]:
    """Parse a spec source into a dict."""
    if isinstance(source, dict):
        return source

    text = source
    # Check if it's a file path
    p = Path(source)
    if p.exists() and p.is_file():
        text = p.read_text()

    # Try JSON first, then YAML
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    return yaml.safe_load(text)


def _validate_spec(spec: dict[str, Any]) -> None:
    """Validate spec against the JSON Schema (best-effort)."""
    schema_path = Path(__file__).parent / "schemas" / "neograph-pipeline.schema.json"
    if not schema_path.exists():
        log.warning("spec_validation_skipped", reason="schema file not found")
        return
    try:
        from jsonschema import validate
        schema = json.loads(schema_path.read_text())
        validate(instance=spec, schema=schema)
    except ImportError:
        log.warning("spec_validation_skipped", reason="jsonschema not installed")


# -- Builder -----------------------------------------------------------------


def _build_construct(spec: dict[str, Any]) -> Construct:
    """Build a Construct from a validated spec dict."""
    # Build all node definitions. Track which have explicit inputs.
    node_defs: dict[str, Node] = {}
    explicit_inputs: set[str] = set()
    for node_spec in spec.get("nodes", []):
        node = _build_node(node_spec)
        field_name = node.name.replace("-", "_")
        node_defs[field_name] = node
        if node_spec.get("inputs"):
            explicit_inputs.add(field_name)

    # Build sub-constructs
    construct_defs: dict[str, Construct] = {}
    for construct_spec in spec.get("constructs", []):
        sub = _build_sub_construct(construct_spec, node_defs, explicit_inputs)
        field_name = sub.name.replace("-", "_")
        construct_defs[field_name] = sub

    # Build pipeline from ordered node/construct references
    pipeline_nodes: list[Any] = []
    for ref in spec["pipeline"]["nodes"]:
        field_ref = ref.replace("-", "_")
        if field_ref in construct_defs:
            pipeline_nodes.append(construct_defs[field_ref])
        elif field_ref in node_defs:
            pipeline_nodes.append(node_defs[field_ref])
        else:
            msg = (
                f"Pipeline references '{ref}' but no node or construct "
                f"with that name exists in the spec."
            )
            raise ConfigurationError(msg)

    return Construct(
        name=spec["name"],
        description=spec.get("description", ""),
        nodes=pipeline_nodes,
    )


def _build_node(node_spec: dict[str, Any]) -> Node:
    """Build a Node from a node spec dict."""
    name = node_spec["name"]
    mode = node_spec.get("mode", "scripted")
    outputs = lookup_type(node_spec["outputs"])

    # Resolve explicit inputs if provided. When omitted, default to
    # outputs type so _extract_input can use type scanning. This works
    # for simple pipelines; sub-constructs override per-node inputs below.
    inputs_spec = node_spec.get("inputs")
    if inputs_spec:
        inputs = {k: lookup_type(v) for k, v in inputs_spec.items()}
    else:
        inputs = outputs  # single-type: _extract_input does type scanning

    node = Node(
        name=name,
        mode=mode,
        inputs=inputs,
        outputs=outputs,
        prompt=node_spec.get("prompt"),
        model=node_spec.get("model"),
        scripted_fn=node_spec.get("scripted_fn"),
        context=node_spec.get("context"),
        llm_config=node_spec.get("llm_config", {}),
    )

    # Apply modifiers
    node = _apply_modifiers(node, node_spec)

    return node


def _build_sub_construct(
    construct_spec: dict[str, Any],
    all_nodes: dict[str, Node],
    explicit_inputs: set[str] | None = None,
) -> Construct:
    """Build a sub-Construct from a construct spec dict."""
    name = construct_spec["name"]
    input_type = lookup_type(construct_spec["input"])
    output_type = lookup_type(construct_spec["output"])

    # Collect nodes for this sub-construct. Wire inputs so that each node
    # can see the input port AND all preceding peer nodes via dict-form inputs.
    # First node: inputs from sub-construct input port (neo_subgraph_input).
    # Subsequent nodes: inputs from input port + all preceding peer outputs.
    nodes: list[Node] = []
    for i, ref in enumerate(construct_spec["nodes"]):
        field_ref = ref.replace("-", "_")
        if field_ref not in all_nodes:
            msg = (
                f"Construct '{name}' references node '{ref}' "
                f"but no node with that name exists."
            )
            raise ConfigurationError(msg)
        node = all_nodes[field_ref]
        if field_ref not in (explicit_inputs or set()):
            if i == 0:
                # First node: reads from sub-construct input port
                node.inputs = input_type
            else:
                # Subsequent nodes: dict-form inputs from input port + peers
                inputs_dict: dict[str, Any] = {"neo_subgraph_input": input_type}
                for prev_ref in construct_spec["nodes"][:i]:
                    prev_field = prev_ref.replace("-", "_")
                    prev_node = all_nodes[prev_field]
                    inputs_dict[prev_field] = prev_node.outputs
                node.inputs = inputs_dict
        nodes.append(node)

    sub = Construct(
        name=name,
        input=input_type,
        output=output_type,
        nodes=nodes,
    )

    # Apply construct-level modifiers
    sub = _apply_modifiers(sub, construct_spec)

    return sub


def _apply_modifiers(item: Any, spec: dict[str, Any]) -> Any:
    """Apply oracle/each/loop/operator modifiers from the spec."""
    oracle_spec = spec.get("oracle")
    if oracle_spec:
        kwargs: dict[str, Any] = {}
        if "n" in oracle_spec:
            kwargs["n"] = oracle_spec["n"]
        if "models" in oracle_spec:
            kwargs["models"] = oracle_spec["models"]
        if "merge_fn" in oracle_spec:
            kwargs["merge_fn"] = oracle_spec["merge_fn"]
        if "merge_prompt" in oracle_spec:
            kwargs["merge_prompt"] = oracle_spec["merge_prompt"]
        if "merge_model" in oracle_spec:
            kwargs["merge_model"] = oracle_spec["merge_model"]
        item = item | Oracle(**kwargs)

    each_spec = spec.get("each")
    if each_spec:
        item = item | Each(over=each_spec["over"], key=each_spec["key"])

    loop_spec = spec.get("loop")
    if loop_spec:
        condition = parse_condition(loop_spec["when"])
        item = item | Loop(
            when=condition,
            max_iterations=loop_spec.get("max_iterations", 10),
            on_exhaust=loop_spec.get("on_exhaust", "error"),
        )

    operator_spec = spec.get("operator")
    if operator_spec:
        item = item | Operator(when=operator_spec["when"])

    return item
