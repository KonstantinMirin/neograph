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
import warnings
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from neograph._normalize import normalize_outputs, primary_output_field
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
from neograph.spec_types import (
    _structural_type_name,
    agent_spec_properties_to_types,
    load_project_types,
    lookup_type,
)
from neograph.tool import Tool

log = structlog.get_logger()


def _import_agent_spec_import_classes() -> Any:
    """Function-local import of pyagentspec's Flow/node classes for import.

    Copies ``_agent_spec._import_agent_spec_flow_classes()``'s exact
    import-guard shape so ``src/neograph`` core stays Agent-Spec-free by
    default -- only calling ``from_agent_spec()`` pulls in the optional
    ``[agent-spec]`` extra.
    """
    try:
        import pyagentspec.flows.nodes as nodes_mod
    except ImportError as exc:
        raise ConfigurationError.build(
            "pyagentspec is not installed",
            expected="the [agent-spec] optional extra",
            found="ImportError on pyagentspec.flows.nodes",
            hint="install with: uv sync --extra agent-spec (or pip install neograph[agent-spec])",
        ) from exc
    return nodes_mod


def _agent_spec_props_to_type(props: Any) -> Any:
    """Register + look up a Pydantic model from a list of Agent Spec
    ``Property`` objects, or ``None`` if there are none.

    Reuses ``spec_types.agent_spec_properties_to_types`` (the neograph-nkjv9
    import-direction bridge) -- never a second Property walker. The
    registration NAME is derived structurally (``spec_types._structural_type_name``,
    the SAME canonical helper the nested-object reconstruction branch uses),
    not from the node's own name -- so a type appearing in two different
    places (e.g. a self-loop's own output feeding back as one of its own
    inputs) reconstructs to ONE shared class, not two incompatible ones.
    """
    if not props:
        return None

    name = _structural_type_name(props)
    agent_spec_properties_to_types(props, name)
    return lookup_type(name)


def _inputs_from_data_edges(dest_name: str, flow: Any, output_types: dict[str, Any]) -> dict[str, Any] | None:
    """Build a dict-form ``Node.inputs`` mapping from a Flow's
    ``DataFlowEdge``s targeting *dest_name*, keyed by upstream item name.

    Only edges whose source is a already-reconstructed TOP-LEVEL item (i.e.
    present in *output_types*) are considered -- this naturally excludes a
    modifier group's own INTERNAL edges (e.g. an Oracle group's variant ->
    merge fan-in), since variant/check/pause nodes never get an
    ``output_types`` entry of their own.
    """
    edges = [e for e in (flow.data_flow_connections or []) if e.destination_node.name == dest_name]
    if not edges:
        return None
    inputs: dict[str, Any] = {}
    for edge in edges:
        source_name = edge.source_node.name
        if source_name in output_types:
            inputs[source_name] = output_types[source_name]
    return inputs or None


def _reconstruct_primitive_node(spec_node: Any, flow: Any, output_types: dict[str, Any]) -> Node:
    """Reconstruct a bare (unmodified) neograph Node from an Agent Spec
    primitive node -- the inverse of ``_agent_spec._lower_node``."""
    cls_name = type(spec_node).__name__

    if cls_name == "LlmNode":
        mode, prompt, model, scripted_fn = "think", spec_node.prompt_template, spec_node.llm_config.model_id, None
    elif cls_name == "ToolNode":
        mode, prompt, model, scripted_fn = "scripted", None, None, spec_node.tool.name
    elif cls_name == "AgentNode":
        # Symmetric with the export-side gate (neograph-i3zsh.1's marker
        # exists, but the IMPORT-side reconstruction of an AgentNode+tools
        # composite back into an agent/act Node is a separate follow-up --
        # fail loud rather than silently downgrade to a plain scripted/think
        # stand-in that would drop the ReAct tool-loop semantics.
        raise ConfigurationError.build(
            f"Flow node {spec_node.name!r} is an AgentNode -- agent/act mode import "
            "is not yet supported",
            expected="LlmNode, ToolNode, or a recognized modifier composite",
            found="AgentNode",
            hint="from_agent_spec cannot yet reconstruct agent/act nodes from an "
            "AgentNode+tools composite; this is a tracked follow-up, not a silent drop",
        )
    else:
        raise ConfigurationError.build(
            f"Flow node {spec_node.name!r} has unsupported type {cls_name!r} for primitive import",
            expected="LlmNode, ToolNode, AgentNode, or FlowNode",
            found=cls_name,
        )

    outputs = _agent_spec_props_to_type(spec_node.outputs)
    # DataFlowEdges name the PRODUCER (dict-form, keyed by upstream name) --
    # but a self-contained node with no external upstream (e.g. Each's inner
    # node in its own single-node sub-flow) has no edges at all, even though
    # its OWN Property list still declares its input shape. Fall back to
    # that single-type reconstruction rather than silently dropping it.
    inputs = _inputs_from_data_edges(spec_node.name, flow, output_types) or _agent_spec_props_to_type(
        spec_node.inputs
    )
    output_types[spec_node.name] = outputs

    return Node(name=spec_node.name, mode=mode, inputs=inputs, outputs=outputs, prompt=prompt, model=model,
                scripted_fn=scripted_fn)


def _reconstruct_oracle_group(group: list[Any], flow: Any, output_types: dict[str, Any]) -> Node | None:
    """Reconstruct an Oracle-modified Node from its exported variant+merge
    group -- the inverse of ``_agent_spec._lower_oracle``.

    Returns ``None`` (and WARNs) if the marker's ``n`` no longer matches the
    ACTUAL number of variant nodes present -- a stale/hand-edited marker
    must never be blindly trusted into a silently-wrong reconstruction
    (per the Core Invariant's per-group re-lower-and-diff discipline). The
    caller falls back to importing every node in the group as a bare
    primitive.
    """

    merge_node = group[-1]
    variant_nodes = group[:-1]
    spec = merge_node.metadata["neograph/oracle_spec"]

    if len(variant_nodes) != spec["n"]:
        warnings.warn(
            f"Oracle group {merge_node.name!r}: marker declares n={spec['n']!r} but "
            f"{len(variant_nodes)} variant node(s) are actually present -- the marker is "
            "stale (hand-edited Flow). Falling back to primitive-level import for this group.",
            stacklevel=2,
        )
        return None

    base_variant = variant_nodes[0]
    base_prompt = base_variant.prompt_template
    base_model = spec.get("models")[0] if spec.get("models") else base_variant.llm_config.model_id

    outputs = _agent_spec_props_to_type(merge_node.outputs)
    inputs = _inputs_from_data_edges(merge_node.name, flow, output_types)
    output_types[merge_node.name] = outputs

    base_node = Node(name=merge_node.name, mode="think", inputs=inputs, outputs=outputs, prompt=base_prompt,
                      model=base_model)

    oracle_kwargs: dict[str, Any] = {"n": spec["n"]}
    if spec.get("models"):
        oracle_kwargs["models"] = spec["models"]
    if spec.get("merge_prompt"):
        oracle_kwargs["merge_prompt"] = spec["merge_prompt"]
        if spec.get("merge_model"):
            oracle_kwargs["merge_model"] = spec["merge_model"]
    elif spec.get("merge_fn"):
        oracle_kwargs["merge_fn"] = spec["merge_fn"]

    return base_node | Oracle(**oracle_kwargs)


def _reconstruct_each_node(map_node: Any, flow: Any, output_types: dict[str, Any]) -> Node:
    """Reconstruct an Each-modified Node from its exported MapNode --
    the inverse of ``_agent_spec._lower_each``."""

    each_spec = map_node.metadata["neograph/each_spec"]
    inner_nodes = [n for n in map_node.subflow.nodes if type(n).__name__ not in ("StartNode", "EndNode")]
    if len(inner_nodes) != 1:
        raise ConfigurationError.build(
            f"Each group {map_node.name!r}'s sub-flow has {len(inner_nodes)} inner nodes, expected 1",
            expected="exactly one inner node (Each wraps a single Node)",
            found=f"{len(inner_nodes)} inner nodes",
        )
    inner_output_types: dict[str, Any] = {}
    inner = _reconstruct_primitive_node(inner_nodes[0], map_node.subflow, inner_output_types)
    # Rename only -- KEEP the inner node's own reconstructed `inputs` (its
    # per-item Property signature, e.g. Tagged): Each's fan-out mechanism
    # feeds each item via `neo_each_item` state, not a dict-form upstream
    # mapping, so overwriting inputs with the MapNode's EXTERNAL data edges
    # (the collection producer, e.g. "seed") would be wrong -- that external
    # edge names the COLLECTION's owner, not the fanned-out item's shape.
    inner = inner.model_copy(update={"name": map_node.name})

    # _lower_each's MapNode never sets its own outputs= (only the wrapped
    # inner node's SpecNode carries the per-item output Properties) -- the
    # per-item output type is the inner node's, not the MapNode's (unset).
    if not normalize_outputs(inner.outputs).is_none:
        output_types[map_node.name] = normalize_outputs(inner.outputs).primary

    return inner | Each(over=each_spec["over"], key=each_spec.get("key"))


def _reconstruct_loop_item(body_spec: Any, check_spec: Any, flow: Any, output_types: dict[str, Any]) -> Node:
    """Reconstruct a Loop-modified Node from its exported body+check pair --
    the inverse of ``_agent_spec._lower_loop``."""

    loop_spec = check_spec.metadata["neograph/loop_spec"]
    inner_output_types: dict[str, Any] = {}
    body = _reconstruct_primitive_node(body_spec, flow, inner_output_types)

    outputs = body.outputs
    inputs = _inputs_from_data_edges(body_spec.name, flow, output_types)
    output_types[body_spec.name] = outputs
    body = body.model_copy(update={"inputs": inputs, "outputs": outputs} if inputs is not None else {})

    condition = parse_condition(loop_spec["when"]) if isinstance(loop_spec["when"], str) else loop_spec["when"]
    return body | Loop(when=condition, max_iterations=loop_spec["max_iterations"], on_exhaust=loop_spec["on_exhaust"])


def _reconstruct_operator_item(
    primary_spec: Any, check_spec: Any, flow: Any, output_types: dict[str, Any]
) -> Node:
    """Reconstruct an Operator-modified Node from its exported
    primary+check+pause composite -- the inverse of ``_agent_spec._lower_operator``."""
    operator_spec = check_spec.metadata["neograph/operator_spec"]
    inner_output_types: dict[str, Any] = {}
    primary = _reconstruct_primitive_node(primary_spec, flow, inner_output_types)

    inputs = _inputs_from_data_edges(check_spec.name, flow, output_types)
    if not normalize_outputs(primary.outputs).is_none:
        output_types[primary_spec.name] = normalize_outputs(primary.outputs).primary
    if inputs is not None:
        primary = primary.model_copy(update={"inputs": inputs})

    return primary | Operator(when=operator_spec["when"])


def _group_flow_items(flow: Any) -> list[tuple[str, Any]]:
    """Walk ``Flow.nodes`` in order, skipping Start/End sentinels, and group
    contiguous nodes into the same shapes ``to_agent_spec`` emits: a bare
    primitive, an Oracle variant+merge run (shared ``neograph/group_id``), an
    Each MapNode, a Loop body+check pair, or an Operator primary+check+pause
    triple. Returns a list of ``(kind, payload)`` tuples in item order.
    """
    nodes = flow.nodes
    n = len(nodes)
    items: list[tuple[str, Any]] = []
    i = 0
    while i < n:
        node = nodes[i]
        cls_name = type(node).__name__
        if cls_name in ("StartNode", "EndNode"):
            i += 1
            continue

        metadata = node.metadata or {}
        modifier = metadata.get("neograph/modifier")

        if modifier == "oracle":
            group_id = metadata["neograph/group_id"]
            group = [node]
            j = i + 1
            while j < n and ((nodes[j].metadata or {}).get("neograph/group_id") == group_id):
                group.append(nodes[j])
                j += 1
            items.append(("oracle", group))
            i = j
            continue

        if modifier == "each":
            items.append(("each", node))
            i += 1
            continue

        if modifier in ("loop", "operator"):
            # A floating check node with no preceding body (the lookahead
            # below always consumes body+check together) means the marker
            # doesn't match the actual structure -- fall back to primitive.
            items.append(("bare", node))
            i += 1
            continue

        # A bare node MAY be the body of a following Loop/Operator check --
        # peek ahead and confirm the control-flow edge actually connects them
        # (per the Core Invariant: never trust a marker without checking the
        # structure it claims to describe).
        nxt = nodes[i + 1] if i + 1 < n else None
        if nxt is not None:
            nxt_name = nxt.name
            nxt_modifier = (nxt.metadata or {}).get("neograph/modifier")
            edge_to_nxt = any(
                e.from_node.name == node.name and e.to_node.name == nxt_name for e in flow.control_flow_connections
            )
            if nxt_modifier == "loop" and edge_to_nxt:
                back_edge = any(
                    e.from_node.name == nxt_name and e.from_branch == "continue" and e.to_node.name == node.name
                    for e in flow.control_flow_connections
                )
                if back_edge:
                    items.append(("loop", (node, nxt)))
                    i += 2
                    continue
            if nxt_modifier == "operator" and edge_to_nxt:
                pause = nodes[i + 2] if i + 2 < n else None
                pause_edge = pause is not None and any(
                    e.from_node.name == nxt_name and e.from_branch == "pause" and e.to_node.name == pause.name
                    for e in flow.control_flow_connections
                )
                if pause_edge:
                    items.append(("operator", (node, nxt)))
                    i += 3
                    continue

        items.append(("bare", node))
        i += 1

    return items


def from_agent_spec(flow: Any) -> Construct:
    """Import an Open Agent Spec ``Flow`` into a neograph ``Construct`` --
    the inverse of ``to_agent_spec()``.

    Sibling of ``load_spec()``: import-guarded (mirrors
    ``_agent_spec._import_agent_spec_flow_classes()``) so ``src/neograph``
    core stays Agent-Spec-free by default.

    Per-group ``neograph/*_spec`` metadata markers (emitted by
    ``to_agent_spec``) are read and STRUCTURALLY VALIDATED against the
    actual primitives around them (never blindly trusted) to losslessly
    reconstruct Oracle/Each/Loop/Operator. A Flow with no markers (a
    foreign/third-party Agent Spec) imports as plain primitives. There is
    no whole-pipeline ``Flow.metadata['neograph/source']`` blob to read --
    fidelity rides only on the per-group markers, matching what
    ``to_agent_spec`` actually emits.

    Agent/act (``AgentNode``) reconstruction and Swarm-onto-Portal-mesh
    import are explicitly NOT implemented here (both fail loud rather than
    silently downgrade) -- tracked as separate follow-ups.
    """
    _import_agent_spec_import_classes()

    output_types: dict[str, Any] = {}
    pipeline_items: list[Any] = []

    for kind, payload in _group_flow_items(flow):
        if kind == "bare":
            spec_node = payload
            if type(spec_node).__name__ == "FlowNode":
                sub = from_agent_spec(spec_node.subflow)
                sub = sub.model_copy(update={"name": spec_node.name})
                output_types[spec_node.name] = sub.output
                pipeline_items.append(sub)
            else:
                pipeline_items.append(_reconstruct_primitive_node(spec_node, flow, output_types))
        elif kind == "oracle":
            reconstructed = _reconstruct_oracle_group(payload, flow, output_types)
            if reconstructed is not None:
                pipeline_items.append(reconstructed)
            else:
                # Stale marker -- fall back to importing every node in the
                # group as a bare primitive (per the Core Invariant: never
                # silently reconstruct a modifier that diverges from the
                # actual structure).
                for spec_node in payload:
                    pipeline_items.append(_reconstruct_primitive_node(spec_node, flow, output_types))
        elif kind == "each":
            pipeline_items.append(_reconstruct_each_node(payload, flow, output_types))
        elif kind == "loop":
            body_spec, check_spec = payload
            pipeline_items.append(_reconstruct_loop_item(body_spec, check_spec, flow, output_types))
        elif kind == "operator":
            primary_spec, check_spec = payload
            pipeline_items.append(_reconstruct_operator_item(primary_spec, check_spec, flow, output_types))

    return Construct(name=flow.name, nodes=pipeline_items)


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
                    # Wire each upstream to the EXACT state-bus field the
                    # validator registers, via the monopolized helpers: a
                    # dict-form producer lives at {prev}_{primary_key} with the
                    # PRIMARY type, not the raw outputs dict at the base field.
                    # Behavior-identical for single-type.
                    producer_field = primary_output_field(prev_field, prev_node.outputs)
                    inputs_dict[producer_field] = normalize_outputs(prev_node.outputs).primary
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
