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
from typing import Any, Literal

import structlog
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError, create_model

from neograph._agent_spec import (
    _MARK_AGENT_SPEC,
    _MARK_EACH_SPEC,
    _MARK_GROUP_ID,
    _MARK_LOOP_SPEC,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    _MARK_ORACLE_SPEC,
    _MARK_PORTAL_OPERATOR_SPEC,
    _MARK_PORTAL_SPEC,
    _MARK_PROMPT_SPEC,
    _MARK_TOOL_SPEC,
)
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
from neograph.modifiers import (
    COMBO_DECOMPOSITION,
    Each,
    Loop,
    Operator,
    Oracle,
    Portal,
    PrimaryShape,
    combo_for_modifier_names,
    is_each_oracle_fused,
)
from neograph.naming import field_name_for
from neograph.node import Node
from neograph.spec_types import (
    _import_agent_spec_property_classes,
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


def _dict_form_inputs_from_props(props: Any) -> dict[str, Any] | None:
    """Reconstruct a dict-form ``Node.inputs`` from dotted-prefixed input
    Properties, the inverse of ``_agent_spec._properties_for``'s dict-form
    ``p.title = f"{key}.{p.title}"`` prefixing.

    ``to_agent_spec`` flattens a dict-form ``inputs={'k': SomeModel}`` into one
    Property per field titled ``"k.field"``. A FLAT reconstruction
    (``_agent_spec_props_to_type``) would build a single model with dotted field
    names (``{'k.field': ...}``) whose structural type hash does NOT match the
    producer's — the neograph-3lk2l / qtfof.4 type-identity loss. Grouping the
    Properties back by their ``k`` prefix and reconstructing each group's model
    from the UN-prefixed field Properties restores the original per-key type, so
    the Each fan-out receiver reconstructs to the SAME structural class as the
    producer's list element. Returns ``None`` when no Property is dotted (leave
    the caller's default single-type reconstruction in charge)."""
    if not props:
        return None
    groups: dict[str, list[Any]] = {}
    for p in props:
        if "." not in p.title:
            return None
        key, _, rest = p.title.partition(".")
        groups.setdefault(key, []).append(p.model_copy(update={"title": rest}))
    return {key: _agent_spec_props_to_type(group) for key, group in groups.items()} or None


def _augment_inputs_from_prompt_marker(
    inputs: Any, marker: dict[str, Any], output_types: dict[str, Any]
) -> dict[str, Any] | None:
    """Restore the ORIGINAL dict-form ``Node.inputs`` from a ``neograph/prompt_spec``
    marker's JSON-native ``original_inputs`` (Option F, neograph-cbpyx).

    An Option-F-translated ``LlmNode``/variant declares only the referenced FLAT
    Properties, so its own ``inputs`` (and its DataFlowEdges) have lost every input
    the prompt never referenced. Starting from the edge-derived ``inputs`` (which
    already carry the CORRECT producer types for referenced inputs -- the SAME
    structural type object the producer registered), this ADDS every original input
    key the edges missed. A dict-form input KEY is the upstream NODE name, so a
    missing key's type is taken from that producer's already-reconstructed output
    (``output_types[key]``) -- guaranteeing type-identity with the producer for the
    fan-in validator, and staying stable across a JSON wire round trip (both sides
    reconstruct from the SAME producer node). Only when no same-named producer
    exists does it fall back to rebuilding the type from the marker's JSON-native
    ``json_schema`` (a bare ``Property`` that ``spec_types._property_to_field_type``
    resolves via ``json_schema``), regrouped by the EXISTING
    ``_dict_form_inputs_from_props``."""
    entries = marker.get("original_inputs")
    if not entries:
        return inputs if isinstance(inputs, dict) else None

    ordered_keys: list[str] = []
    for e in entries:
        key = e["title"].split(".", 1)[0]
        if key not in ordered_keys:
            ordered_keys.append(key)

    result: dict[str, Any] = dict(inputs) if isinstance(inputs, dict) else {}
    for key in ordered_keys:
        if key in result:
            continue
        if key in output_types and output_types[key] is not None:
            result[key] = output_types[key]
            continue
        pas = _import_agent_spec_property_classes()
        group = [
            pas.Property.model_construct(
                title=e["title"].partition(".")[2],
                json_schema=e["json_schema"],
                type=e["json_schema"].get("type"),
                description=None,
                default=None,
            )
            for e in entries
            if e["title"].split(".", 1)[0] == key
        ]
        result[key] = _agent_spec_props_to_type(group)
    return result or None


def _tools_from_marker(marker_tools: list[dict[str, Any]]) -> list[Tool]:
    """Rebuild neograph ``Tool`` specs from the flat ``neograph/agent_spec``
    tools list (the EXACT inverse of ``_agent_spec._agent_spec_marker``'s
    ``tools=[{name, budget, config, idempotent}, ...]`` blob)."""
    return [
        Tool(t["name"], budget=t["budget"], config=t["config"], idempotent=t["idempotent"])
        for t in marker_tools
    ]


def _tools_from_foreign_agent(agent: Any) -> list[Tool]:
    """Best-effort rebuild of ``Tool`` specs from a foreign ``Agent``'s
    ``tools`` (each a ``ServerTool``). A ServerTool that still carries a
    ``neograph/tool_spec`` marker restores budget/config/idempotent; a truly
    foreign one is reconstructed name-only. Returns an empty list (never None --
    ``Node.tools`` rejects None) when the agent declares no tools."""
    tools: list[Tool] = []
    for st in getattr(agent, "tools", None) or []:
        ts = (getattr(st, "metadata", None) or {}).get(_MARK_TOOL_SPEC)
        if ts:
            tools.append(Tool(ts["name"], budget=ts["budget"], config=ts["config"], idempotent=ts["idempotent"]))
        else:
            tools.append(Tool(st.name))
    return tools


def _node_from_spec_agent(
    name: str, agent: Any, marker: dict[str, Any] | None, inputs: Any, outputs: Any
) -> Node:
    """Build an agent/act ``Node`` from an Agent Spec agent, dispatching on
    marker presence (refinement addendum MEDIUM-2).

    * ``marker`` present (a neograph-exported ``AgentNode``) -> LOSSLESS gap-1
      inversion: mode/prompt/model/tools (incl. each Tool's
      budget/config/idempotent)/gate_tools_when/context all restored from the
      ``neograph/agent_spec`` blob.
    * ``marker`` absent (a FOREIGN ``Agent``, e.g. a Swarm member) -> best-effort
      gap-3 reconstruction: mode='agent' (read-only, conservative) built from
      the plain ``Agent``'s ``system_prompt`` / ``llm_config.model_id`` /
      ``tools``.

    Shared by BOTH the AgentNode branch (gaps 1/3) and the Swarm member builder
    (gap 2) -- a raw ``metadata[_MARK_AGENT_SPEC]`` read in the Swarm path would
    ``KeyError`` on foreign agents that carry no marker.
    """
    if marker is not None:
        return Node(
            name=name,
            mode=marker["mode"],
            inputs=inputs,
            outputs=outputs,
            prompt=marker["prompt"],
            model=marker["model"],
            tools=_tools_from_marker(marker["tools"]),
            gate_tools_when=marker["gate_tools_when"],
            context=marker["context"],
        )

    llm_config = getattr(agent, "llm_config", None)
    return Node(
        name=name,
        mode="agent",
        inputs=inputs,
        outputs=outputs,
        prompt=getattr(agent, "system_prompt", None) or None,
        model=getattr(llm_config, "model_id", None),
        tools=_tools_from_foreign_agent(agent),
    )


# Per-family endpoint attribute names for the client-initiated remote-agent
# best-effort branch -- verified against the installed
# pyagentspec SDK, NOT a blind two-field getattr shared across families.
# RemoteAgent itself is abstract (no endpoint fields of its own) and has no
# instantiable concrete form beyond the two families below.
_REMOTE_AGENT_ENDPOINT_ATTRS: dict[str, tuple[str, ...]] = {
    "A2AAgent": ("agent_url", "connection_config"),
    "OciAgent": ("agent_endpoint_id", "client_config"),
}


def _reconstruct_agent_node(spec_node: Any, inputs: Any, outputs: Any) -> Node:
    """Reconstruct a neograph Node from an ``AgentNode`` (gaps 1 & 3).

    Dispatches on the ``neograph/agent_spec`` marker and the wrapped agent's
    runtime type:

    * marker present -> LOSSLESS agent/act reconstruction (gap 1).
    * marker absent, ``.agent`` is a plain ``Agent`` -> best-effort agent-mode
      (gap 3, foreign agent).
    * marker absent, ``.agent`` is a client-initiated
      ``RemoteAgent``/``A2AAgent``/``OciAgent`` -> name-bound scripted stand-in
      + WARNING (gap 3, ratification §3b): never a silent drop, never fail-loud.
    * anything else (e.g. a ServerTool-as-agent, an orchestrator-side surface)
      -> FAIL LOUD (the principled line, ratification §3b).
    """
    marker = (getattr(spec_node, "metadata", None) or {}).get(_MARK_AGENT_SPEC)
    agent = spec_node.agent
    agent_type = type(agent).__name__

    if marker is not None or agent_type == "Agent":
        return _node_from_spec_agent(spec_node.name, agent, marker, inputs, outputs)

    if agent_type in ("RemoteAgent", "A2AAgent", "OciAgent"):
        warnings.warn(
            f"AgentNode {spec_node.name!r} wraps a client-initiated {agent_type} with no "
            "neograph agent-spec marker -- importing best-effort as a name-bound scripted "
            f"Node (scripted_fn={spec_node.name!r}); the runtime binds the endpoint at compile "
            "time. This is a downgrade of the remote-agent semantics, not a lossless import.",
            stacklevel=2,
        )
        node = Node(
            name=spec_node.name,
            mode="scripted",
            inputs=inputs,
            outputs=outputs,
            scripted_fn=spec_node.name,
        )
        attr_names = _REMOTE_AGENT_ENDPOINT_ATTRS.get(agent_type, ())
        node._remote_agent_endpoint = (agent_type, {name: getattr(agent, name) for name in attr_names})
        return node

    raise ConfigurationError.build(
        f"Flow node {spec_node.name!r} is an AgentNode wrapping a {agent_type!r} with no "
        "neograph agent-spec marker -- no best-effort lowering",
        expected="a neograph-exported Agent (marker), a plain Agent, or a client-initiated "
        "RemoteAgent/A2AAgent/OciAgent",
        found=f"AgentNode.agent is a {agent_type}",
        hint="orchestrator-side agents (e.g. a ServerTool-as-agent) have no client-initiated "
        "handoff semantics to lower -- this fails loud rather than silently downgrade "
        "(ratification agent-spec-ratification-2026-07-13.md §3b)",
    )


def _reconstruct_primitive_node(spec_node: Any, flow: Any, output_types: dict[str, Any]) -> Node:
    """Reconstruct a bare (unmodified) neograph Node from an Agent Spec
    primitive node -- the inverse of ``_agent_spec._lower_node``."""
    cls_name = type(spec_node).__name__

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

    if cls_name == "AgentNode":
        # gap 1 (lossless marker inversion) + gap 3 (foreign/remote best-effort).
        # Option F neograph-cbpyx: a translated agent/act node declares only the
        # flat placeholder inputs, so restore the ORIGINAL input TypeSpec from the
        # neograph/prompt_spec marker (an Each fan-out receiver must round-trip to
        # the producer's list element type -- neograph-3lk2l).
        prompt_marker = (getattr(spec_node, "metadata", None) or {}).get(_MARK_PROMPT_SPEC)
        if prompt_marker is not None:
            inputs = _augment_inputs_from_prompt_marker(inputs, prompt_marker, output_types)
        return _reconstruct_agent_node(spec_node, inputs, outputs)

    if cls_name == "LlmNode":
        # Option F neograph-cbpyx: prefer the neograph/prompt_spec marker -- it
        # carries the UNtranslated ${var} prompt and the FULL original inputs
        # (incl. any the prompt never referenced, whose flat LlmNode dropped both
        # Property and DataFlowEdge). Fall back to the translated prompt_template /
        # data-edge inputs for a pre-Option-F or foreign LlmNode with no marker.
        marker = (getattr(spec_node, "metadata", None) or {}).get(_MARK_PROMPT_SPEC)
        if marker is not None:
            prompt = marker["original_text"]
            inputs = _augment_inputs_from_prompt_marker(inputs, marker, output_types)
        else:
            prompt = spec_node.prompt_template
        mode, model, scripted_fn = "think", spec_node.llm_config.model_id, None
    elif cls_name == "ToolNode":
        mode, prompt, model, scripted_fn = "scripted", None, None, spec_node.tool.name
    else:
        raise ConfigurationError.build(
            f"Flow node {spec_node.name!r} has unsupported type {cls_name!r} for primitive import",
            expected="LlmNode, ToolNode, AgentNode, or FlowNode",
            found=cls_name,
        )

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
    spec = merge_node.metadata[_MARK_ORACLE_SPEC]

    if len(variant_nodes) != spec["n"]:
        warnings.warn(
            f"Oracle group {merge_node.name!r}: marker declares n={spec['n']!r} but "
            f"{len(variant_nodes)} variant node(s) are actually present -- the marker is "
            "stale (hand-edited Flow). Falling back to primitive-level import for this group.",
            stacklevel=2,
        )
        return None

    # neograph-m57mn Option A: variants no longer lower unconditionally to
    # LlmNode (_agent_spec._lower_oracle now dispatches per node.mode), so
    # reconstruction must dispatch per the variant's ACTUAL Agent Spec type
    # too -- the inverse of that same dispatch, mirroring
    # _reconstruct_primitive_node's LlmNode/ToolNode branching.
    base_variant = variant_nodes[0]
    base_cls = type(base_variant).__name__
    base_prompt_marker = (getattr(base_variant, "metadata", None) or {}).get(_MARK_PROMPT_SPEC)

    outputs = _agent_spec_props_to_type(merge_node.outputs)
    # Option F neograph-cbpyx: a think/agent variant's external inputs are
    # translated to flat placeholder names, so prefer the variant marker's
    # ORIGINAL dict-form inputs (fallback to the merge node's data-edge inputs
    # for scripted/foreign).
    inputs = _inputs_from_data_edges(merge_node.name, flow, output_types)
    if base_prompt_marker is not None:
        inputs = _augment_inputs_from_prompt_marker(inputs, base_prompt_marker, output_types)
    output_types[merge_node.name] = outputs

    if base_cls == "AgentNode":
        # Design B round-trip, neograph-i7k7j: the variant is a real AgentNode+Agent
        # (the inverse of _lower_oracle's agent/act branch). Reuse the primitive
        # AgentNode reconstructor to recover mode/prompt/model/tools/gate_tools_when/
        # context from the neograph/agent_spec marker, then rename it to the GROUP
        # name (the merge node) and attach Oracle below. The per-variant model tier
        # is discarded here -- it round-trips via the Oracle marker's `models`, so the
        # marker (built from the BASE node) already carries the base model.
        base_node = _reconstruct_agent_node(base_variant, inputs, outputs).model_copy(
            update={"name": merge_node.name}
        )
    else:
        if base_cls == "LlmNode":
            # Option F neograph-cbpyx: prefer the variant's neograph/prompt_spec
            # marker for the UNtranslated base prompt (fallback to the translated
            # prompt_template for a pre-Option-F/foreign variant).
            base_prompt = base_prompt_marker["original_text"] if base_prompt_marker else base_variant.prompt_template
            base_mode, base_scripted_fn = "think", None
            base_model = spec.get("models")[0] if spec.get("models") else base_variant.llm_config.model_id
        elif base_cls == "ToolNode":
            base_mode, base_prompt, base_scripted_fn = "scripted", None, base_variant.tool.name
            base_model = spec.get("models")[0] if spec.get("models") else None
        else:
            raise ConfigurationError.build(
                f"Oracle group {merge_node.name!r}'s variant node has unsupported type {base_cls!r}",
                expected="LlmNode, ToolNode, or AgentNode",
                found=base_cls,
            )
        base_node = Node(name=merge_node.name, mode=base_mode, inputs=inputs, outputs=outputs, prompt=base_prompt,
                          model=base_model, scripted_fn=base_scripted_fn)

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

    each_spec = map_node.metadata[_MARK_EACH_SPEC]
    inner_nodes = _subflow_inner_nodes(map_node)
    if len(inner_nodes) != 1:
        raise ConfigurationError.build(
            f"Each group {map_node.name!r}'s sub-flow has {len(inner_nodes)} inner nodes, expected 1",
            expected="exactly one inner node (Each wraps a single Node)",
            found=f"{len(inner_nodes)} inner nodes",
        )
    inner_output_types: dict[str, Any] = {}
    inner_spec = inner_nodes[0]
    inner = _reconstruct_primitive_node(inner_spec, map_node.subflow, inner_output_types)
    # Rename only -- KEEP the inner node's own reconstructed `inputs` (its
    # per-item Property signature, e.g. Tagged): Each's fan-out mechanism
    # feeds each item via `neo_each_item` state, not a dict-form upstream
    # mapping, so overwriting inputs with the MapNode's EXTERNAL data edges
    # (the collection producer, e.g. "seed") would be wrong -- that external
    # edge names the COLLECTION's owner, not the fanned-out item's shape.
    update: dict[str, Any] = {"name": map_node.name}
    # PRIMARY @node shape (map_over= / dict-form inputs): the inner node's input
    # Properties are dotted-prefixed ("cluster.v", "source.c"). Un-group them
    # back to per-key types so the fan-out receiver reconstructs to the SAME
    # structural class as the producer's list element (Each's assembly checks
    # then pass through the dict-form fan_out_param skip, exactly like the
    # original @node did) instead of a flat {"cluster.v": ...} model whose hash
    # never matches -- neograph-3lk2l. The single-type inner shape (the legacy
    # ``Node.scripted(inputs=Tagged)`` form, no dot) is left untouched.
    dict_inputs = _dict_form_inputs_from_props(inner_spec.inputs)
    if dict_inputs is not None:
        update["inputs"] = dict_inputs
    inner = inner.model_copy(update=update)

    # _lower_each's MapNode never sets its own outputs= (only the wrapped
    # inner node's SpecNode carries the per-item output Properties) -- the
    # per-item output type is the inner node's, not the MapNode's (unset).
    if not normalize_outputs(inner.outputs).is_none:
        output_types[map_node.name] = normalize_outputs(inner.outputs).primary

    return inner | Each(over=each_spec["over"], key=each_spec.get("key"))


def _reconstruct_loop_item(body_spec: Any, check_spec: Any, flow: Any, output_types: dict[str, Any]) -> Node:
    """Reconstruct a Loop-modified Node from its exported body+check pair --
    the inverse of ``_agent_spec._lower_loop``."""

    loop_spec = check_spec.metadata[_MARK_LOOP_SPEC]
    inner_output_types: dict[str, Any] = {}
    body = _reconstruct_primitive_node(body_spec, flow, inner_output_types)

    outputs = body.outputs
    inputs = _inputs_from_data_edges(body_spec.name, flow, output_types)
    output_types[body_spec.name] = outputs
    body = body.model_copy(update={"inputs": inputs, "outputs": outputs} if inputs is not None else {})

    condition = parse_condition(loop_spec["when"]) if isinstance(loop_spec["when"], str) else loop_spec["when"]
    return body | Loop(when=condition, max_iterations=loop_spec["max_iterations"], on_exhaust=loop_spec["on_exhaust"])


def _reconstruct_operator_primary(primary_spec: Any, flow: Any, output_types: dict[str, Any]) -> Node:
    """Reconstruct the BODY node of a BARE+Operator composite, with its external
    inputs routed -- the inverse of the ``PrimaryShape.BARE`` half of
    ``_agent_spec._lower_construct_item``'s Operator path.

    Returns the node WITHOUT the ``| Operator(...)`` pipe: the caller applies
    that in the one shared postlude, so Operator composes onto every primary
    shape the same way (neograph-s7zt3.10) rather than being fused into one
    shape's reconstructor.
    """
    inner_output_types: dict[str, Any] = {}
    primary = _reconstruct_primitive_node(primary_spec, flow, inner_output_types)

    inputs = _inputs_from_data_edges(primary_spec.name, flow, output_types)
    if not normalize_outputs(primary.outputs).is_none:
        output_types[primary_spec.name] = normalize_outputs(primary.outputs).primary
    if inputs is not None:
        primary = primary.model_copy(update={"inputs": inputs})

    return primary


def _reconstruct_fused_each_oracle_node(map_node: Any, output_types: dict[str, Any]) -> Node:
    """Reconstruct an Each x Oracle FUSED Node from its exported MapNode whose
    sub-flow is an Oracle variant+merge group -- the inverse of
    ``_agent_spec._lower_each(node, each, oracle=...)``.

    Composes the two EXISTING reconstructors rather than adding a third: the
    nested group goes through ``_reconstruct_oracle_group`` (against the SUB-flow,
    which is where its fan-in data edges live), then ``| Each(...)`` is piped on.
    """
    each_spec = map_node.metadata[_MARK_EACH_SPEC]
    inner_nodes = _subflow_oracle_group(map_node)
    if inner_nodes is None:  # pragma: no cover - the walk only routes here on a match
        raise ConfigurationError.build(
            f"Each group {map_node.name!r} was recognized as fused but its sub-flow holds no Oracle group",
            expected="an Oracle variant+merge run sharing one neograph/group_id",
            found="no shared group_id",
        )

    inner_output_types: dict[str, Any] = {}
    inner = _reconstruct_oracle_group(inner_nodes, map_node.subflow, inner_output_types)
    if inner is None:
        raise ConfigurationError.build(
            f"Each group {map_node.name!r}'s nested Oracle group did not reconstruct",
            expected="a variant+merge run whose structure matches its neograph/modifier=oracle marker",
            found="stale or inconsistent oracle markers inside the MapNode sub-flow",
        )

    # neograph-3lk2l inside the fusion: the fan-out receiver's element type is
    # recovered from the FIRST VARIANT's dotted input Properties -- the fused
    # analogue of the un-fused path's "inner spec" (there is no single inner
    # node here, and the merge node's inputs are the variant OUTPUTS, not the
    # per-item input shape). Without this the receiver reconstructs as a flat
    # {"cluster.v": ...} model whose hash never matches the producer's element.
    update: dict[str, Any] = {"name": map_node.name}
    dict_inputs = _dict_form_inputs_from_props(inner_nodes[0].inputs)
    if dict_inputs is not None:
        update["inputs"] = dict_inputs
    inner = inner.model_copy(update=update)

    if not normalize_outputs(inner.outputs).is_none:
        output_types[map_node.name] = normalize_outputs(inner.outputs).primary

    return inner | Each(over=each_spec["over"], key=each_spec.get("key"))


def _subflow_inner_nodes(map_node: Any) -> list[Any]:
    """The real (non-sentinel) nodes inside an Each MapNode's sub-flow.

    The ONE place that descends into a ``MapNode.subflow``, shared by the plain
    Each reconstructor and the fusion recognizer so the descent is not walked
    twice under two names.
    """
    inner_nodes = [n for n in map_node.subflow.nodes if type(n).__name__ not in ("StartNode", "EndNode")]
    return inner_nodes


def _subflow_oracle_group(map_node: Any) -> list[Any] | None:
    """The Oracle variant+merge run nested INSIDE an Each MapNode's sub-flow,
    or None when the MapNode wraps a plain single body.

    This is how an Each x Oracle FUSION is recognized on import -- structurally,
    by descending into the sub-flow and finding a shared ``neograph/group_id``
    run, NOT by a dedicated marker on the MapNode. Per the loader's Core
    Invariant, a marker is never trusted without confirming the structure it
    claims to describe, so there is nothing a new marker could add here.
    """
    inner = _subflow_inner_nodes(map_node)
    if len(inner) < 2:
        return None
    group_id = (inner[0].metadata or {}).get(_MARK_GROUP_ID)
    if group_id is None or (inner[0].metadata or {}).get(_MARK_MODIFIER) != "oracle":
        return None
    if any((node.metadata or {}).get(_MARK_GROUP_ID) != group_id for node in inner):
        return None
    return inner


def _trailing_operator(nodes: list[Any], j: int, primary_spec: Any, flow: Any) -> Any | None:
    """Variable-length lookahead for the Operator pause composite that may
    follow ANY primary group (bare / Each / Oracle / Loop). Returns the check
    ``BranchingNode`` on a match -- the caller then consumes TWO nodes (check +
    pause) -- or None.

    ONE helper, called from every branch of the recognition walk, because
    ``_lower_operator`` emits the SAME three-part composite regardless of what
    it wraps. Structural confirmation, never marker trust: the marker must be
    backed by a real ``primary -> check`` edge AND a real ``check --pause-->``
    edge, or the shape is not an Operator and the nodes import as primitives.
    """
    if j + 1 >= len(nodes):
        return None
    check = nodes[j]
    if (check.metadata or {}).get(_MARK_MODIFIER) != "operator":
        return None
    if not any(
        e.from_node.name == primary_spec.name and e.to_node.name == check.name
        for e in flow.control_flow_connections
    ):
        return None
    pause = nodes[j + 1]
    if not any(
        e.from_node.name == check.name and e.from_branch == "pause" and e.to_node.name == pause.name
        for e in flow.control_flow_connections
    ):
        return None
    return check


def _group_flow_items(flow: Any) -> list[tuple[frozenset[str], dict[str, Any]]]:
    """Walk ``Flow.nodes`` in order, skipping Start/End sentinels, and group
    contiguous nodes into the shapes ``to_agent_spec`` emits.

    RECOGNIZE, then CLASSIFY (the import-side mirror of the export dispatch):
    each group yields the frozenset of modifier NAMES its structure encodes,
    which ``from_agent_spec`` hands to ``combo_for_modifier_names`` and
    dispatches on ``COMBO_DECOMPOSITION``. Emitting a fixed ``kind`` string per
    shape does not survive composition -- there are 12 combos and only five
    primary shapes, so a string enumeration would have to re-derive the
    decomposition the table already owns.

    The payload dict carries the recognized spec nodes: ``primary`` (bare),
    ``group`` (Oracle run), ``map_node`` (Each), ``body``/``check`` (Loop), and
    ``operator_check`` whenever the trailing pause composite was recognized.
    """
    nodes = flow.nodes
    n = len(nodes)
    items: list[tuple[frozenset[str], dict[str, Any]]] = []
    i = 0
    while i < n:
        node = nodes[i]
        cls_name = type(node).__name__
        if cls_name in ("StartNode", "EndNode"):
            i += 1
            continue

        metadata = node.metadata or {}
        modifier = metadata.get(_MARK_MODIFIER)

        names: set[str] = set()
        payload: dict[str, Any] = {}

        if modifier == "oracle":
            group_id = metadata[_MARK_GROUP_ID]
            group = [node]
            j = i + 1
            while j < n and ((nodes[j].metadata or {}).get(_MARK_GROUP_ID) == group_id):
                group.append(nodes[j])
                j += 1
            names.add("oracle")
            payload["group"] = group
            # The merge node (last in the run) is the group's control-flow
            # identity -- the node a trailing Operator check attaches to.
            primary_spec = group[-1]

        elif modifier == "each":
            names.add("each")
            payload["map_node"] = node
            if _subflow_oracle_group(node) is not None:
                names.add("oracle")  # Each x Oracle fusion
            primary_spec = node
            j = i + 1

        elif modifier in ("loop", "operator"):
            # A floating check node with no preceding body (the lookahead
            # below always consumes body+check together) means the marker
            # doesn't match the actual structure -- fall back to primitive.
            payload["primary"] = node
            primary_spec = node
            j = i + 1

        else:
            # A bare node MAY be the body of a following Loop check -- peek ahead
            # and confirm the control-flow edge actually connects them (per the
            # Core Invariant: never trust a marker without checking the structure
            # it claims to describe). A trailing OPERATOR is not special-cased
            # here: it is the shared postlude below, exactly as on the export side.
            payload["primary"] = node
            primary_spec = node
            j = i + 1
            nxt = nodes[i + 1] if i + 1 < n else None
            if nxt is not None and (nxt.metadata or {}).get(_MARK_MODIFIER) == "loop":
                edge_to_nxt = any(
                    e.from_node.name == node.name and e.to_node.name == nxt.name
                    for e in flow.control_flow_connections
                )
                back_edge = any(
                    e.from_node.name == nxt.name and e.from_branch == "continue" and e.to_node.name == node.name
                    for e in flow.control_flow_connections
                )
                if edge_to_nxt and back_edge:
                    names.add("loop")
                    payload.pop("primary")
                    payload["body"] = node
                    payload["check"] = nxt
                    primary_spec = nxt
                    j = i + 2

        # ONE shared trailing-Operator lookahead for every primary group above.
        check = _trailing_operator(nodes, j, primary_spec, flow)
        if check is not None:
            names.add("operator")
            payload["operator_check"] = check
            j += 2

        items.append((frozenset(names), payload))
        i = j

    return items


def _swarm_agents_ordered(swarm: Any) -> list[Any]:
    """The Swarm's agents in mesh order: ``first_agent`` FIRST (it becomes the
    entry member, so ``max_hops``/``on_exhaust`` ride its Portal), then every
    other agent in order of first appearance across the directed
    ``relationships`` tuples. Deduplicated by identity."""
    ordered: list[Any] = []
    seen: set[int] = set()

    def _add(agent: Any) -> None:
        if id(agent) not in seen:
            seen.add(id(agent))
            ordered.append(agent)

    _add(swarm.first_agent)
    for src, dst in swarm.relationships:
        _add(src)
        _add(dst)
    return ordered


def _synthesize_swarm_payload(swarm: Any, agents: list[Any]) -> type[BaseModel]:
    """Synthesize the SINGLE uniform mesh payload model reused (by identity)
    as every member's ``outputs`` AND ``inputs['handoff']`` -- ``_check_portal_mesh``
    checks uniform-payload and the reserved handoff key with ``is`` identity.

    Always carries the Portal ``route`` field (``goto: str``). The Swarm's
    declared Properties (its own ``inputs``/``outputs``) and each agent's output
    schema are FOLDED IN as optional data fields where present, so a
    data-carrying Swarm does not silently lose its data channel (refinement
    addendum: no-silent-downgrade)."""
    fields: dict[str, Any] = {"goto": (str, ...)}

    prop_sources = [getattr(swarm, "outputs", None), getattr(swarm, "inputs", None)]
    prop_sources.extend(getattr(agent, "outputs", None) for agent in agents)
    for props in prop_sources:
        model = _agent_spec_props_to_type(props)
        if model is None:
            continue
        for fname, finfo in model.model_fields.items():
            if fname == "goto" or fname in fields:
                continue
            # Optional (default None) so a member that does not fill every folded
            # field can still emit the uniform payload.
            fields[fname] = (finfo.annotation | None, None)

    return create_model(f"{swarm.name}_SwarmHandoff", __base__=BaseModel, **fields)


def _swarm_trigger(swarm: Any) -> Literal["output", "tool"]:
    """Map a ``Swarm.handoff`` mode onto a native Portal ``trigger`` (C1 import,
    grounded in the swarm-langgraph-compilation-spike).

    ``HandoffMode.OPTIONAL``/``ALWAYS`` (or the bool ``True``) -> ``"tool"``: the
    reference LangGraph adapter compiles BOTH to one bound ``transfer_to_<peer>``
    tool per relationship (byte-identical — do NOT try to split OPTIONAL vs
    ALWAYS, that distinction does not materially exist in the backend), which
    neograph now represents natively as ``Portal(trigger="tool")`` (s7zt3.14).
    ``HandoffMode.NEVER`` (or ``False``/absent) -> ``"output"``: the member never
    initiates a transfer, so it keeps the existing typed-``goto`` routing.
    """
    handoff = getattr(swarm, "handoff", None)
    if handoff is True:
        return "tool"
    if handoff is False or handoff is None:
        return "output"
    value = getattr(handoff, "value", handoff)  # HandoffMode enum -> its str value
    return "tool" if value in ("optional", "always") else "output"


def _flow_member_to_construct(agent: Any, payload: type[BaseModel]) -> Construct:
    """Reconstruct a Flow Swarm member (C1) onto a Construct mesh member whose
    boundary I/O is the synthesized uniform mesh ``payload`` (by identity).

    Reuses the SAME ``from_agent_spec`` FlowNode->Construct recursion the
    bare-FlowNode item path uses, then forces BOTH the Construct boundary
    (``input``/``output``) AND its terminal interior producer's ``outputs`` to
    ``payload`` — the sub-construct output-boundary validator requires an internal
    node to actually produce the declared ``output`` type, and a round-tripped
    interior re-synthesizes its own per-node types (agent-spec export does not
    preserve type identity), so the boundary alone would not satisfy it. This is
    the Construct-member analog of forcing an Agent member's ``outputs=payload``
    (a foreign Swarm member never produced a ``goto`` value either) — a best-
    effort structural import, flagged by the caller's mesh-level warning, not a
    behaviorally-faithful foreign runtime. A non-Node terminal item (e.g. a
    nested Construct) is left untouched: it will fail LOUD at the boundary check
    rather than being silently mis-coerced, per the maintainer's fail-loud-over-
    silent default for the interior-already-fixed case.
    """
    sub = from_agent_spec(agent)
    nodes = list(sub.nodes)
    if nodes and isinstance(nodes[-1], Node):
        nodes[-1] = nodes[-1].model_copy(update={"outputs": payload})
    return sub.model_copy(update={"name": agent.name, "input": payload, "output": payload, "nodes": nodes})


def _reconstruct_swarm_mesh(swarm: Any) -> Construct:
    """Import a foreign pyagentspec ``Swarm`` onto a native Portal peer mesh
    (gap 2, ratification §3a).

    A ``Swarm`` is a top-level ``AgenticComponent`` (NOT a ``Flow`` node), so it
    is dispatched here at the ``from_agent_spec`` entry, not through
    ``_reconstruct_primitive_node``. Each Swarm agent becomes an agent-mode
    member ``Node(inputs={'handoff': Payload}, outputs=Payload) | Portal(to=[peers])``;
    ``first_agent`` is the entry (``nodes[0]``). ``handoff_param``/``handoff_channel``
    are NOT set here -- ``normalize_ir`` is their sole writer (fires in
    ``Construct.__init__``) -- and the assembled mesh is validated by the same
    ``_check_portal_mesh`` a hand-written mesh gets.

    This is a best-effort/warning arm (Core Invariant no-silent-downgrade): the
    synthesized payload is route-centric and the members are name-bound live-LLM
    agents needing a factory at compile.
    """
    agents = _swarm_agents_ordered(swarm)
    payload = _synthesize_swarm_payload(swarm, agents)

    # B3: the entry-only Portal knobs (max_hops/on_exhaust/route) ride the
    # neograph/portal_spec marker on export (_lower_portal_mesh_to_swarm) but
    # were silently dropped on re-import. Read them back the SAME way the sibling
    # reconstructors read their _MARK_*_SPEC markers (e.g. _reconstruct_oracle_group,
    # _reconstruct_loop_item) and apply them to the ENTRY member's Portal only --
    # max_hops/on_exhaust are entry-only per _check_portal_mesh, and route is
    # taken from the entry. Foreign Swarms have no marker -> plain Portal(to=peers).
    portal_spec = (getattr(swarm, "metadata", None) or {}).get(_MARK_PORTAL_SPEC)
    mesh_trigger = _swarm_trigger(swarm)

    members: list[Any] = []
    for idx, agent in enumerate(agents):
        peers = [dst.name for (src, dst) in swarm.relationships if src is agent]
        member: Node | Construct
        member_trigger: Literal["output", "tool"]
        if type(agent).__name__ == "Flow":
            # C1 import: a Flow Swarm member (a neograph Construct exported via
            # _lower_portal_mesh_to_swarm, or any foreign sub-Flow agent)
            # reconstructs to a Construct mesh member, reusing the SAME
            # FlowNode->Construct recursion the bare-FlowNode item path uses.
            # Its boundary I/O is forced to the synthesized uniform payload (by
            # identity) so _check_portal_mesh's uniform-payload rule holds. A
            # Construct member cannot be trigger="tool" (s7zt3.14 validation:
            # tool-trigger requires an agent/act member with a ReAct turn), so it
            # always routes via typed output regardless of the Swarm's HandoffMode.
            member = _flow_member_to_construct(agent, payload)
            member_trigger = "output"
        else:
            # The reserved mesh-channel input key is the literal "handoff" (design
            # §3.3, mirrored in example 28's declarative form and _ir_normalize's
            # sole-writer check); normalize_ir derives handoff_param/handoff_channel.
            member = _node_from_spec_agent(agent.name, agent, None, {"handoff": payload}, payload)
            # Option F (neograph-s7zt3.1): a neograph-exported mesh Agent carries
            # the untranslated ${var} prompt in a neograph/prompt_spec marker (its
            # system_prompt is the translated {{ flat }} wire form) -- prefer it so
            # the round trip recovers the original grammar. Foreign Swarms have no
            # marker and keep the system_prompt as-is.
            prompt_marker = (getattr(agent, "metadata", None) or {}).get(_MARK_PROMPT_SPEC)
            if prompt_marker is not None:
                member = member.model_copy(update={"prompt": prompt_marker["original_text"] or None})
            member_trigger = mesh_trigger
        if idx == 0 and portal_spec is not None:
            portal = Portal(
                to=peers,
                trigger=member_trigger,
                max_hops=portal_spec["max_hops"],
                on_exhaust=portal_spec["on_exhaust"],
                route=portal_spec["route"],
            )
        else:
            portal = Portal(to=peers, trigger=member_trigger)
        members.append(member | portal)

    # Trigger-aware best-effort warning (ACTION ITEM, retargeted now that
    # s7zt3.14's Portal(trigger="tool") exists). pyagentspec Swarms route via
    # handoff/send_message TOOLS called mid-conversation (HandoffMode
    # NEVER/OPTIONAL/ALWAYS). The precise remaining mismatch depends on the mode:
    #   - OPTIONAL/ALWAYS -> Portal(trigger="tool"): neograph NOW represents the
    #     mid-loop optional handoff faithfully (a synthesized transfer_to_<peer>
    #     tool), so the old "OPTIONAL collapses into a must-always-emit-a-goto
    #     contract" caveat NO LONGER applies. The residual downgrade is only that
    #     members are name-bound live-LLM agents needing an LLM factory at compile.
    #   - NEVER -> Portal(trigger="output"): the member routes via a typed 'goto'
    #     final output, so it IS forced to always decide a goto as structured
    #     output (the genuine tool-call-vs-typed-output mismatch that remains).
    if mesh_trigger == "tool":
        detail = (
            "its OPTIONAL/ALWAYS handoff maps to Portal(trigger='tool') (a synthesized "
            "transfer_to_<peer> tool per relationship), so the optional mid-loop handoff is "
            "represented faithfully; the residual downgrade is only that members are name-bound "
            "live-LLM agents requiring an LLM factory at compile"
        )
    else:
        detail = (
            "pyagentspec Swarms route via handoff/send_message TOOLS called mid-conversation, but a "
            "HandoffMode.NEVER member maps to Portal(trigger='output') typed routing — so it is "
            "forced to ALWAYS decide a 'goto' as structured final output (a tool-call-vs-typed-output "
            "mismatch), and the payload is a route-only synthesis (a 'goto' field plus any folded "
            "Swarm/agent data properties)"
        )
    warnings.warn(
        f"Swarm {swarm.name!r} imported onto a native Portal mesh (best-effort): {detail}. This is a "
        "structural downgrade of the Swarm's own runtime, not a lossless import.",
        stacklevel=2,
    )
    return Construct(name=swarm.name, nodes=members)


def _reconstruct_swarm_mesh_with_operator_gates(flow: Any) -> Construct | None:
    """Recognize the Phase 6 mesh-exit pause composite
    (``AgentNode(Swarm)`` -> ``BranchingNode['portal_operator']`` ->
    ``InputMessageNode``) and reconstruct the underlying Portal mesh with
    per-member Operator gates re-attached (neograph-s7zt3.2, inverse of
    ``_agent_spec._lower_portal_mesh_to_swarm``'s gated arm).

    Returns ``None`` if the structure does not match -- the caller falls back to
    treating ``flow`` as a plain (possibly foreign) Flow, never trusting the
    marker blindly (same "confirm the structure, don't trust the marker"
    discipline ``_group_flow_items`` applies to Loop/Operator lookahead).
    """
    agent_nodes = [n for n in flow.nodes if type(n).__name__ == "AgentNode"]
    if len(agent_nodes) != 1 or type(agent_nodes[0].agent).__name__ != "Swarm":
        return None
    agent_node = agent_nodes[0]
    swarm = agent_node.agent

    check = next(
        (n for n in flow.nodes if (n.metadata or {}).get(_MARK_MODIFIER) == "portal_operator"),
        None,
    )
    if check is None or _MARK_PORTAL_OPERATOR_SPEC not in (check.metadata or {}):
        return None

    # Structural confirmation, not marker trust: the AgentNode really leads into
    # this specific check via a real ControlFlowEdge.
    edge_ok = any(
        e.from_node.name == agent_node.name and e.to_node.name == check.name
        for e in flow.control_flow_connections
    )
    if not edge_ok:
        return None

    base = _reconstruct_swarm_mesh(swarm)  # existing helper, unchanged
    gated: dict[str, str] = check.metadata[_MARK_PORTAL_OPERATOR_SPEC]
    # Operator.when is always a plain str (never a callable, unlike Loop.when),
    # so the marker's string passes straight through -- exactly as
    # _reconstruct_operator_item does for the single-node case, no parse_condition.
    # _reconstruct_swarm_mesh only ever yields Node members (Agent -> Node |
    # Portal), but base.nodes is typed list[ConstructItem]; the isinstance guard
    # narrows to Node for the `| Operator` compose (a _BranchNode has no `|`).
    updated_nodes = [
        (member | Operator(when=gated[member.name]))
        if isinstance(member, Node) and member.name in gated
        else member
        for member in base.nodes
    ]
    return base.model_copy(update={"nodes": updated_nodes})


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

    Agent/act (``AgentNode``) reconstruction is implemented per neograph-aa5gq:
    a ``neograph/agent_spec`` marker reconstructs the exact agent/act Node
    losslessly; a foreign ``Agent`` imports best-effort as agent-mode; a
    client-initiated ``RemoteAgent``/``A2AAgent``/``OciAgent`` imports
    best-effort as a name-bound scripted Node WITH a warning. A top-level
    ``Swarm`` imports onto a native Portal mesh (also best-effort, with a
    warning). Orchestrator-side surfaces (e.g. a ServerTool-as-agent) still
    fail loud rather than silently downgrade.
    """
    _import_agent_spec_import_classes()

    # A Swarm is a top-level AgenticComponent (NOT a Flow node), so it never
    # reaches the Flow.nodes walk below -- dispatch it onto a Portal mesh here.
    if type(flow).__name__ == "Swarm":
        return _reconstruct_swarm_mesh(flow)

    # A PORTAL_OPERATOR mesh export is a Flow wrapping a Swarm (the mesh-exit
    # pause composite, neograph-s7zt3.2), not a bare Swarm -- recognize that
    # whole-Flow shape here, before the per-item _group_flow_items walk (a
    # top-level dispatch decision like the bare-Swarm check above, not a
    # per-item concern). Returns None if the structure does not match, so a
    # foreign Flow falls through to the generic walk.
    gated_mesh = _reconstruct_swarm_mesh_with_operator_gates(flow)
    if gated_mesh is not None:
        return gated_mesh

    output_types: dict[str, Any] = {}
    pipeline_items: list[Any] = []

    # RECOGNIZE -> CLASSIFY -> dispatch on the DECOMPOSED shape: the exact mirror
    # of _agent_spec._lower_construct_item's export dispatch. The walk recognizes
    # which modifier NAMES a node grouping encodes; combo_for_modifier_names maps
    # that to a ModifierCombo via the single _COMBO_MAP; COMBO_DECOMPOSITION says
    # what the combo means. No local re-derivation of combo semantics lives here.
    for names, payload in _group_flow_items(flow):
        combo = combo_for_modifier_names(names, context=flow.name)
        decomp = COMBO_DECOMPOSITION[combo]

        item: Any
        # The fusion split runs BEFORE the shape match, exactly as on the export
        # side, and asks the ONE shared presence predicate rather than open-coding
        # it -- see neograph-c265k. ``names`` is the recognized modifier-NAME set (the
        # loader recognizes structure, so it has no Modifier instances to hand
        # over); ``dict.fromkeys(names, True)`` presents it in the {name: <present>}
        # mapping shape ``classify_modifiers`` returns. The sentinel must be
        # non-None: the predicate tests ``mods.get(k) is not None``, so a
        # None-valued key would read as ABSENT and silently un-fuse the import.
        if is_each_oracle_fused(dict.fromkeys(names, True)):
            # Fused Each x Oracle -- the MapNode's sub-flow IS an Oracle group.
            item = _reconstruct_fused_each_oracle_node(payload["map_node"], output_types)
        else:
            match decomp.primary:
                case PrimaryShape.ORACLE:
                    reconstructed = _reconstruct_oracle_group(payload["group"], flow, output_types)
                    if reconstructed is None:
                        # Stale marker -- fall back to importing every node in the
                        # group as a bare primitive (per the Core Invariant: never
                        # silently reconstruct a modifier that diverges from the
                        # actual structure). A trailing Operator is meaningless
                        # against a group that did not reconstruct, so skip it too.
                        for spec_node in payload["group"]:
                            pipeline_items.append(_reconstruct_primitive_node(spec_node, flow, output_types))
                        continue
                    item = reconstructed

                case PrimaryShape.EACH:
                    item = _reconstruct_each_node(payload["map_node"], flow, output_types)

                case PrimaryShape.LOOP:
                    item = _reconstruct_loop_item(payload["body"], payload["check"], flow, output_types)

                case PrimaryShape.BARE:
                    spec_node = payload["primary"]
                    if type(spec_node).__name__ == "FlowNode":
                        sub = from_agent_spec(spec_node.subflow)
                        sub = sub.model_copy(update={"name": spec_node.name})
                        output_types[spec_node.name] = sub.output
                        item = sub
                    elif decomp.has_operator:
                        # External inputs land on the PRIMARY node, not the
                        # property-less check BranchingNode -- mirror
                        # to_agent_spec's input_targets routing for BARE+Operator.
                        item = _reconstruct_operator_primary(spec_node, flow, output_types)
                    else:
                        item = _reconstruct_primitive_node(spec_node, flow, output_types)

                case PrimaryShape.PORTAL:  # pragma: no cover - a Portal mesh never reaches this walk
                    raise ConfigurationError.build(
                        f"Flow {flow.name!r} encodes a Portal at item level — no primitive import",
                        expected="a peer-mode Portal mesh, imported from a Swarm before this walk",
                        found=combo.name,
                    )

        # ONE shared Operator postlude, mirroring the export side's.
        if decomp.has_operator:
            item = item | Operator(when=payload["operator_check"].metadata[_MARK_OPERATOR_SPEC]["when"])

        pipeline_items.append(item)

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
