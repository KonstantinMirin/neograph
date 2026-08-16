"""Agent Spec IMPORT: primitive node reconstruction.

Extracted from ``loader.py`` (neograph-s7zt3.11), which had reached its exact
ratchet ceiling. This half rebuilds ONE primitive Agent Spec node (LlmNode /
ToolNode / AgentNode) into a neograph ``Node``, plus the Property/data-edge/
prompt-marker helpers that only that job needs.

Import-side counterpart of ``_agent_spec_node_lowering.py``. Layering is
strictly one-way and was derived from the AST, not by eye: nothing here calls
back into ``loader``, and ``_agent_spec_group_import`` sits above this module.
"""

from __future__ import annotations

import warnings
from typing import Any

from neograph._agent_spec_markers import (
    _MARK_AGENT_SPEC,
    _MARK_PROMPT_SPEC,
    _MARK_TOOL_SPEC,
)
from neograph._agent_spec_placeholders import split_property_title
from neograph.errors import ConfigurationError
from neograph.node import Node
from neograph.spec_types import (
    _import_agent_spec_property_classes,
    _normalize_erased_property,
    _structural_type_name,
    agent_spec_properties_to_types,
    lookup_type,
)
from neograph.tool import Tool

# best-effort branch -- verified against the installed
# pyagentspec SDK, NOT a blind two-field getattr shared across families.
# RemoteAgent itself is abstract (no endpoint fields of its own) and has no
# instantiable concrete form beyond the two families below.
_REMOTE_AGENT_ENDPOINT_ATTRS: dict[str, tuple[str, ...]] = {
    "A2AAgent": ("name", "agent_url", "connection_config"),
    "OciAgent": ("name", "agent_endpoint_id", "client_config"),
}


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
    """Reconstruct a dict-form ``Node.inputs`` from key-qualified input
    Properties, the inverse of ``_agent_spec_placeholders._properties_for``'s
    ``compose_property_title(key, field)`` qualification.

    ``to_agent_spec`` flattens a dict-form ``inputs={'k': SomeModel}`` into one
    Property per field titled ``"k:field"``. A FLAT reconstruction
    (``_agent_spec_props_to_type``) would build a single model with dotted field
    names (``{'k.field': ...}``) whose structural type hash does NOT match the
    producer's — the neograph-3lk2l / qtfof.4 type-identity loss. Grouping the
    Properties back by their ``k`` prefix and reconstructing each group's model
    from the UN-prefixed field Properties restores the original per-key type, so
    the Each fan-out receiver reconstructs to the SAME structural class as the
    producer's list element. Returns ``None`` when a Property is unqualified
    (leave the caller's default single-type reconstruction in charge)."""
    if not props:
        return None
    pas = _import_agent_spec_property_classes()
    groups: dict[str, list[Any]] = {}
    for p in props:
        key, rest = split_property_title(p.title)
        if key is not None:
            groups.setdefault(key, []).append(p.model_copy(update={"title": rest}))
            continue
        # neograph-qtfof.7: a bare-titled ObjectProperty is the RESHAPED
        # fan-out receiver (one Property for the whole per-item model,
        # replacing the old dotted-per-field flattening) -- its OWN
        # `.properties` (already unprefixed, exactly like a dotted group's
        # un-prefixed members) becomes that key's group. A bare-titled
        # NON-object Property is the legacy single-type shape (no dot,
        # nothing to group) and must still fall through to the caller's
        # default reconstruction -- discriminating on "ObjectProperty",
        # not "bare title" alone, is load-bearing here.
        inflated = _normalize_erased_property(p)
        if isinstance(inflated, pas.ObjectProperty) and inflated.properties:
            groups[p.title] = list(inflated.properties.values())
            continue
        return None
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
        key = split_property_title(e["title"])[0] or e["title"]
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
                title=split_property_title(e["title"])[1],
                json_schema=e["json_schema"],
                type=e["json_schema"].get("type"),
                description=None,
                default=None,
            )
            for e in entries
            if split_property_title(e["title"])[0] == key
        ]
        result[key] = _agent_spec_props_to_type(group)
    return result or None


def _tools_from_marker(marker_tools: list[dict[str, Any]]) -> list[Tool]:
    """Rebuild neograph ``Tool`` specs from the flat ``neograph/agent_spec``
    tools list (the EXACT inverse of ``_agent_spec._agent_spec_marker``'s
    ``tools=[{name, budget, config, idempotent}, ...]`` blob)."""
    return [Tool(t["name"], budget=t["budget"], config=t["config"], idempotent=t["idempotent"]) for t in marker_tools]


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


def _node_from_spec_agent(name: str, agent: Any, marker: dict[str, Any] | None, inputs: Any, outputs: Any) -> Node:
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
    inputs = _inputs_from_data_edges(spec_node.name, flow, output_types) or _agent_spec_props_to_type(spec_node.inputs)
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
            expected="LlmNode, ToolNode, or AgentNode",
            found=cls_name,
        )

    return Node(
        name=spec_node.name,
        mode=mode,
        inputs=inputs,
        outputs=outputs,
        prompt=prompt,
        model=model,
        scripted_fn=scripted_fn,
    )
