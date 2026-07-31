"""Agent Spec IMPORT: modifier-group reconstruction, and the Construct seam.

Extracted from ``loader.py`` (neograph-s7zt3.11). This half rebuilds a MODIFIER
GROUP -- Each / Loop / Oracle / Operator, and their fusions -- from the several
Agent Spec nodes each one lowers to, and owns ``_construct_from_subflow``: the
single site on the import path that recurses into a sub-flow.

``from_agent_spec`` is INJECTED as a callable rather than imported. That is what
keeps the layering acyclic: ``loader`` imports this module, so this module must
not import ``loader``. It is also why the seam lives here and not in ``loader``
-- if it stayed there, every reconstructor below would call upward and the
one-way property would break.

Layer order, verified from the AST: ``_agent_spec_node_import`` <- this module
<- ``loader``. No back-edges.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from neograph._agent_spec_markers import (
    _MARK_EACH_SPEC,
    _MARK_GROUP_ID,
    _MARK_LOOP_SPEC,
    _MARK_MODIFIER,
    _MARK_ORACLE_SPEC,
    _MARK_PROMPT_SPEC,
)
from neograph._agent_spec_node_import import (
    _agent_spec_props_to_type,
    _augment_inputs_from_prompt_marker,
    _dict_form_inputs_from_props,
    _inputs_from_data_edges,
    _reconstruct_agent_node,
    _reconstruct_primitive_node,
)
from neograph._normalize import _declared_output, _with_declared_io, normalize_outputs
from neograph.conditions import parse_condition
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import Each, Loop, Oracle
from neograph.node import Node


def _construct_from_subflow(subflow: Any, name: str, from_spec: Callable[[Any], Construct]) -> Construct:
    """Rebuild a sub-``Flow`` into a ``Construct``, boundary port intact.

    THE single site that recurses through ``from_agent_spec`` on a sub-flow.
    ``from_spec`` is injected rather than imported so this can live beside the
    other reconstructors without calling back into its own parent module.

    Restoring ``input``/``output`` is the point, not a detail. Without it a
    sub-Construct declared ``input=A, output=B`` came back with BOTH set to
    ``None`` and the reimported parent then failed to compile with
    "sub-construct has no output type" -- while still passing an is-a-Construct
    plus combo-matches check, so nothing noticed. The exported sub-flow carries
    the boundary all along (``Flow.inputs``/``Flow.outputs``); it was simply
    never read back. Read via ``_agent_spec_props_to_type``, the same canonical
    Property bridge ``_reconstruct_primitive_node`` uses, so the boundary type
    is the SAME structural class a peer producer reconstructs to.

    Reads the boundary off the sub-FLOW, not off the wrapping FlowNode, because
    ``_flow_member_to_construct`` (Swarm C1 import) has only a ``Flow``.
    """
    sub = from_spec(subflow)
    restored: dict[str, Any] = {"name": name}
    port_in = _agent_spec_props_to_type(getattr(subflow, "inputs", None))
    port_out = _agent_spec_props_to_type(getattr(subflow, "outputs", None))
    if port_in is not None:
        restored["input"] = port_in
    if port_out is not None:
        restored["output"] = port_out
    return sub.model_copy(update=restored)


def _reconstruct_item_body(
    spec_node: Any, flow: Any, output_types: dict[str, Any], from_spec: Callable[[Any], Construct]
) -> Node | Construct:
    """Reconstruct ONE construct item's body, whether it is a Node or a Construct.

    The import-side inverse of ``_agent_spec._lower_item_body``, and the single
    seam every modifier reconstructor routes through -- so a Construct item is
    handled once here rather than re-derived per modifier. A ``FlowNode`` unwraps
    to its sub-flow via ``_construct_from_subflow``; anything else is a primitive.

    Registers the item's REAL boundary output into ``output_types``. The inline
    version this replaced registered ``sub.output``, which was always ``None``,
    so a downstream consumer's fan-in reconstruction came out typeless too.
    """
    if type(spec_node).__name__ == "FlowNode":
        sub = _construct_from_subflow(spec_node.subflow, spec_node.name, from_spec)
        output_types[spec_node.name] = sub.output
        return sub
    return _reconstruct_primitive_node(spec_node, flow, output_types)


def _oracle_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    """Build ``Oracle(...)`` kwargs from a ``neograph/oracle_spec`` marker.

    One builder read by BOTH the Node-variant and Construct-variant arms of
    ``_reconstruct_oracle_group`` -- the per-variant model tier lives here, so a
    Construct variant recovers it identically without the arm re-deriving it.
    """
    kwargs: dict[str, Any] = {"n": spec["n"]}
    if spec.get("models"):
        kwargs["models"] = spec["models"]
    if spec.get("merge_prompt"):
        kwargs["merge_prompt"] = spec["merge_prompt"]
        if spec.get("merge_model"):
            kwargs["merge_model"] = spec["merge_model"]
    elif spec.get("merge_fn"):
        kwargs["merge_fn"] = spec["merge_fn"]
    return kwargs


def _reconstruct_oracle_group(
    group: list[Any], flow: Any, output_types: dict[str, Any], from_spec: Callable[[Any], Construct]
) -> Node | Construct | None:
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

    base_node: Node | Construct
    if base_cls == "FlowNode":
        # A CONSTRUCT variant. _lower_oracle emits N FlowNode copies of the SAME
        # sub-flow, so the base variant IS the item and variants 1..n carry no
        # distinct content to recover -- unlike a Node variant, whose per-variant
        # model tier is a real difference. The tier still round-trips: it rides
        # the oracle marker's `models`, read into oracle_kwargs below, so nothing
        # about the ORACLE_OPERATOR representation needs to change here.
        base_node = _construct_from_subflow(base_variant.subflow, merge_node.name, from_spec)
        output_types[merge_node.name] = base_node.output
        return base_node | Oracle(**_oracle_kwargs(spec))

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
        base_node = _reconstruct_agent_node(base_variant, inputs, outputs).model_copy(update={"name": merge_node.name})
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
        base_node = Node(
            name=merge_node.name,
            mode=base_mode,
            inputs=inputs,
            outputs=outputs,
            prompt=base_prompt,
            model=base_model,
            scripted_fn=base_scripted_fn,
        )

    return base_node | Oracle(**_oracle_kwargs(spec))


def _reconstruct_each_node(
    map_node: Any, flow: Any, output_types: dict[str, Any], from_spec: Callable[[Any], Construct]
) -> Node | Construct:
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
    inner = _reconstruct_item_body(inner_spec, map_node.subflow, inner_output_types, from_spec)
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
    dict_inputs = _dict_form_inputs_from_props(inner_spec.inputs) if isinstance(inner, Node) else None
    if dict_inputs is not None:
        update["inputs"] = dict_inputs
    inner = inner.model_copy(update=update)

    # _lower_each's MapNode never sets its own outputs= (only the wrapped
    # inner node's SpecNode carries the per-item output Properties) -- the
    # per-item output type is the inner node's, not the MapNode's (unset).
    declared = _declared_output(inner)
    if not normalize_outputs(declared).is_none:
        output_types[map_node.name] = normalize_outputs(declared).primary

    return inner | Each(over=each_spec["over"], key=each_spec.get("key"))


def _reconstruct_loop_item(
    body_spec: Any, check_spec: Any, flow: Any, output_types: dict[str, Any], from_spec: Callable[[Any], Construct]
) -> Node | Construct:
    """Reconstruct a Loop-modified Node from its exported body+check pair --
    the inverse of ``_agent_spec._lower_loop``."""

    loop_spec = check_spec.metadata[_MARK_LOOP_SPEC]
    inner_output_types: dict[str, Any] = {}
    body = _reconstruct_item_body(body_spec, flow, inner_output_types, from_spec)

    outputs = _declared_output(body)
    inputs = _inputs_from_data_edges(body_spec.name, flow, output_types)
    output_types[body_spec.name] = outputs
    if inputs is not None:
        body = _with_declared_io(body, inputs=inputs, outputs=outputs)

    # A string ``when`` is EITHER an expression OR a registered condition name --
    # ``Loop.when`` is declared ``str | Callable`` and the str form is documented as
    # a registry key. Both round-trip, so discriminate instead of assuming:
    # ``parse_condition`` succeeds on an expression (yielding the callable the
    # expression path has always relied on) and raises ValueError on anything else,
    # which is then a registry NAME and is passed through for ``compile()`` to
    # resolve -- the same thing the Operator postlude does with its marker string.
    # Assuming "expression" unconditionally was neograph-ijyjr: a registry name (the
    # canonical form, the one lint's loop_condition_unregistered rule enforces) died
    # with ValueError on import and could never be reimported at all.
    when: Any = loop_spec["when"]
    if isinstance(when, str):
        try:
            when = parse_condition(when)
        except ValueError:
            pass
    return body | Loop(when=when, max_iterations=loop_spec["max_iterations"], on_exhaust=loop_spec["on_exhaust"])


def _reconstruct_operator_primary(
    primary_spec: Any, flow: Any, output_types: dict[str, Any], from_spec: Callable[[Any], Construct]
) -> Node | Construct:
    """Reconstruct the BODY node of a BARE+Operator composite, with its external
    inputs routed -- the inverse of the ``PrimaryShape.BARE`` half of
    ``_agent_spec._lower_construct_item``'s Operator path.

    Returns the node WITHOUT the ``| Operator(...)`` pipe: the caller applies
    that in the one shared postlude, so Operator composes onto every primary
    shape the same way (neograph-s7zt3.10) rather than being fused into one
    shape's reconstructor.
    """
    inner_output_types: dict[str, Any] = {}
    primary = _reconstruct_item_body(primary_spec, flow, inner_output_types, from_spec)

    inputs = _inputs_from_data_edges(primary_spec.name, flow, output_types)
    declared = _declared_output(primary)
    if not normalize_outputs(declared).is_none:
        output_types[primary_spec.name] = normalize_outputs(declared).primary
    if inputs is not None:
        primary = _with_declared_io(primary, inputs=inputs)

    return primary


def _reconstruct_fused_each_oracle_node(
    map_node: Any, output_types: dict[str, Any], from_spec: Callable[[Any], Construct]
) -> Node | Construct | None:
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
    inner = _reconstruct_oracle_group(inner_nodes, map_node.subflow, inner_output_types, from_spec)
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

    # _declared_output, not `.outputs`: _reconstruct_oracle_group can now return a
    # Construct (a Construct Oracle variant), and a Construct declares `.output`.
    # EACH_ORACLE on a Construct is rejected upstream by
    # SUB_CONSTRUCT_UNSUPPORTED_COMBOS, so this is defence in depth rather than a
    # reachable path -- but reading through the monopoly costs nothing and keeps
    # the Node/Construct split in one place.
    fused_out = _declared_output(inner)
    if not normalize_outputs(fused_out).is_none:
        output_types[map_node.name] = normalize_outputs(fused_out).primary

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
