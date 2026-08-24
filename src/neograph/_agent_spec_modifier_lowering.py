"""Agent Spec EXPORT: the per-MODIFIER lowerings — Oracle, Each, Loop, Operator.

Extracted from ``_agent_spec.py`` (neograph-s7zt3.15), the symmetric sibling of
``_agent_spec_node_lowering.py``: that module owns "what does ONE node lower
to", this one owns "what does ONE MODIFIER wrap it in". What stays behind in
``_agent_spec.py`` is the item DISPATCH (``_lower_construct_item``) and the Flow
ASSEMBLY (``to_agent_spec``) — the two things that need to see every modifier at
once.

``to_agent_spec`` is INJECTED as ``export_flow`` rather than imported, the same
one-way-layering device ``_agent_spec_portal.py`` already uses on this side and
``_agent_spec_group_import.py``'s ``from_spec`` uses on the import side:
``_agent_spec`` imports this module, so this module must not import it back.

Layer order, verified from the AST: ``_agent_spec_markers`` /
``_agent_spec_placeholders`` / ``_agent_spec_node_lowering`` <- this module <-
``_agent_spec``. No back-edges.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from neograph._agent_spec_boundary import close_sub_flow
from neograph._agent_spec_each_fanout import each_receiver_properties
from neograph._agent_spec_loop_predicate import synthesize_loop_predicate
from neograph._agent_spec_lowering_types import _ExportFlow
from neograph._agent_spec_markers import (
    _MARK_BRANCH,
    _MARK_EACH_SPEC,
    _MARK_GROUP_ID,
    _MARK_LOOP_SPEC,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    _MARK_ORACLE_SPEC,
    _MARK_VARIANT,
    Branch,
    _import_agent_spec_flow_classes,
)
from neograph._agent_spec_node_lowering import (
    _lower_generation_step,
    _lower_node,
    _make_llm_config,
    _reject_unrepresentable_fields,
)
from neograph._agent_spec_placeholders import (
    _is_translation_eligible,
    _item_inputs,
    _item_outputs,
    _node_translation,
    _properties_for,
    _translate_placeholders,
    compose_property_title,
    property_title_to_prompt_path,
)
from neograph._ir_branch import _BranchNode
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import Each, Loop, Operator, Oracle
from neograph.naming import field_name_for
from neograph.node import Node

if TYPE_CHECKING:
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.node import Node as SpecNode

    from neograph._agent_spec_provider import ApiProviderResolver  # noqa: F401


def _lower_item_body(item: Node | Construct, export_flow: _ExportFlow, provider: ApiProviderResolver) -> SpecNode:
    """Lower one construct item to the SpecNode a modifier wraps.

    A ``Node`` lowers via ``_lower_node`` (per-mode think/agent/scripted
    dispatch). A ``Construct`` used as one item lowers to the SAME
    ``FlowNode`` the BARE-Construct branch emits — its ``subflow`` is the
    recursively-exported sub-``Flow`` (``export_flow``). Shared by
    ``_lower_each`` / ``_lower_loop`` / ``_lower_oracle`` and the
    LOOP/OPERATOR/BARE arms of ``_lower_construct_item`` so a Construct-item
    modifier wraps its sub-flow EXACTLY as a Node modifier wraps its lowered
    primitive — one body-lowering seam, never a per-modifier re-derivation."""
    if isinstance(item, Construct):
        nodes_mod, _flow_mod, _edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
        return nodes_mod.FlowNode(name=item.name, subflow=export_flow(item))
    return _lower_node(item, provider)


def _lower_oracle(
    node: Node | Construct, oracle: Oracle, export_flow: _ExportFlow, provider: ApiProviderResolver
) -> tuple[list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge]]:
    """Lower an Oracle-modified item: N variant bodies + merge node.

    Oracle is the flagship irreversible gap — no single Agent Spec node
    represents it. Lowers to a ``ParallelFlowNode`` of N single-node flows
    (one ``LlmConfig`` per ``Oracle.models`` entry, or N copies) + a merge
    node, stamped with the full ``neograph/modifier=oracle`` marker (incl.
    ``models``, which has no primitive representation).
    """
    nodes_mod, flow_mod, edges_mod, _property_mod, tools_mod = _import_agent_spec_flow_classes()

    # B2: the per-variant loop lowers ``node`` N times but never guarded its
    # unrepresentable fields — unlike ``_lower_node``, which calls this before
    # lowering. Without it an Oracle-modified node carrying a raw_fn / custom
    # renderer / skip_when (etc.) exported silently. Guard once, up front. A
    # Construct item has none of those callable-valued Node fields (its own
    # child nodes get the same guard during the recursive ``to_agent_spec``
    # that ``_lower_item_body`` runs), so the guard is Node-only.
    if isinstance(node, Node):
        _reject_unrepresentable_fields(node)

    if oracle.merge_pre_process or oracle.merge_post_process or oracle.merge_fallback:
        raise ConfigurationError.build(
            f"node {node.name!r}'s Oracle uses merge_pre_process/merge_post_process/merge_fallback "
            "— Python callables with no Agent Spec representation",
            expected="Oracle without merge hooks",
            found="one or more merge hooks set",
            hint="Oracle merge hooks cannot be exported to Agent Spec (callable-valued field, doc s6)",
        )

    group_id = f"{node.name}__oracle"
    # A Construct item declares no ``.model`` (per-variant model swap rides the
    # oracle_spec marker + runtime _inject_oracle_config, not the export body);
    # ``.oracle_gen_type`` is likewise a Node-only IR field. Read both defensively
    # so the shared lowering covers Node and Construct alike.
    node_model = getattr(node, "model", None)
    variant_models = oracle.models if oracle.models else [node_model] * oracle.n
    oracle_gen_type = getattr(node, "oracle_gen_type", None)
    gen_outputs = _properties_for(oracle_gen_type) if oracle_gen_type else _properties_for(_item_outputs(node))

    variant_nodes: list[SpecNode] = []
    for i, model_tier in enumerate(variant_models):
        variant_name = f"{node.name}__variant_{i}"
        variant_metadata = {_MARK_MODIFIER: "oracle", _MARK_GROUP_ID: group_id, _MARK_VARIANT: i}

        if isinstance(node, Construct):
            # A Construct variant is a copy of the sub-flow run N times (the runtime
            # shape make_oracle_redirect_fn produces over the subgraph); per-variant
            # Oracle.models rides the oracle_spec marker, not a FlowNode field.
            # neograph-15rpw: built through the shared body seam instead of a second
            # inline FlowNode, and called INSIDE the loop -- model_copy is shallow, so
            # a hoisted body would alias ONE sub-Flow across all N variants.
            body = _lower_item_body(node, export_flow, provider)
            variant_nodes.append(body.model_copy(update={"name": variant_name, "metadata": variant_metadata}))
            continue

        # Unified per-node.mode dispatch neograph-2s2o6: each Oracle variant
        # lowers through the SAME _lower_generation_step _lower_node uses -- one
        # dispatch, not two. The variant carries the oracle group/variant markers
        # (base metadata) plus its per-variant Oracle.models tier; think/agent-act/
        # scripted are all handled identically to the top-level node, so the merge
        # node + variant->merge edges below stay mode-agnostic. (An unconditional
        # LlmNode was the root cause of the scripted-mode Oracle export bug --
        # neograph-m57mn; the shared dispatch prevents that class of drift.)
        variant_nodes.append(
            _lower_generation_step(
                node,
                name=variant_name,
                outputs=gen_outputs,
                metadata=variant_metadata,
                model_tier=model_tier,
                tool_description=f"Oracle variant {i} for {node.name!r}",
                provider=provider,
            )
        )

    outputs = _properties_for(_item_outputs(node))
    # Option F neograph-cbpyx: the merge LlmNode's prompt references the variant
    # outputs via ${...}; translate to {{ flat }} and route the variant->merge
    # fan-in DataFlowEdges through the SAME flat map. merge_orig_to_flat stays empty
    # (no translation) for the merge_fn ToolNode branch, so its fan-in edges keep the
    # raw gen_output titles.
    merge_orig_to_flat: dict[str, str] = {}
    if oracle.merge_prompt:
        # Gated on oracle.merge_prompt truthiness, NOT node.mode -- a
        # scripted-mode node can legally carry merge_prompt=... (neograph-
        # m57mn addendum, translated at the 4th Option-F site).
        merge_rewritten, merge_ref_props, merge_flat_to_orig = _translate_placeholders(
            oracle.merge_prompt, gen_outputs, node.name
        )
        merge_orig_to_flat = {path: flat for flat, path in merge_flat_to_orig.items()}
        merge_node = nodes_mod.LlmNode(
            name=f"{node.name}",
            inputs=merge_ref_props or None,
            outputs=outputs or None,
            llm_config=_make_llm_config(Node(name=node.name, model=oracle.merge_model), provider),
            prompt_template=merge_rewritten,
            metadata={
                _MARK_MODIFIER: "oracle",
                _MARK_GROUP_ID: group_id,
                _MARK_ORACLE_SPEC: {
                    "n": oracle.n,
                    "models": oracle.models,
                    "merge_prompt": oracle.merge_prompt,
                    "merge_model": oracle.merge_model,
                },
            },
        )
    else:
        merge_node = nodes_mod.ToolNode(
            name=f"{node.name}",
            inputs=gen_outputs or None,
            outputs=outputs or None,
            tool=tools_mod.ServerTool(
                name=oracle.merge_fn or f"{node.name}_merge",
                description=f"Oracle merge for {node.name!r}",
                inputs=gen_outputs or None,
                outputs=outputs or None,
            ),
            metadata={
                _MARK_MODIFIER: "oracle",
                _MARK_GROUP_ID: group_id,
                _MARK_ORACLE_SPEC: {
                    "n": oracle.n,
                    "models": oracle.models,
                    "merge_fn": oracle.merge_fn,
                },
            },
        )

    control_edges: list[ControlFlowEdge] = []
    data_edges: list[DataFlowEdge] = []
    for i, variant in enumerate(variant_nodes):
        # neograph-s7zt3.15: the variants run as a CHAIN -- variant_0 -> ... ->
        # variant_{n-1} -> merge -- so every variant has an inbound control edge.
        # The pre-fix shape pointed every variant AT the merge and gave the group's
        # single inbound edge to the merge itself, so no variant was reachable at
        # all and a literal edge-walk merged outputs nothing had produced.
        #
        # A chain, not a parallel fan-out, because Agent Spec's only parallel
        # primitive (ParallelFlowNode) forbids two subflows exposing the same
        # output title, and N Oracle variants expose IDENTICAL titles by
        # construction (pyagentspec 26.1.2 _validate_outputs_with_same_name_do_not_exist).
        # The variants are mutually independent, so sequential IS a valid
        # topological order and the merge sees the same N inputs; the fan-out-of-n
        # intent that re-import needs rides the oracle marker.
        control_edges.append(
            edges_mod.ControlFlowEdge(
                name=f"{group_id}_variant_run_{i}",
                from_node=variant,
                to_node=variant_nodes[i + 1] if i + 1 < len(variant_nodes) else merge_node,
            )
        )
        for prop in gen_outputs:
            # When the merge node is a translated LlmNode (merge_prompt), its
            # declared input is the flat placeholder name; route the fan-in edge
            # through the SAME flat map and drop it if the merge prompt never
            # referenced this variant output (unreferenced -> no data path).
            if oracle.merge_prompt:
                dest_input = merge_orig_to_flat.get(property_title_to_prompt_path(prop.title))
                if dest_input is None:
                    continue
            else:
                dest_input = prop.title
            data_edges.append(
                edges_mod.DataFlowEdge(
                    name=f"{group_id}_fanin_{i}_{prop.title}",
                    source_node=variant,
                    source_output=prop.title,
                    destination_node=merge_node,
                    destination_input=dest_input,
                )
            )

    return [*variant_nodes, merge_node], control_edges, data_edges


def _lower_each(
    node: Node | Construct,
    each: Each,
    export_flow: _ExportFlow,
    provider: ApiProviderResolver,
    oracle: Oracle | None = None,
) -> SpecNode:
    """Lower an Each-modified item: MapNode wrapping a single-body sub-Flow.

    The wrapped body is a Node's lowered primitive OR a Construct-item's
    ``FlowNode`` (``_lower_item_body``), so ``Construct(...) | Each(...)`` used
    as one item lowers to the SAME MapNode shape as ``node | Each(...)``.
    ``over``/``key``/``on_error`` have no primitive representation — ride in
    the ``neograph/modifier=each`` marker (``EachSpec``).

    ``oracle`` is the Each x Oracle FUSION seam (neograph-s7zt3.10), and mirrors
    ``_lower_loop(node, loop, body)``'s caller-lowered-body seam: when set, the
    sub-Flow's body is the variant-fan-out + merge group ``_lower_oracle``
    already produces, instead of a single primitive. Composition, not a second
    lowering — the fused MapNode adds ZERO new node-construction sites.
    """
    nodes_mod, flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()

    # The MapNode infers its OWN inputs as ``iterated_{title}`` for every
    # property in ``subflow.inputs`` (pyagentspec MapNode._get_inferred_inputs,
    # which reads the sub-flow's StartNode inputs). Declare the inner node's
    # input Properties on the StartNode so a NON-fan-out context input (e.g.
    # ``verify(source: RawText, cluster: Elem)`` with ``map_over``) has a valid
    # ``iterated_source.text`` destination for its top-level DataFlowEdge — the
    # fan-out-receiver-only case stays valid too (its inferred input is simply
    # left unconnected, populated per-item from the iterated collection).
    # neograph-hf505.
    #
    # Option F consumer sweep (neograph-cbpyx, MEDIUM-1): the StartNode is a
    # NON-DataFlowEdge consumer of _properties_for(node.inputs). When the inner
    # node is placeholder-translated (LLM mode), its declared inputs are the flat
    # ${var}->{{ flat }} names, so the StartNode MUST use the SAME flat titles or
    # the sub-flow ships an unfillable ``{{ item_v }}`` (the inner's inferred input
    # and the StartNode's declared input would not match). Scripted inners keep the
    # untranslated dotted Properties.
    if _is_translation_eligible(node):
        _rewritten, inner_inputs, _flat = _node_translation(node)
    else:
        inner_inputs = each_receiver_properties(
            _item_inputs(node), getattr(node, "fan_out_param", None), _properties_for
        )
    start_node = nodes_mod.StartNode(name=f"{node.name}__each_start", inputs=inner_inputs or None)

    if oracle is None:
        inner = _lower_item_body(node, export_flow, provider)
        body_nodes: list[SpecNode] = [inner]
        body_control = [
            edges_mod.ControlFlowEdge(name=f"{node.name}__each_start_edge", from_node=start_node, to_node=inner),
        ]
        body_data: list[DataFlowEdge] = []
    else:
        # FUSED Each x Oracle. The Oracle group's own variant-chain control edges
        # and fan-in data edges come through verbatim; only the Flow's boundary
        # wiring is added here. A pyagentspec StartNode must have EXACTLY ONE
        # outgoing control edge, and _lower_oracle now chains the variants
        # (neograph-s7zt3.15), so it spends that edge on the FIRST VARIANT — the
        # head of the chain, exactly as to_agent_spec enters an un-fused ORACLE
        # item — and the chain's tail, the merge, exits to the EndNode. Aiming it
        # at the merge instead (the pre-fix shape) orphaned every variant. The
        # inner merge keeps the MapNode's own name, which validates and round-trips.
        body_nodes, body_control, body_data = _lower_oracle(node, oracle, export_flow, provider)
        body_control = [
            edges_mod.ControlFlowEdge(
                name=f"{node.name}__each_start_edge", from_node=start_node, to_node=body_nodes[0]
            ),
            *body_control,
        ]

    # neograph-qtfof.11: close_sub_flow owns the terminal boundary -- the EndNode
    # DECLARES the terminal producer's outputs and is FED by a real DataFlowEdge
    # (a MapNode infers its OWN outputs from ``subflow.outputs`` <- that EndNode,
    # so an empty one erased the entire fan-out result from the exported artifact).
    return nodes_mod.MapNode(
        name=node.name,
        subflow=close_sub_flow(
            "each", node.name, start_node, body_nodes, body_control, body_data, (nodes_mod, flow_mod, edges_mod)
        ),
        metadata={
            _MARK_MODIFIER: "each",
            _MARK_EACH_SPEC: {"over": each.over, "key": each.key, "on_error": each.on_error},
        },
    )


def _lower_loop(
    node: Node | Construct, loop: Loop, body: SpecNode
) -> tuple[SpecNode, list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge]]:
    """Lower a Loop-modified item: BranchingNode({continue: back-edge, done: next}).
    ``body`` is the caller-lowered primitive. Returns ``(branch, extra_nodes,
    control_edges, data_edges)`` -- see ``_agent_spec_loop_predicate`` for what
    ``extra_nodes`` holds; caller folds it between ``body``/``branch``.
    """
    nodes_mod, _flow_mod, edges_mod, property_mod, tools_mod = _import_agent_spec_flow_classes()
    if callable(loop.when):
        raise ConfigurationError.build(
            f"node {node.name!r}'s Loop.when is a callable — no Agent Spec representation",
            expected="a registered condition NAME (str)",
            found="Loop.when is a callable",
            hint="only registered-string conditions serialize (callable-valued field, doc s6)",
        )
    branch = nodes_mod.BranchingNode(
        name=f"{node.name}__loop_check",
        mapping={Branch.CONTINUE: Branch.CONTINUE, Branch.DONE: Branch.DONE},
        metadata={
            _MARK_MODIFIER: "loop",
            _MARK_LOOP_SPEC: {"when": loop.when, "max_iterations": loop.max_iterations, "on_exhaust": loop.on_exhaust},
        },
    )
    # neograph-qtfof.6: see _agent_spec_loop_predicate's module docstring.
    extra_nodes, predicate_control, predicate_data, check_entry = synthesize_loop_predicate(
        node, loop, body, branch, (nodes_mod, edges_mod, property_mod, tools_mod)
    )
    check_tag = "predicate" if extra_nodes else "loop_body"
    control_edges = [
        *predicate_control,
        edges_mod.ControlFlowEdge(name=f"{node.name}__{check_tag}_to_check", from_node=check_entry, to_node=branch),
        edges_mod.ControlFlowEdge(
            name=f"{node.name}__loop_back", from_node=branch, from_branch=Branch.CONTINUE, to_node=body
        ),
    ]
    # Dict-form inputs qualify each Property title with its upstream key (per
    # _properties_for's dict-form convention) -- the body node's real input
    # Property is the qualified title, never the bare "{field}", so the
    # self-edge's destination_input must be resolved against the SAME key the
    # runtime feeds the re-entry value into. That key is whichever dict-form
    # inputs entry has a type compatible with the node's own output type
    # (mirrors the single-type upstream-resolution scan below: a Loop-fed key
    # could be a self-reference — "key matching the node's own name" per the
    # validator's Loop rule — OR the ORIGINAL upstream producer's name, e.g.
    # inputs={'seed': Draft} — either way it's the key whose declared type
    # matches the fed-back output).
    ni = normalize_inputs(_item_inputs(node))
    no_self = normalize_outputs(_item_outputs(node))
    dest_key: str | None = None
    if ni.is_dict_form and not no_self.is_dict_form:
        self_field = field_name_for(node.name)
        if self_field in ni.by_name:
            dest_key = self_field
        else:
            for key, typ in ni.by_name.items():
                if isinstance(typ, type) and (issubclass(no_self.primary, typ) or issubclass(typ, no_self.primary)):
                    dest_key = key
                    break

    # Option F consumer sweep neograph-cbpyx: when the loop body is a
    # placeholder-translated LLM node, its declared inputs are flat ${var}->{{ flat }}
    # names, so the self-feedback edge's destination_input must route through the
    # body's flat map -- keyed by the dotted ${...} PROMPT path, NOT by a Property
    # title (drop it if the fed-back output isn't referenced in the prompt).
    body_orig_to_flat = _node_translation(node)[2] if _is_translation_eligible(node) else {}

    # A DATA edge only where the fed-back output has a castable destination Property on
    # the body's OWN input port. LangGraph's loop-back (_wiring._add_subgraph_loop) is a
    # pure CONTROL edge over a shared accumulating state dict -- input and output occupy
    # separate slots -- so a body with differing boundary types (a Construct item with
    # input != output) maps ZERO fields and loops on control alone. Decided per field via
    # pyagentspec's OWN property_is_castable_to, never by suppressing it -- neograph-rh5fb.
    body_inputs = {p.title: p for p in (body.inputs or [])}
    body_outputs = {p.title: p for p in (body.outputs or [])}
    data_edges: list[DataFlowEdge] = list(predicate_data)
    for prop in _properties_for(_item_outputs(node)):
        if _is_translation_eligible(node):
            dest_input = body_orig_to_flat.get(f"{dest_key}.{prop.title}" if dest_key else prop.title)
            if dest_input is None:
                continue
        else:
            dest_input = compose_property_title(dest_key, prop.title) if dest_key else prop.title
        source_prop, dest_prop = body_outputs.get(prop.title), body_inputs.get(dest_input)
        if source_prop is None or dest_prop is None:
            continue  # no destination property for this field -> control-only loop-back
        if not property_mod.property_is_castable_to(source_prop, dest_prop):
            continue  # same name, incompatible type -> not a structurally identical field
        data_edges.append(
            edges_mod.DataFlowEdge(
                name=f"{node.name}__loop_self_{prop.title}",
                source_node=body,
                source_output=prop.title,
                destination_node=body,
                destination_input=dest_input,
            )
        )
    return branch, extra_nodes, control_edges, data_edges


def _lower_branch(
    branch_node: _BranchNode,
    lower_item: Callable[[Any], Any],
) -> tuple[
    list[SpecNode],
    list[ControlFlowEdge],
    list[DataFlowEdge],
    SpecNode,
    list[tuple[SpecNode, str | None]],
    dict[str, Any],
    dict[str, list[tuple[SpecNode, bool]]],
    dict[str, SpecNode],
]:
    """Lower a ``_BranchNode`` (a ForwardConstruct ``if``/``else``) into a real
    ``BranchingNode`` with divergent true/false ``ControlFlowEdge``s into each
    arm, reconverging on the successor — neograph-s7zt3.17.

    Pre-fix, ``to_agent_spec`` walked ``iter_with_arms``, which DROPS the
    ``_BranchNode`` sentinel and flattens ``true_arm_nodes`` then
    ``false_arm_nodes`` in place — so the exporter never saw a branch at all,
    and both arms were wired to run unconditionally in sequence.

    ``lower_item`` is ``_lower_construct_item``, INJECTED (mirrors
    ``export_flow`` on ``_lower_item_body`` — this module must not import
    ``_agent_spec`` back) so each arm's own items dispatch through the SAME
    modifier machinery the top-level loop uses: a Construct sub-pipeline or an
    Oracle/Each/Loop/Operator-wrapped node inside an arm gets its existing
    lowering for free, chained sequentially within the arm via the identical
    adjacent-pair reconvergence ``to_agent_spec`` already uses between
    top-level items. A nested ``_BranchNode`` inside an arm is not reachable —
    the forward tracer never nests branches (``_merge_sequential_branches``
    handles only sequential, non-nested branches) — so ``lower_item`` raising
    on one is a correct fail-loud boundary, not a gap.

    Returns ``(all_nodes, control_edges, data_edges, entry, exits,
    item_by_name, input_targets_by_item_name, data_node_by_item_name)`` — the
    last three are per-ARM-ITEM bookkeeping the caller merges into its own
    dicts, since ``to_agent_spec``'s data-edge resolution phase
    (``_emit_input_edges``) must still find an arm-internal node by name —
    exactly the visibility ``iter_with_arms`` used to give for free by
    flattening the arm into the top-level item list.
    """
    nodes_mod, _flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
    meta = branch_node._neo_branch_meta

    branch = nodes_mod.BranchingNode(
        name=branch_node.name,
        mapping={Branch.TRUE: Branch.TRUE, Branch.FALSE: Branch.FALSE},
        metadata={_MARK_BRANCH: True},
    )

    all_nodes: list[SpecNode] = [branch]
    control_edges: list[ControlFlowEdge] = []
    data_edges: list[DataFlowEdge] = []
    exits: list[tuple[SpecNode, str | None]] = []
    item_by_name: dict[str, Any] = {}
    input_targets_by_item_name: dict[str, list[tuple[SpecNode, bool]]] = {}
    data_node_by_item_name: dict[str, SpecNode] = {}

    for arm_items, arm_label in ((meta.true_arm_nodes, Branch.TRUE), (meta.false_arm_nodes, Branch.FALSE)):
        if not arm_items:
            # An empty arm has no body -- the branch itself is the exit for
            # that side, mirroring _wiring_branch.py's true_target/
            # false_target = END fallback for an empty arm.
            exits.append((branch, arm_label))
            continue

        arm_entries: list[SpecNode] = []
        arm_exits: list[list[tuple[SpecNode, str | None]]] = []
        for item in arm_items:
            item_by_name[item.name] = item
            lowered_nodes, extra_control, extra_data, entry, item_exits, data_node, input_targets = lower_item(item)
            all_nodes.extend(lowered_nodes)
            control_edges.extend(extra_control)
            data_edges.extend(extra_data)
            arm_entries.append(entry)
            arm_exits.append(item_exits)
            data_node_by_item_name[item.name] = data_node
            input_targets_by_item_name[item.name] = input_targets

        control_edges.append(
            edges_mod.ControlFlowEdge(
                name=f"{branch.name}_{arm_label}",
                from_node=branch,
                from_branch=arm_label,
                to_node=arm_entries[0],
            )
        )
        # Same adjacent-pair reconvergence to_agent_spec uses between
        # top-level items, scoped to this arm's own item list.
        for prev_exits, next_entry in zip(arm_exits, arm_entries[1:], strict=False):
            for from_node, from_branch in prev_exits:
                control_edges.append(
                    edges_mod.ControlFlowEdge(
                        name=f"{from_node.name}_to_{next_entry.name}",
                        from_node=from_node,
                        from_branch=from_branch,
                        to_node=next_entry,
                    )
                )
        exits.extend(arm_exits[-1])

    return (
        all_nodes,
        control_edges,
        data_edges,
        branch,
        exits,
        item_by_name,
        input_targets_by_item_name,
        data_node_by_item_name,
    )


def _lower_top_level_item(
    item: Any,
    lower_item: Callable[[Any], Any],
) -> tuple[
    list[SpecNode],
    list[ControlFlowEdge],
    list[DataFlowEdge],
    SpecNode,
    list[tuple[SpecNode, str | None]],
    dict[str, Any],
    dict[str, list[tuple[SpecNode, bool]]],
    dict[str, SpecNode],
]:
    """Dispatch one top-level ``construct.nodes`` item, incl. ``_BranchNode``,
    to a UNIFORM per-item-bookkeeping shape ``to_agent_spec`` merges into its
    own dicts — keeps the ``isinstance(item, _BranchNode)`` branch out of the
    main loop, since a plain item and a branch's several arm-items need
    different-shaped bookkeeping (one name vs several) unified here instead.
    """
    if isinstance(item, _BranchNode):
        lowered_nodes, extra_control, extra_data, entry, item_exits, item_by_name, targets_by_name, data_by_name = (
            _lower_branch(item, lower_item)
        )
        return (
            lowered_nodes,
            extra_control,
            extra_data,
            entry,
            item_exits,
            item_by_name,
            targets_by_name,
            {**data_by_name, item.name: entry},
        )
    lowered_nodes, extra_control, extra_data, entry, item_exits, data_node, input_targets = lower_item(item)
    return (
        lowered_nodes,
        extra_control,
        extra_data,
        entry,
        item_exits,
        {item.name: item},
        {item.name: input_targets},
        {item.name: data_node},
    )


def _lower_operator(node: Node | Construct, operator: Operator) -> tuple[SpecNode, SpecNode, list[ControlFlowEdge]]:
    """Lower an Operator-modified item: the FULLY PINNED HITL-pause composite
    (neograph-03djs, verified against real pyagentspec 26.1.2 source).

    Reads only ``item.name`` + ``operator.when`` (the caller lowers the primary
    body separately), so it applies uniformly to a Node or a Construct item.

    ``BranchingNode(mapping={<condition-string>: PAUSE_BRANCH})`` +
    ``ControlFlowEdge(from_branch=PAUSE_BRANCH) -> InputMessageNode``. The
    boolean-to-string-key coercion is REQUIRED: the condition's truthy
    result must render to the literal mapping-key string, or the composite
    silently always takes DEFAULT_BRANCH (never pauses).

    Returns the ``(check, pause)`` pair, not an opaque node list, because BOTH
    are exits of the composite — the caller reconverges the DEFAULT_BRANCH and
    the post-pause continuation on the item's successor.
    """
    nodes_mod, _flow_mod, edges_mod, property_mod, _tools_mod = _import_agent_spec_flow_classes()

    check = nodes_mod.BranchingNode(
        name=f"{node.name}__operator_check",
        mapping={Branch.TRUE: Branch.PAUSE, Branch.FALSE: Branch.DEFAULT},
        metadata={_MARK_MODIFIER: "operator", _MARK_OPERATOR_SPEC: {"when": operator.when}},
    )
    input_message = nodes_mod.InputMessageNode(
        name=f"{node.name}__operator_pause",
        outputs=[property_mod.StringProperty(title="user_input")],
    )
    pause_edge = edges_mod.ControlFlowEdge(
        name=f"{node.name}__operator_to_pause", from_node=check, from_branch=Branch.PAUSE, to_node=input_message
    )
    return check, input_message, [pause_edge]
