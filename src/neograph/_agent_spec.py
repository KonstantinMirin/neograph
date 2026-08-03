"""``to_agent_spec()`` — export neograph IR (``Construct``) to an Open Agent
Spec ``Flow``.

A free function, NOT a ``Construct``/``Node`` method (CLAUDE.md layer
discipline, design doc agent-spec-interop-2026-07-09.md §7). Walks the IR via
the existing ``iter_with_arms`` (``_ir_branch.py``) — the same arm-aware walk
the compiler/runner/lint already use — and LOWERS each modifier to the flat
Agent Spec primitives it already lowers to for LangGraph compilation (Oracle
fan-out/barrier, Each router/Send/barrier, Loop back-edge, Operator's
check-node-with-interrupt), per the exporter's Core Invariant: this is the
SAME lowering neograph performs when compiling, expressed in Agent Spec
vocabulary instead of LangGraph's — never a second, divergent lowering.

Every irreversible flattening that CAN round-trip rides in
``neograph/``-prefixed ``metadata`` markers (per-group modifier markers:
``neograph/oracle_spec`` / ``each_spec`` / ``loop_spec`` / ``operator_spec``)
so the export stays BOTH a portable flat Agent Spec (markers are ignorable by
foreign runtimes) AND a neograph round-trip source for those constructs.
There is NO whole-pipeline ``Flow.metadata['neograph/source']`` fallback —
round-trip fidelity comes from the per-group markers, not a full-IR blob.
Constructs that cannot be lowered round-trip-safely FAIL LOUD via
``ConfigurationError`` rather than emit a lossy placeholder — never a silent
downgrade or truncation: ``raw_fn``, ``skip_when``/``skip_value``, a callable
``Loop.when``, Oracle merge hooks, ``renderer``, Portal
``handoff_param``/``handoff_channel``, a callable ``gate_tools_when`` (no Agent
Spec representation at all). ``agent``/``act`` mode lowers to a real
``AgentNode``+``Agent``+``ServerTool`` composite, stamped with a
``neograph/agent_spec`` marker carrying every field the plain primitives
cannot represent (mode/prompt/model/tools/gate_tools_when/context) — EXPORT
SIDE ONLY: the actual export->import round trip is deferred to
``neograph-01i0g``, which owns the ``from_agent_spec()`` importer.

Import-guarded (mirrors ``spec_types._import_agent_spec_property_classes()``)
so ``src/neograph`` core stays Agent-Spec-free by default — only calling
``to_agent_spec()`` pulls in the optional ``[agent-spec]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, assert_never, cast

from neograph._ir_branch import _BranchNode, iter_with_arms
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import (
    COMBO_DECOMPOSITION,
    SUB_CONSTRUCT_UNSUPPORTED_COMBOS,
    Each,
    Loop,
    Operator,
    Oracle,
    PrimaryShape,
    classify_modifiers,
    is_each_oracle_fused,
)
from neograph.naming import field_name_for, split_output_field
from neograph.node import Node

if TYPE_CHECKING:
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.flow import Flow
    from pyagentspec.flows.node import Node as SpecNode

__all__ = ["to_agent_spec"]

# --- extracted clusters (neograph-3ffdg.3), re-exported so the public import
# --- surface of neograph._agent_spec is byte-identical to before the split.
from neograph._agent_spec_markers import (  # noqa: E402,F401
    _MARK_AGENT_SPEC,
    _MARK_BRANCH,
    _MARK_EACH_SPEC,
    _MARK_GROUP_ID,
    _MARK_LOOP_SPEC,
    _MARK_MODE,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    _MARK_ORACLE_SPEC,
    _MARK_PORTAL_OPERATOR_SPEC,
    _MARK_PORTAL_SPEC,
    _MARK_PROMPT_SPEC,
    _MARK_TOOL_SPEC,
    _MARK_VARIANT,
    _import_agent_spec_flow_classes,
)
from neograph._agent_spec_node_lowering import (  # noqa: E402,F401
    _agent_spec_marker,
    _lower_generation_step,
    _lower_node,
    _make_agent,
    _make_llm_config,
    _make_server_tool,
    _reject_unrepresentable_fields,
    _tool_to_server_tool,
)
from neograph._agent_spec_placeholders import (  # noqa: E402,F401
    _is_translation_eligible,
    _node_translation,
    _prompt_spec_marker,
    _properties_for,
    _translate_placeholders,
    compose_property_title,
    property_title_to_prompt_path,
    split_property_title,
)
from neograph._agent_spec_portal import (  # noqa: E402,F401
    _is_peer_mesh_member,
    _lower_portal_mesh_to_swarm,
)

_DEFAULT_BRANCH = "default"
_PAUSE_BRANCH = "pause"


def _item_inputs(item: Node | Construct) -> Any:
    """The input TypeSpec a modifier lowerer reads, uniform across the two
    item kinds it now wraps: a ``Node`` declares plural fan-in ``inputs``
    (``dict|type|None``); a ``Construct`` used as one item declares the
    singular boundary port ``input`` (``type|None``). Both feed
    ``_properties_for`` / ``normalize_inputs`` the same way, so the modifier
    helpers never branch on ``isinstance`` for I/O access."""
    return item.input if isinstance(item, Construct) else item.inputs


def _item_outputs(item: Node | Construct) -> Any:
    """The output TypeSpec counterpart of ``_item_inputs`` — ``Node.outputs``
    (plural) vs ``Construct.output`` (singular boundary port)."""
    return item.output if isinstance(item, Construct) else item.outputs


def _lower_item_body(item: Node | Construct) -> SpecNode:
    """Lower one construct item to the SpecNode a modifier wraps.

    A ``Node`` lowers via ``_lower_node`` (per-mode think/agent/scripted
    dispatch). A ``Construct`` used as one item lowers to the SAME
    ``FlowNode`` the BARE-Construct branch emits — its ``subflow`` is the
    recursively-exported sub-``Flow`` (``to_agent_spec``). Shared by
    ``_lower_each`` / ``_lower_loop`` / ``_lower_oracle`` and the
    LOOP/OPERATOR/BARE arms of ``_lower_construct_item`` so a Construct-item
    modifier wraps its sub-flow EXACTLY as a Node modifier wraps its lowered
    primitive — one body-lowering seam, never a per-modifier re-derivation."""
    if isinstance(item, Construct):
        nodes_mod, _flow_mod, _edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
        return nodes_mod.FlowNode(name=item.name, subflow=to_agent_spec(item))
    return _lower_node(item)


def _lower_oracle(
    node: Node | Construct, oracle: Oracle
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
            body = _lower_item_body(node)
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
            llm_config=_make_llm_config(Node(name=node.name, model=oracle.merge_model)),
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
        control_edges.append(
            edges_mod.ControlFlowEdge(name=f"{group_id}_fanout_{i}", from_node=variant, to_node=merge_node)
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


def _lower_each(node: Node | Construct, each: Each, oracle: Oracle | None = None) -> SpecNode:
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
        inner_inputs = _properties_for(_item_inputs(node))
    start_node = nodes_mod.StartNode(name=f"{node.name}__each_start", inputs=inner_inputs or None)
    end_node = nodes_mod.EndNode(name=f"{node.name}__each_end")

    if oracle is None:
        inner = _lower_item_body(node)
        body_nodes: list[SpecNode] = [inner]
        body_control = [
            edges_mod.ControlFlowEdge(name=f"{node.name}__each_start_edge", from_node=start_node, to_node=inner),
            edges_mod.ControlFlowEdge(name=f"{node.name}__each_end_edge", from_node=inner, to_node=end_node),
        ]
        body_data: list[DataFlowEdge] = []
    else:
        # FUSED Each x Oracle. The Oracle group's own variant->merge control edges
        # and fan-in data edges come through verbatim; only the Flow's boundary
        # wiring is added here. A pyagentspec StartNode must have EXACTLY ONE
        # outgoing control edge, so it points at the MERGE node — the same shape
        # to_agent_spec already builds at top level for an un-fused ORACLE item
        # (primary = the merge node; variants carry no inbound edge). The inner
        # merge node keeps the MapNode's own name, which validates and round-trips.
        body_nodes, body_control, body_data = _lower_oracle(node, oracle)
        merge = body_nodes[-1]  # _lower_oracle returns [*variants, merge]
        body_control = [
            edges_mod.ControlFlowEdge(name=f"{node.name}__each_start_edge", from_node=start_node, to_node=merge),
            *body_control,
            edges_mod.ControlFlowEdge(name=f"{node.name}__each_end_edge", from_node=merge, to_node=end_node),
        ]

    sub_flow = flow_mod.Flow(
        name=f"{node.name}__each_body",
        start_node=start_node,
        nodes=[start_node, *body_nodes, end_node],
        control_flow_connections=body_control,
        data_flow_connections=body_data or None,
    )
    return nodes_mod.MapNode(
        name=node.name,
        subflow=sub_flow,
        metadata={
            _MARK_MODIFIER: "each",
            _MARK_EACH_SPEC: {"over": each.over, "key": each.key, "on_error": each.on_error},
        },
    )


def _lower_loop(
    node: Node | Construct, loop: Loop, body: SpecNode
) -> tuple[SpecNode, list[ControlFlowEdge], list[DataFlowEdge]]:
    """Lower a Loop-modified item: BranchingNode({continue: back-edge, done: next}).

    ``body`` is the caller-lowered primitive (a Node's lowered node or a
    Construct-item's ``FlowNode``), so ``Construct(...) | Loop(...)`` loops its
    sub-flow the same way ``node | Loop(...)`` loops its body node.

    A bare BranchingNode+back-edge is ambiguous (loop vs branch) without the
    ``neograph/modifier=loop`` marker (per the Core Invariant's marker
    requirement) — always stamped.
    """
    nodes_mod, _flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()

    if callable(loop.when):
        raise ConfigurationError.build(
            f"node {node.name!r}'s Loop.when is a callable — no Agent Spec representation",
            expected="a registered condition NAME (str)",
            found="Loop.when is a callable",
            hint="only registered-string conditions serialize (callable-valued field, doc s6)",
        )

    branch = nodes_mod.BranchingNode(
        name=f"{node.name}__loop_check",
        mapping={"continue": "continue", "done": "done"},
        metadata={
            _MARK_MODIFIER: "loop",
            _MARK_LOOP_SPEC: {
                "when": loop.when,
                "max_iterations": loop.max_iterations,
                "on_exhaust": loop.on_exhaust,
            },
        },
    )
    control_edges = [
        edges_mod.ControlFlowEdge(name=f"{node.name}__loop_body_to_check", from_node=body, to_node=branch),
        edges_mod.ControlFlowEdge(
            name=f"{node.name}__loop_back", from_node=branch, from_branch="continue", to_node=body
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
    data_edges: list[DataFlowEdge] = []
    for prop in _properties_for(_item_outputs(node)):
        if _is_translation_eligible(node):
            dest_input = body_orig_to_flat.get(f"{dest_key}.{prop.title}" if dest_key else prop.title)
            if dest_input is None:
                continue
        else:
            dest_input = compose_property_title(dest_key, prop.title) if dest_key else prop.title
        data_edges.append(
            edges_mod.DataFlowEdge(
                name=f"{node.name}__loop_self_{prop.title}",
                source_node=body,
                source_output=prop.title,
                destination_node=body,
                destination_input=dest_input,
            )
        )
    return branch, control_edges, data_edges


def _lower_operator(
    node: Node | Construct, operator: Operator
) -> tuple[SpecNode, list[SpecNode], list[ControlFlowEdge]]:
    """Lower an Operator-modified item: the FULLY PINNED HITL-pause composite
    (neograph-03djs, verified against real pyagentspec 26.1.2 source).

    Reads only ``item.name`` + ``operator.when`` (the caller lowers the primary
    body separately), so it applies uniformly to a Node or a Construct item.

    ``BranchingNode(mapping={<condition-string>: PAUSE_BRANCH})`` +
    ``ControlFlowEdge(from_branch=PAUSE_BRANCH) -> InputMessageNode`` +
    ``ControlFlowEdge(from_branch=DEFAULT_BRANCH) -> reconverge``. The
    boolean-to-string-key coercion is REQUIRED: the condition's truthy
    result must render to the literal mapping-key string, or the composite
    silently always takes DEFAULT_BRANCH (never pauses).
    """
    nodes_mod, _flow_mod, edges_mod, property_mod, _tools_mod = _import_agent_spec_flow_classes()

    check = nodes_mod.BranchingNode(
        name=f"{node.name}__operator_check",
        mapping={"true": _PAUSE_BRANCH, "false": _DEFAULT_BRANCH},
        metadata={_MARK_MODIFIER: "operator", _MARK_OPERATOR_SPEC: {"when": operator.when}},
    )
    input_message = nodes_mod.InputMessageNode(
        name=f"{node.name}__operator_pause",
        outputs=[property_mod.StringProperty(title="user_input")],
    )
    pause_edge = edges_mod.ControlFlowEdge(
        name=f"{node.name}__operator_to_pause", from_node=check, from_branch=_PAUSE_BRANCH, to_node=input_message
    )
    return check, [input_message], [pause_edge]


_LoweredItem: TypeAlias = (
    "tuple[list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge], SpecNode, SpecNode, list[tuple[SpecNode, bool]]]"
)
"""What one lowered construct item is: (all_spec_nodes, extra_control_edges,
extra_data_edges, primary_node, data_node, input_targets). Named so the
per-shape arms of ``_lower_construct_item`` can BIND this shape and let the
shared Operator postlude rewrite it, instead of each arm returning its own."""


def _lower_construct_item(item: Any) -> _LoweredItem:
    """Lower one top-level construct item (Node/Construct/_BranchNode) to
    (all_spec_nodes, extra_control_edges, extra_data_edges, primary_node,
    data_node, input_targets).

    ``primary_node`` is the node other items' ControlFlowEdges attach to
    (the item's DX-visible identity — e.g. an Operator's check node, or an
    Oracle's merge node). ``data_node`` is the node that OTHER items read this
    item's OUTPUT Properties FROM (usually the same as ``primary_node``, except
    for LOOP, where the control-flow ``primary`` — the check ``BranchingNode``
    — declares no Properties, so the wrapped ``body`` is the output source).

    ``input_targets`` is the modifier-aware answer to "when a downstream edge
    feeds THIS item an external input, which SpecNode(s) receive it, and does
    the destination_input need the MapNode ``iterated_`` prefix?" — the single
    place every modifier destination's input routing lives, so the dict-form /
    single-type edge loops in ``to_agent_spec`` never re-derive it per-symptom:

      * BARE / LOOP / Construct / _BranchNode → the node that carries the input
        Properties (``data_node``), bare titles.
      * EACH → the MapNode, ``iterated_``-prefixed (its inputs are inferred as
        ``iterated_{title}`` from the sub-flow StartNode). neograph-hf505.
      * OPERATOR → the PRIMARY node (the real lowered node with Properties), NOT
        the ``check`` BranchingNode (which declares none).
      * ORACLE → EVERY variant node (each variant independently consumes the
        external input); the merge node consumes only the variant fan-in.
    """
    nodes_mod, flow_mod, _edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()

    if isinstance(item, _BranchNode):
        branch = nodes_mod.BranchingNode(
            name=item.name,
            mapping={"true": "true", "false": "false"},
            metadata={_MARK_BRANCH: True},
        )
        return [branch], [], [], branch, branch, [(branch, False)]

    if not isinstance(item, (Node, Construct)):
        raise ConfigurationError.build(
            f"unrecognized construct item {item!r} — no Agent Spec lowering",
            expected="Node, Construct, or _BranchNode",
            found=type(item).__name__,
        )

    # Node AND Construct items go through the SAME modifier dispatch — a
    # Construct item's modifiers are NOT silently dropped (the pre-fix bug: the
    # Construct branch wrapped a FlowNode and returned before classify_modifiers
    # ran). The wrapped body differs (a Node lowers per-mode; a Construct lowers
    # to a FlowNode over its sub-Flow), but that difference is absorbed once by
    # _lower_item_body, so every arm below is item-kind-agnostic.
    combo, mods = classify_modifiers(item)

    # A Construct item carrying an Each x Oracle fusion has no Construct-level
    # lowering — mirror compiler.py's OWN permanent rejection (_add_subgraph's
    # EACH_ORACLE | EACH_ORACLE_OPERATOR arm) rather than silently dropping it or
    # inventing a meaning. The fusion is defined via a single Node's
    # map_over/ensemble_n M x N Send topology, which a multi-node Construct
    # structurally lacks. SUB_CONSTRUCT_UNSUPPORTED_COMBOS is the single source
    # of truth both consumers consult (modifiers.py).
    if isinstance(item, Construct) and combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS:
        raise ConfigurationError.build(
            f"sub-construct {item.name!r} has modifier combination {combo.name} — "
            "Each x Oracle fusion is not supported on sub-constructs",
            expected="Each x Oracle only on a bare Node (map_over + ensemble_n), never a Construct",
            found=combo.name,
            hint="mirrors compiler.py's permanent Each x Oracle sub-construct rejection — the fusion "
            "is defined entirely via a single Node's map_over/ensemble_n fields, which a multi-node "
            "Construct has no equivalent for",
        )

    # Dispatch on the DECOMPOSED shape, never on combo members — COMBO_DECOMPOSITION
    # in modifiers.py is the single source of truth for what a combo means
    # (neograph-tjpn4, closing the epic's last hand-written enumeration).
    #
    # Three-part structure, mirroring compiler.py's _add_subgraph (:584) and
    # _add_node_to_graph (:703) one-for-one (neograph-s7zt3.10 / Phase 7):
    #   1. split the Each x Oracle FUSION out BEFORE the match (it decomposes to
    #      primary=EACH with has_operator=False, so no arm test would catch it);
    #   2. match on the primary shape — each arm BINDS its 6-tuple instead of
    #      returning, and is item-kind-agnostic and Operator-agnostic;
    #   3. ONE unconditional Operator postlude after the match.
    #
    # The postlude goes AFTER the match, not into a `case ... if ...` guard: a
    # guard would make `case _` reachable and swap ConfigurationError for
    # assert_never's AssertionError. And it hoists the LOWERING, never a raise —
    # the PORTAL arm still raises its own dispatch-mode message from inside the
    # arm, so PORTAL_OPERATOR keeps its specific text rather than a generic one.
    decomp = COMBO_DECOMPOSITION[combo]

    if is_each_oracle_fused(mods):
        # Each x Oracle: ONE MapNode whose subflow IS the un-fused Oracle
        # variant-fan-out + merge that _lower_oracle already produces. Composed,
        # not re-implemented — _lower_each grows an optional caller-lowered body
        # group exactly the way _lower_loop(node, loop, body) already takes one.
        map_node = _lower_each(item, mods["each"], oracle=mods["oracle"])
        arm: _LoweredItem = ([map_node], [], [], map_node, map_node, [(map_node, True)])
    else:
        match decomp.primary:
            case PrimaryShape.ORACLE:
                variant_and_merge, control_edges, data_edges = _lower_oracle(item, mods["oracle"])
                variants = variant_and_merge[:-1]
                merge = variant_and_merge[-1]
                arm = (variant_and_merge, control_edges, data_edges, merge, merge, [(v, False) for v in variants])

            case PrimaryShape.EACH:
                map_node = _lower_each(item, mods["each"])
                arm = ([map_node], [], [], map_node, map_node, [(map_node, True)])

            case PrimaryShape.LOOP:
                body = _lower_item_body(item)
                branch, extra_control, extra_data = _lower_loop(item, mods["loop"], body)
                arm = ([body, branch], extra_control, extra_data, branch, body, [(body, False)])

            case PrimaryShape.BARE:
                # BARE and OPERATOR are the SAME primary shape; has_operator is what
                # distinguishes them, and that difference now lives entirely in the
                # shared postlude below.
                primary = _lower_item_body(item)
                arm = ([primary], [], [], primary, primary, [(primary, False)])

            case PrimaryShape.PORTAL:
                # C2 (neograph-s7zt3.12): a DISPATCH-mode Portal (route="decide") reaching
                # here is genuinely unrepresentable in Agent Spec — fail LOUD, not "not yet".
                # Peer-mode Portal meshes were already intercepted in to_agent_spec (the
                # Swarm path), so any PORTAL combo at this point is dispatch mode. Its runtime
                # semantics — SYNTHESIZE an Agent Spec Flow from emitted data at runtime,
                # then compile+run it — has no static-spec primitive: every subflow-bearing
                # pyagentspec node (FlowNode/MapNode/ParallelFlowNode/CatchExceptionNode)
                # takes a STATIC ``subflow: Flow`` declared at authoring time, and
                # BranchingNode only selects among pre-declared branches (verified against
                # installed pyagentspec 26.1.2, tests/agent_spec_capabilities.py registry).
                # A dispatch flow is not knowable until runtime, so it is a permanent,
                # evidence-backed scope boundary — mirrors the Each x Oracle sub-construct
                # rejection's fail-loud shape.
                #
                # Raised from INSIDE the arm, deliberately: hoisting it into the shared
                # postlude would swap PORTAL_OPERATOR's dispatch-mode message for a
                # generic one. The arm covers PORTAL_OPERATOR only vacuously — a
                # dispatch-mode Portal carrying an Operator is rejected at IR level by
                # ModifierSet.with_modifier, so PORTAL_OPERATOR provably cannot reach
                # here. That is WHY the message may say "dispatch-mode" unconditionally.
                raise ConfigurationError.build(
                    f"node {item.name!r} is a dispatch-mode Portal (route='decide') — no Agent Spec lowering",
                    expected="a peer-mode Portal mesh (exported as a Swarm) or a non-Portal node",
                    found=f"dispatch-mode Portal ({combo.name})",
                    node=item.name,
                    hint="dispatch mode synthesizes and runs a flow from runtime-emitted data; Agent Spec has "
                    "no runtime-flow-synthesis primitive (every subflow node takes a static subflow), so it is "
                    "permanently unrepresentable — keep the dispatcher inside neograph, do not export it",
                )

            case _ as unreachable:
                assert_never(unreachable)

    # -- ONE unconditional Operator postlude, orthogonal to every primary shape --
    # Reuses _lower_operator as-is: it reads only item.name + operator.when and is
    # already item-kind- and shape-agnostic. The arm's data_node and input_targets
    # are preserved unchanged; only the control-flow primary moves to the check node
    # (other items' ControlFlowEdges attach to the gate, so the pause is reachable).
    if not decomp.has_operator:
        return arm

    arm_nodes, arm_control, arm_data, arm_primary, arm_data_node, arm_targets = arm
    _nodes_mod, _flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
    check, extra_nodes, extra_control = _lower_operator(item, mods["operator"])
    pre_edge = edges_mod.ControlFlowEdge(name=f"{item.name}__to_operator_check", from_node=arm_primary, to_node=check)
    return (
        [*arm_nodes, check, *extra_nodes],
        [*arm_control, pre_edge, *extra_control],
        arm_data,
        check,
        arm_data_node,
        arm_targets,
    )


def to_agent_spec(construct: Construct) -> Flow:
    """Export a neograph ``Construct`` (IR) to an Open Agent Spec ``Flow``
    (or, for a Portal mode-(a) peer mesh, a top-level ``Swarm``).

    LOWERS every modifier to flat Agent Spec primitives — the same lowering
    neograph performs when compiling to LangGraph, expressed in Agent Spec
    vocabulary. Fails loud (``ConfigurationError``) on any construct it
    cannot represent, rather than silently downgrading. See module
    docstring for the Core Invariant.
    """
    _nodes_mod, flow_mod, edges_mod, _property_mod, tools_mod = _import_agent_spec_flow_classes()

    all_items = list(iter_with_arms(construct))
    mesh_members = [item for item in all_items if _is_peer_mesh_member(item)]
    if mesh_members:
        if len(mesh_members) != len(all_items):
            raise ConfigurationError.build(
                f"construct {construct.name!r} mixes a Portal peer mesh with non-mesh nodes",
                expected="a construct that is EITHER entirely a Portal mesh OR has no Portal mesh members",
                found=f"{len(mesh_members)} mesh member(s) out of {len(all_items)} total node(s)",
                hint="a Swarm is a top-level AgenticComponent, not a Flow node — a mixed "
                "mesh+Flow construct has no single Agent Spec export shape yet",
            )
        # A1 admits a Construct mesh member into DETECTION (do0d9); C1
        # (_lower_portal_mesh_to_swarm) now lowers a Construct member to its
        # recursive sub-Flow (an AgenticComponent) alongside Node members, so the
        # cast widens to the real Node | Construct member union.
        return _lower_portal_mesh_to_swarm(
            construct,
            cast("list[Node | Construct]", mesh_members),
            tools_mod,
            to_agent_spec,
        )

    all_nodes: list[SpecNode] = []
    control_edges: list[ControlFlowEdge] = []
    data_edges: list[DataFlowEdge] = []
    primaries: list[SpecNode] = []
    data_nodes: list[SpecNode] = []
    item_by_name: dict[str, Any] = {}
    input_targets_by_item_name: dict[str, list[tuple[SpecNode, bool]]] = {}

    for item in iter_with_arms(construct):
        item_by_name[item.name] = item
        lowered_nodes, extra_control, extra_data, primary, data_node, input_targets = _lower_construct_item(item)
        all_nodes.extend(lowered_nodes)
        control_edges.extend(extra_control)
        data_edges.extend(extra_data)
        primaries.append(primary)
        data_nodes.append(data_node)
        input_targets_by_item_name[item.name] = input_targets

    # Explicit ControlFlowEdge per adjacent pair in Construct.nodes order.
    for prev_primary, next_primary in zip(primaries, primaries[1:], strict=False):
        control_edges.append(
            edges_mod.ControlFlowEdge(
                name=f"{prev_primary.name}_to_{next_primary.name}",
                from_node=prev_primary,
                to_node=next_primary,
            )
        )

    # Explicit DataFlowEdge per Node.inputs upstream-name mapping. The
    # destination(s) come from the item's modifier-aware ``input_targets`` (see
    # _lower_construct_item): a MapNode wants ``iterated_``-prefixed inputs, an
    # Oracle fans each external input to EVERY variant, an Operator targets its
    # PRIMARY (not the property-less check node) — one rule, no per-modifier
    # re-derivation here. As a SOURCE, the upstream's output still comes from
    # its single ``data_node``.
    ordered_items = list(iter_with_arms(construct))
    data_node_by_item_name = dict(zip((item.name for item in ordered_items), data_nodes, strict=True))

    def _emit_input_edges(
        item_name: str,
        upstream_name: str,
        source_node: SpecNode,
        source_title: str,
        dest_title: str | None = None,
    ) -> None:
        """Emit one DataFlowEdge per (destination target, prefix) for a single
        source Property. ``upstream_name`` is the dict-form key ('' for the
        single-type path, where the destination input title is the bare
        Property title, not the key-qualified one).

        ``dest_title`` decouples the DESTINATION input field from the
        ``source_output`` when they differ — needed for a dict-form-OUTPUT
        producer (B4), whose output Property is qualified by its output KEY but
        whose consumer declares it bare, or qualified by the UPSTREAM name.
        Defaults to ``source_title`` (unchanged for every single-shape caller).

        Option F consumer sweep neograph-cbpyx: when the CONSUMING item is
        placeholder-translated (LLM mode), the destination declares the flat
        ${var}->{{ flat }} name, so the ``${upstream.title}`` path (and the
        MapNode's ``iterated_``-prefixed form) route through the item's flat map —
        and the edge is DROPPED when the source path was never referenced in the
        prompt (a real topology change: the translated primitive has no data path
        to that value). Scripted/raw destinations keep the untranslated form.

        The two name spaces MUST NOT be conflated -- neograph-8zvd1: the flat map
        is keyed by the dotted ``${...}`` PROMPT path, ``destination_input`` by
        the Agent Spec Property TITLE the destination declares.
        """
        dest_item = item_by_name.get(item_name)
        translate = _is_translation_eligible(dest_item)
        orig_to_flat = _node_translation(cast("Node", dest_item))[2] if translate else {}
        dest_core = dest_title if dest_title is not None else source_title
        for target_node, iterated in input_targets_by_item_name[item_name]:
            if translate:
                flat = orig_to_flat.get(f"{upstream_name}.{dest_core}" if upstream_name else dest_core)
                if flat is None:
                    continue
                core = flat
            else:
                # A MapNode infers its inputs as ``iterated_{json_schema title}``
                # from the INNER node's own Properties, so the qualified title is
                # right for the MapNode too — it is what the inner node declares.
                core = compose_property_title(upstream_name, dest_core) if upstream_name else dest_core
            dest_input = f"iterated_{core}" if iterated else core
            data_edges.append(
                edges_mod.DataFlowEdge(
                    name=f"{source_node.name}_to_{target_node.name}_{dest_input}",
                    source_node=source_node,
                    source_output=source_title,
                    destination_node=target_node,
                    destination_input=dest_input,
                )
            )

    for idx, item in enumerate(ordered_items):
        if not isinstance(item, Node):
            continue
        ni = normalize_inputs(item.inputs)
        if ni.is_none:
            continue

        if ni.is_dict_form:
            # Dict-form fan-in: named upstream -> per-field Property edges.
            # upstream_name is the inputs-dict KEY (the upstream NODE'S NAME),
            # never itself a Property title -- resolve the upstream's real
            # output Property titles (mirrors the single-type fallback below
            # and the Oracle/Loop precedent, which all key on prop.title).
            fan_out_key = getattr(item, "fan_out_param", None)
            for upstream_name in ni.by_name:
                if upstream_name == fan_out_key:
                    # The Each fan-out receiver slot is not an upstream NODE
                    # name -- it's populated per-item by the MapNode's own
                    # sub-flow wiring (_lower_each), so no DataFlowEdge here
                    # (mirrors _validation_inputs.py's fan_out_param skip).
                    continue
                upstream_item = item_by_name.get(upstream_name)
                source_node = data_node_by_item_name.get(upstream_name)
                output_key: str | None = None
                if upstream_item is None:
                    # B4: a dict-form-OUTPUT producer is referenced by ONE of its
                    # output keys via the ``{producer}_{key}`` state-field naming
                    # the validator registers producers under (naming.output_field_name).
                    # Recover (producer node, output key) with the canonical inverse
                    # split — never an ad hoc rsplit — then wire ONLY that key's
                    # ``{key}.{field}`` output Properties.
                    for cand in ordered_items:
                        if not isinstance(cand, Node):
                            continue
                        key = split_output_field(upstream_name, field_name_for(cand.name))
                        if key is not None and key in normalize_outputs(cand.outputs).all_keys:
                            upstream_item = cand
                            source_node = data_node_by_item_name.get(cand.name)
                            output_key = key
                            break
                if upstream_item is None or source_node is None or not isinstance(upstream_item, Node):
                    raise ConfigurationError.build(
                        f"node {item.name!r}'s dict-form inputs references upstream "
                        f"{upstream_name!r}, which has no exportable Agent Spec node",
                        expected="an upstream Node producing a resolvable output",
                        found=f"no node named {upstream_name!r} in the construct",
                        hint="dict-form fan-in against a multi-output producer referenced "
                        "via '{upstream}_{key}' naming has no Agent Spec representation yet",
                    )
                no = normalize_outputs(upstream_item.outputs)
                if no.is_none:
                    raise ConfigurationError.build(
                        f"node {item.name!r}'s dict-form inputs references upstream "
                        f"{upstream_name!r}, whose outputs are None",
                        expected="a Node.outputs producing at least one exportable type",
                        found=f"{upstream_name!r}.outputs is None",
                        hint="an upstream with no outputs has no Agent Spec Property to wire",
                    )
                if output_key is not None:
                    # B4: dict-form producer, one key referenced. The producer's
                    # output Property is qualified by ``output_key``, the consumer's
                    # declared input by ``upstream_name`` — decouple source/dest so
                    # the edge is wired, not dropped.
                    for prop in _properties_for({output_key: no.all_keys[output_key]}):
                        field = split_property_title(prop.title)[1]
                        _emit_input_edges(item.name, upstream_name, source_node, prop.title, dest_title=field)
                else:
                    # Single-type producer referenced directly by node name.
                    for prop in _properties_for(upstream_item.outputs):
                        _emit_input_edges(item.name, upstream_name, source_node, prop.title)
            continue

        # Single-type inputs (convenience shorthand): the producer is
        # resolved by an O(N) type-compatibility scan over preceding
        # items, mirroring the assembly-time validator's single-type
        # resolution (_construct_validation.py) rather than a dict key.
        input_props = {p.title for p in _properties_for(ni.single_type)}
        for upstream in reversed(ordered_items[:idx]):
            if not isinstance(upstream, Node):
                continue
            no = normalize_outputs(upstream.outputs)
            if no.is_none:
                continue
            source_node = data_node_by_item_name[upstream.name]
            if no.is_dict_form:
                # B4: a dict-form-output producer is no longer skipped. Match the
                # consumer's single input type against each output KEY; wire the
                # matching key's qualified output Property to the consumer's
                # bare ``{field}`` input (source/dest decoupled, same as the
                # dict-form-input branch above).
                matched = False
                for key, ktype in no.all_keys.items():
                    if not isinstance(ktype, type) or not isinstance(ni.single_type, type):
                        continue
                    if not (issubclass(ktype, ni.single_type) or issubclass(ni.single_type, ktype)):
                        continue
                    for prop in _properties_for({key: ktype}):
                        field = split_property_title(prop.title)[1]
                        if field in input_props:
                            _emit_input_edges(item.name, "", source_node, prop.title, dest_title=field)
                            matched = True
                if matched:
                    break
                continue
            if not (issubclass(no.primary, ni.single_type) or issubclass(ni.single_type, no.primary)):
                continue
            upstream_props = {p.title for p in _properties_for(no.primary)}
            for shared_title in input_props & upstream_props:
                _emit_input_edges(item.name, "", source_node, shared_title)
            break

    if not primaries:
        raise ConfigurationError.build(
            f"construct {construct.name!r} has no nodes — nothing to export",
            expected="at least one node",
            found="empty construct.nodes",
        )

    # A Flow requires exactly one StartNode and >=1 EndNode; neograph's
    # Construct has no explicit start/end sentinels (the node order IS the
    # DAG), so wrap the lowered chain with synthetic boundary nodes.
    #
    # When this construct is a SUB-construct (used as one item inside a parent,
    # so it declares an input/output boundary port), the synthetic StartNode /
    # EndNode carry that boundary I/O — pyagentspec's FlowNode infers its own
    # inputs from the inner StartNode and its outputs from the inner EndNode
    # (FlowNode docstring). Without this a Construct-item modifier that reads the
    # sub-flow's OUTPUT — Loop's self-feedback edge, Oracle's variant fan-in —
    # references a property the FlowNode does not expose, and pyagentspec raises
    # a raw ValidationError. The OUTERMOST construct has input/output=None, so
    # both stay unset there (unchanged behavior). Declaring boundary props needs
    # no extra internal wiring: an unconsumed StartNode input / unfed EndNode
    # output is legal (the payload flows through the item's own peer edges).
    start_props = _properties_for(construct.input)
    end_props = _properties_for(construct.output)
    start_node = _nodes_mod.StartNode(name=f"{construct.name}__start", inputs=start_props or None)
    end_node = _nodes_mod.EndNode(name=f"{construct.name}__end", outputs=end_props or None)
    all_nodes = [start_node, *all_nodes, end_node]
    control_edges = [
        edges_mod.ControlFlowEdge(name=f"{construct.name}__start_edge", from_node=start_node, to_node=primaries[0]),
        *control_edges,
        edges_mod.ControlFlowEdge(name=f"{construct.name}__end_edge", from_node=primaries[-1], to_node=end_node),
    ]

    metadata: dict[str, Any] = {}
    flow = flow_mod.Flow(
        name=construct.name,
        start_node=start_node,
        nodes=all_nodes,
        metadata=metadata,
        control_flow_connections=control_edges,
        data_flow_connections=data_edges or None,
    )
    return flow
