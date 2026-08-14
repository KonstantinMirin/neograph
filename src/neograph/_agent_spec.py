"""``to_agent_spec()`` — export neograph IR (``Construct``) to an Open Agent
Spec ``Flow``.

A free function, NOT a ``Construct``/``Node`` method (CLAUDE.md layer
discipline, design doc agent-spec-interop-2026-07-09.md §7). LOWERS each
modifier to the flat Agent Spec primitives it already lowers to for LangGraph
compilation (Oracle fan-out/barrier, Each router/Send/barrier, Loop back-edge,
Operator's check-node-with-interrupt), per the exporter's Core Invariant: this
is the SAME lowering neograph performs when compiling, expressed in Agent Spec
vocabulary instead of LangGraph's — never a second, divergent lowering.

The main dispatch loop walks RAW ``construct.nodes``, never ``iter_with_arms``
(right for the compiler's/lint's MEMBERSHIP-only consumers, wrong here: the
exporter must see a ``_BranchNode`` boundary to emit a real ``BranchingNode``,
per ``_lower_top_level_item`` / ``_lower_branch``, neograph-s7zt3.17) — the
DATA-edge phase below still uses ``iter_with_arms``, which is right there.

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

import functools
from typing import TYPE_CHECKING, Any, TypeAlias, assert_never, cast

from neograph._ir_branch import iter_with_arms
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import (
    COMBO_DECOMPOSITION,
    SUB_CONSTRUCT_UNSUPPORTED_COMBOS,
    PrimaryShape,
    classify_modifiers,
)
from neograph.naming import field_name_for, split_output_field
from neograph.node import Node

if TYPE_CHECKING:
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.flow import Flow
    from pyagentspec.flows.node import Node as SpecNode

__all__ = ["to_agent_spec"]

# --- extracted clusters (neograph-3ffdg.3), re-exported so this module's public import surface is unchanged.
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
    _MARK_PORTAL_MEMBER_SPEC,
    _MARK_PORTAL_OPERATOR_SPEC,
    _MARK_PORTAL_SPEC,
    _MARK_PROMPT_SPEC,
    _MARK_TOOL_SPEC,
    _MARK_VARIANT,
    Branch,
    _import_agent_spec_flow_classes,
)
from neograph._agent_spec_modifier_lowering import (  # noqa: E402,F401
    _lower_each,
    _lower_item_body,
    _lower_loop,
    _lower_operator,
    _lower_oracle,
    _lower_top_level_item,
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
    _item_inputs,
    _item_outputs,
    _node_translation,
    _prompt_spec_marker,
    _properties_for,
    _translate_placeholders,
    compose_property_title,
    property_title_to_prompt_path,
    split_property_title,
)
from neograph._agent_spec_boundary import resolve_end_node_sources  # noqa: E402
from neograph._agent_spec_portal import (  # noqa: E402,F401
    _is_peer_mesh_member,
    _lower_portal_mesh_to_swarm,
)

_Exit: TypeAlias = "tuple[SpecNode, str | None]"
"""One outgoing control-flow endpoint of a lowered item: the SpecNode a
successor edge leaves FROM, plus the ``from_branch`` it must name (``None`` for
an unconditional node). An item has several only when its lowering genuinely
forks — an Operator reconverges its gate's DEFAULT_BRANCH and its post-pause
continuation on the same successor."""

_LoweredItem: TypeAlias = (
    "tuple[list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge], "
    "SpecNode, list[_Exit], SpecNode, list[tuple[SpecNode, bool]]]"
)
"""What one lowered construct item is: (all_spec_nodes, extra_control_edges,
extra_data_edges, entry_node, exits, data_node, input_targets). Named so the
per-shape arms of ``_lower_construct_item`` can BIND this shape and let the
shared Operator postlude rewrite it, instead of each arm returning its own."""


def _lower_construct_item(item: Any, api_provider: str | None = None) -> _LoweredItem:
    """Lower one top-level construct item (Node/Construct/_BranchNode) to
    (all_spec_nodes, extra_control_edges, extra_data_edges, entry_node, exits,
    data_node, input_targets).

    ``entry_node`` is the node an INCOMING ControlFlowEdge lands on — the first
    node of this item a literal edge-walking executor runs. ``exits`` is the
    dual: the ``(from_node, from_branch)`` endpoints an OUTGOING edge leaves
    from. They are separate because several lowerings are not single-node — a
    Loop is entered at its BODY and left from its check's ``done`` branch; an
    Operator is entered at the body it guards and left from BOTH its gate's
    default branch and its pause node. Collapsing the two roles onto one
    ``primary`` node is what made the Operator body unreachable and the pause
    node a dead end (neograph-s7zt3.15).

    ``data_node`` is the node that OTHER items read this item's OUTPUT Properties
    FROM (usually the entry/exit node, except for LOOP, where the check
    ``BranchingNode`` declares no Properties, so the wrapped ``body`` is the
    output source).

    ``input_targets`` is the modifier-aware answer to "when a downstream edge
    feeds THIS item an external input, which SpecNode(s) receive it, and does
    the destination_input need the MapNode ``iterated_`` prefix?" — the single
    place every modifier destination's input routing lives, so the dict-form /
    single-type edge loops in ``to_agent_spec`` never re-derive it per-symptom:

      * BARE / LOOP / Construct / _BranchNode → the node that carries the input
        Properties (``data_node``), bare titles.
      * EACH → the MapNode, ``iterated_``-prefixed (its inputs are inferred as
        ``iterated_{title}`` from the sub-flow StartNode). neograph-hf505.
      * OPERATOR → whatever the GUARDED arm already targets (the real lowered
        node with Properties), NOT the ``check`` BranchingNode (declares none).
      * ORACLE → EVERY variant node (each variant independently consumes the
        external input); the merge node consumes only the variant fan-in.
    """
    # neograph-qtfof.8: bind api_provider into export_flow so a Construct-item's
    # own recursive sub-export gets it too, without widening every _lower_*'s
    # callable param to also carry a scalar.
    flow_export = functools.partial(to_agent_spec, api_provider=api_provider)
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

    if decomp.fused:
        # Each x Oracle: ONE MapNode whose subflow IS the un-fused Oracle
        # variant-fan-out + merge that _lower_oracle already produces. Composed,
        # not re-implemented — _lower_each grows an optional caller-lowered body
        # group exactly the way _lower_loop(node, loop, body) already takes one.
        map_node = _lower_each(item, mods["each"], flow_export, oracle=mods["oracle"], api_provider=api_provider)
        arm: _LoweredItem = ([map_node], [], [], map_node, [(map_node, None)], map_node, [(map_node, True)])
    else:
        match decomp.primary:
            case PrimaryShape.ORACLE:
                variant_and_merge, control_edges, data_edges = _lower_oracle(item, mods["oracle"], flow_export, api_provider=api_provider)
                variants = variant_and_merge[:-1]
                merge = variant_and_merge[-1]
                # ENTRY is the head of the variant chain, not the merge: the merge is
                # the group's DX identity and its data source, but entering there
                # skips every variant whose output it consumes.
                arm = (
                    variant_and_merge,
                    control_edges,
                    data_edges,
                    variants[0],
                    [(merge, None)],
                    merge,
                    [(v, False) for v in variants],
                )

            case PrimaryShape.EACH:
                map_node = _lower_each(item, mods["each"], flow_export, api_provider=api_provider)
                arm = ([map_node], [], [], map_node, [(map_node, None)], map_node, [(map_node, True)])

            case PrimaryShape.LOOP:
                body = _lower_item_body(item, flow_export, api_provider=api_provider)
                branch, extra_control, extra_data = _lower_loop(item, mods["loop"], body)
                # neograph's Loop is a DO-while (_wiring._add_subgraph_loop wires
                # ``prev -> body`` and only THEN the conditional back-edge), so the
                # group is ENTERED at the body and LEFT through the check's ``done``
                # branch. Entering at the check evaluates ``when`` against state the
                # body has never written -- a different program, not a different
                # spelling of the same one.
                arm = ([body, branch], extra_control, extra_data, body, [(branch, Branch.DONE)], body, [(body, False)])

            case PrimaryShape.BARE:
                # BARE and OPERATOR are the SAME primary shape; has_operator is what
                # distinguishes them, and that difference now lives entirely in the
                # shared postlude below.
                primary = _lower_item_body(item, flow_export, api_provider=api_provider)
                arm = ([primary], [], [], primary, [(primary, None)], primary, [(primary, False)])

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
    # already item-kind- and shape-agnostic. The arm's ENTRY, data_node and
    # input_targets are preserved unchanged -- the gate runs AFTER the body it
    # guards (_wiring._add_operator_check wires ``node -> check``), so it appends
    # itself to the arm's EXITS rather than taking over its entry.
    if not decomp.has_operator:
        return arm

    arm_nodes, arm_control, arm_data, arm_entry, arm_exits, arm_data_node, arm_targets = arm
    _nodes_mod, _flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
    check, pause, extra_control = _lower_operator(item, mods["operator"])
    # EVERY one of the arm's exits feeds the gate, so an Operator over a forking
    # primary would still be gated on all paths rather than on one of them.
    pre_edges = [
        edges_mod.ControlFlowEdge(
            name=f"{exit_node.name}__to_operator_check", from_node=exit_node, from_branch=branch, to_node=check
        )
        for exit_node, branch in arm_exits
    ]
    # The composite exits on BOTH the non-pausing DEFAULT_BRANCH and the pause
    # node, reconverging on the item's successor. At runtime the interrupt happens
    # INSIDE the check node (_wiring._add_operator_check), so once the human
    # answers, execution resumes there and falls through to the next node -- the
    # pause node's outgoing edge IS that fall-through, not an invented path.
    return (
        [*arm_nodes, check, pause],
        [*arm_control, *pre_edges, *extra_control],
        arm_data,
        arm_entry,
        [(check, Branch.DEFAULT), (pause, None)],
        arm_data_node,
        arm_targets,
    )


def to_agent_spec(construct: Construct, api_provider: str | None = None) -> Flow:
    """Export a neograph ``Construct`` (IR) to an Open Agent Spec ``Flow``
    (or, for a Portal mode-(a) peer mesh, a top-level ``Swarm``).

    LOWERS every modifier to flat Agent Spec primitives — the same lowering
    neograph performs when compiling to LangGraph, expressed in Agent Spec
    vocabulary. Fails loud (``ConfigurationError``) on the ENUMERATED fields
    it cannot represent (module docstring) — not raising does NOT mean the
    result is portable; call ``export_conformance(construct)`` for that.

    ``api_provider`` (neograph-qtfof.8, fail-loud opt-in): the real provider is
    unknowable at export time (``Node.model`` is an opaque tier string resolved
    via ``llm_factory`` at runtime). ``None`` (default) keeps every exported
    ``LlmConfig.api_provider`` unset -- honest, ``NEOGRAPH_ROUND_TRIP_ONLY``.
    Pass e.g. ``"openai"`` to make LLM-bearing nodes convertible, at the
    caller's own risk of asserting a provider the pipeline may not run on.
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
            functools.partial(to_agent_spec, api_provider=api_provider),
        )

    all_nodes: list[SpecNode] = []
    control_edges: list[ControlFlowEdge] = []
    data_edges: list[DataFlowEdge] = []
    entries: list[SpecNode] = []
    exits: list[list[_Exit]] = []
    item_by_name: dict[str, Any] = {}
    input_targets_by_item_name: dict[str, list[tuple[SpecNode, bool]]] = {}
    data_node_by_item_name: dict[str, SpecNode] = {}

    # Raw construct.nodes, NOT iter_with_arms — a _BranchNode must be seen as a
    # boundary here (see module docstring). _lower_top_level_item dispatches a
    # plain item vs a _BranchNode uniformly, returning per-item bookkeeping this
    # loop merges in so the DATA-edge phase below -- which walks
    # iter_with_arms(construct) and needs membership-only visibility into
    # arm-internal nodes -- can still resolve them by name exactly as pre-fix.
    for item in construct.nodes:
        lowered = _lower_top_level_item(item, functools.partial(_lower_construct_item, api_provider=api_provider))
        lowered_nodes, extra_control, extra_data, entry, item_exits, names, targets, data_nodes = lowered
        item_by_name.update(names)
        input_targets_by_item_name.update(targets)
        data_node_by_item_name.update(data_nodes)
        all_nodes.extend(lowered_nodes)
        control_edges.extend(extra_control)
        data_edges.extend(extra_data)
        entries.append(entry)
        exits.append(item_exits)

    # Explicit ControlFlowEdge per adjacent pair in Construct.nodes order, from
    # every EXIT of the previous item to the ENTRY of the next. An item with more
    # than one exit (an Operator: its gate's default branch plus its pause node's
    # continuation) reconverges them all on that single entry -- which is what
    # makes both the paused and un-paused paths continue the program.
    for prev_exits, next_entry in zip(exits, entries[1:], strict=False):
        for from_node, from_branch in prev_exits:
            control_edges.append(
                edges_mod.ControlFlowEdge(
                    name=f"{from_node.name}_to_{next_entry.name}",
                    from_node=from_node,
                    from_branch=from_branch,
                    to_node=next_entry,
                )
            )

    # Explicit DataFlowEdge per Node.inputs upstream-name mapping. The
    # destination(s) come from the item's modifier-aware ``input_targets`` (see
    # _lower_construct_item): a MapNode wants ``iterated_``-prefixed inputs, an
    # Oracle fans each external input to EVERY variant, an Operator targets the
    # body it guards (not the property-less check node) — one rule, no per-modifier
    # re-derivation here. As a SOURCE, the upstream's output still comes from
    # its single ``data_node``.
    ordered_items = list(iter_with_arms(construct))

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

    if not entries:
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
    start_node = _nodes_mod.StartNode(name=f"{construct.name}__start", inputs=start_props or None)
    # neograph-qtfof.9: outputs + real DataFlowEdge(s) from the terminal producer.
    end_props, end_sources = resolve_end_node_sources(construct, data_node_by_item_name)
    end_node = _nodes_mod.EndNode(name=f"{construct.name}__end", outputs=end_props or None)
    input_targets_by_item_name[end_node.name] = [(end_node, False)]
    for source_node, prop_title in end_sources:
        _emit_input_edges(end_node.name, "", source_node, prop_title)
    all_nodes = [start_node, *all_nodes, end_node]
    # The last item's every exit terminates at the EndNode, for the same reason
    # the inter-item edges above fan from every exit: a branch-qualified exit that
    # is dropped here is a path with no way to finish. The branch suffix keeps the
    # names unique without renaming the single-exit case every construct has.
    control_edges = [
        edges_mod.ControlFlowEdge(name=f"{construct.name}__start_edge", from_node=start_node, to_node=entries[0]),
        *control_edges,
        *(
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__end_edge" + (f"_{from_branch}" if from_branch else ""),
                from_node=from_node,
                from_branch=from_branch,
                to_node=end_node,
            )
            for from_node, from_branch in exits[-1]
        ),
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
