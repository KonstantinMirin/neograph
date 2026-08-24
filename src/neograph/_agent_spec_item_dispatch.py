"""Per-item modifier dispatch for the Agent Spec export (neograph-qtfof.13 split).

Extracted from ``_agent_spec.py`` when that file crossed its size ceiling. The
boundary is real rather than arithmetic: this module answers "what does ONE
construct item lower to, given its ModifierCombo", while ``_agent_spec.py``
assembles the resulting pieces into a Flow.

``flow_export`` is INJECTED rather than imported. The recursive sub-export lives
in ``_agent_spec`` and importing it here would close a cycle; threading it as a
parameter is the same dependency-ladder rung ``export_flow``, ``shim_factory``
and ``resolve_condition`` already landed on (see AGENTS.md, file-split
procedure). The caller binds it to ``_to_agent_spec_with`` with the SAME resolver
it passes here, so a sub-construct inherits the memoised tier cache instead of
rebuilding one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_never

from neograph._agent_spec_lowering_types import _ExportFlow
from neograph._agent_spec_markers import Branch, _import_agent_spec_flow_classes
from neograph._agent_spec_modifier_lowering import (
    _lower_each,
    _lower_item_body,
    _lower_loop,
    _lower_operator,
    _lower_oracle,
)
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import (
    COMBO_DECOMPOSITION,
    SUB_CONSTRUCT_UNSUPPORTED_COMBOS,
    PrimaryShape,
    classify_modifiers,
)
from neograph.node import Node

if TYPE_CHECKING:
    from neograph._agent_spec_lowering_types import _LoweredItem
    from neograph._agent_spec_provider import ApiProviderResolver

__all__ = ["_lower_construct_item"]


def _lower_construct_item(item: Any, provider: ApiProviderResolver, flow_export: _ExportFlow) -> _LoweredItem:
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
        map_node = _lower_each(item, mods["each"], flow_export, oracle=mods["oracle"], provider=provider)
        arm: _LoweredItem = ([map_node], [], [], map_node, [(map_node, None)], map_node, [(map_node, True)])
    else:
        match decomp.primary:
            case PrimaryShape.ORACLE:
                variant_and_merge, control_edges, data_edges = _lower_oracle(
                    item, mods["oracle"], flow_export, provider=provider
                )
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
                map_node = _lower_each(item, mods["each"], flow_export, provider=provider)
                arm = ([map_node], [], [], map_node, [(map_node, None)], map_node, [(map_node, True)])

            case PrimaryShape.LOOP:
                body = _lower_item_body(item, flow_export, provider=provider)
                branch, extra_nodes, extra_control, extra_data = _lower_loop(item, mods["loop"], body)
                # neograph's Loop is a DO-while (_wiring._add_subgraph_loop wires
                # ``prev -> body`` and only THEN the conditional back-edge), so the
                # group is ENTERED at the body and LEFT through the check's ``done``
                # branch. Entering at the check evaluates ``when`` against state the
                # body has never written -- a different program, not a different
                # spelling of the same one.
                # neograph-qtfof.6: DEFAULT_BRANCH is runtime-reachable too (any
                # unmapped predicate output lands there) -- not a dead end. Routed
                # to DONE's target: no max_iterations cap travels to a foreign
                # runtime, so falling to CONTINUE risks an unbounded loop.
                arm = (
                    [body, *extra_nodes, branch],
                    extra_control,
                    extra_data,
                    body,
                    [(branch, Branch.DONE), (branch, Branch.DEFAULT)],
                    body,
                    [(body, False)],
                )

            case PrimaryShape.BARE:
                # BARE and OPERATOR are the SAME primary shape; has_operator is what
                # distinguishes them, and that difference now lives entirely in the
                # shared postlude below.
                primary = _lower_item_body(item, flow_export, provider=provider)
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
