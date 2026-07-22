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
Spec representation at all), and ``agent``/``act`` mode (would silently drop
prompt/model/tools — real ``AgentNode``+tools lowering tracked in
``neograph-i3zsh.1``).

Import-guarded (mirrors ``spec_types._import_agent_spec_property_classes()``)
so ``src/neograph`` core stays Agent-Spec-free by default — only calling
``to_agent_spec()`` pulls in the optional ``[agent-spec]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neograph._ir_branch import _BranchNode, iter_with_arms
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import Each, Loop, ModifierCombo, Operator, Oracle, classify_modifiers
from neograph.node import Node
from neograph.spec_types import model_to_agent_spec_properties

if TYPE_CHECKING:
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.flow import Flow
    from pyagentspec.flows.node import Node as SpecNode
    from pyagentspec.property import Property

__all__ = ["to_agent_spec"]

_DEFAULT_BRANCH = "default"
_PAUSE_BRANCH = "pause"


def _import_agent_spec_flow_classes() -> Any:
    """Function-local import of pyagentspec's Flow/node/edge classes.

    Copies ``spec_types._import_agent_spec_property_classes()``'s exact
    import-guard shape so ``src/neograph`` core stays Agent-Spec-free by
    default — only calling ``to_agent_spec()`` pulls in the optional
    ``[agent-spec]`` extra.
    """
    try:
        import pyagentspec.flows.edges as edges_mod
        import pyagentspec.flows.flow as flow_mod
        import pyagentspec.flows.nodes as nodes_mod
        import pyagentspec.property as property_mod
        import pyagentspec.tools as tools_mod
    except ImportError as exc:
        raise ConfigurationError.build(
            "pyagentspec is not installed",
            expected="the [agent-spec] optional extra",
            found="ImportError on pyagentspec.flows/property/tools",
            hint="install with: uv sync --extra agent-spec (or pip install neograph[agent-spec])",
        ) from exc
    return nodes_mod, flow_mod, edges_mod, property_mod, tools_mod


def _reject_unrepresentable_fields(node: Node) -> None:
    """Fail loud on any Node field that has no Agent Spec representation.

    Per the Core Invariant, ``to_agent_spec()`` must never silently drop a
    construct it cannot lower. Checked before any lowering attempt.
    """
    if node.raw_fn is not None:
        raise ConfigurationError.build(
            f"node {node.name!r} uses raw_fn — a Python callable with no Agent Spec representation",
            expected="scripted/think/agent/act mode with a name-serializable body",
            found="raw_fn set",
            hint="raw_fn nodes cannot be exported to Agent Spec (callable-valued field, doc s6)",
        )
    if node.skip_when is not None or node.skip_value is not None:
        raise ConfigurationError.build(
            f"node {node.name!r} uses skip_when/skip_value — Python callables with no Agent Spec representation",
            expected="a node without conditional-skip logic",
            found="skip_when and/or skip_value set",
            hint="skip_when/skip_value cannot be exported to Agent Spec (callable-valued field, doc s6)",
        )
    if node.renderer is not None:
        raise ConfigurationError.build(
            f"node {node.name!r} uses a custom renderer — no Agent Spec representation",
            expected="the default rendering pipeline",
            found="renderer set",
            hint="a custom renderer cannot be exported to Agent Spec (callable-valued field, doc s6)",
        )
    if node.handoff_param is not None or node.handoff_channel is not None:
        raise ConfigurationError.build(
            f"node {node.name!r} is a Portal mesh member (handoff_param/handoff_channel set) — "
            "Agent Spec has no combinator for runtime peer-to-peer Command(goto) routing",
            expected="a node outside a Portal mesh",
            found="handoff_param and/or handoff_channel set",
            hint="Portal mesh members cannot be exported to Agent Spec — symmetric to the "
            "ratified Swarm import-reject (agent-spec-ratification-2026-07-13.md)",
        )
    if callable(node.gate_tools_when):
        raise ConfigurationError.build(
            f"node {node.name!r} uses a callable gate_tools_when — no Agent Spec representation",
            expected="a registered condition NAME (str) or no gate_tools_when",
            found="gate_tools_when is a callable",
            hint="only the string (registered-condition-name) form of gate_tools_when serializes",
        )


def _properties_for(type_spec: Any) -> list[Property]:
    """Convert a Node.inputs/outputs TypeSpec (None | type | dict[str, type]) to Properties.

    Reuses ``spec_types.model_to_agent_spec_properties`` for every model —
    never a second type walker, per the Core Invariant.
    """
    if type_spec is None:
        return []
    if isinstance(type_spec, dict):
        result: list[Property] = []
        for key, typ in type_spec.items():
            props = model_to_agent_spec_properties(typ)
            for p in props:
                p.title = f"{key}.{p.title}"
            result.extend(props)
        return result
    return model_to_agent_spec_properties(type_spec)


def _lower_node(node: Node) -> SpecNode:
    """Dispatch a single neograph Node to its Agent Spec primitive by mode."""
    _reject_unrepresentable_fields(node)
    nodes_mod, _flow_mod, _edges_mod, _property_mod, tools_mod = _import_agent_spec_flow_classes()

    inputs = _properties_for(node.inputs)
    outputs = _properties_for(node.outputs)

    if node.mode == "think":
        return nodes_mod.LlmNode(
            name=node.name,
            inputs=inputs or None,
            outputs=outputs or None,
            llm_config=_make_llm_config(node),
            prompt_template=node.prompt or "",
        )

    if node.mode in ("agent", "act"):
        # Agent/act export is NOT yet round-trip-safe. No single Agent Spec node
        # captures the ReAct tool loop, and a ToolNode placeholder would SILENTLY
        # drop the node's prompt/model/tools (`_make_server_tool` fabricates one
        # tool from the I/O signature, not the node's real `tools=[...]`), leaving
        # only a `neograph/mode` marker. Per the Core Invariant (never silently
        # drop), reject it until the real AgentNode+tools lowering lands — rather
        # than ship a lossy placeholder that breaks round-trip once from_agent_spec
        # exists. Tracked in neograph-i3zsh.1.
        raise ConfigurationError.build(
            f"node {node.name!r} is {node.mode!r} mode — agent/act export to Agent Spec "
            "is not yet round-trip-safe (prompt/model/tools would be silently dropped)",
            expected="scripted or think mode",
            found=f"{node.mode!r} mode",
            hint="agent/act -> AgentNode+tools lowering is tracked in neograph-i3zsh.1; "
            "until it lands, export fails loud rather than emit a lossy ToolNode placeholder",
        )

    # scripted / raw already rejected raw_fn above; scripted_fn is name-only.
    return nodes_mod.ToolNode(
        name=node.name,
        inputs=inputs or None,
        outputs=outputs or None,
        tool=_make_server_tool(node, tools_mod, inputs, outputs),
    )


def _make_llm_config(node: Node) -> Any:
    _nodes_mod, _flow_mod, _edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
    from pyagentspec.llms.llmconfig import LlmConfig as SpecLlmConfig

    return SpecLlmConfig(name=f"{node.name}-llm", model_id=node.model or "default")


def _make_server_tool(node: Node, tools_mod: Any, inputs: list[Property], outputs: list[Property]) -> Any:
    return tools_mod.ServerTool(
        name=node.scripted_fn or node.name,
        description=f"neograph node {node.name!r} (mode={node.mode})",
        inputs=inputs or None,
        outputs=outputs or None,
    )


def _lower_oracle(node: Node, oracle: Oracle) -> tuple[list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge]]:
    """Lower an Oracle-modified node: N single-LlmNode flows + merge node.

    Oracle is the flagship irreversible gap — no single Agent Spec node
    represents it. Lowers to a ``ParallelFlowNode`` of N single-node flows
    (one ``LlmConfig`` per ``Oracle.models`` entry, or N copies) + a merge
    node, stamped with the full ``neograph/modifier=oracle`` marker (incl.
    ``models``, which has no primitive representation).
    """
    nodes_mod, flow_mod, edges_mod, _property_mod, tools_mod = _import_agent_spec_flow_classes()

    if oracle.merge_pre_process or oracle.merge_post_process or oracle.merge_fallback:
        raise ConfigurationError.build(
            f"node {node.name!r}'s Oracle uses merge_pre_process/merge_post_process/merge_fallback "
            "— Python callables with no Agent Spec representation",
            expected="Oracle without merge hooks",
            found="one or more merge hooks set",
            hint="Oracle merge hooks cannot be exported to Agent Spec (callable-valued field, doc s6)",
        )

    group_id = f"{node.name}__oracle"
    variant_models = oracle.models if oracle.models else [node.model] * oracle.n
    inputs = _properties_for(node.inputs)
    gen_outputs = _properties_for(node.oracle_gen_type) if node.oracle_gen_type else _properties_for(node.outputs)

    variant_nodes: list[SpecNode] = []
    for i, model_tier in enumerate(variant_models):
        variant_nodes.append(
            nodes_mod.LlmNode(
                name=f"{node.name}__variant_{i}",
                inputs=inputs or None,
                outputs=gen_outputs or None,
                llm_config=_make_llm_config(Node(name=node.name, model=model_tier or node.model)),
                prompt_template=node.prompt or "",
                metadata={"neograph/modifier": "oracle", "neograph/group_id": group_id, "neograph/variant": i},
            )
        )

    outputs = _properties_for(node.outputs)
    if oracle.merge_prompt:
        merge_node = nodes_mod.LlmNode(
            name=f"{node.name}",
            inputs=gen_outputs or None,
            outputs=outputs or None,
            llm_config=_make_llm_config(Node(name=node.name, model=oracle.merge_model)),
            prompt_template=oracle.merge_prompt,
            metadata={
                "neograph/modifier": "oracle",
                "neograph/group_id": group_id,
                "neograph/oracle_spec": {
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
                "neograph/modifier": "oracle",
                "neograph/group_id": group_id,
                "neograph/oracle_spec": {
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
            data_edges.append(
                edges_mod.DataFlowEdge(
                    name=f"{group_id}_fanin_{i}_{prop.title}",
                    source_node=variant,
                    source_output=prop.title,
                    destination_node=merge_node,
                    destination_input=prop.title,
                )
            )

    return [*variant_nodes, merge_node], control_edges, data_edges


def _lower_each(node: Node, each: Each) -> SpecNode:
    """Lower an Each-modified node: MapNode wrapping a single-node sub-Flow.

    ``over``/``key``/``on_error`` have no primitive representation — ride in
    the ``neograph/modifier=each`` marker (``EachSpec``).
    """
    nodes_mod, flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()

    inner = _lower_node(node)
    start_node = nodes_mod.StartNode(name=f"{node.name}__each_start")
    end_node = nodes_mod.EndNode(name=f"{node.name}__each_end")
    sub_flow = flow_mod.Flow(
        name=f"{node.name}__each_body",
        start_node=start_node,
        nodes=[start_node, inner, end_node],
        control_flow_connections=[
            edges_mod.ControlFlowEdge(name=f"{node.name}__each_start_edge", from_node=start_node, to_node=inner),
            edges_mod.ControlFlowEdge(name=f"{node.name}__each_end_edge", from_node=inner, to_node=end_node),
        ],
    )
    return nodes_mod.MapNode(
        name=node.name,
        subflow=sub_flow,
        metadata={
            "neograph/modifier": "each",
            "neograph/each_spec": {"over": each.over, "key": each.key, "on_error": each.on_error},
        },
    )


def _lower_loop(node: Node, loop: Loop, body: SpecNode) -> tuple[SpecNode, list[ControlFlowEdge], list[DataFlowEdge]]:
    """Lower a Loop-modified node: BranchingNode({continue: back-edge, done: next}).

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
            "neograph/modifier": "loop",
            "neograph/loop_spec": {
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
    data_edges: list[DataFlowEdge] = []
    for prop in _properties_for(node.outputs):
        data_edges.append(
            edges_mod.DataFlowEdge(
                name=f"{node.name}__loop_self_{prop.title}",
                source_node=body,
                source_output=prop.title,
                destination_node=body,
                destination_input=prop.title,
            )
        )
    return branch, control_edges, data_edges


def _lower_operator(node: Node, operator: Operator) -> tuple[SpecNode, list[SpecNode], list[ControlFlowEdge]]:
    """Lower an Operator-modified node: the FULLY PINNED HITL-pause composite
    (neograph-03djs, verified against real pyagentspec 26.1.2 source).

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
        metadata={"neograph/modifier": "operator", "neograph/operator_spec": {"when": operator.when}},
    )
    input_message = nodes_mod.InputMessageNode(
        name=f"{node.name}__operator_pause",
        outputs=[property_mod.StringProperty(title="user_input")],
    )
    pause_edge = edges_mod.ControlFlowEdge(
        name=f"{node.name}__operator_to_pause", from_node=check, from_branch=_PAUSE_BRANCH, to_node=input_message
    )
    return check, [input_message], [pause_edge]


def _lower_construct_item(item: Any) -> tuple[list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge], SpecNode]:
    """Lower one top-level construct item (Node/Construct/_BranchNode) to
    (all_spec_nodes, extra_control_edges, extra_data_edges, primary_node).

    ``primary_node`` is the node other items' ControlFlowEdges attach to
    (the item's DX-visible identity — e.g. an Operator's check node, or an
    Oracle's merge node).
    """
    nodes_mod, flow_mod, _edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()

    if isinstance(item, _BranchNode):
        branch = nodes_mod.BranchingNode(
            name=item.name,
            mapping={"true": "true", "false": "false"},
            metadata={"neograph/branch": True},
        )
        return [branch], [], [], branch

    if isinstance(item, Construct):
        sub_flow = to_agent_spec(item)
        flow_node = nodes_mod.FlowNode(name=item.name, subflow=sub_flow)
        return [flow_node], [], [], flow_node

    if not isinstance(item, Node):
        raise ConfigurationError.build(
            f"unrecognized construct item {item!r} — no Agent Spec lowering",
            expected="Node, Construct, or _BranchNode",
            found=type(item).__name__,
        )

    combo, mods = classify_modifiers(item)

    if combo == ModifierCombo.ORACLE:
        variant_and_merge, control_edges, data_edges = _lower_oracle(item, mods["oracle"])
        return variant_and_merge, control_edges, data_edges, variant_and_merge[-1]

    if combo == ModifierCombo.EACH:
        map_node = _lower_each(item, mods["each"])
        return [map_node], [], [], map_node

    if combo == ModifierCombo.LOOP:
        body = _lower_node(item)
        branch, extra_control, extra_data = _lower_loop(item, mods["loop"], body)
        return [body, branch], extra_control, extra_data, branch

    if combo == ModifierCombo.OPERATOR:
        _nodes_mod, _flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
        primary = _lower_node(item)
        check, extra_nodes, extra_control = _lower_operator(item, mods["operator"])
        pre_edge = edges_mod.ControlFlowEdge(name=f"{item.name}__to_operator_check", from_node=primary, to_node=check)
        return [primary, check, *extra_nodes], [pre_edge, *extra_control], [], check

    if combo == ModifierCombo.BARE:
        primary = _lower_node(item)
        return [primary], [], [], primary

    raise ConfigurationError.build(
        f"node {item.name!r} has modifier combination {combo.name} — no Agent Spec lowering yet",
        expected="BARE, ORACLE, EACH, LOOP, or OPERATOR",
        found=combo.name,
        hint="composed modifier lowering (e.g. Each+Oracle) is out of scope for i3zsh's primitive-level export",
    )


def to_agent_spec(construct: Construct) -> Flow:
    """Export a neograph ``Construct`` (IR) to an Open Agent Spec ``Flow``.

    LOWERS every modifier to flat Agent Spec primitives — the same lowering
    neograph performs when compiling to LangGraph, expressed in Agent Spec
    vocabulary. Fails loud (``ConfigurationError``) on any construct it
    cannot represent, rather than silently downgrading. See module
    docstring for the Core Invariant.
    """
    _nodes_mod, flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()

    all_nodes: list[SpecNode] = []
    control_edges: list[ControlFlowEdge] = []
    data_edges: list[DataFlowEdge] = []
    primaries: list[SpecNode] = []
    item_by_name: dict[str, Any] = {}

    for item in iter_with_arms(construct):
        item_by_name[item.name] = item
        lowered_nodes, extra_control, extra_data, primary = _lower_construct_item(item)
        all_nodes.extend(lowered_nodes)
        control_edges.extend(extra_control)
        data_edges.extend(extra_data)
        primaries.append(primary)

    # Explicit ControlFlowEdge per adjacent pair in Construct.nodes order.
    for prev_primary, next_primary in zip(primaries, primaries[1:], strict=False):
        control_edges.append(
            edges_mod.ControlFlowEdge(
                name=f"{prev_primary.name}_to_{next_primary.name}",
                from_node=prev_primary,
                to_node=next_primary,
            )
        )

    # Explicit DataFlowEdge per Node.inputs upstream-name mapping.
    ordered_items = list(iter_with_arms(construct))
    primary_by_item_name = dict(zip((item.name for item in ordered_items), primaries, strict=True))
    for idx, item in enumerate(ordered_items):
        if not isinstance(item, Node):
            continue
        ni = normalize_inputs(item.inputs)
        if ni.is_none:
            continue
        dest_primary = primary_by_item_name[item.name]

        if ni.is_dict_form:
            # Dict-form fan-in: named upstream -> Property title per key.
            for upstream_name in ni.by_name:
                source_primary = primary_by_item_name.get(upstream_name)
                if source_primary is None:
                    continue
                data_edges.append(
                    edges_mod.DataFlowEdge(
                        name=f"{upstream_name}_to_{item.name}",
                        source_node=source_primary,
                        source_output=upstream_name,
                        destination_node=dest_primary,
                        destination_input=upstream_name,
                    )
                )
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
            if no.is_none or no.is_dict_form:
                continue
            if not (issubclass(no.primary, ni.single_type) or issubclass(ni.single_type, no.primary)):
                continue
            source_primary = primary_by_item_name[upstream.name]
            upstream_props = {p.title for p in _properties_for(no.primary)}
            for shared_title in input_props & upstream_props:
                data_edges.append(
                    edges_mod.DataFlowEdge(
                        name=f"{upstream.name}_to_{item.name}_{shared_title}",
                        source_node=source_primary,
                        source_output=shared_title,
                        destination_node=dest_primary,
                        destination_input=shared_title,
                    )
                )
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
    start_node = _nodes_mod.StartNode(name=f"{construct.name}__start")
    end_node = _nodes_mod.EndNode(name=f"{construct.name}__end")
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
