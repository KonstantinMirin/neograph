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
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from neograph._ir_branch import iter_with_arms
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.naming import field_name_for, split_output_field
from neograph.node import Node

if TYPE_CHECKING:
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.flow import Flow
    from pyagentspec.flows.node import Node as SpecNode

__all__ = ["to_agent_spec"]

# --- extracted clusters (neograph-3ffdg.3), re-exported so this module's public import surface is unchanged.
from neograph._agent_spec_boundary import resolve_end_node_sources  # noqa: E402
from neograph._agent_spec_each_fanout import each_fanout_edge_source  # noqa: E402
from neograph._agent_spec_item_dispatch import _lower_construct_item  # noqa: E402
from neograph._agent_spec_lowering_types import _LoweredItem  # noqa: E402,F401
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
from neograph._agent_spec_portal import (  # noqa: E402,F401
    _is_peer_mesh_member,
    _lower_portal_mesh_to_swarm,
)
from neograph._agent_spec_provider import ApiProviderResolver, build_resolver  # noqa: E402

_Exit: TypeAlias = "tuple[SpecNode, str | None]"
"""One outgoing control-flow endpoint of a lowered item: the SpecNode a
successor edge leaves FROM, plus the ``from_branch`` it must name (``None`` for
an unconditional node). An item has several only when its lowering genuinely
forks — an Operator reconverges its gate's DEFAULT_BRANCH and its post-pause
continuation on the same successor."""


def to_agent_spec(construct: Construct, api_provider: str | None = None, *, llm_factory: Any = None) -> Flow:
    """Export a neograph ``Construct`` (IR) to an Open Agent Spec ``Flow``
    (or, for a Portal mode-(a) peer mesh, a top-level ``Swarm``).

    LOWERS every modifier to flat Agent Spec primitives — the same lowering
    neograph performs when compiling to LangGraph, expressed in Agent Spec
    vocabulary. Fails loud (``ConfigurationError``) on the ENUMERATED fields
    it cannot represent (module docstring) — not raising does NOT mean the
    result is portable; call ``export_conformance(construct)`` for that.

    ``llm_factory`` (qtfof.13, the same factory ``compile()``/``lint()``/
    ``Node.run_isolated()`` take) resolves each node's opaque tier PER NODE and
    exports the real provider/model/url; ``api_provider`` (qtfof.8) overrides it
    and stays the only route for an unclassifiable client. Precedence and the
    one-row provider table live in ``_agent_spec_provider``. With NEITHER
    argument the export keeps its ``NEOGRAPH_ROUND_TRIP_ONLY`` shape, byte
    for byte.
    """
    return _to_agent_spec_with(construct, provider=build_resolver(api_provider, llm_factory))


def _to_agent_spec_with(construct: Construct, *, provider: ApiProviderResolver) -> Flow:
    """Recursion entry point: exports against an ALREADY-BUILT resolver, so every
    sub-construct, Swarm member and Each body shares one memoised tier cache."""
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
            functools.partial(_to_agent_spec_with, provider=provider),
            provider,
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
        lowered = _lower_top_level_item(
            item,
            functools.partial(
                _lower_construct_item,
                provider=provider,
                flow_export=functools.partial(_to_agent_spec_with, provider=provider),
            ),
        )
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
                    # neograph-qtfof.7: a real DataFlowEdge when Each.over
                    # resolves to a single-segment path AND the edge is SAFE
                    # (each_fanout_edge_source's own scope boundary + the
                    # json_schemas_have_same_type pre-check) -- else stays
                    # metadata-only, mirroring _validation_inputs.py's
                    # fan_out_param skip.
                    each_mod = item.modifier_set.each
                    resolved = (
                        each_fanout_edge_source(
                            each_mod.over,
                            fan_out_key,
                            ni.by_name[upstream_name],
                            item_by_name,
                            data_node_by_item_name,
                            _properties_for,
                        )
                        if each_mod is not None
                        else None
                    )
                    if resolved is not None:
                        _emit_input_edges(item.name, "", resolved[0], resolved[1], dest_title=fan_out_key)
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
