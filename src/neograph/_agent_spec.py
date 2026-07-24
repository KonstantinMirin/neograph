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

from typing import TYPE_CHECKING, Any, cast

from neograph._ir_branch import _BranchNode, iter_with_arms
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import Each, Loop, ModifierCombo, Operator, Oracle, classify_modifiers
from neograph.naming import field_name_for
from neograph.node import Node
from neograph.spec_types import model_to_agent_spec_properties
from neograph.tool import Tool

if TYPE_CHECKING:
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.flow import Flow
    from pyagentspec.flows.node import Node as SpecNode
    from pyagentspec.property import Property

__all__ = ["to_agent_spec"]

_DEFAULT_BRANCH = "default"
_PAUSE_BRANCH = "pause"

# ── Agent-Spec metadata marker keys (neograph-aa5gq Step 0) ──────────────────
# The SINGLE source of truth for every ``neograph/*`` metadata marker key. Both
# the export side (this module) and the import side (``loader.py``, which imports
# these) reference these named constants — NEVER a re-inlined string literal — so
# a typo cannot silently split the export<->import contract and downgrade a
# marker-bearing primitive to the fail-loud/foreign path. Pinned (no re-inlined
# literals + exact wire-value asserts) by tests/test_guards_agent_spec_markers.py.
_MARK_MODE = "neograph/mode"
_MARK_AGENT_SPEC = "neograph/agent_spec"
_MARK_TOOL_SPEC = "neograph/tool_spec"
_MARK_REMOTE_AGENT = "neograph/remote_agent"
_MARK_MODIFIER = "neograph/modifier"
_MARK_GROUP_ID = "neograph/group_id"
_MARK_VARIANT = "neograph/variant"
_MARK_ORACLE_SPEC = "neograph/oracle_spec"
_MARK_EACH_SPEC = "neograph/each_spec"
_MARK_LOOP_SPEC = "neograph/loop_spec"
_MARK_OPERATOR_SPEC = "neograph/operator_spec"
_MARK_BRANCH = "neograph/branch"
_MARK_PORTAL_SPEC = "neograph/portal_spec"


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
        # Lossless lowering (neograph-i3zsh.1): a real pyagentspec
        # AgentNode+Agent+ServerTool composite, never the ToolNode placeholder
        # that used to silently drop prompt/model/tools. The `neograph/agent_spec`
        # marker carries everything the from_agent_spec() importer needs to
        # reconstruct the node exactly -- the export->import round trip is now
        # implemented (neograph-aa5gq, loader._reconstruct_agent_node).
        agent = _make_agent(node, tools_mod, inputs, outputs)
        return nodes_mod.AgentNode(
            name=node.name,
            inputs=inputs or None,
            outputs=outputs or None,
            agent=agent,
            metadata={_MARK_MODE: node.mode, _MARK_AGENT_SPEC: _agent_spec_marker(node)},
        )

    # scripted / raw already rejected raw_fn above; scripted_fn is name-only.
    return nodes_mod.ToolNode(
        name=node.name,
        inputs=inputs or None,
        outputs=outputs or None,
        tool=_make_server_tool(node, tools_mod, inputs, outputs),
    )


def _make_agent(node: Node, tools_mod: Any, inputs: list[Property], outputs: list[Property]) -> Any:
    from pyagentspec.agent import Agent

    # node.tools is declared list[Tool | BaseTool], but _normalize_raw_base_tools
    # (node.py) normalizes any raw BaseTool to Tool at construction time -- same
    # cast precedent as _agent_cycle.py:235.
    tools = cast("list[Tool]", node.tools)
    return Agent(
        name=f"{node.name}-agent",
        llm_config=_make_llm_config(node),
        system_prompt=node.prompt or "",
        tools=[_tool_to_server_tool(tool, tools_mod) for tool in tools],
        inputs=inputs or None,
        outputs=outputs or None,
        human_in_the_loop=False,
    )


def _tool_to_server_tool(tool: Any, tools_mod: Any) -> Any:
    """Lower one neograph ``Tool`` to a ``ServerTool``, name-only.

    Mirrors ``_make_server_tool``'s ``ServerTool`` shape but is a standalone
    helper: this is an agent/act node's ``tools=[...]`` list attaching to its
    ``Agent``, not a scripted/think node's own ``ToolNode.tool=`` field (a
    different Agent Spec primitive entirely). ``ServerTool`` is used
    UNIFORMLY for every neograph ``Tool`` export (MCP-bound or not) --
    MCP-ness is a runtime factory-registration detail
    (``mcp_tool_factory``), never a wire-format distinction (pyagentspec has
    no ``MCPTool`` class; doc s7/s8's 'do not own the MCP gateway'
    positioning).

    Stamps a ``neograph/tool_spec`` marker on the ServerTool itself
    (budget/config/idempotent -- name-only fields with no plain-``ServerTool``
    equivalent), mirroring ``ToolSpec``'s shape (``_spec_schema.py:66``) and
    the established ``metadata['neograph/*_spec']`` marker convention. Per
    the Core Invariant, ``tool._bound_tool`` (a live callable) is NEVER
    referenced here -- factory binding is exclusively a runtime,
    post-deserialization concern.
    """
    return tools_mod.ServerTool(
        name=tool.name,
        description=f"neograph tool {tool.name!r}",
        inputs=None,
        outputs=None,
        metadata={
            _MARK_TOOL_SPEC: {
                "name": tool.name,
                "budget": tool.budget,
                "config": tool.config,
                "idempotent": tool.idempotent,
            }
        },
    )


def _agent_spec_marker(node: Node) -> dict[str, Any]:
    """Build the ``neograph/agent_spec`` reconstruction blob for an agent/act
    node — every field the plain ``Agent``/``ServerTool`` primitives cannot
    represent, so a future ``from_agent_spec()`` importer can rebuild the
    exact node. Callable ``gate_tools_when`` is already rejected by
    ``_reject_unrepresentable_fields`` before this runs; only the string form
    reaches here.
    """
    # node.tools is declared list[Tool | BaseTool], but _normalize_raw_base_tools
    # (node.py) normalizes any raw BaseTool to Tool at construction time -- same
    # cast precedent as _agent_cycle.py:235.
    tools = cast(list[Tool], node.tools)
    return {
        "mode": node.mode,
        "prompt": node.prompt,
        "model": node.model,
        "tools": [
            {"name": tool.name, "budget": tool.budget, "config": tool.config, "idempotent": tool.idempotent}
            for tool in tools
        ],
        "gate_tools_when": node.gate_tools_when,
        "context": node.context,
    }


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
                metadata={_MARK_MODIFIER: "oracle", _MARK_GROUP_ID: group_id, _MARK_VARIANT: i},
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
    # The MapNode infers its OWN inputs as ``iterated_{title}`` for every
    # property in ``subflow.inputs`` (pyagentspec MapNode._get_inferred_inputs,
    # which reads the sub-flow's StartNode inputs). Declare the inner node's
    # input Properties on the StartNode so a NON-fan-out context input (e.g.
    # ``verify(source: RawText, cluster: Elem)`` with ``map_over``) has a valid
    # ``iterated_source.text`` destination for its top-level DataFlowEdge — the
    # fan-out-receiver-only case stays valid too (its inferred input is simply
    # left unconnected, populated per-item from the iterated collection).
    # neograph-hf505.
    inner_inputs = _properties_for(node.inputs)
    start_node = nodes_mod.StartNode(name=f"{node.name}__each_start", inputs=inner_inputs or None)
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
            _MARK_MODIFIER: "each",
            _MARK_EACH_SPEC: {"over": each.over, "key": each.key, "on_error": each.on_error},
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
    # Dict-form inputs prefix each Property title as "{upstream}.{field}"
    # (per _properties_for's dict-form convention) -- the body node's real
    # input Property is "{key}.{field}", never the bare "{field}", so the
    # self-edge's destination_input must be resolved against the SAME key
    # the runtime feeds the re-entry value into. That key is whichever
    # dict-form inputs entry has a type compatible with the node's own
    # output type (mirrors the single-type upstream-resolution scan below:
    # a Loop-fed key could be a self-reference — "key matching the node's
    # own name" per the validator's Loop rule — OR the ORIGINAL upstream
    # producer's name, e.g. inputs={'seed': Draft} — either way it's the
    # key whose declared type matches the fed-back output).
    ni = normalize_inputs(node.inputs)
    no_self = normalize_outputs(node.outputs)
    dest_prefix = ""
    if ni.is_dict_form and not no_self.is_dict_form:
        self_field = field_name_for(node.name)
        if self_field in ni.by_name:
            dest_prefix = f"{self_field}."
        else:
            for key, typ in ni.by_name.items():
                if isinstance(typ, type) and (issubclass(no_self.primary, typ) or issubclass(typ, no_self.primary)):
                    dest_prefix = f"{key}."
                    break

    data_edges: list[DataFlowEdge] = []
    for prop in _properties_for(node.outputs):
        data_edges.append(
            edges_mod.DataFlowEdge(
                name=f"{node.name}__loop_self_{prop.title}",
                source_node=body,
                source_output=prop.title,
                destination_node=body,
                destination_input=f"{dest_prefix}{prop.title}",
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


def _lower_construct_item(
    item: Any,
) -> tuple[list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge], SpecNode, SpecNode, list[tuple[SpecNode, bool]]]:
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

    if isinstance(item, Construct):
        sub_flow = to_agent_spec(item)
        flow_node = nodes_mod.FlowNode(name=item.name, subflow=sub_flow)
        return [flow_node], [], [], flow_node, flow_node, [(flow_node, False)]

    if not isinstance(item, Node):
        raise ConfigurationError.build(
            f"unrecognized construct item {item!r} — no Agent Spec lowering",
            expected="Node, Construct, or _BranchNode",
            found=type(item).__name__,
        )

    combo, mods = classify_modifiers(item)

    if combo == ModifierCombo.ORACLE:
        variant_and_merge, control_edges, data_edges = _lower_oracle(item, mods["oracle"])
        variants = variant_and_merge[:-1]
        merge = variant_and_merge[-1]
        return variant_and_merge, control_edges, data_edges, merge, merge, [(v, False) for v in variants]

    if combo == ModifierCombo.EACH:
        map_node = _lower_each(item, mods["each"])
        return [map_node], [], [], map_node, map_node, [(map_node, True)]

    if combo == ModifierCombo.LOOP:
        body = _lower_node(item)
        branch, extra_control, extra_data = _lower_loop(item, mods["loop"], body)
        return [body, branch], extra_control, extra_data, branch, body, [(body, False)]

    if combo == ModifierCombo.OPERATOR:
        _nodes_mod, _flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
        primary = _lower_node(item)
        check, extra_nodes, extra_control = _lower_operator(item, mods["operator"])
        pre_edge = edges_mod.ControlFlowEdge(name=f"{item.name}__to_operator_check", from_node=primary, to_node=check)
        return [primary, check, *extra_nodes], [pre_edge, *extra_control], [], check, primary, [(primary, False)]

    if combo == ModifierCombo.BARE:
        primary = _lower_node(item)
        return [primary], [], [], primary, primary, [(primary, False)]

    raise ConfigurationError.build(
        f"node {item.name!r} has modifier combination {combo.name} — no Agent Spec lowering yet",
        expected="BARE, ORACLE, EACH, LOOP, or OPERATOR",
        found=combo.name,
        hint="composed modifier lowering (e.g. Each+Oracle) is out of scope for i3zsh's primitive-level export",
    )


def _lower_portal_mesh_to_swarm(construct: Construct, members: list[Node], tools_mod: Any) -> Any:
    """Export a Portal mode-(a) peer mesh to a top-level pyagentspec ``Swarm``
    -- the export-direction mirror of ``loader.py``'s ``_reconstruct_swarm_mesh``
    Swarm import.

    Swarm.first_agent/relationships are typed ``AgenticComponent`` (pyagentspec
    swarm.py/agent.py), so each member lowers to a real ``Agent`` (via the
    SAME ``_make_agent`` helper agent/act-mode Flow nodes use), never an
    ``LlmNode``. The entry-only knobs (``max_hops``/``on_exhaust``/``route``)
    have no native ``Swarm`` field -- they ride a ``neograph/portal_spec``
    metadata marker (mirrors the Oracle/Each/Loop per-group marker
    convention), so the information is not lost even though the current
    Swarm importer does not read it back yet.

    ``construct.nodes`` is trusted here: ``_check_portal_mesh`` (construct-
    assembly validation) has ALREADY enforced contiguity/entry-first/uniform-
    payload/reachability for any Construct reaching export, so every ``to``
    peer reference is guaranteed to name a real member of this same mesh.
    """
    entry = members[0]
    entry_portal = entry.modifier_set.portal
    assert entry_portal is not None  # collected as Portal-modified

    # pyagentspec's Agent ties inputs/outputs Properties to {{placeholder}}
    # names in its own system_prompt (ComponentWithIO._validate_no_extra_
    # property) -- a mesh member's payload Properties (e.g. dict-form-
    # prefixed "handoff.note") are not prompt placeholders, so passing them
    # unconditionally raises "did not expect any properties" for any member
    # without a matching prompt template. Mesh Agents carry no I/O Properties;
    # the payload/routing shape rides the neograph/portal_spec marker instead.
    agents_by_name: dict[str, Any] = {}
    for member in members:
        agents_by_name[member.name] = _make_agent(member, tools_mod, [], [])

    relationships = [
        (agents_by_name[member.name], agents_by_name[peer])
        for member in members
        for peer in (member.modifier_set.portal.to or [])  # type: ignore[union-attr]
    ]

    from pyagentspec.swarm import Swarm

    return Swarm(
        name=construct.name,
        first_agent=agents_by_name[entry.name],
        relationships=relationships,
        metadata={
            _MARK_PORTAL_SPEC: {
                "max_hops": entry_portal.max_hops,
                "on_exhaust": entry_portal.on_exhaust,
                "route": entry_portal.route,
            }
        },
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
    mesh_members = [
        item
        for item in all_items
        if isinstance(item, Node)
        and item.modifier_set.portal is not None
        and not item.modifier_set.portal.is_dispatch
    ]
    if mesh_members:
        if len(mesh_members) != len(all_items):
            raise ConfigurationError.build(
                f"construct {construct.name!r} mixes a Portal peer mesh with non-mesh nodes",
                expected="a construct that is EITHER entirely a Portal mesh OR has no Portal mesh members",
                found=f"{len(mesh_members)} mesh member(s) out of {len(all_items)} total node(s)",
                hint="a Swarm is a top-level AgenticComponent, not a Flow node — a mixed "
                "mesh+Flow construct has no single Agent Spec export shape yet",
            )
        return _lower_portal_mesh_to_swarm(construct, mesh_members, tools_mod)

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

    def _emit_input_edges(item_name: str, upstream_name: str, source_node: SpecNode, source_title: str) -> None:
        """Emit one DataFlowEdge per (destination target, prefix) for a single
        source Property. ``upstream_name`` is the dict-form key ('' for the
        single-type path, where the destination input title is the bare
        Property title, not '{upstream}.{title}')."""
        for target_node, iterated in input_targets_by_item_name[item_name]:
            if iterated:
                # A MapNode infers its inputs as ``iterated_{json_schema title}``
                # — and pyagentspec forbids dots in json_schema titles, so the
                # inner node's dict-form ``{key}.{field}`` prefix lives only on
                # Property.title; the inferred MapNode input is the BARE
                # ``iterated_{field}``. Target that, not the dotted form.
                dest_input = f"iterated_{source_title}"
            elif upstream_name:
                dest_input = f"{upstream_name}.{source_title}"
            else:
                dest_input = source_title
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
                if no.is_none or no.is_dict_form:
                    raise ConfigurationError.build(
                        f"node {item.name!r}'s dict-form inputs references upstream "
                        f"{upstream_name!r}, whose outputs are not a single exportable type",
                        expected="a single-type Node.outputs on the upstream node",
                        found=f"{upstream_name!r}.outputs is dict-form or None",
                        hint="multi-output (dict-form outputs) producers referenced by a "
                        "downstream dict-form input have no Agent Spec representation yet",
                    )
                for prop in _properties_for(no.primary):
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
            if no.is_none or no.is_dict_form:
                continue
            if not (issubclass(no.primary, ni.single_type) or issubclass(ni.single_type, no.primary)):
                continue
            source_node = data_node_by_item_name[upstream.name]
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
