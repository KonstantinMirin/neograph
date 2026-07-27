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

from typing import TYPE_CHECKING, Any, TypeGuard, cast

from neograph._ir_branch import _BranchNode, iter_with_arms
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._placeholders import DOLLAR_RE, apply_scanner
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.modifiers import (
    SUB_CONSTRUCT_UNSUPPORTED_COMBOS,
    Each,
    Loop,
    ModifierCombo,
    Operator,
    Oracle,
    classify_modifiers,
)
from neograph.naming import field_name_for, split_output_field
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
_MARK_MODIFIER = "neograph/modifier"
_MARK_GROUP_ID = "neograph/group_id"
_MARK_VARIANT = "neograph/variant"
_MARK_ORACLE_SPEC = "neograph/oracle_spec"
_MARK_EACH_SPEC = "neograph/each_spec"
_MARK_LOOP_SPEC = "neograph/loop_spec"
_MARK_OPERATOR_SPEC = "neograph/operator_spec"
_MARK_BRANCH = "neograph/branch"
_MARK_PORTAL_SPEC = "neograph/portal_spec"
_MARK_PORTAL_OPERATOR_SPEC = "neograph/portal_operator_spec"
_MARK_PROMPT_SPEC = "neograph/prompt_spec"


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


def _translate_placeholders(
    prompt_text: str, input_props: list[Property], node_name: str
) -> tuple[str, list[Property], dict[str, str]]:
    """Translate neograph ``${path}`` placeholders to pyagentspec ``{{ flat }}``
    form (Option F, neograph-cbpyx — amends m57mn's Option-B fail-loud guard).

    neograph's ``${var}``/``${var.field}`` and pyagentspec's ``{{ var }}`` are two
    syntaxes for the IDENTICAL flat, non-recursive text substitution. This rewrites
    each ``${path}`` to ``{{ path_with_dots_as_underscores }}`` and returns the
    Properties the exported ``LlmNode``/``Agent`` should declare — exactly the
    scanned names, so pyagentspec's own placeholder inference/validation
    (``ComponentWithIO._get_inferred_inputs`` / ``_validate_inputs``) passes by
    construction. REUSES the ONE ``${...}`` scanner (``_placeholders.DOLLAR_RE`` +
    ``apply_scanner``) — never a second grammar (the anti-duplication invariant).

    Returns ``(rewritten_text, referenced_props, flat_to_original)``:
      * ``rewritten_text`` — prompt with every ``${path}`` -> ``{{ flat }}``.
      * ``referenced_props`` — one ``StringProperty(title=flat)`` per unique scanned
        path (names only; pyagentspec infers inputs by NAME, and round-trip type
        fidelity rides the ``neograph/prompt_spec`` marker, not these props).
      * ``flat_to_original`` — ``{flat_name: original_dotted_path}``, consumed by the
        input-edge / StartNode consumer sweep to route ``destination_input`` through
        the SAME flat name (drop an edge whose source path is unreferenced).

    Fail loud (``ConfigurationError``) on a ``${path}`` whose first segment is not a
    declared input (dangling), and on two distinct paths flattening to one name
    (collision — names both paths AND the collided flat name).
    """
    from pyagentspec.property import StringProperty

    declared_keys = {p.title.split(".", 1)[0] for p in input_props}
    flat_to_original: dict[str, str] = {}
    ordered: list[str] = []

    def resolve(raw: str) -> str:
        path = raw.strip()
        first = path.split(".", 1)[0]
        if first not in declared_keys:
            raise ConfigurationError.build(
                f"node {node_name!r}'s prompt references ${{{path}}}, whose first segment "
                f"{first!r} is not a declared input",
                expected=f"a ${{...}} path rooted at one of the declared inputs {sorted(declared_keys)}",
                found=f"dangling placeholder ${{{path}}}",
                hint="every inline ${var} placeholder in an exported LLM-mode prompt must "
                "resolve to a declared Node.input (the value has no other data path).",
            )
        flat = path.replace(".", "_")
        prev = flat_to_original.get(flat)
        if prev is not None and prev != path:
            raise ConfigurationError.build(
                f"node {node_name!r}'s prompt has two distinct placeholders {prev!r} and "
                f"{path!r} that both flatten to the same name {flat!r}",
                expected="each ${path} to flatten (. -> _) to a unique placeholder name",
                found=f"collision: {prev!r} and {path!r} both -> {flat!r}",
                hint="rename one of the colliding upstream inputs/fields so the flattened "
                "pyagentspec placeholder names stay distinct.",
            )
        if flat not in flat_to_original:
            flat_to_original[flat] = path
            ordered.append(flat)
        return f"{{{{ {flat} }}}}"

    rewritten = apply_scanner(prompt_text, DOLLAR_RE, resolve)
    referenced_props: list[Property] = [StringProperty(title=flat) for flat in ordered]
    return rewritten, referenced_props, flat_to_original


def _node_translation(node: Node) -> tuple[str, list[Property], dict[str, str]]:
    """Recompute a node's placeholder translation (``rewritten_text``,
    ``referenced_props``, ``original_to_flat``) from its prompt + declared inputs.

    The SINGLE per-node translation seam every construction site AND every
    consumer (input edges, Loop self-edge, Oracle fan-in, Each StartNode)
    re-derives from — so ``destination_input`` names are computed by the ONE
    translator, never re-inferred per-symptom. Idempotent: the node was already
    translated during ``_lower_construct_item`` (any collision/dangling already
    raised there), so re-running here cannot introduce a new raise. Returns
    ``original_to_flat`` (dotted path -> flat name) — the inverse of
    ``_translate_placeholders``'s ``flat_to_original`` — for edge routing.
    """
    _, ref_props, flat_to_original = _translate_placeholders(
        node.prompt or "", _properties_for(node.inputs), node.name
    )
    original_to_flat = {path: flat for flat, path in flat_to_original.items()}
    return node.prompt or "", ref_props, original_to_flat


def _is_translation_eligible(item: Any) -> TypeGuard[Node]:
    """A construct item whose exported prompt is placeholder-translated: an
    LLM-mode (``think``/``agent``/``act``) ``Node``. Gates the consumer sweep on
    the CONSUMING ITEM's mode — NOT the destination SpecNode class (a MapNode
    wrapping a translated inner is still a translation target). Scripted/raw
    nodes have no ${var} prompt, so their edges keep the untranslated dotted form.

    A ``TypeGuard[Node]`` so the ``node`` arg of ``_lower_each``/``_lower_loop``
    (now ``Node | Construct``) narrows to ``Node`` inside the guarded branch that
    calls the Node-only ``_node_translation`` — a Construct item is never
    translation-eligible (it carries no ${var} prompt of its own).
    """
    return isinstance(item, Node) and item.mode in ("think", "agent", "act")


def _prompt_spec_marker(node: Node, flat_to_original: dict[str, str]) -> dict[str, Any]:
    """Build the strictly JSON-native ``neograph/prompt_spec`` round-trip marker.

    Carries the UNtranslated ``${var}`` text + the full original input TypeSpec so
    ``from_agent_spec`` reconstructs the exact original ``Node`` — including inputs
    the prompt never referenced (whose translated ``LlmNode`` drops both Property
    and DataFlowEdge, a real topology change). MUST stay JSON-native (str / dict /
    list only): ``p.json_schema`` is the plain JSON-Schema dict (NOT a live
    pyagentspec ``Property`` object, which would degrade to a dict across a
    JSON/YAML wire round trip and break the loader's un-flatten). The dotted
    ``title`` (``"{key}.{field}"``) is stored alongside so the loader can regroup
    by upstream key via the EXISTING ``_dict_form_inputs_from_props``.
    """
    return {
        "original_text": node.prompt or "",
        "placeholder_map": dict(flat_to_original),
        "original_inputs": [
            {"title": p.title, "json_schema": p.json_schema} for p in _properties_for(node.inputs)
        ],
    }


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


def _lower_generation_step(
    node: Node,
    *,
    name: str,
    outputs: list[Property],
    metadata: dict[str, Any],
    model_tier: str | None = None,
    tool_description: str | None = None,
) -> SpecNode:
    """The SINGLE per-node.mode generation dispatch (think / agent-act / scripted-raw).

    Shared by BOTH callers (neograph-2s2o6, retiring the two hand-written copies that
    were the 'one validator, not two' anti-pattern CLAUDE.md bans elsewhere):
    ``_lower_node`` passes ``name=node.name``/``metadata={}``; ``_lower_oracle``'s
    variant loop passes ``name=f'{node.name}__variant_{i}'``, the oracle group/variant
    markers as ``metadata``, and the per-variant ``model_tier`` (Oracle.models). A new
    mode branch or a fix now lands in ONE place for every caller -- if agent/act-mode
    Oracle variants get a lowering change, they get it for free, never a third copy.
    """
    nodes_mod, _flow_mod, _edges_mod, _property_mod, tools_mod = _import_agent_spec_flow_classes()

    inputs = _properties_for(node.inputs)

    if node.mode == "think":
        # Option F neograph-cbpyx: translate ${path} -> {{ flat }}; the LlmNode
        # declares ONLY the referenced flat Properties, and the neograph/prompt_spec
        # marker carries the untranslated text + full original inputs for round trip.
        rewritten, ref_props, flat_to_original = _translate_placeholders(node.prompt or "", inputs, name)
        return nodes_mod.LlmNode(
            name=name,
            inputs=ref_props or None,
            outputs=outputs or None,
            llm_config=_make_llm_config(Node(name=node.name, model=model_tier or node.model)),
            prompt_template=rewritten,
            metadata={**metadata, _MARK_PROMPT_SPEC: _prompt_spec_marker(node, flat_to_original)},
        )

    if node.mode in ("agent", "act"):
        # Lossless lowering (neograph-i3zsh.1): a real pyagentspec
        # AgentNode+Agent+ServerTool composite, never the ToolNode placeholder
        # that used to silently drop prompt/model/tools. The `neograph/agent_spec`
        # marker carries everything the from_agent_spec() importer needs to
        # reconstruct the node exactly -- the export->import round trip is now
        # implemented (neograph-aa5gq, loader._reconstruct_agent_node). Option F:
        # the Agent's system_prompt is placeholder-translated and the AgentNode
        # declares the referenced flat Properties; the original ${var} text rides
        # the existing neograph/agent_spec marker (marker["prompt"]). Per-variant
        # Oracle.models tier + a unique component name ride the model_copy
        # (agent_source); the neograph/agent_spec + prompt_spec markers keep the
        # ORIGINAL node (model=node.model), so _reconstruct_oracle_group recovers
        # base_node | Oracle(models=...).
        rewritten, ref_props, flat_to_original = _translate_placeholders(node.prompt or "", inputs, name)
        agent_source = node.model_copy(update={"name": name, "model": model_tier or node.model})
        agent = _make_agent(agent_source, tools_mod, ref_props, outputs, rewritten)
        return nodes_mod.AgentNode(
            name=name,
            inputs=ref_props or None,
            outputs=outputs or None,
            agent=agent,
            metadata={
                **metadata,
                _MARK_MODE: node.mode,
                _MARK_AGENT_SPEC: _agent_spec_marker(node),
                # The neograph/prompt_spec marker carries the FULL original inputs so
                # a translated agent/act node (whose declared inputs are the flat
                # placeholder names) reconstructs its true input TypeSpec -- e.g. an
                # Each fan-out receiver must round-trip to the SAME element type as
                # the producer's list element neograph-3lk2l. marker["prompt"] on
                # _MARK_AGENT_SPEC already carries the untranslated ${var} text.
                _MARK_PROMPT_SPEC: _prompt_spec_marker(node, flat_to_original),
            },
        )

    # scripted / raw already rejected raw_fn upstream; scripted_fn is name-only.
    return nodes_mod.ToolNode(
        name=name,
        inputs=inputs or None,
        outputs=outputs or None,
        tool=_make_server_tool(node, tools_mod, inputs, outputs, description=tool_description),
        metadata=metadata,
    )


def _lower_node(node: Node) -> SpecNode:
    """Dispatch a single neograph Node to its Agent Spec primitive by mode.

    Thin wrapper over the shared ``_lower_generation_step`` neograph-2s2o6: the
    per-mode dispatch lives in ONE place. ``_lower_node`` adds only the top-level
    ``_reject_unrepresentable_fields`` guard that the Oracle-variant path deliberately
    omits, and passes the node's own name + empty base metadata.
    """
    _reject_unrepresentable_fields(node)
    return _lower_generation_step(node, name=node.name, outputs=_properties_for(node.outputs), metadata={})


def _make_agent(
    node: Node, tools_mod: Any, inputs: list[Property], outputs: list[Property], system_prompt: str
) -> Any:
    """Build the pyagentspec ``Agent`` for an agent/act node. ``inputs`` are the
    Option-F-translated referenced flat Properties and ``system_prompt`` is the
    ``{{ flat }}``-rewritten text (both computed once in ``_lower_node``); the
    original ``${var}`` text rides ``neograph/agent_spec`` marker["prompt"]."""
    from pyagentspec.agent import Agent

    # node.tools is declared list[Tool | BaseTool], but _normalize_raw_base_tools
    # (node.py) normalizes any raw BaseTool to Tool at construction time -- same
    # cast precedent as _agent_cycle.py:235.
    tools = cast("list[Tool]", node.tools)
    return Agent(
        name=f"{node.name}-agent",
        llm_config=_make_llm_config(node),
        system_prompt=system_prompt,
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


def _make_server_tool(
    node: Node,
    tools_mod: Any,
    inputs: list[Property],
    outputs: list[Property],
    description: str | None = None,
) -> Any:
    return tools_mod.ServerTool(
        name=node.scripted_fn or node.name,
        description=description if description is not None else f"neograph node {node.name!r} (mode={node.mode})",
        inputs=inputs or None,
        outputs=outputs or None,
    )


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
            # A Construct variant is a copy of the sub-flow run N times (the
            # runtime shape make_oracle_redirect_fn produces over the subgraph);
            # each copy is a FlowNode over the recursively-exported sub-Flow.
            # Per-variant Oracle.models rides the oracle_spec marker, not a
            # FlowNode field (the sub-flow has no single model to swap here).
            variant_nodes.append(
                nodes_mod.FlowNode(
                    name=variant_name, subflow=to_agent_spec(node), metadata=variant_metadata
                )
            )
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
                dest_input = merge_orig_to_flat.get(prop.title)
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


def _lower_each(node: Node | Construct, each: Each) -> SpecNode:
    """Lower an Each-modified item: MapNode wrapping a single-body sub-Flow.

    The wrapped body is a Node's lowered primitive OR a Construct-item's
    ``FlowNode`` (``_lower_item_body``), so ``Construct(...) | Each(...)`` used
    as one item lowers to the SAME MapNode shape as ``node | Each(...)``.
    ``over``/``key``/``on_error`` have no primitive representation — ride in
    the ``neograph/modifier=each`` marker (``EachSpec``).
    """
    nodes_mod, flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()

    inner = _lower_item_body(node)
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
    ni = normalize_inputs(_item_inputs(node))
    no_self = normalize_outputs(_item_outputs(node))
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

    # Option F consumer sweep neograph-cbpyx: when the loop body is a
    # placeholder-translated LLM node, its declared inputs are flat ${var}->{{ flat }}
    # names, so the self-feedback edge's destination_input must route through the
    # body's flat map (drop it if the fed-back output isn't referenced in the prompt).
    body_orig_to_flat = _node_translation(node)[2] if _is_translation_eligible(node) else {}
    data_edges: list[DataFlowEdge] = []
    for prop in _properties_for(_item_outputs(node)):
        dotted = f"{dest_prefix}{prop.title}"
        if _is_translation_eligible(node):
            dest_input = body_orig_to_flat.get(dotted)
            if dest_input is None:
                continue
        else:
            dest_input = dotted
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

    if combo == ModifierCombo.ORACLE:
        variant_and_merge, control_edges, data_edges = _lower_oracle(item, mods["oracle"])
        variants = variant_and_merge[:-1]
        merge = variant_and_merge[-1]
        return variant_and_merge, control_edges, data_edges, merge, merge, [(v, False) for v in variants]

    if combo == ModifierCombo.EACH:
        map_node = _lower_each(item, mods["each"])
        return [map_node], [], [], map_node, map_node, [(map_node, True)]

    if combo == ModifierCombo.LOOP:
        body = _lower_item_body(item)
        branch, extra_control, extra_data = _lower_loop(item, mods["loop"], body)
        return [body, branch], extra_control, extra_data, branch, body, [(body, False)]

    if combo == ModifierCombo.OPERATOR:
        _nodes_mod, _flow_mod, edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
        primary = _lower_item_body(item)
        check, extra_nodes, extra_control = _lower_operator(item, mods["operator"])
        pre_edge = edges_mod.ControlFlowEdge(name=f"{item.name}__to_operator_check", from_node=primary, to_node=check)
        return [primary, check, *extra_nodes], [pre_edge, *extra_control], [], check, primary, [(primary, False)]

    if combo == ModifierCombo.BARE:
        primary = _lower_item_body(item)
        return [primary], [], [], primary, primary, [(primary, False)]

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
    if combo in (ModifierCombo.PORTAL, ModifierCombo.PORTAL_OPERATOR):
        raise ConfigurationError.build(
            f"node {item.name!r} is a dispatch-mode Portal (route='decide') — no Agent Spec lowering",
            expected="a peer-mode Portal mesh (exported as a Swarm) or a non-Portal node",
            found=f"dispatch-mode Portal ({combo.name})",
            node=item.name,
            hint="dispatch mode synthesizes and runs a flow from runtime-emitted data; Agent Spec has "
            "no runtime-flow-synthesis primitive (every subflow node takes a static subflow), so it is "
            "permanently unrepresentable — keep the dispatcher inside neograph, do not export it",
        )

    raise ConfigurationError.build(
        f"node {item.name!r} has modifier combination {combo.name} — no Agent Spec lowering yet",
        expected="BARE, ORACLE, EACH, LOOP, or OPERATOR",
        found=combo.name,
        hint="composed modifier lowering (e.g. Each+Oracle) is out of scope for i3zsh's primitive-level export",
    )


def _lower_portal_mesh_to_swarm(construct: Construct, members: list[Node | Construct], tools_mod: Any) -> Any:
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

    # pyagentspec's Agent ties inputs Properties to {{placeholder}} names in its
    # own system_prompt (ComponentWithIO._validate_no_extra_property), so a mesh
    # member's prompt is Option-F-translated exactly like every other _make_agent
    # caller (neograph-s7zt3.1): the Agent declares ONLY the referenced flat
    # Properties, which match the rewritten {{ flat }} names by construction. A
    # member may reference the reserved 'handoff' input (${handoff.field}) --
    # shipping it raw would hand a foreign Swarm runtime a placeholder it can
    # neither fill nor flag. Outputs stay [] -- the payload/routing shape rides
    # the neograph/portal_spec marker, and the untranslated ${var} text rides a
    # per-member neograph/prompt_spec marker on the Agent itself so the Swarm
    # import recovers the original prompt grammar.
    agents_by_name: dict[str, Any] = {}
    for member in members:
        # C1 (do0d9): a Construct mesh member lowers to its recursively-exported
        # sub-Flow. A pyagentspec Flow IS an AgenticComponent (Flow.__mro__
        # includes AgenticComponent), so it drops straight into
        # Swarm.first_agent/relationships with NO Agent wrapper — the same
        # recursive-Flow-production pattern _lower_item_body uses for a bare
        # Construct item's FlowNode subflow. A Construct has no .prompt (its
        # interior prompts are Option-F-translated inside the recursive
        # to_agent_spec call) and cannot carry an Operator gate (rejected at
        # assembly), so it never needs the prompt marker or enters `gated`.
        if isinstance(member, Construct):
            agents_by_name[member.name] = to_agent_spec(member)
            continue
        rewritten, ref_props, flat_to_original = _translate_placeholders(
            member.prompt or "", _properties_for(member.inputs), member.name
        )
        agent = _make_agent(member, tools_mod, ref_props, [], rewritten)
        agent.metadata = {
            **(agent.metadata or {}),
            _MARK_PROMPT_SPEC: _prompt_spec_marker(member, flat_to_original),
        }
        agents_by_name[member.name] = agent

    relationships = [
        (agents_by_name[member.name], agents_by_name[peer])
        for member in members
        for peer in (member.modifier_set.portal.to or [])  # type: ignore[union-attr]
    ]

    from pyagentspec.swarm import HandoffMode, Swarm

    # C1/s7zt3.14 round-trip: carry the mesh's trigger sub-mode on Swarm.handoff so
    # _reconstruct_swarm_mesh's _swarm_trigger reads it back. A tool-triggered Node
    # member (Portal(trigger="tool")) -> HandoffMode.OPTIONAL (the reference
    # adapter's tool-bound-handoff mode; OPTIONAL/ALWAYS are byte-identical there,
    # so OPTIONAL is the canonical choice); an all-output mesh -> HandoffMode.NEVER
    # (typed-goto routing, no bound transfer tool). A Construct member is never
    # tool-triggered (validation), so only Node members are polled.
    any_tool = any(
        isinstance(m, Node) and m.modifier_set.portal is not None and m.modifier_set.portal.is_tool_triggered
        for m in members
    )
    swarm = Swarm(
        name=construct.name,
        first_agent=agents_by_name[entry.name],
        relationships=relationships,
        handoff=HandoffMode.OPTIONAL if any_tool else HandoffMode.NEVER,
        metadata={
            _MARK_PORTAL_SPEC: {
                "max_hops": entry_portal.max_hops,
                "on_exhaust": entry_portal.on_exhaust,
                "route": entry_portal.route,
            }
        },
    )

    # Phase 6 (neograph-s7zt3.2): a mesh member's Operator (HITL approval gate)
    # must NOT silently vanish on export. Collect every gated member's `when` in
    # member order (deterministic, matches every other _MARK_*_SPEC builder).
    #
    # Keyed by the AGENT name (`{member.name}-agent`, set by _make_agent), NOT
    # the raw member name: _reconstruct_swarm_mesh names each imported member by
    # its Swarm agent's `.name`, so keying by the agent name is what makes the
    # loader's `member.name in gated` match on round-trip. (The design doc's §4
    # said "member name"; its repro used ad-hoc agents and so never exercised
    # _make_agent's `-agent` suffix -- the agent name is the identity that
    # actually survives export->import.)
    gated: dict[str, str] = {
        agents_by_name[member.name].name: member.modifier_set.operator.when
        for member in members
        if member.modifier_set.operator is not None
    }
    if not gated:
        return swarm  # unchanged today's-behavior path: pure PORTAL cell untouched

    # Mesh-exit pause composite. Swarm has no interior per-member pause primitive
    # (verified live, pyagentspec 26.1.2 -- Swarm.relationships is a flat
    # AgenticComponent adjacency list, no Node graph to splice a check into), so
    # the gate is approximated at the point control returns to the enclosing
    # Flow, mirroring _lower_operator's existing BranchingNode + InputMessageNode
    # shape one-for-one, with the Swarm wrapped in an AgentNode (legal:
    # AgentNode.agent: SerializeAsAny[AgenticComponent], and Swarm IS an
    # AgenticComponent). ALL gated members ride one shared marker dict on the
    # single check node -- a mesh is one connected component (_check_portal_mesh),
    # so "one shared composite" is always well-defined. This is a round-trip-
    # lossless serialization, NOT a behaviorally-faithful foreign runtime: a
    # foreign engine sees one exit-point BranchingNode, not neograph's own
    # per-member interior gate (factory.make_portal_approval_fn).
    nodes_mod, flow_mod, edges_mod, property_mod, _tools_mod = _import_agent_spec_flow_classes()

    agent_node = nodes_mod.AgentNode(name=f"{construct.name}__mesh", agent=swarm)
    check = nodes_mod.BranchingNode(
        name=f"{construct.name}__portal_operator_check",
        mapping={"true": _PAUSE_BRANCH, "false": _DEFAULT_BRANCH},
        metadata={
            _MARK_MODIFIER: "portal_operator",
            _MARK_PORTAL_OPERATOR_SPEC: gated,
        },
    )
    input_message = nodes_mod.InputMessageNode(
        name=f"{construct.name}__portal_operator_pause",
        outputs=[property_mod.StringProperty(title="user_input")],
    )
    start = nodes_mod.StartNode(name=f"{construct.name}__start")
    end_default = nodes_mod.EndNode(name=f"{construct.name}__end_default")
    end_paused = nodes_mod.EndNode(name=f"{construct.name}__end_paused")

    return flow_mod.Flow(
        name=construct.name,
        start_node=start,
        nodes=[start, agent_node, check, input_message, end_default, end_paused],
        control_flow_connections=[
            edges_mod.ControlFlowEdge(name=f"{construct.name}__start_to_mesh", from_node=start, to_node=agent_node),
            edges_mod.ControlFlowEdge(name=f"{construct.name}__mesh_to_check", from_node=agent_node, to_node=check),
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__check_to_pause",
                from_node=check,
                from_branch=_PAUSE_BRANCH,
                to_node=input_message,
            ),
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__check_to_default",
                from_node=check,
                from_branch=_DEFAULT_BRANCH,
                to_node=end_default,
            ),
            edges_mod.ControlFlowEdge(
                name=f"{construct.name}__pause_to_end", from_node=input_message, to_node=end_paused
            ),
        ],
    )


def _is_peer_mesh_member(item: Any) -> bool:
    """True iff ``item`` carries a PEER-mode (non-dispatch) Portal — i.e. it is
    a Portal mesh member — using the SAME modifier-agnostic detection the IR
    normalizer and ``_check_portal_mesh`` use, never an ``isinstance(Node)``
    gate (A1).

    ``classify_modifiers`` reads ``.modifier_set`` on Node AND Construct, so a
    Construct mesh member (do0d9) is correctly detected as a member rather than
    misclassified as a "non-mesh node" and false-rejecting the whole mesh.
    """
    _combo, mods = classify_modifiers(item)
    portal = mods.get("portal")
    return portal is not None and not portal.is_dispatch


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
        return _lower_portal_mesh_to_swarm(construct, cast("list[Node | Construct]", mesh_members), tools_mod)

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
        Property title, not '{upstream}.{title}').

        ``dest_title`` decouples the DESTINATION input field from the
        ``source_output`` when they differ — needed for a dict-form-OUTPUT
        producer (B4), whose output Property is ``{key}.{field}`` but whose
        downstream consumer declares the input as the bare ``{field}`` (or
        ``{upstream}.{field}``). Defaults to ``source_title`` (unchanged for
        every single-shape caller).

        Option F consumer sweep neograph-cbpyx: when the CONSUMING item is
        placeholder-translated (LLM mode), the destination declares the flat
        ${var}->{{ flat }} name, so the dotted ``{upstream}.{title}`` (and the
        MapNode's ``iterated_``-prefixed form) route through the item's flat map —
        and the edge is DROPPED when the source path was never referenced in the
        prompt (a real topology change: the translated primitive has no data path
        to that value). Scripted/raw destinations keep the untranslated form.
        """
        dest_item = item_by_name.get(item_name)
        translate = _is_translation_eligible(dest_item)
        orig_to_flat = _node_translation(cast("Node", dest_item))[2] if translate else {}
        dest_core = dest_title if dest_title is not None else source_title
        dotted = f"{upstream_name}.{dest_core}" if upstream_name else dest_core
        for target_node, iterated in input_targets_by_item_name[item_name]:
            if translate:
                flat = orig_to_flat.get(dotted)
                if flat is None:
                    continue
                core = flat
            elif iterated:
                # A MapNode infers its inputs as ``iterated_{json_schema title}``
                # — and pyagentspec forbids dots in json_schema titles, so the
                # inner node's dict-form ``{key}.{field}`` prefix lives only on
                # Property.title; the inferred MapNode input is the BARE
                # ``iterated_{field}``. Target that, not the dotted form.
                core = source_title
            else:
                core = dotted
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
                    # output Property is ``{key}.{field}`` (the SAME prefix
                    # ``_properties_for`` applies); the consumer's declared input is
                    # ``{upstream_name}.{field}`` — decouple source/dest so the edge
                    # is wired, not dropped.
                    for prop in _properties_for({output_key: no.all_keys[output_key]}):
                        field = prop.title[len(output_key) + 1 :]
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
                # matching key's ``{key}.{field}`` output Property to the consumer's
                # bare ``{field}`` input (source/dest decoupled, same as the
                # dict-form-input branch above).
                matched = False
                for key, ktype in no.all_keys.items():
                    if not isinstance(ktype, type) or not isinstance(ni.single_type, type):
                        continue
                    if not (issubclass(ktype, ni.single_type) or issubclass(ni.single_type, ktype)):
                        continue
                    for prop in _properties_for({key: ktype}):
                        field = prop.title[len(key) + 1 :]
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
