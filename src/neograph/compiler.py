"""Graph compiler — Construct -> LangGraph StateGraph.

    graph = compile(my_construct)

Reads the Construct's node list, resolves modifiers (Oracle, Each, Operator),
and builds a LangGraph StateGraph with correct topology, checkpointing, and state bus.

Wiring helpers (Each, Oracle, Loop, Branch, Operator topology) live in _wiring.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from neograph._compiled import CompiledNeograph
from neograph._dev_warnings import DEV_MODE
from neograph._fan_agent_wrap import wrap_fan_over_agents
from neograph._ir_branch import _BranchNode, iter_with_arms
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime, check_llm_kwargs_or_raise
from neograph._oracle import (
    make_each_redirect_fn,
    make_oracle_merge_fn,
    make_oracle_redirect_fn,
)
from neograph._runtime_registry import _decoration_registry
from neograph._sidecar import _get_param_res
from neograph._state_keys import StateKeys
from neograph._subconstruct import make_subgraph_fn
from neograph._trace import named
from neograph._wiring import (
    _add_agent_cycle,
    _add_branch_to_graph,
    _add_each_oracle_fused,
    _add_loop_back_edge,
    _add_operator_check,
    _add_portal_dispatch,
    _add_portal_mesh,
    _add_subgraph_loop,
    _contiguous_portal_mesh,
    _wire_each,
    _wire_oracle,
)
from neograph.construct import Construct, iter_nodes
from neograph.di import DIKind
from neograph.errors import CompileError, ConfigurationError
from neograph.factory import make_node_fn
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.naming import field_name_for
from neograph.node import Node
from neograph.state import (
    build_output_schema_model,
    compile_state_model,
    compute_node_fingerprints,
    compute_schema_fingerprint,
)
from neograph.tool import register_bound_tool_factories

log = structlog.get_logger()


def compile(
    construct: Construct,
    checkpointer: Any = None,
    _context_types: dict[str, type] | None = None,
    *,
    llm_factory: Any = None,
    prompt_compiler: Any = None,
    renderer: Any = None,
    cost_callback: Any = None,
    scripted: dict[str, Callable] | None = None,
    conditions: dict[str, Callable] | None = None,
    tool_factories: dict[str, Callable] | None = None,
    _runtime: LlmRuntime | None = None,
    _scripted_lookup: dict[str, Callable] | None = None,
) -> CompiledNeograph:
    """Compile a Construct into an executable LangGraph StateGraph.

    Args:
        construct: The Construct to compile.
        checkpointer: LangGraph checkpointer for persistence/resume support.
                      Required if any node uses Operator (interrupt/resume).
        llm_factory: Per-tier LLM client factory used by LLM-mode nodes.
        prompt_compiler: Callable that turns a prompt template + input into
            a message list for the LLM. Required by LLM-mode nodes.
        renderer: Optional global renderer used for input rendering when a
            node has no per-node renderer set.
        cost_callback: Optional telemetry hook called after each LLM
            invocation with token usage.
        _runtime: Internal — sub-construct compiles thread the parent's
            runtime through this argument instead of rebuilding it.

    The LLM kwargs are closed over into a frozen `LlmRuntime` bundle and
    threaded into every factory closure (`make_node_fn`,
    `make_oracle_merge_fn`, etc.). Two `compile()` calls with different
    kwargs produce fully isolated graphs (§2 multi-tenant guarantee).

    Retry concerns live in their own layers (see docs/design/architecture-decisions.md §3):
      - Transient API failures: user's llm_factory via model.with_retry(...)
      - LLM output-quality failures: per-node LlmConfig.max_retries
      - Flaky external calls in scripted nodes: inside the node function
    """
    compile_log = log.bind(construct=construct.name, nodes=len(construct.nodes))

    # Resolve runtime: explicit kwargs > internal pass-through > legacy compat.
    if _runtime is not None:
        runtime = _runtime
    elif any(x is not None for x in (llm_factory, prompt_compiler, renderer, cost_callback)):
        runtime = LlmRuntime.build(
            llm_factory=llm_factory,
            prompt_compiler=prompt_compiler,
            renderer=renderer,
            cost_callback=cost_callback,
        )
    else:
        # No runtime configuration supplied — start empty. Fail-loud check
        # below will raise CompileError if any LLM-mode nodes are present.
        runtime = EMPTY_RUNTIME

    # Per-compile scripted lookup: walk the construct, collect each Node's
    # `_scripted_shim` PrivateAttr into a fresh dict. Factory closures close
    # over this dict instead of consulting the deprecated fallback registry.
    # When called recursively for a sub-construct, the parent passes its
    # collected lookup so child-graph factory closures see the same shims.
    if _scripted_lookup is not None:
        scripted_lookup: dict[str, Callable] = _scripted_lookup
    else:
        scripted_lookup = _collect_scripted_shims(construct)
        # Merge in decoration-time shims for inline body-merge / @merge_fn /
        # interrupt_when callables. These are registered at decoration time
        # (in the _runtime_registry leaf) and flow to compile()'s per-compile dict.
        scripted_lookup.update(_decoration_registry.scripted)
        # Merge in callers' explicit `scripted=` kwargs LAST so they win.
        if scripted:
            scripted_lookup.update(scripted)

    # Build per-compile condition + tool_factory dicts. Both seed from
    # decoration-time registrations (for inline interrupt_when callables and
    # `@tool` decorations) and merge in explicit kwargs.
    condition_lookup: dict[str, Callable] = dict(_decoration_registry.condition)
    if conditions:
        condition_lookup.update(conditions)
    tool_factory_lookup: dict[str, Callable] = dict(_decoration_registry.tool_factory)
    if tool_factories:
        tool_factory_lookup.update(tool_factories)
    # Pre-pass: fan-over-agent auto-wrap (neograph-m6d3.6). Oracle over a
    # self-contained agent/act node is rewritten into an isolated single-node
    # sub-construct so the fan runs over isolated subgraph state — the ONLY
    # correct mechanism (an inline fan Sends into the ReAct cycle's SHARED
    # reducer channels and collapses N>1 branches; see _fan_agent +
    # docs/design/fan-over-agent-node-2026-07-07.md). Unsupported shapes already
    # failed loud at assembly. Runs BEFORE the state model + all validation below
    # so everything downstream sees the wrapped sub-construct uniformly.
    construct = wrap_fan_over_agents(construct, scripted_lookup)

    # Auto-register factories for raw LangChain BaseTools passed via tools=.
    # Explicit tool_factories= (merged above) win; bound tools fill the gaps.
    register_bound_tool_factories(construct, tool_factory_lookup)

    compile_log.info(
        "compile_start",
        node_names=[n.name for n in construct.nodes],
        modifiers={
            n.name: n.modifier_set.combo.name
            for n in construct.nodes
            if isinstance(n, Node) and n.modifier_set.combo.name != "BARE"
        },
    )

    # Validate: Operator string conditions are registered.
    # Check BEFORE the checkpointer guard so the real error isn't masked.
    # iter_with_arms so a bare arm Operator node's condition is validated too —
    # bare arm Nodes bypass make_node_fn validation entirely (arm Constructs
    # self-validate via _wiring's recursive _compile). See neograph-vn5f (6-8).
    for item in iter_with_arms(construct):
        if isinstance(item, (Node, Construct)):
            _, item_mods = classify_modifiers(item)
            op = item_mods.get("operator")
            if op is not None and isinstance(op.when, str):
                if op.when not in condition_lookup:
                    raise ConfigurationError.build(
                        f"Condition '{op.when}' not registered",
                        hint=("Pass conditions={'" + op.when + "': fn} to compile() to register the condition."),
                    )

    # Validate: Operator requires checkpointer
    has_operator = any(
        "operator" in classify_modifiers(item)[1]
        for item in iter_with_arms(construct)
        if isinstance(item, (Node, Construct))
    )
    if has_operator and checkpointer is None:
        raise CompileError.build(
            "Operator modifier requires a checkpointer",
            expected="checkpointer passed to compile()",
            found="checkpointer=None",
            hint="Pass checkpointer= to compile()",
            construct=construct.name,
        )

    # Validate: LLM nodes require runtime configuration (fail-loud per §2).
    check_llm_kwargs_or_raise(
        construct,
        runtime.llm_factory,
        runtime.prompt_compiler,
        source="compile()",
    )

    # Validate: tool factory registrations
    # iter_with_arms so a bare arm agent/act node's tool factories are validated
    # at compile time. See neograph-vn5f (site 8).
    for item in iter_with_arms(construct):
        if isinstance(item, Node) and item.mode in ("agent", "act") and item.tools:
            for t in item.tools:
                if t.name not in tool_factory_lookup:
                    raise CompileError.build(
                        f"tool '{t.name}' has no registered factory",
                        expected=f"compile(..., tool_factories={{'{t.name}': factory_fn}})",
                        found="no factory registered",
                        hint=f"Pass tool_factories={{'{t.name}': factory_fn}} to compile().",
                        node=item.name,
                        construct=construct.name,
                    )

    # output_strategy values are now enforced at Node construction via the
    # typed LlmConfig.output_strategy Literal field (pej0). The prior runtime
    # _VALID_STRATEGIES check is redundant and was removed.

    # 1. Generate state model from node I/O
    state_model = compile_state_model(construct, context_types=_context_types)

    # 2. Build graph. Declare output_schema = non-neo_ fields so the ENGINE filters
    # framework plumbing out of invoke/ainvoke results (neograph-pjqe: declare, don't
    # wrap). This is recursive — the sub-construct compile at _add_subgraph re-enters
    # compile(), so child graphs get the same declaration and their invoke() results
    # are neo_-free without a _strip_internals wrap. Stream chunks are NOT covered by
    # output_schema in langgraph 1.2.4 (see runner._finalize_by_mode).
    graph = StateGraph(state_model, output_schema=build_output_schema_model(state_model))

    prev_node: str | None = None

    # A contiguous Portal run is a mesh: lowered ONCE at its ENTRY by
    # _add_portal_mesh; the remaining members are recorded in `meshed` so the
    # walk skips them (review M1 — prevents double-adding). See design §4.1.
    meshed: set[int] = set()

    for item in construct.nodes:
        if id(item) in meshed:
            continue  # a non-entry mesh member, already lowered at its entry
        if isinstance(item, Node) and classify_modifiers(item)[0] in (
            ModifierCombo.PORTAL,
            ModifierCombo.PORTAL_OPERATOR,
        ):
            portal = item.modifier_set.portal
            if portal is not None and portal.is_dispatch:
                # Dispatch mode (design §4.2): a standalone LINEAR node (plain
                # add_node + static edge, NO Command), never a mesh member —
                # _contiguous_portal_mesh / _validation_portal exclude it.
                prev_node = _add_portal_dispatch(
                    graph, item, prev_node, runtime=runtime,
                    scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup,
                )
                continue
            members = _contiguous_portal_mesh(construct.nodes, item)
            prev_node = _add_portal_mesh(
                graph,
                members,
                prev_node,
                checkpointer=checkpointer,
                parent_state_model=state_model,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                tool_factory_lookup=tool_factory_lookup,
                condition_lookup=condition_lookup,
            )
            meshed.update(id(m) for m in members)
        elif isinstance(item, _BranchNode):
            prev_node = _add_branch_to_graph(
                graph,
                item,
                prev_node,
                checkpointer=checkpointer,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                condition_lookup=condition_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )
        elif isinstance(item, Construct):
            prev_node = _add_subgraph(
                graph,
                item,
                prev_node,
                checkpointer=checkpointer,
                parent_state_model=state_model,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                condition_lookup=condition_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )
        else:
            assert isinstance(item, Node)  # narrow ConstructItem Protocol to Node
            prev_node = _add_node_to_graph(
                graph,
                item,
                prev_node,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                condition_lookup=condition_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )

    # Final edge to END
    if prev_node:
        graph.add_edge(prev_node, END)

    # 3. Compile
    # NOTE: LangGraph 1.x has no public API for narrowing the checkpointer's
    # msgpack allowlist (only langgraph._internal._serde.build_serde_allowlist
    # exists, which is private and explicitly off-limits per architecture
    # decision §1). We accept the default warn-all behavior; "Deserializing
    # unregistered type" warnings on checkpoint resume are expected. Revisit
    # if LangGraph exposes a public allowlist API.
    compiled = graph.compile(checkpointer=checkpointer)
    compile_log.info("compile_complete", state_fields=list(state_model.model_fields.keys()))

    # 4-6. Wrap the LangGraph graph in the typed CompiledNeograph facade with
    # all framework metadata as typed fields (no _neo_* monkey-patching):
    #   - required_di: pre-flight DI validation in run()
    #   - schema/node fingerprints: checkpoint-resume schema validation
    #   - construct/runtime/scripted/conditions/tool_factories: verify_compiled
    result = CompiledNeograph(
        graph=compiled,
        required_di=_collect_required_di(construct),
        schema_fingerprint=compute_schema_fingerprint(state_model),
        node_fingerprints=compute_node_fingerprints(construct),
        construct=construct,
        runtime=runtime,
        scripted=scripted_lookup,
        conditions=condition_lookup,
        tool_factories=tool_factory_lookup,
    )

    # 7. Dev-mode DAG visualization
    if DEV_MODE:
        _print_dag_summary(result, construct)

    return result


def _collect_scripted_shims(construct: Construct) -> dict[str, Any]:
    """Walk the construct tree and build the per-compile scripted dict.

    For each Node with a `_scripted_shim` PrivateAttr (attached by the
    `@node` decorator path via `_register_node_scripted`), insert the
    shim under `node.scripted_fn` into the returned dict. Sub-constructs
    are walked recursively.
    """
    lookup: dict[str, Any] = {}
    for item in iter_nodes(construct):
        shim = getattr(item, "_scripted_shim", None)
        if shim is not None and item.scripted_fn:
            lookup[item.scripted_fn] = shim
    return lookup


def _collect_required_di(construct: Construct) -> dict[str, set[str]]:
    """Walk all nodes and collect required DI param names by source (input/config).

    Returns {"input": {"topic", "node_id"}, "config": {"limiter"}} — the set of
    param names that must be present in run(input=) or config['configurable'].
    """
    required: dict[str, set[str]] = {"input": set(), "config": set()}
    for item in iter_nodes(construct):
        param_res = _get_param_res(item)
        if not param_res:
            continue
        for _pname, binding in param_res.items():
            if not binding.required:
                continue
            if binding.kind in (DIKind.FROM_INPUT, DIKind.FROM_INPUT_MODEL):
                if binding.kind == DIKind.FROM_INPUT_MODEL:
                    # Bundled model — each field is a required input key
                    model_cls = binding.model_cls
                    if model_cls is not None:
                        for fname in model_cls.model_fields:
                            required["input"].add(fname)
                else:
                    required["input"].add(binding.name)
            elif binding.kind in (DIKind.FROM_CONFIG, DIKind.FROM_CONFIG_MODEL):
                if binding.kind == DIKind.FROM_CONFIG_MODEL:
                    model_cls = binding.model_cls
                    if model_cls is not None:
                        for fname in model_cls.model_fields:
                            required["config"].add(fname)
                else:
                    required["config"].add(binding.name)
    return required


def describe_graph(compiled: Any) -> str:
    """Return a Mermaid diagram string for a compiled graph.

    Usage::

        graph = compile(pipeline)
        print(describe_graph(graph))

    Paste the output into any Mermaid renderer (GitHub, docs, mermaid.live).
    """
    try:
        return compiled.get_graph().draw_mermaid()
    except (AttributeError, TypeError, ValueError) as exc:
        log.debug("describe_graph_failed", error=str(exc))
        return "(graph visualization not available)"


def _print_dag_summary(compiled: Any, construct: Any) -> None:
    """Print a human-readable DAG summary to stderr in dev mode."""
    import sys

    try:
        lg_graph = compiled.get_graph()
    except (AttributeError, TypeError, ValueError):
        return

    nodes = [n for n in lg_graph.nodes if n not in ("__start__", "__end__")]
    edges = lg_graph.edges

    lines = [f"[neograph-dev] Compiled '{construct.name}' ({len(nodes)} nodes):"]

    for edge in edges:
        src = edge.source.replace("__start__", "START").replace("__end__", "END")
        tgt = edge.target.replace("__start__", "START").replace("__end__", "END")
        cond = " [conditional]" if edge.conditional else ""
        lines.append(f"  {src} -> {tgt}{cond}")

    print("\n".join(lines), file=sys.stderr)


def _add_subgraph(
    graph: StateGraph,
    sub: Construct,
    prev_node: str | None,
    checkpointer: Any = None,
    parent_state_model: type[BaseModel] | None = None,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> str:
    """Compile a sub-Construct as an isolated subgraph node, with modifier support."""
    if sub.input is None:
        raise CompileError.build(
            "sub-construct has no input type",
            expected="input=SomeModel declared on the sub-construct",
            found="input=None",
            hint="Declare input=SomeModel on the sub-construct",
            construct=sub.name,
        )

    sub_log = log.bind(subgraph=sub.name)
    output_name = sub.output.__name__ if sub.output is not None else "None"
    sub_log.info("subgraph_compile", input=sub.input.__name__, output=output_name)

    # Build context_types from parent state model — gives subconstruct concrete
    # types for context fields instead of Any.
    _context_types: dict[str, type] | None = None
    if parent_state_model is not None:
        _context_types = {}
        for fname, finfo in parent_state_model.model_fields.items():
            if finfo.annotation is not None:
                _context_types[fname] = finfo.annotation

    # Compile the sub-construct into its own graph (recursive, thread checkpointer + runtime)
    sub_graph = compile(
        sub,
        checkpointer=checkpointer,
        _context_types=_context_types,
        _runtime=runtime,
        _scripted_lookup=scripted_lookup,
        conditions=condition_lookup,
        tool_factories=tool_factory_lookup,
    )
    field_name = field_name_for(sub.name)

    # Build the subgraph node function via factory. compile() returns the
    # CompiledNeograph facade; make_subgraph_fn drives the raw LangGraph graph.
    # `named` binds run_name=sub.name here (not inside make_subgraph_fn, whose
    # bare dual-path return is pinned by the async guard) so the sub-construct's
    # engine span reads as the construct name across every modifier branch below
    # (bare / oracle-redirect / each-redirect / loop). See neograph-3fm1.
    subgraph_fn = named(
        make_subgraph_fn(sub, sub_graph.graph),
        sub.name,
        mode="subgraph",
        output_type=sub.output.__name__ if sub.output is not None else None,
    )

    from typing import assert_never

    combo, mods = classify_modifiers(sub)
    operator = mods.get("operator")

    match combo:
        case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
            # Each x Oracle fusion on Constructs: not supported.
            raise CompileError.build(
                "Each x Oracle fusion is not supported on sub-constructs",
                found="both Oracle and Each modifiers on a sub-construct",
                hint="Use a Node with map_over + ensemble_n instead",
                construct=sub.name,
            )
        case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
            oracle = mods["oracle"]
            collector_field = StateKeys.oracle_collector(field_name)
            redirect_fn = make_oracle_redirect_fn(
                subgraph_fn,
                field_name,
                collector_field,
                item=sub,
            )
            merge_fn = make_oracle_merge_fn(
                oracle,
                field_name,
                collector_field,
                sub.output,
                llm_config=sub.llm_config or None,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
            )
            last_name = _wire_oracle(graph, sub.name, redirect_fn, merge_fn, oracle, prev_node)
        case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
            each = mods["each"]
            each_fn = make_each_redirect_fn(subgraph_fn, field_name, each, item=sub)
            last_name = _wire_each(graph, sub.name, each_fn, each, prev_node)
        case ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
            loop = mods["loop"]
            last_name = _add_subgraph_loop(graph, sub, subgraph_fn, loop, prev_node, condition_lookup=condition_lookup)
        case ModifierCombo.BARE | ModifierCombo.OPERATOR:
            # Plain subgraph — no modifiers (or Operator only)
            graph.add_node(sub.name, subgraph_fn)
            if prev_node:
                graph.add_edge(prev_node, sub.name)
            else:
                graph.add_edge(START, sub.name)
            last_name = sub.name
        case ModifierCombo.PORTAL | ModifierCombo.PORTAL_OPERATOR:
            # Portal (with or without Operator) on a sub-construct is illegal in
            # v1 (D-MESH-LEVEL); already rejected at assembly — this arm is
            # defense-in-depth + exhaustiveness.
            raise CompileError.build(
                "Portal on a sub-construct is not supported",
                expected="mesh members must be sibling Nodes (D-MESH-LEVEL)",
                found=f"Portal modifier on sub-construct '{sub.name}'",
            )
        case _ as unreachable:
            assert_never(unreachable)

    # Operator stacking: add interrupt check after the primary modifier
    if operator:
        last_name = _add_operator_check(graph, last_name, operator, condition_lookup=condition_lookup)

    return last_name


def _add_node_to_graph(
    graph: StateGraph,
    node: Node,
    prev_node: str | None,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> str:
    """Add a single node (with its modifiers) to the graph. Returns the last node name."""
    from typing import assert_never

    combo, mods = classify_modifiers(node)
    operator = mods.get("operator")

    # NOTE: a fan (Each/Oracle/Loop) modifier over an agent/act node never
    # reaches this dispatch as a bare Node. Unsupported shapes are rejected at
    # assembly time (_construct_validation -> _fan_agent); the one supported
    # shape (Oracle over a self-contained agent) is rewritten into a sub-construct
    # by the _wrap_fan_over_agents pre-pass in compile() before the state model is
    # built, so it arrives here as a Construct via _add_subgraph. By here an
    # agent/act Node is guaranteed BARE or OPERATOR-only. See neograph-m6d3.6.

    match combo:
        case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
            # Each x Oracle fusion: flat M x N Send topology
            last_name = _add_each_oracle_fused(
                graph,
                node,
                mods["each"],
                mods["oracle"],
                prev_node,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )
        case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
            # Oracle: expand to fan-out + merge
            last_name = _add_oracle_nodes(
                graph,
                node,
                mods["oracle"],
                prev_node,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )
        case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
            # Each: expand to fan-out + barrier
            last_name = _add_each_nodes(
                graph,
                node,
                mods["each"],
                prev_node,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )
        case ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
            # Loop: conditional back-edge
            last_name = _add_loop_back_edge(
                graph,
                node,
                mods["loop"],
                prev_node,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                condition_lookup=condition_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )
        case ModifierCombo.BARE | ModifierCombo.OPERATOR:
            if node.mode in ("agent", "act"):
                # Agent/act: inline ReAct cycle (agent/tools/parse + conditional router).
                last_name = _add_agent_cycle(
                    graph,
                    node,
                    prev_node,
                    runtime=runtime,
                    tool_factory_lookup=tool_factory_lookup,
                    condition_lookup=condition_lookup,
                )
            else:
                # Simple node — no modifiers (or Operator only)
                node_name = node.name
                node_fn = make_node_fn(
                    node, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup
                )
                graph.add_node(node_name, node_fn)
                if prev_node:
                    graph.add_edge(prev_node, node_name)
                else:
                    graph.add_edge(START, node_name)
                last_name = node_name
        case ModifierCombo.PORTAL | ModifierCombo.PORTAL_OPERATOR:
            # Unreachable: the mesh-aware walk (M1) lowers a contiguous mesh via
            # _add_portal_mesh before per-node dispatch. Arm kept for match
            # exhaustiveness; fails loud if the walk ever regresses.
            raise CompileError.build(
                "Portal member reached per-node dispatch",
                found=f"Portal node '{node.name}' dispatched individually",
                hint="the compile walk must collapse the contiguous mesh (M1)",
            )
        case _ as unreachable:
            assert_never(unreachable)

    # Operator stacking: add interrupt check after the primary modifier
    if operator:
        last_name = _add_operator_check(graph, last_name, operator, condition_lookup=condition_lookup)

    return last_name


def _add_oracle_nodes(
    graph: StateGraph,
    node: Node,
    oracle: Any,
    prev_node: str | None,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> str:
    """Expand Oracle modifier into fan-out generators + merge barrier."""
    field_name = field_name_for(node.name)
    collector_field = StateKeys.oracle_collector(field_name)

    raw_fn = make_node_fn(
        node, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup
    )
    redirect_fn = make_oracle_redirect_fn(
        raw_fn,
        field_name,
        collector_field,
        item=node,
    )
    merge_fn = make_oracle_merge_fn(
        oracle,
        field_name,
        collector_field,
        node.outputs,
        node_inputs=node.inputs,
        llm_config=node.llm_config or None,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
    )

    return _wire_oracle(graph, node.name, redirect_fn, merge_fn, oracle, prev_node)


def _add_each_nodes(
    graph: StateGraph,
    node: Node,
    each: Any,
    prev_node: str | None,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> str:
    """Expand Each modifier into fan-out dispatch + barrier."""
    node_fn = make_node_fn(
        node, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup
    )
    return _wire_each(graph, node.name, node_fn, each, prev_node)
