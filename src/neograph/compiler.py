"""Graph compiler — Construct -> LangGraph StateGraph.

    graph = compile(my_construct)

Reads the Construct's node list, resolves modifiers (Oracle, Each, Operator),
and builds a LangGraph StateGraph with correct topology, checkpointing, and state bus.

Wiring helpers (Each, Oracle, Loop, Branch, Operator topology) live in _wiring.py.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from neograph._registry import registry
from neograph._wiring import (  # noqa: F401 — re-exported for backward compat
    _add_branch_to_graph,
    _add_each_oracle_fused,
    _add_loop_back_edge,
    _add_operator_check,
    _add_subgraph_loop,
    _merge_one_group,
    _wire_each,
    _wire_oracle,
)
from neograph._dev_warnings import DEV_MODE
from neograph.construct import Construct
from neograph.di import DIKind
from neograph.errors import CompileError
from neograph.factory import (
    lookup_condition,
    make_each_redirect_fn,
    make_node_fn,
    make_oracle_merge_fn,
    make_oracle_redirect_fn,
    make_subgraph_fn,
)
from neograph.forward import _BranchNode
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.naming import field_name_for
from neograph.node import Node
from neograph.state import compile_state_model, compute_node_fingerprints, compute_schema_fingerprint

log = structlog.get_logger()


def _register_msgpack_types(checkpointer: Any, state_model: type) -> None:
    """Register node output types with the checkpointer's msgpack serializer.

    Converts the serializer from warn-all mode (allowed_msgpack_modules=True)
    to an explicit allowlist so LangGraph doesn't emit 'Deserializing
    unregistered type' warnings on checkpoint resume.
    """
    try:
        from langgraph._internal._serde import build_serde_allowlist
        from langgraph.checkpoint.serde._msgpack import SAFE_MSGPACK_TYPES
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    except ImportError:  # pragma: no cover
        return  # LangGraph version doesn't have this API

    serde = getattr(checkpointer, "serde", None)
    if not isinstance(serde, JsonPlusSerializer):
        return  # pragma: no cover

    # Collect types from the state model (includes nested Pydantic models)
    allowlist = build_serde_allowlist(schemas=[state_model])
    allowlist = allowlist | SAFE_MSGPACK_TYPES

    # Merge with existing allowlist if present
    existing = getattr(serde, "_allowed_msgpack_modules", None)
    if isinstance(existing, (set, frozenset)):
        allowlist = allowlist | existing

    # Replace serde with the combined allowlist
    checkpointer.serde = JsonPlusSerializer(allowed_msgpack_modules=allowlist)


def compile(construct: Construct, checkpointer: Any = None, retry_policy: Any = None, _context_types: dict[str, type] | None = None) -> Any:
    """Compile a Construct into an executable LangGraph StateGraph.

    Args:
        construct: The Construct to compile.
        checkpointer: LangGraph checkpointer for persistence/resume support.
                      Required if any node uses Operator (interrupt/resume).
        retry_policy: LangGraph RetryPolicy applied to all LLM-calling nodes
                      (think/agent/act). Handles malformed JSON, validation
                      errors, and transient API failures. Scripted nodes are
                      not retried.
    """
    compile_log = log.bind(construct=construct.name, nodes=len(construct.nodes))
    compile_log.info("compile_start",
                     node_names=[n.name for n in construct.nodes],
                     modifiers={n.name: n.modifier_set.combo.name
                                for n in construct.nodes
                                if isinstance(n, Node) and n.modifier_set.combo.name != "BARE"})

    # Validate: Operator string conditions are registered.
    # Check BEFORE the checkpointer guard so the real error isn't masked.
    for item in construct.nodes:
        if isinstance(item, (Node, Construct)):
            _, item_mods = classify_modifiers(item)
            op = item_mods.get("operator")
            if op is not None and isinstance(op.when, str):
                lookup_condition(op.when)  # raises ConfigurationError if not registered

    # Validate: Operator requires checkpointer
    has_operator = any(
        "operator" in classify_modifiers(item)[1] for item in construct.nodes
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

    # Validate: LLM nodes require configure_llm()
    has_llm_node = any(
        isinstance(item, Node) and item.mode in ("think", "agent", "act")
        for item in construct.nodes
    )
    if has_llm_node:
        from neograph._llm import _llm_factory, _prompt_compiler
        if _llm_factory is None or _prompt_compiler is None:
            missing = []
            if _llm_factory is None:
                missing.append("llm_factory")
            if _prompt_compiler is None:
                missing.append("prompt_compiler")
            raise CompileError.build(
                "LLM nodes require configure_llm()",
                expected="llm_factory and prompt_compiler configured",
                found=f"{' and '.join(missing)} not set",
                hint="Call neograph.configure_llm() before compile()",
                construct=construct.name,
            )

    # Validate: tool factory registrations
    for item in construct.nodes:
        if isinstance(item, Node) and item.mode in ("agent", "act") and item.tools:
            for t in item.tools:
                if t.name not in registry.tool_factory:
                    raise CompileError.build(
                        f"tool '{t.name}' has no registered factory",
                        expected=f"register_tool_factory('{t.name}', factory_fn) called",
                        found="no factory registered",
                        hint=f"Use register_tool_factory('{t.name}', factory_fn) before compile()",
                        node=item.name,
                        construct=construct.name,
                    )

    # Validate: output_strategy values
    _VALID_STRATEGIES = {"structured", "json_mode", "text"}
    for item in construct.nodes:
        if isinstance(item, Node) and item.mode in ("think", "agent", "act"):
            strategy = item.llm_config.get("output_strategy")
            if strategy is not None and strategy not in _VALID_STRATEGIES:
                raise CompileError.build(
                    "invalid output_strategy",
                    expected=f"one of {', '.join(sorted(_VALID_STRATEGIES))}",
                    found=f"output_strategy='{strategy}'",
                    node=item.name,
                    construct=construct.name,
                )

    # 1. Generate state model from node I/O
    state_model = compile_state_model(construct, context_types=_context_types)

    # 2. Build graph
    graph = StateGraph(state_model)

    prev_node: str | None = None

    for item in construct.nodes:
        if isinstance(item, _BranchNode):
            prev_node = _add_branch_to_graph(graph, item, prev_node)
        elif isinstance(item, Construct):
            prev_node = _add_subgraph(graph, item, prev_node, checkpointer=checkpointer, retry_policy=retry_policy, parent_state_model=state_model)
        else:
            prev_node = _add_node_to_graph(graph, item, prev_node, retry_policy=retry_policy)

    # Final edge to END
    if prev_node:
        graph.add_edge(prev_node, END)

    # 3. Register output types with checkpointer's msgpack serializer
    if checkpointer is not None:
        _register_msgpack_types(checkpointer, state_model)

    # 4. Compile
    compiled = graph.compile(checkpointer=checkpointer)
    compile_log.info("compile_complete", state_fields=list(state_model.model_fields.keys()))

    # 5. Collect required DI params for pre-flight validation in run()
    compiled._neo_required_di = _collect_required_di(construct)  # type: ignore[attr-defined]

    # 6. Schema fingerprint for checkpoint validation
    compiled._neo_schema_fingerprint = compute_schema_fingerprint(state_model)  # type: ignore[attr-defined]
    compiled._neo_node_fingerprints = compute_node_fingerprints(construct)  # type: ignore[attr-defined]

    # 7. Stash Construct for post-compile verification (verify_compiled)
    compiled._neo_construct = construct  # type: ignore[attr-defined]

    # 8. Dev-mode DAG visualization
    if DEV_MODE:
        _print_dag_summary(compiled, construct)

    return compiled


def _collect_required_di(construct: Construct) -> dict[str, set[str]]:
    """Walk all nodes and collect required DI param names by source (input/config).

    Returns {"input": {"topic", "node_id"}, "config": {"limiter"}} — the set of
    param names that must be present in run(input=) or config['configurable'].
    """
    required: dict[str, set[str]] = {"input": set(), "config": set()}

    def _walk(nodes: list) -> None:
        for item in nodes:
            if isinstance(item, Construct):
                _walk(item.nodes)
                continue
            if not isinstance(item, Node):
                continue
            param_res = item._param_res
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

    _walk(construct.nodes)
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
    retry_policy: Any = None,
    parent_state_model: type[BaseModel] | None = None,
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
            _context_types[fname] = finfo.annotation

    # Compile the sub-construct into its own graph (recursive, thread checkpointer)
    sub_graph = compile(sub, checkpointer=checkpointer, retry_policy=retry_policy, _context_types=_context_types)
    field_name = field_name_for(sub.name)

    # Build the subgraph node function via factory
    subgraph_fn = make_subgraph_fn(sub, sub_graph)

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
            collector_field = f"neo_oracle_{field_name}"
            redirect_fn = make_oracle_redirect_fn(subgraph_fn, field_name, collector_field)
            merge_fn = make_oracle_merge_fn(oracle, field_name, collector_field, sub.output)
            last_name = _wire_oracle(graph, sub.name, redirect_fn, merge_fn, oracle, prev_node)
        case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
            each = mods["each"]
            each_fn = make_each_redirect_fn(subgraph_fn, field_name, each)
            last_name = _wire_each(graph, sub.name, each_fn, each, prev_node)
        case ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
            loop = mods["loop"]
            last_name = _add_subgraph_loop(graph, sub, subgraph_fn, loop, prev_node)
        case ModifierCombo.BARE | ModifierCombo.OPERATOR:
            # Plain subgraph — no modifiers (or Operator only)
            graph.add_node(sub.name, subgraph_fn)
            if prev_node:
                graph.add_edge(prev_node, sub.name)
            else:
                graph.add_edge(START, sub.name)
            last_name = sub.name
        case _ as unreachable:
            assert_never(unreachable)

    # Operator stacking: add interrupt check after the primary modifier
    if operator:
        last_name = _add_operator_check(graph, last_name, operator)

    return last_name


def _add_node_to_graph(
    graph: StateGraph,
    node: Node,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Add a single node (with its modifiers) to the graph. Returns the last node name."""
    from typing import assert_never

    # Retry applies to LLM-calling nodes only (think/agent/act).
    # Scripted nodes are deterministic — retrying won't help.
    rp = retry_policy if node.mode in ("think", "agent", "act") else None

    combo, mods = classify_modifiers(node)
    operator = mods.get("operator")

    match combo:
        case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
            # Each x Oracle fusion: flat M x N Send topology
            last_name = _add_each_oracle_fused(graph, node, mods["each"], mods["oracle"], prev_node, retry_policy=rp)
        case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:
            # Oracle: expand to fan-out + merge
            last_name = _add_oracle_nodes(graph, node, mods["oracle"], prev_node, retry_policy=rp)
        case ModifierCombo.EACH | ModifierCombo.EACH_OPERATOR:
            # Each: expand to fan-out + barrier
            last_name = _add_each_nodes(graph, node, mods["each"], prev_node, retry_policy=rp)
        case ModifierCombo.LOOP | ModifierCombo.LOOP_OPERATOR:
            # Loop: conditional back-edge
            last_name = _add_loop_back_edge(graph, node, mods["loop"], prev_node, retry_policy=rp)
        case ModifierCombo.BARE | ModifierCombo.OPERATOR:
            # Simple node — no modifiers (or Operator only)
            node_name = node.name
            node_fn = make_node_fn(node)
            graph.add_node(node_name, node_fn, retry_policy=rp)
            if prev_node:
                graph.add_edge(prev_node, node_name)
            else:
                graph.add_edge(START, node_name)
            last_name = node_name
        case _ as unreachable:
            assert_never(unreachable)

    # Operator stacking: add interrupt check after the primary modifier
    if operator:
        last_name = _add_operator_check(graph, last_name, operator)

    return last_name


def _add_oracle_nodes(
    graph: StateGraph,
    node: Node,
    oracle: Any,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Expand Oracle modifier into fan-out generators + merge barrier."""
    field_name = field_name_for(node.name)
    collector_field = f"neo_oracle_{field_name}"

    raw_fn = make_node_fn(node)
    redirect_fn = make_oracle_redirect_fn(raw_fn, field_name, collector_field)
    merge_fn = make_oracle_merge_fn(oracle, field_name, collector_field, node.outputs,
                                    node_inputs=node.inputs)

    return _wire_oracle(graph, node.name, redirect_fn, merge_fn, oracle, prev_node, retry_policy=retry_policy)


def _add_each_nodes(
    graph: StateGraph,
    node: Node,
    each: Any,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Expand Each modifier into fan-out dispatch + barrier."""
    node_fn = make_node_fn(node)
    return _wire_each(graph, node.name, node_fn, each, prev_node, retry_policy=retry_policy)
