"""Graph compiler — Construct → LangGraph StateGraph.

    graph = compile(my_construct)

Reads the Construct's node list, resolves modifiers (Oracle, Each, Operator),
and builds a LangGraph StateGraph with correct topology, checkpointing, and state bus.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from neograph.construct import Construct
from neograph.errors import CompileError, ExecutionError
from neograph.factory import (
    _tool_factory_registry,
    lookup_condition,
    make_each_redirect_fn,
    make_eachoracle_redirect_fn,
    make_node_fn,
    make_oracle_merge_fn,
    make_oracle_redirect_fn,
    make_subgraph_fn,
)
from neograph.forward import _BranchNode
from neograph.modifiers import Each, Loop, Operator, Oracle, split_each_path
from neograph.node import Node
from neograph.state import compile_state_model

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
    except ImportError:
        return  # LangGraph version doesn't have this API

    serde = getattr(checkpointer, "serde", None)
    if not isinstance(serde, JsonPlusSerializer):
        return
    if getattr(serde, "_allowed_msgpack_modules", None) is not True:
        return  # already has an explicit allowlist

    # Collect types from the state model
    allowlist = build_serde_allowlist(schemas=[state_model])
    allowlist = allowlist | SAFE_MSGPACK_TYPES

    # Replace serde with one that has the explicit allowlist
    checkpointer.serde = JsonPlusSerializer(allowed_msgpack_modules=allowlist)


def compile(construct: Construct, checkpointer: Any = None, retry_policy: Any = None) -> Any:
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
                     modifiers={n.name: [type(m).__name__ for m in n.modifiers]
                                for n in construct.nodes
                                if isinstance(n, Node) and n.modifiers})

    # Validate: Operator string conditions are registered (neograph-6dh9).
    # Check BEFORE the checkpointer guard so the real error isn't masked.
    for item in construct.nodes:
        if isinstance(item, (Node, Construct)):
            op = item.get_modifier(Operator)
            if op is not None and isinstance(op.when, str):
                lookup_condition(op.when)  # raises ConfigurationError if not registered

    # Validate: Operator requires checkpointer
    has_operator = any(
        item.has_modifier(Operator) for item in construct.nodes
        if isinstance(item, (Node, Construct))
    )
    if has_operator and checkpointer is None:
        msg = (
            f"Construct '{construct.name}' uses Operator (interrupt/resume) "
            "but no checkpointer provided. Pass checkpointer= to compile()."
        )
        raise CompileError(msg)

    # Validate: LLM nodes require configure_llm() (neograph-fn5x)
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
            msg = (
                f"Construct '{construct.name}' has LLM nodes but "
                f"{' and '.join(missing)} not set. "
                f"Call neograph.configure_llm() before compile()."
            )
            raise CompileError(msg)

    # Validate: tool factory registrations (neograph-9513)
    for item in construct.nodes:
        if isinstance(item, Node) and item.mode in ("agent", "act") and item.tools:
            for t in item.tools:
                if t.name not in _tool_factory_registry:
                    msg = (
                        f"Node '{item.name}' references tool '{t.name}' "
                        f"but no factory is registered for it. "
                        f"Use register_tool_factory('{t.name}', factory_fn) "
                        f"before compile()."
                    )
                    raise CompileError(msg)

    # Validate: output_strategy values (neograph-0b2m)
    _VALID_STRATEGIES = {"structured", "json_mode", "text"}
    for item in construct.nodes:
        if isinstance(item, Node) and item.mode in ("think", "agent", "act"):
            strategy = item.llm_config.get("output_strategy")
            if strategy is not None and strategy not in _VALID_STRATEGIES:
                msg = (
                    f"Node '{item.name}' has output_strategy='{strategy}'. "
                    f"Valid values: {', '.join(sorted(_VALID_STRATEGIES))}."
                )
                raise CompileError(msg)

    # 1. Generate state model from node I/O
    state_model = compile_state_model(construct)

    # 2. Build graph
    graph = StateGraph(state_model)

    prev_node: str | None = None

    for item in construct.nodes:
        if isinstance(item, _BranchNode):
            prev_node = _add_branch_to_graph(graph, item, prev_node)
        elif isinstance(item, Construct):
            prev_node = _add_subgraph(graph, item, prev_node, checkpointer=checkpointer, retry_policy=retry_policy)
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
    return compiled


def _add_subgraph(
    graph: StateGraph,
    sub: Construct,
    prev_node: str | None,
    checkpointer: Any = None,
    retry_policy: Any = None,
) -> str:
    """Compile a sub-Construct as an isolated subgraph node, with modifier support."""
    if sub.input is None:
        msg = f"Sub-construct '{sub.name}' has no input type. Declare input=SomeModel."
        raise CompileError(msg)

    sub_log = log.bind(subgraph=sub.name)
    sub_log.info("subgraph_compile", input=sub.input.__name__, output=sub.output.__name__)

    # Compile the sub-construct into its own graph (recursive, thread checkpointer)
    sub_graph = compile(sub, checkpointer=checkpointer, retry_policy=retry_policy)
    field_name = sub.name.replace("-", "_")

    # Build the subgraph node function via factory
    subgraph_fn = make_subgraph_fn(sub, sub_graph)

    # Check for modifiers on the Construct
    oracle = sub.get_modifier(Oracle)
    each = sub.get_modifier(Each)
    operator = sub.get_modifier(Operator)

    if oracle:
        collector_field = f"neo_oracle_{field_name}"
        redirect_fn = make_oracle_redirect_fn(subgraph_fn, field_name, collector_field)
        merge_fn = make_oracle_merge_fn(oracle, field_name, collector_field, sub.output)
        return _wire_oracle(graph, sub.name, redirect_fn, merge_fn, oracle, prev_node)

    if each:
        each_fn = make_each_redirect_fn(subgraph_fn, field_name, each)
        return _wire_each(graph, sub.name, each_fn, each, prev_node)

    # ── Loop on Construct: conditional back-edge ──
    loop = sub.get_modifier(Loop)
    if loop:
        last_name = _add_subgraph_loop(graph, sub, subgraph_fn, loop, prev_node)
        if operator:
            last_name = _add_operator_check(graph, last_name, operator)
        return last_name

    # Plain subgraph — no modifiers (or Operator handled after)
    graph.add_node(sub.name, subgraph_fn)

    if prev_node:
        graph.add_edge(prev_node, sub.name)
    else:
        graph.add_edge(START, sub.name)

    last_name = sub.name

    if operator:
        last_name = _add_operator_check(graph, sub.name, operator)

    return last_name


def _add_node_to_graph(
    graph: StateGraph,
    node: Node,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Add a single node (with its modifiers) to the graph. Returns the last node name."""

    # Retry applies to LLM-calling nodes only (think/agent/act).
    # Scripted nodes are deterministic — retrying won't help.
    rp = retry_policy if node.mode in ("think", "agent", "act") else None

    oracle = node.get_modifier(Oracle)
    each = node.get_modifier(Each)
    operator = node.get_modifier(Operator)

    # ── Each×Oracle fusion: flat M×N Send topology (neograph-tpgi) ──
    if oracle and each:
        last_name = _add_each_oracle_fused(graph, node, each, oracle, prev_node, retry_policy=rp)
        if operator:
            last_name = _add_operator_check(graph, last_name, operator)
        return last_name

    # ── Oracle: expand to fan-out + merge ──
    if oracle:
        last_name = _add_oracle_nodes(graph, node, oracle, prev_node, retry_policy=rp)
        if operator:
            last_name = _add_operator_check(graph, last_name, operator)
        return last_name

    # ── Each: expand to fan-out + barrier ──
    if each:
        last_name = _add_each_nodes(graph, node, each, prev_node, retry_policy=rp)
        if operator:
            last_name = _add_operator_check(graph, last_name, operator)
        return last_name

    # ── Loop: conditional back-edge ──
    loop = node.get_modifier(Loop)
    if loop:
        last_name = _add_loop_back_edge(graph, node, loop, prev_node, retry_policy=rp)
        if operator:
            last_name = _add_operator_check(graph, last_name, operator)
        return last_name

    # ── Simple node ──
    node_name = node.name
    node_fn = make_node_fn(node)
    graph.add_node(node_name, node_fn, retry_policy=rp)

    if prev_node:
        graph.add_edge(prev_node, node_name)
    else:
        graph.add_edge(START, node_name)

    last_name = node_name

    # ── Operator: add interrupt check after node ──
    if operator:
        last_name = _add_operator_check(graph, node_name, operator)

    return last_name


def _add_oracle_nodes(
    graph: StateGraph,
    node: Node,
    oracle: Oracle,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Expand Oracle modifier into fan-out generators + merge barrier."""
    field_name = node.name.replace("-", "_")
    collector_field = f"neo_oracle_{field_name}"

    raw_fn = make_node_fn(node)
    redirect_fn = make_oracle_redirect_fn(raw_fn, field_name, collector_field)
    merge_fn = make_oracle_merge_fn(oracle, field_name, collector_field, node.outputs)

    return _wire_oracle(graph, node.name, redirect_fn, merge_fn, oracle, prev_node, retry_policy=retry_policy)


def _add_each_nodes(
    graph: StateGraph,
    node: Node,
    each: Each,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Expand Each modifier into fan-out dispatch + barrier."""
    node_fn = make_node_fn(node)
    return _wire_each(graph, node.name, node_fn, each, prev_node, retry_policy=retry_policy)


def _add_each_oracle_fused(
    graph: StateGraph,
    node: Node,
    each: Each,
    oracle: Oracle,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Each×Oracle fusion: flat M×N Send topology (neograph-tpgi).

    Instead of nesting Each → Sub-graph → Oracle, dispatches M×N generators
    in a single router and groups by each.key in the merge barrier.

    Topology: prev → flat_router → M×N Send(gen) → group_merge(defer) → next
    """
    field_name = node.name.replace("-", "_")
    collector_field = f"neo_eachoracle_{field_name}"
    gen_name = node.name
    barrier_name = f"merge_{node.name}"

    # Generator function — tagged redirect for Each×Oracle fusion
    raw_fn = make_node_fn(node)
    redirect_fn = make_eachoracle_redirect_fn(raw_fn, field_name, collector_field, each.key)
    graph.add_node(gen_name, redirect_fn, retry_policy=retry_policy)

    # Flat router: M items × N generators = M×N Send() calls
    root, segments = split_each_path(each.over)
    models = oracle.models

    def flat_router(state: Any) -> list:
        # Navigate dotted path to collection
        obj = getattr(state, root) if hasattr(state, root) else state[root]
        for part in segments:
            obj = getattr(obj, part) if hasattr(obj, part) else obj[part]

        # Dedup items by each.key
        seen_keys: dict[str, int] = {}
        items = list(obj)
        unique_items: list = []
        for idx, item in enumerate(items):
            key_val = getattr(item, each.key, str(item))
            if key_val in seen_keys:
                log.warning("each_duplicate_key", fan_out=gen_name, key=key_val)
                continue
            seen_keys[key_val] = idx
            unique_items.append(item)

        # Dispatch M×N
        state_dict = {k: getattr(state, k) for k in state.__class__.model_fields}
        sends = []
        for item in unique_items:
            for i in range(oracle.n):
                send_state = {
                    **state_dict,
                    "neo_each_item": item,
                    "neo_oracle_gen_id": f"gen-{i}",
                }
                if models:
                    send_state["neo_oracle_model"] = models[i % len(models)]
                sends.append(Send(gen_name, send_state))
        return sends

    if prev_node:
        graph.add_conditional_edges(prev_node, flat_router, path_map=[gen_name])
    else:
        graph.add_conditional_edges(START, flat_router, path_map=[gen_name])

    # Group-merge barrier: partitions by each.key, calls merge_fn per group
    merge_fn_impl = make_oracle_merge_fn(oracle, field_name, collector_field, node.outputs)

    def group_merge_barrier(state: Any, config: RunnableConfig | None = None) -> dict:
        from collections import defaultdict
        collector = getattr(state, collector_field, [])

        # Group tagged results by each_key
        groups: dict[str, list] = defaultdict(list)
        for key, result in collector:
            groups[key].append(result)

        # Call merge per group
        merged: dict[str, Any] = {}
        for key, variants in groups.items():
            # Build a mini-state with just this group's variants for the merge_fn
            merged[key] = _merge_one_group(oracle, node, variants, config)

        return {field_name: merged}

    graph.add_node(barrier_name, group_merge_barrier, defer=True)
    graph.add_edge([gen_name], barrier_name)

    return barrier_name


def _merge_one_group(oracle: Oracle, node: Node, variants: list, config: Any) -> Any:
    """Merge one group of Oracle variants (used by Each×Oracle fusion)."""
    from neograph.factory import lookup_scripted
    from neograph.decorators import get_merge_fn_metadata, _resolve_merge_args

    output_model = node.outputs
    if isinstance(output_model, dict):
        output_model = next(iter(output_model.values()))

    if oracle.merge_prompt:
        from neograph._llm import invoke_structured
        return invoke_structured(
            model_tier=oracle.merge_model,
            prompt_template=oracle.merge_prompt,
            input_data=variants[0] if variants else None,
            output_model=output_model,
            config=config,
        )

    if oracle.merge_fn:
        meta = get_merge_fn_metadata(oracle.merge_fn)
        if meta is not None:
            user_fn, param_res = meta
            # For fused path, no state available for from_state params
            # (they'd need the outer state which we don't thread here).
            # DI params (from_input/from_config) work via config.
            args = _resolve_merge_args(param_res, config, None)
            return user_fn(variants, *args)
        else:
            scripted_merge = lookup_scripted(oracle.merge_fn)
            return scripted_merge(variants, config)

    return variants[0] if variants else None


def _add_loop_back_edge(
    graph: StateGraph,
    node: Node,
    loop: Loop,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Wire Loop modifier: conditional back-edge with iteration tracking.

    Adds the node, a loop_router conditional edge (back-edge or exit),
    and a pass-through exit node so the compile loop can wire forward normally.
    """
    node_name = node.name
    node_fn = make_node_fn(node)
    field_name = node_name.replace('-', '_')
    count_field = f'neo_loop_count_{field_name}'

    graph.add_node(node_name, node_fn, retry_policy=retry_policy)

    if prev_node:
        graph.add_edge(prev_node, node_name)
    else:
        graph.add_edge(START, node_name)

    # Resolve the when condition (string name or callable)
    if isinstance(loop.when, str):
        condition = lookup_condition(loop.when)
    else:
        condition = loop.when

    reenter_target = node_name  # always self-loop (multi-node loops use Construct)
    exit_name = f"__loop_exit_{node_name}"

    def loop_exit(state: Any) -> dict:
        return {}  # pass-through

    graph.add_node(exit_name, loop_exit)

    def loop_router(state: Any) -> str:
        count = getattr(state, count_field, 0)
        if count >= loop.max_iterations:
            if loop.on_exhaust == 'error':
                raise ExecutionError(
                    f"Loop on '{node_name}' exceeded max_iterations={loop.max_iterations}"
                )
            return exit_name  # on_exhaust='last' — exit with last result
        # The loop condition receives the node's latest output (unwrapped
        # from the append-list). This lets the user write simple lambdas
        # like `lambda draft: draft.score < 0.8` without list awareness.
        # Dict-form outputs: primary key is {field}_{first_key} (neograph-ltqj).
        if isinstance(node.outputs, dict):
            primary_key = next(iter(node.outputs))
            state_field = f"{field_name}_{primary_key}"
        else:
            state_field = field_name
        own_val = getattr(state, state_field, None)
        if isinstance(own_val, list) and own_val:
            latest = own_val[-1]
        elif isinstance(own_val, list):
            # Empty list — no output yet (e.g. skip_when with no skip_value).
            # Pass None so user conditions like `lambda d: d is None or ...` work.
            latest = None
        else:
            latest = own_val
        try:
            should_continue = condition(latest)
        except (AttributeError, TypeError) as exc:
            raise ExecutionError(
                f"Loop condition on '{node_name}' raised {type(exc).__name__} "
                f"when called with value {latest!r}: {exc}"
            ) from exc
        if should_continue:
            return reenter_target
        return exit_name

    graph.add_conditional_edges(
        node_name, loop_router, path_map=[reenter_target, exit_name],
    )

    return exit_name


def _add_subgraph_loop(
    graph: StateGraph,
    sub: Construct,
    subgraph_fn: Any,
    loop: Loop,
    prev_node: str | None,
) -> str:
    """Wire Loop modifier on a sub-construct: conditional back-edge.

    Same pattern as _add_loop_back_edge but for sub-constructs.
    The subgraph_fn runs the sub-construct; the loop_router checks
    the condition on the latest output and decides to re-enter or exit.
    """
    field_name = sub.name.replace('-', '_')
    count_field = f'neo_loop_count_{field_name}'

    graph.add_node(sub.name, subgraph_fn)

    if prev_node:
        graph.add_edge(prev_node, sub.name)
    else:
        graph.add_edge(START, sub.name)

    # Resolve condition
    if isinstance(loop.when, str):
        condition = lookup_condition(loop.when)
    else:
        condition = loop.when

    exit_name = f"__loop_exit_{sub.name}"

    def loop_exit(state: Any) -> dict:
        return {}

    graph.add_node(exit_name, loop_exit)

    def loop_router(state: Any) -> str:
        count = getattr(state, count_field, 0)
        if count >= loop.max_iterations:
            if loop.on_exhaust == 'error':
                raise ExecutionError(
                    f"Loop on '{sub.name}' exceeded max_iterations={loop.max_iterations}"
                )
            return exit_name
        val = getattr(state, field_name, None)
        if isinstance(val, list) and val:
            val = val[-1]
        try:
            should_continue = condition(val)
        except (AttributeError, TypeError) as exc:
            raise ExecutionError(
                f"Loop condition on '{sub.name}' raised {type(exc).__name__} "
                f"when called with value {val!r}: {exc}"
            ) from exc
        if should_continue:
            return sub.name
        return exit_name

    graph.add_conditional_edges(
        sub.name, loop_router, path_map=[sub.name, exit_name],
    )

    return exit_name


# ── Shared topology wiring ──────────────────────────────────────────────


def _wire_oracle(
    graph: StateGraph,
    gen_name: str,
    gen_fn: Any,
    merge_fn: Any,
    oracle: Oracle,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Shared Oracle wiring used by both Node and Construct paths.

    Adds generator node, oracle_router with Send, merge barrier with defer=True.
    """
    merge_name = f"merge_{gen_name}"

    # Generator node (called N times via Send) — retryable
    graph.add_node(gen_name, gen_fn, retry_policy=retry_policy)

    # Router that dispatches N generators
    models = oracle.models

    def oracle_router(state: Any) -> list:
        state_dict = {
            k: getattr(state, k)
            for k in state.__class__.model_fields
        }
        sends = []
        for i in range(oracle.n):
            send_state = {**state_dict, "neo_oracle_gen_id": f"gen-{i}"}
            if models:
                send_state["neo_oracle_model"] = models[i % len(models)]
            sends.append(Send(gen_name, send_state))
        if models and oracle.n % len(models) != 0:
            log.info("oracle_uneven_distribution",
                     node=gen_name, n=oracle.n, models=models,
                     msg=f"Uneven distribution: {oracle.n} generators across {len(models)} models")
        return sends

    if prev_node:
        graph.add_conditional_edges(prev_node, oracle_router, path_map=[gen_name])
    else:
        graph.add_conditional_edges(START, oracle_router, path_map=[gen_name])

    # Merge barrier
    graph.add_node(merge_name, merge_fn, defer=True)
    graph.add_edge([gen_name], merge_name)

    return merge_name


def _wire_each(
    graph: StateGraph,
    fan_name: str,
    fan_fn: Any,
    each: Each,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Shared Each wiring used by both Node and Construct paths.

    Adds fan-out node, each_router with Send (dotted path navigation),
    barrier with defer=True.
    """
    barrier_name = f"assemble_{fan_name}"

    graph.add_node(fan_name, fan_fn, retry_policy=retry_policy)

    # Router that iterates over the collection
    root, segments = split_each_path(each.over)

    def each_router(state: Any) -> list:
        # Navigate dotted path to get collection
        obj = getattr(state, root) if hasattr(state, root) else state[root]
        for part in segments:
            obj = getattr(obj, part) if hasattr(obj, part) else obj[part]

        # Dedup: keep first occurrence of each dispatch key, warn on duplicates
        seen_keys: dict[str, int] = {}
        items = list(obj)
        unique_items: list = []
        for idx, item in enumerate(items):
            key_val = getattr(item, each.key, str(item))
            if key_val in seen_keys:
                log.warning(
                    "each_duplicate_key",
                    fan_out=fan_name,
                    key=key_val,
                    kept_index=seen_keys[key_val],
                    dropped_index=idx,
                )
                continue
            seen_keys[key_val] = idx
            unique_items.append(item)

        state_dict = {
            k: getattr(state, k)
            for k in state.__class__.model_fields
        }
        return [
            Send(fan_name, {**state_dict, "neo_each_item": item})
            for item in unique_items
        ]

    if prev_node:
        graph.add_conditional_edges(prev_node, each_router, path_map=[fan_name])
    else:
        graph.add_conditional_edges(START, each_router, path_map=[fan_name])

    # Barrier node (collects fan-out results)
    def barrier_fn(state: Any) -> dict:
        return {}  # pass-through; fan-out results collected via reducer

    graph.add_node(barrier_name, barrier_fn, defer=True)
    graph.add_edge([fan_name], barrier_name)

    return barrier_name


def _add_branch_to_graph(
    graph: StateGraph,
    branch_node: _BranchNode,
    prev_node: str | None,
) -> str:
    """Lower a _BranchNode into conditional edges in the graph.

    Adds all nodes from both arms, then wires:
        prev_node → conditional_edge → (true_arm_first | false_arm_first)
        true_arm_last → join_node
        false_arm_last → join_node

    The router function evaluates the branch condition against live state
    and returns the name of the first node in the appropriate arm.
    """
    meta = branch_node._neo_branch_meta
    cond_spec = meta.condition_spec
    true_nodes = meta.true_arm_nodes
    false_nodes = meta.false_arm_nodes

    # Add all arm nodes to the graph. Arms can contain both Nodes and
    # Constructs (e.g., self.loop() produces a Construct in the arm).
    # We only add the node function here — edge wiring is handled below.
    for item in true_nodes:
        if isinstance(item, Construct):
            sub_graph = compile(item, checkpointer=None)
            subgraph_fn = make_subgraph_fn(item, sub_graph)
            graph.add_node(item.name, subgraph_fn)
        else:
            node_fn = make_node_fn(item)
            graph.add_node(item.name, node_fn)

    for item in false_nodes:
        if isinstance(item, Construct):
            sub_graph = compile(item, checkpointer=None)
            subgraph_fn = make_subgraph_fn(item, sub_graph)
            graph.add_node(item.name, subgraph_fn)
        else:
            node_fn = make_node_fn(item)
            graph.add_node(item.name, node_fn)

    # Wire sequential edges within each arm
    for i in range(1, len(true_nodes)):
        graph.add_edge(true_nodes[i - 1].name, true_nodes[i].name)

    for i in range(1, len(false_nodes)):
        graph.add_edge(false_nodes[i - 1].name, false_nodes[i].name)

    # Build the router function
    true_target = true_nodes[0].name if true_nodes else END
    false_target = false_nodes[0].name if false_nodes else END

    # Build a runtime condition evaluator from the condition spec
    source_node = cond_spec.source_node
    attr_chain = cond_spec.attr_chain
    op_fn = cond_spec.op_fn
    threshold = cond_spec.threshold

    if source_node is not None:
        field_name = source_node.name.replace("-", "_")
    else:
        field_name = None

    def branch_router(state: Any) -> str:
        """Evaluate the branch condition against live state and route."""
        if field_name is not None:
            value = getattr(state, field_name, None)
            # Navigate attribute chain (e.g., .score, .items.first)
            for attr in attr_chain:
                if value is None:
                    break
                value = getattr(value, attr, None)
        else:
            value = None

        try:
            result = op_fn(value, threshold)
        except (AttributeError, TypeError) as exc:
            raise ExecutionError(
                f"Branch condition on '{branch_node.name}' raised "
                f"{type(exc).__name__} when called with value {value!r}: {exc}"
            ) from exc
        if result:
            return true_target
        return false_target

    # Wire conditional edge from previous node
    path_map = {true_target: true_target, false_target: false_target}
    if prev_node:
        graph.add_conditional_edges(prev_node, branch_router, path_map)
    else:
        graph.add_conditional_edges(START, branch_router, path_map)

    # Create a join node that both arms converge to
    join_name = f"__join_{branch_node.name}"

    def join_fn(state: Any) -> dict:
        return {}  # pass-through

    graph.add_node(join_name, join_fn)

    # Wire arm endings to join
    true_last = true_nodes[-1].name if true_nodes else None
    false_last = false_nodes[-1].name if false_nodes else None

    if true_last:
        graph.add_edge(true_last, join_name)
    if false_last:
        graph.add_edge(false_last, join_name)

    return join_name


def _add_operator_check(
    graph: StateGraph,
    node_name: str,
    operator: Operator,
) -> str:
    """Add an interrupt check node after the given node."""
    check_name = f"{node_name}__operator"

    condition_fn = lookup_condition(operator.when)

    def operator_check(state: Any) -> dict:
        should_pause = condition_fn(state)
        if should_pause:
            human_input = interrupt(should_pause)
            return {"human_feedback": human_input}
        return {}

    graph.add_node(check_name, operator_check)
    graph.add_edge(node_name, check_name)

    return check_name
