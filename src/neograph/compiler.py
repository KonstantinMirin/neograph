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
    lookup_condition,
    make_each_redirect_fn,
    make_node_fn,
    make_oracle_merge_fn,
    make_oracle_redirect_fn,
    make_subgraph_fn,
)
from neograph.forward import _BranchNode
from neograph.modifiers import Each, Operator, Oracle, split_each_path
from neograph.node import Node
from neograph.state import compile_state_model

log = structlog.get_logger()


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

    # 3. Compile
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
    def oracle_router(state: Any) -> list:
        state_dict = {
            k: getattr(state, k)
            for k in state.__class__.model_fields
        }
        return [
            Send(gen_name, {**state_dict, "neo_oracle_gen_id": f"gen-{i}"})
            for i in range(oracle.n)
        ]

    if prev_node:
        graph.add_conditional_edges(prev_node, oracle_router)
    else:
        graph.add_conditional_edges(START, oracle_router)

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

        # Guard: detect duplicate dispatch keys before fan-out
        seen_keys: dict[str, int] = {}
        items = list(obj)
        for idx, item in enumerate(items):
            key_val = getattr(item, each.key, str(item))
            if key_val in seen_keys:
                msg = (
                    f"Each fan-out for '{fan_name}' produced duplicate key '{key_val}' "
                    f"(items at index {seen_keys[key_val]} and {idx}). "
                    f"Each item's '{each.key}' must be unique."
                )
                raise ExecutionError(msg)
            seen_keys[key_val] = idx

        state_dict = {
            k: getattr(state, k)
            for k in state.__class__.model_fields
        }
        return [
            Send(fan_name, {**state_dict, "neo_each_item": item})
            for item in items
        ]

    if prev_node:
        graph.add_conditional_edges(prev_node, each_router)
    else:
        graph.add_conditional_edges(START, each_router)

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

    # Add all arm nodes to the graph
    for node in true_nodes:
        node_fn = make_node_fn(node)
        graph.add_node(node.name, node_fn)

    for node in false_nodes:
        node_fn = make_node_fn(node)
        graph.add_node(node.name, node_fn)

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

        if op_fn(value, threshold):
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
