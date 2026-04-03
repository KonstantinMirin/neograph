"""Graph compiler — Construct → LangGraph StateGraph.

    graph = compile(my_construct)

Reads the Construct's node list, resolves modifiers (Oracle, Replicate, Operator),
and builds a LangGraph StateGraph with correct topology, checkpointing, and state bus.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from neograph.construct import Construct
from neograph.factory import _condition_registry, make_node_fn
from neograph.modifiers import Operator, Oracle, Replicate
from neograph.node import Node
from neograph.state import compile_state_model


def compile(construct: Construct) -> Any:
    """Compile a Construct into an executable LangGraph StateGraph.

    Validates type chains, builds the graph topology from node definitions
    and modifiers, auto-generates the state bus, and returns a compiled graph.
    """
    # 1. Generate state model from node I/O
    state_model = compile_state_model(construct)

    # 2. Build graph
    graph = StateGraph(state_model)

    prev_node: str | None = None

    for node in construct.nodes:
        prev_node = _add_node_to_graph(graph, node, prev_node)

    # Final edge to END
    if prev_node:
        graph.add_edge(prev_node, END)

    # 3. Compile
    return graph.compile()


def _add_node_to_graph(
    graph: StateGraph,
    node: Node,
    prev_node: str | None,
) -> str:
    """Add a single node (with its modifiers) to the graph. Returns the last node name."""

    oracle = node.get_modifier(Oracle)
    replicate = node.get_modifier(Replicate)
    operator = node.get_modifier(Operator)

    # ── Oracle: expand to fan-out + merge ──
    if oracle:
        return _add_oracle_nodes(graph, node, oracle, prev_node)

    # ── Replicate: expand to fan-out + barrier ──
    if replicate:
        return _add_replicate_nodes(graph, node, replicate, prev_node)

    # ── Simple node ──
    node_name = node.name
    node_fn = make_node_fn(node)
    graph.add_node(node_name, node_fn)

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
) -> str:
    """Expand Oracle modifier into fan-out generators + merge barrier."""
    from langgraph.constants import Send

    gen_name = node.name
    merge_name = f"merge-{node.name}"

    node_fn = make_node_fn(node)

    # Generator node (called N times via Send)
    graph.add_node(gen_name, node_fn)

    # Router that dispatches N generators
    def oracle_router(state: Any) -> list:
        return [
            Send(gen_name, {**state.__dict__, "_generator_id": f"gen-{i}"})
            for i in range(oracle.n)
        ]

    if prev_node:
        graph.add_conditional_edges(prev_node, oracle_router)
    else:
        graph.add_conditional_edges(START, oracle_router)

    # Merge barrier (judge LLM combines N variants)
    # For now, create a produce node for the merge
    merge_node = Node(
        name=merge_name,
        mode="produce",
        output=node.output,
        model=oracle.merge_model,
        prompt=oracle.merge_prompt,
    )
    merge_fn = make_node_fn(merge_node)
    graph.add_node(merge_name, merge_fn, defer=True)
    graph.add_edge([gen_name], merge_name)

    return merge_name


def _add_replicate_nodes(
    graph: StateGraph,
    node: Node,
    replicate: Replicate,
    prev_node: str | None,
) -> str:
    """Expand Replicate modifier into fan-out dispatch + barrier."""
    from langgraph.constants import Send

    fan_name = node.name
    barrier_name = f"assemble-{node.name}"

    node_fn = make_node_fn(node)
    graph.add_node(fan_name, node_fn)

    # Router that iterates over the collection
    def replicate_router(state: Any) -> list:
        # Navigate dotted path to get collection
        obj = state
        for part in replicate.over.split("."):
            obj = getattr(obj, part) if hasattr(obj, part) else obj[part]

        return [
            Send(fan_name, {**state.__dict__, f"_replicate_item": item})
            for item in obj
        ]

    if prev_node:
        graph.add_conditional_edges(prev_node, replicate_router)
    else:
        graph.add_conditional_edges(START, replicate_router)

    # Barrier node (collects fan-out results)
    def barrier_fn(state: Any) -> dict:
        return {}  # pass-through; fan-out results collected via reducer

    graph.add_node(barrier_name, barrier_fn, defer=True)
    graph.add_edge([fan_name], barrier_name)

    return barrier_name


def _add_operator_check(
    graph: StateGraph,
    node_name: str,
    operator: Operator,
) -> str:
    """Add an interrupt check node after the given node."""
    check_name = f"{node_name}__operator"

    condition_fn = _condition_registry.get(operator.when)
    if condition_fn is None:
        msg = f"Condition '{operator.when}' not registered. Use register_condition()."
        raise ValueError(msg)

    def operator_check(state: Any) -> dict:
        should_pause = condition_fn(state)
        if should_pause:
            human_input = interrupt(should_pause)
            return {"human_feedback": human_input}
        return {}

    graph.add_node(check_name, operator_check)
    graph.add_edge(node_name, check_name)

    return check_name
