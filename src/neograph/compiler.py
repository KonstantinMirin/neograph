"""Graph compiler — Construct → LangGraph StateGraph.

    graph = compile(my_construct)

Reads the Construct's node list, resolves modifiers (Oracle, Each, Operator),
and builds a LangGraph StateGraph with correct topology, checkpointing, and state bus.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from neograph.construct import Construct
from neograph.factory import _condition_registry, _scripted_registry, make_node_fn
from neograph.modifiers import Operator, Oracle, Each
from neograph.node import Node
from neograph.state import compile_state_model

log = structlog.get_logger()


def compile(construct: Construct, checkpointer: Any = None) -> Any:
    """Compile a Construct into an executable LangGraph StateGraph.

    Args:
        construct: The Construct to compile.
        checkpointer: LangGraph checkpointer for persistence/resume support.
                      Required if any node uses Operator (interrupt/resume).
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
        raise ValueError(msg)

    # 1. Generate state model from node I/O
    state_model = compile_state_model(construct)

    # 2. Build graph
    graph = StateGraph(state_model)

    prev_node: str | None = None

    for item in construct.nodes:
        if isinstance(item, Construct):
            prev_node = _add_subgraph(graph, item, prev_node, checkpointer=checkpointer)
        else:
            prev_node = _add_node_to_graph(graph, item, prev_node)

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
) -> str:
    """Compile a sub-Construct as an isolated subgraph node, with modifier support."""
    from langchain_core.runnables import RunnableConfig

    from neograph.runner import _strip_internals

    if sub.input is None:
        msg = f"Sub-construct '{sub.name}' must declare input type."
        raise ValueError(msg)

    sub_log = log.bind(subgraph=sub.name)
    sub_log.info("subgraph_compile", input=sub.input.__name__, output=sub.output.__name__)

    # Compile the sub-construct into its own graph (recursive, thread checkpointer)
    sub_graph = compile(sub, checkpointer=checkpointer)
    field_name = sub.name.replace("-", "_")

    # Build the subgraph node function — runs the sub-pipeline in isolation
    def _make_subgraph_fn():
        def subgraph_node(state: Any, config: RunnableConfig) -> dict:
            sub_log.info("subgraph_start")

            # Extract input from parent state by type
            input_data = None
            if isinstance(state, dict):
                for val in state.values():
                    if val is not None and isinstance(val, sub.input):
                        input_data = val
                        break
            else:
                for attr_name in state.__class__.model_fields:
                    val = getattr(state, attr_name, None)
                    if val is not None and isinstance(val, sub.input):
                        input_data = val
                        break

            # Run sub-graph with isolated state
            sub_input: dict[str, Any] = {"node_id": state.get("node_id", "") if isinstance(state, dict) else getattr(state, "node_id", "")}
            if input_data is not None:
                sub_input["neo_subgraph_input"] = input_data

            sub_result = _strip_internals(sub_graph.invoke(sub_input, config=config))

            # Extract the declared output type from sub result
            output_val = None
            for val in sub_result.values():
                if isinstance(val, sub.output):
                    output_val = val
                    break

            sub_log.info("subgraph_complete")
            return {field_name: output_val}

        subgraph_node.__name__ = field_name
        return subgraph_node

    subgraph_fn = _make_subgraph_fn()

    # Check for modifiers on the Construct
    oracle = sub.get_modifier(Oracle)
    each = sub.get_modifier(Each)
    operator = sub.get_modifier(Operator)

    if oracle:
        return _add_oracle_subgraph(graph, sub, oracle, subgraph_fn, field_name, prev_node)
    if each:
        return _add_each_subgraph(graph, sub, each, subgraph_fn, field_name, prev_node)

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


def _add_oracle_subgraph(
    graph: StateGraph,
    sub: Construct,
    oracle: Oracle,
    subgraph_fn: Any,
    field_name: str,
    prev_node: str | None,
) -> str:
    """Oracle on a Construct: run the sub-pipeline N times, merge outputs."""
    from langgraph.types import Send

    gen_name = sub.name
    merge_name = f"merge_{sub.name}"
    collector_field = f"neo_oracle_{field_name}"

    # The subgraph function already writes to {field_name}
    # For Oracle, we need it to write to the collector instead
    def oracle_subgraph_fn(state: Any, config=None) -> dict:
        result = subgraph_fn(state, config) if config else subgraph_fn(state)
        # Redirect output to collector
        val = result.get(field_name)
        return {collector_field: val} if val is not None else {}

    graph.add_node(gen_name, oracle_subgraph_fn)

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
    if oracle.merge_prompt:
        from langchain_core.runnables import RunnableConfig

        def merge_fn(state: Any, config: RunnableConfig) -> dict:
            from neograph._llm import invoke_structured

            results = getattr(state, collector_field, [])
            return {field_name: invoke_structured(
                model_tier=oracle.merge_model,
                prompt_template=oracle.merge_prompt,
                input_data=results,
                output_model=sub.output,
                config=config,
            )}
    else:
        from langchain_core.runnables import RunnableConfig

        scripted_merge = _scripted_registry.get(oracle.merge_fn)
        if scripted_merge is None:
            msg = f"Merge function '{oracle.merge_fn}' not registered."
            raise ValueError(msg)

        def merge_fn(state: Any, config: RunnableConfig) -> dict:
            results = getattr(state, collector_field, [])
            return {field_name: scripted_merge(results, config)}

    graph.add_node(merge_name, merge_fn, defer=True)
    graph.add_edge([gen_name], merge_name)

    return merge_name


def _add_each_subgraph(
    graph: StateGraph,
    sub: Construct,
    each: Each,
    subgraph_fn: Any,
    field_name: str,
    prev_node: str | None,
) -> str:
    """Each on a Construct: run the sub-pipeline per collection item."""
    from langgraph.types import Send

    fan_name = sub.name
    barrier_name = f"assemble_{sub.name}"

    # The subgraph function runs per-item; wrap to key the result
    def each_subgraph_fn(state: Any, config=None) -> dict:
        # Get the item being processed
        each_item = state.get("neo_each_item") if isinstance(state, dict) else getattr(state, "neo_each_item", None)

        # Override the subgraph input with the specific item
        from langchain_core.runnables import RunnableConfig as RC
        result = subgraph_fn(state, config) if config else subgraph_fn(state)
        val = result.get(field_name)

        if val is not None and each_item is not None:
            key_val = getattr(each_item, each.key, str(each_item))
            return {field_name: {key_val: val}}
        return result

    graph.add_node(fan_name, each_subgraph_fn)

    def each_router(state: Any) -> list:
        obj = state
        for part in each.over.split("."):
            obj = getattr(obj, part) if hasattr(obj, part) else obj[part]

        state_dict = {
            k: getattr(state, k)
            for k in state.__class__.model_fields
        }
        return [
            Send(fan_name, {**state_dict, "neo_each_item": item})
            for item in obj
        ]

    if prev_node:
        graph.add_conditional_edges(prev_node, each_router)
    else:
        graph.add_conditional_edges(START, each_router)

    def barrier_fn(state: Any) -> dict:
        return {}

    graph.add_node(barrier_name, barrier_fn, defer=True)
    graph.add_edge([fan_name], barrier_name)

    return barrier_name


def _add_node_to_graph(
    graph: StateGraph,
    node: Node,
    prev_node: str | None,
) -> str:
    """Add a single node (with its modifiers) to the graph. Returns the last node name."""

    oracle = node.get_modifier(Oracle)
    replicate = node.get_modifier(Each)
    operator = node.get_modifier(Operator)

    # ── Oracle: expand to fan-out + merge ──
    if oracle:
        return _add_oracle_nodes(graph, node, oracle, prev_node)

    # ── Each: expand to fan-out + barrier ──
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
    from langgraph.types import Send

    gen_name = node.name
    merge_name = f"merge_{node.name}"
    field_name = node.name.replace("-", "_")
    collector_field = f"neo_oracle_{field_name}"

    raw_fn = make_node_fn(node)

    # Wrap node function to redirect output to collector field
    from langchain_core.runnables import RunnableConfig

    def node_fn(state: Any, config: RunnableConfig) -> dict:
        result = raw_fn(state, config)
        val = result.get(field_name)
        if val is not None:
            return {collector_field: val}
        return result

    node_fn.__name__ = raw_fn.__name__

    # Generator node (called N times via Send)
    graph.add_node(gen_name, node_fn)

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

    # Merge barrier — combines fan-out results, writes to consumer-facing field
    if oracle.merge_prompt:
        from langchain_core.runnables import RunnableConfig

        def merge_fn(state: Any, config: RunnableConfig) -> dict:
            from neograph._llm import invoke_structured

            results = getattr(state, collector_field, [])
            return {field_name: invoke_structured(
                model_tier=oracle.merge_model,
                prompt_template=oracle.merge_prompt,
                input_data=results,
                output_model=node.output,
                config=config,
            )}
    else:
        from langchain_core.runnables import RunnableConfig

        scripted_merge = _scripted_registry.get(oracle.merge_fn)
        if scripted_merge is None:
            msg = f"Merge function '{oracle.merge_fn}' not registered. Use register_scripted()."
            raise ValueError(msg)

        def merge_fn(state: Any, config: RunnableConfig) -> dict:
            results = getattr(state, collector_field, [])
            return {field_name: scripted_merge(results, config)}

    graph.add_node(merge_name, merge_fn, defer=True)
    graph.add_edge([gen_name], merge_name)

    return merge_name


def _add_replicate_nodes(
    graph: StateGraph,
    node: Node,
    replicate: Each,
    prev_node: str | None,
) -> str:
    """Expand Each modifier into fan-out dispatch + barrier."""
    from langgraph.types import Send

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

        state_dict = {
            k: getattr(state, k)
            for k in state.__class__.model_fields
        }
        return [
            Send(fan_name, {**state_dict, "neo_each_item": item})
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
