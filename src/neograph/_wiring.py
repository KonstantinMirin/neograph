"""Wiring helpers — lower modifiers into LangGraph topology.

Extracted from compiler.py. These functions build the LangGraph node/edge
topology for Each, Oracle, Each×Oracle fusion, Loop, Branch, and Operator
modifiers. They are called by _add_node_to_graph / _add_subgraph in compiler.py.
"""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from neograph.construct import Construct
from neograph.di import _unwrap_loop_value
from neograph.errors import ExecutionError
from neograph.factory import (
    lookup_condition,
    make_node_fn,
    make_oracle_merge_fn,
    make_subgraph_fn,
)
from neograph.forward import _BranchNode
from neograph.modifiers import Each, Loop, Oracle, split_each_path
from neograph.naming import field_name_for
from neograph.node import Node

log = structlog.get_logger()


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
    empty_name = f"__each_empty_{fan_name}"

    graph.add_node(fan_name, fan_fn, retry_policy=retry_policy)

    # Empty-collection bypass: writes empty dict to the Each field so
    # downstream nodes proceed. Follows the __loop_exit_ pattern.
    def empty_bypass(state: Any) -> dict:
        return {field_name_for(fan_name): {}}

    graph.add_node(empty_name, empty_bypass)

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

        # Empty collection: skip fan-out, route to bypass node
        if not unique_items:
            state_dict = {
                k: getattr(state, k)
                for k in state.__class__.model_fields
            }
            return [Send(empty_name, state_dict)]

        state_dict = {
            k: getattr(state, k)
            for k in state.__class__.model_fields
        }
        return [
            Send(fan_name, {**state_dict, "neo_each_item": item})
            for item in unique_items
        ]

    if prev_node:
        graph.add_conditional_edges(prev_node, each_router, path_map=[fan_name, empty_name])
    else:
        graph.add_conditional_edges(START, each_router, path_map=[fan_name, empty_name])

    # Barrier node (collects fan-out results)
    def barrier_fn(state: Any) -> dict:
        return {}  # pass-through; fan-out results collected via reducer

    graph.add_node(barrier_name, barrier_fn, defer=True)
    graph.add_edge([fan_name], barrier_name)
    graph.add_edge(empty_name, barrier_name)

    return barrier_name


def _add_each_oracle_fused(
    graph: StateGraph,
    node: Node,
    each: Each,
    oracle: Oracle,
    prev_node: str | None,
    retry_policy: Any = None,
) -> str:
    """Each x Oracle fusion: flat M x N Send topology.

    Instead of nesting Each -> Sub-graph -> Oracle, dispatches M x N generators
    in a single router and groups by each.key in the merge barrier.

    Topology: prev -> flat_router -> M x N Send(gen) -> group_merge(defer) -> next
    """
    field_name = field_name_for(node.name)
    collector_field = f"neo_eachoracle_{field_name}"
    gen_name = node.name
    barrier_name = f"merge_{node.name}"
    empty_name = f"__each_empty_{node.name}"

    # Generator function — tagged redirect for Each x Oracle fusion
    from neograph.factory import make_eachoracle_redirect_fn

    raw_fn = make_node_fn(node)
    redirect_fn = make_eachoracle_redirect_fn(raw_fn, field_name, collector_field, each.key)
    graph.add_node(gen_name, redirect_fn, retry_policy=retry_policy)

    # Empty-collection bypass for Each x Oracle fusion
    def empty_bypass(state: Any) -> dict:
        return {field_name: {}}

    graph.add_node(empty_name, empty_bypass)

    # Flat router: M items x N generators = M x N Send() calls
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

        # Empty collection: skip fan-out, route to bypass
        if not unique_items:
            state_dict = {k: getattr(state, k) for k in state.__class__.model_fields}
            return [Send(empty_name, state_dict)]

        # Dispatch M x N
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
        graph.add_conditional_edges(prev_node, flat_router, path_map=[gen_name, empty_name])
    else:  # pragma: no cover — EachOracle as first node requires pre-populated state
        graph.add_conditional_edges(START, flat_router, path_map=[gen_name, empty_name])

    # Group-merge barrier: partitions by each.key, calls merge_fn per group
    merge_fn_impl = make_oracle_merge_fn(oracle, field_name, collector_field, node.outputs)

    def group_merge_barrier(state: Any, config: RunnableConfig) -> dict:
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

        # For dict-form outputs: write to per-key fields
        if isinstance(node.outputs, dict):
            update: dict[str, Any] = {}
            for each_key, per_item_result in merged.items():
                if isinstance(per_item_result, dict):
                    for output_key, val in per_item_result.items():
                        key_field = f"{field_name}_{output_key}"
                        update.setdefault(key_field, {})[each_key] = val
                else:
                    update.setdefault(field_name, {})[each_key] = per_item_result
            return update
        return {field_name: merged}

    graph.add_node(barrier_name, group_merge_barrier, defer=True)
    graph.add_edge([gen_name], barrier_name)
    graph.add_edge(empty_name, barrier_name)

    return barrier_name


def _merge_one_group(oracle: Oracle, node: Node, variants: list, config: Any) -> Any:
    """Merge one group of Oracle variants (used by Each x Oracle fusion)."""
    from neograph.decorators import _resolve_merge_args, get_merge_fn_metadata
    from neograph.factory import lookup_scripted

    output_model = node.outputs
    if isinstance(output_model, dict):
        output_model = next(iter(output_model.values()))
    assert output_model is not None, f"Oracle merge on '{node.name}' requires outputs"

    if oracle.merge_prompt:
        from neograph._llm import invoke_structured
        return invoke_structured(
            model_tier=oracle.merge_model,
            prompt_template=oracle.merge_prompt,
            input_data=variants[0] if variants else None,
            output_model=output_model,  # type: ignore[arg-type]
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

    return variants[0] if variants else None  # pragma: no cover — Oracle validation requires merge_fn or merge_prompt


def _make_loop_router(
    item_name: str,
    field_name: str,
    count_field: str,
    loop: Loop,
    condition: Any,
    exit_name: str,
    reenter_target: str,
    unwrap_fn: Any,
) -> Any:
    """Build a loop_router closure shared by Node and Construct loop wiring.

    Parameters
    ----------
    unwrap_fn:
        ``(state, field_name) -> value`` -- extracts the latest output from
        state.  Node path handles dict-form outputs and list unwrapping;
        Construct path reads the field and delegates to _unwrap_loop_value.
    """

    def loop_router(state: Any) -> str:
        count = getattr(state, count_field, 0)
        if count >= loop.max_iterations:
            if loop.on_exhaust == 'error':
                raise ExecutionError.build(
                    "loop exceeded max_iterations",
                    expected=f"convergence within {loop.max_iterations} iterations",
                    found=f"{loop.max_iterations} iterations exhausted",
                    node=item_name,
                )
            return exit_name
        val = unwrap_fn(state, field_name)
        try:
            should_continue = condition(val)
        except (AttributeError, TypeError) as exc:
            raise ExecutionError.build(
                f"loop condition raised {type(exc).__name__}",
                found=f"value {val!r}",
                hint=str(exc),
                node=item_name,
            ) from exc
        if should_continue:
            return reenter_target
        return exit_name

    return loop_router


def _node_loop_unwrap(node: Node, field_name: str) -> Any:
    """Unwrap callback for Node loop routers.

    Handles dict-form outputs (primary key) and list unwrapping from the
    append-reducer that Loop uses.
    """

    def unwrap(state: Any, _field_name: str) -> Any:
        # Dict-form outputs: primary key is {field}_{first_key}.
        if isinstance(node.outputs, dict):
            primary_key = next(iter(node.outputs))
            state_field = f"{_field_name}_{primary_key}"
        else:
            state_field = _field_name
        own_val = getattr(state, state_field, None)
        if isinstance(own_val, list) and own_val:
            return own_val[-1]
        elif isinstance(own_val, list):
            # Empty list — no output yet (e.g. skip_when with no skip_value).
            # Pass None so user conditions like `lambda d: d is None or ...` work.
            return None
        else:  # pragma: no cover — Loop reducer always produces a list
            return own_val

    return unwrap


def _construct_loop_unwrap(state: Any, field_name: str) -> Any:
    """Unwrap callback for Construct loop routers."""
    val = getattr(state, field_name, None)
    return _unwrap_loop_value(val, object)


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
    field_name = field_name_for(node_name)
    count_field = f'neo_loop_count_{field_name}'

    graph.add_node(node_name, node_fn, retry_policy=retry_policy)

    if prev_node:
        graph.add_edge(prev_node, node_name)
    else:
        graph.add_edge(START, node_name)

    if isinstance(loop.when, str):
        condition = lookup_condition(loop.when)
    else:
        condition = loop.when

    reenter_target = node_name
    exit_name = f"__loop_exit_{node_name}"

    def loop_exit(state: Any) -> dict:
        return {}

    graph.add_node(exit_name, loop_exit)

    router = _make_loop_router(
        item_name=node_name,
        field_name=field_name,
        count_field=count_field,
        loop=loop,
        condition=condition,
        exit_name=exit_name,
        reenter_target=reenter_target,
        unwrap_fn=_node_loop_unwrap(node, field_name),
    )

    graph.add_conditional_edges(
        node_name, router, path_map=[reenter_target, exit_name],
    )

    return exit_name


def _add_subgraph_loop(
    graph: StateGraph,
    sub: Construct,
    subgraph_fn: Any,
    loop: Loop,
    prev_node: str | None,
) -> str:
    """Wire Loop modifier on a sub-construct: conditional back-edge."""
    field_name = field_name_for(sub.name)
    count_field = f'neo_loop_count_{field_name}'

    graph.add_node(sub.name, subgraph_fn)

    if prev_node:
        graph.add_edge(prev_node, sub.name)
    else:
        graph.add_edge(START, sub.name)

    if isinstance(loop.when, str):
        condition = lookup_condition(loop.when)
    else:
        condition = loop.when

    exit_name = f"__loop_exit_{sub.name}"

    def loop_exit(state: Any) -> dict:
        return {}

    graph.add_node(exit_name, loop_exit)

    router = _make_loop_router(
        item_name=sub.name,
        field_name=field_name,
        count_field=count_field,
        loop=loop,
        condition=condition,
        exit_name=exit_name,
        reenter_target=sub.name,
        unwrap_fn=_construct_loop_unwrap,
    )

    graph.add_conditional_edges(
        sub.name, router, path_map=[sub.name, exit_name],
    )

    return exit_name


def _add_branch_to_graph(
    graph: StateGraph,
    branch_node: _BranchNode,
    prev_node: str | None,
) -> str:
    """Lower a _BranchNode into conditional edges in the graph.

    Adds all nodes from both arms, then wires:
        prev_node -> conditional_edge -> (true_arm_first | false_arm_first)
        true_arm_last -> join_node
        false_arm_last -> join_node

    The router function evaluates the branch condition against live state
    and returns the name of the first node in the appropriate arm.
    """
    # Circular import: _add_branch_to_graph calls compile() for sub-constructs
    # inside branch arms. Import here to avoid import cycle.
    from neograph.compiler import compile as _compile

    meta = branch_node._neo_branch_meta
    cond_spec = meta.condition_spec
    true_nodes = meta.true_arm_nodes
    false_nodes = meta.false_arm_nodes

    # Add all arm nodes to the graph. Arms can contain both Nodes and
    # Constructs (e.g., self.loop() produces a Construct in the arm).
    # We only add the node function here — edge wiring is handled below.
    for item in true_nodes:
        if isinstance(item, Construct):
            sub_graph = _compile(item, checkpointer=None)
            subgraph_fn = make_subgraph_fn(item, sub_graph)
            graph.add_node(item.name, subgraph_fn)
        else:
            node_fn = make_node_fn(item)
            graph.add_node(item.name, node_fn)

    for item in false_nodes:
        if isinstance(item, Construct):
            sub_graph = _compile(item, checkpointer=None)
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
        field_name = field_name_for(source_node.name)
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
            raise ExecutionError.build(
                f"branch condition raised {type(exc).__name__}",
                found=f"value {value!r}",
                hint=str(exc),
                node=branch_node.name,
            ) from exc
        if result:
            return true_target
        return false_target

    # Wire conditional edge from previous node
    path_map: dict[str, str] = {true_target: true_target, false_target: false_target}
    if prev_node:
        graph.add_conditional_edges(prev_node, branch_router, path_map)  # type: ignore[arg-type]
    else:
        graph.add_conditional_edges(START, branch_router, path_map)  # type: ignore[arg-type]

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
    operator: Any,
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
