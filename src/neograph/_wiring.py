"""Wiring helpers — lower modifiers into LangGraph topology.

Extracted from compiler.py. These functions build the LangGraph node/edge
topology for Each, Oracle, Each×Oracle fusion, Loop, Branch, and Operator
modifiers. They are called by _add_node_to_graph / _add_subgraph in compiler.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import structlog
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from neograph._ir_branch import _BranchNode
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import normalize_outputs, primary_output_field
from neograph._oracle import (
    _assert_merge_fn_registered,
    _build_upstream_context,
    _merge_variants,
    make_eachoracle_redirect_fn,
)
from neograph._state_bus import StateBus, adapt_state, snapshot_state
from neograph._state_keys import StateKeys
from neograph._subconstruct import make_subgraph_fn
from neograph.construct import Construct
from neograph.di import _unwrap_loop_value
from neograph.errors import ConfigurationError, ExecutionError
from neograph.factory import (
    make_node_fn,
)
from neograph.modifiers import Each, Loop, Operator, Oracle, split_each_path
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node

log = structlog.get_logger()

# A graph node function is either a plain (state, config) callable or a Runnable
# (post-Phase-1a the factory/redirect wrappers return RunnableLambda for the
# driver-selected sync/async dual path). Both are accepted by add_node.
LangGraphNodeFn = Callable[[Any, RunnableConfig], dict[str, Any]] | Runnable
LangGraphRouterFn = Callable[[Any], str]
LangGraphLoopUnwrapFn = Callable[[StateBus, str], Any]


def _collect_each_items(bus: StateBus, each: Each, *, fan_out: str) -> list:
    """Navigate the Each ``over`` dotted path and dedup the collection.

    The SINGLE source of the Each-router navigation+dedup rule, shared by
    ``each_router`` (single Each) and ``flat_router`` (Each×Oracle fusion) so
    the two topologies cannot drift. Reads the root collection through the
    StateBus; the remaining dotted segments navigate the resolved VALUE
    (``getattr(obj, part)`` is value navigation, not a state read).

    Dedup keeps the first occurrence of each ``each.key`` value and emits an
    ``each_duplicate_key`` warning (with kept/dropped indices) for the rest.
    """
    root, segments = split_each_path(each.over)
    # StateBus.get optional: the Each ``over`` root is validated at assembly
    # time; a runtime-absent root surfaces as an empty/None collection below.
    obj = bus.get(root)
    for part in segments:
        obj = getattr(obj, part) if hasattr(obj, part) else obj[part]

    seen_keys: dict[str, int] = {}
    unique_items: list = []
    for idx, item in enumerate(list(obj)):
        key_val = getattr(item, each.key, str(item))
        if key_val in seen_keys:
            log.warning(
                "each_duplicate_key",
                fan_out=fan_out,
                key=key_val,
                kept_index=seen_keys[key_val],
                dropped_index=idx,
            )
            continue
        seen_keys[key_val] = idx
        unique_items.append(item)
    return unique_items


def _wire_oracle(
    graph: StateGraph,
    gen_name: str,
    gen_fn: LangGraphNodeFn,
    merge_fn: LangGraphNodeFn,
    oracle: Oracle,
    prev_node: str | None,
) -> str:
    """Shared Oracle wiring used by both Node and Construct paths.

    Adds generator node, oracle_router with Send, merge barrier with defer=True.
    """
    merge_name = f"merge_{gen_name}"

    # Generator node (called N times via Send)
    graph.add_node(gen_name, cast(Any, gen_fn))

    # Router that dispatches N generators
    models = oracle.models

    def oracle_router(state: Any) -> list:
        state_dict = snapshot_state(adapt_state(state))
        sends = []
        for i in range(oracle.n):
            send_state = {**state_dict, StateKeys.ORACLE_GEN_ID: f"gen-{i}"}
            if models:
                send_state[StateKeys.ORACLE_MODEL] = models[i % len(models)]
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
    graph.add_node(merge_name, cast(Any, merge_fn), defer=True)
    graph.add_edge([gen_name], merge_name)

    return merge_name


def _wire_each(
    graph: StateGraph,
    fan_name: str,
    fan_fn: LangGraphNodeFn,
    each: Each,
    prev_node: str | None,
) -> str:
    """Shared Each wiring used by both Node and Construct paths.

    Adds fan-out node, each_router with Send (dotted path navigation),
    barrier with defer=True.
    """
    barrier_name = f"assemble_{fan_name}"
    empty_name = f"__each_empty_{fan_name}"

    graph.add_node(fan_name, cast(Any, fan_fn))

    # Empty-collection bypass: writes empty dict to the Each field so
    # downstream nodes proceed. Follows the __loop_exit_ pattern.
    def empty_bypass(state: Any) -> dict:
        return {field_name_for(fan_name): {}}

    graph.add_node(empty_name, empty_bypass)

    def each_router(state: Any) -> list:
        bus = adapt_state(state)
        unique_items = _collect_each_items(bus, each, fan_out=fan_name)
        state_dict = snapshot_state(bus)

        # Empty collection: skip fan-out, route to bypass node
        if not unique_items:
            return [Send(empty_name, state_dict)]

        return [
            Send(fan_name, {**state_dict, StateKeys.EACH_ITEM: item})
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
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> str:
    """Each x Oracle fusion: flat M x N Send topology.

    Instead of nesting Each -> Sub-graph -> Oracle, dispatches M x N generators
    in a single router and groups by each.key in the merge barrier.

    Topology: prev -> flat_router -> M x N Send(gen) -> group_merge(defer) -> next
    """
    field_name = field_name_for(node.name)
    collector_field = StateKeys.eachoracle_collector(field_name)
    gen_name = node.name
    barrier_name = f"merge_{node.name}"
    empty_name = f"__each_empty_{node.name}"

    # Generator function — tagged redirect for Each x Oracle fusion
    raw_fn = make_node_fn(node, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup)
    redirect_fn = make_eachoracle_redirect_fn(
        raw_fn, field_name, collector_field, each.key, item=node,
    )
    graph.add_node(gen_name, redirect_fn)

    # Empty-collection bypass for Each x Oracle fusion
    def empty_bypass(state: Any) -> dict:
        return {field_name: {}}

    graph.add_node(empty_name, empty_bypass)

    # Flat router: M items x N generators = M x N Send() calls
    models = oracle.models

    def flat_router(state: Any) -> list:
        bus = adapt_state(state)
        unique_items = _collect_each_items(bus, each, fan_out=gen_name)
        state_dict = snapshot_state(bus)

        # Empty collection: skip fan-out, route to bypass
        if not unique_items:
            return [Send(empty_name, state_dict)]

        # Dispatch M x N
        sends = []
        for item in unique_items:
            for i in range(oracle.n):
                send_state = {
                    **state_dict,
                    StateKeys.EACH_ITEM: item,
                    StateKeys.ORACLE_GEN_ID: f"gen-{i}",
                }
                if models:
                    send_state[StateKeys.ORACLE_MODEL] = models[i % len(models)]
                sends.append(Send(gen_name, send_state))
        return sends

    if prev_node:
        graph.add_conditional_edges(prev_node, flat_router, path_map=[gen_name, empty_name])
    else:  # pragma: no cover — EachOracle as first node requires pre-populated state
        graph.add_conditional_edges(START, flat_router, path_map=[gen_name, empty_name])

    # Fail-fast at compile time when a scripted merge_fn is unregistered
    # (parity with the standard Oracle merge barrier build).
    _assert_merge_fn_registered(oracle, scripted_lookup)

    # Group-merge barrier: partitions by each.key, delegates each group to the
    # canonical merge step in _oracle.py (no merge algorithm lives here).
    def group_merge_barrier(state: Any, config: RunnableConfig) -> dict:
        from collections import defaultdict
        bus = adapt_state(state)
        # StateBus.get optional: collector is unbound until the first fused
        # generator writes a tagged result; empty-list default is the zero.
        collector = bus.get(collector_field, [])

        # Group tagged results by each_key
        groups: dict[str, list] = defaultdict(list)
        for key, result in collector:
            groups[key].append(result)

        # Upstream-context for merge_prompt injection — built ONCE from the
        # barrier's state (parity with make_oracle_merge_fn's single-group path).
        upstream_context = _build_upstream_context(bus, node.inputs)

        # Call merge per group
        merged: dict[str, Any] = {}
        for key, variants in groups.items():
            merged[key] = _merge_one_group(
                oracle, node, variants, config,
                upstream_context=upstream_context,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                state=state,
            )

        # For dict-form outputs: write to per-key fields
        if normalize_outputs(node.outputs).is_dict_form:
            update: dict[str, Any] = {}
            for each_key, per_item_result in merged.items():
                if isinstance(per_item_result, dict):
                    for output_key, val in per_item_result.items():
                        key_field = output_field_name(field_name, output_key)
                        update.setdefault(key_field, {})[each_key] = val
                else:
                    update.setdefault(field_name, {})[each_key] = per_item_result
            return update
        return {field_name: merged}

    graph.add_node(barrier_name, group_merge_barrier, defer=True)
    graph.add_edge([gen_name], barrier_name)
    graph.add_edge(empty_name, barrier_name)

    return barrier_name


def _merge_one_group(
    oracle: Oracle,
    node: Node,
    variants: list,
    config: RunnableConfig,
    *,
    upstream_context: dict[str, Any] | None = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    state: Any = None,
) -> Any:
    """Merge one group of Oracle variants (Each×Oracle fusion, per-group).

    Pure delegation to the canonical merge step in ``_oracle._merge_variants``;
    this function only adapts the per-group call shape (variants pre-extracted,
    returns the raw merged value — the barrier assembles per-key). It does NOT
    re-implement any merge step. ``state`` supplies from_state DI params on the
    fused path (parity with the standard merge).
    """
    output_model = node.outputs
    assert output_model is not None, f"Oracle merge on '{node.name}' requires outputs"
    return _merge_variants(
        oracle, variants, output_model, config,
        upstream_context=upstream_context,
        llm_config=node.llm_config,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        state_for_di=state,
    )


def _make_loop_router(
    item_name: str,
    field_name: str,
    count_field: str,
    loop: Loop,
    condition: Callable[[Any], bool],
    exit_name: str,
    reenter_target: str,
    unwrap_fn: LangGraphLoopUnwrapFn,
) -> LangGraphRouterFn:
    """Build a loop_router closure shared by Node and Construct loop wiring.

    Parameters
    ----------
    unwrap_fn:
        ``(state, field_name) -> value`` -- extracts the latest output from
        state.  Node path handles dict-form outputs and list unwrapping;
        Construct path reads the field and delegates to _unwrap_loop_value.
    """

    def loop_router(state: Any) -> str:
        bus = adapt_state(state)
        # Counter bootstrap (absent/None -> 0) lives in StateBus.get_counter.
        count = bus.get_counter(count_field)
        if count >= loop.max_iterations:
            if loop.on_exhaust == 'error':
                raise ExecutionError.build(
                    "loop exceeded max_iterations",
                    expected=f"convergence within {loop.max_iterations} iterations",
                    found=f"{loop.max_iterations} iterations exhausted",
                    node=item_name,
                )
            return exit_name
        val = unwrap_fn(bus, field_name)
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


def _node_loop_unwrap(node: Node, field_name: str) -> LangGraphLoopUnwrapFn:
    """Unwrap callback for Node loop routers.

    Handles dict-form outputs (primary key) and list unwrapping from the
    append-reducer that Loop uses.
    """

    def unwrap(state: StateBus, _field_name: str) -> Any:
        # Dict-form outputs: primary value lands on {field}_{first_key}.
        state_field = primary_output_field(_field_name, node.outputs)
        # StateBus.get optional: loop-bootstrap — first router pass may have
        # not-yet-populated list; user condition expected to handle None.
        # Empty list -> None (no output yet, e.g. skip_when with no skip_value)
        # so user conditions like `lambda d: d is None or ...` work; the
        # construct-loop path delegates to the same helper.
        own_val = state.get(state_field)
        return _unwrap_loop_value(own_val, object)

    return unwrap


def _construct_loop_unwrap(state: StateBus, field_name: str) -> Any:
    """Unwrap callback for Construct loop routers.

    Receives a pre-adapted StateBus from ``loop_router``.
    """
    # StateBus.get optional: loop-bootstrap — sub-construct output absent on
    # first pass; condition handles None.
    val = state.get(field_name)
    return _unwrap_loop_value(val, object)


def _resolve_condition(
    name_or_fn: str | Callable,
    condition_lookup: dict[str, Callable] | None,
) -> Callable:
    """Resolve a condition reference: string → per-compile dict;
    callable → identity. Raises ConfigurationError when a string condition
    isn't in the per-compile dict (post-§2: no fallback registry)."""
    if not isinstance(name_or_fn, str):
        return name_or_fn
    per_compile = condition_lookup or {}
    fn = per_compile.get(name_or_fn)
    if fn is not None:
        return fn
    raise ConfigurationError.build(
        f"Condition '{name_or_fn}' not registered",
        hint=f"Pass conditions={{'{name_or_fn}': fn}} to compile().",
    )


def _add_loop_back_edge(
    graph: StateGraph,
    node: Node,
    loop: Loop,
    prev_node: str | None,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> str:
    """Wire Loop modifier: conditional back-edge with iteration tracking.

    Adds the node, a loop_router conditional edge (back-edge or exit),
    and a pass-through exit node so the compile loop can wire forward normally.
    """
    node_name = node.name
    node_fn = make_node_fn(node, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup)
    field_name = field_name_for(node_name)
    count_field = StateKeys.loop_count(field_name)

    graph.add_node(node_name, node_fn)

    if prev_node:
        graph.add_edge(prev_node, node_name)
    else:
        graph.add_edge(START, node_name)

    condition = _resolve_condition(loop.when, condition_lookup)

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
    subgraph_fn: LangGraphNodeFn,
    loop: Loop,
    prev_node: str | None,
    *,
    condition_lookup: dict[str, Callable] | None = None,
) -> str:
    """Wire Loop modifier on a sub-construct: conditional back-edge."""
    field_name = field_name_for(sub.name)
    count_field = StateKeys.loop_count(field_name)

    graph.add_node(sub.name, cast(Any, subgraph_fn))

    if prev_node:
        graph.add_edge(prev_node, sub.name)
    else:
        graph.add_edge(START, sub.name)

    condition = _resolve_condition(loop.when, condition_lookup)

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
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
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
            sub_graph = _compile(item, checkpointer=None, _runtime=runtime, _scripted_lookup=scripted_lookup, tool_factories=tool_factory_lookup)
            subgraph_fn = make_subgraph_fn(item, sub_graph.graph)
            graph.add_node(item.name, subgraph_fn)
        else:
            node_fn = make_node_fn(item, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup)
            graph.add_node(item.name, node_fn)

    for item in false_nodes:
        if isinstance(item, Construct):
            sub_graph = _compile(item, checkpointer=None, _runtime=runtime, _scripted_lookup=scripted_lookup, tool_factories=tool_factory_lookup)
            subgraph_fn = make_subgraph_fn(item, sub_graph.graph)
            graph.add_node(item.name, subgraph_fn)
        else:
            node_fn = make_node_fn(item, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup)
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
            # StateBus.get optional: the branch source field may be unbound
            # when the condition fires before its producer ran; None routes
            # to the false arm via the op_fn below.
            value = adapt_state(state).get(field_name)
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
    operator: Operator,
    *,
    condition_lookup: dict[str, Callable] | None = None,
) -> str:
    """Add an interrupt check node after the given node."""
    check_name = f"{node_name}__operator"

    condition_fn = _resolve_condition(operator.when, condition_lookup)

    def operator_check(state: Any) -> dict:
        should_pause = condition_fn(state)
        if should_pause:
            human_input = interrupt(should_pause)
            return {StateKeys.HUMAN_FEEDBACK: human_input}
        return {}

    graph.add_node(check_name, operator_check)
    graph.add_edge(node_name, check_name)

    return check_name
