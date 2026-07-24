"""Wiring helpers — lower modifiers into LangGraph topology.

Extracted from compiler.py. These functions build the LangGraph node/edge
topology for Each, Oracle, Each×Oracle fusion, Loop, Branch, and Operator
modifiers. They are called by _add_node_to_graph / _add_subgraph in compiler.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import structlog
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel

from neograph._agent_cycle import make_agent_cycle_bodies, make_tool_gate_bodies
from neograph._ir_branch import _BranchNode
from neograph._ir_protocols import ConstructItem
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import normalize_outputs, primary_output_field
from neograph._oracle import (
    _amerge_variants,
    _assert_merge_fn_registered,
    _build_upstream_context,
    _merge_variants,
    make_eachoracle_redirect_fn,
)
from neograph._state_bus import StateBus, adapt_state, snapshot_state
from neograph._state_keys import StateKeys
from neograph._subconstruct import make_subgraph_fn
from neograph._trace import named
from neograph.construct import Construct
from neograph.di import _unwrap_loop_value
from neograph.errors import ConfigurationError, ExecutionError
from neograph.factory import (
    make_node_fn,
    make_portal_agent_cycle_fn,
    make_portal_dispatch_fn,
    make_portal_fn,
    make_portal_subgraph_fn,
)
from neograph.modifiers import (
    Each,
    Loop,
    ModifierCombo,
    Operator,
    Oracle,
    Portal,
    classify_modifiers,
    split_each_path,
)
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
    # A None/absent root (untaken branch arm, skip_when with no skip_value) or a
    # dotted path whose intermediate attr is None must fail CLOSED to [] so
    # each_router/flat_router route to their empty_bypass — never crash the
    # navigation before the bypass can fire.
    obj = bus.get(root)
    for part in segments:
        if obj is None:
            break
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            try:
                obj = obj[part]
            except (TypeError, KeyError, IndexError):
                obj = None
    if obj is None:
        # Diagnosable-empty: an unexpected zero-item fan-out is logged so the
        # companion ran-and-returned-None contract violation is not masked by a
        # silent empty collection.
        log.info("each_over_absent", fan_out=fan_out, path=each.over)
        return []

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


def _empty_each_bypass(field: str) -> Callable[[Any], dict]:
    """Build the empty-collection bypass body for an Each fan-out.

    Writes an empty dict to the Each ``field`` so downstream nodes proceed when
    the collection is empty. Follows the ``__loop_exit_`` pass-through pattern.
    Single source shared by ``_wire_each`` (single Each) and
    ``_add_each_oracle_fused`` (Each×Oracle fusion) so the one bypass rule cannot
    drift between the two topologies (DRY-08 / neograph-7w0d).
    """

    def empty_bypass(state: Any) -> dict:
        return {field: {}}

    return empty_bypass


def _add_arm_nodes(
    graph: StateGraph,
    nodes: list,
    *,
    checkpointer: Any = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> None:
    """Add every node of one branch arm to the graph.

    Arms can contain both Nodes and Constructs (e.g. ``self.loop()`` produces a
    Construct in the arm). Constructs are compiled with the parent checkpointer +
    condition/tool lookups threaded in — exactly like the main sub-construct path
    (``compiler.py::_add_subgraph``). Without the checkpointer an Operator
    sub-construct in an arm fails the "Operator requires a checkpointer" compile
    guard; without conditions a string Operator condition inside the arm cannot
    resolve. See neograph-faf8.

    Single source for both branch arms (DRY-07 / neograph-7w0d) — the true-arm
    and false-arm node-add loops were verbatim-identical, the exact shape the
    vn5f arm-descent primitives exist to kill. Edge wiring is handled separately
    by :func:`_wire_arm_edges`.
    """
    # Circular import: arm Constructs compile via compile(). Import here to avoid
    # the import cycle (compiler.py imports this module).
    from neograph.compiler import compile as _compile

    for item in nodes:
        if isinstance(item, Construct):
            sub_graph = _compile(
                item,
                checkpointer=checkpointer,
                _runtime=runtime,
                _scripted_lookup=scripted_lookup,
                conditions=condition_lookup,
                tool_factories=tool_factory_lookup,
            )
            subgraph_fn = make_subgraph_fn(item, sub_graph.graph)
            # `named` so the arm sub-construct's engine span reads as the construct
            # name (not the leaking `subgraph_node` __name__). See neograph-3fm1.
            graph.add_node(
                item.name,
                cast(
                    Any,
                    named(
                        subgraph_fn,
                        item.name,
                        mode="subgraph",
                        output_type=item.output.__name__ if item.output is not None else None,
                    ),
                ),
            )
        else:
            # make_node_fn already self-names its wrapper per neograph-3fm1.
            node_fn = make_node_fn(
                item,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )
            graph.add_node(item.name, node_fn)


def _wire_arm_edges(graph: StateGraph, nodes: list) -> None:
    """Wire the sequential edges within one branch arm.

    Single source for both arms (DRY-07 / neograph-7w0d) — the two duplicated
    ``range(1, len(...))`` edge loops collapse here.
    """
    for i in range(1, len(nodes)):
        graph.add_edge(nodes[i - 1].name, nodes[i].name)


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

    # Generator node (called N times via Send). `named` so the engine span reads
    # as the node (not the leaking redirect __name__). See neograph-3fm1.
    graph.add_node(gen_name, cast(Any, named(cast(Runnable, gen_fn), gen_name, mode="oracle")))

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
            log.info(
                "oracle_uneven_distribution",
                node=gen_name,
                n=oracle.n,
                models=models,
                msg=f"Uneven distribution: {oracle.n} generators across {len(models)} models",
            )
        return sends

    if prev_node:
        graph.add_conditional_edges(prev_node, oracle_router, path_map=[gen_name])
    else:
        graph.add_conditional_edges(START, oracle_router, path_map=[gen_name])

    # Merge barrier
    graph.add_node(merge_name, cast(Any, named(cast(Runnable, merge_fn), merge_name, mode="oracle_merge")), defer=True)
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

    # `named` so the fan-out node's engine span reads as the node (not the
    # leaking wrapper __name__). See neograph-3fm1.
    graph.add_node(fan_name, cast(Any, named(cast(Runnable, fan_fn), fan_name, mode="each")))

    # Empty-collection bypass: writes empty dict to the Each field so
    # downstream nodes proceed. Follows the __loop_exit_ pattern.
    graph.add_node(empty_name, cast(Any, _empty_each_bypass(field_name_for(fan_name))))

    def each_router(state: Any) -> list:
        bus = adapt_state(state)
        unique_items = _collect_each_items(bus, each, fan_out=fan_name)
        state_dict = snapshot_state(bus)

        # Empty collection: skip fan-out, route to bypass node
        if not unique_items:
            return [Send(empty_name, state_dict)]

        return [Send(fan_name, {**state_dict, StateKeys.EACH_ITEM: item}) for item in unique_items]

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
    raw_fn = make_node_fn(
        node, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup
    )
    redirect_fn = make_eachoracle_redirect_fn(
        raw_fn,
        field_name,
        collector_field,
        each.key,
        item=node,
    )
    graph.add_node(gen_name, cast(Any, named(redirect_fn, gen_name, mode="each_oracle")))

    # Empty-collection bypass for Each x Oracle fusion
    graph.add_node(empty_name, cast(Any, _empty_each_bypass(field_name)))

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
    # Dual-path per neograph-p3c7: sync + async twins share the group-collection
    # (_collect_groups) and result-shaping (_shape_merged) helpers so an LLM-judge
    # merge_prompt runs on the loop under graph.ainvoke instead of blocking it.
    def _collect_groups(state: Any) -> tuple[dict[str, list], Any]:
        from collections import defaultdict

        bus = adapt_state(state)
        # StateBus.get optional: collector is unbound until the first fused
        # generator writes a tagged result; empty-list default is the zero.
        collector = bus.get(collector_field, [])
        groups: dict[str, list] = defaultdict(list)
        for key, result in collector:
            groups[key].append(result)
        # Upstream-context for merge_prompt injection — built ONCE from the
        # barrier's state (parity with make_oracle_merge_fn's single-group path).
        upstream_context = _build_upstream_context(bus, node.inputs)
        return groups, upstream_context

    def _shape_merged(merged: dict[str, Any]) -> dict:
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

    def group_merge_barrier(state: Any, config: RunnableConfig) -> dict:
        groups, upstream_context = _collect_groups(state)
        merged: dict[str, Any] = {}
        for key, variants in groups.items():
            merged[key] = _merge_one_group(
                oracle,
                node,
                variants,
                config,
                upstream_context=upstream_context,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                state=state,
            )
        return _shape_merged(merged)

    async def agroup_merge_barrier(state: Any, config: RunnableConfig) -> dict:
        groups, upstream_context = _collect_groups(state)
        merged: dict[str, Any] = {}
        for key, variants in groups.items():
            merged[key] = await _amerge_one_group(
                oracle,
                node,
                variants,
                config,
                upstream_context=upstream_context,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                state=state,
            )
        return _shape_merged(merged)

    graph.add_node(
        barrier_name,
        cast(
            Any,
            named(
                RunnableLambda(group_merge_barrier, afunc=agroup_merge_barrier),
                barrier_name,
                mode="each_oracle_merge",
            ),
        ),
        defer=True,
    )
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
        oracle,
        variants,
        output_model,
        config,
        upstream_context=upstream_context,
        llm_config=node.llm_config,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        state_for_di=state,
    )


async def _amerge_one_group(
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
    """Async twin of :func:`_merge_one_group` per neograph-p3c7.

    Pure delegation to the async canonical merge step ``_oracle._amerge_variants``
    — same per-group call shape as the sync twin, only the seam differs.
    """
    output_model = node.outputs
    assert output_model is not None, f"Oracle merge on '{node.name}' requires outputs"
    return await _amerge_variants(
        oracle,
        variants,
        output_model,
        config,
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
            if loop.on_exhaust == "error":
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
    node_fn = make_node_fn(
        node, runtime=runtime, scripted_lookup=scripted_lookup, tool_factory_lookup=tool_factory_lookup
    )
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
        node_name,
        router,
        path_map=[reenter_target, exit_name],
    )

    return exit_name


def _contiguous_portal_mesh(nodes: list[ConstructItem], entry: Node) -> list[ConstructItem]:
    """Collect the contiguous run of Portal-modified members starting at ``entry``.

    Called by the compile walk when it reaches a mesh ENTRY. ``entry`` is located
    by identity, then the run is collected forward while each item is a
    Portal-modified member — a Node OR a sub-``Construct`` (do0d9, §3.1 site 4):
    a Construct mesh member currently TERMINATED the run, so this relaxation lets
    it be included. Contiguity is guaranteed by assembly validation
    (design §3.1 r2). Takes the node LIST as a parameter (not ``construct.nodes``),
    so it does not add a second raw ``.nodes`` walk to the compiler.
    """
    start = next(i for i, n in enumerate(nodes) if n is entry)
    members: list[ConstructItem] = []
    for item in nodes[start:]:
        if classify_modifiers(item)[0] != ModifierCombo.PORTAL:
            break
        # A dispatch-mode Portal (route="decide") is NOT a mesh member — it is a
        # standalone linear node lowered by _add_portal_dispatch (review M2). Stop
        # the run here so a dispatch node contiguous with a peer mesh is never
        # absorbed into `members` (which would mesh-wire it and skip its dispatch
        # wiring). The assembly-side collector (_validation_portal) agrees.
        km = item.modifier_set.portal
        if km is not None and km.is_dispatch:
            break
        members.append(item)
    return members


def _make_portal_subgraph_member_fn(
    sub: Construct,
    portal: Portal,
    entry_field: str,
    exit_name: str,
    *,
    max_hops: int,
    on_exhaust: str,
    entry_name: str,
    target_resolve: dict[str, str],
    checkpointer: Any = None,
    parent_state_model: type[BaseModel] | None = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> Runnable:
    """Compile a sub-Construct mesh member and wrap its boundary as a Portal fn.

    The sub-construct is compiled into its own isolated graph — the SAME
    recursive ``compile()`` threading ``_add_subgraph`` uses (checkpointer +
    runtime + scripted/condition/tool lookups + parent-derived context types) —
    and its boundary runnable is piped through ``factory.make_portal_subgraph_fn``
    (do0d9, §3.1 sites 1/4/7). No ``Command`` is constructed here (guard G1);
    the factory delegates to the shared ``_portal_route_to_command``.
    """
    # Circular import: the sub-construct compiles via compile(). Import here to
    # avoid the cycle (compiler.py imports this module), mirroring _add_arm_nodes.
    from neograph.compiler import compile as _compile

    # Build context_types from the parent state model so context fields get their
    # concrete parent types instead of Any (parity with _add_subgraph).
    _context_types: dict[str, type] | None = None
    if parent_state_model is not None:
        _context_types = {
            fname: finfo.annotation
            for fname, finfo in parent_state_model.model_fields.items()
            if finfo.annotation is not None
        }

    sub_graph = _compile(
        sub,
        checkpointer=checkpointer,
        _context_types=_context_types,
        _runtime=runtime,
        _scripted_lookup=scripted_lookup,
        conditions=condition_lookup,
        tool_factories=tool_factory_lookup,
    )
    return make_portal_subgraph_fn(
        sub,
        sub_graph.graph,
        portal,
        entry_field,
        exit_name,
        max_hops=max_hops,
        on_exhaust=on_exhaust,
        entry_name=entry_name,
        target_resolve=target_resolve,
    )


def _add_portal_mesh(
    graph: StateGraph,
    members: list[ConstructItem],
    prev_node: str | None,
    *,
    checkpointer: Any = None,
    parent_state_model: type[BaseModel] | None = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
) -> str:
    """Wire a Portal mesh: dynamic Command(goto) handoff (design §4.1, D3).

    ``members`` is the contiguous run of Portal-modified sibling Nodes at one
    construct level; ``members[0]`` is the mesh ENTRY. Unlike Loop (a conditional
    back-edge router), the mesh has NO static inter-member edges and NO router:
    each member returns ``Command(goto=peer_or_exit)`` and is registered with
    ``destinations=`` so LangGraph validates the target set at compile time. The
    single static edge into the mesh is ``prev → entry``; a pass-through exit node
    (``__handoff_exit_<entry>``, mirroring Loop's ``__loop_exit_``) is where the
    linear chain resumes, so the compile walk threads ``prev_node`` forward from
    it unchanged. Returns the exit node name.

    ENTRY-LABEL MAP (mesh-local): a DX-visible peer name
    (``member.name``) may not be the member's real LangGraph node name — an
    agent/act member's real entry is ``{member.name}__agent`` (its Portal-
    visible boundary port, per the Core Invariant: routing resolves to an
    entry, never a region's interior ``__tools``/loopback nodes). Every
    ``Command(goto=...)`` target AND the mesh's own static entry edge
    (``prev → entry``, below) resolve through this ONE map — atomic members
    map to themselves. Built as a plain local dict scoped to this mesh; the
    graph-wide generalization (reused across sub-construct boundaries) is
    left to neograph-do0d9/a37vk, not built here.
    """
    entry = members[0]
    entry_field = field_name_for(entry.name)
    exit_name = f"__handoff_exit_{entry.name}"

    # ENTRY-LABEL MAP: a DX-visible peer name may not be its real LangGraph node
    # name. An agent/act member's real entry is ``{name}__agent``; an atomic Node
    # OR a sub-``Construct`` member maps to ITSELF (a Construct compiles to ONE
    # opaque boundary node added under ``sub.name`` — do0d9). ``getattr(...,
    # "mode")`` because a ``Construct`` has no ``.mode`` field.
    entry_label_map = {
        member.name: (
            f"{member.name}__agent" if getattr(member, "mode", None) in ("agent", "act") else member.name
        )
        for member in members
    }

    # max_hops/on_exhaust are ENTRY-only knobs (T1 validation), but the wrapper
    # runs per member — source the budget from the entry and thread it into every
    # member's wrapper as closure params (design §3.4, decisions D11/D12).
    entry_portal = entry.modifier_set.portal
    assert isinstance(entry_portal, Portal)
    entry_max_hops = entry_portal.max_hops
    entry_on_exhaust = entry_portal.on_exhaust

    # Pass-through exit node — the mesh's single re-join point (design §3.1 r2).
    def handoff_exit(state: Any) -> dict:
        return {}

    graph.add_node(exit_name, handoff_exit)

    for member in members:
        portal = member.modifier_set.portal
        assert isinstance(portal, Portal)  # collected as Portal-modified
        if isinstance(member, Construct):
            # A sub-Construct mesh member (do0d9, §3.1 site 4): compile the
            # sub-construct into its OWN isolated graph exactly as _add_subgraph
            # does (threading checkpointer + runtime + lookups), then wrap the
            # boundary node with make_portal_subgraph_fn so its declared-output
            # payload drives parent routing through the SAME Command(goto) path.
            subgraph_fn = _make_portal_subgraph_member_fn(
                member,
                portal,
                entry_field,
                exit_name,
                max_hops=entry_max_hops,
                on_exhaust=entry_on_exhaust,
                entry_name=entry.name,
                target_resolve=entry_label_map,
                checkpointer=checkpointer,
                parent_state_model=parent_state_model,
                runtime=runtime,
                scripted_lookup=scripted_lookup,
                condition_lookup=condition_lookup,
                tool_factory_lookup=tool_factory_lookup,
            )
            destinations = tuple(entry_label_map.get(t, t) for t in (portal.to or ())) + (exit_name,)
            graph.add_node(member.name, cast(Any, subgraph_fn), destinations=destinations)
            continue
        # Past the Construct branch every remaining member is a Portal-modified
        # Node (atomic or agent/act) — narrow for the Node-typed wiring calls.
        assert isinstance(member, Node)
        if member.mode in ("agent", "act"):
            _add_portal_agent_cycle_member(
                graph,
                member,
                portal,
                entry_field,
                exit_name,
                prev_node=None,  # the mesh entry edge is wired once, below — not per member
                max_hops=entry_max_hops,
                on_exhaust=entry_on_exhaust,
                entry_name=entry.name,
                runtime=runtime,
                tool_factory_lookup=tool_factory_lookup,
                condition_lookup=condition_lookup,
                target_resolve=entry_label_map,
            )
            continue
        member_fn = make_portal_fn(
            member,
            portal,
            entry_field,
            exit_name,
            max_hops=entry_max_hops,
            on_exhaust=entry_on_exhaust,
            entry_name=entry.name,
            runtime=runtime,
            scripted_lookup=scripted_lookup,
            tool_factory_lookup=tool_factory_lookup,
            target_resolve=entry_label_map,
        )
        # destinations = declared peers ∪ {exit}, resolved through the
        # entry-label map so an agent/act peer's destination is its real
        # entry node name. HANDOFF_END is a route VALUE mapped to exit_name
        # inside the wrapper, so exit_name (not HANDOFF_END) is the goto
        # target that must appear here.
        destinations = tuple(entry_label_map.get(t, t) for t in (portal.to or ())) + (exit_name,)
        graph.add_node(member.name, cast(Any, member_fn), destinations=destinations)

    # The only static edge into the mesh: prev → entry, resolved through the
    # SAME entry-label map — an agent/act ENTRY's real node is
    # {entry.name}__agent, not entry.name (this is the same map applied to
    # the entry as well as every peer, not a separate mechanism).
    entry_target = entry_label_map[entry.name]
    if prev_node:
        graph.add_edge(prev_node, entry_target)
    else:
        graph.add_edge(START, entry_target)

    return exit_name


def _add_portal_dispatch(
    graph: StateGraph,
    node: Node,
    prev_node: str | None,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    tool_factory_lookup: dict[str, Callable] | None = None,
) -> str:
    """Wire a Portal DISPATCH node (``route="decide"``, design §4.2, reduced v1).

    ``on_invalid='raise'`` (default): a dispatch node is a plain LINEAR node —
    it runs its body, validates+compiles+invokes the emitted flow inside
    :func:`make_portal_dispatch_fn`, and returns a plain state-update dict. So
    it wires exactly like a bare node — a static ``prev → node`` edge in, and
    the walk threads ``prev_node`` forward so the next item adds the
    ``node → next`` edge. NO ``Command`` (keeps the G1 monopoly narrow).

    ``on_invalid='route_to_error'``: mirrors the mesh's
    pass-through-exit-node pattern (``_add_portal_mesh``) — a synthetic
    ``__dispatch_exit_<node>`` node is the single re-join point; BOTH the
    success and error paths return ``Command(goto=...)`` (never a static
    out-edge alongside a conditional Command on the same node), registered
    with ``destinations=(exit_name, error_handler)``. Returns the exit node
    name so the walk continues the linear chain from there.
    """
    portal = node.modifier_set.portal
    assert isinstance(portal, Portal)  # dispatched by the PORTAL walk arm

    if portal.on_invalid == "route_to_error":
        exit_name = f"__dispatch_exit_{node.name}"

        def dispatch_exit(state: Any) -> dict:
            return {}

        graph.add_node(exit_name, dispatch_exit)

        dispatch_fn = make_portal_dispatch_fn(
            node,
            portal,
            runtime=runtime,
            scripted_lookup=scripted_lookup,
            tool_factory_lookup=tool_factory_lookup,
            exit_name=exit_name,
        )
        assert portal.error_handler is not None  # T1 validation (route_to_error requires it)
        graph.add_node(node.name, cast(Any, dispatch_fn), destinations=(exit_name, portal.error_handler))
        if prev_node:
            graph.add_edge(prev_node, node.name)
        else:
            graph.add_edge(START, node.name)
        return exit_name

    dispatch_fn = make_portal_dispatch_fn(
        node,
        portal,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        tool_factory_lookup=tool_factory_lookup,
    )
    graph.add_node(node.name, cast(Any, dispatch_fn))
    if prev_node:
        graph.add_edge(prev_node, node.name)
    else:
        graph.add_edge(START, node.name)
    return node.name


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
        sub.name,
        router,
        path_map=[sub.name, exit_name],
    )

    return exit_name


def _add_branch_to_graph(
    graph: StateGraph,
    branch_node: _BranchNode,
    prev_node: str | None,
    *,
    checkpointer: Any = None,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    scripted_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
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
    meta = branch_node._neo_branch_meta
    cond_spec = meta.condition_spec
    true_nodes = meta.true_arm_nodes
    false_nodes = meta.false_arm_nodes

    # Add all arm nodes to the graph, then wire each arm's sequential edges.
    # Both arms go through the single-source arm-descent primitives (DRY-07 /
    # neograph-7w0d): one node-add path, one edge-wiring path, called per arm.
    arm_kwargs: dict[str, Any] = {
        "checkpointer": checkpointer,
        "runtime": runtime,
        "scripted_lookup": scripted_lookup,
        "condition_lookup": condition_lookup,
        "tool_factory_lookup": tool_factory_lookup,
    }
    _add_arm_nodes(graph, true_nodes, **arm_kwargs)
    _add_arm_nodes(graph, false_nodes, **arm_kwargs)

    _wire_arm_edges(graph, true_nodes)
    _wire_arm_edges(graph, false_nodes)

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


def _wire_agent_cycle_body(
    graph: StateGraph,
    node: Node,
    parts: dict[str, Any],
    prev_node: str | None,
    *,
    condition_lookup: dict[str, Callable] | None = None,
    parse_destinations: tuple[str, ...] | None = None,
    add_static_entry_edge: bool = True,
) -> str:
    """Shared agent/tools/gate/router wiring for ONE ReAct cycle.

    Used by both ``_add_agent_cycle`` (a linear agent/act node) and
    ``_add_portal_agent_cycle_member`` (an agent/act Portal mesh member)
    — the two lowerings diverge ONLY in the parse node's
    registration (plain ``add_node`` vs ``destinations=`` for a
    Command-returning body) and what its body returns; every other wire
    (agent/tools node registration, the optional gate arm, the 3-way router,
    the tools→agent loopback) is identical, so it is single-sourced here
    rather than copy-then-maybe-merged per call site.

    Adds three parent nodes — ``{node}__agent`` / ``{node}__tools`` /
    ``{node}__parse`` — with a 3-way conditional router and a tools→agent
    loopback. Every ReAct turn is a checkpointed superstep, so a mid-loop
    interrupt pauses at a turn boundary (turn-boundary idempotency by
    construction).

    ``add_static_entry_edge=False`` skips the ``prev_node``/``START -> agent``
    edge entirely — a NON-entry Portal mesh member is reachable only via a
    peer's ``Command(goto=...)``, never a static edge (the mesh's single
    static edge is ``prev -> entry``, wired once by ``_add_portal_mesh``
    itself, not per member).
    """
    names = parts["names"]

    agent_sync, agent_async = parts["agent"]
    tools_sync, tools_async = parts["tools"]
    parse_sync, parse_async = parts["parse"]

    # `named` so each ReAct-cycle body's engine span reads as {node}__agent /
    # {node}__tools / {node}__parse (not the leaking body __name__). See
    # neograph-3fm1.
    graph.add_node(
        names.agent, cast(Any, named(RunnableLambda(agent_sync, afunc=agent_async), names.agent, mode=node.mode))
    )
    graph.add_node(
        names.tools, cast(Any, named(RunnableLambda(tools_sync, afunc=tools_async), names.tools, mode=node.mode))
    )
    parse_runnable = named(RunnableLambda(parse_sync, afunc=parse_async), names.parse, mode=node.mode)
    if parse_destinations is not None:
        graph.add_node(names.parse, cast(Any, parse_runnable), destinations=parse_destinations)
    else:
        graph.add_node(names.parse, cast(Any, parse_runnable))

    if add_static_entry_edge:
        if prev_node:
            graph.add_edge(prev_node, names.agent)
        else:
            graph.add_edge(START, names.agent)

    base_router = parts["router"]

    if node.gate_tools_when is not None:
        # Tool-gating HITL (neograph-m6d3.4 + neograph-whq0): insert a gate node
        # on the router's tools arm so a human can approve BEFORE the {node}__tools
        # body — and its side effects — run. The gate runs the predicate; a truthy
        # result triggers interrupt(payload) (pausing at this turn-boundary
        # superstep, so the tool has not executed yet). On resume the gate HONORS
        # the decision: approve → {node}__tools; deny (fail-closed) → back to
        # {node}__agent with denial ToolMessages appended so the loop continues.
        # The decision is a Layer-1 conditional edge (gate_router), not an in-body
        # check in the tools node; the gate body lives in _agent_cycle where the
        # message channel is owned.
        gate_name = f"{node.name}__tools_gate"
        gate_condition = _resolve_condition(node.gate_tools_when, condition_lookup)
        gate_parts = make_tool_gate_bodies(node, gate_condition)

        def gated_router(state: Any) -> str:
            # Preserve the base 3-way decision, but send the tools branch through
            # the gate first.
            dest = base_router(state)
            return gate_name if dest == names.tools else dest

        graph.add_node(gate_name, gate_parts["gate"])
        graph.add_conditional_edges(
            names.agent,
            gated_router,
            path_map=[gate_name, names.parse],
        )
        graph.add_conditional_edges(
            gate_name,
            gate_parts["router"],
            path_map=[names.tools, names.agent],
        )
    else:
        # 3-way router after the agent turn: tools (loop) | parse (done/forced-final).
        graph.add_conditional_edges(
            names.agent,
            base_router,
            path_map=[names.tools, names.parse],
        )

    # ReAct loopback: after executing tools, take another agent turn.
    graph.add_edge(names.tools, names.agent)

    return names.parse


def _add_agent_cycle(
    graph: StateGraph,
    node: Node,
    prev_node: str | None,
    *,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    tool_factory_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
) -> str:
    """Expand an agent/act node into an inline ReAct cycle of supersteps.

    Mirrors the four other expanders (Each/Oracle/Branch/Loop): one IR node
    → several parent nodes + reducer channels + conditional routing. The
    node bodies live in ``_agent_cycle`` (Layer-2 cognition; no engine verb);
    the actual graph wiring is shared with the Portal mesh-member lowering
    via ``_wire_agent_cycle_body``.
    """
    parts = make_agent_cycle_bodies(node, runtime=runtime, tool_factory_lookup=tool_factory_lookup)
    return _wire_agent_cycle_body(graph, node, parts, prev_node, condition_lookup=condition_lookup)


def _add_portal_agent_cycle_member(
    graph: StateGraph,
    node: Node,
    portal: Portal,
    entry_field: str,
    exit_name: str,
    prev_node: str | None,
    *,
    max_hops: int,
    on_exhaust: str,
    entry_name: str,
    runtime: LlmRuntime = EMPTY_RUNTIME,
    tool_factory_lookup: dict[str, Callable] | None = None,
    condition_lookup: dict[str, Callable] | None = None,
    target_resolve: dict[str, str] | None = None,
) -> None:
    """Wire an agent/act Portal mesh member's ReAct cycle.

    The mesh-member counterpart of ``_add_portal_mesh``'s per-member
    ``make_portal_fn`` call for atomic members: the member's DX-visible
    Portal identity is its entry (``{node}__agent``, the destination other
    peers' ``Command(goto=...)`` must resolve to via ``target_resolve``) and
    its reconverging exit (``{node}__parse``, the ONLY node that returns a
    mesh ``Command`` — the interior ``__tools``/loopback nodes never do),
    per the Core Invariant (route-to-entry-port, emit-from-exit-port).
    Reuses ``_wire_agent_cycle_body`` for everything except the parse node's
    registration (``destinations=`` + Command-returning body, built by
    ``factory.make_portal_agent_cycle_fn``).
    """
    parts = make_portal_agent_cycle_fn(
        node,
        portal,
        entry_field,
        exit_name,
        max_hops=max_hops,
        on_exhaust=on_exhaust,
        entry_name=entry_name,
        runtime=runtime,
        tool_factory_lookup=tool_factory_lookup,
        target_resolve=target_resolve,
    )
    # destinations = declared peers ∪ {exit}, resolved through the entry-label
    # map so an agent/act peer's destination is its real entry node name —
    # mirrors the atomic member's `graph.add_node(member.name, fn,
    # destinations=...)` in `_add_portal_mesh`.
    resolve = target_resolve or {}
    destinations = tuple(resolve.get(t, t) for t in (portal.to or ())) + (exit_name,)
    # A Portal mesh member (entry or peer) is never reached via a static
    # prev-node edge — the mesh's single static edge (prev -> entry) is
    # wired once by `_add_portal_mesh` itself, resolved through the SAME
    # entry-label map; every other member is reachable only via a peer's
    # `Command(goto=...)`.
    _wire_agent_cycle_body(
        graph,
        node,
        parts,
        prev_node,
        condition_lookup=condition_lookup,
        parse_destinations=destinations,
        add_static_entry_edge=False,
    )


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
