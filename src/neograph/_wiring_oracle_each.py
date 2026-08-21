"""Oracle / Each wiring — lower Oracle, Each, and their fusion into topology.

Extracted from ``_wiring.py`` (neograph-3ffdg.1) as a pure file split: the
functions below are byte-identical to their previous definitions, only their
home changed. ``_wiring.py`` re-exports every public name here, so existing
``from neograph._wiring import ...`` call sites keep working; ``compiler.py``
imports them from this module directly, which is their real home.

Scope note: the Branch/Loop/Portal/Operator wiring helpers deliberately stayed
in ``_wiring.py``. The Oracle/Each cluster was interleaved with the branch-arm
helpers, so this extraction moved two non-contiguous ranges and left
``_add_arm_nodes`` / ``_wire_arm_edges`` in place for neograph-3ffdg.2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import structlog
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.graph import START, StateGraph
from langgraph.types import Send

from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._normalize import normalize_outputs
from neograph._oracle import (
    _amerge_variants,
    _assert_merge_fn_registered,
    _build_upstream_context,
    _merge_variants,
    make_eachoracle_redirect_fn,
)
from neograph._state_bus import StateBus, adapt_state, snapshot_state
from neograph._state_keys import StateKeys
from neograph._trace import add_traced_node, named
from neograph.factory import make_node_fn
from neograph.modifiers import Each, Oracle, split_each_path
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node

log = structlog.get_logger()

# Mirrors the alias in _wiring.py: a graph node function is either a plain
# (state, config) callable or a Runnable.
LangGraphNodeFn = Callable[[Any, RunnableConfig], dict[str, Any]] | Runnable


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


def _wire_oracle(
    graph: StateGraph,
    gen_name: str,
    gen_fn: LangGraphNodeFn,
    merge_fn: LangGraphNodeFn,
    oracle: Oracle,
    prev_node: str | None,
    subgraph_meta: dict[str, str] | None = None,
) -> str:
    """Shared Oracle wiring used by both Node and Construct paths.

    Adds generator node, oracle_router with Send, merge barrier with defer=True.
    """
    merge_name = f"merge_{gen_name}"

    # Generator node (called N times via Send). `named` so the engine span reads
    # as the node (not the leaking redirect __name__). See neograph-3fm1.
    add_traced_node(
        graph, gen_name, cast(Any, gen_fn), mode="oracle", subgraph_meta=subgraph_meta
    )

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
    subgraph_meta: dict[str, str] | None = None,
) -> str:
    """Shared Each wiring used by both Node and Construct paths.

    Adds fan-out node, each_router with Send (dotted path navigation),
    barrier with defer=True.
    """
    barrier_name = f"assemble_{fan_name}"
    empty_name = f"__each_empty_{fan_name}"

    # `named` so the fan-out node's engine span reads as the node (not the
    # leaking wrapper __name__). See neograph-3fm1.
    add_traced_node(
        graph, fan_name, cast(Any, fan_fn), mode="each", subgraph_meta=subgraph_meta
    )

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


