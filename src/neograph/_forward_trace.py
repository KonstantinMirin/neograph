"""Trace orchestration and branch merge for ForwardConstruct.

Extracted from ``forward.py`` (neograph-3ffdg.12) as a pure file split — the
functions below are unchanged except for one injected parameter, described below.

What lives here: the re-trace strategy. ``forward()`` is run N times under
different branch decisions and the resulting traces are diffed into one flat node
list carrying ``_BranchNode`` sentinels.

``_ForwardSelf`` (the shim ``self`` swapped in during tracing) stays in
``forward.py``: its ``.loop()/.each()/.ensemble()/.interrupt()`` factories
construct the DX builder classes that also stay, so moving it here would invert
the cycle rather than break it. Instead ``_run_trace`` takes it as a
``shim_factory`` parameter, threaded from ``_trace_forward``, whose only caller is
``ForwardConstruct.__init__``. Same resolution as the Portal exporter in
neograph-3ffdg.3 and the loop condition resolver in .2 — no cycle to document and
no allowlist grew. ``ForwardConstruct`` is annotation-only here, so it rides
under ``TYPE_CHECKING``.
"""

from __future__ import annotations

import operator as op_module
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from neograph._forward_proxy import (
    _BranchPoint,
    _BranchTrace,
    _ConditionProxy,
    _Proxy,
    _Tracer,
)
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from neograph.construct import Construct
from neograph.modifiers import Each
from neograph.node import Node

if TYPE_CHECKING:
    from neograph.forward import ForwardConstruct


def _run_trace(
    instance: ForwardConstruct,
    node_attrs: dict[str, Node],
    branch_decisions: dict[int, bool] | None = None,
    *,
    shim_factory: Callable[..., Any],
) -> tuple[_Tracer, list[Node | Construct]]:
    """Run a single trace pass of forward() and return (tracer, nodes)."""
    tracer = _Tracer(branch_decisions=branch_decisions)
    shim = shim_factory(node_attrs, tracer, real_self=instance)
    seed = _Proxy(source_node=None, name="forward_input", tracer=tracer)
    type(instance).forward(shim, seed)  # type: ignore[arg-type]
    nodes = _apply_loop_modifiers(tracer)
    return tracer, nodes


def _apply_loop_modifiers(tracer: _Tracer) -> list[Node | Construct]:
    """Replace loop-body nodes with Each-modified copies.

    Nodes recorded during a for-loop iteration over a proxy get an
    Each modifier attached. Non-loop nodes pass through unchanged.
    """
    if not tracer._loop_body_nodes:
        return tracer.nodes

    result = []
    for node in tracer._ordered:
        key = id(node)
        if key in tracer._loop_body_nodes:
            over_path = tracer._loop_body_nodes[key]
            node = node | Each(over=over_path, key="label")
        result.append(node)
    return result


def _trace_forward(
    instance: ForwardConstruct,
    node_attrs: dict[str, Node],
    *,
    shim_factory: Callable[..., Any],
) -> list[Node | Construct]:
    """Trace forward() to discover node call order.

    For straight-line pipelines (no if/else), returns nodes in call order.

    For branching pipelines, uses the re-trace strategy:
    1. First trace: all branches take True arm → discover true-arm nodes
    2. For each branch: re-trace with that branch flipped to False
    3. Diff traces to identify true-only and false-only nodes
    4. Build _BranchNode sentinels that the compiler lowers to conditional edges
    """
    # Pass 1: all branches True (default)
    true_tracer, true_nodes = _run_trace(instance, node_attrs, shim_factory=shim_factory)
    branches = true_tracer.branches

    if not branches:
        # Straight-line pipeline — no branches, return as-is
        return true_nodes

    # Re-trace for each branch with that branch flipped to False
    branch_traces: list[_BranchTrace] = []
    for branch in branches:
        false_tracer, false_nodes = _run_trace(
            instance,
            node_attrs,
            branch_decisions={branch.branch_id: False},
            shim_factory=shim_factory,
        )
        branch_traces.append(
            _BranchTrace(
                branch=branch,
                true_nodes=true_nodes,
                false_nodes=false_nodes,
            )
        )

    return _merge_branch_traces(true_nodes, branch_traces, branches)


def _merge_branch_traces(
    true_nodes: list[Node | Construct],
    branch_traces: list[_BranchTrace],
    branches: list[_BranchPoint],
) -> list:
    """Merge true and false trace results into a node list with branch metadata.

    For each branch, identifies:
    - Shared prefix: nodes that appear in both traces (before the divergence)
    - True-only nodes: nodes unique to the true arm
    - False-only nodes: nodes unique to the false arm
    - Shared suffix: nodes that appear in both traces (after the convergence)

    Returns a flat list containing Node instances for shared nodes and
    _BranchNode sentinels for branches. The compiler recognizes _BranchNode
    and emits conditional edges.
    """
    if len(branch_traces) == 1:
        return _merge_single_branch(
            branch_traces[0],
            branches[0],
        )

    # Multiple sequential branches: merge one at a time
    # Each branch splits the linear flow; we process them in order
    return _merge_sequential_branches(branch_traces, branches)


def _build_condition_spec(condition: _ConditionProxy | _Proxy) -> _ConditionSpec:
    """Resolve a branch condition into a _ConditionSpec.

    A _ConditionProxy carries a comparison and builds its own spec; a plain
    _Proxy used directly as a bool becomes a truthy check. Shared by both branch
    merge paths so the truthy-fallback default lives in one place.
    """
    if isinstance(condition, _ConditionProxy):
        return condition._build_runtime_condition()
    # Plain proxy used as bool — less common, create a truthy-check spec.
    return _ConditionSpec(
        source_node=condition._neo_source,
        attr_chain=[],
        op_fn=op_module.truth,
        op_str="truthy",
        threshold=None,
    )


def _merge_single_branch(
    trace: _BranchTrace,
    branch: _BranchPoint,
) -> list:
    """Merge a single branch into a node list with a _BranchNode sentinel."""
    true_names = [n.name for n in trace.true_nodes]
    false_names = [n.name for n in trace.false_nodes]

    true_set = set(true_names)
    false_set = set(false_names)

    # Find shared prefix (nodes before the branch point)
    shared_prefix = []
    for node in trace.true_nodes:
        if node.name in false_set:
            shared_prefix.append(node)
        else:
            break

    prefix_names = {n.name for n in shared_prefix}

    # True-only and false-only nodes
    true_only = [n for n in trace.true_nodes if n.name not in false_set]
    false_only = [n for n in trace.false_nodes if n.name not in true_set]

    # Build condition spec from the branch's condition
    cond_spec = _build_condition_spec(branch.condition)

    branch_meta = _BranchMeta(
        condition_spec=cond_spec,
        true_arm_nodes=true_only,
        false_arm_nodes=false_only,
    )

    # Shared suffix: nodes in both traces that aren't in prefix or branch arms
    branch_arm_names = {n.name for n in true_only} | {n.name for n in false_only}
    shared_suffix = [n for n in trace.true_nodes if n.name not in prefix_names and n.name not in branch_arm_names]

    # Build the result: prefix + branch sentinel + suffix
    result: list = list(shared_prefix)
    result.append(_BranchNode(branch_meta, branch.branch_id))
    result.extend(shared_suffix)
    return result


def _merge_sequential_branches(
    branch_traces: list[_BranchTrace],
    branches: list[_BranchPoint],
) -> list:
    """Merge multiple sequential branches.

    For sequential branches (not nested), each branch adds a _BranchNode
    sentinel at the appropriate position in the node list.
    """
    # Use the first (all-true) trace as the base ordering
    base_nodes = branch_traces[0].true_nodes

    # For each branch, compute its true-only and false-only nodes
    result: list = []
    processed_names: set[str] = set()

    for _i, (trace, branch) in enumerate(zip(branch_traces, branches, strict=True)):
        true_names = {n.name for n in trace.true_nodes}
        false_names = {n.name for n in trace.false_nodes}

        # True-only nodes for this branch
        true_only = [n for n in trace.true_nodes if n.name not in false_names]
        false_only = [n for n in trace.false_nodes if n.name not in true_names]

        # Build condition spec
        cond_spec = _build_condition_spec(branch.condition)

        branch_meta = _BranchMeta(
            condition_spec=cond_spec,
            true_arm_nodes=true_only,
            false_arm_nodes=false_only,
        )

        # Add shared nodes before this branch's divergence point
        for node in base_nodes:
            if node.name in processed_names:
                continue
            if node.name in {n.name for n in true_only}:
                # Hit the divergence — insert branch sentinel
                result.append(_BranchNode(branch_meta, branch.branch_id))
                processed_names.update(n.name for n in true_only)
                processed_names.update(n.name for n in false_only)
                break
            result.append(node)
            processed_names.add(node.name)
        else:  # pragma: no cover — defensive: divergence always found in practice
            # Branch divergence not found in remaining base nodes — append sentinel
            result.append(_BranchNode(branch_meta, branch.branch_id))
            processed_names.update(n.name for n in true_only)
            processed_names.update(n.name for n in false_only)

    # Add any remaining shared nodes from the base
    for node in base_nodes:
        if node.name not in processed_names:
            result.append(node)
            processed_names.add(node.name)

    return result
