"""Branch wiring — lower a ``_BranchNode`` into LangGraph arm topology.

Extracted from ``_wiring.py`` (neograph-3ffdg.2) as a pure file split — the
functions below are unchanged, only their home moved. ``_wiring.py`` re-exports
them, so existing ``from neograph._wiring import ...`` call sites keep working
and ``compiler.py`` is untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import structlog
from langgraph.graph import END, START, StateGraph

from neograph._ir_branch import _BranchNode
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._state_bus import adapt_state
from neograph._subconstruct import make_subgraph_fn
from neograph._trace import subgraph_metadata
from neograph.construct import Construct
from neograph.errors import ExecutionError
from neograph.factory import make_node_fn
from neograph.naming import field_name_for

log = structlog.get_logger()

LangGraphNodeFn = Callable[[Any, Any], dict[str, Any]] | Any


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
            # NOT wrapped in `named(...)` -- same reason as the top-level
            # sub-construct: `.with_config` returns a RunnableBinding, which
            # LangGraph's subgraph walker cannot see through, hiding an arm
            # sub-construct's interior. See neograph-xunot / GH #6.
            graph.add_node(
                item.name,
                cast(Any, subgraph_fn),
                metadata=subgraph_metadata(item.name, item.output),
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
