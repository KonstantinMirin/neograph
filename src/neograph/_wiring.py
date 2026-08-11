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
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from neograph._agent_cycle import make_agent_cycle_bodies, make_tool_gate_bodies
from neograph._ir_protocols import ConstructItem
from neograph._llm_runtime import EMPTY_RUNTIME, LlmRuntime
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph._portal_route import MeshContext, MeshDeps, PortalRouteSpec
from neograph._state_bus import StateBus
from neograph._state_keys import StateKeys
from neograph._trace import named
from neograph.construct import Construct
from neograph.errors import ConfigurationError
from neograph.factory import (
    make_node_fn,
    make_portal_agent_cycle_fn,
    make_portal_agent_cycle_tool_handoff_fn,
    make_portal_approval_fn,
    make_portal_dispatch_fn,
    make_portal_fn,
    make_portal_subgraph_fn,
)
from neograph.modifiers import (
    Loop,
    Operator,
    Portal,
    PrimaryShape,
    _group_portal_members,
    primary_shape,
)
from neograph.naming import field_name_for
from neograph.node import Node

log = structlog.get_logger()

# --- extracted clusters (neograph-3ffdg.2), re-exported so existing
# --- `from neograph._wiring import ...` call sites keep resolving unchanged.
from neograph._wiring_branch import (  # noqa: E402,F401
    _add_arm_nodes,
    _add_branch_to_graph,
    _wire_arm_edges,
)
from neograph._wiring_loop import (  # noqa: E402,F401
    _construct_loop_unwrap,
    _make_loop_router,
    _node_loop_unwrap,
)

# --- Oracle/Each cluster: extracted to _wiring_oracle_each.py (neograph-3ffdg.1) ---
# Re-exported here so existing `from neograph._wiring import _wire_oracle` (and
# the six siblings) keep resolving. compiler.py imports them from their real
# home, _wiring_oracle_each. Pure file split: no behavior changed.
from neograph._wiring_oracle_each import (  # noqa: E402
    _add_each_oracle_fused,
    _amerge_one_group,
    _collect_each_items,
    _empty_each_bypass,
    _merge_one_group,
    _wire_each,
    _wire_oracle,
)

__all__ = [
    "_add_each_oracle_fused",
    "_amerge_one_group",
    "_collect_each_items",
    "_empty_each_bypass",
    "_merge_one_group",
    "_wire_each",
    "_wire_oracle",
]

# A graph node function is either a plain (state, config) callable or a Runnable
# (post-Phase-1a the factory/redirect wrappers return RunnableLambda for the
# driver-selected sync/async dual path). Both are accepted by add_node.
LangGraphNodeFn = Callable[[Any, RunnableConfig], dict[str, Any]] | Runnable
LangGraphRouterFn = Callable[[Any], str]
LangGraphLoopUnwrapFn = Callable[[StateBus, str], Any]


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
    entry_portal = entry.modifier_set.portal
    entry_group = entry_portal.name if entry_portal is not None else None
    start = next(i for i, n in enumerate(nodes) if n is entry)
    candidates: list[ConstructItem] = []
    for item in nodes[start:]:
        if primary_shape(item) is not PrimaryShape.PORTAL:
            break
        # A dispatch-mode Portal (route="decide") is NOT a mesh member — it is a
        # standalone linear node lowered by _add_portal_dispatch (review M2). Stop
        # the run here so a dispatch node contiguous with a peer mesh is never
        # absorbed into `candidates` (which would mesh-wire it and skip its
        # dispatch wiring). The assembly-side collector (_validation_portal) agrees.
        if portal_member_class(item) is PortalMemberClass.DISPATCH:
            break
        candidates.append(item)
    # neograph-fefar: `candidates` may span >1 NAMED mesh if a different-named
    # mesh sits immediately adjacent (no gap) -- _group_portal_members (the SAME
    # shared grouping helper the validator/normalizer use, never a re-derived
    # inline grouping) isolates just the entry's own group. Already validated
    # contiguous WITHIN itself by _check_portal_mesh, so this lookup is safe.
    return _group_portal_members(candidates)[entry_group]


def _make_portal_subgraph_member_fn(
    sub: Construct,
    portal: Portal,
    ctx: MeshContext,
    deps: MeshDeps,
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
    if deps.parent_state_model is not None:
        _context_types = {
            fname: finfo.annotation
            for fname, finfo in deps.parent_state_model.model_fields.items()
            if finfo.annotation is not None
        }

    sub_graph = _compile(
        sub,
        checkpointer=deps.checkpointer,
        _context_types=_context_types,
        _runtime=deps.runtime,
        _scripted_lookup=deps.scripted_lookup,
        conditions=deps.condition_lookup,
        tool_factories=deps.tool_factory_lookup,
    )
    spec = PortalRouteSpec.for_sub_construct(sub, portal, ctx)
    return make_portal_subgraph_fn(sub, sub_graph.graph, portal, spec, ctx)


# --- Portal member-class adapter table (neograph-dgbqv.4, P9) ----------------
#
# Replaces a hand-written if/elif chain in `_add_portal_mesh` with a table
# total over every REACHABLE `PortalMemberClass` (DISPATCH is excluded --
# `_contiguous_portal_mesh` breaks on a dispatch-mode Portal, so it can never
# be a mesh member; see `_portal_member.py`'s module docstring). A sixth
# member class therefore cannot be added without an adapter -- the table's
# totality is enforced by `tests/test_guards_portal_route_plumbing.py`.
#
# Each adapter has ONE uniform signature `(graph, member, portal, ctx, deps)
# -> None`. ATOMIC and ATOMIC_OPERATOR share `_atomic_adapter`: the body
# already branches on `member.modifier_set.operator is not None`, so the two
# classes need no separate code path -- only two separate table KEYS pointing
# at it (keeping the split at the table, not inside the body). Likewise
# AGENT_CYCLE_OUTPUT and AGENT_CYCLE_TOOL share `_agent_cycle_adapter`, which
# defers to `_add_portal_agent_cycle_member`'s own `portal_member_class`
# check for the tool-vs-output distinction.


def _atomic_adapter(graph: StateGraph, member: ConstructItem, portal: Portal, ctx: MeshContext, deps: MeshDeps) -> None:
    """ATOMIC and ATOMIC_OPERATOR: a plain node, or one with an Operator
    approval gate spliced onto its outgoing routes (neograph-kdr1u, D4)."""
    assert isinstance(member, Node)
    operator = member.modifier_set.operator
    approve_name = f"{member.name}__approve" if operator is not None else None
    spec = PortalRouteSpec.for_node(member, portal, ctx, approve_name=approve_name)
    member_fn = make_portal_fn(
        member,
        portal,
        spec,
        ctx,
        runtime=deps.runtime,
        scripted_lookup=deps.scripted_lookup,
        tool_factory_lookup=deps.tool_factory_lookup,
    )
    if approve_name is not None:
        assert operator is not None  # approve_name is derived from operator above
        # neograph-kdr1u (D4 lift): the member's OWN destinations become
        # ONLY the approval node + exit -- ALL peer routes now detour
        # through {member}__approve (HANDOFF_END stays direct/unguarded,
        # wired inside the member's own Command via _portal_route_to_command).
        # The approval node's OWN destinations are the declared peers ∪
        # {exit} (whichever peer gets approved).
        assert spec.proposed_field is not None
        approval_fn = make_portal_approval_fn(
            member.name,
            operator,
            count_field=ctx.count_field,
            proposed_field=spec.proposed_field,
            exit_name=ctx.exit_name,
            condition_lookup=deps.condition_lookup,
        )
        approval_destinations = ctx.destinations_for(portal)
        graph.add_node(approve_name, cast(Any, approval_fn), destinations=approval_destinations)
        graph.add_node(member.name, cast(Any, member_fn), destinations=(approve_name, ctx.exit_name))
    else:
        # destinations = declared peers ∪ {exit}, resolved through the
        # entry-label map so an agent/act peer's destination is its real
        # entry node name. HANDOFF_END is a route VALUE mapped to exit_name
        # inside the wrapper, so exit_name (not HANDOFF_END) is the goto
        # target that must appear here.
        graph.add_node(member.name, cast(Any, member_fn), destinations=ctx.destinations_for(portal))


def _agent_cycle_adapter(graph: StateGraph, member: ConstructItem, portal: Portal, ctx: MeshContext, deps: MeshDeps) -> None:
    """AGENT_CYCLE_OUTPUT and AGENT_CYCLE_TOOL: an agent/act mesh member's
    ReAct cycle. The mesh entry edge is wired once, below -- not per member."""
    assert isinstance(member, Node)
    _add_portal_agent_cycle_member(graph, member, portal, ctx, deps, prev_node=None)


def _sub_construct_adapter(graph: StateGraph, member: ConstructItem, portal: Portal, ctx: MeshContext, deps: MeshDeps) -> None:
    """SUB_CONSTRUCT: a whole ``Construct`` mesh member (do0d9, §3.1 site 4)."""
    assert isinstance(member, Construct)
    subgraph_fn = _make_portal_subgraph_member_fn(member, portal, ctx, deps)
    graph.add_node(member.name, cast(Any, subgraph_fn), destinations=ctx.destinations_for(portal))


Adapter = Callable[[StateGraph, ConstructItem, Portal, MeshContext, MeshDeps], None]

_PORTAL_MEMBER_ADAPTERS: dict[PortalMemberClass, Adapter] = {
    PortalMemberClass.ATOMIC: _atomic_adapter,
    PortalMemberClass.ATOMIC_OPERATOR: _atomic_adapter,
    PortalMemberClass.AGENT_CYCLE_OUTPUT: _agent_cycle_adapter,
    PortalMemberClass.AGENT_CYCLE_TOOL: _agent_cycle_adapter,
    PortalMemberClass.SUB_CONSTRUCT: _sub_construct_adapter,
}


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
    ``destinations=`` (NOT validated by LangGraph, neograph-dgbqv.7 -- rendering
    only; keep it complete, never lean on it). The single static edge into the
    mesh is ``prev → entry``; a pass-through exit node (``__handoff_exit_<entry>``,
    mirroring Loop's ``__loop_exit_``) is where the
    linear chain resumes, so the compile walk threads ``prev_node`` forward from
    it unchanged. Returns the exit node name.

    Every fact the mesh members need (the entry-label map, the hop budget, the
    shared channel/counter keys, the resolved destination tuples) is derived
    ONCE into a ``MeshContext`` (neograph-dgbqv.4, P9) rather than re-derived
    per member -- dispatch to the right per-member wiring is a table lookup on
    ``PortalMemberClass``, not an if/elif chain.
    """
    ctx = MeshContext.build(members)
    deps = MeshDeps(
        checkpointer=checkpointer,
        parent_state_model=parent_state_model,
        runtime=runtime,
        scripted_lookup=scripted_lookup,
        condition_lookup=condition_lookup,
        tool_factory_lookup=tool_factory_lookup,
    )

    # Pass-through exit node — the mesh's single re-join point (design §3.1 r2).
    def handoff_exit(state: Any) -> dict:
        return {}

    graph.add_node(ctx.exit_name, handoff_exit)

    for member in members:
        portal = member.modifier_set.portal
        assert isinstance(portal, Portal)  # collected as Portal-modified
        member_class = portal_member_class(member)
        adapter = _PORTAL_MEMBER_ADAPTERS.get(member_class) if member_class is not None else None
        if adapter is None:
            raise ConfigurationError.build(
                "Unreachable PortalMemberClass in a mesh",
                found=f"{member_class!r} for member {member.name!r}",
                hint="DISPATCH cannot be a mesh member; every other class has a declared adapter",
            )
        adapter(graph, member, portal, ctx, deps)

    # The only static edge into the mesh: prev → entry, resolved through the
    # SAME entry-label map — an agent/act ENTRY's real node is
    # {entry.name}__agent, not entry.name (this is the same map applied to
    # the entry as well as every peer, not a separate mechanism).
    entry_target = ctx.entry_label_map[ctx.entry_name]
    if prev_node:
        graph.add_edge(prev_node, entry_target)
    else:
        graph.add_edge(START, entry_target)

    return ctx.exit_name


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


def _wire_agent_cycle_body(
    graph: StateGraph,
    node: Node,
    parts: dict[str, Any],
    prev_node: str | None,
    *,
    condition_lookup: dict[str, Callable] | None = None,
    parse_destinations: tuple[str, ...] | None = None,
    tools_destinations: tuple[str, ...] | None = None,
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

    ``tools_destinations`` (a tool-triggered Portal member, design
    portal-tool-triggered-handoff §3.4): register ``{node}__tools`` as a
    ``Command``-emitting node with these ``destinations=`` (declared peers ∪
    ``{node}__agent`` loopback) and SKIP the static ``tools -> agent`` edge.
    LangGraph does NOT reject a node that has both a static out-edge AND a
    ``destinations=``-registered ``Command`` body — it silently double-executes
    BOTH targets in one superstep (verified live), so the static edge MUST be
    omitted, not merely tolerated. ``None`` (every non-tool-triggered member)
    keeps the plain tools node + static loopback, unchanged.

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
    tools_runnable = named(RunnableLambda(tools_sync, afunc=tools_async), names.tools, mode=node.mode)
    if tools_destinations is not None:
        graph.add_node(names.tools, cast(Any, tools_runnable), destinations=tools_destinations)
    else:
        graph.add_node(names.tools, cast(Any, tools_runnable))
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

    # ReAct loopback: after executing tools, take another agent turn. A
    # tool-triggered member (tools_destinations set) expresses this loopback as a
    # dynamic Command(goto={node}__agent) from its Command-emitting tools body
    # instead — a static edge here would coexist with the destinations= registration
    # and silently double-execute (design portal-tool-triggered-handoff §3.4), so it
    # is omitted for that member class only.
    if tools_destinations is None:
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
    ctx: MeshContext,
    deps: MeshDeps,
    *,
    prev_node: str | None,
) -> None:
    """Wire an agent/act Portal mesh member's ReAct cycle.

    The mesh-member counterpart of ``_add_portal_mesh``'s per-member
    ``make_portal_fn`` call for atomic members: the member's DX-visible
    Portal identity is its entry (``{node}__agent``, the destination other
    peers' ``Command(goto=...)`` must resolve to via ``ctx.entry_label_map``)
    and its reconverging exit (``{node}__parse``, the ONLY node that returns a
    mesh ``Command`` — the interior ``__tools``/loopback nodes never do),
    per the Core Invariant (route-to-entry-port, emit-from-exit-port).
    Reuses ``_wire_agent_cycle_body`` for everything except the parse node's
    registration (``destinations=`` + Command-returning body, built by
    ``factory.make_portal_agent_cycle_fn``).

    A ``trigger="tool"`` member (design portal-tool-triggered-handoff §3.4)
    additionally lowers ``{node}__tools`` into a ``Command``-emitting handoff exit
    (``factory.make_portal_agent_cycle_tool_handoff_fn`` + ``tools_destinations=``)
    with NO static ``tools -> agent`` edge; its ``parse`` node is still wired
    exactly like a ``trigger="output"`` member (the normal-completion exit).
    """
    # parse_destinations = declared peers ∪ {exit} for BOTH trigger kinds —
    # mirrors the atomic member's `graph.add_node(member.name, fn,
    # destinations=...)` in `_add_portal_mesh`.
    parse_destinations = ctx.destinations_for(portal)
    tools_destinations: tuple[str, ...] | None = None
    if portal_member_class(node) is PortalMemberClass.AGENT_CYCLE_TOOL:
        spec = PortalRouteSpec.for_tool_member(node, portal, ctx)
        parts = make_portal_agent_cycle_tool_handoff_fn(
            node,
            portal,
            spec,
            ctx,
            runtime=deps.runtime,
            tool_factory_lookup=deps.tool_factory_lookup,
        )
        # Peer entry, mesh EXIT, or back to {node}__agent — all dynamic, so the static
        # tools -> agent edge is dropped (§3.4). peers ∪ exit: the exit IS emitted (dgbqv.7).
        tools_destinations = ctx.destinations_for(portal) + (parts["names"].agent,)
    else:
        spec = PortalRouteSpec.for_node(node, portal, ctx)
        parts = make_portal_agent_cycle_fn(
            node,
            portal,
            spec,
            ctx,
            runtime=deps.runtime,
            tool_factory_lookup=deps.tool_factory_lookup,
        )
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
        condition_lookup=deps.condition_lookup,
        parse_destinations=parse_destinations,
        tools_destinations=tools_destinations,
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
