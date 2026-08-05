"""Portal-mesh / agent-cycle recursion-limit budget.

Extracted from ``runner.py`` (neograph-3ffdg.9) as a pure file split — the
functions and constants below are unchanged, only their home moved.

Why this is its own module: the budget special-cases a Portal mesh as an opaque
multi-hop region, which is a distinct concern from the run verbs that consume it.
``runner.py`` re-exports every name here, so existing imports keep resolving.
"""

from __future__ import annotations

from typing import cast

from langchain_core.runnables import RunnableConfig

from neograph._compiled import CompiledNeograph
from neograph._ir_branch import iter_with_arms
from neograph._llm_config import _coerce_llm_config
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph.construct import Construct, iter_nodes
from neograph.modifiers import PrimaryShape, primary_shape
from neograph.node import Node

# LangGraph's default per-invoke superstep ceiling. Agent/act nodes compile to an
# inline ReAct cycle (2 supersteps/turn: agent + tools), so a loop near its
# max_iterations bound can exceed this ceiling BEFORE the graceful budget-exhaust
# forced-final fires. This bites most in a NESTED agent (a sub-construct invoke
# defaults to 25); the raised top-level limit propagates into the child invoke.
# _ensure_agent_recursion_limit raises the ceiling so the forced-final edge is
# reachable at any max_iterations.
_LANGGRAPH_DEFAULT_RECURSION_LIMIT = 25
# Supersteps a single agent turn costs (agent node + tools node).
_SUPERSTEPS_PER_AGENT_TURN = 2
# Per-agent overhead beyond the turns: the parse node + the forced-final turn.
_AGENT_CYCLE_OVERHEAD = 3


def _member_hop_cost(member: Node | Construct) -> int:
    """Superstep cost of ONE hop through a Portal mesh member.

    Atomic members (scripted/think/raw) cost exactly 1 superstep per hop —
    the pre-nnds9 assumption. An agent/act member's hop re-enters its own
    ReAct cycle (agent<->tools loop, up to ``max_iterations`` turns) before
    its parse node ever emits the mesh ``Command`` — so ONE hop through an
    agent/act member can itself cost ``max_iterations * 2 + overhead``
    supersteps. Mirrors the flat per-node agent/act cost computed in
    ``_ensure_agent_recursion_limit``.

    A sub-``Construct`` mesh member (do0d9, §3.1 site 6) costs exactly 1 opaque
    boundary superstep per hop — its interior runs as a SEPARATE isolated Pregel
    invocation reusing the shared config ``recursion_limit`` (Q4:
    sub-construct-internal work contributes 0 to the PARENT budget), so the
    parent floor must NOT fold the interior worst-case in.
    """
    if portal_member_class(member) in (PortalMemberClass.AGENT_CYCLE_OUTPUT, PortalMemberClass.AGENT_CYCLE_TOOL):
        max_iters = _coerce_llm_config(cast(Node, member).llm_config).max_iterations
        return max_iters * _SUPERSTEPS_PER_AGENT_TURN + _AGENT_CYCLE_OVERHEAD
    # An Operator-guarded member detours through its
    # {member}__approve node before reaching the peer — one extra superstep
    # per hop over the un-guarded atomic case. Checked independently of
    # portal_member_class's SUB_CONSTRUCT precedence: a Construct member that
    # ALSO carries an Operator costs 2 here, same as an atomic one -- this
    # function's operator surcharge is a strictly separate question from the
    # classifier's lossy member-kind reduction.
    ms = getattr(member, "modifier_set", None)
    if ms is not None and ms.operator is not None:
        return 2
    return 1


def _mesh_hop_cost(construct: Construct) -> int:
    """Sum the worst-case superstep cost of every Portal mesh, recursing
    sub-constructs.

    A K-hop mesh consumes up to K hops (one ``Command(goto)`` per hop), each
    landing on WHICHEVER member the route target names — so the worst case is
    every hop landing on the mesh's MOST EXPENSIVE member (per
    ``_member_hop_cost``), not a flat 1-superstep-per-hop assumption (which
    undercounts an agent/act member's own internal ReAct supersteps). A
    contiguous run of Portal-modified sibling Nodes is ONE mesh (design
    §3.1 r2) and only its ENTRY (first member) carries the real ``max_hops``
    (entry-only, T1) — non-entry members default to 10. ``iter_nodes``
    leaf-flattens and cannot identify the entry (nor a mesh whose entry left
    ``max_hops`` at the default), so this uses the level-preserving
    ``iter_with_arms`` walk that mirrors the compiler's mesh detection.
    """
    total = 0
    current_run: list[Node | Construct] = []

    def _flush() -> None:
        nonlocal total
        if not current_run:
            return
        entry_portal = current_run[0].modifier_set.portal
        if entry_portal is not None:
            per_hop = max(_member_hop_cost(m) for m in current_run)
            total += entry_portal.max_hops * per_hop
        current_run.clear()

    for item in iter_with_arms(construct):
        # A Portal-carrying Construct member (do0d9, §3.1 site 6) is a mesh
        # member kept IN the parent contiguous run (cost 1, opaque boundary) —
        # NOT flushed and re-costed as its own standalone nested mesh, which
        # would mis-segment the parent mesh (excluding the boundary hop) and the
        # members after it. This branch precedes the plain-Construct recursion.
        if isinstance(item, Construct) and primary_shape(item) is PrimaryShape.PORTAL:
            current_run.append(item)
            continue
        if isinstance(item, Construct):
            _flush()
            total += _mesh_hop_cost(item)
            continue
        if isinstance(item, Node) and primary_shape(item) is PrimaryShape.PORTAL:
            current_run.append(item)
        else:
            _flush()
    _flush()
    return total


def _portal_mesh_member_ids(construct: Construct) -> set[int]:
    """``id()`` of every Node that is a Portal mesh member, recursing
    sub-constructs.

    An agent/act mesh member's cost is captured ENTIRELY by ``_mesh_hop_cost``
    (its per-hop ReAct-cycle cost, times the mesh's ``max_hops``) — so the flat
    per-node ``agent_cost`` loop in ``_ensure_agent_recursion_limit`` must
    EXCLUDE mesh members, or an agent/act mesh member's cost is double-counted
    (once flat, once mesh-aware).
    """
    ids: set[int] = set()
    for item in iter_with_arms(construct):
        if isinstance(item, Construct) and primary_shape(item) is PrimaryShape.PORTAL:
            # A Portal-carrying Construct mesh member (do0d9, §3.1 site 6): its
            # interior runs as a separate isolated invoke (0 parent-budget
            # contribution, Q4) and its per-hop cost is 1 in _mesh_hop_cost — so
            # EXCLUDE its boundary AND every interior leaf node from the flat
            # per-node agent-cost loop, or a nested agent/act node would be
            # double-counted (mesh member cost 1, yet also flat-counted).
            ids.add(id(item))
            ids |= {id(n) for n in iter_nodes(item)}
        elif isinstance(item, Construct):
            ids |= _portal_mesh_member_ids(item)
        elif isinstance(item, Node) and primary_shape(item) is PrimaryShape.PORTAL:
            ids.add(id(item))
    return ids


def _ensure_agent_recursion_limit(
    graph: CompiledNeograph,
    config: RunnableConfig | None,
) -> RunnableConfig | None:
    """Raise ``recursion_limit`` so an agent/act cycle OR a Portal mesh can
    reach its graceful budget-exhaust edge instead of hitting LangGraph's default
    superstep ceiling first.

    Each STANDALONE agent/act node's cycle can cost ``max_iterations * 2 +
    overhead`` supersteps; each Portal mesh can cost up to
    ``entry.max_hops * worst_member_hop_cost`` supersteps (an agent/act mesh
    member's own hop cost, per ``_mesh_hop_cost``/``_member_hop_cost``). Both
    run in distinct supersteps, so their costs ADD across the run. The floor
    sums every STANDALONE agent/act node's worst case AND every mesh's
    worst-case hop budget on top of the default (which already covers the
    surrounding non-agent nodes) — an agent/act node that is ALSO a Portal
    mesh member is excluded from the flat per-node sum
    (``_portal_mesh_member_ids``) so its cost is not double-counted (once
    flat, once mesh-aware). Only RAISES to the floor — a larger
    user-supplied ``recursion_limit`` is kept. Pure config mutation (no
    engine verb); shared verbatim by ``_prepare`` and ``_aprepare``.
    """
    construct = getattr(graph, "construct", None)
    if construct is None:
        return config

    mesh_member_ids = _portal_mesh_member_ids(construct)
    agent_cost = 0
    for node in iter_nodes(construct):
        if isinstance(node, Node) and node.mode in ("agent", "act") and id(node) not in mesh_member_ids:
            max_iters = _coerce_llm_config(node.llm_config).max_iterations
            agent_cost += max_iters * _SUPERSTEPS_PER_AGENT_TURN + _AGENT_CYCLE_OVERHEAD

    mesh_cost = _mesh_hop_cost(construct)

    if agent_cost == 0 and mesh_cost == 0:
        return config  # no agent/act nodes and no mesh — leave the default untouched

    floor = _LANGGRAPH_DEFAULT_RECURSION_LIMIT + agent_cost + mesh_cost
    current = (config or {}).get("recursion_limit", _LANGGRAPH_DEFAULT_RECURSION_LIMIT)
    if current >= floor:
        return config  # user asked for at least what agents/mesh need — keep theirs

    new_config: RunnableConfig = {**(config or {})}
    new_config["recursion_limit"] = floor
    return new_config
