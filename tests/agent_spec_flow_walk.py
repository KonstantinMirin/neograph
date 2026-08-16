"""The ONE test-side reader of an exported Agent Spec ``Flow``'s control graph.

Core Invariant: a test-side graph walk must see EVERY node and EVERY labelled
edge the exported Flow actually contains -- across both subflow spellings,
preserving ``from_branch`` AND its multiplicity -- and must be reachable from any
test module without dragging the marker-reading importer (``from_agent_spec``)
into its import graph.

Why this module exists at all (the layering half). The canonical walk helpers
lived in ``tests/test_agent_spec_reachability.py``, which imports
``tests.test_agent_spec_matrix``, which imports ``neograph.loader.from_agent_spec``.
Any module wanting to reuse the walk therefore inherited the IMPORTER -- and the
whole point of a test-side walk is to prove the EXPORT is independently
executable, without the marker-reading importer in the picture. A live consumer
was already paying for it: ``tests/agent_spec_loader_harness.py`` took an
``all_flows_fn`` parameter in three functions purely to avoid that import.

IMPORT PURITY IS ABSOLUTE -- this module imports NOTHING from ``neograph``, not
even ``Branch``. That is not fastidiousness, it is arithmetic: importing ANY
``neograph`` submodule executes ``src/neograph/__init__.py``, whose line 65 is
``from neograph.loader import from_agent_spec, load_spec``. So a single
``from neograph._agent_spec_markers import Branch`` would put ``from_agent_spec``
straight back into this module's import graph and contradict the invariant above.
Consequently every ``branch`` parameter here is a plain ``str`` and CALLERS pass
``Branch.*`` in. ``tests/test_agent_spec_flow_walk.py`` proves the purity by
subprocess ``sys.modules`` inspection, because an AST guard structurally cannot.

It does not import ``pyagentspec`` either: every Flow/node/edge here is read
duck-typed, so this module is always importable (the Tier-A property
``tests/agent_spec_capabilities.py`` has).

The one rule that shapes the API: ``branch_adjacency`` is the SINGLE edge read,
and every other edge question is a pure PROJECTION of it. Nothing else iterates
``control_flow_connections``. That gives ``neograph-dgbqv.10`` exactly one seam to
convert when it adds labelled assertions.

This module does NOT widen coverage on its own -- the five migrated ``pairs``
sites still assert on unlabelled tuples, so a branch-relabel mutation stays as
invisible as it was. It ENABLES ``dgbqv.10`` to fix that; it does not do it here.

Data flow (``wired_edges``) joined later, under a DIFFERENT rule and for a
different reason (``neograph-qtfof.6``). It is NOT governed by the control-flow
monopoly guard: ``tests/test_guards_agent_spec_data_flow_reads.py`` owns every
``data_flow_connections`` read (the read must be the left operand of a
``BoolOp(Or)``, because the exporter writes ``<edges> or None``), and its
``TestWiredEdgeProjectionIsNotDuplicated`` caps the 4-tuple projection at ONE
inline site tree-wide. That cap is why the projection lives here rather than as a
second local copy: its own comment names "a copy in a THIRD file" as the signal to
promote. Two guards over two attributes under two rules -- not one attribute under
two rules, which is the shape AGENTS.md's Portal lesson bans.
"""

from __future__ import annotations

from typing import Any

_BOUNDARY_TYPE_NAMES = ("StartNode", "EndNode")


def sub_flows(flow: Any) -> list[Any]:
    """Every nested sub-``Flow`` one level down, from any subflow-bearing node.

    Reads BOTH pyagentspec subflow-holder spellings -- the singular ``subflow``
    (``FlowNode``/``MapNode``/``CatchExceptionNode``) and the plural ``subflows``
    (``ParallelFlowNode``) -- so a fused Each x Oracle body cannot hide from the
    walk behind whichever holder the lowering happens to pick.
    """
    found: list[Any] = []
    for node in flow.nodes:
        found.extend(holder_flows(node))
    return found


def all_flows(flow: Any) -> list[Any]:
    """The flow and every sub-flow beneath it, breadth-first."""
    flows = [flow]
    i = 0
    while i < len(flows):
        flows.extend(sub_flows(flows[i]))
        i += 1
    return flows


def inner_nodes(holder: Any) -> list[Any]:
    """A subflow holder's interior nodes, minus the Start/End boundary sentinels.

    Routes through the same both-spellings read ``sub_flows`` uses, so a plural
    holder cannot hide -- which is the point: the copies this replaces
    (``test_agent_spec_each_oracle.py``, ``test_agent_spec_each_oracle_operator.py``)
    each read the SINGULAR ``.subflow`` only and would go blind the day a
    lowering picks a plural holder. This is ``neograph-498gr``'s seam.

    Takes the HOLDER NODE (not a Flow); a node holding no sub-flow yields [].
    """
    return [
        node for sub in holder_flows(holder) for node in sub.nodes if type(node).__name__ not in _BOUNDARY_TYPE_NAMES
    ]


def branch_adjacency(flow: Any) -> list[tuple[str, str | None, str]]:
    """THE single edge read: ``(from_node, from_branch, to_node)`` per control edge.

    LIST-valued, never a set, and that is load-bearing rather than incidental:
    ``tests/test_agent_spec_export.py`` asserts ``len(back_edges) == 1`` at :783,
    :1138 and :1201 and ``len(self_edges) == 1`` at :1157. A set silently
    collapses duplicate ``(from, branch, to)`` triples and would weaken all four
    from "exactly one such edge" to "at least one".

    ``from_branch`` is preserved (``None`` for an unconditional edge) so the arm
    label is never dropped AT THE SOURCE -- every other edge question below is a
    projection of this one, so there is exactly one place a label could be lost.
    """
    return [(e.from_node.name, e.from_branch, e.to_node.name) for e in flow.control_flow_connections]


def edge_pairs(flow: Any) -> set[tuple[str, str]]:
    """Unlabelled ``{(from, to)}`` -- a pure projection of ``branch_adjacency``.

    The form five test modules spelled by hand. A set here is correct: callers
    ask "is this pair wired?", not "how many times".
    """
    return {(frm, to) for frm, _branch, to in branch_adjacency(flow)}


def successors_of(flow: Any, name: str) -> list[str]:
    """Targets of every edge leaving ``name``, on any branch. Multiplicity kept."""
    return [to for frm, _branch, to in branch_adjacency(flow) if frm == name]


def arm_targets(flow: Any, name: str, branch: str) -> list[str]:
    """Targets of edges leaving ``name`` on the ``branch`` arm specifically.

    ``branch`` is a plain ``str``; callers pass ``Branch.CONTINUE`` etc. in, since
    this module cannot import ``Branch`` (see the module docstring).
    """
    return [to for frm, arm, to in branch_adjacency(flow) if frm == name and arm == branch]


def entered_nodes(flow: Any) -> set[str]:
    """The nodes a literal executor runs FIRST -- the StartNode's edge targets.

    Asserts a THREE-WAY BIND before answering, rather than merely that a
    StartNode is unique. The two readings this unifies are different questions:
    ``walk`` seeds from the DECLARED ``flow.start_node``, while the hand-rolled
    sites it replaces took successors of whatever node's TYPE NAME is
    ``StartNode``. pyagentspec annotates ``Flow.start_node`` as ``Node`` -- not
    ``StartNode`` -- so a Flow whose declared start is not the StartNode is
    representable, and today's exporters merely happen to pass the same object.
    Each reading catches a defect the other misses: the TYPE form notices a
    second StartNode appearing, the ``start_node`` form notices the declared
    start drifting off it. Asserting only uniqueness would keep one net and
    silently drop the other -- a narrowing hidden inside a move.
    """
    starts = [n for n in flow.nodes if type(n).__name__ == "StartNode"]
    assert len(starts) == 1, f"expected exactly one StartNode, found {[n.name for n in starts]}"
    assert flow.start_node is starts[0], (
        f"flow.start_node ({flow.start_node.name!r}) is not the StartNode-typed node "
        f"({starts[0].name!r}) -- the declared start and the structural start disagree"
    )
    return set(successors_of(flow, starts[0].name))


def walk(flow: Any) -> tuple[set[str], set[str]]:
    """``(reachable-from-start, can-reach-an-EndNode)`` node names for ONE flow.

    Seeds reachability from the DECLARED ``flow.start_node`` -- deliberately not
    the same question ``entered_nodes`` asks, which is why that one checks the
    bind.

    The adjacency maps are PRE-SEEDED from ``flow.nodes`` rather than built with
    a ``defaultdict``. That is intentional and worth keeping: an edge naming an
    endpoint the flow does not list raises ``KeyError`` here, whereas a
    ``defaultdict`` would silently absorb the dangling reference -- exactly the
    class of defect this walk exists to surface.
    """
    successors: dict[str, set[str]] = {n.name: set() for n in flow.nodes}
    predecessors: dict[str, set[str]] = {n.name: set() for n in flow.nodes}
    for frm, _branch, to in branch_adjacency(flow):
        successors[frm].add(to)
        predecessors[to].add(frm)

    def closure(seeds: set[str], adjacency: dict[str, set[str]]) -> set[str]:
        seen = set(seeds)
        stack = list(seeds)
        while stack:
            for nxt in adjacency[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    ends = {n.name for n in flow.nodes if type(n).__name__ == "EndNode"}
    return closure({flow.start_node.name}, successors), closure(ends, predecessors)


def wired_edges(flow: Any) -> list[tuple[str, str, str, str]]:
    """THE single data-edge read: ``(source, source_output, dest, dest_input)`` per edge.

    ``data_flow_connections`` is None-able -- the exporter writes ``<edges> or None``
    at both construction sites, so a control-only shape genuinely exports ``None``
    and a raw dereference raises ``TypeError`` instead of failing an assertion
    (``neograph-p0dn8``). The ``or []`` here is the guarded form
    ``tests/test_guards_agent_spec_data_flow_reads.py`` requires.

    LIST-valued for the same reason ``branch_adjacency`` is: callers assert on
    multiplicity ("exactly one edge feeds this input"), which a set would silently
    collapse to "at least one".
    """
    return [
        (e.source_node.name, e.source_output, e.destination_node.name, e.destination_input)
        for e in (flow.data_flow_connections or [])
    ]


def fed_inputs(flow: Any, name: str) -> set[str]:
    """The ``destination_input`` titles on ``name`` that a real DataFlowEdge feeds.

    A pure projection of ``wired_edges``. The question it exists to answer is the
    one a metadata-blind Agent Spec runtime asks: does this node's declared input
    arrive over a declared data path, or does it silently resolve to nothing?
    """
    return {dest_input for _src, _out, dst, dest_input in wired_edges(flow) if dst == name}


def data_flow_predecessors(flow: Any, name: str) -> set[str]:
    """Source node names of every DataFlowEdge terminating on ``name``."""
    return {src for src, _out, dst, _dest_input in wired_edges(flow) if dst == name}


def declared_input_titles(node: Any) -> list[str]:
    """A node's declared input Property titles, duck-typed and None-safe.

    ``inputs`` is None-able on pyagentspec nodes exactly as
    ``data_flow_connections`` is on the Flow, and an inferred-input node (a
    ``BranchingNode``, whose ``branching_mapping_key`` is derived rather than
    authored) still reports it here -- which is the point: an INFERRED input is
    still an input a runtime must be handed a value for.
    """
    return [p.title for p in (getattr(node, "inputs", None) or [])]


def holder_flows(node: Any) -> list[Any]:
    """The sub-flows ONE node holds, reading both holder spellings.

    The single place the singular/plural distinction is handled, so ``sub_flows``
    and ``inner_nodes`` cannot drift apart on it.

    Public because callers legitimately need the sub-Flow ITSELF rather than its
    interior: a test asserting on the sub-flow's own ``StartNode``, or on
    per-variant sub-Flow object identity, cannot use ``inner_nodes`` (which
    strips exactly those boundary sentinels) and must not fall back to a raw
    ``.subflow`` read, which is singular-only and goes blind the day a lowering
    picks a plural holder (neograph-498gr).
    """
    found: list[Any] = []
    one = getattr(node, "subflow", None)
    if one is not None:
        found.append(one)
    found.extend(getattr(node, "subflows", None) or [])
    return found
