"""Behavioural suite for ``tests/agent_spec_flow_walk.py`` -- neograph-dgbqv.9.

``tests/agent_spec_flow_walk.py`` is the ONE place a test-side reader asks a
question of an exported Agent Spec ``Flow``'s control graph. Its Core Invariant:

    a test-side graph walk must see EVERY node and EVERY labelled edge the
    exported Flow actually contains -- across both subflow spellings, preserving
    ``from_branch`` AND its multiplicity -- and must be reachable from any test
    module without dragging the marker-reading importer (``from_agent_spec``)
    into its import graph.

This file is that module's own suite. Three of its requirements carry the risk
and get the sharpest tests:

* **R0 -- import purity is absolute.** The walk module imports NOTHING from
  ``neograph``, not even ``Branch``. Verified empirically, not assumed:
  ``import neograph._agent_spec_markers`` already puts ``neograph.loader`` into
  ``sys.modules``, because importing ANY neograph submodule executes
  ``src/neograph/__init__.py``, whose line 65 is
  ``from neograph.loader import from_agent_spec, load_spec``. So even a
  ``Branch`` import would put the marker-reading importer back into the walk
  module's import graph and contradict the Core Invariant -- while the AST guard
  shipping alongside would NOT catch it. Only a subprocess ``sys.modules``
  inspection can prove this, which is why ``TestWalkModuleImportPurity`` has the
  same shape as ``tests/test_guards_agent_spec_core_purity.py``.

* **R1 -- ``branch_adjacency`` is LIST-valued, never a set.** Multiplicity is
  load-bearing: ``test_agent_spec_export.py`` asserts ``len(back_edges) == 1`` at
  :783, :1138 and :1201 and ``len(self_edges) == 1`` at :1157. A set silently
  collapses duplicate ``(from, branch, to)`` triples and weakens all four.

* **R2 -- ``entered_nodes`` asserts a THREE-WAY BIND, not merely uniqueness.**
  ``walk`` seeds from the DECLARED ``flow.start_node``; the two entered sites it
  replaces (``test_agent_spec_reachability.py``:165 and :223) compute successors
  of whatever node's TYPE NAME is ``StartNode``. Those are different questions,
  and pyagentspec annotates ``Flow.start_node`` as ``Node`` -- not ``StartNode``
  -- so a Flow whose declared start is not the StartNode is representable.
  Unifying on uniqueness alone would be a silent narrowing, so ``entered_nodes``
  must fail LOUD on every way the bind can break.

NOT ``importorskip``-gated, deliberately. ``pyagentspec`` lives in
``[dependency-groups].dev`` and is therefore always installed by the gate
(``uv run pytest``), so a guard here would never fire -- it would only buy the
ability to skip silently. That is the same reasoning CLAUDE.md records for the
~32 unguarded Agent Spec tests: a loud failure beats a silent skip, and R0's
proof is exactly the kind that must never quietly not run.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from collections import Counter
from typing import Any

import pytest
from pyagentspec.flows.edges import ControlFlowEdge
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import (
    BranchingNode,
    EndNode,
    FlowNode,
    MapNode,
    ParallelFlowNode,
    StartNode,
)

from neograph._agent_spec import to_agent_spec
from neograph._agent_spec_markers import Branch
from neograph.construct import Construct
from tests.agent_spec_flow_walk import (
    all_flows,
    arm_targets,
    branch_adjacency,
    edge_pairs,
    entered_nodes,
    inner_nodes,
    sub_flows,
    successors_of,
    walk,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# -- Synthetic flows ---------------------------------------------------------
#
# Built from raw pyagentspec primitives rather than through ``to_agent_spec``,
# because the shapes that matter here -- a duplicate edge triple, a Flow whose
# declared start is not its StartNode, an edge naming a node the Flow does not
# list -- are exactly the ones neograph's own exporters never emit today. That
# is the point: the helpers must be correct for every Flow the SPEC permits, not
# only for the ones our exporter happens to produce.


def _leaf_flow(name: str) -> Flow:
    """A minimal valid Flow: ``start -> end``."""
    start = StartNode(name=f"{name}__start")
    end = EndNode(name=f"{name}__end")
    return Flow(
        name=name,
        start_node=start,
        nodes=[start, end],
        control_flow_connections=[ControlFlowEdge(name=f"{name}__e", from_node=start, to_node=end)],
    )


def _duplicate_edge_flow() -> Flow:
    """``start -> gate``; ``gate --true--> end`` TWICE; ``gate --false--> gate``.

    The duplicated arm is the R1 fixture: a set-valued adjacency collapses the
    two ``(gate, true, end)`` triples into one and cannot tell a doubled edge
    from a single one.
    """
    start = StartNode(name="start")
    gate = BranchingNode(name="gate", mapping={Branch.TRUE: Branch.TRUE, Branch.FALSE: Branch.FALSE})
    end = EndNode(name="end")
    return Flow(
        name="duplicate-arm-flow",
        start_node=start,
        nodes=[start, gate, end],
        control_flow_connections=[
            ControlFlowEdge(name="c0", from_node=start, to_node=gate),
            ControlFlowEdge(name="c1", from_node=gate, from_branch=Branch.TRUE, to_node=end),
            ControlFlowEdge(name="c2", from_node=gate, from_branch=Branch.TRUE, to_node=end),
            ControlFlowEdge(name="c3", from_node=gate, from_branch=Branch.FALSE, to_node=gate),
        ],
    )


def _misdeclared_start_flow() -> Flow:
    """A Flow whose ``start_node`` is NOT its ``StartNode``.

    pyagentspec's model validator rejects this at construction time, so the Flow
    is built valid and then mutated -- which is precisely what makes the shape
    reachable in practice, and why ``entered_nodes`` cannot lean on the
    constructor having checked. ``start_node`` is re-pointed at ``gate``, whose
    successors ({'end'}) differ from the StartNode's ({'gate'}), so an
    implementation that answers either question silently is answering a
    different one than the other.
    """
    flow = _duplicate_edge_flow()
    flow.start_node = next(n for n in flow.nodes if n.name == "gate")
    return flow


def _nested_flow() -> Flow:
    """Three levels, and BOTH subflow spellings at level 1.

    ``root`` holds a ``FlowNode`` (singular ``.subflow``) and a
    ``ParallelFlowNode`` (plural ``.subflows``); the FlowNode's subflow itself
    holds a ``FlowNode``, giving a grandchild for the BFS ordering proof.
    """
    grandchild = _leaf_flow("grandchild")
    child_a = _leaf_flow("child_a")
    child_a.nodes = [*child_a.nodes, FlowNode(name="child_a__holder", subflow=grandchild)]

    child_b = _leaf_flow("child_b")
    child_c = _leaf_flow("child_c")

    start = StartNode(name="root__start")
    singular = FlowNode(name="singular_holder", subflow=child_a)
    plural = ParallelFlowNode(name="plural_holder", subflows=[child_b, child_c])
    end = EndNode(name="root__end")
    return Flow(
        name="root",
        start_node=start,
        nodes=[start, singular, plural, end],
        control_flow_connections=[
            ControlFlowEdge(name="r0", from_node=start, to_node=singular),
            ControlFlowEdge(name="r1", from_node=singular, to_node=plural),
            ControlFlowEdge(name="r2", from_node=plural, to_node=end),
        ],
    )


def _loop_flow() -> Any:
    """A real exported Flow -- the do-while Loop shape from the reachability suite."""
    from neograph.modifiers import Loop
    from neograph.node import Node
    from tests.schemas import Claims

    item = Node.scripted("refine", fn="refine_fn", inputs=Claims, outputs=Claims) | Loop(
        when="claims_incomplete", max_iterations=3
    )
    return to_agent_spec(Construct("loop-entry-pipeline", nodes=[item]))


def _each_flow() -> Any:
    """A real exported Flow carrying a ``MapNode`` -- a live singular-subflow holder."""
    from neograph.modifiers import Each
    from tests.schemas import Claims, RawText, _consumer

    item = _consumer("verify", RawText, Claims) | Each(over="items", key="label")
    return to_agent_spec(Construct("each-pipeline", nodes=[item]))


# -- R0 ----------------------------------------------------------------------


class TestWalkModuleImportPurity:
    """``tests/agent_spec_flow_walk.py`` imports NOTHING from ``neograph``.

    Not even ``Branch``: a single neograph submodule import executes
    ``neograph/__init__.py`` and puts ``neograph.loader`` -- the marker-reading
    importer whose blindness this whole family of tests exists to route around --
    into the walk module's import graph. Every branch parameter is therefore a
    plain ``str``, and CALLERS pass ``Branch.*`` in from their own modules.
    """

    @staticmethod
    def _module_roots_after(import_line: str) -> set[str]:
        """Top-level ``sys.modules`` roots present after ``import_line``, in a
        SUBPROCESS -- this test process has already imported neograph, so an
        in-process check would be vacuous."""
        code = (
            f"import sys\n{import_line}\nprint('ROOTS:' + ','.join(sorted({{m.split('.')[0] for m in sys.modules}})))\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"subprocess failed:\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
        line = next(ln for ln in result.stdout.splitlines() if ln.startswith("ROOTS:"))
        return set(line[len("ROOTS:") :].split(","))

    def test_importing_the_walk_module_pulls_in_no_neograph_module(self) -> None:
        roots = self._module_roots_after("import tests.agent_spec_flow_walk")
        assert "neograph" not in roots, (
            "tests/agent_spec_flow_walk.py imported something from neograph. Importing ANY "
            "neograph submodule executes neograph/__init__.py, which imports "
            "neograph.loader.from_agent_spec -- the marker-reading importer this module must "
            "stay independent of. Every branch parameter is a plain str; callers pass Branch.* in."
        )

    def test_importing_the_walk_module_pulls_in_no_pyagentspec_module(self) -> None:
        """Tier-A, like ``tests/agent_spec_capabilities.py``: the module reads Flows
        duck-typed (``getattr``/``type(n).__name__``), so it stays importable in an
        environment without the optional dependency."""
        roots = self._module_roots_after("import tests.agent_spec_flow_walk")
        assert "pyagentspec" not in roots, (
            "tests/agent_spec_flow_walk.py imported pyagentspec. It reads Flows duck-typed "
            "(getattr for subflow/subflows, type(n).__name__ for StartNode/EndNode), so it must "
            "stay always-importable."
        )

    def test_meta_a_single_neograph_submodule_import_really_drags_in_the_loader(self) -> None:
        """Non-vacuity: the detector above fires on the shape it bans.

        This is the empirical fact R0 rests on -- importing the LEAF marker module
        is enough to pull ``neograph.loader`` in.
        """
        code = (
            "import sys\n"
            "import neograph._agent_spec_markers\n"
            "assert 'neograph.loader' in sys.modules, 'premise broken: no loader import'\n"
            "print('LOADER_DRAGGED_IN')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, f"STDOUT:{result.stdout}\nSTDERR:{result.stderr}"
        assert "LOADER_DRAGGED_IN" in result.stdout


# -- R1 ----------------------------------------------------------------------


class TestBranchAdjacencyIsTheSingleEdgeRead:
    """``branch_adjacency`` is THE control-edge read: one ``(from, branch, to)``
    triple per ``ControlFlowEdge``, in a LIST, with the label kept."""

    def test_branch_adjacency_returns_a_list(self) -> None:
        assert isinstance(branch_adjacency(_duplicate_edge_flow()), list)

    def test_branch_adjacency_preserves_duplicate_triples(self) -> None:
        counts = Counter(branch_adjacency(_duplicate_edge_flow()))
        assert counts[("gate", Branch.TRUE, "end")] == 2, (
            "a set-valued adjacency collapses duplicate (from, branch, to) triples. "
            "Multiplicity is load-bearing: test_agent_spec_export.py asserts len(back_edges) == 1 "
            "at :783, :1138 and :1201, and len(self_edges) == 1 at :1157."
        )

    def test_branch_adjacency_emits_exactly_one_entry_per_control_flow_edge(self) -> None:
        flow = _duplicate_edge_flow()
        assert len(branch_adjacency(flow)) == len(flow.control_flow_connections) == 4

    def test_branch_adjacency_keeps_the_label_and_none_for_an_unlabelled_edge(self) -> None:
        adjacency = branch_adjacency(_duplicate_edge_flow())
        assert ("start", None, "gate") in adjacency, "an unlabelled edge must keep from_branch=None"
        assert ("gate", Branch.FALSE, "gate") in adjacency

    def test_branch_adjacency_reads_a_real_exported_flow(self) -> None:
        adjacency = branch_adjacency(_loop_flow())
        assert ("refine__loop_check", Branch.CONTINUE, "refine") in adjacency
        assert any(frm == "refine__loop_check" and branch == Branch.DONE for frm, branch, _ in adjacency), (
            f"the loop's labelled done-arm vanished from the adjacency: {adjacency}"
        )


class TestEdgePairsIsAProjectionOfBranchAdjacency:
    """``edge_pairs`` drops the label -- and nothing else. Deriving it from
    ``branch_adjacency`` centralizes WHERE the label is dropped (which is what
    gives neograph-dgbqv.10 exactly one seam to convert)."""

    def test_edge_pairs_is_the_unlabelled_projection_of_branch_adjacency(self) -> None:
        for flow in (_duplicate_edge_flow(), _loop_flow()):
            assert edge_pairs(flow) == {(frm, to) for frm, _, to in branch_adjacency(flow)}

    def test_edge_pairs_is_a_set_and_collapses_the_duplicate_arm(self) -> None:
        pairs = edge_pairs(_duplicate_edge_flow())
        assert isinstance(pairs, set)
        assert pairs == {("start", "gate"), ("gate", "end"), ("gate", "gate")}

    def test_edge_pairs_matches_the_hand_rolled_expression_it_replaces(self) -> None:
        """The five migrated sites all spell
        ``{(e.from_node.name, e.to_node.name) for e in flow.control_flow_connections}``."""
        flow = _loop_flow()
        assert edge_pairs(flow) == {(e.from_node.name, e.to_node.name) for e in flow.control_flow_connections}


class TestSuccessorsOfAndArmTargets:
    """Both are projections of the single edge read, so both keep multiplicity."""

    def test_successors_of_preserves_multiplicity(self) -> None:
        assert Counter(successors_of(_duplicate_edge_flow(), "gate")) == Counter(["end", "end", "gate"])

    def test_successors_of_returns_a_list(self) -> None:
        assert isinstance(successors_of(_duplicate_edge_flow(), "gate"), list)

    def test_successors_of_is_empty_for_a_node_with_no_outgoing_edges(self) -> None:
        assert successors_of(_duplicate_edge_flow(), "end") == []

    def test_arm_targets_selects_one_arm_and_keeps_multiplicity(self) -> None:
        flow = _duplicate_edge_flow()
        assert Counter(arm_targets(flow, "gate", Branch.TRUE)) == Counter(["end", "end"])
        assert arm_targets(flow, "gate", Branch.FALSE) == ["gate"]

    def test_arm_targets_takes_the_branch_as_a_plain_string_from_the_caller(self) -> None:
        """R0's consequence at the call site: the walk module cannot import
        ``Branch``, so THIS module does and passes the plain string in."""
        flow = _loop_flow()
        assert arm_targets(flow, "refine__loop_check", Branch.CONTINUE) == ["refine"]
        assert arm_targets(flow, "refine__loop_check", "continue") == ["refine"]

    def test_arm_targets_is_empty_for_an_arm_the_node_does_not_take(self) -> None:
        assert arm_targets(_duplicate_edge_flow(), "gate", Branch.PAUSE) == []


# -- R2 ----------------------------------------------------------------------


class TestEnteredNodesAssertsTheThreeWayBind:
    """``entered_nodes`` returns the StartNode's outgoing targets -- but only
    after proving the StartNode and the DECLARED start are the same object.

    The two questions differ. ``walk`` seeds from ``flow.start_node``; the sites
    this replaces filter on ``type(e.from_node).__name__ == 'StartNode'``. The
    TYPE form catches a second StartNode appearing; the ``start_node`` form
    catches the declared start drifting off it. Asserting only uniqueness keeps
    one of those two nets and silently drops the other.
    """

    def test_entered_nodes_returns_the_start_nodes_outgoing_targets(self) -> None:
        assert entered_nodes(_loop_flow()) == {"refine"}

    def test_entered_nodes_matches_the_hand_rolled_expression_it_replaces(self) -> None:
        flow = _loop_flow()
        assert entered_nodes(flow) == {
            e.to_node.name for e in flow.control_flow_connections if type(e.from_node).__name__ == "StartNode"
        }

    def test_entered_nodes_fails_loud_when_the_declared_start_is_not_the_start_node(self) -> None:
        flow = _misdeclared_start_flow()
        # The narrowing this catches: the two readings genuinely disagree here.
        assert successors_of(flow, "start") != successors_of(flow, flow.start_node.name)
        with pytest.raises(AssertionError):
            entered_nodes(flow)

    def test_entered_nodes_fails_loud_when_a_second_start_node_appears(self) -> None:
        flow = _duplicate_edge_flow()
        flow.nodes = [*flow.nodes, StartNode(name="second_start")]
        with pytest.raises(AssertionError):
            entered_nodes(flow)

    def test_entered_nodes_fails_loud_when_no_start_node_exists(self) -> None:
        flow = _duplicate_edge_flow()
        flow.nodes = [n for n in flow.nodes if type(n).__name__ != "StartNode"]
        with pytest.raises(AssertionError):
            entered_nodes(flow)


# -- walk / sub_flows / all_flows (moved verbatim, pinned here) --------------


class TestWalk:
    """``walk`` keeps its NAME and its two-set answer: reachable-from-the-declared
    -start, and can-reach-an-EndNode."""

    def test_walk_returns_the_reachable_and_live_name_sets(self) -> None:
        reachable, live = walk(_duplicate_edge_flow())
        assert reachable == {"start", "gate", "end"}
        assert live == {"start", "gate", "end"}

    def test_walk_reports_an_unreachable_node_and_a_dead_end(self) -> None:
        flow = _duplicate_edge_flow()
        orphan = BranchingNode(name="orphan", mapping={Branch.TRUE: Branch.TRUE})
        flow.nodes = [*flow.nodes, orphan]
        reachable, live = walk(flow)
        assert "orphan" not in reachable, "a node nothing points at never executes"
        assert "orphan" not in live, "a node with no path to an EndNode is a dead end"

    def test_walk_seeds_reachability_from_the_declared_start_node(self) -> None:
        """Deliberately NOT the same question as ``entered_nodes``: ``walk`` trusts
        ``flow.start_node``, which is why ``entered_nodes`` has to check the bind."""
        flow = _misdeclared_start_flow()
        reachable, _ = walk(flow)
        assert "start" not in reachable, "walk seeds from flow.start_node, which here is 'gate'"
        assert reachable == {"gate", "end"}

    def test_walk_fails_loud_on_an_edge_naming_a_node_the_flow_does_not_list(self) -> None:
        """The adjacency dicts are PRE-SEEDED from ``flow.nodes``. A defaultdict
        would silently absorb an edge endpoint that is not a node of the flow --
        exactly the kind of dangling reference the walk exists to surface."""
        flow = _duplicate_edge_flow()
        flow.nodes = [n for n in flow.nodes if n.name != "end"]
        with pytest.raises(KeyError):
            walk(flow)


class TestSubFlowsReadsBothSpellings:
    """A fused Each x Oracle body must not hide behind whichever holder the
    lowering happens to pick."""

    def test_sub_flows_reads_the_singular_subflow_holder(self) -> None:
        flow = _leaf_flow("host")
        flow.nodes = [*flow.nodes, FlowNode(name="holder", subflow=_leaf_flow("inner"))]
        assert [f.name for f in sub_flows(flow)] == ["inner"]

    def test_sub_flows_reads_the_plural_subflows_holder(self) -> None:
        flow = _leaf_flow("host")
        flow.nodes = [
            *flow.nodes,
            ParallelFlowNode(name="holder", subflows=[_leaf_flow("inner_a"), _leaf_flow("inner_b")]),
        ]
        assert [f.name for f in sub_flows(flow)] == ["inner_a", "inner_b"]

    def test_sub_flows_reads_both_spellings_in_one_flow(self) -> None:
        found = sub_flows(_nested_flow())
        assert len(found) == 3
        assert {f.name for f in found} == {"child_a", "child_b", "child_c"}

    def test_sub_flows_is_empty_for_a_flow_with_no_holders(self) -> None:
        assert sub_flows(_leaf_flow("plain")) == []

    def test_sub_flows_reads_a_real_exported_map_node(self) -> None:
        flow = _each_flow()
        map_node = next(n for n in flow.nodes if isinstance(n, MapNode))
        assert [f.name for f in sub_flows(flow)] == [map_node.subflow.name]

    def test_sub_flows_is_one_level_only(self) -> None:
        assert "grandchild" not in {f.name for f in sub_flows(_nested_flow())}


class TestAllFlowsIsTheBfsClosure:
    def test_all_flows_starts_with_the_flow_itself(self) -> None:
        root = _nested_flow()
        assert all_flows(root)[0] is root

    def test_all_flows_covers_every_nested_flow(self) -> None:
        assert {f.name for f in all_flows(_nested_flow())} == {
            "root",
            "child_a",
            "child_b",
            "child_c",
            "grandchild",
        }

    def test_all_flows_is_breadth_first(self) -> None:
        names = [f.name for f in all_flows(_nested_flow())]
        assert names.index("grandchild") > max(names.index(n) for n in ("child_a", "child_b", "child_c"))

    def test_all_flows_of_a_leaf_is_just_the_leaf(self) -> None:
        leaf = _leaf_flow("plain")
        assert all_flows(leaf) == [leaf]


class TestInnerNodes:
    """``inner_nodes`` is neograph-498gr's seam: the holder's own sub-flow nodes,
    minus the boundary ``StartNode``/``EndNode``. It routes through the same
    both-spellings read ``sub_flows`` uses, so a plural holder cannot hide."""

    def test_inner_nodes_excludes_the_start_and_end_boundary_nodes(self) -> None:
        inner = _leaf_flow("inner")
        inner.nodes = [*inner.nodes, BranchingNode(name="body", mapping={Branch.TRUE: Branch.TRUE})]
        assert [n.name for n in inner_nodes(FlowNode(name="holder", subflow=inner))] == ["body"]

    def test_inner_nodes_reads_the_plural_subflows_spelling(self) -> None:
        first, second = _leaf_flow("a"), _leaf_flow("b")
        first.nodes = [*first.nodes, BranchingNode(name="body_a", mapping={Branch.TRUE: Branch.TRUE})]
        second.nodes = [*second.nodes, BranchingNode(name="body_b", mapping={Branch.TRUE: Branch.TRUE})]
        holder = ParallelFlowNode(name="holder", subflows=[first, second])
        assert {n.name for n in inner_nodes(holder)} == {"body_a", "body_b"}

    def test_inner_nodes_matches_the_map_node_expression_it_replaces(self) -> None:
        """The two local copies (``test_agent_spec_each_oracle.py``:113,
        ``test_agent_spec_each_oracle_operator.py``:118) spell exactly this."""
        map_node = next(n for n in _each_flow().nodes if isinstance(n, MapNode))
        assert inner_nodes(map_node) == [
            n for n in map_node.subflow.nodes if type(n).__name__ not in ("StartNode", "EndNode")
        ]

    def test_inner_nodes_is_empty_for_a_node_that_holds_no_sub_flow(self) -> None:
        assert inner_nodes(BranchingNode(name="plain", mapping={Branch.TRUE: Branch.TRUE})) == []
