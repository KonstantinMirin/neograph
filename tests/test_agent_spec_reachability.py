"""Control-flow TOPOLOGY proof for ``to_agent_spec`` -- neograph-s7zt3.15.

The maintainer's policy decision (2026-08-03) on this ticket is that an exported
``Flow`` must be "a faithful, independently executable program": correct when
executed by ANY real Agent Spec runtime that walks ``ControlFlowEdge``s
literally, not merely round-trippable through neograph's own marker-reading
importer (which never executes a control edge at all, so it is blind to every
defect below).

That policy has a structural consequence a literal edge-walk can check, and
this module checks it over the WHOLE mechanically-derived GREEN matrix, not a
hand-picked shape:

  * REACHABILITY -- every node in a Flow is reachable from its ``StartNode`` by
    following control edges. A node nothing points at never executes.
  * LIVENESS -- every node can reach an ``EndNode``. A node with no path out is
    a dead end: a literal executor stops there forever.

Both hold recursively for every nested sub-``Flow`` (``MapNode.subflow``,
``FlowNode.subflow``, ``ParallelFlowNode.subflows``), which is where the fused
Each x Oracle body lives.

The three defects this pinned (all pre-existing, all found by the
neograph-s7zt3.10 architect review):

  1. OPERATOR -- the top-level incoming edge targeted the check ``BranchingNode``
     and the body's only edge was ``body -> check``, so the BODY node had no
     inbound edge at all: unreachable, never executed.
  2. The pause ``InputMessageNode`` had no outgoing edge: a dead end.
  3. ORACLE (top level AND fused inside an Each ``MapNode``) -- the incoming edge
     targeted the MERGE node and every variant's only edge was ``variant ->
     merge``, so no variant had an inbound edge: unreachable, yet the merge
     consumes their outputs.

LOOP's defect is NOT visible to a reachability walk (the body is reachable
through the check's ``continue`` branch), so it has its own do-while entry test
in ``TestLoopEntersTheBodyBeforeTheCheck`` below -- see that class for why the
distinction is semantic rather than structural.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pyagentspec")

from neograph._agent_spec import Branch, to_agent_spec  # noqa: E402
from neograph.construct import Construct  # noqa: E402
from tests.agent_spec_flow_walk import (  # noqa: E402
    all_flows,
    arm_targets,
    branch_adjacency,
    entered_nodes,
    successors_of,
    walk,
)
from tests.test_agent_spec_matrix import GREEN, build_cell  # noqa: E402


def _topology_defects(flow: Any) -> list[str]:
    """Every unreachable or dead-end node across the flow and its sub-flows."""
    defects: list[str] = []
    for sub in all_flows(flow):
        reachable, live = walk(sub)
        for node in sub.nodes:
            if node.name not in reachable:
                defects.append(f"{sub.name}: {node.name} ({type(node).__name__}) is UNREACHABLE from the StartNode")
            if node.name not in live:
                defects.append(f"{sub.name}: {node.name} ({type(node).__name__}) cannot reach any EndNode")
    return defects


class TestEveryExportedFlowIsLiterallyExecutable:
    """Every GREEN matrix cell exports a Flow whose control graph a naive
    edge-walking executor can traverse: nothing orphaned, nothing dead-ended.

    Parametrized over the matrix's mechanically-derived GREEN set rather than a
    hand-typed shape list -- the same completeness discipline
    ``test_agent_spec_matrix.py`` exists to enforce. A new mode/combo axis lands
    here automatically instead of silently escaping the topology net.
    """

    @pytest.mark.parametrize("cell_id", sorted(GREEN))
    def test_every_node_is_reachable_and_can_reach_an_end_node(self, cell_id: str) -> None:
        from tests.test_agent_spec_matrix import CELLS

        mode, combo, config, shape = CELLS[cell_id]
        flow = to_agent_spec(build_cell(mode, combo, config, shape))
        if type(flow).__name__ != "Flow":
            pytest.skip(f"{cell_id} exports a {type(flow).__name__}, not a Flow")

        defects = _topology_defects(flow)
        assert not defects, f"{cell_id} exports a Flow a literal executor cannot run:\n  " + "\n  ".join(defects)


class TestLoopEntersTheBodyBeforeTheCheck:
    """neograph's ``Loop`` is a DO-while, not a while-do.

    ``_wiring._add_subgraph_loop`` wires ``prev -> body`` and only then attaches
    the conditional back-edge, so the body always runs at least once and the
    condition is evaluated against what the body produced. The exported Flow
    used to invert that -- the incoming control edge targeted the check
    ``BranchingNode`` -- which a reachability walk cannot see (the body is still
    reachable through the ``continue`` branch) but which changes what the program
    DOES: a literal executor would evaluate ``when`` against state the body has
    not written yet, and on a false-y first read would skip the body entirely.
    """

    @staticmethod
    def _pipeline() -> Construct:
        from neograph.modifiers import Loop
        from neograph.node import Node
        from tests.schemas import Claims

        node = Node.scripted("refine", fn="refine_fn", inputs=Claims, outputs=Claims) | Loop(
            when="claims_incomplete", max_iterations=3
        )
        return Construct("loop-entry-pipeline", nodes=[node])

    def test_the_flow_enters_the_loop_body_not_the_check_node(self) -> None:
        flow = to_agent_spec(self._pipeline())
        entered = entered_nodes(flow)
        assert entered == {"refine"}, (
            "a do-while loop must be ENTERED at its body; entering at the check node evaluates "
            f"the condition before the body has ever run (entered: {entered})"
        )

    def test_the_loop_exits_through_the_check_nodes_done_branch(self) -> None:
        flow = to_agent_spec(self._pipeline())
        exits = [
            e
            for e in flow.control_flow_connections
            if e.from_node.name == "refine__loop_check" and type(e.to_node).__name__ == "EndNode"
        ]
        # neograph-qtfof.6: DEFAULT_BRANCH now ALSO exits here (a real edge, not a
        # dead end, for a metadata-blind runtime landing there on an unmapped
        # predicate output) -- both labelled, neither ambiguous.
        by_branch = {e.from_branch for e in exits}
        assert by_branch == {Branch.DONE, Branch.DEFAULT}, (
            "the loop's exit edges must be exactly {done, default}, each labelled -- an "
            f"unlabelled edge out of a BranchingNode is ambiguous to a literal executor (got {by_branch})"
        )


class TestOperatorPauseResumesIntoTheFlow:
    """The HITL pause node must continue where the un-paused path continues.

    At runtime (``_wiring._add_operator_check``) the interrupt happens INSIDE the
    check node: once the human answers, execution resumes from that same node and
    proceeds to the next node in the pipeline. So the exported pause
    ``InputMessageNode`` and the check's non-pausing branch must RECONVERGE on the
    same successor -- which is exactly what ``_lower_operator``'s own docstring
    has always claimed ("ControlFlowEdge(from_branch=DEFAULT_BRANCH) ->
    reconverge") but the lowering never emitted.
    """

    @staticmethod
    def _flow() -> Any:
        from neograph.modifiers import Operator
        from tests.schemas import Claims, _producer

        gate = _producer("gate", Claims) | Operator(when="needs_review")
        return to_agent_spec(Construct("operator-resume-pipeline", nodes=[gate]))

    def test_pause_node_and_default_branch_reconverge_on_the_same_successor(self) -> None:
        flow = self._flow()
        after_pause = set(successors_of(flow, "gate__operator_pause"))
        after_default = set(arm_targets(flow, "gate__operator_check", Branch.DEFAULT))
        assert after_pause, "the pause node dead-ends -- a literal executor never resumes"
        assert after_pause == after_default, (
            "the paused and un-paused paths must reconverge on the same successor "
            f"(pause -> {after_pause}, default -> {after_default})"
        )

    def test_the_operator_body_is_entered_before_the_gate(self) -> None:
        flow = self._flow()
        entered = entered_nodes(flow)
        assert entered == {"gate"}, (
            "the Operator gate runs AFTER the body it guards (_wiring._add_operator_check wires "
            f"node -> check), so the flow must enter the body (entered: {entered})"
        )


def _branch_label_defects(flow: Any) -> list[str]:
    """Every way a BranchingNode's outgoing arm labels can disagree with the
    ``mapping`` it declares, across the flow and all its sub-flows."""
    defects: list[str] = []
    for sub in all_flows(flow):
        adjacency = branch_adjacency(sub)
        for node in sub.nodes:
            mapping = getattr(node, "mapping", None)
            if not mapping:
                continue
            # Branch.DEFAULT is ALWAYS a structurally-valid arm on a BranchingNode
            # (pyagentspec's fallback for an unmapped key) independent of what
            # `mapping` declares -- an outgoing edge on it is never "invented"
            # (neograph-qtfof.6 wires one so a metadata-blind runtime landing
            # there, e.g. from a malfunctioning predicate tool, isn't a dead end).
            declared = {*mapping.values(), Branch.DEFAULT}
            emitted = [branch for frm, branch, _to in adjacency if frm == node.name]
            labels = [b for b in emitted if b is not None]
            where = f"{sub.name}: {node.name}"
            if None in emitted:
                defects.append(f"{where} has an UNLABELLED outgoing edge -- ambiguous to a literal executor")
            if invented := sorted(set(labels) - declared):
                defects.append(f"{where} emits {invented}, not in its mapping {sorted(declared)}")
            if missing := sorted(declared - set(labels)):
                defects.append(f"{where} declares {missing} in its mapping with NO outgoing edge")
            if len(labels) != len(set(labels)):
                defects.append(f"{where} emits a DUPLICATE label among {sorted(labels)}")
    return defects


class TestEveryBranchingNodeEmitsTheArmsItDeclares:
    """neograph-dgbqv.10: the matrix validated reachability and liveness but never
    the ARM LABELS, so a branch-relabelling bug was invisible to it.

    Proven by controlled mutation: relabelling the loop's exit arm 'done' ->
    'continue' produces a BranchingNode with two 'continue' successors and no
    'done' successor -- unrunnable for any literal executor, because the executor
    cannot pick a successor for the 'done' outcome and has two candidates for
    'continue'. That mutation was caught by exactly ONE test in the whole tree,
    and only by luck. Under the check below it is caught in many cells at once,
    on two independent clauses.

    The expectation is SELF-DESCRIBING -- it comes from each node's own
    ``mapping`` -- so this needs no per-cell table of expected labels and no
    maintenance as matrix cells are added. That is what makes it worth having
    over per-shape assertions: it scales with the matrix for free.
    """

    @pytest.mark.parametrize("cell_id", sorted(GREEN))
    def test_branch_arms_match_the_declared_mapping(self, cell_id: str) -> None:
        from tests.test_agent_spec_matrix import CELLS

        mode, combo, config, shape = CELLS[cell_id]
        flow = to_agent_spec(build_cell(mode, combo, config, shape))
        if type(flow).__name__ != "Flow":
            pytest.skip(f"{cell_id} exports a {type(flow).__name__}, not a Flow")

        defects = _branch_label_defects(flow)
        assert not defects, (
            f"{cell_id} exports a BranchingNode whose outgoing arms disagree with its own "
            "mapping -- a literal executor cannot resolve the branch:\n  " + "\n  ".join(defects)
        )

    def test_the_matrix_actually_exercises_branching_nodes(self) -> None:
        """Non-vacuity: the check above would pass trivially if no GREEN cell
        exported a BranchingNode at all."""
        from tests.test_agent_spec_matrix import CELLS

        seen = 0
        for cell_id in sorted(GREEN):
            mode, combo, config, shape = CELLS[cell_id]
            flow = to_agent_spec(build_cell(mode, combo, config, shape))
            if type(flow).__name__ != "Flow":
                continue
            seen += sum(
                1 for sub in all_flows(flow) for n in sub.nodes if getattr(n, "mapping", None)
            )
        assert seen >= 20, f"expected the GREEN matrix to exercise many BranchingNodes, saw {seen}"
