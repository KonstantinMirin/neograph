"""A ``BranchingNode``'s decision must arrive over a REAL data path (neograph-qtfof.6).

Core Invariant (from the ticket): an exported Agent Spec Flow must be correctly
wired and executable by a METADATA-BLIND runtime using ONLY ControlFlowEdge and
DataFlowEdge -- every BranchingNode's branch-key input must arrive over a declared
data path from a real node output, and every branch value plus DEFAULT_BRANCH must
have an outgoing ControlFlowEdge.

Why this is a SILENT seam rather than missing metadata. pyagentspec's
``BranchingNodeExecutor._execute`` does ``inputs.get(title, DEFAULT_BRANCH)`` and
then ``mapping.get(input_branch_name, DEFAULT_BRANCH)``. With no DataFlowEdge
feeding the branch-key input, a metadata-blind runtime does not error -- it
silently falls through to the default arm. For a Loop that means the back-edge is
never taken and the loop body runs exactly once, while the exported Flow LOOKS
fully wired. Under the North Star that is the single-silent-seam class the project
forbids, so the fix must be a real edge (or a loud conformance finding), never a
richer ``neograph/loop_spec`` metadata mirror.

SCOPE: Loop only, deliberately. Operator is out of scope for this ticket and keeps
firing ``modifier_metadata_only_branch`` -- its ``when`` is an opaque registry NAME
whose ``condition_fn(state)`` may read any field of the whole state bus, so its
predicate's source node is genuinely undeterminable and no honest DataFlowEdge can
be synthesized for it. Loop is the asymmetric case: ``_make_loop_router`` calls
``condition(unwrap(bus, field_name))`` against the loop BODY's own latest output,
so the source is statically determinable in every case.

NOT CLAIMED: that the exported Flow reaches ``ConformanceTier.PORTABLE``. A
third-party runtime still cannot EXECUTE a neograph predicate; it must resolve the
declared ServerTool by name. The gain this pins is that failure becomes LOUD
(unresolved tool at load) instead of silent (wrong branch).

Gated on ``pyagentspec`` (the ``[agent-spec]`` extra keeps ``src/neograph``
dependency-light)::

    uv run pytest tests/test_agent_spec_branch_predicate_data_flow.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyagentspec")

from pydantic import BaseModel  # noqa: E402

from neograph import Construct, Loop, Node  # noqa: E402
from neograph._agent_spec import Branch, to_agent_spec  # noqa: E402
from tests.agent_spec_flow_walk import (  # noqa: E402
    arm_targets,
    data_flow_predecessors,
    declared_input_titles,
    fed_inputs,
    successors_of,
    wired_edges,
)
from tests.fakes import register_scripted  # noqa: E402

_BODY = "refine"


class Draft(BaseModel, frozen=True):
    content: str = ""
    score: float = 0.0


def _loop_pipeline() -> Construct:
    """``seed -> refine``, with ``refine`` self-looping until ``score >= 0.8``.

    The minimal shape that exercises the seam: a single Loop-modified body whose
    ``when`` is an EXPRESSION over the body's own output field, i.e. exactly the
    case whose predicate source IS statically determinable.
    """
    register_scripted("qtfof6_seed", lambda i, c: Draft())
    register_scripted("qtfof6_refine", lambda i, c: Draft())
    return Construct(
        "qtfof6-loop",
        nodes=[
            Node.scripted("seed", fn="qtfof6_seed", outputs=Draft),
            Node.scripted(_BODY, fn="qtfof6_refine", inputs=Draft, outputs=Draft)
            | Loop(when="score < 0.8", max_iterations=3),
        ],
    )


def _branching_node(flow: object) -> object:
    branching = [n for n in flow.nodes if type(n).__name__ == "BranchingNode"]
    assert len(branching) == 1, f"expected exactly one BranchingNode, got {[n.name for n in branching]}"
    return branching[0]


class TestLoopBranchDecisionIsWired:
    """The three properties a metadata-blind runtime needs to route a Loop."""

    def test_the_branching_node_has_an_incoming_data_flow_edge(self):
        """(a) The branch-key input arrives over a declared data path.

        Asserted against the node's OWN declared input titles rather than the
        literal ``"branching_mapping_key"``: pyagentspec honours
        ``inputs[0].title``, so the title is a CHOICE the lowering makes, not a
        constraint it must satisfy. Pinning the literal would pin the wrong thing
        and would go blind the day the lowering picks another title.
        """
        flow = to_agent_spec(_loop_pipeline())
        branch = _branching_node(flow)

        unfed = [t for t in declared_input_titles(branch) if t not in fed_inputs(flow, branch.name)]
        assert not unfed, (
            f"BranchingNode {branch.name!r} declares input(s) {unfed} that NO DataFlowEdge feeds. "
            "pyagentspec's BranchingNodeExecutor does inputs.get(title, DEFAULT_BRANCH), so a "
            "metadata-blind runtime does not error -- it silently takes the default arm and the "
            "loop back-edge is never followed. The decision currently lives only in "
            f"metadata['neograph/loop_spec']. Wired edges: {wired_edges(flow)}"
        )

    def test_every_branch_value_and_the_default_branch_has_an_outgoing_control_edge(self):
        """(b) No branch the executor can select is a dead end.

        ``mapping`` values are the arms neograph intends; ``Branch.DEFAULT`` is the
        arm the executor falls to for ANY unmapped key -- reachable at runtime
        whether or not neograph means it to be, so it needs a target too.
        """
        flow = to_agent_spec(_loop_pipeline())
        branch = _branching_node(flow)

        selectable = {*branch.mapping.values(), Branch.DEFAULT}
        missing = sorted(arm for arm in selectable if not arm_targets(flow, branch.name, arm))
        assert not missing, (
            f"BranchingNode {branch.name!r} can select branch(es) {missing} that have no outgoing "
            f"ControlFlowEdge -- a runtime landing there stalls. mapping={branch.mapping}, "
            f"declared branches={branch.branches}, successors={successors_of(flow, branch.name)}"
        )

    def test_the_predicate_source_node_has_its_own_inputs_fed(self):
        """(c) The synthesized predicate node is not a NEW unfed seam one hop up.

        Moving the unfed input from the BranchingNode onto a synthesized predicate
        node would relocate the silent seam rather than close it: the predicate
        would receive nothing and a metadata-blind runtime would be exactly as
        wrong, one node further from where anyone would look.
        """
        flow = to_agent_spec(_loop_pipeline())
        branch = _branching_node(flow)
        by_name = {n.name: n for n in flow.nodes}

        sources = data_flow_predecessors(flow, branch.name)
        assert sources, (
            f"nothing feeds BranchingNode {branch.name!r} at all, so there is no predicate node to "
            "check -- close (a) first, then this assertion becomes the real one"
        )

        unfed = {
            src: [t for t in declared_input_titles(by_name[src]) if t not in fed_inputs(flow, src)] for src in sources
        }
        offenders = {src: titles for src, titles in unfed.items() if titles}
        assert not offenders, (
            "the node(s) feeding the branch decision declare input(s) that no DataFlowEdge feeds: "
            f"{offenders}. The seam moved one hop upstream instead of closing."
        )

    def test_the_branch_key_is_not_fed_straight_from_the_loop_body(self):
        """The branch key is a mapping KEY (continue/done), not the body's value.

        Wiring ``refine``'s output directly into the branch-key input would produce
        a structurally-complete-LOOKING Flow that is silently wrong: the input is a
        StringProperty and the body emits a Draft model, so the value never matches
        a mapping key and every iteration hits the default arm. That is strictly
        WORSE than today's honest gap, because the wiring assertions above would
        then pass while the routing stayed broken.
        """
        flow = to_agent_spec(_loop_pipeline())
        branch = _branching_node(flow)
        titles = set(declared_input_titles(branch))

        feeders = [
            (src, out) for src, out, dst, dest_input in wired_edges(flow) if dst == branch.name and dest_input in titles
        ]
        assert feeders, (
            f"no edge feeds any declared input of {branch.name!r} -- see the (a) assertion; "
            "this test only becomes meaningful once one exists"
        )
        assert _BODY not in {src for src, _out in feeders}, (
            f"the loop body {_BODY!r} feeds the branch key directly ({feeders}). The body emits a "
            "Draft model; the branch key is a mapping KEY (continue/done). That edge type-mismatches "
            "and always resolves to the default arm -- a Flow that looks wired and routes wrong."
        )
