"""A gated Portal mesh's exported Flow: what is broken, and what BLOCKS fixing it
(neograph-qtfof.12).

Two separable halves, deliberately graded differently.

**Half 1 -- the MISATTRIBUTION (fixable now, asserted RED->GREEN).** The UNGATED
peer mesh exports a ``Swarm``, and ``_flow_findings`` early-returns
``portal_mesh_unverified`` for it -- an honest "this shape has no GREEN cell, so
absence of a failure signal is not evidence". The GATED mesh exports a **Flow**
(the mesh-exit pause composite), which falls PAST that early return into
``_check_outermost_end_node`` and is reported as ``outermost_end_node_unwired``,
i.e. attributed to **neograph-qtfof.9** -- a ticket that has LANDED and whose fix
provably does not cover this shape. A closed bead is the wrong address for an open
gap: it makes the report say "already fixed" about something that is not.

**Half 2 -- the DATA PLANE (blocked, asserted as measured BLOCKERS).** The same
Flow's two EndNodes declare no outputs and it carries no
``data_flow_connections`` at all. qtfof.12's own acceptance requires that fix be
"verified by executing it under the real pyagentspec AgentSpecLoader", and its
description states the prerequisite plainly: fixing an unexercised path blind is
how a second silent seam gets added. That prerequisite is currently UNMEETABLE,
and the tests below pin exactly why, against the installed SDK rather than by
assertion:

  * an UNGATED mesh's ``Swarm`` carries ``handoff=NEVER``, which the LangGraph
    adapter refuses outright (``_swarm_convert_to_langgraph``);
  * a GATED mesh's Flow LOADS, but running it raises ``TypeError:
    AgentNodeExecutor can only be used with AgentSpecAgent agents`` -- pyagentspec
    26.1.2 accepts ``AgentNode(agent=Swarm)`` at the SCHEMA level
    (``SerializeAsAny[AgenticComponent]``) while its own reference runtime cannot
    execute it.

These blocker tests are written to FAIL LOUD the day either stops being true --
an SDK upgrade or an export-shape change. That failure is the signal that
qtfof.12's prerequisite has been met and its data-plane half can finally be fixed
against real evidence. Until then the half stays unfixed ON PURPOSE, and this
module is the record of why, in place of a permanently-red test or a blind patch.

Gated on ``pyagentspec``::

    uv run pytest tests/test_agent_spec_portal_gated_boundary.py
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pyagentspec")

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from pyagentspec.adapters.langgraph import AgentSpecLoader  # noqa: E402

from neograph import Construct, Node, Portal, export_conformance  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from neograph._agent_spec_conformance import CONFORMANCE_PREDICATE_META, ConformanceTier  # noqa: E402
from tests.agent_spec_flow_walk import wired_edges  # noqa: E402
from tests.test_agent_spec_portal_operator import (  # noqa: E402
    _one_gated_mesh,
    _Payload,
    _register,
)


def _ungated_mesh() -> Construct:
    """The sibling shape: same mesh, no Operator on any member -- exports as a
    bare ``Swarm`` rather than the gated Flow composite."""
    _register()
    return Construct(
        "pg_ungated",
        nodes=[
            Node.scripted("a", fn="_po_a", inputs={"handoff": _Payload}, outputs=_Payload) | Portal(to=["b"]),
            Node.scripted("b", fn="_po_b", inputs={"handoff": _Payload}, outputs=_Payload) | Portal(to=[]),
        ],
    )


def _predicates(construct: Construct) -> list[str]:
    return [f.predicate for f in export_conformance(construct, strict=False).findings]


def _beads(construct: Construct) -> set[str]:
    return {CONFORMANCE_PREDICATE_META[p].bead for p in _predicates(construct)}


def _end_nodes(flow: Any) -> list[Any]:
    return [n for n in flow.nodes if type(n).__name__ == "EndNode"]


class TestTheGatedMeshIsNotAttributedToAClosedTicket:
    """Half 1: the report must address the gap that is actually open."""

    def test_a_gated_mesh_is_not_reported_as_the_landed_qtfof_9_gap(self) -> None:
        beads = _beads(_one_gated_mesh())

        assert "neograph-qtfof.9" not in beads, (
            "a gated Portal mesh's Flow is reported under neograph-qtfof.9 "
            f"(outermost_end_node_unwired). Findings: {_predicates(_one_gated_mesh())}. That ticket has "
            "LANDED and its fix (a BARE/ORACLE/EACH terminal producer wired into the EndNode) provably "
            "does not reach this shape -- the gated mesh has no construct.nodes terminal to wire, it has "
            "an AgentNode wrapping a Swarm. Reporting an open gap against a closed bead makes the "
            "conformance report say 'already fixed' about something that is not."
        )

    def test_a_gated_mesh_carries_the_same_unverified_predicate_its_ungated_sibling_does(self) -> None:
        """The two shapes differ only by an Operator on a member; both are peer
        meshes with zero GREEN cells, so both deserve the SAME honest verdict."""
        assert "portal_mesh_unverified" in _predicates(_one_gated_mesh()), (
            f"gated mesh findings: {_predicates(_one_gated_mesh())}; "
            f"ungated sibling findings: {_predicates(_ungated_mesh())}"
        )

    def test_the_verdict_itself_is_unchanged_by_the_re_attribution(self) -> None:
        """Re-attribution must not quietly PROMOTE the shape: it is still not
        portable, it is just filed under the right open gap now."""
        assert export_conformance(_one_gated_mesh(), strict=False).verdict is ConformanceTier.NEOGRAPH_ROUND_TRIP_ONLY


class TestTheDataPlaneHalfIsBlockedByTheSdk:
    """Half 2: the prerequisite, measured -- these pass TODAY and must fail loudly
    the day the blocker lifts, which is the signal to finish neograph-qtfof.12."""

    def test_an_ungated_peer_mesh_cannot_be_converted_at_all(self) -> None:
        swarm = to_agent_spec(_ungated_mesh())
        assert type(swarm).__name__ == "Swarm"

        with pytest.raises(ValueError, match="Handoff mode NEVER is not supported"):
            AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver()).load_dict(swarm.to_dict())

    def test_a_gated_mesh_loads_but_cannot_run(self) -> None:
        """LOADING is not evidence -- this is the exact "did not raise == is
        portable" substitution the conformance epic exists to retire."""
        flow = to_agent_spec(_one_gated_mesh())
        graph = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver()).load_dict(flow.to_dict())

        with pytest.raises(TypeError, match="AgentNodeExecutor can only be used with AgentSpecAgent"):
            graph.invoke({}, config={"configurable": {"thread_id": "qtfof12"}})

    def test_the_data_plane_is_still_empty_pending_that_prerequisite(self) -> None:
        """The unfixed half, recorded as a FACT rather than as a red test.

        Flipping this to declared outputs + real edges is qtfof.12's remaining
        work; doing it while the two tests above still pass would be a blind fix
        of an unexecutable path.
        """
        flow = to_agent_spec(_one_gated_mesh())

        assert [n.name for n in _end_nodes(flow)] == ["po_mesh1__end_default", "po_mesh1__end_paused"]
        assert all(not (n.outputs or n.inputs) for n in _end_nodes(flow)), (
            "the gated mesh's EndNodes now declare I/O -- if that was done deliberately, "
            "neograph-qtfof.12's data-plane half has landed and this test must be replaced by the "
            "execution-level assertion its acceptance actually requires"
        )
        assert wired_edges(flow) == [], (
            "the gated mesh Flow now carries data_flow_connections -- same as above: replace this "
            "record with the executable proof qtfof.12's acceptance demands"
        )
