"""SPIKE proof (neograph-airpr): Portal+Operator approval gate on the dynamic path.

Portal + Operator on the same node is ILLEGAL in v1 (D-NO-OPERATOR-COMBO/D4):
``_add_operator_check`` wires a STATIC ``node -> {node}__operator`` edge, and a
``Command(goto)``-returning Portal member routes straight to its target without
ever traversing an appended static edge — the approval node would be wired,
look correct, and silently never fire.

This module is the runnable proof that the v2 shape — a dedicated approval node
spliced ONTO the dynamic path — works, and that the naive alternative does not.
It pins mechanics at the raw-LangGraph level (the neograph surface still bans
the combo), mirroring exactly what ``factory.make_portal_fn`` /
``_wiring._add_portal_mesh`` would produce once the feature lands:

  CLEAN (target) shape, proven here to satisfy the spike acceptance:
    member computes its handoff and returns
        Command(goto="{member}__approve", update={proposed target})
    the approval node is the ONLY ``interrupt()`` site; on resume it returns
        Command(goto=approved_target)   (approve)
        Command(goto=exit)              (reject)
    so the pause sits ON the routed path (member -> approve -> target) and a
    resume re-runs ONLY the cheap approval node — the member's LLM/tool spend
    happens EXACTLY ONCE across pause+resume.

  NAIVE (rejected) shape, proven here to FAIL the exactly-once crux:
    ``interrupt()`` inside the member wrapper right before returning the
    Command. LangGraph re-enters the interrupted node FROM THE TOP on resume,
    so the member body (= the LLM/tool calls) re-fires: called twice.

CORE INVARIANT (bead neograph-airpr): the approval pause must sit ON the
dynamic ``Command(goto)`` path, never on a statically-appended edge.

Checkpointer convention: REAL file-backed sqlite (``SqliteSaver`` /
``AsyncSqliteSaver`` via ``tmp_path``), no MemorySaver — copied from
``tests/test_checkpoint_portal.py``. Design decision + surface recommendation:
``docs/design/portal-operator-approval-gate-2026-07-14.md``.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


def _last(_old: Any, new: Any) -> Any:
    return new


class MeshState(TypedDict, total=False):
    proposed_target: Annotated[str, _last]
    handled_by: Annotated[str, _last]


APPROVE_NAME = "triage__approve"
EXIT_NAME = "__handoff_exit_triage"


def _build_clean_graph(saver: Any) -> tuple[Any, list[int]]:
    """The v2 target shape: approval node spliced onto the dynamic path.

    ``member_calls`` counts executions of the member body — the stand-in for
    the LLM/tool spend that must happen exactly once across pause+resume.
    """
    member_calls: list[int] = []

    def triage(state: MeshState) -> Command:
        # Simulates make_portal_fn's wrapper: the inner node (LLM/tools) runs,
        # then instead of goto=<peer> directly, the guarded member routes
        # through its approval node carrying the proposed target.
        member_calls.append(1)
        proposed = "billing"  # "the LLM decided"
        return Command(goto=APPROVE_NAME, update={"proposed_target": proposed})

    def approve(state: MeshState) -> Command:
        # The ONLY interrupt() site. Cheap by construction: no model calls, so
        # LangGraph's re-run-from-the-top resume semantics re-run only this.
        decision = interrupt({"proposed_target": state["proposed_target"]})
        if decision == "approve":
            return Command(goto=state["proposed_target"])
        return Command(goto=EXIT_NAME)

    def billing(state: MeshState) -> dict:
        return {"handled_by": "billing"}

    def handoff_exit(state: MeshState) -> dict:
        return {"handled_by": "exit"}

    g = StateGraph(MeshState)
    g.add_node("triage", triage, destinations=(APPROVE_NAME,))
    g.add_node(APPROVE_NAME, approve, destinations=("billing", EXIT_NAME))
    g.add_node("billing", billing)
    g.add_node(EXIT_NAME, handoff_exit)
    g.add_edge(START, "triage")
    g.add_edge("billing", END)
    g.add_edge(EXIT_NAME, END)
    return g.compile(checkpointer=saver), member_calls


def _build_naive_graph(saver: Any) -> tuple[Any, list[int]]:
    """The rejected shape: interrupt() inside the member, before the Command."""
    member_calls: list[int] = []

    def triage(state: MeshState) -> Command:
        member_calls.append(1)  # the LLM/tool spend...
        proposed = "billing"
        decision = interrupt({"proposed_target": proposed})  # ...then the pause
        target = proposed if decision == "approve" else EXIT_NAME
        return Command(goto=target)

    def billing(state: MeshState) -> dict:
        return {"handled_by": "billing"}

    def handoff_exit(state: MeshState) -> dict:
        return {"handled_by": "exit"}

    g = StateGraph(MeshState)
    g.add_node("triage", triage, destinations=("billing", EXIT_NAME))
    g.add_node("billing", billing)
    g.add_node(EXIT_NAME, handoff_exit)
    g.add_edge(START, "triage")
    g.add_edge("billing", END)
    g.add_edge(EXIT_NAME, END)
    return g.compile(checkpointer=saver), member_calls


class TestApprovalNodeOnDynamicPath:
    """Spike acceptance #1 + #2: pause on the routed path, exactly-once member."""

    def test_pause_surfaces_proposed_target_then_approval_routes_there(self, tmp_path):
        with SqliteSaver.from_conn_string(str(tmp_path / "clean.db")) as saver:
            graph, member_calls = _build_clean_graph(saver)
            cfg = {"configurable": {"thread_id": "t-approve"}}

            paused = graph.invoke({}, cfg)
            assert "__interrupt__" in paused, "run must pause at the approval node"
            assert paused["__interrupt__"][0].value == {"proposed_target": "billing"}, (
                "the interrupt payload must surface the PROPOSED handoff target"
            )
            assert len(member_calls) == 1, "member ran once before the pause"

            done = graph.invoke(Command(resume="approve"), cfg)
            assert "__interrupt__" not in done
            assert done["handled_by"] == "billing", "approval must route to the approved target"
            # THE CRUX (acceptance #2): resume re-ran only the approval node.
            assert len(member_calls) == 1, (
                f"member executed {len(member_calls)}x across pause+resume — "
                "the LLM/tool spend must happen EXACTLY ONCE"
            )

    def test_rejection_routes_to_exit_without_rerunning_member(self, tmp_path):
        with SqliteSaver.from_conn_string(str(tmp_path / "reject.db")) as saver:
            graph, member_calls = _build_clean_graph(saver)
            cfg = {"configurable": {"thread_id": "t-reject"}}

            paused = graph.invoke({}, cfg)
            assert "__interrupt__" in paused

            done = graph.invoke(Command(resume="reject"), cfg)
            assert done["handled_by"] == "exit", "rejection must route to the mesh exit"
            assert len(member_calls) == 1

    @pytest.mark.asyncio
    async def test_async_parity_exactly_once_member(self, tmp_path):
        async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "async.db")) as saver:
            graph, member_calls = _build_clean_graph(saver)
            cfg = {"configurable": {"thread_id": "t-async"}}

            paused = await graph.ainvoke({}, cfg)
            assert "__interrupt__" in paused
            assert paused["__interrupt__"][0].value == {"proposed_target": "billing"}

            done = await graph.ainvoke(Command(resume="approve"), cfg)
            assert done["handled_by"] == "billing"
            assert len(member_calls) == 1


class TestNaiveInWrapperShapeFailsTheCrux:
    """The contrast that justifies rejecting the naive shape: LangGraph resumes
    an interrupted node FROM THE TOP, so an interrupt inside the member wrapper
    re-fires the member body (the LLM/tool spend) on resume.
    """

    def test_member_body_reruns_on_resume(self, tmp_path):
        with SqliteSaver.from_conn_string(str(tmp_path / "naive.db")) as saver:
            graph, member_calls = _build_naive_graph(saver)
            cfg = {"configurable": {"thread_id": "t-naive"}}

            paused = graph.invoke({}, cfg)
            assert "__interrupt__" in paused
            assert len(member_calls) == 1

            done = graph.invoke(Command(resume="approve"), cfg)
            assert done["handled_by"] == "billing", "routing still works — cost is the problem"
            assert len(member_calls) == 2, (
                "expected the naive shape to RE-RUN the member on resume "
                "(double LLM spend) — if this is now 1, LangGraph's resume "
                "semantics changed and the D4 design should be revisited"
            )
