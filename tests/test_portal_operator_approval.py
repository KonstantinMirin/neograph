"""Regression test for kdr1u (Portal+Operator first-class human-approval gate).

Pins the v2 shape proven by the spike (neograph-airpr,
``tests/test_spike_portal_operator_approval.py``) on the REAL neograph
surface -- not raw LangGraph. Full design:
``docs/design/portal-operator-approval-gate-2026-07-14.md``.

CORE INVARIANT under test: an Operator-guarded Portal member routes
``Command(goto="{member}__approve")``; the approval node is the ONLY
``interrupt()`` site and, on resume, itself emits
``Command(goto=approved_target)`` (approve) or ``Command(goto=<mesh exit>)``
(reject) -- so the pause sits ON the dynamic ``Command(goto)`` path and the
member's body (its LLM/tool spend) runs EXACTLY ONCE across pause+resume.

MUST FAIL NOW: Portal + Operator on the same node is still banned by THREE
reciprocal excludes in ``modifiers.py`` (the Operator row + the Portal row in
``_SLOT_RULES``, and the pairwise arm in ``ModifierSet.model_post_init``).
Every test below that pipes ``Node.scripted(...) | Portal(...) |
Operator(when=...)`` (or the reverse order, or the ``@node`` decorator
equivalent) raises ``ConstructError: Cannot combine Portal and Operator on
the same item`` at construct-assembly time, before the graph ever compiles or
runs -- this IS the correct TDD-red failure: the feature kdr1u implements
does not exist yet.

Checkpointer convention: REAL file-backed sqlite (``SqliteSaver`` /
``AsyncSqliteSaver`` via ``tmp_path``, no ``MemorySaver``) -- copied from
``tests/test_checkpoint_portal.py``. Resume shape (``run(graph,
resume={"approved": bool}, config=cfg)``) copied from the same file's
``TestCounterPersistsAcrossResume`` (D4-era Operator-after-exit convention);
the crux is that once kdr1u lands, that SAME resume shape must resolve the
ROUTING decision (approve -> proposed peer, reject -> mesh exit) instead of
merely unblocking a static downstream edge.
"""

from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

import neograph
from neograph import HANDOFF_END, Construct, Node, Operator, Portal, compile, node, run
from neograph._state_keys import StateKeys
from tests.fakes import build_test_compile_kwargs, register_condition, register_scripted


class RouteHop(BaseModel, frozen=True):
    """Uniform mesh payload -- mirrors test_checkpoint_portal.py's RouteHop."""

    goto: str


def _register_approval_mesh(tag: str) -> tuple[Node, Node, list[int]]:
    """One-hop mesh: ``triage`` (Operator-guarded entry) -> ``billing`` (peer).

    ``member_calls`` counts executions of the ``triage`` body -- the
    stand-in for the LLM/tool spend that must happen EXACTLY ONCE across
    pause+resume (the spike's crux, ported to the real surface).
    """
    member_calls: list[int] = []

    def triage_fn(input_data: object, config: object) -> RouteHop:
        member_calls.append(1)
        return RouteHop(goto="billing")

    def billing_fn(input_data: object, config: object) -> RouteHop:
        return RouteHop(goto=HANDOFF_END)

    register_scripted(f"{tag}_triage", triage_fn)
    register_scripted(f"{tag}_billing", billing_fn)
    # The approval gate: pauses on triage's FIRST arrival, surfacing the
    # proposed handoff target as the interrupt payload (mirrors
    # _add_operator_check's `interrupt(should_pause)` -- should_pause IS the
    # payload the human sees).
    register_condition(
        f"{tag}_gate",
        lambda state: (
            {"proposed_target": state.triage.goto} if getattr(state, "triage", None) is not None else None
        ),
    )

    entry = (
        Node.scripted("triage", fn=f"{tag}_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
        | Portal(to=["billing"])
        | Operator(when=f"{tag}_gate")
    )
    billing = Node.scripted("billing", fn=f"{tag}_billing", inputs={"handoff": RouteHop}, outputs=RouteHop) | Portal(
        to=["triage"]
    )
    return entry, billing, member_calls


class TestApprovalGateOnDynamicPath:
    """The crux: pause sits ON the routed path; member runs exactly once."""

    def test_sync_approve_routes_to_peer_and_runs_member_exactly_once(self, tmp_path):
        db = str(tmp_path / "approve_sync.db")
        cfg = {"configurable": {"thread_id": "kdr1u-approve-sync"}}
        with SqliteSaver.from_conn_string(db) as saver:
            entry, billing, member_calls = _register_approval_mesh("kdr1u_as")
            graph = compile(
                Construct("kdr1u-approve-sync", nodes=[entry, billing]),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )

            paused = run(graph, input={}, config=cfg)
            assert "__interrupt__" in paused, "the approval node must be the ONLY interrupt() site"
            assert paused["__interrupt__"][0].value == {"proposed_target": "billing"}, (
                "the interrupt payload must surface the PROPOSED handoff target"
            )
            assert len(member_calls) == 1, "triage's body ran once before the pause"

            completed = run(graph, resume={"approved": True}, config=cfg)
            assert "__interrupt__" not in completed, "approval must resume the run to completion"
            assert graph.get_state(cfg).values.get("billing") == RouteHop(goto=HANDOFF_END), (
                "approval must route to the PROPOSED peer (billing), not re-derive it"
            )
            # THE CRUX: resume re-ran ONLY the (cheap) approval node.
            assert len(member_calls) == 1, (
                f"triage executed {len(member_calls)}x across pause+resume -- the "
                "member's LLM/tool spend must happen EXACTLY ONCE (D4 anti-band-aid)"
            )
            # An approved hop costs exactly one peer-hop.
            assert graph.get_state(cfg).values.get(StateKeys.handoff_hops("triage")) == 1

    def test_sync_reject_routes_to_mesh_exit_without_rerunning_member(self, tmp_path):
        db = str(tmp_path / "reject_sync.db")
        cfg = {"configurable": {"thread_id": "kdr1u-reject-sync"}}
        with SqliteSaver.from_conn_string(db) as saver:
            entry, billing, member_calls = _register_approval_mesh("kdr1u_rs")
            graph = compile(
                Construct("kdr1u-reject-sync", nodes=[entry, billing]),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )

            paused = run(graph, input={}, config=cfg)
            assert "__interrupt__" in paused
            assert len(member_calls) == 1

            completed = run(graph, resume={"approved": False}, config=cfg)
            assert "__interrupt__" not in completed
            # Rejection must NOT reach the (never-approved) peer.
            assert "billing" not in completed or completed.get("billing") is None
            assert len(member_calls) == 1, "rejection must not re-run the member either"
            # A rejected hop must not be counted as a completed peer-hop.
            assert graph.get_state(cfg).values.get(StateKeys.handoff_hops("triage"), 0) == 0

    async def test_async_approve_parity_exactly_once_member(self, tmp_path):
        db = str(tmp_path / "approve_async.db")
        cfg = {"configurable": {"thread_id": "kdr1u-approve-async"}}
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            entry, billing, member_calls = _register_approval_mesh("kdr1u_aa")
            graph = compile(
                Construct("kdr1u-approve-async", nodes=[entry, billing]),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )

            paused = await neograph.arun(graph, input={}, config=cfg)
            assert "__interrupt__" in paused
            assert paused["__interrupt__"][0].value == {"proposed_target": "billing"}

            completed = await neograph.arun(graph, resume={"approved": True}, config=cfg)
            assert "__interrupt__" not in completed
            assert len(member_calls) == 1, "async parity: member must also run exactly once"


class TestBanLiftBothPipeOrders:
    """LOW finding: 3 reciprocal edit sites; both pipe orders must assemble."""

    def test_portal_then_operator_assembles(self):
        register_scripted("kdr1u_order_a_fn", lambda i, c: RouteHop(goto="peer"))
        register_condition("kdr1u_order_a_gate", lambda state: None)
        Node.scripted("member_a", fn="kdr1u_order_a_fn", outputs=RouteHop) | Portal(to=["peer"]) | Operator(
            when="kdr1u_order_a_gate"
        )

    def test_operator_then_portal_assembles(self):
        register_scripted("kdr1u_order_b_fn", lambda i, c: RouteHop(goto="peer"))
        register_condition("kdr1u_order_b_gate", lambda state: None)
        Node.scripted("member_b", fn="kdr1u_order_b_fn", outputs=RouteHop) | Operator(when="kdr1u_order_b_gate") | Portal(
            to=["peer"]
        )


class TestNodeDecoratorSurfaceParity:
    """Three-surface parity: the @node decorator's `interrupt_when=` sugar
    (which attaches an Operator under the hood) must compose with a piped
    Portal exactly like the programmatic ``Node.scripted | Portal |
    Operator`` form above -- same exactly-once crux.
    """

    def test_node_decorator_interrupt_when_piped_with_portal(self, tmp_path):
        db = str(tmp_path / "decorator_surface.db")
        cfg = {"configurable": {"thread_id": "kdr1u-decorator-surface"}}
        member_calls: list[int] = []

        @node(
            mode="scripted",
            outputs=RouteHop,
            interrupt_when=lambda state: (
                {"proposed_target": state.triage.goto} if getattr(state, "triage", None) is not None else None
            ),
        )
        def triage() -> RouteHop:
            member_calls.append(1)
            return RouteHop(goto="billing")

        triage_member = triage | Portal(to=["billing"])

        def billing_fn(input_data: object, config: object) -> RouteHop:
            return RouteHop(goto=HANDOFF_END)

        register_scripted("kdr1u_decorator_billing", billing_fn)
        billing_member = Node.scripted(
            "billing", fn="kdr1u_decorator_billing", inputs={"handoff": RouteHop}, outputs=RouteHop
        ) | Portal(to=["triage"])

        with SqliteSaver.from_conn_string(db) as saver:
            graph = compile(
                Construct("kdr1u-decorator-surface", nodes=[triage_member, billing_member]),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )

            paused = run(graph, input={}, config=cfg)
            assert "__interrupt__" in paused
            assert len(member_calls) == 1

            completed = run(graph, resume={"approved": True}, config=cfg)
            assert "__interrupt__" not in completed
            assert len(member_calls) == 1, "the @node decorator surface must also honor the exactly-once crux"
