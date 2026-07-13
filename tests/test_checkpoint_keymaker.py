"""KEYMAKER hop-budget checkpoint semantics (T3 — neograph-0umvg).

Pins the DURABILITY half of the hop budget against a REAL file-backed sqlite
checkpointer (per the project's checkpoint-test convention — no MemorySaver, no
mocks): every peer-hop is one checkpointed superstep, and the shared entry-keyed
hop counter (``neo_handoff_hops_<entry_field>``) persists verbatim across an
interrupt+resume on the same ``thread_id``.

BINDING SEMANTICS (design §3.4/§6/§7, decision-log D11/D12):
  - A "hop" = a member routing to a PEER. The entry's own first execution is NOT
    a hop; the first peer route is hop 1. The counter counts peer-hops.
  - The counter is a plain ``(int, 0)`` state field, ``neo_``-prefixed, so it is
    excluded from the schema fingerprint (no resume-invalidation) but DOES
    persist in the checkpoint like any other state field.
  - D4: Keymaker + Operator on the SAME node is illegal, so the interrupt for the
    resume test is an Operator on the node placed AFTER the mesh's exit node.

These pin behavior that does not exist yet — the T3 counter RMW at the
factory.py make_keymaker_fn seam (~lines 156-160) is a comment — so they MUST
FAIL now. File-backed sqlite setup copied from
``tests/test_checkpoint_auto_rewind.py`` (sync ``SqliteSaver`` /
async ``AsyncSqliteSaver`` via ``tmp_path``); interrupt+resume shape copied from
``tests/test_checkpoint_sqlite_async.py``.
"""

from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

import neograph
from neograph import HANDOFF_END, Construct, Keymaker, Node, compile, run
from neograph._state_keys import StateKeys
from tests.fakes import build_test_compile_kwargs, register_condition, register_scripted


class RouteHop(BaseModel, frozen=True):
    """Uniform mesh payload with a plain-str route field + carried hop count."""

    goto: str
    hops: int = 0


def _register_counting_mesh(tag: str, threshold: int) -> tuple[Node, Node]:
    """Register a ping-pong mesh that makes EXACTLY ``threshold`` peer-hops then
    routes to HANDOFF_END. Entry is ``triage``, peer is ``billing``. Names are
    ``tag``-scoped so multiple meshes coexist within a test module.
    """

    def triage_fn(input_data, config):
        incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
        h = 0 if incoming is None else incoming.hops
        if h >= threshold:
            return RouteHop(goto=HANDOFF_END, hops=h)
        return RouteHop(goto="billing", hops=h + 1)

    def billing_fn(input_data, config):
        incoming = input_data["handoff"]
        h = incoming.hops
        if h >= threshold:
            return RouteHop(goto=HANDOFF_END, hops=h)
        return RouteHop(goto="triage", hops=h + 1)

    register_scripted(f"{tag}_triage", triage_fn)
    register_scripted(f"{tag}_billing", billing_fn)

    entry = (
        Node.scripted("triage", fn=f"{tag}_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
        | Keymaker(peers=["billing"], max_hops=threshold + 5)  # budget slack: this mesh self-terminates
    )
    billing = (
        Node.scripted("billing", fn=f"{tag}_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
        | Keymaker(peers=["triage"])
    )
    return entry, billing


_COUNTER_KEY = StateKeys.handoff_hops("triage")


# ═══════════════════════════════════════════════════════════════════════════
# (a) hops-as-supersteps — each peer-hop is one checkpointed superstep
# ═══════════════════════════════════════════════════════════════════════════
#
# Reviewer LOW: do NOT assert len(history) == raw hop count — get_state_history
# also includes the initial input checkpoint + the __handoff_exit_ pass-through.
# Instead: (1) run a 1-hop and a 3-hop mesh, assert the history-length DELTA
# equals the hop DELTA (2) — a fixed per-run overhead cancels out; and (2) pin
# the persisted counter equals the peer-hop count (the T3 property).


class TestHopsAsSupersteps:
    def test_sync_history_grows_one_superstep_per_peer_hop(self, tmp_path):
        db = str(tmp_path / "km_hops_sync.db")
        with SqliteSaver.from_conn_string(db) as saver:
            e1, b1 = _register_counting_mesh("km_hs1", threshold=1)
            g1 = compile(Construct("km-hs-1", nodes=[e1, b1]), checkpointer=saver, **build_test_compile_kwargs())
            cfg1 = {"configurable": {"thread_id": "km-hops-sync-1"}}
            run(g1, input={}, config=cfg1)
            hist1 = list(g1.get_state_history(cfg1))

            e3, b3 = _register_counting_mesh("km_hs3", threshold=3)
            g3 = compile(Construct("km-hs-3", nodes=[e3, b3]), checkpointer=saver, **build_test_compile_kwargs())
            cfg3 = {"configurable": {"thread_id": "km-hops-sync-3"}}
            run(g3, input={}, config=cfg3)
            hist3 = list(g3.get_state_history(cfg3))

            # Fixed per-run overhead cancels: 2 more peer-hops => 2 more supersteps.
            assert len(hist3) - len(hist1) == 3 - 1, (len(hist1), len(hist3))
            # And the persisted counter equals the peer-hop count (T3).
            assert g1.get_state(cfg1).values.get(_COUNTER_KEY) == 1
            assert g3.get_state(cfg3).values.get(_COUNTER_KEY) == 3

    async def test_async_history_grows_one_superstep_per_peer_hop(self, tmp_path):
        db = str(tmp_path / "km_hops_async.db")
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            e1, b1 = _register_counting_mesh("km_ha1", threshold=1)
            g1 = compile(Construct("km-ha-1", nodes=[e1, b1]), checkpointer=saver, **build_test_compile_kwargs())
            cfg1 = {"configurable": {"thread_id": "km-hops-async-1"}}
            await neograph.arun(g1, input={}, config=cfg1)
            hist1 = [s async for s in g1.aget_state_history(cfg1)]

            e3, b3 = _register_counting_mesh("km_ha3", threshold=3)
            g3 = compile(Construct("km-ha-3", nodes=[e3, b3]), checkpointer=saver, **build_test_compile_kwargs())
            cfg3 = {"configurable": {"thread_id": "km-hops-async-3"}}
            await neograph.arun(g3, input={}, config=cfg3)
            hist3 = [s async for s in g3.aget_state_history(cfg3)]

            assert len(hist3) - len(hist1) == 3 - 1, (len(hist1), len(hist3))
            assert (await g1.aget_state(cfg1)).values.get(_COUNTER_KEY) == 1
            assert (await g3.aget_state(cfg3)).values.get(_COUNTER_KEY) == 3


# ═══════════════════════════════════════════════════════════════════════════
# (b) counter persistence across an interrupt+resume (Operator AFTER the exit)
# ═══════════════════════════════════════════════════════════════════════════
#
# D4: Operator on a mesh MEMBER is illegal, so the interrupt is an Operator on
# the ``gate`` node placed AFTER the mesh's pass-through exit. The mesh runs to
# completion (counter set) BEFORE the interrupt; we read the counter at the
# interrupt point, resume on the same thread_id, and assert the run completes
# and the counter value persisted verbatim.


def _register_gate_and_finalize(tag: str) -> tuple[Node, Node]:
    def gate_fn(input_data, config):
        return RouteHop(goto="review")

    def finalize_fn(input_data, config):
        return RouteHop(goto="done")

    register_scripted(f"{tag}_gate", gate_fn)
    register_scripted(f"{tag}_finalize", finalize_fn)
    # Interrupt on first arrival at the gate (Operator inserts a check node after
    # ``gate`` that calls interrupt()); resume consumes it and continues.
    register_condition(f"{tag}_pause", lambda state: {"pause": True} if getattr(state, "gate", None) else None)

    from neograph import Operator

    gate = Node.scripted("gate", fn=f"{tag}_gate", inputs=RouteHop, outputs=RouteHop) | Operator(when=f"{tag}_pause")
    finalize = Node.scripted("finalize", fn=f"{tag}_finalize", inputs=RouteHop, outputs=RouteHop)
    return gate, finalize


class TestCounterPersistsAcrossResume:
    def test_sync_counter_survives_interrupt_and_resume(self, tmp_path):
        db = str(tmp_path / "km_resume_sync.db")
        cfg = {"configurable": {"thread_id": "km-resume-sync"}}
        with SqliteSaver.from_conn_string(db) as saver:
            entry, billing = _register_counting_mesh("km_rs", threshold=3)
            gate, finalize = _register_gate_and_finalize("km_rs")
            graph = compile(
                Construct("km-resume", nodes=[entry, billing, gate, finalize]),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )
            paused = run(graph, input={}, config=cfg)
            assert "__interrupt__" in paused, "Operator-after-exit must interrupt"

            # Counter was written by the mesh (3 peer-hops) BEFORE the interrupt.
            mid = graph.get_state(cfg).values.get(_COUNTER_KEY)
            assert mid == 3, f"{_COUNTER_KEY}={mid!r} at interrupt, expected 3"

            completed = run(graph, resume={"approved": True}, config=cfg)
            assert "__interrupt__" not in completed, "resume must complete the run"
            # The counter persisted verbatim across the resume.
            assert graph.get_state(cfg).values.get(_COUNTER_KEY) == 3

    async def test_async_counter_survives_interrupt_and_resume(self, tmp_path):
        db = str(tmp_path / "km_resume_async.db")
        cfg = {"configurable": {"thread_id": "km-resume-async"}}
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            entry, billing = _register_counting_mesh("km_ra", threshold=3)
            gate, finalize = _register_gate_and_finalize("km_ra")
            graph = compile(
                Construct("km-resume-async", nodes=[entry, billing, gate, finalize]),
                checkpointer=saver,
                **build_test_compile_kwargs(),
            )
            paused = await neograph.arun(graph, input={}, config=cfg)
            assert "__interrupt__" in paused, "Operator-after-exit must interrupt"

            mid = (await graph.aget_state(cfg)).values.get(_COUNTER_KEY)
            assert mid == 3, f"{_COUNTER_KEY}={mid!r} at interrupt, expected 3"

            completed = await neograph.arun(graph, resume={"approved": True}, config=cfg)
            assert "__interrupt__" not in completed, "resume must complete the run"
            assert (await graph.aget_state(cfg)).values.get(_COUNTER_KEY) == 3
