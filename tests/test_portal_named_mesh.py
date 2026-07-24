"""Portal: named meshes -- >1 mesh per construct level (fefar, TDD-red pin).

v1 enforces D-SINGLE-MESH: exactly one Portal mesh (one connected component
under the peer relation) per construct level -- see
``tests/test_portal_validation.py::TestMeshStructureErrors::
test_two_meshes_at_one_level_rejected``. D-SINGLE-MESH exists because the
reserved ``handoff`` inputs key is ambiguous when two disjoint meshes share a
level: the IR normalizer (``_ir_normalize.py``) blindly stamps ONE
``handoff_channel`` (keyed off the FIRST Portal member at the level) onto
EVERY Portal member at that level, so a second disjoint mesh would silently
inherit the first mesh's channel key.

fefar closes this gap with ``Portal(name=...)``: ``name=None`` members form
the implicit default group (unchanged back-compat behavior); ``name="foo"``
members form a distinct named group. Grouping must be computed by exactly ONE
shared helper reused identically by the validator (``_validation_portal.py``),
the IR normalizer (``_ir_normalize.py``), and the compiler's contiguous-mesh
collector (``_wiring.py:_contiguous_portal_mesh``) -- see ``bd show fefar``
Core Invariant. Each named group gets its OWN contiguity check, OWN
connected-component check, OWN ``Command(goto)`` exit node, and OWN hop
counter, so two named meshes coexist and run at one level WITHOUT cross-talk.

TDD-red status: ``Portal`` has no ``name`` field yet, so a ``name=`` kwarg is
silently dropped by pydantic's default ``extra="ignore"`` behavior. Both
groups below therefore fall into ONE flattened members list at assembly time,
and the EXISTING D-SINGLE-MESH connected-component check raises "two disjoint
Portal meshes" for what fefar's design says should be two legally-named,
independently-running meshes. This test currently FAILS at
``Construct(...)`` assembly (a ``ConstructError`` neither the design nor this
test wants).

Design refs: bd show fefar; docs/design/dynamic-handoff-2026-07-13.md
Sec3.1-3.3, Sec5.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import (
    HANDOFF_END,
    Construct,
    ConstructError,
    Node,
    Portal,
    compile,
    run,
)
from neograph._state_keys import StateKeys
from tests.fakes import build_test_compile_kwargs, register_scripted


class RouteHop(BaseModel, frozen=True):
    """Uniform mesh payload with a plain-str route field + hop counter."""

    goto: str
    hops: int = 0


def _triage_fn(own: str, peer: str, threshold: int, visits: list[str]):
    def fn(input_data, config):
        visits.append(own)
        incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
        h = 0 if incoming is None else incoming.hops
        if h >= threshold:
            return RouteHop(goto=HANDOFF_END, hops=h)
        return RouteHop(goto=peer, hops=h + 1)

    return fn


def _billing_fn(own: str, peer: str, threshold: int, visits: list[str]):
    def fn(input_data, config):
        visits.append(own)
        incoming = input_data["handoff"]
        h = incoming.hops
        if h >= threshold:
            return RouteHop(goto=HANDOFF_END, hops=h)
        return RouteHop(goto=peer, hops=h + 1)

    return fn


class TestNamedMeshesCoexistAtOneLevel:
    """fefar: ``Portal(name=...)`` partitions one construct level into >1 mesh.

    Two DISJOINT named meshes must run independently -- own connected-
    component validation, own ``Command(goto)`` exit node, own hop counter --
    with NO cross-talk. The two meshes below use DIFFERENT ``max_hops``
    budgets specifically so "falls out for free" (the fefar review's
    load-bearing concern) is PROVEN rather than assumed: if grouping or
    hop-counter keying were wrong (one shared global counter, or mesh_b's
    entry silently inheriting mesh_a's ``handoff_channel``), the two distinct
    thresholds would collide and the assertions below would fail.
    """

    def test_two_named_meshes_run_independently_with_distinct_hop_counters(self):
        visits: list[str] = []
        threshold_a, threshold_b = 2, 4

        register_scripted("fefar_a_triage", _triage_fn("a_triage", "a_billing", threshold_a, visits))
        register_scripted("fefar_a_billing", _billing_fn("a_billing", "a_triage", threshold_a, visits))
        register_scripted("fefar_b_triage", _triage_fn("b_triage", "b_billing", threshold_b, visits))
        register_scripted("fefar_b_billing", _billing_fn("b_billing", "b_triage", threshold_b, visits))

        a_entry = (
            Node.scripted("a_triage", fn="fefar_a_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["a_billing"], max_hops=threshold_a, name="mesh_a")
        )
        a_billing = (
            Node.scripted("a_billing", fn="fefar_a_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["a_triage"], name="mesh_a")
        )
        b_entry = (
            Node.scripted("b_triage", fn="fefar_b_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["b_billing"], max_hops=threshold_b, name="mesh_b")
        )
        b_billing = (
            Node.scripted("b_billing", fn="fefar_b_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["b_triage"], name="mesh_b")
        )

        # Two named meshes BACK-TO-BACK at one construct level, no plain node
        # between them -- the exact "two adjacent named meshes with no gap"
        # scenario the compiler-side contiguous-run collector must bound by
        # mesh name, not merely by "any Portal member" (fefar risks section).
        mesh = Construct("two-named-meshes", nodes=[a_entry, a_billing, b_entry, b_billing])
        graph = compile(mesh, **build_test_compile_kwargs(checkpointer=MemorySaver()))
        cfg = {"configurable": {"thread_id": "fefar-two-mesh"}}
        run(graph, input={}, config=cfg)

        # Each mesh reached its OWN budget, not a shared/merged counter.
        counter_a = graph.get_state(cfg).values.get(StateKeys.handoff_hops("a_triage"))
        counter_b = graph.get_state(cfg).values.get(StateKeys.handoff_hops("b_triage"))
        assert counter_a == threshold_a, f"mesh_a counter={counter_a!r}, expected {threshold_a}"
        assert counter_b == threshold_b, f"mesh_b counter={counter_b!r}, expected {threshold_b}"
        assert counter_a != counter_b  # distinct budgets prove independent counters, not one shared key

        # Distinct exit nodes: BOTH entries' terminal payloads are on the
        # final state, each carrying its OWN mesh's result -- neither mesh's
        # Command(goto) exit clobbered or merged with the other's.
        final = graph.get_state(cfg).values
        assert final["a_triage"].goto == HANDOFF_END
        assert final["a_triage"].hops == threshold_a
        assert final["b_triage"].goto == HANDOFF_END
        assert final["b_triage"].hops == threshold_b

        # Both meshes' ping-pong ran independently to completion -- no
        # member from mesh_a interleaved into mesh_b's cycle or vice versa
        # (each visited at least once; the hop counters above already pin
        # exact independent counts per mesh).
        assert visits.count("a_triage") >= 1 and visits.count("a_billing") >= 1
        assert visits.count("b_triage") >= 1 and visits.count("b_billing") >= 1

    def test_unnamed_two_meshes_still_rejected(self):
        """Back-compat pin: ``name=None`` default-group behavior is UNCHANGED
        -- two disjoint UNNAMED meshes at one level still raise (the existing
        D-SINGLE-MESH rule governs the implicit default group exactly as
        before fefar)."""
        register_scripted("fefar_unnamed_f", lambda i, c: RouteHop(goto=HANDOFF_END))
        a1 = Node.scripted("u_a1", fn="fefar_unnamed_f", outputs=RouteHop) | Portal(to=["u_a2"])
        a2 = (
            Node.scripted("u_a2", fn="fefar_unnamed_f", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["u_a1"])
        )
        b1 = Node.scripted("u_b1", fn="fefar_unnamed_f", outputs=RouteHop) | Portal(to=["u_b2"])
        b2 = (
            Node.scripted("u_b2", fn="fefar_unnamed_f", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["u_b1"])
        )
        with pytest.raises(ConstructError):
            Construct("two-unnamed-meshes", nodes=[a1, a2, b1, b2])
