"""Portal modifier-legality tests (T1 half — neograph-rwion).

Pins the modifier-slot legality of ``Portal``:

- slot conflicts with Each/Oracle/Loop/Operator raise ``ConstructError`` on BOTH
  the pipe path (`_SLOT_RULES` excludes) AND the direct-`ModifierSet(...)` path
  (`model_post_init` arms — review M2 parity hazard);
- CRITICALLY both pipe orders — Portal FIRST and Portal SECOND — raise
  ``ConstructError`` (not KeyError); pins review MEDIUM-2, the reciprocal
  `_SLOT_RULES` excludes;
- duplicate Portal is rejected;
- mode discrimination raises ``ConfigurationError`` on neither/both, and
  `max_hops >= 1` is enforced.

Runtime routing / budget behavior lands in T2/T3 (separate homes). This file is
the modifier-legality half only.

Design ref: docs/design/dynamic-handoff-2026-07-13.md §2.1, §4.1 (modifier decl
row), §5.6.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import (
    HANDOFF_END,
    ConfigurationError,
    ConstructError,
    Each,
    Loop,
    Node,
    Operator,
    Oracle,
    Portal,
)
from neograph.modifiers import ModifierSet
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_scripted("f", lambda i, c: Handoff(goto="__end__"))
register_scripted("mrg", lambda variants, c: variants[0])


def _base() -> Node:
    return Node.scripted("member", fn="f", outputs=Handoff)


# ═══════════════════════════════════════════════════════════════════════════
# MODE DISCRIMINATION (§2.1 model_post_init)
# ═══════════════════════════════════════════════════════════════════════════


class TestPublicSurface:
    """HANDOFF_END sentinel is public and equals '__end__' (design §2.1)."""

    def test_handoff_end_sentinel_value(self):
        assert HANDOFF_END == "__end__"


class TestModeDiscrimination:
    """Portal discriminates peer mode vs dispatch mode in model_post_init."""

    def test_neither_peers_nor_decide_raises(self):
        """No peers and route != 'decide' — neither mode — raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            Portal()

    def test_both_peers_and_decide_raises(self):
        """peers set AND route=='decide' — both modes — raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            Portal(to=["a"], route="decide")

    def test_peer_mode_route_decide_forbidden(self):
        """Peer mode with route=='decide' is a contradiction — raises."""
        with pytest.raises(ConfigurationError):
            Portal(to=["a"], route="decide", spec_field="s", input_field="i", output=Handoff)

    def test_dispatch_mode_requires_spec_and_input_and_output(self):
        """route=='decide' without spec_field/input_field/output raises."""
        with pytest.raises(ConfigurationError):
            Portal(route="decide")

    def test_dispatch_mode_forbids_peer_knobs(self):
        """route=='decide' with peer-mode knobs (max_hops/on_exhaust) raises."""
        with pytest.raises(ConfigurationError):
            Portal(
                route="decide",
                spec_field="s",
                input_field="i",
                output=Handoff,
                on_exhaust="exit",
            )

    def test_max_hops_below_one_raises(self):
        """max_hops < 1 is rejected (mirrors Loop.max_iterations)."""
        with pytest.raises(ConfigurationError):
            Portal(to=["a"], max_hops=0)

    def test_peer_mode_defaults(self):
        """Peer mode with defaults: route='goto', max_hops=10, on_exhaust='error'."""
        km = Portal(to=["a"])
        assert km.route == "goto"
        assert km.max_hops == 10
        assert km.on_exhaust == "error"

    def test_dispatch_mode_constructs(self):
        """A well-formed dispatch-mode Portal constructs cleanly."""
        km = Portal(
            route="decide", spec_field="spec", input_field="dispatch_input", output=Handoff, max_depth=5
        )
        assert km.route == "decide"
        assert km.to is None


# ═══════════════════════════════════════════════════════════════════════════
# SLOT CONFLICTS — pipe path, BOTH orders (review MEDIUM-2)
# ═══════════════════════════════════════════════════════════════════════════


class TestPipeSlotConflicts:
    """Portal × Each/Oracle/Loop/Operator on the pipe path, BOTH orders.

    Both orders must raise ConstructError (not KeyError) — this pins the
    reciprocal `_SLOT_RULES` excludes (review MEDIUM-2).
    """

    def test_portal_then_each_raises(self):
        """node | Portal() | Each() — Portal FIRST — raises ConstructError."""
        with pytest.raises(ConstructError):
            _base() | Portal(to=["x"]) | Each(over="src.items", key="k")

    def test_each_then_portal_raises(self):
        """node | Each() | Portal() — Portal SECOND — raises ConstructError."""
        with pytest.raises(ConstructError):
            _base() | Each(over="src.items", key="k") | Portal(to=["x"])

    def test_portal_then_oracle_raises(self):
        with pytest.raises(ConstructError):
            _base() | Portal(to=["x"]) | Oracle(n=3, merge_fn="mrg")

    def test_oracle_then_portal_raises(self):
        with pytest.raises(ConstructError):
            _base() | Oracle(n=3, merge_fn="mrg") | Portal(to=["x"])

    def test_portal_then_loop_raises(self):
        with pytest.raises(ConstructError):
            _base() | Portal(to=["x"]) | Loop(when=lambda d: False, max_iterations=2)

    def test_loop_then_portal_raises(self):
        with pytest.raises(ConstructError):
            _base() | Loop(when=lambda d: False, max_iterations=2) | Portal(to=["x"])

    def test_portal_peer_then_operator_assembles(self):
        """Portal PEER mode + Operator is now legal (neograph-kdr1u, D4 lift):
        a human-approval gate spliced onto the dynamic path. Both pipe orders
        must assemble (review LOW finding: 3 reciprocal edit sites)."""
        register_scripted("cond", lambda d: True)
        node = _base() | Portal(to=["x"]) | Operator(when="cond")
        assert node.modifier_set.portal is not None
        assert node.modifier_set.operator is not None

    def test_operator_then_portal_peer_assembles(self):
        register_scripted("cond2", lambda d: True)
        node = _base() | Operator(when="cond2") | Portal(to=["x"])
        assert node.modifier_set.portal is not None
        assert node.modifier_set.operator is not None

    def test_portal_dispatch_then_operator_raises(self):
        """Portal DISPATCH mode (route='decide') + Operator stays illegal:
        dispatch has no peer to approve a handoff TO and no mesh-exit analog
        for a rejection to route to."""
        register_scripted("cond3", lambda d: True)
        with pytest.raises(ConstructError, match="dispatch mode"):
            _base() | Portal(
                route="decide", spec_field="s", input_field="i", output=Handoff, max_depth=5
            ) | Operator(when="cond3")

    def test_operator_then_portal_dispatch_raises(self):
        register_scripted("cond4", lambda d: True)
        with pytest.raises(ConstructError, match="dispatch mode"):
            _base() | Operator(when="cond4") | Portal(
                route="decide", spec_field="s", input_field="i", output=Handoff, max_depth=5
            )


class TestDuplicatePortal:
    """A second Portal on the same node is rejected (occupied slot)."""

    def test_duplicate_portal_raises(self):
        with pytest.raises(ConstructError):
            _base() | Portal(to=["x"]) | Portal(to=["y"])


# ═══════════════════════════════════════════════════════════════════════════
# SLOT CONFLICTS — direct-construct path (review M2 parity hazard)
# ═══════════════════════════════════════════════════════════════════════════


class TestDirectModifierSetConflicts:
    """Direct ModifierSet(portal=..., other=...) must ALSO reject.

    The pipe path reads `_SLOT_RULES`; the direct-construct path uses
    hard-coded pairwise checks in `model_post_init`. Without explicit portal
    arms the direct path would silently pass while the pipe rejects — the M2
    parity hazard. Both must reject with the "Cannot combine ..." message.

    Note the exception SHAPE differs by path (established convention, see
    ``test_modifier_edge_cases.test_each_loop_rejected_at_construction``): the
    pipe path raises ``ConstructError`` directly from ``with_modifier``, while
    the direct ``ModifierSet(...)`` construction raises it from ``model_post_init``
    where Pydantic wraps it into a ``ValidationError`` (a ``ValueError``
    subclass). We assert on the message so both wrapped and unwrapped forms pass.
    """

    def test_portal_and_each_direct_raises(self):
        with pytest.raises(Exception, match="Cannot combine Portal and Each"):
            ModifierSet(portal=Portal(to=["x"]), each=Each(over="s.i", key="k"))

    def test_portal_and_oracle_direct_raises(self):
        with pytest.raises(Exception, match="Cannot combine Portal and Oracle"):
            ModifierSet(portal=Portal(to=["x"]), oracle=Oracle(n=3, merge_fn="mrg"))

    def test_portal_and_loop_direct_raises(self):
        with pytest.raises(Exception, match="Cannot combine Portal and Loop"):
            ModifierSet(portal=Portal(to=["x"]), loop=Loop(when=lambda d: False, max_iterations=2))

    def test_portal_peer_and_operator_direct_assembles(self):
        """Direct-ModifierSet-construction path parity with the pipe path
        (review M2): Portal PEER + Operator is legal here too (neograph-kdr1u)."""
        register_scripted("cond5", lambda d: True)
        ms = ModifierSet(portal=Portal(to=["x"]), operator=Operator(when="cond5"))
        assert ms.portal is not None
        assert ms.operator is not None

    def test_portal_dispatch_and_operator_direct_raises(self):
        register_scripted("cond6", lambda d: True)
        with pytest.raises(Exception, match="dispatch mode"):
            ModifierSet(
                portal=Portal(route="decide", spec_field="s", input_field="i", output=Handoff, max_depth=5),
                operator=Operator(when="cond6"),
            )


# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME ROUTING (T2 — neograph-on6jt): Command(goto) mesh lowering
# ═══════════════════════════════════════════════════════════════════════════
#
# These are integration tests through the REAL compile() + run() surface: a
# scripted mesh routes hop-to-hop via Command(goto), reads each hop's payload
# off the shared mesh channel via the reserved 'handoff' inputs key, and closes
# LangGraph's silent-drop hole with an ExecutionError on an out-of-set target.
# No mocks — scripted members are pure functions registered via register_scripted.
# Budget (max_hops error/exit) + checkpoint semantics land in T3; @node sugar +
# full three-surface parity land in T4.

from neograph import Construct, ExecutionError, compile, run  # noqa: E402
from tests.fakes import build_test_compile_kwargs  # noqa: E402


class RouteHop(BaseModel, frozen=True):
    """Uniform mesh payload with a plain-str route field (design §3.2)."""

    goto: str
    hops: int = 0


class TestMeshRoutesEndToEndWithGenuineCycle:
    """A Portal mesh routes hop-to-hop and completes a GENUINE cycle.

    Flow: START -> triage (channel empty -> route to billing)
                -> billing (reads triage's payload -> route back to triage)
                -> triage (channel now populated -> route to HANDOFF_END/exit)
                -> exit -> END.
    triage is visited TWICE (the cycle); billing once. The visit order is
    recorded by the scripted bodies, proving the cycle actually executed rather
    than a lucky terminal read. Every member consumes the shared mesh channel
    via the reserved 'handoff' inputs key — the D10 read-side threading.
    """

    def _build_mesh(self, visits: list[str]) -> Construct:
        def triage_fn(input_data, config):
            visits.append("triage")
            incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
            # First activation: channel empty -> hand off to billing.
            # Re-entry (billing routed back): channel populated -> leave the mesh.
            if incoming is None:
                return RouteHop(goto="billing", hops=1)
            return RouteHop(goto=HANDOFF_END, hops=incoming.hops + 1)

        def billing_fn(input_data, config):
            visits.append("billing")
            incoming = input_data["handoff"]
            return RouteHop(goto="triage", hops=incoming.hops + 1)

        register_scripted("km_triage", triage_fn)
        register_scripted("km_billing", billing_fn)

        entry = (
            Node.scripted("triage", fn="km_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["billing"], max_hops=6)
        )
        billing = (
            Node.scripted("billing", fn="km_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["triage"])
        )
        return Construct("swarm", nodes=[entry, billing])

    def test_mesh_completes_cycle_and_exits(self):
        visits: list[str] = []
        mesh = self._build_mesh(visits)
        graph = compile(mesh, **build_test_compile_kwargs())
        run(graph, input={})
        # triage visited twice (the genuine cycle), billing once, in order.
        assert visits == ["triage", "billing", "triage"]


class TestOutOfSetTargetRaises:
    """A route value outside peers ∪ {HANDOFF_END} raises ExecutionError.

    This is the INVARIANT: the wrapper checks the target BEFORE emitting the
    goto, so an out-of-set target fails LOUD instead of LangGraph silently
    dropping the update (langgraph _algo.py:312, the research's #1 constraint).
    """

    def test_unknown_target_raises_execution_error(self):
        register_scripted("km_ghost", lambda i, c: RouteHop(goto="ghost"))
        register_scripted("km_sink", lambda i, c: RouteHop(goto=HANDOFF_END))
        entry = (
            Node.scripted("triage", fn="km_ghost", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["billing"], max_hops=6)
        )
        billing = (
            Node.scripted("billing", fn="km_sink", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["triage"])
        )
        mesh = Construct("swarm-bad", nodes=[entry, billing])
        graph = compile(mesh, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError, match="ghost"):
            run(graph, input={})


class TestMeshLowersWithoutDoubleAddingNodes:
    """The mesh-aware walk collapses the contiguous mesh to ONE dispatch.

    Review M1: without the walk skip, every non-entry member is double-added
    (once by the mesh helper, once by the outer walk). This pins that each
    member node name appears EXACTLY ONCE in the compiled LangGraph.
    """

    def test_each_member_appears_once_in_compiled_graph(self):
        register_scripted("km_once", lambda i, c: RouteHop(goto=HANDOFF_END))
        entry = (
            Node.scripted("triage", fn="km_once", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["billing"], max_hops=6)
        )
        billing = (
            Node.scripted("billing", fn="km_once", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["triage"])
        )
        mesh = Construct("swarm-walk", nodes=[entry, billing])
        graph = compile(mesh, **build_test_compile_kwargs())
        node_names = list(graph.graph.get_graph().nodes)
        assert node_names.count("triage") == 1
        assert node_names.count("billing") == 1


class TestHandoffEndLeavesViaExitNode:
    """A member routing to HANDOFF_END leaves the mesh via the pass-through
    exit node, so a downstream node AFTER the mesh still runs (HANDOFF_END is
    byte-identical to LangGraph END, so it must map to the exit node — not
    terminate the whole graph)."""

    def test_downstream_runs_after_handoff_end(self):
        after: list[str] = []

        def entry_fn(input_data, config):
            return RouteHop(goto=HANDOFF_END)

        def after_fn(input_data, config):
            after.append("after")
            return RouteHop(goto="done")

        register_scripted("km_exit_entry", entry_fn)
        register_scripted("km_after", after_fn)

        entry = (
            Node.scripted("triage", fn="km_exit_entry", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["billing"], max_hops=6)
        )
        billing = (
            Node.scripted("billing", fn="km_after", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["triage"])
        )
        # A plain scripted node AFTER the mesh — reached only if the mesh exits
        # via its pass-through exit node rather than terminating the graph.
        downstream = Node.scripted("report", fn="km_after", inputs=RouteHop, outputs=RouteHop)
        mesh = Construct("swarm-exit", nodes=[entry, billing, downstream])
        graph = compile(mesh, **build_test_compile_kwargs())
        run(graph, input={})
        assert after == ["after"]  # 'report' ran after the mesh exited


# ═══════════════════════════════════════════════════════════════════════════
# HOP BUDGET (T3 — neograph-0umvg): max_hops + on_exhaust error/exit
# ═══════════════════════════════════════════════════════════════════════════
#
# BINDING SEMANTICS (design §3.4, decision-log D11/D12/D13):
#   - A "hop" = a member routing to a PEER (mesh continuation). The entry
#     member's own first execution is NOT a hop; the first peer route out of
#     ANY member is hop 1.
#   - Shared entry-keyed counter neo_handoff_hops_<entry_field>. The budget is
#     CHECKED BEFORE emitting a peer goto: current >= max_hops -> exhaust (Loop
#     parity). max_hops=N ⇒ exactly N peer-hops allowed, then exhaust.
#   - A member routing to HANDOFF_END exits cleanly, is NEVER budget-gated and
#     does NOT increment the counter.
#   - on_exhaust="error" (default) -> ExecutionError naming the ENTRY node.
#   - on_exhaust="exit" -> routes to the mesh exit node with the last payload on
#     the bus (NO raise); a downstream node after the mesh still runs.
#   - Budget knobs are ENTRY-only; a non-entry member's Portal never carries
#     them (defaults max_hops=10/on_exhaust='error' are irrelevant to the mesh
#     budget, which uses the ENTRY's values).
#
# The T3 seam (factory.py make_portal_fn, ~lines 156-160) is currently a
# comment — the counter RMW + max_hops check do not exist yet, so these tests
# MUST FAIL now.

import re  # noqa: E402

from neograph._state_keys import StateKeys  # noqa: E402


class TestBudgetExhaustRaisesError:
    """A ping-pong mesh that NEVER routes to HANDOFF_END exhausts the ENTRY's
    max_hops budget and raises ExecutionError naming the entry (Loop parity).

    Boundary pin (D11/D12): with 2 ping-pong members and entry max_hops=3, the
    counter is checked before each peer goto. hops 1,2,3 emit
    (triage->billing->triage->billing), then the 4th member execution's peer
    route sees current(3) >= 3 and exhausts. So EXACTLY 4 member executions
    occur, and the raise happens on the 4th's route (its body already ran, so it
    IS recorded in ``visits``).
    """

    def _build_pingpong(self, visits, *, on_exhaust="error", max_hops=3):
        def triage_fn(input_data, config):
            visits.append("triage")
            incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
            nxt = 0 if incoming is None else incoming.hops
            return RouteHop(goto="billing", hops=nxt + 1)  # forever -> billing

        def billing_fn(input_data, config):
            visits.append("billing")
            incoming = input_data["handoff"]
            return RouteHop(goto="triage", hops=incoming.hops + 1)  # forever -> triage

        register_scripted("km_err_triage", triage_fn)
        register_scripted("km_err_billing", billing_fn)

        entry = (
            Node.scripted("triage", fn="km_err_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["billing"], max_hops=max_hops, on_exhaust=on_exhaust)
        )
        billing = (
            Node.scripted("billing", fn="km_err_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["triage"])
        )
        return Construct("swarm-budget-err", nodes=[entry, billing])

    def test_budget_exhaust_raises_execution_error_naming_entry(self):
        visits: list[str] = []
        mesh = self._build_pingpong(visits, on_exhaust="error", max_hops=3)
        graph = compile(mesh, **build_test_compile_kwargs())
        with pytest.raises(ExecutionError) as exc:
            run(graph, input={})
        msg = str(exc.value)
        assert re.search(r"max_hops|handoff exceeded", msg), msg
        assert "triage" in msg, f"error must name the ENTRY node: {msg}"
        # Exactly 4 member executions before the raise (the boundary).
        assert visits == ["triage", "billing", "triage", "billing"], visits


class TestBudgetExhaustExits:
    """on_exhaust='exit' on the ENTRY: the same never-terminating ping-pong mesh
    leaves via the pass-through exit node (NO raise) with the last payload on the
    shared channel, and a plain scripted node placed AFTER the mesh still runs.
    """

    def test_budget_exhaust_exit_routes_downstream_with_last_payload(self):
        visits: list[str] = []
        after: list[str] = []

        def triage_fn(input_data, config):
            visits.append("triage")
            incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
            nxt = 0 if incoming is None else incoming.hops
            return RouteHop(goto="billing", hops=nxt + 1)

        def billing_fn(input_data, config):
            visits.append("billing")
            incoming = input_data["handoff"]
            return RouteHop(goto="triage", hops=incoming.hops + 1)

        def report_fn(input_data, config):
            after.append("report")
            return RouteHop(goto="done")

        register_scripted("km_exit_triage", triage_fn)
        register_scripted("km_exit_billing", billing_fn)
        register_scripted("km_exit_report", report_fn)

        entry = (
            Node.scripted("triage", fn="km_exit_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["billing"], max_hops=3, on_exhaust="exit")
        )
        billing = (
            Node.scripted("billing", fn="km_exit_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["triage"])
        )
        # Plain scripted node AFTER the mesh — reached only if 'exit' routes
        # through the pass-through exit node rather than raising.
        downstream = Node.scripted("report", fn="km_exit_report", inputs=RouteHop, outputs=RouteHop)
        mesh = Construct("swarm-budget-exit", nodes=[entry, billing, downstream])
        graph = compile(mesh, **build_test_compile_kwargs())

        result = run(graph, input={})  # NO exception on exhaust
        assert after == ["report"], "downstream node must run after the 'exit' exhaust"
        # The last payload survives for downstream consumption (design §3.4). The
        # neo_-prefixed shared channel is stripped from the final result by the
        # _strip_internals invariant (state.py:410-419), so the observable locus
        # is the last member's output field — 'billing' ran last before exhaust.
        last = result.get("billing")
        assert isinstance(last, RouteHop), result
        assert last.hops == 4, last  # 3 peer-hops made, exhaust wrote hops=4 payload


class TestBudgetBoundaryExactlyMaxHops:
    """Off-by-one pin: a mesh that makes EXACTLY max_hops peer-hops then routes
    to HANDOFF_END exits cleanly (no exhaust), and the shared counter equals
    max_hops. This pins 'max_hops=N ⇒ N peer-hops allowed'.

    Members ping-pong while the carried payload hop count < 3, then route to
    HANDOFF_END: triage->billing (hop1), billing->triage (hop2),
    triage->billing (hop3), billing->HANDOFF_END (exit, not a hop). 3 peer-hops
    == max_hops, so the counter must read 3 and NO exhaust fires.
    """

    def test_exactly_max_hops_then_handoff_end_exits_clean(self):
        from langgraph.checkpoint.memory import MemorySaver

        visits: list[str] = []
        threshold = 3

        def triage_fn(input_data, config):
            visits.append("triage")
            incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
            h = 0 if incoming is None else incoming.hops
            if h >= threshold:
                return RouteHop(goto=HANDOFF_END, hops=h)
            return RouteHop(goto="billing", hops=h + 1)

        def billing_fn(input_data, config):
            visits.append("billing")
            incoming = input_data["handoff"]
            h = incoming.hops
            if h >= threshold:
                return RouteHop(goto=HANDOFF_END, hops=h)
            return RouteHop(goto="triage", hops=h + 1)

        register_scripted("km_bnd_triage", triage_fn)
        register_scripted("km_bnd_billing", billing_fn)

        entry = (
            Node.scripted("triage", fn="km_bnd_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["billing"], max_hops=threshold)
        )
        billing = (
            Node.scripted("billing", fn="km_bnd_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["triage"])
        )
        mesh = Construct("swarm-budget-boundary", nodes=[entry, billing])
        graph = compile(mesh, **build_test_compile_kwargs(checkpointer=MemorySaver()))
        cfg = {"configurable": {"thread_id": "km-boundary"}}
        run(graph, input={}, config=cfg)  # clean exit, no exhaust

        # 3 peer-hops (+ the entry's non-hop first exec) => 4 member executions.
        assert visits == ["triage", "billing", "triage", "billing"], visits
        # The shared entry-keyed counter counts EXACTLY the peer-hops == max_hops.
        counter_key = StateKeys.handoff_hops("triage")
        counter = graph.get_state(cfg).values.get(counter_key)
        assert counter == threshold, f"{counter_key}={counter!r}, expected {threshold}"


class TestRecursionFloorForLargeMesh:
    """D13 recursion floor: a mesh-ONLY construct (no agent/act nodes) whose
    entry max_hops EXCEEDS LangGraph's default recursion ceiling (25) must run
    to on_exhaust WITHOUT a GraphRecursionError — the runner raises the
    recursion floor by the mesh's max_hops.

    Without the floor, a 30-hop ping-pong hits LangGraph's 25-superstep ceiling
    and raises GraphRecursionError before the budget fork can fire.
    """

    def test_large_max_hops_reaches_budget_not_recursion_ceiling(self):
        def triage_fn(input_data, config):
            incoming = input_data.get("handoff") if isinstance(input_data, dict) else None
            nxt = 0 if incoming is None else incoming.hops
            return RouteHop(goto="billing", hops=nxt + 1)

        def billing_fn(input_data, config):
            incoming = input_data["handoff"]
            return RouteHop(goto="triage", hops=incoming.hops + 1)

        register_scripted("km_floor_triage", triage_fn)
        register_scripted("km_floor_billing", billing_fn)

        entry = (
            Node.scripted("triage", fn="km_floor_triage", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["billing"], max_hops=30, on_exhaust="error")
        )
        billing = (
            Node.scripted("billing", fn="km_floor_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Portal(to=["triage"])
        )
        mesh = Construct("swarm-floor", nodes=[entry, billing])
        graph = compile(mesh, **build_test_compile_kwargs())
        # Reaches the budget fork (ExecutionError), NOT GraphRecursionError.
        with pytest.raises(ExecutionError) as exc:
            run(graph, input={})
        assert re.search(r"max_hops|handoff exceeded", str(exc.value)), str(exc.value)
