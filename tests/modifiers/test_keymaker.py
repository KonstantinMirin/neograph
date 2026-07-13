"""Keymaker modifier-legality tests (T1 half — neograph-rwion).

Pins the modifier-slot legality of ``Keymaker``:

- slot conflicts with Each/Oracle/Loop/Operator raise ``ConstructError`` on BOTH
  the pipe path (`_SLOT_RULES` excludes) AND the direct-`ModifierSet(...)` path
  (`model_post_init` arms — review M2 parity hazard);
- CRITICALLY both pipe orders — Keymaker FIRST and Keymaker SECOND — raise
  ``ConstructError`` (not KeyError); pins review MEDIUM-2, the reciprocal
  `_SLOT_RULES` excludes;
- duplicate Keymaker is rejected;
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
    Keymaker,
    Loop,
    Node,
    Operator,
    Oracle,
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
    """Keymaker discriminates peer mode vs dispatch mode in model_post_init."""

    def test_neither_peers_nor_decide_raises(self):
        """No peers and route != 'decide' — neither mode — raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            Keymaker()

    def test_both_peers_and_decide_raises(self):
        """peers set AND route=='decide' — both modes — raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            Keymaker(peers=["a"], route="decide")

    def test_peer_mode_route_decide_forbidden(self):
        """Peer mode with route=='decide' is a contradiction — raises."""
        with pytest.raises(ConfigurationError):
            Keymaker(peers=["a"], route="decide", spec_field="s", input_field="i", output=Handoff)

    def test_dispatch_mode_requires_spec_and_input_and_output(self):
        """route=='decide' without spec_field/input_field/output raises."""
        with pytest.raises(ConfigurationError):
            Keymaker(route="decide")

    def test_dispatch_mode_forbids_peer_knobs(self):
        """route=='decide' with peer-mode knobs (max_hops/on_exhaust) raises."""
        with pytest.raises(ConfigurationError):
            Keymaker(
                route="decide",
                spec_field="s",
                input_field="i",
                output=Handoff,
                on_exhaust="exit",
            )

    def test_max_hops_below_one_raises(self):
        """max_hops < 1 is rejected (mirrors Loop.max_iterations)."""
        with pytest.raises(ConfigurationError):
            Keymaker(peers=["a"], max_hops=0)

    def test_peer_mode_defaults(self):
        """Peer mode with defaults: route='goto', max_hops=10, on_exhaust='error'."""
        km = Keymaker(peers=["a"])
        assert km.route == "goto"
        assert km.max_hops == 10
        assert km.on_exhaust == "error"

    def test_dispatch_mode_constructs(self):
        """A well-formed dispatch-mode Keymaker constructs cleanly."""
        km = Keymaker(route="decide", spec_field="spec", input_field="dispatch_input", output=Handoff)
        assert km.route == "decide"
        assert km.peers is None


# ═══════════════════════════════════════════════════════════════════════════
# SLOT CONFLICTS — pipe path, BOTH orders (review MEDIUM-2)
# ═══════════════════════════════════════════════════════════════════════════


class TestPipeSlotConflicts:
    """Keymaker × Each/Oracle/Loop/Operator on the pipe path, BOTH orders.

    Both orders must raise ConstructError (not KeyError) — this pins the
    reciprocal `_SLOT_RULES` excludes (review MEDIUM-2).
    """

    def test_keymaker_then_each_raises(self):
        """node | Keymaker() | Each() — Keymaker FIRST — raises ConstructError."""
        with pytest.raises(ConstructError):
            _base() | Keymaker(peers=["x"]) | Each(over="src.items", key="k")

    def test_each_then_keymaker_raises(self):
        """node | Each() | Keymaker() — Keymaker SECOND — raises ConstructError."""
        with pytest.raises(ConstructError):
            _base() | Each(over="src.items", key="k") | Keymaker(peers=["x"])

    def test_keymaker_then_oracle_raises(self):
        with pytest.raises(ConstructError):
            _base() | Keymaker(peers=["x"]) | Oracle(n=3, merge_fn="mrg")

    def test_oracle_then_keymaker_raises(self):
        with pytest.raises(ConstructError):
            _base() | Oracle(n=3, merge_fn="mrg") | Keymaker(peers=["x"])

    def test_keymaker_then_loop_raises(self):
        with pytest.raises(ConstructError):
            _base() | Keymaker(peers=["x"]) | Loop(when=lambda d: False, max_iterations=2)

    def test_loop_then_keymaker_raises(self):
        with pytest.raises(ConstructError):
            _base() | Loop(when=lambda d: False, max_iterations=2) | Keymaker(peers=["x"])

    def test_keymaker_then_operator_raises(self):
        """Keymaker + Operator is ILLEGAL in v1 (D-NO-OPERATOR-COMBO)."""
        register_scripted("cond", lambda d: True)
        with pytest.raises(ConstructError):
            _base() | Keymaker(peers=["x"]) | Operator(when="cond")

    def test_operator_then_keymaker_raises(self):
        register_scripted("cond2", lambda d: True)
        with pytest.raises(ConstructError):
            _base() | Operator(when="cond2") | Keymaker(peers=["x"])


class TestDuplicateKeymaker:
    """A second Keymaker on the same node is rejected (occupied slot)."""

    def test_duplicate_keymaker_raises(self):
        with pytest.raises(ConstructError):
            _base() | Keymaker(peers=["x"]) | Keymaker(peers=["y"])


# ═══════════════════════════════════════════════════════════════════════════
# SLOT CONFLICTS — direct-construct path (review M2 parity hazard)
# ═══════════════════════════════════════════════════════════════════════════


class TestDirectModifierSetConflicts:
    """Direct ModifierSet(keymaker=..., other=...) must ALSO reject.

    The pipe path reads `_SLOT_RULES`; the direct-construct path uses
    hard-coded pairwise checks in `model_post_init`. Without explicit keymaker
    arms the direct path would silently pass while the pipe rejects — the M2
    parity hazard. Both must reject with the "Cannot combine ..." message.

    Note the exception SHAPE differs by path (established convention, see
    ``test_modifier_edge_cases.test_each_loop_rejected_at_construction``): the
    pipe path raises ``ConstructError`` directly from ``with_modifier``, while
    the direct ``ModifierSet(...)`` construction raises it from ``model_post_init``
    where Pydantic wraps it into a ``ValidationError`` (a ``ValueError``
    subclass). We assert on the message so both wrapped and unwrapped forms pass.
    """

    def test_keymaker_and_each_direct_raises(self):
        with pytest.raises(Exception, match="Cannot combine Keymaker and Each"):
            ModifierSet(keymaker=Keymaker(peers=["x"]), each=Each(over="s.i", key="k"))

    def test_keymaker_and_oracle_direct_raises(self):
        with pytest.raises(Exception, match="Cannot combine Keymaker and Oracle"):
            ModifierSet(keymaker=Keymaker(peers=["x"]), oracle=Oracle(n=3, merge_fn="mrg"))

    def test_keymaker_and_loop_direct_raises(self):
        with pytest.raises(Exception, match="Cannot combine Keymaker and Loop"):
            ModifierSet(keymaker=Keymaker(peers=["x"]), loop=Loop(when=lambda d: False, max_iterations=2))

    def test_keymaker_and_operator_direct_raises(self):
        register_scripted("cond3", lambda d: True)
        with pytest.raises(Exception, match="Cannot combine Keymaker and Operator"):
            ModifierSet(keymaker=Keymaker(peers=["x"]), operator=Operator(when="cond3"))


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
    """A Keymaker mesh routes hop-to-hop and completes a GENUINE cycle.

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
            | Keymaker(peers=["billing"], max_hops=6)
        )
        billing = (
            Node.scripted("billing", fn="km_billing", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Keymaker(peers=["triage"])
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
            | Keymaker(peers=["billing"], max_hops=6)
        )
        billing = (
            Node.scripted("billing", fn="km_sink", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Keymaker(peers=["triage"])
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
            | Keymaker(peers=["billing"], max_hops=6)
        )
        billing = (
            Node.scripted("billing", fn="km_once", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Keymaker(peers=["triage"])
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
            | Keymaker(peers=["billing"], max_hops=6)
        )
        billing = (
            Node.scripted("billing", fn="km_after", inputs={"handoff": RouteHop}, outputs=RouteHop)
            | Keymaker(peers=["triage"])
        )
        # A plain scripted node AFTER the mesh — reached only if the mesh exits
        # via its pass-through exit node rather than terminating the graph.
        downstream = Node.scripted("report", fn="km_after", inputs=RouteHop, outputs=RouteHop)
        mesh = Construct("swarm-exit", nodes=[entry, billing, downstream])
        graph = compile(mesh, **build_test_compile_kwargs())
        run(graph, input={})
        assert after == ["after"]  # 'report' ran after the mesh exited
