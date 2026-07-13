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
