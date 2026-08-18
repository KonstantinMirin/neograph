"""Branch-arm modifier validation — Loop/Each/Oracle/Operator (neograph-ftnxl.19).

Sibling of ``TestPortalInsideBranchArmRejected`` in ``tests/test_portal_validation.py``
(neograph-ftnxl.12). Same root cause, four more modifiers.

``_add_arm_nodes`` / ``_wire_arm_edges`` (``_wiring_branch.py``) never dispatch on
``primary_shape(item)`` — every branch-arm item goes through the plain
``make_node_fn`` (Node) or ``_compile()`` + ``make_subgraph_fn`` (Construct) path
plus a static ``add_edge``, regardless of the modifier it carries. Compare
``compiler.py``'s main loop, whose ``match COMBO_DECOMPOSITION[combo].primary``
routes ORACLE/EACH/LOOP/BARE/PORTAL to five different graph-builders and then
appends ``_add_operator_check`` for the orthogonal Operator wrapper.

Empirically confirmed inert (compiled topology, arm vs. top level):

===========  ==========================================  ==================
Modifier     Top-level wiring                            Inside a branch arm
===========  ==========================================  ==================
Loop         ``looped -> looped`` back-edge +             plain node, NO
             ``__loop_exit_looped`` router                back-edge, no router
Each         ``__each_empty_*`` + ``assemble_*`` barrier  plain node only
Oracle       ``merge_*`` judge node                       plain node only
Operator     ``<node>__operator`` interrupt check node    plain node only
===========  ==========================================  ==================

These are integration-level tests through the REAL ``Construct(...)`` / ``Node`` /
modifier surface — assembly validation is pure in-process, no mocks.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, ConstructError, Node, Portal
from neograph._each import Each
from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec
from neograph.modifiers import Loop, ModifierCombo, Operator, Oracle, modifier_names_for_combo
from tests.fakes import register_condition, register_scripted


class Seed(BaseModel, frozen=True):
    n: int = 1


class Item(BaseModel, frozen=True):
    v: str = "x"


class Bag(BaseModel, frozen=True):
    items: list[str] = ("a", "b")  # type: ignore[assignment]


def _construct_with_arm_item(name: str, arm_item, *, trigger_out=Seed) -> Construct:
    """Assemble a Construct whose true-arm holds exactly ``arm_item``.

    Builds the ``_BranchNode`` IR directly (same shape ``ForwardConstruct``'s
    tracer emits for ``if ...:``) so the test does not depend on the DX layer.
    """
    register_scripted(f"trig_{name}", lambda _in, _cfg: trigger_out())
    trigger = Node.scripted(f"trigger_{name}", fn=f"trig_{name}", outputs=trigger_out)
    cond = _ConditionSpec(
        source_node=trigger,
        attr_chain=[],
        op_fn=lambda value, _t: bool(value),
        op_str="truthy",
        threshold=None,
    )
    meta = _BranchMeta(condition_spec=cond, true_arm_nodes=[arm_item], false_arm_nodes=[])
    return Construct(f"c_{name}", nodes=[trigger, _BranchNode(meta, 0)])


class TestNonPortalModifiersInsideBranchArmRejected:
    """neograph-ftnxl.19 — Loop/Each/Oracle/Operator on a branch-arm item are
    INERT at compile time (no error, no wiring), exactly like the Portal case
    ftnxl.12 already closed.

    A silent seam falsifies the "safer than LangGraph" claim (AGENTS.md north
    star: unrepresentable > fail-loud > silent), so the shape must be rejected
    at assembly rather than accepted and silently un-wired.
    """

    def test_loop_modified_node_inside_branch_arm_raises_construct_error(self):
        """``Node | Loop(...)`` in an arm compiles with ZERO self-loop back-edge
        and no ``__loop_exit_*`` router — the Loop never runs more than once."""
        register_scripted("arm_loop_body", lambda _in, _cfg: Seed())
        looped = Node.scripted("looped", fn="arm_loop_body", inputs=Seed, outputs=Seed) | Loop(
            when=lambda d: d is None or d.n < 3, max_iterations=3
        )

        with pytest.raises(ConstructError) as exc:
            _construct_with_arm_item("loop", looped)
        msg = str(exc.value).lower()
        assert "loop" in msg and "arm" in msg

    def test_each_modified_node_inside_branch_arm_raises_construct_error(self):
        """``Node | Each(...)`` in an arm compiles with no fan-out router and no
        ``assemble_*`` barrier — the node runs once, not once per item."""
        register_scripted("arm_each_body", lambda _in, _cfg: Item())
        eached = Node.scripted("eached", fn="arm_each_body", inputs={"item": str}, outputs=Item) | Each(
            over="trigger_each.items", key="v"
        )

        with pytest.raises(ConstructError) as exc:
            _construct_with_arm_item("each", eached, trigger_out=Bag)
        msg = str(exc.value).lower()
        assert "each" in msg and "arm" in msg

    def test_oracle_modified_node_inside_branch_arm_raises_construct_error(self):
        """``Node | Oracle(...)`` in an arm compiles with no variant fan-out and
        no merge node — a single sample masquerades as an N-way ensemble."""
        register_scripted("arm_oracle_body", lambda _in, _cfg: Item())
        register_scripted("arm_oracle_merge", lambda _in, _cfg: Item())
        oracled = Node.scripted("oracled", fn="arm_oracle_body", inputs=Seed, outputs=Item) | Oracle(
            n=3, merge_fn="arm_oracle_merge"
        )

        with pytest.raises(ConstructError) as exc:
            _construct_with_arm_item("oracle", oracled)
        msg = str(exc.value).lower()
        assert "oracle" in msg and "arm" in msg

    def test_operator_modified_node_inside_branch_arm_raises_construct_error(self):
        """``Node | Operator(...)`` in an arm compiles with no ``<node>__operator``
        check node — the human-in-the-loop gate never fires. Worse than merely
        inert: the compile-time "Operator requires a checkpointer" guard DOES
        observe the arm Operator, so the author gets a signal the modifier was
        seen while none of its wiring is ever emitted."""
        register_scripted("arm_op_body", lambda _in, _cfg: Item())
        register_condition("arm_always", lambda _d: True)
        gated = Node.scripted("opnode", fn="arm_op_body", inputs=Seed, outputs=Item) | Operator(when="arm_always")

        with pytest.raises(ConstructError) as exc:
            _construct_with_arm_item("op", gated)
        msg = str(exc.value).lower()
        assert "operator" in msg and "arm" in msg

    def test_loop_carrying_construct_inside_branch_arm_raises_construct_error(self):
        """The realistic DX shape: ``self.loop(body=[...])`` inside an ``if:``
        emits a ``Construct | Loop`` into the arm. ``_add_arm_nodes`` recompiles
        an arm Construct via plain ``_compile()`` + ``make_subgraph_fn``, never
        through ``_add_subgraph_loop`` — same inertness as the Node form."""
        register_scripted("arm_sub_body", lambda _in, _cfg: Seed())
        inner = Node.scripted("inner", fn="arm_sub_body", inputs=Seed, outputs=Seed)
        arm_construct = Construct("arm-loop-body", nodes=[inner], input=Seed, output=Seed) | Loop(
            when=lambda d: d is None or d.n < 3, max_iterations=3
        )

        with pytest.raises(ConstructError) as exc:
            _construct_with_arm_item("subloop", arm_construct)
        msg = str(exc.value).lower()
        assert "loop" in msg and "arm" in msg

    def test_unmodified_branch_arm_items_still_assemble(self):
        """Should-pass companion: the guard must reject only NON-BARE arm items.
        An ordinary unmodified Node (and an unmodified sub-Construct) in an arm
        stays legal — the arm-descent path handles both today."""
        register_scripted("arm_plain_body", lambda _in, _cfg: Item())
        plain = Node.scripted("plain", fn="arm_plain_body", inputs=Seed, outputs=Item)

        # Must not raise.
        _construct_with_arm_item("plain", plain)

        register_scripted("arm_plain_sub", lambda _in, _cfg: Item())
        sub_inner = Node.scripted("sub-inner", fn="arm_plain_sub", inputs=Seed, outputs=Item)
        plain_sub = Construct("arm-plain-sub", nodes=[sub_inner], input=Seed, output=Item)

        # Must not raise.
        _construct_with_arm_item("plainsub", plain_sub)


# ═══════════════════════════════════════════════════════════════════════════
# Totality — the guard must cover EVERY ModifierCombo, not just today's five
# ═══════════════════════════════════════════════════════════════════════════


def _item_carrying(combo: ModifierCombo, name: str):
    """Build a Node carrying exactly ``combo``'s modifier set.

    Derived from ``modifier_names_for_combo`` (the one ``_COMBO_MAP`` reader),
    so a NEW ModifierCombo value is built here automatically and only needs a
    row in ``_MODIFIER_FOR_NAME`` — never a new hand-written test case.
    """
    register_scripted(f"tot_{name}", lambda _in, _cfg: Seed())
    item = Node.scripted(name, fn=f"tot_{name}", outputs=Seed)
    for mod_name in sorted(modifier_names_for_combo(combo)):
        item = item | _MODIFIER_FOR_NAME[mod_name]()
    return item


register_condition("tot_always", lambda _d: True)
register_scripted("tot_merge", lambda _in, _cfg: Seed())

#: One representative instance per modifier NAME. Keyed by the same names
#: ``modifier_names_for_combo`` returns, so the two cannot drift silently:
#: a new modifier name with no row here raises KeyError in ``_item_carrying``,
#: which is the intended loud failure.
_MODIFIER_FOR_NAME = {
    "each": lambda: Each(over="trigger_tot.items", key="v"),
    "oracle": lambda: Oracle(n=2, merge_fn="tot_merge"),
    "loop": lambda: Loop(when=lambda d: d is None, max_iterations=2),
    "operator": lambda: Operator(when="tot_always"),
    "portal": lambda: Portal(to=["__end__"]),
}


class TestBranchArmGuardIsTotalOverModifierCombo:
    """The guard's predicate is read from ``COMBO_DECOMPOSITION`` (``primary is
    not BARE`` or ``has_operator``), never a hand-typed modifier list -- which
    is what makes it total over FUTURE ``PrimaryShape`` values.

    This parametrization proves that totality instead of asserting it: every
    ``ModifierCombo`` value is built and placed in an arm, and exactly one
    (``BARE``) is expected to survive. A new combo added to the enum is
    exercised here automatically.
    """

    @pytest.mark.parametrize("combo", list(ModifierCombo), ids=lambda c: c.name)
    def test_every_non_bare_combo_is_rejected_in_a_branch_arm(self, combo):
        name = f"tot-{combo.name.lower()}"
        item = _item_carrying(combo, name)

        if combo is ModifierCombo.BARE:
            # Must NOT raise — the guard rejects modifiers, not arm items.
            _construct_with_arm_item(f"tot_{combo.name.lower()}", item, trigger_out=Bag)
            return

        with pytest.raises(ConstructError) as exc:
            _construct_with_arm_item(f"tot_{combo.name.lower()}", item, trigger_out=Bag)
        msg = str(exc.value).lower()
        assert "arm" in msg
        for mod_name in modifier_names_for_combo(combo):
            assert mod_name in msg, f"error must name the carried modifier {mod_name!r}: {msg}"
