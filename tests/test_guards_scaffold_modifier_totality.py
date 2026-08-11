"""Structural guard: the scaffold captures EVERY modifier slot (neograph-wvp7j).

`scaffold.py` hand-enumerated oracle/each/loop/operator at four sites, so a
Portal-modified node produced no modifier assertions, no EXPECTED_MODIFIED entry, and
no drift tracking -- the scaffold reported covering the construct while silently not
covering it.

The fix derives `MODIFIER_CAPTURE` from `modifiers._SLOT_RULES`, the declared roster.
That makes a set-equality check between them VACUOUSLY TRUE, so this guard does NOT
assert set equality. It asserts NON-VACUITY instead: every slot's capture must return
a real dict for a node actually carrying that modifier, and `_node_info` must emit a
key per slot. A registry entry that exists but captures nothing would pass a
set-equality check and fail these.
"""

from __future__ import annotations

import pathlib

from neograph.modifiers import _SLOT_RULES
from neograph.testing._scaffold_capture import MODIFIER_CAPTURE, capture_modifiers

_SCAFFOLD_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph" / "testing"


class TestEveryModifierSlotIsCaptured:
    def test_the_registry_covers_the_declared_roster(self):
        """Derived, so this is near-vacuous on its own -- it exists to fail LOUDLY
        with a readable message if the derivation is ever replaced by a hand-list."""
        assert set(MODIFIER_CAPTURE) == {r.slot for r in _SLOT_RULES}

    def test_every_slot_captures_something_for_a_node_that_has_it(self):
        """The non-vacuity check: a registry entry that returns None for a node
        carrying that modifier would satisfy set-equality and still cover nothing."""
        from neograph.modifiers import Each, Loop, Operator, Oracle, Portal
        from neograph.node import Node
        from tests.schemas import Claims

        def _n() -> Node:
            return Node.scripted("m", fn="f", inputs=Claims, outputs=Claims)

        built = {
            "oracle": _n() | Oracle(n=2, merge_fn="mrg"),
            "each": _n() | Each(over="src.items", key="k"),
            "loop": _n() | Loop(when="cond", max_iterations=2),
            "operator": _n() | Operator(when="cond"),
            "portal": _n() | Portal(to=["peer"]),
        }
        assert set(built) == set(MODIFIER_CAPTURE), (
            "this test must exercise EVERY registered slot; add the missing one "
            f"({set(MODIFIER_CAPTURE) - set(built)})"
        )
        for slot, node in built.items():
            captured = capture_modifiers(node)
            assert captured[slot] is not None, f"{slot} captured nothing for a node that has it"
            assert isinstance(captured[slot], dict), f"{slot} must capture a dict"

    def test_capture_returns_a_key_for_every_slot_even_when_absent(self):
        """`_node_info` spreads capture_modifiers(), and the generators index
        `n[slot]` -- a missing key would be a KeyError, not a silent miss."""
        from neograph.node import Node
        from tests.schemas import Claims

        captured = capture_modifiers(Node.scripted("plain", fn="f", inputs=Claims, outputs=Claims))
        assert set(captured) == set(MODIFIER_CAPTURE)
        assert all(v is None for v in captured.values())


class TestNoHandRolledSlotEnumerationSurvives:
    """The four sites that used to enumerate the roster by hand."""

    def test_scaffold_has_no_inline_four_way_slot_disjunct(self):
        offenders: list[str] = []
        for py in sorted(_SCAFFOLD_DIR.glob("*.py")):
            for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                # the disease shape: two or more slot names OR-ed together on one line
                hits = sum(f'"{r.slot}"' in line for r in _SLOT_RULES)
                if hits >= 2 and " or " in line:
                    offenders.append(f"{py.name}:{i}: {line.strip()}")
        assert not offenders, (
            "enumerate the roster via MODIFIER_CAPTURE, never a hand-written disjunct -- "
            "the four-way version is what made Portal invisible (neograph-wvp7j).\n  "
            + "\n  ".join(offenders)
        )

    def test_the_emitted_drift_check_is_total_by_construction(self):
        """The GENERATED file freezes at scaffold time, so it is the one site a later
        fix here could never reach. It must not enumerate at all."""
        src = (_SCAFFOLD_DIR / "scaffold.py").read_text(encoding="utf-8")
        assert "item.modifier_set.to_list()" in src, (
            "the emitted has_mod check must use modifier_set.to_list(), which walks "
            "_SLOT_RULES and is total for modifiers that do not exist yet"
        )
        assert "item.modifier_set.oracle or item.modifier_set.each" not in src, (
            "the emitted 4-way enumeration is back -- a generated suite freezes, so this "
            "one would rot again on the sixth modifier"
        )

    def test_the_guard_detects_a_reintroduced_disjunct(self):
        """Mutation check: the scan must fire on the disease shape."""
        sample = 'modified = [n for n in nodes if n["oracle"] or n["each"]]'
        hits = sum(f'"{r.slot}"' in sample for r in _SLOT_RULES)
        assert hits >= 2 and " or " in sample


class TestScaffoldEmitsRealPortalCoverage:
    """should_pass-style: a Portal pipeline actually produces assertions.

    The acceptance criteria say the scaffold must "cover" Portal, and the failure
    mode this ticket exists to refuse is satisfying that to the letter while emitting
    nothing. So this asserts on the GENERATED TEXT, not on the capture dict.
    """

    @staticmethod
    def _generated_modifiers_file() -> str:
        import tempfile

        from pydantic import BaseModel

        from neograph import Construct, Node
        from neograph.modifiers import Portal
        from neograph.testing.scaffold import scaffold_tests

        class Handoff(BaseModel, frozen=True):
            goto: str

        triage = Node(
            name="triage", mode="agent", model="r", prompt="p",
            inputs={"handoff": Handoff}, outputs=Handoff, tools=[],
        ) | Portal(to=["scribe"], trigger="tool", max_hops=6)
        scribe = Node(
            name="scribe", mode="agent", model="w", prompt="p",
            inputs={"handoff": Handoff}, outputs=Handoff, tools=[],
        ) | Portal(to=["triage"])

        out = tempfile.mkdtemp()
        scaffold_tests(Construct("mesh-pipeline", nodes=[triage, scribe]), out, construct_import="x.y", overwrite=True)
        return (pathlib.Path(out) / "test_modifiers.py").read_text(encoding="utf-8")

    def test_a_portal_mesh_generates_mesh_assertions(self):
        text = self._generated_modifiers_file()
        assert "class TestPortalMesh" in text, "a Portal mesh must generate a mesh test class"
        assert "['triage', 'scribe']" in text, (
            "ordered membership must be baked in as a LITERAL and compared against the "
            "live grouping -- comparing two live calls would be a tautology"
        )
        assert "max_hops == 6" in text, "the entry's declared budget must be asserted"

    def test_an_unset_knob_generates_no_assertion(self):
        """The R1 rule: never emit a value the author did not write. Asserting
        `on_exhaust == "error"` would freeze a LIBRARY default into a USER suite --
        change DEFAULT_ON_EXHAUST and every such suite goes red, blaming a neograph
        change on the user's pipeline."""
        text = self._generated_modifiers_file()
        assert "on_exhaust ==" not in text, (
            "the mesh entry never set on_exhaust, so no assertion may be emitted for it"
        )
