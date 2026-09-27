"""Guard: an AUTHORED reference is never taken on trust.

An authored reference is a NAME the author writes and the framework resolves --
``Construct.output_from``, ``Node.input_from``. It is the cheapest thing in the IR
to get wrong (a typo) and the most expensive to get wrong silently: the name
addresses a state field, a field nobody writes reads as absent, and absence used to
mean ``None`` on a green run.

``output_from`` shipped with ``check_output_from`` from the start. ``input_from``
shipped without a checker and with a comment CITING one -- "type-checked at assembly
by _validation_inputs" -- that named a function which did not exist. So a misspelled
``input_from`` did not merely fail to resolve; it DISPLACED a compatible producer
that was sitting right there, and the body received ``None``.

The lesson generalises past the two fields: the defect was not a missing check, it
was that nothing forced the question to be asked. This guard asks it. A new
``*_from`` field on ``Node`` or ``Construct`` must name the function that verifies
it, and that function must exist -- so "who checks this?" is answered at review time
rather than discovered by a run that returns the wrong answer quietly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from neograph import Construct, Node

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "neograph"

#: model field -> the function that verifies the name resolves. Both entries are
#: checked to EXIST below, so a renamed checker fails here instead of leaving a
#: field silently unverified.
AUTHORED_REFERENCE_CHECKERS: dict[str, str] = {
    "Construct.output_from": "check_output_from",
    # Resolves the name against the visible producers PLUS the branch-arm
    # producers, and refuses when it matches none or when the one it matches
    # produces an incompatible type. Arm producers are nameable on purpose: a
    # post-join read that an arm shadows is refused, and the refusal tells the
    # author to name the producer, so this is what makes that advice followable.
    "Node.input_from": "_resolve_named_port",
}


def _authored_reference_fields() -> list[str]:
    """Every ``*_from`` field on the two IR models, as ``Model.field``.

    Derived from the live models rather than listed, which is the whole point: a
    field added tomorrow appears here on its own and fails the assertion until
    somebody says who checks it.
    """
    return sorted(
        f"{model.__name__}.{name}"
        for model in (Node, Construct)
        for name in model.model_fields
        if name.endswith("_from")
    )


def _defines(function_name: str) -> list[str]:
    """Files under ``src/neograph`` defining ``function_name`` at module level."""
    hits = []
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        for node in ast.parse(py_file.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                hits.append(py_file.name)
    return hits


class TestEveryAuthoredReferenceNamesItsChecker:
    """The ratchet, stated over the live models."""

    def test_no_authored_reference_field_is_unaccounted_for(self):
        undeclared = [field for field in _authored_reference_fields() if field not in AUTHORED_REFERENCE_CHECKERS]
        assert undeclared == [], (
            "an authored reference has no named checker: "
            f"{undeclared}. A name the author writes and the framework resolves must be verified at "
            "assembly -- an unchecked one addresses a field nobody writes, which reads as absent and "
            "used to mean None on a green run (the input_from defect). Add the checker and name it in "
            "AUTHORED_REFERENCE_CHECKERS."
        )

    @pytest.mark.parametrize("field", sorted(AUTHORED_REFERENCE_CHECKERS))
    def test_the_named_checker_exists(self, field):
        checker = AUTHORED_REFERENCE_CHECKERS[field]
        assert _defines(checker), (
            f"{field} names {checker!r} as its checker, and no module under src/neograph defines it. "
            "Either the checker was renamed (update the row) or it was deleted (the field is now "
            "unverified, which is the defect this guard exists to catch)."
        )

    def test_both_tabled_fields_are_still_real_model_fields(self):
        """A row for a field that no longer exists certifies nothing."""
        live = set(_authored_reference_fields())
        stale = sorted(set(AUTHORED_REFERENCE_CHECKERS) - live)
        assert stale == [], f"rows name fields that are gone: {stale} -- delete them"


class TestTheRatchetActuallyFires:
    """Meta-tests. A ratchet nobody can see fire is decoration."""

    def test_a_simulated_new_field_with_no_row_is_caught(self):
        simulated = [*_authored_reference_fields(), "Node.seeded_from"]
        undeclared = [field for field in simulated if field not in AUTHORED_REFERENCE_CHECKERS]
        assert undeclared == ["Node.seeded_from"], (
            "the rule must flag a new authored reference that no row accounts for"
        )

    def test_a_checker_that_does_not_exist_is_caught(self):
        assert _defines("check_a_function_nobody_wrote") == [], (
            "the existence check must report a missing checker rather than pass vacuously"
        )

    def test_the_derivation_reads_the_live_models(self):
        """Both known fields are found by derivation, not by the table."""
        assert _authored_reference_fields() == ["Construct.output_from", "Node.input_from"]
