"""Structural guard: the ForwardConstruct parity ratchet must be
non-tautological (neograph-zrcln).

THE DISEASE: a coverage/ratchet "required" set built by comprehension over
the SAME collection the coverage assertion checks. Both sides of the check
then come from one source, so the set difference is structurally empty —
deleting a corpus row or shipping a capability without its parity twin can
never fail. tests/test_forward_parity.py:766 shipped exactly this form
(``REQUIRED_CAPABILITIES = frozenset(row.capability for row in
PARITY_CORPUS)``); a load-bearing row was deleted in an adversarial probe
and the ratchet stayed green.

THE GUARD: AST-walk the assignment of ``REQUIRED_CAPABILITIES`` in the
parity module and assert its value expression references no
``PARITY_CORPUS`` name — the required set must be a self-contained literal
(or derive from an independent production source), never from the corpus it
validates. AST-based, not regex, so positive + negative meta-tests suffice
(no regex-slip case exists).
"""

from __future__ import annotations

import ast
from pathlib import Path

PARITY_MODULE = Path(__file__).parent / "test_forward_parity.py"
REQUIRED_NAME = "REQUIRED_CAPABILITIES"
CORPUS_NAME = "PARITY_CORPUS"


def _names_referenced_by_assignment(source: str, target: str) -> set[str]:
    """Names referenced in the VALUE expression of the module-level
    assignment to ``target``. Raises StopIteration-free LookupError-style
    failure via an assert if the assignment is missing (a renamed ratchet
    set must update this guard, not silently pass)."""
    tree = ast.parse(source)
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == target for t in node.targets)
    ]
    assert assigns, f"no module-level assignment to {target} found"
    return {
        n.id for assign in assigns for n in ast.walk(assign.value) if isinstance(n, ast.Name)
    }


class TestParityRatchetIndependence:
    """The parity ratchet's required set must not derive from its corpus."""

    def test_required_capabilities_is_independent_of_corpus(self):
        referenced = _names_referenced_by_assignment(
            PARITY_MODULE.read_text(), REQUIRED_NAME
        )
        assert CORPUS_NAME not in referenced, (
            f"{REQUIRED_NAME} is derived from {CORPUS_NAME} — the ratchet is "
            "tautological (both sides of the coverage check come from the same "
            "list). Define it as an independent literal frozenset."
        )


class TestGuardMetaTests:
    """Meta-tests: the guard's predicate catches the disease form (positive)
    and stays quiet on the healthy form (negative)."""

    def test_meta_positive_derived_form_is_detected(self):
        diseased = (
            "PARITY_CORPUS = [1, 2]\n"
            "REQUIRED_CAPABILITIES = frozenset(row.capability for row in PARITY_CORPUS)\n"
        )
        referenced = _names_referenced_by_assignment(diseased, REQUIRED_NAME)
        assert CORPUS_NAME in referenced, (
            "guard predicate failed to detect the tautological derivation form"
        )

    def test_meta_positive_indirect_call_form_is_detected(self):
        # Laundering the derivation through a call still references the corpus
        # name inside the value expression — the walker must see through it.
        diseased = (
            "PARITY_CORPUS = [1, 2]\n"
            "REQUIRED_CAPABILITIES = frozenset(map(str, PARITY_CORPUS))\n"
        )
        referenced = _names_referenced_by_assignment(diseased, REQUIRED_NAME)
        assert CORPUS_NAME in referenced, (
            "guard predicate failed to detect a call-wrapped derivation"
        )

    def test_meta_negative_literal_form_is_clean(self):
        healthy = (
            "PARITY_CORPUS = [1, 2]\n"
            'REQUIRED_CAPABILITIES = frozenset({"straight_line", "fan_in"})\n'
        )
        referenced = _names_referenced_by_assignment(healthy, REQUIRED_NAME)
        assert CORPUS_NAME not in referenced

    def test_meta_missing_assignment_fails_loud(self):
        import pytest

        with pytest.raises(AssertionError, match="no module-level assignment"):
            _names_referenced_by_assignment("X = 1\n", REQUIRED_NAME)
