"""Structural guard: the Agent Spec coverage matrix must stay MECHANICALLY
DERIVED, never regress to a hand-typed enumeration (neograph-00447 --
codebase-scan:complete).

Disease pattern (the sdfgz blind spot): the export/round-trip coverage matrix
``tests/test_agent_spec_matrix.py`` enumerated its cells by hand, so a whole
closed axis (the Oracle ``merge_prompt`` variant, the entire ``mode`` axis) could
silently go uncovered -- the completeness of the matrix depended on a human
remembering every axis. neograph-00447 replaced the hand-typed ``CELLS`` dict with
``CELLS = _generate_cells()`` (an introspected cross product) plus loud partition
assertions.

This guard is the class-level ratchet that keeps the local fix from silently
regressing: it bans re-introducing a hand-typed ``CELLS`` literal and pins that
the loud-completeness machinery stays present. AST-based (not regex), so a
positive + negative meta-test pair is sufficient -- there is no regex-slip case.
"""

from __future__ import annotations

import ast
import pathlib

MATRIX_FILE = (
    pathlib.Path(__file__).resolve().parent / "test_agent_spec_matrix.py"
)

# The loud-completeness assertions that make the matrix self-guarding. If any of
# these test methods disappears, the matrix could silently under-cover again.
REQUIRED_COMPLETENESS_TESTS = frozenset(
    {
        "test_pyagentspec_registry_is_complete",
        "test_modifier_combo_axis_is_a_loud_partition",
        "test_every_generated_cell_is_classified",
        "test_prior_green_cells_are_still_generated_and_green",
    }
)


def _cells_rhs_is_generator_call(source: str) -> bool:
    """True iff the module-level ``CELLS`` binding is assigned from a CALL
    (e.g. ``CELLS = _generate_cells()``) rather than a hand-typed Dict literal.

    Returns False when CELLS is a dict literal (the disease) or absent.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        target_names: list[str] = []
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value = node.value
        elif isinstance(node, ast.Assign):
            target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        if "CELLS" in target_names and value is not None:
            # Disease = a Dict literal; cure = any Call (a generator function).
            return isinstance(value, ast.Call)
    return False


def _test_method_names(source: str) -> set[str]:
    """All ``def test_*`` names defined anywhere in the source."""
    tree = ast.parse(source)
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }


class TestAgentSpecMatrixStaysGenerated:
    """The matrix's cell set must be generated, and its completeness machinery
    must remain, so it can never silently under-cover an axis again."""

    def test_cells_is_generated_not_hand_typed(self) -> None:
        source = MATRIX_FILE.read_text()
        assert _cells_rhs_is_generator_call(source), (
            "tests/test_agent_spec_matrix.py's module-level CELLS must be assigned "
            "from a generator call (CELLS = _generate_cells()), NOT a hand-typed "
            "dict literal -- a hand-typed matrix silently under-covers closed axes "
            "(the neograph-sdfgz disease neograph-00447 cured)."
        )

    def test_loud_completeness_assertions_present(self) -> None:
        present = _test_method_names(MATRIX_FILE.read_text())
        missing = REQUIRED_COMPLETENESS_TESTS - present
        assert not missing, (
            "the matrix's loud-completeness assertions were removed: "
            f"{sorted(missing)}. These make the generated matrix self-guarding "
            "(partition vs ModifierCombo, generated-cell classification, registry "
            "completeness); without them a new axis under-covers silently."
        )


class TestGuardMetaTests:
    """Meta-tests: the guard must ACCEPT the current generated form and REJECT a
    hand-typed dict-literal regression (positive + negative)."""

    def test_positive_generated_form_is_accepted(self) -> None:
        good = "CELLS: dict[str, int] = _generate_cells()\n"
        assert _cells_rhs_is_generator_call(good)

    def test_negative_hand_typed_dict_is_rejected(self) -> None:
        # The exact disease: a hand-typed CELLS dict literal.
        bad = 'CELLS: dict = {\n    "none-single": build_none_single,\n}\n'
        assert not _cells_rhs_is_generator_call(bad)

    def test_negative_bare_assign_dict_is_rejected(self) -> None:
        # Same disease without a type annotation.
        bad = 'CELLS = {"a": 1, "b": 2}\n'
        assert not _cells_rhs_is_generator_call(bad)

    def test_missing_completeness_test_is_detected(self) -> None:
        # If the file defined no completeness tests, the guard must flag it.
        empty = "def helper():\n    pass\n"
        present = _test_method_names(empty)
        assert REQUIRED_COMPLETENESS_TESTS - present == REQUIRED_COMPLETENESS_TESTS
