"""Structural guard: a check-fixture whose claim is a RESOLVED VALUE declares ``EXPECT``.

Root-cause pin for neograph-36302. ``test_should_pass`` used to assert only that
a fixture imports and compiles, so a fixture could assemble the right graph,
deliver the wrong value at run time, and stay green -- a green signal compatible
with the thing you care about not happening. ``input_from_names_the_port`` was
written with two producers tagged FIRST and SECOND *precisely* to discriminate,
and the harness threw its run away.

The harness now runs any fixture declaring a module-level ``EXPECT``. Nothing
else stops an author deleting one and silently restoring the blind state -- the
fixture would still compile, and the suite would still be green. This guard is
that stop. ``VALUE_RESOLUTION_FIXTURES`` is the MIGRATE column of that ticket's
disease scan; it may GROW when a new value-resolution fixture lands, and shrinks
only when a fixture is deleted outright.

Detection is AST-based, not a substring scan, because the two ways an ``EXPECT``
stops being live -- commented out, or demoted into a function body -- both leave
the text in the file. Both are exercised as would-be-missed meta-tests below.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SHOULD_PASS = pathlib.Path(__file__).resolve().parent / "check_fixtures" / "should_pass"

# Fixtures whose defining claim is a resolved VALUE (which producer, port, arm
# or peer was selected), not merely a valid shape.
VALUE_RESOLUTION_FIXTURES = (
    "input_from_names_the_port.py",  # which producer a single-type input port reads
    "output_from_names_a_real_item.py",  # which member IS the boundary
    "named_port_dotted_address.py",  # which dict-form output KEY is the boundary
    "branch_output_boundary_every_arm.py",  # which arm satisfies the boundary
    "portal_mesh_minimal.py",  # which peer a mesh hands off to
)

EXPECT_ATTR = "EXPECT"


def declares_expect(source: str) -> bool:
    """True when ``source`` binds ``EXPECT`` at MODULE level.

    Module level, because that is the only place the harness looks: an ``EXPECT``
    inside a function is dead, and one inside a comment was never code.
    """
    for stmt in ast.parse(source).body:
        targets: list[ast.expr] = []
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        if any(isinstance(t, ast.Name) and t.id == EXPECT_ATTR for t in targets):
            return True
    return False


class TestValueResolutionFixturesDeclareExpectations:
    """The real assertion: every listed fixture still declares its expectation."""

    @pytest.mark.parametrize("fixture_name", VALUE_RESOLUTION_FIXTURES)
    def test_the_fixture_declares_an_expectation(self, fixture_name: str):
        path = SHOULD_PASS / fixture_name
        assert path.exists(), (
            f"{fixture_name} is listed as a value-resolution fixture but does not exist. "
            f"If it was deleted, drop its row -- do not leave a dangling name."
        )
        assert declares_expect(path.read_text()), (
            f"{fixture_name}'s whole point is a resolved value, so it must declare a module-level "
            f"{EXPECT_ATTR} and be RUN. Without one the harness only compiles it, and the fixture "
            f"would pass identically if the feature it exercises were ignored entirely."
        )


class TestDeclaresExpectDetector:
    """Meta-tests for the detector itself -- positive, negative, would-be-missed."""

    def test_detects_a_module_level_assignment(self):
        assert declares_expect('EXPECT = {"sink": 1}\n')

    def test_detects_an_annotated_module_level_assignment(self):
        assert declares_expect('EXPECT: dict = {"sink": 1}\n')

    def test_does_not_detect_a_fixture_with_no_expectation(self):
        assert not declares_expect("pipeline = 1\n")

    def test_does_not_detect_a_commented_out_expectation(self):
        """Would-be-missed by a substring scan: the text is still in the file."""
        source = 'pipeline = 1\n# EXPECT = {"sink": 1}\n'
        assert EXPECT_ATTR in source  # a substring scan would call this declared
        assert not declares_expect(source)

    def test_does_not_detect_an_expectation_demoted_into_a_function(self):
        """Would-be-missed by a substring scan: assigned, but never at module level."""
        source = 'def helper():\n    EXPECT = {"sink": 1}\n    return EXPECT\n'
        assert EXPECT_ATTR in source  # a substring scan would call this declared
        assert not declares_expect(source)

    def test_does_not_detect_a_differently_named_binding(self):
        assert not declares_expect('EXPECTATIONS = {"sink": 1}\n')
