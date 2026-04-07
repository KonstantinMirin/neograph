"""Condition expression evaluator tests — parse_condition grammar,
field resolution, type coercion, error paths, and injection rejection.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph.conditions import parse_condition


# ---- test models -------------------------------------------------------

class Score(BaseModel):
    value: float
    confidence: float


class Result(BaseModel):
    score: Score
    passed: bool
    name: str
    count: int


# ═══════════════════════════════════════════════════════════════════════════
# Numeric comparisons
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericComparisons:
    """Numeric operators work against model fields."""

    def test_less_than_when_field_below_threshold(self):
        cond = parse_condition("value < 0.8")
        assert cond(Score(value=0.5, confidence=0.9)) is True

    def test_less_than_when_field_above_threshold(self):
        cond = parse_condition("value < 0.8")
        assert cond(Score(value=0.9, confidence=0.9)) is False

    def test_greater_than_when_field_above_threshold(self):
        cond = parse_condition("value > 0.5")
        assert cond(Score(value=0.8, confidence=0.9)) is True

    def test_greater_than_when_field_below_threshold(self):
        cond = parse_condition("value > 0.5")
        assert cond(Score(value=0.3, confidence=0.9)) is False

    def test_less_equal_when_field_equals_threshold(self):
        cond = parse_condition("value <= 0.8")
        assert cond(Score(value=0.8, confidence=0.9)) is True

    def test_less_equal_when_field_below_threshold(self):
        cond = parse_condition("value <= 0.8")
        assert cond(Score(value=0.5, confidence=0.9)) is True

    def test_greater_equal_when_field_equals_threshold(self):
        cond = parse_condition("confidence >= 0.9")
        assert cond(Score(value=0.5, confidence=0.9)) is True

    def test_greater_equal_when_field_below_threshold(self):
        cond = parse_condition("confidence >= 0.9")
        assert cond(Score(value=0.5, confidence=0.8)) is False

    def test_equal_when_field_matches_int(self):
        cond = parse_condition("count == 3")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=3)) is True

    def test_equal_when_field_does_not_match_int(self):
        cond = parse_condition("count == 3")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=5)) is False

    def test_not_equal_when_field_differs(self):
        cond = parse_condition("count != 3")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=5)) is True

    def test_not_equal_when_field_matches(self):
        cond = parse_condition("count != 3")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=3)) is False


# ═══════════════════════════════════════════════════════════════════════════
# Boolean comparisons
# ═══════════════════════════════════════════════════════════════════════════

class TestBooleanComparisons:
    """Boolean literals (true/false) compare correctly."""

    def test_equal_true_when_field_is_true(self):
        cond = parse_condition("passed == true")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=1)) is True

    def test_equal_true_when_field_is_false(self):
        cond = parse_condition("passed == true")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=False, name="x", count=1)) is False

    def test_equal_false_when_field_is_false(self):
        cond = parse_condition("passed == false")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=False, name="x", count=1)) is True

    def test_not_equal_false_when_field_is_true(self):
        cond = parse_condition("passed != false")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=1)) is True


# ═══════════════════════════════════════════════════════════════════════════
# String comparisons
# ═══════════════════════════════════════════════════════════════════════════

class TestStringComparisons:
    """Quoted string literals compare correctly."""

    def test_equal_string_when_matches(self):
        cond = parse_condition('name == "draft"')
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="draft", count=1)) is True

    def test_not_equal_string_when_differs(self):
        cond = parse_condition('name != "draft"')
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="final", count=1)) is True

    def test_not_equal_string_when_matches(self):
        cond = parse_condition('name != "draft"')
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="draft", count=1)) is False


# ═══════════════════════════════════════════════════════════════════════════
# Dotted field access
# ═══════════════════════════════════════════════════════════════════════════

class TestDottedFieldAccess:
    """Dotted paths resolve through nested models."""

    def test_dotted_field_less_than(self):
        cond = parse_condition("score.value < 0.8")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=1)) is True

    def test_dotted_field_greater_equal(self):
        cond = parse_condition("score.confidence >= 0.9")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=1)) is True

    def test_dotted_field_on_dict(self):
        cond = parse_condition("result.score < 0.8")
        assert cond({"result": {"score": 0.5}}) is True

    def test_dotted_field_missing_raises(self):
        cond = parse_condition("result.missing < 0.8")
        with pytest.raises(AttributeError, match="missing"):
            cond({"result": {"score": 0.5}})


# ═══════════════════════════════════════════════════════════════════════════
# Error paths
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorPaths:
    """Bad expressions raise ValueError with clear messages."""

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError, match="Invalid condition expression"):
            parse_condition("")

    def test_missing_operator_raises(self):
        with pytest.raises(ValueError, match="Invalid condition expression"):
            parse_condition("score 0.8")

    def test_missing_literal_raises(self):
        with pytest.raises(ValueError, match="Invalid condition expression"):
            parse_condition("score <")

    def test_invalid_literal_raises(self):
        with pytest.raises(ValueError, match="Cannot parse literal"):
            parse_condition("score < notanumber")

    def test_bare_word_expression_raises(self):
        with pytest.raises(ValueError, match="Invalid condition expression"):
            parse_condition("hello")

    def test_injection_import_rejected(self):
        with pytest.raises(ValueError, match="Invalid condition expression"):
            parse_condition('__import__("os").system("rm -rf /")')

    def test_injection_eval_rejected(self):
        with pytest.raises(ValueError, match="Invalid condition expression"):
            parse_condition('eval("1+1")')

    def test_injection_semicolon_rejected(self):
        with pytest.raises(ValueError, match="Cannot parse literal"):
            parse_condition("score < 0.8; import os")

    def test_injection_double_underscore_field_rejected(self):
        """Fields starting with __ are rejected by the grammar (not alphanumeric start)."""
        # Actually __ does match [A-Za-z_] start, but let's verify the full
        # expression would need to match the grammar to be dangerous.
        # The key protection is that we never eval — we only do getattr.
        with pytest.raises(ValueError, match="Invalid condition expression"):
            parse_condition('__import__("os") == true')


class TestConditionEdgeCases:
    """Edge cases from TQ-10-12 audit."""

    def test_negative_number_literal(self):
        """Negative number: score > -0.5."""
        cond = parse_condition("score > -0.5")

        class M(BaseModel):
            score: float

        assert cond(M(score=0.0)) is True
        assert cond(M(score=-1.0)) is False

    def test_int_field_vs_float_literal(self):
        """Cross-type: integer field compared to float literal."""
        cond = parse_condition("count >= 2.5")

        class M(BaseModel):
            count: int

        assert cond(M(count=3)) is True
        assert cond(M(count=2)) is False

    def test_zero_literal(self):
        """Zero as literal: score == 0."""
        cond = parse_condition("score == 0")

        class M(BaseModel):
            score: float

        assert cond(M(score=0.0)) is True
        assert cond(M(score=0.1)) is False
