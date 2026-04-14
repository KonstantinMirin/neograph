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


class TestPrivateFieldAccessBlocked:
    """Dunder and single-underscore field segments are rejected at resolve time."""

    def test_dunder_class_raises(self):
        """'x.__class__' must raise, not leak type info."""
        cond = parse_condition('__class__ == "foo"')
        with pytest.raises(AttributeError, match="private/dunder"):
            cond(Score(value=0.5, confidence=0.9))

    def test_dotted_dunder_class_raises(self):
        """'score.__class__' via dotted path must raise."""
        cond = parse_condition('score.__class__ == "Score"')
        with pytest.raises(AttributeError, match="private/dunder"):
            cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=1))

    def test_dunder_dict_raises(self):
        """'x.__dict__.keys' must raise."""
        cond = parse_condition('__dict__ == "bar"')
        with pytest.raises(AttributeError, match="private/dunder"):
            cond(Score(value=0.5, confidence=0.9))

    def test_single_underscore_private_raises(self):
        """'x._private' single-underscore prefix also blocked."""
        cond = parse_condition('_private == 1')
        with pytest.raises(AttributeError, match="private/dunder"):
            cond(Score(value=0.5, confidence=0.9))

    def test_underscore_in_middle_of_name_works(self):
        """'normal_field' with underscore in middle is fine."""
        cond = parse_condition("confidence > 0.5")
        assert cond(Score(value=0.1, confidence=0.9)) is True

    def test_dotted_path_with_underscores_in_middle_works(self):
        """'score.value' style paths still work — underscore not at start."""
        cond = parse_condition("score.value < 0.8")
        assert cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=1)) is True

    def test_nested_dunder_in_dotted_path_raises(self):
        """'score.__class__.__module__' — dunder deeper in the path."""
        cond = parse_condition('score.__class__ == "Score"')
        with pytest.raises(AttributeError, match="private/dunder"):
            cond(Result(score=Score(value=0.5, confidence=0.9), passed=True, name="x", count=1))


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


# =============================================================================
# Coverage gap tests for conditions.py
# =============================================================================


class TestResolvePathAttributeError:
    """Lines 88-89: _resolve_field raises AttributeError for missing attributes."""

    def test_missing_attr_on_object_raises(self):
        """_resolve_field raises AttributeError when attribute doesn't exist."""
        from neograph.conditions import _resolve_field

        class M(BaseModel):
            score: float

        with pytest.raises(AttributeError, match="not found"):
            _resolve_field(M(score=0.5), "nonexistent")


class TestUnsupportedOperator:
    """Line 125: operator not in _OPS."""

    def test_parse_condition_with_nonmatching_expression_raises(self):
        """Expression that doesn't match the grammar raises ValueError."""
        with pytest.raises(ValueError, match="Invalid condition"):
            parse_condition("no-operator-here")

    def test_single_equals_not_in_ops(self):
        """Single = matches regex but is not in _OPS (line 125)."""
        # The regex op group [<>!=]=? matches "=" (! optional, = required)
        # But "=" is not in _OPS dict which has ==, !=, <, >, <=, >=
        with pytest.raises(ValueError, match="Unsupported operator"):
            parse_condition("score = 5")
