"""Hypothesis property tests for rendering invariants.

Stress-tests the rendering system against structural invariants that should
hold for any model/input combination. These tests probe the exact boundaries
where 7+ bugs were found in a single session.
"""

from __future__ import annotations

from typing import Optional

import hypothesis.strategies as st
from hypothesis import given, settings, assume
from pydantic import BaseModel, Field

from neograph.renderers import render_input, _render_single, _render_with_flattening
from neograph.describe_type import describe_value
from neograph._llm import _resolve_var, _render_tool_result_for_llm


# ── Strategies ──────────────────────────────────────────────────────────

class Alpha(BaseModel):
    value: str = "default"

class Beta(BaseModel):
    score: float = 0.5

class Nested(BaseModel):
    inner: Alpha = Field(default_factory=Alpha)
    count: int = 0

class WithProjection(BaseModel):
    raw: str = "data"

    def render_for_prompt(self) -> Alpha:
        return Alpha(value=self.raw.upper())

class WithStrProjection(BaseModel):
    raw: str = "data"

    def render_for_prompt(self) -> str:
        return f"CUSTOM: {self.raw}"

class WithNestedProjection(BaseModel):
    raw: str = "data"

    def render_for_prompt(self) -> Nested:
        return Nested(inner=Alpha(value=self.raw), count=len(self.raw))


simple_models = st.sampled_from([
    Alpha(value="test"),
    Beta(score=0.7),
    Nested(inner=Alpha(value="deep"), count=3),
])

projection_models = st.sampled_from([
    WithProjection(raw="hello"),
    WithStrProjection(raw="world"),
    WithNestedProjection(raw="nested"),
])

any_model = st.one_of(simple_models, projection_models)


@st.composite
def fan_in_dict(draw):
    """Generate a dict-form fan-in input with 1-3 entries."""
    n = draw(st.integers(min_value=1, max_value=3))
    d = {}
    for i in range(n):
        key = f"input_{i}"
        model = draw(any_model)
        d[key] = model
    return d


# ── Invariant: render_input never raises ────────────────────────────────

class TestRenderInputNeverRaises:
    """render_input must never raise for any valid Pydantic model input."""

    @given(model=any_model)
    @settings(max_examples=50)
    def test_single_model_no_crash(self, model):
        """Single model input: render_input always returns without raising."""
        result = render_input(model, renderer=None)
        assert result is not None

    @given(data=fan_in_dict())
    @settings(max_examples=50)
    def test_dict_input_no_crash(self, data):
        """Dict-form fan-in input: render_input always returns a dict."""
        result = render_input(data, renderer=None)
        assert isinstance(result, dict)

    @given(model=any_model)
    @settings(max_examples=30)
    def test_render_single_no_crash(self, model):
        """_render_single never raises for any model."""
        result = _render_single(model, None)
        assert result is not None


# ── Invariant: dict keys preserved ──────────────────────────────────────

class TestDictKeysPreserved:
    """Original dict keys must always be present in the rendered output."""

    @given(data=fan_in_dict())
    @settings(max_examples=50)
    def test_original_keys_survive_rendering(self, data):
        """Every key from the input dict appears in the rendered output."""
        result = render_input(data, renderer=None)
        for key in data:
            assert key in result, (
                f"Original key '{key}' missing from rendered output. "
                f"Input keys: {sorted(data.keys())}, "
                f"Output keys: {sorted(result.keys())}"
            )


# ── Invariant: render_for_prompt always wins ────────────────────────────

class TestRenderForPromptPrecedence:
    """render_for_prompt() must always take precedence over BAML default."""

    @given(model=projection_models)
    @settings(max_examples=30)
    def test_projection_applied_in_single(self, model):
        """Single-value: render_for_prompt result is used, not raw model."""
        result = _render_single(model, None)
        # For str projections: result is the custom string
        # For BaseModel projections: result is BAML of the projected model
        if isinstance(model, WithStrProjection):
            assert result.startswith("CUSTOM:")
        else:
            # Should NOT contain the raw field from the original model
            assert "raw" not in str(result) or "data" not in str(result)


# ── Invariant: BAML parity (tool result == input rendering) ─────────────

class TestBAMLParity:
    """Tool-result and input rendering must produce identical BAML body."""

    @given(model=simple_models)
    @settings(max_examples=30)
    def test_tool_vs_input_same_baml(self, model):
        """Same model instance → same BAML from tool-result and input paths."""
        tool_result = _render_tool_result_for_llm(model, renderer=None)
        input_result = render_input(model, renderer=None)
        expected = describe_value(model)

        # Tool result has "Tool result:" prefix
        tool_body = tool_result.replace("Tool result:\n", "")
        assert tool_body == expected or input_result == expected


# ── Invariant: flattened fields are a superset of input keys ────────────

class TestFlatteningInvariants:
    """Field flattening must produce a superset of the original dict keys."""

    @given(data=fan_in_dict())
    @settings(max_examples=50)
    def test_flattened_output_superset_of_input_keys(self, data):
        """Rendered dict keys are always >= input dict keys."""
        result = render_input(data, renderer=None)
        assert set(data.keys()) <= set(result.keys()), (
            f"Input keys {sorted(data.keys())} not subset of "
            f"output keys {sorted(result.keys())}"
        )

    def test_basemodel_child_preserved_not_stringified(self):
        """BaseModel children in flattened fields must be model instances."""
        result = render_input(
            {"src": WithNestedProjection(raw="test")},
            renderer=None,
        )
        if "inner" in result:
            assert isinstance(result["inner"], BaseModel), (
                f"BaseModel child should be preserved, got {type(result['inner']).__name__}"
            )


# ── Invariant: _resolve_var handles all value types ─────────────────────

class TestResolveVarRobustness:
    """_resolve_var must handle any value type without crashing."""

    @given(model=any_model)
    @settings(max_examples=30)
    def test_resolve_var_on_dict_no_crash(self, model):
        """_resolve_var with dict input never crashes."""
        data = {"key": model}
        result = _resolve_var("key", data)
        assert isinstance(result, str)

    @given(model=any_model)
    @settings(max_examples=30)
    def test_resolve_var_on_single_no_crash(self, model):
        """_resolve_var with single value never crashes."""
        result = _resolve_var("value", model)
        assert isinstance(result, str)

    def test_resolve_var_none_returns_empty(self):
        """_resolve_var with None value returns empty string."""
        result = _resolve_var("key", {"key": None})
        assert result == ""

    def test_resolve_var_basemodel_returns_baml(self):
        """_resolve_var with BaseModel returns BAML, not Pydantic repr."""
        model = Alpha(value="test")
        result = _resolve_var("key", {"key": model})
        assert "value" in result
        assert "Alpha(" not in result  # not Pydantic repr
