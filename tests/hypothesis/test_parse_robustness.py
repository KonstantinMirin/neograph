"""Hypothesis property-based tests for _parse_json_response robustness.

Generates random Pydantic models and malformed JSON responses to verify
that the parsing layer either returns a valid model or raises ExecutionError
— never crashes with an unhandled exception.

TASK neograph-f0vp: zero Hypothesis coverage for LLM output parsing.
"""

from __future__ import annotations

import json

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from pydantic import BaseModel, Field

from neograph._llm import _apply_null_defaults, _build_retry_msg, _parse_json_response
from neograph.errors import ExecutionError


# ── Fixed test models (varying complexity) ────────────────────────────────

class Simple(BaseModel):
    name: str
    score: float

class WithDefaults(BaseModel):
    name: str
    note: str = Field(default="")
    count: int = 0
    active: bool = True

class Nested(BaseModel):
    label: str
    inner: Simple

class WithList(BaseModel):
    items: list[Simple]

class DeepNested(BaseModel):
    title: str = ""
    sections: list[Nested] = Field(default_factory=list)

class MixedDefaults(BaseModel):
    required_str: str
    required_int: int
    optional_note: str = ""
    optional_score: float = 0.0
    optional_flag: bool = False


ALL_MODELS = [Simple, WithDefaults, Nested, WithList, DeepNested, MixedDefaults]


# ── Strategies ────────────────────────────────────────────────────────────

def st_simple_instance() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries({
        "name": st.text(min_size=1, max_size=20),
        "score": st.floats(min_value=-100, max_value=100, allow_nan=False),
    })

def st_with_defaults_instance() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries({
        "name": st.text(min_size=1, max_size=20),
        "note": st.one_of(st.text(max_size=50), st.just(None)),
        "count": st.one_of(st.integers(min_value=0, max_value=1000), st.just(None)),
        "active": st.one_of(st.booleans(), st.just(None)),
    })

def st_nested_instance() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries({
        "label": st.text(min_size=1, max_size=20),
        "inner": st_simple_instance(),
    })

def st_with_list_instance() -> st.SearchStrategy[dict]:
    return st.fixed_dictionaries({
        "items": st.lists(st_simple_instance(), min_size=0, max_size=5),
    })


@st.composite
def st_corrupt_json(draw, base_strategy, corruption_type=None):
    """Take a valid dict and apply a random corruption."""
    data = draw(base_strategy)
    json_str = json.dumps(data)

    corruption = corruption_type or draw(st.sampled_from([
        "wrap_markdown",
        "add_trailing_comma",
        "truncate",
        "nullify_random_field",
        "add_preamble",
        "stringify_number",
    ]))

    if corruption == "wrap_markdown":
        return f"```json\n{json_str}\n```"
    elif corruption == "add_trailing_comma":
        # Add trailing comma before last }
        idx = json_str.rfind("}")
        if idx > 0:
            return json_str[:idx] + "," + json_str[idx:]
        return json_str
    elif corruption == "truncate":
        # Cut off last 1-10 chars
        cut = draw(st.integers(min_value=1, max_value=min(10, len(json_str) // 2)))
        return json_str[:-cut]
    elif corruption == "nullify_random_field":
        if isinstance(data, dict) and data:
            key = draw(st.sampled_from(list(data.keys())))
            data[key] = None
            return json.dumps(data)
        return json_str
    elif corruption == "add_preamble":
        preamble = draw(st.sampled_from([
            "Here is the JSON:\n",
            "Sure! Let me help:\n\n",
            "Based on my analysis:\n",
        ]))
        return preamble + json_str
    elif corruption == "stringify_number":
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    data[k] = str(v)
                    break
            return json.dumps(data)
        return json_str
    return json_str


# ── Property tests ────────────────────────────────────────────────────────

class TestParseNeverCrashes:
    """_parse_json_response must either return a valid model or raise ExecutionError."""

    @given(data=st_simple_instance())
    @settings(max_examples=50)
    def test_valid_simple_always_parses(self, data):
        """Valid Simple JSON always produces a Simple instance."""
        text = json.dumps(data)
        result = _parse_json_response(text, Simple)
        assert isinstance(result, Simple)
        assert result.name == data["name"]

    @given(data=st_with_defaults_instance())
    @settings(max_examples=50)
    def test_nulls_coerced_to_defaults(self, data):
        """Null values for defaulted fields become the default, not failures."""
        text = json.dumps(data)
        result = _parse_json_response(text, WithDefaults)
        assert isinstance(result, WithDefaults)
        # Null fields should have their defaults
        if data["note"] is None:
            assert result.note == ""
        if data["count"] is None:
            assert result.count == 0
        if data["active"] is None:
            assert result.active is True

    @given(text=st_corrupt_json(st_simple_instance()))
    @settings(max_examples=100)
    def test_corrupted_simple_never_crashes(self, text):
        """Corrupted JSON either parses or raises ExecutionError — never crashes."""
        try:
            result = _parse_json_response(text, Simple)
            assert isinstance(result, Simple)
        except ExecutionError:
            pass  # expected for truly broken input

    @given(text=st_corrupt_json(st_nested_instance()))
    @settings(max_examples=50)
    def test_corrupted_nested_never_crashes(self, text):
        """Corrupted nested JSON either parses or raises ExecutionError."""
        try:
            result = _parse_json_response(text, Nested)
            assert isinstance(result, Nested)
        except ExecutionError:
            pass

    @given(text=st_corrupt_json(st_with_list_instance()))
    @settings(max_examples=50)
    def test_corrupted_list_never_crashes(self, text):
        """Corrupted list JSON either parses or raises ExecutionError."""
        try:
            result = _parse_json_response(text, WithList)
            assert isinstance(result, WithList)
        except ExecutionError:
            pass


class TestApplyNullDefaultsIdempotent:
    """_apply_null_defaults applied twice gives same result as once."""

    @given(data=st_with_defaults_instance())
    @settings(max_examples=50)
    def test_idempotent(self, data):
        """Applying null defaults twice is the same as once."""
        import copy
        d1 = copy.deepcopy(data)
        _apply_null_defaults(d1, WithDefaults)
        d2 = copy.deepcopy(d1)
        _apply_null_defaults(d2, WithDefaults)
        assert d1 == d2


class TestRetryMsgAlwaysProducesOutput:
    """_build_retry_msg never returns empty string."""

    @given(model=st.sampled_from(ALL_MODELS))
    @settings(max_examples=20)
    def test_retry_msg_with_model_nonempty(self, model):
        """Retry message with any model produces non-empty output."""
        err = ExecutionError("test")
        err.validation_errors = "some.field: Input should be a valid string"
        msg = _build_retry_msg(err, output_model=model)
        assert len(msg) > 50
        assert "schema" in msg.lower() or "fix" in msg.lower()

    def test_retry_msg_without_model_nonempty(self):
        """Retry message without model still produces useful output."""
        err = ExecutionError("test")
        msg = _build_retry_msg(err)
        assert len(msg) > 20
