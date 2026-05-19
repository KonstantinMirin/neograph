"""Unit tests for ``neograph._normalize``.

Covers every branch of the ``type | dict[str, type] | None`` trichotomy for
both ``normalize_outputs`` and ``normalize_inputs``. The normalizer is the
single discrimination point used by 18+ call sites; a bug here silently
corrupts three-surface output/input handling.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel

from neograph._normalize import (
    NormalizedInputs,
    NormalizedOutputs,
    normalize_inputs,
    normalize_outputs,
)


class A(BaseModel):
    x: int


class B(BaseModel):
    y: str


class C(BaseModel):
    z: float


class Only(BaseModel):
    value: int


class X(BaseModel):
    a: int


class Y(BaseModel):
    b: str


class TestNormalizeOutputs:
    """``normalize_outputs`` discriminates None / dict / single-type."""

    def test_returns_single_form_when_outputs_is_single_type(self):
        result = normalize_outputs(A)

        assert isinstance(result, NormalizedOutputs)
        assert result.primary is A
        assert result.primary_key is None
        assert result.secondary == {}
        assert result.all_keys == {}
        assert result.is_dict_form is False
        assert result.is_none is False

    def test_returns_none_form_when_outputs_is_none(self):
        result = normalize_outputs(None)

        assert result.is_none is True
        assert result.primary is None
        assert result.primary_key is None
        assert result.secondary == {}
        assert result.all_keys == {}
        assert result.is_dict_form is False

    def test_returns_dict_form_with_first_key_as_primary_when_outputs_is_multi_key_dict(self):
        result = normalize_outputs({"a": A, "b": B, "c": C})

        assert result.primary is A
        assert result.primary_key == "a"
        assert result.secondary == {"b": B, "c": C}
        assert result.all_keys == {"a": A, "b": B, "c": C}
        assert result.is_dict_form is True
        assert result.is_none is False

    def test_returns_dict_form_with_empty_secondary_when_outputs_is_single_key_dict(self):
        result = normalize_outputs({"only": Only})

        assert result.primary is Only
        assert result.primary_key == "only"
        assert result.secondary == {}
        assert result.all_keys == {"only": Only}
        assert result.is_dict_form is True
        assert result.is_none is False

    def test_raises_when_outputs_is_empty_dict(self):
        # Empty dict reaches the items[0] unpack and IndexErrors. The 18
        # production call sites never construct a Node with outputs={}, so
        # the normalizer correctly treats this as a programmer error rather
        # than silently masking it as is_none.
        with pytest.raises(IndexError):
            normalize_outputs({})

    def test_all_keys_preserves_insertion_order_for_dict_form(self):
        result = normalize_outputs({"first": A, "second": B, "third": C})

        assert list(result.all_keys.keys()) == ["first", "second", "third"]
        assert list(result.secondary.keys()) == ["second", "third"]

    def test_all_keys_is_empty_for_single_type_form(self):
        result = normalize_outputs(A)

        assert result.all_keys == {}

    def test_all_keys_is_empty_for_none_form(self):
        result = normalize_outputs(None)

        assert result.all_keys == {}

    def test_returns_frozen_dataclass(self):
        result = normalize_outputs(A)

        with pytest.raises(FrozenInstanceError):
            result.primary = B  # type: ignore[misc]

    def test_secondary_dict_is_independent_copy(self):
        source = {"a": A, "b": B}
        result = normalize_outputs(source)

        # Mutating the source must not corrupt the normalized view.
        source["c"] = C
        assert "c" not in result.all_keys
        assert "c" not in result.secondary


class TestNormalizeInputs:
    """``normalize_inputs`` discriminates None / dict / single-type."""

    def test_returns_none_form_when_inputs_is_none(self):
        result = normalize_inputs(None)

        assert isinstance(result, NormalizedInputs)
        assert result.is_none is True
        assert result.by_name == {}
        assert result.single_type is None
        assert result.is_dict_form is False

    def test_returns_single_form_when_inputs_is_single_type(self):
        result = normalize_inputs(X)

        assert result.single_type is X
        assert result.by_name == {}
        assert result.is_dict_form is False
        assert result.is_none is False

    def test_returns_dict_form_when_inputs_is_dict(self):
        result = normalize_inputs({"x": X, "y": Y})

        assert result.by_name == {"x": X, "y": Y}
        assert result.single_type is None
        assert result.is_dict_form is True
        assert result.is_none is False

    def test_returns_empty_dict_form_when_inputs_is_empty_dict(self):
        # Empty dict is a valid (if degenerate) dict — the normalizer reports
        # is_dict_form=True with an empty by_name. Callers that need at least
        # one input are responsible for their own validation.
        result = normalize_inputs({})

        assert result.by_name == {}
        assert result.is_dict_form is True
        assert result.is_none is False
        assert result.single_type is None

    def test_by_name_preserves_insertion_order(self):
        result = normalize_inputs({"first": X, "second": Y})

        assert list(result.by_name.keys()) == ["first", "second"]

    def test_by_name_is_independent_copy(self):
        source = {"x": X}
        result = normalize_inputs(source)

        source["y"] = Y
        assert "y" not in result.by_name

    def test_returns_frozen_dataclass(self):
        result = normalize_inputs(X)

        with pytest.raises(FrozenInstanceError):
            result.single_type = Y  # type: ignore[misc]
