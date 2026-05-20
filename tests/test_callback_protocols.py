"""Regression baseline for PEP 696-parameterized callback Protocols.

When the SkipPredicate / SkipValueFactory / Merge* Protocols are parameterized
with TypeVar defaults, ``runtime_checkable`` + generic Protocol has known edge
cases (CPython issue tracker covers a handful). These tests pin the runtime
isinstance behavior we depend on, BOTH un-subscripted and after subscription
has been forced.

The same tests pass against the un-parameterized form (current baseline) and
against the parameterized form (after PEP 696 defaults are added) -- they
isolate the runtime contract, not the static-type contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph.modifiers import MergeFallback, MergePostProcess, MergePreProcess
from neograph.node import SkipPredicate, SkipValueFactory


class _Claims(BaseModel):
    items: list[str] = []


def _skip_fn(input_data: Any) -> bool:
    return False


def _skip_value_fn(input_data: Any) -> _Claims:
    return _Claims()


def _merge_pre_fn(variants: list[Any]) -> dict[str, Any]:
    return {"variants": variants}


def _merge_post_fn(result: Any, variants: list[Any]) -> Any:
    return result


def _merge_fallback_fn(variants: list[Any], error: Exception) -> _Claims:
    return _Claims()


class TestSkipPredicate:
    def test_isinstance_passes_unsubscripted(self):
        assert isinstance(_skip_fn, SkipPredicate)

    def test_isinstance_passes_after_subscript_reference(self):
        # Force subscripted access; isinstance MUST still operate on the un-parameterized class.
        _ = SkipPredicate[_Claims]  # noqa: F841 -- side effect: forces __class_getitem__
        assert isinstance(_skip_fn, SkipPredicate)

    def test_subscripted_isinstance_is_an_error(self):
        # Subscripted generic Protocols are not directly usable as isinstance args.
        # This is the same rule as `isinstance(x, list[int])`.
        with pytest.raises(TypeError):
            isinstance(_skip_fn, SkipPredicate[_Claims])  # type: ignore[misc]


class TestSkipValueFactory:
    def test_isinstance_passes_unsubscripted(self):
        assert isinstance(_skip_value_fn, SkipValueFactory)

    def test_isinstance_passes_after_subscript_reference(self):
        _ = SkipValueFactory[_Claims, _Claims]
        assert isinstance(_skip_value_fn, SkipValueFactory)

    def test_subscripted_isinstance_is_an_error(self):
        with pytest.raises(TypeError):
            isinstance(_skip_value_fn, SkipValueFactory[_Claims, _Claims])  # type: ignore[misc]


class TestMergePreProcess:
    def test_isinstance_passes_unsubscripted(self):
        assert isinstance(_merge_pre_fn, MergePreProcess)

    def test_isinstance_passes_after_subscript_reference(self):
        _ = MergePreProcess[_Claims]
        assert isinstance(_merge_pre_fn, MergePreProcess)

    def test_subscripted_isinstance_is_an_error(self):
        with pytest.raises(TypeError):
            isinstance(_merge_pre_fn, MergePreProcess[_Claims])  # type: ignore[misc]


class TestMergePostProcess:
    def test_isinstance_passes_unsubscripted(self):
        assert isinstance(_merge_post_fn, MergePostProcess)

    def test_isinstance_passes_after_subscript_reference(self):
        _ = MergePostProcess[_Claims, _Claims]
        assert isinstance(_merge_post_fn, MergePostProcess)

    def test_subscripted_isinstance_is_an_error(self):
        with pytest.raises(TypeError):
            isinstance(_merge_post_fn, MergePostProcess[_Claims, _Claims])  # type: ignore[misc]


class TestMergeFallback:
    def test_isinstance_passes_unsubscripted(self):
        assert isinstance(_merge_fallback_fn, MergeFallback)

    def test_isinstance_passes_after_subscript_reference(self):
        _ = MergeFallback[_Claims, _Claims]
        assert isinstance(_merge_fallback_fn, MergeFallback)

    def test_subscripted_isinstance_is_an_error(self):
        with pytest.raises(TypeError):
            isinstance(_merge_fallback_fn, MergeFallback[_Claims, _Claims])  # type: ignore[misc]


class TestProtocolsRejectNonCallables:
    """isinstance must reject non-callables that lack __call__ — same as before."""

    @pytest.mark.parametrize(
        "proto",
        [SkipPredicate, SkipValueFactory, MergePreProcess, MergePostProcess, MergeFallback],
    )
    def test_rejects_non_callable(self, proto):
        assert not isinstance(42, proto)
        assert not isinstance("hello", proto)
        assert not isinstance(_Claims(), proto)
