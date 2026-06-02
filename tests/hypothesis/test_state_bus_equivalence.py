"""Property-based equivalence tests for StateBus adapter.

Pins the invariant: for any state shape (dict or BaseModel) and any (key, default),
``_state_get_legacy(state, key, default)`` is observationally identical to
``adapt_state(state).get(key, default)``.

These tests guarantee the StateBus adapter introduced in Batch 2 (neograph-036p)
preserves the exact semantics of the prior duplicated ``_state_get`` helpers in
``factory.py`` and ``_oracle.py``.

Legacy semantics (the contract being preserved):
- dict state: ``state.get(key, default)`` -> default when missing, returns value when present
- BaseModel state: ``getattr(state, key, default)`` -> default when missing, returns value when present
- None values in state are returned as None (NOT treated as missing)
- Empty dict / empty BaseModel return ``default`` for any key
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from neograph._state_bus import adapt_state
from neograph.errors import StateMissingError


# ── The legacy reference implementation ────────────────────────────────────
# Mirrors the pre-Batch-2 ``_state_get`` (factory.py + _oracle.py) verbatim.
def _state_get_legacy(state: Any, key: str, default: Any = None) -> Any:
    """Reference behavior: what _state_get did before the adapter."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


# ── State shape strategies ─────────────────────────────────────────────────

KEYS = st.sampled_from(["alpha", "beta", "gamma", "neo_each_item", "missing_key", "node_id"])

# Allow None/sentinel/list/dict/int/str values to exercise the "value is None" branch
VALUES = st.one_of(
    st.none(),
    st.integers(min_value=-100, max_value=100),
    st.text(max_size=20),
    st.lists(st.integers(min_value=0, max_value=10), max_size=3),
    st.dictionaries(st.text(min_size=1, max_size=5), st.integers(min_value=0, max_value=10), max_size=3),
)

DEFAULTS = st.one_of(
    st.none(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=5),
    st.just(0),
    st.just("__SENTINEL__"),
)


@st.composite
def dict_states(draw):
    """Random dict states with str keys and arbitrary values."""
    keys = draw(st.lists(st.text(min_size=1, max_size=10), min_size=0, max_size=6, unique=True))
    items = {k: draw(VALUES) for k in keys}
    return items


class _DynamicModel(BaseModel):
    """Open-ended BaseModel for hypothesis-driven attribute composition."""
    model_config = {"extra": "allow", "arbitrary_types_allowed": True}


@st.composite
def model_states(draw):
    """Random BaseModel states constructed with extra='allow'."""
    keys = draw(st.lists(st.text(
        alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
        min_size=1,
        max_size=10,
    ), min_size=0, max_size=6, unique=True))
    items = {k: draw(VALUES) for k in keys}
    return _DynamicModel(**items)


STATES = st.one_of(dict_states(), model_states())


# ── Equivalence properties ─────────────────────────────────────────────────


@given(state=STATES, key=KEYS, default=DEFAULTS)
def test_get_value_equivalent_to_state_get(state, key, default):
    """For arbitrary state/key/default, adapter.get matches legacy _state_get."""
    legacy = _state_get_legacy(state, key, default)
    new = adapt_state(state).get(key, default)
    assert legacy == new


@given(state=STATES, key=KEYS)
def test_get_with_implicit_none_default(state, key):
    """Calling .get without a default returns None, matching legacy."""
    legacy = _state_get_legacy(state, key)
    new = adapt_state(state).get(key)
    assert legacy == new


@given(state=dict_states(), default=DEFAULTS)
def test_missing_key_returns_default_dict(state, default):
    """For dict state, a guaranteed-missing key returns the default."""
    key = "__definitely_not_in_state__"
    assert adapt_state(state).get(key, default) == default
    assert _state_get_legacy(state, key, default) == default


@given(state=model_states(), default=DEFAULTS)
def test_missing_key_returns_default_model(state, default):
    """For BaseModel state, a guaranteed-missing key returns the default."""
    key = "__definitely_not_in_state__"
    assert adapt_state(state).get(key, default) == default
    assert _state_get_legacy(state, key, default) == default


@given(state=STATES)
def test_none_value_is_returned_not_treated_as_missing(state):
    """If state has key bound to None, .get returns None (not default)."""
    # Inject the key with value None
    if isinstance(state, dict):
        state_with_none = {**state, "explicit_none": None}
    else:
        state_with_none = state.model_copy(update={"explicit_none": None})

    assert adapt_state(state_with_none).get("explicit_none", "DEFAULT") is None
    assert _state_get_legacy(state_with_none, "explicit_none", "DEFAULT") is None


def test_empty_dict_states():
    """Empty dict: any key returns default."""
    assert adapt_state({}).get("anything") is None
    assert adapt_state({}).get("anything", "default") == "default"


def test_empty_model_states():
    """Empty BaseModel: any key returns default."""
    empty = _DynamicModel()
    assert adapt_state(empty).get("anything") is None
    assert adapt_state(empty).get("anything", "default") == "default"


# ── Pinning specific call patterns from factory.py / _oracle.py ────────────


@given(default=DEFAULTS)
def test_neo_each_item_pattern(default):
    """The most common state-bus pattern: read 'neo_each_item' from state.

    Used by _extract_each_item, _build_state_update Each branch, _classify_input_shape,
    and the Each redirect_fn closures. Anchor on the exact key.
    """
    state_dict = {"neo_each_item": {"item_id": "abc"}}
    state_model = _DynamicModel(neo_each_item={"item_id": "abc"})

    assert adapt_state(state_dict).get("neo_each_item", default) == {"item_id": "abc"}
    assert adapt_state(state_model).get("neo_each_item", default) == {"item_id": "abc"}


@given(default=DEFAULTS)
def test_neo_loop_count_pattern(default):
    """Loop count field pattern: 'neo_loop_count_{field}'. Default of 0 is critical
    because callers do ``current_count = adapt_state(state).get(field) or 0``."""
    state_dict: dict[str, Any] = {}
    state_model = _DynamicModel()
    # Default-missing path used in _build_state_update line 269 and _apply_skip_when line 198
    assert adapt_state(state_dict).get("neo_loop_count_x") is None
    assert adapt_state(state_model).get("neo_loop_count_x") is None
    # `... or 0` idiom collapses None to 0 consistently
    assert (adapt_state(state_dict).get("neo_loop_count_x") or 0) == 0
    assert (adapt_state(state_model).get("neo_loop_count_x") or 0) == 0


# ── get_required properties (izo1-D / neograph-tzzi) ───────────────────────


@given(state_dict=dict_states(), key=KEYS)
def test_get_required_matches_get_when_key_present_dict(state_dict, key):
    """When the key exists in dict state, get_required and get return the
    same value (including None — explicit None binds are permitted)."""
    if key in state_dict:
        bus = adapt_state(state_dict)
        assert bus.get_required(key) == bus.get(key)


@given(state_dict=dict_states(), key=KEYS)
def test_get_required_raises_when_key_missing_dict(state_dict, key):
    """When the key is absent from dict state, get_required raises
    StateMissingError carrying the key and the node_label."""
    if key not in state_dict:
        bus = adapt_state(state_dict)
        with pytest.raises(StateMissingError) as exc_info:
            bus.get_required(key, node_label="test_node")
        msg = str(exc_info.value)
        assert key in msg
        assert "test_node" in msg


@given(state_model=model_states(), key=KEYS)
def test_get_required_matches_get_when_field_present_model(state_model, key):
    """Same equivalence as the dict case for BaseModel state."""
    if hasattr(state_model, key):
        bus = adapt_state(state_model)
        assert bus.get_required(key) == bus.get(key)


@given(state_model=model_states(), key=KEYS)
def test_get_required_raises_when_field_missing_model(state_model, key):
    """BaseModel state: missing attribute raises with the same message shape."""
    if not hasattr(state_model, key):
        bus = adapt_state(state_model)
        with pytest.raises(StateMissingError) as exc_info:
            bus.get_required(key, node_label="model_node")
        msg = str(exc_info.value)
        assert key in msg
        assert "model_node" in msg


# ── get_counter equivalence (neograph-ylk9) ────────────────────────────────


def _get_counter_legacy(state: Any, key: str) -> int:
    """Reference: the 'None/absent-means-zero' counter rule pre-get_counter."""
    return _state_get_legacy(state, key, None) or 0


@given(state=STATES, key=KEYS)
def test_get_counter_equivalent_to_legacy_or_zero(state, key):
    """For any state shape and key, get_counter matches the legacy
    'get(k) or 0' result whenever the stored value is an int/None/absent —
    the only shapes a monotonic counter field ever holds."""
    legacy = _get_counter_legacy(state, key)
    # get_counter only claims equivalence for counter-shaped values (int/None/
    # absent). Restrict the property to those to avoid comparing against
    # legacy's accidental coercion of non-int falsy values.
    raw = _state_get_legacy(state, key, None)
    if raw is None or isinstance(raw, int):
        assert adapt_state(state).get_counter(key) == legacy


@given(state=STATES, key=KEYS)
def test_get_counter_always_returns_int(state, key):
    assert isinstance(adapt_state(state).get_counter(key), int)
