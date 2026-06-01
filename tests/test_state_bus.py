"""Unit tests for StateBus adapters.

Covers the `get_required` accessor added per izo1-A (neograph-xwdl) — §7
of the architecture-decisions backlog. `get_required` raises
`StateMissingError` when the key is absent; explicit `None` values are
permitted (mirrors `dict` semantics).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph._state_bus import _DictStateBus, _ModelStateBus, adapt_state
from neograph.errors import NeographError, StateMissingError


class _StateModel(BaseModel):
    field_a: str = "default-a"
    field_b: int | None = None


class TestStateMissingErrorHierarchy:
    def test_state_missing_error_subclasses_neograph_error(self):
        err = StateMissingError.build(key="x")
        assert isinstance(err, NeographError)

    def test_build_with_node_label_includes_prefix(self):
        err = StateMissingError.build(key="topic", node_label="summarize")
        assert "[Node 'summarize']" in str(err)
        assert "topic" in str(err)

    def test_build_without_node_label_has_no_node_prefix(self):
        err = StateMissingError.build(key="topic")
        msg = str(err)
        assert "[Node" not in msg
        assert "topic" in msg
        assert msg.strip()


class TestDictStateBusGetRequired:
    def test_present_key_returns_value(self):
        bus = _DictStateBus({"k": 42})
        assert bus.get_required("k") == 42

    def test_explicit_none_value_returned(self):
        bus = _DictStateBus({"k": None})
        # Explicit None is a valid bound value; only absence raises.
        assert bus.get_required("k") is None

    def test_missing_key_raises_state_missing_error(self):
        bus = _DictStateBus({"present": 1})
        with pytest.raises(StateMissingError, match="absent_key"):
            bus.get_required("absent_key")

    def test_missing_key_message_includes_node_label(self):
        bus = _DictStateBus({})
        with pytest.raises(StateMissingError, match=r"\[Node 'summarize'\]"):
            bus.get_required("topic", node_label="summarize")


class TestModelStateBusGetRequired:
    def test_present_field_returns_value(self):
        bus = _ModelStateBus(_StateModel(field_a="hello"))
        assert bus.get_required("field_a") == "hello"

    def test_explicit_none_field_returned(self):
        bus = _ModelStateBus(_StateModel(field_b=None))
        assert bus.get_required("field_b") is None

    def test_missing_field_raises_state_missing_error(self):
        bus = _ModelStateBus(_StateModel())
        with pytest.raises(StateMissingError, match="not_a_field"):
            bus.get_required("not_a_field")

    def test_missing_field_message_includes_node_label(self):
        bus = _ModelStateBus(_StateModel())
        with pytest.raises(StateMissingError, match=r"\[Node 'scorer'\]"):
            bus.get_required("nope", node_label="scorer")


class TestAdaptStateParity:
    def test_dict_and_model_buses_share_get_required_semantics(self):
        dict_bus = adapt_state({"x": 1})
        model_bus = adapt_state(_StateModel(field_a="a"))
        assert dict_bus.get_required("x") == 1
        assert model_bus.get_required("field_a") == "a"
        with pytest.raises(StateMissingError):
            dict_bus.get_required("missing")
        with pytest.raises(StateMissingError):
            model_bus.get_required("missing")
