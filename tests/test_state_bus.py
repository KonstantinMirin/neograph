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


class TestOracleEachLabelInError:
    """neograph-7nan: when StateMissingError fires from inside an Oracle
    or Each redirect closure, the error must name the USER's node (e.g.
    'score-claim'), not the framework's internal closure name (e.g.
    'eachoracle_redirect_fn'). The closures previously passed
    ``raw_fn.__name__`` which surfaces a confusing framework label."""

    def test_eachoracle_redirect_error_carries_user_node_name(self):
        from neograph._oracle import make_eachoracle_redirect_fn
        from tests.schemas import MatchResult

        def user_node_fn(_state, _config):
            return {"score_claim": MatchResult(cluster_label="c", matched=["x"])}

        # In production, factory.py sets raw_fn.__name__ to the MANGLED
        # field_name (hyphens replaced with underscores). The error must
        # surface the user's original node name, not this mangled form.
        user_node_fn.__name__ = "score_claim"  # what factory.py would set
        redirect = make_eachoracle_redirect_fn(
            user_node_fn, field_name="score_claim",
            collector_field="score_claim_collector", each_key="claim_id",
            node_name="score-claim",
        )
        # Empty dict state lacks EACH_ITEM — closure's get_required must raise.
        with pytest.raises(StateMissingError) as exc_info:
            redirect({}, {})
        msg = str(exc_info.value)
        # The closure must surface the user's node name, not the closure name.
        assert "score-claim" in msg, msg
        assert "eachoracle_redirect_fn" not in msg, msg

    def test_each_redirect_error_carries_user_node_name(self):
        from neograph._oracle import make_each_redirect_fn
        from neograph.modifiers import Each
        from tests.schemas import MatchResult

        def user_node_fn(_state, _config):
            return {"score_claim": MatchResult(cluster_label="c", matched=["x"])}

        # In production, factory.py sets raw_fn.__name__ to the MANGLED
        # field_name (hyphens replaced with underscores). The error must
        # surface the user's original node name, not this mangled form.
        user_node_fn.__name__ = "score_claim"  # what factory.py would set
        redirect = make_each_redirect_fn(
            user_node_fn, field_name="score_claim",
            each=Each(over="src.items", key="claim_id"),
            node_name="score-claim",
        )
        with pytest.raises(StateMissingError) as exc_info:
            redirect({}, {})
        msg = str(exc_info.value)
        assert "score-claim" in msg, msg
        assert "each_redirect_fn" not in msg, msg


class TestEndToEndRequiredStateMiss:
    """Behavioral: when a node's runtime requires a state field that no
    upstream produced, the pipeline raises StateMissingError naming the
    consuming node — rather than silently rendering None or AttributeError.

    Anchors izo1-D against the live §7 pathology fixed in izo1-B at
    _execute.py:46 (`_extract_context`)."""

    def test_extract_context_on_dict_state_with_missing_field_raises(self):
        """Integration-style: feed _extract_context a dict-shaped StateBus
        (the sub-construct/runtime-isolated path) missing the declared
        context field. The helper must raise StateMissingError naming the
        consuming node — proving the live §7 pathology fixed in izo1-B
        (no more silent 'None' rendering at _execute.py:46) is locked down.
        """
        from neograph._execute import _extract_context
        from neograph.node import Node
        from tests.schemas import MatchResult, RawText

        consumer = Node(
            name="stb-consumer", mode="think", outputs=MatchResult,
            inputs={"upstream": RawText},
            context=["never_produced"],
            prompt="dummy", model="fast",
        )
        # Dict state with the input bound but the context field absent —
        # the exact shape sub-construct dispatch produces when parent state
        # doesn't forward a declared context (cf. _subconstruct.py:101-107).
        state = adapt_state({"upstream": RawText(text="x")})

        with pytest.raises(StateMissingError) as exc_info:
            _extract_context(state, consumer)
        msg = str(exc_info.value)
        assert "never_produced" in msg
        assert "stb-consumer" in msg
