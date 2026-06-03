"""Unit tests for StateBus adapters.

Covers the `get_required` accessor added per izo1-A (neograph-xwdl) — §7
of the architecture-decisions backlog. `get_required` raises
`StateMissingError` when the key is absent; explicit `None` values are
permitted (mirrors `dict` semantics).
"""

from __future__ import annotations

from types import SimpleNamespace

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


class TestGetCounter:
    """neograph-ylk9: StateBus.get_counter(key) internalizes the
    'None/absent-means-zero' rule for monotonic counters, replacing the
    'get(k) or 0' / 'get(k, 0)' idioms scattered across the runtime.

    Contract: absent key -> 0, explicit None -> 0, bound int -> that int.
    """

    def test_dict_bus_absent_key_returns_zero(self):
        assert _DictStateBus({}).get_counter("loop_count") == 0

    def test_dict_bus_explicit_none_returns_zero(self):
        assert _DictStateBus({"loop_count": None}).get_counter("loop_count") == 0

    def test_dict_bus_bound_int_returned(self):
        assert _DictStateBus({"loop_count": 3}).get_counter("loop_count") == 3

    def test_dict_bus_zero_is_returned_as_zero(self):
        # A genuinely-stored 0 reads as 0, same as absent — the contract is
        # identical either way for a monotonic counter.
        assert _DictStateBus({"loop_count": 0}).get_counter("loop_count") == 0

    def test_model_bus_absent_field_returns_zero(self):
        assert _ModelStateBus(_StateModel()).get_counter("not_a_field") == 0

    def test_model_bus_explicit_none_returns_zero(self):
        assert _ModelStateBus(_StateModel(field_b=None)).get_counter("field_b") == 0

    def test_model_bus_bound_int_returned(self):
        assert _ModelStateBus(_StateModel(field_b=5)).get_counter("field_b") == 5

    def test_return_type_is_always_int(self):
        assert isinstance(_DictStateBus({}).get_counter("x"), int)
        assert isinstance(_DictStateBus({"x": None}).get_counter("x"), int)

    def test_dict_and_model_buses_share_get_counter_semantics(self):
        assert adapt_state({"c": 2}).get_counter("c") == 2
        assert adapt_state(_StateModel(field_b=2)).get_counter("field_b") == 2
        assert adapt_state({}).get_counter("c") == 0
        assert adapt_state(_StateModel()).get_counter("field_b") == 0


class TestOracleEachLabelInError:
    """neograph-7nan / y20i: when StateMissingError fires from inside an
    Oracle or Each redirect closure, the error must name the USER's node
    (e.g. 'score-claim'), not the framework's internal closure name (e.g.
    'eachoracle_redirect_fn').

    Post-y20i the closure sources the label from the captured IR object's
    ``.name`` (Information Expert) — passed as ``item``, NOT threaded as a
    ``node_name`` string kwarg. The closure no longer relies on
    ``raw_fn.__name__`` at all (factory.py no longer mangles it)."""

    def test_eachoracle_redirect_error_carries_user_node_name(self):
        from neograph._oracle import make_eachoracle_redirect_fn
        from tests.schemas import MatchResult

        def user_node_fn(_state, _config):
            return {"score_claim": MatchResult(cluster_label="c", matched=["x"])}

        # The wrapper's __name__ is informational only post-y20i. The closure
        # must surface item.name, never the wrapper function name.
        user_node_fn.__name__ = "eachoracle_redirect_fn"
        redirect = make_eachoracle_redirect_fn(
            user_node_fn, field_name="score_claim",
            collector_field="score_claim_collector", each_key="claim_id",
            item=SimpleNamespace(name="score-claim"),
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

        user_node_fn.__name__ = "each_redirect_fn"
        redirect = make_each_redirect_fn(
            user_node_fn, field_name="score_claim",
            each=Each(over="src.items", key="claim_id"),
            item=SimpleNamespace(name="score-claim"),
        )
        with pytest.raises(StateMissingError) as exc_info:
            redirect({}, {})
        msg = str(exc_info.value)
        assert "score-claim" in msg, msg
        assert "each_redirect_fn" not in msg, msg


class TestOracleClosureNodeNameRequiredAndComplete:
    """neograph-y20i: the three redirect closure factories source the
    user-facing label from the captured IR object (``item.name``), not from
    a threaded ``node_name`` string and not from ``raw_fn.__name__``.

    The previous source-text guards (asserting the literal ``node_name=...``
    appears in compiler.py/_wiring.py) and the "node_name is a required
    kwarg" guards are deleted: they pinned the band-aid this epic removes.
    The behavioral contract is verified end-to-end by the compile()+invoke
    test below, which exercises the real production wire path."""

    def test_compile_then_invoke_subgraph_each_closure_surfaces_user_name(self):
        """Actual compile() + invoke the wired closure: build a Construct
        containing a sub-construct under Each, run compile(), reach into the
        compiled graph to fetch the redirect closure that LangGraph received
        from make_each_redirect_fn, invoke it with a state lacking
        neo_each_item, and assert the resulting StateMissingError names the
        sub-construct via the user's hyphenated form. If a future regression
        stops passing the IR object (item=sub) to make_each_redirect_fn at
        compiler.py:439, this test fails for a behavioral (not textual)
        reason."""
        from neograph import Construct, Node, compile
        from neograph.modifiers import Each
        from tests.fakes import build_test_compile_kwargs, register_scripted
        from tests.hypothesis.conftest import FanCollection, FanItem
        from tests.schemas import RawText

        register_scripted(
            "kg8l_seed",
            lambda _i, _c: FanCollection(items=[FanItem(item_id="a")]),
        )
        register_scripted("kg8l_inner", lambda _i, _c: RawText(text="processed"))

        sub = Construct(
            "my-sub-hyphen", input=FanItem, output=RawText,
            nodes=[Node.scripted("kg8l-inner", fn="kg8l_inner", outputs=RawText)],
        )
        sub_each = sub | Each(over="kg8l_seed.items", key="item_id")
        parent = Construct(
            "kg8l-parent", nodes=[
                Node.scripted("kg8l-seed", fn="kg8l_seed", outputs=FanCollection),
                sub_each,
            ],
        )
        graph = compile(parent, **build_test_compile_kwargs())
        # Reach into the compiled LangGraph to find the each redirect closure
        # for the sub-construct. Its name is sub.name per compiler.py:440.
        # compile() returns the CompiledNeograph facade; the raw LangGraph graph
        # (with its pregel `.nodes`) lives on the `.graph` field.
        lg_graph = graph.graph
        nodes_dict = lg_graph.get_graph().nodes
        target_key = next(
            k for k in nodes_dict if "my-sub-hyphen" in k or k == "my-sub-hyphen"
        )
        # Pull the runnable for that node and invoke its bound function with
        # an empty dict to force the get_required call inside the closure.
        runnable = nodes_dict[target_key]
        # The runnable wraps the closure; invoke via the graph's pregel node.
        pregel_node = lg_graph.nodes[target_key]
        bound = pregel_node.bound
        with pytest.raises(StateMissingError) as exc_info:
            bound.invoke({}, config={})
        msg = str(exc_info.value)
        assert "my-sub-hyphen" in msg, msg


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
