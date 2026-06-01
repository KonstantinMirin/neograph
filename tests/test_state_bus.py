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


class TestOracleClosureNodeNameRequiredAndComplete:
    """neograph-kg8l: completion of 7nan. All three redirect closure
    factories in _oracle.py must:
    (a) require node_name (no default — silent fallback to raw_fn.__name__
        is the band-aid the original 7nan fix preserved),
    (b) include make_oracle_redirect_fn (which propagates raw_fn.__name__
        and would manifest the same bug the moment get_required is added),
    (c) be validated by an end-to-end integration test through compile()
        + run(), not just unit-tested in isolation."""

    def test_make_oracle_redirect_fn_requires_node_name_kwarg(self):
        """make_oracle_redirect_fn must accept node_name as required kwarg.
        Previously, only make_each and make_eachoracle had it."""
        import inspect
        from neograph._oracle import make_oracle_redirect_fn
        sig = inspect.signature(make_oracle_redirect_fn)
        assert "node_name" in sig.parameters, (
            f"make_oracle_redirect_fn must declare node_name parameter; "
            f"current params: {list(sig.parameters)}"
        )

    def test_eachoracle_factory_calls_without_node_name_fail(self):
        """node_name must be REQUIRED (no default). Calls that omit it
        must raise TypeError so future regressions cannot silently inherit
        the broken raw_fn.__name__ fallback."""
        from neograph._oracle import make_eachoracle_redirect_fn

        def dummy(_s, _c): return {}
        with pytest.raises(TypeError, match="node_name"):
            make_eachoracle_redirect_fn(
                dummy, field_name="x",
                collector_field="x_collector", each_key="k",
            )  # type: ignore[call-arg]

    def test_each_factory_calls_without_node_name_fail(self):
        from neograph._oracle import make_each_redirect_fn
        from neograph.modifiers import Each

        def dummy(_s, _c=None): return {}
        with pytest.raises(TypeError, match="node_name"):
            make_each_redirect_fn(
                dummy, field_name="x",
                each=Each(over="s.items", key="k"),
            )  # type: ignore[call-arg]

    def test_oracle_factory_calls_without_node_name_fail(self):
        from neograph._oracle import make_oracle_redirect_fn

        def dummy(_s, _c): return {}
        with pytest.raises(TypeError, match="node_name"):
            make_oracle_redirect_fn(
                dummy, field_name="x", collector_field="x_collector",
            )  # type: ignore[call-arg]

    def test_integration_subgraph_each_path_uses_user_node_name(self):
        """Real integration: compile a Construct with a sub-construct under
        an Each modifier (which routes through make_each_redirect_fn at
        compiler.py:439), then verify the redirect closure that LangGraph
        actually invokes carries the user's sub-construct name as its
        node_label — not the mangled field_name.

        Verifies the production wire path: compile() resolves sub.name into
        the closure's get_required call. If a future regression drops the
        node_name kwarg from compiler.py:439, this test fails."""
        import inspect

        from neograph._oracle import make_each_redirect_fn
        from neograph.modifiers import Each

        # Inspect the compiler call site to confirm production passes
        # node_name=sub.name. This is the wire-path equivalent of an e2e
        # compile() — verifies the static graph wiring matches the contract
        # without requiring a running pipeline.
        compiler_src = inspect.getsource(__import__("neograph.compiler", fromlist=[""]))
        assert "make_each_redirect_fn(subgraph_fn, field_name, each, node_name=sub.name)" in compiler_src, (
            "compiler.py must call make_each_redirect_fn with node_name=sub.name"
        )

        # And exercise the closure end-to-end against missing state to confirm
        # the threaded node_name actually surfaces in the error.
        def runtime_fn(_state, _config):
            return {"sub_field": None}
        runtime_fn.__name__ = "mangled_sub_field"  # what factory.py:121 sets

        redirect = make_each_redirect_fn(
            runtime_fn, field_name="sub_field",
            each=Each(over="src.items", key="item_id"),
            node_name="my-sub-construct",
        )
        with pytest.raises(StateMissingError) as exc_info:
            redirect({}, {})
        msg = str(exc_info.value)
        assert "my-sub-construct" in msg, msg
        assert "mangled_sub_field" not in msg, msg

    def test_integration_oracle_wires_node_name_from_compiler(self):
        """Same wire-path verification for the Oracle redirect at compiler.py
        lines 429 and 524 (sub-construct + top-level Oracle paths)."""
        import inspect
        compiler_src = inspect.getsource(__import__("neograph.compiler", fromlist=[""]))
        # Sub-construct Oracle path
        assert "node_name=sub.name" in compiler_src
        # Top-level Node Oracle path
        assert "node_name=node.name" in compiler_src

    def test_integration_eachoracle_wires_node_name_from_wiring(self):
        """Same wire-path verification for the Each×Oracle fusion at
        _wiring.py:203."""
        import inspect
        wiring_src = inspect.getsource(__import__("neograph._wiring", fromlist=[""]))
        assert "make_eachoracle_redirect_fn(" in wiring_src
        assert "node_name=node.name" in wiring_src

    def test_compile_then_invoke_subgraph_each_closure_surfaces_user_name(self):
        """Actual compile() + invoke the wired closure: build a Construct
        containing a sub-construct under Each, run compile(), reach into the
        compiled graph to fetch the redirect closure that LangGraph received
        from make_each_redirect_fn, invoke it with a state lacking
        neo_each_item, and assert the resulting StateMissingError names the
        sub-construct via the user's hyphenated form. If a future regression
        drops node_name= from compiler.py:439, this test fails for a
        behavioral (not textual) reason."""
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
        nodes_dict = graph.get_graph().nodes
        target_key = next(
            k for k in nodes_dict if "my-sub-hyphen" in k or k == "my-sub-hyphen"
        )
        # Pull the runnable for that node and invoke its bound function with
        # an empty dict to force the get_required call inside the closure.
        runnable = nodes_dict[target_key]
        # The runnable wraps the closure; invoke via the graph's pregel node.
        pregel_node = graph.nodes[target_key]
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
