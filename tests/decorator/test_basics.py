"""@node decorator tests — decoration basics, mode inference, tool decorator"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Node,
    Tool,
    compile,
    construct_from_module,
    node,
    run,
    tool,
)
from neograph.factory import register_scripted
from tests.fakes import configure_fake_llm
from tests.schemas import (
    Claims,
    ClassifiedClaims,
    ClusterGroup,
    Clusters,
    RawText,
)


class TestToolDecorator:
    """@tool decorator: signature-inferred tool schemas."""

    def test_tool_registers_and_invokes_when_decorated_with_budget(self):
        """@tool wraps a function, auto-registers the factory, returns a Tool spec."""
        from langchain_core.messages import AIMessage

        call_log = []

        @tool(budget=3)
        def search_codebase(query: str) -> str:
            """Search the codebase for a query."""
            call_log.append(query)
            return f"Results for: {query}"

        # The decorator returns a Tool instance
        assert isinstance(search_codebase, Tool)
        assert search_codebase.name == "search_codebase"
        assert search_codebase.budget == 3

        # Build a pipeline using it directly (no register_tool_factory needed)
        counter = {"n": 0}

        class FakeGatherLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                counter["n"] += 1
                if counter["n"] <= 2:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{
                        "name": "search_codebase",
                        "args": {"query": f"q{counter['n']}"},
                        "id": f"c{counter['n']}",
                    }]
                    return msg
                return AIMessage(content="done")

            def with_structured_output(self, model, **kwargs):
                self._model = model
                return self

        configure_fake_llm(lambda tier: FakeGatherLLM())

        researcher = Node(
            name="research",
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[search_codebase],  # decorator output used directly
        )

        pipeline = Construct("test-tool-decorator", nodes=[researcher])
        graph = compile(pipeline)
        run(graph, input={})

        # The decorated function was called twice (within budget)
        assert len(call_log) == 2
        assert call_log == ["q1", "q2"]

    def test_tool_returns_spec_when_decorated_without_parens(self):
        """@tool (no parens) also works."""
        @tool
        def noop(x: str) -> str:
            """A no-op tool."""
            return x

        assert isinstance(noop, Tool)
        assert noop.name == "noop"
        assert noop.budget == 0  # unlimited by default


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecorator — @node + construct_from_module (Dagster-style signatures)
#
# Parameter names in the decorated function name the upstream nodes. The
# decorator produces a plain Node; construct_from_module walks a module's
# @node-built nodes and topologically sorts them into a Construct. No new
# IR path — compile()/run() handle the result unchanged.
# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecorator — @node + construct_from_module (Dagster-style signatures)
#
# Parameter names in the decorated function name the upstream nodes. The
# decorator produces a plain Node; construct_from_module walks a module's
# @node-built nodes and topologically sorts them into a Construct. No new
# IR path — compile()/run() handle the result unchanged.
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecorator:
    """@node decorator: parameter-name-based dependency inference."""

    @staticmethod
    def _fresh_module(name: str):
        """Create a throwaway module object for construct_from_module to walk."""
        import types as _types
        return _types.ModuleType(name)

    def test_chain_compiles_and_runs_when_two_nodes_wired_by_param_name(self):
        """Two @node-decorated scripted functions wired by parameter name,
        assembled via construct_from_module, compile and run end-to-end."""
        from neograph import compile, run

        mod = self._fresh_module("test_basic_chain_mod")

        @node(mode="scripted", outputs=RawText)
        def seed() -> RawText:
            return RawText(text="hello world")

        @node(mode="scripted", outputs=Claims)
        def split(seed: RawText) -> Claims:
            return Claims(items=[w for w in seed.text.split() if w])

        mod.seed = seed
        mod.split = split

        pipeline = construct_from_module(mod)

        # It is a Construct, with nodes in dependency order
        assert isinstance(pipeline, Construct)
        assert [n.name for n in pipeline.nodes] == ["seed", "split"]

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "basic-chain"})

        assert isinstance(result["split"], Claims)
        assert result["split"].items == ["hello", "world"]

    def test_fan_in_produces_result_when_three_upstreams_wired(self):
        """A node with three parameters gets wired to three upstream nodes,
        and topological sort puts all upstreams before the fan-in."""

        class A(BaseModel, frozen=True):
            value: str

        class B(BaseModel, frozen=True):
            value: str

        class C(BaseModel, frozen=True):
            value: str

        class Report(BaseModel, frozen=True):
            summary: str

        mod = self._fresh_module("test_fan_in_mod")

        @node(mode="scripted", outputs=A)
        def alpha() -> A:
            return A(value="a")

        @node(mode="scripted", outputs=B)
        def beta() -> B:
            return B(value="b")

        @node(mode="scripted", outputs=C)
        def gamma() -> C:
            return C(value="c")

        @node(mode="scripted", outputs=Report)
        def report(alpha: A, beta: B, gamma: C) -> Report:
            return Report(summary=f"{alpha.value}-{beta.value}-{gamma.value}")

        mod.alpha = alpha
        mod.beta = beta
        mod.gamma = gamma
        mod.report = report

        pipeline = construct_from_module(mod)
        names = [n.name for n in pipeline.nodes]

        # All three upstreams appear before the fan-in consumer.
        assert set(names[:3]) == {"alpha", "beta", "gamma"}
        assert names[-1] == "report"

        # Register the scripted fns and run end-to-end.
        from neograph import compile, run

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fan-in"})
        assert result["report"].summary == "a-b-c"

    def test_explicit_outputs_mismatch_raises(self):
        """@node(outputs=X) with -> Y (X != Y) raises ConstructError (neograph-pcdp)."""

        class Bogus(BaseModel, frozen=True):
            nope: str

        with pytest.raises(ConstructError, match="outputs=.*differs from return annotation"):
            @node(mode="scripted", outputs=Claims)
            def producer() -> Bogus:
                return Claims(items=["overridden"])

    def test_construct_raises_when_param_names_unknown_node(self):
        """A parameter that doesn't name any @node in the module raises
        ConstructError with a helpful message."""
        from neograph import ConstructError

        mod = self._fresh_module("test_unknown_param_mod")

        @node(mode="scripted", outputs=Claims)
        def orphan(ghost: RawText) -> Claims:
            return Claims(items=["x"])

        mod.orphan = orphan

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "ghost" in msg
        assert "orphan" in msg

    def test_topo_sort_orders_correctly_when_declared_out_of_order(self):
        """Out-of-declaration-order dependencies get sorted correctly."""

        mod = self._fresh_module("test_topo_mod")

        # Attach to module in a SHUFFLED order (report, seed, split).
        # Declaration order inside the function body is also shuffled: the
        # downstream-most node is declared first.

        @node(mode="scripted", outputs=ClassifiedClaims)
        def report(split: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[{"claim": c, "category": "x"} for c in split.items],
            )

        @node(mode="scripted", outputs=RawText)
        def seed() -> RawText:
            return RawText(text="a b c")

        @node(mode="scripted", outputs=Claims)
        def split(seed: RawText) -> Claims:
            return Claims(items=seed.text.split())

        # Assign in a different order from their dependency DAG.
        mod.report = report
        mod.seed = seed
        mod.split = split

        pipeline = construct_from_module(mod)
        names = [n.name for n in pipeline.nodes]
        assert names == ["seed", "split", "report"]

        from neograph import compile, run

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "topo"})
        assert [c["claim"] for c in result["report"].classified] == ["a", "b", "c"]

    def test_node_name_hyphenated_when_function_uses_underscores(self):
        """Function `make_clusters` becomes node 'make-clusters'; downstream
        parameter `make_clusters` resolves to it."""
        from neograph import compile, run

        mod = self._fresh_module("test_name_convention_mod")

        @node(mode="scripted", outputs=Claims)
        def seed_text() -> Claims:
            return Claims(items=["one", "two"])

        @node(mode="scripted", outputs=Clusters)
        def make_clusters(seed_text: Claims) -> Clusters:
            return Clusters(
                groups=[ClusterGroup(label="g", claim_ids=list(seed_text.items))],
            )

        @node(mode="scripted", outputs=ClassifiedClaims)
        def summarize(make_clusters: Clusters) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[
                    {"claim": cid, "category": g.label}
                    for g in make_clusters.groups
                    for cid in g.claim_ids
                ],
            )

        mod.seed_text = seed_text
        mod.make_clusters = make_clusters
        mod.summarize = summarize

        # Node names are hyphenated.
        assert make_clusters.name == "make-clusters"
        assert seed_text.name == "seed-text"

        pipeline = construct_from_module(mod)
        assert [n.name for n in pipeline.nodes] == [
            "seed-text",
            "make-clusters",
            "summarize",
        ]

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "name-conv"})
        # Output field uses underscore form of the node name.
        classified = result["summarize"].classified
        assert len(classified) == 2
        assert classified[0]["category"] == "g"





class TestNodeDecoratorModeInference:
    """@node mode inference: mode=None infers from prompt/model presence."""

    def test_mode_infers_scripted_when_no_prompt_or_model(self):
        """@node(outputs=X) with no prompt/model infers mode='scripted'."""

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="hello")

        assert seed.mode == "scripted"

    def test_mode_infers_produce_when_prompt_and_model_present(self):
        """@node(outputs=X, prompt='...', model='...') infers mode='produce'."""

        @node(outputs=Claims, prompt="rw/decompose", model="reason")
        def decompose(topic: RawText) -> Claims: ...

        assert decompose.mode == "think"

    def test_decoration_raises_when_produce_mode_missing_prompt(self):
        """@node(mode='produce', outputs=X, model='reason') with no prompt raises at decoration time."""
        from neograph import ConstructError

        with pytest.raises(ConstructError, match="requires prompt="):

            @node(mode="think", outputs=Claims, model="reason")
            def decompose(topic: RawText) -> Claims: ...

    def test_decoration_raises_when_gather_mode_missing_model(self):
        """@node(mode='gather', outputs=X, prompt='...') with no model raises at decoration time."""
        from neograph import ConstructError

        with pytest.raises(ConstructError, match="requires model="):

            @node(mode="agent", outputs=Claims, prompt="rw/decompose")
            def decompose(topic: RawText) -> Claims: ...

    def test_warning_emitted_when_produce_mode_has_nontrivial_body(self):
        """@node(mode='produce', ...) with a real function body emits UserWarning."""


        with pytest.warns(UserWarning, match="body.*not executed"):

            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason")
            def decompose(topic: RawText) -> Claims:
                return Claims(items=topic.text.split("."))

    def test_no_warning_when_produce_mode_has_ellipsis_body(self):
        """@node(mode='produce', ...) with `...` body does NOT warn."""
        import warnings as _warnings


        with _warnings.catch_warnings():
            _warnings.simplefilter("error")

            @node(mode="think", outputs=Claims, prompt="rw/decompose", model="reason")
            def decompose(topic: RawText) -> Claims: ...





# ═══════════════════════════════════════════════════════════════════════════
# TEST: @node scalar parameters — FromInput, FromConfig, default constants
#
# Not every @node parameter must name an upstream @node. Three additional
# parameter resolution mechanisms:
#   1. Annotated[T, FromInput]  — value from run(input={param: ...})
#   2. Annotated[T, FromConfig] — value from config["configurable"][param]
#   3. default value  — compile-time constant, no upstream needed
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TEST: @node scalar parameters — FromInput, FromConfig, default constants
#
# Not every @node parameter must name an upstream @node. Three additional
# parameter resolution mechanisms:
#   1. Annotated[T, FromInput]  — value from run(input={param: ...})
#   2. Annotated[T, FromConfig] — value from config["configurable"][param]
#   3. default value  — compile-time constant, no upstream needed
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecoratorParams:
    """Scalar parameter support: FromInput, FromConfig, default constants."""

    def test_from_input_delivers_value_when_present_in_run_input(self):
        """Annotated[str, FromInput] param is delivered via run(input={'topic': 'x'})."""
        import types as _types

        from neograph import FromInput, compile, run

        mod = _types.ModuleType("test_from_input_mod")

        @node(mode="scripted", outputs=RawText)
        def greet(topic: Annotated[str, FromInput]) -> RawText:
            return RawText(text=f"Hello, {topic}!")

        mod.greet = greet

        pipeline = construct_from_module(mod, name="test-from-input")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t-001", "topic": "world"})

        assert result["greet"] == RawText(text="Hello, world!")

    def test_from_config_delivers_resource_when_present_in_configurable(self):
        """Annotated[RateLimiter, FromConfig] param is delivered via config['configurable']."""
        import types as _types

        from neograph import FromConfig, compile, run

        mod = _types.ModuleType("test_from_config_mod")

        class FakeRateLimiter:
            def __init__(self):
                self.calls = 0

            def call(self):
                self.calls += 1

        limiter = FakeRateLimiter()

        @node(mode="scripted", outputs=Claims)
        def process(rate_limiter: Annotated[FakeRateLimiter, FromConfig]) -> Claims:
            rate_limiter.call()
            return Claims(items=[f"calls={rate_limiter.calls}"])

        mod.process = process

        pipeline = construct_from_module(mod, name="test-from-config")
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "t-002"},
            config={"configurable": {"rate_limiter": limiter}},
        )

        assert limiter.calls == 1
        assert result["process"] == Claims(items=["calls=1"])

    def test_default_used_when_param_has_default_value(self):
        """Param with default value not matching any @node is used as compile-time constant."""
        import types as _types

        from neograph import compile, run

        mod = _types.ModuleType("test_default_const_mod")

        @node(mode="scripted", outputs=RawText)
        def greet(greeting: str = "Hi") -> RawText:
            return RawText(text=f"{greeting}, friend!")

        mod.greet = greet

        pipeline = construct_from_module(mod, name="test-default-const")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t-003"})

        assert result["greet"] == RawText(text="Hi, friend!")

    def test_all_param_types_resolve_when_mixed_in_one_function(self):
        """One function with upstream + FromInput + FromConfig + default."""
        import types as _types

        from neograph import FromConfig, FromInput, compile, run

        mod = _types.ModuleType("test_mixed_mod")

        class FakeLogger:
            def __init__(self):
                self.logged: list[str] = []

            def log(self, msg: str):
                self.logged.append(msg)

        logger = FakeLogger()

        @node(mode="scripted", outputs=RawText)
        def seed() -> RawText:
            return RawText(text="base")

        @node(mode="scripted", outputs=Claims)
        def combine(
            seed: RawText,
            topic: Annotated[str, FromInput],
            logger: Annotated[FakeLogger, FromConfig],
            separator: str = " | ",
        ) -> Claims:
            logger.log(f"combining {seed.text} with {topic}")
            return Claims(items=[f"{seed.text}{separator}{topic}"])

        mod.seed = seed
        mod.combine = combine

        pipeline = construct_from_module(mod, name="test-mixed")
        graph = compile(pipeline)
        result = run(
            graph,
            input={"node_id": "t-004", "topic": "science"},
            config={"configurable": {"logger": logger}},
        )

        assert result["combine"] == Claims(items=["base | science"])
        assert len(logger.logged) == 1
        assert "combining base with science" in logger.logged[0]

    def test_none_returned_when_from_input_key_missing_optional(self):
        """FromInput(required=False) param not in run(input=...) returns None."""
        import types as _types

        from neograph import FromInput, compile, run

        mod = _types.ModuleType("test_from_input_missing_mod")

        @node(mode="scripted", outputs=RawText)
        def greet(topic: Annotated[str, FromInput(required=False)]) -> RawText:
            if topic is None:
                return RawText(text="no topic")
            return RawText(text=f"Hello, {topic}!")

        mod.greet = greet

        pipeline = construct_from_module(mod, name="test-from-input-missing")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t-005"})

        assert result["greet"] == RawText(text="no topic")





class TestNodeDecoratorErrorLocation:
    """@node errors include the decorated function's source file:line."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_error_message_includes_source_location_when_param_unknown(self):
        """Unknown-param error includes 'test_basics.py:<line>'
        pointing at the decorated function definition."""
        from neograph import ConstructError

        mod = self._fresh_module("test_src_loc_mod")

        @node(mode="scripted", outputs=Claims)
        def orphan(ghost: RawText) -> Claims:
            return Claims(items=["x"])

        mod.orphan = orphan

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "test_basics.py:" in msg

    def test_error_message_includes_source_location_when_cycle_detected(self):
        """Cycle error includes source locations for the involved nodes."""
        from neograph import ConstructError

        mod = self._fresh_module("test_cycle_loc_mod")

        @node(mode="scripted", outputs=RawText)
        def ping(pong: Claims) -> RawText:
            return RawText(text="p")

        @node(mode="scripted", outputs=Claims)
        def pong(ping: RawText) -> Claims:
            return Claims(items=["q"])

        mod.ping = ping
        mod.pong = pong

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "test_basics.py:" in msg

    def test_source_location_uses_basename_when_reporting_errors(self):
        """Source location uses basename, not the full absolute path."""
        from neograph import ConstructError

        mod = self._fresh_module("test_basename_mod")

        @node(mode="scripted", outputs=Claims)
        def orphan(ghost: RawText) -> Claims:
            return Claims(items=["x"])

        mod.orphan = orphan

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        # Must contain basename, not an absolute path with directory separators
        assert "test_basics.py:" in msg
        assert "/tests/test_basics.py:" not in msg





# ═══════════════════════════════════════════════════════════════════════════
# @node(mode='raw') — LangGraph escape hatch via unified @node decorator
#
# Raw mode folds @raw_node into @node: the user writes a classic
# (state, config) -> state_update function, and @node wires edges +
# observability. No parameter-name topology — the function body manages
# its own state access.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# @node(mode='raw') — LangGraph escape hatch via unified @node decorator
#
# Raw mode folds @raw_node into @node: the user writes a classic
# (state, config) -> state_update function, and @node wires edges +
# observability. No parameter-name topology — the function body manages
# its own state access.
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecoratorRawMode:
    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_raw_mode_filters_state_when_reading_and_writing(self):
        """@node(mode='raw') reads state and returns a filtered update dict."""
        from neograph import compile, run

        register_scripted(
            "make_claims",
            lambda input_data, config: Claims(items=["a", "b", "c"]),
        )

        mod = self._fresh_module("test_raw_mode_basic")

        make = Node.scripted("make-claims", fn="make_claims", outputs=Claims)

        @node(mode="raw", inputs=Claims, outputs=Claims)
        def filter_claims(state, config):
            claims = None
            for field_name in state.__class__.model_fields:
                val = getattr(state, field_name, None)
                if isinstance(val, Claims):
                    claims = val
                    break
            if claims is None:
                return {"filter_claims": Claims(items=[])}
            filtered = Claims(items=[c for c in claims.items if c != "b"])
            return {"filter_claims": filtered}

        pipeline = Construct("test-raw-mode", nodes=[make, filter_claims])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        filtered = result.get("filter_claims")
        assert isinstance(filtered, Claims)
        assert "b" not in filtered.items
        assert "a" in filtered.items
        assert "c" in filtered.items

    def test_raw_mode_raises_when_signature_invalid(self):
        """@node(mode='raw') rejects functions with wrong parameter count or names."""

        # Three parameters — too many
        with pytest.raises(ConstructError, match="exactly two parameters"):
            @node(mode="raw", inputs=Claims, outputs=Claims)
            def bad_three(state, config, extra):
                pass

        # Wrong parameter names
        with pytest.raises(ConstructError, match="named 'state' and 'config'"):
            @node(mode="raw", inputs=Claims, outputs=Claims)
            def bad_names(s, c):
                pass

        # One parameter — too few
        with pytest.raises(ConstructError, match="exactly two parameters"):
            @node(mode="raw", inputs=Claims, outputs=Claims)
            def bad_one(state):
                pass

    def test_downstream_consumes_when_raw_node_produces_output(self):
        """Raw node output is consumed by a downstream scripted @node via param name."""
        from neograph import compile, run

        mod = self._fresh_module("test_raw_downstream")

        @node(mode="raw", inputs=Claims, outputs=Claims)
        def produce_claims(state, config):
            return {"produce_claims": Claims(items=["x", "y"])}

        @node(mode="scripted", outputs=RawText)
        def summarize(produce_claims: Claims) -> RawText:
            return RawText(text=f"count={len(produce_claims.items)}")

        mod.produce_claims = produce_claims
        mod.summarize = summarize

        pipeline = construct_from_module(mod, name="test-raw-downstream")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        summary = result.get("summarize")
        assert isinstance(summary, RawText)
        assert summary.text == "count=2"

    def test_pipeline_runs_when_raw_and_scripted_mixed(self):
        """Pipeline with both raw and scripted @nodes in the same module."""
        from neograph import compile, run

        mod = self._fresh_module("test_mixed_raw_scripted")

        @node(mode="scripted", outputs=RawText)
        def extract() -> RawText:
            return RawText(text="hello world")

        @node(mode="raw", inputs=RawText, outputs=Claims)
        def process(state, config):
            return {"process": Claims(items=["from-raw"])}

        @node(mode="scripted", outputs=ClassifiedClaims)
        def classify(process: Claims) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[{"claim": c, "category": "raw"} for c in process.items]
            )

        mod.extract = extract
        mod.process = process
        mod.classify = classify

        pipeline = construct_from_module(mod, name="test-mixed")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        classified = result.get("classify")
        assert isinstance(classified, ClassifiedClaims)
        assert len(classified.classified) == 1
        assert classified.classified[0]["claim"] == "from-raw"


# ═══════════════════════════════════════════════════════════════════════════
# @node interrupt_when — Operator human-in-loop via @node decorator
#
# The interrupt_when= kwarg on @node composes the node with Operator(when=...).
# String form uses a pre-registered condition name; callable form auto-registers.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# @node interrupt_when — Operator human-in-loop via @node decorator
#
# The interrupt_when= kwarg on @node composes the node with Operator(when=...).
# String form uses a pre-registered condition name; callable form auto-registers.
# ═══════════════════════════════════════════════════════════════════════════

