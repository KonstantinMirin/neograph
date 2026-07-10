"""ForwardConstruct tests — base class, tracer, compilation, branching, loops.

Merged from test_e2e_piarch_ready.py (3 classes) and
test_forward_construct_branching.py (8 classes).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, ConstructError, Each, ForwardConstruct, Node, compile, run
from tests.fakes import StructuredFake, build_test_compile_kwargs, configure_fake_llm, register_scripted
from tests.schemas import Claims, ClassifiedClaims, Clusters, MatchResult, RawText


class Confidence(BaseModel, frozen=True):
    score: float


class HighResult(BaseModel, frozen=True):
    label: str


class LowResult(BaseModel, frozen=True):
    label: str


class FinalResult(BaseModel, frozen=True):
    summary: str


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: _Proxy attribute access + comparison dunders
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# BASE / TRACER / COMPILE (from monolith)
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardConstructBase:
    """Task neograph-di0: ForwardConstruct base class and node discovery."""

    def test_nodes_discovered_when_class_has_node_attributes(self):
        """Class with 3 Node attrs — all discovered in declaration order."""
        from neograph import ForwardConstruct, Node

        class Triple(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)
            b = Node.scripted("b", fn="b_fn", outputs=Claims)
            c = Node.scripted("c", fn="c_fn", outputs=ClassifiedClaims)

            def forward(self, topic):
                x = self.a(topic)
                y = self.b(x)
                return self.c(y)

        discovered = Triple._discover_node_attrs()
        assert list(discovered.keys()) == ["a", "b", "c"]
        assert discovered["a"] is Triple.a
        assert discovered["b"] is Triple.b
        assert discovered["c"] is Triple.c

    def test_isinstance_true_when_checking_construct_subclass(self):
        """isinstance(pipeline, Construct) is True."""
        from neograph import Construct, ForwardConstruct, Node

        class Simple(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)

            def forward(self, topic):
                return self.a(topic)

        pipeline = Simple()
        assert isinstance(pipeline, Construct)
        assert isinstance(pipeline, ForwardConstruct)

    def test_init_raises_when_forward_method_missing(self):
        """Subclass without forward() method raises ConstructError."""
        from neograph import ForwardConstruct, Node

        class NoForward(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)

        with pytest.raises(ConstructError, match="must override forward"):
            NoForward()


# ═══════════════════════════════════════════════════════════════════════════
# ForwardConstruct — Task 2: symbolic proxy tracer
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardConstructTracer:
    """Task neograph-3us: symbolic proxy tracer for straight-line forward()."""

    def test_trace_matches_sequence_when_two_nodes_called(self):
        """Two nodes called in sequence — traced order matches."""
        from neograph import ForwardConstruct, Node

        class TwoStep(ForwardConstruct):
            extract = Node.scripted("extract", fn="extract_fn", outputs=RawText)
            classify = Node.scripted("classify", fn="classify_fn", outputs=Claims)

            def forward(self, topic):
                raw = self.extract(topic)
                return self.classify(raw)

        pipeline = TwoStep()
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[0].name == "extract"
        assert pipeline.nodes[1].name == "classify"

    def test_trace_matches_sequence_when_three_nodes_chained(self):
        """A -> B -> C traced as [A, B, C]."""
        from neograph import ForwardConstruct, Node

        class ThreeChain(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)
            b = Node.scripted("b", fn="b_fn", outputs=Claims)
            c = Node.scripted("c", fn="c_fn", outputs=ClassifiedClaims)

            def forward(self, topic):
                x = self.a(topic)
                y = self.b(x)
                return self.c(y)

        pipeline = ThreeChain()
        assert [n.name for n in pipeline.nodes] == ["a", "b", "c"]

    def test_traced_node_is_same_object_when_class_attr_checked(self):
        """traced_nodes[0] is MyPipeline.extract (same object)."""
        from neograph import ForwardConstruct, Node

        class Identity(ForwardConstruct):
            extract = Node.scripted("extract", fn="extract_fn", outputs=RawText)
            classify = Node.scripted("classify", fn="classify_fn", outputs=Claims)

            def forward(self, topic):
                raw = self.extract(topic)
                return self.classify(raw)

        pipeline = Identity()
        assert pipeline.nodes[0] is Identity.extract
        assert pipeline.nodes[1] is Identity.classify

    def test_trace_has_two_when_third_node_unused(self):
        """Class has 3 nodes, forward() only calls 2 — trace has 2."""
        from neograph import ForwardConstruct, Node

        class PartialUse(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)
            b = Node.scripted("b", fn="b_fn", outputs=Claims)
            unused = Node.scripted("unused", fn="unused_fn", outputs=ClassifiedClaims)

            def forward(self, topic):
                x = self.a(topic)
                return self.b(x)

        pipeline = PartialUse()
        assert len(pipeline.nodes) == 2
        assert [n.name for n in pipeline.nodes] == ["a", "b"]


# ═══════════════════════════════════════════════════════════════════════════
# ForwardConstruct — Task 3: compile() integration
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardConstructCompile:
    """Task neograph-vxv: compile() integration for traced ForwardConstruct."""

    def test_pipeline_produces_output_when_compiled_and_run(self):
        """Full end-to-end: ForwardConstruct with 2 scripted nodes, compile, run."""
        from neograph import ForwardConstruct, Node, compile, run
        from tests.fakes import register_scripted

        register_scripted("fc_extract", lambda input_data, config: RawText(text="hello world"))
        register_scripted("fc_split", lambda input_data, config: Claims(items=["claim-1", "claim-2"]))

        class ScriptedPipeline(ForwardConstruct):
            extract = Node.scripted("fc-extract", fn="fc_extract", outputs=RawText)
            split = Node.scripted("fc-split", fn="fc_split", outputs=Claims)

            def forward(self, topic):
                raw = self.extract(topic)
                return self.split(raw)

        pipeline = ScriptedPipeline()
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "fc-test-001"})

        assert isinstance(result["fc_split"], Claims)
        assert result["fc_split"].items == ["claim-1", "claim-2"]
        assert isinstance(result["fc_extract"], RawText)
        assert result["fc_extract"].text == "hello world"

    def test_produce_node_runs_when_forward_construct_with_fake_llm(self):
        """ForwardConstruct with a produce node + FakeLLM."""
        from neograph import ForwardConstruct, Node, compile, run
        from tests.fakes import register_scripted

        register_scripted("fc_prep", lambda input_data, config: RawText(text="topic"))

        fake = StructuredFake(lambda model: model(items=["classified-a", "classified-b"]))
        _llm_kw = configure_fake_llm(lambda tier: fake)

        class ProducePipeline(ForwardConstruct):
            prep = Node.scripted("fc-prep", fn="fc_prep", outputs=RawText)
            classify = Node("fc-classify", mode="think", outputs=Claims, prompt="rw/classify", model="fast")

            def forward(self, topic):
                raw = self.prep(topic)
                return self.classify(raw)

        pipeline = ProducePipeline()
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "fc-test-002"})

        assert isinstance(result["fc_classify"], Claims)
        assert result["fc_classify"].items == ["classified-a", "classified-b"]

    def test_output_identical_when_compared_to_declarative_construct(self):
        """Same pipeline as ForwardConstruct and Construct(nodes=[...]) — identical output."""
        from neograph import Construct, ForwardConstruct, Node, compile, run
        from tests.fakes import register_scripted

        register_scripted("fc_equiv_a", lambda input_data, config: RawText(text="extracted"))
        register_scripted("fc_equiv_b", lambda input_data, config: Claims(items=["x", "y"]))

        node_a = Node.scripted("fc-equiv-a", fn="fc_equiv_a", outputs=RawText)
        node_b = Node.scripted("fc-equiv-b", fn="fc_equiv_b", outputs=Claims)

        # Declarative
        declarative = Construct("fc-equiv-test", nodes=[node_a, node_b])
        graph_decl = compile(declarative, **build_test_compile_kwargs())
        result_decl = run(graph_decl, input={"node_id": "equiv-001"})

        # ForwardConstruct
        class ForwardPipeline(ForwardConstruct):
            a = node_a
            b = node_b

            def forward(self, topic):
                raw = self.a(topic)
                return self.b(raw)

        forward_pipe = ForwardPipeline()
        graph_fwd = compile(forward_pipe, **build_test_compile_kwargs())
        result_fwd = run(graph_fwd, input={"node_id": "equiv-001"})

        # Both produce identical output
        assert result_decl["fc_equiv_a"] == result_fwd["fc_equiv_a"]
        assert result_decl["fc_equiv_b"] == result_fwd["fc_equiv_b"]
        assert isinstance(result_fwd["fc_equiv_a"], RawText)
        assert isinstance(result_fwd["fc_equiv_b"], Claims)


# ═══════════════════════════════════════════════════════════════════════════
# BRANCHING / LOOPS (from test_forward_construct_branching.py)
# ═══════════════════════════════════════════════════════════════════════════


class TestProxyAttributeAccess:
    """Proxy gains __getattr__ for dotted access and comparison operators."""

    def test_child_proxy_created_when_attribute_accessed(self):
        """classified.confidence returns a child proxy with dotted name."""
        from neograph.forward import _Proxy

        parent = _Proxy(source_node=None, name="classified")
        child = parent.confidence
        assert child._neo_name == "classified.confidence"
        assert child._neo_source is parent._neo_source

    def test_dotted_name_correct_when_attributes_chained(self):
        """classified.items.severity chains produce correct names."""
        from neograph.forward import _Proxy

        parent = _Proxy(source_node=None, name="result")
        child = parent.items.severity
        assert child._neo_name == "result.items.severity"

    def test_attribute_error_raised_when_neo_prefix_accessed(self):
        """Accessing _neo_* attributes raises AttributeError, not child proxy."""
        from neograph.forward import _Proxy

        proxy = _Proxy(source_node=None, name="x")
        with pytest.raises(AttributeError):
            _ = proxy._neo_nonexistent

    def test_condition_proxy_returned_when_less_than_used(self):
        """proxy < 0.7 returns a _ConditionProxy, not a plain bool."""
        from neograph.forward import _ConditionProxy, _Proxy

        proxy = _Proxy(source_node=None, name="score")
        result = proxy < 0.7
        assert isinstance(result, _ConditionProxy)
        assert result._op == "<"
        assert result._right == 0.7

    def test_condition_proxy_created_when_any_comparison_used(self):
        """All 6 comparison operators produce correct _ConditionProxy."""
        from neograph.forward import _ConditionProxy, _Proxy

        proxy = _Proxy(source_node=None, name="val")
        ops = {
            "<": proxy < 1,
            "<=": proxy <= 2,
            ">": proxy > 3,
            ">=": proxy >= 4,
            "==": proxy == 5,
            "!=": proxy != 6,
        }
        for op_str, cond in ops.items():
            assert isinstance(cond, _ConditionProxy), f"Failed for {op_str}"
            assert cond._op == op_str

    def test_type_error_raised_when_proxy_used_as_bool_outside_tracing(self):
        """Using proxy in boolean context outside tracing raises TypeError."""
        from neograph.forward import _Proxy

        proxy = _Proxy(source_node=None, name="x")
        with pytest.raises(TypeError, match="boolean context"):
            bool(proxy)

    def test_type_error_raised_when_condition_used_as_bool_outside_tracing(self):
        """Using condition in boolean context outside tracing raises TypeError."""
        from neograph.forward import _Proxy

        proxy = _Proxy(source_node=None, name="x")
        cond = proxy < 0.5
        with pytest.raises(TypeError, match="boolean context"):
            bool(cond)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Tracer branch recording
# ═══════════════════════════════════════════════════════════════════════════


class TestTracerBranchRecording:
    """_Tracer records branch decisions during forward() tracing."""

    def test_branch_recorded_when_condition_evaluated(self):
        """record_branch stores the condition and returns a bool."""
        from neograph.forward import _ConditionProxy, _Proxy, _Tracer

        tracer = _Tracer()
        proxy = _Proxy(source_node=None, name="score", tracer=tracer)
        cond = _ConditionProxy(proxy, "<", 0.7)
        cond._neo_tracer = tracer

        result = tracer.record_branch(cond)
        assert result is True  # default: take true arm
        assert len(tracer.branches) == 1
        assert tracer.branches[0].branch_id == 0

    def test_false_returned_when_branch_decision_overridden(self):
        """Pre-configured branch decisions override the default True."""
        from neograph.forward import _ConditionProxy, _Proxy, _Tracer

        tracer = _Tracer(branch_decisions={0: False})
        proxy = _Proxy(source_node=None, name="score", tracer=tracer)
        cond = _ConditionProxy(proxy, "<", 0.7)
        cond._neo_tracer = tracer

        result = tracer.record_branch(cond)
        assert result is False

    def test_tracer_records_when_condition_bool_called(self):
        """if proxy.score < 0.7 calls tracer.record_branch via __bool__."""
        from neograph.forward import _Proxy, _Tracer

        tracer = _Tracer()
        proxy = _Proxy(source_node=None, name="score", tracer=tracer)
        cond = proxy < 0.7  # returns _ConditionProxy with tracer
        # Simulating `if cond:` — which calls __bool__
        result = bool(cond)
        assert isinstance(result, bool)
        assert len(tracer.branches) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Simple if-branch tracing (re-trace strategy)
# ═══════════════════════════════════════════════════════════════════════════


class TestIfBranchSimple:
    """Single if/else in forward() produces branch metadata for the compiler."""

    def _make_branching_pipeline(self):
        """Helper: pipeline with if self.check(x).passed: path_a else path_b."""

        register_scripted(
            "br_check",
            lambda input_data, config: Confidence(score=0.9),
        )
        register_scripted(
            "br_high",
            lambda input_data, config: HighResult(label="high-confidence"),
        )
        register_scripted(
            "br_low",
            lambda input_data, config: LowResult(label="low-confidence"),
        )

        class BranchPipeline(ForwardConstruct):
            check = Node.scripted("br-check", fn="br_check", outputs=Confidence)
            high_path = Node.scripted("br-high", fn="br_high", outputs=HighResult)
            low_path = Node.scripted("br-low", fn="br_low", outputs=LowResult)

            def forward(self, topic):
                result = self.check(topic)
                if result.score > 0.5:
                    return self.high_path(result)
                else:
                    return self.low_path(result)

        return BranchPipeline

    def test_pipeline_instantiates_when_forward_has_if_else(self):
        """Pipeline with if/else in forward() can be instantiated."""
        Pipeline = self._make_branching_pipeline()
        pipeline = Pipeline()
        # Should have nodes: check + branch metadata
        assert any(n.name == "br-check" for n in pipeline.nodes)

    def test_graph_compiles_when_forward_has_if_else(self):
        """Pipeline with if/else compiles to a LangGraph graph."""
        Pipeline = self._make_branching_pipeline()
        pipeline = Pipeline()
        compile(pipeline, **build_test_compile_kwargs())  # succeeds = test passes; raises = test fails

    def test_high_path_runs_when_condition_true(self):
        """When condition is true at runtime, high_path runs."""
        # Override to produce score > 0.5
        register_scripted(
            "br_check",
            lambda input_data, config: Confidence(score=0.9),
        )
        Pipeline = self._make_branching_pipeline()
        pipeline = Pipeline()
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "br-true-test"})
        assert "br_high" in result
        assert result["br_high"].label == "high-confidence"

    def test_low_path_runs_when_condition_false(self):
        """When condition is false at runtime, low_path runs."""
        Pipeline = self._make_branching_pipeline()
        pipeline = Pipeline()
        # Override AFTER pipeline instantiation so compile-time tracing
        # used the default, but the runtime scripted fn returns low score
        register_scripted(
            "br_check",
            lambda input_data, config: Confidence(score=0.2),
        )
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "br-false-test"})
        assert "br_low" in result
        assert result["br_low"].label == "low-confidence"


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Sequential if-blocks (two branches)
# ═══════════════════════════════════════════════════════════════════════════


class TestSequentialBranches:
    """Two sequential if-blocks in forward()."""

    def test_correct_paths_run_when_two_independent_branches(self):
        """Pipeline with two independent if-blocks compiles and runs."""
        register_scripted(
            "seq_check1",
            lambda input_data, config: Confidence(score=0.9),
        )
        register_scripted(
            "seq_check2",
            lambda input_data, config: Confidence(score=0.3),
        )
        register_scripted(
            "seq_a",
            lambda input_data, config: HighResult(label="path-a"),
        )
        register_scripted(
            "seq_b",
            lambda input_data, config: LowResult(label="path-b"),
        )
        register_scripted(
            "seq_c",
            lambda input_data, config: HighResult(label="path-c"),
        )
        register_scripted(
            "seq_d",
            lambda input_data, config: LowResult(label="path-d"),
        )

        class TwoBranch(ForwardConstruct):
            check1 = Node.scripted("seq-check1", fn="seq_check1", outputs=Confidence)
            check2 = Node.scripted("seq-check2", fn="seq_check2", outputs=Confidence)
            a = Node.scripted("seq-a", fn="seq_a", outputs=HighResult)
            b = Node.scripted("seq-b", fn="seq_b", outputs=LowResult)
            c = Node.scripted("seq-c", fn="seq_c", outputs=HighResult)
            d = Node.scripted("seq-d", fn="seq_d", outputs=LowResult)

            def forward(self, topic):
                r1 = self.check1(topic)
                if r1.score > 0.5:
                    out1 = self.a(r1)
                else:
                    out1 = self.b(r1)

                r2 = self.check2(out1)
                if r2.score > 0.5:
                    return self.c(r2)
                else:
                    return self.d(r2)

        pipeline = TwoBranch()
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "seq-test"})

        # check1 returns 0.9 (> 0.5) → seq-a, check2 returns 0.3 (<= 0.5) → seq-d
        assert "seq_a" in result
        assert "seq_d" in result
        assert result["seq_a"].label == "path-a"
        assert result["seq_d"].label == "path-d"


# ═══════════════════════════════════════════════════════════════════════════
# Part 5: Branch count limit
# ═══════════════════════════════════════════════════════════════════════════


class TestBranchCountLimit:
    """More than 8 branches raises a clear error."""

    def test_tracer_raises_when_nine_branches_recorded(self):
        """9 branches in forward() raises an error."""
        # We can't easily create 9 real branches in a single forward(),
        # but we can test that the tracer enforces the limit.
        from neograph.forward import _Tracer

        tracer = _Tracer()
        # Simulate 9 branch recordings
        from neograph.forward import _ConditionProxy, _Proxy

        proxy = _Proxy(source_node=None, name="val", tracer=tracer)
        for i in range(8):
            cond = _ConditionProxy(proxy, "<", i)
            cond._neo_tracer = tracer
            tracer.record_branch(cond)

        # 9th branch should raise
        cond = _ConditionProxy(proxy, "<", 99)
        cond._neo_tracer = tracer
        with pytest.raises(ValueError, match="[Bb]ranch.*limit|[Tt]oo many branch"):
            tracer.record_branch(cond)


# ═══════════════════════════════════════════════════════════════════════════
# Part 6: Proxy attribute chains for condition building
# ═══════════════════════════════════════════════════════════════════════════


class TestProxyAttributeChains:
    """Attribute chains on proxies produce correct condition metadata."""

    def test_full_path_captured_when_dotted_proxy_compared(self):
        """result.score > 0.5 captures the full attribute path."""
        from neograph.forward import _ConditionProxy, _Proxy

        proxy = _Proxy(source_node=None, name="out_of_check")
        cond = proxy.score > 0.5
        assert isinstance(cond, _ConditionProxy)
        assert cond._left._neo_name == "out_of_check.score"
        assert cond._op == ">"
        assert cond._right == 0.5

    def test_deep_path_captured_when_multi_level_chain_compared(self):
        """result.items.first.severity < 3 captures deep chain."""
        from neograph.forward import _Proxy

        proxy = _Proxy(source_node=None, name="out_of_classify")
        cond = proxy.items.first.severity < 3
        assert cond._left._neo_name == "out_of_classify.items.first.severity"
        assert cond._op == "<"
        assert cond._right == 3


# ═══════════════════════════════════════════════════════════════════════════
# Part 7: For-loop support (Proxy.__iter__ → Each fan-out)
# ═══════════════════════════════════════════════════════════════════════════


class Report(BaseModel, frozen=True):
    summary: str


class TestForwardConstructLoops:
    """For-loop over a proxy attribute lowers to Each fan-out modifier."""

    def test_each_traced_when_for_loop_over_proxy_attr(self):
        """for item in clusters.groups: self.verify(item) traces verify with Each."""
        register_scripted(
            "loop_make",
            lambda input_data, config: Clusters(groups=["a", "b"]),
        )
        register_scripted(
            "loop_verify",
            lambda input_data, config: MatchResult(label="ok"),
        )
        register_scripted(
            "loop_report",
            lambda input_data, config: Report(summary="done"),
        )

        class LoopPipeline(ForwardConstruct):
            make = Node.scripted("loop-make", fn="loop_make", outputs=Clusters)
            verify = Node.scripted("loop-verify", fn="loop_verify", outputs=MatchResult)
            report = Node.scripted("loop-report", fn="loop_report", outputs=Report)

            def forward(self, topic):
                clusters = self.make(topic)
                for cluster in clusters.groups:
                    self.verify(cluster)
                return self.report(clusters)

        pipeline = LoopPipeline()
        verify_node = next(n for n in pipeline.nodes if n.name == "loop-verify")
        assert verify_node.has_modifier(Each), "for-loop should trace to Each modifier"
        compile(pipeline, **build_test_compile_kwargs())  # also confirm it compiles cleanly

    def test_step_deduped_when_for_loop_over_range(self):
        """for i in range(3): self.step(x) traces step once (dedup). No Each."""
        register_scripted(
            "range_step",
            lambda input_data, config: RawText(text="stepped"),
        )

        class RangePipeline(ForwardConstruct):
            step = Node.scripted("range-step", fn="range_step", outputs=RawText)

            def forward(self, topic):
                for _ in range(3):
                    self.step(topic)
                return topic

        pipeline = RangePipeline()
        # step should appear once (dedup), and should NOT have Each modifier
        step_node = [n for n in pipeline.nodes if isinstance(n, Node) and n.name == "range-step"]
        assert len(step_node) == 1
        assert not step_node[0].has_modifier(Each)

    def test_each_modifier_present_when_for_loop_over_proxy(self):
        """Loop-body node has Each modifier with correct over path."""
        register_scripted(
            "each_make",
            lambda input_data, config: Clusters(groups=["x", "y"]),
        )
        register_scripted(
            "each_verify",
            lambda input_data, config: MatchResult(label="checked"),
        )

        class EachCheckPipeline(ForwardConstruct):
            make = Node.scripted("each-make", fn="each_make", outputs=Clusters)
            verify = Node.scripted("each-verify", fn="each_verify", outputs=MatchResult)

            def forward(self, topic):
                clusters = self.make(topic)
                for cluster in clusters.groups:
                    self.verify(cluster)
                return clusters

        pipeline = EachCheckPipeline()
        verify_node = [n for n in pipeline.nodes if isinstance(n, Node) and n.name == "each-verify"]
        assert len(verify_node) == 1
        assert verify_node[0].has_modifier(Each)
        each_mod = verify_node[0].get_modifier(Each)
        assert each_mod.over == "each_make.groups"

    def test_post_loop_node_has_no_each_when_after_for_loop(self):
        """Nodes after the for loop are NOT wrapped with Each."""
        register_scripted(
            "post_make",
            lambda input_data, config: Clusters(groups=["a"]),
        )
        register_scripted(
            "post_verify",
            lambda input_data, config: MatchResult(label="ok"),
        )
        register_scripted(
            "post_report",
            lambda input_data, config: Report(summary="final"),
        )

        class PostLoopPipeline(ForwardConstruct):
            make = Node.scripted("post-make", fn="post_make", outputs=Clusters)
            verify = Node.scripted("post-verify", fn="post_verify", outputs=MatchResult)
            report = Node.scripted("post-report", fn="post_report", outputs=Report)

            def forward(self, topic):
                clusters = self.make(topic)
                for cluster in clusters.groups:
                    self.verify(cluster)
                return self.report(clusters)

        pipeline = PostLoopPipeline()
        report_node = [n for n in pipeline.nodes if isinstance(n, Node) and n.name == "post-report"]
        assert len(report_node) == 1
        assert not report_node[0].has_modifier(Each)


# ═══════════════════════════════════════════════════════════════════════════
# Part 8: try/except in forward() — v1 behavior
# ═══════════════════════════════════════════════════════════════════════════


class ErrorResult(BaseModel, frozen=True):
    message: str


class TestForwardConstructExceptions:
    """try/except in forward() — v1 traces the try body; except is dead code.

    In v1, proxy operations in the try block never raise (they are symbolic),
    so the except block is unreachable during tracing. Only real Python errors
    (e.g., division by zero) before any node call can route tracing into the
    except block.

    This is a documented limitation: try/except does not compile to a
    fallback graph. The except block only runs if tracing itself fails.
    """

    def test_try_body_nodes_traced_when_no_exception(self):
        """try body with node calls, no exception -> nodes recorded correctly."""
        register_scripted(
            "te_extract",
            lambda input_data, config: RawText(text="extracted"),
        )
        register_scripted(
            "te_classify",
            lambda input_data, config: Confidence(score=0.8),
        )

        class TryPipeline(ForwardConstruct):
            extract = Node.scripted("te-extract", fn="te_extract", outputs=RawText)
            classify = Node.scripted("te-classify", fn="te_classify", outputs=Confidence)

            def forward(self, topic):
                try:
                    raw = self.extract(topic)
                    result = self.classify(raw)
                except Exception:
                    result = self.classify(topic)
                return result

        pipeline = TryPipeline()
        node_names = [n.name for n in pipeline.nodes]
        assert "te-extract" in node_names
        assert "te-classify" in node_names
        assert len(node_names) == 2

    def test_except_body_traced_when_real_python_error_raised(self):
        """try body raises a real Python error (division by zero) before a
        node call -> except body calls a fallback node -> fallback is recorded."""
        register_scripted(
            "te_fallback",
            lambda input_data, config: ErrorResult(message="fallback"),
        )

        class FallbackPipeline(ForwardConstruct):
            fallback = Node.scripted(
                "te-fallback",
                fn="te_fallback",
                outputs=ErrorResult,
            )

            def forward(self, topic):
                try:
                    _ = 1 / 0  # real Python error before any node call
                    return self.fallback(topic)  # never reached
                except ZeroDivisionError:
                    return self.fallback(topic)

        pipeline = FallbackPipeline()
        node_names = [n.name for n in pipeline.nodes]
        assert "te-fallback" in node_names
        assert len(node_names) == 1

    def test_try_body_only_traced_when_proxy_ops_in_try(self):
        """Proxy operations in try block don't raise -> try-body nodes
        are recorded, except body is skipped."""
        register_scripted(
            "te_primary",
            lambda input_data, config: RawText(text="primary"),
        )
        register_scripted(
            "te_backup",
            lambda input_data, config: RawText(text="backup"),
        )
        register_scripted(
            "te_report",
            lambda input_data, config: FinalResult(summary="done"),
        )

        class ProxyTryPipeline(ForwardConstruct):
            primary = Node.scripted("te-primary", fn="te_primary", outputs=RawText)
            backup = Node.scripted("te-backup", fn="te_backup", outputs=RawText)
            report = Node.scripted("te-report", fn="te_report", outputs=FinalResult)

            def forward(self, topic):
                try:
                    result = self.primary(topic)
                except Exception:
                    result = self.backup(topic)
                return self.report(result)

        pipeline = ProxyTryPipeline()
        node_names = [n.name for n in pipeline.nodes]
        # primary is traced (try body runs), backup is NOT (except is dead code)
        assert "te-primary" in node_names
        assert "te-backup" not in node_names
        assert "te-report" in node_names
        assert len(node_names) == 2

    def test_except_body_skipped_when_proxy_ops_never_raise(self):
        """If both try and except call nodes, only try-body nodes appear
        in the trace. The except block is dead code during tracing.

        This documents expected v1 behavior — try/except does not compile
        to a fallback graph.
        """
        register_scripted(
            "te_main",
            lambda input_data, config: RawText(text="main"),
        )
        register_scripted(
            "te_rescue",
            lambda input_data, config: RawText(text="rescue"),
        )
        register_scripted(
            "te_final",
            lambda input_data, config: FinalResult(summary="final"),
        )

        class LimitationPipeline(ForwardConstruct):
            main = Node.scripted("te-main", fn="te_main", outputs=RawText)
            rescue = Node.scripted("te-rescue", fn="te_rescue", outputs=RawText)
            final = Node.scripted("te-final", fn="te_final", outputs=FinalResult)

            def forward(self, topic):
                try:
                    result = self.main(topic)
                except Exception:
                    result = self.rescue(topic)
                return self.final(result)

        pipeline = LimitationPipeline()
        node_names = [n.name for n in pipeline.nodes]

        # v1 limitation: only try-body nodes are traced
        assert "te-main" in node_names, "try-body node must be traced"
        assert "te-rescue" not in node_names, "v1 limitation: except-body nodes are dead code during tracing"
        assert "te-final" in node_names, "post-try/except nodes must be traced"

        # The pipeline should still compile despite dead except-body nodes
        compile(pipeline, **build_test_compile_kwargs())


# ═══════════════════════════════════════════════════════════════════════════
# Part 9: self.loop() primitive — tracing and validation
# ═══════════════════════════════════════════════════════════════════════════


class Draft(BaseModel, frozen=True):
    content: str
    iteration: int = 0
    score: float = 0.0


class ReviewResult(BaseModel, frozen=True):
    score: float
    feedback: str


class TestSelfLoopTracing:
    """self.loop() primitive: tracing, node list, and error handling."""

    def test_empty_body_raises_construct_error(self):
        """self.loop(body=[], when=...) raises ConstructError at trace time."""
        from neograph import ForwardConstruct

        class EmptyLoop(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_empty", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=5,
                )(d)
                return d

        register_scripted(
            "fc_seed_empty",
            lambda _in, _cfg: Draft(content="seed", score=0.0),
        )

        with pytest.raises(ConstructError, match="at least one node"):
            EmptyLoop()

    def test_loop_construct_appears_in_traced_nodes(self):
        """After tracing, the node list contains a Construct with Loop modifier
        for the loop body, not the individual body nodes."""
        from neograph import ForwardConstruct
        from neograph.modifiers import Loop

        class LoopPipeline(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_trace", outputs=Draft)
            review = Node.scripted("review", fn="fc_review_trace", outputs=ReviewResult)
            revise = Node.scripted("revise", fn="fc_revise_trace", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.review, self.revise],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=10,
                )(d)
                return d

        register_scripted(
            "fc_seed_trace",
            lambda _in, _cfg: Draft(content="trace-seed", score=0.0),
        )
        register_scripted(
            "fc_review_trace",
            lambda _in, _cfg: ReviewResult(score=0.5, feedback="ok"),
        )
        register_scripted(
            "fc_revise_trace",
            lambda _in, _cfg: Draft(content="trace-revised", score=0.5),
        )

        pipeline = LoopPipeline()

        # Should have 2 entries: the seed Node and the loop Construct
        assert len(pipeline.nodes) == 2

        # First is the seed node
        assert isinstance(pipeline.nodes[0], Node)
        assert pipeline.nodes[0].name == "seed"

        # Second is a Construct (the loop body) with a Loop modifier
        loop_entry = pipeline.nodes[1]
        assert isinstance(loop_entry, Construct)
        assert loop_entry.has_modifier(Loop)

        loop_mod = loop_entry.get_modifier(Loop)
        assert loop_mod.max_iterations == 10

        # The loop construct's internal nodes are the body nodes
        inner_names = [n.name for n in loop_entry.nodes]
        assert "review" in inner_names
        assert "revise" in inner_names


# ═══════════════════════════════════════════════════════════════════════════
# Part 9b: self.each() primitive — fan-out over a sub-construct (neograph-e9zse.1)
# ═══════════════════════════════════════════════════════════════════════════


class TestSelfEachTracing:
    """self.each() two-step tracing surface (neograph-e9zse.1).

    Contract: ``each_x = self.each(body=[...], key=..., on_error=...)`` returns
    a deferred callable; calling it with a proxy attribute (or a raw dotted
    string) builds ``Construct(input=item_type, output=..., nodes=body) |
    Each(over=..., key=..., on_error=...)``, records it into the tracer, and
    returns the output proxy.

    Core Invariant pinned here: the traced IR must be IDENTICAL to the
    declarative ``Construct(input=, output=, nodes=) | Each(over, key)`` twin
    (see tests/test_composition.py TestEachOnErrorCollect._build_parent) —
    no ForwardConstruct-only IR, zero compiler/validator changes.
    """

    @staticmethod
    def _assert_ir_identical(traced_sub, declarative_sub):
        """Structural IR equality between the traced Each'd sub-construct and
        its declarative twin: name, boundary ports, node list, Each params."""
        assert isinstance(traced_sub, Construct), (
            f"self.each() must emit a Construct sub-construct, got {type(traced_sub).__name__}"
        )
        assert traced_sub.has_modifier(Each), "self.each() sub-construct must carry an Each modifier"

        # Each modifier params pass through verbatim
        traced_each = traced_sub.get_modifier(Each)
        decl_each = declarative_sub.get_modifier(Each)
        assert traced_each.over == decl_each.over
        assert traced_each.key == decl_each.key
        assert traced_each.on_error == decl_each.on_error

        # Boundary ports (Construct.input / Construct.output — singular)
        assert traced_sub.input is declarative_sub.input
        assert traced_sub.output is declarative_sub.output

        # Node list: same length, names, inputs, outputs, scripted fns
        assert len(traced_sub.nodes) == len(declarative_sub.nodes)
        for traced_node, decl_node in zip(traced_sub.nodes, declarative_sub.nodes, strict=True):
            assert isinstance(traced_node, Node)
            assert traced_node.name == decl_node.name
            assert traced_node.inputs == decl_node.inputs
            assert traced_node.outputs == decl_node.outputs
            assert traced_node.scripted_fn == decl_node.scripted_fn

    def test_proxy_form_traces_ir_identical_to_declarative_twin(self):
        """(a) Two-step self.each() with a proxy attr — the traced node list is
        IR-identical to the declarative Construct|Each twin."""
        from tests.hypothesis.conftest import FanCollection, FanItem

        register_scripted(
            "fc_each_seed_a",
            lambda _in, _cfg: FanCollection(items=[FanItem(item_id="i1"), FanItem(item_id="i2")]),
        )
        register_scripted(
            "fc_each_verify_a",
            lambda item, _cfg: RawText(text=f"ok-{item.item_id}"),
        )

        class FanPipeline(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_each_seed_a", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_each_verify_a", outputs=RawText)

            def forward(self, topic):
                items = self.seed(topic)
                each_verify = self.each(body=[self.verify], key="item_id")
                return each_verify(items.items)

        pipeline = FanPipeline()

        # Two entries: the seed Node and the Each'd sub-construct
        assert len(pipeline.nodes) == 2
        assert isinstance(pipeline.nodes[0], Node)
        assert pipeline.nodes[0].name == "seed"
        traced_sub = pipeline.nodes[1]

        # The declarative twin — the exact shape self.each() must emit
        # (deterministic name each-{body_slug}, input=item type inferred from
        # seed.items -> list[FanItem], output=last body node's output).
        declarative_sub = Construct(
            "each-verify",
            input=FanItem,
            output=RawText,
            nodes=[
                Node.scripted("verify", fn="fc_each_verify_a", inputs=FanItem, outputs=RawText),
            ],
        ) | Each(over="seed.items", key="item_id", on_error="raise")

        assert traced_sub.name == declarative_sub.name
        self._assert_ir_identical(traced_sub, declarative_sub)

    def test_raw_string_over_infers_item_type_from_traced_root(self):
        """(b) Raw-string over ('rawseed.items') — the tracer reverse-resolves
        the root segment to the traced producer and infers the item type,
        emitting the same IR as the proxy form (Option C)."""
        from tests.hypothesis.conftest import FanCollection, FanItem

        register_scripted(
            "fc_each_seed_b",
            lambda _in, _cfg: FanCollection(items=[FanItem(item_id="i1")]),
        )
        register_scripted(
            "fc_each_verify_b",
            lambda item, _cfg: RawText(text=f"ok-{item.item_id}"),
        )

        class RawStringFan(ForwardConstruct):
            rawseed = Node.scripted("rawseed", fn="fc_each_seed_b", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_each_verify_b", outputs=RawText)

            def forward(self, topic):
                self.rawseed(topic)
                each_verify = self.each(body=[self.verify], key="item_id")
                return each_verify("rawseed.items")

        pipeline = RawStringFan()

        assert len(pipeline.nodes) == 2
        traced_sub = pipeline.nodes[1]

        declarative_sub = Construct(
            "each-verify",
            input=FanItem,
            output=RawText,
            nodes=[
                Node.scripted("verify", fn="fc_each_verify_b", inputs=FanItem, outputs=RawText),
            ],
        ) | Each(over="rawseed.items", key="item_id", on_error="raise")

        assert traced_sub.name == declarative_sub.name
        self._assert_ir_identical(traced_sub, declarative_sub)

    def test_on_error_collect_passes_through_to_each_modifier(self):
        """(c) on_error='collect' passes through verbatim to the Each modifier."""
        from tests.hypothesis.conftest import FanCollection, FanItem

        register_scripted(
            "fc_each_seed_c",
            lambda _in, _cfg: FanCollection(items=[FanItem(item_id="i1")]),
        )
        register_scripted(
            "fc_each_verify_c",
            lambda item, _cfg: RawText(text=f"ok-{item.item_id}"),
        )

        class CollectFan(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_each_seed_c", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_each_verify_c", outputs=RawText)

            def forward(self, topic):
                items = self.seed(topic)
                each_verify = self.each(body=[self.verify], key="item_id", on_error="collect")
                return each_verify(items.items)

        pipeline = CollectFan()

        traced_sub = pipeline.nodes[1]
        assert isinstance(traced_sub, Construct)
        assert traced_sub.has_modifier(Each)
        each_mod = traced_sub.get_modifier(Each)
        assert each_mod.on_error == "collect"
        assert each_mod.over == "seed.items"
        assert each_mod.key == "item_id"

        # The declarative on_error='collect' twin (test_composition.py
        # TestEachOnErrorCollect shape) must be IR-identical too.
        declarative_sub = Construct(
            "each-verify",
            input=FanItem,
            output=RawText,
            nodes=[
                Node.scripted("verify", fn="fc_each_verify_c", inputs=FanItem, outputs=RawText),
            ],
        ) | Each(over="seed.items", key="item_id", on_error="collect")
        self._assert_ir_identical(traced_sub, declarative_sub)

    def test_each_pipeline_compiles_and_fans_out_end_to_end(self):
        """Round-trip: the self.each() pipeline compiles through the real
        compiler and runs on the real LangGraph runtime — the barrier field
        is a dict keyed by the CUSTOM key (item_id), one entry per item,
        observed through run()'s public result surface."""
        from tests.hypothesis.conftest import FanCollection, FanItem

        register_scripted(
            "fc_each_seed_e2e",
            lambda _in, _cfg: FanCollection(
                items=[FanItem(item_id="alpha"), FanItem(item_id="beta")]
            ),
        )
        register_scripted(
            "fc_each_verify_e2e",
            lambda item, _cfg: RawText(text=f"ok-{item.item_id}"),
        )

        class E2EFan(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_each_seed_e2e", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_each_verify_e2e", outputs=RawText)

            def forward(self, topic):
                items = self.seed(topic)
                each_verify = self.each(body=[self.verify], key="item_id")
                return each_verify(items.items)

        pipeline = E2EFan()
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test-self-each-e2e"})

        barrier = result["each_verify"]
        assert isinstance(barrier, dict)
        assert set(barrier.keys()) == {"alpha", "beta"}
        assert isinstance(barrier["alpha"], RawText)
        assert barrier["alpha"].text == "ok-alpha"
        assert barrier["beta"].text == "ok-beta"


class TestForwardTracingVerifiedCapabilities:
    """neograph-e9zse.3: pins the three capabilities the parity table marked
    'verify' — believed to pass through ForwardConstruct tracing but
    previously untested. Each is asserted against the declarative twin
    (the epic's acceptance principle), not just 'doesn't crash'.

    1. Fan-in: a traced node with dict-form inputs consuming TWO upstreams.
    2. Multi-output dict nodes: outputs={...} and {node}_{key} downstream wiring.
    3. skip_when / skip_value surviving tracing (incl. the loop-copy path).
    """

    def test_fan_in_node_traces_ir_identical_to_declarative_twin(self):
        """A forward() node consuming two upstream producers via dict-form
        inputs traces to the same node list as the declarative Construct."""
        register_scripted("vcap_left_a", lambda _i, _c: RawText(text="L"))
        register_scripted("vcap_right_a", lambda _i, _c: Claims(items=["R"]))
        register_scripted(
            "vcap_fuse_a",
            lambda data, _c: FinalResult(summary=f"{data['left'].text}+{len(data['right'].items)}"),
        )

        fan_in_inputs = {"left": RawText, "right": Claims}

        class FanIn(ForwardConstruct):
            left = Node.scripted("left", fn="vcap_left_a", outputs=RawText)
            right = Node.scripted("right", fn="vcap_right_a", outputs=Claims)
            fuse = Node.scripted(
                "fuse", fn="vcap_fuse_a", inputs=fan_in_inputs, outputs=FinalResult
            )

            def forward(self, topic):
                left_out = self.left(topic)
                right_out = self.right(topic)
                return self.fuse(left_out, right_out)

        pipeline = FanIn()

        declarative = Construct(
            "FanIn",
            nodes=[
                Node.scripted("left", fn="vcap_left_a", outputs=RawText),
                Node.scripted("right", fn="vcap_right_a", outputs=Claims),
                Node.scripted(
                    "fuse", fn="vcap_fuse_a", inputs=fan_in_inputs, outputs=FinalResult
                ),
            ],
        )

        assert len(pipeline.nodes) == len(declarative.nodes)
        for traced, decl in zip(pipeline.nodes, declarative.nodes, strict=True):
            assert isinstance(traced, Node)
            assert traced.name == decl.name
            assert traced.inputs == decl.inputs
            assert traced.outputs == decl.outputs
            assert traced.scripted_fn == decl.scripted_fn

        # Runtime round-trip: the fan-in consumer sees BOTH upstream values.
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "vcap-fanin"})
        assert result["fuse"] == FinalResult(summary="L+1")

    def test_multi_output_dict_node_traces_ir_identical_and_runs(self):
        """A traced node with dict-form outputs keeps the outputs dict
        verbatim, and a downstream node wired via the {node}_{key} field
        name receives that key's value at runtime."""
        multi_outputs = {"summary": RawText, "count": Claims}

        register_scripted(
            "vcap_analyze_b",
            lambda _i, _c: {"summary": RawText(text="S"), "count": Claims(items=["x", "y"])},
        )
        register_scripted(
            "vcap_report_b",
            lambda data, _c: FinalResult(summary=data["analyze_summary"].text + "!"),
        )

        class MultiOut(ForwardConstruct):
            analyze = Node.scripted("analyze", fn="vcap_analyze_b", outputs=multi_outputs)
            report = Node.scripted(
                "report",
                fn="vcap_report_b",
                inputs={"analyze_summary": RawText},
                outputs=FinalResult,
            )

            def forward(self, topic):
                analyzed = self.analyze(topic)
                return self.report(analyzed.summary)

        pipeline = MultiOut()

        declarative = Construct(
            "MultiOut",
            nodes=[
                Node.scripted("analyze", fn="vcap_analyze_b", outputs=multi_outputs),
                Node.scripted(
                    "report",
                    fn="vcap_report_b",
                    inputs={"analyze_summary": RawText},
                    outputs=FinalResult,
                ),
            ],
        )

        assert len(pipeline.nodes) == len(declarative.nodes)
        for traced, decl in zip(pipeline.nodes, declarative.nodes, strict=True):
            assert isinstance(traced, Node)
            assert traced.name == decl.name
            assert traced.inputs == decl.inputs
            assert traced.outputs == decl.outputs

        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "vcap-multiout"})
        assert result["analyze_summary"] == RawText(text="S")
        assert result["analyze_count"] == Claims(items=["x", "y"])
        assert result["report"] == FinalResult(summary="S!")

    def test_skip_when_and_skip_value_survive_tracing(self):
        """Node-level skip_when/skip_value callables pass through tracing
        unchanged — both on the straight-line path (identity: the traced
        node IS the class attribute) and through the loop-body model_copy
        path (copies carry the same callables)."""

        def _skip_pred(data):
            return False

        def _skip_val(data):
            return RawText(text="skipped")

        register_scripted("vcap_seed_c", lambda _i, _c: Draft(content="seed"))
        register_scripted("vcap_guarded_c", lambda _i, _c: RawText(text="ran"))

        class Guarded(ForwardConstruct):
            seed = Node.scripted("seed", fn="vcap_seed_c", outputs=Draft)
            guarded = Node(
                "guarded",
                mode="scripted",
                scripted_fn="vcap_guarded_c",
                inputs=Draft,
                outputs=RawText,
                skip_when=_skip_pred,
                skip_value=_skip_val,
            )

            def forward(self, topic):
                d = self.seed(topic)
                return self.guarded(d)

        pipeline = Guarded()
        traced_guarded = pipeline.nodes[1]
        assert isinstance(traced_guarded, Node)
        assert traced_guarded.skip_when is _skip_pred
        assert traced_guarded.skip_value is _skip_val

        # Loop-copy path: the body copy created by _LoopCall._materialize
        # (model_copy) must preserve the skip callables too.
        register_scripted("vcap_loop_seed_c", lambda _i, _c: Draft(content="seed"))
        register_scripted("vcap_loop_body_c", lambda _i, _c: Draft(content="looped"))

        class GuardedLoop(ForwardConstruct):
            seed = Node.scripted("seed", fn="vcap_loop_seed_c", outputs=Draft)
            body = Node(
                "body",
                mode="scripted",
                scripted_fn="vcap_loop_body_c",
                outputs=Draft,
                skip_when=_skip_pred,
                skip_value=_skip_val,
            )

            def forward(self, topic):
                d = self.seed(topic)
                return self.loop(
                    body=[self.body],
                    when=lambda r: r is None,
                    max_iterations=2,
                    on_exhaust="last",
                )(d)

        loop_pipeline = GuardedLoop()
        loop_sub = loop_pipeline.nodes[1]
        assert isinstance(loop_sub, Construct)
        body_copy = loop_sub.nodes[0]
        assert isinstance(body_copy, Node)
        assert body_copy is not GuardedLoop.body  # a copy, not the class attr
        assert body_copy.skip_when is _skip_pred
        assert body_copy.skip_value is _skip_val


class TestSelfEachEdgeCases:
    """self.each() error paths, multi-node bodies, non-mutation, and
    deterministic naming (neograph-e9zse.1 plan step 6)."""

    @staticmethod
    def _register_fan_fns(suffix):
        from tests.hypothesis.conftest import FanCollection, FanItem

        register_scripted(
            f"fc_eachedge_seed_{suffix}",
            lambda _in, _cfg: FanCollection(items=[FanItem(item_id="i1")]),
        )
        register_scripted(
            f"fc_eachedge_verify_{suffix}",
            lambda item, _cfg: RawText(text=f"ok-{item.item_id}"),
        )

    def test_empty_body_raises_construct_error(self):
        from tests.hypothesis.conftest import FanCollection

        self._register_fan_fns("empty")

        class EmptyBody(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_empty", outputs=FanCollection)

            def forward(self, topic):
                items = self.seed(topic)
                return self.each(body=[], key="item_id")(items.items)

        with pytest.raises(ConstructError, match="at least one node"):
            EmptyBody()

    def test_non_node_body_item_raises_construct_error(self):
        from tests.hypothesis.conftest import FanCollection

        self._register_fan_fns("nonnode")

        class NonNodeBody(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_nonnode", outputs=FanCollection)

            def forward(self, topic):
                items = self.seed(topic)
                return self.each(body=["not-a-node"], key="item_id")(items.items)

        with pytest.raises(ConstructError, match="node reference"):
            NonNodeBody()

    def test_non_list_over_path_raises_construct_error(self):
        """Fanning out over a scalar field fails loud at trace time."""
        from tests.hypothesis.conftest import FanCollection

        self._register_fan_fns("scalar")

        class ScalarOver(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_scalar", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_eachedge_verify_scalar", outputs=RawText)

            def forward(self, topic):
                items = self.seed(topic)
                # FanCollection has no scalar 'items.item_id' list — walk a
                # non-list terminal: 'label' on FanItem is str
                each_verify = self.each(body=[self.verify], key="item_id")
                return each_verify(items)  # the collection root itself, not a list field

        with pytest.raises(ConstructError, match="not a list field"):
            ScalarOver()

    def test_unresolvable_over_attr_raises_construct_error(self):
        from tests.hypothesis.conftest import FanCollection

        self._register_fan_fns("badattr")

        class BadAttr(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_badattr", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_eachedge_verify_badattr", outputs=RawText)

            def forward(self, topic):
                items = self.seed(topic)
                each_verify = self.each(body=[self.verify], key="item_id")
                return each_verify(items.nonexistent_field)

        with pytest.raises(ConstructError, match="does not resolve"):
            BadAttr()

    def test_raw_string_unmatched_root_raises_construct_error(self):
        from tests.hypothesis.conftest import FanCollection

        self._register_fan_fns("badroot")

        class BadRoot(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_badroot", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_eachedge_verify_badroot", outputs=RawText)

            def forward(self, topic):
                self.seed(topic)
                each_verify = self.each(body=[self.verify], key="item_id")
                return each_verify("nosuchnode.items")

        with pytest.raises(ConstructError, match="does not match any traced node"):
            BadRoot()

    def test_forward_input_proxy_as_over_raises_construct_error(self):
        """The raw forward() input seed has no traced producer to infer from."""
        from tests.hypothesis.conftest import FanCollection

        self._register_fan_fns("seedproxy")

        class SeedProxyOver(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_seedproxy", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_eachedge_verify_seedproxy", outputs=RawText)

            def forward(self, topic):
                self.seed(topic)
                each_verify = self.each(body=[self.verify], key="item_id")
                return each_verify(topic)

        with pytest.raises(ConstructError, match="node output attribute"):
            SeedProxyOver()

    def test_multi_node_body_wraps_all_nodes_in_one_sub_construct(self):
        from tests.hypothesis.conftest import FanCollection, FanItem

        self._register_fan_fns("multi")
        register_scripted(
            "fc_eachedge_score_multi",
            lambda raw, _cfg: MatchResult(label="scored", matched=True),
        )

        class MultiBody(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_multi", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_eachedge_verify_multi", outputs=RawText)
            score = Node.scripted("score", fn="fc_eachedge_score_multi", outputs=MatchResult)

            def forward(self, topic):
                items = self.seed(topic)
                each_vs = self.each(body=[self.verify, self.score], key="item_id")
                return each_vs(items.items)

        pipeline = MultiBody()

        assert len(pipeline.nodes) == 2
        sub = pipeline.nodes[1]
        assert isinstance(sub, Construct)
        assert sub.name == "each-verify-score"
        assert [n.name for n in sub.nodes] == ["verify", "score"]
        # First body node gets the item port; the second keeps its own wiring
        assert sub.nodes[0].inputs is FanItem
        assert sub.input is FanItem
        assert sub.output is MatchResult

    def test_each_call_does_not_mutate_class_level_nodes(self):
        """Same copy-not-mutate rule as _LoopCall (neograph-2o9n)."""
        from tests.hypothesis.conftest import FanCollection

        self._register_fan_fns("nomut")

        class NoMutate(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_nomut", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_eachedge_verify_nomut", outputs=RawText)

            def forward(self, topic):
                items = self.seed(topic)
                return self.each(body=[self.verify], key="item_id")(items.items)

        NoMutate()
        assert NoMutate.verify.inputs is None, "Class-level Node.inputs was mutated by _EachCall"
        # A second instantiation must produce the same IR
        second = NoMutate()
        assert second.nodes[1].name == "each-verify"

    def test_loop_and_each_over_same_body_slug_get_distinct_names(self):
        """The occurrence counter is keyed per (kind, slug): a loop and an
        each over the same body do not share a counter, and re-trace passes
        produce identical deterministic names."""
        from tests.hypothesis.conftest import FanCollection

        self._register_fan_fns("kinds")

        class LoopAndEach(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_eachedge_seed_kinds", outputs=FanCollection)
            verify = Node.scripted("verify", fn="fc_eachedge_verify_kinds", outputs=RawText)

            def forward(self, topic):
                items = self.seed(topic)
                looped = self.loop(
                    body=[self.verify],
                    when=lambda r: r is None,
                    max_iterations=2,
                )(items)
                fanned = self.each(body=[self.verify], key="item_id")(items.items)
                return fanned

        pipeline = LoopAndEach()

        names = [n.name for n in pipeline.nodes]
        assert names == ["seed", "loop-verify", "each-verify"]
        # Determinism across a fresh trace of the same class
        assert [n.name for n in LoopAndEach().nodes] == names


class TestSelfEnsembleEdgeCases:
    """self.ensemble() error paths + end-to-end run (neograph-e9zse.5).
    IR-equality twins live in tests/test_forward_parity.py (corpus rows
    oracle_ensemble / oracle_ensemble_sub_construct + form-aware nesting)."""

    def test_no_merge_strategy_fails_loud_at_construction(self):
        """Oracle's own validation fires when self.ensemble() is built —
        before the proxy call, so the trace fails at the definition site."""
        from neograph.errors import ConfigurationError

        register_scripted("ens_seed_a", lambda _i, _c: RawText(text="s"))
        register_scripted("ens_gen_a", lambda _i, _c: RawText(text="g"))

        class NoMerge(ForwardConstruct):
            seed = Node.scripted("seed", fn="ens_seed_a", outputs=RawText)
            gen = Node.scripted("gen", fn="ens_gen_a", inputs=RawText, outputs=RawText)

            def forward(self, topic):
                t = self.seed(topic)
                return self.ensemble(self.gen, n=3)(t)

        with pytest.raises(ConfigurationError, match="merge strategy"):
            NoMerge()

    def test_empty_body_list_raises_construct_error(self):
        register_scripted("ens_seed_b", lambda _i, _c: RawText(text="s"))
        register_scripted("ens_merge_b", lambda variants, _c: variants[0])

        class EmptyBody(ForwardConstruct):
            seed = Node.scripted("seed", fn="ens_seed_b", outputs=RawText)

            def forward(self, topic):
                t = self.seed(topic)
                return self.ensemble([], n=2, merge_fn="ens_merge_b")(t)

        with pytest.raises(ConstructError, match="at least one node"):
            EmptyBody()

    def test_non_node_body_item_raises_construct_error(self):
        register_scripted("ens_seed_c", lambda _i, _c: RawText(text="s"))
        register_scripted("ens_merge_c", lambda variants, _c: variants[0])

        class BadBody(ForwardConstruct):
            seed = Node.scripted("seed", fn="ens_seed_c", outputs=RawText)

            def forward(self, topic):
                t = self.seed(topic)
                return self.ensemble(["not-a-node"], n=2, merge_fn="ens_merge_c")(t)

        with pytest.raises(ConstructError, match="node references"):
            BadBody()

    def test_invalid_target_type_raises_construct_error(self):
        register_scripted("ens_seed_d", lambda _i, _c: RawText(text="s"))
        register_scripted("ens_merge_d", lambda variants, _c: variants[0])

        class BadTarget(ForwardConstruct):
            seed = Node.scripted("seed", fn="ens_seed_d", outputs=RawText)

            def forward(self, topic):
                t = self.seed(topic)
                return self.ensemble("gen", n=2, merge_fn="ens_merge_d")(t)

        with pytest.raises(ConstructError, match="node reference or a list"):
            BadTarget()

    def test_models_list_infers_n_when_n_omitted(self):
        """models= without n= infers n from len(models) — Oracle's own
        inference must survive the kwarg pass-through."""
        from neograph.modifiers import Oracle

        register_scripted("ens_seed_e", lambda _i, _c: RawText(text="s"))
        register_scripted("ens_gen_e", lambda _i, _c: RawText(text="g"))
        register_scripted("ens_merge_e", lambda variants, _c: variants[0])

        class ModelsOnly(ForwardConstruct):
            seed = Node.scripted("seed", fn="ens_seed_e", outputs=RawText)
            gen = Node.scripted("gen", fn="ens_gen_e", inputs=RawText, outputs=RawText)

            def forward(self, topic):
                t = self.seed(topic)
                return self.ensemble(
                    self.gen, models=["reason", "fast"], merge_fn="ens_merge_e"
                )(t)

        pipeline = ModelsOnly()
        gen_member = pipeline.nodes[1]
        oracle = gen_member.get_modifier(Oracle)
        assert oracle.n == 2
        assert oracle.models == ["reason", "fast"]

    def test_node_form_ensemble_compiles_and_runs_end_to_end(self):
        """Traced node-form ensemble runs: N variants generated, scripted
        merge_fn collapses them, downstream sees the merged value."""
        register_scripted("ens_seed_f", lambda _i, _c: RawText(text="seed"))
        register_scripted("ens_gen_f", lambda t, _c: RawText(text=f"var-of-{t.text}"))
        register_scripted("ens_merge_f", lambda variants, _c: variants[0])

        class Ensembled(ForwardConstruct):
            seed = Node.scripted("seed", fn="ens_seed_f", outputs=RawText)
            gen = Node.scripted("gen", fn="ens_gen_f", inputs=RawText, outputs=RawText)

            def forward(self, topic):
                t = self.seed(topic)
                return self.ensemble(self.gen, n=3, merge_fn="ens_merge_f")(t)

        pipeline = Ensembled()
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "ens-e2e"})
        assert result["gen"] == RawText(text="var-of-seed")


class TestV1LimitsFailLoud:
    """neograph-e9zse.7 (Phase 3 CAP decision): branch richness and
    try/except stay documented v1 limits with a LOUD escape to the
    declarative form. These tests pin the loud trace-time errors."""

    def test_non_constant_comparison_raises_construct_error_naming_the_escape(self):
        """proxy-vs-proxy comparison in an if condition fails loud at trace
        time (previously it traced silently and misbehaved at runtime —
        the threshold would be a _Proxy object)."""
        register_scripted("v1lim_a", lambda _i, _c: Confidence(score=0.1))
        register_scripted("v1lim_b", lambda _i, _c: Confidence(score=0.2))
        register_scripted("v1lim_c", lambda _i, _c: Confidence(score=0.3))

        class NonConstant(ForwardConstruct):
            a = Node.scripted("a", fn="v1lim_a", outputs=Confidence)
            b = Node.scripted("b", fn="v1lim_b", outputs=Confidence)
            c = Node.scripted("c", fn="v1lim_c", outputs=Confidence)

            def forward(self, topic):
                x = self.a(topic)
                y = self.b(topic)
                if x.score < y.score:  # non-constant right-hand side
                    return self.c(x)
                return y

        with pytest.raises(ConstructError, match="constant") as exc_info:
            NonConstant()
        assert "declarative" in str(exc_info.value)

    def test_too_many_branches_error_names_the_declarative_escape(self):
        """Exceeding _MAX_BRANCHES points the author to the declarative form."""
        register_scripted("v1lim_seed", lambda _i, _c: Confidence(score=0.5))
        register_scripted("v1lim_n", lambda _i, _c: Confidence(score=0.5))

        class ManyBranches(ForwardConstruct):
            seed = Node.scripted("seed", fn="v1lim_seed", outputs=Confidence)
            n0 = Node.scripted("n0", fn="v1lim_n", outputs=Confidence)

            def forward(self, topic):
                x = self.seed(topic)
                for _ in range(9):  # 9 > _MAX_BRANCHES (8)
                    if x.score > 0.5:
                        x = self.n0(x)
                return x

        with pytest.raises(ConstructError, match="too many branches") as exc_info:
            ManyBranches()
        assert "declarative" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════
# Part 10: _LoopCall must not mutate class-level Node.inputs (neograph-2o9n)
# ═══════════════════════════════════════════════════════════════════════════


class TestLoopCallDoesNotMutateClassNodes:
    """neograph-2o9n: _LoopCall.__call__ must copy body nodes instead of
    mutating class-level Node.inputs, which leaks across instances."""

    def test_class_node_inputs_unchanged_after_instantiation(self):
        """After creating an instance, the class-level Node.inputs must
        remain None (not mutated by _LoopCall)."""
        register_scripted(
            "fc_seed_2o9n_a",
            lambda _in, _cfg: Draft(content="seed", score=0.0),
        )
        register_scripted(
            "fc_review_2o9n_a",
            lambda _in, _cfg: ReviewResult(score=0.5, feedback="ok"),
        )
        register_scripted(
            "fc_revise_2o9n_a",
            lambda _in, _cfg: Draft(content="revised", score=0.9),
        )

        class Writer(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_2o9n_a", outputs=Draft)
            review = Node.scripted("review", fn="fc_review_2o9n_a", outputs=ReviewResult)
            revise = Node.scripted("revise", fn="fc_revise_2o9n_a", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.review, self.revise],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=5,
                )(d)
                return d

        Writer()

        # Class-level nodes must not have been mutated
        assert Writer.review.inputs is None, "Class-level Node.inputs was mutated by _LoopCall"
        assert Writer.revise.inputs is None, "Class-level Node.inputs was mutated by _LoopCall"

    def test_second_instance_works_when_same_class_instantiated_twice(self):
        """Two instances of the same ForwardConstruct subclass must both
        trace correctly — the second must not see mutated inputs from the first."""
        register_scripted(
            "fc_seed_2o9n_b",
            lambda _in, _cfg: Draft(content="seed", score=0.0),
        )
        register_scripted(
            "fc_review_2o9n_b",
            lambda _in, _cfg: ReviewResult(score=0.5, feedback="ok"),
        )
        register_scripted(
            "fc_revise_2o9n_b",
            lambda _in, _cfg: Draft(content="revised", score=0.9),
        )

        class Writer(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_2o9n_b", outputs=Draft)
            review = Node.scripted("review", fn="fc_review_2o9n_b", outputs=ReviewResult)
            revise = Node.scripted("revise", fn="fc_revise_2o9n_b", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.review, self.revise],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=5,
                )(d)
                return d

        p1 = Writer()
        p2 = Writer()

        # Both instances should have valid loop constructs
        from neograph.modifiers import Loop

        loop1 = [n for n in p1.nodes if isinstance(n, Construct) and n.has_modifier(Loop)]
        loop2 = [n for n in p2.nodes if isinstance(n, Construct) and n.has_modifier(Loop)]
        assert len(loop1) == 1, "First instance missing loop construct"
        assert len(loop2) == 1, "Second instance missing loop construct"

    def test_retrace_does_not_corrupt_class_nodes_with_branch(self):
        """A ForwardConstruct with self.loop() and a branch — re-trace
        (branch discovery) must not corrupt class-level nodes."""
        register_scripted(
            "fc_seed_2o9n_c",
            lambda _in, _cfg: Draft(content="seed", score=0.0),
        )
        register_scripted(
            "fc_review_2o9n_c",
            lambda _in, _cfg: ReviewResult(score=0.5, feedback="ok"),
        )
        register_scripted(
            "fc_revise_2o9n_c",
            lambda _in, _cfg: Draft(content="revised", score=0.9),
        )
        register_scripted(
            "fc_final_2o9n_c",
            lambda _in, _cfg: FinalResult(summary="done"),
        )

        class BranchWriter(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_2o9n_c", outputs=Draft)
            review = Node.scripted("review", fn="fc_review_2o9n_c", outputs=ReviewResult)
            revise = Node.scripted("revise", fn="fc_revise_2o9n_c", outputs=Draft)
            finish = Node.scripted("finish", fn="fc_final_2o9n_c", outputs=FinalResult)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.review, self.revise],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=5,
                )(d)
                return self.finish(d)

        BranchWriter()

        # Class-level nodes still untouched
        assert BranchWriter.review.inputs is None
        assert BranchWriter.revise.inputs is None


# ═══════════════════════════════════════════════════════════════════════════
# Part 11: Nested self.loop() and invalid body items (neograph-ndm1)
# ═══════════════════════════════════════════════════════════════════════════


class TestLoopBodyValidation:
    """neograph-ndm1: self.loop() body members must be node references or
    deferred self.each()/self.loop() builders (nesting legal since
    neograph-e9zse.2). Passing proxies (CALL results) or non-node objects
    still raises ConstructError with a clear message instead of a cryptic
    AttributeError — see TestLoopOverSubConstructBody for the legal
    nested forms."""

    def test_proxy_in_body_raises_construct_error(self):
        """Passing a _Proxy (result of self.some_node(x)) as body item
        raises ConstructError, not AttributeError."""
        from neograph import ForwardConstruct

        register_scripted(
            "fc_seed_ndm1_a",
            lambda _in, _cfg: Draft(content="seed", score=0.0),
        )
        register_scripted(
            "fc_review_ndm1_a",
            lambda _in, _cfg: ReviewResult(score=0.5, feedback="ok"),
        )

        class ProxyInBody(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_ndm1_a", outputs=Draft)
            review = Node.scripted("review", fn="fc_review_ndm1_a", outputs=ReviewResult)

            def forward(self, topic):
                d = self.seed(topic)
                r = self.review(d)  # r is a _Proxy
                d = self.loop(
                    body=[r],  # bad: _Proxy instead of _NodeCall
                    when=lambda x: x is None or x.score < 0.8,
                    max_iterations=3,
                )(d)
                return d

        with pytest.raises(ConstructError, match="node references"):
            ProxyInBody()

    def test_non_node_object_in_body_raises_construct_error(self):
        """Passing a plain string as body item raises ConstructError."""
        from neograph import ForwardConstruct

        register_scripted(
            "fc_seed_ndm1_b",
            lambda _in, _cfg: Draft(content="seed", score=0.0),
        )

        class StringInBody(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_ndm1_b", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=["not_a_node"],  # bad: string
                    when=lambda x: x is None or x.score < 0.8,
                    max_iterations=3,
                )(d)
                return d

        with pytest.raises(ConstructError, match="node references"):
            StringInBody()

    def test_normal_loop_body_still_works(self):
        """Normal usage: body=[self.review, self.revise] still traces correctly."""
        from neograph.modifiers import Loop

        register_scripted(
            "fc_seed_ndm1_c",
            lambda _in, _cfg: Draft(content="seed", score=0.0),
        )
        register_scripted(
            "fc_review_ndm1_c",
            lambda _in, _cfg: ReviewResult(score=0.5, feedback="ok"),
        )
        register_scripted(
            "fc_revise_ndm1_c",
            lambda _in, _cfg: Draft(content="revised", score=0.9),
        )

        class NormalLoop(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_ndm1_c", outputs=Draft)
            review = Node.scripted("review", fn="fc_review_ndm1_c", outputs=ReviewResult)
            revise = Node.scripted("revise", fn="fc_revise_ndm1_c", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.review, self.revise],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=5,
                )(d)
                return d

        pipeline = NormalLoop()
        assert len(pipeline.nodes) == 2
        loop_entry = pipeline.nodes[1]
        assert isinstance(loop_entry, Construct)
        assert loop_entry.has_modifier(Loop)

    def test_single_node_body_works(self):
        """body=[self.refine] with a single _NodeCall works."""
        from neograph.modifiers import Loop

        register_scripted(
            "fc_seed_ndm1_d",
            lambda _in, _cfg: Draft(content="seed", score=0.0),
        )
        register_scripted(
            "fc_refine_ndm1_d",
            lambda _in, _cfg: Draft(content="refined", score=0.9),
        )

        class SingleBody(ForwardConstruct):
            seed = Node.scripted("seed", fn="fc_seed_ndm1_d", outputs=Draft)
            refine = Node.scripted("refine", fn="fc_refine_ndm1_d", outputs=Draft)

            def forward(self, topic):
                d = self.seed(topic)
                d = self.loop(
                    body=[self.refine],
                    when=lambda r: r is None or r.score < 0.8,
                    max_iterations=5,
                )(d)
                return d

        pipeline = SingleBody()
        assert len(pipeline.nodes) == 2
        loop_entry = pipeline.nodes[1]
        assert isinstance(loop_entry, Construct)
        assert loop_entry.has_modifier(Loop)


# =============================================================================
# Coverage gap tests for forward.py
# =============================================================================

from neograph.forward import (
    _BranchMeta,
    _BranchNode,
    _ConditionProxy,
    _ConditionSpec,
    _ForwardSelf,
    _Proxy,
    _Tracer,
)
from neograph.modifiers import Loop


class TestForwardConstructDirectInstantiation:
    """Lines 113-117: direct ForwardConstruct() raises ConstructError."""

    def test_direct_instantiation_raises_type_error(self):
        """Instantiating ForwardConstruct directly (not a subclass) raises ConstructError."""
        with pytest.raises(ConstructError, match="cannot be instantiated directly"):
            ForwardConstruct()


class TestForwardNotOverridden:
    """Line 160: forward() not overridden raises NotImplementedError."""

    def test_forward_not_overridden_raises(self):
        """Subclass without forward() override raises ConstructError at init."""

        class NoForward(ForwardConstruct):
            a = Node.scripted("a", fn="f", outputs=RawText)

        with pytest.raises(ConstructError, match="must override forward"):
            NoForward()

    def test_base_forward_raises_not_implemented(self):
        """Calling ForwardConstruct.forward() directly raises NotImplementedError."""
        # This hits line 160 — the base class's forward() method body
        with pytest.raises(NotImplementedError, match="must override forward"):
            ForwardConstruct.forward(None)


class TestProxyHashBoolIter:
    """Lines 215, 223, 228, 232-233: __hash__, __bool__/__iter__ outside tracing."""

    def test_proxy_hash(self):
        """Proxy.__hash__ returns id-based hash (line 215)."""
        p = _Proxy(source_node=None, name="test")
        assert hash(p) == id(p)

    def test_proxy_bool_outside_tracing_raises(self):
        """Proxy.__bool__ outside tracing raises TypeError (line 223)."""
        p = _Proxy(source_node=None, name="test")
        with pytest.raises(TypeError, match="outside tracing"):
            bool(p)

    def test_proxy_iter_outside_tracing_raises(self):
        """Proxy.__iter__ outside tracing raises TypeError (line 228)."""
        p = _Proxy(source_node=None, name="test")
        with pytest.raises(TypeError, match="outside tracing"):
            iter(p)

    def test_proxy_repr_with_source(self):
        """Proxy.__repr__ with source node (line 232)."""
        node = Node.scripted("mynode", fn="f", outputs=RawText)
        p = _Proxy(source_node=node, name="out_of_mynode")
        assert "mynode" in repr(p)

    def test_proxy_repr_without_source(self):
        """Proxy.__repr__ with None source shows <input> (line 232)."""
        p = _Proxy(source_node=None, name="forward_input")
        assert "<input>" in repr(p)


class TestConditionProxyEmptyAttrChain:
    """Lines 290-292: ConditionProxy with no prefix match yields empty attr_chain."""

    def test_empty_attr_chain_when_name_doesnt_match_prefix(self):
        """_build_runtime_condition returns empty attr_chain when proxy name
        doesn't match the expected prefix pattern (lines 290-292)."""
        node = Node.scripted("mynode", fn="f", outputs=RawText)
        p = _Proxy(source_node=node, name="weird_prefix")
        cond = _ConditionProxy(p, "<", 0.5)
        spec = cond._build_runtime_condition()
        assert spec.attr_chain == []

    def test_empty_attr_chain_when_source_is_none(self):
        """ConditionProxy with source_node=None yields empty attr_chain."""
        p = _Proxy(source_node=None, name="test_name")
        cond = _ConditionProxy(p, ">", 0.5)
        spec = cond._build_runtime_condition()
        assert spec.attr_chain == []


class TestBranchNodeStubs:
    """Lines 358, 361: _BranchNode.has_modifier/get_modifier stubs."""

    def test_has_modifier_returns_false(self):
        """_BranchNode.has_modifier always returns False (line 358)."""
        meta = _BranchMeta(
            condition_spec=_ConditionSpec(None, [], None, "<", 0.5),
            true_arm_nodes=[],
            false_arm_nodes=[],
        )
        bn = _BranchNode(meta, 0)
        assert bn.has_modifier(Loop) is False

    def test_get_modifier_returns_none(self):
        """_BranchNode.get_modifier always returns None (line 361)."""
        meta = _BranchMeta(
            condition_spec=_ConditionSpec(None, [], None, "<", 0.5),
            true_arm_nodes=[],
            false_arm_nodes=[],
        )
        bn = _BranchNode(meta, 0)
        assert bn.get_modifier(Loop) is None


class TestTracingProxyGetSetAttr:
    """Lines 628-633: _ForwardSelf __getattr__ and __setattr__."""

    def test_getattr_falls_through_to_real_self(self):
        """Attribute access on _ForwardSelf falls through to the real instance
        for non-node attributes (line 628-629)."""
        # Test the _ForwardSelf directly — the fallthrough reads from real_self
        node_a = Node.scripted("a", fn="f", outputs=RawText)
        tracer = _Tracer()

        # Build a minimal real object with a custom attr
        class Dummy:
            custom_value = 99

        dummy = Dummy()
        shim = _ForwardSelf({"a": node_a}, tracer, real_self=dummy)
        # Node access returns _NodeCall
        from neograph.forward import _NodeCall

        assert isinstance(shim.a, _NodeCall)
        # Non-node access falls through to real_self
        assert shim.custom_value == 99

    def test_setattr_delegates_to_real_self(self):
        """__setattr__ on _ForwardSelf sets on the real instance (lines 632-633)."""
        node_a = Node.scripted("a", fn="f", outputs=RawText)
        tracer = _Tracer()

        # Use a plain object as real_self to avoid Pydantic validation
        class PlainObj:
            pass

        real = PlainObj()
        shim = _ForwardSelf({"a": node_a}, tracer, real_self=real)
        shim.custom_attr = "set_value"
        assert real.custom_attr == "set_value"


class TestLoopIterationInputInference:
    """Lines 429-431, 557, 560: loop iteration/input inference failures."""

    def test_loop_empty_body_raises(self):
        """self.loop() with empty body raises ConstructError (line 544-545)."""
        from neograph.forward import _LoopCall

        tracer = _Tracer()
        lc = _LoopCall(
            body=[],
            when=lambda r: r is not None and r.score < 0.8,
            max_iterations=5,
            on_exhaust="error",
            tracer=tracer,
        )
        p = _Proxy(source_node=None, name="test")
        with pytest.raises(ConstructError, match="at least one node"):
            lc(p)

    def test_loop_non_node_call_in_body_raises(self):
        """self.loop() with non-_NodeCall items raises ConstructError."""
        from neograph.forward import _LoopCall

        tracer = _Tracer()
        lc = _LoopCall(
            body=["not_a_node_call"],
            when=lambda r: True,
            max_iterations=5,
            on_exhaust="error",
            tracer=tracer,
        )
        p = _Proxy(source_node=None, name="test")
        with pytest.raises(ConstructError, match="node references"):
            lc(p)

    def test_record_iteration_source_none(self):
        """_Tracer.record_iteration with source_node=None uses full_name (line 431)."""
        tracer = _Tracer()
        p = _Proxy(source_node=None, name="some_input_name")
        it = tracer.record_iteration(p)
        # Consume the iterator — it yields one proxy item
        items = list(it)
        assert len(items) == 1
        assert tracer._loop_stack == []  # stack is popped after iteration

    def test_record_iteration_name_mismatch(self):
        """_Tracer.record_iteration when name doesn't match prefix uses field_name (line 429)."""
        tracer = _Tracer()
        node = Node.scripted("mynode", fn="f", outputs=RawText)
        # Name that doesn't start with "out_of_mynode"
        p = _Proxy(source_node=node, name="different_prefix")
        it = tracer.record_iteration(p)
        items = list(it)
        assert len(items) == 1

    def test_loop_input_type_none_raises(self):
        """When loop can't infer input type, raises ConstructError (line 560)."""
        from neograph.forward import _LoopCall, _NodeCall

        tracer = _Tracer()

        # A node with outputs=None
        bad_node = Node("bad", mode="scripted", outputs=RawText)
        nc = _NodeCall(bad_node, tracer)

        lc = _LoopCall(
            body=[nc],
            when=lambda r: True,
            max_iterations=5,
            on_exhaust="error",
            tracer=tracer,
        )
        # Proxy with source_node=None so input_type falls back to output_type
        # which is RawText, so this actually works. We need a node with outputs=None
        bad_node2 = Node("bad2", mode="scripted", outputs=None)
        nc2 = _NodeCall(bad_node2, tracer)
        lc2 = _LoopCall(
            body=[nc2],
            when=lambda r: True,
            max_iterations=5,
            on_exhaust="error",
            tracer=tracer,
        )
        p = _Proxy(source_node=None, name="test")
        with pytest.raises(ConstructError, match="input type could not be inferred"):
            lc2(p)


class TestBranchLoweringEdgeCases:
    """Lines 788, 845, 873-875, 880-881: branch lowering edge cases."""

    def test_plain_proxy_as_branch_condition_truthy_check(self):
        """Using a plain _Proxy (not a comparison) as a branch condition
        produces a truthy-check _ConditionSpec (line 788)."""
        register_scripted("br_a_fn", lambda _i, _c: Confidence(score=0.9))
        register_scripted("br_hi_fn", lambda _i, _c: HighResult(label="high"))
        register_scripted("br_lo_fn", lambda _i, _c: LowResult(label="low"))
        register_scripted("br_fin_fn", lambda _i, _c: FinalResult(summary="done"))

        class TruthyBranch(ForwardConstruct):
            check = Node.scripted("check", fn="br_a_fn", outputs=Confidence)
            hi = Node.scripted("hi", fn="br_hi_fn", outputs=HighResult)
            lo = Node.scripted("lo", fn="br_lo_fn", outputs=LowResult)
            fin = Node.scripted("fin", fn="br_fin_fn", outputs=FinalResult)

            def forward(self, topic):
                c = self.check(topic)
                if c:  # plain proxy as boolean — not a comparison
                    h = self.hi(c)
                else:
                    h = self.lo(c)
                return self.fin(h)

        pipe = TruthyBranch()
        # Should have a _BranchNode in the node list
        from neograph.forward import _BranchNode

        has_branch = any(isinstance(n, _BranchNode) for n in pipe.nodes)
        assert has_branch

    def test_sequential_branches_merge(self):
        """Two sequential if/else branches produce two _BranchNode sentinels
        (lines 873-875, 880-881)."""
        register_scripted("sb_a_fn", lambda _i, _c: Confidence(score=0.5))
        register_scripted("sb_b_fn", lambda _i, _c: HighResult(label="b"))
        register_scripted("sb_c_fn", lambda _i, _c: LowResult(label="c"))
        register_scripted("sb_d_fn", lambda _i, _c: HighResult(label="d"))
        register_scripted("sb_e_fn", lambda _i, _c: LowResult(label="e"))
        register_scripted("sb_f_fn", lambda _i, _c: FinalResult(summary="f"))

        class TwoBranches(ForwardConstruct):
            check1 = Node.scripted("check1", fn="sb_a_fn", outputs=Confidence)
            b = Node.scripted("b", fn="sb_b_fn", outputs=HighResult)
            c = Node.scripted("c", fn="sb_c_fn", outputs=LowResult)
            check2 = Node.scripted("check2", fn="sb_a_fn", outputs=Confidence)
            d = Node.scripted("d", fn="sb_d_fn", outputs=HighResult)
            e = Node.scripted("e", fn="sb_e_fn", outputs=LowResult)
            fin = Node.scripted("fin", fn="sb_f_fn", outputs=FinalResult)

            def forward(self, topic):
                c1 = self.check1(topic)
                if c1.score > 0.5:
                    r1 = self.b(c1)
                else:
                    r1 = self.c(c1)
                c2 = self.check2(r1)
                if c2.score > 0.7:
                    r2 = self.d(c2)
                else:
                    r2 = self.e(c2)
                return self.fin(r2)

        pipe = TwoBranches()
        branch_nodes = [n for n in pipe.nodes if isinstance(n, _BranchNode)]
        assert len(branch_nodes) == 2

    def test_sequential_branches_with_truthy_condition(self):
        """Two sequential branches where one uses a plain proxy as bool (line 845)."""
        register_scripted("sb2_a_fn", lambda _i, _c: Confidence(score=0.5))
        register_scripted("sb2_b_fn", lambda _i, _c: HighResult(label="b"))
        register_scripted("sb2_c_fn", lambda _i, _c: LowResult(label="c"))
        register_scripted("sb2_d_fn", lambda _i, _c: HighResult(label="d"))
        register_scripted("sb2_e_fn", lambda _i, _c: LowResult(label="e"))
        register_scripted("sb2_f_fn", lambda _i, _c: FinalResult(summary="f"))

        class TruthySequential(ForwardConstruct):
            check1 = Node.scripted("check1", fn="sb2_a_fn", outputs=Confidence)
            b = Node.scripted("b", fn="sb2_b_fn", outputs=HighResult)
            c = Node.scripted("c", fn="sb2_c_fn", outputs=LowResult)
            check2 = Node.scripted("check2", fn="sb2_a_fn", outputs=Confidence)
            d = Node.scripted("d", fn="sb2_d_fn", outputs=HighResult)
            e = Node.scripted("e", fn="sb2_e_fn", outputs=LowResult)
            fin = Node.scripted("fin", fn="sb2_f_fn", outputs=FinalResult)

            def forward(self, topic):
                c1 = self.check1(topic)
                if c1.score > 0.5:
                    r1 = self.b(c1)
                else:
                    r1 = self.c(c1)
                c2 = self.check2(r1)
                if c2:  # Plain proxy as bool — truthy check (line 845)
                    r2 = self.d(c2)
                else:
                    r2 = self.e(c2)
                return self.fin(r2)

        pipe = TruthySequential()
        branch_nodes = [n for n in pipe.nodes if isinstance(n, _BranchNode)]
        assert len(branch_nodes) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Part 12: self.loop() with sub-construct body members (neograph-e9zse.2)
# ═══════════════════════════════════════════════════════════════════════════


class CascadeClaim(BaseModel, frozen=True):
    cid: str
    text: str = ""


class CascadeClaimBatch(BaseModel, frozen=True):
    claims: list[CascadeClaim]


class CascadeDiagnosis(BaseModel, frozen=True):
    summary: str
    score: float = 0.0


class NestedVerdict(BaseModel, frozen=True):
    score: float


# Module-level exit conditions shared between the forward() twin and its
# declarative companion so `Loop.when` can be asserted by IDENTITY.
def _cascade_when(d):
    return d is None or d.score < 0.5


def _outer_loop_when(v):
    return v is None or v.score < 0.5


def _inner_loop_when(d):
    return d is None or d.score < 0.8


class TestLoopOverSubConstructBody:
    """neograph-e9zse.2: self.loop() body accepts sub-construct-producing
    members — deferred ``self.each(...)`` / ``self.loop(...)`` objects placed
    alongside node references — and traces to IR IDENTICAL to the declarative
    ``Construct(input=, output=, nodes=[Node, Construct|Each, Node]) | Loop``
    twin. ``Construct(nodes=...)`` already accepts ``list[Node | Construct]``;
    the tracer must not be stricter than the IR it targets.

    A nested ``self.each(...)`` is never __call__'d, so its ``over`` path is
    supplied at construction as a RAW dotted string whose root is either a
    loop-body peer's state field name (e.g. ``'get_claims.claims'``) or
    ``'neo_subgraph_input.<field>'`` — the exact two root forms the assembly
    validator accepts inside a sub-construct.

    Collector param-naming coupling (documented DX edge): inside the loop,
    the each construct's state field is ``field_name_for('each-verify') ==
    'each_verify'``, so a post-each consumer must declare
    ``inputs={'each_verify': dict[str, X]}`` (or name its @node param
    ``each_verify``).

    Mixed-body inputs-fill scope (refinement, 2026-07-11): the blanket
    "fill inputs=input_type when inputs is None" rule applies only to node
    members that PRECEDE the first construct member; a None-inputs node
    member AFTER a construct member fails loud at trace time.
    """

    @staticmethod
    def _register_cascade_fns(suffix):
        register_scripted(
            f"lsc_intake_{suffix}",
            lambda _i, _c: CascadeClaimBatch(claims=[CascadeClaim(cid="c1"), CascadeClaim(cid="c2")]),
        )
        register_scripted(f"lsc_get_claims_{suffix}", lambda batch, _c: batch)
        register_scripted(
            f"lsc_verify_{suffix}",
            lambda claim, _c: MatchResult(cluster_label=claim.cid, matched=[]),
        )
        register_scripted(
            f"lsc_collect_{suffix}",
            lambda _d, _c: CascadeDiagnosis(summary="done", score=1.0),
        )

    @staticmethod
    def _declarative_cascade_loop(suffix):
        """The exact declarative shape the traced cascade must emit
        (verified compiling end-to-end — see test_composition.py
        TestEachOnErrorCollect._build_parent for the Each'd-sub pattern)."""
        from neograph.modifiers import Loop

        each_sub = Construct(
            "each-verify",
            input=CascadeClaim,
            output=MatchResult,
            nodes=[
                Node.scripted(
                    "verify",
                    fn=f"lsc_verify_{suffix}",
                    inputs=CascadeClaim,
                    outputs=MatchResult,
                ),
            ],
        ) | Each(over="get_claims.claims", key="cid", on_error="raise")
        return Construct(
            "loop-get_claims-each-verify-collect",
            input=CascadeClaimBatch,
            output=CascadeDiagnosis,
            nodes=[
                Node.scripted(
                    "get_claims",
                    fn=f"lsc_get_claims_{suffix}",
                    inputs=CascadeClaimBatch,
                    outputs=CascadeClaimBatch,
                ),
                each_sub,
                Node.scripted(
                    "collect",
                    fn=f"lsc_collect_{suffix}",
                    inputs={"each_verify": dict[str, MatchResult]},
                    outputs=CascadeDiagnosis,
                ),
            ],
        ) | Loop(when=_cascade_when, max_iterations=3, on_exhaust="last")

    @classmethod
    def _assert_ir_identical(cls, traced, decl):
        """Structural IR equality, RECURSING into nested Construct members
        (extends TestSelfEachTracing._assert_ir_identical for mixed bodies)."""
        from neograph.modifiers import Loop

        if isinstance(decl, Construct):
            assert isinstance(traced, Construct), (
                f"expected a Construct member named {decl.name!r}, "
                f"got {type(traced).__name__}"
            )
            assert traced.name == decl.name

            # Modifier presence must match exactly
            assert traced.has_modifier(Each) == decl.has_modifier(Each)
            assert traced.has_modifier(Loop) == decl.has_modifier(Loop)

            if decl.has_modifier(Each):
                traced_each = traced.get_modifier(Each)
                decl_each = decl.get_modifier(Each)
                assert traced_each.over == decl_each.over
                assert traced_each.key == decl_each.key
                assert traced_each.on_error == decl_each.on_error

            if decl.has_modifier(Loop):
                traced_loop = traced.get_modifier(Loop)
                decl_loop = decl.get_modifier(Loop)
                assert traced_loop.when is decl_loop.when
                assert traced_loop.max_iterations == decl_loop.max_iterations
                assert traced_loop.on_exhaust == decl_loop.on_exhaust

            # Boundary ports (Construct.input / Construct.output — singular)
            assert traced.input is decl.input
            assert traced.output is decl.output

            # Recurse into members in order
            assert len(traced.nodes) == len(decl.nodes)
            for traced_member, decl_member in zip(traced.nodes, decl.nodes, strict=True):
                cls._assert_ir_identical(traced_member, decl_member)
        else:
            assert isinstance(traced, Node)
            assert not isinstance(traced, Construct)
            assert traced.name == decl.name
            assert traced.inputs == decl.inputs
            assert traced.outputs == decl.outputs
            assert traced.scripted_fn == decl.scripted_fn

    def test_cascade_loop_body_with_each_traces_ir_identical_to_declarative_twin(self):
        """(a) The cascade twin: self.loop(body=[node, deferred self.each(...),
        collector]) traces IR-identical to the declarative Construct|Loop
        containing an Each'd sub-construct."""
        self._register_cascade_fns("a")

        # Parity oracle: the declarative twin is legal IR (assembles+compiles).
        declarative_loop = self._declarative_cascade_loop("a")
        declarative_parent = Construct(
            "cascade-parent",
            nodes=[
                Node.scripted("intake", fn="lsc_intake_a", outputs=CascadeClaimBatch),
                declarative_loop,
            ],
        )
        compile(declarative_parent, **build_test_compile_kwargs())

        class Cascade(ForwardConstruct):
            intake = Node.scripted("intake", fn="lsc_intake_a", outputs=CascadeClaimBatch)
            get_claims = Node.scripted(
                "get_claims", fn="lsc_get_claims_a", outputs=CascadeClaimBatch
            )
            verify = Node.scripted("verify", fn="lsc_verify_a", outputs=MatchResult)
            collect = Node.scripted(
                "collect",
                fn="lsc_collect_a",
                inputs={"each_verify": dict[str, MatchResult]},
                outputs=CascadeDiagnosis,
            )

            def forward(self, topic):
                batch = self.intake(topic)
                return self.loop(
                    body=[
                        self.get_claims,
                        self.each(body=[self.verify], key="cid", over="get_claims.claims"),
                        self.collect,
                    ],
                    when=_cascade_when,
                    max_iterations=3,
                    on_exhaust="last",
                )(batch)

        pipeline = Cascade()

        # Two top-level entries: intake Node + the loop Construct. The nested
        # each construct lands INSIDE the loop's nodes, never at top level.
        assert len(pipeline.nodes) == 2
        assert isinstance(pipeline.nodes[0], Node)
        assert pipeline.nodes[0].name == "intake"

        self._assert_ir_identical(pipeline.nodes[1], declarative_loop)

    def test_loop_in_loop_traces_ir_identical_to_declarative_twin(self):
        """(b) Loop-in-loop: a deferred self.loop(...) as a body member traces
        IR-identical to the declarative nested Construct|Loop twin. The
        declarative companion is compiled FIRST so the parity target is
        proven in-suite, not assumed (refinement item 2)."""
        from neograph.modifiers import Loop

        register_scripted("lsc_seed_b", lambda _i, _c: Draft(content="seed"))
        register_scripted("lsc_refine_b", lambda d, _c: Draft(content=d.content + "+"))
        register_scripted(
            "lsc_polish_b", lambda d, _c: Draft(content=d.content + "*", score=0.9)
        )
        register_scripted("lsc_grade_b", lambda _d, _c: NestedVerdict(score=1.0))

        inner_loop = Construct(
            "loop-polish",
            input=Draft,
            output=Draft,
            nodes=[Node.scripted("polish", fn="lsc_polish_b", inputs=Draft, outputs=Draft)],
        ) | Loop(when=_inner_loop_when, max_iterations=2, on_exhaust="last")

        outer_loop = Construct(
            "loop-refine-loop-polish-grade",
            input=Draft,
            output=NestedVerdict,
            nodes=[
                Node.scripted("refine", fn="lsc_refine_b", inputs=Draft, outputs=Draft),
                inner_loop,
                Node.scripted(
                    "grade",
                    fn="lsc_grade_b",
                    inputs={"loop_polish": Draft},
                    outputs=NestedVerdict,
                ),
            ],
        ) | Loop(when=_outer_loop_when, max_iterations=3, on_exhaust="last")

        # Declarative loop-in-loop legality: compile+assemble companion.
        declarative_parent = Construct(
            "loopinloop-parent",
            nodes=[Node.scripted("seed", fn="lsc_seed_b", outputs=Draft), outer_loop],
        )
        compile(declarative_parent, **build_test_compile_kwargs())

        class LoopInLoop(ForwardConstruct):
            seed = Node.scripted("seed", fn="lsc_seed_b", outputs=Draft)
            refine = Node.scripted("refine", fn="lsc_refine_b", outputs=Draft)
            polish = Node.scripted("polish", fn="lsc_polish_b", outputs=Draft)
            grade = Node.scripted(
                "grade",
                fn="lsc_grade_b",
                inputs={"loop_polish": Draft},
                outputs=NestedVerdict,
            )

            def forward(self, topic):
                d = self.seed(topic)
                return self.loop(
                    body=[
                        self.refine,
                        self.loop(
                            body=[self.polish],
                            when=_inner_loop_when,
                            max_iterations=2,
                            on_exhaust="last",
                        ),
                        self.grade,
                    ],
                    when=_outer_loop_when,
                    max_iterations=3,
                    on_exhaust="last",
                )(d)

        pipeline = LoopInLoop()

        assert len(pipeline.nodes) == 2
        assert isinstance(pipeline.nodes[0], Node)
        assert pipeline.nodes[0].name == "seed"

        self._assert_ir_identical(pipeline.nodes[1], outer_loop)

    def test_none_inputs_node_after_construct_member_fails_loud(self):
        """(c) Fail-loud: a None-inputs node member AFTER a construct member
        cannot receive the blanket inputs=input_type fill (it consumes the
        construct's state field, not the loop port) — trace-time
        ConstructError telling the author to declare inputs= explicitly."""
        self._register_cascade_fns("c")

        class BadCollector(ForwardConstruct):
            intake = Node.scripted("intake", fn="lsc_intake_c", outputs=CascadeClaimBatch)
            get_claims = Node.scripted(
                "get_claims", fn="lsc_get_claims_c", outputs=CascadeClaimBatch
            )
            verify = Node.scripted("verify", fn="lsc_verify_c", outputs=MatchResult)
            # inputs=None AND placed after the each member: the loop port type
            # (CascadeClaimBatch) would be a silent misfill — must fail loud.
            collect = Node.scripted(
                "collect", fn="lsc_collect_c", outputs=CascadeDiagnosis
            )

            def forward(self, topic):
                batch = self.intake(topic)
                return self.loop(
                    body=[
                        self.get_claims,
                        self.each(body=[self.verify], key="cid", over="get_claims.claims"),
                        self.collect,
                    ],
                    when=_cascade_when,
                    max_iterations=3,
                    on_exhaust="last",
                )(batch)

        with pytest.raises(ConstructError, match="declare inputs"):
            BadCollector()
