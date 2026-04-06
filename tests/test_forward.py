"""ForwardConstruct tests — base class, tracer, compilation, branching, loops.

Merged from test_e2e_piarch_ready.py (3 classes) and
test_forward_construct_branching.py (8 classes).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph import Construct, Each, ForwardConstruct, Node, compile, run
from neograph.factory import register_scripted
from tests.fakes import StructuredFake, configure_fake_llm
from tests.schemas import RawText, Claims, ClassifiedClaims, Clusters, MatchResult


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

    def test_forward_construct_discovers_node_attributes(self):
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

    def test_forward_construct_is_construct_subclass(self):
        """isinstance(pipeline, Construct) is True."""
        from neograph import Construct, ForwardConstruct, Node

        class Simple(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)

            def forward(self, topic):
                return self.a(topic)

        pipeline = Simple()
        assert isinstance(pipeline, Construct)
        assert isinstance(pipeline, ForwardConstruct)

    def test_forward_construct_without_forward_raises(self):
        """Subclass without forward() method raises TypeError."""
        from neograph import ForwardConstruct, Node

        class NoForward(ForwardConstruct):
            a = Node.scripted("a", fn="a_fn", outputs=RawText)

        with pytest.raises(TypeError, match="must override forward"):
            NoForward()


# ═══════════════════════════════════════════════════════════════════════════
# ForwardConstruct — Task 2: symbolic proxy tracer
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardConstructTracer:
    """Task neograph-3us: symbolic proxy tracer for straight-line forward()."""

    def test_trace_straight_line_two_nodes(self):
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

    def test_trace_three_nodes_chain(self):
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

    def test_trace_preserves_node_identity(self):
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

    def test_trace_skips_unused_nodes(self):
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

    def test_forward_construct_compile_and_run(self):
        """Full end-to-end: ForwardConstruct with 2 scripted nodes, compile, run."""
        from neograph import ForwardConstruct, Node, compile, run
        from neograph.factory import register_scripted

        register_scripted("fc_extract", lambda input_data, config: RawText(text="hello world"))
        register_scripted("fc_split", lambda input_data, config: Claims(items=["claim-1", "claim-2"]))

        class ScriptedPipeline(ForwardConstruct):
            extract = Node.scripted("fc-extract", fn="fc_extract", outputs=RawText)
            split = Node.scripted("fc-split", fn="fc_split", outputs=Claims)

            def forward(self, topic):
                raw = self.extract(topic)
                return self.split(raw)

        pipeline = ScriptedPipeline()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fc-test-001"})

        assert isinstance(result["fc_split"], Claims)
        assert result["fc_split"].items == ["claim-1", "claim-2"]
        assert isinstance(result["fc_extract"], RawText)
        assert result["fc_extract"].text == "hello world"

    def test_forward_construct_compile_with_produce_mode(self):
        """ForwardConstruct with a produce node + FakeLLM."""
        from neograph import ForwardConstruct, Node, compile, run
        from neograph.factory import register_scripted

        register_scripted("fc_prep", lambda input_data, config: RawText(text="topic"))

        fake = StructuredFake(lambda model: model(items=["classified-a", "classified-b"]))
        configure_fake_llm(lambda tier: fake)

        class ProducePipeline(ForwardConstruct):
            prep = Node.scripted("fc-prep", fn="fc_prep", outputs=RawText)
            classify = Node("fc-classify", mode="produce", outputs=Claims, prompt="rw/classify", model="fast")

            def forward(self, topic):
                raw = self.prep(topic)
                return self.classify(raw)

        pipeline = ProducePipeline()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fc-test-002"})

        assert isinstance(result["fc_classify"], Claims)
        assert result["fc_classify"].items == ["classified-a", "classified-b"]

    def test_forward_construct_equivalent_to_declarative(self):
        """Same pipeline as ForwardConstruct and Construct(nodes=[...]) — identical output."""
        from neograph import Construct, ForwardConstruct, Node, compile, run
        from neograph.factory import register_scripted

        register_scripted("fc_equiv_a", lambda input_data, config: RawText(text="extracted"))
        register_scripted("fc_equiv_b", lambda input_data, config: Claims(items=["x", "y"]))

        node_a = Node.scripted("fc-equiv-a", fn="fc_equiv_a", outputs=RawText)
        node_b = Node.scripted("fc-equiv-b", fn="fc_equiv_b", outputs=Claims)

        # Declarative
        declarative = Construct("fc-equiv-test", nodes=[node_a, node_b])
        graph_decl = compile(declarative)
        result_decl = run(graph_decl, input={"node_id": "equiv-001"})

        # ForwardConstruct
        class ForwardPipeline(ForwardConstruct):
            a = node_a
            b = node_b

            def forward(self, topic):
                raw = self.a(topic)
                return self.b(raw)

        forward_pipe = ForwardPipeline()
        graph_fwd = compile(forward_pipe)
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

    def test_proxy_attribute_access_returns_child_proxy(self):
        """classified.confidence returns a child proxy with dotted name."""
        from neograph.forward import _Proxy

        parent = _Proxy(source_node=None, name="classified")
        child = parent.confidence
        assert child._neo_name == "classified.confidence"
        assert child._neo_source is parent._neo_source

    def test_proxy_chained_attribute_access(self):
        """classified.items.severity chains produce correct names."""
        from neograph.forward import _Proxy

        parent = _Proxy(source_node=None, name="result")
        child = parent.items.severity
        assert child._neo_name == "result.items.severity"

    def test_proxy_internal_attrs_raise(self):
        """Accessing _neo_* attributes raises AttributeError, not child proxy."""
        from neograph.forward import _Proxy

        proxy = _Proxy(source_node=None, name="x")
        with pytest.raises(AttributeError):
            proxy._neo_nonexistent

    def test_proxy_comparison_returns_condition_proxy(self):
        """proxy < 0.7 returns a _ConditionProxy, not a plain bool."""
        from neograph.forward import _ConditionProxy, _Proxy

        proxy = _Proxy(source_node=None, name="score")
        result = proxy < 0.7
        assert isinstance(result, _ConditionProxy)
        assert result._op == "<"
        assert result._right == 0.7

    def test_proxy_all_comparison_operators(self):
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

    def test_proxy_bool_outside_tracing_raises(self):
        """Using proxy in boolean context outside tracing raises TypeError."""
        from neograph.forward import _Proxy

        proxy = _Proxy(source_node=None, name="x")
        with pytest.raises(TypeError, match="boolean context"):
            bool(proxy)

    def test_condition_proxy_bool_outside_tracing_raises(self):
        """Using condition in boolean context outside tracing raises TypeError."""
        from neograph.forward import _ConditionProxy, _Proxy

        proxy = _Proxy(source_node=None, name="x")
        cond = proxy < 0.5
        with pytest.raises(TypeError, match="boolean context"):
            bool(cond)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Tracer branch recording
# ═══════════════════════════════════════════════════════════════════════════


class TestTracerBranchRecording:
    """_Tracer records branch decisions during forward() tracing."""

    def test_tracer_records_branch_point(self):
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

    def test_tracer_respects_branch_decisions(self):
        """Pre-configured branch decisions override the default True."""
        from neograph.forward import _ConditionProxy, _Proxy, _Tracer

        tracer = _Tracer(branch_decisions={0: False})
        proxy = _Proxy(source_node=None, name="score", tracer=tracer)
        cond = _ConditionProxy(proxy, "<", 0.7)
        cond._neo_tracer = tracer

        result = tracer.record_branch(cond)
        assert result is False

    def test_proxy_bool_delegates_to_tracer(self):
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

    def test_branch_pipeline_instantiates(self):
        """Pipeline with if/else in forward() can be instantiated."""
        Pipeline = self._make_branching_pipeline()
        pipeline = Pipeline()
        # Should have nodes: check + branch metadata
        assert any(n.name == "br-check" for n in pipeline.nodes)

    def test_branch_pipeline_compiles(self):
        """Pipeline with if/else compiles to a LangGraph graph."""
        Pipeline = self._make_branching_pipeline()
        pipeline = Pipeline()
        graph = compile(pipeline)
        assert graph is not None

    def test_branch_true_arm(self):
        """When condition is true at runtime, high_path runs."""
        # Override to produce score > 0.5
        register_scripted(
            "br_check",
            lambda input_data, config: Confidence(score=0.9),
        )
        Pipeline = self._make_branching_pipeline()
        pipeline = Pipeline()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "br-true-test"})
        assert "br_high" in result
        assert result["br_high"].label == "high-confidence"

    def test_branch_false_arm(self):
        """When condition is false at runtime, low_path runs."""
        Pipeline = self._make_branching_pipeline()
        pipeline = Pipeline()
        # Override AFTER pipeline instantiation so compile-time tracing
        # used the default, but the runtime scripted fn returns low score
        register_scripted(
            "br_check",
            lambda input_data, config: Confidence(score=0.2),
        )
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "br-false-test"})
        assert "br_low" in result
        assert result["br_low"].label == "low-confidence"


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: Sequential if-blocks (two branches)
# ═══════════════════════════════════════════════════════════════════════════


class TestSequentialBranches:
    """Two sequential if-blocks in forward()."""

    def test_two_sequential_branches(self):
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
        graph = compile(pipeline)
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

    def test_too_many_branches_raises(self):
        """9 branches in forward() raises an error."""
        # We can't easily create 9 real branches in a single forward(),
        # but we can test that the tracer enforces the limit.
        from neograph.forward import _Tracer

        tracer = _Tracer()
        # Simulate 9 branch recordings
        from neograph.forward import _BranchPoint, _ConditionProxy, _Proxy

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

    def test_condition_captures_attribute_path(self):
        """result.score > 0.5 captures the full attribute path."""
        from neograph.forward import _ConditionProxy, _Proxy

        proxy = _Proxy(source_node=None, name="out_of_check")
        cond = proxy.score > 0.5
        assert isinstance(cond, _ConditionProxy)
        assert cond._left._neo_name == "out_of_check.score"
        assert cond._op == ">"
        assert cond._right == 0.5

    def test_deep_attribute_chain_in_condition(self):
        """result.items.first.severity < 3 captures deep chain."""
        from neograph.forward import _ConditionProxy, _Proxy

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

    def test_for_loop_over_proxy_attr(self):
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
        graph = compile(pipeline)
        assert graph is not None

    def test_for_loop_literal_range_unrolls(self):
        """for i in range(3): self.step(x) traces step once (dedup). No Each."""
        register_scripted(
            "range_step",
            lambda input_data, config: RawText(text="stepped"),
        )

        class RangePipeline(ForwardConstruct):
            step = Node.scripted("range-step", fn="range_step", outputs=RawText)

            def forward(self, topic):
                for i in range(3):
                    self.step(topic)
                return topic

        pipeline = RangePipeline()
        # step should appear once (dedup), and should NOT have Each modifier
        step_node = [n for n in pipeline.nodes if isinstance(n, Node) and n.name == "range-step"]
        assert len(step_node) == 1
        assert not step_node[0].has_modifier(Each)

    def test_for_loop_each_modifier_attached(self):
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

    def test_for_loop_with_post_loop_node(self):
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

    def test_try_block_traces_normally(self):
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

    def test_except_block_catches_tracing_error(self):
        """try body raises a real Python error (division by zero) before a
        node call -> except body calls a fallback node -> fallback is recorded."""
        register_scripted(
            "te_fallback",
            lambda input_data, config: ErrorResult(message="fallback"),
        )

        class FallbackPipeline(ForwardConstruct):
            fallback = Node.scripted(
                "te-fallback", fn="te_fallback", outputs=ErrorResult,
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

    def test_try_except_with_proxy_operations(self):
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

    def test_try_except_documented_limitation(self):
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
        assert "te-rescue" not in node_names, (
            "v1 limitation: except-body nodes are dead code during tracing"
        )
        assert "te-final" in node_names, "post-try/except nodes must be traced"

        # The pipeline should still compile successfully
        graph = compile(pipeline)
        assert graph is not None
