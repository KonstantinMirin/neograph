"""Tests for ForwardConstruct if/else branching support.

Task neograph-w5z: Python `if` branching in forward() compiles to
LangGraph conditional edges via re-trace strategy (torch.fx pattern).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph import ForwardConstruct, Node, compile, run
from neograph.factory import register_scripted


# ═══════════════════════════════════════════════════════════════════════════
# Test schemas
# ═══════════════════════════════════════════════════════════════════════════


class RawText(BaseModel, frozen=True):
    text: str


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
            check = Node.scripted("br-check", fn="br_check", output=Confidence)
            high_path = Node.scripted("br-high", fn="br_high", output=HighResult)
            low_path = Node.scripted("br-low", fn="br_low", output=LowResult)

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
            check1 = Node.scripted("seq-check1", fn="seq_check1", output=Confidence)
            check2 = Node.scripted("seq-check2", fn="seq_check2", output=Confidence)
            a = Node.scripted("seq-a", fn="seq_a", output=HighResult)
            b = Node.scripted("seq-b", fn="seq_b", output=LowResult)
            c = Node.scripted("seq-c", fn="seq_c", output=HighResult)
            d = Node.scripted("seq-d", fn="seq_d", output=LowResult)

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
