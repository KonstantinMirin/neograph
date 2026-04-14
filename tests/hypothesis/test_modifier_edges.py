"""Hypothesis property-based tests for modifier interaction edge cases.

Focuses on degenerate and boundary conditions that are hard to discover
with example-based tests: empty collections, skip-all loops, type-shifting
merges, and modifier combinations at their limits.
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings, assume
from pydantic import BaseModel, Field

from neograph import (
    Construct,
    Each,
    Loop,
    Node,
    Oracle,
    compile,
    run,
)
from neograph.factory import register_scripted


# ── Test models ───────────────────────────────────────────────────────────

class Item(BaseModel, frozen=True):
    key: str
    value: int = 0

class Collection(BaseModel, frozen=True):
    items: list[Item] = Field(default_factory=list)

class Result(BaseModel, frozen=True):
    text: str
    count: int = 0

class Draft(BaseModel, frozen=True):
    content: str
    score: float = 0.0
    iteration: int = 0


# ── Each edge cases ──────────────────────────────────────────────────────

class TestEachBoundaryConditions:
    """Each modifier at boundary conditions: 0, 1, many items."""

    @given(n_items=st.integers(min_value=0, max_value=10))
    @settings(max_examples=20)
    def test_each_fanout_with_n_items_produces_n_keys(self, n_items):
        """Each over N items always produces a dict with exactly N keys."""
        items = [Item(key=f"k{i}", value=i) for i in range(n_items)]

        register_scripted(
            "each_src",
            lambda _i, _c: Collection(items=items),
        )
        register_scripted(
            "each_proc",
            lambda _i, _c: Result(text=f"processed-{getattr(_i, 'key', '?')}"),
        )

        pipeline = Construct("each-boundary", nodes=[
            Node.scripted("src", fn="each_src", outputs=Collection),
            Node.scripted("proc", fn="each_proc", inputs=Item, outputs=Result)
            | Each(over="src.items", key="key"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        proc_result = result.get("proc")
        if n_items == 0:
            assert proc_result == {} or proc_result is None
        else:
            assert isinstance(proc_result, dict)
            assert len(proc_result) == n_items

    @given(n_items=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10)
    def test_each_produces_correct_keys(self, n_items):
        """Each result dict keys match the each.key field values."""
        items = [Item(key=f"item-{i}", value=i) for i in range(n_items)]

        register_scripted(
            "key_src",
            lambda _i, _c: Collection(items=items),
        )
        register_scripted(
            "key_proc",
            lambda _i, _c: Result(text=_i.key if hasattr(_i, 'key') else "?"),
        )

        pipeline = Construct("each-keys", nodes=[
            Node.scripted("src", fn="key_src", outputs=Collection),
            Node.scripted("proc", fn="key_proc", inputs=Item, outputs=Result)
            | Each(over="src.items", key="key"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        proc_result = result["proc"]
        expected_keys = {f"item-{i}" for i in range(n_items)}
        assert set(proc_result.keys()) == expected_keys


# ── Loop edge cases ──────────────────────────────────────────────────────

class TestLoopBoundaryConditions:
    """Loop modifier at boundary conditions: immediate exit, max iterations, skip-all."""

    def test_loop_exits_immediately_when_condition_false_on_first_check(self):
        """Loop with when=always_false runs the body once then exits."""
        call_count = [0]

        def loop_body(input_data, config):
            call_count[0] += 1
            return Draft(content="done", score=1.0, iteration=call_count[0])

        register_scripted("imm_seed", lambda _i, _c: Draft(content="seed", score=0.0))
        register_scripted("imm_loop", loop_body)

        pipeline = Construct("loop-imm-exit", nodes=[
            Node.scripted("seed", fn="imm_seed", outputs=Draft),
            Node.scripted("loop", fn="imm_loop", inputs=Draft, outputs=Draft)
            | Loop(when=lambda d: d is None, max_iterations=10),  # False after first run (d is not None)
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert call_count[0] == 1  # body ran once, condition was False, loop exited

    @given(max_iter=st.integers(min_value=1, max_value=5))
    @settings(max_examples=10)
    def test_loop_respects_max_iterations_with_stop(self, max_iter):
        """Loop with on_exhaust='stop' exits at max_iterations without error."""
        call_count = [0]

        def counting_body(input_data, config):
            call_count[0] += 1
            return Draft(content=f"iter-{call_count[0]}", score=0.0)

        register_scripted("max_seed", lambda _i, _c: Draft(content="seed"))
        register_scripted("max_loop", counting_body)

        call_count[0] = 0
        pipeline = Construct("loop-max", nodes=[
            Node.scripted("seed", fn="max_seed", outputs=Draft),
            Node.scripted("loop", fn="max_loop", inputs=Draft, outputs=Draft)
            | Loop(when=lambda d: True, max_iterations=max_iter, on_exhaust="last"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})
        assert call_count[0] == max_iter

    @given(max_iter=st.integers(min_value=1, max_value=3))
    @settings(max_examples=5)
    def test_loop_raises_on_exhaust_error(self, max_iter):
        """Loop with on_exhaust='error' (default) raises ExecutionError at max."""
        from neograph.errors import ExecutionError

        register_scripted("err_seed", lambda _i, _c: Draft(content="seed"))
        register_scripted("err_loop", lambda _i, _c: Draft(content="x"))

        pipeline = Construct("loop-err", nodes=[
            Node.scripted("seed", fn="err_seed", outputs=Draft),
            Node.scripted("loop", fn="err_loop", inputs=Draft, outputs=Draft)
            | Loop(when=lambda d: True, max_iterations=max_iter),  # default on_exhaust="error"
        ])
        graph = compile(pipeline)
        with pytest.raises(ExecutionError, match="max_iterations"):
            run(graph, input={"node_id": "test"})

    def test_loop_skip_when_fires_every_iteration(self):
        """skip_when that always fires + Loop(on_exhaust='stop'):
        counter increments via skip path, loop exits at max without error."""
        call_count = [0]

        def always_skipped(input_data, config):
            call_count[0] += 1
            return Draft(content="should-not-run")

        register_scripted("skip_seed", lambda _i, _c: Draft(content="seed"))
        register_scripted("skip_loop", always_skipped)

        pipeline = Construct("loop-skip-all", nodes=[
            Node.scripted("seed", fn="skip_seed", outputs=Draft),
            Node.scripted("loop", fn="skip_loop", inputs=Draft, outputs=Draft)
            | Loop(when=lambda d: True, max_iterations=3, on_exhaust="last"),
        ])

        # Add skip_when to the loop node — always skip
        loop_node = pipeline.nodes[1]
        loop_node_with_skip = loop_node.model_copy(update={
            "skip_when": lambda _input: True,
            "skip_value": lambda _input: Draft(content="skipped"),
        })
        pipeline_with_skip = Construct("loop-skip-all", nodes=[
            pipeline.nodes[0],
            loop_node_with_skip,
        ])

        graph = compile(pipeline_with_skip)
        result = run(graph, input={"node_id": "test"})

        # Body should never be called (always skipped)
        assert call_count[0] == 0
        # But loop still exits (counter increments via skip path)
        loop_result = result.get("loop")
        assert loop_result is not None


# ── Oracle edge cases ────────────────────────────────────────────────────

class TestOracleBoundaryConditions:
    """Oracle modifier with type-shifting merge."""

    def test_oracle_merge_fn_wrong_type_raises_execution_error(self):
        """Oracle merge_fn returning wrong type is caught at runtime."""
        from neograph.errors import ExecutionError
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(lambda tier: StructuredFake(lambda model: model(text="gen", count=1)))

        def type_shift_merge(variants, config):
            """Returns Draft instead of Result — wrong type."""
            return Draft(content=f"merged-{len(variants)}")

        register_scripted("shift_merge", type_shift_merge)

        pipeline = Construct("oracle-shift", nodes=[
            Node("gen", mode="think", outputs=Result, model="fast", prompt="test",
                 llm_config={"output_strategy": "structured"})
            | Oracle(n=2, merge_fn="shift_merge"),
        ])
        graph = compile(pipeline)
        with pytest.raises(ExecutionError, match="wrong type"):
            run(graph, input={"node_id": "test"})

    def test_oracle_merge_fn_correct_type_succeeds(self):
        """Oracle merge_fn returning the declared output type succeeds."""
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(lambda tier: StructuredFake(lambda model: model(text="gen", count=1)))

        def correct_merge(variants, config):
            return Result(text=f"merged-{len(variants)}", count=len(variants))

        register_scripted("correct_merge", correct_merge)

        pipeline = Construct("oracle-correct", nodes=[
            Node("gen", mode="think", outputs=Result, model="fast", prompt="test",
                 llm_config={"output_strategy": "structured"})
            | Oracle(n=2, merge_fn="correct_merge"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})

        assert isinstance(result["gen"], Result)
        assert result["gen"].count == 2
