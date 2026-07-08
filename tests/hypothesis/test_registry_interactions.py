"""Hypothesis property-based tests for test-side registry lifecycle.

Post-§2 (ticket ezqz): src/ no longer has a module-level registry. The
test-side `register_scripted` helper in `tests/fakes.py` writes into a
test-local dict that compile() consumes via the `scripted=` kwarg.

Tests in this file exercise the test-helper round-trip and the §2
multi-tenant compile() guarantee.
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from pydantic import BaseModel

from neograph import Construct, Node, compile, run
from neograph.errors import ConfigurationError
from tests.fakes import (
    build_test_compile_kwargs,
    lookup_scripted,
    register_scripted,
)

# ── Test models ───────────────────────────────────────────────────────────


class Input(BaseModel, frozen=True):
    text: str


class Output(BaseModel, frozen=True):
    result: str


# ── Strategies ────────────────────────────────────────────────────────────

st_fn_name = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
    min_size=3,
    max_size=15,
).filter(lambda s: s[0] != "_")  # no leading underscore


# ── Property tests ────────────────────────────────────────────────────────


class TestRegistryWriteReadContract:
    """register_scripted → lookup_scripted must round-trip."""

    @given(name=st_fn_name)
    @settings(max_examples=50)
    def test_register_then_lookup_returns_same_function(self, name):
        """Registered function is retrievable by name."""

        def sentinel(_i, _c):
            return Output(result="x")

        register_scripted(name, sentinel)
        assert lookup_scripted(name) is sentinel

    @given(name=st_fn_name)
    @settings(max_examples=30)
    def test_lookup_unregistered_raises_configuration_error(self, name):
        """Looking up an unregistered name raises ConfigurationError."""
        unique = f"__hyp_missing_{name}_{id(name):x}"
        with pytest.raises(ConfigurationError):
            lookup_scripted(unique)

    @given(name=st_fn_name)
    @settings(max_examples=30)
    def test_overwrite_replaces_previous(self, name):
        """Re-registering the same name replaces the function."""

        def fn1(_i, _c):
            return Output(result="first")

        def fn2(_i, _c):
            return Output(result="second")

        register_scripted(name, fn1)
        register_scripted(name, fn2)
        assert lookup_scripted(name) is fn2


class TestCompileRunContract:
    """Compiled graph is self-contained — registry mutations after compile don't affect it."""

    def test_run_succeeds_when_scripted_fn_deregistered_after_compile(self):
        """Compiled graph captures function references at compile time.

        Post-§2: the scripted_lookup is closure-captured at compile time.
        Mutations to the test-side `register_scripted` dict after compile
        don't break the compiled graph.
        """
        fn_name = "ephemeral_fn"
        register_scripted(fn_name, lambda _i, _c: Output(result="ok"))

        pipeline = Construct(
            "ephemeral",
            nodes=[
                Node.scripted("a", fn=fn_name, outputs=Output),
            ],
        )
        graph = compile(pipeline, **build_test_compile_kwargs())

        # Even after we deregister, the graph still works.
        from tests.fakes import _TEST_SCRIPTED

        _TEST_SCRIPTED.pop(fn_name, None)

        result = run(graph, input={"node_id": "test"})
        assert result["a"].result == "ok"

    def test_compile_fails_when_scripted_fn_not_registered(self):
        """compile() validates that scripted_fn names resolve to a callable."""
        pipeline = Construct(
            "missing-fn",
            nodes=[
                Node.scripted("a", fn="nonexistent_fn_xyz", outputs=Output),
            ],
        )
        with pytest.raises(ConfigurationError, match="not registered"):
            compile(pipeline, **build_test_compile_kwargs())


class TestMultiplePipelineNameCollision:
    """Two pipelines compiled in sequence must not interfere through shared state."""

    @given(data=st.data())
    @settings(max_examples=20)
    def test_two_pipelines_with_same_fn_names_dont_collide(self, data):
        """Compiling pipeline B doesn't break pipeline A's captured shims."""
        name = data.draw(st_fn_name)
        fn_a = f"pipe_a_{name}"
        fn_b = f"pipe_b_{name}"

        register_scripted(fn_a, lambda _i, _c: Output(result="from-a"))
        register_scripted(fn_b, lambda _i, _c: Output(result="from-b"))

        pipe_a = Construct(
            "pipe-a",
            nodes=[
                Node.scripted("a", fn=fn_a, outputs=Output),
            ],
        )
        pipe_b = Construct(
            "pipe-b",
            nodes=[
                Node.scripted("b", fn=fn_b, outputs=Output),
            ],
        )

        graph_a = compile(pipe_a, **build_test_compile_kwargs())
        graph_b = compile(pipe_b, **build_test_compile_kwargs())

        result_a = run(graph_a, input={"node_id": "a"})
        result_b = run(graph_b, input={"node_id": "b"})

        assert result_a["a"].result == "from-a"
        assert result_b["b"].result == "from-b"
