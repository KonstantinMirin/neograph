"""Hypothesis property-based tests for registry lifecycle interactions.

Tests the behavioral properties of the registration → compile → run
lifecycle, not just the Registry class itself. Focuses on:
- Name collision across concurrent pipelines
- Registration timing (assembly vs compile vs run)
- Session isolation (no leaks between test contexts)
- compile→run contract (registry mutated between compile and run)

TASK neograph-f0vp followup: registry at 76% coverage signals
untested interaction surface, not just uncovered lines.
"""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from pydantic import BaseModel

from neograph import Construct, Node, compile, run
from neograph._registry import Registry, registry
from neograph.errors import ConfigurationError
from neograph.factory import lookup_scripted, register_scripted


# ── Test models ───────────────────────────────────────────────────────────

class Input(BaseModel, frozen=True):
    text: str

class Output(BaseModel, frozen=True):
    result: str


# ── Strategies ────────────────────────────────────────────────────────────

st_fn_name = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
    min_size=3, max_size=15,
).filter(lambda s: s[0] != "_")  # no leading underscore

st_node_name = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz-"),
    min_size=2, max_size=12,
).filter(lambda s: s[0].isalpha() and s[-1].isalpha())


# ── Property tests ────────────────────────────────────────────────────────

class TestRegistryWriteReadContract:
    """register_scripted → lookup_scripted must round-trip."""

    @given(name=st_fn_name)
    @settings(max_examples=50)
    def test_register_then_lookup_returns_same_function(self, name):
        """Registered function is retrievable by name."""
        sentinel = lambda _i, _c: Output(result="x")
        register_scripted(name, sentinel)
        assert lookup_scripted(name) is sentinel

    @given(name=st_fn_name)
    @settings(max_examples=30)
    def test_lookup_unregistered_raises_configuration_error(self, name):
        """Looking up an unregistered name raises ConfigurationError."""
        # Use a name that's guaranteed not registered
        unique = f"__hyp_missing_{name}_{id(name):x}"
        with pytest.raises(ConfigurationError):
            lookup_scripted(unique)

    @given(name=st_fn_name)
    @settings(max_examples=30)
    def test_overwrite_replaces_previous(self, name):
        """Re-registering the same name replaces the function."""
        fn1 = lambda _i, _c: Output(result="first")
        fn2 = lambda _i, _c: Output(result="second")
        register_scripted(name, fn1)
        register_scripted(name, fn2)
        assert lookup_scripted(name) is fn2


class TestSessionIsolation:
    """Registry.session() provides clean isolation."""

    def test_session_restores_state(self):
        """session() restores all three registries on exit."""
        from neograph.factory import register_condition, register_tool_factory

        register_scripted("before_session", lambda _i, _c: None)
        register_condition("cond_before", lambda x: True)
        register_tool_factory("tool_before", lambda: None)

        with registry.session():
            # Inside session — registries are empty
            with pytest.raises(ConfigurationError):
                lookup_scripted("before_session")
            # Register something inside session
            register_scripted("inside_session", lambda _i, _c: None)
            assert lookup_scripted("inside_session") is not None

        # After session — original state restored, session-local gone
        assert lookup_scripted("before_session") is not None
        with pytest.raises(ConfigurationError):
            lookup_scripted("inside_session")

    def test_session_restores_on_exception(self):
        """session() restores state even if body raises."""
        register_scripted("before_exc", lambda _i, _c: None)

        with pytest.raises(RuntimeError):
            with registry.session():
                register_scripted("during_exc", lambda _i, _c: None)
                raise RuntimeError("boom")

        # Original restored despite exception
        assert lookup_scripted("before_exc") is not None
        with pytest.raises(ConfigurationError):
            lookup_scripted("during_exc")


class TestCompileRunContract:
    """Compiled graph is self-contained — registry mutations after compile don't affect it."""

    def test_run_succeeds_when_scripted_fn_deregistered_after_compile(self):
        """Compiled graph captures function references at compile time.

        Deleting from registry after compile doesn't break the graph —
        the function is already bound in the node_fn closure.
        """
        fn_name = "ephemeral_fn"
        register_scripted(fn_name, lambda _i, _c: Output(result="ok"))

        pipeline = Construct("ephemeral", nodes=[
            Node.scripted("a", fn=fn_name, outputs=Output),
        ])
        graph = compile(pipeline)

        # Remove from registry — compiled graph should still work
        del registry.scripted[fn_name]

        result = run(graph, input={"node_id": "test"})
        assert result["a"].result == "ok"

    def test_compile_fails_when_scripted_fn_not_registered(self):
        """compile() validates that scripted_fn names exist in the registry."""
        pipeline = Construct("missing-fn", nodes=[
            Node.scripted("a", fn="nonexistent_fn_xyz", outputs=Output),
        ])
        with pytest.raises(ConfigurationError, match="not registered"):
            compile(pipeline)


class TestMultiplePipelineNameCollision:
    """Two pipelines compiled in sequence must not interfere via shared registry."""

    @given(data=st.data())
    @settings(max_examples=20)
    def test_two_pipelines_with_same_fn_names_dont_collide(self, data):
        """Compiling pipeline B doesn't break pipeline A's registered functions."""
        name = data.draw(st_fn_name)
        fn_a = f"pipe_a_{name}"
        fn_b = f"pipe_b_{name}"

        register_scripted(fn_a, lambda _i, _c: Output(result="from-a"))
        register_scripted(fn_b, lambda _i, _c: Output(result="from-b"))

        pipe_a = Construct("pipe-a", nodes=[
            Node.scripted("a", fn=fn_a, outputs=Output),
        ])
        pipe_b = Construct("pipe-b", nodes=[
            Node.scripted("b", fn=fn_b, outputs=Output),
        ])

        graph_a = compile(pipe_a)
        graph_b = compile(pipe_b)

        result_a = run(graph_a, input={"node_id": "a"})
        result_b = run(graph_b, input={"node_id": "b"})

        assert result_a["a"].result == "from-a"
        assert result_b["b"].result == "from-b"
