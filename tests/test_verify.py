"""Tests for verify_compiled() — post-compile structural verification.

TDD red: these tests define expected behavior for verify_compiled().
Written BEFORE implementation.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    Node,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph.factory import register_scripted
from neograph.modifiers import Loop


class RawText(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]


class TestVerifyCompiled:
    """verify_compiled() validates structural integrity after compile()."""

    def test_valid_pipeline_returns_empty_issues(self):
        """A correctly wired pipeline produces no verification issues."""
        from neograph.verify import verify_compiled

        register_scripted("_vc_a", lambda i, c: RawText(text="ok"))
        register_scripted("_vc_b", lambda i, c: Claims(items=["x"]))

        a = Node.scripted("seed", fn="_vc_a", outputs=RawText)
        b = Node.scripted("classify", fn="_vc_b",
                          inputs={"seed": RawText}, outputs=Claims)
        pipeline = Construct("valid", nodes=[a, b])
        graph = compile(pipeline)

        issues = verify_compiled(graph)
        assert issues == []

    def test_detects_missing_scripted_fn_after_registry_clear(self):
        """If registry is cleared after compile, verify catches missing fn."""
        from neograph.verify import verify_compiled
        from neograph._registry import registry

        register_scripted("_vc_temp", lambda i, c: RawText(text="ok"))
        a = Node.scripted("temp-node", fn="_vc_temp", outputs=RawText)
        pipeline = Construct("temp", nodes=[a])
        graph = compile(pipeline)

        # Clear the registry after compile — simulates session reset
        saved = dict(registry.scripted)
        del registry.scripted["_vc_temp"]
        try:
            issues = verify_compiled(graph)
            scripted_issues = [i for i in issues if i.kind == "scripted_fn_missing"]
            assert len(scripted_issues) >= 1
            assert any("_vc_temp" in i.message for i in scripted_issues)
        finally:
            registry.scripted.update(saved)

    def test_detects_missing_loop_condition(self):
        """Verify catches unregistered Loop condition post-compile."""
        from neograph.verify import verify_compiled
        from neograph import register_condition
        from neograph._registry import registry

        register_condition("_vc_cond", lambda d: d is None or d.text == "")
        register_scripted("_vc_loop", lambda i, c: RawText(text="ok"))

        a = Node.scripted("loop-node", fn="_vc_loop", outputs=RawText)
        a = a | Loop(when="_vc_cond", max_iterations=3)
        pipeline = Construct("loop-test", nodes=[a])
        graph = compile(pipeline)

        # Remove the condition after compile
        saved = registry.condition.pop("_vc_cond")
        try:
            issues = verify_compiled(graph)
            cond_issues = [i for i in issues if i.kind == "condition_missing"]
            assert len(cond_issues) >= 1
        finally:
            registry.condition["_vc_cond"] = saved

    def test_detects_missing_llm_factory(self):
        """Verify catches missing LLM factory for graphs with LLM nodes."""
        from neograph.verify import verify_compiled

        # Node with mode=think requires LLM factory
        think_node = Node("llm-node", mode="think", outputs=Claims,
                          prompt="test/prompt", model="fast")
        pipeline = Construct("llm-test", nodes=[think_node])

        # Compile requires LLM factory — need to set one up then clear
        from neograph._llm import _llm_factory
        from neograph import configure_llm
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["x"])))
        graph = compile(pipeline)

        # Clear LLM factory after compile
        import neograph._llm as llm_mod
        saved_factory = llm_mod._llm_factory
        llm_mod._llm_factory = None
        try:
            issues = verify_compiled(graph)
            llm_issues = [i for i in issues if i.kind == "llm_factory_missing"]
            assert len(llm_issues) >= 1
        finally:
            llm_mod._llm_factory = saved_factory

    def test_state_field_cross_check(self):
        """Every node output has a corresponding state field."""
        from neograph.verify import verify_compiled

        register_scripted("_vc_sf", lambda i, c: RawText(text="ok"))
        a = Node.scripted("producer", fn="_vc_sf", outputs=RawText)
        pipeline = Construct("sf-test", nodes=[a])
        graph = compile(pipeline)

        issues = verify_compiled(graph)
        # Valid pipeline — no state field issues
        state_issues = [i for i in issues if i.kind == "state_field_missing"]
        assert state_issues == []

    def test_construct_is_stashed_on_compiled_graph(self):
        """compile() stashes _neo_construct for post-compile introspection."""
        register_scripted("_vc_stash", lambda i, c: RawText(text="ok"))
        a = Node.scripted("stash-test", fn="_vc_stash", outputs=RawText)
        pipeline = Construct("stash", nodes=[a])
        graph = compile(pipeline)

        assert hasattr(graph, "_neo_construct")
        assert graph._neo_construct is pipeline

    def test_verify_compiled_with_node_decorator(self):
        """Works with @node decorated pipelines."""
        from neograph.verify import verify_compiled

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="hello")

        @node(outputs=Claims)
        def classify(seed: RawText) -> Claims:
            return Claims(items=[seed.text])

        pipeline = construct_from_functions("deco-test", [seed, classify])
        graph = compile(pipeline)

        issues = verify_compiled(graph)
        assert issues == []
