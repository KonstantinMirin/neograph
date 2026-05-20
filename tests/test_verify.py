"""Tests for verify_compiled() — post-compile structural verification.

TDD red: these tests define expected behavior for verify_compiled().
Written BEFORE implementation.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import (
    Construct,
    Node,
    compile,
    construct_from_functions,
    node,
)
from neograph.modifiers import Loop
from tests.fakes import build_test_compile_kwargs, register_scripted


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
        graph = compile(pipeline, **build_test_compile_kwargs())

        issues = verify_compiled(graph)
        assert issues == []

    def test_detects_missing_scripted_fn_after_registry_clear(self):
        """If the per-compile scripted snapshot is mutated, verify catches missing fn."""
        from neograph.verify import verify_compiled

        register_scripted("_vc_temp", lambda i, c: RawText(text="ok"))
        a = Node.scripted("temp-node", fn="_vc_temp", outputs=RawText)
        pipeline = Construct("temp", nodes=[a])
        graph = compile(pipeline, **build_test_compile_kwargs())

        # Post-§2: compile() captures a frozen snapshot into graph._neo_scripted.
        # Simulate "registry cleared after compile" by mutating the snapshot directly.
        snapshot = getattr(graph, "_neo_scripted", {})
        saved = snapshot.pop("_vc_temp", None)
        try:
            issues = verify_compiled(graph)
            scripted_issues = [i for i in issues if i.kind == "scripted_fn_missing"]
            assert len(scripted_issues) >= 1
            assert any("_vc_temp" in i.message for i in scripted_issues)
        finally:
            if saved is not None:
                snapshot["_vc_temp"] = saved

    def test_detects_missing_loop_condition(self):
        """Verify catches unregistered Loop condition post-compile."""
        from neograph.verify import verify_compiled
        from tests.fakes import _TEST_CONDITIONS as _condition_dict
        from tests.fakes import register_condition

        register_condition("_vc_cond", lambda d: d is None or d.text == "")
        register_scripted("_vc_loop", lambda i, c: RawText(text="ok"))

        a = Node.scripted("loop-node", fn="_vc_loop", outputs=RawText)
        a = a | Loop(when="_vc_cond", max_iterations=3)
        pipeline = Construct("loop-test", nodes=[a])
        graph = compile(pipeline, **build_test_compile_kwargs())

        # Remove the condition from the compiled graph's stashed dict to
        # simulate "condition deregistered after compile".
        compiled_conditions = getattr(graph, "_neo_conditions", {})
        if "_vc_cond" in compiled_conditions:
            saved = compiled_conditions.pop("_vc_cond")
        else:
            saved = _condition_dict.pop("_vc_cond")
        try:
            issues = verify_compiled(graph)
            cond_issues = [i for i in issues if i.kind == "condition_missing"]
            assert len(cond_issues) >= 1
        finally:
            _condition_dict["_vc_cond"] = saved

    def test_detects_missing_llm_factory(self):
        """Verify catches missing LLM factory for graphs with LLM nodes."""
        from neograph._llm_runtime import LlmRuntime
        from neograph.verify import verify_compiled

        # Node with mode=think requires LLM factory
        think_node = Node("llm-node", mode="think", outputs=Claims,
                          prompt="test/prompt", model="fast")
        pipeline = Construct("llm-test", nodes=[think_node])

        from tests.fakes import StructuredFake, configure_fake_llm

        llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["x"])))
        graph = compile(pipeline, **build_test_compile_kwargs(), **llm_kw)

        # Post-§2: clear the per-compile runtime snapshot to simulate
        # "LLM factory missing".
        saved_runtime = getattr(graph, "_neo_runtime", None)
        graph._neo_runtime = LlmRuntime(
            llm_factory=None,
            prompt_compiler=saved_runtime.prompt_compiler if saved_runtime else None,
            renderer=saved_runtime.renderer if saved_runtime else None,
            cost_callback=saved_runtime.cost_callback if saved_runtime else None,
        )
        try:
            issues = verify_compiled(graph)
            llm_issues = [i for i in issues if i.kind == "llm_factory_missing"]
            assert len(llm_issues) >= 1
        finally:
            graph._neo_runtime = saved_runtime

    def test_state_field_cross_check(self):
        """Every node output has a corresponding state field."""
        from neograph.verify import verify_compiled

        register_scripted("_vc_sf", lambda i, c: RawText(text="ok"))
        a = Node.scripted("producer", fn="_vc_sf", outputs=RawText)
        pipeline = Construct("sf-test", nodes=[a])
        graph = compile(pipeline, **build_test_compile_kwargs())

        issues = verify_compiled(graph)
        # Valid pipeline — no state field issues
        state_issues = [i for i in issues if i.kind == "state_field_missing"]
        assert state_issues == []

    def test_construct_is_stashed_on_compiled_graph(self):
        """compile() stashes _neo_construct for post-compile introspection."""
        register_scripted("_vc_stash", lambda i, c: RawText(text="ok"))
        a = Node.scripted("stash-test", fn="_vc_stash", outputs=RawText)
        pipeline = Construct("stash", nodes=[a])
        graph = compile(pipeline, **build_test_compile_kwargs())

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
        graph = compile(pipeline, **build_test_compile_kwargs())

        issues = verify_compiled(graph)
        assert issues == []
