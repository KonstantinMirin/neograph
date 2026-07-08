"""Parity + behavior tests for the public ``compile_prompt`` (neograph-v569).

``compile_prompt`` promotes the graph's internal ``render_prompt``/``_compile_prompt``
inspection path into a first-class STANDALONE function that produces byte-identical
messages to what a compiled ``think`` node actually sends — the eval-parity unlock
(survey F4). Eval harnesses that today rebuild prompts outside the graph (a second
schema mechanism, a second renderer) call this and get the REAL prompt.

The load-bearing test is ``TestCompilePromptParity``: it captures the message list a
compiled think node hands its LLM and asserts ``compile_prompt`` reproduces it exactly.
"""

from __future__ import annotations

import types
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from neograph import (
    DefaultPromptCompiler,
    FromInput,
    compile,
    construct_from_module,
    node,
    run,
)
from tests.fakes import build_test_compile_kwargs
from tests.schemas import Claims, RawText


class CapturingFake:
    """StructuredFake that records the message list handed to ``invoke`` so a test
    can assert what the graph actually sent to the model."""

    def __init__(self, respond, captured: list | None = None):
        self._respond = respond
        self._model: type[BaseModel] | None = None
        self.captured: list = captured if captured is not None else []

    def with_structured_output(self, model: type[BaseModel], **kwargs) -> CapturingFake:
        clone = CapturingFake(self._respond, self.captured)
        clone._model = model
        return clone

    def bind(self, **kwargs: Any) -> CapturingFake:
        clone = CapturingFake(self._respond, self.captured)
        clone._model = self._model
        return clone

    def invoke(self, messages: list, **kwargs) -> Any:
        self.captured.append(messages)
        assert self._model is not None
        return self._respond(self._model)

    async def ainvoke(self, *a, **k) -> Any:
        return self.invoke(*a, **k)


def _run_think_node_capturing_messages(tmp_path, *, di_input: dict | None = None):
    """Compile + run a single think node backed by a DefaultPromptCompiler and
    return (captured_messages, prompts_dir, run_input) so a parity test can
    reconstruct the standalone call with the same compiler."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "analyze.md").write_text("Analyze: {seed}\n\nSchema:\n{json_schema}\n")

    mod = types.ModuleType("test_compile_prompt_mod")

    @node(outputs=RawText)
    def seed() -> RawText:
        return RawText(text="hello world")

    @node(outputs=Claims, mode="think", model="fast", prompt="analyze")
    def analyze(seed: RawText) -> Claims: ...

    mod.seed = seed
    mod.analyze = analyze
    pipeline = construct_from_module(mod)

    fake = CapturingFake(lambda m: m(items=["ok"]))
    graph = compile(
        pipeline,
        llm_factory=lambda tier: fake,
        prompt_compiler=DefaultPromptCompiler(prompts),
        **build_test_compile_kwargs(),
    )
    run_input = {"node_id": "e2e"}
    if di_input:
        run_input.update(di_input)
    run(graph, input=run_input)
    return fake.captured, prompts, run_input


class TestCompilePromptParity:
    """compile_prompt output == the messages a compiled think node sends."""

    def test_compile_prompt_matches_graph_messages_byte_for_byte(self, tmp_path):
        from neograph import compile_prompt

        captured, prompts, _ = _run_think_node_capturing_messages(tmp_path)
        assert len(captured) == 1
        graph_messages = captured[0]

        standalone = compile_prompt(
            "analyze",
            {"seed": RawText(text="hello world")},
            output_model=Claims,
            prompt_compiler=DefaultPromptCompiler(prompts),
        )

        assert standalone == graph_messages

    def test_di_inputs_layering_matches_graph(self, tmp_path):
        """A think node with a FromInput param resolves di_inputs into the template
        at runtime; compile_prompt reproduces the same messages when handed the
        same di_inputs map."""
        from neograph import compile_prompt

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "leaf.md").write_text("Domain: {domain}\n\nSchema:\n{json_schema}\n")

        mod = types.ModuleType("test_compile_prompt_di_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="leaf")
        def analyze(domain: Annotated[str, FromInput]) -> Claims: ...

        mod.analyze = analyze
        pipeline = construct_from_module(mod)

        fake = CapturingFake(lambda m: m(items=["ok"]))
        graph = compile(
            pipeline,
            llm_factory=lambda tier: fake,
            prompt_compiler=DefaultPromptCompiler(prompts),
            **build_test_compile_kwargs(),
        )
        run(graph, input={"domain": "oncology", "node_id": "leaf"})
        graph_messages = fake.captured[0]

        standalone = compile_prompt(
            "leaf",
            None,
            output_model=Claims,
            di_inputs={"domain": "oncology"},
            prompt_compiler=DefaultPromptCompiler(prompts),
        )
        assert standalone == graph_messages


class TestCompilePromptRoutesThroughSharedSeam:
    """Anti-duplication guard: compile_prompt MUST call the same internal
    ``_compile_prompt`` the graph uses, not a parallel reimplementation."""

    def test_compile_prompt_calls_internal_compile_prompt(self, tmp_path, monkeypatch):
        import neograph._llm_render as render_mod

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "greet.md").write_text("Hi {seed}\n{json_schema}\n")

        calls: list = []
        real = render_mod._compile_prompt

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real(*args, **kwargs)

        monkeypatch.setattr(render_mod, "_compile_prompt", spy)

        from neograph import compile_prompt

        compile_prompt(
            "greet",
            {"seed": RawText(text="x")},
            output_model=Claims,
            prompt_compiler=DefaultPromptCompiler(prompts),
        )
        assert len(calls) == 1


class TestCompilePromptTemplateSourceOverride:
    """The template-source override (template_text / loader) lets eval harnesses
    parameterize prompt VARIANTS without wiring a production compiler."""

    def test_template_text_override_renders_with_default_rendering(self):
        from neograph import compile_prompt, describe_type

        messages = compile_prompt(
            "ignored-name",
            {"seed": RawText(text="hi there")},
            output_model=Claims,
            template_text="Variant A: {seed}\n{json_schema}",
        )
        content = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "Variant A" in content
        assert "hi there" in content
        assert describe_type(Claims) in content

    def test_loader_override_selects_variant_by_name(self, tmp_path):
        from neograph import compile_prompt

        variants = tmp_path / "variants"
        variants.mkdir()
        (variants / "explain-v7.txt").write_text("v7: {seed}")
        (variants / "explain-v8.txt").write_text("v8 BETTER: {seed}")

        loader = DefaultPromptCompiler(variants, suffix=".txt").load_template
        messages = compile_prompt("explain-v8", {"seed": RawText(text="z")}, loader=loader)
        content = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "v8 BETTER" in content
        assert "v7" not in content

    def test_override_with_explicit_compiler_is_rejected(self, tmp_path):
        from neograph import compile_prompt

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "x.md").write_text("hi")
        with pytest.raises(Exception) as exc:
            compile_prompt(
                "x",
                {},
                prompt_compiler=DefaultPromptCompiler(prompts),
                template_text="Variant: {seed}",
            )
        assert "template" in str(exc.value).lower()

    def test_inline_template_needs_no_compiler(self):
        from neograph import compile_prompt

        messages = compile_prompt("Just say hi to ${seed}", {"seed": "Ada"})
        assert messages == [{"role": "user", "content": "Just say hi to Ada"}]
