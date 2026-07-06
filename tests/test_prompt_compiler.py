"""Regression tests for the exported fail-loud prompt primitives + DefaultPromptCompiler.

Ticket: neograph-hjwv (GH issue 5, layers 1-2). TDD-red step neograph-938w.6.

These pin the behavior the hjwv implementation plan creates: a NEW public module
``src/neograph/prompt.py`` exporting four composable primitives
(``substitute``, ``render_inputs``, ``inject_schema``, ``DefaultPromptCompiler``)
plus the typed ``PromptVarMissing`` error, all re-exported from ``neograph``.

Layer-2 node-internal DX. Per the plan the three-surface parity rule is EXEMPT
here (this is primitive logic, not IR behavior) — unit coverage of the primitives
+ one end-to-end compile()/run() proof + one opt-in proof is the right shape.

The five new public names do not exist yet, so every test that imports them is
red now (ImportError inside the test body -> FAILED). Each test also carries the
behavioral assertion that will fail if the primitive is implemented but WRONG.

Load-bearing case: BRACE-SAFETY. ``substitute(..., syntax='brace')`` must render a
template whose injected value (a JSON schema) contains literal ``{}`` intact — the
exact agent-stark regression that motivated this ticket (a naive ``str.format`` /
``.format_map`` crashes or mangles those braces; single-pass ``re.sub`` over a
token-only pattern does not).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from neograph import (
    ExecutionError,
    NeographError,
    compile,
    construct_from_module,
    describe_type,
    node,
    run,
)
from neograph.renderers import build_rendered_input
from tests.fakes import (
    StructuredFake,
    build_test_compile_kwargs,
)
from tests.schemas import Claims, RawText

# ═══════════════════════════════════════════════════════════════════════════
# 1. Public exports
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptPrimitiveExports:
    """The four primitives + PromptVarMissing are importable from `neograph`."""

    def test_public_names_importable_when_module_shipped(self):
        """`from neograph import substitute, render_inputs, inject_schema,
        DefaultPromptCompiler, PromptVarMissing` succeeds."""
        from neograph import (  # noqa: F401
            DefaultPromptCompiler,
            PromptVarMissing,
            inject_schema,
            render_inputs,
            substitute,
        )

    def test_prompt_var_missing_is_a_neograph_execution_error_when_defined(self):
        """PromptVarMissing is a typed runtime error in the neograph hierarchy
        (plan: subclasses ExecutionError)."""
        from neograph import PromptVarMissing

        assert issubclass(PromptVarMissing, NeographError)
        assert issubclass(PromptVarMissing, ExecutionError)


# ═══════════════════════════════════════════════════════════════════════════
# 2. BRACE-SAFETY — the load-bearing case (agent-stark regression)
# ═══════════════════════════════════════════════════════════════════════════


class TestSubstituteBraceSafety:
    """substitute(syntax='brace') must never crash or mangle literal braces in
    the injected values — single-pass, token-only substitution."""

    def test_injected_schema_braces_survive_when_syntax_is_brace(self):
        """A {json_schema} placeholder whose VALUE contains literal `{}` / JSON
        braces renders intact (the agent-stark bug: str.format would explode)."""
        from neograph import substitute

        schema_value = '{ "foo": "bar", "nested": { "n": 1 }, "empty": {} }'
        template = "Respond in JSON:\n{json_schema}\nEnd."
        out = substitute(template, {"json_schema": schema_value}, syntax="brace")

        assert out == f"Respond in JSON:\n{schema_value}\nEnd."
        # The injected braces are present and untouched.
        assert schema_value in out
        assert "{json_schema}" not in out

    def test_empty_braces_left_verbatim_when_not_a_token(self):
        """A bare `{}` is not a placeholder token and must pass through even
        under strict=True (it names no variable)."""
        from neograph import substitute

        template = "before {} after"
        assert substitute(template, {}, syntax="brace", strict=True) == "before {} after"

    def test_json_fragment_not_treated_as_token_when_brace_has_space_or_quote(self):
        """`{ "a": 1 }` (space/quote after brace) is not a var token — no lookup,
        no PromptVarMissing, rendered verbatim."""
        from neograph import substitute

        template = 'config = { "a": 1, "b": 2 }'
        assert substitute(template, {}, syntax="brace", strict=True) == template


# ═══════════════════════════════════════════════════════════════════════════
# 3. STRICTNESS — fail-loud by default, opt-out leaves token verbatim
# ═══════════════════════════════════════════════════════════════════════════


class TestSubstituteStrictness:
    """strict=True raises typed PromptVarMissing(var, available); strict=False
    leaves the unfilled token in place."""

    def test_raises_prompt_var_missing_with_structured_attrs_when_strict(self):
        """The {domain}-reaches-the-model class is fail-loud: a missing var raises
        PromptVarMissing carrying var:str and available:list[str] (sorted)."""
        from neograph import PromptVarMissing, substitute

        with pytest.raises(PromptVarMissing) as exc_info:
            substitute("Hi {name}!", {"zeta": "z", "alpha": "a"}, strict=True)

        err = exc_info.value
        assert err.var == "name"
        assert err.available == ["alpha", "zeta"]  # sorted

    def test_available_is_empty_sorted_list_when_no_vars(self):
        from neograph import PromptVarMissing, substitute

        with pytest.raises(PromptVarMissing) as exc_info:
            substitute("Hi {name}", {}, strict=True)
        assert exc_info.value.var == "name"
        assert exc_info.value.available == []

    def test_leaves_token_verbatim_when_not_strict(self):
        """strict=False is the explicit opt-out — the unfilled token survives."""
        from neograph import substitute

        assert substitute("Hi {name}", {}, syntax="brace", strict=False) == "Hi {name}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. SYNTAX — brace / dollar / Callable
# ═══════════════════════════════════════════════════════════════════════════


class TestSubstituteSyntax:
    """Engine-agnostic syntax: brace ({var}), dollar (${var}), pluggable Callable."""

    def test_brace_syntax_resolves_when_default(self):
        from neograph import substitute

        assert substitute("Hi {name}", {"name": "World"}) == "Hi World"

    def test_dollar_syntax_resolves_when_selected(self):
        """syntax='dollar' resolves ${var} and leaves bare {curly} untouched."""
        from neograph import substitute

        out = substitute("Hi ${name}, keep {curly}", {"name": "World"}, syntax="dollar")
        assert out == "Hi World, keep {curly}"

    def test_callable_syntax_resolves_when_custom_tokenizer(self):
        """A Callable syntax provides its own single-pass scanner while substitute
        keeps the resolver/strict policy.

        Contract: ``syntax(template, resolve)`` where ``resolve(var_name) -> str``
        is neograph's per-match resolver.
        """
        from neograph import substitute

        def angle(template: str, resolve):
            return re.sub(r"<<(\w+)>>", lambda m: resolve(m.group(1)), template)

        assert substitute("Hi <<name>>", {"name": "World"}, syntax=angle) == "Hi World"


# ═══════════════════════════════════════════════════════════════════════════
# 5. render_inputs — thin wrapper over the existing BAML rendering (reuse proof)
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderInputs:
    """render_inputs is the exported view of build_rendered_input(...).for_template_ref,
    NOT a re-implementation."""

    def test_matches_build_rendered_input_for_template_ref_when_dict_input(self):
        from neograph import render_inputs

        input_data = {"claim": RawText(text="the sky is blue"), "count": 3}
        expected = build_rendered_input(input_data).for_template_ref
        assert render_inputs(input_data) == expected


# ═══════════════════════════════════════════════════════════════════════════
# 6. inject_schema — rides describe_type
# ═══════════════════════════════════════════════════════════════════════════


class TestInjectSchema:
    """inject_schema sets vars['json_schema'] to describe_type(output_model)."""

    def test_sets_json_schema_via_describe_type_when_output_model_given(self):
        from neograph import inject_schema

        out = inject_schema({}, Claims)
        assert out["json_schema"] == describe_type(Claims)


# ═══════════════════════════════════════════════════════════════════════════
# 7. END-TO-END — DefaultPromptCompiler renders a file-ref template, zero app code
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaultPromptCompilerEndToEnd:
    """DefaultPromptCompiler(Path('prompts')) is the 90%-case file-ref handler —
    load a .md, render inputs, inject schema, substitute — with no app code."""

    def _write_prompt(self, prompt_dir: Path) -> None:
        prompt_dir.mkdir(parents=True, exist_ok=True)
        # {json_schema}'s VALUE (describe_type output) contains literal braces —
        # if the compiler used str.format this template would crash at run time.
        (prompt_dir / "greet.md").write_text(
            "Analyze the text: {seed}\n\nRespond per schema:\n{json_schema}\n"
        )

    def test_renders_file_ref_template_end_to_end_when_default_compiler(self, tmp_path):
        """compile(construct, prompt_compiler=DefaultPromptCompiler(prompts_dir))
        runs a think node whose file-ref prompt is loaded + rendered with zero app
        compiler code. Brace-safety holds through the full stack."""
        from neograph import DefaultPromptCompiler

        prompts = tmp_path / "prompts"
        self._write_prompt(prompts)

        import types

        mod = types.ModuleType("test_default_compiler_mod")

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="hello world")

        @node(outputs=Claims, mode="think", model="fast", prompt="greet")
        def analyze(seed: RawText) -> Claims: ...

        mod.seed = seed
        mod.analyze = analyze
        pipeline = construct_from_module(mod)

        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=DefaultPromptCompiler(prompts),
            **build_test_compile_kwargs(),
        )
        result = run(graph, input={"node_id": "e2e"})

        assert isinstance(result["analyze"], Claims)
        assert result["analyze"].items == ["ok"]

    def test_call_returns_rendered_message_list_when_invoked_directly(self, tmp_path):
        """The compiler satisfies the PromptCompiler protocol: __call__ loads the
        template, renders inputs, injects the schema, and substitutes — returning
        a message list with the placeholders resolved and schema braces intact."""
        from neograph import DefaultPromptCompiler

        prompts = tmp_path / "prompts"
        self._write_prompt(prompts)

        compiler = DefaultPromptCompiler(prompts)
        messages = compiler("greet", {"seed": RawText(text="hello world")}, output_model=Claims)

        assert isinstance(messages, list) and messages
        content = " ".join(
            m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user"
        )
        # placeholders resolved, none left verbatim
        assert "{seed}" not in content
        assert "{json_schema}" not in content
        # rendered input + schema present, schema braces survived
        assert "hello world" in content
        assert describe_type(Claims) in content
        assert "{" in content  # the schema's own literal braces


# ═══════════════════════════════════════════════════════════════════════════
# 8. OPT-IN proof — an existing custom prompt_compiler is untouched
# ═══════════════════════════════════════════════════════════════════════════


class TestExistingCompilerUnchanged:
    """DefaultPromptCompiler is opt-in: a consumer passing their OWN callable sees
    zero behavior change. (Control test — passes now AND after; proves the seam
    is not altered by the new primitives.)"""

    def test_custom_prompt_compiler_still_works_when_passed(self, tmp_path):
        received: dict[str, object] = {}

        def custom_compiler(template, data, **kw):
            received["template"] = template
            received["data"] = data
            return [{"role": "user", "content": f"custom::{template}"}]

        import types

        mod = types.ModuleType("test_custom_compiler_mod")

        @node(outputs=RawText)
        def seed2() -> RawText:
            return RawText(text="x")

        @node(outputs=Claims, mode="think", model="fast", prompt="mytemplate")
        def analyze2(seed2: RawText) -> Claims: ...

        mod.seed2 = seed2
        mod.analyze2 = analyze2
        pipeline = construct_from_module(mod)

        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["done"])),
            prompt_compiler=custom_compiler,
            **build_test_compile_kwargs(),
        )
        result = run(graph, input={"node_id": "optin"})

        assert received["template"] == "mytemplate"
        assert isinstance(result["analyze2"], Claims)
        assert result["analyze2"].items == ["done"]
