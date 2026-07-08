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

    def test_returns_empty_dict_when_input_is_none(self):
        """An all-DI leaf think node yields input_data=None (no upstream). The
        primitive owns a total dict contract: None -> {} so the downstream
        ``inject_schema``/``substitute`` never sees a non-dict (neograph-4tsd:
        ``dict(None)`` TypeError before the fix)."""
        from neograph import render_inputs

        assert render_inputs(None) == {}

    def test_returns_empty_dict_when_single_nondict_value(self):
        """Single-type (non-dict) inputs have no bindable var name; template-ref
        prompts address vars by name, so a nameless value contributes no vars.
        The contract is uniform with None: any non-dict view collapses to {}
        (before the fix these returned a bare str -> ``dict(str)`` ValueError)."""
        from neograph import render_inputs

        assert render_inputs("bare string") == {}
        assert render_inputs(RawText(text="x")) == {}


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
        (prompt_dir / "greet.md").write_text("Analyze the text: {seed}\n\nRespond per schema:\n{json_schema}\n")

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

    def test_all_di_think_node_runs_when_input_is_none(self, tmp_path):
        """A leaf think node whose params are ALL DI (no upstream node) yields
        input_data=None through the compile()+run() seam. DefaultPromptCompiler
        must render it (schema-only vars) rather than crashing on ``dict(None)``.

        This is the agent-stark ``{domain}`` leaf shape (neograph-4tsd): a think
        node driven purely by ``run(input=...)`` DI, no upstream producer."""
        from typing import Annotated

        from neograph import DefaultPromptCompiler, FromInput

        prompts = tmp_path / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        # No upstream var to reference — the only var is the injected schema.
        (prompts / "leaf.md").write_text("Analyze the domain.\n\nRespond per schema:\n{json_schema}\n")

        import types

        mod = types.ModuleType("test_all_di_think_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="leaf")
        def analyze(domain: Annotated[str, FromInput]) -> Claims: ...

        mod.analyze = analyze
        pipeline = construct_from_module(mod)

        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=DefaultPromptCompiler(prompts),
            **build_test_compile_kwargs(),
        )
        result = run(graph, input={"domain": "finance", "node_id": "leaf"})

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
        content = " ".join(m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user")
        # placeholders resolved, none left verbatim
        assert "{seed}" not in content
        assert "{json_schema}" not in content
        # rendered input + schema present, schema braces survived
        assert "hello world" in content
        assert describe_type(Claims) in content
        assert "{" in content  # the schema's own literal braces


# ═══════════════════════════════════════════════════════════════════════════
# 7b. di_inputs — resolved FromInput/FromConfig values reach the template
#     (neograph-euyh, GH issue 5 layer 3). The agent-stark {domain} incident.
# ═══════════════════════════════════════════════════════════════════════════


def _user_content(messages: list) -> str:
    """Join the user-role message content(s) into one string."""
    return " ".join(m["content"] for m in messages if isinstance(m, dict) and m.get("role") == "user")


class TestDefaultPromptCompilerDiInputs:
    """build_vars exposes di_inputs as a BASE layer; upstream outputs shadow it."""

    def test_di_input_var_renders_when_no_upstream_output(self):
        """A ``{domain}`` placeholder fed purely by di_inputs renders — the
        agent-stark leaf shape where a FromInput param is the only var."""
        from neograph import DefaultPromptCompiler

        compiler = DefaultPromptCompiler(lambda name: "The domain is {domain}.")
        messages = compiler("t", None, di_inputs={"domain": "oncology"})

        content = _user_content(messages)
        assert "The domain is oncology." in content
        assert "{domain}" not in content

    def test_upstream_output_shadows_di_input_on_name_collision(self):
        """Precedence decision (neograph-euyh): on a name collision the upstream
        node OUTPUT wins over the di_input — the node-local, dataflow-derived
        value is more specific than run-wide ambient DI context.

        This is the zero-behavior-change rule: di_inputs only fills names NOT
        already produced upstream, so no existing pipeline's {name} binding
        changes meaning when a FromInput param happens to collide."""
        from neograph import DefaultPromptCompiler

        compiler = DefaultPromptCompiler(lambda name: "value={domain}")
        # 'domain' is present BOTH as an upstream output (input_data) and a
        # di_input. The output must win.
        messages = compiler("t", {"domain": "FROM_OUTPUT"}, di_inputs={"domain": "FROM_DI"})

        content = _user_content(messages)
        assert "value=FROM_OUTPUT" in content
        assert "FROM_DI" not in content

    def test_build_vars_layers_di_inputs_under_rendered_inputs(self):
        """Unit-level precedence proof on build_vars directly: di_inputs is the
        base, render_inputs(input_data) overlays it."""
        from neograph import DefaultPromptCompiler

        compiler = DefaultPromptCompiler(lambda name: "x")
        vars = compiler.build_vars({"topic": "sky"}, di_inputs={"domain": "finance", "topic": "SHADOWED"})
        assert vars["domain"] == "finance"  # di_input survives (no collision)
        assert vars["topic"] == "sky"  # rendered output shadows di_input

    def test_di_inputs_none_preserves_total_dict_contract(self):
        """di_inputs=None collapses to {} — no crash, mirrors render_inputs(None)."""
        from neograph import DefaultPromptCompiler

        compiler = DefaultPromptCompiler(lambda name: "no vars")
        messages = compiler("t", None, di_inputs=None)
        assert _user_content(messages) == "no vars"


class TestDiInputReachesModelEndToEnd:
    """The production incident, fixed: a think node references a FromInput param
    in its template and the RESOLVED value reaches the model — no seed node."""

    def test_from_input_value_reaches_model_via_template_when_compiler_opts_in(self, tmp_path):
        """agent-stark shape end-to-end: ``domain: Annotated[str, FromInput]`` on a
        think node whose ``{domain}`` template placeholder is filled with the value
        from ``run(input={'domain': ...})`` — with NO scripted seed node copying
        run-input onto the bus."""
        from typing import Annotated

        from neograph import DefaultPromptCompiler, FromInput

        prompts = tmp_path / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "leaf.md").write_text("Analyze the {domain} domain.\n\nRespond per schema:\n{json_schema}\n")

        # Wrap DefaultPromptCompiler to capture the messages handed to the LLM.
        base = DefaultPromptCompiler(prompts)
        captured: dict[str, object] = {}

        def capturing_compiler(*a, **kw):
            messages = base(*a, **kw)
            captured["messages"] = messages
            captured["di_inputs"] = kw.get("di_inputs")
            return messages

        import types

        mod = types.ModuleType("test_di_reaches_model_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="leaf")
        def analyze(domain: Annotated[str, FromInput]) -> Claims: ...

        mod.analyze = analyze
        pipeline = construct_from_module(mod)

        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=capturing_compiler,
            **build_test_compile_kwargs(),
        )
        result = run(graph, input={"domain": "oncology", "node_id": "leaf"})

        assert isinstance(result["analyze"], Claims)
        # The resolved FromInput value reached the compiler as di_inputs...
        assert captured["di_inputs"] == {"domain": "oncology"}
        # ...and is rendered into the user message the model received.
        content = _user_content(captured["messages"])  # type: ignore[arg-type]
        assert "Analyze the oncology domain." in content
        assert "{domain}" not in content

    def test_from_input_dropped_when_compiler_does_not_opt_in(self, tmp_path):
        """Opt-in preserved: a compiler that does NOT declare di_inputs never
        receives it (the introspection gate). The literal ``{domain}`` would ship
        unresolved — this is exactly what lint flags as unresolvable."""
        from typing import Annotated

        from neograph import FromInput

        received: dict[str, object] = {}

        # Explicit params only — no **kwargs, no di_inputs. The gate must not
        # pass di_inputs to this compiler.
        def strict_compiler(
            template, input_data, *, output_model=None, output_schema=None, config=None, node_name="", llm_config=None
        ):
            received["saw_di_inputs"] = False
            return [{"role": "user", "content": "static"}]

        import types

        mod = types.ModuleType("test_no_optin_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="leaf")
        def analyze(domain: Annotated[str, FromInput]) -> Claims: ...

        mod.analyze = analyze
        pipeline = construct_from_module(mod)

        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=strict_compiler,
            **build_test_compile_kwargs(),
        )
        # Must not raise a TypeError from an unexpected di_inputs kwarg.
        result = run(graph, input={"domain": "oncology", "node_id": "leaf"})
        assert isinstance(result["analyze"], Claims)
        assert received["saw_di_inputs"] is False


class TestDiInputReachesAgentModelEndToEnd:
    """neograph-jhz4 (TDD RED): the SAME production incident as the think-mode
    ``TestDiInputReachesModelEndToEnd`` above, but for an AGENT-mode node — a
    ``domain: Annotated[str, FromInput]`` param whose ``{domain}`` template
    placeholder must be filled with the value from ``run(input={'domain': ...})``,
    with NO scripted seed node.

    ``euyh`` wired ``di_inputs`` for think mode only. Agent/act nodes compile to
    the ReAct cycle (``_agent_cycle.py`` / ``_tool_loop.py``) and bypass
    ``_dispatch._inject_di_inputs``, so the resolved DI value never rides ``config``
    into the cycle's ``_compile_prompt`` — the ``di_inputs`` column is ``None`` and
    ``{domain}`` ships unresolved. RED now; passes once
    ``_agent_cycle._turn_prep_kwargs`` calls the same injector (neograph-jhz4).

    Three-surface parity is EXEMPT by construction: ``di_inputs`` is sourced from
    ``node._param_res``, populated only by ``@node`` ``_classify_di_params``.
    Declarative/programmatic nodes carry empty ``_param_res``, so
    ``_inject_di_inputs`` is a no-op for them — matching the think-mode precedent.
    Hence this E2E is ``@node``-built only.
    """

    def test_from_input_value_reaches_agent_model_via_template_when_no_seed_node(self, tmp_path):
        """agent-stark shape end-to-end on an ``@node(mode='agent')`` node: the
        resolved ``FromInput`` value reaches the agent cycle's prompt compiler as
        ``di_inputs`` and is rendered into the user message the model receives —
        via a TEMPLATE-REF ``{domain}`` prompt (inline ``${domain}`` never gets the
        di_inputs column), with a ``domain`` name distinct from every upstream
        field (so upstream-output-shadows-di_inputs precedence cannot mask the
        path) and NO scripted seed node copying run-input onto the bus."""
        from typing import Annotated

        from neograph import (
            DefaultPromptCompiler,
            FromInput,
            Tool,
            construct_from_functions,
        )
        from tests.fakes import FakeTool, ReActFake, register_tool_factory

        prompts = tmp_path / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "explore.md").write_text("Analyze the {domain} domain.\n")

        # strict=False so the RED run COMPLETES (an unresolved {domain} ships
        # verbatim instead of raising PromptVarMissing) and the failure surfaces
        # as the BEHAVIORAL di_inputs assertion below, not a crash. In GREEN the
        # injected di_inputs fills {domain} regardless of strict.
        base = DefaultPromptCompiler(prompts, strict=False)
        captured: dict[str, object] = {}

        def capturing_compiler(*a, **kw):
            captured["di_inputs"] = kw.get("di_inputs")
            messages = base(*a, **kw)
            captured["messages"] = messages
            return messages

        lookup = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: lookup)

        fake = ReActFake(
            tool_calls=[
                [{"name": "lookup", "args": {"q": "x"}, "id": "c1"}],
                [],  # stop — final structured turn
            ],
            final=lambda m: m(items=["done"]),
            output_model=Claims,
        )

        @node(
            mode="agent",
            outputs=Claims,
            model="reason",
            prompt="explore",
            tools=[Tool(name="lookup", budget=2)],
        )
        def explore(domain: Annotated[str, FromInput]) -> Claims: ...

        graph = compile(
            construct_from_functions("p", [explore]),
            **build_test_compile_kwargs(
                llm_factory=lambda tier: fake,
                prompt_compiler=capturing_compiler,
            ),
        )
        result = run(graph, input={"domain": "oncology", "node_id": "explore"})

        assert isinstance(result["explore"], Claims)
        # The resolved FromInput value reached the agent cycle's prompt compiler...
        assert captured["di_inputs"] == {"domain": "oncology"}
        # ...and is rendered into the user message the model received.
        content = _user_content(captured["messages"])  # type: ignore[arg-type]
        assert "Analyze the oncology domain." in content
        assert "{domain}" not in content


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
