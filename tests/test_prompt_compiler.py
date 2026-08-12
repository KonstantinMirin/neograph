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
    describe_value,
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

    def test_keys_a_bare_value_by_its_type_name(self):
        """A bare (non-dict) input is KEYED by its type name, not dropped.

        Changed by neograph-l2a7w. The old contract collapsed any non-dict view to
        {}, which silently removed the node's only input from the template
        namespace -- a quiet seam of the same family as the bug this ticket fixes.
        Keying it makes the value addressable and keeps the mapping total; `None`
        still yields {} because there is nothing to name."""
        from neograph import render_inputs

        assert render_inputs("bare string") == {"str": "bare string"}
        assert render_inputs(RawText(text="x")) == {"RawText": describe_value(RawText(text="x"))}
        assert render_inputs(None) == {}


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


# ═══════════════════════════════════════════════════════════════════════════
# 8. Loader convenience (.txt suffix) + message-shaping recipe (node_name)
#    neograph-rndl / survey F3.2-3.3.
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaultPromptCompilerSuffix:
    """The dir-loader defaults to `{name}.md`; consumers with `.txt` template
    dirs pass `suffix='.txt'` instead of hand-rolling a callable loader."""

    def test_dir_loader_loads_txt_when_suffix_given(self, tmp_path):
        from neograph import DefaultPromptCompiler

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "greet.txt").write_text("Hello {seed}")

        compiler = DefaultPromptCompiler(prompts, suffix=".txt")
        assert compiler.load_template("greet") == "Hello {seed}"

    def test_dir_loader_defaults_to_md_when_no_suffix(self, tmp_path):
        from neograph import DefaultPromptCompiler

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "greet.md").write_text("Hi {seed}")

        compiler = DefaultPromptCompiler(prompts)
        assert compiler.load_template("greet") == "Hi {seed}"

    def test_txt_template_renders_end_to_end(self, tmp_path):
        from neograph import DefaultPromptCompiler

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "greet.txt").write_text("Analyze: {seed}\n\n{json_schema}\n")

        compiler = DefaultPromptCompiler(prompts, suffix=".txt")
        messages = compiler("greet", {"seed": RawText(text="hi there")}, output_model=Claims)
        content = _user_content(messages)
        assert "hi there" in content
        assert describe_type(Claims) in content


class TestRenderMessagesReceivesNodeName:
    """render_messages is handed the node_name so per-node role shaping (piarch's
    explore -> single user message; else system + user with a node-specific line)
    is a ~10-line override. The graph passes node.name at runtime."""

    def test_render_messages_override_receives_node_name_on_direct_call(self, tmp_path):
        from neograph import DefaultPromptCompiler

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "explore.txt").write_text("Explore {seed}")
        (prompts / "score.txt").write_text("Score {seed}")

        seen: list[str] = []

        class RoleCompiler(DefaultPromptCompiler):
            def render_messages(self, template_text, vars, *, node_name=""):
                seen.append(node_name)
                from neograph.prompt import substitute

                body = substitute(template_text, vars, strict=self.strict, syntax=self.syntax)
                if node_name == "explore":
                    return [{"role": "user", "content": body}]
                return [
                    {"role": "system", "content": f"You are the {node_name} node."},
                    {"role": "user", "content": body},
                ]

        compiler = RoleCompiler(prompts, suffix=".txt")

        explore_msgs = compiler("explore", {"seed": RawText(text="x")}, node_name="explore")
        score_msgs = compiler("score", {"seed": RawText(text="y")}, node_name="score")

        assert seen == ["explore", "score"]
        assert [m["role"] for m in explore_msgs] == ["user"]
        assert [m["role"] for m in score_msgs] == ["system", "user"]
        assert "score node" in score_msgs[0]["content"]

    def test_node_name_reaches_override_at_runtime(self, tmp_path):
        """End-to-end: the compiled think node passes node.name into the compiler,
        so the override's node_name branch fires during run()."""
        import types

        from neograph import DefaultPromptCompiler

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "analyze.txt").write_text("Analyze {seed}\n\n{json_schema}\n")

        captured: list[str] = []

        class RoleCompiler(DefaultPromptCompiler):
            def render_messages(self, template_text, vars, *, node_name=""):
                captured.append(node_name)
                from neograph.prompt import substitute

                body = substitute(template_text, vars, strict=self.strict, syntax=self.syntax)
                return [
                    {"role": "system", "content": f"node={node_name}"},
                    {"role": "user", "content": body},
                ]

        mod = types.ModuleType("test_node_name_runtime_mod")

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="hello")

        @node(outputs=Claims, mode="think", model="fast", prompt="analyze")
        def analyze(seed: RawText) -> Claims: ...

        mod.seed = seed
        mod.analyze = analyze
        pipeline = construct_from_module(mod)

        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=RoleCompiler(prompts, suffix=".txt"),
            **build_test_compile_kwargs(),
        )
        result = run(graph, input={"node_id": "e2e"})

        assert result["analyze"].items == ["ok"]
        assert "analyze" in captured


class TestPromptCompilerReceivesOneShape:
    """neograph-l2a7w — every value handed to a prompt_compiler is prompt-ready
    TEXT, on every channel.

    Today the shape depends on which call site invoked the compiler: the
    think/agent paths pre-render their input, while the Oracle merge_prompt path
    (`_oracle._merge_prompt_input`) hands the SAME compiler raw Pydantic models,
    and `di_inputs` are never rendered at all. A compiler written the obvious way
    (`getattr(value, 'text', '')`) silently yields an empty payload on whichever
    channel it did not expect, and the model answers coherently about nothing.

    Design: docs/design/prompt-compiler-input-shape-2026-08-11.md
    """

    @staticmethod
    def _capturing_compiler(calls: list) -> object:
        """A prompt_compiler that records what it was handed, per call."""

        def compiler(template, input_data, *, node_name="", di_inputs=None, **kw):
            calls.append(
                {
                    "template": template,
                    "node_name": node_name,
                    "data": input_data,
                    "di_inputs": di_inputs,
                }
            )
            return [{"role": "user", "content": "compiled"}]

        return compiler

    @staticmethod
    def _shape(value) -> str:
        return type(value).__name__

    def test_merge_prompt_hands_the_compiler_text_like_the_think_path_does(self):
        """One compiler, one pipeline, two invocations — the think generation call
        and the Oracle merge call — must agree on the shape of the SAME upstream.

        Fails today: the think call gets `seed` as a rendered str, the merge call
        gets the raw RawText model, and `variants` arrives as a raw list of Claims.
        """
        import types

        calls: list = []

        mod = types.ModuleType("test_l2a7w_merge_shape_mod")

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="a prose diagnosis, 5334 chars in the real incident")

        @node(
            outputs=Claims,
            mode="think",
            model="fast",
            prompt="extract",
            ensemble_n=2,
            merge_prompt="merge",
            merge_model="fast",
        )
        def analyze(seed: RawText) -> Claims: ...

        mod.seed = seed
        mod.analyze = analyze
        pipeline = construct_from_module(mod)

        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=self._capturing_compiler(calls),
            **build_test_compile_kwargs(),
        )
        run(graph, input={"node_id": "l2a7w"})

        think_calls = [c for c in calls if c["template"] == "extract"]
        merge_calls = [c for c in calls if c["template"] == "merge"]
        assert think_calls, f"no generation call captured; saw {[c['template'] for c in calls]}"
        assert merge_calls, f"no merge call captured; saw {[c['template'] for c in calls]}"

        think_seed = think_calls[0]["data"]["seed"]
        merge_seed = merge_calls[0]["data"]["seed"]

        assert isinstance(think_seed, str), f"think path handed {self._shape(think_seed)}, expected rendered text"
        assert isinstance(merge_seed, str), (
            f"merge path handed {self._shape(merge_seed)} for the SAME upstream the "
            f"think path rendered to {self._shape(think_seed)} — one compiler cannot "
            f"serve two shapes without an isinstance dance (neograph-l2a7w)"
        )

        variants = merge_calls[0]["data"]["variants"]
        assert isinstance(variants, str), (
            f"merge path handed variants as {self._shape(variants)}; every value a "
            f"prompt_compiler receives is prompt-ready text"
        )

    def test_di_inputs_reach_the_compiler_as_text_when_a_model_is_bundled(self):
        """`di_inputs` share the template var namespace with rendered upstream
        outputs, so they obey the same rule.

        Fails today: a bundled FromInput model arrives as a live BaseModel, so a
        `{ctx}` placeholder ships a Pydantic repr while `{seed}` ships BAML.
        """
        import types
        from typing import Annotated

        from pydantic import BaseModel

        from neograph import FromInput

        calls: list = []

        class RunCtx(BaseModel):
            node_id: str
            project_root: str

        mod = types.ModuleType("test_l2a7w_di_shape_mod")

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="hello")

        @node(outputs=Claims, mode="think", model="fast", prompt="extract")
        def analyze(seed: RawText, ctx: Annotated[RunCtx, FromInput]) -> Claims: ...

        mod.seed = seed
        mod.analyze = analyze
        pipeline = construct_from_module(mod)

        graph = compile(
            pipeline,
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=self._capturing_compiler(calls),
            **build_test_compile_kwargs(),
        )
        run(graph, input={"node_id": "l2a7w", "project_root": "/tmp"})

        analyze_calls = [c for c in calls if c["template"] == "extract"]
        assert analyze_calls, "no think call captured"
        di = analyze_calls[0]["di_inputs"] or {}
        assert "ctx" in di, f"expected a bundled ctx di_input; saw {sorted(di)}"

        assert isinstance(di["ctx"], str), (
            f"di_inputs handed ctx as {self._shape(di['ctx'])}; it shares the template "
            f"var namespace with rendered upstream outputs and obeys the same rule "
            f"(neograph-l2a7w)"
        )


class TestOneRenderingRule:
    """neograph-l2a7w — the properties the fix rests on, beyond the reported bug.

    The reproduction (TestPromptCompilerReceivesOneShape) proves the defect is
    gone. These pin the invariants that keep a FOURTH partial implementation of
    the rendering rule from appearing.
    """

    def test_rendering_twice_equals_rendering_once(self):
        """Rung 0: the ladder is idempotent.

        This is what lets a call site render where the node's renderer= is in
        scope AND the seam render again. Without it the invariant would be
        'nobody may render twice', which is unenforceable across consumer code --
        the public render_inputs is exported and the docs teach calling it.
        """
        from neograph._rendered import Rendered
        from neograph.renderers import to_rendered

        once = to_rendered(RawText(text="prose"), None)
        twice = to_rendered(once, None)

        assert isinstance(once, Rendered)
        assert twice is once, "a second render must be a no-op, not a re-render"
        # And it holds through the public primitive a consumer compiler calls.
        from neograph import render_inputs

        view = render_inputs({"seed": RawText(text="prose")})
        assert render_inputs(view) == view

    def test_rendered_text_refuses_attribute_access_loudly(self):
        """The acceptance criterion: the obvious getattr must not yield ''.

        getattr-with-default and hasattr swallow ONLY AttributeError, so the
        error deliberately is not one. A dunder probe still gets AttributeError
        so copy/pickle/deepcopy keep working on what is otherwise a str.
        """
        import copy
        import pickle

        from neograph._rendered import Rendered
        from neograph.errors import PromptInputError

        r = Rendered("the prose the model needed")

        with pytest.raises(PromptInputError):
            getattr(r, "text", "")
        with pytest.raises(PromptInputError):
            hasattr(r, "model_dump")
        assert not issubclass(PromptInputError, AttributeError)

        # Behaves exactly like str everywhere else.
        assert copy.deepcopy(r) == r
        assert pickle.loads(pickle.dumps(r)) == r
        assert f"{r}" == str(r) and isinstance(f"{r}", str)

    def test_every_channel_hands_the_compiler_rendered_text(self):
        """Totality across channels: node inputs, di_inputs, a bare single-type
        value, and the Oracle merge payload all arrive Rendered."""
        import types
        from typing import Annotated

        from neograph import FromInput
        from neograph._rendered import Rendered

        seen: list[dict] = []

        def compiler(template, input_data, *, di_inputs=None, **kw):
            seen.append({"data": input_data, "di": di_inputs or {}})
            return [{"role": "user", "content": "x"}]

        mod = types.ModuleType("test_l2a7w_totality_mod")

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="prose")

        @node(
            outputs=Claims,
            mode="think",
            model="fast",
            prompt="extract",
            ensemble_n=2,
            merge_prompt="merge",
            merge_model="fast",
        )
        def analyze(seed: RawText, domain: Annotated[str, FromInput]) -> Claims: ...

        mod.seed = seed
        mod.analyze = analyze
        graph = compile(
            construct_from_module(mod),
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["ok"])),
            prompt_compiler=compiler,
            **build_test_compile_kwargs(),
        )
        run(graph, input={"node_id": "t", "domain": "oncology"})

        assert seen, "compiler never invoked"
        for call in seen:
            for key, value in call["data"].items():
                assert isinstance(value, Rendered), f"{key} arrived as {type(value).__name__}"
            for key, value in call["di"].items():
                assert isinstance(value, Rendered), f"di_inputs[{key}] arrived as {type(value).__name__}"

    def test_bare_values_are_keyed_from_every_producer_of_them(self):
        """A bare unkeyed value reaches the seam from four different producers.
        All four must be keyed by the RAW type name -- not by 'Rendered', which
        is what a naive keying at the seam would produce for a pre-rendered one.
        """
        from neograph import compile_prompt, render_inputs
        from neograph.renderers import build_rendered_input, to_prompt_input

        payload = RawText(text="prose")

        # 1. the public render_inputs primitive (single-type node input)
        assert set(render_inputs(payload)) == {"RawText"}
        # 2. a value that has ALREADY been rendered by a call site
        pre_rendered = build_rendered_input(payload).for_template_ref
        assert set(pre_rendered) == {"RawText"}
        # 3. the seam normalizer, given a raw model (the merge_pre_process shape)
        assert set(to_prompt_input(payload)) == {"RawText"}
        # 4. the public compile_prompt entry point
        captured: list = []

        def compiler(template, input_data, **kw):
            captured.append(input_data)
            return [{"role": "user", "content": "x"}]

        compile_prompt("tmpl", payload, prompt_compiler=compiler)
        assert set(captured[0]) == {"RawText"}


# ═══════════════════════════════════════════════════════════════════════════
# 9. THE CONTEXT CHANNEL reaches the template — neograph-cbfd9
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaultCompilerThreadsContext:
    """A node's declared ``context=`` must reach its template's namespace.

    neograph-cbfd9. ``_compile_prompt`` passes ``context=`` to the compiler under
    the same introspection gate ``di_inputs`` rides, and
    ``DefaultPromptCompiler.__call__`` accepted it into ``**_kw`` and dropped it
    on the floor. So a channel the node DECLARES is dead in the framework's own
    90%-case compiler: with ``strict=True`` the run dies naming a var the author
    did declare, and with ``strict=False`` the literal ``{brief}`` ships to the
    model. That is the failure mode neograph-hjwv's strict substitution exists to
    prevent, arriving one layer earlier -- the var never enters the namespace at
    all, so strictness has nothing left to catch.

    The context field is deliberately a node that is NOT an input of the consumer.
    A first draft used the consumer's own upstream and proved nothing: that name is
    already in the namespace via ``render_inputs``, so the template resolved through
    the input channel and the context channel was never exercised.
    """

    def _run(self, tmp_path, *, strict: bool, mode: str = "think"):
        import types

        from neograph import DefaultPromptCompiler, Tool
        from tests.fakes import FakeTool, ReActFake, register_tool_factory

        seen: list[str] = []

        def _record(messages):
            seen.extend(m["content"] for m in messages if isinstance(m, dict) and "content" in m)

        class RecordingFake:
            """Records the compiled messages, then answers structurally.

            NOT a StructuredFake subclass: that fake's ``with_structured_output``
            returns a NEW StructuredFake rather than ``self``, so an override on
            the subclass is discarded before ``invoke`` is ever reached and the
            recorder silently captures nothing. (It did: the first run of this
            test failed on an empty transcript, not on the bug.) Returning
            ``self`` is the idiom the other capture fakes in this suite use.
            """

            def __init__(self, respond):
                self._respond = respond
                self._model = None

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def bind(self, **kw):
                return self

            def invoke(self, messages, **kw):
                _record(messages)
                return self._respond(self._model)

            async def ainvoke(self, *a, **k):
                return self.invoke(*a, **k)

        class RecordingReAct(ReActFake):
            def invoke(self, messages, **kw):
                _record(messages)
                return super().invoke(messages, **kw)

        prompts = tmp_path / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "use-ctx.md").write_text("Seed {seed}. Given {brief}, respond per {json_schema}\n")

        mod = types.ModuleType(f"test_ctx_thread_{mode}_mod")

        @node(outputs=RawText)
        def seed() -> RawText:
            return RawText(text="the seed")

        @node(outputs=RawText)
        def brief() -> RawText:
            return RawText(text="ship it")

        is_agent = mode in ("agent", "act")
        if is_agent:
            register_tool_factory("noop", lambda config, tool_config: FakeTool("noop", response="-"))
        extra = {"tools": [Tool("noop", budget=1)]} if is_agent else {}

        # `brief` is context-only: NOT a parameter of analyze, so {brief} can only
        # be satisfied by the context channel.
        @node(outputs=Claims, mode=mode, model="fast", prompt="use-ctx", context=["brief"], **extra)
        def analyze(seed: RawText) -> Claims: ...

        mod.seed = seed
        mod.brief = brief
        mod.analyze = analyze
        graph = compile(
            construct_from_module(mod),
            llm_factory=lambda tier: (
                RecordingReAct(tool_calls=[[]], final=lambda m: m(items=["ok"]), output_model=Claims)
                if is_agent
                else RecordingFake(lambda m: m(items=["ok"]))
            ),
            prompt_compiler=DefaultPromptCompiler(prompts, strict=strict),
            **build_test_compile_kwargs(),
        )
        run(graph, input={"node_id": "ctx-thread"})
        return "\n".join(seen)

    def test_context_var_resolves_when_default_compiler_is_strict(self, tmp_path):
        """strict=True: the run must not die on a var the author DID declare."""
        joined = self._run(tmp_path, strict=True)
        assert "ship it" in joined, f"declared context never reached the prompt:\n{joined}"

    def test_context_reaches_the_prompt_in_agent_mode_too(self, tmp_path):
        """The fix sits BELOW the think/agent fork -- proven, not argued.

        think/raw reach the compiler via ThinkDispatch and agent/act via the ReAct
        cycle's own turn prep; both hand context to the same _compile_prompt. The
        trace-similar atom reasoned that a compiler-level fix therefore covers all
        modes. Reasoning is not evidence, and the two paths have diverged before
        (that is precisely how di_inputs needed a second wiring in _agent_cycle).
        """
        joined = self._run(tmp_path, strict=True, mode="agent")
        assert "ship it" in joined, f"declared context never reached the agent prompt:\n{joined}"

    def test_context_var_is_not_left_literal_when_default_compiler_is_lenient(self, tmp_path):
        """strict=False: the literal placeholder must not ship to the model.

        The lenient half matters on its own -- strict=True turns the bug into a
        loud crash, which is survivable; strict=False turns it into a prompt that
        silently asks the model about '{brief}'.
        """
        joined = self._run(tmp_path, strict=False)
        assert "{brief}" not in joined, f"literal placeholder shipped to the model:\n{joined}"
        assert "ship it" in joined, f"declared context never reached the prompt:\n{joined}"
        # Pin the SHAPE, not just the substring. Context values are raw models
        # today, and substitute() stringifies with str(), so what actually reaches
        # the model is the Pydantic repr -- ugly, and strictly better than the
        # crash or the literal placeholder it replaces. Asserting only "ship it"
        # would pass either way and would NOT notice when neograph-ufqr7 makes the
        # channel obey the one rendering rule. When ufqr7 lands, this assertion is
        # SUPPOSED to fail; flip it to the rendered form then.
        assert "text='ship it'" in joined, (
            "expected the current verbatim-raw-model contract (Pydantic repr).\n"
            f"If neograph-ufqr7 has landed, this is the expected break -- update to the\n"
            f"rendered form. Got:\n{joined}"
        )
