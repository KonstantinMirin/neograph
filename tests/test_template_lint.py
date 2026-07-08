"""lint() template-placeholder validation: inline ${var} and template-ref {var}."""

from __future__ import annotations

from pydantic import BaseModel, Field

from neograph import (
    Construct,
    Node,
    construct_from_functions,
    construct_from_module,
    node,
)


class TestTemplatePlaceholderLint:
    """lint() validates inline prompt ${var} placeholders against predicted input keys.

    TASK neograph-0h3x: a template referencing ${original_param} inside a sub-construct
    crashes at runtime because the key is neo_subgraph_input. lint must catch this.
    """

    # ── Basic valid / invalid ───────────────────────────────────────────

    def test_valid_inline_placeholder_no_issue(self):
        """Inline prompt ${seed} matching input key → no lint issue."""
        from neograph.lint import lint

        class Claims(BaseModel):
            items: list[str]

        class Summary(BaseModel):
            text: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Claims),
                Node(
                    "summarize", prompt="Summarize: ${seed}", model="default", outputs=Summary, inputs={"seed": Claims}
                ),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_invalid_inline_placeholder_flagged(self):
        """Inline prompt ${nonexistent} not matching any input key → lint issue."""
        from neograph.lint import lint

        class Claims(BaseModel):
            items: list[str]

        class Summary(BaseModel):
            text: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Claims),
                Node(
                    "summarize",
                    prompt="Summarize: ${nonexistent}",
                    model="default",
                    outputs=Summary,
                    inputs={"seed": Claims},
                ),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert len(template_issues) == 1
        assert "nonexistent" in template_issues[0].message
        assert template_issues[0].required is True

    def test_multiple_invalid_placeholders_all_flagged(self):
        """Every invalid placeholder in a prompt gets its own issue."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node(
                    "proc", prompt="A: ${bad1}, B: ${bad2}, OK: ${seed}", model="default", outputs=B, inputs={"seed": A}
                ),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged_params = {i.param for i in template_issues}
        assert flagged_params == {"bad1", "bad2"}

    # ── Sub-construct / neo_subgraph_input ──────────────────────────────

    def test_sub_construct_port_remapping_flagged(self):
        """Inside a sub-construct, placeholder referencing original param name
        that was remapped to neo_subgraph_input → lint issue."""
        from neograph.lint import lint

        class Input(BaseModel):
            text: str

        class Output(BaseModel):
            result: str

        sub = Construct(
            "sub",
            input=Input,
            output=Output,
            nodes=[
                Node(
                    "proc",
                    prompt="Process: ${original_param}",
                    model="default",
                    outputs=Output,
                    inputs={"neo_subgraph_input": Input},
                ),
            ],
        )
        parent = Construct(
            "parent",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Input),
                sub,
            ],
        )
        issues = lint(parent)
        template_issues = [i for i in issues if "template" in i.kind]
        assert len(template_issues) >= 1
        assert "original_param" in template_issues[0].message

    def test_sub_construct_neo_subgraph_input_valid(self):
        """${neo_subgraph_input} inside a sub-construct is valid — it's the actual key."""
        from neograph.lint import lint

        class Input(BaseModel):
            text: str

        class Output(BaseModel):
            result: str

        sub = Construct(
            "sub",
            input=Input,
            output=Output,
            nodes=[
                Node(
                    "proc",
                    prompt="Process: ${neo_subgraph_input}",
                    model="default",
                    outputs=Output,
                    inputs={"neo_subgraph_input": Input},
                ),
            ],
        )
        parent = Construct(
            "parent",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Input),
                sub,
            ],
        )
        issues = lint(parent)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_node_decorator_sub_construct_remapping(self):
        """@node inside construct_from_functions(input=, output=) — port param
        remapped to neo_subgraph_input. Invalid placeholder caught."""
        from neograph.lint import lint

        class Input(BaseModel):
            text: str

        class Output(BaseModel):
            result: str

        @node(mode="think", outputs=Output, model="default", prompt="Process: ${text_input}")
        def proc(text_input: Input) -> Output: ...

        sub = construct_from_functions("sub", [proc], input=Input, output=Output)
        parent = Construct(
            "parent",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Input),
                sub,
            ],
        )
        issues = lint(parent)
        template_issues = [i for i in issues if "template" in i.kind]
        # After @node assembly, port param 'text_input' is remapped to neo_subgraph_input
        # If lint predicts correctly, ${text_input} is unresolvable
        # (the actual IR key is neo_subgraph_input)
        assert len(template_issues) >= 1
        assert "text_input" in template_issues[0].message

    def test_sub_construct_port_alias_valid_in_template_ref(self):
        """Template-ref {PortType} -- the friendly alias for neo_subgraph_input
        (the port's declared type name) -- resolves without a lint issue, and
        {neo_subgraph_input} itself stays valid too (back-compat, neograph-bluv,
        F3.4). Lint's third-column prediction must stay in lockstep with the
        runtime alias added in renderers.build_rendered_input."""
        from neograph.lint import lint

        class Input(BaseModel):
            text: str

        class Output(BaseModel):
            result: str

        sub = Construct(
            "sub",
            input=Input,
            output=Output,
            nodes=[
                Node(
                    "proc",
                    prompt="tmpl/proc",
                    model="default",
                    outputs=Output,
                    inputs={"neo_subgraph_input": Input},
                ),
            ],
        )
        parent = Construct(
            "parent",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Input),
                sub,
            ],
        )

        def resolver(name):
            return "Alias: {Input}, internal: {neo_subgraph_input}" if name == "tmpl/proc" else None

        issues = lint(parent, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        errors = [i for i in template_issues if i.required]
        assert errors == []

    # ── Known extras & custom vars ──────────────────────────────────────

    def test_known_extras_not_flagged_in_template_ref(self):
        """Template-ref {node_id}, {project_root} are framework extras -> no issue.
        Note: these are NOT available in inline prompts (no config access)."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/analyze", model="default", outputs=B, inputs={"seed": A}),
            ],
        )

        def resolver(name):
            return "ID: {node_id}, root: {project_root}, data: {seed}" if name == "rw/analyze" else None

        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        errors = [i for i in template_issues if i.required]
        assert errors == []

    def test_custom_known_vars_prevents_error_but_warns(self):
        """Consumer-supplied known_template_vars prevents ERROR but emits WARN."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="Topic: ${topic}, data: ${seed}", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        issues = lint(c, known_template_vars={"topic"})
        template_issues = [i for i in issues if "template" in i.kind]
        # ${topic} is not an ERROR (not unresolvable) but IS a WARN (known_vars only)
        errors = [i for i in template_issues if i.required]
        warns = [i for i in template_issues if not i.required]
        assert errors == [], "Should not be an ERROR"
        assert len(warns) == 1
        assert "topic" in warns[0].message
        assert "known_vars" in warns[0].kind

    def test_custom_known_vars_not_supplied_flagged(self):
        """Without known_template_vars, consumer-specific var IS flagged."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="Topic: ${topic}, data: ${seed}", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        issues = lint(c)  # no known_template_vars
        template_issues = [i for i in issues if "template" in i.kind]
        assert len(template_issues) == 1
        assert "topic" in template_issues[0].message

    def test_known_vars_only_placeholder_warns(self):
        """BUG neograph-yws3: placeholder resolved ONLY via known_vars (not
        in input keys or framework extras) should emit WARN, not silently pass.

        This catches the piarch pattern where bridge aliases like
        {research_packet} passed lint via --known-vars but crashed at runtime.
        """
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node(
                    "proc",
                    prompt="Packet: ${research_packet}, data: ${seed}",
                    model="default",
                    outputs=B,
                    inputs={"seed": A},
                ),
            ],
        )
        # ${research_packet} only resolvable via known_vars, not input keys
        issues = lint(c, known_template_vars={"research_packet"})
        template_issues = [i for i in issues if "template" in i.kind]
        assert len(template_issues) == 1
        assert "research_packet" in template_issues[0].message
        assert template_issues[0].required is False  # WARN, not ERROR
        assert "known_vars" in template_issues[0].kind

    def test_known_vars_overlapping_input_key_no_warn(self):
        """known_vars that overlap with actual input keys → no warning (redundant but harmless)."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="Data: ${seed}", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        # "seed" is both an input key AND in known_vars — no warning
        issues = lint(c, known_template_vars={"seed"})
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_known_vars_overlapping_framework_extra_no_warn(self):
        """known_vars that overlap with framework extras (node_id) in template-ref -> no warning."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/proc", model="default", outputs=B, inputs={"seed": A}),
            ],
        )

        def resolver(name):
            return "ID: {node_id}" if name == "rw/proc" else None

        issues = lint(c, known_template_vars={"node_id"}, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    # ── Dotted access ───────────────────────────────────────────────────

    def test_dotted_placeholder_validates_first_segment(self):
        """${seed.items} — first segment 'seed' must match input key."""
        from neograph.lint import lint

        class Claims(BaseModel):
            items: list[str]

        class Summary(BaseModel):
            text: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Claims),
                Node(
                    "summarize",
                    prompt="Items: ${seed.items}",
                    model="default",
                    outputs=Summary,
                    inputs={"seed": Claims},
                ),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_dotted_placeholder_invalid_first_segment_flagged(self):
        """${bad.field} — first segment 'bad' not in input keys → flagged."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="Val: ${bad.field}", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert len(template_issues) == 1
        assert template_issues[0].param == "bad"

    # ── Edge cases: skip conditions ─────────────────────────────────────

    def test_scripted_node_skipped(self):
        """Scripted nodes have no LLM prompt — not checked."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_template_ref_prompt_skipped(self):
        """Template-ref prompts (no space, no ${}) are opaque — skip."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/summarize", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_no_prompt_skipped(self):
        """Node with mode=think but prompt=None — no crash, no issues."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt=None, model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_inline_prompt_without_placeholders_skipped(self):
        """Inline prompt with spaces but no ${} — nothing to validate."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="Just a plain instruction", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    # ── Input shape edge cases ──────────────────────────────────────────

    def test_node_with_no_inputs_flags_all_placeholders(self):
        """Source node with prompt and ${var} — no input keys, all flagged."""
        from neograph.lint import lint

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node("gen", prompt="Generate about: ${topic}", model="default", outputs=B, inputs=None),
            ],
        )
        # ${topic} not in empty predicted keys and not a known extra
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert len(template_issues) == 1
        assert "topic" in template_issues[0].message

    def test_node_with_no_inputs_known_extra_ok_in_template_ref(self):
        """Source node with template-ref {node_id} — framework extra is fine.
        Note: ${node_id} in inline prompts IS flagged (no config access)."""
        from neograph.lint import lint

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node("gen", prompt="rw/gen", model="default", outputs=B, inputs=None),
            ],
        )

        def resolver(name):
            return "Generate for: {node_id}" if name == "rw/gen" else None

        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_fan_in_multiple_upstreams_all_valid(self):
        """Fan-in with multiple upstream keys — all valid in template."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        class C(BaseModel):
            z: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("alpha", fn="noop", outputs=A),
                Node.scripted("beta", fn="noop", outputs=B),
                Node(
                    "merge",
                    prompt="A: ${alpha}, B: ${beta}",
                    model="default",
                    outputs=C,
                    inputs={"alpha": A, "beta": B},
                ),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    # ── predict_input_keys public API ───────────────────────────────────

    def test_predict_input_keys_dict_form(self):
        """_predict_input_keys returns the dict keys for dict-form inputs."""
        from neograph.lint import _predict_input_keys

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        n = Node("test", outputs=B, inputs={"alpha": A, "beta": A})
        assert _predict_input_keys(n) == {"alpha", "beta"}

    def test_predict_input_keys_none(self):
        """_predict_input_keys returns empty set for inputs=None."""
        from neograph.lint import _predict_input_keys

        class B(BaseModel):
            y: str

        n = Node("test", outputs=B, inputs=None)
        assert _predict_input_keys(n) == set()

    def test_predict_input_keys_single_type(self):
        """_predict_input_keys returns empty set for single-type inputs."""
        from neograph.lint import _predict_input_keys

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        # DeprecationWarning fires at Construct assembly, not Node creation
        n = Node("test", outputs=B, inputs=A)
        assert _predict_input_keys(n) == set()

    # ── render_for_prompt return annotation introspection ─────────────

    def test_predict_input_keys_includes_flattened_fields(self):
        """_predict_input_keys includes fields from render_for_prompt() return model."""
        from neograph.lint import _predict_input_keys

        class ViewModel(BaseModel):
            claim_statement: str
            score: float

        class FullModel(BaseModel):
            raw: str
            internal_id: int

            def render_for_prompt(self) -> ViewModel:
                return ViewModel(claim_statement=self.raw, score=0.0)

        n = Node("test", outputs=FullModel, inputs={"data": FullModel})
        keys = _predict_input_keys(n)
        # Must include the input key AND the flattened fields from ViewModel
        assert "data" in keys
        assert "claim_statement" in keys
        assert "score" in keys
        # Internal fields of FullModel should NOT be included
        assert "internal_id" not in keys

    def test_predict_input_keys_no_render_for_prompt_no_extra(self):
        """Without render_for_prompt, only the input dict keys are returned."""
        from neograph.lint import _predict_input_keys

        class Plain(BaseModel):
            x: str

        n = Node("test", outputs=Plain, inputs={"item": Plain})
        assert _predict_input_keys(n) == {"item"}

    def test_predict_input_keys_str_return_no_extra(self):
        """render_for_prompt returning str — no flattening, no extra keys."""
        from neograph.lint import _predict_input_keys

        class WithStr(BaseModel):
            x: str

            def render_for_prompt(self) -> str:
                return f"CUSTOM: {self.x}"

        n = Node("test", outputs=WithStr, inputs={"data": WithStr})
        keys = _predict_input_keys(n)
        assert keys == {"data"}  # no extra fields from str return

    def test_predict_input_keys_exclude_fields_skipped(self):
        """Excluded fields on the return model are not added to predicted keys."""
        from neograph.lint import _predict_input_keys

        class View(BaseModel):
            visible: str
            hidden: str = Field(exclude=True, default="x")

        class Source(BaseModel):
            raw: str

            def render_for_prompt(self) -> View:
                return View(visible=self.raw)

        n = Node("test", outputs=Source, inputs={"src": Source})
        keys = _predict_input_keys(n)
        assert "visible" in keys
        assert "hidden" not in keys

    def test_predict_input_keys_no_return_annotation_fallback(self):
        """render_for_prompt with no return annotation — graceful fallback."""
        from neograph.lint import _predict_input_keys

        class NoAnnotation(BaseModel):
            x: str

            def render_for_prompt(self):
                return "plain"

        n = Node("test", outputs=NoAnnotation, inputs={"data": NoAnnotation})
        keys = _predict_input_keys(n)
        assert keys == {"data"}  # no extra — can't introspect without annotation

    def test_lint_accepts_flattened_placeholder_in_template_ref(self):
        """lint() should not flag {claim_statement} in a template-ref prompt when
        input model's render_for_prompt returns a ViewModel with that field."""
        from neograph.lint import lint

        class ViewModel(BaseModel):
            claim_statement: str

        class FullModel(BaseModel):
            raw: str

            def render_for_prompt(self) -> ViewModel:
                return ViewModel(claim_statement=self.raw)

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=FullModel),
                Node("proc", prompt="rw/claim", model="default", outputs=FullModel, inputs={"seed": FullModel}),
            ],
        )

        def resolver(name):
            return "Claim: {claim_statement}" if name == "rw/claim" else None

        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        errors = [i for i in template_issues if i.required]
        assert errors == [], f"Flattened field in template-ref should be valid: {errors}"

    # ── Inline vs template-ref key set distinction ─────────────────────

    def test_inline_prompt_rejects_flattened_field(self):
        """Inline ${summary} referencing a flattened field from render_for_prompt
        must be flagged -- inline prompts skip flattening."""
        from neograph.lint import lint

        class Presentation(BaseModel):
            summary: str

        class Claims(BaseModel):
            raw: str

            def render_for_prompt(self) -> Presentation:
                return Presentation(summary=self.raw.upper())

        class Result(BaseModel):
            text: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Claims),
                Node("proc", prompt="Summarize: ${summary}", model="default", outputs=Result, inputs={"seed": Claims}),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged = {i.param for i in template_issues}
        assert "summary" in flagged, f"Flattened field in inline prompt should be flagged: {template_issues}"

    def test_template_ref_still_accepts_flattened_field(self):
        """Template-ref {summary} referencing a flattened field IS valid."""
        from neograph.lint import lint

        class Presentation(BaseModel):
            summary: str

        class Claims(BaseModel):
            raw: str

            def render_for_prompt(self) -> Presentation:
                return Presentation(summary=self.raw.upper())

        class Result(BaseModel):
            text: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=Claims),
                Node("proc", prompt="rw/summarize", model="default", outputs=Result, inputs={"seed": Claims}),
            ],
        )

        def resolver(name):
            return "Summary: {summary}" if name == "rw/summarize" else None

        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        errors = [i for i in template_issues if i.required]
        assert not errors, f"Template-ref flattened field should be valid: {errors}"

    def test_inline_prompt_rejects_known_extras(self):
        """Inline ${node_id} must be flagged -- _resolve_var has no config access."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="ID: ${node_id}, data: ${seed}", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged = {i.param for i in template_issues}
        assert "node_id" in flagged, f"Known extra in inline prompt should be flagged: {template_issues}"

    def test_template_ref_still_accepts_known_extras(self):
        """Template-ref {node_id} IS valid -- prompt_compiler has config access."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/analyze", model="default", outputs=B, inputs={"seed": A}),
            ],
        )

        def resolver(name):
            return "ID: {node_id}, data: {seed}" if name == "rw/analyze" else None

        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        errors = [i for i in template_issues if i.required]
        assert not errors, f"Known extra in template-ref should be valid: {errors}"

    # ── Consumer integration: template-ref prompt validation ────────────

    def test_consumer_validates_template_ref_with_predicted_keys(self):
        """Consumer uses _predict_input_keys to validate template-ref prompt
        placeholders — the pattern piarch needs for its prompt_compiler.

        This simulates a consumer who:
        1. Loads a template file with {placeholder} markers
        2. Uses _predict_input_keys to get the runtime dict keys
        3. Validates template placeholders against those keys
        """
        import re

        from neograph.lint import _predict_input_keys

        class Research(BaseModel):
            findings: str

        class Draft(BaseModel):
            content: str

        class Review(BaseModel):
            score: float

        # Consumer's template (loaded from file)
        template_content = "Review this draft: {draft}\nBased on: {research}"
        template_placeholders = set(re.findall(r"\{(\w+)\}", template_content))
        # → {"draft", "research"}

        # Node with matching fan-in inputs
        good_node = Node(
            "review", prompt="rw/review", model="default", outputs=Review, inputs={"draft": Draft, "research": Research}
        )
        predicted = _predict_input_keys(good_node)
        assert template_placeholders <= predicted, (
            f"Template needs {template_placeholders} but node provides {predicted}"
        )

        # Node with MISMATCHED inputs (the piarch bug pattern)
        bad_node = Node(
            "review", prompt="rw/review", model="default", outputs=Review, inputs={"neo_subgraph_input": Draft}
        )
        predicted_bad = _predict_input_keys(bad_node)
        unresolvable = template_placeholders - predicted_bad
        assert unresolvable == {"draft", "research"}, f"Expected unresolvable placeholders, got {unresolvable}"


class TestTemplateRefLint:
    """lint() validates template-ref prompt {placeholder} names when a resolver is provided.

    BUG neograph-vkiw: templates referencing field-level names like {existing_si}
    (a field INSIDE a model, not a parameter name) pass lint but crash at runtime.
    """

    def _resolver(self, templates: dict[str, str]):
        """Create a template_resolver from a dict of name → text."""
        return lambda name: templates.get(name)

    def test_valid_template_placeholders_no_issue(self):
        """Template with {seed} matching input key → no lint issue."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/summarize", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        resolver = self._resolver({"rw/summarize": "Summarize this: {seed}"})
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_invalid_template_placeholder_flagged(self):
        """Template with {nonexistent} → lint ERROR."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/summarize", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        resolver = self._resolver({"rw/summarize": "Data: {seed}, Extra: {nonexistent}"})
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        assert len(template_issues) == 1
        assert "nonexistent" in template_issues[0].message
        assert template_issues[0].required is True

    def test_field_inside_model_flagged(self):
        """Template referencing a field inside a model (not the param name) → ERROR.

        This is the exact piarch bug: {existing_si} is a field inside UCComposite,
        not a top-level input key. After BAML rendering, the key is the parameter
        name, not the field name.
        """
        from neograph.lint import lint

        class UCComposite(BaseModel):
            existing_si: str
            title: str

        class Output(BaseModel):
            result: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=UCComposite),
                Node("writer", prompt="rw/write-si", model="default", outputs=Output, inputs={"seed": UCComposite}),
            ],
        )
        # Template references {existing_si} — a field inside UCComposite, not the param name "seed"
        resolver = self._resolver({"rw/write-si": "Write SI for: {existing_si}\nTitle: {title}"})
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        assert len(template_issues) == 2  # both {existing_si} and {title}
        flagged = {i.param for i in template_issues}
        assert flagged == {"existing_si", "title"}

    def test_no_resolver_skips_template_ref(self):
        """Without template_resolver, template-ref prompts remain opaque."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/summarize", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        # No resolver → no template inspection → no issues
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_resolver_returns_none_skips(self):
        """Resolver returning None for unknown template → skip gracefully."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/unknown", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        resolver = self._resolver({})  # empty — returns None for everything
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_known_vars_accepted_in_template_ref(self):
        """Template {topic} resolved via known_vars → WARN (not ERROR)."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/summarize", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        resolver = self._resolver({"rw/summarize": "Topic: {topic}, Data: {seed}"})
        issues = lint(c, known_template_vars={"topic"}, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        # {topic} is known_vars-only → WARN, not ERROR
        errors = [i for i in template_issues if i.required]
        warns = [i for i in template_issues if not i.required]
        assert errors == []
        assert len(warns) == 1
        assert "topic" in warns[0].message

    def test_framework_extras_accepted_in_template_ref(self):
        """{node_id} in template is a framework extra → no issue."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/summarize", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        resolver = self._resolver({"rw/summarize": "ID: {node_id}, Data: {seed}"})
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_dotted_placeholder_validates_first_segment(self):
        """{seed.items} in template — first segment 'seed' valid → no issue."""
        from neograph.lint import lint

        class A(BaseModel):
            items: list[str]

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/summarize", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        resolver = self._resolver({"rw/summarize": "Items: {seed.items}"})
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    def test_multiple_unresolvable_all_flagged(self):
        """Multiple bad placeholders in one template → all flagged."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct(
            "test",
            nodes=[
                Node.scripted("seed", fn="noop", outputs=A),
                Node("proc", prompt="rw/proc", model="default", outputs=B, inputs={"seed": A}),
            ],
        )
        resolver = self._resolver({"rw/proc": "A: {bad1}, B: {bad2}, OK: {seed}"})
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged = {i.param for i in template_issues}
        assert flagged == {"bad1", "bad2"}


class TestDiInputTemplateRefColumn:
    """Third column of the inline/template-ref key asymmetry (neograph-euyh).

    A ``FromInput``/``FromConfig`` parameter name is a VALID template-ref
    placeholder ONLY when the app's prompt_compiler opts into ``di_inputs`` (the
    dispatch layer resolves the value and the compiler binds it by param name).
    When the compiler does not declare ``di_inputs``, the resolved value never
    reaches the template, so the placeholder stays unresolvable.

    Requires ``@node``: DI bindings live in the decorator sidecar (``_param_res``).
    Declarative/programmatic Nodes carry no bindings, so the column is empty for
    them — the documented three-surface exemption.
    """

    def _build(self):
        import types
        from typing import Annotated

        from neograph import FromInput

        class Claims(BaseModel):
            items: list[str]

        mod = types.ModuleType("test_di_lint_col_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="leaf")
        def analyze(domain: Annotated[str, FromInput]) -> Claims: ...

        mod.analyze = analyze
        return construct_from_module(mod)

    @staticmethod
    def _resolver(name):
        return "Analyze the {domain} domain." if name == "leaf" else None

    def test_di_param_valid_template_ref_when_compiler_accepts_di_inputs(self):
        """ACCEPT case: compiler declares ``di_inputs`` → ``{domain}`` is valid."""
        from neograph.lint import lint

        def compiler_with_di(template, input_data, *, di_inputs=None):
            return [{"role": "user", "content": "x"}]

        c = self._build()
        issues = lint(
            c,
            template_resolver=self._resolver,
            prompt_compiler=compiler_with_di,
            llm_factory=lambda tier: None,
            config={"domain": "oncology"},
        )
        unresolvable = [i for i in issues if i.kind == "template_placeholder_unresolvable"]
        assert unresolvable == []

    def test_di_param_unresolvable_when_compiler_lacks_di_inputs(self):
        """NOT-ACCEPT case: compiler has no ``di_inputs`` param → ``{domain}`` is
        flagged unresolvable (the literal would ship to the model — agent-stark)."""
        from neograph.lint import lint

        def compiler_no_di(template, input_data, *, output_model=None):
            return [{"role": "user", "content": "x"}]

        c = self._build()
        issues = lint(
            c,
            template_resolver=self._resolver,
            prompt_compiler=compiler_no_di,
            llm_factory=lambda tier: None,
            config={"domain": "oncology"},
        )
        unresolvable = [i for i in issues if i.kind == "template_placeholder_unresolvable"]
        assert len(unresolvable) == 1
        assert unresolvable[0].param == "domain"

    def test_accept_all_kwargs_compiler_enables_di_column(self):
        """A ``**kwargs`` compiler accepts di_inputs (the _ACCEPT_ALL sentinel), so
        the column is enabled and ``{domain}`` is valid."""
        from neograph.lint import lint

        def compiler_kwargs(template, input_data, **kw):
            return [{"role": "user", "content": "x"}]

        c = self._build()
        issues = lint(
            c,
            template_resolver=self._resolver,
            prompt_compiler=compiler_kwargs,
            llm_factory=lambda tier: None,
            config={"domain": "oncology"},
        )
        unresolvable = [i for i in issues if i.kind == "template_placeholder_unresolvable"]
        assert unresolvable == []


class TestResourceTemplateRefColumn:
    """FROM_RESOURCE as a template-ref var (neograph-3q6j).

    Runtime serves a ``{history}`` FROM_RESOURCE placeholder only on the async
    ``arun()`` driver (the fetch is awaited); the sync ``run()`` driver fails
    loud. Lint mirrors that in lockstep:

    - with a ``di_inputs``-aware compiler the placeholder is VALID (it resolves on
      async), but earns a WARN (``template_var_requires_async_driver``) naming the
      param, because it needs arun();
    - without one the value never reaches the template, so it is
      ``template_placeholder_unresolvable`` — same as any other DI kind.
    """

    def _build(self):
        import types
        from typing import Annotated

        from neograph import FromResource

        class Out(BaseModel):
            ok: bool = True

        mod = types.ModuleType("test_resource_lint_col_mod")

        @node(outputs=Out, mode="think", model="fast", prompt="leaf")
        def analyze(history: Annotated[str, FromResource("crm://history")]) -> Out: ...

        mod.analyze = analyze
        return construct_from_module(mod)

    @staticmethod
    def _resolver(name):
        return "History:\n{history}" if name == "leaf" else None

    def test_resource_var_valid_but_warns_async_driver_when_compiler_accepts_di_inputs(self):
        """COVERED case: compiler declares ``di_inputs`` → ``{history}`` is valid,
        but flagged with the async-driver WARN (runtime serves it only on arun())."""
        from neograph.lint import lint

        def compiler_with_di(template, input_data, *, di_inputs=None):
            return [{"role": "user", "content": "x"}]

        c = self._build()
        issues = lint(
            c,
            template_resolver=self._resolver,
            prompt_compiler=compiler_with_di,
            llm_factory=lambda tier: None,
            config={},
        )
        unresolvable = [i for i in issues if i.kind == "template_placeholder_unresolvable"]
        assert unresolvable == [], "a covered FROM_RESOURCE template var must not be unresolvable"
        async_warn = [i for i in issues if i.kind == "template_var_requires_async_driver"]
        assert len(async_warn) == 1
        assert async_warn[0].param == "history"
        assert async_warn[0].required is False

    def test_resource_var_unresolvable_when_compiler_lacks_di_inputs(self):
        """UNCOVERED case: compiler has no ``di_inputs`` param → ``{history}`` never
        reaches the template, so it is flagged unresolvable (no async-driver warn)."""
        from neograph.lint import lint

        def compiler_no_di(template, input_data, *, output_model=None):
            return [{"role": "user", "content": "x"}]

        c = self._build()
        issues = lint(
            c,
            template_resolver=self._resolver,
            prompt_compiler=compiler_no_di,
            llm_factory=lambda tier: None,
            config={},
        )
        unresolvable = [i for i in issues if i.kind == "template_placeholder_unresolvable"]
        assert len(unresolvable) == 1
        assert unresolvable[0].param == "history"
        assert [i for i in issues if i.kind == "template_var_requires_async_driver"] == []
