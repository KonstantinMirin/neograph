"""Modifier tests — Oracle ensemble modifier"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel

from neograph import (
    ConfigurationError,
    Construct,
    ConstructError,
    ExecutionError,
    Node,
    Oracle,
    Tool,
    compile,
    construct_from_module,
    merge_fn,
    node,
    run,
)
from tests.fakes import (
    FakeTool,
    ReActFake,
    StructuredFake,
    build_test_compile_kwargs,
    configure_fake_llm,
    register_scripted,
    register_tool_factory,
)
from tests.schemas import (
    Claims,
    RawText,
)

# ═════════════════════════════════���═════════════════════════════════════════
# SHARED SCHEMAS
# ══════════════════��════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══���═══════════════���════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Oracle — 3-way ensemble + merge
#
# Three parallel generators produce variants, barrier merges them.
# This proves: Oracle modifier expands to fan-out Send() + merge barrier,
# all three generators run, results converge.
# ═══════════════════════════════════════════════════════════════════════════

class TestOracle:
    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_merge_combines_variants_when_three_generators_run(self):
        """Oracle dispatches 3 generators and merges results."""

        from tests.fakes import register_scripted

        gen_call_count = [0]

        # Merge: combine all variant items into one Claims
        def combine_variants(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("combine_variants", combine_variants)

        @node(outputs=Claims, ensemble_n=3, merge_fn="combine_variants")
        def generate() -> Claims:
            gen_call_count[0] += 1
            return Claims(items=[f"variant-{gen_call_count[0]}"])

        mod = self._fresh_module("test_oracle_ensemble")
        mod.generate = generate

        pipeline = construct_from_module(mod, name="test-oracle")
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test-001"})

        # All 3 generators ran
        assert gen_call_count[0] == 3
        # Merge combined all 3 variants
        merged = result.get("generate")
        assert isinstance(merged, Claims)
        assert len(merged.items) == 3

    def test_llm_judge_merges_when_merge_prompt_set(self):
        """Oracle with merge_prompt calls LLM to judge-merge variants."""

        from tests.fakes import register_scripted

        register_scripted("gen_llm", lambda input_data, config: Claims(items=["v1"]))

        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["merged-consensus"])))

        @node(outputs=Claims, ensemble_n=2, merge_prompt="test/merge")
        def generate() -> Claims:
            return Claims(items=["v1"])

        mod = self._fresh_module("test_oracle_llm")
        mod.generate = generate

        pipeline = construct_from_module(mod, name="test-oracle-llm")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "test-001"})

        merged = result.get("generate")
        assert isinstance(merged, Claims)
        assert merged.items == ["merged-consensus"]

    def test_oracle_raises_when_no_merge_option_given(self):
        """Oracle without merge_prompt or merge_fn is a ConfigurationError."""
        with pytest.raises(ConfigurationError, match="merge_prompt.*merge_fn"):
            Oracle(n=3)

    def test_oracle_raises_when_both_merge_options_given(self):
        """Oracle with both merge_prompt and merge_fn is a ConfigurationError."""
        with pytest.raises(ConfigurationError, match="not both"):
            Oracle(n=3, merge_prompt="x", merge_fn="y")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Each — dynamic fan-out over collection
#
# Node fans out over a list of clusters, processes each in parallel,
# results collected as dict[key, result].
# This proves: Each modifier expands to Send() per item,
# barrier collects, dict reducer merges results.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Each — dynamic fan-out over collection
#
# Node fans out over a list of clusters, processes each in parallel,
# results collected as dict[key, result].
# This proves: Each modifier expands to Send() per item,
# barrier collects, dict reducer merges results.
# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════
# CONSTRUCT + MODIFIER COMPOSITIONS
#
# Every modifier × Construct target, plus deep nesting combos.
# ═══════════════════════════════════════════════════════════════════════════


class TestConstructOracle:
    """Construct | Oracle — run entire sub-pipeline N times, merge outputs."""

    def test_sub_pipeline_runs_n_times_when_oracle_with_scripted_merge(self):
        """Sub-pipeline runs 3 times, scripted merge combines outputs."""
        from tests.fakes import register_scripted

        register_scripted("sub_step_a", lambda input_data, config: Claims(items=["step-a"]))
        register_scripted("sub_step_b", lambda input_data, config: RawText(
            text=f"processed: {input_data.items[0]}" if input_data else "processed: none"
        ))

        def merge_sub_outputs(variants, config):
            all_texts = [v.text for v in variants]
            return RawText(text=" | ".join(all_texts))

        register_scripted("merge_sub", merge_sub_outputs)

        sub = Construct(
            "enrich",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("step-a", fn="sub_step_a", outputs=Claims),
                Node.scripted("step-b", fn="sub_step_b", inputs=Claims, outputs=RawText),
            ],
        ) | Oracle(n=3, merge_fn="merge_sub")

        register_scripted("make_input", lambda input_data, config: Claims(items=["raw"]))

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="make_input", outputs=Claims),
            sub,
        ])
        graph = compile(parent, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "test-001"})

        # 3 variants merged into one
        assert isinstance(result["enrich"], RawText)
        assert result["enrich"].text.count("processed") == 3

    def test_sub_pipeline_runs_n_times_when_oracle_with_llm_merge(self):
        """Sub-pipeline runs 2 times, LLM merge combines outputs."""
        from tests.fakes import register_scripted

        register_scripted("gen_claim", lambda input_data, config: Claims(items=["variant"]))

        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["llm-merged"])))

        sub = Construct(
            "gen-pipeline",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("gen", fn="gen_claim", outputs=Claims)],
        ) | Oracle(n=2, merge_prompt="test/merge")

        register_scripted("seed", lambda input_data, config: Claims(items=["seed"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", outputs=Claims),
            sub,
        ])
        graph = compile(parent, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "test-001"})

        assert result["gen_pipeline"].items == ["llm-merged"]

    def test_oracle_models_forwarded_to_sub_construct_inner_nodes(self):
        """Oracle(models=) on a Construct must forward model override to inner nodes.

        Bug neograph-e481: make_subgraph_fn builds sub_input without
        neo_oracle_model, so all variants use the same model. The fix
        injects _oracle_model into the config passed to sub_graph.invoke.
        """
        seen_models = []

        def model_capturing_step(input_data, config):
            model = config.get("configurable", {}).get("_oracle_model")
            seen_models.append(model)
            return RawText(text=f"from-{model}")

        register_scripted("capture_model_step", model_capturing_step)

        def merge_models(variants, config):
            return RawText(text=" | ".join(v.text for v in variants))

        register_scripted("merge_model_variants", merge_models)

        sub = Construct(
            "model-sub",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("capture", fn="capture_model_step", inputs=Claims, outputs=RawText),
            ],
        ) | Oracle(models=["reason", "fast"], merge_fn="merge_model_variants")

        register_scripted("seed_claims", lambda input_data, config: Claims(items=["x"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed_claims", outputs=Claims),
            sub,
        ])
        graph = compile(parent, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "model-fwd-test"})

        # Each variant must have received a distinct model override
        assert len(seen_models) == 2, f"Expected 2 inner calls, got {len(seen_models)}"
        assert set(seen_models) == {"reason", "fast"}, f"Expected {{reason, fast}}, got {seen_models}"

    def test_oracle_without_models_on_construct_no_model_override(self):
        """Oracle(n=3) without models= must NOT inject _oracle_model (backward compat)."""
        seen_models = []

        def check_no_model(input_data, config):
            model = config.get("configurable", {}).get("_oracle_model")
            seen_models.append(model)
            return RawText(text="ok")

        register_scripted("check_no_model_step", check_no_model)

        def merge_no_model(variants, config):
            return RawText(text="merged")

        register_scripted("merge_no_model", merge_no_model)

        sub = Construct(
            "no-model-sub",
            input=Claims,
            output=RawText,
            nodes=[
                Node.scripted("check", fn="check_no_model_step", inputs=Claims, outputs=RawText),
            ],
        ) | Oracle(n=3, merge_fn="merge_no_model")

        register_scripted("seed_nmo", lambda input_data, config: Claims(items=["x"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed_nmo", outputs=Claims),
            sub,
        ])
        graph = compile(parent, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "no-model-test"})

        # All 3 variants should have seen None for _oracle_model
        assert len(seen_models) == 3
        assert all(m is None for m in seen_models), f"Expected all None, got {seen_models}"





# =============================================================================
# Oracle models= — multi-model ensemble (neograph-beyr)
# =============================================================================


class TestOracleModels:
    """Oracle with models= parameter for multi-model ensemble."""

    def test_oracle_assigns_model_per_generator_when_models_set(self):
        """Each generator gets a different model from the models list."""
        from tests.fakes import register_scripted

        seen_models = []

        def gen(input_data, config):
            # The generator should see a model override in config or state
            model = config.get("configurable", {}).get("_oracle_model")
            seen_models.append(model)
            return Claims(items=[f"from-{model}"])

        register_scripted("models_gen", gen)

        def merge(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("models_merge", merge)

        gen_node = (
            Node.scripted("models-gen", fn="models_gen", outputs=Claims)
            | Oracle(models=["reason", "fast", "creative"], merge_fn="models_merge")
        )
        pipeline = Construct("test-oracle-models", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "oracle-models"})

        # 3 generators, one per model
        assert len(seen_models) == 3
        assert set(seen_models) == {"reason", "fast", "creative"}
        # Merge combined all 3
        merged = result["models_gen"]
        assert isinstance(merged, Claims)
        assert len(merged.items) == 3

    def test_oracle_round_robins_models_when_n_exceeds_models_count(self):
        """When n > len(models), models are assigned round-robin."""
        from tests.fakes import register_scripted

        seen_models = []

        def rr_gen(input_data, config):
            model = config.get("configurable", {}).get("_oracle_model")
            seen_models.append(model)
            return Claims(items=[f"from-{model}"])

        register_scripted("rr_gen", rr_gen)

        def rr_merge(variants, config):
            return Claims(items=[f"{len(variants)} variants"])

        register_scripted("rr_merge", rr_merge)

        gen_node = (
            Node.scripted("rr-gen", fn="rr_gen", outputs=Claims)
            | Oracle(n=7, models=["reason", "fast", "creative"], merge_fn="rr_merge")
        )
        pipeline = Construct("test-rr", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "rr"})

        # 7 generators, round-robin across 3 models
        assert len(seen_models) == 7
        assert seen_models.count("reason") == 3   # 0,3,6
        assert seen_models.count("fast") == 2      # 1,4
        assert seen_models.count("creative") == 2  # 2,5

    def test_oracle_infers_n_from_models_length(self):
        """When only models= is set, n defaults to len(models)."""
        oracle = Oracle(models=["a", "b"], merge_fn="some_merge")
        assert oracle.n == 2

    def test_body_as_merge_when_models_set_on_node_decorator(self):
        """@node with models= uses the function body as the merge function.
        The body receives list[OutputType] at runtime (the collected variants)."""
        from tests.fakes import register_scripted

        gen_count = [0]

        def bam_gen(input_data, config):
            gen_count[0] += 1
            model = config.get("configurable", {}).get("_oracle_model", "unknown")
            return Claims(items=[f"from-{model}"])

        register_scripted("bam_gen", bam_gen)

        # Body-as-merge: function body IS the merge function
        # models= triggers Oracle, body receives list[Claims]
        gen_node = (
            Node.scripted("bam-gen", fn="bam_gen", outputs=Claims)
            | Oracle(models=["reason", "fast"], merge_fn="bam_body_merge")
        )

        # Register a merge that simulates what the body-as-merge would do
        def bam_body_merge(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("bam_body_merge", bam_body_merge)

        pipeline = Construct("body-merge", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "body-merge"})

        assert gen_count[0] == 2
        merged = result["bam_gen"]
        assert isinstance(merged, Claims)
        assert len(merged.items) == 2
        assert "from-reason" in merged.items
        assert "from-fast" in merged.items

    def test_node_decorator_models_registers_body_as_merge(self):
        """@node(models=...) without merge_fn uses the function body as merge."""

        # The body receives list[Claims] and merges them
        @node(outputs=Claims, models=["reason", "fast"])
        def ensemble(data: Claims) -> Claims:
            # At runtime, 'data' is list[Claims] (the collected variants)
            all_items = []
            for v in data:
                all_items.extend(v.items)
            return Claims(items=all_items)

        # Should have Oracle modifier with body-as-merge registered
        assert ensemble.has_modifier(Oracle)
        oracle = ensemble.get_modifier(Oracle)
        assert oracle.models == ["reason", "fast"]
        assert oracle.n == 2
        assert isinstance(oracle.merge_fn, str), "body should be registered as merge_fn (string key)"

    def test_oracle_models_on_think_mode_node(self):
        """Oracle(models=) must override model tier for think-mode (produce) nodes.

        Bug (pre-neograph-y8ww): the produce wrapper read _oracle_model from
        config but never transferred neo_oracle_model from state to config.
        Now all modes share _execute_node preamble which calls
        _inject_oracle_config. Regression test for neograph-lbsf.
        """
        from tests.fakes import register_scripted

        seen_tiers = []

        def tier_capturing_factory(tier):
            seen_tiers.append(tier)
            return StructuredFake(lambda m: m(items=[f"from-{tier}"]))

        __llm_kw = configure_fake_llm(tier_capturing_factory)

        # Merge function (scripted) to combine variants
        def think_merge(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("think_models_merge", think_merge)

        gen_node = (
            Node(
                name="think-gen",
                mode="think",
                outputs=Claims,
                model="default-tier",
                prompt="test/generate",
            )
            | Oracle(models=["reason", "fast", "creative"], merge_fn="think_models_merge")
        )
        pipeline = Construct("test-think-oracle-models", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "think-oracle-models"})

        # Filter to only the generator tiers (exclude merge LLM calls)
        gen_tiers = [t for t in seen_tiers if t in ("reason", "fast", "creative")]
        assert len(gen_tiers) == 3, f"Expected 3 generator calls with model overrides, got tiers: {seen_tiers}"
        assert set(gen_tiers) == {"reason", "fast", "creative"}

    def _agent_oracle_fake(self, tier):
        """A history-driven ReAct fake: call the tool once, then answer.

        Fresh per factory call, so each of the N isolated subgraph invocations
        (the auto-wrapped ReAct cycles) drives its own tool loop independently.
        """
        return ReActFake(
            tool_calls=[[{"name": "agent_search", "args": {}, "id": "s1"}], []],
            final=lambda m: m(items=["variant"]),
            output_model=Claims,
        )

    def test_oracle_over_self_contained_agent_node_runs(self):
        """Oracle(n) over a SELF-CONTAINED agent/act node compiles AND runs.

        neograph-m6d3.6: the inline ``(entry, exit)`` fan target the ticket
        prescribed is impossible — an agent's ReAct cycle keeps per-turn state in
        SHARED reducer channels, so ``Send``-ing N branches into it collapses them
        (see docs/design/fan-over-agent-node-2026-07-07.md). The isolation-correct
        fix is the AUTO-WRAP: the ReAct cycle is wrapped in an isolated single-node
        sub-construct and Oracle fans over THAT via the existing subgraph path.

        This is the declarative surface (``Node(mode="agent") | Oracle``). The
        ``merged-2`` assertion detects variant-count COLLAPSE: if the two ReAct
        cycles had shared one tangled message channel instead of isolating, the
        fan would collapse to fewer than 2 variants at the merge. It does NOT
        prove each cycle saw distinct CONTENT — that (content isolation) is the
        echo-style ``TestOracleOverAgentValueDelivery`` suite below, modeled on
        the Each content-isolation suite (``tests/modifiers/test_each.py``).
        """
        fake_tool = FakeTool("agent_search", response="found")
        register_tool_factory("agent_search", lambda config, tool_config: fake_tool)
        register_scripted(
            "agent_merge_count",
            lambda variants, config: Claims(items=[f"merged-{len(variants)}"]),
        )

        gen_node = (
            Node(
                name="agent-gen",
                mode="agent",
                outputs=Claims,
                model="default-tier",
                prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Oracle(n=2, merge_fn="agent_merge_count")
        )
        pipeline = Construct("test-agent-oracle", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs(),
                        **configure_fake_llm(self._agent_oracle_fake))
        result = run(graph, input={"node_id": "agent-oracle"})
        assert result["agent_gen"] == Claims(items=["merged-2"]), result

    def test_oracle_over_self_contained_agent_node_runs_via_node_decorator(self):
        """Same auto-wrap, the ``@node`` agent surface (three-surface parity).

        A ``@node(mode="agent")`` function with no params is self-contained; the
        decorator path must reach the same isolated-subgraph fan as the
        declarative ``Node(...)`` above.
        """
        fake_tool = FakeTool("agent_search", response="found")
        register_tool_factory("agent_search", lambda config, tool_config: fake_tool)
        register_scripted(
            "agent_merge_count_dec",
            lambda variants, config: Claims(items=[f"merged-{len(variants)}"]),
        )

        @node(
            mode="agent",
            outputs=Claims,
            model="default-tier",
            prompt="test/search",
            tools=[Tool(name="agent_search", budget=5)],
        )
        def researcher() -> Claims: ...

        gen_node = researcher | Oracle(n=2, merge_fn="agent_merge_count_dec")
        pipeline = Construct("test-agent-oracle-dec", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs(),
                        **configure_fake_llm(self._agent_oracle_fake))
        result = run(graph, input={"node_id": "agent-oracle-dec"})
        assert result["researcher"] == Claims(items=["merged-2"]), result

    def test_oracle_over_agent_node_with_single_type_input_runs(self):
        """neograph-qot6: Oracle over an agent that consumes a SINGLE-TYPE
        upstream input compiles AND runs.

        Input-port synthesis: the auto-wrap synthesizes ``input=RawText`` from
        the agent's single-type ``inputs``. The parent's upstream ``RawText`` is
        found by the subgraph's type-based input scan and delivered as
        ``neo_subgraph_input``; inside the isolated sub-construct the bare agent
        reads it via single-type extraction. The ``merged-2`` assertion detects
        variant-count COLLAPSE (2 variants reach the merge), not per-cycle content
        isolation — the latter is the echo-style
        ``TestOracleOverAgentValueDelivery`` suite below. Declarative surface.
        """
        fake_tool = FakeTool("agent_search", response="found")
        register_tool_factory("agent_search", lambda config, tool_config: fake_tool)
        register_scripted("agent_seed_raw", lambda input_data, config: RawText(text="seed"))
        register_scripted(
            "agent_merge_in",
            lambda variants, config: Claims(items=[f"merged-{len(variants)}"]),
        )

        gen_node = (
            Node(
                name="agent-gen",
                mode="agent",
                inputs=RawText,
                outputs=Claims,
                model="default-tier",
                prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Oracle(n=2, merge_fn="agent_merge_in")
        )
        pipeline = Construct("test-agent-oracle-single-in", nodes=[
            Node.scripted("seed", fn="agent_seed_raw", outputs=RawText),
            gen_node,
        ])
        graph = compile(pipeline, **build_test_compile_kwargs(),
                        **configure_fake_llm(self._agent_oracle_fake))
        result = run(graph, input={"node_id": "agent-oracle-single-in"})
        assert result["agent_gen"] == Claims(items=["merged-2"]), result

    def test_oracle_over_agent_node_with_dict_form_input_runs(self):
        """neograph-qot6: Oracle over an agent with DICT-FORM (single-key) fan-in
        input compiles AND runs. Declarative surface.

        The auto-wrap synthesizes ``input=RawText`` from the single dict-form
        input key and rewrites the inner agent's read to ``neo_subgraph_input``
        (the proven ``@node`` sub-construct port convention). N=2 isolated cycles.
        """
        fake_tool = FakeTool("agent_search", response="found")
        register_tool_factory("agent_search", lambda config, tool_config: fake_tool)
        register_scripted("agent_seed_raw2", lambda input_data, config: RawText(text="seed"))
        register_scripted(
            "agent_merge_dict",
            lambda variants, config: Claims(items=[f"merged-{len(variants)}"]),
        )

        gen_node = (
            Node(
                name="agent-gen",
                mode="agent",
                inputs={"seed": RawText},  # dict-form key names the upstream producer
                outputs=Claims,
                model="default-tier",
                prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Oracle(n=2, merge_fn="agent_merge_dict")
        )
        pipeline = Construct("test-agent-oracle-dict-in", nodes=[
            Node.scripted("seed", fn="agent_seed_raw2", outputs=RawText),
            gen_node,
        ])
        graph = compile(pipeline, **build_test_compile_kwargs(),
                        **configure_fake_llm(self._agent_oracle_fake))
        result = run(graph, input={"node_id": "agent-oracle-dict-in"})
        assert result["agent_gen"] == Claims(items=["merged-2"]), result

    def test_oracle_over_agent_node_with_upstream_input_via_node_decorator(self):
        """neograph-qot6: same input-port synthesis, the ``@node`` agent surface
        (three-surface parity). A ``@node(mode='agent')`` function with a typed
        upstream param is wrapped and fanned over the isolated subgraph.
        """
        fake_tool = FakeTool("agent_search", response="found")
        register_tool_factory("agent_search", lambda config, tool_config: fake_tool)
        register_scripted("agent_seed_dec", lambda input_data, config: RawText(text="seed"))
        register_scripted(
            "agent_merge_dec_in",
            lambda variants, config: Claims(items=[f"merged-{len(variants)}"]),
        )

        @node(
            mode="agent",
            outputs=Claims,
            model="default-tier",
            prompt="test/search",
            tools=[Tool(name="agent_search", budget=5)],
        )
        def researcher(seed: RawText) -> Claims: ...

        gen_node = researcher | Oracle(n=2, merge_fn="agent_merge_dec_in")
        pipeline = Construct("test-agent-oracle-dec-in", nodes=[
            Node.scripted("seed", fn="agent_seed_dec", outputs=RawText),
            gen_node,
        ])
        graph = compile(pipeline, **build_test_compile_kwargs(),
                        **configure_fake_llm(self._agent_oracle_fake))
        result = run(graph, input={"node_id": "agent-oracle-dec-in"})
        assert result["researcher"] == Claims(items=["merged-2"]), result

    def test_oracle_over_agent_node_with_input_and_di_param_runs(self):
        """neograph-qot6: an agent with BOTH an upstream input AND a FromInput DI
        param is wrapped and fanned. The DI param rides the forwarded config into
        the isolated sub-construct (it never appears in ``inputs``), so only the
        real upstream edge needs port synthesis.
        """
        from neograph import FromInput

        fake_tool = FakeTool("agent_search", response="found")
        register_tool_factory("agent_search", lambda config, tool_config: fake_tool)
        register_scripted("agent_seed_di", lambda input_data, config: RawText(text="seed"))
        register_scripted(
            "agent_merge_di",
            lambda variants, config: Claims(items=[f"merged-{len(variants)}"]),
        )

        @node(
            mode="agent",
            outputs=Claims,
            model="default-tier",
            prompt="test/search",
            tools=[Tool(name="agent_search", budget=5)],
        )
        def researcher(seed: RawText, topic: Annotated[str, FromInput]) -> Claims: ...

        gen_node = researcher | Oracle(n=2, merge_fn="agent_merge_di")
        pipeline = Construct("test-agent-oracle-di-in", nodes=[
            Node.scripted("seed", fn="agent_seed_di", outputs=RawText),
            gen_node,
        ])
        graph = compile(pipeline, **build_test_compile_kwargs(),
                        **configure_fake_llm(self._agent_oracle_fake))
        result = run(graph, input={"node_id": "agent-oracle-di-in", "topic": "birds"})
        assert result["researcher"] == Claims(items=["merged-2"]), result

    def test_oracle_over_agent_node_with_multiple_producers_runs_via_node_decorator(self):
        """neograph-qzrv: multi-producer Oracle over an agent now compiles AND runs
        via packer-port synthesis. The ``@node`` surface (three-surface parity): a
        function with TWO typed upstream params is bundled by a synthesized parent
        packer and unbundled by inner per-key unpackers, so both reach the fan."""
        fake_tool = FakeTool("agent_search", response="found")
        register_tool_factory("agent_search", lambda config, tool_config: fake_tool)
        register_scripted("agent_seed_claims", lambda input_data, config: Claims(items=["c"]))
        register_scripted("agent_seed_raw_m", lambda input_data, config: RawText(text="r"))
        register_scripted(
            "agent_merge_multi",
            lambda variants, config: Claims(items=[f"merged-{len(variants)}"]),
        )

        @node(
            mode="agent",
            outputs=Claims,
            model="default-tier",
            prompt="test/search",
            tools=[Tool(name="agent_search", budget=5)],
        )
        def researcher(claims: Claims, raw: RawText) -> Claims: ...

        gen_node = researcher | Oracle(n=2, merge_fn="agent_merge_multi")
        pipeline = Construct("agent-oracle-multi-dec", nodes=[
            Node.scripted("claims", fn="agent_seed_claims", outputs=Claims),
            Node.scripted("raw", fn="agent_seed_raw_m", outputs=RawText),
            gen_node,
        ])
        graph = compile(pipeline, **build_test_compile_kwargs(),
                        **configure_fake_llm(self._agent_oracle_fake))
        result = run(graph, input={"node_id": "agent-oracle-multi-dec"})
        assert result["researcher"] == Claims(items=["merged-2"]), result

    def test_each_over_agent_with_multiple_producers_fails_loud(self):
        """neograph-qzrv: packer synthesis is wired for Oracle only. Each over an
        agent with MULTIPLE producers stays fail-loud (Each also delivers a fanned
        item, which competes for the single-value boundary the packer occupies)."""
        from neograph import Each

        gen_node = (
            Node(
                name="agent-gen",
                mode="agent",
                inputs={"claims": Claims, "raw": RawText},
                outputs=Claims,
                model="default-tier",
                prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Each(over="upstream", key="text")
        )
        with pytest.raises(ConstructError, match="multiple upstream inputs"):
            Construct("bad-each-agent-multi", nodes=[gen_node])

    def test_oracle_models_round_robin_on_think_mode(self):
        """Round-robin model assignment works on think-mode nodes when n > len(models)."""
        from tests.fakes import register_scripted

        seen_tiers = []

        def tier_capturing_factory(tier):
            seen_tiers.append(tier)
            return StructuredFake(lambda m: m(items=[f"from-{tier}"]))

        __llm_kw = configure_fake_llm(tier_capturing_factory)

        def rr_think_merge(variants, config):
            return Claims(items=[f"{len(variants)} variants"])

        register_scripted("rr_think_merge", rr_think_merge)

        gen_node = (
            Node(
                name="rr-think",
                mode="think",
                outputs=Claims,
                model="default-tier",
                prompt="test/generate",
            )
            | Oracle(n=5, models=["alpha", "beta"], merge_fn="rr_think_merge")
        )
        pipeline = Construct("test-rr-think", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "rr-think"})

        gen_tiers = [t for t in seen_tiers if t in ("alpha", "beta")]
        assert len(gen_tiers) == 5, f"Expected 5 generator calls, got tiers: {seen_tiers}"
        assert gen_tiers.count("alpha") == 3  # 0,2,4
        assert gen_tiers.count("beta") == 2   # 1,3

    def test_oracle_model_does_not_leak_to_merge_node(self):
        """Merge node must use merge_model, not a generator's oracle model override."""

        seen_tiers = []

        def tier_capturing_factory(tier):
            seen_tiers.append(tier)
            return StructuredFake(lambda m: m(items=[f"from-{tier}"]))

        __llm_kw = configure_fake_llm(tier_capturing_factory)

        gen_node = (
            Node(
                name="leak-gen",
                mode="think",
                outputs=Claims,
                model="default-tier",
                prompt="test/generate",
            )
            | Oracle(
                models=["reason", "fast"],
                merge_prompt="test/merge",
                merge_model="judge-tier",
            )
        )
        pipeline = Construct("test-leak", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "leak-test"})

        # The merge call should use "judge-tier", not "reason" or "fast"
        # Generator calls use the oracle model overrides
        gen_tiers = [t for t in seen_tiers if t in ("reason", "fast")]
        merge_tiers = [t for t in seen_tiers if t == "judge-tier"]
        assert len(gen_tiers) == 2, f"Expected 2 generator tiers, got: {seen_tiers}"
        assert len(merge_tiers) == 1, f"Expected 1 merge call with judge-tier, got: {seen_tiers}"

    def test_oracle_raises_when_models_is_empty_list(self):
        """Oracle(models=[]) must raise ConfigurationError, not silently fall back to n=3."""
        with pytest.raises(ConfigurationError, match="models= must not be empty"):
            Oracle(models=[], merge_fn="x")

    def test_oracle_accepts_single_model(self):
        """Oracle(models=["a"]) is valid — single model ensemble."""
        oracle = Oracle(models=["a"], merge_fn="x")
        assert oracle.models == ["a"]
        assert oracle.n == 1

    def test_oracle_accepts_multiple_models(self):
        """Oracle(models=["a", "b"]) is valid."""
        oracle = Oracle(models=["a", "b"], merge_fn="x")
        assert oracle.models == ["a", "b"]
        assert oracle.n == 2

    def test_oracle_accepts_none_models(self):
        """Oracle(models=None) is valid — means no model override, uses default n."""
        oracle = Oracle(models=None, merge_fn="x")
        assert oracle.models is None
        assert oracle.n == 3

    def test_body_as_merge_receives_list_not_single_type(self):
        """Body-as-merge: param annotation says upstream type T, but body
        receives list[T] at runtime. This documents the intentional mismatch
        (neograph-qr9v) — the annotation is for compile-time wiring, not a
        runtime type contract."""
        from tests.fakes import register_scripted

        received_type = [None]

        # Generator: registered scripted so we control exactly what it returns
        def bam_typed_gen(input_data, config):
            model = config.get("configurable", {}).get("_oracle_model", "x")
            return Claims(items=[f"from-{model}"])

        register_scripted("bam_typed_gen", bam_typed_gen)

        gen_node = (
            Node.scripted("bam-typed-gen", fn="bam_typed_gen", outputs=Claims)
            | Oracle(models=["reason", "fast"], merge_fn="bam_typed_merge")
        )

        def bam_typed_merge(variants, config):
            received_type[0] = type(variants)
            # Verify each variant is a Claims instance
            assert all(isinstance(v, Claims) for v in variants)
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("bam_typed_merge", bam_typed_merge)

        pipeline = Construct("body-merge-typed", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "body-merge-typed"})

        # The merge function receives a list, not a single Claims
        assert received_type[0] is list
        merged = result["bam_typed_gen"]
        assert len(merged.items) == 2

    def test_body_as_merge_decorator_param_annotation_is_upstream_type(self):
        """The @node body-as-merge parameter is annotated as the upstream type
        (for compile-time wiring) but receives list[OutputType] at runtime.
        This documents the intentional mismatch (neograph-qr9v)."""
        from typing import get_type_hints

        from neograph.decorators import _get_sidecar

        @node(outputs=Claims, models=["reason", "fast"])
        def merge_check(data: Claims) -> Claims:
            # data is list[Claims] at runtime, not Claims
            all_items = []
            for v in data:
                all_items.extend(v.items)
            return Claims(items=all_items)

        # Compile-time: the annotation says Claims (for wiring)
        sidecar = _get_sidecar(merge_check)
        assert isinstance(sidecar, tuple), "Sidecar should be a (fn, param_names) tuple"
        original_fn = sidecar[0]
        hints = get_type_hints(original_fn)
        assert hints["data"] is Claims, (
            "Parameter annotation should be the upstream type Claims"
        )

        # The node has an Oracle modifier attached
        assert merge_check.has_modifier(Oracle)

    def test_node_decorator_raises_when_models_is_empty_list(self):
        """@node(models=[]) must raise at decoration time."""

        with pytest.raises((ConfigurationError, ConstructError)):
            @node(outputs=Claims, models=[])
            def bad_ensemble(data: Claims) -> Claims:
                return data


# =============================================================================
# BUG REGRESSION: neograph-bglm
# merge_fn exceptions must surface, not produce silent garbage
# =============================================================================





# =============================================================================
# BUG REGRESSION: neograph-bglm
# merge_fn exceptions must surface, not produce silent garbage
# =============================================================================


class TestOracleMergeFnErrors:
    """When a merge_fn throws an exception, neograph must propagate it —
    not silently continue with whatever state the node had before the merge."""

    def test_exception_propagates_when_merge_fn_raises(self):
        """merge_fn that raises AttributeError must crash the pipeline,
        not produce silent garbage results."""
        from tests.fakes import register_scripted

        register_scripted("bglm_gen", lambda input_data, config: Claims(items=["v1"]))

        def bad_merge(variants, config):
            raise AttributeError("ModelRole.FAST doesn't exist")

        register_scripted("bglm_bad_merge", bad_merge)

        gen_node = (
            Node.scripted("gen", fn="bglm_gen", outputs=Claims)
            | Oracle(n=2, merge_fn="bglm_bad_merge")
        )
        pipeline = Construct("test-merge-error", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs())

        # The pipeline MUST fail — not silently produce garbage
        with pytest.raises(Exception, match="ModelRole.FAST"):
            run(graph, input={"node_id": "bglm-test"})

    def test_wrong_return_type_raises_when_merge_fn_returns_bad_type(self):
        """merge_fn that returns the wrong type should be caught."""
        from tests.fakes import register_scripted

        register_scripted("bglm_gen2", lambda input_data, config: Claims(items=["v1"]))

        def wrong_type_merge(variants, config):
            # Returns a string instead of Claims
            return "this is not a Claims object"

        register_scripted("bglm_wrong_merge", wrong_type_merge)

        gen_node = (
            Node.scripted("gen2", fn="bglm_gen2", outputs=Claims)
            | Oracle(n=2, merge_fn="bglm_wrong_merge")
        )
        pipeline = Construct("test-merge-type", nodes=[gen_node])
        graph = compile(pipeline, **build_test_compile_kwargs())

        # Should raise because merge result doesn't match output type
        with pytest.raises(ExecutionError, match="(?i)merge.*type|expected.*Claims"):
            run(graph, input={"node_id": "bglm-test2"})


# ═══════════════════════════════════════════════════════════════════════════
# MERGE_FN STATE PARAMS (neograph-jg2g)
#
# @merge_fn can auto-wire params from graph state by name.
# ═══════════════════════════════════════════════════════════════════════════





# ═══════════════════════════════════════════════════════════════════════════
# MERGE_FN STATE PARAMS (neograph-jg2g)
#
# @merge_fn can auto-wire params from graph state by name.
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeFnStateParams:
    """@merge_fn can access upstream node outputs via state params."""

    def test_merge_fn_receives_upstream_state_value(self):
        """A @merge_fn param named after an upstream node reads from state."""
        from pydantic import BaseModel

        from tests.fakes import register_scripted

        class Context(BaseModel, frozen=True):
            topic: str

        class Result(BaseModel, frozen=True):
            text: str
            topic: str

        register_scripted("jg2g_ctx", lambda i, c: Context(topic="AI safety"))
        register_scripted("jg2g_gen", lambda i, c: Result(text="draft", topic=""))

        @merge_fn
        def merge_with_context(
            variants: list[Result],
            context: Context,  # auto-wired from state field "context"
        ) -> Result:
            return Result(text=variants[0].text, topic=context.topic)

        pipeline = Construct("merge-state", nodes=[
            Node.scripted("context", fn="jg2g_ctx", outputs=Context),
            Node.scripted("gen", fn="jg2g_gen", outputs=Result)
            | Oracle(n=2, merge_fn="merge_with_context"),
        ])
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "jg2g-test"})

        assert result["gen"].topic == "AI safety"

    def test_merge_fn_mixes_state_and_di_params(self):
        """@merge_fn can have both state params and DI params."""
        from pydantic import BaseModel

        from neograph import FromInput
        from tests.fakes import register_scripted

        class Metadata(BaseModel, frozen=True):
            source: str

        class Result(BaseModel, frozen=True):
            text: str
            label: str

        register_scripted("jg2g_meta", lambda i, c: Metadata(source="api"))
        register_scripted("jg2g_gen2", lambda i, c: Result(text="v1", label=""))

        @merge_fn
        def merge_mixed(
            variants: list[Result],
            metadata: Metadata,  # from state
            node_id: Annotated[str, FromInput],  # from config
        ) -> Result:
            return Result(
                text=variants[0].text,
                label=f"{metadata.source}:{node_id}",
            )

        pipeline = Construct("merge-mixed", nodes=[
            Node.scripted("metadata", fn="jg2g_meta", outputs=Metadata),
            Node.scripted("gen2", fn="jg2g_gen2", outputs=Result)
            | Oracle(n=2, merge_fn="merge_mixed"),
        ])
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "mixed-test"})

        assert result["gen2"].label == "api:mixed-test"

    def test_merge_fn_state_param_rejects_unknown_field(self):
        """State param for a field not in state raises ConstructError."""
        from pydantic import BaseModel

        from tests.fakes import register_scripted

        class Result(BaseModel, frozen=True):
            text: str

        register_scripted("jg2g_gen3", lambda i, c: Result(text="ok"))

        @merge_fn
        def merge_missing(
            variants: list[Result],
            nonexistent_field: str | None,  # no upstream with this name
        ) -> Result:
            return variants[0]

        with pytest.raises(ConstructError, match="does not match any upstream"):
            Construct("merge-missing", nodes=[
                Node.scripted("gen3", fn="jg2g_gen3", outputs=Result)
                | Oracle(n=2, merge_fn="merge_missing"),
            ])


    def test_merge_fn_from_state_unwraps_loop_append_list(self):
        """Regression neograph-8i1g: from_state on a Loop node must unwrap [-1]."""
        from pydantic import BaseModel

        from neograph.modifiers import Loop
        from tests.fakes import register_scripted

        class Ctx(BaseModel, frozen=True):
            topic: str

        class Result(BaseModel, frozen=True):
            text: str

        register_scripted("8i1g_ctx", lambda i, c: Ctx(topic="safety"))
        register_scripted("8i1g_gen", lambda i, c: Result(text="v1"))

        captured_ctx = [None]

        @merge_fn
        def ctx_aware_merge(variants: list[Result], ctx: Ctx) -> Result:
            captured_ctx[0] = ctx
            return variants[0]

        pipeline = Construct("8i1g-test", nodes=[
            Node.scripted("ctx", fn="8i1g_ctx", outputs=Ctx)
            | Loop(when=lambda c: c is None or c.topic == "first_only", max_iterations=2),
            Node.scripted("gen", fn="8i1g_gen", outputs=Result)
            | Oracle(n=2, merge_fn="ctx_aware_merge"),
        ])
        graph = compile(pipeline, **build_test_compile_kwargs())
        run(graph, input={"node_id": "8i1g"})

        # from_state should unwrap the Loop append-list to get Ctx, not list[Ctx]
        assert isinstance(captured_ctx[0], Ctx), (
            f"Expected Ctx, got {type(captured_ctx[0])}: {captured_ctx[0]}"
        )
        assert captured_ctx[0].topic == "safety"


# ═══════════════════════════════════════════════════════════════════════════
# MERGE_PROMPT UPSTREAM CONTEXT (neograph-26eg)
#
# merge_prompt receives upstream context alongside variant list.
# Templates reference ${variants} for the N drafts and ${upstream.field}
# for upstream data via dotted access.
# ═══════════════════════════════════════════════════════════════════════════


class UpstreamContext(BaseModel, frozen=True):
    site_name: str
    tone: str


class Draft(BaseModel, frozen=True):
    text: str


class TestMergePromptUpstreamContext:
    """merge_prompt should receive upstream context alongside variant list."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_merge_prompt_receives_upstream_context_as_dict(self):
        """merge_prompt input_data should be dict with 'variants' + upstream keys."""
        captured_input = {}

        class CaptureMerge:
            """Fake that captures the merge prompt's input."""
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                # Capture merge call (tier="reason") vs generator calls
                if self._tier == "reason":
                    captured_input["messages"] = messages
                return self._model(text="merged")

        __llm_kw = configure_fake_llm(lambda tier: CaptureMerge(tier))

        @node(outputs=UpstreamContext)
        def enrich() -> UpstreamContext:
            return UpstreamContext(site_name="Acme Corp", tone="professional")

        @node(outputs=Draft, ensemble_n=2,
              merge_prompt="Pick the best considering ${enrich.site_name}: ${variants}",
              prompt="draft using ${enrich.site_name}", model="fast")
        def write(enrich: UpstreamContext) -> Draft: ...

        mod = self._fresh_module("test_merge_ctx")
        mod.enrich = enrich
        mod.write = write

        from neograph import construct_from_module
        pipeline = construct_from_module(mod, name="merge-ctx")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t1"})

        # The merge prompt should have received upstream context
        msgs = captured_input.get("messages", [])
        prompt_text = msgs[0]["content"] if msgs else ""
        # The upstream 'enrich' value should appear in the prompt
        assert "Acme Corp" in prompt_text, (
            f"Upstream context 'Acme Corp' not found in merge prompt: {prompt_text}"
        )

    def test_merge_prompt_variants_key_contains_all_variants(self):
        """The 'variants' key should contain all N generator outputs."""
        captured_data = {}

        class InspectMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    captured_data["prompt"] = messages[0]["content"] if messages else ""
                return self._model(text="variant-output")

        __llm_kw = configure_fake_llm(lambda tier: InspectMerge(tier))

        @node(outputs=Draft, ensemble_n=3, merge_prompt="Judge: ${variants}",
              prompt="generate a draft", model="fast")
        def generate() -> Draft: ...

        mod = self._fresh_module("test_variants")
        mod.generate = generate

        from neograph import construct_from_module
        pipeline = construct_from_module(mod, name="variants-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t2"})

        # Variants should be rendered in the prompt (BAML notation of Draft list)
        prompt = captured_data.get("prompt", "")
        assert "variant-output" in prompt, f"Variants not rendered: {prompt}"

    def test_merge_fn_path_unchanged(self):
        """@merge_fn still receives raw list, not dict."""
        captured_args = {}

        @merge_fn
        def pick_best(variants: list[Draft]) -> Draft:
            captured_args["type"] = type(variants).__name__
            captured_args["len"] = len(variants)
            return variants[0] if variants else Draft(text="empty")

        @node(outputs=Draft, ensemble_n=2, merge_fn="pick_best")
        def generate() -> Draft:
            return Draft(text="v1")

        mod = self._fresh_module("test_mfn")
        mod.generate = generate

        from neograph import construct_from_module
        pipeline = construct_from_module(mod, name="mfn-test")
        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "t3"})

        # merge_fn should still receive list, not dict
        assert captured_args["type"] == "list"
        assert captured_args["len"] == 2

    def test_programmatic_oracle_merge_prompt_gets_upstream(self):
        """Programmatic Node | Oracle(merge_prompt=...) also gets upstream context."""
        captured_input = {}

        class CaptureMerge3:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    captured_input["prompt"] = messages[0]["content"] if messages else ""
                return self._model(text="merged")

        __llm_kw = configure_fake_llm(lambda tier: CaptureMerge3(tier))

        register_scripted("_mp_seed", lambda i, c: UpstreamContext(site_name="Test Site", tone="casual"))

        seed = Node.scripted("seed", fn="_mp_seed", outputs=UpstreamContext)
        writer = Node(
            "writer", mode="think", outputs=Draft,
            inputs={"seed": UpstreamContext},
            prompt="write", model="fast",
        ) | Oracle(n=2, merge_prompt="Pick best: ${seed.site_name} ${variants}")

        pipeline = Construct("prog-merge", nodes=[seed, writer])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t4"})

        prompt = captured_input.get("prompt", "")
        assert "Test Site" in prompt, (
            f"Upstream context 'Test Site' missing from programmatic merge prompt: {prompt}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MERGE HOOKS: pre_process, post_process, fallback (neograph-apki)
#
# Optional callbacks that bracket the merge_prompt LLM dispatch.
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeHooks:
    """merge_pre_process, merge_post_process, merge_fallback on Oracle."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_pre_process_transforms_variants_before_llm(self):
        """merge_pre_process should replace default input_data for the merge prompt."""
        captured_input = {}

        class CaptureMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    captured_input["prompt"] = messages[0]["content"] if messages else ""
                return self._model(text="merged")

        __llm_kw = configure_fake_llm(lambda tier: CaptureMerge(tier))

        def tag_variants(variants):
            tagged = [f"[v{i}] {v.text}" for i, v in enumerate(variants)]
            return {"tagged_items": "\n".join(tagged)}

        @node(outputs=Draft, ensemble_n=2,
              prompt="generate draft", model="fast",
              merge_prompt="Merge these tagged items: ${tagged_items}",
              merge_pre_process=tag_variants)
        def write() -> Draft: ...

        mod = self._fresh_module("test_pre")
        mod.write = write

        pipeline = construct_from_module(mod, name="pre-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-pre"})

        prompt = captured_input.get("prompt", "")
        assert "[v0]" in prompt, f"Pre-processed tag not in prompt: {prompt}"
        assert "[v1]" in prompt, f"Pre-processed tag not in prompt: {prompt}"

    def test_post_process_transforms_merged_result(self):
        """merge_post_process should transform the LLM result before state write."""
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(text="draft-output")))

        post_called = []

        def uppercase_result(result, variants):
            post_called.append(True)
            return Draft(text=result.text.upper())

        @node(outputs=Draft, ensemble_n=2,
              prompt="generate draft", model="fast",
              merge_prompt="pick best: ${variants}",
              merge_post_process=uppercase_result)
        def write() -> Draft: ...

        mod = self._fresh_module("test_post")
        mod.write = write

        pipeline = construct_from_module(mod, name="post-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-post"})

        assert len(post_called) == 1, "post_process should have been called once"
        assert result["write"].text == "DRAFT-OUTPUT"

    def test_fallback_catches_llm_error(self):
        """merge_fallback should catch LLM errors and return a deterministic result."""
        class FailOnMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    raise RuntimeError("LLM merge failed")
                return self._model(text="variant")

        __llm_kw = configure_fake_llm(lambda tier: FailOnMerge(tier))

        def fallback_fn(variants, error):
            return Draft(text=f"fallback-{len(variants)}")

        @node(outputs=Draft, ensemble_n=2,
              prompt="generate draft", model="fast",
              merge_prompt="merge: ${variants}",
              merge_fallback=fallback_fn)
        def write() -> Draft: ...

        mod = self._fresh_module("test_fallback")
        mod.write = write

        pipeline = construct_from_module(mod, name="fallback-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-fb"})

        assert result["write"].text == "fallback-2"

    def test_hooks_rejected_with_merge_fn(self):
        """Hooks must raise ConfigurationError when combined with merge_fn."""
        with pytest.raises(ConfigurationError):
            Oracle(
                merge_fn="some_fn",
                merge_pre_process=lambda v: {"x": v},
            )

        with pytest.raises(ConfigurationError):
            Oracle(
                merge_fn="some_fn",
                merge_post_process=lambda r, v: r,
            )

        with pytest.raises(ConfigurationError):
            Oracle(
                merge_fn="some_fn",
                merge_fallback=lambda v, e: v[0],
            )

    def test_programmatic_oracle_with_hooks(self):
        """Programmatic Node | Oracle(merge_prompt=..., hooks=...) should work."""
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(text="gen-output")))

        post_called = []

        def track_post(result, variants):
            post_called.append(len(variants))
            return result

        writer = Node(
            "writer", mode="think", outputs=Draft,
            prompt="write", model="fast",
        ) | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_post_process=track_post,
        )

        pipeline = Construct("hook-prog", nodes=[writer])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-prog"})

        assert len(post_called) == 1
        assert post_called[0] == 2

    def test_no_fallback_error_propagates(self):
        """Without fallback, LLM errors propagate with original type and message."""
        class FailMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    raise RuntimeError("LLM unavailable")
                return self._model(text="v")

        __llm_kw = configure_fake_llm(lambda tier: FailMerge(tier))

        @node(outputs=Draft, ensemble_n=2,
              prompt="gen", model="fast",
              merge_prompt="merge: ${variants}")
        def write() -> Draft: ...

        mod = self._fresh_module("test_nofb")
        mod.write = write

        pipeline = construct_from_module(mod, name="nofb-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            run(graph, input={"node_id": "t-nofb"})

    def test_dict_form_outputs_with_post_process(self):
        """post_process receives primary values from dict-form outputs."""
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(text="llm")))

        post_variants = []

        def track_post(result, variants):
            post_variants.extend(variants)
            return Draft(text=result.text + "-pp")

        writer = Node(
            "writer", mode="think",
            outputs={"result": Draft, "meta": Draft},
            prompt="write", model="fast",
        ) | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_post_process=track_post,
        )

        pipeline = Construct("dict-hook", nodes=[writer])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-dict"})

        # post_process should have received primary values (Draft instances)
        assert len(post_variants) == 2
        assert all(isinstance(v, Draft) for v in post_variants)
        # The merged result goes to the primary field
        assert result["writer_result"].text == "llm-pp"

    def test_body_as_merge_with_hooks_rejected(self):
        """body-as-merge converts to merge_fn; hooks should be rejected."""
        with pytest.raises(ConfigurationError, match="merge hooks"):
            @node(outputs=Draft, models=["fast", "reason"],
                  merge_pre_process=lambda v: {"x": v})
            def write(variants) -> Draft:
                return variants[0]

    def test_pre_process_displaces_upstream_context(self):
        """When pre_process is set, upstream context is NOT auto-injected."""
        captured = {}

        class CaptureMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    captured["prompt"] = messages[0]["content"] if messages else ""
                return self._model(text="merged")

        __llm_kw = configure_fake_llm(lambda tier: CaptureMerge(tier))

        def custom_pre(variants):
            return {"custom_key": "custom_value"}

        @node(outputs=UpstreamContext)
        def enrich() -> UpstreamContext:
            return UpstreamContext(site_name="ShouldNotAppear", tone="x")

        @node(outputs=Draft, ensemble_n=2,
              prompt="gen", model="fast",
              merge_prompt="Merge: ${custom_key}",
              merge_pre_process=custom_pre)
        def write(enrich: UpstreamContext) -> Draft: ...

        mod = self._fresh_module("test_displace")
        mod.enrich = enrich
        mod.write = write

        pipeline = construct_from_module(mod, name="displace-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        run(graph, input={"node_id": "t-disp"})

        prompt = captured.get("prompt", "")
        assert "custom_value" in prompt, f"pre_process key missing: {prompt}"
        assert "ShouldNotAppear" not in prompt, (
            f"Upstream context should be displaced by pre_process: {prompt}"
        )

    def test_upstream_context_preserved_without_pre_process(self):
        """Without pre_process, upstream context still auto-injected into merge prompt."""
        captured = {}

        class CaptureMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    captured["prompt"] = messages[0]["content"] if messages else ""
                return self._model(text="merged")

        __llm_kw = configure_fake_llm(lambda tier: CaptureMerge(tier))

        @node(outputs=UpstreamContext)
        def enrich() -> UpstreamContext:
            return UpstreamContext(site_name="AcmeCorp", tone="formal")

        @node(outputs=Draft, ensemble_n=2,
              prompt="gen", model="fast",
              merge_prompt="Merge considering ${enrich.site_name}: ${variants}")
        def write(enrich: UpstreamContext) -> Draft: ...

        mod = self._fresh_module("test_preserve")
        mod.enrich = enrich
        mod.write = write

        pipeline = construct_from_module(mod, name="preserve-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        run(graph, input={"node_id": "t-pres"})

        prompt = captured.get("prompt", "")
        assert "AcmeCorp" in prompt, f"Upstream context missing: {prompt}"

    def test_all_three_hooks_success_path(self):
        """All hooks set, LLM succeeds: pre + post called, fallback NOT called."""
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(text="llm")))

        call_log = []

        def pre(variants):
            call_log.append("pre")
            return {"items": str(len(variants))}

        def post(result, variants):
            call_log.append("post")
            return Draft(text=result.text + "-post")

        def fb(variants, error):
            call_log.append("fallback")
            return Draft(text="fb")

        @node(outputs=Draft, ensemble_n=2,
              prompt="gen", model="fast",
              merge_prompt="Merge ${items}",
              merge_pre_process=pre,
              merge_post_process=post,
              merge_fallback=fb)
        def write() -> Draft: ...

        mod = self._fresh_module("test_all3_ok")
        mod.write = write

        pipeline = construct_from_module(mod, name="all3-ok")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-3ok"})

        assert call_log == ["pre", "post"]
        assert result["write"].text == "llm-post"

    def test_all_three_hooks_failure_path(self):
        """All hooks set, LLM fails: pre + fallback called, post NOT called."""
        class FailMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    raise RuntimeError("boom")
                return self._model(text="v")

        __llm_kw = configure_fake_llm(lambda tier: FailMerge(tier))

        call_log = []

        def pre(variants):
            call_log.append("pre")
            return {"items": str(len(variants))}

        def post(result, variants):
            call_log.append("post")
            return result

        def fb(variants, error):
            call_log.append("fallback")
            return Draft(text="fb")

        @node(outputs=Draft, ensemble_n=2,
              prompt="gen", model="fast",
              merge_prompt="Merge ${items}",
              merge_pre_process=pre,
              merge_post_process=post,
              merge_fallback=fb)
        def write() -> Draft: ...

        mod = self._fresh_module("test_all3_fail")
        mod.write = write

        pipeline = construct_from_module(mod, name="all3-fail")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-3f"})

        assert call_log == ["pre", "fallback"]
        assert result["write"].text == "fb"

    def test_programmatic_pre_process(self):
        """Programmatic Node | Oracle with merge_pre_process."""
        captured = {}

        class CaptureMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    captured["prompt"] = messages[0]["content"] if messages else ""
                return self._model(text="merged")

        __llm_kw = configure_fake_llm(lambda tier: CaptureMerge(tier))

        def pre(variants):
            return {"count": str(len(variants))}

        writer = Node(
            "writer", mode="think", outputs=Draft,
            prompt="write", model="fast",
        ) | Oracle(
            n=2,
            merge_prompt="Merge ${count} variants",
            merge_pre_process=pre,
        )

        pipeline = Construct("prog-pre", nodes=[writer])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        run(graph, input={"node_id": "t-pp"})

        assert "2" in captured.get("prompt", ""), f"pre_process not applied: {captured}"

    def test_programmatic_fallback(self):
        """Programmatic Node | Oracle with merge_fallback."""
        class FailMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    raise RuntimeError("merge boom")
                return self._model(text="v")

        __llm_kw = configure_fake_llm(lambda tier: FailMerge(tier))

        writer = Node(
            "writer", mode="think", outputs=Draft,
            prompt="write", model="fast",
        ) | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_fallback=lambda v, e: Draft(text=f"fb-{len(v)}"),
        )

        pipeline = Construct("prog-fb", nodes=[writer])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-pfb"})

        assert result["writer"].text == "fb-2"

    def test_declarative_node_with_hooks(self):
        """Declarative Node.scripted() + Oracle with hooks (three-surface rule)."""
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(text="llm-out")))

        post_called = []

        def track_post(result, variants):
            post_called.append(True)
            return Draft(text=result.text + "-post")

        register_scripted("_decl_hook_gen", lambda i, c: Draft(text="gen"))

        writer = Node(
            "writer", mode="think", outputs=Draft,
            prompt="write", model="fast",
        ) | Oracle(
            n=2,
            merge_prompt="merge: ${variants}",
            merge_post_process=track_post,
        )

        pipeline = Construct("decl-hook", nodes=[writer])
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-decl"})

        assert len(post_called) == 1
        assert result["writer"].text == "llm-out-post"

    def test_merge_prompt_without_hooks_unchanged(self):
        """merge_prompt without hooks works exactly as before (regression guard)."""
        captured_input = {}

        class CaptureMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    captured_input["prompt"] = messages[0]["content"] if messages else ""
                return self._model(text="merged")

        __llm_kw = configure_fake_llm(lambda tier: CaptureMerge(tier))

        @node(outputs=Draft, ensemble_n=2,
              prompt="generate draft", model="fast",
              merge_prompt="Judge: ${variants}")
        def write() -> Draft: ...

        mod = self._fresh_module("test_nochange")
        mod.write = write

        pipeline = construct_from_module(mod, name="nochange-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "t-nc"})

        prompt = captured_input.get("prompt", "")
        assert "Judge:" in prompt

    def test_post_process_error_not_caught_by_fallback(self):
        """post_process errors must propagate, NOT be caught by fallback."""
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(text="ok")))

        fallback_called = []

        def bad_post_process(result, variants):
            raise ValueError("post_process bug")

        def track_fallback(variants, error):
            fallback_called.append(str(error))
            return Draft(text="fallback")

        @node(outputs=Draft, ensemble_n=2,
              prompt="gen", model="fast",
              merge_prompt="merge: ${variants}",
              merge_post_process=bad_post_process,
              merge_fallback=track_fallback)
        def write() -> Draft: ...

        mod = self._fresh_module("test_pp_err")
        mod.write = write

        pipeline = construct_from_module(mod, name="pp-err-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)

        with pytest.raises(ValueError, match="post_process bug"):
            run(graph, input={"node_id": "t-pp-err"})

        assert fallback_called == [], (
            f"fallback should NOT catch post_process errors, but got: {fallback_called}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# EACH×ORACLE FUSION WITH HOOKS
#
# map_over + ensemble_n + merge hooks on the same @node.
# Tests the _merge_one_group path in _wiring.py.
# ═══════════════════════════════════════════════════════════════════════════


class ChunkItem(BaseModel, frozen=True):
    idx: str
    text: str


class ChunkList(BaseModel, frozen=True):
    items: list[ChunkItem]


class ChunkResult(BaseModel, frozen=True):
    summary: str


class TestEachOracleFusionHooks:
    """Hooks on the EachxOracle fused path (_merge_one_group)."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_post_process_in_fused_path(self):
        """merge_post_process works in EachxOracle fused path."""
        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(summary="raw")))

        post_calls = []

        def upper_post(result, variants):
            post_calls.append(len(variants))
            return ChunkResult(summary=result.summary.upper())

        @node(outputs=ChunkList)
        def make_chunks() -> ChunkList:
            return ChunkList(items=[
                ChunkItem(idx="A", text="alpha"),
                ChunkItem(idx="B", text="beta"),
            ])

        @node(outputs=ChunkResult, prompt="score", model="fast",
              map_over="make_chunks.items", map_key="idx",
              ensemble_n=2, merge_prompt="merge: ${variants}",
              merge_post_process=upper_post)
        def score(chunk: ChunkItem) -> ChunkResult: ...

        mod = self._fresh_module("test_fused_post")
        mod.make_chunks = make_chunks
        mod.score = score

        pipeline = construct_from_module(mod, name="fused-post")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "f-post"})

        scores = result["score"]
        assert isinstance(scores, dict)
        assert scores["A"].summary == "RAW"
        assert scores["B"].summary == "RAW"
        assert len(post_calls) == 2  # once per Each group

    def test_fallback_in_fused_path(self):
        """merge_fallback works in EachxOracle fused path."""
        class FailOnMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    raise RuntimeError("fused merge failed")
                return self._model(summary="gen")

        __llm_kw = configure_fake_llm(lambda tier: FailOnMerge(tier))

        def fused_fallback(variants, error):
            return ChunkResult(summary=f"fb-{len(variants)}")

        @node(outputs=ChunkList)
        def make_chunks() -> ChunkList:
            return ChunkList(items=[ChunkItem(idx="X", text="x")])

        @node(outputs=ChunkResult, prompt="score", model="fast",
              map_over="make_chunks.items", map_key="idx",
              ensemble_n=2, merge_prompt="merge: ${variants}",
              merge_fallback=fused_fallback)
        def score(chunk: ChunkItem) -> ChunkResult: ...

        mod = self._fresh_module("test_fused_fb")
        mod.make_chunks = make_chunks
        mod.score = score

        pipeline = construct_from_module(mod, name="fused-fb")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "f-fb"})

        scores = result["score"]
        assert scores["X"].summary == "fb-2"

    def test_pre_process_in_fused_path(self):
        """merge_pre_process works in EachxOracle fused path."""
        captured = {}

        class CaptureMerge:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                if self._tier == "reason":
                    captured.setdefault("prompts", []).append(
                        messages[0]["content"] if messages else "")
                return self._model(summary="merged")

        __llm_kw = configure_fake_llm(lambda tier: CaptureMerge(tier))

        def tag_pre(variants):
            return {"tagged": f"count={len(variants)}"}

        @node(outputs=ChunkList)
        def make_chunks() -> ChunkList:
            return ChunkList(items=[ChunkItem(idx="Z", text="z")])

        @node(outputs=ChunkResult, prompt="score", model="fast",
              map_over="make_chunks.items", map_key="idx",
              ensemble_n=2, merge_prompt="Merge tagged: ${tagged}",
              merge_pre_process=tag_pre)
        def score(chunk: ChunkItem) -> ChunkResult: ...

        mod = self._fresh_module("test_fused_pre")
        mod.make_chunks = make_chunks
        mod.score = score

        pipeline = construct_from_module(mod, name="fused-pre")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        result = run(graph, input={"node_id": "f-pre"})

        assert any("count=2" in p for p in captured.get("prompts", [])), (
            f"pre_process tag not in prompts: {captured}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# ORACLE MERGE LLM_CONFIG PROPAGATION (neograph-63g9)
# ═══════════════════════════════════════════════════════════════════════════


class TestOracleMergeLlmConfig:
    """Oracle merge_prompt must propagate the node's llm_config to invoke_structured."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_merge_prompt_receives_node_llm_config(self, monkeypatch):
        """invoke_structured in the merge path should receive node.llm_config."""
        captured_calls = []

        __llm_kw = configure_fake_llm(lambda tier: StructuredFake(lambda m: m(text="out")))

        # Patch _llm.invoke_structured to capture all calls
        from neograph import _llm
        original = _llm.invoke_structured

        def capture_invoke(
            *args,
            model_tier=None, prompt_template=None, input_data=None,
            output_model=None, config=None,
            node_name="", llm_config=None, context=None, runtime=None,
            **kwargs,
        ):
            captured_calls.append({
                "model_tier": model_tier,
                "prompt_template": prompt_template,
                "llm_config": llm_config,
            })
            return output_model(text="fake")

        monkeypatch.setattr(_llm, "invoke_structured", capture_invoke)

        @node(outputs=Draft, ensemble_n=2,
              prompt="gen", model="fast",
              merge_prompt="merge: ${variants}",
              llm_config={
                  "output_strategy": "json_mode",
                  "provider_kwargs": {"temperature": 0.5},
              })
        def write() -> Draft: ...

        mod = self._fresh_module("test_llm_cfg")
        mod.write = write

        pipeline = construct_from_module(mod, name="llm-cfg-test")
        graph = compile(pipeline, **build_test_compile_kwargs(), **__llm_kw)
        run(graph, input={"node_id": "t-cfg"})

        # Should have 3 calls: 2 generators + 1 merge
        assert len(captured_calls) == 3, f"Expected 3 calls, got {len(captured_calls)}"

        # The merge call is the last one (tier="reason", prompt contains "merge")
        merge_call = captured_calls[-1]
        assert merge_call["model_tier"] == "reason", f"Last call should be merge: {merge_call}"

        merge_llm_config = merge_call["llm_config"]
        assert merge_llm_config is not None, (
            "merge invoke_structured did not receive llm_config"
        )
        # Post-pej0: llm_config is a typed LlmConfig at this boundary.
        assert merge_llm_config.output_strategy == "json_mode", (
            f"merge should inherit node's output_strategy, got: {merge_llm_config}"
        )




# ═══════════════════════════════════════════════════════════════════════════
# neograph-qzrv: bundle-port synthesis (multi-input) + value-delivery pins
# ═══════════════════════════════════════════════════════════════════════════


class _Alpha(BaseModel, frozen=True):
    a: str


class _Beta(BaseModel, frozen=True):
    b: str


def _echo_oracle_prompt_compiler(template, input_data, **kwargs):
    """Render the agent's resolved input into the user turn so the fake's echo
    proves the upstream value(s) reached THIS isolated ReAct cycle across the
    subgraph boundary (neograph-qzrv wave-8 value-delivery pin)."""
    return [{"role": "user", "content": f"UPSTREAM={input_data!r}"}]


class _EchoOracleReActFake:
    """A ReAct fake whose FINAL turn reflects the rendered prompt (the upstream
    value it SAW) into ``Claims.items``. One tool call first exercises the
    multi-superstep cycle. Fresh per factory call so each isolated Oracle variant
    drives its own loop."""

    def __init__(self, *, call_tool: str | None = "agent_search"):
        self._model = None
        self._structured = False
        self._call_tool = call_tool

    def bind_tools(self, tools):
        return self

    def abind_tools(self, *a, **k):
        return self

    def _seen(self, messages) -> str:
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if content and "UPSTREAM=" in str(content):
                return str(content)
        return ""

    def invoke(self, messages, **kwargs):
        import json

        from langchain_core.messages import AIMessage, ToolMessage

        n_tool_results = sum(isinstance(m, ToolMessage) for m in messages)
        if self._call_tool and n_tool_results == 0:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": self._call_tool, "args": {}, "id": "t1"}]
            return msg
        return AIMessage(content=json.dumps({"items": [self._seen(messages)]}))

    async def ainvoke(self, *a, **k):
        return self.invoke(*a, **k)

    def with_structured_output(self, model, **kwargs):
        clone = _EchoOracleReActFake(call_tool=self._call_tool)
        clone._model = model
        clone._structured = True
        return clone


def _echo_merge_concat(variants, config):
    """Concatenate every variant's seen-content — so the assertion can confirm
    EACH isolated cycle saw the upstream value (not just merged-N wiring)."""
    seen = [item for v in variants for item in v.items]
    return Claims(items=seen)


class TestOracleOverAgentValueDelivery:
    """Wave-8 pin (neograph-qzrv): the qot6 agent-fan tests assert merged-N wiring
    but NOT that each isolated cycle SAW its upstream value. These echo-style tests
    render the upstream into the prompt, reflect it via the fake, and concatenate in
    the merge — so 'each isolated cycle saw its upstream value' is committed, for the
    single-type, dict-form, and DI shapes."""

    def _compile(self, pipeline):
        return compile(
            pipeline,
            **build_test_compile_kwargs(),
            **configure_fake_llm(
                lambda tier: _EchoOracleReActFake(),
                _echo_oracle_prompt_compiler,
            ),
        )

    def _register_common(self):
        register_tool_factory(
            "agent_search", lambda config, tool_config: FakeTool("agent_search", response="found")
        )
        register_scripted("echo_merge", _echo_merge_concat)

    def test_single_type_input_value_reaches_each_isolated_cycle(self):
        self._register_common()
        register_scripted("seed_raw_vd", lambda input_data, config: RawText(text="SEEDVAL"))
        gen = (
            Node(
                name="agent-gen", mode="agent", inputs=RawText, outputs=Claims,
                model="default-tier", prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Oracle(n=2, merge_fn="echo_merge")
        )
        pipeline = Construct("vd-single", nodes=[
            Node.scripted("seed", fn="seed_raw_vd", outputs=RawText),
            gen,
        ])
        result = run(self._compile(pipeline), input={"node_id": "vd-single"})
        items = result["agent_gen"].items
        assert len(items) == 2, items  # merged-2 wiring
        assert all("SEEDVAL" in seen for seen in items), items  # value delivery

    def test_dict_form_input_value_reaches_each_isolated_cycle(self):
        self._register_common()
        register_scripted("seed_raw_vd2", lambda input_data, config: RawText(text="SEEDVAL"))
        gen = (
            Node(
                name="agent-gen", mode="agent", inputs={"seed": RawText}, outputs=Claims,
                model="default-tier", prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Oracle(n=2, merge_fn="echo_merge")
        )
        pipeline = Construct("vd-dict", nodes=[
            Node.scripted("seed", fn="seed_raw_vd2", outputs=RawText),
            gen,
        ])
        result = run(self._compile(pipeline), input={"node_id": "vd-dict"})
        items = result["agent_gen"].items
        assert len(items) == 2, items
        assert all("SEEDVAL" in seen for seen in items), items

    def test_di_param_value_reaches_each_isolated_cycle(self):
        from neograph import FromInput

        self._register_common()
        register_scripted("seed_raw_vd3", lambda input_data, config: RawText(text="SEEDVAL"))

        @node(
            mode="agent", outputs=Claims, model="default-tier", prompt="test/search",
            tools=[Tool(name="agent_search", budget=5)],
        )
        def researcher(seed: RawText, topic: Annotated[str, FromInput]) -> Claims: ...

        gen = researcher | Oracle(n=2, merge_fn="echo_merge")
        pipeline = Construct("vd-di", nodes=[
            Node.scripted("seed", fn="seed_raw_vd3", outputs=RawText),
            gen,
        ])
        result = run(self._compile(pipeline), input={"node_id": "vd-di", "topic": "BIRDS"})
        items = result["researcher"].items
        assert len(items) == 2, items
        # The upstream (seed) value reached each cycle.
        assert all("SEEDVAL" in seen for seen in items), items


class TestOracleOverAgentMultiInputBundle:
    """neograph-qzrv: Oracle over an agent with MULTIPLE distinct dict-form
    upstream producers is now supported via packer-port synthesis — a synthesized
    parent 'packer' node bundles the N upstreams into one port model, and inner
    per-key 'unpacker' nodes re-expose the ORIGINAL keys so the agent's prompt
    surface is unchanged. Both bundled values reach each isolated cycle."""

    def test_multiple_dict_form_producers_bundle_and_deliver(self):
        register_tool_factory(
            "agent_search", lambda config, tool_config: FakeTool("agent_search", response="found")
        )
        register_scripted("alpha_fn_vd", lambda input_data, config: _Alpha(a="AVAL"))
        register_scripted("beta_fn_vd", lambda input_data, config: _Beta(b="BVAL"))
        register_scripted("echo_merge", _echo_merge_concat)

        gen = (
            Node(
                name="agent-gen", mode="agent",
                inputs={"alpha": _Alpha, "beta": _Beta}, outputs=Claims,
                model="default-tier", prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Oracle(n=2, merge_fn="echo_merge")
        )
        pipeline = Construct("vd-multi", nodes=[
            Node.scripted("alpha", fn="alpha_fn_vd", outputs=_Alpha),
            Node.scripted("beta", fn="beta_fn_vd", outputs=_Beta),
            gen,
        ])
        graph = compile(
            pipeline,
            **build_test_compile_kwargs(),
            **configure_fake_llm(
                lambda tier: _EchoOracleReActFake(),
                _echo_oracle_prompt_compiler,
            ),
        )
        result = run(graph, input={"node_id": "vd-multi"})
        items = result["agent_gen"].items
        assert len(items) == 2, items  # merged-2 isolated variants
        # BOTH bundled upstream values reached each isolated cycle.
        assert all("AVAL" in seen and "BVAL" in seen for seen in items), items

    def test_multi_output_agent_stays_fail_loud(self):
        """Dict-form (multi-output) agent OUTPUTS stay fail-loud: the sub-construct
        has a single output boundary port, and an N-way merge of secondary outputs
        (e.g. tool_log) is undefined. Design call (neograph-qzrv): keep fail-loud
        rather than silently drop declared outputs."""
        from neograph import ToolInteraction

        gen = (
            Node(
                name="agent-gen", mode="agent", inputs=RawText,
                outputs={"result": Claims, "tool_log": list[ToolInteraction]},
                model="default-tier", prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Oracle(n=2, merge_fn="echo_merge")
        )
        with pytest.raises(ConstructError, match="multi-output"):
            Construct("bad-multi-output", nodes=[
                Node.scripted("seed", fn="seed_raw_vd", outputs=RawText),
                gen,
            ])
