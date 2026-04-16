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
from neograph.factory import register_scripted, register_tool_factory
from tests.fakes import FakeTool, ReActFake, StructuredFake, configure_fake_llm
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

        from neograph.factory import register_scripted

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
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # All 3 generators ran
        assert gen_call_count[0] == 3
        # Merge combined all 3 variants
        merged = result.get("generate")
        assert isinstance(merged, Claims)
        assert len(merged.items) == 3

    def test_llm_judge_merges_when_merge_prompt_set(self):
        """Oracle with merge_prompt calls LLM to judge-merge variants."""

        from neograph.factory import register_scripted

        register_scripted("gen_llm", lambda input_data, config: Claims(items=["v1"]))

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["merged-consensus"])))

        @node(outputs=Claims, ensemble_n=2, merge_prompt="test/merge")
        def generate() -> Claims:
            return Claims(items=["v1"])

        mod = self._fresh_module("test_oracle_llm")
        mod.generate = generate

        pipeline = construct_from_module(mod, name="test-oracle-llm")
        graph = compile(pipeline)
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
        from neograph.factory import register_scripted

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
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # 3 variants merged into one
        assert isinstance(result["enrich"], RawText)
        assert result["enrich"].text.count("processed") == 3

    def test_sub_pipeline_runs_n_times_when_oracle_with_llm_merge(self):
        """Sub-pipeline runs 2 times, LLM merge combines outputs."""
        from neograph.factory import register_scripted

        register_scripted("gen_claim", lambda input_data, config: Claims(items=["variant"]))

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["llm-merged"])))

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
        graph = compile(parent)
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
        graph = compile(parent)
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
        graph = compile(parent)
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
        from neograph.factory import register_scripted

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
        graph = compile(pipeline)
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
        from neograph.factory import register_scripted

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
        graph = compile(pipeline)
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
        from neograph.factory import register_scripted

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
        graph = compile(pipeline)
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
        from neograph.factory import register_scripted

        seen_tiers = []

        def tier_capturing_factory(tier):
            seen_tiers.append(tier)
            return StructuredFake(lambda m: m(items=[f"from-{tier}"]))

        configure_fake_llm(tier_capturing_factory)

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
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "think-oracle-models"})

        # Filter to only the generator tiers (exclude merge LLM calls)
        gen_tiers = [t for t in seen_tiers if t in ("reason", "fast", "creative")]
        assert len(gen_tiers) == 3, f"Expected 3 generator calls with model overrides, got tiers: {seen_tiers}"
        assert set(gen_tiers) == {"reason", "fast", "creative"}

    def test_oracle_models_on_agent_mode_node(self):
        """Oracle(models=) must override model tier for agent-mode (tool) nodes.

        Bug (pre-neograph-y8ww): the tool wrapper read _oracle_model from
        config but never transferred neo_oracle_model from state to config.
        Now all modes share _execute_node preamble which calls
        _inject_oracle_config. Regression test for neograph-lbsf.
        """
        from neograph.factory import register_scripted

        seen_tiers = []
        fake_tool = FakeTool("agent_search", response="found")
        register_tool_factory("agent_search", lambda config, tool_config: fake_tool)

        def tier_capturing_factory(tier):
            seen_tiers.append(tier)
            return ReActFake(
                tool_calls=[
                    [{"name": "agent_search", "args": {}, "id": "c1"}],
                    [],  # stop
                ],
                final=lambda m: m(items=[f"from-{tier}"]),
            )

        configure_fake_llm(tier_capturing_factory)

        def agent_merge(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("agent_models_merge", agent_merge)

        gen_node = (
            Node(
                name="agent-gen",
                mode="agent",
                outputs=Claims,
                model="default-tier",
                prompt="test/search",
                tools=[Tool(name="agent_search", budget=5)],
            )
            | Oracle(models=["reason", "fast"], merge_fn="agent_models_merge")
        )
        pipeline = Construct("test-agent-oracle-models", nodes=[gen_node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "agent-oracle-models"})

        # Filter to only the generator tiers (exclude merge/final-parse calls)
        gen_tiers = [t for t in seen_tiers if t in ("reason", "fast")]
        assert len(gen_tiers) == 2, f"Expected 2 generator calls with model overrides, got tiers: {seen_tiers}"
        assert set(gen_tiers) == {"reason", "fast"}

    def test_oracle_models_round_robin_on_think_mode(self):
        """Round-robin model assignment works on think-mode nodes when n > len(models)."""
        from neograph.factory import register_scripted

        seen_tiers = []

        def tier_capturing_factory(tier):
            seen_tiers.append(tier)
            return StructuredFake(lambda m: m(items=[f"from-{tier}"]))

        configure_fake_llm(tier_capturing_factory)

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
        graph = compile(pipeline)
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

        configure_fake_llm(tier_capturing_factory)

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
        graph = compile(pipeline)
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
        from neograph.factory import register_scripted

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
        graph = compile(pipeline)
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
        from neograph.factory import register_scripted

        register_scripted("bglm_gen", lambda input_data, config: Claims(items=["v1"]))

        def bad_merge(variants, config):
            raise AttributeError("ModelRole.FAST doesn't exist")

        register_scripted("bglm_bad_merge", bad_merge)

        gen_node = (
            Node.scripted("gen", fn="bglm_gen", outputs=Claims)
            | Oracle(n=2, merge_fn="bglm_bad_merge")
        )
        pipeline = Construct("test-merge-error", nodes=[gen_node])
        graph = compile(pipeline)

        # The pipeline MUST fail — not silently produce garbage
        with pytest.raises(Exception, match="ModelRole.FAST"):
            run(graph, input={"node_id": "bglm-test"})

    def test_wrong_return_type_raises_when_merge_fn_returns_bad_type(self):
        """merge_fn that returns the wrong type should be caught."""
        from neograph.factory import register_scripted

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
        graph = compile(pipeline)

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

        from neograph.factory import register_scripted

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
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "jg2g-test"})

        assert result["gen"].topic == "AI safety"

    def test_merge_fn_mixes_state_and_di_params(self):
        """@merge_fn can have both state params and DI params."""
        from pydantic import BaseModel

        from neograph import FromInput
        from neograph.factory import register_scripted

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
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "mixed-test"})

        assert result["gen2"].label == "api:mixed-test"

    def test_merge_fn_state_param_rejects_unknown_field(self):
        """State param for a field not in state raises ConstructError."""
        from pydantic import BaseModel

        from neograph.factory import register_scripted

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

        from neograph.factory import register_scripted
        from neograph.modifiers import Loop

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
        graph = compile(pipeline)
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

        configure_fake_llm(lambda tier: CaptureMerge(tier))

        @node(outputs=UpstreamContext)
        def enrich() -> UpstreamContext:
            return UpstreamContext(site_name="Acme Corp", tone="professional")

        @node(outputs=Draft, ensemble_n=2,
              merge_prompt="Pick the best considering ${enrich.site_name}: ${variants}")
        def write(enrich: UpstreamContext) -> Draft: ...

        mod = self._fresh_module("test_merge_ctx")
        mod.enrich = enrich
        mod.write = write

        from neograph import construct_from_module
        pipeline = construct_from_module(mod, name="merge-ctx")
        graph = compile(pipeline)
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

        configure_fake_llm(lambda tier: InspectMerge(tier))

        @node(outputs=Draft, ensemble_n=3, merge_prompt="Judge: ${variants}",
              prompt="generate a draft", model="fast")
        def generate() -> Draft: ...

        mod = self._fresh_module("test_variants")
        mod.generate = generate

        from neograph import construct_from_module
        pipeline = construct_from_module(mod, name="variants-test")
        graph = compile(pipeline)
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
        graph = compile(pipeline)
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

        configure_fake_llm(lambda tier: CaptureMerge3(tier))

        register_scripted("_mp_seed", lambda i, c: UpstreamContext(site_name="Test Site", tone="casual"))

        seed = Node.scripted("seed", fn="_mp_seed", outputs=UpstreamContext)
        writer = Node(
            "writer", mode="think", outputs=Draft,
            inputs={"seed": UpstreamContext},
            prompt="write", model="fast",
        ) | Oracle(n=2, merge_prompt="Pick best: ${seed.site_name} ${variants}")

        pipeline = Construct("prog-merge", nodes=[seed, writer])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "t4"})

        prompt = captured_input.get("prompt", "")
        assert "Test Site" in prompt, (
            f"Upstream context 'Test Site' missing from programmatic merge prompt: {prompt}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# EACH×ORACLE FUSION (neograph-tpgi)
#
# map_over + ensemble_n on the same @node: flat M×N Send topology.
# ═══════════════════════════════════════════════════════════════════════════


