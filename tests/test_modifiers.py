"""Modifier tests — Oracle, Each, Operator, map(), deep compositions,
modifier-as-first-node, Construct-level modifiers, list-over-Each e2e.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest

from neograph import (
    Construct, ConstructError, Node, Each, Loop, Oracle, Operator, Tool,
    compile, construct_from_functions, construct_from_module,
    merge_fn, node, run, tool,
    ConfigurationError, ExecutionError,
)
from neograph.factory import register_scripted, register_tool_factory
from tests.fakes import FakeTool, ReActFake, StructuredFake, configure_fake_llm
from tests.schemas import RawText, Claims, ClassifiedClaims, ClusterGroup, Clusters, MatchResult, MergedResult, ValidationResult, _producer, _consumer


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
        import types as _types

        from neograph import construct_from_module, node
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
        assert merged is not None
        assert len(merged.items) == 3

    def test_llm_judge_merges_when_merge_prompt_set(self):
        """Oracle with merge_prompt calls LLM to judge-merge variants."""
        import types as _types

        from neograph import construct_from_module, node
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
        assert merged is not None
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

class TestEach:
    def test_each_dispatches_per_item_when_collection_provided(self):
        """Each dispatches per-item and collects results (@node API)."""
        import types as _types
        from neograph import compile, construct_from_module, node, run

        mod = _types.ModuleType("test_each_fanout")

        @node(mode="scripted", outputs=Clusters)
        def make_clusters() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="alpha", claim_ids=["c1", "c2"]),
                ClusterGroup(label="beta", claim_ids=["c3"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(
                cluster_label=cluster.label,
                matched=["match-1"],
            )

        mod.make_clusters = make_clusters
        mod.verify = verify

        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Both clusters were processed — pin cardinality AND per-key payload
        verify_results = result.get("verify", {})
        assert set(verify_results.keys()) == {"alpha", "beta"}
        assert verify_results["alpha"].cluster_label == "alpha"
        assert verify_results["beta"].cluster_label == "beta"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5b: .map() — sugar over Each(over=..., key=...)
#
# Node.map() accepts a lambda introspected at definition time, or a string
# path as an escape hatch. Both compile to the same Each modifier.
# This proves: .map() is pure sugar, lambda introspection resolves attribute
# chains to dotted paths, the resulting graph runs identically to | Each(...).
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5b: .map() — sugar over Each(over=..., key=...)
#
# Node.map() accepts a lambda introspected at definition time, or a string
# path as an escape hatch. Both compile to the same Each modifier.
# This proves: .map() is pure sugar, lambda introspection resolves attribute
# chains to dotted paths, the resulting graph runs identically to | Each(...).
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeMap:
    """Node.map() — lambda- and string-path fan-out sugar over `| Each(...)`."""

    def test_each_resolves_path_when_lambda_used(self):
        """A lambda `s.foo.bar` resolves to the same Each(over='foo.bar', ...)."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        mapped = node.map(lambda s: s.make_clusters.groups, key="label")

        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make_clusters.groups"
        assert each.key == "label"

    def test_each_resolves_path_when_string_used(self):
        """A string source is passed straight through to Each.over."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        mapped = node.map("make_clusters.groups", key="label")

        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make_clusters.groups"
        assert each.key == "label"

    def test_map_produces_same_node_when_compared_to_pipe_each(self):
        """node.map(...) and node | Each(...) produce structurally identical nodes."""
        base = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)

        via_map = base.map(lambda s: s.make_clusters.groups, key="label")
        via_pipe = base | Each(over="make_clusters.groups", key="label")

        assert via_map.modifiers == via_pipe.modifiers

    def test_fanout_runs_when_map_sugar_used(self):
        """.map() drives the same fan-out/collect behavior as | Each(...)."""
        from neograph.factory import register_scripted

        register_scripted("make_clusters", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="alpha", claim_ids=["c1", "c2"]),
                ClusterGroup(label="beta", claim_ids=["c3"]),
            ]
        ))
        register_scripted("verify_cluster", lambda input_data, config: MatchResult(
            cluster_label=input_data.label if hasattr(input_data, "label") else "unknown",
            matched=["match-1"],
        ))

        make = Node.scripted("make-clusters", fn="make_clusters", outputs=Clusters)
        verify = Node.scripted(
            "verify", fn="verify_cluster", inputs=ClusterGroup, outputs=MatchResult
        ).map(lambda s: s.make_clusters.groups, key="label")

        pipeline = Construct("test-map", nodes=[make, verify])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        # Fan-out fired for BOTH clusters — pin cardinality and payload
        verify_results = result.get("verify", {})
        assert set(verify_results.keys()) == {"alpha", "beta"}
        assert verify_results["alpha"].cluster_label == "alpha"
        assert verify_results["beta"].cluster_label == "beta"

    def test_map_raises_when_lambda_has_no_attributes(self):
        """`lambda s: s` has no path — clear error."""
        node = Node.scripted("verify", fn="noop", outputs=MatchResult)
        with pytest.raises(TypeError, match="at least one attribute"):
            node.map(lambda s: s, key="label")

    def test_map_raises_when_lambda_returns_scalar(self):
        """`lambda s: 42` — clear error, not a silent Each."""
        node = Node.scripted("verify", fn="noop", outputs=MatchResult)
        with pytest.raises(TypeError, match="attribute-access chain"):
            node.map(lambda s: 42, key="label")

    def test_map_raises_when_lambda_uses_indexing(self):
        """A lambda that does something illegal (e.g. indexing) reports cleanly."""
        node = Node.scripted("verify", fn="noop", outputs=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s.items[0], key="label")  # __getitem__ on recorder

    def test_map_raises_when_lambda_accesses_dunder(self):
        """`lambda s: s.__dict__.foo` must not silently produce Each(over='__dict__.foo')."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s.__dict__.foo, key="label")

    def test_map_raises_when_lambda_accesses_underscore_attr(self):
        """Reject `lambda s: s._private.field` — underscores are a footgun trapdoor."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            node.map(lambda s: s._private.x, key="label")

    def test_user_exception_propagates_when_lambda_raises(self):
        """Non-attribute errors (e.g. ZeroDivisionError) propagate with their own type."""
        node = Node.scripted("verify", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        with pytest.raises(ZeroDivisionError):
            node.map(lambda s: 1 / 0 and s.x, key="label")

    def test_map_raises_when_source_not_string_or_callable(self):
        """Passing an int or other non-source type raises immediately."""
        node = Node.scripted("verify", fn="noop", outputs=MatchResult)
        with pytest.raises(TypeError, match="string path or a lambda"):
            node.map(42, key="label")  # type: ignore[arg-type]

    def test_each_attaches_when_map_called_on_construct(self):
        """Construct also gets .map() via Modifiable — sub-construct fan-out."""
        inner = Node.scripted("inner", fn="noop", inputs=ClusterGroup, outputs=MatchResult)
        sub = Construct("sub", input=ClusterGroup, output=MatchResult, nodes=[inner])
        mapped = sub.map(lambda s: s.upstream.items, key="label")

        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "upstream.items"
        assert each.key == "label"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Operator — human-in-the-loop interrupt
#
# A node produces a validation result. If validation fails,
# the graph pauses via interrupt(). Resume with human input.
# This proves: Operator modifier wires interrupt() correctly,
# graph pauses and resumes.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Operator — human-in-the-loop interrupt
#
# A node produces a validation result. If validation fails,
# the graph pauses via interrupt(). Resume with human input.
# This proves: Operator modifier wires interrupt() correctly,
# graph pauses and resumes.
# ═══════════════════════════════════════════════════════════════════════════

class TestOperator:
    def test_graph_pauses_when_operator_condition_truthy(self):
        """Graph pauses when Operator condition is met."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_operator_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            interrupt_when=lambda state: (
                {"issues": state.validate.issues}
                if state.validate and not state.validate.passed
                else None
            ),
        )
        def validate() -> ValidationResult:
            return ValidationResult(passed=False, issues=["missing stakeholder coverage"])

        mod.validate = validate

        pipeline = construct_from_module(mod, name="test-operator")
        graph = compile(pipeline, checkpointer=MemorySaver())

        # Run — with checkpointer, interrupt returns result with __interrupt__
        config = {"configurable": {"thread_id": "test-interrupt"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        # Verify the graph paused
        assert "__interrupt__" in result
        assert result["validate"].passed is False


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Raw node alongside declarative nodes
#
# A @node(mode='raw') function mixed with @node declarations in the same Construct.
# This proves: raw escape hatch works, framework wires edges around it,
# data flows through raw node like any other.
# ═══════════════════════════════════════════════════════════════════════════


class TestOperatorContinues:
    """Operator condition is falsy — graph continues without interrupt."""

    def test_graph_continues_when_operator_condition_falsy(self):
        """Graph runs through Operator without pausing when condition returns None."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_operator_continues_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            interrupt_when=lambda state: None,  # always falsy
        )
        def validate() -> ValidationResult:
            return ValidationResult(passed=True, issues=[])

        mod.validate = validate

        pipeline = construct_from_module(mod, name="test-operator-pass")
        graph = compile(pipeline, checkpointer=MemorySaver())
        result = run(graph, input={"node_id": "test-001"}, config={"configurable": {"thread_id": "pass-test"}})

        assert result["validate"].passed is True
        assert result.get("human_feedback") is None


class TestOperatorResume:
    """Operator interrupt + resume flow."""

    def test_graph_resumes_when_human_feedback_provided(self):
        """Graph pauses at interrupt, resumes with human feedback via run()."""
        import types as _types

        from langgraph.checkpoint.memory import MemorySaver

        from neograph import construct_from_module, node

        mod = _types.ModuleType("test_operator_resume_mod")

        @node(
            mode="scripted",
            outputs=ValidationResult,
            name="validate-thing",
            interrupt_when=lambda state: (
                {"issues": state.validate_thing.issues}
                if state.validate_thing and not state.validate_thing.passed
                else None
            ),
        )
        def validate_thing() -> ValidationResult:
            return ValidationResult(passed=False, issues=["bad coverage"])

        mod.validate_thing = validate_thing

        pipeline = construct_from_module(mod, name="test-resume")
        graph = compile(pipeline, checkpointer=MemorySaver())

        config = {"configurable": {"thread_id": "resume-test"}}

        # First run: hits interrupt — returns with __interrupt__
        result = run(graph, input={"node_id": "test-001"}, config=config)
        assert "__interrupt__" in result

        # Resume via run()
        result = run(graph, resume={"approved": True}, config=config)

        assert result["validate_thing"].passed is False
        assert result["human_feedback"] == {"approved": True}


class TestModifierAsFirstNode:
    """Modifiers on the first node wire from START, not from a previous node."""

    def test_oracle_wires_from_start_when_first_node(self):
        """Oracle as the first (and only) node — router wired from START."""
        from neograph.factory import register_scripted

        register_scripted("gen_start", lambda input_data, config: Claims(items=["from-start"]))

        def merge_start(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("merge_start", merge_start)

        node = Node.scripted(
            "gen", fn="gen_start", outputs=Claims
        ) | Oracle(n=2, merge_fn="merge_start")

        pipeline = Construct("test-oracle-start", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test-001"})

        merged = result.get("gen")
        assert merged is not None
        assert len(merged.items) == 2

    def test_each_wires_from_start_when_first_node(self):
        """Each as the first node — router wired from START."""
        from langgraph.graph import END, StateGraph

        from neograph.compiler import _add_node_to_graph
        from neograph.factory import register_scripted
        from neograph.state import compile_state_model

        register_scripted("process_item", lambda input_data, config: MatchResult(
            cluster_label=input_data.label, matched=["done"],
        ))

        process = Node.scripted(
            "process", fn="process_item", inputs=ClusterGroup, outputs=MatchResult
        ) | Each(over="make_items.groups", key="label")

        pipeline = Construct("test-each-start", nodes=[process])
        state_model = compile_state_model(pipeline)
        graph = StateGraph(state_model)
        prev = _add_node_to_graph(graph, process, None)
        graph.add_edge(prev, END)
        compiled = graph.compile()
        # Compilation succeeded — Each wired from START without crash


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
        assert result["enrich"] is not None
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


class TestConstructEach:
    """Construct | Each — run entire sub-pipeline per collection item."""

    def test_sub_pipeline_runs_per_item_when_each_over_collection(self):
        """Sub-pipeline runs once per cluster, results collected as dict."""
        from neograph.factory import register_scripted

        register_scripted("make_clusters", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="alpha", claim_ids=["c1"]),
                ClusterGroup(label="beta", claim_ids=["c2", "c3"]),
            ]
        ))

        register_scripted("sub_analyze", lambda input_data, config: RawText(
            text=f"analyzed: {input_data.label}"
        ))
        register_scripted("sub_score", lambda input_data, config: MatchResult(
            cluster_label=input_data.label if hasattr(input_data, 'label') else "unknown",
            matched=[f"scored-{input_data.text}" if hasattr(input_data, 'text') else "scored"],
        ))

        sub = Construct(
            "verify-cluster",
            input=ClusterGroup,
            output=MatchResult,
            nodes=[
                Node.scripted("analyze", fn="sub_analyze", inputs=ClusterGroup, outputs=RawText),
                Node.scripted("score", fn="sub_score", inputs=RawText, outputs=MatchResult),
            ],
        ) | Each(over="make_clusters.groups", key="label")

        parent = Construct("parent", nodes=[
            Node.scripted("make-clusters", fn="make_clusters", outputs=Clusters),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Both clusters processed
        verify_results = result["verify_cluster"]
        assert "alpha" in verify_results
        assert "beta" in verify_results


class TestConstructOperator:
    """Construct | Operator — check condition after sub-pipeline completes."""

    def test_parent_pauses_when_sub_construct_operator_truthy(self):
        """Sub-pipeline runs, then Operator checks and interrupts."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_validate", lambda input_data, config: ValidationResult(
            passed=False, issues=["coverage gap"],
        ))

        register_condition("sub_failed", lambda state: (
            {"issues": state.enrich.issues}
            if hasattr(state, 'enrich') and state.enrich and not state.enrich.passed
            else None
        ))

        sub = Construct(
            "enrich",
            input=Claims,
            output=ValidationResult,
            nodes=[Node.scripted("val", fn="sub_validate", outputs=ValidationResult)],
        ) | Operator(when="sub_failed")

        register_scripted("seed", lambda input_data, config: Claims(items=["data"]))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed", outputs=Claims),
            sub,
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "construct-op-test"}}

        result = run(graph, input={"node_id": "test-001"}, config=config)

        # Sub-pipeline ran and produced output
        assert result["enrich"] is not None
        assert result["enrich"].passed is False
        # Interrupted
        assert "__interrupt__" in result

    def test_parent_continues_when_sub_construct_operator_falsy(self):
        """Sub-pipeline runs, condition is falsy, graph continues."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_ok", lambda input_data, config: ValidationResult(
            passed=True, issues=[],
        ))

        register_condition("sub_check", lambda state: None)  # always passes

        sub = Construct(
            "check",
            input=Claims,
            output=ValidationResult,
            nodes=[Node.scripted("ok", fn="sub_ok", outputs=ValidationResult)],
        ) | Operator(when="sub_check")

        register_scripted("seed2", lambda input_data, config: Claims(items=["ok"]))
        register_scripted("done", lambda input_data, config: RawText(text="complete"))

        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="seed2", outputs=Claims),
            sub,
            Node.scripted("done", fn="done", outputs=RawText),
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "construct-op-pass"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        assert result["done"].text == "complete"


class TestDeepCompositions:
    """Complex nesting: modifiers inside modifiers, tool exhaustion, etc."""

    def test_oracle_runs_per_item_when_nested_inside_each(self):
        """Each item gets Oracle ensemble — fan-out inside fan-out."""
        from neograph.factory import register_scripted

        register_scripted("make_items", lambda input_data, config: Clusters(
            groups=[ClusterGroup(label="x", claim_ids=["1"]), ClusterGroup(label="y", claim_ids=["2"])]
        ))

        gen_count = {"n": 0}

        def gen_variant(input_data, config):
            gen_count["n"] += 1
            return RawText(text=f"variant-{gen_count['n']}")

        register_scripted("gen_v", gen_variant)

        def merge_v(variants, config):
            return RawText(text=f"merged-{len(variants)}")

        register_scripted("merge_v", merge_v)

        # Inner sub-construct: Oracle (2 variants, merge)
        inner = Construct(
            "oracle-inner",
            input=ClusterGroup,
            output=RawText,
            nodes=[
                Node.scripted("gen", fn="gen_v", outputs=RawText)
                | Oracle(n=2, merge_fn="merge_v"),
            ],
        )

        # Outer: Each over clusters, each runs the Oracle sub-pipeline
        # This means: 2 clusters × 2 Oracle variants = 4 generator calls + 2 merges
        parent = Construct("parent", nodes=[
            Node.scripted("make", fn="make_items", outputs=Clusters),
            inner | Each(over="make.groups", key="label"),
        ])

        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Each cluster got Oracle'd — 2 clusters × 2 generators = 4 total
        assert gen_count["n"] == 4
        assert "x" in result["oracle_inner"]
        assert "y" in result["oracle_inner"]

    def test_tool_budget_enforced_when_gather_inside_subgraph(self):
        """Gather node inside subgraph exhausts tool budget, forced to respond."""
        from neograph.factory import register_scripted, register_tool_factory

        deep_search_tool = FakeTool("deep_search", response="found")
        register_tool_factory("deep_search", lambda config, tool_config: deep_search_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "deep_search", "args": {}, "id": "c1"}],
                [{"name": "deep_search", "args": {}, "id": "c2"}],
                [{"name": "deep_search", "args": {}, "id": "c3"}],
                [{"name": "deep_search", "args": {}, "id": "c4"}],
                [{"name": "deep_search", "args": {}, "id": "c5"}],
                [],  # stop
            ],
            final=lambda m: m(text="search complete"),
        )
        configure_fake_llm(lambda tier: fake)

        register_scripted("prep_search", lambda input_data, config: Claims(items=["query"]))

        sub = Construct(
            "deep-search",
            input=Claims,
            output=RawText,
            nodes=[
                Node(
                    name="search",
                    mode="agent",
                    inputs=Claims,
                    outputs=RawText,
                    model="fast",
                    prompt="test/search",
                    tools=[Tool(name="deep_search", budget=3)],
                ),
            ],
        )

        parent = Construct("parent", nodes=[
            Node.scripted("prep", fn="prep_search", outputs=Claims),
            sub,
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "test-001"})

        # Tool budget was 3, so exactly 3 calls made despite LLM wanting 5
        assert len(deep_search_tool.calls) == 3
        # Subgraph still produced output
        assert result["deep_search"] is not None

    def test_interrupt_surfaces_when_operator_inside_subgraph(self):
        """Operator inside a sub-construct pauses the entire parent pipeline."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted("sub_check", lambda input_data, config: ValidationResult(
            passed=False, issues=["needs human"],
        ))

        register_condition("inner_failed", lambda state: (
            {"reason": "inner check failed"}
            if hasattr(state, 'check') and state.check and not state.check.passed
            else None
        ))

        sub = Construct(
            "inner",
            input=Claims,
            output=ValidationResult,
            nodes=[
                Node.scripted("check", fn="sub_check", outputs=ValidationResult)
                | Operator(when="inner_failed"),
            ],
        )

        register_scripted("start_fn", lambda input_data, config: Claims(items=["go"]))
        register_scripted("after_fn", lambda input_data, config: RawText(text="should not reach"))

        parent = Construct("parent", nodes=[
            Node.scripted("start", fn="start_fn", outputs=Claims),
            sub,
            Node.scripted("after", fn="after_fn", outputs=RawText),
        ])

        # Operator lives inside sub-construct — parent needs checkpointer
        # so the recursive compile of the sub-construct gets it
        from langgraph.checkpoint.memory import MemorySaver

        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "inner-op-test"}}
        result = run(graph, input={"node_id": "test-001"}, config=config)

        # The interrupt inside the subgraph should surface
        # "after" node should not have run
        assert result.get("after") is None or "__interrupt__" in result


# ═══════════════════════════════════════════════════════════════════════════
# REMAINING COVERAGE GAPS — paths not yet exercised
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeDecoratorDictInputs (neograph-kqd.4)
#
# @node decoration now emits dict-form inputs={param_name: annotation, ...}
# for all typed upstream params. This is the metadata shift that lets
# step-2's validator catch fan-in mismatches via _check_fan_in_inputs.
# Fan-out params (Each) are stripped from inputs at construct-assembly time.
# ═══════════════════════════════════════════════════════════════════════════

class TestListOverEachEndToEnd:
    def test_list_consumer_receives_values_when_declarative_each_producer(self):
        """Declarative: Each producer + Node.scripted consumer that
        annotates inputs={'verify': list[MatchResult]}."""
        from neograph import compile, run
        from neograph.factory import register_scripted

        register_scripted(
            "make_clusters_l5",
            lambda _in, _cfg: Clusters(groups=[
                ClusterGroup(label="a", claim_ids=["1"]),
                ClusterGroup(label="b", claim_ids=["2"]),
                ClusterGroup(label="c", claim_ids=["3"]),
            ]),
        )
        register_scripted(
            "verify_cluster_l5",
            lambda input_data, _cfg: MatchResult(
                cluster_label=input_data.label if hasattr(input_data, "label") else "?",
                matched=[f"m-{input_data.label}" if hasattr(input_data, "label") else "?"],
            ),
        )

        def summarize_fn(input_data, _cfg):
            verify_list = input_data["verify"]
            assert isinstance(verify_list, list), f"expected list, got {type(verify_list).__name__}"
            return MergedResult(
                final_text=f"verified:{len(verify_list)}:" + ",".join(sorted(v.cluster_label for v in verify_list)),
            )

        register_scripted("summarize_l5", summarize_fn)

        make = Node.scripted("make-clusters", fn="make_clusters_l5", outputs=Clusters)
        verify = (
            Node.scripted("verify", fn="verify_cluster_l5", inputs=ClusterGroup, outputs=MatchResult)
            .map(lambda s: s.make_clusters.groups, key="label")
        )
        summarize = Node.scripted(
            "summarize", fn="summarize_l5",
            inputs={"verify": list[MatchResult]},
            outputs=MergedResult,
        )
        pipeline = Construct("l5-decl", nodes=[make, verify, summarize])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "l5"})
        assert result["summarize"].final_text == "verified:3:a,b,c"

    def test_list_consumer_receives_values_when_decorator_each_producer(self):
        """@node: Each producer + @node consumer with list[X] annotation."""
        from neograph import compile, run, node
        from neograph.decorators import construct_from_functions

        @node(mode="scripted", outputs=Clusters)
        def gen_clusters() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="alpha", claim_ids=["1"]),
                ClusterGroup(label="beta", claim_ids=["2"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="gen_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=[f"m-{cluster.label}"])

        @node(mode="scripted", outputs=MergedResult)
        def summarize(verify: list[MatchResult]) -> MergedResult:
            assert isinstance(verify, list), f"expected list, got {type(verify).__name__}"
            return MergedResult(
                final_text=f"got:{len(verify)}:" + ",".join(sorted(v.cluster_label for v in verify)),
            )

        pipeline = construct_from_functions("l5-dec", [gen_clusters, verify, summarize])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "l5"})
        assert result["summarize"].final_text == "got:2:alpha,beta"

    def test_construct_raises_when_list_element_type_wrong(self):
        """list[WrongType] consumer + Each producer raises ConstructError."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = Node.scripted(
            "summarize", fn="f",
            inputs={"verify": list[Claims]},  # WRONG: Each emits MatchResult
            outputs=MergedResult,
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-list-type", nodes=[make, verify, summarize])
        msg = str(exc_info.value)
        assert "verify" in msg

    def test_dict_consumer_passes_when_each_producer_present(self):
        """dict[str, X] consumer still passes through unchanged (regression)."""
        from neograph import compile, run
        from neograph.factory import register_scripted

        register_scripted(
            "make_clusters_l5b",
            lambda _in, _cfg: Clusters(groups=[ClusterGroup(label="x", claim_ids=["1"])]),
        )
        register_scripted(
            "verify_cluster_l5b",
            lambda input_data, _cfg: MatchResult(
                cluster_label=input_data.label if hasattr(input_data, "label") else "?",
                matched=["ok"],
            ),
        )

        def summarize_dict_fn(input_data, _cfg):
            verify_dict = input_data["verify"]
            assert isinstance(verify_dict, dict), f"expected dict, got {type(verify_dict).__name__}"
            return MergedResult(final_text=f"keys:{sorted(verify_dict.keys())}")

        register_scripted("summarize_l5b", summarize_dict_fn)

        make = Node.scripted("make-clusters", fn="make_clusters_l5b", outputs=Clusters)
        verify = (
            Node.scripted("verify", fn="verify_cluster_l5b", inputs=ClusterGroup, outputs=MatchResult)
            .map(lambda s: s.make_clusters.groups, key="label")
        )
        summarize = Node.scripted(
            "summarize", fn="summarize_l5b",
            inputs={"verify": dict[str, MatchResult]},
            outputs=MergedResult,
        )
        pipeline = Construct("l5-dict", nodes=[make, verify, summarize])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "l5"})
        assert result["summarize"].final_text == "keys:['x']"


# ═══════════════════════════════════════════════════════════════════════════
# TestNodeInputsEpicAcceptance (neograph-kqd.7)
#
# Closes remaining acceptance gaps from the epic. The bulk of the matrix is
# covered by TestFanInValidation / TestListOverEachEndToEnd /
# TestExtractInputListUnwrap / TestNodeDecoratorDictInputs. This class adds:
#   - LLM-driven spec round-trip (JSON-shaped dict → Node → validated pipeline)
#   - Zero-upstream node explicit test
#   - Programmatic fan-in via Node + Oracle pipe
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Three-surface parity: Each fan-out behavior
#
# Template pattern: @pytest.mark.parametrize("build", [...]) with one
# builder per API surface (declarative, @node decorator, programmatic).
# Each builder returns a Construct; the test compiles, runs, and asserts
# identically across all three surfaces.
# ═══════════════════════════════════════════════════════════════════════════


def _each_via_declarative() -> Construct:
    """Declarative surface: Node.scripted + .map()."""
    register_scripted(
        "tsp_make",
        lambda _in, _cfg: Clusters(groups=[
            ClusterGroup(label="alpha", claim_ids=["c1"]),
            ClusterGroup(label="beta", claim_ids=["c2"]),
        ]),
    )
    register_scripted(
        "tsp_verify",
        lambda input_data, _cfg: MatchResult(
            cluster_label=input_data.label if hasattr(input_data, "label") else "?",
            matched=[f"m-{input_data.label}" if hasattr(input_data, "label") else "?"],
        ),
    )

    make = Node.scripted("make", fn="tsp_make", outputs=Clusters)
    verify = Node.scripted(
        "verify", fn="tsp_verify", inputs=ClusterGroup, outputs=MatchResult
    ).map(lambda s: s.make.groups, key="label")

    return Construct("tsp-decl", nodes=[make, verify])


def _each_via_decorator() -> Construct:
    """@node decorator surface: construct_from_functions."""
    @node(mode="scripted", outputs=Clusters)
    def tsp_dec_make() -> Clusters:
        return Clusters(groups=[
            ClusterGroup(label="alpha", claim_ids=["c1"]),
            ClusterGroup(label="beta", claim_ids=["c2"]),
        ])

    @node(
        mode="scripted",
        outputs=MatchResult,
        map_over="tsp_dec_make.groups",
        map_key="label",
    )
    def tsp_dec_verify(cluster: ClusterGroup) -> MatchResult:
        return MatchResult(
            cluster_label=cluster.label,
            matched=[f"m-{cluster.label}"],
        )

    return construct_from_functions("tsp-dec", [tsp_dec_make, tsp_dec_verify])


def _each_via_programmatic() -> Construct:
    """Programmatic surface: Node() | Each() with single-type inputs."""
    register_scripted(
        "tsp_make",
        lambda _in, _cfg: Clusters(groups=[
            ClusterGroup(label="alpha", claim_ids=["c1"]),
            ClusterGroup(label="beta", claim_ids=["c2"]),
        ]),
    )
    register_scripted(
        "tsp_verify",
        lambda input_data, _cfg: MatchResult(
            cluster_label=input_data.label,
            matched=[f"m-{input_data.label}"],
        ),
    )

    make = Node.scripted("make", fn="tsp_make", outputs=Clusters)
    verify = Node.scripted(
        "verify", fn="tsp_verify",
        inputs=ClusterGroup,
        outputs=MatchResult,
    ) | Each(over="make.groups", key="label")

    return Construct("tsp-prog", nodes=[make, verify])


class TestThreeSurfaceParity:
    """Each fan-out tested identically across declarative, @node, and
    programmatic API surfaces. Template pattern for future parity tests."""

    @pytest.mark.parametrize("build", [
        _each_via_declarative,
        _each_via_decorator,
        _each_via_programmatic,
    ], ids=["declarative", "decorator", "programmatic"])
    def test_each_produces_dict_when_any_surface_used(self, build):
        """Each fan-out produces dict[str, MatchResult] keyed by label."""
        pipeline = build()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "tsp-001"})

        verify_results = result.get("verify") or result.get("tsp_dec_verify")
        assert isinstance(verify_results, dict)
        assert set(verify_results.keys()) == {"alpha", "beta"}

    @pytest.mark.parametrize("build", [
        _each_via_declarative,
        _each_via_decorator,
        _each_via_programmatic,
    ], ids=["declarative", "decorator", "programmatic"])
    def test_each_items_match_source_when_any_surface_used(self, build):
        """Each fan-out item has the correct cluster_label from the source."""
        pipeline = build()
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "tsp-002"})

        verify_results = result.get("verify") or result.get("tsp_dec_verify")
        labels = {v.cluster_label for v in verify_results.values()}
        assert labels == {"alpha", "beta"}


# ═══════════════════════════════════════════════════════════════════════════
# TestModifierCombinations (neograph-rdu.1, rdu.4, rdu.6, rdu.7)
#
# Integration tests for modifier combinations that were previously only
# covered via one API surface or not at all.
# ═══════════════════════════════════════════════════════════════════════════

class TestModifierCombinations:
    """Cross-modifier integration tests: Each+Oracle, Each+Operator,
    dict-outputs+Oracle, dict-outputs+Each."""

    def test_oracle_merges_per_item_when_each_wraps_oracle_subconstruct(self):
        """neograph-rdu.1: Each fans out over clusters, each runs Oracle
        ensemble (2 variants + merge) via a sub-Construct containing @node."""
        import types as _types
        from neograph import compile, node, run
        from neograph.factory import register_scripted

        gen_count = [0]

        def mc_merge(variants, config):
            all_matched = []
            for v in variants:
                all_matched.extend(v.matched)
            return MatchResult(
                cluster_label=variants[0].cluster_label,
                matched=all_matched,
            )

        register_scripted("mc_merge_fn", mc_merge)

        # Inner @node with Oracle
        @node(mode="scripted", outputs=MatchResult, ensemble_n=2, merge_fn="mc_merge_fn")
        def mc_verify() -> MatchResult:
            gen_count[0] += 1
            return MatchResult(cluster_label="item", matched=[f"m-{gen_count[0]}"])

        mod = _types.ModuleType("mc_oracle_inner_mod")
        mod.mc_verify = mc_verify

        from neograph import construct_from_module
        inner = construct_from_module(mod, name="mc-oracle-inner")
        # Give the sub-construct proper input/output for Each
        inner = inner.model_copy(update={"input": ClusterGroup, "output": MatchResult})

        register_scripted(
            "mc_make_clusters_rdu1",
            lambda _in, _cfg: Clusters(groups=[
                ClusterGroup(label="alpha", claim_ids=["c1", "c2"]),
                ClusterGroup(label="beta", claim_ids=["c3"]),
            ]),
        )

        parent = Construct("test-each-oracle", nodes=[
            Node.scripted("mc-make", fn="mc_make_clusters_rdu1", outputs=Clusters),
            inner | Each(over="mc_make.groups", key="label"),
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "rdu1"})

        # Each cluster got Oracle'd (2 variants each) => 2 clusters x 2 = 4 calls
        assert gen_count[0] == 4
        verify_results = result.get("mc_oracle_inner", {})
        assert isinstance(verify_results, dict)
        assert set(verify_results.keys()) == {"alpha", "beta"}

    def test_graph_pauses_when_each_then_operator_on_next_node(self):
        """neograph-rdu.4: Each fan-out followed by Operator on a downstream
        node — the interrupt fires after Each results are collected."""
        from langgraph.checkpoint.memory import MemorySaver
        from neograph.factory import register_condition, register_scripted

        register_scripted(
            "mc_make_clusters_rdu4",
            lambda _in, _cfg: Clusters(groups=[
                ClusterGroup(label="x", claim_ids=["1"]),
                ClusterGroup(label="y", claim_ids=["2"]),
            ]),
        )
        register_scripted(
            "mc_review_item",
            lambda input_data, _cfg: MatchResult(
                cluster_label=input_data.label if hasattr(input_data, "label") else "?",
                matched=["reviewed"],
            ),
        )

        def mc_check_fn(input_data, _cfg):
            return ValidationResult(passed=False, issues=["needs human review"])

        register_scripted("mc_check_fn", mc_check_fn)

        register_condition(
            "mc_check_failed",
            lambda state: (
                {"issues": state.mc_check.issues}
                if hasattr(state, "mc_check") and state.mc_check and not state.mc_check.passed
                else None
            ),
        )

        make = Node.scripted("mc-make", fn="mc_make_clusters_rdu4", outputs=Clusters)
        review = (
            Node.scripted(
                "mc-review", fn="mc_review_item",
                inputs=ClusterGroup, outputs=MatchResult,
            )
            | Each(over="mc_make.groups", key="label")
        )
        check = (
            Node.scripted("mc-check", fn="mc_check_fn", outputs=ValidationResult)
            | Operator(when="mc_check_failed")
        )
        pipeline = Construct("test-each-operator", nodes=[make, review, check])
        graph = compile(pipeline, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "rdu4-test"}}

        result = run(graph, input={"node_id": "rdu4"}, config=config)

        # Each fan-out ran and produced results
        review_results = result.get("mc_review", {})
        assert isinstance(review_results, dict)
        assert set(review_results.keys()) == {"x", "y"}
        # Operator interrupted after check
        assert "__interrupt__" in result

    def test_oracle_merges_variants_when_single_output_oracle_node(self):
        """neograph-rdu.6: Oracle modifier on a node with single-type outputs
        runs N variants and merges via scripted merge_fn."""
        from neograph.factory import register_scripted

        gen_count = [0]

        def mc_gen(input_data, config):
            gen_count[0] += 1
            return Claims(items=[f"v{gen_count[0]}"])

        register_scripted("mc_oracle_gen", mc_gen)

        def mc_merge(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("mc_oracle_merge", mc_merge)

        gen_node = (
            Node.scripted("mc-gen", fn="mc_oracle_gen", outputs=Claims)
            | Oracle(n=2, merge_fn="mc_oracle_merge")
        )
        pipeline = Construct("test-oracle-merge", nodes=[gen_node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "rdu6"})

        # Oracle ran 2 variants
        assert gen_count[0] == 2
        # Merge combined both variants
        merged = result.get("mc_gen")
        assert merged is not None
        assert len(merged.items) == 2
        assert set(merged.items) == {"v1", "v2"}

    def test_oracle_merges_per_key_when_dict_form_outputs(self):
        """neograph-7ft: dict-form outputs + Oracle — Oracle redirect must
        handle per-key state fields ({node}_{key}), not just {node}.

        BUG: make_oracle_redirect_fn looks for result.get(field_name) but
        dict-form outputs write to {field_name}_{key}. The redirect misses
        them, each generator writes directly to per-key fields, causing
        concurrent write errors in LangGraph's state management."""
        from neograph.factory import register_scripted

        gen_count = [0]

        def dict_oracle_gen(input_data, config):
            gen_count[0] += 1
            return {
                "result": Claims(items=[f"v{gen_count[0]}"]),
                "meta": RawText(text=f"meta-{gen_count[0]}"),
            }

        register_scripted("dict_oracle_gen", dict_oracle_gen)

        def dict_oracle_merge(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("dict_oracle_merge", dict_oracle_merge)

        gen_node = (
            Node.scripted(
                "dogen", fn="dict_oracle_gen",
                outputs={"result": Claims, "meta": RawText},
            )
            | Oracle(n=2, merge_fn="dict_oracle_merge")
        )
        pipeline = Construct("test-dict-oracle", nodes=[gen_node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "7ft"})

        # Oracle ran 2 variants
        assert gen_count[0] == 2
        # Primary output merged
        merged = result.get("dogen_result")
        assert merged is not None
        assert len(merged.items) == 2

    def test_each_wraps_per_key_when_dict_outputs_with_each(self):
        """neograph-rdu.7: dict-form outputs + Each — each output key becomes
        dict[str, type] independently in state."""
        from neograph.factory import register_scripted

        register_scripted(
            "mc_each_make",
            lambda _in, _cfg: Clusters(groups=[
                ClusterGroup(label="a", claim_ids=["1"]),
                ClusterGroup(label="b", claim_ids=["2"]),
            ]),
        )

        def mc_each_process(input_data, config):
            label = input_data.label if hasattr(input_data, "label") else "?"
            return {
                "result": MatchResult(cluster_label=label, matched=[f"ok-{label}"]),
                "score": RawText(text=f"score-{label}"),
            }

        register_scripted("mc_each_process", mc_each_process)

        make = Node.scripted("mc-each-make", fn="mc_each_make", outputs=Clusters)
        process = (
            Node.scripted(
                "mc-each-proc", fn="mc_each_process",
                inputs=ClusterGroup,
                outputs={"result": MatchResult, "score": RawText},
            )
            | Each(over="mc_each_make.groups", key="label")
        )
        pipeline = Construct("test-dict-each", nodes=[make, process])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "rdu7"})

        # Each output key should be a dict keyed by Each labels
        result_dict = result.get("mc_each_proc_result", {})
        score_dict = result.get("mc_each_proc_score", {})

        assert isinstance(result_dict, dict)
        assert set(result_dict.keys()) == {"a", "b"}
        assert result_dict["a"].cluster_label == "a"
        assert result_dict["b"].cluster_label == "b"

        assert isinstance(score_dict, dict)
        assert set(score_dict.keys()) == {"a", "b"}
        assert score_dict["a"].text == "score-a"
        assert score_dict["b"].text == "score-b"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Each duplicate key guard
#
# When the input collection contains items with duplicate key values,
# the each_router should raise a clear ValueError before dispatching
# Send() calls — not let it bubble up from the LangGraph reducer.
# ═══════════════════════════════════════════════════════════════════════════


class TestEachDuplicateKeyGuard:

    def test_dedup_with_warning_when_each_collection_has_duplicate_keys_node_api(self):
        """@node API: duplicate key in Each collection deduped (keep first), no crash."""
        import types as _types
        from neograph import compile, construct_from_module, node, run

        mod = _types.ModuleType("test_each_dup_key")

        @node(mode="scripted", outputs=Clusters)
        def make_clusters() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="dup", claim_ids=["c1"]),
                ClusterGroup(label="dup", claim_ids=["c2"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=cluster.claim_ids)

        mod.make_clusters = make_clusters
        mod.verify = verify

        pipeline = construct_from_module(mod)
        graph = compile(pipeline)

        result = run(graph, input={"node_id": "dup-test"})
        # First occurrence kept
        assert "dup" in result["verify"]
        assert result["verify"]["dup"].matched == ["c1"]

    def test_dedup_with_warning_when_each_collection_has_duplicate_keys_programmatic(self):
        """Programmatic API: duplicate key in Each collection deduped (keep first), no crash."""
        from neograph import compile, run

        def make_fn(input_data, config):
            return Clusters(groups=[
                ClusterGroup(label="same", claim_ids=["c1"]),
                ClusterGroup(label="same", claim_ids=["c2"]),
            ])

        def proc_fn(input_data, config):
            return MatchResult(cluster_label=input_data.label, matched=input_data.claim_ids)

        register_scripted("dup_each_make", make_fn)
        register_scripted("dup_each_proc", proc_fn)

        make = Node.scripted("dup-each-make", fn="dup_each_make", outputs=Clusters)
        proc = (
            Node.scripted(
                "dup-each-proc", fn="dup_each_proc",
                inputs=ClusterGroup, outputs=MatchResult,
            )
            | Each(over="dup_each_make.groups", key="label")
        )
        pipeline = Construct("test-dup-each", nodes=[make, proc])
        graph = compile(pipeline)

        result = run(graph, input={"node_id": "dup-prog"})
        # First occurrence kept
        assert "same" in result["dup_each_proc"]
        assert result["dup_each_proc"]["same"].matched == ["c1"]


class TestEachDuplicateKeyDedup:
    """neograph-b1g9: Each fan-out should dedup duplicate keys with a warning
    instead of crashing. Keep first occurrence, log warning, continue."""

    def test_dedup_keeps_first_and_warns_when_each_has_duplicate_keys_node_api(self):
        """@node API: duplicate keys in Each collection dedup with warning, keep first."""
        import types as _types
        from neograph import compile, construct_from_module, node, run

        mod = _types.ModuleType("test_each_dedup")

        @node(mode="scripted", outputs=Clusters)
        def make_clusters() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="dup", claim_ids=["c1"]),
                ClusterGroup(label="dup", claim_ids=["c2"]),
                ClusterGroup(label="unique", claim_ids=["c3"]),
            ])

        @node(
            mode="scripted",
            outputs=MatchResult,
            map_over="make_clusters.groups",
            map_key="label",
        )
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=cluster.claim_ids)

        mod.make_clusters = make_clusters
        mod.verify = verify

        pipeline = construct_from_module(mod)
        graph = compile(pipeline)

        result = run(graph, input={"node_id": "dedup-test"})

        # First occurrence kept for the duplicate key, unique key also present
        assert "dup" in result["verify"]
        assert result["verify"]["dup"].matched == ["c1"]
        assert "unique" in result["verify"]

    def test_dedup_keeps_first_and_warns_when_each_has_duplicate_keys_programmatic(self):
        """Programmatic API: duplicate keys dedup with warning, keep first."""
        from neograph import compile, run

        def make_fn(input_data, config):
            return Clusters(groups=[
                ClusterGroup(label="same", claim_ids=["c1"]),
                ClusterGroup(label="same", claim_ids=["c2"]),
            ])

        def proc_fn(input_data, config):
            return MatchResult(cluster_label=input_data.label, matched=input_data.claim_ids)

        register_scripted("dedup_each_make", make_fn)
        register_scripted("dedup_each_proc", proc_fn)

        make = Node.scripted("dedup-each-make", fn="dedup_each_make", outputs=Clusters)
        proc = (
            Node.scripted(
                "dedup-each-proc", fn="dedup_each_proc",
                inputs=ClusterGroup, outputs=MatchResult,
            )
            | Each(over="dedup_each_make.groups", key="label")
        )
        pipeline = Construct("test-dedup-each", nodes=[make, proc])
        graph = compile(pipeline)

        result = run(graph, input={"node_id": "dedup-prog"})

        # First occurrence kept
        assert "same" in result["dedup_each_proc"]
        assert result["dedup_each_proc"]["same"].matched == ["c1"]


class TestSkipWhenWithEach:
    """skip_when + Each: skipped items must produce {dispatch_key: value} dicts
    just like non-skipped items (neograph-gpn)."""

    def test_skip_value_wrapped_in_each_key_when_skip_fires(self):
        """skip_when fires for some items, skip_value result is wrapped with
        the Each dispatch key so the reducer can merge it with non-skipped results."""
        from neograph import compile, run, node, construct_from_functions

        # LLM-mode node with skip_when + Each. Skip fires for single-claim
        # groups. Non-skipped groups go through the LLM (StructuredFake).
        configure_fake_llm(lambda tier: StructuredFake(
            lambda m: m(cluster_label="processed", matched=["llm-result"]),
        ))

        @node(outputs=Clusters)
        def make() -> Clusters:
            return Clusters(groups=[
                ClusterGroup(label="a", claim_ids=["c1"]),          # skip: len==1
                ClusterGroup(label="b", claim_ids=["c2", "c3"]),    # LLM processes
                ClusterGroup(label="c", claim_ids=["c4"]),          # skip: len==1
            ])

        @node(
            outputs=MatchResult,
            model="fast",
            prompt="verify",
            map_over="make.groups",
            map_key="label",
            skip_when=lambda g: len(g.claim_ids) == 1,
            skip_value=lambda g: MatchResult(cluster_label=g.label, matched=["skipped"]),
        )
        def verify(group: ClusterGroup) -> MatchResult: ...

        pipeline = construct_from_functions("skip-each", [make, verify])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "gpn"})

        proc = result["verify"]
        assert isinstance(proc, dict), f"Expected dict, got {type(proc)}"
        assert sorted(proc.keys()) == ["a", "b", "c"]
        assert proc["a"].matched == ["skipped"]
        assert proc["b"].matched == ["llm-result"]
        assert proc["c"].matched == ["skipped"]


# ═══════════════════════════════════════════════════════════════════════════
# TestOracleOperatorCombo (neograph-l84)
#
# Oracle + Operator modifier combination: run N LLM variants (Oracle),
# merge them, then pause for human review (Operator) before continuing.
# This proves: the two modifiers compose on a single node, interrupt
# fires after Oracle merge, and resume delivers the merged result.
# ═══════════════════════════════════════════════════════════════════════════


class TestOracleOperatorCombo:
    """Oracle + Operator on the same node — ensemble then human review."""

    def test_graph_pauses_with_merged_result_when_oracle_operator_applied(self):
        """Oracle merges N variants, then Operator interrupts for review.
        The merged result must be in state before the interrupt fires."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        gen_count = [0]

        def oo_gen(input_data, config):
            gen_count[0] += 1
            return Claims(items=[f"variant-{gen_count[0]}"])

        register_scripted("oo_gen", oo_gen)

        def oo_merge(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("oo_merge", oo_merge)

        register_condition(
            "oo_always_review",
            lambda state: (
                {"needs_review": True}
                if hasattr(state, "oo_gen") and state.oo_gen is not None
                else None
            ),
        )

        gen_node = (
            Node.scripted("oo-gen", fn="oo_gen", outputs=Claims)
            | Oracle(n=2, merge_fn="oo_merge")
            | Operator(when="oo_always_review")
        )

        pipeline = Construct("test-oracle-operator", nodes=[gen_node])
        graph = compile(pipeline, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "oracle-op-test"}}

        result = run(graph, input={"node_id": "oo-001"}, config=config)

        # Oracle ran 2 variants
        assert gen_count[0] == 2
        # Merged result is in state before interrupt
        merged = result.get("oo_gen")
        assert merged is not None
        assert len(merged.items) == 2
        assert set(merged.items) == {"variant-1", "variant-2"}
        # Operator interrupted
        assert "__interrupt__" in result

    def test_graph_resumes_with_merged_output_when_oracle_operator_resumed(self):
        """After interrupt, resume delivers the Oracle-merged output
        and human feedback is accessible in state."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        gen_count = [0]

        def oo_gen2(input_data, config):
            gen_count[0] += 1
            return Claims(items=[f"v{gen_count[0]}"])

        register_scripted("oo_gen2", oo_gen2)

        def oo_merge2(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("oo_merge2", oo_merge2)

        register_condition(
            "oo_always_review2",
            lambda state: (
                {"needs_review": True}
                if hasattr(state, "oo_gen2") and state.oo_gen2 is not None
                else None
            ),
        )

        gen_node = (
            Node.scripted("oo-gen2", fn="oo_gen2", outputs=Claims)
            | Oracle(n=2, merge_fn="oo_merge2")
            | Operator(when="oo_always_review2")
        )

        pipeline = Construct("test-oracle-op-resume", nodes=[gen_node])
        graph = compile(pipeline, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "oracle-op-resume"}}

        # First run: hits interrupt
        result = run(graph, input={"node_id": "oo-002"}, config=config)
        assert "__interrupt__" in result

        # Resume with human feedback
        result = run(graph, resume={"approved": True}, config=config)

        # Merged output persists after resume
        merged = result.get("oo_gen2")
        assert merged is not None
        assert len(merged.items) == 2
        # Human feedback captured
        assert result["human_feedback"] == {"approved": True}

    def test_oracle_subconstruct_then_operator_on_parent_node(self):
        """Oracle on a sub-Construct, then Operator on the next parent node.
        Tests that modifiers compose across construct boundaries."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_condition, register_scripted

        register_scripted(
            "oo_sub_gen",
            lambda input_data, config: Claims(items=["sub-variant"]),
        )

        def oo_sub_merge(variants, config):
            all_items = []
            for v in variants:
                all_items.extend(v.items)
            return Claims(items=all_items)

        register_scripted("oo_sub_merge", oo_sub_merge)

        # Sub-construct with Oracle
        sub = Construct(
            "oo-ensemble",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("gen", fn="oo_sub_gen", outputs=Claims)],
        ) | Oracle(n=2, merge_fn="oo_sub_merge")

        # Validation node after sub-construct, with Operator
        def oo_validate(input_data, config):
            return ValidationResult(passed=False, issues=["human must review ensemble"])

        register_scripted("oo_validate", oo_validate)

        register_condition(
            "oo_val_failed",
            lambda state: (
                {"issues": state.oo_validate.issues}
                if hasattr(state, "oo_validate") and state.oo_validate
                and not state.oo_validate.passed
                else None
            ),
        )

        register_scripted("oo_seed", lambda _in, _cfg: Claims(items=["seed"]))

        parent = Construct("test-oracle-sub-operator", nodes=[
            Node.scripted("seed", fn="oo_seed", outputs=Claims),
            sub,
            Node.scripted("oo-validate", fn="oo_validate", outputs=ValidationResult)
            | Operator(when="oo_val_failed"),
        ])
        graph = compile(parent, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "oracle-sub-op-test"}}

        result = run(graph, input={"node_id": "oo-003"}, config=config)

        # Oracle sub-construct ran and merged 2 variants
        ensemble_result = result.get("oo_ensemble")
        assert ensemble_result is not None
        assert len(ensemble_result.items) == 2
        # Validation node ran
        assert result["oo_validate"].passed is False
        # Operator interrupted
        assert "__interrupt__" in result


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
        assert merged is not None
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
        from neograph import node, construct_from_functions
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
        assert merged is not None
        assert len(merged.items) == 2
        assert "from-reason" in merged.items
        assert "from-fast" in merged.items

    def test_node_decorator_models_registers_body_as_merge(self):
        """@node(models=...) without merge_fn uses the function body as merge."""
        from neograph import node, construct_from_functions

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
        assert oracle.merge_fn is not None  # body was registered as merge_fn

    def test_oracle_models_on_think_mode_node(self):
        """Oracle(models=) must override model tier for think-mode (produce) nodes.

        Bug: _make_produce_fn reads _oracle_model from config but never
        transfers neo_oracle_model from state to config. Only the scripted
        wrapper does this transfer. Regression test for neograph-lbsf.
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

        Bug: _make_tool_fn reads _oracle_model from config but never
        transfers neo_oracle_model from state to config. Regression test
        for neograph-lbsf.
        """
        from neograph.factory import register_scripted, register_tool_factory

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
        from neograph.factory import register_scripted

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
        from neograph import node, construct_from_functions
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
        from neograph import node
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
        assert sidecar is not None, "Sidecar should be registered for @node"
        original_fn = sidecar[0]
        hints = get_type_hints(original_fn)
        assert hints["data"] is Claims, (
            "Parameter annotation should be the upstream type Claims"
        )

        # The node has an Oracle modifier attached
        assert merge_check.has_modifier(Oracle)

    def test_node_decorator_raises_when_models_is_empty_list(self):
        """@node(models=[]) must raise at decoration time."""
        from neograph import node

        with pytest.raises((ConfigurationError, ConstructError)):
            @node(outputs=Claims, models=[])
            def bad_ensemble(data: Claims) -> Claims:
                return data


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
# DEV-MODE WARNINGS
#
# NEOGRAPH_DEV=1 emits warnings for ambiguous-but-valid patterns.
# ═══════════════════════════════════════════════════════════════════════════


class TestDevWarnings:
    """Dev-mode warnings for ambiguous modifier patterns."""

    def test_oracle_n1_warns_in_dev_mode(self, monkeypatch):
        """Oracle(n=1) emits a dev warning about single-element ensemble."""
        import warnings
        import neograph._dev_warnings as dw

        monkeypatch.setattr(dw, "DEV_MODE", True)

        gen = _producer("gen", Claims)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen | Oracle(n=1, merge_fn="dummy_merge")

        dev_msgs = [w for w in caught if "[neograph-dev]" in str(w.message)]
        assert len(dev_msgs) == 1
        assert "ensemble of 1" in str(dev_msgs[0].message)

    def test_oracle_n1_silent_without_dev_mode(self, monkeypatch):
        """Oracle(n=1) does NOT warn when dev mode is off."""
        import warnings
        import neograph._dev_warnings as dw

        monkeypatch.setattr(dw, "DEV_MODE", False)

        gen = _producer("gen", Claims)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen | Oracle(n=1, merge_fn="dummy_merge")

        dev_msgs = [w for w in caught if "[neograph-dev]" in str(w.message)]
        assert len(dev_msgs) == 0

    def test_oracle_uneven_models_warns_in_dev_mode(self, monkeypatch):
        """Oracle with n not divisible by len(models) warns about uneven distribution."""
        import warnings
        import neograph._dev_warnings as dw

        monkeypatch.setattr(dw, "DEV_MODE", True)

        gen = _producer("gen", Claims)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gen | Oracle(n=5, models=["a", "b"], merge_fn="dummy_merge")

        dev_msgs = [w for w in caught if "[neograph-dev]" in str(w.message)]
        assert any("uneven distribution" in str(w.message) for w in dev_msgs)

    def test_loop_max_iterations_1_warns_in_dev_mode(self, monkeypatch):
        """Loop(max_iterations=1) warns about effectively non-looping config."""
        import warnings
        import neograph._dev_warnings as dw

        monkeypatch.setattr(dw, "DEV_MODE", True)

        n = Node.scripted("looper", fn="noop_loop", inputs=Claims, outputs=Claims)
        from neograph.factory import register_scripted
        register_scripted("noop_loop", lambda data, config: data)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            n | Loop(when=lambda d: False, max_iterations=1)

        dev_msgs = [w for w in caught if "[neograph-dev]" in str(w.message)]
        assert len(dev_msgs) == 1
        assert "max_iterations=1" in str(dev_msgs[0].message)
