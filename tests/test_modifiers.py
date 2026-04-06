"""Modifier tests — Oracle, Each, Operator, map(), deep compositions,
modifier-as-first-node, Construct-level modifiers, list-over-Each e2e.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest

from neograph import (
    Construct, ConstructError, Node, Each, Oracle, Operator, Tool,
    compile, construct_from_functions, construct_from_module,
    merge_fn, node, run, tool,
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
        """Oracle without merge_prompt or merge_fn is a ValueError."""
        with pytest.raises(ValueError, match="merge_prompt.*merge_fn"):
            Oracle(n=3)

    def test_oracle_raises_when_both_merge_options_given(self):
        """Oracle with both merge_prompt and merge_fn is a ValueError."""
        with pytest.raises(ValueError, match="not both"):
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
                    mode="gather",
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
