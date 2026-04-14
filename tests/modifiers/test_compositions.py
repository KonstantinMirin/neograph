"""Modifier tests — modifier compositions, map, deep combos, fusion"""

from __future__ import annotations

from typing import Annotated

import pytest

from neograph import (
    Construct,
    Each,
    Node,
    Operator,
    Oracle,
    Tool,
    compile,
    construct_from_functions,
    construct_from_module,
    merge_fn,
    node,
    run,
)
from neograph.factory import register_scripted, register_tool_factory
from tests.fakes import FakeTool, ReActFake, configure_fake_llm
from tests.schemas import (
    Claims,
    ClusterGroup,
    Clusters,
    MatchResult,
    RawText,
    ValidationResult,
)

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

        assert via_map.modifier_set == via_pipe.modifier_set
        assert via_map.modifiers == via_pipe.modifiers  # backward compat bridge

    def test_fanout_runs_when_map_sugar_used(self):
        """.map() drives the same fan-out/collect behavior as | Each(...)."""
        from neograph.factory import register_scripted

        register_scripted("make_clusters", lambda input_data, config: Clusters(
            groups=[
                ClusterGroup(label="alpha", claim_ids=["c1", "c2"]),
                ClusterGroup(label="beta", claim_ids=["c3"]),
            ]
        ))
        def _verify_cluster(input_data, config):
            assert isinstance(input_data, ClusterGroup), f"Expected ClusterGroup, got {type(input_data)}"
            return MatchResult(cluster_label=input_data.label, matched=["match-1"])
        register_scripted("verify_cluster", _verify_cluster)

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
        assert isinstance(merged, Claims)
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
        from neograph.factory import register_scripted

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
        assert isinstance(result["deep_search"], RawText)

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
    def _tsp_verify(input_data, _cfg):
        assert isinstance(input_data, ClusterGroup), f"Expected ClusterGroup, got {type(input_data)}"
        return MatchResult(cluster_label=input_data.label, matched=[f"m-{input_data.label}"])
    register_scripted("tsp_verify", _tsp_verify)

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
        def _mc_review_item(input_data, _cfg):
            assert isinstance(input_data, ClusterGroup), f"Expected ClusterGroup, got {type(input_data)}"
            return MatchResult(cluster_label=input_data.label, matched=["reviewed"])
        register_scripted("mc_review_item", _mc_review_item)

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
        assert isinstance(merged, Claims)
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
        assert isinstance(merged, Claims)
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
            assert isinstance(input_data, ClusterGroup), f"Expected ClusterGroup, got {type(input_data)}"
            return {
                "result": MatchResult(cluster_label=input_data.label, matched=[f"ok-{input_data.label}"]),
                "score": RawText(text=f"score-{input_data.label}"),
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
        assert isinstance(merged, Claims)
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
        assert isinstance(merged, Claims)
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
        assert isinstance(ensemble_result, Claims)
        assert len(ensemble_result.items) == 2
        # Validation node ran
        assert result["oo_validate"].passed is False
        # Operator interrupted
        assert "__interrupt__" in result


# =============================================================================
# Oracle models= — multi-model ensemble (neograph-beyr)
# =============================================================================





# ═══════════════════════════════════════════════════════════════════════════
# EACH×ORACLE FUSION (neograph-tpgi)
#
# map_over + ensemble_n on the same @node: flat M×N Send topology.
# ═══════════════════════════════════════════════════════════════════════════


class TestEachOracleFusion:
    """Each×Oracle fusion: map_over + ensemble_n on the same node."""

    def test_each_oracle_fusion_e2e_scripted(self):
        """M items × N generators, grouped merge, dict output."""
        from pydantic import BaseModel

        from neograph.factory import register_scripted

        class Chunk(BaseModel, frozen=True):
            chunk_idx: str
            text: str

        class ChunkList(BaseModel, frozen=True):
            items: list[Chunk]

        class Result(BaseModel, frozen=True):
            text: str
            model: str

        call_log = []

        register_scripted("tpgi_chunks", lambda i, c: ChunkList(items=[
            Chunk(chunk_idx="A", text="auth section"),
            Chunk(chunk_idx="B", text="billing section"),
        ]))

        register_scripted("tpgi_gen", lambda i, c: Result(
            text="processed",
            model=c.get("configurable", {}).get("_oracle_model", "default"),
        ))

        register_scripted("tpgi_merge", lambda variants, c: Result(
            text=f"merged({len(variants)})",
            model="merged",
        ))

        pipeline = Construct("fusion-test", nodes=[
            Node.scripted("chunks", fn="tpgi_chunks", outputs=ChunkList),
            Node.scripted("decompose", fn="tpgi_gen",
                          inputs=Chunk, outputs=Result)
            | Oracle(n=3, merge_fn="tpgi_merge")
            | Each(over="chunks.items", key="chunk_idx"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fusion-1"})

        decompose = result["decompose"]
        assert isinstance(decompose, dict), f"Expected dict, got {type(decompose)}"
        assert set(decompose.keys()) == {"A", "B"}
        # Each chunk got 3 variants merged into 1
        assert decompose["A"].text == "merged(3)"
        assert decompose["B"].text == "merged(3)"

    def test_each_oracle_fusion_via_node_decorator(self):
        """@node with map_over + ensemble_n produces fused topology."""
        from pydantic import BaseModel

        class Item(BaseModel, frozen=True):
            item_id: str
            value: int

        class ItemBatch(BaseModel, frozen=True):
            items: list[Item]

        class Scored(BaseModel, frozen=True):
            item_id: str
            score: float

        @node(outputs=ItemBatch)
        def make_items() -> ItemBatch:
            return ItemBatch(items=[
                Item(item_id="x", value=10),
                Item(item_id="y", value=20),
            ])

        @merge_fn
        def pick_best(variants: list[Scored]) -> Scored:
            return max(variants, key=lambda v: v.score)

        @node(
            outputs=Scored,
            map_over="make_items.items",
            map_key="item_id",
            ensemble_n=2,
            merge_fn="pick_best",
        )
        def score(item: Item) -> Scored:
            return Scored(item_id=item.item_id, score=item.value * 0.1)

        pipeline = construct_from_functions("decorator-fusion", [make_items, score])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fusion-2"})

        scored = result["score"]
        assert isinstance(scored, dict)
        assert set(scored.keys()) == {"x", "y"}
        assert scored["x"].score == 1.0
        assert scored["y"].score == 2.0

    def test_each_oracle_fusion_downstream_list_consumer(self):
        """list[X] consumer of Each×Oracle dict[str, X] works (merge-after-fanout)."""
        from pydantic import BaseModel

        from neograph.factory import register_scripted

        class Chunk(BaseModel, frozen=True):
            chunk_idx: str

        class ChunkList(BaseModel, frozen=True):
            items: list[Chunk]

        class Result(BaseModel, frozen=True):
            label: str

        register_scripted("tpgi_c", lambda i, c: ChunkList(items=[
            Chunk(chunk_idx="A"), Chunk(chunk_idx="B"),
        ]))
        register_scripted("tpgi_g", lambda i, c: Result(label="gen"))
        register_scripted("tpgi_m", lambda v, c: Result(label=f"m({len(v)})"))
        register_scripted("tpgi_collect", lambda i, c: RawText(
            text=f"collected {len(i['decompose'])} items",
        ))

        pipeline = Construct("fusion-list", nodes=[
            Node.scripted("chunks", fn="tpgi_c", outputs=ChunkList),
            Node.scripted("decompose", fn="tpgi_g", inputs=Chunk, outputs=Result)
            | Oracle(n=2, merge_fn="tpgi_m")
            | Each(over="chunks.items", key="chunk_idx"),
            Node.scripted("collect", fn="tpgi_collect",
                          inputs={"decompose": list[Result]}, outputs=RawText),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fusion-list"})

        assert result["collect"].text == "collected 2 items"

    def test_each_oracle_order_irrelevant_on_node(self):
        """Node | Oracle | Each produces same result as Node | Each | Oracle."""
        from pydantic import BaseModel

        from neograph.factory import register_scripted

        class Tpgi2Chunk(BaseModel, frozen=True):
            chunk_idx: str

        class Tpgi2ChunkList(BaseModel, frozen=True):
            items: list[Tpgi2Chunk]

        class Tpgi2Result(BaseModel, frozen=True):
            label: str

        register_scripted("tpgi2_src", lambda i, c: Tpgi2ChunkList(items=[
            Tpgi2Chunk(chunk_idx="P"), Tpgi2Chunk(chunk_idx="Q"),
        ]))
        register_scripted("tpgi2_gen", lambda i, c: Tpgi2Result(label="gen"))
        register_scripted("tpgi2_mrg", lambda v, c: Tpgi2Result(
            label=f"merged({len(v)})",
        ))

        # Order 1: Oracle first, then Each
        p1 = Construct("order-oe", nodes=[
            Node.scripted("chunks", fn="tpgi2_src", outputs=Tpgi2ChunkList),
            Node.scripted("proc", fn="tpgi2_gen", inputs=Tpgi2Chunk, outputs=Tpgi2Result)
            | Oracle(n=2, merge_fn="tpgi2_mrg")
            | Each(over="chunks.items", key="chunk_idx"),
        ])
        g1 = compile(p1)
        r1 = run(g1, input={"node_id": "order-1"})

        # Order 2: Each first, then Oracle
        p2 = Construct("order-eo", nodes=[
            Node.scripted("chunks", fn="tpgi2_src", outputs=Tpgi2ChunkList),
            Node.scripted("proc", fn="tpgi2_gen", inputs=Tpgi2Chunk, outputs=Tpgi2Result)
            | Each(over="chunks.items", key="chunk_idx")
            | Oracle(n=2, merge_fn="tpgi2_mrg"),
        ])
        g2 = compile(p2)
        r2 = run(g2, input={"node_id": "order-2"})

        # Both produce same shape and values
        assert isinstance(r1["proc"], dict)
        assert isinstance(r2["proc"], dict)
        assert set(r1["proc"].keys()) == {"P", "Q"}
        assert set(r2["proc"].keys()) == {"P", "Q"}
        assert r1["proc"]["P"].label == r2["proc"]["P"].label == "merged(2)"
        assert r1["proc"]["Q"].label == r2["proc"]["Q"].label == "merged(2)"

    def test_each_oracle_models_routed_to_generators(self):
        """Oracle models= routes different models to different generators within each item."""
        from pydantic import BaseModel

        from neograph.factory import register_scripted

        class Tpgi2Item(BaseModel, frozen=True):
            item_id: str

        class Tpgi2Items(BaseModel, frozen=True):
            items: list[Tpgi2Item]

        class Tpgi2ModelResult(BaseModel, frozen=True):
            model_used: str

        register_scripted("tpgi2_items", lambda i, c: Tpgi2Items(items=[
            Tpgi2Item(item_id="a"), Tpgi2Item(item_id="b"),
        ]))

        # Generator captures the oracle model from config
        register_scripted("tpgi2_model_gen", lambda i, c: Tpgi2ModelResult(
            model_used=c.get("configurable", {}).get("_oracle_model", "none"),
        ))

        # Merge collects which models were used
        def tpgi2_model_merge(variants, c):
            models = sorted(v.model_used for v in variants)
            return Tpgi2ModelResult(model_used=",".join(models))

        register_scripted("tpgi2_model_merge", tpgi2_model_merge)

        pipeline = Construct("models-test", nodes=[
            Node.scripted("items", fn="tpgi2_items", outputs=Tpgi2Items),
            Node.scripted("proc", fn="tpgi2_model_gen",
                          inputs=Tpgi2Item, outputs=Tpgi2ModelResult)
            | Oracle(n=3, models=["reason", "fast", "creative"],
                     merge_fn="tpgi2_model_merge")
            | Each(over="items.items", key="item_id"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "models-1"})

        proc = result["proc"]
        assert isinstance(proc, dict)
        # Each item got all 3 models
        for key in ("a", "b"):
            models = proc[key].model_used.split(",")
            assert sorted(models) == ["creative", "fast", "reason"]

    def test_each_oracle_single_item_collection(self):
        """Each with 1-item collection + Oracle still works (M=1, N=3)."""
        from pydantic import BaseModel

        from neograph.factory import register_scripted

        class Tpgi2Solo(BaseModel, frozen=True):
            solo_id: str

        class Tpgi2SoloList(BaseModel, frozen=True):
            items: list[Tpgi2Solo]

        class Tpgi2SoloResult(BaseModel, frozen=True):
            count: int

        gen_counter = [0]

        register_scripted("tpgi2_solo_src", lambda i, c: Tpgi2SoloList(
            items=[Tpgi2Solo(solo_id="only")],
        ))

        def tpgi2_solo_gen(i, c):
            gen_counter[0] += 1
            return Tpgi2SoloResult(count=1)

        register_scripted("tpgi2_solo_gen", tpgi2_solo_gen)
        register_scripted("tpgi2_solo_merge", lambda v, c: Tpgi2SoloResult(
            count=len(v),
        ))

        pipeline = Construct("solo-fusion", nodes=[
            Node.scripted("source", fn="tpgi2_solo_src", outputs=Tpgi2SoloList),
            Node.scripted("proc", fn="tpgi2_solo_gen",
                          inputs=Tpgi2Solo, outputs=Tpgi2SoloResult)
            | Oracle(n=3, merge_fn="tpgi2_solo_merge")
            | Each(over="source.items", key="solo_id"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "solo-1"})

        proc = result["proc"]
        assert isinstance(proc, dict)
        assert set(proc.keys()) == {"only"}
        # 3 generators ran, merge saw 3 variants
        assert proc["only"].count == 3
        assert gen_counter[0] == 3

    def test_each_oracle_with_decorated_merge_fn(self):
        """@merge_fn with FromInput DI works in fused topology."""
        from pydantic import BaseModel

        from neograph import FromInput
        from neograph import merge_fn as merge_fn_deco
        from neograph.factory import register_scripted

        class Tpgi2DIChunk(BaseModel, frozen=True):
            chunk_idx: str

        class Tpgi2DIChunks(BaseModel, frozen=True):
            items: list[Tpgi2DIChunk]

        class Tpgi2DIResult(BaseModel, frozen=True):
            text: str
            tag: str

        register_scripted("tpgi2_di_src", lambda i, c: Tpgi2DIChunks(items=[
            Tpgi2DIChunk(chunk_idx="X"), Tpgi2DIChunk(chunk_idx="Y"),
        ]))
        register_scripted("tpgi2_di_gen", lambda i, c: Tpgi2DIResult(
            text="draft", tag="",
        ))

        captured_node_ids: list[str] = []

        @merge_fn_deco
        def tpgi2_di_merge(
            variants: list[Tpgi2DIResult],
            node_id: Annotated[str, FromInput],
        ) -> Tpgi2DIResult:
            captured_node_ids.append(node_id)
            return Tpgi2DIResult(
                text=f"merged({len(variants)})",
                tag=f"id={node_id}",
            )

        pipeline = Construct("di-fusion", nodes=[
            Node.scripted("chunks", fn="tpgi2_di_src", outputs=Tpgi2DIChunks),
            Node.scripted("proc", fn="tpgi2_di_gen",
                          inputs=Tpgi2DIChunk, outputs=Tpgi2DIResult)
            | Oracle(n=2, merge_fn="tpgi2_di_merge")
            | Each(over="chunks.items", key="chunk_idx"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "di-test-42"})

        proc = result["proc"]
        assert isinstance(proc, dict)
        assert set(proc.keys()) == {"X", "Y"}

        # DI resolved node_id from input for each group's merge
        assert all(nid == "di-test-42" for nid in captured_node_ids)
        assert len(captured_node_ids) == 2  # one merge per each-item

        # Merged values reflect the DI-injected node_id
        for key in ("X", "Y"):
            assert proc[key].text == "merged(2)"
            assert proc[key].tag == "id=di-test-42"

    def test_each_oracle_merge_fn_with_bundled_pydantic_from_input(self):
        """FromInput[PydanticModel] (bundled resolution) in Each+Oracle merge_fn.

        When FromInput wraps a BaseModel subclass, DIBinding.resolve constructs
        the model by pulling each field from config['configurable']. This test
        verifies the bundled path works inside the fused Each+Oracle topology.
        """
        from pydantic import BaseModel

        from neograph import FromInput
        from neograph import merge_fn as merge_fn_deco
        from neograph.factory import register_scripted

        class RunCtx(BaseModel):
            node_id: str
            project_root: str

        class BundledChunk(BaseModel, frozen=True):
            idx: str

        class BundledChunks(BaseModel, frozen=True):
            items: list[BundledChunk]

        class BundledResult(BaseModel, frozen=True):
            text: str

        register_scripted("bundled_src", lambda i, c: BundledChunks(items=[
            BundledChunk(idx="a"), BundledChunk(idx="b"),
        ]))
        register_scripted("bundled_gen", lambda i, c: BundledResult(text="draft"))

        captured_ctx: list[RunCtx] = []

        @merge_fn_deco
        def bundled_merge(
            variants: list[BundledResult],
            ctx: Annotated[RunCtx, FromInput],
        ) -> BundledResult:
            captured_ctx.append(ctx)
            return BundledResult(text=f"merged({len(variants)}) root={ctx.project_root}")

        pipeline = Construct("bundled-di-fusion", nodes=[
            Node.scripted("src", fn="bundled_src", outputs=BundledChunks),
            Node.scripted("proc", fn="bundled_gen",
                          inputs=BundledChunk, outputs=BundledResult)
            | Oracle(n=2, merge_fn="bundled_merge")
            | Each(over="src.items", key="idx"),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "ctx-test", "project_root": "/proj"})

        proc = result["proc"]
        assert isinstance(proc, dict)
        assert set(proc.keys()) == {"a", "b"}

        # Bundled DI resolved RunCtx from individual config fields
        assert len(captured_ctx) == 2
        for ctx in captured_ctx:
            assert isinstance(ctx, RunCtx)
            assert ctx.node_id == "ctx-test"
            assert ctx.project_root == "/proj"

        for key in ("a", "b"):
            assert proc[key].text == "merged(2) root=/proj"

    def test_programmatic_pipe_both_orders(self):
        """Programmatic Node.scripted() | Oracle() | Each() and reverse both compile and run."""
        from pydantic import BaseModel

        from neograph.factory import register_scripted

        class Tpgi2ProgItem(BaseModel, frozen=True):
            prog_id: str

        class Tpgi2ProgList(BaseModel, frozen=True):
            items: list[Tpgi2ProgItem]

        class Tpgi2ProgResult(BaseModel, frozen=True):
            value: str

        register_scripted("tpgi2_prog_src", lambda i, c: Tpgi2ProgList(items=[
            Tpgi2ProgItem(prog_id="r"), Tpgi2ProgItem(prog_id="s"),
        ]))
        register_scripted("tpgi2_prog_gen", lambda i, c: Tpgi2ProgResult(
            value="generated",
        ))
        register_scripted("tpgi2_prog_merge", lambda v, c: Tpgi2ProgResult(
            value=f"merged({len(v)})",
        ))

        # Order 1: Oracle | Each
        node_oe = (
            Node.scripted("proc", fn="tpgi2_prog_gen",
                          inputs=Tpgi2ProgItem, outputs=Tpgi2ProgResult)
            | Oracle(n=2, merge_fn="tpgi2_prog_merge")
            | Each(over="source.items", key="prog_id")
        )
        p1 = Construct("prog-oe", nodes=[
            Node.scripted("source", fn="tpgi2_prog_src", outputs=Tpgi2ProgList),
            node_oe,
        ])
        g1 = compile(p1)
        r1 = run(g1, input={"node_id": "prog-oe"})

        # Order 2: Each | Oracle
        node_eo = (
            Node.scripted("proc", fn="tpgi2_prog_gen",
                          inputs=Tpgi2ProgItem, outputs=Tpgi2ProgResult)
            | Each(over="source.items", key="prog_id")
            | Oracle(n=2, merge_fn="tpgi2_prog_merge")
        )
        p2 = Construct("prog-eo", nodes=[
            Node.scripted("source", fn="tpgi2_prog_src", outputs=Tpgi2ProgList),
            node_eo,
        ])
        g2 = compile(p2)
        r2 = run(g2, input={"node_id": "prog-eo"})

        # Both produce same dict shape
        assert isinstance(r1["proc"], dict)
        assert isinstance(r2["proc"], dict)
        assert set(r1["proc"].keys()) == {"r", "s"}
        assert set(r2["proc"].keys()) == {"r", "s"}
        assert r1["proc"]["r"].value == r2["proc"]["r"].value == "merged(2)"
        assert r1["proc"]["s"].value == r2["proc"]["s"].value == "merged(2)"


# ═══════════════════════════════════════════════════════════════════════════
# DEV-MODE WARNINGS
#
# NEOGRAPH_DEV=1 emits warnings for ambiguous-but-valid patterns.
# ═══════════════════════════════════════════════════════════════════════════


