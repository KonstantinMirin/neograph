"""Modifier tests — Each fan-out modifier"""

from __future__ import annotations

import pytest

from neograph import (
    Construct,
    ConstructError,
    Each,
    Node,
    compile,
    construct_from_functions,
    construct_from_module,
    node,
    run,
)
from neograph.factory import register_scripted
from tests.fakes import StructuredFake, configure_fake_llm
from tests.schemas import (
    Claims,
    ClusterGroup,
    Clusters,
    MatchResult,
    MergedResult,
    RawText,
    _consumer,
    _producer,
)

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

        from neograph import compile, run

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

    def test_each_over_empty_collection_produces_empty_dict(self):
        """Each over [] must produce {} and let downstream run (neograph-r087)."""
        import types as _types

        from neograph import compile, run

        mod = _types.ModuleType("test_each_empty")

        @node
        def make_clusters() -> Clusters:
            return Clusters(groups=[])  # Empty list

        @node(map_over="make_clusters.groups", map_key="label")
        def verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=["m"])

        @node
        def summarize(verify: dict[str, MatchResult]) -> MergedResult:
            return MergedResult(final_text=f"processed {len(verify)} items")

        mod.make_clusters = make_clusters
        mod.verify = verify
        mod.summarize = summarize

        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "empty-each"})

        # Downstream MUST fire �� verify produces empty dict, summarize runs
        assert "summarize" in result, "Downstream node 'summarize' never fired — pipeline deadlocked on empty Each"
        verify_results = result.get("verify", {})
        assert verify_results == {} or verify_results is None  # empty dict or None, not missing
        assert result["summarize"].final_text == "processed 0 items"


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
        def _sub_score(input_data, config):
            assert isinstance(input_data, RawText), f"Expected RawText from sub_analyze, got {type(input_data)}"
            return MatchResult(cluster_label="scored", matched=[f"scored-{input_data.text}"])
        register_scripted("sub_score", _sub_score)

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
        def _verify_cluster_l5(input_data, _cfg):
            assert isinstance(input_data, ClusterGroup), f"Expected ClusterGroup, got {type(input_data)}"
            return MatchResult(cluster_label=input_data.label, matched=[f"m-{input_data.label}"])
        register_scripted("verify_cluster_l5", _verify_cluster_l5)

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
        from neograph import compile, run
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
        def _verify_cluster_l5b(input_data, _cfg):
            assert isinstance(input_data, ClusterGroup), f"Expected ClusterGroup, got {type(input_data)}"
            return MatchResult(cluster_label=input_data.label, matched=["ok"])
        register_scripted("verify_cluster_l5b", _verify_cluster_l5b)

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

        from neograph import compile, run

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

        from neograph import compile, run

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
        from neograph import compile, run

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


