"""Validation tests — assembly-time type checking, fan-in validation,
Each path resolution, effective_producer_type, list/dict compatibility,
dict-form outputs validation, and Oracle error paths.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from tests.schemas import (
    RawText, Claims, ClassifiedClaims, ClusterGroup, Clusters,
    MatchResult, MergedResult, ValidationResult, _producer, _consumer,
)
from neograph import (
    Construct, ConstructError, Node, Each, Oracle, Operator,
    compile, run,
)


# ═══════════════════════════════════════════════════════════════════════════
# Construct assembly-time validation
# ═══════════════════════════════════════════════════════════════════════════

class TestConstructValidation:
    """Input/output compatibility is checked at Construct assembly time."""

    def test_valid_chain_assembles_when_types_match(self):
        """A correctly typed chain assembles without error."""
        a = _producer("a", RawText)
        b = _consumer("b", RawText, Claims)
        c = _consumer("c", Claims, ClassifiedClaims)
        pipeline = Construct("good", nodes=[a, b, c])
        assert len(pipeline.nodes) == 3
        assert [n.name for n in pipeline.nodes] == ["a", "b", "c"]

    def test_mismatch_raises_when_no_compatible_upstream(self):
        """Downstream input with no compatible upstream raises ConstructError
        AND the error message lists the upstream producers."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad", nodes=[a, b])
        msg = str(exc_info.value)
        assert "declares inputs=Claims" in msg
        assert "node 'a': RawText" in msg

    def test_hint_suggests_map_when_upstream_has_list_field(self):
        """When upstream has list[input_type] field, hint names the correct path."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult)
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-fanout", nodes=[a, b])
        msg = str(exc_info.value)
        assert "did you forget to fan out" in msg
        assert "s.a.groups" in msg

    def test_error_includes_source_location_when_mismatch(self):
        """Error message includes a file:line pointer to the user call site."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ConstructError, match=r"at test_validation\.py:\d+"):
            Construct("bad-loc", nodes=[a, b])

    def test_each_assembles_when_path_resolves_to_list(self):
        """Each whose path resolves to list[input_type] assembles AND attaches the modifier."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult).map(
            lambda s: s.a.groups, key="label"
        )
        pipeline = Construct("good-each", nodes=[a, b])
        assert len(pipeline.nodes) == 2
        each = pipeline.nodes[1].get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "a.groups"
        assert each.key == "label"

    def test_each_raises_when_field_missing(self):
        """Each path that walks to a non-existent field raises."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.nonexistent", key="label"
        )
        with pytest.raises(ConstructError, match="has no field 'nonexistent'"):
            Construct("bad-each-field", nodes=[a, b])

    def test_each_raises_when_terminal_not_list(self):
        """Each whose terminal field isn't a list is flagged."""
        a = _producer("a", RawText)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.text", key="label"
        )
        with pytest.raises(ConstructError, match="not a list"):
            Construct("bad-each-terminal", nodes=[a, b])

    def test_each_raises_when_list_element_type_wrong(self):
        """Each whose list element type doesn't match input raises."""
        a = _producer("a", Claims)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.items", key="label"
        )
        with pytest.raises(ConstructError, match=r"list\[str\]"):
            Construct("bad-each-element", nodes=[a, b])

    def test_first_item_deferred_when_has_input(self):
        """First-of-chain with declared input is NOT flagged -- runtime-seeded."""
        b = _consumer("b", Claims, ClassifiedClaims)
        pipeline = Construct("top-level", nodes=[b])
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].inputs is Claims

    def test_top_level_each_deferred_when_root_unknown(self):
        """Each at position 0 whose root isn't a known producer defers cleanly."""
        process = _consumer("process", ClusterGroup, MatchResult) | Each(
            over="seeded_from_runtime.groups", key="label"
        )
        pipeline = Construct("top-each", nodes=[process])
        assert len(pipeline.nodes) == 1
        each = pipeline.nodes[0].get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "seeded_from_runtime.groups"

    def test_sub_construct_input_port_satisfies_inner_node(self):
        """Inner node reading from the sub-construct's input port validates."""
        inner = _consumer("inner", Claims, Claims)
        sub = Construct("sub", input=Claims, output=Claims, nodes=[inner])
        assert sub.input is Claims
        assert sub.output is Claims
        assert len(sub.nodes) == 1

    def test_sub_construct_validates_when_chained_in_parent(self):
        """Parent producing sub.input satisfies the sub-construct's input check."""
        upstream = _producer("upstream", Claims)
        sub = Construct(
            "sub", input=Claims, output=ClassifiedClaims,
            nodes=[_consumer("inner", Claims, ClassifiedClaims)],
        )
        parent = Construct("parent", nodes=[upstream, sub])
        assert len(parent.nodes) == 2
        assert parent.nodes[1].input is Claims

    def test_sub_construct_raises_when_parent_type_incompatible(self):
        """Parent's upstream output incompatible with sub.input raises with
        a tight error pinning BOTH the construct name and the clause."""
        upstream = _producer("upstream", RawText)
        sub = Construct(
            "sub", input=Claims, output=ClassifiedClaims,
            nodes=[_consumer("inner", Claims, ClassifiedClaims)],
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("parent", nodes=[upstream, sub])
        msg = str(exc_info.value)
        assert "'sub' in construct 'parent'" in msg
        assert "declares input=Claims" in msg

    def test_construct_error_is_valueerror(self):
        """ConstructError subclasses ValueError for existing except clauses."""
        a = _producer("a", RawText)
        b = _consumer("b", Claims, ClassifiedClaims)
        with pytest.raises(ValueError):
            Construct("bad", nodes=[a, b])

    def test_dict_input_skipped_when_multi_field(self):
        """Nodes with dict[str, type] input spec aren't statically validated."""
        step_a = _producer("step-a", Claims)
        step_b = _producer("step-b", RawText)
        step_c = Node.scripted(
            "step-c", fn="f",
            inputs={"step_a": Claims, "step_b": RawText},
            outputs=RawText,
        )
        pipeline = Construct("multi-input", nodes=[step_a, step_b, step_c])
        assert len(pipeline.nodes) == 3
        assert isinstance(pipeline.nodes[2].inputs, dict)

    def test_dict_class_input_deferred_when_raw_dict(self):
        """input=dict (raw class) defers to runtime isinstance scan."""
        a = _producer("a", RawText)
        b = Node.scripted("b", fn="f", inputs=dict, outputs=Claims)
        pipeline = Construct("dict-class", nodes=[a, b])
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[1].inputs is dict

    def test_dict_generic_input_deferred_when_parameterized(self):
        """input=dict[str, X] (parameterized generic) defers to runtime."""
        a = _producer("a", RawText)
        b = Node.scripted("b", fn="f", inputs=dict[str, Claims], outputs=Claims)
        pipeline = Construct("dict-generic", nodes=[a, b])
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[1].inputs == dict[str, Claims]

    def test_each_downstream_rejected_when_raw_input(self):
        """Consumer declaring raw input=X after an Each-modified producer
        that emits dict[str, X] must be rejected at assembly time."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = _consumer("summarize", MatchResult, MergedResult)
        with pytest.raises(ConstructError, match=r"dict\[str, MatchResult\]"):
            Construct("bad", nodes=[make, verify, summarize])

    def test_each_downstream_accepted_when_dict_input(self):
        """Consumer with input=dict (raw class) after Each-modified producer passes."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = Node.scripted("summarize", fn="f", inputs=dict, outputs=MergedResult)
        pipeline = Construct("good-dict", nodes=[make, verify, summarize])
        assert len(pipeline.nodes) == 3

    def test_each_downstream_accepted_when_typed_dict_input(self):
        """Consumer with input=dict[str, X] matching Each output passes."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = Node.scripted(
            "summarize", fn="f",
            inputs=dict[str, MatchResult], outputs=MergedResult,
        )
        pipeline = Construct("good-typed-dict", nodes=[make, verify, summarize])
        assert len(pipeline.nodes) == 3

    def test_each_downstream_rejected_when_wrong_element_type(self):
        """Consumer with input=dict[str, WrongType] after Each is rejected."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = Node.scripted(
            "summarize", fn="f",
            inputs=dict[str, ValidationResult], outputs=MergedResult,
        )
        with pytest.raises(ConstructError):
            Construct("bad-element", nodes=[make, verify, summarize])

    def test_each_hint_mentions_dict_when_raw_consumer(self):
        """Error for raw-type consumer after Each mentions 'via Each'
        and suggests using dict input."""
        make = _producer("make", Clusters)
        verify = _consumer("verify", ClusterGroup, MatchResult).map(
            lambda s: s.make.groups, key="label"
        )
        summarize = _consumer("summarize", MatchResult, MergedResult)
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-hint", nodes=[make, verify, summarize])
        msg = str(exc_info.value)
        assert "via Each" in msg
        assert "dict" in msg


# ═══════════════════════════════════════════════════════════════════════════
# Construct | Oracle error paths
# ═══════════════════════════════════════════════════════════════════════════

class TestConstructOracleErrors:
    """Error paths for Construct | Oracle."""

    def test_unregistered_merge_fn_raises_when_compiled(self):
        """Construct | Oracle with unregistered merge_fn raises at compile."""
        from neograph.factory import register_scripted

        register_scripted("gen_err", lambda input_data, config: Claims(items=[]))

        sub = Construct(
            "bad-oracle-sub",
            input=Claims,
            output=Claims,
            nodes=[Node.scripted("g", fn="gen_err", outputs=Claims)],
        ) | Oracle(n=2, merge_fn="nonexistent_merge_fn")

        parent = Construct("parent", nodes=[sub])

        with pytest.raises(ValueError, match="not registered"):
            compile(parent)


# ═══════════════════════════════════════════════════════════════════════════
# @node fan-in validation
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecoratorFanInValidation:
    """@node fan-in: ALL parameter types must match their upstream's output."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types
        return _types.ModuleType(name)

    def test_mismatch_raises_when_second_param_wrong(self):
        """3-way fan-in where param 2's annotation doesn't match upstream output."""
        from neograph import ConstructError, construct_from_module, node

        mod = self._fresh_module("test_fan_in_mismatch_2nd")

        @node(mode="scripted", outputs=RawText)
        def alpha() -> RawText:
            return RawText(text="a")

        @node(mode="scripted", outputs=RawText)
        def beta() -> RawText:
            return RawText(text="wrong-type")

        @node(mode="scripted", outputs=ClassifiedClaims)
        def gamma() -> ClassifiedClaims:
            return ClassifiedClaims(classified=[])

        @node(mode="scripted", outputs=MergedResult)
        def report(alpha: RawText, beta: Claims, gamma: ClassifiedClaims) -> MergedResult:
            return MergedResult(final_text="unreachable")

        mod.alpha = alpha
        mod.beta = beta
        mod.gamma = gamma
        mod.report = report

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "beta" in msg
        assert "report" in msg
        assert "Claims" in msg
        assert "RawText" in msg

    def test_mismatch_raises_when_last_param_wrong(self):
        """4-way fan-in where the LAST param mismatches -- proves we check ALL params."""
        from neograph import ConstructError, construct_from_module, node

        mod = self._fresh_module("test_fan_in_mismatch_last")

        @node(mode="scripted", outputs=RawText)
        def a_src() -> RawText:
            return RawText(text="a")

        @node(mode="scripted", outputs=Claims)
        def b_src() -> Claims:
            return Claims(items=["b"])

        @node(mode="scripted", outputs=ClassifiedClaims)
        def c_src() -> ClassifiedClaims:
            return ClassifiedClaims(classified=[])

        @node(mode="scripted", outputs=RawText)
        def d_src() -> RawText:
            return RawText(text="wrong")

        @node(mode="scripted", outputs=MergedResult)
        def sink(a_src: RawText, b_src: Claims, c_src: ClassifiedClaims, d_src: Clusters) -> MergedResult:
            return MergedResult(final_text="unreachable")

        mod.a_src = a_src
        mod.b_src = b_src
        mod.c_src = c_src
        mod.d_src = d_src
        mod.sink = sink

        with pytest.raises(ConstructError) as exc_info:
            construct_from_module(mod)
        msg = str(exc_info.value)
        assert "d_src" in msg or "d-src" in msg
        assert "sink" in msg
        assert "Clusters" in msg
        assert "RawText" in msg

    def test_fan_in_passes_when_all_types_match(self):
        """3-way fan-in with correct types -- no error (regression guard)."""
        from neograph import construct_from_module, node

        mod = self._fresh_module("test_fan_in_match")

        @node(mode="scripted", outputs=RawText)
        def alpha() -> RawText:
            return RawText(text="a")

        @node(mode="scripted", outputs=Claims)
        def beta() -> Claims:
            return Claims(items=["b"])

        @node(mode="scripted", outputs=ClassifiedClaims)
        def gamma() -> ClassifiedClaims:
            return ClassifiedClaims(classified=[])

        @node(mode="scripted", outputs=MergedResult)
        def report(alpha: RawText, beta: Claims, gamma: ClassifiedClaims) -> MergedResult:
            return MergedResult(final_text="ok")

        mod.alpha = alpha
        mod.beta = beta
        mod.gamma = gamma
        mod.report = report

        pipeline = construct_from_module(mod)
        assert [n.name for n in pipeline.nodes][-1] == "report"

    def test_unannotated_param_skipped_when_no_type_hint(self):
        """A param with no annotation is skipped (not flagged as mismatch)."""
        from neograph import construct_from_module, node

        mod = self._fresh_module("test_fan_in_unannotated")

        @node(mode="scripted", outputs=RawText)
        def alpha() -> RawText:
            return RawText(text="a")

        @node(mode="scripted", outputs=Claims)
        def beta() -> Claims:
            return Claims(items=["b"])

        @node(mode="scripted", outputs=MergedResult)
        def report(alpha: RawText, beta) -> MergedResult:
            return MergedResult(final_text="ok")

        mod.alpha = alpha
        mod.beta = beta
        mod.report = report

        pipeline = construct_from_module(mod)
        assert [n.name for n in pipeline.nodes][-1] == "report"


# ═══════════════════════════════════════════════════════════════════════════
# @node fan-in + Each interop
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeDecoratorFanInEachInterop:
    """Regression -- _validate_fan_in_types must unwrap Each output as dict[key, item]."""

    def test_each_result_consumed_as_dict_when_typed(self):
        """Downstream @node parameter `dict[str, UpstreamOut]` is compatible
        with an Each-modified upstream producing UpstreamOut per item."""
        from neograph import compile, construct_from_functions, node, run

        @node(outputs=Clusters)
        def fie_source() -> Clusters:
            return Clusters(
                groups=[
                    ClusterGroup(label="alpha", claim_ids=["c1"]),
                    ClusterGroup(label="beta", claim_ids=["c2"]),
                ]
            )

        @node(
            outputs=MatchResult,
            map_over="fie_source.groups",
            map_key="label",
        )
        def fie_verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(
                cluster_label=cluster.label,
                matched=[f"m-{cluster.label}"],
            )

        @node(outputs=ClassifiedClaims)
        def fie_summarize(fie_verify: dict[str, MatchResult]) -> ClassifiedClaims:
            return ClassifiedClaims(
                classified=[
                    {"claim": label, "category": result.cluster_label}
                    for label, result in fie_verify.items()
                ]
            )

        pipeline = construct_from_functions(
            "fie-pipeline", [fie_source, fie_verify, fie_summarize]
        )
        assert [n.name for n in pipeline.nodes] == [
            "fie-source",
            "fie-verify",
            "fie-summarize",
        ]

        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fie-001"})

        assert isinstance(result["fie_summarize"], ClassifiedClaims)
        categories = {c["category"] for c in result["fie_summarize"].classified}
        assert categories == {"alpha", "beta"}

    def test_each_result_consumed_as_raw_dict_when_unparameterized(self):
        """Downstream parameter typed as plain `dict` (unparameterized) is
        also compatible with an Each-modified upstream."""
        from neograph import construct_from_functions, node

        @node(outputs=Clusters)
        def fier_source() -> Clusters:
            return Clusters(
                groups=[ClusterGroup(label="alpha", claim_ids=["c1"])]
            )

        @node(
            outputs=MatchResult,
            map_over="fier_source.groups",
            map_key="label",
        )
        def fier_verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=["m"])

        @node(outputs=ClassifiedClaims)
        def fier_summarize(fier_verify: dict) -> ClassifiedClaims:
            return ClassifiedClaims(classified=[])

        construct_from_functions(
            "fier-pipeline", [fier_source, fier_verify, fier_summarize]
        )

    def test_raw_type_rejected_when_upstream_has_each(self):
        """If a downstream param is annotated as the raw upstream output type
        (NOT wrapped in dict), it must be rejected when the upstream has Each."""
        from neograph import ConstructError, construct_from_functions, node

        @node(outputs=Clusters)
        def fieraw_source() -> Clusters:
            return Clusters(
                groups=[ClusterGroup(label="alpha", claim_ids=["c1"])]
            )

        @node(
            outputs=MatchResult,
            map_over="fieraw_source.groups",
            map_key="label",
        )
        def fieraw_verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=["m"])

        @node(outputs=ClassifiedClaims)
        def fieraw_summarize(fieraw_verify: MatchResult) -> ClassifiedClaims:
            return ClassifiedClaims(classified=[])

        with pytest.raises(ConstructError) as exc_info:
            construct_from_functions(
                "fieraw-pipeline",
                [fieraw_source, fieraw_verify, fieraw_summarize],
            )
        msg = str(exc_info.value)
        assert "fieraw_verify" in msg
        assert "dict[str, MatchResult]" in msg or "dict" in msg

    def test_wrong_dict_element_rejected_when_each_upstream(self):
        """Downstream annotated `dict[str, WrongType]` against an Each upstream
        producing `RightType` must still be rejected."""
        from neograph import ConstructError, construct_from_functions, node

        @node(outputs=Clusters)
        def fiew_source() -> Clusters:
            return Clusters(
                groups=[ClusterGroup(label="alpha", claim_ids=["c1"])]
            )

        @node(
            outputs=MatchResult,
            map_over="fiew_source.groups",
            map_key="label",
        )
        def fiew_verify(cluster: ClusterGroup) -> MatchResult:
            return MatchResult(cluster_label=cluster.label, matched=["m"])

        @node(outputs=ClassifiedClaims)
        def fiew_summarize(fiew_verify: dict[str, RawText]) -> ClassifiedClaims:
            return ClassifiedClaims(classified=[])

        with pytest.raises(ConstructError) as exc_info:
            construct_from_functions(
                "fiew-pipeline",
                [fiew_source, fiew_verify, fiew_summarize],
            )
        assert "fiew_verify" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════
# effective_producer_type helper
# ═══════════════════════════════════════════════════════════════════════════

class TestEffectiveProducerType:
    """Single source of truth for 'what type does this producer write to
    the state bus, accounting for modifiers'."""

    def test_plain_node_returns_declared_output(self):
        """A node without any modifier has its raw output type."""
        from neograph._construct_validation import effective_producer_type

        n = Node.scripted("plain", fn="_x_plain", outputs=Claims)
        assert effective_producer_type(n) is Claims

    def test_each_wraps_as_dict_when_modifier_present(self):
        """A node with an Each modifier writes dict[str, output] to the state bus."""
        from neograph._construct_validation import effective_producer_type

        n = Node.scripted(
            "each-node", fn="_x_each", inputs=ClusterGroup, outputs=MatchResult
        ) | Each(over="upstream.items", key="label")
        effective = effective_producer_type(n)
        assert effective == dict[str, MatchResult]

    def test_oracle_keeps_raw_output(self):
        """Oracle merges N variants into ONE output. Effective type is unchanged."""
        from neograph._construct_validation import effective_producer_type

        n = Node.scripted("ens", fn="_x_ens", outputs=Claims) | Oracle(
            n=3, merge_fn="nonexistent_ok_for_this_test"
        )
        assert effective_producer_type(n) is Claims

    def test_operator_keeps_raw_output(self):
        """Operator is an interrupt modifier -- it doesn't reshape state."""
        from neograph._construct_validation import effective_producer_type

        n = Node.scripted("op", fn="_x_op", outputs=Claims) | Operator(
            when="_nonexistent_for_helper_test"
        )
        assert effective_producer_type(n) is Claims

    def test_sub_construct_each_wraps_as_dict(self):
        """An Each modifier on a sub-Construct wraps its output the same way."""
        from neograph._construct_validation import effective_producer_type

        inner = Node.scripted(
            "inner", fn="_x_inner", inputs=Claims, outputs=Claims
        )
        sub = Construct(
            "sub", input=Claims, output=Claims, nodes=[inner]
        ) | Each(over="upstream.items", key="label")
        assert effective_producer_type(sub) == dict[str, Claims]

    def test_none_output_returns_none(self):
        """A node with no declared output returns None -- the helper is total."""
        from neograph._construct_validation import effective_producer_type

        class OutputlessStub:
            output = None
            modifiers: list = []

            def has_modifier(self, _):
                return False

            def get_modifier(self, _):
                return None

        assert effective_producer_type(OutputlessStub()) is None


# ═══════════════════════════════════════════════════════════════════════════
# Fan-in validation (dict-form inputs)
# ═══════════════════════════════════════════════════════════════════════════

class TestFanInValidation:
    """Fan-in dict-form inputs validation against upstream producers."""

    def test_matching_upstreams_pass_when_all_types_correct(self):
        """Consumer with inputs={'a': A, 'b': B, 'c': C} validates against
        three upstream producers by name."""
        a = _producer("a", Claims)
        b = _producer("b", RawText)
        c = _producer("c", ClusterGroup)
        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"a": Claims, "b": RawText, "c": ClusterGroup},
            outputs=MatchResult,
        )
        pipeline = Construct("fan-in-ok", nodes=[a, b, c, consumer])
        assert len(pipeline.nodes) == 4
        resolved = pipeline.nodes[3]
        assert resolved.inputs == {"a": Claims, "b": RawText, "c": ClusterGroup}
        assert resolved.outputs is MatchResult

    def test_unknown_upstream_rejected_when_name_missing(self):
        """Consumer declaring inputs['nonexistent'] raises ConstructError."""
        a = _producer("a", Claims)
        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"a": Claims, "nonexistent": RawText},
            outputs=MatchResult,
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-name", nodes=[a, consumer])
        msg = str(exc_info.value)
        assert "'nonexistent'" in msg
        assert "no upstream node" in msg

    def test_type_mismatch_rejected_when_upstream_produces_different(self):
        """Consumer with matching upstream name but wrong type raises ConstructError."""
        a = _producer("a", Claims)
        b = _producer("b", RawText)
        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"a": Claims, "b": Claims},  # b produces RawText, not Claims
            outputs=MatchResult,
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-type", nodes=[a, b, consumer])
        msg = str(exc_info.value)
        assert "'b'" in msg
        assert "Claims" in msg
        assert "RawText" in msg

    def test_programmatic_each_with_dict_inputs_passes_when_fan_out_key(self):
        """Programmatic Node(inputs={...}) | Each(...) must NOT reject the
        fan-out key as an unknown upstream (neograph-ts7)."""
        make = _producer("make", Clusters)
        canonicalize = Node.scripted(
            "canonicalize", fn="f",
            inputs={"group": ClusterGroup},
            outputs=MatchResult,
        ) | Each(over="make.groups", key="label")
        pipeline = Construct("ts7", nodes=[make, canonicalize])
        assert len(pipeline.nodes) == 2
        each = pipeline.nodes[1].get_modifier(Each)
        assert each is not None

    def test_mixed_upstream_and_fan_out_when_programmatic(self):
        """Programmatic fan-in + fan-out: upstream keys validate, fan-out
        key is skipped (neograph-ts7)."""
        a = _producer("a", RawText)
        make = _producer("make", Clusters)
        process = Node.scripted(
            "process", fn="f",
            inputs={"a": RawText, "group": ClusterGroup},
            outputs=MatchResult,
        ) | Each(over="make.groups", key="label")
        pipeline = Construct("ts7-mixed", nodes=[a, make, process])
        assert len(pipeline.nodes) == 3

    def test_empty_dict_inputs_assembles_when_no_upstreams_declared(self):
        """inputs={} (empty dict) is treated as 'no upstream needed'."""
        a = _producer("a", Claims)
        seed = Node.scripted("seed", fn="f", inputs={}, outputs=RawText)
        pipeline = Construct("empty-inputs", nodes=[a, seed])
        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[1].inputs == {}


# ═══════════════════════════════════════════════════════════════════════════
# list[X] <-> dict[str, X] compatibility rule
# ═══════════════════════════════════════════════════════════════════════════

class TestTypesCompatibleListOverDict:
    """The list[X] / dict[str, X] compatibility rule."""

    def test_list_consumer_accepts_when_dict_producer_element_matches(self):
        """dict[str, MatchResult] producer satisfies list[MatchResult] consumer."""
        from neograph._construct_validation import _types_compatible

        assert _types_compatible(dict[str, MatchResult], list[MatchResult]) is True

    def test_list_consumer_rejects_when_dict_producer_element_wrong(self):
        """dict[str, MatchResult] producer does NOT satisfy list[Claims]."""
        from neograph._construct_validation import _types_compatible

        assert _types_compatible(dict[str, MatchResult], list[Claims]) is False

    def test_list_consumer_accepts_when_dict_producer_element_is_subclass(self):
        """dict[str, MatchResult] producer satisfies list[BaseModel] -- subclass ok."""
        from neograph._construct_validation import _types_compatible

        assert _types_compatible(dict[str, MatchResult], list[BaseModel]) is True

    def test_non_dict_producer_rejected_when_list_consumer(self):
        """A plain-class (non-dict) producer does NOT match list[X] via this rule."""
        from neograph._construct_validation import _types_compatible

        assert _types_compatible(Claims, list[Claims]) is False


# ═══════════════════════════════════════════════════════════════════════════
# Dict-form outputs validation
# ═══════════════════════════════════════════════════════════════════════════

class TestDictOutputsValidator:
    """Validator resolves dict-form outputs as per-key producers."""

    def test_downstream_references_dict_output_key(self):
        """A downstream node can consume a specific output key via inputs dict."""
        upstream = Node("explore", outputs={"result": RawText, "meta": Claims})
        downstream = Node("score", inputs={"explore_result": RawText}, outputs=ClassifiedClaims)
        Construct("p", nodes=[upstream, downstream])

    def test_unknown_output_key_raises_when_referenced(self):
        """Referencing a non-existent output key raises ConstructError."""
        upstream = Node("explore", outputs={"result": RawText})
        downstream = Node("score", inputs={"explore_bogus": RawText}, outputs=ClassifiedClaims)
        with pytest.raises(ConstructError):
            Construct("p", nodes=[upstream, downstream])

    def test_single_type_outputs_still_validates(self):
        """Single-type outputs still produce {node_name} as upstream name."""
        upstream = Node("extract", outputs=RawText)
        downstream = Node("score", inputs={"extract": RawText}, outputs=ClassifiedClaims)
        Construct("p", nodes=[upstream, downstream])
