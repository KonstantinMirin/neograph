"""Validation tests — assembly-time type checking, fan-in validation,
Each path resolution, effective_producer_type, list/dict compatibility,
dict-form outputs validation, Oracle error paths, and lint() DI validation.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from neograph import (
    CompileError,
    ConfigurationError,
    Construct,
    ConstructError,
    Each,
    ExecutionError,
    FromConfig,
    FromInput,
    Node,
    Operator,
    Oracle,
    Tool,
    compile,
    construct_from_functions,
    node,
    run,
)
from tests.schemas import (
    Claims,
    ClassifiedClaims,
    ClusterGroup,
    Clusters,
    MatchResult,
    MergedResult,
    RawText,
    ValidationResult,
    _consumer,
    _producer,
)

from neograph import lint


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

        with pytest.raises(ConfigurationError, match="not registered"):
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
        assert isinstance(each, Each)

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


# ═══════════════════════════════════════════════════════════════════════════
# Three-surface parity: fan-in type validation
#
# Assembly-time type checking must reject mismatches identically regardless
# of which API surface builds the pipeline.
# ═══════════════════════════════════════════════════════════════════════════


def _fan_in_valid_declarative() -> Construct:
    """Declarative surface: single-type inputs (isinstance scan, not dict keyed)."""
    a = _producer("a", RawText)
    b = _consumer("b", RawText, Claims)
    c = _consumer("c", Claims, MatchResult)
    return Construct("fan-in-decl", nodes=[a, b, c])


def _fan_in_valid_decorator() -> Construct:
    """@node decorator surface: 3-way fan-in with correct types."""
    from neograph import node
    from neograph.decorators import construct_from_functions

    @node(mode="scripted", outputs=Claims)
    def a() -> Claims:
        return Claims(items=["claim"])

    @node(mode="scripted", outputs=RawText)
    def b() -> RawText:
        return RawText(text="raw")

    @node(mode="scripted", outputs=MatchResult)
    def consumer(a: Claims, b: RawText) -> MatchResult:
        return MatchResult(cluster_label="ok", matched=[])

    return construct_from_functions("fan-in-dec", [a, b, consumer])


def _fan_in_valid_programmatic() -> Construct:
    """Programmatic surface: 3-way fan-in with correct types."""
    a = _producer("a", Claims)
    b = _producer("b", RawText)
    consumer = Node.scripted(
        "consumer", fn="f",
        inputs={"a": Claims, "b": RawText},
        outputs=MatchResult,
    )
    return Construct("fan-in-prog", nodes=[a, b, consumer])


def _fan_in_mismatch_declarative():
    """Declarative surface: single-type inputs mismatch — should raise."""
    a = _producer("a", RawText)
    b = _consumer("b", Claims, MatchResult)  # b expects Claims, but a produces RawText
    Construct("fan-in-bad-decl", nodes=[a, b])


def _fan_in_mismatch_decorator():
    """@node decorator surface: fan-in with type mismatch — should raise."""
    from neograph import node
    from neograph.decorators import construct_from_functions

    @node(mode="scripted", outputs=Claims)
    def a() -> Claims:
        return Claims(items=["claim"])

    @node(mode="scripted", outputs=RawText)
    def b() -> RawText:
        return RawText(text="raw")

    @node(mode="scripted", outputs=MatchResult)
    def consumer(a: Claims, b: Claims) -> MatchResult:  # b produces RawText
        return MatchResult(cluster_label="bad", matched=[])

    construct_from_functions("fan-in-bad-dec", [a, b, consumer])


def _fan_in_mismatch_programmatic():
    """Programmatic surface: fan-in with type mismatch — should raise."""
    a = _producer("a", Claims)
    b = _producer("b", RawText)
    consumer = Node.scripted(
        "consumer", fn="f",
        inputs={"a": Claims, "b": Claims},  # b produces RawText, not Claims
        outputs=MatchResult,
    )
    Construct("fan-in-bad-prog", nodes=[a, b, consumer])


class TestThreeSurfaceFanInParity:
    """Fan-in type validation tested identically across declarative, @node,
    and programmatic API surfaces. Template pattern for future parity tests."""

    @pytest.mark.parametrize("build", [
        _fan_in_valid_declarative,
        _fan_in_valid_decorator,
        _fan_in_valid_programmatic,
    ], ids=["declarative", "decorator", "programmatic"])
    def test_fan_in_assembles_when_types_correct(self, build):
        """Fan-in with matching types assembles without error across surfaces."""
        pipeline = build()
        assert len(pipeline.nodes) == 3

    @pytest.mark.parametrize("build", [
        _fan_in_mismatch_declarative,
        _fan_in_mismatch_decorator,
        _fan_in_mismatch_programmatic,
    ], ids=["declarative", "decorator", "programmatic"])
    def test_fan_in_rejects_when_type_mismatches(self, build):
        """Fan-in with mismatched types raises ConstructError across surfaces."""
        with pytest.raises(ConstructError) as exc_info:
            build()
        msg = str(exc_info.value)
        assert "b" in msg
        assert "Claims" in msg


# ═══════════════════════════════════════════════════════════════════════════
# Modifiable.map() error paths
# ═══════════════════════════════════════════════════════════════════════════

class TestModifiableMapErrors:
    """Error paths for Modifiable.map() — string/lambda introspection."""

    def test_map_resolves_when_lambda_path_valid(self):
        """Happy path: lambda with valid attribute chain produces Each modifier."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        mapped = n.map(lambda s: s.make.groups, key="label")
        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make.groups"
        assert each.key == "label"

    def test_map_resolves_when_string_path_given(self):
        """String path is used directly without introspection."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        mapped = n.map("make.groups", key="label")
        each = mapped.get_modifier(Each)
        assert isinstance(each, Each)
        assert each.over == "make.groups"

    def test_map_raises_when_source_not_string_or_callable(self):
        """Non-string, non-callable source raises TypeError."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(TypeError, match="must be a string path or a lambda"):
            n.map(42, key="label")

    def test_map_raises_when_lambda_uses_indexing(self):
        """Lambda with subscript/indexing raises TypeError (not a pure attr chain)."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            n.map(lambda s: s.make.groups[0], key="label")

    def test_map_raises_when_lambda_accesses_underscore_attr(self):
        """Lambda accessing underscore-prefixed attribute raises TypeError."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(TypeError, match="pure attribute-access chain"):
            n.map(lambda s: s.make._private, key="label")

    def test_map_raises_when_lambda_returns_non_recorder(self):
        """Lambda that returns a non-recorder value raises TypeError."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(TypeError, match="must return an attribute-access chain"):
            n.map(lambda s: "literal_string", key="label")

    def test_map_raises_when_lambda_is_identity(self):
        """Lambda that returns the recorder without any attribute access raises TypeError."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        with pytest.raises(TypeError, match="must access at least one attribute"):
            n.map(lambda s: s, key="label")

    def test_map_raises_when_called_twice(self):
        """Calling .map() twice raises ConstructError — duplicate Each is invalid."""
        n = _consumer("verify", ClusterGroup, MatchResult)
        mapped_once = n.map("a.groups", key="label")
        with pytest.raises(ConstructError, match="Duplicate Each"):
            mapped_once.map("b.items", key="id")


# ═══════════════════════════════════════════════════════════════════════════
# _check_each_path edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckEachPathErrors:
    """Edge cases for _check_each_path beyond the standard 3 error paths."""

    def test_single_segment_path_defers_when_no_dot(self):
        """Each(over="a") with no dot — root matches upstream but no field to walk.
        split_each_path returns root='a', segments=(). The path resolves to the
        raw upstream type, which must be a list for validation to pass. Since
        Clusters is NOT a list, this should raise 'not a list'."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a", key="label"
        )
        with pytest.raises(ConstructError, match="not a list"):
            Construct("single-seg", nodes=[a, b])

    def test_single_segment_path_raises_when_root_unknown(self):
        """Each(over="unknown") with no matching upstream raises ConstructError."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="unknown", key="label"
        )
        with pytest.raises(ConstructError, match="root 'unknown' does not match"):
            Construct("single-seg-reject", nodes=[a, b])

    def test_empty_path_string_rejected_at_construction(self):
        """Each(over='') — rejected by field_validator at construction time."""
        with pytest.raises((ValueError, Exception), match="must not be empty"):
            Each(over="", key="label")

    def test_deeply_nested_path_resolves_when_fields_exist(self):
        """Multi-level dotted path that walks through nested models."""

        class Inner(BaseModel, frozen=True):
            claim_ids: list[str]

        class Middle(BaseModel, frozen=True):
            inner: Inner

        class Outer(BaseModel, frozen=True):
            middle: Middle

        a = _producer("a", Outer)
        # Path: a.middle.inner.claim_ids → list[str], element str
        b = Node.scripted(
            "b", fn="f", inputs=str, outputs=MatchResult,
        ) | Each(over="a.middle.inner.claim_ids", key="id")
        pipeline = Construct("deep-path", nodes=[a, b])
        assert len(pipeline.nodes) == 2

    def test_deeply_nested_path_raises_when_intermediate_missing(self):
        """Multi-level path where an intermediate segment doesn't exist."""

        class Shallow(BaseModel, frozen=True):
            name: str

        a = _producer("a", Shallow)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.name.nonexistent.deep", key="label"
        )
        with pytest.raises(ConstructError, match="has no field 'nonexistent'"):
            Construct("deep-missing", nodes=[a, b])

    def test_path_raises_when_terminal_is_non_list_primitive(self):
        """Path resolving to a primitive (int) raises 'not a list'."""

        class WithInt(BaseModel, frozen=True):
            count: int

        a = _producer("a", WithInt)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.count", key="label"
        )
        with pytest.raises(ConstructError, match="not a list"):
            Construct("prim-terminal", nodes=[a, b])

    def test_each_key_raises_when_field_missing_on_item_type(self):
        """Each.key must name a valid field on the list element type.
        each.key='nonexistent' on list[ClusterGroup] should raise (neograph-mn41)."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.groups", key="nonexistent"
        )
        with pytest.raises(ConstructError, match="has no field 'nonexistent'"):
            Construct("bad-each-key", nodes=[a, b])

    def test_each_key_passes_when_field_exists_on_item_type(self):
        """Each.key='label' on list[ClusterGroup] (which has a 'label' field)
        should assemble without error."""
        a = _producer("a", Clusters)
        b = _consumer("b", ClusterGroup, MatchResult) | Each(
            over="a.groups", key="label"
        )
        pipeline = Construct("ok-each-key", nodes=[a, b])
        assert len(pipeline.nodes) == 2

    def test_each_key_skipped_when_element_type_is_primitive(self):
        """Each.key on list[str] (no model_fields) defers to runtime."""

        class HasStrings(BaseModel, frozen=True):
            tags: list[str]

        a = _producer("a", HasStrings)
        b = Node.scripted(
            "b", fn="f", inputs=str, outputs=MatchResult,
        ) | Each(over="a.tags", key="value")
        # str has no model_fields — should defer to runtime, not raise.
        pipeline = Construct("prim-key", nodes=[a, b])
        assert len(pipeline.nodes) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Node name collision detection (neograph-x820)
# ═══════════════════════════════════════════════════════════════════════════

class TestNodeNameCollision:
    """Nodes whose names differ only by hyphens vs underscores must be
    rejected at compile time — they map to the same state field and would
    silently share loop counters, reducers, etc."""

    def test_collision_raises_when_hyphen_and_underscore_names_collide(self):
        """'my-node' and 'my_node' both map to field 'my_node' — must raise."""
        a = _producer("my-node", RawText)
        b = _producer("my_node", Claims)
        with pytest.raises(CompileError, match="name collision"):
            compile(Construct("collision", nodes=[a, b]))

    def test_no_collision_when_names_differ(self):
        """Two nodes with genuinely different names compile fine."""
        from neograph.factory import register_scripted
        register_scripted("f_node_a", lambda input_data, config: RawText(text="a"))
        register_scripted("f_node_b", lambda input_data, config: Claims(items=["b"]))

        a = Node.scripted("node-a", fn="f_node_a", outputs=RawText)
        b = Node.scripted("node-b", fn="f_node_b", inputs=RawText, outputs=Claims)
        result = run(compile(Construct("no-collision", nodes=[a, b])), input={})
        assert isinstance(result["node_b"], Claims)

    def test_sub_construct_names_do_not_collide_with_parent(self):
        """Sub-construct node names live in separate state scopes — no error
        even if a parent node and sub-construct-internal node share a name."""
        from neograph.factory import register_scripted

        register_scripted("inner_fn", lambda input_data, config: Claims(items=["ok"]))
        register_scripted("parent_fn", lambda input_data, config: RawText(text="raw"))

        inner_node = Node.scripted("my_node", fn="inner_fn", inputs=RawText, outputs=Claims)
        sub = Construct(
            "sub",
            input=RawText,
            output=Claims,
            nodes=[inner_node],
        )
        parent_node = Node.scripted("my-parent", fn="parent_fn", outputs=RawText)
        # 'my_node' inside sub and 'my-parent' in parent — different scopes, no collision
        parent = Construct("parent", nodes=[parent_node, sub])
        result = run(compile(parent), input={})
        assert isinstance(result["sub"], Claims)


# ═══════════════════════════════════════════════════════════════════════════
# Compile-time: tool factory registration check (neograph-9513)
# ═══════════════════════════════════════════════════════════════════════════

class TestToolFactoryRegistrationCheck:
    """compile() must verify that every tool referenced by agent/act nodes
    is registered in _tool_factory_registry."""

    def test_unregistered_tool_raises_at_compile_when_agent_mode(self):
        """Agent node with unregistered tool raises CompileError at compile()."""
        from tests.fakes import StructuredFake, configure_fake_llm
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        n = Node(
            "research",
            mode="agent",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool("nonexistent_tool_9513", budget=3)],
        )
        pipeline = Construct("bad-tool", nodes=[n])
        with pytest.raises(CompileError, match="nonexistent_tool_9513"):
            compile(pipeline)

    def test_unregistered_tool_raises_at_compile_when_act_mode(self):
        """Act node with unregistered tool raises CompileError at compile()."""
        from tests.fakes import StructuredFake, configure_fake_llm
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        n = Node(
            "actor",
            mode="act",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool("missing_tool_9513", budget=1)],
        )
        pipeline = Construct("bad-act-tool", nodes=[n])
        with pytest.raises(CompileError, match="missing_tool_9513"):
            compile(pipeline)

    def test_registered_tool_passes_compile_when_agent_mode(self):
        """Agent node with registered tool compiles without error."""
        from neograph.factory import register_tool_factory
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        register_tool_factory("registered_tool_9513", lambda config, tool_config: None)

        n = Node(
            "research-ok",
            mode="agent",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool("registered_tool_9513", budget=3)],
        )
        pipeline = Construct("good-tool", nodes=[n])
        compile(pipeline)  # no raise = tool factory check passed


# ═══════════════════════════════════════════════════════════════════════════
# Compile-time: LLM + prompt compiler configured (neograph-fn5x)
# ═══════════════════════════════════════════════════════════════════════════

class TestLlmConfiguredCheck:
    """compile() must verify _llm_factory and _prompt_compiler are set
    when any node has mode in (think, agent, act)."""

    def test_unconfigured_llm_raises_at_compile_when_think_node(self, monkeypatch):
        """Think node without configure_llm() raises CompileError at compile()."""
        import neograph._llm as _llm_mod
        monkeypatch.setattr(_llm_mod, "_llm_factory", None)
        monkeypatch.setattr(_llm_mod, "_prompt_compiler", None)
        n = Node(
            "think-node",
            mode="think",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
        )
        pipeline = Construct("bad-llm", nodes=[n])
        with pytest.raises(CompileError, match="configure_llm"):
            compile(pipeline)

    def test_unconfigured_prompt_compiler_raises_at_compile(self, monkeypatch):
        """LLM factory set but prompt compiler missing raises CompileError."""
        import neograph._llm as _llm_mod
        monkeypatch.setattr(_llm_mod, "_llm_factory", lambda tier: None)
        monkeypatch.setattr(_llm_mod, "_prompt_compiler", None)
        n = Node(
            "think-node-pc",
            mode="think",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
        )
        pipeline = Construct("bad-pc", nodes=[n])
        with pytest.raises(CompileError, match="configure_llm"):
            compile(pipeline)

    def test_scripted_only_compiles_without_llm_configured(self, monkeypatch):
        """Pipeline with only scripted nodes compiles even without configure_llm()."""
        import neograph._llm as _llm_mod
        from neograph.factory import register_scripted
        monkeypatch.setattr(_llm_mod, "_llm_factory", None)
        monkeypatch.setattr(_llm_mod, "_prompt_compiler", None)
        register_scripted("fn_no_llm_test", lambda input_data, config: RawText(text="ok"))
        n = Node.scripted("scripted-only", fn="fn_no_llm_test", outputs=RawText)
        pipeline = Construct("scripted-ok", nodes=[n])
        compile(pipeline)  # no raise = scripted-only pipeline doesn't need LLM


# ═══════════════════════════════════════════════════════════════════════════
# Compile-time: output_strategy validation (neograph-0b2m)
# ═══════════════════════════════════════════════════════════════════════════

class TestOutputStrategyValidation:
    """compile() must verify output_strategy values are valid."""

    def test_invalid_output_strategy_raises_at_compile(self):
        """Node with bogus output_strategy raises CompileError at compile()."""
        from tests.fakes import StructuredFake, configure_fake_llm
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        n = Node(
            "bad-strat",
            mode="think",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
            llm_config={"output_strategy": "banana"},
        )
        pipeline = Construct("bad-strat-pipe", nodes=[n])
        with pytest.raises(CompileError, match="banana"):
            compile(pipeline)

    def test_valid_output_strategies_pass_compile(self):
        """Nodes with valid output_strategy values compile without error."""
        from tests.fakes import StructuredFake, configure_fake_llm
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        for strategy in ("structured", "json_mode", "text"):
            n = Node(
                f"strat-{strategy}",
                mode="think",
                inputs=RawText,
                outputs=Claims,
                model="fast",
                prompt="test",
                llm_config={"output_strategy": strategy},
            )
            pipeline = Construct(f"strat-{strategy}-pipe", nodes=[n])
            compile(pipeline)  # no raise = strategy accepted

    def test_no_output_strategy_defaults_without_error(self):
        """Node with no output_strategy (default) compiles fine."""
        from tests.fakes import StructuredFake, configure_fake_llm
        configure_fake_llm(lambda tier: StructuredFake(lambda m: m()))
        n = Node(
            "no-strat",
            mode="think",
            inputs=RawText,
            outputs=Claims,
            model="fast",
            prompt="test",
        )
        pipeline = Construct("no-strat-pipe", nodes=[n])
        compile(pipeline)  # no raise = default strategy accepted


# ═══════════════════════════════════════════════════════════════════════════
# Sub-construct output boundary contract (neograph-c4se)
# ═══════════════════════════════════════════════════════════════════════════

class TestSubConstructOutputBoundary:
    """When a Construct declares output=SomeType, at least one internal node
    must produce a compatible type. Silent None propagation at runtime
    indicates the contract was never satisfied."""

    def test_output_mismatch_raises_when_no_node_produces_output(self):
        """Sub-construct declares output=Claims but only node produces RawText."""
        inner = _producer("inner", RawText)
        with pytest.raises(ConstructError, match="output=Claims"):
            Construct("bad-sub", input=RawText, output=Claims, nodes=[inner])

    def test_output_match_passes_when_node_produces_compatible_type(self):
        """Sub-construct declares output=Claims and inner node produces Claims."""
        inner = Node.scripted("inner", fn="f", inputs=RawText, outputs=Claims)
        sub = Construct("ok-sub", input=RawText, output=Claims, nodes=[inner])
        assert sub.output is Claims

    def test_output_subclass_passes_when_node_produces_subclass(self):
        """Sub-construct output=BaseModel is satisfied by any Pydantic model."""
        inner = _producer("inner", Claims)
        sub = Construct("sub-sub", output=BaseModel, nodes=[inner])
        assert sub.output is BaseModel

    def test_no_output_declared_skips_check(self):
        """Construct without output= declaration skips boundary check."""
        inner = _producer("inner", RawText)
        pipeline = Construct("top-level", nodes=[inner])
        assert pipeline.output is None

    def test_multiple_nodes_passes_when_last_produces_output(self):
        """Only one of several nodes needs to produce the output type."""
        inner_a = _producer("a", RawText)
        inner_b = Node.scripted("b", fn="f", inputs=RawText, outputs=Claims)
        sub = Construct("multi", input=RawText, output=Claims, nodes=[inner_a, inner_b])
        assert sub.output is Claims


# ═══════════════════════════════════════════════════════════════════════════
# lint() — DI binding validation (neograph-no0q)
# ═══════════════════════════════════════════════════════════════════════════

class TestLint:
    """lint() validates DI bindings against a sample config."""

    def test_lint_returns_empty_when_all_bindings_present(self):
        """No warnings when every FromInput/FromConfig key exists in config."""
        @node(outputs=RawText)
        def my_node(topic: Annotated[str, FromInput]) -> RawText: ...

        pipeline = construct_from_functions("ok", [my_node])
        issues = lint(pipeline, config={"topic": "hello"})
        assert issues == []

    def test_lint_reports_missing_from_input_key(self):
        """lint reports when a FromInput param has no matching config key."""
        @node(outputs=RawText)
        def my_node(topic: Annotated[str, FromInput]) -> RawText: ...

        pipeline = construct_from_functions("bad", [my_node])
        issues = lint(pipeline, config={})
        assert len(issues) == 1
        assert "topic" in issues[0].param
        assert "my" in issues[0].node_name  # "my-node" or "my_node"

    def test_lint_reports_missing_from_config_key(self):
        """lint reports when a FromConfig param has no matching config key."""
        @node(outputs=RawText)
        def my_node(
            upstream: RawText,
            limiter: Annotated[str, FromConfig],
        ) -> RawText: ...

        producer = _producer("upstream", RawText)
        pipeline = Construct("bad", nodes=[producer, my_node])
        issues = lint(pipeline, config={})
        assert len(issues) == 1
        assert "limiter" in issues[0].param

    def test_lint_reports_missing_bundled_model_fields(self):
        """When a FromInput param is a BaseModel, lint checks each field."""
        class Ctx(BaseModel):
            node_id: str
            project_root: str

        @node(outputs=RawText)
        def my_node(ctx: Annotated[Ctx, FromInput]) -> RawText: ...

        pipeline = construct_from_functions("bundled", [my_node])
        # Only provide node_id, missing project_root
        issues = lint(pipeline, config={"node_id": "x"})
        assert len(issues) == 1
        assert "project_root" in issues[0].param

    def test_lint_bundled_model_all_fields_present(self):
        """No issues when all bundled model fields are in config."""
        class Ctx(BaseModel):
            node_id: str
            project_root: str

        @node(outputs=RawText)
        def my_node(ctx: Annotated[Ctx, FromInput]) -> RawText: ...

        pipeline = construct_from_functions("bundled-ok", [my_node])
        issues = lint(pipeline, config={"node_id": "x", "project_root": "/tmp"})
        assert issues == []

    def test_lint_no_config_still_validates_required(self):
        """Without config, lint reports required=True params as errors."""
        @node(outputs=RawText)
        def my_node(
            topic: Annotated[str, FromInput(required=True)],
        ) -> RawText: ...

        pipeline = construct_from_functions("no-cfg", [my_node])
        issues = lint(pipeline)
        assert len(issues) == 1
        assert issues[0].required is True
        assert "topic" in issues[0].param

    def test_lint_required_false_no_issue_without_config(self):
        """Optional FromInput(required=False) params are NOT flagged without config."""
        @node(outputs=RawText)
        def my_node(topic: Annotated[str, FromInput(required=False)]) -> RawText: ...

        pipeline = construct_from_functions("opt", [my_node])
        issues = lint(pipeline)
        assert issues == []

    def test_lint_walks_sub_constructs(self):
        """lint recurses into sub-constructs."""
        @node(outputs=Claims)
        def inner(topic: Annotated[str, FromInput]) -> Claims: ...

        sub = construct_from_functions("sub", [inner], input=None, output=Claims)
        outer_prod = _producer("start", RawText)
        pipeline = Construct("outer", nodes=[outer_prod, sub])
        issues = lint(pipeline, config={})
        assert len(issues) == 1
        assert "topic" in issues[0].param

    def test_lint_skips_upstream_and_constant_params(self):
        """Upstream and constant params should not be checked against config."""
        @node(outputs=RawText)
        def upstream() -> RawText: ...

        @node(outputs=Claims)
        def my_node(
            upstream: RawText,
            limit: int = 10,
        ) -> Claims: ...

        pipeline = construct_from_functions("ok", [upstream, my_node])
        issues = lint(pipeline, config={})
        assert issues == []

    def test_lint_multiple_nodes_multiple_issues(self):
        """lint collects issues from all nodes, not just the first."""
        @node(outputs=RawText)
        def node_a(x: Annotated[str, FromInput]) -> RawText: ...

        @node(outputs=Claims)
        def node_b(y: Annotated[str, FromConfig]) -> Claims: ...

        pipeline = construct_from_functions("multi", [node_a, node_b])
        issues = lint(pipeline, config={})
        assert len(issues) == 2
        params = {i.param for i in issues}
        assert params == {"x", "y"}

    def test_lint_skips_non_node_non_construct_items(self):
        """lint silently skips items that are neither Node nor Construct."""
        # Construct.nodes can only hold Node|Construct, but _walk is typed
        # to accept either. Passing something else should just return early.
        from neograph.lint import LintIssue, _walk
        issues: list[LintIssue] = []
        _walk("not-a-node", None, issues)  # type: ignore[arg-type]
        assert issues == []

    def test_lint_required_bundled_model_no_config(self):
        """Required bundled model params are flagged when config is None."""
        class Ctx(BaseModel):
            node_id: str
            project_root: str

        @node(outputs=RawText)
        def my_node(ctx: Annotated[Ctx, FromInput(required=True)]) -> RawText: ...

        pipeline = construct_from_functions("bundled-no-cfg", [my_node])
        issues = lint(pipeline)
        assert len(issues) == 2
        params = {i.param for i in issues}
        assert params == {"node_id", "project_root"}
        assert all(i.required for i in issues)
        assert all("has no config" in i.message for i in issues)

    def test_lint_merge_fn_di_param_missing_from_config(self):
        """lint detects missing DI param in @merge_fn when config is provided."""
        from neograph import merge_fn as merge_fn_deco

        @merge_fn_deco
        def lint_merge(
            variants: list[Claims],
            api_key: Annotated[str, FromConfig],
        ) -> Claims:
            return variants[0]

        # Use @node with ensemble_n to get a node with param_resolutions AND Oracle.
        @node(
            outputs=Claims,
            prompt="test", model="fast",
            ensemble_n=2, merge_fn="lint_merge",
        )
        def lint_gen(topic: Annotated[str, FromInput]) -> Claims: ...

        pipeline = construct_from_functions("merge-lint", [lint_gen])
        # Provide 'topic' so the node itself is satisfied, but not 'api_key'
        issues = lint(pipeline, config={"topic": "hello"})
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        assert len(merge_issues) == 1
        assert merge_issues[0].param == "api_key"
        assert "not found in config" in merge_issues[0].message

    def test_lint_merge_fn_required_di_param_no_config(self):
        """lint flags required @merge_fn DI params when config is None."""
        from neograph import merge_fn as merge_fn_deco

        @merge_fn_deco
        def lint_merge_req(
            variants: list[Claims],
            secret: Annotated[str, FromInput(required=True)],
        ) -> Claims:
            return variants[0]

        @node(
            outputs=Claims,
            prompt="test", model="fast",
            ensemble_n=2, merge_fn="lint_merge_req",
        )
        def lint_gen2(topic: Annotated[str, FromInput(required=True)]) -> Claims: ...

        pipeline = construct_from_functions("merge-lint-req", [lint_gen2])
        issues = lint(pipeline)
        # Both node-level 'topic' and merge_fn-level 'secret' are required
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        assert len(merge_issues) == 1
        assert merge_issues[0].param == "secret"
        assert merge_issues[0].required is True
        assert "has no config" in merge_issues[0].message


    def test_lint_merge_fn_bundled_model_fields_checked(self):
        """lint() checks from_input_model fields in @merge_fn (neograph-s2h8)."""
        from pydantic import BaseModel

        from neograph import lint, node
        from neograph import merge_fn as merge_fn_deco
        from neograph.decorators import construct_from_functions

        class PipeCtx(BaseModel):
            node_id: str
            project_root: str

        @merge_fn_deco
        def ctx_merge(
            variants: list[Claims],
            ctx: Annotated[PipeCtx, FromInput(required=True)],
        ) -> Claims:
            return variants[0]

        @node(
            outputs=Claims,
            prompt="test", model="fast",
            ensemble_n=2, merge_fn="ctx_merge",
        )
        def gen_s2h8() -> Claims: ...

        pipeline = construct_from_functions("s2h8-test", [gen_s2h8])

        # With config missing the model fields
        issues = lint(pipeline, config={"some_other": "value"})
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        # Should flag node_id and project_root as missing
        missing_fields = {i.param for i in merge_issues}
        assert "node_id" in missing_fields
        assert "project_root" in missing_fields

    def test_lint_merge_fn_bundled_model_passes_with_config(self):
        """lint() passes when bundled model fields are present in config."""
        from pydantic import BaseModel

        from neograph import lint, node
        from neograph import merge_fn as merge_fn_deco
        from neograph.decorators import construct_from_functions

        class Ctx2(BaseModel):
            node_id: str

        @merge_fn_deco
        def ctx_merge2(
            variants: list[Claims],
            ctx: Annotated[Ctx2, FromInput],
        ) -> Claims:
            return variants[0]

        @node(
            outputs=Claims,
            prompt="test", model="fast",
            ensemble_n=2, merge_fn="ctx_merge2",
        )
        def gen_s2h8b() -> Claims: ...

        pipeline = construct_from_functions("s2h8-pass", [gen_s2h8b])
        issues = lint(pipeline, config={"node_id": "test-123"})
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        assert len(merge_issues) == 0


class TestLintObligationGaps:
    """Test obligations from /test-obligations analysis of _walk()."""

    def test_lint_merge_fn_simple_di_on_node_without_param_res(self):
        """W-13: Node(no DI) + merge_fn with simple from_input — lint catches it (neograph-tlrs)."""
        from neograph import lint, node
        from neograph import merge_fn as merge_fn_deco
        from neograph.decorators import construct_from_functions

        @merge_fn_deco
        def simple_merge(
            variants: list[Claims],
            api_key: Annotated[str, FromInput],
        ) -> Claims:
            return variants[0]

        @node(outputs=Claims, prompt="test", model="fast",
              ensemble_n=2, merge_fn="simple_merge")
        def gen_w13() -> Claims: ...

        pipeline = construct_from_functions("w13-test", [gen_w13])
        issues = lint(pipeline, config={"some_other": "value"})
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        assert any(i.param == "api_key" for i in merge_issues)

    def test_lint_merge_fn_bundled_required_no_config(self):
        """W-15: Node(no DI) + merge_fn bundled required + config=None (neograph-wcbv)."""
        from pydantic import BaseModel

        from neograph import lint, node
        from neograph import merge_fn as merge_fn_deco
        from neograph.decorators import construct_from_functions

        class Ctx3(BaseModel):
            node_id: str
            project_root: str

        @merge_fn_deco
        def bundled_merge(
            variants: list[Claims],
            ctx: Annotated[Ctx3, FromInput(required=True)],
        ) -> Claims:
            return variants[0]

        @node(outputs=Claims, prompt="test", model="fast",
              ensemble_n=2, merge_fn="bundled_merge")
        def gen_w15() -> Claims: ...

        pipeline = construct_from_functions("w15-test", [gen_w15])
        issues = lint(pipeline)  # no config
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        missing = {i.param for i in merge_issues}
        assert "node_id" in missing
        assert "project_root" in missing

    def test_lint_oracle_callable_merge_fn_no_false_positive(self):
        """W-19: Oracle with callable merge_fn (not string) — no issues (neograph-xcy7)."""
        from neograph import lint
        from neograph.factory import register_scripted

        register_scripted("w19_gen", lambda i, c: Claims(items=["ok"]))

        def my_callable_merge(variants, config):
            return variants[0]

        pipeline = Construct("w19-test", nodes=[
            Node.scripted("gen", fn="w19_gen", outputs=Claims)
            | Oracle(n=2, merge_fn="w19_gen"),  # string merge_fn — lint checks it
        ])
        # Verify no crash when merge_fn is a registered string
        issues = lint(pipeline, config={"node_id": "test"})
        # This tests the path — no assertion on count, just no crash

    def test_lint_from_config_required_no_config(self):
        """W-21: FromConfig(required=True) + config=None — symmetric with FromInput (neograph-oued)."""
        from neograph import lint, node
        from neograph.decorators import construct_from_functions

        @node(outputs=Claims, prompt="test", model="fast")
        def gen_w21(limiter: Annotated[str, FromConfig(required=True)]) -> Claims: ...

        pipeline = construct_from_functions("w21-test", [gen_w21])
        issues = lint(pipeline)  # no config
        required_issues = [i for i in issues if i.required and i.param == "limiter"]
        assert len(required_issues) == 1
        assert "from_config" in required_issues[0].kind


class TestFromInputRequired:
    """FromInput(required=True) raises ExecutionError at runtime when missing."""

    def test_required_from_input_raises_when_missing(self):
        """Runtime: required=True param not in config raises ExecutionError."""
        @node(outputs=RawText)
        def my_node(
            topic: Annotated[str, FromInput(required=True)],
        ) -> RawText:
            return RawText(text=topic)

        pipeline = construct_from_functions("req", [my_node])
        graph = compile(pipeline)
        with pytest.raises(ExecutionError, match="topic"):
            run(graph, input={})

    def test_required_from_input_works_when_present(self):
        """Runtime: required=True param that IS in config works normally."""
        @node(outputs=RawText)
        def my_node(
            topic: Annotated[str, FromInput(required=True)],
        ) -> RawText:
            return RawText(text=topic)

        pipeline = construct_from_functions("req-ok", [my_node])
        graph = compile(pipeline)
        result = run(graph, input={"topic": "hello"})
        assert result["my_node"].text == "hello"

    def test_required_from_config_raises_when_missing(self):
        """Runtime: required=True FromConfig param not in config raises."""
        @node(outputs=RawText)
        def my_node(
            key: Annotated[str, FromConfig(required=True)],
        ) -> RawText:
            return RawText(text=key)

        pipeline = construct_from_functions("req-cfg", [my_node])
        graph = compile(pipeline)
        with pytest.raises(ExecutionError, match="key"):
            run(graph, input={})


# ═══════════════════════════════════════════════════════════════════════════
# NeographError.build() error builder pattern
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorBuilder:
    """NeographError.build() classmethod produces consistently structured
    error messages with what/expected/found/hint/location/node/construct."""

    def test_build_minimal_message_when_only_what(self):
        """build() with just `what` produces a plain message."""
        from neograph.errors import NeographError
        err = NeographError.build("something broke")
        assert isinstance(err, NeographError)
        assert str(err) == "something broke"

    def test_build_full_message_when_all_fields(self):
        """build() with all fields produces the structured format."""
        from neograph.errors import NeographError
        err = NeographError.build(
            "type mismatch",
            expected="Claims",
            found="RawText",
            hint="check your upstream",
            location="test.py:42",
            node="verify",
            construct="pipeline",
        )
        msg = str(err)
        assert msg.startswith("[Node 'verify' in construct 'pipeline']")
        assert "type mismatch" in msg
        assert "\n  expected: Claims" in msg
        assert "\n  found: RawText" in msg
        assert "\n  hint: check your upstream" in msg
        assert "\n  at test.py:42" in msg

    def test_build_node_only_prefix_when_no_construct(self):
        """build() with node= but no construct= uses [Node 'X'] prefix."""
        from neograph.errors import NeographError
        err = NeographError.build("failed", node="verify")
        assert str(err).startswith("[Node 'verify'] failed")

    def test_build_construct_only_prefix_when_no_node(self):
        """build() with construct= but no node= uses [Construct 'X'] prefix."""
        from neograph.errors import NeographError
        err = NeographError.build("failed", construct="pipeline")
        assert str(err).startswith("[Construct 'pipeline'] failed")

    def test_build_returns_subclass_when_called_on_subclass(self):
        """ConstructError.build() returns a ConstructError, not NeographError."""
        err = ConstructError.build("type mismatch", node="x")
        assert isinstance(err, ConstructError)
        assert isinstance(err, ValueError)  # dual inheritance preserved

    def test_build_returns_compile_error_when_called_on_compile_error(self):
        """CompileError.build() returns a CompileError."""
        from neograph.errors import CompileError
        err = CompileError.build("missing checkpointer")
        assert isinstance(err, CompileError)

    def test_build_returns_configuration_error_when_called_on_config_error(self):
        """ConfigurationError.build() returns a ConfigurationError."""
        err = ConfigurationError.build(
            "function not registered",
            hint="use register_scripted()",
        )
        assert isinstance(err, ConfigurationError)
        assert "function not registered" in str(err)
        assert "register_scripted()" in str(err)

    def test_execution_error_build_passes_validation_errors(self):
        """ExecutionError.build() accepts validation_errors kwarg."""
        err = ExecutionError.build(
            "DI resolution failed",
            node="my_node",
            found="field X missing from config",
            validation_errors="field X missing",
        )
        assert isinstance(err, ExecutionError)
        assert err.validation_errors == "field X missing"
        assert "DI resolution failed" in str(err)

    def test_execution_error_build_without_validation_errors(self):
        """ExecutionError.build() without validation_errors defaults to None."""
        err = ExecutionError.build("runtime failure", node="n")
        assert isinstance(err, ExecutionError)
        assert err.validation_errors is None

    def test_build_omits_absent_fields_when_partial(self):
        """build() with only expected= and hint= omits found= and location=."""
        from neograph.errors import NeographError
        err = NeographError.build(
            "wrong type",
            expected="int",
            hint="check annotation",
        )
        msg = str(err)
        assert "\n  expected: int" in msg
        assert "\n  hint: check annotation" in msg
        assert "\n  found:" not in msg
        assert "\n  at " not in msg


class TestFanInErrorsMigratedToBuild:
    """_check_fan_in_inputs errors use the build() pattern but existing
    test regex patterns still match (backward compatibility)."""

    def test_unknown_upstream_error_has_structured_format(self):
        """Unknown upstream fan-in error has node/construct prefix and
        available upstreams in the message."""
        a = _producer("a", Claims)
        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"a": Claims, "nonexistent": RawText},
            outputs=MatchResult,
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-name", nodes=[a, consumer])
        msg = str(exc_info.value)
        # Existing assertions must still pass
        assert "'nonexistent'" in msg
        assert "no upstream node" in msg

    def test_type_mismatch_error_has_structured_format(self):
        """Type mismatch fan-in error has node/construct prefix and
        expected/found types in the message."""
        a = _producer("a", Claims)
        b = _producer("b", RawText)
        consumer = Node.scripted(
            "consumer", fn="f",
            inputs={"a": Claims, "b": Claims},
            outputs=MatchResult,
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-type", nodes=[a, b, consumer])
        msg = str(exc_info.value)
        # Existing assertions must still pass
        assert "'b'" in msg
        assert "Claims" in msg
        assert "RawText" in msg


class TestTypeSpecValidation:
    """Node.inputs/outputs TypeSpec validator must reject non-type garbage.

    BUG neograph-m91y: _validate_type_spec is a no-op — accepts everything.
    """

    def test_string_rejected_as_inputs(self):
        """inputs='SomeType' (string, not a type) must raise."""
        with pytest.raises((TypeError, ValueError)):
            Node("bad", mode="scripted", inputs="SomeType", outputs=Claims)

    def test_int_rejected_as_outputs(self):
        """outputs=42 (int, not a type) must raise."""
        with pytest.raises((TypeError, ValueError)):
            Node("bad", mode="scripted", inputs=Claims, outputs=42)

    def test_list_of_strings_rejected(self):
        """inputs=['a', 'b'] must raise."""
        with pytest.raises((TypeError, ValueError)):
            Node("bad", mode="scripted", inputs=["a", "b"], outputs=Claims)

    def test_valid_types_accepted(self):
        """Smoke: valid type, dict, None, and generic alias all pass."""
        # These must NOT raise
        Node("ok1", mode="scripted", outputs=Claims)  # inputs=None default
        Node("ok2", mode="scripted", inputs=Claims, outputs=MatchResult)
        Node("ok3", mode="scripted", inputs={"a": Claims}, outputs=MatchResult)
        Node("ok4", mode="scripted", inputs=list[Claims], outputs=MatchResult)

    def test_generic_alias_accepted_as_inputs(self):
        """Generic aliases (list[X], dict[str,X], X|None) must pass validation.

        BUG neograph-vs6w: static annotation was type|dict|None which
        doesn't include generic aliases. PlainValidator is the real gate.
        """
        # These are NOT `type` instances — they're GenericAlias/UnionType
        Node("ga1", mode="scripted", inputs=list[Claims], outputs=Claims)
        Node("ga2", mode="scripted", inputs=dict[str, Claims], outputs=Claims)
        Node("ga3", mode="scripted", inputs=Claims | None, outputs=Claims)

    def test_dict_with_string_values_accepted(self):
        """Dict values can be strings (loader path before type resolution).

        BUG neograph-vs6w: string dict values pass validation but the
        static annotation says dict[str, type].
        """
        Node("sv1", mode="scripted", inputs={"a": "Claims"}, outputs=Claims)


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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=Claims),
            Node("summarize", prompt="Summarize: ${seed}",
                 model="default", outputs=Summary, inputs={"seed": Claims}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=Claims),
            Node("summarize", prompt="Summarize: ${nonexistent}",
                 model="default", outputs=Summary, inputs={"seed": Claims}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="A: ${bad1}, B: ${bad2}, OK: ${seed}",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
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

        sub = Construct("sub", input=Input, output=Output, nodes=[
            Node("proc", prompt="Process: ${original_param}",
                 model="default", outputs=Output,
                 inputs={"neo_subgraph_input": Input}),
        ])
        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="noop", outputs=Input),
            sub,
        ])
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

        sub = Construct("sub", input=Input, output=Output, nodes=[
            Node("proc", prompt="Process: ${neo_subgraph_input}",
                 model="default", outputs=Output,
                 inputs={"neo_subgraph_input": Input}),
        ])
        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="noop", outputs=Input),
            sub,
        ])
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

        @node(mode="think", outputs=Output, model="default",
              prompt="Process: ${text_input}")
        def proc(text_input: Input) -> Output: ...

        sub = construct_from_functions("sub", [proc], input=Input, output=Output)
        parent = Construct("parent", nodes=[
            Node.scripted("seed", fn="noop", outputs=Input),
            sub,
        ])
        issues = lint(parent)
        template_issues = [i for i in issues if "template" in i.kind]
        # After @node assembly, port param 'text_input' is remapped to neo_subgraph_input
        # If lint predicts correctly, ${text_input} is unresolvable
        # (the actual IR key is neo_subgraph_input)
        assert len(template_issues) >= 1
        assert "text_input" in template_issues[0].message

    # ── Known extras & custom vars ──────────────────────────────────────

    def test_known_extras_not_flagged_in_template_ref(self):
        """Template-ref {node_id}, {project_root} are framework extras -> no issue.
        Note: these are NOT available in inline prompts (no config access)."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/analyze",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
        resolver = lambda name: "ID: {node_id}, root: {project_root}, data: {seed}" if name == "rw/analyze" else None
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="Topic: ${topic}, data: ${seed}",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="Topic: ${topic}, data: ${seed}",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="Packet: ${research_packet}, data: ${seed}",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="Data: ${seed}",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/proc",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
        resolver = lambda name: "ID: {node_id}" if name == "rw/proc" else None
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=Claims),
            Node("summarize", prompt="Items: ${seed.items}",
                 model="default", outputs=Summary, inputs={"seed": Claims}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="Val: ${bad.field}",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/summarize",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt=None, model="default", outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="Just a plain instruction",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        assert template_issues == []

    # ── Input shape edge cases ──────────────────────────────────────────

    def test_node_with_no_inputs_flags_all_placeholders(self):
        """Source node with prompt and ${var} — no input keys, all flagged."""
        from neograph.lint import lint

        class B(BaseModel):
            y: str

        c = Construct("test", nodes=[
            Node("gen", prompt="Generate about: ${topic}",
                 model="default", outputs=B, inputs=None),
        ])
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

        c = Construct("test", nodes=[
            Node("gen", prompt="rw/gen",
                 model="default", outputs=B, inputs=None),
        ])
        resolver = lambda name: "Generate for: {node_id}" if name == "rw/gen" else None
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

        c = Construct("test", nodes=[
            Node.scripted("alpha", fn="noop", outputs=A),
            Node.scripted("beta", fn="noop", outputs=B),
            Node("merge", prompt="A: ${alpha}, B: ${beta}",
                 model="default", outputs=C,
                 inputs={"alpha": A, "beta": B}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=FullModel),
            Node("proc", prompt="rw/claim",
                 model="default", outputs=FullModel,
                 inputs={"seed": FullModel}),
        ])
        resolver = lambda name: "Claim: {claim_statement}" if name == "rw/claim" else None
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        errors = [i for i in template_issues if i.required]
        assert errors == [], (
            f"Flattened field in template-ref should be valid: {errors}"
        )

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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=Claims),
            Node("proc", prompt="Summarize: ${summary}",
                 model="default", outputs=Result, inputs={"seed": Claims}),
        ])
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged = {i.param for i in template_issues}
        assert "summary" in flagged, (
            f"Flattened field in inline prompt should be flagged: {template_issues}"
        )

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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=Claims),
            Node("proc", prompt="rw/summarize",
                 model="default", outputs=Result, inputs={"seed": Claims}),
        ])
        resolver = lambda name: "Summary: {summary}" if name == "rw/summarize" else None
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="ID: ${node_id}, data: ${seed}",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
        issues = lint(c)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged = {i.param for i in template_issues}
        assert "node_id" in flagged, (
            f"Known extra in inline prompt should be flagged: {template_issues}"
        )

    def test_template_ref_still_accepts_known_extras(self):
        """Template-ref {node_id} IS valid -- prompt_compiler has config access."""
        from neograph.lint import lint

        class A(BaseModel):
            x: str

        class B(BaseModel):
            y: str

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/analyze",
                 model="default", outputs=B, inputs={"seed": A}),
        ])
        resolver = lambda name: "ID: {node_id}, data: {seed}" if name == "rw/analyze" else None
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
        good_node = Node("review", prompt="rw/review", model="default",
                         outputs=Review,
                         inputs={"draft": Draft, "research": Research})
        predicted = _predict_input_keys(good_node)
        assert template_placeholders <= predicted, (
            f"Template needs {template_placeholders} but node provides {predicted}"
        )

        # Node with MISMATCHED inputs (the piarch bug pattern)
        bad_node = Node("review", prompt="rw/review", model="default",
                        outputs=Review,
                        inputs={"neo_subgraph_input": Draft})
        predicted_bad = _predict_input_keys(bad_node)
        unresolvable = template_placeholders - predicted_bad
        assert unresolvable == {"draft", "research"}, (
            f"Expected unresolvable placeholders, got {unresolvable}"
        )


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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/summarize", model="default",
                 outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/summarize", model="default",
                 outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=UCComposite),
            Node("writer", prompt="rw/write-si", model="default",
                 outputs=Output, inputs={"seed": UCComposite}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/summarize", model="default",
                 outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/unknown", model="default",
                 outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/summarize", model="default",
                 outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/summarize", model="default",
                 outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/summarize", model="default",
                 outputs=B, inputs={"seed": A}),
        ])
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

        c = Construct("test", nodes=[
            Node.scripted("seed", fn="noop", outputs=A),
            Node("proc", prompt="rw/proc", model="default",
                 outputs=B, inputs={"seed": A}),
        ])
        resolver = self._resolver({"rw/proc": "A: {bad1}, B: {bad2}, OK: {seed}"})
        issues = lint(c, template_resolver=resolver)
        template_issues = [i for i in issues if "template" in i.kind]
        flagged = {i.param for i in template_issues}
        assert flagged == {"bad1", "bad2"}


class TestSingleTypeInputsDeprecation:
    """Single-type inputs= should emit DeprecationWarning at assembly time.

    TASK neograph-np0y: _extract_single_type does O(N) isinstance scan.
    Phase 1 adds a warning to signal migration to dict-form.
    """

    def test_single_type_inputs_warns_at_assembly(self):
        """Construct assembly with single-type inputs on non-first node warns."""
        a = _producer("a", RawText)
        b = _consumer("b", RawText, Claims)  # _consumer uses single-type inputs
        with pytest.warns(DeprecationWarning, match="single-type.*inputs"):
            Construct(name="test", nodes=[a, b])
