"""Fan-in validation: dict-form inputs, Each interop, effective_producer_type,
list/dict compatibility, dict-form outputs, three-surface parity."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Each,
    Node,
    Operator,
    Oracle,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph._normalize import normalize_inputs
from tests.fakes import build_test_compile_kwargs
from tests.schemas import (
    Claims,
    ClassifiedClaims,
    ClusterGroup,
    Clusters,
    MatchResult,
    MergedResult,
    RawText,
    _consumer,
    _producer,
)


class TestNodeDecoratorFanInValidation:
    """@node fan-in: ALL parameter types must match their upstream's output."""

    @staticmethod
    def _fresh_module(name: str):
        import types as _types

        return _types.ModuleType(name)

    def test_mismatch_raises_when_second_param_wrong(self):
        """3-way fan-in where param 2's annotation doesn't match upstream output."""
        from neograph import ConstructError, construct_from_module

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
        from neograph import ConstructError, construct_from_module

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
        from neograph import construct_from_module

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
        from neograph import construct_from_module

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
                classified=[{"claim": label, "category": result.cluster_label} for label, result in fie_verify.items()]
            )

        pipeline = construct_from_functions("fie-pipeline", [fie_source, fie_verify, fie_summarize])
        assert [n.name for n in pipeline.nodes] == [
            "fie-source",
            "fie-verify",
            "fie-summarize",
        ]

        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "fie-001"})

        assert isinstance(result["fie_summarize"], ClassifiedClaims)
        categories = {c["category"] for c in result["fie_summarize"].classified}
        assert categories == {"alpha", "beta"}

    def test_each_result_consumed_as_raw_dict_when_unparameterized(self):
        """Downstream parameter typed as plain `dict` (unparameterized) is
        also compatible with an Each-modified upstream."""

        @node(outputs=Clusters)
        def fier_source() -> Clusters:
            return Clusters(groups=[ClusterGroup(label="alpha", claim_ids=["c1"])])

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

        construct_from_functions("fier-pipeline", [fier_source, fier_verify, fier_summarize])

    def test_raw_type_rejected_when_upstream_has_each(self):
        """If a downstream param is annotated as the raw upstream output type
        (NOT wrapped in dict), it must be rejected when the upstream has Each."""
        from neograph import ConstructError

        @node(outputs=Clusters)
        def fieraw_source() -> Clusters:
            return Clusters(groups=[ClusterGroup(label="alpha", claim_ids=["c1"])])

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
        from neograph import ConstructError

        @node(outputs=Clusters)
        def fiew_source() -> Clusters:
            return Clusters(groups=[ClusterGroup(label="alpha", claim_ids=["c1"])])

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

        n = Node.scripted("each-node", fn="_x_each", inputs=ClusterGroup, outputs=MatchResult) | Each(
            over="upstream.items", key="label"
        )
        effective = effective_producer_type(n)
        assert effective == dict[str, MatchResult]

    def test_oracle_keeps_raw_output(self):
        """Oracle merges N variants into ONE output. Effective type is unchanged."""
        from neograph._construct_validation import effective_producer_type

        n = Node.scripted("ens", fn="_x_ens", outputs=Claims) | Oracle(n=3, merge_fn="nonexistent_ok_for_this_test")
        assert effective_producer_type(n) is Claims

    def test_operator_keeps_raw_output(self):
        """Operator is an interrupt modifier -- it doesn't reshape state."""
        from neograph._construct_validation import effective_producer_type

        n = Node.scripted("op", fn="_x_op", outputs=Claims) | Operator(when="_nonexistent_for_helper_test")
        assert effective_producer_type(n) is Claims

    def test_sub_construct_each_wraps_as_dict(self):
        """An Each modifier on a sub-Construct wraps its output the same way."""
        from neograph._construct_validation import effective_producer_type

        inner = Node.scripted("inner", fn="_x_inner", inputs=Claims, outputs=Claims)
        sub = Construct("sub", input=Claims, output=Claims, nodes=[inner]) | Each(over="upstream.items", key="label")
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


class TestEffectiveProducerTypeFor:
    """Per-key core (neograph-etxo): the single implementation of the
    modifier-to-bus rule, shared by the whole-node and dict-form paths."""

    def test_no_modifier_set_returns_declared_type(self):
        from neograph._construct_validation import effective_producer_type_for

        assert effective_producer_type_for(Claims, None) is Claims

    def test_each_modifier_wraps_single_type_as_dict(self):
        from neograph._construct_validation import effective_producer_type_for

        n = Node.scripted("each-node", fn="_x_each2", inputs=ClusterGroup, outputs=MatchResult) | Each(
            over="upstream.items", key="label"
        )
        assert effective_producer_type_for(MatchResult, n.modifier_set) == dict[str, MatchResult]

    def test_non_each_modifier_leaves_type_unchanged(self):
        from neograph._construct_validation import effective_producer_type_for

        n = Node.scripted("op2", fn="_x_op2", outputs=Claims) | Operator(when="_nonexistent_for_helper_test")
        assert effective_producer_type_for(Claims, n.modifier_set) is Claims

    def test_dict_form_each_wraps_each_key_independently(self):
        """The dict-form branch applies the rule PER KEY, not to the whole dict."""
        from neograph._construct_validation import effective_producer_type_for

        n = Node.scripted(
            "multi",
            fn="_x_multi",
            inputs=ClusterGroup,
            outputs={"a": MatchResult, "b": Claims},
        ) | Each(over="upstream.items", key="label")
        assert effective_producer_type_for(MatchResult, n.modifier_set) == dict[str, MatchResult]
        assert effective_producer_type_for(Claims, n.modifier_set) == dict[str, Claims]


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
            "consumer",
            fn="f",
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
            "consumer",
            fn="f",
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
            "consumer",
            fn="f",
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
            "canonicalize",
            fn="f",
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
            "process",
            fn="f",
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
        "consumer",
        fn="f",
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
        "consumer",
        fn="f",
        inputs={"a": Claims, "b": Claims},  # b produces RawText, not Claims
        outputs=MatchResult,
    )
    Construct("fan-in-bad-prog", nodes=[a, b, consumer])


class TestThreeSurfaceFanInParity:
    """Fan-in type validation tested identically across declarative, @node,
    and programmatic API surfaces. Template pattern for future parity tests."""

    @pytest.mark.parametrize(
        "build",
        [
            _fan_in_valid_declarative,
            _fan_in_valid_decorator,
            _fan_in_valid_programmatic,
        ],
        ids=["declarative", "decorator", "programmatic"],
    )
    def test_fan_in_assembles_when_types_correct(self, build):
        """Fan-in with matching types assembles without error across surfaces."""
        pipeline = build()
        assert len(pipeline.nodes) == 3

    @pytest.mark.parametrize(
        "build",
        [
            _fan_in_mismatch_declarative,
            _fan_in_mismatch_decorator,
            _fan_in_mismatch_programmatic,
        ],
        ids=["declarative", "decorator", "programmatic"],
    )
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


class TestFanInErrorsMigratedToBuild:
    """_check_fan_in_inputs errors use the build() pattern but existing
    test regex patterns still match (backward compatibility)."""

    def test_unknown_upstream_error_has_structured_format(self):
        """Unknown upstream fan-in error has node/construct prefix and
        available upstreams in the message."""
        a = _producer("a", Claims)
        consumer = Node.scripted(
            "consumer",
            fn="f",
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
            "consumer",
            fn="f",
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


class TestDictOutputParamPortPriority:
    """Dict-output reference params take priority over port classification.

    The regression shape (ox wnp.4): construct input=Base, an upstream node emits dict-form
    outputs whose 'result' type SUBCLASSES the construct input (an enriched Base), and a
    downstream node consumes '{upstream}_result'. Port identification must recognise the
    dict-output reference — a param is a port ONLY when it matches no producer, mirroring
    the existing 'peer @node takes priority' rule."""

    def test_dict_output_param_subclassing_construct_input_binds_to_the_output(self):
        class Base(BaseModel):
            x: str = "b"

        class Enriched(Base):
            verdict: str = "ok"

        @node(mode="scripted", outputs={"result": Enriched, "meta": RawText})
        def judge(inp: Base) -> Enriched:
            return Enriched(x=inp.x)

        @node(mode="scripted", outputs=Enriched)
        def seal(judge_result: Enriched) -> Enriched:
            # judge_result references judge's dict output; Enriched subclasses the construct
            # input, so pre-fix this param was misclassified as a port (neo_subgraph_input).
            return judge_result

        # Must build, and the param must bind to the dict output — not the subgraph input port.
        built = construct_from_functions("dict-output-port-priority", [judge, seal], input=Base, output=Enriched)
        seal_node = next(n for n in built.nodes if n.name == "seal")
        seal_inputs = normalize_inputs(seal_node.inputs).by_name
        assert "judge_result" in seal_inputs
        assert "neo_subgraph_input" not in seal_inputs

    def test_true_port_param_still_binds_when_dict_outputs_exist(self):
        """The port path is untouched: a param typed as the construct input with a
        non-producer name still reads neo_subgraph_input."""

        class Base(BaseModel):
            x: str = "b"

        class Enriched(Base):
            verdict: str = "ok"

        @node(mode="scripted", outputs={"result": RawText, "meta": Claims})
        def explore(inp: Base) -> RawText:
            return RawText(text=inp.x)

        @node(mode="scripted", outputs=Enriched)
        def seal(explore_result: RawText, original: Base) -> Enriched:
            return Enriched(x=original.x)

        built = construct_from_functions("port-still-works", [explore, seal], input=Base, output=Enriched)
        seal_node = next(n for n in built.nodes if n.name == "seal")
        seal_inputs = normalize_inputs(seal_node.inputs).by_name
        assert "neo_subgraph_input" in seal_inputs  # `original: Base` still reads the port
        assert "explore_result" in seal_inputs
