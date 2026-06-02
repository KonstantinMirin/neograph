"""Behavioral contract for neograph._ir_normalize (neograph-20xq).

The epic unifies IR-level field inference (fan_out_param, oracle_gen_type)
behind a single entry point, normalize_ir(construct), called once from
Construct.__init__ regardless of which API surface built the Construct.

These tests pin the BEHAVIORAL contract of that entry point:

  1. normalize_ir is the inference owner — given a Construct whose IR-level
     fields have been cleared, it re-derives them. (Proves the logic moved
     into _ir_normalize, not just that Construct.__init__ happens to set them.)
  2. normalize_ir is idempotent — a second call is a no-op.
  3. Three-surface parity — @node, programmatic pipe, and a programmatic
     equivalent all produce identical IR-level field values.

The three-surface tests guard EXISTING behavior across the refactor; tests 1-2
define the NEW public contract of the extracted module.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import (
    Each,
    Node,
    compile,
    construct_from_module,
    merge_fn,
    node,
    run,
)
from neograph.construct import Construct
from tests.fakes import build_test_compile_kwargs, register_scripted


class Alpha(BaseModel, frozen=True):
    value: str


class Beta(BaseModel, frozen=True):
    label: str


class Gamma(BaseModel, frozen=True):
    score: float


def _node_by_name(construct: Construct, name: str) -> Node:
    for n in construct.nodes:
        if getattr(n, "name", None) == name:
            assert isinstance(n, Node)
            return n
    raise AssertionError(f"node {name!r} not found in {construct.name}")


class TestDeclaredOutputFields:
    """declared_output_fields is the single source of producer field NAMES,
    shared by normalize_ir's peer set and the validator's producer loop, so the
    two can't drift (neograph-bcct)."""

    def test_dict_output_yields_per_key_fields_no_bare_base(self):
        from neograph._ir_normalize import declared_output_fields

        class _X(BaseModel, frozen=True):
            a: str

        class _Y(BaseModel, frozen=True):
            b: str

        n = Node.scripted("gen", fn="dof_gen", outputs={"result": _X, "log": _Y})
        fields = declared_output_fields(n)
        assert fields == {"gen_result", "gen_log"}, (
            "dict-output node must contribute per-key fields, NOT the bare base "
            "'gen' — exact parity with the validator's producer registration"
        )

    def test_single_output_yields_bare_base(self):
        from neograph._ir_normalize import declared_output_fields

        n = Node.scripted("extract", fn="dof_extract", outputs=Alpha)
        assert declared_output_fields(n) == {"extract"}

    def test_none_output_yields_empty(self):
        from neograph._ir_normalize import declared_output_fields

        n = Node.scripted("sink", fn="dof_sink")  # outputs=None
        assert declared_output_fields(n) == set(), (
            "a None-output node registers no producer; its field must not leak "
            "into the known set (would diverge from the validator)"
        )

    def test_hyphenated_name_uses_field_form(self):
        from neograph._ir_normalize import declared_output_fields

        n = Node.scripted("my-gen", fn="dof_mygen", outputs={"result": Alpha})
        assert declared_output_fields(n) == {"my_gen_result"}


class TestFanOutMultiOutputRuntime:
    """End-to-end: an Each consumer fanning over a multi-output upstream's
    per-key field must route the fanned item at runtime (neograph-bcct). Before
    the fix, fan_out_param was None and _extract_fan_in_dict could not route."""

    def test_each_over_multi_output_field_runs(self):
        class _Item(BaseModel, frozen=True):
            v: str

        class _Log(BaseModel, frozen=True):
            msg: str

        class _Summary(BaseModel, frozen=True):
            s: str

        register_scripted(
            "bcct_gen_rt",
            lambda _i, _c: {"result": [_Item(v="a"), _Item(v="b")], "log": _Log(msg="ok")},
        )
        register_scripted("bcct_cons_rt", lambda input_data, _c: _Summary(s=input_data["item"].v))

        gen = Node.scripted(
            "gen", fn="bcct_gen_rt",
            outputs={"result": list[_Item], "log": _Log},
        )
        consumer = Node.scripted(
            "consumer", fn="bcct_cons_rt",
            inputs={"gen_result": list[_Item], "item": _Item},
            outputs=_Summary,
        ) | Each(over="gen_result", key="v")

        pipeline = Construct("bcct-rt", nodes=[gen, consumer])
        # fan_out_param must be set for the runtime to route the fanned item.
        assert _node_by_name(pipeline, "consumer").fan_out_param == "item"

        graph = compile(pipeline, **build_test_compile_kwargs())
        result = run(graph, input={})
        # Each produces a dict keyed by `key`; both items were processed.
        produced = result["consumer"]
        summaries = list(produced.values()) if isinstance(produced, dict) else produced
        assert {s.s for s in summaries} == {"a", "b"}, (
            f"both fanned items must be processed; got {produced!r}"
        )


class TestFanOutCandidates:
    """fan_out_candidates is the SINGLE definition of 'which dict-form input
    key is an Each fan-out receiver candidate' (neograph-k7bg). Both
    _FanOutParamNormalizer (the writer of fan_out_param) and the validator's
    _check_fan_in_inputs (the tolerator of the unmatched key) derive their
    candidates from it. A candidate is an input key whose field name is neither
    a known producer/peer field nor the node's own field; the caller supplies
    its own ``known_field_names`` (the normalizer has peer node fields, the
    validator has the full producer field set — different pipeline stages)."""

    def _each_consumer(self, name: str, inputs: dict, over: str) -> Node:
        n = Node.scripted(name, fn=f"{name}_fn", inputs=inputs, outputs=Gamma)
        return n | Each(over=over, key="value")

    def test_returns_single_unmatched_key_as_candidate(self):
        from neograph._ir_normalize import fan_out_candidates

        consumer = self._each_consumer("consumer", {"claims": Alpha, "item": Beta}, "claims")
        cands = fan_out_candidates(consumer, {"claims", "consumer"})
        assert cands == ["item"], (
            f"the one input key naming no known field is the fan-out candidate; got {cands}"
        )

    def test_excludes_the_nodes_own_field(self):
        from neograph._ir_normalize import fan_out_candidates

        # A node consuming its own output (loop self-reference shape): the
        # self field must NOT be reported as a fan-out candidate.
        n = Node.scripted("refine", fn="refine_fn", inputs={"seed": Alpha, "refine": Gamma}, outputs=Gamma)
        cands = fan_out_candidates(n, {"seed", "refine"})
        assert cands == [], f"self field 'refine' must be excluded; got {cands}"

    def test_caller_supplied_known_set_drives_result(self):
        from neograph._ir_normalize import fan_out_candidates

        consumer = self._each_consumer("consumer", {"a": Alpha, "b": Beta}, "a")
        # Both known -> nothing unmatched.
        assert fan_out_candidates(consumer, {"a", "b", "consumer"}) == []
        # Only 'a' known -> 'b' is the candidate. Proves the caller's set drives
        # the result (the normalizer and validator pass different sets).
        assert fan_out_candidates(consumer, {"a", "consumer"}) == ["b"]

    def test_normalizer_writes_fan_out_param_via_the_shared_helper(self):
        """Integration: the normalizer still resolves fan_out_param (it now
        derives candidates from fan_out_candidates). normalize_ir runs inside
        Construct.__init__."""
        consumer = self._each_consumer("consumer", {"claims": Alpha, "item": Beta}, "claims")
        producer = Node.scripted("claims", fn="k7bg_p", outputs=Alpha)
        pipeline = Construct("k7bg-int", nodes=[producer, consumer])
        assert _node_by_name(pipeline, "consumer").fan_out_param == "item"


class TestNormalizeIrOwnsFanOutParam:
    """normalize_ir is the single inference site for fan_out_param."""

    def test_normalize_ir_rederives_cleared_fan_out_param(self):
        """Given a constructed Each + dict-form node whose fan_out_param has
        been cleared, normalize_ir must re-derive the single unknown input
        key as the fan-out receiver. This proves the inference lives in
        _ir_normalize, not merely as a side effect of Construct.__init__."""
        from neograph._ir_normalize import normalize_ir

        consumer = Node.scripted(
            "consumer",
            fn="ir_norm_consumer",
            inputs={"claims": Alpha, "item": Beta},
            outputs=Gamma,
        )
        consumer = consumer | Each(over="claims", key="value")
        producer = Node.scripted("claims", fn="ir_norm_producer", outputs=Alpha)

        pipeline = Construct("ir-norm-fanout", nodes=[producer, consumer])

        # Existing behavior: the single unknown input key ("item") is the
        # fan-out receiver.
        idx = next(
            i for i, n in enumerate(pipeline.nodes)
            if getattr(n, "name", None) == "consumer"
        )
        assert pipeline.nodes[idx].fan_out_param == "item"

        # Clear it to simulate the pre-normalization IR a non-@node surface
        # produces, then prove normalize_ir restores it.
        pipeline.nodes[idx] = pipeline.nodes[idx].model_copy(
            update={"fan_out_param": None}
        )
        assert pipeline.nodes[idx].fan_out_param is None

        normalize_ir(pipeline)

        assert pipeline.nodes[idx].fan_out_param == "item", (
            "normalize_ir must re-derive fan_out_param for an Each + dict-form "
            "node with a single unknown input key"
        )

    def test_fan_out_param_set_when_each_consumes_multi_output_upstream(self):
        """neograph-bcct: an Each node that fans over a MULTI-OUTPUT upstream's
        per-output-key field must still get fan_out_param set.

        Bug: the normalizer computed candidates against peer NODE field names
        ({field_name_for(name)}), which excludes per-output-key producer fields
        like 'gen_result'. So for a consumer with inputs={'gen_result', 'item'}
        fanning over 'gen_result', the normalizer saw BOTH keys as unknown (two
        candidates) and declined -> fan_out_param=None, while the validator
        (using the full producer field set) correctly tolerated it. The
        assembled IR was wrong: fan_out_param must be 'item'."""
        class _Item(BaseModel, frozen=True):
            v: str

        class _Log(BaseModel, frozen=True):
            msg: str

        class _Summary(BaseModel, frozen=True):
            s: str

        gen = Node.scripted(
            "gen", fn="bcct_gen",
            outputs={"result": list[_Item], "log": _Log},
        )
        consumer = Node.scripted(
            "consumer", fn="bcct_cons",
            inputs={"gen_result": list[_Item], "item": _Item},
            outputs=_Summary,
        ) | Each(over="gen_result", key="v")

        pipeline = Construct("bcct-multi-output", nodes=[gen, consumer])

        assert _node_by_name(pipeline, "consumer").fan_out_param == "item", (
            "an Each consumer over a multi-output upstream's per-key field must "
            "have fan_out_param set to the fan-out receiver ('item'), not None"
        )

    def test_normalize_ir_idempotent_after_rederivation(self):
        """Idempotency of the inference itself (neograph-hede / TQ-04).

        The weak version of this test only re-ran normalize_ir on an
        already-set node, so _FanOutParamNormalizer.applies_to short-circuited
        and `apply` was never exercised — it would pass even if `apply` were
        broken. This version CLEARS the field first (forcing `apply` to run on
        the first normalize), then normalizes AGAIN, asserting the second pass
        is a true no-op and the value is stable across both passes."""
        from neograph._ir_normalize import normalize_ir

        consumer = Node.scripted(
            "consumer",
            fn="ir_norm_consumer2",
            inputs={"claims": Alpha, "item": Beta},
            outputs=Gamma,
        )
        consumer = consumer | Each(over="claims", key="value")
        producer = Node.scripted("claims", fn="ir_norm_producer2", outputs=Alpha)
        pipeline = Construct("ir-norm-idem", nodes=[producer, consumer])

        idx = next(
            i for i, n in enumerate(pipeline.nodes)
            if getattr(n, "name", None) == "consumer"
        )
        # Clear so the FIRST normalize_ir actually runs apply (not short-circuit).
        pipeline.nodes[idx] = pipeline.nodes[idx].model_copy(update={"fan_out_param": None})

        normalize_ir(pipeline)            # pass 1: apply runs, sets "item"
        after_first = pipeline.nodes[idx].fan_out_param
        normalize_ir(pipeline)            # pass 2: must be a no-op
        after_second = pipeline.nodes[idx].fan_out_param

        assert after_first == "item", "first normalize_ir (post-clear) must re-derive via apply"
        assert after_second == after_first, "second normalize_ir must be a no-op (idempotent)"

    def test_node_and_normalizer_agree_on_fan_out_param(self):
        """The @node signature-derived fan_out_param (written at
        _construct_builder.py:568) and the inputs-dict normalizer must AGREE on
        the receiver key (neograph-hede / DRY-02).

        Background: the refactor keeps two derivations of fan_out_param — the
        @node signature path and the normalizer's inputs-dict heuristic. The
        original justification was that the signature path is "richer" (handles
        multi-unknown). Empirically (recorded in the bead) a multi-unknown
        @node Each is rejected by the validator, so for every VALID @node Each
        the normalizer reproduces the builder's value. This test pins that
        AGREEMENT — the actually-true invariant — so a future change to either
        derivation that makes them diverge is caught. (It also means builder:568
        is redundant-but-consistent; deleting it is a safe future simplification
        tracked under neograph-k7bg, out of scope here.)"""
        from neograph._ir_normalize import _FanOutParamNormalizer
        from neograph.naming import field_name_for

        @node(outputs=Alpha)
        def produce() -> Alpha: ...

        @node(outputs=Gamma, map_over="produce", map_key="value")
        def summarize(produce: Alpha, item: Beta) -> Gamma: ...

        import types as t
        mod = t.ModuleType("test_agree_fanout_mod")
        mod.produce = produce
        mod.summarize = summarize
        construct = construct_from_module(mod)

        node_ir = _node_by_name(construct, "summarize")
        builder_value = node_ir.fan_out_param
        assert builder_value == "item", "sanity: @node builder resolves the fan-out receiver"

        # Independent re-derivation by the normalizer, given the same peers.
        peers = {
            field_name_for(n.name) for n in construct.nodes
            if getattr(n, "name", None) is not None
        }
        cleared = node_ir.model_copy(update={"fan_out_param": None})
        normalizer_update = _FanOutParamNormalizer().apply(cleared, peers)

        assert normalizer_update == {"fan_out_param": builder_value}, (
            f"@node builder set fan_out_param={builder_value!r} but the "
            f"normalizer independently derived {normalizer_update!r} — the two "
            f"derivations must agree for every valid @node Each"
        )


class TestNormalizeIrOwnsOracleGenType:
    """normalize_ir is the single inference site for oracle_gen_type."""

    def test_normalize_ir_rederives_cleared_oracle_gen_type(self):
        """Given a constructed Oracle node whose oracle_gen_type has been
        cleared, normalize_ir must re-infer the per-generator type from the
        merge_fn signature."""
        from neograph._ir_normalize import normalize_ir

        class GenType(BaseModel, frozen=True):
            raw: str

        class MergedType(BaseModel, frozen=True):
            combined: str

        @node(outputs=MergedType, model="fast", prompt="gen",
              ensemble_n=2, merge_fn="ir_norm_merge")
        def generate() -> MergedType: ...

        @merge_fn
        def ir_norm_merge(variants: list[GenType]) -> MergedType:
            return MergedType(combined=",".join(v.raw for v in variants))

        import types as t
        mod = t.ModuleType("test_ir_norm_oracle_mod")
        mod.generate = generate
        pipeline = construct_from_module(mod)

        gen_node = pipeline.nodes[0]
        assert gen_node.oracle_gen_type is GenType

        idx = next(
            i for i, n in enumerate(pipeline.nodes)
            if getattr(n, "name", None) == gen_node.name
        )
        pipeline.nodes[idx] = pipeline.nodes[idx].model_copy(
            update={"oracle_gen_type": None}
        )
        assert pipeline.nodes[idx].oracle_gen_type is None

        normalize_ir(pipeline)

        assert pipeline.nodes[idx].oracle_gen_type is GenType, (
            "normalize_ir must re-infer oracle_gen_type from the merge_fn "
            "first-parameter list element type"
        )


class TestThreeSurfaceIrParity:
    """The same logical pipeline built via different API surfaces must produce
    identical IR-level field values. Guards existing behavior across the
    refactor (the Core Invariant)."""

    def test_fan_out_param_identical_across_node_and_programmatic(self):
        """@node decoration and programmatic pipe must agree on fan_out_param."""
        # Programmatic surface
        prog_consumer = Node.scripted(
            "summarize",
            fn="parity_consumer",
            inputs={"produce": Alpha, "item": Beta},
            outputs=Gamma,
        ) | Each(over="produce", key="value")
        prog_producer = Node.scripted("produce", fn="parity_producer", outputs=Alpha)
        prog = Construct("parity-prog", nodes=[prog_producer, prog_consumer])
        prog_fop = _node_by_name(prog, "summarize").fan_out_param

        # @node surface — equivalent topology
        @node(outputs=Alpha)
        def produce() -> Alpha: ...

        @node(outputs=Gamma, map_over="produce", map_key="value")
        def summarize(produce: Alpha, item: Beta) -> Gamma: ...

        import types as t
        mod = t.ModuleType("test_parity_fanout_mod")
        mod.produce = produce
        mod.summarize = summarize
        deco = construct_from_module(mod)
        deco_fop = _node_by_name(deco, "summarize").fan_out_param

        assert prog_fop == deco_fop == "item", (
            f"fan_out_param diverged across surfaces: "
            f"programmatic={prog_fop!r}, @node={deco_fop!r}"
        )
