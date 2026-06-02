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
    construct_from_module,
    merge_fn,
    node,
)
from neograph.construct import Construct


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

    def test_normalize_ir_idempotent_on_fan_out_param(self):
        """A second normalize_ir call must not change an already-set
        fan_out_param (idempotency — required because the @node surface sets
        it before Construct.__init__ runs normalize_ir)."""
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

        before = _node_by_name(pipeline, "consumer").fan_out_param
        normalize_ir(pipeline)
        after = _node_by_name(pipeline, "consumer").fan_out_param
        assert before == after == "item"


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
