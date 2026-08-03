"""Phase 7 / neograph-s7zt3.10 -- ModifierCombo.EACH_ORACLE (the Each x Oracle
FUSION) must survive an Agent Spec round-trip (export AND import, in lockstep).

At HEAD ``_lower_construct_item``'s EACH arm rejects the fusion outright
(``is_each_oracle_fused(mods)`` -> ``ConfigurationError: node 'fanned' has
modifier combination EACH_ORACLE -- no Agent Spec lowering yet``), and the
import walk never descends into ``MapNode.subflow``, so even a hand-built fused
Flow dies with "Each group 'fanned''s sub-flow has 3 inner nodes, expected 1".

The lowering is COMPOSITION, not a new primitive: a ``MapNode`` whose subflow is
what ``_lower_oracle`` already produces un-fused (``StartNode -> merge`` as the
single start edge -- pyagentspec requires exactly one -- plus the variant ->
merge control edges, the fan-in data edges, and ``merge -> EndNode``).

Binding evidence rules:

  R1  full ``Flow.from_dict(flow.to_dict())`` survival is asserted, not the
      weaker in-memory-Flow-only path.
  R3  ``classify_modifiers(<reimported item>)[0] is ModifierCombo.EACH_ORACLE``
      is asserted explicitly -- a silent downgrade to plain EACH (or to plain
      ORACLE) would pass every structural assertion.
  R2 does not bind this module (no Operator, so no checkpointer is required);
      it binds the EACH_ORACLE_OPERATOR sibling.

ELEMENT-TYPE PIN (the decision the ticket makes UP FRONT): the fused import
must apply ``_dict_form_inputs_from_props`` to the FIRST VARIANT node's inputs
-- the fused analogue of "the inner spec" that ``_reconstruct_each_node`` uses.
``_reconstruct_oracle_group`` instead derives inputs from the merge node's data
edges, and the two do NOT compose, so without this the fan-out receiver's
element type silently stops matching the producer's ``list[X]`` element and
neograph-3lk2l regresses INSIDE the fusion. The node therefore declares the
PRIMARY dict-form ``inputs={'cluster': Tagged}`` shape and the round-trip
asserts the identity directly.
"""

from __future__ import annotations

import typing
import warnings

from pydantic import BaseModel

from neograph import Construct, Each, Node, Oracle, compile, run
from neograph._agent_spec import (
    _MARK_EACH_SPEC,
    _MARK_GROUP_ID,
    _MARK_MODIFIER,
    _MARK_ORACLE_SPEC,
    _MARK_VARIANT,
    to_agent_spec,
)
from neograph.loader import from_agent_spec
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.testing.fakes import StructuredFake
from tests.fakes import build_fake_llm_kwargs, build_test_compile_kwargs, register_scripted


class Tagged(BaseModel, frozen=True):
    label: str


class Bag(BaseModel, frozen=True):
    items: list[Tagged]


class Claims(BaseModel, frozen=True):
    items: list[str]


def _register() -> None:
    def merge_fn(variants, config):
        all_items: list[str] = []
        for v in variants:
            all_items.extend(v.items)
        return type(variants[0])(items=all_items)

    register_scripted("eor_seed", lambda input_data, config: Bag(items=[Tagged(label="a"), Tagged(label="b")]))
    register_scripted("eor_merge", merge_fn)


def _each_oracle_pipeline() -> Construct:
    """seed -> a fanned-out node that ALSO runs a 2-variant ensemble per item."""
    _register()
    return Construct(
        "each-oracle",
        nodes=[
            Node.scripted("seed", fn="eor_seed", outputs=Bag),
            Node(
                name="fanned",
                mode="think",
                model="fast",
                prompt="rw/fanned",
                inputs={"cluster": Tagged},
                outputs=Claims,
            )
            | Oracle(n=2, merge_fn="eor_merge")
            | Each(over="seed.items", key="label"),
        ],
    )


def _import(flow):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_agent_spec(flow)


def _by_name(construct: Construct) -> dict[str, Node]:
    return {n.name: n for n in construct.nodes}


def _inner_nodes(map_node) -> list:
    return [n for n in map_node.subflow.nodes if type(n).__name__ not in ("StartNode", "EndNode")]


class TestEachOracleExportShape:
    """The fusion lowers to ONE MapNode whose subflow IS the un-fused Oracle
    lowering -- nesting two existing lowerings, not a new primitive."""

    def test_export_emits_one_map_node_carrying_the_oracle_group_inside(self):
        flow = to_agent_spec(_each_oracle_pipeline())
        by_name = {n.name: n for n in flow.nodes}

        map_node = by_name["fanned"]
        assert type(map_node).__name__ == "MapNode"
        assert map_node.metadata[_MARK_MODIFIER] == "each"
        assert map_node.metadata[_MARK_EACH_SPEC]["over"] == "seed.items"
        assert map_node.metadata[_MARK_EACH_SPEC]["key"] == "label"

        assert not [n for n in flow.nodes if "variant" in n.name], (
            "the ensemble variants belong INSIDE the MapNode's subflow -- a "
            "top-level variant means the fusion was flattened, not nested"
        )

        inner = _inner_nodes(map_node)
        variants = [n for n in inner if _MARK_VARIANT in (n.metadata or {})]
        assert len(variants) == 2
        assert {type(v).__name__ for v in variants} == {"LlmNode"}

        merge = next(n for n in inner if _MARK_ORACLE_SPEC in (n.metadata or {}))
        assert merge.metadata[_MARK_ORACLE_SPEC]["n"] == 2
        assert merge.metadata[_MARK_ORACLE_SPEC]["merge_fn"] == "eor_merge"

        group_ids = {n.metadata[_MARK_GROUP_ID] for n in inner}
        assert len(group_ids) == 1, "variants and merge must share ONE group id inside the subflow"

    def test_subflow_start_edge_enters_the_variant_chain_and_exits_at_the_merge(self):
        """neograph-s7zt3.15: pyagentspec requires exactly one StartNode outgoing
        control edge, and the fused subflow spends it on the HEAD OF THE VARIANT
        CHAIN -- variant_0 -> ... -> variant_{n-1} -> merge -> EndNode.

        It used to be spent on the merge instead, which left every variant with
        no inbound control edge at all: a foreign runtime walking these edges
        literally would run the merge over outputs nothing had produced.
        """
        flow = to_agent_spec(_each_oracle_pipeline())
        map_node = {n.name: n for n in flow.nodes}["fanned"]
        subflow = map_node.subflow

        start_edges = [e for e in subflow.control_flow_connections if type(e.from_node).__name__ == "StartNode"]
        assert len(start_edges) == 1

        merge = next(n for n in _inner_nodes(map_node) if _MARK_ORACLE_SPEC in (n.metadata or {}))
        variants = [n for n in _inner_nodes(map_node) if _MARK_VARIANT in (n.metadata or {})]
        variants.sort(key=lambda n: n.metadata[_MARK_VARIANT])
        assert start_edges[0].to_node.name == variants[0].name

        pairs = {(e.from_node.name, e.to_node.name) for e in subflow.control_flow_connections}
        chain = [*variants, merge]
        for src, dst in zip(chain, chain[1:], strict=False):
            assert (src.name, dst.name) in pairs, f"missing chain edge {src.name} -> {dst.name}"

        end_edges = [e for e in subflow.control_flow_connections if type(e.to_node).__name__ == "EndNode"]
        assert [e.from_node.name for e in end_edges] == [merge.name]


class TestEachOracleRoundTrip:
    """Export -> import recovers BOTH modifiers on ONE node, and the reimported
    pipeline compiles and runs the per-item ensemble."""

    def test_round_trip_preserves_the_each_oracle_combo(self):
        imported = _import(to_agent_spec(_each_oracle_pipeline()))
        item = _by_name(imported)["fanned"]
        assert classify_modifiers(item)[0] is ModifierCombo.EACH_ORACLE, (
            "reimporting the fusion as plain EACH (or plain ORACLE) would satisfy "
            "every structure-shape assertion while silently dropping half of it"
        )

    def test_round_trip_preserves_both_modifier_specs(self):
        imported = _import(to_agent_spec(_each_oracle_pipeline()))
        item = _by_name(imported)["fanned"]

        each = item.modifier_set.each
        assert each is not None
        assert each.over == "seed.items"
        assert each.key == "label"

        oracle = item.modifier_set.oracle
        assert oracle is not None
        assert oracle.n == 2
        assert oracle.merge_fn == "eor_merge"

    def test_round_trip_preserves_the_fan_out_receiver_element_type(self):
        """neograph-3lk2l, inside the fusion: the reimported receiver's type
        must be the SAME synthesized class as the producer's ``list[X]``
        element -- derived from the FIRST VARIANT node's dotted input
        Properties, since the merge node's data edges carry the fan-in shape
        instead."""
        imported = _import(to_agent_spec(_each_oracle_pipeline()))
        by_name = _by_name(imported)
        producer, receiver_node = by_name["seed"], by_name["fanned"]

        assert receiver_node.fan_out_param == "cluster"
        receiver_type = receiver_node.inputs[receiver_node.fan_out_param]

        element_type = typing.get_args(producer.outputs.model_fields["items"].annotation)[0]
        assert receiver_type is element_type, (
            "the fan-out receiver must reconstruct to the producer's list element "
            "class; a flat {'cluster.label': ...} model would never match"
        )
        assert set(receiver_type.model_fields) == {"label"}

    def test_flow_from_dict_to_dict_round_trip_then_reimport(self):
        """R1: full serialization survival, not just the in-memory Flow."""
        flow = to_agent_spec(_each_oracle_pipeline())
        rebuilt = type(flow).from_dict(flow.to_dict())
        item = _by_name(_import(rebuilt))["fanned"]
        assert classify_modifiers(item)[0] is ModifierCombo.EACH_ORACLE
        assert item.modifier_set.oracle.n == 2

    def test_reimported_pipeline_compiles_and_runs_the_per_item_ensemble(self):
        _register()
        imported = _import(to_agent_spec(_each_oracle_pipeline()))

        fake_llm = StructuredFake(lambda m: m(items=["variant"]))
        graph = compile(
            imported,
            **build_test_compile_kwargs(),
            **build_fake_llm_kwargs(lambda tier: fake_llm),
        )
        result = run(graph, input={"node_id": "each-oracle-rt"})

        collected = result["fanned"]
        assert set(collected) == {"a", "b"}, collected
        assert len(collected["a"].items) == 2, "two variants merged per fanned-out item"
        assert len(collected["b"].items) == 2
