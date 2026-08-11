"""Phase 7 / neograph-s7zt3.10 -- ModifierCombo.EACH_ORACLE_OPERATOR (the
three-way Each x Oracle fusion PLUS a human-approval gate) must survive an
Agent Spec round-trip (export AND import, in lockstep).

At HEAD ``_lower_construct_item``'s EACH arm rejects it outright
(``ConfigurationError: node 'fanned' has modifier combination
EACH_ORACLE_OPERATOR -- no Agent Spec lowering yet``) and the import walk has
neither the ``MapNode.subflow`` descent nor the trailing-operator lookahead.

By design this combo needs NO third code path: the unconditional Operator
postlude composes with the fused Each x Oracle arm, and the shared
``_trailing_operator`` lookahead folds the pause composite onto the fused
MapNode. This module is therefore the proof that the composition actually
composes -- all THREE modifiers on ONE node, out the far side of a round-trip.

Binding evidence rules:

  R1  full ``Flow.from_dict(flow.to_dict())`` survival is asserted.
  R2  Operator needs a checkpointer to compile, hence
      ``build_test_compile_kwargs(checkpointer=MemorySaver())`` plus a
      deliberately-FALSE ``when``. **NOT CLAIMED**: pause/resume semantics of a
      HITL gate under a fan-out barrier -- the hardest unclaimed case of the
      five, and deliberately out of scope. What is claimed is only that the
      modifier survives export -> import -> compile -> run.
  R3  ``classify_modifiers(<reimported item>)[0] is
      ModifierCombo.EACH_ORACLE_OPERATOR`` is asserted explicitly: with three
      modifiers there are three distinct silent-downgrade targets
      (EACH_ORACLE, EACH_OPERATOR, EACH), every one of which would satisfy a
      structure-only assertion.
"""

from __future__ import annotations

import warnings

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import Construct, Each, Node, Operator, Oracle, compile, run
from neograph._agent_spec import (
    _MARK_EACH_SPEC,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    _MARK_ORACLE_SPEC,
    _MARK_VARIANT,
    Branch,
    to_agent_spec,
)
from neograph.loader import from_agent_spec
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.testing.fakes import StructuredFake
from tests.agent_spec_flow_walk import arm_targets, edge_pairs, inner_nodes
from tests.fakes import (
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_condition,
    register_scripted,
)


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

    register_scripted("eoo_seed", lambda input_data, config: Bag(items=[Tagged(label="a"), Tagged(label="b")]))
    register_scripted("eoo_merge", merge_fn)
    # Deliberately FALSE: the gate must not actually interrupt the run (R2).
    register_condition("eoo_never", lambda state: None)


def _each_oracle_operator_pipeline() -> Construct:
    """seed -> a fanned-out per-item ensemble, gated by an Operator."""
    _register()
    return Construct(
        "each-oracle-operator",
        nodes=[
            Node.scripted("seed", fn="eoo_seed", outputs=Bag),
            Node(
                name="fanned",
                mode="think",
                model="fast",
                prompt="rw/fanned",
                inputs={"cluster": Tagged},
                outputs=Claims,
            )
            | Oracle(n=2, merge_fn="eoo_merge")
            | Each(over="seed.items", key="label")
            | Operator(when="eoo_never"),
        ],
    )


def _import(flow):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_agent_spec(flow)


def _by_name(construct: Construct) -> dict[str, Node]:
    return {n.name: n for n in construct.nodes}



class TestEachOracleOperatorExportShape:
    """The fused MapNode AND the Operator pause composite, together -- the
    postlude must apply to the fused arm exactly as it does to every other."""

    def test_export_emits_the_fused_map_node_plus_the_pause_composite(self):
        flow = to_agent_spec(_each_oracle_operator_pipeline())
        by_name = {n.name: n for n in flow.nodes}

        map_node = by_name["fanned"]
        assert type(map_node).__name__ == "MapNode"
        assert map_node.metadata[_MARK_MODIFIER] == "each"
        assert map_node.metadata[_MARK_EACH_SPEC]["over"] == "seed.items"

        inner = inner_nodes(map_node)
        assert len([n for n in inner if _MARK_VARIANT in (n.metadata or {})]) == 2
        merge = next(n for n in inner if _MARK_ORACLE_SPEC in (n.metadata or {}))
        assert merge.metadata[_MARK_ORACLE_SPEC]["n"] == 2

        check = by_name["fanned__operator_check"]
        assert type(check).__name__ == "BranchingNode"
        assert check.metadata[_MARK_MODIFIER] == "operator"
        assert check.metadata[_MARK_OPERATOR_SPEC] == {"when": "eoo_never"}

        assert type(by_name["fanned__operator_pause"]).__name__ == "InputMessageNode"

    def test_fused_map_node_chains_into_the_operator_check(self):
        flow = to_agent_spec(_each_oracle_operator_pipeline())
        pairs = edge_pairs(flow)
        assert ("fanned", "fanned__operator_check") in pairs
        assert ("fanned__operator_check", "fanned__operator_pause") in pairs

        assert "fanned__operator_pause" in arm_targets(flow, "fanned__operator_check", Branch.PAUSE)


class TestEachOracleOperatorRoundTrip:
    """Export -> import recovers ALL THREE modifiers on ONE node.

    NOT CLAIMED (R2): pause/resume semantics of a HITL gate under a fan-out
    barrier. The gate condition is deliberately false so the run completes.
    """

    def test_round_trip_preserves_the_each_oracle_operator_combo(self):
        imported = _import(to_agent_spec(_each_oracle_operator_pipeline()))
        item = _by_name(imported)["fanned"]
        assert classify_modifiers(item)[0] is ModifierCombo.EACH_ORACLE_OPERATOR, (
            "three modifiers means three silent-downgrade targets (EACH_ORACLE, "
            "EACH_OPERATOR, EACH) -- only the combo assertion rules all three out"
        )

    def test_round_trip_preserves_all_three_modifier_specs(self):
        imported = _import(to_agent_spec(_each_oracle_operator_pipeline()))
        item = _by_name(imported)["fanned"]

        each = item.modifier_set.each
        assert each is not None
        assert each.over == "seed.items"
        assert each.key == "label"

        oracle = item.modifier_set.oracle
        assert oracle is not None
        assert oracle.n == 2
        assert oracle.merge_fn == "eoo_merge"

        operator = item.modifier_set.operator
        assert operator is not None
        assert operator.when == "eoo_never"

    def test_flow_from_dict_to_dict_round_trip_then_reimport(self):
        """R1: full serialization survival, not just the in-memory Flow."""
        flow = to_agent_spec(_each_oracle_operator_pipeline())
        rebuilt = type(flow).from_dict(flow.to_dict())
        item = _by_name(_import(rebuilt))["fanned"]
        assert classify_modifiers(item)[0] is ModifierCombo.EACH_ORACLE_OPERATOR
        assert item.modifier_set.operator.when == "eoo_never"

    def test_reimported_pipeline_compiles_and_runs_the_gated_per_item_ensemble(self):
        _register()
        imported = _import(to_agent_spec(_each_oracle_operator_pipeline()))

        fake_llm = StructuredFake(lambda m: m(items=["variant"]))
        graph = compile(
            imported,
            **build_test_compile_kwargs(checkpointer=MemorySaver()),
            **build_fake_llm_kwargs(lambda tier: fake_llm),
        )
        result = run(
            graph,
            input={"node_id": "each-oracle-operator-rt"},
            config={"configurable": {"thread_id": "each-oracle-operator-rt"}},
        )

        collected = result["fanned"]
        assert set(collected) == {"a", "b"}, collected
        assert len(collected["a"].items) == 2, "two variants merged per fanned-out item"
        assert "__interrupt__" not in result, "the gate condition is false -- the run must not pause"
