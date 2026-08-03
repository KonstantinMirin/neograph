"""Phase 7 / neograph-s7zt3.10 -- ModifierCombo.ORACLE_OPERATOR must survive an
Agent Spec round-trip (export AND import, landed in lockstep).

At HEAD ``_lower_construct_item``'s ORACLE arm raises
``ConfigurationError: node 'ensemble' has modifier combination ORACLE_OPERATOR
-- no Agent Spec lowering yet`` even though ``_lower_oracle`` and
``_lower_operator`` both exist; on the import side the walk's oracle branch
emits its group immediately with no trailing-operator lookahead, so the exported
``BranchingNode`` lands in the orphan branch and import dies.

Binding evidence rules (identical across the five Phase-7 combo modules):

  R1  full ``Flow.from_dict(flow.to_dict())`` survival is asserted, not the
      weaker in-memory-Flow-only path.
  R2  Operator needs a checkpointer to compile, hence
      ``build_test_compile_kwargs(checkpointer=MemorySaver())`` plus a
      deliberately-FALSE ``when``. **NOT CLAIMED**: pause/resume semantics of a
      HITL gate placed after an ensemble merge (or under a fan-out barrier);
      only that the modifier survives export -> import -> compile -> run.
  R3  ``classify_modifiers(<reimported item>)[0] is ModifierCombo.ORACLE_OPERATOR``
      is asserted explicitly -- a silent downgrade to plain ORACLE would pass
      every structural assertion.

Oracle round-trips only for "think" mode today (``_lower_oracle`` lowers
variants to ``LlmNode`` and the loader reconstructs ``mode="think"``), so this
fixture uses a think-mode node with a ``StructuredFake`` -- matching
``test_agent_spec_roundtrip.py::test_oracle_round_trips_and_runs``.
"""

from __future__ import annotations

import warnings

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import Construct, Node, Operator, Oracle, compile, run
from neograph._agent_spec import (
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    _MARK_ORACLE_SPEC,
    _PAUSE_BRANCH,
    to_agent_spec,
)
from neograph.loader import from_agent_spec
from neograph.modifiers import ModifierCombo, classify_modifiers
from neograph.testing.fakes import StructuredFake
from tests.fakes import (
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_condition,
    register_scripted,
)


class Claims(BaseModel, frozen=True):
    items: list[str]


def _register() -> None:
    def merge_fn(variants, config):
        # The reconstructed output type is a freshly-synthesized class, so the
        # merge operates on whatever type it actually receives.
        all_items: list[str] = []
        for v in variants:
            all_items.extend(v.items)
        return type(variants[0])(items=all_items)

    register_scripted("oo_merge", merge_fn)
    # Deliberately FALSE: the gate must not actually interrupt the run (R2).
    register_condition("oo_never", lambda state: None)


def _oracle_operator_pipeline() -> Construct:
    """A 3-variant ensemble whose merged result is gated by an Operator."""
    _register()
    return Construct(
        "oracle-operator",
        nodes=[
            Node(name="ensemble", mode="think", model="fast", outputs=Claims, prompt="rw/ensemble")
            | Oracle(n=3, merge_fn="oo_merge")
            | Operator(when="oo_never"),
        ],
    )


def _plain_oracle_pipeline() -> Construct:
    """The SAME pipeline without the gate -- the zero-behavior-change control."""
    _register()
    return Construct(
        "oracle-plain",
        nodes=[
            Node(name="ensemble", mode="think", model="fast", outputs=Claims, prompt="rw/ensemble")
            | Oracle(n=3, merge_fn="oo_merge"),
        ],
    )


def _import(flow):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_agent_spec(flow)


def _by_name(construct: Construct) -> dict[str, Node]:
    return {n.name: n for n in construct.nodes}


class TestOracleOperatorExportShape:
    """The variant fan-out + merge node AND the Operator pause composite must
    both appear -- composition of two lowerings that already exist."""

    def test_export_emits_variants_merge_and_pause_composite(self):
        flow = to_agent_spec(_oracle_operator_pipeline())
        by_name = {n.name: n for n in flow.nodes}

        variants = [n for n in flow.nodes if n.name.startswith("ensemble__variant_")]
        assert len(variants) == 3
        assert {type(v).__name__ for v in variants} == {"LlmNode"}

        merge = by_name["ensemble"]
        assert merge.metadata[_MARK_MODIFIER] == "oracle"
        assert merge.metadata[_MARK_ORACLE_SPEC]["n"] == 3
        assert merge.metadata[_MARK_ORACLE_SPEC]["merge_fn"] == "oo_merge"

        check = by_name["ensemble__operator_check"]
        assert type(check).__name__ == "BranchingNode"
        assert check.metadata[_MARK_MODIFIER] == "operator"
        assert check.metadata[_MARK_OPERATOR_SPEC] == {"when": "oo_never"}

        assert type(by_name["ensemble__operator_pause"]).__name__ == "InputMessageNode"

    def test_merge_node_chains_into_the_operator_check(self):
        """The postlude's pre-edge attaches to the ORACLE arm's EXIT (the merge
        node), leaving the variant chain untouched -- the last variant is what
        reaches the merge (neograph-s7zt3.15)."""
        flow = to_agent_spec(_oracle_operator_pipeline())
        pairs = {(e.from_node.name, e.to_node.name) for e in flow.control_flow_connections}
        assert ("ensemble", "ensemble__operator_check") in pairs
        assert ("ensemble__operator_check", "ensemble__operator_pause") in pairs
        assert ("ensemble__variant_0", "ensemble__variant_1") in pairs
        assert ("ensemble__variant_2", "ensemble") in pairs

        pause_edge = next(e for e in flow.control_flow_connections if e.to_node.name == "ensemble__operator_pause")
        assert pause_edge.from_branch == _PAUSE_BRANCH

    def test_plain_oracle_export_is_unchanged_when_ungated(self):
        """Zero behavior change: an Oracle node WITHOUT an Operator keeps
        today's variants + merge export -- no stray pause composite."""
        flow = to_agent_spec(_plain_oracle_pipeline())
        names = [n.name for n in flow.nodes]
        assert "ensemble" in names
        assert not [n for n in names if "operator" in n]


class TestOracleOperatorRoundTrip:
    """Export -> import recovers BOTH modifiers, and the reimported pipeline
    compiles and runs the ensemble.

    NOT CLAIMED (R2): pause/resume semantics of a HITL gate after an ensemble
    merge. The gate condition is deliberately false so the run completes.
    """

    def test_round_trip_preserves_the_oracle_operator_combo(self):
        imported = _import(to_agent_spec(_oracle_operator_pipeline()))
        item = _by_name(imported)["ensemble"]
        assert classify_modifiers(item)[0] is ModifierCombo.ORACLE_OPERATOR, (
            "a silent downgrade to plain ORACLE would satisfy every structure-shape "
            "assertion while dropping the human-approval gate"
        )

    def test_round_trip_preserves_both_modifier_specs(self):
        imported = _import(to_agent_spec(_oracle_operator_pipeline()))
        item = _by_name(imported)["ensemble"]

        oracle = item.modifier_set.oracle
        assert oracle is not None
        assert oracle.n == 3
        assert oracle.merge_fn == "oo_merge"

        operator = item.modifier_set.operator
        assert operator is not None
        assert operator.when == "oo_never"

    def test_flow_from_dict_to_dict_round_trip_then_reimport(self):
        """R1: full serialization survival, not just the in-memory Flow."""
        flow = to_agent_spec(_oracle_operator_pipeline())
        rebuilt = type(flow).from_dict(flow.to_dict())
        item = _by_name(_import(rebuilt))["ensemble"]
        assert classify_modifiers(item)[0] is ModifierCombo.ORACLE_OPERATOR
        assert item.modifier_set.operator.when == "oo_never"

    def test_reimported_pipeline_compiles_and_runs_the_ensemble(self):
        _register()
        imported = _import(to_agent_spec(_oracle_operator_pipeline()))

        fake_llm = StructuredFake(lambda m: m(items=["variant"]))
        graph = compile(
            imported,
            **build_test_compile_kwargs(checkpointer=MemorySaver()),
            **build_fake_llm_kwargs(lambda tier: fake_llm),
        )
        result = run(
            graph,
            input={"node_id": "oracle-operator-rt"},
            config={"configurable": {"thread_id": "oracle-operator-rt"}},
        )

        merged = result["ensemble"]
        assert len(merged.items) == 3, "three variants merged into one result"
        assert "__interrupt__" not in result, "the gate condition is false -- the run must not pause"
