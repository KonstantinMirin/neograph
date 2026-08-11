"""Phase 7 / neograph-s7zt3.10 -- ModifierCombo.EACH_OPERATOR must survive an
Agent Spec round-trip (export AND import, landed in lockstep).

At HEAD ``_lower_construct_item``'s EACH arm raises
``ConfigurationError: node 'each_step' has modifier combination EACH_OPERATOR --
no Agent Spec lowering yet`` even though BOTH halves already exist
(``_lower_each`` + ``_lower_operator``); the import side then has no
trailing-operator lookahead, so the exported ``BranchingNode`` lands in
``_group_flow_items``' orphan branch and import dies with "unsupported type
'BranchingNode'". This module pins the composed behavior on BOTH sides.

Structure copies ``tests/test_agent_spec_portal_operator.py`` (export-shape
asserts + a zero-behavior-change assert + a ``from_dict(to_dict())`` survival
test + a round-trip class) and the behavioral round-trip pattern of
``tests/test_agent_spec_roundtrip.py::TestEachOracleLoopRoundTripPreservesBehavior``.

The three evidence rules the ticket binds every new fixture to:

  R1  full ``Flow.from_dict(flow.to_dict())`` survival is asserted, not the
      weaker in-memory-Flow-only path.
  R2  an Operator-modified node needs a checkpointer to compile, so the run
      uses ``build_test_compile_kwargs(checkpointer=MemorySaver())`` plus a
      deliberately-FALSE ``when`` condition. **NOT CLAIMED**: pause/resume
      semantics of a HITL gate sitting behind a fan-out barrier. These
      fixtures claim only that the modifier survives export -> import ->
      compile -> run.
  R3  the reimported item's ``classify_modifiers(...)[0]`` is asserted to be
      exactly ``ModifierCombo.EACH_OPERATOR`` -- a structure-shape assert
      alone cannot tell a correct import from a silent downgrade to EACH.

Out of scope (pre-existing, neograph-s7zt3.15): the exported topology has the
incoming control edge target the check node rather than the body, and the pause
``InputMessageNode`` has no outgoing edge. Nothing here asserts foreign-runtime
executable semantics.
"""

from __future__ import annotations

import warnings

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import Construct, Each, Node, Operator, compile, run
from neograph._agent_spec import (
    _MARK_EACH_SPEC,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    Branch,
    to_agent_spec,
)
from neograph.loader import from_agent_spec
from neograph.modifiers import ModifierCombo, classify_modifiers
from tests.fakes import build_test_compile_kwargs, register_condition, register_scripted


class Tagged(BaseModel, frozen=True):
    label: str


class Bag(BaseModel, frozen=True):
    items: list[Tagged]


class Result(BaseModel, frozen=True):
    value: str


def _register() -> None:
    register_scripted("eo_seed", lambda input_data, config: Bag(items=[Tagged(label="a"), Tagged(label="b")]))
    register_scripted("eo_step", lambda input_data, config: Result(value=f"tagged-{input_data.label}"))
    # Deliberately FALSE: the gate must not actually interrupt the run (R2).
    register_condition("eo_never", lambda state: None)


def _each_operator_pipeline() -> Construct:
    """seed -> fan-out step, with a human-approval gate on the fan-out node."""
    _register()
    return Construct(
        "each-operator",
        nodes=[
            Node.scripted("seed", fn="eo_seed", outputs=Bag),
            Node.scripted("each_step", fn="eo_step", inputs=Tagged, outputs=Result)
            | Each(over="seed.items", key="label")
            | Operator(when="eo_never"),
        ],
    )


def _plain_each_pipeline() -> Construct:
    """The SAME pipeline without the gate -- the zero-behavior-change control."""
    _register()
    return Construct(
        "each-plain",
        nodes=[
            Node.scripted("seed", fn="eo_seed", outputs=Bag),
            Node.scripted("each_step", fn="eo_step", inputs=Tagged, outputs=Result)
            | Each(over="seed.items", key="label"),
        ],
    )


def _import(flow):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_agent_spec(flow)


def _by_name(construct: Construct) -> dict[str, Node]:
    return {n.name: n for n in construct.nodes}


class TestEachOperatorExportShape:
    """The Each MapNode and the Operator pause composite must BOTH appear --
    the composition of two lowerings that already exist independently."""

    def test_export_emits_map_node_and_pause_composite(self):
        flow = to_agent_spec(_each_operator_pipeline())
        by_name = {n.name: n for n in flow.nodes}

        assert type(by_name["each_step"]).__name__ == "MapNode"
        assert by_name["each_step"].metadata[_MARK_MODIFIER] == "each"
        assert by_name["each_step"].metadata[_MARK_EACH_SPEC]["over"] == "seed.items"

        check = by_name["each_step__operator_check"]
        assert type(check).__name__ == "BranchingNode"
        assert check.metadata[_MARK_MODIFIER] == "operator"
        assert check.metadata[_MARK_OPERATOR_SPEC] == {"when": "eo_never"}

        assert type(by_name["each_step__operator_pause"]).__name__ == "InputMessageNode"

    def test_map_node_chains_into_the_operator_check(self):
        """The postlude's pre-edge: the arm's primary (the MapNode) flows into
        the check, and the check's pause branch reaches the InputMessageNode."""
        flow = to_agent_spec(_each_operator_pipeline())
        pairs = {(e.from_node.name, e.to_node.name) for e in flow.control_flow_connections}
        assert ("each_step", "each_step__operator_check") in pairs
        assert ("each_step__operator_check", "each_step__operator_pause") in pairs

        pause_edge = next(e for e in flow.control_flow_connections if e.to_node.name == "each_step__operator_pause")
        assert pause_edge.from_branch == Branch.PAUSE

    def test_plain_each_export_is_unchanged_when_ungated(self):
        """Zero behavior change: an Each node WITHOUT an Operator keeps today's
        MapNode-only export -- no stray pause composite."""
        flow = to_agent_spec(_plain_each_pipeline())
        names = [n.name for n in flow.nodes]
        assert "each_step" in names
        assert not [n for n in names if "operator" in n]


class TestEachOperatorRoundTrip:
    """Export -> import recovers BOTH modifiers, and the reimported pipeline
    compiles and runs with the fan-out intact.

    NOT CLAIMED (R2): pause/resume semantics of a HITL gate under a fan-out
    barrier. The gate condition is deliberately false so the run completes;
    what is pinned is that the Operator survives at all.
    """

    def test_round_trip_preserves_the_each_operator_combo(self):
        imported = _import(to_agent_spec(_each_operator_pipeline()))
        item = _by_name(imported)["each_step"]
        assert classify_modifiers(item)[0] is ModifierCombo.EACH_OPERATOR, (
            "a silent downgrade to plain EACH would satisfy every structure-shape "
            "assertion while dropping the human-approval gate"
        )

    def test_round_trip_preserves_both_modifier_specs(self):
        imported = _import(to_agent_spec(_each_operator_pipeline()))
        item = _by_name(imported)["each_step"]

        each = item.modifier_set.each
        assert each is not None
        assert each.over == "seed.items"
        assert each.key == "label"

        operator = item.modifier_set.operator
        assert operator is not None
        assert operator.when == "eo_never"

    def test_flow_from_dict_to_dict_round_trip_preserves_the_operator_composite(self):
        """R1: full serialization survival.

        WIDENED back to the full ``_import(rebuilt)`` assertion (neograph-s7zt3.16
        landed). This test was temporarily narrowed to structure-only because
        ``Flow.from_dict`` erases Property SUBCLASSES (``ListProperty`` -> base
        ``Property``, json_schema intact) and the ``spec_types`` bridge dispatched
        on the class, so a ``list[Tagged]`` producer reimported as an opaque
        ``AgentSpecType_<hash>`` and Each's element-identity check
        (neograph-3lk2l) rejected it. The bridge now normalizes an erased Property
        before dispatching, so the reimport works and R3 (the combo must survive as
        EACH_OPERATOR, not silently downgrade to EACH) is assertable again.
        """
        flow = to_agent_spec(_each_operator_pipeline())
        rebuilt = type(flow).from_dict(flow.to_dict())

        # The Operator composite -- the thing THIS ticket added -- survives the
        # full JSON round trip: gate node, its marker, and the pause target.
        names = [n.name for n in rebuilt.nodes]
        assert "each_step" in names, names
        assert "each_step__operator_check" in names, names
        check = next(n for n in rebuilt.nodes if n.name == "each_step__operator_check")
        assert check.metadata[_MARK_OPERATOR_SPEC]["when"] == "eo_never"
        assert any(
            e.from_node.name == "each_step" and e.to_node.name == "each_step__operator_check"
            for e in rebuilt.control_flow_connections
        ), "the MapNode must still chain into the gate after serialization"

        # R3: the serialized-and-reimported item is still the COMPOSITE. A
        # structure-shape assert alone cannot tell a correct import from a silent
        # downgrade to plain EACH.
        item = _by_name(_import(rebuilt))["each_step"]
        assert classify_modifiers(item)[0] is ModifierCombo.EACH_OPERATOR

    def test_reimported_pipeline_compiles_and_runs_with_fan_out_intact(self):
        _register()
        imported = _import(to_agent_spec(_each_operator_pipeline()))

        graph = compile(imported, **build_test_compile_kwargs(checkpointer=MemorySaver()))
        result = run(
            graph,
            input={"node_id": "each-operator-rt"},
            config={"configurable": {"thread_id": "each-operator-rt"}},
        )

        collected = result["each_step"]
        assert set(collected) == {"a", "b"}, collected
        assert collected["a"].value == "tagged-a"
        assert collected["b"].value == "tagged-b"
        assert "__interrupt__" not in result, "the gate condition is false -- the run must not pause"
