"""Phase 7 / neograph-s7zt3.10 -- ModifierCombo.LOOP_OPERATOR must survive an
Agent Spec round-trip (export AND import, landed in lockstep).

At HEAD ``_lower_construct_item``'s LOOP arm raises
``ConfigurationError: node 'refine' has modifier combination LOOP_OPERATOR --
no Agent Spec lowering yet`` even though ``_lower_loop`` and ``_lower_operator``
both exist; on the import side the walk's bare+loop lookahead consumes exactly
two nodes and never continues into a trailing operator composite, so the
``BranchingNode`` falls into the orphan branch and import dies.

Same three binding evidence rules as the sibling combo modules:

  R1  full ``Flow.from_dict(flow.to_dict())`` survival is asserted.
      This fixture uses the single-type ``inputs=Draft`` form -- exactly what
      ``test_agent_spec_roundtrip.py::test_loop_round_trips_and_runs`` uses.
      It was originally forced: a DICT-FORM body then exported dotted Property
      titles that pyagentspec's deserializer rejected. neograph-8zvd1 fixed that
      (titles are qualified with ``:`` at construction), and dict-form loop
      bodies now survive the wire -- pinned by
      ``test_agent_spec_roundtrip.py::TestExportedPropertyTitlesAreAgentSpecLegal``.
  R2  Operator needs a checkpointer to compile, hence
      ``build_test_compile_kwargs(checkpointer=MemorySaver())`` plus a
      deliberately-FALSE ``when``. **NOT CLAIMED**: pause/resume semantics of a
      HITL gate wrapped around a loop (or under a fan-out barrier); only that
      the modifier survives export -> import -> compile -> run.
  R3  ``classify_modifiers(<reimported item>)[0] is ModifierCombo.LOOP_OPERATOR``
      is asserted explicitly -- a silent downgrade to plain LOOP would pass
      every structural assertion.

Out of scope (pre-existing, neograph-s7zt3.15): the exported LOOP topology
already makes the check ``BranchingNode`` the item's primary, so the incoming
control edge bypasses the body. Nothing here asserts foreign-runtime semantics.
"""

from __future__ import annotations

import warnings

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import Construct, Loop, Node, Operator, compile, node, run
from neograph._agent_spec import (
    _MARK_LOOP_SPEC,
    _MARK_MODIFIER,
    _MARK_OPERATOR_SPEC,
    Branch,
    to_agent_spec,
)
from neograph.decorators import construct_from_functions
from neograph.loader import from_agent_spec
from neograph.modifiers import ModifierCombo, classify_modifiers
from tests.fakes import build_test_compile_kwargs, register_condition, register_scripted


class Draft(BaseModel, frozen=True):
    content: str
    iteration: int
    score: float


_CALLS: list[int] = []


def _register() -> None:
    _CALLS.clear()

    def seed_fn(input_data, config):
        return Draft(content="v0", iteration=0, score=0.0)

    def refine_fn(input_data, config):
        # from_agent_spec always reconstructs inputs as dict-form, so the
        # upstream/reentry value arrives under the "seed" key.
        _CALLS.append(1)
        prev = input_data["seed"] if isinstance(input_data, dict) else input_data
        return Draft(content=f"v{len(_CALLS)}", iteration=prev.iteration + 1, score=prev.score + 0.3)

    register_scripted("lo_seed", seed_fn)
    register_scripted("lo_refine", refine_fn)
    # Deliberately FALSE: the gate must not actually interrupt the run (R2).
    register_condition("lo_never", lambda state: None)


def _loop_operator_pipeline() -> Construct:
    """seed -> refine (self-loop until score >= 0.8), gated by an Operator."""
    _register()
    return Construct(
        "loop-operator",
        nodes=[
            Node.scripted("seed", fn="lo_seed", outputs=Draft),
            Node.scripted("refine", fn="lo_refine", inputs=Draft, outputs=Draft)
            | Loop(when="score < 0.8", max_iterations=10)
            | Operator(when="lo_never"),
        ],
    )


def _plain_loop_pipeline() -> Construct:
    """The SAME pipeline without the gate -- the zero-behavior-change control."""
    _register()
    return Construct(
        "loop-plain",
        nodes=[
            Node.scripted("seed", fn="lo_seed", outputs=Draft),
            Node.scripted("refine", fn="lo_refine", inputs=Draft, outputs=Draft)
            | Loop(when="score < 0.8", max_iterations=10),
        ],
    )


def _import(flow):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_agent_spec(flow)


def _by_name(construct: Construct) -> dict[str, Node]:
    return {n.name: n for n in construct.nodes}


class TestLoopOperatorExportShape:
    """The loop body + loop check + the Operator pause composite must ALL
    appear -- composition of two lowerings that already exist."""

    def test_export_emits_loop_check_and_pause_composite(self):
        flow = to_agent_spec(_loop_operator_pipeline())
        by_name = {n.name: n for n in flow.nodes}

        loop_check = by_name["refine__loop_check"]
        assert type(loop_check).__name__ == "BranchingNode"
        assert loop_check.metadata[_MARK_MODIFIER] == "loop"
        assert loop_check.metadata[_MARK_LOOP_SPEC]["when"] == "score < 0.8"
        assert loop_check.metadata[_MARK_LOOP_SPEC]["max_iterations"] == 10

        check = by_name["refine__operator_check"]
        assert type(check).__name__ == "BranchingNode"
        assert check.metadata[_MARK_MODIFIER] == "operator"
        assert check.metadata[_MARK_OPERATOR_SPEC] == {"when": "lo_never"}

        assert type(by_name["refine__operator_pause"]).__name__ == "InputMessageNode"

    def test_loop_check_chains_into_the_operator_check(self):
        """The postlude's pre-edge attaches to the LOOP arm's primary (the loop
        check), not to the body -- the arm's own primary is unchanged."""
        flow = to_agent_spec(_loop_operator_pipeline())
        pairs = {(e.from_node.name, e.to_node.name) for e in flow.control_flow_connections}
        assert ("refine__loop_check", "refine__operator_check") in pairs
        assert ("refine__operator_check", "refine__operator_pause") in pairs

        pause_edge = next(e for e in flow.control_flow_connections if e.to_node.name == "refine__operator_pause")
        assert pause_edge.from_branch == Branch.PAUSE

    def test_plain_loop_export_is_unchanged_when_ungated(self):
        """Zero behavior change: a Loop node WITHOUT an Operator keeps today's
        body + check export -- no stray pause composite."""
        flow = to_agent_spec(_plain_loop_pipeline())
        names = [n.name for n in flow.nodes]
        assert "refine__loop_check" in names
        assert not [n for n in names if "operator" in n]


class TestLoopOperatorRoundTrip:
    """Export -> import recovers BOTH modifiers, and the reimported pipeline
    compiles and runs the loop to completion.

    NOT CLAIMED (R2): pause/resume semantics of a HITL gate wrapped around a
    loop. The gate condition is deliberately false so the run completes.
    """

    def test_round_trip_preserves_the_loop_operator_combo(self):
        imported = _import(to_agent_spec(_loop_operator_pipeline()))
        item = _by_name(imported)["refine"]
        assert classify_modifiers(item)[0] is ModifierCombo.LOOP_OPERATOR, (
            "a silent downgrade to plain LOOP would satisfy every structure-shape "
            "assertion while dropping the human-approval gate"
        )

    def test_round_trip_preserves_both_modifier_specs(self):
        imported = _import(to_agent_spec(_loop_operator_pipeline()))
        item = _by_name(imported)["refine"]

        loop = item.modifier_set.loop
        assert loop is not None
        assert loop.max_iterations == 10

        operator = item.modifier_set.operator
        assert operator is not None
        assert operator.when == "lo_never"

    def test_flow_from_dict_to_dict_round_trip_then_reimport(self):
        """R1: full serialization survival, not just the in-memory Flow."""
        flow = to_agent_spec(_loop_operator_pipeline())
        rebuilt = type(flow).from_dict(flow.to_dict())
        item = _by_name(_import(rebuilt))["refine"]
        assert classify_modifiers(item)[0] is ModifierCombo.LOOP_OPERATOR
        assert item.modifier_set.operator.when == "lo_never"

    def test_reimported_pipeline_compiles_and_runs_the_loop_to_completion(self):
        _register()
        imported = _import(to_agent_spec(_loop_operator_pipeline()))

        graph = compile(imported, **build_test_compile_kwargs(checkpointer=MemorySaver()))
        result = run(
            graph,
            input={"node_id": "loop-operator-rt"},
            config={"configurable": {"thread_id": "loop-operator-rt"}},
        )

        assert len(_CALLS) == 3, "0.0 -> 0.3 -> 0.6 -> 0.9: the loop body runs exactly three times"
        value = result["refine"]
        final = value[-1] if isinstance(value, list) else value
        assert final.score >= 0.8
        assert final.iteration == 3


class TestNodeLevelLoopRegisteredNameRoundTrip:
    """neograph-d3x4j: the NODE-level sibling of this file's Operator
    registered-name coverage (``test_round_trip_preserves_both_modifier_specs``
    above asserts ``operator.when == "lo_never"`` after reimport; Loop's own
    ``when`` had no Node-level pin, only the Construct-level
    ``_loop_matching_boundary`` row in
    ``test_agent_spec_construct_item_roundtrip.py``, proven only indirectly via
    a successful compile).

    ``_reconstruct_loop_item`` (``_agent_spec_group_import.py``) is the single
    shared reconstruction path for Loop on both Node- and Construct-level items
    (neograph-ijyjr / commit 110e8d6), so this was expected to already work --
    this test pins it directly on a plain ``@node(loop_when=...)`` fixture with
    a direct string assertion, mirroring how the Operator sibling asserts its
    ``when`` above.
    """

    def test_node_level_loop_registered_name_survives_round_trip(self):
        # Node names deliberately underscore-free (plain "seed"/"refine") --
        # @node infers DICT-FORM inputs from the function signature
        # (inputs={"seed": Draft}), and the exporter's dict-form fan-in
        # resolves that key against the upstream item's literal (hyphenated)
        # .name; an underscored multi-word name would hyphenate and mismatch
        # the key, which is an orthogonal pre-existing export limitation this
        # ticket does not touch.
        register_condition("d3x4j_continue", lambda draft: draft is None or draft.score < 0.8)

        @node(outputs=Draft)
        def seed() -> Draft:
            return Draft(content="v0", iteration=0, score=0.0)

        @node(outputs=Draft, loop_when="d3x4j_continue", max_iterations=10)
        def refine(seed: Draft) -> Draft:
            return Draft(content=f"v{seed.iteration + 1}", iteration=seed.iteration + 1, score=seed.score + 0.3)

        pipeline = construct_from_functions("d3x4j-node-loop", [seed, refine])
        flow = to_agent_spec(pipeline)
        imported = _import(flow)

        item = _by_name(imported)["refine"]
        loop = item.modifier_set.loop
        assert loop is not None
        assert loop.when == "d3x4j_continue", (
            "the registered NAME must survive reimport unresolved (matching "
            "how the Operator sibling test asserts operator.when == 'lo_never' "
            "above) -- compile() resolves it, not the importer"
        )

        def seed_fn(input_data, config):
            return Draft(content="v0", iteration=0, score=0.0)

        def refine_fn(input_data, config):
            prev = input_data["seed"] if isinstance(input_data, dict) else input_data
            return Draft(content=f"v{prev.iteration + 1}", iteration=prev.iteration + 1, score=prev.score + 0.3)

        register_scripted("seed", seed_fn)
        register_scripted("refine", refine_fn)

        graph = compile(imported, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "d3x4j-node-loop-rt"})

        value = result["refine"]
        final = value[-1] if isinstance(value, list) else value
        assert final.score >= 0.8
        assert final.iteration == 3, "0.0 -> 0.3 -> 0.6 -> 0.9: the registered condition resolved and drove the loop"
