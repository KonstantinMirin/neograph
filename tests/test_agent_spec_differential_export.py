"""DIFFERENTIAL-EXPORT HARNESS: a change in RESOLUTION must change the export.

Core Invariant (neograph-9axw6.1, plan item 5): two constructs that differ in
exactly one ADDRESS -- which upstream value a demand resolves to -- must produce
exports that differ in the RESOLUTION-BEARING part of the document, and two that
differ in anything else must not. The epic's section-15 success claim is
"exporter behaviour changes anyway, proven by the harness"; this module is that
harness, and it is only worth the name if it can tell those two cases apart.

## Why the assertion is a PROJECTION and not the whole document

The obvious form -- ``export(A) != export(B)`` over the whole canonicalized
document -- is a FALSE-GREEN GENERATOR for exactly the claim this file exists to
support. MEASURED on the ``output_from`` pair below (a two-member sub-construct
declaring ``output=Claims`` with ``output_from='a'`` vs ``'b'``):

  * the two exported documents DIFFER -- in exactly one key,
    ``metadata['neograph/boundary_spec']``, an INERT marker;
  * the two data-flow edge sets are IDENTICAL: BOTH wire
    ``('boundary', 'b', 'items', 'boundary__end', 'items')``, including the
    construct that declared ``output_from='a'``.

So a whole-document assertion reports "the exporter distinguishes these" while
the wire still names the wrong producer -- which IS the live neograph-fnlrx /
neograph-kgndo defect. The assertion therefore reads a projection:

  * ``to_agent_spec`` -> the canonical data-flow EDGE SET (every flow, every
    sub-flow), via the single sanctioned readers ``agent_spec_flow_walk``;
  * ``dump_spec``     -> the dumped node/construct declarations with LIST ORDER
    projected out, because order is the one thing a resolution change
    incidentally moves in a dump (measured below).

## Why there is a NEGATIVE control

Two constructs that resolve differently must ALWAYS differ somewhere else too
(the swap that changes the resolution also changes declaration order), so "these
exports differ" is not evidence about the resolution unless the harness also
proves it stays QUIET on a difference that is not one. The negative control
applies the IDENTICAL perturbation -- ``b`` and ``sink`` trade places -- to a
corpus where ``b``'s output type cannot satisfy ``sink``'s demand, so no
resolution moves. It asserts BOTH halves: the whole document genuinely differs
(else the control proves nothing) and the projection is EQUAL.

## The cells, each measured rather than assumed

| axis                 | format         | state                              |
|----------------------|----------------|------------------------------------|
| ``input_source_field`` | ``to_agent_spec`` | GREEN -- the non-vacuity proof  |
| ``input_source_field`` | ``dump_spec``     | DROPPED (see ``_DROPPED_CELLS``) |
| ``output_from``        | ``to_agent_spec`` | ``xfail(strict=True)``           |
| ``output_from``        | ``dump_spec``     | ``xfail(strict=True)``           |

``xfail`` is PER-CELL, never per-axis. A per-axis mark would XPASS on
``(output_from, to_agent_spec)`` at the document level and turn the gate RED on
step 0, whose entire point is "changes no behaviour". Never ``skip``:
``scripts/check_skips.py`` fails on ANY skip and has no allowlist; ``xfail`` is
reported distinctly and is not a skip.

Both xfails are SELF-REMOVING, but NOT at the step first written here. The
original text credited "step 2 (neograph-9axw6.2)", conflating the ticket's child
suffix with the step number: ``neograph-9axw6.2`` is STEP 1, and step 1 only adds
an assembly-time REFUSAL -- a type mismatch on a named port, or a multi-output
member named without a port key. It does not touch either exporter's edge
derivation.

It also cannot refuse THIS pair, which is why the pair still measures red after
step 1 lands: both members produce ``Claims``, so both are type-eligible and
``output_from`` is the only thing naming which is meant. Step 1 has nothing to
object to.

The remover is STEP 2 = ``neograph-9axw6.3`` ("stamp the boundary address and read
it from both exporters"). When the exporters read the stamped address, the edge
set follows the declared port, ``(output_from, to_agent_spec)`` XPASSes,
strict=True fails, and the exemption must be deleted. That is the ratchet, and it
is why the cell is written as a live red rather than omitted.

## The leg this harness deliberately does NOT have yet

EXPORT-ONLY grading is insufficient: the roadmap-validation amendment
(``docs/design/ir-ownership-roadmap-validation-2026-08-27.md``:118-121) records
that "the loader can silently undo a distinction the dumper preserves". The
missing leg is IMPORT re-normalisation equality --
``load_spec(dump_spec(x))`` and ``from_agent_spec(to_agent_spec(x))`` re-exported
and compared in the same projection. **STEP 6 (neograph-9axw6.7) adds it**, and
the ``_FORMATS`` registry below is shaped so it lands as one added ROW (a third
``_Format`` whose ``document``/``project`` route through the importer first), not
as a rewrite. Deferring it is defensible; forgetting it is not.

## Corpus

``tests/schemas.py`` models plus its ``_producer``/``_consumer`` helpers -- the
same corpus ``tests/test_single_type_input_resolution.py`` and
``tests/test_agent_spec_matrix.py``'s ``build_cell`` draw on. No third corpus is
minted. Every build is MULTI-NODE (four members at the level under test) so a
positional rule cannot agree with a resolution by coincidence, and no build
needs ``register_scripted``: the export path never resolves a scripted body
(measured -- both formats export cleanly with ``fn='f'`` unregistered).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from neograph import Construct, dump_spec, to_agent_spec
from neograph._spec_dump import LOSSES_KEY
from neograph._spec_schema import ConstructSpec
from tests.agent_spec_canonical import _canonicalize
from tests.agent_spec_flow_walk import all_flows, wired_edges
from tests.schemas import Claims, MergedResult, RawText, ValidationResult, _consumer, _producer

# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------


def _agent_spec_document(construct: Construct) -> Any:
    """The WHOLE exported Flow, folded deterministic (UUIDs resolved and dropped).

    ``Flow.model_dump(mode='json')`` RAISES, so ``_canonicalize`` is the only
    working comparison key -- see ``tests/agent_spec_canonical.py``.
    """
    return _canonicalize(to_agent_spec(construct))


def _agent_spec_edges(construct: Construct) -> list[tuple[Any, ...]]:
    """The RESOLUTION-BEARING projection: every data-flow edge, every flow.

    Reads through ``agent_spec_flow_walk``'s ``wired_edges`` -- the single
    sanctioned ``data_flow_connections`` reader, capped tree-wide by
    ``tests/test_guards_agent_spec_data_flow_reads.py``. The flow name is carried
    so a sub-flow's boundary edge is not confused with the parent's, and the
    result is sorted because "which edges exist" is the question, not "in what
    order the exporter emitted them".
    """
    flow = to_agent_spec(construct)
    return sorted((f.name, *edge) for f in all_flows(flow) for edge in wired_edges(f))


def _dumped_declarations(construct: Construct) -> dict[str, Any]:
    """``dump_spec`` with LIST ORDER projected out.

    Order must go, and the measurement is the reason: an ``input_source_field``
    pair's two dumps differ ONLY in ``nodes[]`` / ``pipeline.nodes`` order while
    the consumer's own dumped dict is byte-identical. Comparing raw dumps would
    therefore report ORDER as if it were the resolution -- the same
    incidental-difference failure the whole-document agent-spec assertion has.
    What survives the fold is exactly the declarations: each node's dumped body
    and each sub-construct's boundary (``name``/``input``/``output``/members).
    """
    spec = dump_spec(construct)
    return {
        "types": spec["types"],
        "nodes": {n["name"]: n for n in spec["nodes"]},
        "constructs": {c["name"]: c for c in spec["constructs"]},
        "pipeline": sorted(spec["pipeline"]["nodes"]),
        LOSSES_KEY: spec[LOSSES_KEY],
    }


@dataclass(frozen=True)
class _Format:
    """One export format: its whole document, and its resolution-bearing view.

    Both halves are needed. ``project`` is what a cell asserts on; ``document``
    is what the negative control uses to prove its pair differs AT ALL, so a
    control that is silently comparing two identical constructs cannot pass.

    Step 6's import leg is a third instance of this dataclass, not a rewrite.
    """

    name: str
    document: Callable[[Construct], Any]
    project: Callable[[Construct], Any]
    projection_is: str


_FORMATS: dict[str, _Format] = {
    fmt.name: fmt
    for fmt in (
        _Format(
            name="to_agent_spec",
            document=_agent_spec_document,
            project=_agent_spec_edges,
            projection_is="the canonical data-flow edge set",
        ),
        _Format(
            name="dump_spec",
            document=dump_spec,
            project=_dumped_declarations,
            projection_is="the order-free dumped node and sub-construct declarations",
        ),
    )
}


# ---------------------------------------------------------------------------
# Corpus: pairs differing in exactly one address
# ---------------------------------------------------------------------------


def _single_type_pair(*, b_output: type) -> tuple[Construct, Construct]:
    """Four members; ``b`` and ``sink`` trade places between A and B.

    ``sink`` declares the single-type shorthand ``inputs=Claims``, whose one
    answer is ``_ir_normalize.resolve_single_type_source`` -- the LAST compatible
    producer declared before it. So the swap moves ``sink``'s address iff ``b``'s
    output can satisfy ``Claims``:

      * ``b_output=Claims``     -> A resolves ``sink`` to ``b``, B to ``a``. The
        POSITIVE pair: exactly one address differs.
      * ``b_output=ValidationResult`` -> both resolve ``sink`` to ``a``. The
        NEGATIVE control: the same perturbation, no address moves.

    ``tail`` is last in BOTH orders on purpose. Without it the swap would also
    move which node feeds the flow's End boundary, adding a second difference to
    the projection and muddying what the positive cell proves. With it, the
    measured projection delta is exactly one edge:
    ``('isf', 'b', 'items', 'sink', 'items')`` vs ``('isf', 'a', 'items', 'sink',
    'items')``.
    """
    a = _producer("a", Claims)
    b = _consumer("b", Claims, b_output)
    sink = _consumer("sink", Claims, MergedResult)
    tail = _consumer("tail", MergedResult, ValidationResult)
    return (
        Construct("isf", nodes=[a, b, sink, tail]),
        Construct("isf", nodes=[a, sink, b, tail]),
    )


def _input_source_field_pair() -> tuple[Construct, Construct]:
    return _single_type_pair(b_output=Claims)


def _incidental_pair() -> tuple[Construct, Construct]:
    return _single_type_pair(b_output=ValidationResult)


def _output_from_pair() -> tuple[Construct, Construct]:
    """A nested sub-construct whose declared boundary port is ``a`` vs ``b``.

    Both members produce ``Claims``, so both are ELIGIBLE for the ``output=``
    boundary and ``output_from=`` is the only thing naming which one is meant.
    Nested rather than top-level because that is where BOTH formats have
    something to say: ``_spec_dump._dump_sub_construct`` is the function that
    drops ``output_from``, and the sub-flow is where the boundary edge is wired.

    MEASURED today, and this is the defect the harness exists to detect: both
    exports wire ``('boundary', 'b', 'items', 'boundary__end', 'items')`` -- the
    positional last-eligible-member rule -- including the construct that declared
    ``output_from='a'``.
    """

    def build(port: str) -> Construct:
        sub = Construct(
            "boundary",
            input=RawText,
            output=Claims,
            output_from=port,
            nodes=[_consumer("a", RawText, Claims), _consumer("b", Claims, Claims)],
        )
        return Construct("of", nodes=[_producer("seed", RawText), sub, _consumer("tail", Claims, MergedResult)])

    return build("a"), build("b")


@dataclass(frozen=True)
class _Axis:
    """One ADDRESS the pair differs in, and what resolves it."""

    name: str
    pair: Callable[[], tuple[Construct, Construct]]
    resolved_by: str


_AXES: dict[str, _Axis] = {
    axis.name: axis
    for axis in (
        _Axis(
            name="input_source_field",
            pair=_input_source_field_pair,
            resolved_by="_ir_normalize.resolve_single_type_source, stamped onto Node.input_source_field",
        ),
        _Axis(
            name="output_from",
            pair=_output_from_pair,
            resolved_by="Construct.output_from, the USER-declared boundary port",
        ),
    )
}


# ---------------------------------------------------------------------------
# The cell table
# ---------------------------------------------------------------------------

#: Cells the harness asserts on, each ``(axis, export_format)``.
_CELLS: tuple[tuple[str, str], ...] = (
    ("input_source_field", "to_agent_spec"),
    ("output_from", "to_agent_spec"),
    ("output_from", "dump_spec"),
)

#: Cells that are live RED today, with the measurement that makes them red.
#: PER-CELL, never per-axis -- a per-axis mark on ``output_from`` would XPASS at
#: the document level and turn step 0's gate red.
_XFAIL_CELLS: dict[tuple[str, str], str] = {
    ("output_from", "to_agent_spec"): (
        "the exporter's edge derivation ignores output_from: both variants wire "
        "('boundary', 'b', 'items', 'boundary__end', 'items') from the positional "
        "last-eligible-member rule. The only document-level difference is the inert "
        "metadata['neograph/boundary_spec'] marker. neograph-9axw6.2 (step 2) closes it."
    ),
    ("output_from", "dump_spec"): (
        "_spec_dump._dump_sub_construct iterates ('input', 'output') only and never "
        "emits output_from, although _spec_schema.ConstructSpec declares the field "
        "(:157) and _spec_loader reads it back (:271). The two dumps are BYTE-identical, "
        "not merely equal under the projection."
    ),
}

#: Cells deliberately NOT asserted, with the measurement that disqualified them.
#: Kept in the table rather than deleted so the next reader does not re-add one
#: believing it proves something.
_DROPPED_CELLS: dict[tuple[str, str], str] = {
    ("input_source_field", "dump_spec"): (
        "dump_spec serializes input_source_field NOWHERE (it appears only in "
        "_agent_spec.py). MEASURED: the two dumps differ only in nodes[] and "
        "pipeline.nodes ORDER and the consumer's own dumped dict is byte-identical, so "
        "under _dumped_declarations the pair compares EQUAL -- and a raw-dump assertion "
        "would stay green even if resolve_single_type_source were deleted outright."
    ),
}


def _cell_params() -> list[Any]:
    """The parametrize argvalues, carrying each cell's own xfail mark."""
    params = []
    for axis_name, format_name in _CELLS:
        reason = _XFAIL_CELLS.get((axis_name, format_name))
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(axis_name, format_name, marks=marks, id=f"{axis_name}-{format_name}"))
    return params


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


class TestAResolutionChangeChangesTheExport:
    """The instrument: one address moves, the resolution-bearing projection moves."""

    @pytest.mark.parametrize(("axis_name", "format_name"), _cell_params())
    def test_two_constructs_differing_in_one_address_project_differently(
        self, axis_name: str, format_name: str
    ) -> None:
        axis = _AXES[axis_name]
        fmt = _FORMATS[format_name]
        a, b = axis.pair()
        projected_a, projected_b = fmt.project(a), fmt.project(b)
        assert projected_a != projected_b, (
            f"{fmt.name} exports the SAME {fmt.projection_is} for two constructs whose "
            f"{axis.name} address differs (resolved by {axis.resolved_by}). The exported "
            "artifact therefore wires an edge that does not follow the declared address.\n"
            f"  A == B == {projected_a!r}"
        )

    def test_the_green_cell_moves_exactly_the_resolved_edge_and_nothing_else(self) -> None:
        """ "They differ" is the weak form; this is the strong one.

        A projection that differs could still be differing for an incidental
        reason (the negative control covers that direction). This pins the
        opposite direction: the ONE edge that moves is the one feeding the
        consumer whose address changed, and no other edge moves with it. If a
        future exporter change adds a second delta here, the green cell would
        keep passing while no longer isolating the resolution -- which is the
        whole failure mode this file was rewritten to avoid.
        """
        a, b = _input_source_field_pair()
        edges_a, edges_b = set(_agent_spec_edges(a)), set(_agent_spec_edges(b))
        assert edges_a - edges_b == {("isf", "b", "items", "sink", "items")}
        assert edges_b - edges_a == {("isf", "a", "items", "sink", "items")}


class TestAnIncidentalDifferenceDoesNotMoveTheProjection:
    """The negative control -- without it the harness cannot tell a resolution
    change from noise, which is its only job.

    Same perturbation as the positive pair (``b`` and ``sink`` trade places); the
    only difference is that ``b``'s output type cannot satisfy ``sink``'s demand,
    so no address moves.
    """

    @pytest.mark.parametrize("format_name", sorted(_FORMATS))
    def test_a_pair_that_resolves_identically_projects_identically(self, format_name: str) -> None:
        fmt = _FORMATS[format_name]
        a, b = _incidental_pair()
        assert fmt.document(a) != fmt.document(b), (
            f"the negative control is VACUOUS in {fmt.name}: the two constructs export "
            "identical documents, so 'the projection is equal' proves nothing about the "
            "projection. The pair must differ incidentally (it reorders two members) "
            "while resolving identically."
        )
        assert fmt.project(a) == fmt.project(b), (
            f"{fmt.name}'s {fmt.projection_is} moved for a pair in which NO address "
            "moved -- so the harness is reporting incidental structure (declaration "
            "order) as if it were a resolution change, and every positive cell above is "
            "unfalsifiable.\n"
            f"  A: {fmt.project(a)!r}\n  B: {fmt.project(b)!r}"
        )


class TestTheHarnessIsNotAPassingRatchet:
    """A harness made only of xfails, or of no cells at all, reads as green while
    measuring nothing. These are PLAIN assertions, deliberately: ``pyproject.toml``
    sets ``empty_parameter_set_mark = 'fail_at_collect'`` and AGENTS.md bans
    encoding an invariant in the ABSENCE of parameters.
    """

    def test_the_axis_set_is_non_empty(self) -> None:
        assert _AXES, "no address axis is declared -- the parametrized cells would range over nothing"

    def test_at_least_one_cell_is_not_xfail(self) -> None:
        live = [cell for cell in _CELLS if cell not in _XFAIL_CELLS]
        assert live, (
            "every cell is xfail(strict=True), so the harness proves only that the "
            "exporter is broken -- never that it can detect a resolution change at all. "
            "(input_source_field, to_agent_spec) is the non-vacuity proof and is not "
            "optional decoration."
        )

    def test_every_axis_is_exercised_by_at_least_one_cell(self) -> None:
        uncovered = sorted(set(_AXES) - {axis for axis, _fmt in _CELLS})
        assert not uncovered, f"axes declared but never asserted on: {uncovered}"

    def test_a_dropped_cell_is_not_also_asserted(self) -> None:
        both = sorted(set(_CELLS) & set(_DROPPED_CELLS))
        assert not both, (
            f"cells are both asserted and recorded as dropped: {both}. A dropped cell "
            "carries the measurement that disqualified it -- read it before re-adding."
        )

    def test_every_dropped_cell_names_a_real_axis_and_format(self) -> None:
        """A dropped cell whose axis or format no longer exists is a stale note
        masquerading as a measurement."""
        stale = sorted(cell for cell in _DROPPED_CELLS if cell[0] not in _AXES or cell[1] not in _FORMATS)
        assert not stale, f"dropped cells naming an axis or format that no longer exists: {stale}"


class TestTheMeasurementsBehindTheXfails:
    """Each xfail's REASON pinned as an assertion, so a cell cannot keep failing
    for a different reason than the one recorded next to it.
    """

    def test_dump_spec_drops_output_from_although_the_spec_schema_declares_it(self) -> None:
        """The exact asymmetry that makes ``(output_from, dump_spec)`` red.

        This goes red alongside that cell's XPASS when the dumper is fixed --
        deliberately: the reason and the exemption are removed together.
        """
        assert "output_from" in ConstructSpec.model_fields, (
            "ConstructSpec.output_from is gone -- the dump_spec xfail's recorded reason "
            "(schema declares it, dumper drops it) no longer describes reality"
        )
        a, _b = _output_from_pair()
        dumped = {c["name"]: c for c in dump_spec(a)["constructs"]}
        assert "boundary" in dumped, f"expected a dumped sub-construct named 'boundary', got {sorted(dumped)}"
        assert "output" in dumped["boundary"], "the boundary TYPE is dumped -- only the port name is missing"
        assert "output_from" not in dumped["boundary"], (
            "_dump_sub_construct now emits output_from -- delete the "
            "(output_from, dump_spec) xfail, which has become an XPASS"
        )

    def test_the_output_from_pair_differs_only_in_the_inert_boundary_marker(self) -> None:
        """Why the assertion is a projection: the documents DO differ, in one
        inert key, while the wires do not. A whole-document ``!=`` would pass
        here and call that success."""
        a, b = _output_from_pair()
        assert _agent_spec_document(a) != _agent_spec_document(b), (
            "the output_from pair's documents no longer differ -- the recorded "
            "false-green measurement (an inert metadata marker moved) is stale"
        )
        assert _agent_spec_edges(a) == _agent_spec_edges(b), (
            "the exporter now wires output_from -- the (output_from, to_agent_spec) cell "
            "has become an XPASS; delete its xfail"
        )
