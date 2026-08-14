"""Calibration: the STATIC conformance verdict must be grounded in EMPIRICAL execution.

neograph-ftnxl.1. The disease this ticket exists to delete is ``to_agent_spec()
did not raise`` standing in for ``is portable to a third-party runtime``. A
classifier whose top tier means "none of my hand-written predicates fired"
reproduces that exact category error one layer out: PORTABLE would again be an
absence-of-evidence verdict, just with more machinery attached.

So PORTABLE is not allowed to be self-certified. This module binds it to the one
authority in the repo that measures portability by MEASURING it: the EXECUTE +
COMPARE tier (``tests/test_agent_spec_execute.py``), which loads every exported
Flow into a genuinely third-party, independently-authored runtime
(``pyagentspec.adapters.langgraph.AgentSpecLoader``) and compares its answer
against neograph's own ``run(compile(construct))``.

**The binding is POSITIVE, never the complement of an exemption dict**
(architect review HIGH-3). ``EXEC_EXEMPT`` / ``COMPARE_EXEMPT`` are evidence of
FAILURE, and membership in them is sound evidence a cell is not portable. Their
*complement* is not evidence of success: ``_compute_compare_exempt`` skips any
cell with no declared body (``if cell_id not in _CELL_BODIES: continue``), so
"not exempt" can mean "never probed". Today that is invisible because
``COMPARE_CELLS`` happens to equal exactly the four bodied cells -- but the whole
point of this ticket is that ``EXEC_EXEMPT`` SHRINKS as neograph-qtfof.6/.7/.8
land, at which moment ``COMPARE_CELLS`` widens to unbodied cells and "not
exempt" silently starts meaning "unprobed". Harvesting a free PORTABLE from that
is the ticket's own disease, reproduced at the authority it depends on.

Hence a cell counts as empirically portable only when it is PROBED **and** LOADS
**and** COMPARES EQUAL. Cells that cannot be probed (no declared body) go to an
explicit UNKNOWN bucket which is excluded from the biconditional and asserted
EMPTY, so a future widening forces someone to write the missing bodies instead.

The calibration is meaningful only with the third-party loader installed;
without ``pyagentspec`` this module skips (``pyagentspec`` is in
``[dependency-groups].dev``, so the bare ``uv run pytest`` gate does run it).

Run with::

    uv run pytest tests/test_agent_spec_conformance.py
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("pyagentspec")

from neograph import ConformanceTier, export_conformance  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from tests.agent_spec_loader_harness import (  # noqa: E402
    compare_registry,
    run_via_agent_spec_loader,
    run_via_neograph,
)
from tests.test_agent_spec_execute import (  # noqa: E402
    _CELL_BODIES,
    COMPARE_CELLS,
    EXEC_EXEMPT,
)
from tests.test_agent_spec_matrix import CELLS, GREEN, build_cell  # noqa: E402


@dataclass(frozen=True)
class _EmpiricalGrading:
    """What a real third-party runtime actually did with each GREEN cell.

    ``passed`` is the POSITIVE set -- probed, loaded, and compared equal against
    neograph's own run. ``failed`` carries a per-cell reason (never a bare bool,
    so a soundness violation prints WHY the cell is not portable). ``unknown`` is
    the honest third bucket: cells nothing probed, which may never be silently
    folded into either of the other two.
    """

    passed: frozenset[str]
    failed: dict[str, str]
    unknown: frozenset[str]


def _grade_green_cells_empirically() -> _EmpiricalGrading:
    """Grade every GREEN cell by EXECUTION, reusing the one body map (``_CELL_BODIES``).

    The load-failure half reuses ``EXEC_EXEMPT`` -- positive evidence that the
    third-party loader rejected the exported Flow for a tracked reason. The
    success half is re-derived here rather than inferred from ``COMPARE_EXEMPT``'s
    complement, because only an actual successful compare proves portability.
    """
    passed: set[str] = set()
    failed: dict[str, str] = dict(EXEC_EXEMPT)
    unknown: set[str] = set()

    for cell_id in sorted(COMPARE_CELLS):
        bodies = _CELL_BODIES.get(cell_id)
        if bodies is None:
            unknown.add(cell_id)
            continue

        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        try:
            third_party = run_via_agent_spec_loader(flow, cell_id, compare_registry(flow, bodies))
        except Exception as exc:  # noqa: BLE001 -- any failure is empirical non-portability
            failed[cell_id] = f"third-party loader raised {type(exc).__name__}: {exc}"
            continue

        outputs = third_party.get("outputs")
        if not outputs:
            failed[cell_id] = (
                "the third-party runtime finished with no result at all (empty outputs) -- "
                "a consumer gets an empty dict back (neograph-qtfof.9)"
            )
            continue

        neograph_side = run_via_neograph(build_cell(*CELLS[cell_id]), bodies)
        terminal = list(neograph_side)[-1]
        expected = neograph_side[terminal].model_dump()
        if outputs != expected:
            failed[cell_id] = (
                f"the two independently-authored runtimes disagree -- third-party {outputs!r} "
                f"vs neograph {expected!r} [terminal node {terminal!r}]"
            )
            continue

        passed.add(cell_id)

    return _EmpiricalGrading(frozenset(passed), failed, frozenset(unknown))


class TestTheStaticConformanceVerdictIsCalibratedAgainstEmpiricalExecution:
    """``export_conformance()`` may call a construct PORTABLE only when a real
    third-party Agent Spec runtime demonstrably executes it correctly.

    Parametrization is deliberately absent: the biconditional is a property of
    the WHOLE partition (both directions, over every graded cell at once), and
    per-cell parametrization would let a single mis-tiered cell read as one
    unrelated red among 100 greens rather than as a broken calibration.
    """

    def test_a_portable_verdict_holds_exactly_for_the_cells_a_third_party_runtime_really_executes(
        self,
    ) -> None:
        grading = _grade_green_cells_empirically()

        assert not grading.unknown, (
            "the empirical authority is not total over the cells it is asked about: "
            f"{sorted(grading.unknown)} load under the third-party runtime but have no declared "
            "body in test_agent_spec_execute._CELL_BODIES, so nothing ever compared their results. "
            "COMPARE_CELLS has widened (a tracked export gap landed) -- write the missing bodies. "
            "Until then these cells have NO empirical verdict, and inferring PORTABLE from the "
            "absence of a recorded failure is exactly the false-green neograph-ftnxl.1 exists to kill."
        )

        graded = GREEN - grading.unknown
        unsound: dict[str, str] = {}
        over_pessimistic: dict[str, ConformanceTier] = {}
        for cell_id in sorted(graded):
            verdict = export_conformance(build_cell(*CELLS[cell_id]), strict=False).verdict
            statically_portable = verdict is ConformanceTier.PORTABLE
            empirically_portable = cell_id in grading.passed
            if statically_portable and not empirically_portable:
                unsound[cell_id] = grading.failed[cell_id]
            elif empirically_portable and not statically_portable:
                over_pessimistic[cell_id] = verdict

        assert not unsound, (
            "UNSOUND: export_conformance() certified these cells PORTABLE, but a real third-party "
            "Agent Spec runtime does not execute them correctly:\n"
            + "\n".join(f"  {cell_id}: {reason}" for cell_id, reason in sorted(unsound.items()))
            + "\nA PORTABLE verdict that a real runtime contradicts is the original disease "
            "('did not raise' == 'is portable') relocated into the classifier. This direction is an "
            "unconditional failure -- it may never be exempted."
        )

        assert not over_pessimistic, (
            "OVER-PESSIMISTIC: a third-party runtime loads and correctly executes these cells, but "
            "export_conformance() refuses to call them PORTABLE:\n"
            + "\n".join(f"  {cell_id}: {verdict}" for cell_id, verdict in sorted(over_pessimistic.items()))
            + "\nA predicate is firing on a construct the empirical authority has cleared -- either "
            "the tracked gap it cites has landed and the predicate must retire, or it over-matches. "
            "Slack in this direction is tolerable ONLY as a computed, per-entry, reasoned exemption "
            "in the style of _compute_exec_exempt -- never as silent slack added here."
        )

        assert grading.passed == frozenset(), (
            "VACUITY RATCHET: the empirical PASS set is no longer empty -- "
            f"{sorted(grading.passed)} now load AND compare equal under a third-party runtime. "
            "Until now the biconditional above only proved 'the classifier calls nobody PORTABLE' "
            "(every export hits neograph-qtfof.9), so it could pass without ever exercising the "
            "PORTABLE branch. A tracked export gap has landed: re-read the whole biconditional, "
            "confirm export_conformance now genuinely certifies these cells, then update this literal."
        )
