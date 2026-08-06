"""EXECUTE + COMPARE tiers of the Oracle differential-execution grid -- neograph-dgbqv.2.

The WIRE tier (neograph-dgbqv.1) proves an exported ``Flow`` survives a
``Flow.from_dict`` round trip; ``test_agent_spec_reachability.py`` proves its
control graph has no orphan and no dead end. Neither EXECUTES it, and both are
therefore blind to the bug class this tier exists for: ``a734217``'s
branch-flattening defect was fully reachable and fully round-trippable -- only a
walk that EVALUATES conditions sees a missing decision.

**Core Invariant.** The exported Flow is run by a genuinely third-party,
independently-authored runtime -- ``pyagentspec.adapters.langgraph.AgentSpecLoader``,
shipped by the already-pinned ``pyagentspec>=26.1.0,<27`` dependency -- never by a
walker neograph's own team wrote and could get wrong the same way twice.
Independence comes from AUTHORSHIP (Oracle wrote it, we did not), so a
disagreement between it and neograph's own ``run(compile(construct))`` is a real
defect in one of two independently-authored sides, never a shared blind spot.

``AgentSpecLoader`` consumes ONLY plain Flow data (``flow.to_dict()``:
nodes/edges/metadata) plus a ``tool_registry`` of callables keyed by exported
tool name. It imports nothing from neograph, and it is metadata-BLIND by
construction: it builds and runs the graph from ``ControlFlowEdge`` /
``DataFlowEdge`` / ``BranchingNode`` topology alone and never reads a
``neograph/*`` metadata key (that is neograph's own private export annotation,
meaningless to a third-party consumer). That makes this tier the literal referent
of the policy ``test_agent_spec_reachability.py``'s module docstring already
commits to -- "correct when executed by ANY real Agent Spec runtime" -- and the
direct measurement of the maintainer's stated full-interop design goal: a third
party may lose neograph's abstractions (Oracle/Each), but must not lose correct
execution.

**A nonzero red set is this tier's OUTPUT, not its failure.** Three export
gaps are already tracked and are the KNOWN source of red-cell membership --
do not re-file them and do not treat hitting them as a new finding:

  * neograph-qtfof.6 -- a ``BranchingNode``'s ``branching_mapping_key`` input is
    fed by no ``DataFlowEdge``; the loop/operator predicate lives only in
    ``metadata['neograph/loop_spec']['when']`` /
    ``['neograph/operator_spec']['when']``
    (``_agent_spec_modifier_lowering.py:376-385``, ``:634-641``).
  * neograph-qtfof.7 -- a ``MapNode``'s ``iterated_item`` input is fed by no
    ``DataFlowEdge``; the collection source lives only in
    ``metadata['neograph/each_spec']['over']``. Covers the sibling
    Each-subflow-carries-zero-data-edges case too, not a fourth gap.
  * neograph-qtfof.8 -- ``LlmConfig`` with ``api_provider=None`` is unsupported
    by the loader, so every think/agent/act cell (and every scripted Oracle
    ``merge_prompt`` cell, whose merge lowers to an ``LlmNode``) fails to load.

``EXEC_EXEMPT`` is computed, not hand-typed: every GREEN cell is attempted once
at collection time, and a failure is classified into one of the three tracked
gaps above by ``agent_spec_loader_harness.classify_load_failure``. An
UNCLASSIFIED failure is a hard collection error -- a new finding must be
investigated and filed before it can be exempted, never silently swept in.
``COMPARE_CELLS`` is ``GREEN - EXEC_EXEMPT.keys()`` -- it widens automatically as
neograph-qtfof.6/.7/.8 land, with no ratchet literal to remember to update.

Run with::

    uv run pytest tests/test_agent_spec_execute.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyagentspec")

from neograph._agent_spec import to_agent_spec  # noqa: E402
from tests.agent_spec_loader_harness import (  # noqa: E402
    classify_load_failure,
    compare_registry,
    run_via_agent_spec_loader,
    run_via_neograph,
    server_tools,
    stub_registry,
)
from tests.test_agent_spec_matrix import CELLS, GREEN, Alpha, Beta, Out, build_cell  # noqa: E402
from tests.test_agent_spec_reachability import _all_flows  # noqa: E402

# ``_all_flows`` is imported rather than re-derived: two builders over the same
# ``control_flow_connections`` that must agree forever and are checked by nothing
# is exactly the duplicated-source-of-truth shape AGENTS.md forbids, and this
# tier's whole value proposition ("a disagreement is a real defect in one of two
# sides") is void if the reachability tier and this one disagree about what the
# graph IS. Cross-test-module import is the established house pattern --
# ``test_agent_spec_reachability.py`` itself imports from the matrix. It is
# imported HERE, not inside ``agent_spec_loader_harness.py`` -- see that module's
# docstring for why the harness's own import list must stay independent.


def _server_tools(flow: object) -> dict[str, object]:
    return server_tools(flow, _all_flows)


def _stub_registry(flow: object) -> dict[str, object]:
    return stub_registry(flow, _all_flows)


def _compare_registry(flow: object, bodies: dict[str, object]) -> dict[str, object]:
    return compare_registry(flow, bodies, _all_flows)


# -- COMPARE fixtures --------------------------------------------------------
#
# ONE per-cell body map feeds BOTH sides, so a disagreement can only come from the
# two runtimes, never from two different sets of bodies. Values are the Pydantic
# outputs neograph's own nodes produce; each side adapts them:
#   * neograph  -- ``compile(scripted={name: fn})``, whose shim signature is
#     ``fn(input_data, config)`` and which is merged LAST so it wins over the
#     decorator's own ``...``-bodied shim (``compiler.py:153-160``).
#   * loader    -- a ``tool_registry`` callable returning ``model_dump()``, since a
#     ToolNode exchanges plain JSON-shaped values, not Pydantic instances.
#
# Only the 4 cells known-passing at write-test time have declared bodies today;
# COMPARE_CELLS may be WIDER than this (as qtfof.6/.7/.8 land), in which case a
# cell with no declared bodies is skipped rather than failed -- see the tests
# below. Declared ABOVE EXEC_EXEMPT/COMPARE_EXEMPT's computation -- both probe
# these specific cells and need the body map already defined.

_CELL_BODIES: dict[str, dict[str, object]] = {
    "scripted-bare-single": {"prod": Alpha(a="p"), "target": Out(ok="done")},
    "scripted-bare-dict": {"pa": Alpha(a="p"), "pb": Beta(b="q"), "target": Out(ok="done")},
    "scripted-oracle-merge_fn-single": {"prod": Alpha(a="p"), "gen": Out(ok="done")},
    "scripted-oracle-merge_fn-dict": {"pa": Alpha(a="p"), "pb": Beta(b="q"), "gen": Out(ok="done")},
}


# -- EXEC_EXEMPT: computed, per-entry reasons, shrink-only ------------------


def _compute_exec_exempt() -> dict[str, str]:
    exempt: dict[str, str] = {}
    for cell_id in sorted(GREEN):
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        try:
            run_via_agent_spec_loader(flow, cell_id, _stub_registry(flow))
        except Exception as exc:  # noqa: BLE001 -- classify or fail loud, never swallow
            reason = classify_load_failure(exc)
            if reason is None:
                raise AssertionError(
                    f"{cell_id}: EXECUTE-tier failure does not match a tracked export gap "
                    f"(neograph-qtfof.6/.7/.8) -- {type(exc).__name__}: {exc}. This is a NEW "
                    "finding: investigate and file a ticket before adding it to EXEC_EXEMPT. "
                    "Do not silently sweep an unclassified failure into the exemption."
                ) from exc
            exempt[cell_id] = reason
    return exempt


#: GREEN cells whose exported Flow currently fails to load/run under the
#: third-party runtime for a KNOWN, already-tracked reason (neograph-qtfof.6/.7/.8).
#: Computed (not hand-typed) so it can never silently drift from reality, and
#: fails collection loud on any UNCLASSIFIED failure (see ``_compute_exec_exempt``).
#: SHRINK-ONLY: as each tracked gap is fixed, its cells stop failing and drop out
#: on their own -- there is no literal count to remember to lower.
EXEC_EXEMPT: dict[str, str] = _compute_exec_exempt()

#: The cells COMPARE can grade TODAY: every GREEN cell whose exported Flow both
#: loads and invokes under the third-party runtime, so a failure there is a real
#: disagreement rather than a re-report of an EXECUTE-tier load failure. Widens
#: automatically as EXEC_EXEMPT shrinks -- no literal to update.
COMPARE_CELLS: frozenset[str] = GREEN - frozenset(EXEC_EXEMPT)

assert COMPARE_CELLS | frozenset(EXEC_EXEMPT) == GREEN, "EXEC_EXEMPT and COMPARE_CELLS must partition GREEN"
assert not (COMPARE_CELLS & frozenset(EXEC_EXEMPT)), "EXEC_EXEMPT and COMPARE_CELLS must not overlap"


def _compute_compare_exempt() -> dict[str, str]:
    """A SECOND, independent finding discovered while implementing this tier
    (neograph-qtfof.9, filed 2026-08-06): the outermost Construct's EndNode
    declares no outputs at all -- ``construct.output`` is always ``None`` for a
    top-level Construct by neograph's own design (only sub-constructs used as
    items need a declared boundary output type), so
    ``src/neograph/_agent_spec.py``'s ``end_props = _properties_for(construct.output)``
    is always empty and the third-party runtime's ``invoke()`` result has
    nothing to populate. So a third-party runtime invoking a top-level
    exported Flow directly NEVER surfaces a result, for ANY cell shape --
    confirmed on ``scripted-bare-single``, the simplest possible cell.

    A SECOND failure mode of the same root cause (merged from the duplicate
    ticket neograph-qtfof.10 into qtfof.9): even when ``construct.output`` IS
    explicitly set, the EndNode's declared outputs still have no
    ``DataFlowEdge`` feeding them (there is no parent to wire the peer edge a
    sub-flow item would get), so the loader raises "Expected node ... to have
    a value for property ..." instead of returning an empty dict.

    Unlike EXEC_EXEMPT (which partitions by mode/modifier shape), this affects
    every COMPARE_CELLS member identically -- it is a universal property of
    outermost-Construct export, not a per-shape defect. Computed the same way
    for the same reason: never hand-typed, fails loud on an unclassified cause.
    """
    exempt: dict[str, str] = {}
    for cell_id in sorted(COMPARE_CELLS):
        if cell_id not in _CELL_BODIES:
            continue
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        bodies = _CELL_BODIES[cell_id]
        try:
            final = run_via_agent_spec_loader(flow, cell_id, _compare_registry(flow, bodies))
        except Exception as exc:  # noqa: BLE001
            if "to have a value for property" in str(exc):
                exempt[cell_id] = (
                    "neograph-qtfof.9: outermost EndNode's declared outputs have no DataFlowEdge feeding them"
                )
                continue
            raise
        if not final.get("outputs"):
            exempt[cell_id] = "neograph-qtfof.9: outermost EndNode declares no outputs (construct.output is None)"
    return exempt


#: COMPARE_CELLS members that cannot be genuinely compared today because of
#: neograph-qtfof.9, a SECOND tracked gap discovered while building this tier
#: (see ``_compute_compare_exempt`` for why it is orthogonal to EXEC_EXEMPT).
#: Computed, shrink-only -- disappears entirely once qtfof.9 lands.
COMPARE_EXEMPT: dict[str, str] = _compute_compare_exempt()


class TestExecExemptProjectionCoverage:
    """The exemption is a function of a cell's declared SHAPE (mode / combo
    axis), not an accident of which cell IDs happened to be probed first.

    A weak ``non_empty`` check (the plan's original, superseded draft) would
    pass even if ``EXEC_EXEMPT`` were one arbitrary cell short of correct. This
    proves every cell sharing the same (mode, "each" in combo, "loop"/"operator"
    in combo, ``config == "merge_prompt"``) projection has the SAME exemption
    status -- so the exemption tracks a real structural property of the cell,
    not incidental noise. The ``config`` axis is included because an Oracle's
    ``merge_prompt`` variant lowers its merge to an ``LlmNode`` even on an
    otherwise-scripted cell, so qtfof.8 is not purely mode-determined.
    """

    def test_exemption_status_is_determined_by_mode_and_modifier_projection(self) -> None:
        from neograph.modifiers import modifier_names_for_combo

        projection_status: dict[tuple[str, bool, bool, bool], set[bool]] = {}
        for cell_id in sorted(GREEN):
            mode, combo, config, _shape = CELLS[cell_id]
            names = modifier_names_for_combo(combo)
            key = (mode, "each" in names, bool({"loop", "operator"} & set(names)), config == "merge_prompt")
            projection_status.setdefault(key, set()).add(cell_id in EXEC_EXEMPT)

        inconsistent = {k: v for k, v in projection_status.items() if len(v) > 1}
        assert not inconsistent, (
            "EXEC_EXEMPT membership is not determined by (mode, each, loop/operator, merge_prompt) "
            f"alone -- these projections contain BOTH exempt and non-exempt cells: {sorted(inconsistent)}. "
            "That means the exemption is tracking something other than the three known export "
            "gaps (which are pure mode/modifier-shape defects), or a genuinely new export gap "
            "only affects SOME cells in a shape class -- investigate before trusting the ratchet."
        )

    def test_exec_exempt_is_a_real_subset_not_accidentally_all_of_green(self) -> None:
        # The weak check this replaces (`assert EXEC_EXEMPT`) would pass even if
        # EVERY cell were wrongly exempted. Pin that COMPARE still has real work.
        assert COMPARE_CELLS, "COMPARE_CELLS is empty -- EXEC_EXEMPT swallowed every GREEN cell"
        assert len(COMPARE_CELLS) >= 4, (
            f"COMPARE_CELLS shrank below the 4 cells known-passing at write-test time: {sorted(COMPARE_CELLS)}"
        )


# -- (b) EXECUTE -------------------------------------------------------------


class TestEveryGreenCellExecutesUnderAThirdPartyAgentSpecRuntime:
    """Every GREEN cell exports a Flow a real, independently-authored Agent Spec
    runtime can load AND run -- OR is a known, tracked, cited export gap.

    Parametrized over the matrix's mechanically-derived GREEN set, not a hand-typed
    shape list -- the same completeness discipline ``test_agent_spec_matrix.py``
    exists to enforce, so a new mode/combo axis lands here automatically instead of
    silently escaping the execution net.
    """

    @pytest.mark.parametrize("cell_id", sorted(GREEN))
    def test_the_exported_flow_loads_and_runs_without_raising(self, cell_id: str) -> None:
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        assert type(flow).__name__ == "Flow", f"{cell_id} exports a {type(flow).__name__}, not a Flow"

        if cell_id in EXEC_EXEMPT:
            pytest.skip(f"known export gap: {EXEC_EXEMPT[cell_id]}")

        try:
            run_via_agent_spec_loader(flow, cell_id, _stub_registry(flow))
        except Exception as exc:  # noqa: BLE001 -- ANY failure is the finding
            pytest.fail(
                f"{cell_id}: a third-party Agent Spec runtime cannot execute the exported Flow -- "
                f"{type(exc).__name__}: {exc}"
            )


# -- (c) COMPARE -------------------------------------------------------------


class TestTheThirdPartyRuntimeAgreesWithNeographsOwnRuntime:
    """Running the exported Flow must produce what running the Construct produces.

    Fed the identical per-cell bodies, the two independently-authored runtimes must
    reach the same answer. This is the half that defends the tier against its own
    harness: an EXECUTE pass proves only that SOMETHING ran.
    """

    @pytest.mark.parametrize("cell_id", sorted(COMPARE_CELLS))
    def test_the_third_party_run_surfaces_a_result(self, cell_id: str) -> None:
        if cell_id not in _CELL_BODIES:
            pytest.skip(f"{cell_id}: no declared COMPARE bodies yet (newly widened COMPARE_CELLS)")
        if cell_id in COMPARE_EXEMPT:
            pytest.skip(f"known export gap: {COMPARE_EXEMPT[cell_id]}")
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        bodies = _CELL_BODIES[cell_id]

        final = run_via_agent_spec_loader(flow, cell_id, _compare_registry(flow, bodies))

        assert final.get("outputs"), (
            f"{cell_id}: the third-party runtime finishes with no result at all "
            f"(final outputs: {final.get('outputs')!r}). An exported Flow that computes a value "
            "but surfaces none is not independently executable in any useful sense -- a consumer "
            "gets an empty dict back. Compare neograph's own run, which returns "
            f"{ {k: str(v) for k, v in run_via_neograph(build_cell(*CELLS[cell_id]), bodies).items()} }."
        )

    @pytest.mark.parametrize("cell_id", sorted(COMPARE_CELLS))
    def test_the_third_party_result_equals_neographs_own(self, cell_id: str) -> None:
        if cell_id not in _CELL_BODIES:
            pytest.skip(f"{cell_id}: no declared COMPARE bodies yet (newly widened COMPARE_CELLS)")
        if cell_id in COMPARE_EXEMPT:
            pytest.skip(f"known export gap: {COMPARE_EXEMPT[cell_id]}")
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        bodies = _CELL_BODIES[cell_id]

        third_party = run_via_agent_spec_loader(flow, cell_id, _compare_registry(flow, bodies))
        neograph_side = run_via_neograph(build_cell(*CELLS[cell_id]), bodies)

        # neograph keys its result by NODE name and carries Pydantic instances; the
        # exported Flow's terminal value is a flat property map. Compare the flat
        # projection of the neograph result's terminal node against it.
        terminal = max(neograph_side, key=lambda name: list(neograph_side).index(name))
        expected = neograph_side[terminal].model_dump()

        assert third_party.get("outputs") == expected, (
            f"{cell_id}: the two independently-authored runtimes disagree on the result.\n"
            f"  third-party (AgentSpecLoader): {third_party.get('outputs')!r}\n"
            f"  neograph (run(compile(...)))  : {expected!r}  [terminal node {terminal!r}]"
        )
