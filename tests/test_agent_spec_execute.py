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
    ``merge_prompt`` cell, whose merge lowers to an ``LlmNode``) fails to load
    BY DEFAULT. FIXED as a fail-loud OPT-IN (maintainer decision): neograph
    cannot know the real provider at export time (``Node.model`` is an opaque
    tier string, resolved via ``llm_factory`` at runtime), so
    ``to_agent_spec(construct)`` keeps this honest default unchanged --
    ``to_agent_spec(construct, api_provider="openai")`` is what becomes
    convertible. This tier's cells all call the zero-arg form, so qtfof.8
    landing does NOT widen ``COMPARE_CELLS`` here; see
    ``tests/test_agent_spec_llm_config_api_provider.py`` for the opt-in path.

``EXEC_EXEMPT`` is computed, not hand-typed: every GREEN cell is attempted once
at collection time, and a failure is classified into one of the three tracked
gaps above by ``agent_spec_loader_harness.classify_load_failure``. An
UNCLASSIFIED failure is a hard collection error -- a new finding must be
investigated and filed before it can be exempted, never silently swept in.
``COMPARE_CELLS`` is ``GREEN - EXEC_EXEMPT.keys()`` -- it widens automatically as
neograph-qtfof.6/.7 land (real DataFlowEdge fixes, no opt-in needed); qtfof.8's
opt-in shape means its cells stay in EXEC_EXEMPT for the zero-arg export this
tier exercises, by design.

Run with::

    uv run pytest tests/test_agent_spec_execute.py
"""

from __future__ import annotations

from typing import Any

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
    terminal_property_map,
)
from tests.test_agent_spec_matrix import CELLS, GREEN, Alpha, Beta, Coll, Ctx, Elem, Out, build_cell  # noqa: E402

# The shared graph walk is NOT re-derived here: two builders over the same
# ``control_flow_connections`` that must agree forever and are checked by nothing
# is exactly the duplicated-source-of-truth shape AGENTS.md forbids, and this
# tier's whole value proposition ("a disagreement is a real defect in one of two
# sides") is void if the reachability tier and this one disagree about what the
# graph IS. Since neograph-dgbqv.9 both tiers read
# ``tests.agent_spec_flow_walk``, so the agreement is structural rather than
# maintained by hand -- and the harness can import it directly, because that
# module imports nothing from ``neograph`` (the reason the walk used to be
# injected as a parameter).


def _server_tools(flow: object) -> dict[str, object]:
    return server_tools(flow)


def _stub_registry(flow: object) -> dict[str, object]:
    return stub_registry(flow)


def _compare_registry(flow: object, bodies: dict[str, object]) -> dict[str, object]:
    return compare_registry(flow, bodies)


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
    # neograph-qtfof.6 widened COMPARE_CELLS to include these two. `a="not-x"`
    # makes the loop's `a == "x"` condition false on iteration 1 (DONE) on
    # BOTH sides -- the third-party side's synthesized predicate tool has no
    # declared body (compare_registry's pass-through echoes its kwargs, not a
    # real evaluation), so it lands on DEFAULT, which this ticket routes to the
    # SAME exit as DONE. Both still land empty (qtfof.9's Loop-terminal EndNode
    # gap, explicitly out of that ticket's scope) -- see EXEC_EXEMPT below.
    "scripted-loop-single": {"prod": Alpha(a="p"), "refine": Alpha(a="not-x")},
    "scripted-loop-dict": {"pa": Alpha(a="p"), "pb": Beta(b="q"), "refine": Alpha(a="not-x")},
    # neograph-qtfof.7 widened COMPARE_CELLS to include these three. `target`'s
    # body is a CONSTANT (both `_const`/`_body` ignore kwargs), so every fanned-
    # out item produces the identical Out -- the comparison is on that shared
    # value winding up on both sides via real iteration, not per-item content.
    "scripted-each-single": {"prod": Coll(groups=[Elem(v="a"), Elem(v="b"), Elem(v="c")]), "target": Out(ok="done")},
    "scripted-each-dict": {
        "prod": Coll(groups=[Elem(v="a"), Elem(v="b"), Elem(v="c")]),
        "pb": Beta(b="q"),
        "target": Out(ok="done"),
    },
    "scripted-each-context": {
        "prod": Coll(groups=[Elem(v="a"), Elem(v="b"), Elem(v="c")]),
        "source": Ctx(c="ctx"),
        "target": Out(ok="done"),
    },
    # neograph-qtfof.11 widened COMPARE_CELLS again, to the FUSED Each x Oracle
    # cells. Their sub-Flow already shipped explicit data edges (_lower_oracle's
    # variant fan-in), so the loader's same-title auto-wiring was already off and
    # their `item` input was fed by nothing -- that ticket's StartNode->body edges
    # are what made them loadable. Same bodies as the plain Each cells above: the
    # keys are TOOL names, and both Oracle variants carry the one `target` tool
    # (the merge carries `m_combine`, left to compare_registry's pass-through,
    # which mirrors neograph's own `variants[0]` merge registration).
    "scripted-each_oracle-merge_fn-single": {
        "prod": Coll(groups=[Elem(v="a"), Elem(v="b"), Elem(v="c")]),
        "target": Out(ok="done"),
    },
    "scripted-each_oracle-merge_fn-dict": {
        "prod": Coll(groups=[Elem(v="a"), Elem(v="b"), Elem(v="c")]),
        "pb": Beta(b="q"),
        "target": Out(ok="done"),
    },
    "scripted-each_oracle-merge_fn-context": {
        "prod": Coll(groups=[Elem(v="a"), Elem(v="b"), Elem(v="c")]),
        "source": Ctx(c="ctx"),
        "target": Out(ok="done"),
    },
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

        projection_status: dict[tuple[str, bool, bool, bool, bool], set[bool]] = {}
        for cell_id in sorted(GREEN):
            mode, combo, config, _shape = CELLS[cell_id]
            names = modifier_names_for_combo(combo)
            # neograph-qtfof.6: "loop" alone no longer determines exemption --
            # a Loop's predicate is statically determinable and now gets a real
            # DataFlowEdge, closing that gap. "operator" still does: its `when`
            # reads the whole state bus, genuinely undeterminable (out of this
            # ticket's scope), so its check node's input stays permanently
            # unfed regardless of whether Loop is also present.
            # neograph-qtfof.7: "each" alone likewise no longer determines
            # exemption for the PLAIN (non-fused) case -- the MapNode's
            # iterated_item now gets a real, conditional DataFlowEdge. The
            # FUSED Each x Oracle sub-flow is a genuinely different closure
            # status (its own inner variant node stays unfed -- the fused
            # branch's non-empty body_data disables auto-wiring one level
            # deeper, out of this ticket's budget), so "oracle" (whether this
            # combo also carries Oracle) is a real, separate axis now, mirroring
            # how qtfof.6 added "operator" alongside "loop".
            key = (mode, "each" in names, "each" in names and "oracle" in names, "operator" in names, config == "merge_prompt")
            projection_status.setdefault(key, set()).add(cell_id in EXEC_EXEMPT)

        inconsistent = {k: v for k, v in projection_status.items() if len(v) > 1}
        assert not inconsistent, (
            "EXEC_EXEMPT membership is not determined by (mode, each, operator, merge_prompt) "
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
    def test_the_exported_flow_loads_and_runs_without_raising(self, cell_id: str, request: Any) -> None:
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        assert type(flow).__name__ == "Flow", f"{cell_id} exports a {type(flow).__name__}, not a Flow"

        if cell_id in EXEC_EXEMPT:
            # neograph-qtfof.13: xfail(strict=True), NOT skip. Two reasons, and the
            # second is the one that matters. (1) scripts/check_skips.py fails on ANY
            # skip and has no allowlist, so the skip form made release-gate red here.
            # (2) EXEC_EXEMPT is documented SHRINK-ONLY: an xfail that starts PASSING
            # turns the suite RED (XPASS is a failure under strict), so a gap that
            # closes cannot silently drop out of the count. A skip can never do that.
            # The body below still RUNS -- that is what makes the XPASS reachable.
            request.node.add_marker(
                pytest.mark.xfail(strict=True, reason=f"known export gap: {EXEC_EXEMPT[cell_id]}")
            )

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
    def test_the_third_party_run_surfaces_a_result(self, cell_id: str, request: Any) -> None:
        if cell_id not in _CELL_BODIES:
            pytest.skip(f"{cell_id}: no declared COMPARE bodies yet (newly widened COMPARE_CELLS)")
        if cell_id in COMPARE_EXEMPT:
            # neograph-qtfof.13: same reasoning as EXEC_EXEMPT above -- strict xfail so a
            # closed gap turns RED instead of dropping silently out of a skip count.
            request.node.add_marker(
                pytest.mark.xfail(strict=True, reason=f"known export gap: {COMPARE_EXEMPT[cell_id]}")
            )
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
    def test_the_third_party_result_equals_neographs_own(self, cell_id: str, request: Any) -> None:
        if cell_id not in _CELL_BODIES:
            pytest.skip(f"{cell_id}: no declared COMPARE bodies yet (newly widened COMPARE_CELLS)")
        if cell_id in COMPARE_EXEMPT:
            # neograph-qtfof.13: same reasoning as EXEC_EXEMPT above -- strict xfail so a
            # closed gap turns RED instead of dropping silently out of a skip count.
            request.node.add_marker(
                pytest.mark.xfail(strict=True, reason=f"known export gap: {COMPARE_EXEMPT[cell_id]}")
            )
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        bodies = _CELL_BODIES[cell_id]

        third_party = run_via_agent_spec_loader(flow, cell_id, _compare_registry(flow, bodies))
        neograph_side = run_via_neograph(build_cell(*CELLS[cell_id]), bodies)

        # neograph keys its result by NODE name and carries Pydantic instances; the
        # exported Flow's terminal value is a flat property map. Compare the flat
        # projection of the neograph result's terminal node against it -- via the
        # SHARED projection, so this tier and the conformance calibration cannot
        # drift into two different notions of "equal".
        terminal = max(neograph_side, key=lambda name: list(neograph_side).index(name))
        expected = terminal_property_map(neograph_side)

        assert third_party.get("outputs") == expected, (
            f"{cell_id}: the two independently-authored runtimes disagree on the result.\n"
            f"  third-party (AgentSpecLoader): {third_party.get('outputs')!r}\n"
            f"  neograph (run(compile(...)))  : {expected!r}  [terminal node {terminal!r}]"
        )


# -- neograph-qtfof.9 reproduction -------------------------------------------
#
# Both failure modes bypass COMPARE_EXEMPT deliberately -- the exemption exists
# so the ambient GREEN-cell matrix stays honest about a KNOWN gap; these two
# tests are the gap itself, asserted directly, and must FAIL until the
# outermost Construct's EndNode gets real declared outputs wired by a real
# DataFlowEdge from its terminal producer node(s).


class TestOutermostConstructEndNodeHasNoOutputs:
    """neograph-qtfof.9 failure mode (a): construct.output is None (the default,
    universal case for every top-level Construct passed to compile()/run()
    directly) -- so end_props is empty and the third party's invoke() always
    returns {}, regardless of what the graph actually computed."""

    def test_third_party_invoke_returns_empty_outputs_for_a_simple_outermost_construct(self) -> None:
        cell_id = "scripted-oracle-merge_fn-single"
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        bodies = _CELL_BODIES[cell_id]

        third_party = run_via_agent_spec_loader(flow, cell_id, _compare_registry(flow, bodies))
        neograph_side = run_via_neograph(build_cell(*CELLS[cell_id]), bodies)
        terminal = max(neograph_side, key=lambda name: list(neograph_side).index(name))
        expected = neograph_side[terminal].model_dump()

        assert third_party.get("outputs") == expected, (
            f"{cell_id}: a third-party Agent Spec runtime must surface what the graph computed, "
            f"not an empty dict regardless of it.\n"
            f"  third-party (AgentSpecLoader): {third_party.get('outputs')!r}\n"
            f"  neograph (run(compile(...)))  : {expected!r}  [terminal node {terminal!r}]"
        )


class TestOutermostConstructEndNodeWithDeclaredOutputRaises:
    """neograph-qtfof.9 failure mode (b) (merged from the duplicate ticket
    neograph-qtfof.10): even when construct.output IS explicitly set on an
    outermost Construct, the EndNode's declared outputs have zero DataFlowEdge
    feeding them -- there is no parent to wire the peer edge a sub-flow item's
    parent would provide -- so the loader raises instead of returning a value."""

    def test_third_party_invoke_raises_when_outermost_output_is_declared_but_unwired(self) -> None:
        from neograph import node
        from neograph.decorators import construct_from_functions
        from tests.test_agent_spec_matrix import Alpha, Out

        @node(outputs=Alpha)
        def prod() -> Alpha: ...

        @node(outputs=Out)
        def target(prod: Alpha) -> Out: ...

        construct = construct_from_functions("qtfof9-mode-b", [prod, target], output=Out)
        flow = to_agent_spec(construct)
        bodies = {"prod": Alpha(a="p"), "target": Out(ok="done")}

        third_party = run_via_agent_spec_loader(flow, "qtfof9-mode-b", _compare_registry(flow, bodies))
        neograph_side = run_via_neograph(construct, bodies)
        terminal = max(neograph_side, key=lambda name: list(neograph_side).index(name))
        expected = neograph_side[terminal].model_dump()

        assert third_party.get("outputs") == expected, (
            f"qtfof9-mode-b: declaring construct.output on an outermost Construct must make the "
            f"third party's invoke() return the computed value, not raise or return an empty dict.\n"
            f"  third-party (AgentSpecLoader): {third_party.get('outputs')!r}\n"
            f"  neograph (run(compile(...)))  : {expected!r}  [terminal node {terminal!r}]"
        )
