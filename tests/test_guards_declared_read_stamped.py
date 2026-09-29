"""Guard: every DECLARED READ in the corpus holds a stamped ``Source``.

``neograph-4cvx8`` step 0 (child ``neograph-4cvx8.1``). The Core Invariant it makes
executable: *every declared read leaves ``Construct()`` either holding exactly one
stamped ``Source`` or refused with a reason*. Eight separately-found instances of
one defect say that today it can do neither -- the node declares an input, nothing
resolves it, nothing refuses it, and the body receives ``None`` on a green run.

This file is the VERDICT; ``tests/stamp_instrument.py`` is the rule and the
collector, and the root ``conftest.py`` arms it for the whole session. Read that
module's docstring before changing anything here -- in particular what the
instrument deliberately CANNOT see (``run_isolated``, a stale-but-present stamp)
and why a green verdict means "nothing is unaddressed", never "every address is
right".

The corpus is three sources, and the union is what the verdict ranges over:

1. the whole test suite (armed in ``conftest.pytest_configure``; this module's
   items are ordered LAST so they see it),
2. ``tests/check_fixtures/should_pass`` (driven below, so this file is meaningful
   when run alone),
3. the keyless ``examples/`` -- the same list ``make examples`` runs -- driven by
   IMPORT, because every one of them builds its constructs at module level.

``EXPECTED_UNSTAMPED`` is the shrink-only allowlist, and the measurement it was
baselined from, over 4785 constructs at the red step::

    dict-key/peer            2737    addressed by NAME (disease-scan row 47)
    single-type/each-item     223    step 3 -- stamp EachItem                 [GONE]
    dict-key/framework-port   111    the DX port-KEY rewrite (neograph-d8ac9)
    port/sub-construct        110    step 8 -- Construct.port_source
    single-type/unfed           5    steps 2, 4 and 5 -- refuse, do not stamp [GONE]
    single-type/mesh-member     1    step 5 -- refuse (sdqsv)                [GONE]
    single-type/loop            0    already stamped to the seed

Every later step DELETES rows, and ``ROW_CEILING`` moves down with them in the same
commit. ``single-type/each-item`` is the first retired, and the largest: step 3 stamps
the fan-out channel, so those 223 reads now hold an ``EachItem`` address instead of
either nothing or a peer that no run reads.

``single-type/unfed`` is the instructive case: steps 2 and 4 turned two of its three populations into refusals (an
``input_from`` naming nothing; ``inputs=dict`` / ``dict[str, X]`` / a non-class
annotation no producer could satisfy), taking it from 5 measured reads to 3. The
remainder is one SHAPE, not a residue -- a ``route="decide"`` DISPATCH Portal, which
the validator exempts on a membership claim ``portal_member_class`` denies -- so the
row survives with a narrowed reason and step 5's name on it.

That is why the ceiling counts SHAPES and the reasons name steps: a row that shrank
by two thirds still licenses the same shape, and only the reason can say which
population is left.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from neograph import Construct, Each, Loop, Portal
from neograph._state_keys import StateKeys
from tests import stamp_instrument
from tests.schemas import Claims, MergedResult, RawText, _consumer, _producer
from tests.stamp_instrument import SHAPES, DeclaredRead, audit_construct, outside_allowlist

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"

# The keyless examples, kept in step with the Makefile's `examples` target. They
# are IMPORTED, not run: each builds its constructs at module level and guards
# main() behind `if __name__ == "__main__"`, so assembly -- the only thing this
# instrument measures -- happens on import without an LLM or a run.
KEYLESS_EXAMPLE_GLOBS = (
    "01_*.py",
    "01c_*.py",
    "02_*.py",
    "03_*.py",
    "04_*.py",
    "05_*.py",
    "06_*.py",
    "08_*.py",
    "09_*.py",
    "10_*.py",
    "11_*.py",
    "32_*.py",
    "33_*.py",
)


# ═══════════════════════════════════════════════════════════════════════════
# THE SHRINK-ONLY ALLOWLIST
#
# shape -> why that shape is expected unstamped TODAY, naming the step that
# deletes the row. A row is a licence for a WHOLE SHAPE, so it is the most
# expensive kind of exemption in this repo: keep the vocabulary narrow
# (stamp_instrument.SHAPES) so a row cannot quietly cover a defect it was never
# written for.
#
# The ratchet is stated as an ASSERTION over the dict and is exercised against a
# SIMULATED over-ceiling allowlist, an unknown shape and an unstamped read -- never
# as an empty parametrize, which reports as a skip and cannot be told from a test
# that failed to run (AGENTS.md / neograph-e8wiv). So it keeps its teeth at every
# size, including the empty one it is shrinking toward.
# ═══════════════════════════════════════════════════════════════════════════
EXPECTED_UNSTAMPED: dict[str, str] = {
    # NOT step 7's, which is what re-measuring showed. Step 7 retired the framework
    # TAIL -- the fallback that served SINGLE-TYPE reads whose stamp was stale or
    # absent (the fan-agent wrapper, run_isolated), and those are stamped Port() now.
    # These are DICT-FORM keys literally named `neo_subgraph_input`, minted by the DX
    # layer's port-param rewrite (_param_classify / _construct_builder) and stamped by
    # nobody: the same channel with a second spelling. neograph-d8ac9 owns it.
    "dict-key/framework-port": "neograph-d8ac9 -- the DX layer's port KEY rewrite is unstamped",
    # Step 8 (neograph-4cvx8.9) makes `Construct.port_source` the stamped Source
    # for the whole `_build_sub_input` ladder (N3, N7, xejyn half 2).
    "port/sub-construct": "step 8 / neograph-4cvx8.9 -- stamp Construct.port_source",
    # Disease-scan row 47: a named dict-form key IS the producer's name, and
    # `_check_fan_in_inputs` requires that producer at assembly, so today's form
    # is the dispositioned target, not a defect. The elegance review's C1 end
    # state stamps these too; it is filed with step 9's end-state follow-up and
    # this row goes when it lands.
    "dict-key/peer": "disease-scan row 47 -- named, validated by _check_fan_in_inputs; C1 end state stamps it",
}

# Shrink-only ceiling. The implement atom (neograph-4cvx8.1) raises this ONCE, to
# the row count it measures; from then on it may only go DOWN, and the step that
# deletes a row lowers it in the same commit. A step that grows it is a step that
# widened the disease.
#
# `single-type/loop` is deliberately NOT a row: the 86 measured Loop sites all
# stamp their SEED, so a presence check already sees nothing there. Step 6 adds
# the CARRY rung, which a presence check cannot see either way.
ROW_CEILING = 3


def _load_module(path: Path, prefix: str) -> None:
    """Exec ``path`` as a module so its module-level constructs get built.

    Registered in ``sys.modules`` BEFORE exec because several examples call
    ``construct_from_module(sys.modules[__name__])`` and Pydantic forward refs
    resolve against the live module namespace. Removed afterwards so nothing is
    left behind for a later test to import by accident.
    """
    name = f"{prefix}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)


def _drive_self_corpus() -> dict[str, int]:
    """Build corpus sources 2 and 3 in-process; return constructs seen per source.

    Isolation is the same snapshot the check-fixture harness uses: a fixture or
    example registering a ``@merge_fn`` at a different def site would otherwise
    trip the fail-loud collision guard against a neighbour's residue.

    The per-source counts are returned, not discarded, because a corpus that
    silently STOPS producing constructs (a renamed example glob, a fixture
    directory that moved) would turn this guard green for the one reason a guard
    must never go green.
    """
    from tests.test_check_fixtures import SHOULD_PASS, _isolated_registries, _load_fixture

    before = stamp_instrument.constructs_seen()
    for path in SHOULD_PASS:
        with _isolated_registries():
            _load_fixture(path)
    after_fixtures = stamp_instrument.constructs_seen()

    for glob in KEYLESS_EXAMPLE_GLOBS:
        for path in sorted(EXAMPLES.glob(glob)):
            with _isolated_registries():
                _load_module(path, "_stamp_corpus_example")
    return {
        "check_fixtures": after_fixtures - before,
        "examples": stamp_instrument.constructs_seen() - after_fixtures,
    }


def test_every_declared_read_in_the_corpus_holds_a_stamp():
    """No declared read may leave assembly with neither a stamp nor a refusal.

    THE red test for neograph-4cvx8. It fails today by NAMING the unaddressed
    shapes; it goes green when every one of them is either stamped (Each item,
    Loop carry, sub-construct port) or refused (mesh single-type reads, the
    unresolvable generics), with the residue carried as an explicit allowlist row.
    """
    per_source = _drive_self_corpus()
    assert all(per_source.values()), (
        f"a corpus source produced no constructs at all ({per_source}) -- a green "
        "verdict would then be an artefact of the corpus vanishing, not of the reads "
        "being stamped."
    )

    counts = stamp_instrument.observed()
    assert stamp_instrument.constructs_seen() > 0, (
        "the instrument audited ZERO constructs -- it was not armed, so a green "
        "verdict here would mean nothing. Check conftest.pytest_configure."
    )

    offending = outside_allowlist(counts, EXPECTED_UNSTAMPED)
    assert not offending, (
        "declared reads left assembly with NO stamped Source and no refusal "
        f"(over {stamp_instrument.constructs_seen()} constructs):\n"
        f"{stamp_instrument.shape_report(offending)}\n"
        "Each shape is either a defect to fix (the read must resolve or the "
        "construct must be refused) or a row for EXPECTED_UNSTAMPED naming the "
        "plan step that deletes it."
    )


def allowlist_defects(allowlist: dict[str, str], ceiling: int) -> list[str]:
    """Everything wrong with ``allowlist`` under ``ceiling``, as messages.

    The ratchet stated as a FUNCTION of the dict, so the real allowlist and a
    simulated one go through the SAME code. That is what keeps the ratchet
    non-vacuous while ``EXPECTED_UNSTAMPED`` is empty: the emptiness is asserted,
    and the rule that guards it is exercised against a set that is not empty
    (``AGENTS.md`` / neograph-e8wiv -- never an empty parametrize).
    """
    defects = []
    if len(allowlist) > ceiling:
        defects.append(
            f"{len(allowlist)} rows against a ceiling of {ceiling}: a new expected-unstamped "
            "SHAPE is a widened disease, not a new exemption -- fix the read or refuse the construct"
        )
    for shape, reason in sorted(allowlist.items()):
        if shape not in SHAPES:
            defects.append(f"{shape!r} is not a shape stamp_instrument can produce -- a dead licence")
        if not reason.strip():
            defects.append(f"{shape!r} carries no reason, so nothing says which step deletes it")
    return defects


class TestExpectedUnstampedIsShrinkOnly:
    """The allowlist is a ratchet, and the ratchet is EXERCISED, not merely asserted."""

    def test_the_real_allowlist_is_within_its_ceiling_and_every_row_is_justified(self):
        assert allowlist_defects(EXPECTED_UNSTAMPED, ROW_CEILING) == []

    def test_the_row_count_is_the_asserted_state_at_this_step(self):
        """The ceiling is EXACT, not an upper bound.

        A ceiling above its dict is silent headroom -- the same defect the file-size
        ratchet refuses a tolerance band for. A step that retires a row lowers this
        in the same commit, so the number always names the current state rather than
        a past one.
        """
        assert len(EXPECTED_UNSTAMPED) == ROW_CEILING

    def test_the_same_rule_refuses_a_simulated_over_ceiling_allowlist(self):
        simulated = dict.fromkeys(sorted(SHAPES)[:2], "a reason")
        defects = allowlist_defects(simulated, ceiling=1)
        assert defects and "ceiling of 1" in defects[0]

    def test_the_same_rule_refuses_a_simulated_unknown_shape_and_an_empty_reason(self):
        simulated = {"single-type/typo-nobody-produces": "stale row", "dict-key/peer": "   "}
        defects = allowlist_defects(simulated, ceiling=len(simulated))
        assert [d.split(" is ")[0].split(" carries ")[0] for d in defects] == [
            "'dict-key/peer'",
            "'single-type/typo-nobody-produces'",
        ]


class TestTheRuleFiresOnASimulatedUnstampedRead:
    """The audit rule and the allowlist filter, exercised end to end on one
    construct whose stamp is removed after assembly.

    Deliberately NOT dependent on any of the eight real instances: these keep
    their meaning after the whole plan lands, which is what stops step 9 from
    deleting the instrument's own test coverage along with the disease.
    """

    @staticmethod
    def _two_node_construct() -> Construct:
        return Construct(
            name="stamped-pair",
            nodes=[_producer("seed", RawText), _consumer("use", RawText, Claims)],
        )

    def test_a_fully_stamped_construct_reports_nothing(self):
        construct = self._two_node_construct()
        reads = audit_construct(construct)
        assert [r.key for r in reads] == ["neo_single_input"], "the consumer's single-type read was not seen"
        assert [r for r in reads if not r.stamped] == [], "a resolved read must not be reported as unstamped"

    def test_removing_the_stamp_makes_the_read_reportable(self):
        construct = self._two_node_construct()
        # Mutate the BUILT ir -- model_copy does not re-run __init__, so the
        # normalizer cannot re-stamp what we removed.
        construct.nodes[1] = construct.nodes[1].model_copy(update={"input_sources": None})

        unstamped = [r for r in audit_construct(construct) if not r.stamped]
        assert [r.shape for r in unstamped] == ["single-type/unfed"]
        assert unstamped[0].site == "stamped-pair.use[neo_single_input]"

    def test_the_allowlist_filter_excuses_exactly_the_rows_it_carries(self):
        counts = {"single-type/unfed": 1}
        assert outside_allowlist(counts, {}) == counts
        assert outside_allowlist(counts, {"single-type/unfed": "a row naming its step"}) == {}
        assert outside_allowlist(counts, {"port/sub-construct": "a different row"}) == counts

    def test_an_unstamped_sub_construct_port_is_a_declared_read(self):
        """The third read kind -- ``Construct.port_source`` -- on a simulated miss."""
        child = Construct(
            name="child",
            input=RawText,
            output=MergedResult,
            nodes=[_consumer("inner", RawText, MergedResult)],
        )
        parent = Construct(name="parent", nodes=[_producer("seed", RawText), child])
        assert [r for r in audit_construct(parent) if not r.stamped] == []

        parent.nodes[1].port_source = None
        unstamped = [r for r in audit_construct(parent) if not r.stamped]
        assert unstamped == [
            DeclaredRead(
                construct="parent",
                member="child",
                key="port",
                shape="port/sub-construct",
                stamped=False,
            )
        ]


def test_every_shape_in_the_vocabulary_is_one_the_classifier_can_emit():
    """A shape nothing can produce is a dead allowlist licence.

    Pins the vocabulary against the CLASSIFIER rather than against a copy of
    itself, so a shape string that stops being reachable must be deleted instead
    of lingering as an exemption nobody can trip. Every one of the seven is
    produced here from a real IR object, not asserted from a literal.
    """
    plain = _consumer("plain", RawText, Claims)
    # A Loop is a self-edge, so its node's output must re-enter its own input.
    looper = _consumer("looper", RawText, RawText)
    produced = {
        stamp_instrument._single_type_shape(plain),
        stamp_instrument._single_type_shape(plain | Each(over="seed.items", key="text")),
        stamp_instrument._single_type_shape(looper | Loop(when=lambda d: d is None, max_iterations=2)),
        stamp_instrument._single_type_shape(plain | Portal(to=["other"], max_hops=3)),
        stamp_instrument._dict_key_shape("upstream"),
        stamp_instrument._dict_key_shape(StateKeys.SUBGRAPH_INPUT),
        # The port read has no classifier branch -- `audit_construct` labels it
        # directly -- so it is produced through the audit, above, and named here.
        "port/sub-construct",
    }
    assert produced == set(SHAPES), (
        "the closed shape vocabulary and what the classifier can emit disagree: "
        f"unreachable={sorted(SHAPES - produced)} unlisted={sorted(produced - SHAPES)}"
    )
