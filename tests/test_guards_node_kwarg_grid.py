"""Structural guard: pins the CURRENT @node modifier-sugar dispatch behavior
before neograph-jtawq.4's ModifierCombo-keyed registry replaces it (design
Phase 0 -- bd show neograph-jtawq.4 for the full refined plan).

Enumerates every 2**5=32 subset of the five modifier-trigger kwarg groups
(each / oracle / operator / loop / portal). For each subset the EXPECTED
outcome is DERIVED from ``_COMBO_MAP`` (``modifiers.py``) -- never
hand-listed -- so the grid cannot drift from the one composition/validity
authority it exists to pin:

  * valid combo (``frozenset(triggered names)`` a key of ``_COMBO_MAP``) ->
    decoration succeeds; asserts BOTH which modifier slots got populated AND
    that the specific kwarg VALUES passed for each triggered group (and one
    satellite per group) actually landed on the built modifier instances.
  * invalid combo -> decoration raises ``ConstructError`` (today, via one of
    decorators.py's eager pre-checks or the pipe-layer's ``ModifierSet``
    mutual-exclusion checks -- this grid does not care which raises, only
    that one does; message-shape is a separate, narrower regression test).

This closes the value-threading gap the research pass found (bd show
neograph-jtawq.4 notes): today Loop and Portal have ZERO @node-path value
assertions anywhere in the suite; Each/Oracle/Operator are only partially
asserted. Loop and Operator's ``self`` on Node forces every cell to share one
input/output type (``_GridIO``) so Loop's self-edge compatibility check never
interferes with an unrelated cell.

Non-vacuity is proven by mutation in the write-test/implement atoms for
neograph-jtawq.4 (edit-mutate the decorator, watch a targeted cell fail,
revert) -- not re-encoded here as a permanent test, per the design's Phase 0
instructions.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import pytest
from pydantic import BaseModel

from neograph import ConstructError, node
from neograph._node_modifier_kwargs import MODIFIER_KWARGS
from neograph._runtime_registry import _decoration_registry
from neograph.modifiers import _COMBO_MAP, Each, Loop, Operator, Oracle, Portal


class _GridIO(BaseModel, frozen=True):
    """Single input/output type for every grid cell -- keeps Loop's
    self-edge compatibility check (output must feed back as input) trivially
    satisfied regardless of which other modifiers are also in the cell."""

    text: str


TRIGGER_NAMES: tuple[str, ...] = ("each", "oracle", "operator", "loop", "portal")

#: Representative kwargs per triggered group -- both the TRIGGER kwarg(s) and
#: at least one SATELLITE kwarg with a non-default value, so presence AND
#: value-fidelity are both provable from the same grid cell. No two groups
#: share a kwarg NAME except ``on_exhaust`` (loop/portal) -- and loop+portal
#: never co-occur in a valid combo, so the two never collide for a cell this
#: grid asserts values on.
_TRIGGER_KWARGS: dict[str, dict[str, object]] = {
    "each": {"map_over": "seed.items", "map_key": "label", "map_on_error": "collect"},
    "oracle": {
        "ensemble_n": 4,
        "merge_fn": "grid_merge",
        "models": ["reason", "fast"],
        "merge_model": "creative",
    },
    "operator": {"interrupt_when": "grid_needs_review"},
    "loop": {"loop_when": "grid_needs_more_work", "max_iterations": 7, "on_exhaust": "last"},
    "portal": {"portal": ["peer-a", "peer-b"], "route": "next_hop", "max_hops": 6, "on_exhaust": "exit"},
}


def _kwargs_for(names: frozenset[str]) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for trigger_name in names:
        kwargs.update(_TRIGGER_KWARGS[trigger_name])
    return kwargs


def _all_subsets() -> list[frozenset[str]]:
    subsets = []
    for bits in itertools.product([False, True], repeat=len(TRIGGER_NAMES)):
        names = frozenset(n for n, b in zip(TRIGGER_NAMES, bits, strict=True) if b)
        subsets.append(names)
    return subsets


ALL_SUBSETS = _all_subsets()
assert len(ALL_SUBSETS) == 32, "must enumerate exactly 2**5 subsets"


def _subset_id(names: frozenset[str]) -> str:
    return "+".join(sorted(names)) or "bare"


class TestNodeKwargGrid:
    """Phase 0 (neograph-jtawq.4): pin today's @node modifier-dispatch
    behavior across all 32 kwarg-trigger subsets before the dispatch
    rewrite lands."""

    @pytest.mark.parametrize("names", ALL_SUBSETS, ids=[_subset_id(s) for s in ALL_SUBSETS])
    def test_grid_cell(self, names: frozenset[str]) -> None:
        kwargs = _kwargs_for(names)
        is_valid = names in _COMBO_MAP

        if not is_valid:
            with pytest.raises(ConstructError):

                @node(outputs=_GridIO, name=f"grid-{_subset_id(names)}", **kwargs)
                def _fn(seed: _GridIO) -> _GridIO: ...

            return

        @node(outputs=_GridIO, name=f"grid-{_subset_id(names)}", **kwargs)
        def _fn(seed: _GridIO) -> _GridIO: ...

        expected_combo = _COMBO_MAP[names]
        assert _fn.modifier_set.combo is expected_combo, (
            f"subset {sorted(names)} built combo {_fn.modifier_set.combo}, expected {expected_combo}"
        )

        if "each" in names:
            each = _fn.get_modifier(Each)
            assert each is not None
            assert each.over == "seed.items"
            assert each.key == "label"
            assert each.on_error == "collect"
        if "oracle" in names:
            oracle = _fn.get_modifier(Oracle)
            assert oracle is not None
            assert oracle.n == 4
            assert oracle.merge_fn == "grid_merge"
            assert oracle.models == ["reason", "fast"]
            assert oracle.merge_model == "creative"
        if "operator" in names:
            operator = _fn.get_modifier(Operator)
            assert operator is not None
            assert operator.when == "grid_needs_review"
        if "loop" in names:
            loop = _fn.get_modifier(Loop)
            assert loop is not None
            assert loop.when == "grid_needs_more_work"
            assert loop.max_iterations == 7
            assert loop.on_exhaust == "last"
        if "portal" in names:
            portal = _fn.get_modifier(Portal)
            assert portal is not None
            assert portal.to == ["peer-a", "peer-b"]
            assert portal.route == "next_hop"
            assert portal.max_hops == 6
            assert portal.on_exhaust == "exit"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 (neograph-5nvb0): the dangling-satellite strictness gate
# ═══════════════════════════════════════════════════════════════════════════
#
# A SATELLITE kwarg configures a modifier once triggered but never triggers it
# (``MODIFIER_KWARGS`` row semantics). Passing one with NONE of its owning
# triggers present means the value can reach no modifier the derived
# ``ModifierCombo`` actually carries -- it is silently dropped today. Phase 3
# makes that a decoration-time ``ConstructError``.
#
# The case set below is DERIVED from ``MODIFIER_KWARGS``, not hand-listed, so a
# satellite added to a row later automatically gains a dangling cell (refined
# plan, Finding 3). This is a SEPARATE parametrization from the 32-subset grid
# above -- ``TRIGGER_NAMES`` / ``ALL_SUBSETS`` are deliberately untouched.
#
# Three of the derived cells (``map_key``, ``route``, ``max_hops``) already
# raise today via ``decorators.py``'s eager pre-checks, whose messages are
# kwarg-named and BETTER than a generic one. They stay in the derived set --
# they cost nothing, they assert the same contract, and excluding them would
# reintroduce the hand-maintained list this parametrization exists to remove.
# The narrower "4 unshadowed patterns only" rule applies to the check-fixture
# ``# CHECK_ERROR:`` files, where matching the wrong message would prove
# nothing.


def _satellite_owners() -> dict[str, frozenset[str]]:
    """satellite kwarg -> every trigger that could legitimately own it.

    Derived by union over ``MODIFIER_KWARGS`` rows, so ``on_exhaust`` (a
    satellite of BOTH the loop and portal rows) correctly requires
    ``loop_when`` OR ``portal`` and is dangling only when neither is present.
    """
    owners: dict[str, set[str]] = {}
    for row in MODIFIER_KWARGS:
        for satellite in row.satellites:
            owners.setdefault(satellite, set()).update(row.triggers)
    return {k: frozenset(v) for k, v in owners.items()}


SATELLITE_OWNERS: dict[str, frozenset[str]] = _satellite_owners()

#: A NON-DEFAULT value per satellite. Values must be hand-written (they are
#: type-specific), but coverage is not: the assertion below fails the module at
#: import if a new ``MODIFIER_KWARGS`` satellite has no value here, so the
#: derived case set can never silently shrink.
_DANGLING_VALUES: dict[str, Any] = {
    "map_key": "label",
    "map_on_error": "collect",
    "merge_pre_process": lambda variants: variants,
    "merge_post_process": lambda merged: merged,
    "merge_fallback": lambda variants: variants[0],
    "merge_model": "creative",
    "max_iterations": 7,
    "on_exhaust": "last",
    "route": "next_hop",
    "max_hops": 6,
}

assert set(_DANGLING_VALUES) == set(SATELLITE_OWNERS), (
    "every MODIFIER_KWARGS satellite needs a non-default dangling value; missing: "
    f"{sorted(set(SATELLITE_OWNERS) - set(_DANGLING_VALUES))}, "
    f"stale: {sorted(set(_DANGLING_VALUES) - set(SATELLITE_OWNERS))}"
)

DANGLING_SATELLITES: list[str] = sorted(SATELLITE_OWNERS)


class TestDanglingSatelliteKwargsRejected:
    """Phase 3 (neograph-5nvb0): a satellite kwarg with none of its owning
    triggers must fail loudly at decoration time, naming the offending kwarg
    and what it requires."""

    @pytest.mark.parametrize("satellite", DANGLING_SATELLITES, ids=DANGLING_SATELLITES)
    def test_satellite_without_owning_trigger_raises(self, satellite: str) -> None:
        owners = SATELLITE_OWNERS[satellite]

        with pytest.raises(ConstructError) as exc_info:

            @node(outputs=_GridIO, name=f"dangling-{satellite}", **{satellite: _DANGLING_VALUES[satellite]})
            def _fn(seed: _GridIO) -> _GridIO: ...

        message = str(exc_info.value)
        assert f"{satellite}=" in message, (
            f"the error must NAME the offending kwarg as '{satellite}=' (the repo's kwarg-named "
            f"idiom, cf. 'max_hops= requires portal='), got:\n{message}"
        )
        assert any(f"{trigger}=" in message for trigger in owners), (
            f"the error must name at least one owning trigger of '{satellite}' "
            f"({sorted(owners)}) so the user knows what to add, got:\n{message}"
        )

    def test_explicitly_default_satellite_value_still_accepted(self) -> None:
        """Positive control -- GREEN both before and after Phase 3.

        ``sugar_kwargs`` is a ``locals()`` snapshot (decorators.py:249), so an
        explicitly-passed default is indistinguishable from unset. ``map_on_error``
        is the ONLY one of node()'s kwargs with a non-None default ('raise'), so
        this cell is what pins value-vs-default rather than the design doc's
        superseded ``is not None`` test -- under which EVERY node in the codebase
        would be rejected.
        """

        @node(outputs=_GridIO, name="dangling-default-ok", map_on_error="raise")
        def _fn(seed: _GridIO) -> _GridIO: ...

        assert _fn.get_modifier(Each) is None


class TestGateRunsBeforeBuilderSideEffects:
    """Phase 3 (neograph-5nvb0), refined plan Finding 1: the Core Invariant's
    "BEFORE any builder side effect fires" clause, made detectable.

    ``_build_oracle_kwargs`` (_node_modifier_kwargs.py:116-133) emits a
    body-as-merge ``UserWarning`` AND calls ``register_scripted`` as a side
    effect. A rejected node must leave neither behind -- the same
    side-effect-leak class Phase 2 fixed for the oracle+loop / oracle+portal
    pairs. Asserting only "ConstructError is raised" would not notice a future
    edit that moves the gate call below decorators.py:546.

    Shape note: the review sketched this as ``merge_fn='m', ensemble_n=3``, but
    that combination is side-effect-FREE (``_build_oracle_kwargs`` only fires
    the warning + shim when ``models=`` is set with neither ``merge_fn`` nor
    ``merge_prompt``), so it could not detect a misplaced gate at all. This uses
    ``models=[...]`` -- verified to leak a ``_body_merge_*`` registry entry and a
    ``UserWarning`` today -- which is the shape the finding was actually about.
    """

    def test_rejected_oracle_node_leaves_no_shim_and_no_warning(self) -> None:
        scripted_before = set(_decoration_registry.scripted)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ConstructError):
                # ORACLE combo (triggered by models=), with max_iterations
                # dangling -- it owns no modifier this combo carries.
                @node(
                    outputs=_GridIO,
                    name="leak-order",
                    prompt="summarize ${seed}",
                    model="reason",
                    models=["reason", "fast"],
                    max_iterations=7,
                )
                def _fn(seed: _GridIO) -> _GridIO: ...

        leaked = set(_decoration_registry.scripted) - scripted_before
        assert leaked == set(), (
            "a rejected @node must not leave a scripted shim behind -- the gate has to run "
            f"BEFORE the builder dispatch (decorators.py, between :544 and :546); leaked: {sorted(leaked)}"
        )
        assert [str(w.message) for w in caught] == [], (
            "a rejected @node must emit no warning -- _build_oracle_kwargs' body-as-merge "
            "UserWarning fired, which means a builder ran before the gate"
        )
