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

import pytest
from pydantic import BaseModel

from neograph import ConstructError, node
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
