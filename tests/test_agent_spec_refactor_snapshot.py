"""BYTE-SNAPSHOT zero-behavior-change proof for neograph-2s2o6 (the pure
structural extract of ``_lower_node``/``_lower_oracle``'s per-``node.mode``
three-way dispatch into one shared ``_lower_generation_step``).

## Why this test exists (the REAL zero-change net -- refinement x02ki.12, MEDIUM-1)

The 109-cell coverage matrix (``test_agent_spec_matrix.py``) only asserts
``to_agent_spec`` does NOT raise + ``flow.nodes`` is non-empty -- grep-confirmed
it NEVER compares serialized output. So the refactor's plan-central trap --
a scripted node's ``ToolNode.metadata == {}`` (NOT ``None``; ``{} != None`` on
the pyagentspec wire) -- plus the ServerTool description strings, the Oracle
per-variant model tiers/markers, and node ordering have ZERO covering assertion
in the matrix. A ``metadata or None`` slip during the extract would leave every
matrix + round-trip cell GREEN. This snapshot is the only artifact that catches
it: it pins the EXACT serialized ``Flow`` for a representative construct set
spanning every dispatch path, captured from CURRENT (pre-refactor develop-HEAD)
output.

Contract:
  * It PASSES on current code (it pins current behavior).
  * After the refactor it must STILL pass BYTE-IDENTICAL -- that is the real
    zero-behavior-change proof.

## Canonicalization

``pyagentspec`` ``Flow.model_dump(mode='json')`` RAISES (serialization-context
error), so the JSON-native form is ``flow.to_dict()``. But ``to_dict()`` stamps
a fresh random UUID on every component ``id`` and references components by that
UUID (``$component_ref`` + a top-level ``$referenced_components`` map), so the
raw dump is non-deterministic run-to-run. ``_canonicalize`` folds that away by
RESOLVING every ``$component_ref`` inline (following the reference into
``$referenced_components``) and DROPPING every ``id`` field -- leaving a fully
expanded, id-free tree whose content (component names, Property titles,
metadata, node ordering, descriptions) is fully deterministic. Scripted Oracle
variants legitimately share a ServerTool NAME across variants, so re-keying by
name would collide -- inlining sidesteps that.

The golden lives in ``tests/fixtures/agent_spec_refactor_snapshot.json`` (a
static, committed artifact captured ONCE from develop HEAD -- NOT regenerated at
test time, which would be a vacuous self-comparison).

Run with::

    uv run --extra dev --extra agent-spec pytest tests/test_agent_spec_refactor_snapshot.py
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

pytest.importorskip("pyagentspec")

from neograph._agent_spec import to_agent_spec  # noqa: E402

# Reuse the matrix's mechanically-derived cell builder (NOT hand-typed constructs
# -- the builder IS the factory, per the test conventions' factory/builder rule).
from tests.test_agent_spec_matrix import CELLS, GREEN, build_cell  # noqa: E402

_GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "fixtures" / "agent_spec_refactor_snapshot.json"

# Representative construct set: EVERY dispatch path the refactor touches --
# scripted/think/agent/act BARE (the _lower_node three-way dispatch) AND each of
# those four modes under Oracle with BOTH merge configs (the _lower_oracle
# per-variant dispatch + the merge LlmNode/ToolNode), single AND dict-form input.
# Every id here is a GREEN cell of the matrix (asserted below), so it builds,
# exports, and is a real steady-state shape -- not a red/unrepresentable one.
REPRESENTATIVE_CELLS: tuple[str, ...] = (
    # -- BARE: exercises _lower_node's think / agent-act / scripted-or-raw dispatch
    "scripted-bare-single",
    "scripted-bare-dict",
    "think-bare-single",
    "think-bare-dict",
    "agent-bare-single",
    "agent-bare-dict",
    "act-bare-single",
    "act-bare-dict",
    # -- ORACLE: exercises _lower_oracle's per-variant dispatch across all 4 modes,
    #    both merge_fn (ToolNode merge) and merge_prompt (LlmNode merge), 2 shapes.
    "scripted-oracle-merge_fn-single",
    "scripted-oracle-merge_fn-dict",
    "scripted-oracle-merge_prompt-single",
    "scripted-oracle-merge_prompt-dict",
    "think-oracle-merge_fn-single",
    "think-oracle-merge_fn-dict",
    "think-oracle-merge_prompt-single",
    "think-oracle-merge_prompt-dict",
    "agent-oracle-merge_fn-single",
    "agent-oracle-merge_fn-dict",
    "agent-oracle-merge_prompt-single",
    "agent-oracle-merge_prompt-dict",
    "act-oracle-merge_fn-single",
    "act-oracle-merge_fn-dict",
    "act-oracle-merge_prompt-single",
    "act-oracle-merge_prompt-dict",
)


def _canonicalize(flow: Any) -> Any:
    """Fold ``flow.to_dict()`` into a deterministic, id-free tree.

    Resolves every ``$component_ref`` inline against the top-level
    ``$referenced_components`` map and drops every random-UUID ``id`` field. A
    ``$cycle``/``$unresolved_ref`` sentinel would surface a structural surprise
    loudly rather than silently corrupt the snapshot (asserted absent below).
    """
    d = flow.to_dict()
    refs: dict[str, Any] = d.get("$referenced_components", {})

    def resolve(obj: Any, seen: frozenset[str]) -> Any:
        if isinstance(obj, dict):
            if set(obj.keys()) == {"$component_ref"}:
                ref_id = obj["$component_ref"]
                if ref_id in seen:
                    return {"$cycle": refs.get(ref_id, {}).get("name", ref_id)}
                target = refs.get(ref_id)
                if target is None:
                    return {"$unresolved_ref": ref_id}
                return resolve(target, seen | {ref_id})
            return {k: resolve(v, seen) for k, v in obj.items() if k != "id"}
        if isinstance(obj, list):
            return [resolve(item, seen) for item in obj]
        return obj

    top = {k: v for k, v in d.items() if k != "$referenced_components"}
    return resolve(top, frozenset())


def _load_golden() -> dict[str, Any]:
    return json.loads(_GOLDEN_PATH.read_text())


class TestRepresentativeSetCoversEveryDispatchPath:
    """The representative set must actually span every dispatch path the refactor
    unifies, and every id must be a buildable GREEN matrix cell -- so the snapshot
    cannot silently under-cover (the sdfgz disease the matrix was built to kill)."""

    def test_every_representative_cell_is_a_green_matrix_cell(self) -> None:
        missing = [c for c in REPRESENTATIVE_CELLS if c not in CELLS]
        assert not missing, f"representative cells not in the generated matrix: {missing}"
        not_green = [c for c in REPRESENTATIVE_CELLS if c not in GREEN]
        assert not not_green, f"representative cells that are not GREEN (won't export cleanly): {not_green}"

    def test_all_four_modes_reach_both_lower_node_and_lower_oracle(self) -> None:
        for mode in ("scripted", "think", "agent", "act"):
            assert f"{mode}-bare-single" in REPRESENTATIVE_CELLS, (
                f"{mode} bare cell (the _lower_node dispatch) missing from the set"
            )
            assert any(c.startswith(f"{mode}-oracle-") for c in REPRESENTATIVE_CELLS), (
                f"{mode} oracle cell (the _lower_oracle per-variant dispatch) missing from the set"
            )

    def test_golden_fixture_covers_exactly_the_representative_set(self) -> None:
        golden = _load_golden()
        assert set(golden) == set(REPRESENTATIVE_CELLS), (
            "golden fixture keys diverged from REPRESENTATIVE_CELLS -- recapture the "
            "golden or reconcile the set (missing: "
            f"{set(REPRESENTATIVE_CELLS) - set(golden)}; extra: {set(golden) - set(REPRESENTATIVE_CELLS)})"
        )


class TestExportByteSnapshotUnchanged:
    """The pinned zero-behavior-change net: for every dispatch path, the canonical
    serialized ``Flow`` must equal the golden captured from pre-refactor HEAD.

    PASSES now (pins current behavior); MUST stay byte-identical after the
    _lower_generation_step extract -- a ``metadata or None`` slip, a dropped
    ServerTool description, a reordered node, or a changed Oracle marker all break
    exactly here, where the matrix stays green."""

    @pytest.mark.parametrize("cell_id", REPRESENTATIVE_CELLS)
    def test_canonical_export_matches_golden(self, cell_id: str) -> None:
        golden = _load_golden()
        mode, combo, config, shape = CELLS[cell_id]
        flow = to_agent_spec(build_cell(mode, combo, config, shape))
        canonical = _canonicalize(flow)

        blob = json.dumps(canonical, sort_keys=True)
        assert "$unresolved_ref" not in blob, "a $component_ref did not resolve -- canonicalization is unsound"
        assert "$cycle" not in blob, "unexpected reference cycle in the exported Flow"

        expected = golden[cell_id]
        assert canonical == expected, (
            f"cell {cell_id!r}: exported Flow diverged from the pre-refactor golden.\n"
            f"--- expected (golden) ---\n{json.dumps(expected, sort_keys=True, indent=2)}\n"
            f"--- actual ---\n{json.dumps(canonical, sort_keys=True, indent=2)}"
        )


class TestScriptedMetadataAndDescriptionTraps:
    """The plan-central trap, asserted directly on the LIVE pyagentspec objects
    (complementing the serialized snapshot): a scripted ToolNode's ``metadata``
    is ``{}`` (the default_factory), NOT ``None`` (``{} != None`` on the wire),
    and its ServerTool ``description`` is the current default -- on BOTH the
    ``_lower_node`` scripted path AND the ``_lower_oracle`` scripted-variant path.

    The refinement's fallback names both paths with ``metadata == {}``; that is
    literally true only for the ``_lower_node`` path (which passes NO metadata).
    The ``_lower_oracle`` variant path deliberately attaches the NON-empty
    ``neograph/variant`` marker dict -- so the accurate variant assertion is
    'non-empty markers, NOT None', with its OWN ``Oracle variant i`` description.
    """

    def test_lower_node_scripted_path_toolnode_metadata_is_empty_dict_not_none(self) -> None:
        from pyagentspec.flows.nodes import ToolNode

        mode, combo, config, shape = CELLS["scripted-bare-single"]
        flow = to_agent_spec(build_cell(mode, combo, config, shape))

        target = next(n for n in flow.nodes if isinstance(n, ToolNode) and n.name == "target")
        assert target.metadata == {}, (
            "scripted _lower_node ToolNode.metadata must be {} (the pyagentspec "
            "default_factory), NEVER None -- {} != None on the wire (the refactor's "
            "central trap: an extracted branch passing `metadata or None` would break here)"
        )
        assert target.metadata is not None
        assert target.tool.description == "neograph node 'target' (mode=scripted)", (
            "the default ServerTool description on the _lower_node scripted path must be preserved"
        )

    def test_lower_oracle_scripted_variant_path_metadata_and_description(self) -> None:
        from pyagentspec.flows.nodes import ToolNode

        mode, combo, config, shape = CELLS["scripted-oracle-merge_fn-single"]
        flow = to_agent_spec(build_cell(mode, combo, config, shape))

        variants = [
            n
            for n in flow.nodes
            if isinstance(n, ToolNode)
            and n.metadata
            and n.metadata.get("neograph/modifier") == "oracle"
            and "neograph/variant" in n.metadata
        ]
        assert len(variants) == 2, "expected 2 scripted Oracle variant ToolNodes"
        for i, variant in enumerate(sorted(variants, key=lambda n: n.metadata["neograph/variant"])):
            assert variant.metadata is not None and variant.metadata != {}, (
                "the _lower_oracle scripted-variant ToolNode carries the NON-empty "
                "neograph/variant marker dict (never None, never bare {})"
            )
            assert variant.metadata["neograph/variant"] == i
            assert variant.tool.description == f"Oracle variant {i} for 'gen'", (
                "the per-variant ServerTool description on the _lower_oracle scripted path must be preserved"
            )
