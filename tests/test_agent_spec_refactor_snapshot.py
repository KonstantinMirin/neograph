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
UUID (``$component_ref`` + a ``$referenced_components`` map), so the raw dump is
non-deterministic run-to-run. ``_canonicalize`` folds that away by RESOLVING
every ``$component_ref`` inline (following the reference into
``$referenced_components``) and DROPPING every ``id`` field -- leaving a fully
expanded, id-free tree whose content (component names, Property titles,
metadata, node ordering, descriptions) is fully deterministic. Scripted Oracle
variants legitimately share a ServerTool NAME across variants, so re-keying by
name would collide -- inlining sidesteps that.

``$referenced_components`` is NOT top-level-only: a subflow-bearing node (the
``MapNode`` an EACH cell lowers to) carries its OWN nested map, whose KEYS are
themselves random UUIDs. Resolving against the top-level map alone left those
refs dangling (the ``$unresolved_ref`` sentinel fired) AND left the nested,
UUID-keyed map in the tree, so an EACH cell could not be snapshotted at all.
``_canonicalize`` therefore MERGES every ``$referenced_components`` map found
anywhere in the dump (UUIDs are globally unique, so the merge is collision-free)
and drops the key at every depth. Verified strictly wider, never different: all
24 pre-existing BARE/ORACLE cells canonicalize BYTE-IDENTICALLY under the merged
form -- they simply have no nested map (neograph-tjpn4).

The golden lives in ``tests/fixtures/agent_spec_refactor_snapshot.json`` (a
static, committed artifact captured ONCE from develop HEAD -- NOT regenerated at
test time, which would be a vacuous self-comparison).

## Extending the set (neograph-tjpn4, R-RM1) -- ADD keys, never regenerate

The golden is a historical artifact: its 24 original BARE/ORACLE values are the
neograph-2s2o6 proof, captured from an earlier develop HEAD. A wholesale
regeneration silently REBASELINES them, destroying that proof if anything (a
pyagentspec version bump, a prior landed change) has drifted since capture --
and there is no regeneration tooling in-repo by design.

So when a new cell is added to ``REPRESENTATIVE_CELLS``, APPEND only its key to
the JSON, on the PRE-change tree, and verify ``git diff --stat`` on the fixture
shows additions and ZERO modifications. If an existing value would change, that
is a pre-existing drift FINDING to report, never something to absorb. The
neograph-tjpn4 EACH/LOOP/OPERATOR extension was captured exactly that way (the
one-off snippet is recorded in that bead's notes).

POLARITY: this whole module is GREEN before and after the tjpn4 migration -- it
pins today's bytes. It is not a TDD-red artifact; do not "fix" it for passing.

## The ONE sanctioned value modification to date (neograph-p7dyq, 2026-08-03)

The append-only rule above governs REFACTORS, whose whole premise is zero
behavior change. It does not cover an intentional BUG FIX that legitimately
changes the exported bytes -- and this golden had captured a bug: the exporter
titled a ``$ref``-shaped field after the referenced DEFINITION rather than the
FIELD, which silently collapsed two same-typed nested fields into one on import.
Fixing it changed exactly 24 values across the 4 EACH cells -- every one of them
a ``list[Elem]`` item schema's ``title``, ``'Elem'`` -> ``'groups'`` (the field
name, which is what a primitive item schema such as ``list[str]`` already
carried). Nothing else in the tree moved; that was verified by a full recursive
diff of golden-vs-actual across every cell BEFORE patching, and the patch touched
only those 24 lines (``git diff --numstat``: 24 insertions, 24 deletions).

The discipline that survives: a changed existing value is still never absorbed
silently. It is either a drift finding, or -- as here -- an intended fix that
names its bead, states the diff's exact extent, and proves the extent.

## The SECOND sanctioned value modification (neograph-8zvd1, 2026-08-03)

Same category, and the golden had captured this bug too: a dict-form input's
``'{key}.{field}'`` qualification was applied by MUTATING ``Property.title``
after construction, so it (a) never reached ``json_schema['title']`` -- the only
thing ``Property`` serializes -- and (b) skipped the title validator, which
rejects ``.``. The exported Flow therefore declared BARE input titles while every
``DataFlowEdge`` still routed to the qualified name: a dangling edge that
``Flow.from_dict`` refuses, for essentially every @node pipeline. The fix
qualifies at construction time and moves the separator to ``:`` (legal in a
title; unambiguous to split, unlike ``_``).

Extent, measured by a full recursive golden-vs-actual diff across all 37 cells
BEFORE patching: 1010 changed values, ZERO added/removed/reordered nodes, and
only three keys touched -- ``title`` (962: bare ``'a'`` -> ``'pa:a'``, the
qualification now actually reaching the wire, plus already-qualified
``'pa.a'`` -> ``'pa:a'``), ``destination_input`` (25) and DataFlowEdge ``name``
(23), both purely the separator. ``git diff --numstat`` on the fixture: 1010
insertions, 1010 deletions (1034/1034 counting the p7dyq rows landed alongside).

## The THIRD sanctioned value modification (neograph-s7zt3.15, 2026-08-03)

Same category again, and again the golden had captured the bug: the exported
control graph was not one a foreign Agent Spec runtime could execute. An
Operator's BODY had no inbound edge (the item's incoming edge went straight to
the gate), the pause ``InputMessageNode`` had no outgoing edge, every Oracle
VARIANT had no inbound edge, and a Loop was entered at its check rather than its
body -- inverting neograph's do-while into a while-do.

Extent, measured by a full recursive golden-vs-actual diff across all 37 cells
BEFORE patching: 22 cells changed -- all 16 ORACLE, all 3 LOOP, all 3 OPERATOR --
and in every one of the 37, the change was confined ENTIRELY to
``control_flow_connections``. The ``nodes`` list, ``data_flow_connections``,
``start_node`` and every metadata marker are byte-identical, and the 8 BARE plus
4 EACH cells did not move at all. That containment was asserted by the patch
script itself, which refuses any cell that differs outside the control edges, so
the modification could not silently widen. Line churn is large
(``git diff --numstat``: 6074 insertions, 4595 deletions) only because
``_canonicalize`` inlines each edge's ``from_node``/``to_node`` in full, so
re-pointing one edge rewrites a whole nested node blob.

## The FOURTH sanctioned value modification (neograph-qtfof.9, 2026-08-14)

Same category: the golden had captured the qtfof.9 bug too -- a synthetic
EndNode was never fed a real DataFlowEdge, so a third-party runtime's
invoke() always returned {} regardless of what the graph computed. The fix
wires the terminal producer (construct.nodes[-1], when BARE or ORACLE) into
the EndNode with real declared outputs+inputs and one new DataFlowEdge.

Extent, measured by a full recursive golden-vs-actual diff across all 34
REPRESENTATIVE_CELLS BEFORE patching: 27 cells changed (every BARE/ORACLE
cell, plus the 3 OPERATOR cells whose guarded body is BARE-shaped), 7
unchanged (all EACH/LOOP cells -- qtfof.9's own documented scope boundary,
see _agent_spec_boundary.py). In every changed cell the diff was confined to
exactly: the Flow's own `.outputs` (+1), the EndNode's `.outputs`/`.inputs`
(+1 each, pyagentspec infers EndNode.inputs from its declared outputs), and
`.data_flow_connections` (+1 edge) -- verified per-cell before patching, not
assumed. Large line churn (`git diff --numstat`: 8942 insertions, 5939
deletions) is `_canonicalize`'s full-inlining artifact, not evidence of a
wider change -- confirmed by an exact `old[k] == new[k]` check on all 7
unchanged keys plus `set(old) == set(new)`.

## The FIFTH sanctioned value modification (neograph-qtfof.6, 2026-08-14)

Same category: the golden had captured the qtfof.6 bug too -- a Loop's exported
``BranchingNode`` decision lived only in ``neograph/loop_spec`` metadata, with no
real ``DataFlowEdge`` feeding it, so a metadata-blind runtime silently fell to
``DEFAULT_BRANCH`` and never followed the back-edge. The fix synthesizes a
predicate ``ToolNode`` (body -> predicate -> check, real edges both hops) when
``Loop.when`` is a statically-determinable expression, and wires
``DEFAULT_BRANCH`` to the same exit as ``done`` (previously a dead end).

Extent, measured by a full recursive golden-vs-actual diff across all 34
REPRESENTATIVE_CELLS BEFORE patching: exactly 3 cells changed (scripted-loop-
single, scripted-loop-dict, think-loop-single -- every LOOP cell in the
representative set; a registered-condition-NAME ``when`` synthesizes no
predicate and is unaffected, out of scope per the ticket), 31 unchanged
(verified `old[k] == new[k]` for all 31 plus `set(old) == set(new)`, zero
added/removed keys). In each changed cell the diff is confined to: the renamed
control edge (``..._loop_body_to_check`` -> ``..._loop_body_to_predicate``),
the new predicate ``ToolNode`` + its ``ServerTool``, two new
``DataFlowEdge``s (body->predicate, predicate->check), and one new
``ControlFlowEdge`` for ``DEFAULT_BRANCH``'s exit.

## The SIXTH sanctioned value modification (neograph-qtfof.7, 2026-08-16)

Same category: the golden had captured the qtfof.7 bug too -- an Each-modified
node's MapNode declared its fan-out receiver as FLATTENED per-field
Properties (``item:v``), so pyagentspec inferred ``iterated_item:v`` instead
of one ``iterated_item`` Property a real collection edge could target. The
fix reshapes the StartNode's (and the inner body's own) receiver into ONE
ObjectProperty titled with the ``fan_out_param`` key, and wires a real
``DataFlowEdge`` from the producer into it -- CONDITIONAL on pyagentspec's own
``json_schemas_have_same_type`` predicate (mirroring qtfof.6's precedent), so
an incompatible producer/receiver shape still emits no edge rather than a
silent broadcast.

Extent, measured by a full recursive golden-vs-actual diff across all 34
REPRESENTATIVE_CELLS BEFORE patching: exactly 3 cells changed (scripted-each-
single, scripted-each-dict, scripted-each-context -- every SCRIPTED plain-Each
cell in the representative set), 31 unchanged (verified `set(old) ==
set(new)` for all 31, zero added/removed top-level keys; think-each-single is
translation-eligible and deliberately carved out of the reshape, so it is
correctly among the unchanged). In each changed cell the diff is confined to
exactly three top-level keys (`nodes`, `control_flow_connections`,
`data_flow_connections`) -- verified `set(old.keys()) == set(new.keys())` per
cell before patching, not assumed. Large line churn
(`git diff --numstat`) is expected and not evidence of a wider change: the new
ObjectProperty receiver shape is structurally DEEPER (nested sub-properties)
than the flattened form it replaces, and `_canonicalize` full-inlines it on
both sides of an `anyOf` union.

Run with::

    uv run --extra agent-spec pytest tests/test_agent_spec_refactor_snapshot.py
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

pytest.importorskip("pyagentspec")

from neograph._agent_spec import _MARK_MODIFIER, _MARK_VARIANT, to_agent_spec  # noqa: E402

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
    # -- EACH / LOOP / OPERATOR (added by neograph-tjpn4, research risk T5).
    #    The original 24 cells above cover only the BARE and ORACLE arms of
    #    _lower_construct_item. neograph-tjpn4 restructures ALL FIVE arms onto
    #    COMBO_DECOMPOSITION, so the three unpinned arms -- whose only prior net
    #    was the matrix's weak "does not raise + flow.nodes non-empty" -- get a
    #    byte-level pin too. Without these, an arm could start emitting a
    #    different MapNode/BranchingNode/InputMessageNode shape (a dropped
    #    each_spec marker, a lost iterated_ prefix, a re-ordered pause composite)
    #    with the whole suite green.
    #    scripted + think per arm covers _lower_item_body's ToolNode and LlmNode
    #    lowerings; agent/act each-dict/context and loop-dict are UNREPRESENTABLE
    #    (neograph's own assembly rejects them), so they cannot be pinned here.
    "scripted-each-single",
    "scripted-each-dict",
    "scripted-each-context",
    "think-each-single",
    "scripted-loop-single",
    "scripted-loop-dict",
    "think-loop-single",
    "scripted-operator-single",
    "scripted-operator-dict",
    "think-operator-single",
)


def _collect_referenced_components(obj: Any, into: dict[str, Any]) -> None:
    """Merge every ``$referenced_components`` map in the dump into one.

    A subflow-bearing node (``MapNode``, ``FlowNode``) nests its own map, so the
    top-level one is incomplete. UUID keys are globally unique, so merging is
    collision-free (neograph-tjpn4 -- without this, EACH cells cannot be
    canonicalized: their refs dangle and the nested UUID-keyed map itself makes
    the tree non-deterministic).
    """
    if isinstance(obj, dict):
        nested = obj.get("$referenced_components")
        if isinstance(nested, dict):
            into.update(nested)
        for value in obj.values():
            _collect_referenced_components(value, into)
    elif isinstance(obj, list):
        for item in obj:
            _collect_referenced_components(item, into)


def _canonicalize(flow: Any) -> Any:
    """Fold ``flow.to_dict()`` into a deterministic, id-free tree.

    Resolves every ``$component_ref`` inline against the MERGED
    ``$referenced_components`` maps (top-level plus every nested subflow map)
    and drops every random-UUID ``id`` field, plus the ``$referenced_components``
    key at every depth. A ``$cycle``/``$unresolved_ref`` sentinel would surface a
    structural surprise loudly rather than silently corrupt the snapshot
    (asserted absent below).
    """
    d = flow.to_dict()
    refs: dict[str, Any] = {}
    _collect_referenced_components(d, refs)

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
            return {k: resolve(v, seen) for k, v in obj.items() if k not in ("id", "$referenced_components")}
        if isinstance(obj, list):
            return [resolve(item, seen) for item in obj]
        return obj

    return resolve(d, frozenset())


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

    def test_all_five_lower_construct_item_arms_are_pinned(self) -> None:
        """Arm-gap closure (neograph-tjpn4, risk T5): the representative set must
        span EVERY arm of ``_lower_construct_item``'s modifier dispatch -- BARE,
        ORACLE, EACH, LOOP and OPERATOR -- not just the BARE/ORACLE pair the
        original 24 cells covered. The tjpn4 migration restructures all five onto
        ``COMBO_DECOMPOSITION``, so an unpinned arm is an unproven arm."""
        for arm in ("bare", "oracle", "each", "loop", "operator"):
            assert any(f"-{arm}-" in c for c in REPRESENTATIVE_CELLS), (
                f"no representative cell exercises the {arm.upper()} arm of "
                "_lower_construct_item -- that arm has no byte-level pin"
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
            and n.metadata.get(_MARK_MODIFIER) == "oracle"
            and _MARK_VARIANT in n.metadata
        ]
        assert len(variants) == 2, "expected 2 scripted Oracle variant ToolNodes"
        for i, variant in enumerate(sorted(variants, key=lambda n: n.metadata[_MARK_VARIANT])):
            assert variant.metadata is not None and variant.metadata != {}, (
                "the _lower_oracle scripted-variant ToolNode carries the NON-empty "
                "neograph/variant marker dict (never None, never bare {})"
            )
            assert variant.metadata[_MARK_VARIANT] == i
            assert variant.tool.description == f"Oracle variant {i} for 'gen'", (
                "the per-variant ServerTool description on the _lower_oracle scripted path must be preserved"
            )
