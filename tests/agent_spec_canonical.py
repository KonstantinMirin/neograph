"""The ONE test-side canonical form of an exported Agent Spec ``Flow``.

Core Invariant: two exported Flows that carry the same CONTENT must compare
equal, and two that differ in content must not -- independent of the random
UUIDs ``to_dict()`` mints on every run.

Why this module exists. ``pyagentspec``'s ``Flow.model_dump(mode='json')``
RAISES (serialization-context error), so ``flow.to_dict()`` is the only
JSON-native form -- and it stamps a fresh random UUID on every component ``id``
and references components through those UUIDs (``$component_ref`` plus a
``$referenced_components`` map). A raw dump is therefore non-deterministic
run-to-run and cannot be compared at all. ``canonicalize`` is the fold that makes
comparison possible, and it is the only working comparison key for an exported
Flow.

It was module-private in ``tests/test_agent_spec_refactor_snapshot.py`` while it
had exactly one caller. It has two now (the snapshot golden and the differential
export harness, ``tests/test_agent_spec_differential_export.py``), and the
sanctioned way to share a test-side reader is a non-test module under ``tests/``
-- exactly as ``tests/agent_spec_flow_walk.py`` does for the control graph. The
extraction is a PURE MOVE: the bodies below are byte-identical to the originals,
and the proof is that ``tests/test_agent_spec_refactor_snapshot.py`` stays
identically green (40 passed) across it, comparing against a 1.5 MB golden that
is append-only and never regenerated.

Division of labour with the two sibling readers, so nothing here overlaps them:

  * ``agent_spec_flow_walk`` reads the Flow OBJECT graph -- control edges
    (``branch_adjacency``) and data edges (``wired_edges``). It is the single
    sanctioned ``data_flow_connections`` reader, capped tree-wide by
    ``tests/test_guards_agent_spec_data_flow_reads.py``.
  * this module reads the SERIALIZED dump -- the whole document, folded to a
    deterministic tree.

A whole-document comparison and an edge-set comparison answer different
questions, and the differential harness needs both: the document to prove a pair
differs AT ALL, the edge set to prove WHERE. Do not use one where the other is
meant -- ``neograph-9axw6.1`` measured an ``output_from`` pair whose documents
differ (an inert metadata marker moved) while the wires stayed identical, so a
document-level "these differ" reads as success over a live defect.

Like ``agent_spec_flow_walk``, this module imports NOTHING from ``neograph`` and
nothing from ``pyagentspec``: the Flow is read duck-typed through ``to_dict()``,
so the module is always importable.
"""

from __future__ import annotations

from typing import Any


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
    (``tests/test_agent_spec_refactor_snapshot.py`` asserts both sentinels
    absent; the "asserted absent below" of the pre-extraction docstring pointed
    at that assertion, which stayed where it was).
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
