"""Structural guards for dump_spec (GH issue #9, part a).

Two ratchets, both aimed at defects that would be SILENT -- the dump keeps
returning a plausible document while quietly lying about what is in it.

1. **Loss ids are registered.** Every id ``dump_spec`` can emit exists in
   ``DUMP_LOSS_META`` and every registered id is reachable. An unregistered id
   means a consumer sees a sentinel it cannot interpret; a stale entry means the
   taxonomy claims coverage it no longer has.

2. **One lowering.** ``dump_spec`` must not use ``iter_with_arms`` for its
   structural walk. That iterator DROPS the ``_BranchNode`` sentinel and yields
   both arms in sequence (``_ir_branch.py`` docstring), so a dumper built on it
   emits a linear pipeline in which both arms look unconditional -- with nothing
   recording that a branch existed. The guard bans that specific lowering
   mistake, not the iterator itself: a later data-edge phase may legitimately use
   it, which is why the check is scoped to the structural walk.
"""

from __future__ import annotations

import ast
import pathlib

from pydantic import BaseModel

from neograph import Construct, Loop, Node, dump_spec
from neograph._spec_dump import DUMP_LOSS_META, UNREPRESENTABLE_KEY
from tests.fakes import register_scripted

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"
DUMP_MODULE = SRC / "_spec_dump.py"


class Thing(BaseModel):
    note: str


def _emitted_ids(payload: dict) -> set[str]:
    """Every loss id appearing anywhere in *payload*, in band or in the index."""
    found: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if UNREPRESENTABLE_KEY in value:
                found.add(str(value[UNREPRESENTABLE_KEY]))
            if "id" in value and "tier" in value:
                found.add(str(value["id"]))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found


class TestLossIdsAreRegistered:
    """A sentinel a consumer cannot interpret is worse than no sentinel."""

    def test_every_emitted_id_is_registered(self):
        register_scripted("guard_dump_fn", lambda _in, _cfg: Thing(note="n"))
        looped = Node.scripted(
            "n", fn="guard_dump_fn", inputs=Thing, outputs=Thing
        ) | Loop(when=lambda d: d is None, max_iterations=2)

        emitted = _emitted_ids(dump_spec(Construct("c", nodes=[looped])))

        unregistered = sorted(emitted - set(DUMP_LOSS_META))
        assert unregistered == [], (
            f"dump_spec emitted loss id(s) {unregistered} with no DUMP_LOSS_META entry. "
            "A consumer reading the sentinel has no way to learn what it means."
        )

    def test_every_registered_id_is_reachable_from_the_dumper(self):
        """The other direction: no stale entry claiming coverage that is gone.

        Only ``lose("id", ...)`` CALL SITES count. Scanning every string constant
        in the module -- the obvious implementation -- is vacuous here, because
        each id also appears as its own key in the ``DUMP_LOSS_META`` literal, so
        an entry whose emitting branch had been deleted still looked "cited".
        That exact false pass happened while widening dict-form outputs.
        """
        tree = ast.parse(DUMP_MODULE.read_text())

        # The registry literal's own keys are excluded; every OTHER string
        # constant counts, because an id legitimately reaches `lose()` through a
        # tuple loop or a default argument, not only as a literal call arg.
        registry_keys: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign | ast.Assign) and isinstance(node.value, ast.Dict):
                targets = (
                    [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
                )
                if any(
                    isinstance(t, ast.Name) and t.id == "DUMP_LOSS_META" for t in targets
                ):
                    registry_keys = {id(k) for k in node.value.keys}

        cited = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in registry_keys
        }

        stale = sorted(set(DUMP_LOSS_META) - cited)
        assert stale == [], (
            f"DUMP_LOSS_META entries {stale} are never emitted by _spec_dump.py. "
            "Remove them, or wire the case that should produce them."
        )

    def test_every_registered_tier_is_known(self):
        assert {meta.tier for meta in DUMP_LOSS_META.values()} <= {"NO_REPR", "NO_SLOT"}


class TestOneLowering:
    """dump_spec must not adopt the walker that erases branch structure."""

    def test_structural_walk_does_not_use_iter_with_arms(self):
        source = DUMP_MODULE.read_text()
        called = {
            n.func.id
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }

        assert "iter_with_arms" not in called, (
            "_spec_dump.py calls iter_with_arms. That iterator DROPS the _BranchNode "
            "sentinel and yields both arms in sequence, so the dump would show both "
            "arms as unconditional with no record that a branch existed. Walk raw "
            "construct.nodes for structural dispatch (GH #9 / the C1 correction)."
        )

    def test_branch_node_is_reported_rather_than_silently_flattened(self):
        """The behavioural half: a branch must leave a mark in the output."""
        from neograph._ir_branch import _BranchMeta, _BranchNode, _ConditionSpec

        register_scripted("guard_seed_b", lambda _in, _cfg: Thing(note="n"))
        register_scripted("guard_arm_b", lambda _in, _cfg: Thing(note="n"))

        seed = Node.scripted("seed", fn="guard_seed_b", outputs=Thing)
        arm = Node.scripted("arm", fn="guard_arm_b", inputs=Thing, outputs=Thing)
        meta = _BranchMeta(
            condition_spec=_ConditionSpec(
                source_node=seed,
                attr_chain=["note"],
                op_fn=lambda value, _t: bool(value),
                op_str="route",
                threshold=None,
            ),
            true_arm_nodes=[arm],
            false_arm_nodes=[],
        )

        payload = dump_spec(Construct("branchy", nodes=[seed, _BranchNode(meta, 0)]))

        assert any(
            isinstance(ref, dict) and ref.get(UNREPRESENTABLE_KEY) == "branch_node"
            for ref in payload["pipeline"]["nodes"]
        ), f"the branch vanished from the pipeline: {payload['pipeline']['nodes']}"

    # --- meta-tests: prove the detectors detect ---

    def test_id_scanner_finds_an_in_band_sentinel(self):
        assert _emitted_ids({"a": {UNREPRESENTABLE_KEY: "raw_fn"}}) == {"raw_fn"}

    def test_id_scanner_finds_a_manifest_entry(self):
        assert _emitted_ids({"neograph/losses": [{"id": "renderer", "tier": "NO_REPR"}]}) == {
            "renderer"
        }

    def test_id_scanner_ignores_an_ordinary_dict(self):
        """'Would-be-missed' inverse: a plain spec node must not read as a loss."""
        assert _emitted_ids({"name": "seed", "outputs": "Thing", "id": "not-a-loss"}) == set()
