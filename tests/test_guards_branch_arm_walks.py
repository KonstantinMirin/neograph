"""Structural guard: no hand-rolled ``construct.nodes`` walk may silently skip
``_BranchNode`` arm contents (neograph-vn5f).

Root-cause pin. Five documented + six scan-found walks iterated a Construct's
node list with an ``isinstance(item, Node)/(item, Construct)`` guard and
short-circuited on the ``_BranchNode`` sentinel, so nodes inside branch arms
(``_BranchMeta.true_arm_nodes`` / ``false_arm_nodes``) were invisible. The fix
routes every such walk through the single arm-descent primitives
``iter_with_arms`` / ``iter_item_slots`` (``_ir_branch.py``); those wrappers do
not textually iterate ``.nodes`` at the call site, so a migrated walk drops out
of this scan.

This guard AST-scans ``src/neograph`` for every raw ``X.nodes`` iteration (for
loops and comprehensions) and asserts the set matches a content-keyed
ALLOWLIST of the walks that legitimately touch ``.nodes`` directly: the
primitives themselves, the gold-standard arm-aware walks in ``state.py``,
``iter_nodes``, telemetry/logging, the LangGraph ``lg_graph.nodes`` (not a
Construct), the spec-level loader walks, and the deferred ``testing.py`` codegen
(neograph-gfoq). Any NEW raw ``.nodes`` walk fails this guard until it is either
routed through a primitive or added to the allowlist with a justification.

The count is line-number-independent (content-keyed Counter) so a later fix that
shifts line numbers does not spuriously trip the guard — matching the
sweep-verify contract (re-localize by content, check the instance count).
"""

from __future__ import annotations

import ast
import pathlib
from collections import Counter

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"


def _iter_iteration_exprs(tree: ast.AST):
    """Yield the iterated expression of every for-loop and comprehension."""
    for n in ast.walk(tree):
        if isinstance(n, ast.For):
            yield n.iter
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in n.generators:
                yield gen.iter


def _is_dot_nodes(expr: ast.AST) -> bool:
    """True for ``X.nodes`` and ``X.nodes[...]`` (a subscript of ``.nodes``).

    A call wrapper such as ``enumerate(x.nodes)`` or ``iter_with_arms(x)`` is
    NOT matched: the disease is a RAW attribute iteration, and both the
    primitives and ``enumerate`` wrapping are the sanctioned escapes.
    """
    e = expr
    if isinstance(e, ast.Subscript):
        e = e.value
    return isinstance(e, ast.Attribute) and e.attr == "nodes"


def _collect_raw_nodes_walks() -> Counter:
    """Content-keyed Counter of every raw ``.nodes`` iteration in the source."""
    found: Counter = Counter()
    # rglob (not glob): cover subpackages too (src/neograph/testing, schemas) so
    # a .nodes walk moved into a subpackage stays visible to this guard (dyp3
    # moved the scaffold codegen into testing/scaffold.py).
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        src = py_file.read_text()
        lines = src.splitlines()
        tree = ast.parse(src)
        for it in _iter_iteration_exprs(tree):
            if _is_dot_nodes(it):
                found[(py_file.name, lines[it.lineno - 1].strip())] += 1
    return found


# The walks that legitimately iterate `.nodes` directly. Every entry has a
# recorded reason; anything NOT here is a suspected arm-blind walk.
_ALLOWLIST: Counter = Counter({
    # The arm-descent primitive itself — the single source of truth.
    ("_ir_branch.py", "for item in construct.nodes:"): 1,
    # Peer-field set for fan_out inference: intentionally TOP-LEVEL only
    # (arm-sibling fan-in unsupported; documented in normalize_ir).
    ("_ir_normalize.py", "for item in construct.nodes:"): 1,
    # Graph-build dispatch loop — already arm-aware (dispatches _BranchNode to
    # _add_branch_to_graph).
    ("compiler.py", "for item in construct.nodes:"): 1,
    # LangGraph compiled-graph node ids — NOT a Construct node list.
    ("compiler.py",
     'nodes = [n for n in lg_graph.nodes if n not in ("__start__", "__end__")]'): 1,
    # compile_start telemetry — display only, no correctness impact.
    ("compiler.py", "node_names=[n.name for n in construct.nodes],"): 1,
    ("compiler.py", "for n in construct.nodes"): 1,
    # iter_nodes — the leaf-flattening SoT walk, already arm-aware (tdbb).
    ("construct.py", "for item in construct.nodes:"): 1,
    # Spec-level walks — YAML spec, pre-IR; _BranchNode never appears in a spec.
    ("loader.py", "for node_spec in spec.nodes:"): 1,
    ("loader.py", "for ref in spec.pipeline.nodes:"): 1,
    ("loader.py", "for prev_ref in construct_spec.nodes[:i]:"): 1,
    # compute_node_fingerprints — already arm-aware (tdbb, _fingerprint_item).
    ("state.py", "for item in construct.nodes:"): 1,
    # compile_state_model — the gold-standard arm-aware walk (handles
    # branch_nodes explicitly right after these partitions).
    ("state.py", "nodes_only = [n for n in construct.nodes if isinstance(n, Node)]"): 1,
    ("state.py",
     "sub_constructs = [n for n in construct.nodes if isinstance(n, Construct)]"): 1,
    ("state.py",
     "branch_nodes = [n for n in construct.nodes if isinstance(n, _BranchNode)]"): 1,
    # Test-scaffold codegen introspection. The two top-level _collect_items /
    # _collect_edges walks were migrated to iter_with_arms in neograph-gfoq; the
    # remaining inner walk descends a sub-construct's own node list one level
    # down (sub-construct arm-descent is owned by the sub-construct itself).
    ("scaffold.py",
     "sub_nodes = [_node_info(n) for n in item.nodes if isinstance(n, Node)]"): 1,
    # wrap_fan_over_agents — the fan-over-agent auto-wrap pre-pass (neograph-m6d3.6).
    # Deliberately TOP-LEVEL only: it rewrites supported fan-over-agent nodes into
    # isolating sub-constructs, but arm nodes CANNOT be wrapped here (they are added
    # verbatim by _add_arm_nodes). Arm-nested cases are caught fail-loud by the
    # iter_with_arms scan immediately after this walk, so routing this walk through
    # iter_with_arms would be wrong (it must NOT descend into arms).
    ("_fan_agent_wrap.py", "for item in construct.nodes:"): 1,
})


class TestNoArmBlindNodesWalks:
    """Every raw ``construct.nodes`` walk is either arm-aware, the primitive, or
    an explicitly-justified allowlist entry."""

    def test_no_unallowlisted_raw_nodes_walk(self):
        found = _collect_raw_nodes_walks()
        new = found - _ALLOWLIST  # Counter difference: positive residue only
        stale = _ALLOWLIST - found
        msg = []
        if new:
            msg.append(
                "NEW raw `.nodes` walk(s) not in the allowlist — route each "
                "through iter_with_arms / iter_item_slots (arm-aware) or add an "
                "allowlist entry with a justification naming why it is exempt:\n"
                + "\n".join(f"    {f}: {line}" for (f, line), _ in new.items())
            )
        if stale:
            msg.append(
                "STALE allowlist entry — the walk moved or was migrated; remove "
                "it or re-localize by content:\n"
                + "\n".join(f"    {f}: {line}" for (f, line), _ in stale.items())
            )
        assert not msg, "\n\n".join(msg)

    def test_migrated_sites_no_longer_iterate_nodes_directly(self):
        """The eleven migrated walks must NOT reappear as raw `.nodes` walks —
        pins that the fix routed them through the primitives, not that it merely
        added arm handling inline (which would re-duplicate the disease)."""
        found = _collect_raw_nodes_walks()
        # These files' migrated walks read the construct via a primitive now, so
        # no raw `.nodes` for-loop should survive in them (lint/verify/runner/
        # _subconstruct had exactly one arm-relevant walk each).
        for fname in ("lint.py", "verify.py", "runner.py", "_subconstruct.py"):
            offenders = {k: v for k, v in found.items() if k[0] == fname}
            assert not offenders, (
                f"{fname} still has a raw `.nodes` walk after migration: "
                f"{offenders}. It must route through iter_with_arms."
            )


class TestArmBlindDetectorMetaTests:
    """Meta-tests proving the AST detector actually distinguishes an arm-blind
    raw walk from a primitive-routed one (else the guard above is vacuous)."""

    @staticmethod
    def _count(source: str) -> int:
        tree = ast.parse(source)
        return sum(1 for it in _iter_iteration_exprs(tree) if _is_dot_nodes(it))

    def test_detector_flags_a_raw_arm_blind_walk(self):
        """A hand-rolled `for x in construct.nodes` IS detected (positive)."""
        blind = (
            "def walk(construct):\n"
            "    for item in construct.nodes:\n"
            "        handle(item)\n"
        )
        assert self._count(blind) == 1

    def test_detector_ignores_primitive_routed_walk(self):
        """A walk routed through the primitive is NOT detected — migration is
        the sanctioned escape from the guard (would-be-missed case)."""
        routed = (
            "def walk(construct):\n"
            "    for item in iter_with_arms(construct):\n"
            "        handle(item)\n"
        )
        assert self._count(routed) == 0

    def test_detector_ignores_enumerate_wrapped_walk(self):
        """`enumerate(x.nodes)` is a Call, not a raw attribute iteration — the
        write-back primitive iter_item_slots uses exactly this form."""
        wrapped = (
            "def walk(construct):\n"
            "    for i, item in enumerate(construct.nodes):\n"
            "        handle(i, item)\n"
        )
        assert self._count(wrapped) == 0

    def test_detector_is_attribute_name_agnostic(self):
        """`self.nodes`, `sub.nodes`, `construct.nodes` and `x.nodes[:i]` are all
        caught — a future arm-blind walk cannot dodge the guard by renaming the
        receiver."""
        variants = (
            "def a(self):\n    [n for n in self.nodes]\n"
            "def b(sub):\n    {n for n in sub.nodes}\n"
            "def c(x):\n    [n for n in x.nodes[:2]]\n"
        )
        assert self._count(variants) == 3
