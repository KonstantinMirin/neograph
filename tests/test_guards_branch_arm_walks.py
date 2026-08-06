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
_ALLOWLIST: Counter = Counter(
    {
        # The arm-descent primitive itself — the single source of truth.
        ("_ir_branch.py", "for item in construct.nodes:"): 1,
        # Peer-field set for fan_out inference: intentionally TOP-LEVEL only
        # (arm-sibling fan-in unsupported; documented in normalize_ir).
        ("_ir_normalize.py", "for item in construct.nodes:"): 1,
        # Graph-build dispatch loop — already arm-aware (dispatches _BranchNode to
        # _add_branch_to_graph).
        ("compiler.py", "for item in construct.nodes:"): 1,
        # Agent Spec export dispatch loop (neograph-s7zt3.17) — mirrors compiler.py's
        # entry directly above: already arm-aware, dispatching _BranchNode to
        # _lower_branch (which recurses into each arm's own items via the injected
        # _lower_construct_item). Pre-fix this walked iter_with_arms, which DROPS the
        # _BranchNode sentinel and silently flattened both arms into an unconditional
        # sequence -- exactly the arm-blindness disease this guard exists to catch,
        # just on the export side instead of the compile side.
        ("_agent_spec.py", "for item in construct.nodes:"): 1,
        # LangGraph compiled-graph node ids — NOT a Construct node list.
        # neograph-3ffdg.8: this walk moved with describe_graph into
        # _compile_diagnostics.py. Re-keyed by (module, content); it walks the
        # COMPILED LangGraph's node names, not the neograph IR, so arm-awareness
        # does not apply -- the exemption is unchanged, only its address.
        ("_compile_diagnostics.py", 'nodes = [n for n in lg_graph.nodes if n not in ("__start__", "__end__")]'): 1,
        # compile_start telemetry — display only, no correctness impact.
        ("compiler.py", "node_names=[n.name for n in construct.nodes],"): 1,
        ("compiler.py", "for n in construct.nodes"): 1,
        # iter_nodes — the leaf-flattening SoT walk, already arm-aware (tdbb).
        ("construct.py", "for item in construct.nodes:"): 1,
        # Spec-level walks — YAML spec, pre-IR; _BranchNode never appears in a spec.
        # neograph-3ffdg.4: these three walk the neograph-Spec SCHEMA objects
        # (Spec.nodes / ConstructSpec.nodes), not the neograph IR, so arm-awareness
        # never applied. They moved with the spec-loader half into _spec_loader.py;
        # re-keyed by (module, content), exemption unchanged.
        ("_spec_loader.py", "for node_spec in spec.nodes:"): 1,
        ("_spec_loader.py", "for ref in spec.pipeline.nodes:"): 1,
        ("_spec_loader.py", "for prev_ref in construct_spec.nodes[:i]:"): 1,
        # compute_node_fingerprints — already arm-aware (tdbb, _fingerprint_item).
        # neograph-3ffdg.14: this walk moved with compute_node_fingerprints into
        # _schema_fingerprint.py. Re-keyed by (module, content); it walks the IR to
        # hash node output types, and the arm-blindness exemption is unchanged.
        ("_schema_fingerprint.py", "for item in construct.nodes:"): 1,
        # compile_state_model — the gold-standard arm-aware walk (handles
        # branch_nodes explicitly right after these partitions).
        ("state.py", "nodes_only = [n for n in construct.nodes if isinstance(n, Node)]"): 1,
        ("state.py", "sub_constructs = [n for n in construct.nodes if isinstance(n, Construct)]"): 1,
        ("state.py", "branch_nodes = [n for n in construct.nodes if isinstance(n, _BranchNode)]"): 1,
        # Portal mesh-member partition (neograph-s7zt3.5) — DELIBERATELY TOP-LEVEL
        # only: a Portal mesh is one contiguous run of sibling members at ONE
        # construct level (D-MESH-LEVEL); a mesh never spans branch arms. Filters
        # construct.nodes DIRECTLY (not nodes_only) so a sub-Construct mesh ENTRY is
        # not excluded and order is preserved for _group_portal_members' entry pick.
        # Routing through an arm-descending iterator would be WRONG (it would admit
        # arm-nested nodes that cannot be mesh members).
        # content re-localized: ruff format rewrapped this comprehension onto one
        # line during neograph-3ffdg.14. Same walk, same exemption.
        ("state.py", "for m in construct.nodes"): 1,
        # Test-scaffold codegen introspection. The two top-level _collect_items /
        # _collect_edges walks were migrated to iter_with_arms in neograph-gfoq; the
        # remaining inner walk descends a sub-construct's own node list one level
        # down (sub-construct arm-descent is owned by the sub-construct itself).
        ("scaffold.py", "sub_nodes = [_node_info(n) for n in item.nodes if isinstance(n, Node)]"): 1,
        # wrap_fan_over_agents — the fan-over-agent auto-wrap pre-pass (neograph-m6d3.6).
        # Deliberately TOP-LEVEL only: it rewrites supported fan-over-agent nodes into
        # isolating sub-constructs, but arm nodes CANNOT be wrapped here (they are added
        # verbatim by _add_arm_nodes). Arm-nested cases are caught fail-loud by the
        # iter_with_arms scan immediately after this walk, so routing this walk through
        # iter_with_arms would be wrong (it must NOT descend into arms).
        ("_fan_agent_wrap.py", "for item in construct.nodes:"): 1,
        # from_agent_spec's Each sub-flow descent (neograph-01i0g): this walks a
        # pyagentspec Flow's OWN `.subflow.nodes` list (the exported MapNode's
        # sub-flow), not a neograph Construct.nodes list -- there is no
        # _BranchNode/arm concept on the Agent Spec side for iter_with_arms to
        # apply to. Count stays 1 across the Each x Oracle fusion work
        # (neograph-s7zt3.10): the fusion recognizer does NOT add a second walk,
        # it calls the same `_subflow_inner_nodes` helper this line lives in.
        # RE-KEYED (neograph-s7zt3.11): `_subflow_inner_nodes` moved to
        # _agent_spec_group_import.py when loader.py was split back under its
        # ceiling. The old loader.py key is REMOVED, not kept alongside -- leaving
        # it would grant a permission to a file that no longer does the thing.
        (
            "_agent_spec_group_import.py",
            'inner_nodes = [n for n in map_node.subflow.nodes if type(n).__name__ not in ("StartNode", "EndNode")]',
        ): 1,
        # from_agent_spec's PORTAL_OPERATOR mesh-exit composite recognition
        # (neograph-s7zt3.2, _reconstruct_swarm_mesh_with_operator_gates): these
        # two walk a pyagentspec Flow's OWN `.nodes` list (to find the wrapping
        # AgentNode and the portal_operator BranchingNode), NOT a neograph
        # Construct.nodes -- there is no _BranchNode/arm concept on the Agent Spec
        # side for iter_with_arms to apply to. Same false-positive category as the
        # map_node.subflow.nodes entry above.
        # RE-KEYED (neograph-jtawq.10): the Swarm-import cluster moved to
        # _agent_spec_swarm_import.py. The old loader.py key is REMOVED, not kept
        # alongside -- leaving it would grant a permission to a file that no
        # longer does the thing.
        ("_agent_spec_swarm_import.py", 'agent_nodes = [n for n in flow.nodes if type(n).__name__ == "AgentNode"]'): 1,
        (
            "_agent_spec_swarm_import.py",
            '(n for n in flow.nodes if (n.metadata or {}).get(_MARK_MODIFIER) == "portal_operator"),',
        ): 1,
        # Same reconstructor's per-member Operator re-attach walk over the
        # reconstructed mesh Construct's node list. DELIBERATELY TOP-LEVEL only
        # (mirrors the state.py "Portal mesh-member partition" entry above): a
        # Portal mesh is one contiguous run of sibling members at ONE construct
        # level (D-MESH-LEVEL) and NEVER spans branch arms, so there is no arm
        # content to descend into. Routing through an arm-descending iterator
        # would be WRONG -- it would flatten sub-construct-member arms whose nodes
        # can never be mesh members receiving a top-level Operator gate. The
        # isinstance(member, Node) guard at the call site further confirms only
        # atomic top-level members are gated (a Construct mesh member cannot carry
        # Operator per _validation_portal.py, so it is never in `gated`).
        ("_agent_spec_swarm_import.py", "for member in base.nodes"): 1,
    }
)


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
                "it or re-localize by content:\n" + "\n".join(f"    {f}: {line}" for (f, line), _ in stale.items())
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
        blind = "def walk(construct):\n    for item in construct.nodes:\n        handle(item)\n"
        assert self._count(blind) == 1

    def test_detector_ignores_primitive_routed_walk(self):
        """A walk routed through the primitive is NOT detected — migration is
        the sanctioned escape from the guard (would-be-missed case)."""
        routed = "def walk(construct):\n    for item in iter_with_arms(construct):\n        handle(item)\n"
        assert self._count(routed) == 0

    def test_detector_ignores_enumerate_wrapped_walk(self):
        """`enumerate(x.nodes)` is a Call, not a raw attribute iteration — the
        write-back primitive iter_item_slots uses exactly this form."""
        wrapped = "def walk(construct):\n    for i, item in enumerate(construct.nodes):\n        handle(i, item)\n"
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
