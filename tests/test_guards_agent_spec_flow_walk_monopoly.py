"""Structural guard: the test-side control-graph walk has ONE home
(neograph-dgbqv.9).

Disease: six test modules hand-rolled the same questions of an exported Agent
Spec ``Flow`` -- the ``{(from, to)}`` edge-pair set appeared VERBATIM in five
files, entry-node resolution was duplicated even within one module, and the
subflow descent existed in three places reading only the SINGULAR ``.subflow``.
Divergence between copies does not raise: a copy that stops seeing a node or an
arm simply asserts less, so the suite keeps passing while covering less.

The fix names them once in ``tests/agent_spec_flow_walk.py``, where
``branch_adjacency`` is the SINGLE edge read and every other edge question is a
pure projection of it. This guard AST-scans ``tests/`` for any remaining raw read
of ``control_flow_connections`` and allows it only in that module or in a NAMED
allowlist entry below.

SCOPE: control flow only. ``data_flow_connections`` is deliberately NOT governed
here -- ``tests/test_guards_agent_spec_data_flow_reads.py`` (neograph-p0dn8)
already governs every one of those reads under a DIFFERENT rule (the read must be
the left operand of a ``BoolOp(Or)``) with a zero-entry allowlist. Two guards with
two rules over one attribute is precisely the duplicated-source-of-truth shape
AGENTS.md's Portal lesson bans. The remaining data-flow duplication is
neograph-dgbqv.11.

WHY THE RULE KEYS ON THE ATTRIBUTE READ rather than the iteration expression: an
iteration-expression scan (the shape ``test_guards_branch_arm_walks.py`` uses) is
vacuous against ``edges = flow.control_flow_connections`` followed by
``for e in edges``. No CURRENT control-flow site uses that binding form, so this
is future-proofing rather than a response to an existing bypass -- recorded
plainly so a later reader does not go looking for the site that forced it.
"""

from __future__ import annotations

import ast
import pathlib

TESTS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = TESTS_DIR.parent

_ATTR = "control_flow_connections"

#: The module that IS the single source; its OWN suite, which must spell the raw
#: form in order to prove each helper EQUALS the expression it replaced (an
#: equivalence proof that named no baseline would prove nothing); and the two
#: guards, which must name the attribute in order to scan for it.
_EXEMPT_MODULES = frozenset(
    {
        "agent_spec_flow_walk.py",
        "test_agent_spec_flow_walk.py",
        "test_guards_agent_spec_flow_walk_monopoly.py",
        "test_guards_agent_spec_control_edge_predicate.py",
    }
)

#: Sites that legitimately read the raw edge list, each with a reason.
#:
#: CONTENT-KEYED (file, stripped source line) -> reason, never line-numbered --
#: the house pattern ``test_guards_branch_arm_walks.py`` established for exactly
#: this reason: a later fix that shifts line numbers must not spuriously trip the
#: guard, and a stale line number must not silently exempt a NEW read that drifts
#: onto it. (Learned the hard way: neograph-498gr's edits moved these very lines.)
#:
#: This list was published in the ticket BEFORE the guard was written, so it
#: cannot be grown to fit whatever the tree happened to contain. It may only
#: SHRINK (AGENTS.md ratchet discipline).
_ALLOWLIST: dict[tuple[str, str], str] = {
    # -- structural-forever: these need the EDGE OBJECTS, not a derived answer.
    # The walk module deliberately does not hand out raw edges (that would widen
    # its contract from "answers graph questions" to "hands out edges" and partly
    # defeat this guard). Type-filtering on the pyagentspec class is a different
    # question from graph shape.
    (
        "test_agent_spec_export.py",
        "control_edges = [e for e in flow.control_flow_connections if isinstance(e, ControlFlowEdge)]",
    ): "isinstance(ControlFlowEdge) type filter, not a graph query (3 sites)",
    (
        "test_agent_spec_export.py",
        "for e in flow.control_flow_connections",
    ): "needs the edge OBJECT -- multiplicity (len(...) == 1) and from_branch off a filtered list (4 sites)",
    (
        "test_agent_spec_reachability.py",
        "for e in flow.control_flow_connections",
    ): "needs the edge OBJECT to read .from_branch off the single loop exit edge",
    (
        "test_agent_spec_each_operator.py",
        "for e in rebuilt.control_flow_connections",
    ): "asserts on a RE-IMPORTED flow (rebuilt) -- a different object under test",
    # -- Start/End boundary edge readers. Dropped from the walk module's API on
    # purpose: they return edge OBJECTS. Two sites only; if a third appears,
    # that is the signal to add a derived projection instead of a third entry.
    (
        "test_agent_spec_each_oracle.py",
        'start_edges = [e for e in subflow.control_flow_connections if type(e.from_node).__name__ == "StartNode"]',
    ): "StartNode-sourced edge objects inside a MapNode subflow",
    (
        "test_agent_spec_each_oracle.py",
        'end_edges = [e for e in subflow.control_flow_connections if type(e.to_node).__name__ == "EndNode"]',
    ): "EndNode-targeted edge objects inside a MapNode subflow",
}


def _attribute_reads(tree: ast.AST) -> list[ast.Attribute]:
    """Every LOAD of ``X.control_flow_connections``.

    A keyword ARGUMENT of the same name -- ``Flow(control_flow_connections=[...])``
    when a test CONSTRUCTS a fixture flow -- is an ``ast.keyword``, never an
    ``ast.Attribute``, so fixture construction is structurally out of scope.
    """
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == _ATTR and isinstance(n.ctx, ast.Load)
    ]


def _collect_raw_reads() -> list[tuple[str, int, str]]:
    """(file, lineno, stripped source line) for every raw read outside the exempt modules."""
    found: list[tuple[str, int, str]] = []
    for py_file in sorted(TESTS_DIR.rglob("*.py")):
        if py_file.name in _EXEMPT_MODULES:
            continue
        src = py_file.read_text(encoding="utf-8")
        if _ATTR not in src:
            continue
        lines = src.splitlines()
        for node in _attribute_reads(ast.parse(src)):
            found.append((py_file.name, node.lineno, lines[node.lineno - 1].strip()))
    return found


class TestFlowWalkIsTheSingleTestSideReader:
    """Only ``tests/agent_spec_flow_walk.py`` walks the control-flow edge list."""

    def test_no_unallowlisted_raw_control_flow_read_exists(self):
        offenders = [
            f"{name}:{line}: {text}"
            for name, line, text in _collect_raw_reads()
            if (name, text) not in _ALLOWLIST
        ]
        assert not offenders, (
            "ask tests/agent_spec_flow_walk.py instead of walking "
            "control_flow_connections by hand -- branch_adjacency is the single "
            "edge read and edge_pairs/successors_of/arm_targets/entered_nodes/walk "
            "are its projections. A hand-rolled copy that drifts does not raise, it "
            "silently asserts less (neograph-dgbqv.9).\n  " + "\n  ".join(offenders)
        )

    def test_every_allowlist_entry_is_still_a_real_read(self):
        """The ratchet may only SHRINK, so a stale entry must be deleted rather
        than left as silent headroom for a future bypass."""
        real = {(name, text) for name, _line, text in _collect_raw_reads()}
        stale = sorted(set(_ALLOWLIST) - real)
        assert not stale, (
            "these allowlist entries no longer point at a raw read -- delete them "
            f"(the ratchet may only shrink): {stale}"
        )

    def test_the_allowlist_records_a_reason_for_every_entry(self):
        blank = [k for k, v in _ALLOWLIST.items() if not v.strip()]
        assert not blank, f"every allowlist entry needs a written reason: {blank}"

    def test_only_the_walk_module_and_its_own_suite_are_exempt_non_guards(self):
        """Guards may name the attribute; beyond them only the single-source
        module and its own equivalence suite get a blanket exemption. Anything
        else belongs in the per-site allowlist, where it needs a written reason."""
        non_guard = {m for m in _EXEMPT_MODULES if not m.startswith("test_guards_")}
        assert non_guard == {"agent_spec_flow_walk.py", "test_agent_spec_flow_walk.py"}, (
            "only the single-source module and its own suite may be blanket-exempt; "
            f"everything else belongs in the per-site allowlist. Got: {sorted(non_guard)}"
        )

    def test_the_guard_detects_a_hand_rolled_walk(self):
        """Mutation check: the scan must fire on the disease shape, including the
        bound-to-a-local form an iteration-expression scan would miss."""
        inline = ast.parse("pairs = {(e.from_node.name, e.to_node.name) for e in flow.control_flow_connections}\n")
        assert len(_attribute_reads(inline)) == 1, "the inline comprehension form must be detected"

        bound = ast.parse("edges = flow.control_flow_connections\nfor e in edges:\n    pass\n")
        assert len(_attribute_reads(bound)) == 1, (
            "the bind-then-iterate form must ALSO be detected -- this is why the rule "
            "keys on the attribute read rather than the iteration expression"
        )

        construction = ast.parse("f = Flow(name='x', control_flow_connections=[])\n")
        assert not _attribute_reads(construction), (
            "constructing a fixture Flow is not a read and must never be flagged"
        )
