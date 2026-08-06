"""Structural guard: the import-side "is there an edge X -> Y?" predicate has
exactly one implementation (neograph-lp92e).

Disease: the Agent-Spec importer's *never trust a marker -- confirm the
structure* discipline asks one question over and over -- is there a
``ControlFlowEdge`` from X to Y, optionally on branch L? It was spelled out by
hand at five sites (four in ``loader.py``, one that had since moved into
``_agent_spec_swarm_import.py``), so the copies could drift independently. A
drifted copy does not raise: it silently declines to recognize a composite, and
the nodes import as bare primitives -- a SILENT DOWNGRADE.

The fix names it once as ``_agent_spec_graph._has_control_edge``. This guard
AST-scans ``src/neograph`` for any raw iteration over
``X.control_flow_connections`` and allows it only in that module. A new
recognizer that re-inlines the predicate fails here until it calls the helper.

Scope is ``src/`` deliberately. The test side has its own, much larger
graph-walk duplication (~20-25 sites) tracked as neograph-dgbqv.9; folding it
in here would mean shipping this guard pre-loaded with a 20-entry allowlist,
which is the shape of a ratchet that never ratchets.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# The ONE module allowed to iterate the edge list: the helper's own definition.
_ALLOWED_MODULES = frozenset({"_agent_spec_graph.py"})

_EDGE_ATTR = "control_flow_connections"


def _iter_iteration_exprs(tree: ast.AST):
    """Yield the iterated expression of every for-loop and comprehension."""
    for n in ast.walk(tree):
        if isinstance(n, ast.For):
            yield n.iter
        elif isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in n.generators:
                yield gen.iter


def _is_edge_list(expr: ast.AST) -> bool:
    """True for ``X.control_flow_connections`` (and a subscript of it).

    A keyword ARGUMENT of the same name -- ``Flow(control_flow_connections=...)``
    on the export side -- is an ``ast.keyword``, never an ``ast.Attribute``, so
    the exporter's construction sites are structurally out of scope.
    """
    e = expr
    if isinstance(e, ast.Subscript):
        e = e.value
    return isinstance(e, ast.Attribute) and e.attr == _EDGE_ATTR


def _collect_raw_edge_walks() -> list[str]:
    offenders: list[str] = []
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        if py_file.name in _ALLOWED_MODULES:
            continue
        src = py_file.read_text(encoding="utf-8")
        if _EDGE_ATTR not in src:
            continue
        lines = src.splitlines()
        for it in _iter_iteration_exprs(ast.parse(src)):
            if _is_edge_list(it):
                rel = py_file.relative_to(SRC_DIR.parent.parent)
                offenders.append(f"{rel}:{it.lineno}: {lines[it.lineno - 1].strip()}")
    return offenders


class TestControlEdgePredicateHasOneImplementation:
    """Only ``_agent_spec_graph`` walks the control-flow edge list."""

    def test_no_module_re_inlines_the_edge_existence_predicate(self):
        offenders = _collect_raw_edge_walks()
        assert not offenders, (
            "the 'is there a ControlFlowEdge X -> Y (optionally on branch L)?' "
            "predicate belongs to _agent_spec_graph._has_control_edge -- call it "
            "instead of re-inlining the walk. A drifted hand-rolled copy does not "
            "raise: it silently declines to recognize a composite and the nodes "
            "import as bare primitives (neograph-lp92e).\n  " + "\n  ".join(offenders)
        )

    def test_the_helper_itself_is_what_the_allowlist_names(self):
        """The allowlist must point at a module that actually holds the walk --
        otherwise the guard passes because the thing it protects is gone."""
        helper = SRC_DIR / "_agent_spec_graph.py"
        assert helper.exists(), "the allowlisted module must exist"
        src = helper.read_text(encoding="utf-8")
        walks = [it for it in _iter_iteration_exprs(ast.parse(src)) if _is_edge_list(it)]
        assert walks, (
            "_agent_spec_graph.py is allowlisted as the single home of the edge "
            "walk but no longer contains one -- re-point or remove the entry"
        )

    def test_the_guard_detects_a_re_inlined_predicate(self):
        """Mutation check: the scan must actually fire on the disease shape."""
        tree = ast.parse(
            "ok = any(e.from_node.name == a for e in flow.control_flow_connections)\n"
        )
        walks = [it for it in _iter_iteration_exprs(tree) if _is_edge_list(it)]
        assert len(walks) == 1, (
            "a hand-rolled any(...) over flow.control_flow_connections must be "
            "detected -- if this fails, the guard above is vacuous"
        )
