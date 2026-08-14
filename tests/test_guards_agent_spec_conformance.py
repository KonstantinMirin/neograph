"""Structural guards for the Agent Spec conformance classifier (neograph-ftnxl.1).

Pins three invariants the design depends on:

1. **Predicate registry completeness, BOTH directions.** Every predicate id
   ``ConformanceFinding(...)`` can emit has an entry in
   ``CONFORMANCE_PREDICATE_META``, and every registry entry is actually
   emitted somewhere -- a closed bead with a still-live predicate, or an
   emission site with no registry entry, both fail loud. Mirrors
   ``scripts/gen_api_manifest.py``'s bidirectional ``code_kinds != set(
   LINT_KIND_META)`` check (architect review consistency-1's correction: a
   one-way containment misses the shrink-direction drift this ticket's own
   ratchet depends on).
2. **Single-walker invariant.** ``_structural_findings``/``_walk_structural``
   is the only IR-level walker; ``_flow_findings`` and its helpers are the
   only Flow-level walker. No third function may independently walk
   ``iter_with_arms``/``Construct`` recursion or ``flow.nodes``/``subflow``
   for a conformance predicate -- that would re-duplicate the exact
   "two entry points into one predicate set" risk the design named.
3. **Anti-duplication (AST-based, not regex).** ``_agent_spec_conformance.py``
   and ``_agent_spec_conformance_report.py`` contain no re-typed copy of the
   exporter's own NOT_EXPORTABLE raise-list field names (``raw_fn``,
   ``skip_when``, ``handoff_channel``, ``gate_tools_when``) as string
   literals -- NOT_EXPORTABLE must be decided by attempting ``to_agent_spec()``
   and catching ``ConfigurationError``, never by re-deriving what it rejects.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"
CONFORMANCE_MODULE = SRC_DIR / "_agent_spec_conformance.py"
REPORT_MODULE = SRC_DIR / "_agent_spec_conformance_report.py"

# Field names from the exporter's real raise-list (_agent_spec_node_lowering.py's
# _reject_unrepresentable_fields) that must never be re-typed as string literals
# in the classifier -- the classifier decides NOT_EXPORTABLE by attempt-and-catch.
_BANNED_RAISE_LIST_LITERALS = frozenset({"raw_fn", "skip_when", "handoff_channel", "gate_tools_when"})


def _emitted_predicate_ids(tree: ast.Module) -> set[str]:
    """Every string literal passed as the first positional arg to a
    ``ConformanceFinding(...)`` call."""
    ids: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ConformanceFinding"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            ids.add(node.args[0].value)
    return ids


def _registry_keys(tree: ast.Module) -> set[str]:
    """The string keys of the ``CONFORMANCE_PREDICATE_META`` dict literal.

    ``CONFORMANCE_PREDICATE_META: dict[str, ConformancePredicateMeta] = {...}``
    is an ``ast.AnnAssign`` (annotated assignment), not a plain ``ast.Assign``.
    """
    for node in ast.walk(tree):
        target = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0], node.value
        if target is not None and target.id == "CONFORMANCE_PREDICATE_META" and isinstance(value, ast.Dict):
            return {k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    raise AssertionError("CONFORMANCE_PREDICATE_META dict literal not found via AST -- did its shape change?")


def _walker_function_names(tree: ast.Module) -> set[str]:
    """Every top-level function definition in the conformance module."""
    return {node.name for node in ast.iter_child_nodes(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


class TestPredicateRegistryCompleteness:
    def test_every_emitted_predicate_has_a_registry_entry(self):
        tree = ast.parse(CONFORMANCE_MODULE.read_text())
        emitted = _emitted_predicate_ids(tree)
        registered = _registry_keys(tree)
        report_tree = ast.parse(REPORT_MODULE.read_text())
        emitted |= _emitted_predicate_ids(report_tree)
        missing = emitted - registered
        assert not missing, f"predicate id(s) emitted but not in CONFORMANCE_PREDICATE_META: {sorted(missing)}"

    def test_every_registry_entry_is_actually_emitted(self):
        """The reverse direction (consistency-1's correction): a registry row
        for a predicate nothing emits is either dead or was renamed without
        updating the registry -- both are drift this ratchet exists to catch."""
        tree = ast.parse(CONFORMANCE_MODULE.read_text())
        emitted = _emitted_predicate_ids(tree)
        registered = _registry_keys(tree)
        report_tree = ast.parse(REPORT_MODULE.read_text())
        emitted |= _emitted_predicate_ids(report_tree)
        stale = registered - emitted
        assert not stale, f"CONFORMANCE_PREDICATE_META entry with no emission site: {sorted(stale)}"

    def test_every_registry_entry_has_a_non_empty_bead(self):
        from neograph._agent_spec_conformance import CONFORMANCE_PREDICATE_META

        empty = [pid for pid, meta in CONFORMANCE_PREDICATE_META.items() if not meta.bead.strip()]
        assert not empty, f"predicate(s) with an empty bead field: {empty}"


class TestSingleWalkerInvariant:
    """Exactly one IR-level walker function family, one Flow-level walker
    function family -- no second, independently-written walk of the same
    shape hiding elsewhere in the module."""

    def test_only_one_iter_with_arms_call_site_in_the_conformance_module(self):
        tree = ast.parse(CONFORMANCE_MODULE.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "iter_with_arms"
        ]
        assert len(calls) == 1, (
            f"expected exactly one iter_with_arms(...) call (inside _walk_structural, the sole IR-level "
            f"walker) -- found {len(calls)}. A second call site is a re-derived walker."
        )

    def test_flow_nodes_attribute_walked_by_exactly_two_functions(self):
        """`flow.nodes` (or `n.subflow`) is read by exactly the two designated
        Flow-level walker helpers -- `_check_outermost_end_node` and
        `_walk_flow_for_llm_configs` -- never a third."""
        tree = ast.parse(CONFORMANCE_MODULE.read_text())
        readers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Attribute) and inner.attr in ("nodes", "subflow"):
                        readers.add(node.name)
        allowed = {"_check_outermost_end_node", "_walk_flow_for_llm_configs"}
        unexpected = readers - allowed
        assert not unexpected, (
            f"function(s) reading flow.nodes/.subflow outside the two designated Flow-level walkers: "
            f"{sorted(unexpected)} -- this is a second, re-derived Flow walk."
        )


class TestNoRaiseListReDerivation:
    """AST-based (not regex) -- explanatory prose in a docstring may legitimately
    MENTION these field names; only a STRING LITERAL used as a dict/attribute
    comparison target would indicate a re-typed copy of the raise-list."""

    def test_no_banned_field_name_used_as_a_comparison_or_lookup_literal(self):
        for module_path in (CONFORMANCE_MODULE, REPORT_MODULE):
            tree = ast.parse(module_path.read_text())
            for node in ast.walk(tree):
                # A banned name compared via == or used as a dict/attr key/string
                # literal OUTSIDE a docstring/comment context (ast.Constant strings
                # in non-docstring position are the only surface -- comments are
                # never in the AST at all).
                if isinstance(node, ast.Compare):
                    literals = [n.value for n in [node.left, *node.comparators] if isinstance(n, ast.Constant)]
                    hit = _BANNED_RAISE_LIST_LITERALS & {v for v in literals if isinstance(v, str)}
                    assert not hit, f"{module_path.name}: raise-list field name used in a comparison: {hit}"
