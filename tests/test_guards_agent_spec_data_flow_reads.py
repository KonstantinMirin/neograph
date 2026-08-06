"""Structural guard: no read of ``data_flow_connections`` may skip the None guard
(neograph-p0dn8).

Root-cause pin. The exporter writes the attribute as ``<edges> or None`` at BOTH
construction sites (``_agent_spec.py`` and ``_agent_spec_modifier_lowering.py``),
so a control-only shape genuinely exports ``data_flow_connections=None`` -- the
serialized fixture ``tests/fixtures/agent_spec_refactor_snapshot.json`` carries a
literal ``"data_flow_connections": null`` at a dozen-plus sites. Four readers
dereferenced the attribute raw, which raises ``TypeError: 'NoneType' object is not
iterable`` rather than failing an assertion, and were protected only by their
fixtures happening to wire at least one data edge.

The sanctioned form -- already the dominant one -- interposes an ``or`` fallback::

    edges = flow.data_flow_connections or []
    data_edges = [e for e in (flow.data_flow_connections or []) if ...]
    dest = {e["destination_input"] for e in (exported.get("data_flow_connections") or [])}

This guard AST-scans ``src/neograph`` AND ``tests`` (the disease lived on the test
side, so a src-only scan would be blind to it -- the mistake
``test_guards_agent_spec_markers.py`` made for the marker literals) for every READ
of the attribute in its three spellings: attribute access, string-subscript on the
serialized dict, and ``.get()`` on the serialized dict. Each must be the left
operand of a ``BoolOp(Or)``. Writes are structurally invisible to the scan --
``data_flow_connections=<expr>`` is an ``ast.keyword``, not an attribute read --
so the two exporter sites are not false positives.
"""

from __future__ import annotations

import ast
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_DIR = _ROOT / "src" / "neograph"
_TESTS_DIR = _ROOT / "tests"

_KEY = "data_flow_connections"

# This module names the key as a plain string constant to express the rule; it
# performs no read of its own, but is exempted explicitly so a future assertion
# helper here cannot trip the guard it implements.
_EXEMPT_FILES = frozenset({"test_guards_agent_spec_data_flow_reads.py"})


def _is_attribute_read(node: ast.AST) -> bool:
    """``<expr>.data_flow_connections`` -- the in-memory spec object spelling.

    ``Load`` context only: an ASSIGNMENT to the attribute is a write, and a write
    is exactly what makes the value None-able. Flagging it would be backwards.
    """
    return isinstance(node, ast.Attribute) and node.attr == _KEY and isinstance(node.ctx, ast.Load)


def _is_subscript_read(node: ast.AST) -> bool:
    """``<expr>["data_flow_connections"]`` -- the serialized-dict spelling."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == _KEY
        and isinstance(node.ctx, ast.Load)
    )


def _is_get_read(node: ast.AST) -> bool:
    """``<expr>.get("data_flow_connections")`` -- the other serialized spelling.

    ``.get()`` still needs the fallback: the key is PRESENT and its value is
    ``null``, so ``.get()`` returns ``None`` just as the subscript does. A
    ``.get(key, [])`` two-arg form supplies its own fallback and is accepted.
    """
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    if node.func.attr != "get" or not node.args:
        return False
    first = node.args[0]
    if not (isinstance(first, ast.Constant) and first.value == _KEY):
        return False
    return len(node.args) < 2  # a supplied default is its own guard


def _is_read(node: ast.AST) -> bool:
    return _is_attribute_read(node) or _is_subscript_read(node) or _is_get_read(node)


def _guarded_reads(tree: ast.AST) -> set[int]:
    """ids of read nodes that are the left operand of an ``or`` expression."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and node.values:
            guarded.add(id(node.values[0]))
    return guarded


def _collect_unguarded() -> list[str]:
    """Every ``data_flow_connections`` read not wrapped in an ``or`` fallback."""
    offenders: list[str] = []
    files = sorted(_SRC_DIR.rglob("*.py")) + sorted(_TESTS_DIR.rglob("*.py"))
    for py_file in files:
        if py_file.name in _EXEMPT_FILES:
            continue
        src = py_file.read_text()
        if _KEY not in src:  # cheap pre-filter, keeps the scan fast
            continue
        lines = src.splitlines()
        tree = ast.parse(src)
        guarded = _guarded_reads(tree)
        for node in ast.walk(tree):
            if _is_read(node) and id(node) not in guarded:
                rel = py_file.relative_to(_ROOT)
                offenders.append(f"{rel}:{node.lineno}: {lines[node.lineno - 1].strip()}")
    return offenders


class TestDataFlowConnectionsReadsAreNoneGuarded:
    """Every read of the None-able ``data_flow_connections`` carries a fallback."""

    def test_no_unguarded_data_flow_connections_read_exists(self):
        offenders = _collect_unguarded()
        assert not offenders, (
            "data_flow_connections is exported as '<edges> or None' (see "
            "_agent_spec.py and _agent_spec_modifier_lowering.py), so a control-only "
            "shape exports None and a raw dereference raises TypeError instead of "
            "failing an assertion. Use the guarded form -- "
            "'(flow.data_flow_connections or [])' or "
            "'(exported.get(\"data_flow_connections\") or [])'.\n"
            "Unguarded reads:\n  " + "\n  ".join(offenders)
        )

    def test_the_guard_detects_an_unguarded_read(self):
        """Mutation check: the scan must actually fail on the disease shape."""
        tree = ast.parse("edges = [e for e in flow.data_flow_connections if e]\n")
        guarded = _guarded_reads(tree)
        reads = [n for n in ast.walk(tree) if _is_read(n)]
        assert len(reads) == 1, "expected exactly one read in the mutation sample"
        assert id(reads[0]) not in guarded, (
            "the raw attribute read must be reported as unguarded -- if this passes, "
            "the guard above is vacuous"
        )

    def test_the_guard_accepts_every_sanctioned_spelling(self):
        """The three guarded spellings must all be recognised as safe."""
        sample = (
            "a = flow.data_flow_connections or []\n"
            'b = exported["data_flow_connections"] or []\n'
            'c = exported.get("data_flow_connections") or []\n'
            'd = exported.get("data_flow_connections", [])\n'
        )
        tree = ast.parse(sample)
        guarded = _guarded_reads(tree)
        unguarded = [n for n in ast.walk(tree) if _is_read(n) and id(n) not in guarded]
        assert not unguarded, (
            "all four sanctioned spellings must be accepted; "
            f"{len(unguarded)} were flagged"
        )
