"""GUARD: pyagentspec's own INFERENCE conventions are never re-derived in ``src/``
(neograph-qtfof.11).

The disease this pins, in one line: **a naming/shape convention the Agent Spec SDK
computes for itself, hand-rolled on the neograph side instead of read back off the
lowered component.**

The concrete instance is ``MapNode``'s reduction naming. ``MapNode.
_get_inferred_outputs`` retitles every one of its sub-Flow's outputs
``collected_{title}`` and -- under the default APPEND reducer that
``_get_default_reducers`` assigns -- wraps it in an array. neograph never has to
know that: ``ComponentWithIO.model_post_init`` materialises ``.outputs`` at
construction, so the lowered MapNode can simply be ASKED. Writing
``f"collected_{title}"`` anywhere in ``src/neograph`` would fork that rule into a
second copy which a reducer change (or a ``ParallelMapNode`` with different
defaults) silently desynchronises -- and the failure mode is not a crash but an
exported artifact naming a Property the node does not have.

**Why AST over regex.** A regex for ``f"collected_`` misses
``"collected_" + title`` and ``"collected_%s" % title``; this guard walks string
CONSTANTS, so every spelling that puts the prefix in the source text is caught
regardless of how it is later assembled. Prose is untouched for free -- a comment
is not an AST node, and docstrings are excluded explicitly. The meta-tests below
exercise all three: the f-string form, the concatenation form a regex would miss,
and the prose form that must NOT fire.

SCOPE, stated honestly: this catches the prefix appearing as SOURCE TEXT. A fully
dynamic construction (``"".join(["collected", "_", t])``) is out of reach of any
static scan and is not claimed.

Run::

    uv run pytest tests/test_guards_agent_spec_sdk_conventions.py
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

#: Substrings that identify an SDK-inferred naming convention. ``iterated_`` is
#: deliberately ABSENT: ``_agent_spec.py``'s ``dest_input = f"iterated_{core}"`` is
#: the one sanctioned, shipped, execution-tested site (neograph-qtfof.7), where the
#: prefix decision rides on the ``input_targets`` table rather than on a lowered
#: node -- see neograph-qtfof.11's codebase scan, row 1 (ALLOWLIST). Adding it here
#: would demand a refactor of a working path, not prevent a defect.
BANNED_CONVENTION_FRAGMENTS = ("collected_",)

#: Empty, and may only stay empty: a new re-derivation gets the lowered node read,
#: not a row here.
ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """``id()`` of every Constant that is a module/class/function DOCSTRING.

    Prose describing the convention (this module's own subject matter) must not be
    mistaken for code re-deriving it.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    found.add(id(body[0].value))
    return found


def find_convention_literals(source: str) -> list[str]:
    """Every string CONSTANT in ``source`` (docstrings excluded) that spells out an
    SDK-inferred naming convention. The guard's detector, exposed so the meta-tests
    below grade the same function the sweep uses."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in docstrings:
            continue
        for fragment in BANNED_CONVENTION_FRAGMENTS:
            if fragment in node.value:
                hits.append(node.value)
    return hits


class TestNoSdkInferenceConventionIsReDerived:
    """No ``src/neograph`` module spells out a convention pyagentspec infers."""

    def test_no_module_hand_rolls_an_inferred_property_name(self) -> None:
        offenders: dict[str, list[str]] = {}
        for path in sorted(SRC_DIR.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            hits = [
                literal
                for literal in find_convention_literals(path.read_text())
                if (path.relative_to(SRC_DIR).as_posix(), literal) not in ALLOWLIST
            ]
            if hits:
                offenders[path.relative_to(SRC_DIR).as_posix()] = hits

        assert not offenders, (
            "module(s) hand-rolling a pyagentspec-INFERRED naming convention: "
            f"{offenders}. The SDK computes these itself (MapNode._get_inferred_outputs "
            "+ _get_default_reducers) and materialises them onto the lowered component -- "
            "read `map_node.outputs` instead of re-spelling the rule (neograph-qtfof.11)."
        )

    def test_the_scan_is_not_vacuous(self) -> None:
        """A scan that sees no files would pass silently forever."""
        scanned = [p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts]
        assert len(scanned) > 60, f"only {len(scanned)} source files scanned -- wrong root?"


class TestTheDetectorItself:
    """Positive, negative, and the regex-would-miss case."""

    def test_detects_the_f_string_form(self) -> None:
        assert find_convention_literals('t = "x"\nname = f"collected_{t}"\n') == ["collected_"]

    def test_detects_the_concatenation_form_a_regex_would_miss(self) -> None:
        """``"collected_" + title`` never matches an ``f"collected_`` regex.

        This is the guard's reason for being AST-based rather than a grep: the
        prefix is in the source either way, and only a constant-level walk sees
        both spellings.
        """
        assert find_convention_literals('title = "ok"\nname = "collected_" + title\n') == ["collected_"]

    def test_does_not_fire_on_prose(self) -> None:
        """A comment or docstring EXPLAINING the convention is not re-deriving it --
        including this very module, which necessarily names it repeatedly."""
        source = '"""A MapNode retitles its outputs collected_{title}."""\n# collected_ok is inferred\nx = 1\n'
        assert find_convention_literals(source) == []

    def test_does_not_fire_on_an_unrelated_literal(self) -> None:
        assert find_convention_literals('name = f"{node}__each_end"\n') == []
