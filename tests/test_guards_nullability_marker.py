"""Structural guard: the null marker is emitted from ONE source of truth.

neograph-g21jc (GH issue #7). A field's nullability is knowable from two
independent places in a Pydantic model -- the annotation (``X | None``) and
``FieldInfo.is_required()`` -- and ``describe_type`` consulted both, then
combined them by string-appending, so every Optional field in every rendered
schema shipped ``T or null or null`` into the model-facing prompt.

The fix made the annotation the single authority: ``_render_model_body`` only
appends its own ``or null`` when ``_admits_none`` says the annotation did NOT
already contribute one. This guard is what stops the two sources from drifting
apart again -- it fails if the ``is_required()`` branch is ever re-written
without consulting ``_admits_none``.

AST-based, no regex, so there is no regex-slip case to meta-test -- a call is a
call whatever the surrounding text looks like. The "would-be-missed" shape that
IS meta-tested is the nested-``if`` spelling, where the annotation check lives
in the branch body rather than in its test.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"
RENDERER = SRC_DIR / "describe_type.py"
# neograph-bek2f: the lint check asks the SAME question the renderer asks -- "has
# a default AND the type rejects null" -- so it is the second place the two
# sources could drift apart, and the gating test below scans it too. It only
# IMPORTS _admits_none, so the definition test stays renderer-only on purpose.
NULLABLE_DEFAULT_LINT = SRC_DIR / "_lint_nullable_defaults.py"
GATED_FILES = (RENDERER, NULLABLE_DEFAULT_LINT)

# The optionality signal, and the annotation-shape authority that must gate it.
OPTIONALITY_SIGNAL = "is_required"
ANNOTATION_AUTHORITY = "_admits_none"


def _called_names(node: ast.AST) -> set[str]:
    """Every simple function name CALLED anywhere under *node*."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _ungated_branches(source: str) -> list[int]:
    """Line numbers of ``if`` branches that consult the optionality signal
    without ever consulting the annotation-shape authority.

    The authority may appear in the branch's TEST (the shipped spelling) or
    anywhere inside the branch BODY (the nested-``if`` spelling) -- either way
    the annotation still has the final say, so both are accepted.
    """
    offenders: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        if OPTIONALITY_SIGNAL not in _called_names(node.test):
            continue
        if ANNOTATION_AUTHORITY in _called_names(node):
            continue
        offenders.append(node.lineno)
    return offenders


class TestNullMarkerSingleSource:
    """``describe_type`` must never re-derive nullability from ``is_required()``
    alone -- the annotation is the authority (neograph-g21jc / GH issue #7)."""

    def test_annotation_authority_is_defined_in_the_renderer(self):
        """Deleting ``_admits_none`` must fail loudly here, not silently
        re-open the doubled-null defect through the guard below going vacuous."""
        tree = ast.parse(RENDERER.read_text())
        defined = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        }
        assert ANNOTATION_AUTHORITY in defined, (
            f"{ANNOTATION_AUTHORITY}() is gone from describe_type.py -- it is the "
            "single authority on whether an annotation already rendered a null "
            "member. Without it, the is_required() branch doubles the marker."
        )

    @pytest.mark.parametrize("path", GATED_FILES, ids=lambda p: p.name)
    def test_optionality_signal_is_gated_by_the_annotation_authority(self, path):
        """No shipped consumer has an ungated ``is_required()`` branch.

        Both files that decide "has a default AND rejects null" are scanned: the
        renderer, which SAYS `` or null`` to the model, and the lint check, which
        REPORTS the same condition. They must answer it the same way or the
        report and the prompt describe different graphs.
        """
        offenders = _ungated_branches(path.read_text())
        assert offenders == [], (
            f"{path.name} decides an optionality marker from "
            f"{OPTIONALITY_SIGNAL}() without consulting {ANNOTATION_AUTHORITY}() "
            f"at line(s) {offenders}. An `X | None` annotation already renders "
            "its own `null`; adding a second one ships `T or null or null` into "
            "every structured-output prompt (neograph-g21jc / GH issue #7)."
        )

    def test_the_gate_is_not_vacuous_on_any_scanned_file(self):
        """Each scanned file must actually CONTAIN a gated ``is_required()`` branch.

        ``_ungated_branches`` walks ``ast.If`` only, so a predicate spelled as a
        comprehension filter or a bare boolean expression is invisible to it and
        the guard above passes by finding nothing to inspect. This asserts every
        scanned file presents the shape the scanner can see -- otherwise a file
        joins ``GATED_FILES`` and is silently never guarded.
        """
        for path in GATED_FILES:
            branches = [
                node
                for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.If) and OPTIONALITY_SIGNAL in _called_names(node.test)
            ]
            assert branches, (
                f"{path.name} is in GATED_FILES but has no `if ...{OPTIONALITY_SIGNAL}()...` "
                "branch, so the gating test above inspects nothing there. Either the "
                "predicate was rewritten into a form the ast.If scanner cannot see "
                "(a comprehension filter or a bare boolean), or the file no longer "
                "belongs in GATED_FILES."
            )

    # --- meta-tests: prove the guard actually catches regressions ---

    def test_guard_flags_the_pre_fix_shape(self):
        """Negative meta-test: the exact pre-fix code must be flagged."""
        pre_fix = (
            "def f(field_info, type_str):\n"
            "    if not field_info.is_required():\n"
            '        type_str = f"{type_str} or null"\n'
            "    return type_str\n"
        )
        assert _ungated_branches(pre_fix) == [2]

    def test_guard_accepts_the_gated_shape(self):
        """Positive meta-test: the shipped conjunction must pass."""
        fixed = (
            "def f(field_info, type_str):\n"
            "    if not field_info.is_required() and not _admits_none(field_info.annotation):\n"
            '        type_str = f"{type_str} or null"\n'
            "    return type_str\n"
        )
        assert _ungated_branches(fixed) == []

    def test_guard_accepts_the_reversed_conjunction(self):
        """Positive meta-test: operand order is not part of the contract."""
        reversed_order = (
            "def f(field_info, type_str):\n"
            "    if not _admits_none(field_info.annotation) and not field_info.is_required():\n"
            '        type_str = f"{type_str} or null"\n'
            "    return type_str\n"
        )
        assert _ungated_branches(reversed_order) == []

    def test_guard_accepts_the_nested_if_spelling(self):
        """'Would-be-missed' meta-test: a test-only check would flag this
        equally-correct spelling, where the annotation has the final say from
        inside the branch body. The guard must accept it."""
        nested = (
            "def f(field_info, type_str):\n"
            "    if not field_info.is_required():\n"
            "        if not _admits_none(field_info.annotation):\n"
            '            type_str = f"{type_str} or null"\n'
            "    return type_str\n"
        )
        assert _ungated_branches(nested) == []

    def test_guard_flags_a_sibling_call_that_is_not_in_the_branch(self):
        """Negative meta-test: calling the authority elsewhere in the function
        must NOT launder an ungated branch -- the scan is per-``if``, not
        per-file, so a stale call at the top cannot vouch for the branch."""
        laundered = (
            "def f(field_info, type_str):\n"
            "    _ = _admits_none(field_info.annotation)\n"
            "    if not field_info.is_required():\n"
            '        type_str = f"{type_str} or null"\n'
            "    return type_str\n"
        )
        assert _ungated_branches(laundered) == [3]
