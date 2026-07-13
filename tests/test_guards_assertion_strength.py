"""PAT-04/T4b (neograph-m5k19): standing ratcheting guard against lone-non-null-only
test assertions regrowing.

A test_* function whose SOLE ``ast.Assert`` is ``assert <Name/Subscript> is not None``
detects almost no real regression — the T4a sweep (neograph-gwncf) strengthened 6 such
sites. This guard flags any new occurrence that is not on the shrink-only allowlist
(seeded from T4a's benign/crash-only classifications). Pure-AST (no ``re``) so it stays
exempt from test_guards_meta.py's named-regex discipline.

The detector counts ``ast.Assert`` nodes (NOT ``raise AssertionError``). A3 in the
allowlist (test_guards_llm_runtime.test_input_shape_...) is a structural-precondition
guard whose real invariant is a ``raise AssertionError`` AST walk — its sole
``ast.Assert`` is the allowlisted one, so the ``raise`` does not turn it into a
multi-assert function.

Allowlist keying is ``(file, test_name)`` — robust to line-number shifts. The allowlist
is ratcheting: a removal (dead entry when a test is deleted or gains a second assert)
is handled by shrink-only cleanup; growth is blocked (fix the weak assert in-PR).
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent


def _is_not_none_compare(node: ast.expr) -> bool:
    """True iff node is exactly ``<x> is not None`` (single IsNot-None, no bool op)."""
    return (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.IsNot)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Constant)
        and node.comparators[0].value is None
    )


def _find_lone_non_null_tests(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, funcname)`` for each test_* function whose SOLE ast.Assert is
    ``<Name/Subscript> is not None`` (with or without a message)."""
    out: list[tuple[int, str]] = []
    tree = ast.parse(source)
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_")):
            continue
        # Asserts directly in this function (ast.walk includes nested defs, but a
        # lone-non-null helper inside a test is still a weak test — count them all;
        # the allowlist exempts the genuine cases).
        asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
        if len(asserts) == 1 and _is_not_none_compare(asserts[0].test):
            out.append((asserts[0].lineno, fn.name))
    return out


# Ratcheting allowlist (shrink-only). Seeded from T4a (neograph-gwncf) benign/crash-only
# classifications. Keyed by (posix path under tests/, test_name) for line-shift robustness.
ALLOWLIST: dict[tuple[str, str], str] = {
    ("hypothesis/test_topologies.py", "test_loop_inside_sub_terminates_and_surfaces"): (
        "A1: sole assert WITH diagnostic msg; the sub-construct surfacing its output "
        "field IS the property; absence IS the regression."
    ),
    ("hypothesis/test_topologies.py", "test_loop_max_iterations_zero"): (
        "A2: sole assert WITH diagnostic msg; a 0-iter loop still emitting its body "
        "field IS the property; absence IS the regression."
    ),
    ("test_guards_llm_runtime.py", "test_input_shape_does_not_call_state_keys_for_fan_out_detection"): (
        "A3: structural-precondition guard (sole ast.Assert) before a raise-AssertionError "
        "AST walk that is the function's real invariant."
    ),
}


class TestAssertionStrengthRatchet:
    """No test_* function outside the allowlist has a lone-non-null-only assertion."""

    def test_no_unallowlisted_lone_non_null_only_asserts(self) -> None:
        offenders: list[str] = []
        for path in sorted(SRC_DIR.rglob("test_*.py")):
            rel = path.relative_to(SRC_DIR).as_posix()
            for _lineno, name in _find_lone_non_null_tests(path.read_text()):
                if (rel, name) in ALLOWLIST:
                    continue
                offenders.append(f"{rel}::{name}")
        assert not offenders, (
            f"\n{len(offenders)} test function(s) whose SOLE assertion is "
            "'assert <x> is not None' (a weak shape that catches almost no regression):\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nStrengthen each to assert a real property, or — if it is a genuine "
            "existence/surfacing/precondition guard with a diagnostic message — add it "
            "to ALLOWLIST in tests/test_guards_assertion_strength.py with a one-line "
            "reason (ratcheting: the allowlist may only shrink)."
        )

    def test_allowlist_entries_still_match_live_tests(self) -> None:
        """Anti-dead-entry: every allowlisted (file, test) is still a live lone-non-null
        test — if a test is removed or gains a second assert, the entry is dead and must
        be removed (shrink-only)."""
        for (rel, name), reason in ALLOWLIST.items():
            path = SRC_DIR / rel
            assert path.exists(), f"allowlist entry {rel}::{name} points at a missing file"
            hits = [n for _ln, n in _find_lone_non_null_tests(path.read_text()) if n == name]
            assert hits, (
                f"allowlist entry {rel}::{name} is DEAD: the test no longer has a "
                "lone-non-null-only assertion (it was removed or strengthened). Remove "
                f"the entry. Reason was: {reason}"
            )

    # ── anti-vacuity meta-tests: prove the detector actually catches the shape ──

    def test_meta_detector_flags_lone_non_null_only(self) -> None:
        source = "def test_weak():\n    assert result is not None\n"
        assert _find_lone_non_null_tests(source) == [(2, "test_weak")]

    def test_meta_detector_flags_subscript_and_with_message(self) -> None:
        source = (
            "def test_a():\n    assert out['x'] is not None\n"
            "def test_b():\n    assert v is not None, 'why'\n"
        )
        names = {n for _ln, n in _find_lone_non_null_tests(source)}
        assert names == {"test_a", "test_b"}

    def test_meta_detector_passes_real_assertions(self) -> None:
        source = (
            "def test_real():\n    assert result == 5\n"
            "def test_multi():\n    assert x is not None\n    assert x == 1\n"
            "def test_and():\n    assert x is not None and x.done\n"
        )
        assert _find_lone_non_null_tests(source) == []
