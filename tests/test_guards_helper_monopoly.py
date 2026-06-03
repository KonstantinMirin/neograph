"""Structural guards: helper-monopoly enforcement for neograph-wpzg.

neograph-wpzg extracted three single-source-of-truth helpers after finding the
same logic hand-rolled across >=2 call sites in the DX-layer codegen/tracer
files. These guards pin the monopolies so a future PR cannot silently
re-introduce an inline variant:

  - forward.py   _attr_chain_after_prefix  — strips the ``out_of_<node>`` proxy
                  prefix (the parse idiom ``full_name[len(prefix):]`` may live
                  ONLY inside the helper).
  - forward.py   _build_condition_spec     — builds a _ConditionSpec from a
                  branch condition (the truthy-fallback ``op_str="truthy"`` spec
                  may live ONLY inside the helper).
  - testing.py   _emit_set_block           — emits an ``EXPECTED_* = {...}`` set
                  literal (the set-open codegen idiom may live ONLY inside it).

Each monopoly is enforced two ways:
  1. the helper is called at least N times (AST call-site count), and
  2. its unique inline idiom appears exactly once in the owner file (the def).

The idiom check normalizes whitespace before counting, so a spacing variant
(``[len( prefix ):]``) cannot slip — proven by the regex-slip meta-test.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# helper name -> (owner file, minimum call sites excluding the def)
MONOPOLIES = {
    "_attr_chain_after_prefix": ("forward.py", 2),
    "_build_condition_spec": ("forward.py", 2),
    "_emit_set_block": ("testing.py", 4),
}

# helper name -> the unique inline idiom that must appear exactly once (the def).
# Stored whitespace-stripped; the scanner strips whitespace from the source too.
IDIOM_SIGNATURES = {
    "_attr_chain_after_prefix": "full_name[len(prefix):]",
    "_build_condition_spec": 'op_str="truthy"',
    "_emit_set_block": "lines=[f'    {field_name} = {{']",
}


def _count_calls(source: str, func_name: str) -> int:
    """Count call sites ``func_name(...)`` in source (AST, excludes the def)."""
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name
        ):
            count += 1
    return count


def _has_def(source: str, func_name: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(n, ast.FunctionDef) and n.name == func_name
        for n in ast.walk(tree)
    )


def _normalized_idiom_count(source: str, idiom: str) -> int:
    """Count whitespace-insensitive occurrences of ``idiom`` in source."""
    squashed = "".join(source.split())
    needle = "".join(idiom.split())
    return squashed.count(needle)


class TestHelperMonopolyWpzg:
    """neograph-wpzg: extracted helpers are the single source for their logic.

    AST/normalized-text guard. Positive + negative + regex-slip meta-tests.
    """

    def test_helpers_are_defined_and_called_at_least_n_times(self):
        violations: list[str] = []
        for helper, (fname, min_calls) in MONOPOLIES.items():
            source = (SRC_DIR / fname).read_text()
            if not _has_def(source, helper):
                violations.append(f"  MISSING DEF: {helper} not defined in {fname}")
                continue
            calls = _count_calls(source, helper)
            if calls < min_calls:
                violations.append(
                    f"  UNDER-USED: {helper} called {calls}x in {fname} "
                    f"(monopoly requires >= {min_calls}); did a call site get inlined?"
                )
        assert violations == [], (
            "\nHelper-monopoly call-site violations:\n" + "\n".join(violations)
        )

    def test_inline_idiom_appears_only_in_its_helper(self):
        violations: list[str] = []
        for helper, idiom in IDIOM_SIGNATURES.items():
            fname = MONOPOLIES[helper][0]
            source = (SRC_DIR / fname).read_text()
            n = _normalized_idiom_count(source, idiom)
            if n != 1:
                violations.append(
                    f"  BYPASS: idiom for {helper} appears {n}x in {fname} "
                    f"(must be exactly 1 — only inside the helper). Idiom: {idiom!r}"
                )
        assert violations == [], (
            "\nHelper-monopoly inline-bypass violations:\n" + "\n".join(violations)
        )

    # --- meta-tests: prove the guards catch regressions ---

    def test_meta_call_count_catches_inlined_call_site(self):
        """Negative: a helper with too few call sites is flagged."""
        source = "def _emit_set_block(a, b):\n    return [a, b]\n\n_emit_set_block(1, 2)\n"
        # only 1 call; monopoly requires >= 4
        assert _count_calls(source, "_emit_set_block") < MONOPOLIES["_emit_set_block"][1]

    def test_meta_idiom_count_catches_duplicate(self):
        """Negative: the same idiom appearing twice is flagged."""
        dup = (
            "def helper():\n"
            "    remainder = full_name[len(prefix):]\n"
            "def bypass():\n"
            "    remainder = full_name[len(prefix):]\n"
        )
        assert _normalized_idiom_count(dup, "full_name[len(prefix):]") == 2

    def test_meta_idiom_count_resists_whitespace_slip(self):
        """Regex-slip: a spacing variant must NOT evade the idiom scanner."""
        spaced = "def bypass():\n    remainder = full_name[ len( prefix ) : ]\n"
        # A naive substring scan of the exact idiom would return 0 here and let
        # the bypass slip; the normalized scanner must still count it as 1.
        assert spaced.count("full_name[len(prefix):]") == 0  # naive scan misses it
        assert _normalized_idiom_count(spaced, "full_name[len(prefix):]") == 1
