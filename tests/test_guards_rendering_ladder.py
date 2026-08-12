"""Structural guard: ONE rendering ladder, ONE writer of the compiler-facing shape.

neograph-l2a7w. The defect was not a wrong line of code — it was a rule ("how a
pipeline value becomes LLM-facing text") that lived in three places at once and
disagreed pairwise:

    renderers._render_single           honored the presenter, did NOT coerce primitives
    _llm_render._resolve_var           coerced primitives, did NOT honor the presenter
    _tool_loop._render_tool_result...  honored an explicit renderer only, emitted 'None'

plus two channels (the Oracle merge and di_inputs) that skipped rendering
entirely, so the same logical value reached a prompt_compiler as rendered text on
one path and as a live Pydantic model on another. A compiler written the obvious
way returned "" on whichever path its author did not expect, and the model
answered coherently about nothing.

Collapsing the three was the fix. This guard is what stops a FOURTH from
appearing: the rule is only single-sourced for as long as nothing else calls the
BAML leaf or re-derives the compiler-facing shape.

Two ratchets, both AST-based (no regex, so there is no regex-slip case to
meta-test — a call is a call whatever the surrounding text looks like):

1. ``describe_value`` — the BAML leaf — is CALLED only from the ladder module.
2. ``to_prompt_input`` / ``to_raw_inputs`` — the compiler-facing shape — are
   called only from the seam that hands a compiler its input.

Note what is deliberately NOT banned: ``build_rendered_input`` still has several
callers (``_dispatch``, ``_llm_render``, ``prompt``). Under the idempotence
amendment (design A0) a call site may render early, where the node's own
``renderer=`` is in scope; rendering twice is a no-op. What must stay singular is
the SHAPE decision, which is ratchet 2.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# The BAML leaf may be called only here: describe_type.py defines it, renderers.py is
# the one ladder that consumes it.
BAML_CALLER_ALLOWLIST = frozenset({"renderers.py", "describe_type.py"})

# The compiler-facing shape has exactly one writer: the seam that invokes a
# prompt_compiler. renderers.py defines the helpers.
SHAPE_WRITER_ALLOWLIST = frozenset({"_llm_render.py", "renderers.py"})

BAML_LEAF = "describe_value"
SHAPE_FNS = frozenset({"to_prompt_input", "to_raw_inputs"})


def _called_names(tree: ast.AST) -> set[str]:
    """Every simple function name CALLED in *tree* (``f(...)`` and ``mod.f(...)``)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def _offenders(root: pathlib.Path, wanted: frozenset[str], allowlist: frozenset[str]) -> list[str]:
    """Files outside *allowlist* that CALL any name in *wanted*."""
    out: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name in allowlist:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - a broken source file fails elsewhere
            continue
        hit = _called_names(tree) & wanted
        if hit:
            out.append(f"{path.relative_to(root).as_posix()}: calls {sorted(hit)}")
    return out


class TestOneRenderingLadder:
    """The BAML leaf has exactly one consumer, and the shape has exactly one writer."""

    def test_baml_leaf_is_called_only_from_the_ladder(self):
        offenders = _offenders(SRC_DIR, frozenset({BAML_LEAF}), BAML_CALLER_ALLOWLIST)
        assert offenders == [], (
            f"\n{len(offenders)} module(s) call {BAML_LEAF}() outside the one ladder:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nThat is how neograph-l2a7w happened: three modules each grew their own "
            "partial version of the rendering rule and disagreed about the presenter, "
            "primitive coercion and None. Call renderers.to_rendered() instead — it is "
            "idempotent, so calling it from a site that may already have rendered is free."
        )

    def test_compiler_facing_shape_has_one_writer(self):
        offenders = _offenders(SRC_DIR, SHAPE_FNS, SHAPE_WRITER_ALLOWLIST)
        assert offenders == [], (
            f"\n{len(offenders)} module(s) build the compiler-facing shape outside the seam:\n"
            + "\n".join(f"  {o}" for o in offenders)
            + "\n\nThe shape a prompt_compiler receives is decided ONCE, in "
            "_llm_render._compile_prompt. A call site that decides it again is the "
            "original defect: the Oracle merge path skipped rendering because the "
            "decision lived at each call site instead of at the seam."
        )

    # ── meta-tests: the scanner actually detects / actually accepts ───────────

    def test_meta_detects_a_second_ladder(self, tmp_path: pathlib.Path):
        """POSITIVE: a new module that calls the BAML leaf directly is caught."""
        (tmp_path / "_sneaky_renderer.py").write_text(
            "from neograph.describe_type import describe_value\n\n"
            "def render_thing(v):\n"
            "    return describe_value(v)\n"
        )
        offenders = _offenders(tmp_path, frozenset({BAML_LEAF}), BAML_CALLER_ALLOWLIST)
        assert any("_sneaky_renderer.py" in o for o in offenders), (
            f"scanner failed to detect a second ladder; offenders={offenders}"
        )

    def test_meta_detects_an_attribute_call_form(self, tmp_path: pathlib.Path):
        """POSITIVE: the module-qualified form ``renderers.to_prompt_input(...)``
        is caught too — a bare-name scan would miss it."""
        (tmp_path / "_sneaky_shape.py").write_text(
            "from neograph import renderers\n\n"
            "def build(v):\n"
            "    return renderers.to_prompt_input(v)\n"
        )
        offenders = _offenders(tmp_path, SHAPE_FNS, SHAPE_WRITER_ALLOWLIST)
        assert any("_sneaky_shape.py" in o for o in offenders), (
            f"scanner failed to detect the attribute call form; offenders={offenders}"
        )

    def test_meta_accepts_a_delegating_call_site(self, tmp_path: pathlib.Path):
        """NEGATIVE: delegating to the ladder is the CORRECT shape and must pass.

        This is what the fix turned the two former re-implementations into, so a
        guard that flagged it would ban the cure along with the disease.
        """
        (tmp_path / "_good_citizen.py").write_text(
            "from neograph.renderers import to_rendered\n\n"
            "def render_tool_result(v, renderer=None):\n"
            "    return to_rendered(v, renderer, prefix='Tool result:')\n"
        )
        assert _offenders(tmp_path, frozenset({BAML_LEAF}), BAML_CALLER_ALLOWLIST) == []
        assert _offenders(tmp_path, SHAPE_FNS, SHAPE_WRITER_ALLOWLIST) == []

    def test_meta_allowlist_entries_are_not_vacuous(self):
        """An allowlist entry naming a file that does not exist is dead weight and
        hides drift — every entry must name a real module."""
        missing = sorted(
            name
            for name in (BAML_CALLER_ALLOWLIST | SHAPE_WRITER_ALLOWLIST)
            if not (SRC_DIR / name).exists()
        )
        assert missing == [], f"allowlist names non-existent module(s): {missing}"
