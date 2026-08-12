"""Structural guard: every channel the seam OFFERS, the shipped compiler ANSWERS.

neograph-cbfd9. ``_llm_render._compile_prompt`` is the single seam that invokes a
``prompt_compiler``, and it offers a set of named channels — ``output_model``,
``di_inputs``, ``context``, ``raw_inputs`` and friends — filtered by introspection
so a compiler receives only what it declares. ``DefaultPromptCompiler`` declares
``**_kw``, which defeats the filter: it claims every channel and implemented all
but one. ``context`` — a channel the NODE AUTHOR declares — arrived and was
dropped, so a fail-loud compiler raised ``PromptVarMissing`` naming a variable the
author had declared, and a lenient one shipped the literal ``{brief}`` to a model.

The lesson is not "add context". It is that the offered set and the answered set
were related only by someone remembering. This guard derives BOTH sides by AST —
the offers from ``_compile_prompt`` itself, the answers from
``DefaultPromptCompiler.__call__`` — so a channel added to the seam tomorrow
either gets answered, or gets an explicit written reason, or fails CI.

``KNOWN_IGNORED`` is the escape hatch, and it is deliberately a map to a REASON,
not a set: "the default compiler does not need this" is a claim someone has to
type out, which is the whole difference between a decision and an oversight.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

SEAM_FILE = SRC_DIR / "_llm_render.py"
SEAM_FN = "_compile_prompt"
SEAM_DICT = "all_kwargs"

COMPILER_FILE = SRC_DIR / "prompt.py"
COMPILER_CLS = "DefaultPromptCompiler"

# Channels the shipped compiler deliberately does not consume. Each needs a reason.
KNOWN_IGNORED: dict[str, str] = {
    "config": "the RunnableConfig is runtime plumbing, not a template variable",
    "llm_config": "provider/tier settings; nothing a prompt template binds",
    "raw_inputs": (
        "opt-IN by design — a compiler declares raw_inputs to receive the objects "
        "behind the rendered text. DefaultPromptCompiler renders everything itself, "
        "so NOT declaring it is the correct opt-out, not an oversight"
    ),
}


def _offered_channels() -> set[str]:
    """The channel names ``_compile_prompt`` puts into its compiler-kwargs dict.

    Reads the seam itself rather than a hand-copied list -- a list copied from the
    seam is exactly the coupling that let ``context`` be forgotten.
    """
    tree = ast.parse(SEAM_FILE.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == SEAM_FN
    )
    offered: set[str] = set()
    for node in ast.walk(fn):
        # all_kwargs = {"node_name": ..., ...}
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            if any(isinstance(t, ast.Name) and t.id == SEAM_DICT for t in node.targets):
                offered |= {k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        # all_kwargs["context"] = context
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == SEAM_DICT
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    offered.add(target.slice.value)
    return offered


def _declared_params() -> set[str]:
    """Keyword parameters ``DefaultPromptCompiler.__call__`` declares by name."""
    tree = ast.parse(COMPILER_FILE.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == COMPILER_CLS)
    call = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "__call__"
    )
    return {a.arg for a in call.args.kwonlyargs} | {a.arg for a in call.args.args}


class TestSeamChannelsAreAnswered:
    """The offered set and the answered set stay related by CI, not by memory."""

    def test_every_offered_channel_is_declared_or_explicitly_ignored(self):
        offered = _offered_channels()
        unanswered = sorted(offered - _declared_params() - set(KNOWN_IGNORED))
        assert unanswered == [], (
            f"\n_compile_prompt offers channel(s) DefaultPromptCompiler neither declares "
            f"nor explicitly ignores: {unanswered}\n\n"
            "That is neograph-cbfd9: the compiler's **_kw made it claim every channel while "
            "implementing all but one, so a channel the node author DECLARED was silently "
            "dropped. Either declare the parameter and use it, or add it to KNOWN_IGNORED "
            "with a reason someone can disagree with."
        )

    def test_the_seam_scan_actually_finds_the_channels(self):
        """Non-vacuity: an empty or near-empty offer set would make the guard pass
        for nothing. Pin the channels that must be present today."""
        offered = _offered_channels()
        for expected in ("node_name", "output_model", "output_schema", "context", "di_inputs", "raw_inputs"):
            assert expected in offered, f"seam scan missed {expected!r}; found {sorted(offered)}"

    def test_known_ignored_entries_are_real_channels(self):
        """An entry for a channel the seam no longer offers is stale -- delete it."""
        stale = sorted(set(KNOWN_IGNORED) - _offered_channels())
        assert stale == [], f"KNOWN_IGNORED names channel(s) the seam does not offer: {stale}"

    def test_known_ignored_entries_carry_a_reason(self):
        empty = sorted(name for name, why in KNOWN_IGNORED.items() if not why.strip())
        assert empty == [], f"KNOWN_IGNORED entries with no reason: {empty}"
