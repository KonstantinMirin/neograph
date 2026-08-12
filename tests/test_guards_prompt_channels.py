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
    "raw_di_inputs": (
        "opt-IN by design (neograph-fqcm6), the exact sibling of raw_inputs — a "
        "compiler declares raw_di_inputs to receive the typed DI values behind the "
        "rendered di_inputs text. DefaultPromptCompiler renders everything itself, "
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
        for expected in (
            "node_name",
            "output_model",
            "output_schema",
            "context",
            "di_inputs",
            "raw_inputs",
            "raw_di_inputs",
        ):
            assert expected in offered, f"seam scan missed {expected!r}; found {sorted(offered)}"

    def test_known_ignored_entries_are_real_channels(self):
        """An entry for a channel the seam no longer offers is stale -- delete it."""
        stale = sorted(set(KNOWN_IGNORED) - _offered_channels())
        assert stale == [], f"KNOWN_IGNORED names channel(s) the seam does not offer: {stale}"

    def test_known_ignored_entries_carry_a_reason(self):
        empty = sorted(name for name, why in KNOWN_IGNORED.items() if not why.strip())
        assert empty == [], f"KNOWN_IGNORED entries with no reason: {empty}"


# neograph-fqcm6's disease pattern, one level narrower than the guard above: a
# channel written into all_kwargs via to_prompt_input(...) (i.e. a RENDERED
# value, like context/di_inputs -- NOT node_name/config/output_model, which are
# never rendered and so have no "raw form" question at all) with no companion
# raw_<channel> sibling giving a compiler the typed value behind the text. This
# is exactly the shape di_inputs had until fqcm6 added raw_di_inputs.
NO_RAW_SIBLING_ALLOWLIST: dict[str, str] = {
    "context": (
        "same disease, deferred separately (neograph-ebxdg) -- context's claimed "
        "escape hatch (a model's render_for_prompt() override) only customizes "
        "rendered TEXT, it never hands back the live object, so this is a real, "
        "tracked gap, not a false positive"
    ),
}


def _rendered_channels(tree: ast.AST, seam_fn: str, seam_dict: str) -> set[str]:
    """Channels written into ``all_kwargs`` via a ``to_prompt_input(...)`` call --
    i.e. channels that carry a RENDERED value and so raise the "does this need a
    raw sibling" question at all."""
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == seam_fn)
    rendered: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Call) and _call_name(node.value) == "to_prompt_input"):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == seam_dict
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                rendered.add(target.slice.value)
    return rendered


def _raw_sibling_channels(tree: ast.AST, seam_fn: str, seam_dict: str) -> set[str]:
    """The base channel names (``raw_<name>`` stripped to ``<name>``) that DO have
    a raw sibling -- assigned via a ``to_raw_inputs(...)`` call into
    ``all_kwargs["raw_<name>"]``."""
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == seam_fn)
    bases: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Call) and _call_name(node.value) == "to_raw_inputs"):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == seam_dict
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
                and target.slice.value.startswith("raw_")
            ):
                bases.add(target.slice.value[len("raw_") :])
    return bases


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _channels_missing_raw_sibling(tree: ast.AST, *, allowlist: dict[str, str], seam_fn: str = SEAM_FN, seam_dict: str = SEAM_DICT) -> set[str]:
    rendered = _rendered_channels(tree, seam_fn, seam_dict)
    have_sibling = _raw_sibling_channels(tree, seam_fn, seam_dict)
    return rendered - have_sibling - set(allowlist)


class TestRenderedChannelsHaveRawSiblingOrExemption:
    """neograph-fqcm6: a rendered channel with no raw counterpart is a silent
    dead end for a compiler that needs the typed value for logic, not template
    substitution. Every rendered channel either has a raw_<name> sibling, or an
    explicit, reasoned exemption -- never an unexamined gap."""

    def test_every_rendered_channel_has_a_raw_sibling_or_a_reasoned_exemption(self):
        tree = ast.parse(SEAM_FILE.read_text())
        missing = sorted(_channels_missing_raw_sibling(tree, allowlist=NO_RAW_SIBLING_ALLOWLIST))
        assert missing == [], (
            f"\nRendered channel(s) with no raw_<name> sibling and no exemption: {missing}\n\n"
            "That is neograph-fqcm6's disease: di_inputs was rendered with no way for a "
            "compiler to recover the typed value behind the text. Either add a raw_<name> "
            "sibling (to_raw_inputs(...) is generic over any Mapping, no new renderer code "
            "needed), or add a reasoned entry to NO_RAW_SIBLING_ALLOWLIST."
        )

    def test_allowlist_entries_are_real_missing_channels(self):
        """A stale entry (fixed since, or never a real gap) hides in a passing
        suite forever unless something checks it still names a real gap."""
        tree = ast.parse(SEAM_FILE.read_text())
        rendered = _rendered_channels(tree, SEAM_FN, SEAM_DICT)
        have_sibling = _raw_sibling_channels(tree, SEAM_FN, SEAM_DICT)
        actually_missing = rendered - have_sibling
        stale = sorted(set(NO_RAW_SIBLING_ALLOWLIST) - actually_missing)
        assert stale == [], f"NO_RAW_SIBLING_ALLOWLIST names channel(s) that already have a raw sibling (or aren't rendered at all): {stale}"

    def test_allowlist_entries_carry_a_reason(self):
        empty = sorted(name for name, why in NO_RAW_SIBLING_ALLOWLIST.items() if not why.strip())
        assert empty == [], f"NO_RAW_SIBLING_ALLOWLIST entries with no reason: {empty}"

    def test_di_inputs_no_longer_needs_the_allowlist(self):
        """Positive pin: the actual bug this guard exists for. di_inputs must have
        raw_di_inputs as a real sibling in the real file — not merely absent from
        the allowlist by omission."""
        tree = ast.parse(SEAM_FILE.read_text())
        assert "di_inputs" in _rendered_channels(tree, SEAM_FN, SEAM_DICT)
        assert "di_inputs" in _raw_sibling_channels(tree, SEAM_FN, SEAM_DICT)

    def test_detector_flags_a_synthetic_channel_with_no_sibling_and_no_exemption(self):
        """Negative meta-test: a channel added to a from-scratch synthetic seam
        with NO raw sibling and NO allowlist entry must be flagged — proves the
        detector actually fires, not just passes vacuously on the real file."""
        src = '''
def _compile_prompt(runtime, template, input_data):
    all_kwargs = {"node_name": "x"}
    all_kwargs["new_channel"] = to_prompt_input(new_channel, renderer=None)
    return runtime.prompt_compiler(template, input_data, **all_kwargs)
'''
        tree = ast.parse(src)
        missing = _channels_missing_raw_sibling(tree, allowlist={})
        assert missing == {"new_channel"}, f"detector failed to flag an unexempted rendered channel; saw {missing}"

    def test_detector_does_not_flag_a_channel_with_a_real_sibling(self):
        """Positive meta-test: a channel with an actual raw_<name> sibling must
        NOT be flagged (proves the detector isn't just flagging everything)."""
        src = '''
def _compile_prompt(runtime, template, input_data):
    all_kwargs = {"node_name": "x"}
    all_kwargs["new_channel"] = to_prompt_input(new_channel, renderer=None)
    all_kwargs["raw_new_channel"] = to_raw_inputs(new_channel)
    return runtime.prompt_compiler(template, input_data, **all_kwargs)
'''
        tree = ast.parse(src)
        missing = _channels_missing_raw_sibling(tree, allowlist={})
        assert missing == set(), f"detector flagged a channel that has a real raw sibling; saw {missing}"

    def test_detector_does_not_flag_non_rendered_channels(self):
        """A plain passthrough value (never routed through to_prompt_input) never
        raises the "needs a raw sibling" question — node_name/config/output_model
        etc. are correctly out of this guard's scope."""
        src = '''
def _compile_prompt(runtime, template, input_data):
    all_kwargs = {"node_name": node_name, "config": config, "output_model": output_model}
    return runtime.prompt_compiler(template, input_data, **all_kwargs)
'''
        tree = ast.parse(src)
        missing = _channels_missing_raw_sibling(tree, allowlist={})
        assert missing == set(), f"detector flagged a non-rendered channel; saw {missing}"
