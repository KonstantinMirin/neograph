"""POSITIVE exhaustive AST shape-guard: every placeholder-prompt-emitting
construction site in ``src/neograph/_agent_spec.py`` is paired with a preceding
``_translate_placeholders(`` / ``_node_translation(`` call (neograph-tjupj).

This is NOT another symptom-shaped regex pin like the four classes in
``tests/test_guards_agent_spec_lowering.py`` (each of which pins ONE already-found
bug and asserts nothing about the general shape of the risk). It is the POSITIVE,
exhaustive shape-guard the retrospective (docs/design/
agent-spec-oracle-inputs-2026-07-25-architecture-retrospective.md sec 4.1) asked
for: enumerate ALL coupled sites, assert each is paired.

## Core Invariant (pinned here)

Every construction site in ``_agent_spec.py`` that emits a pyagentspec primitive
carrying neograph-authored ``${var}`` prompt text -- ``LlmNode.prompt_template``,
``AgentNode`` via nested ``Agent.system_prompt``, AND every ``_make_agent(...)``
call (which builds ``Agent(system_prompt=...)``) -- MUST be preceded, in the same
enclosing function body, by a ``_translate_placeholders(`` / ``_node_translation(``
call, so no untranslated ``${var}`` ever reaches the Agent Spec wire.

## Always-on (no importorskip)

Pure ``ast`` + the Tier-A ``PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS`` frozenset
(derived from ``NODE_FAMILIES``, which imports nothing from pyagentspec). This
guard runs in the CORE suite without the ``agent-spec`` extra -- a structural
guard silently not running is exactly the silent seam the north star forbids.

## Classification is DERIVED, never hand-copied

The coupled-constructor set is imported straight from the neograph-00447 shared
registry (``PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS`` = text_scan family
``| {"AgentNode"}``). A new text_scan prompt-bearing primitive pyagentspec ships
auto-flows into the walk -- a LOUD add, not a silent miss.

## No unpaired ``_make_agent`` sites remain (s7zt3.1 FIXED)

``_make_agent`` at ``_lower_portal_mesh_to_swarm`` used to ship a raw ``${var}``
to a foreign Swarm agent's ``system_prompt`` (BUG neograph-s7zt3.1, empirically
reproduced 2026-07-27). It rode a one-entry SHRINKING allowlist keyed to that
bug; the fix routed the mesh-member prompt through ``_translate_placeholders``
like every sibling site, the entry was removed, and the guard is now maximally
tight: EVERY ``_make_agent`` call must be translation-paired, no exceptions.
"""

from __future__ import annotations

import ast
import pathlib
from typing import NamedTuple

from tests.agent_spec_capabilities import PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS

AGENT_SPEC_FILE = (
    pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph" / "_agent_spec.py"
)

# The pairing target post-cbpyx (72fa5a1): the retired `_check_placeholder_inputs`
# is GONE (zero src hits); `_translate_placeholders` is the translator, and
# `_node_translation` is its Each/Loop wrapper that calls it internally.
TRANSLATE_CALLEES = frozenset({"_translate_placeholders", "_node_translation"})


class Site(NamedTuple):
    """A collected call site: bare callee name, line, and enclosing function."""

    name: str
    lineno: int
    func: str


def _callee_trailing_name(call: ast.Call) -> str | None:
    """`nodes_mod.LlmNode(...)` -> 'LlmNode'; bare `_make_agent(...)` -> '_make_agent'."""
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _parse(source_text: str) -> tuple[list[Site], list[Site], dict[str, list[int]]]:
    """AST-walk `source_text`, attributing every Call to its NEAREST enclosing
    FunctionDef.

    Returns ``(node_sites, make_agent_sites, translate_linenos_by_func)`` where:

    * ``node_sites`` -- Calls whose trailing name is in the DERIVED
      ``PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS`` set (family a).
    * ``make_agent_sites`` -- every ``_make_agent(`` Call (family b) -- the family
      that catches the mutual blind spot the AgentNode-only walk misses.
    * ``translate_linenos_by_func`` -- per enclosing function, the sorted linenos
      of every ``_translate_placeholders`` / ``_node_translation`` Call.
    """
    tree = ast.parse(source_text)
    all_calls: list[Site] = []
    translate_by_func: dict[str, list[int]] = {}

    def visit(node: ast.AST, func_name: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call):
                name = _callee_trailing_name(child)
                if name is not None:
                    all_calls.append(Site(name, child.lineno, func_name))
                    if name in TRANSLATE_CALLEES:
                        translate_by_func.setdefault(func_name, []).append(child.lineno)
            visit(child, func_name)

    visit(tree, "<module>")

    node_sites = [s for s in all_calls if s.name in PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS]
    make_agent_sites = [s for s in all_calls if s.name == "_make_agent"]
    return node_sites, make_agent_sites, translate_by_func


def _unpaired(sites: list[Site], translate_by_func: dict[str, list[int]]) -> list[Site]:
    """Sites NOT preceded (same enclosing function, lineno <) by a translate call."""
    out: list[Site] = []
    for s in sites:
        tlines = translate_by_func.get(s.func, [])
        if not any(t < s.lineno for t in tlines):
            out.append(s)
    return out


# -- Parse the real file ONCE at module load ---------------------------------
_SOURCE = AGENT_SPEC_FILE.read_text()
NODE_SITES, MAKE_AGENT_SITES, TRANSLATE_BY_FUNC = _parse(_SOURCE)

# -- Named exemptions (req-3) -- NOT prompt-bearing, so no pairing demanded ---
# Keyed by (constructor name, enclosing function) -- NEVER by line number
# (line numbers drift). InputMessageNode in _lower_operator is a Portal
# routing-CHECK node: it carries no neograph-authored ${var} prompt text.
NODE_SITE_EXEMPTIONS: frozenset[tuple[str, str]] = frozenset(
    {("InputMessageNode", "_lower_operator")}
)

# Bare Agent() at :370 inside _make_agent is DELIBERATELY EXCLUDED FROM THE SET
# (not in PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS): it receives an
# already-translated system_prompt PARAM, so a same-body translate would be a
# false positive. Its pairing is enforced at the _make_agent CALL sites (family
# b) below, not here.

# The req-4 SHRINKING allowlist (MAKE_AGENT_UNPAIRED_ALLOWLIST) is GONE: its
# sole entry (_lower_portal_mesh_to_swarm, BUG neograph-s7zt3.1) was removed when
# the fix paired that site with _translate_placeholders, per the allowlist's own
# exactly-1 meta-check instructions. Every _make_agent call is now unconditionally
# translation-paired -- a new unpaired site fails loud with no escape hatch short
# of filing a ticketed bug and re-introducing a one-entry allowlist keyed to it.

# -- Expected counts (req-3 drift tripwire) ----------------------------------
# If any count below changes, a new placeholder-emitting / _make_agent site was
# added -- verify it has a paired _translate_placeholders/_node_translation call
# OR a named exemption OR a ticketed shrinking-allowlist entry; never silent
# drift. Current census (grep-verified, src/neograph/_agent_spec.py):
#   node sites (6): LlmNode _lower_generation_step + _lower_oracle merge_node,
#     AgentNode _lower_generation_step, InputMessageNode Portal-input (_lower_operator),
#     AND the Phase-6 PORTAL_OPERATOR mesh-exit composite (neograph-s7zt3.2):
#     AgentNode(Swarm) + pause InputMessageNode, both in _lower_portal_mesh_to_swarm.
#   _make_agent (2): _lower_generation_step (paired), _lower_portal_mesh_to_swarm
#     (paired since the s7zt3.1 fix).
# neograph-2s2o6 consolidation: _lower_node's + _lower_oracle-variant's two
# hand-written mode dispatches collapsed into ONE _lower_generation_step, so the
# 5 SEMANTIC dispatch branches now map onto 3 PHYSICAL paired coupled sites
# (LlmNode-think + AgentNode-agent/act inside _lower_generation_step, + the
# _lower_oracle merge_prompt LlmNode). Fewer physical sites, SAME pairing
# invariant -- the guard's discriminating power (the _unpaired check + meta-guard)
# is unchanged; these counts are the census tripwire, not the safety net.
# neograph-s7zt3.2 (Phase 6): the PORTAL_OPERATOR mesh-exit pause composite adds
# TWO more collected sites in _lower_portal_mesh_to_swarm -- AgentNode(agent=Swarm)
# and a prompt-less InputMessageNode pause. Both land AFTER that function's own
# per-member _translate_placeholders loop, so both are translation-PAIRED by the
# lineno heuristic (the InputMessageNode carries no ${var} -- outputs-only, like
# _lower_operator's -- so its "paired" tag is a benign over-count, not a safety
# concern). Neither can leak an untranslated ${var} to the wire.
EXPECTED_NODE_SITES = 6
EXPECTED_MAKE_AGENT_SITES = 2
EXPECTED_PAIRED_NODE_SITES = 5  # 6 collected minus the ONE unpaired _lower_operator InputMessageNode (exempt)
EXPECTED_PAIRED_MAKE_AGENT_SITES = 2  # ALL of them -- no allowlist since the s7zt3.1 fix

# Req-4 reconciliation: post-2s2o6 the 5 SEMANTIC matrix dispatch branches
# (_lower_node{think,agent/act} + _lower_oracle{think-variant,agent/act-variant,
# merge_prompt}) lower through the shared _lower_generation_step, collapsing onto
# 3 PHYSICAL placeholder-coupled sites; Phase 6's mesh-exit composite adds 2 more
# paired physical sites (AgentNode(Swarm) + pause InputMessageNode in
# _lower_portal_mesh_to_swarm), for 5 total. The guard's paired NODE-site count
# must equal this physical count (a lost/added physical site still fails loud);
# it is NOT the semantic-branch count.
PHYSICAL_COUPLED_SITES = 5


class TestPlaceholderEmittingSitesAreExhaustivelyCollected:
    """The AST walk must collect exactly the known coupled construction sites."""

    def test_node_site_count_matches_expected_census(self):
        # req-3 count assertion -- drift tripwire (see EXPECTED_NODE_SITES comment).
        assert len(NODE_SITES) == EXPECTED_NODE_SITES, (
            f"placeholder-prompt-emitting node sites drifted: expected "
            f"{EXPECTED_NODE_SITES}, found {len(NODE_SITES)} -> {NODE_SITES}. A new "
            "site was added -- give it a paired _translate_placeholders call OR a "
            "named exemption OR a ticketed shrinking-allowlist entry; never silent drift."
        )

    def test_node_sites_break_down_by_constructor(self):
        by_name: dict[str, int] = {}
        for s in NODE_SITES:
            by_name[s.name] = by_name.get(s.name, 0) + 1
        assert by_name == {"LlmNode": 2, "AgentNode": 2, "InputMessageNode": 2}, (
            f"coupled-constructor breakdown drifted: {by_name}"
        )

    def test_make_agent_site_count_matches_expected_census(self):
        # req-3 count assertion for family (b).
        assert len(MAKE_AGENT_SITES) == EXPECTED_MAKE_AGENT_SITES, (
            f"_make_agent call sites drifted: expected {EXPECTED_MAKE_AGENT_SITES}, "
            f"found {len(MAKE_AGENT_SITES)} -> {MAKE_AGENT_SITES}."
        )

    def test_the_derived_set_is_not_a_hardcoded_llmnode_literal(self):
        # req-1 forward-compat: the target set is DERIVED from NODE_FAMILIES'
        # text_scan family (LOUD add, not silent miss), not {LlmNode}.
        assert "LlmNode" in PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS
        assert "AgentNode" in PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS
        assert "InputMessageNode" in PLACEHOLDER_PROMPT_EMITTING_CONSTRUCTORS, (
            "the derived set must include the broad text_scan family (so a future "
            "coupled primitive is a loud add) -- it is NOT a hardcoded {LlmNode}"
        )


class TestEveryCoupledNodeSiteIsTranslationPaired:
    """Every collected node site is preceded by a translate call in the same
    enclosing function -- except the ONE named, documented exemption."""

    def test_every_unpaired_node_site_is_a_named_exemption(self):
        unpaired = _unpaired(NODE_SITES, TRANSLATE_BY_FUNC)
        offenders = [s for s in unpaired if (s.name, s.func) not in NODE_SITE_EXEMPTIONS]
        assert not offenders, (
            "placeholder-prompt-emitting node site(s) with NO preceding "
            "_translate_placeholders/_node_translation call in the same enclosing "
            f"function -- a raw ${{var}} would reach the Agent Spec wire: {offenders}"
        )

    def test_the_only_exemption_actually_fires(self):
        # The exemption machinery is real, not decorative: InputMessageNode:812
        # IS collected and IS unpaired -- and is the SOLE named exemption.
        unpaired = _unpaired(NODE_SITES, TRANSLATE_BY_FUNC)
        exempt = {(s.name, s.func) for s in unpaired if (s.name, s.func) in NODE_SITE_EXEMPTIONS}
        assert exempt == {("InputMessageNode", "_lower_operator")}, (
            f"expected exactly the InputMessageNode/_lower_operator exemption, got {exempt}"
        )

    def test_paired_node_site_count(self):
        unpaired = _unpaired(NODE_SITES, TRANSLATE_BY_FUNC)
        paired = len(NODE_SITES) - len(unpaired)
        assert paired == EXPECTED_PAIRED_NODE_SITES, (
            f"paired coupled node sites: expected {EXPECTED_PAIRED_NODE_SITES}, got {paired}"
        )


class TestEveryMakeAgentCallIsTranslationPaired:
    """Every _make_agent call is preceded by a translate call -- no exceptions
    (the s7zt3.1 shrinking allowlist was emptied and removed by the fix)."""

    def test_every_make_agent_call_is_translation_paired(self):
        unpaired = _unpaired(MAKE_AGENT_SITES, TRANSLATE_BY_FUNC)
        assert not unpaired, (
            "_make_agent call(s) with NO preceding _translate_placeholders call in "
            "the same enclosing function -- a raw ${var} would reach the nested "
            f"Agent.system_prompt on the wire: {unpaired}. Add the pairing; if a "
            "genuine, ticketed seam, escalate it as its own bug and re-introduce a "
            "one-entry shrinking allowlist keyed to that bug (see neograph-s7zt3.1 "
            "for the precedent)."
        )

    def test_paired_make_agent_count(self):
        unpaired = _unpaired(MAKE_AGENT_SITES, TRANSLATE_BY_FUNC)
        paired = len(MAKE_AGENT_SITES) - len(unpaired)
        assert paired == EXPECTED_PAIRED_MAKE_AGENT_SITES, (
            f"paired _make_agent sites: expected {EXPECTED_PAIRED_MAKE_AGENT_SITES}, got {paired}"
        )


class TestMatrixReconciliation:
    """Req-4: the guard's paired-node-site count must equal the matrix's coupled
    dispatch-branch count, and the Portal->Swarm discrepancy (once RECORDED via
    the shrinking allowlist) is now CLOSED by the s7zt3.1 fix."""

    def test_paired_node_sites_equal_matrix_coupled_dispatch_branches(self):
        unpaired = _unpaired(NODE_SITES, TRANSLATE_BY_FUNC)
        paired = len(NODE_SITES) - len(unpaired)
        assert paired == PHYSICAL_COUPLED_SITES, (
            f"guard paired node sites ({paired}) must equal the "
            f"{PHYSICAL_COUPLED_SITES} PHYSICAL coupled construction sites the 5 "
            "SEMANTIC matrix dispatch branches collapse onto post-2s2o6 "
            "(LlmNode-think + AgentNode-agent/act in _lower_generation_step, + the "
            "_lower_oracle merge_prompt LlmNode). A lost/added physical site fails loud."
        )

    def test_portal_swarm_req4_discrepancy_is_closed(self):
        # Req-4 finding, recorded then FIXED: the matrix classifies Portal as
        # per-node UNSUPPORTED, so it never exercises the successful Portal->Swarm
        # lowering -- guard AND matrix shared that blind spot. Family (b)
        # (_make_agent CALL sites) surfaced the seam, escalated as
        # neograph-s7zt3.1, whose fix paired _lower_portal_mesh_to_swarm's
        # _make_agent with _translate_placeholders. This test pins BOTH halves:
        # the once-discrepant function now carries its own translate call, and no
        # unpaired _make_agent site exists anywhere.
        assert TRANSLATE_BY_FUNC.get("_lower_portal_mesh_to_swarm"), (
            "_lower_portal_mesh_to_swarm must call _translate_placeholders (the "
            "s7zt3.1 fix) -- its disappearance means the mesh-member prompt "
            "translation was removed or moved without updating this guard."
        )
        unpaired_make_agent = _unpaired(MAKE_AGENT_SITES, TRANSLATE_BY_FUNC)
        assert unpaired_make_agent == [], (
            "the req-4 Portal->Swarm discrepancy was closed by neograph-s7zt3.1; "
            f"an unpaired _make_agent site reappeared: {unpaired_make_agent}"
        )


class TestPairingCheckMetaGuard:
    """Positive+negative meta-guard (mirrors the plain-substring meta-tests in
    tests/test_guards_agent_spec_lowering.py): prove the pairing check actually
    FLAGS an unpaired site -- so the guard above is not vacuously green."""

    def test_negative_unpaired_llmnode_is_flagged(self):
        # Synthetic function constructing an LlmNode with NO preceding translate.
        buggy = (
            "def _fake_lower():\n"
            "    return nodes_mod.LlmNode(prompt_template=raw)\n"
        )
        node_sites, _make_agent, translate_by_func = _parse(buggy)
        assert len(node_sites) == 1 and node_sites[0].name == "LlmNode"
        unpaired = _unpaired(node_sites, translate_by_func)
        assert unpaired == node_sites, (
            "an LlmNode with no preceding _translate_placeholders MUST be flagged "
            "unpaired -- otherwise the guard is vacuously green"
        )

    def test_negative_unpaired_make_agent_is_flagged(self):
        # Synthetic function calling _make_agent with NO preceding translate --
        # the exact shape of the neograph-s7zt3.1 seam.
        buggy = (
            "def _fake_swarm():\n"
            "    agent = _make_agent(member, tools_mod, [], [], member.prompt or '')\n"
        )
        _node_sites, make_agent_sites, translate_by_func = _parse(buggy)
        assert len(make_agent_sites) == 1
        unpaired = _unpaired(make_agent_sites, translate_by_func)
        assert unpaired == make_agent_sites, (
            "a _make_agent call with no preceding _translate_placeholders MUST be "
            "flagged unpaired"
        )

    def test_positive_paired_llmnode_is_not_flagged(self):
        # Synthetic function WITH a preceding translate -> must NOT be flagged.
        good = (
            "def _fake_ok():\n"
            "    rewritten, ref, flat = _translate_placeholders(text, inputs, name)\n"
            "    return nodes_mod.LlmNode(prompt_template=rewritten)\n"
        )
        node_sites, _make_agent, translate_by_func = _parse(good)
        assert len(node_sites) == 1
        assert _unpaired(node_sites, translate_by_func) == [], (
            "an LlmNode preceded (same function) by _translate_placeholders MUST NOT "
            "be flagged -- the pairing check must not raise false positives"
        )

    def test_node_translation_wrapper_also_satisfies_pairing(self):
        # _node_translation is the Each/Loop wrapper that calls _translate_placeholders
        # internally -- it counts as a valid pairing source.
        good = (
            "def _fake_each():\n"
            "    _r, inner_inputs, _f = _node_translation(node)\n"
            "    return nodes_mod.LlmNode(prompt_template=inner_inputs)\n"
        )
        node_sites, _make_agent, translate_by_func = _parse(good)
        assert _unpaired(node_sites, translate_by_func) == []
