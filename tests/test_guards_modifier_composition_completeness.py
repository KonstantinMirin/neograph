"""Structural guard: no modifier-composition path may stop after ONE modifier.

Origin (neograph-s7zt3.10, Phase 7). The codebase-scan for that ticket named the
disease in one line:

    a modifier-composition path that stops after handling ONE modifier -- by
    early-return, by a fail-loud has_operator raise, or by an emit-immediately
    walk step -- instead of composing every modifier the item's ModifierCombo
    decomposes to.

It had 13 instances across three files, and the WORST of them was silent:
``decorators.py``'s two ``map_over`` branches returned BEFORE the
``interrupt_when`` tail, so ``@node(map_over=..., interrupt_when=...)`` built a
plain EACH node with the Operator dropped and NO error. That is exactly the
"silent seam" class AGENTS.md's North Star calls an existential defect -- and
nothing in the suite could see it, because every test that could have caught it
asserted on a node built through a surface that did not take the early return.

The other two loci were fail-LOUD (``_agent_spec.py``'s per-arm
``has_operator`` raises, ``loader.py``'s orphan-check fallback), so they were
merely missing features. This guard therefore aims at the SILENT shape:

  RULE 1 (decorators.py) -- the ``@node`` modifier-application chain has exactly
  ONE exit. Every modifier sugar (``map_over``, ``ensemble_n``, ``interrupt_when``,
  ``loop_when``, ``portal``) must FALL THROUGH to the next, so adding a sixth
  cannot silently swallow the five after it. A new ``return`` inside that chain
  is the regrowth signature.

  RULE 2 (_agent_spec.py) -- the Operator postlude in ``_lower_construct_item``
  stays UNCONDITIONAL on the primary shape: it lives after the ``match``, not
  inside an arm. An arm that grows its own Operator handling is the export-side
  regrowth signature (it is how the pre-Phase-7 tree looked).

Pure ``ast``, no ``re`` -- so there is no regex-slip case to meta-test, the same
exemption ``tests/test_guards_combo_decomposition_consumers.py`` claims. Positive
and negative meta-tests are provided for both rules, each against a synthetic
source string so the meta-tests cannot rot with the real file.

NOT in scope: the fail-loud loci. Those are already pinned by equality censuses
(``test_guards_combo_decomposition_consumers.py``'s MIGRATED set,
``test_guards_each_oracle_fusion_predicate.py``'s FUSION_READERS) and by
``test_agent_spec_export.py::TestUnsupportedComboFallthroughRaise``, which asserts
the provisional raiser stays DELETED.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

#: The `@node` kwargs that each build-and-pipe one modifier. A branch testing any
#: of these is a link in the composition chain and must not terminate it.
MODIFIER_SUGAR_TESTS: frozenset[str] = frozenset(
    {"map_over", "has_oracle_kwarg", "interrupt_when", "loop_when", "portal"}
)


def _node_decorator_body(source: str) -> list[ast.stmt] | None:
    """The statement list of ``decorators.py``'s inner ``decorator(f)`` -- the
    function whose body IS the modifier-application chain."""
    for outer in ast.walk(ast.parse(source)):
        if not isinstance(outer, ast.FunctionDef) or outer.name != "node":
            continue
        for inner in outer.body:
            if isinstance(inner, ast.FunctionDef) and inner.name == "decorator":
                return inner.body
    return None


def _returns_inside_modifier_branches(body: list[ast.stmt]) -> list[tuple[int, str]]:
    """Every ``return`` that sits INSIDE a top-level ``if`` whose test mentions a
    modifier-sugar kwarg. The terminal ``return n`` is a sibling of those ifs, not
    inside one, so it is excluded by construction rather than by a name check."""
    offenders: list[tuple[int, str]] = []
    for stmt in body:
        if not isinstance(stmt, ast.If):
            continue
        names = {n.id for n in ast.walk(stmt.test) if isinstance(n, ast.Name)}
        if not (names & MODIFIER_SUGAR_TESTS):
            continue
        for node in ast.walk(stmt):
            # Do not descend into a nested def (a closure's own return is fine).
            if isinstance(node, ast.Return):
                offenders.append((node.lineno, sorted(names & MODIFIER_SUGAR_TESTS)[0]))
    return offenders


def _operator_postlude_is_inside_a_match_arm(source: str) -> list[int]:
    """Lines where ``_lower_operator`` is called from INSIDE a ``match`` arm of
    ``_lower_construct_item`` -- i.e. an arm grew its own Operator handling
    instead of deferring to the one shared postlude after the match."""
    offenders: list[int] = []
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef) or fn.name != "_lower_construct_item":
            continue
        for stmt in ast.walk(fn):
            if not isinstance(stmt, ast.Match):
                continue
            for case in stmt.cases:
                for node in ast.walk(case):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "_lower_operator"
                    ):
                        offenders.append(node.lineno)
    return offenders


class TestModifierCompositionCompleteness:
    """The two structural signatures of "one modifier handled, the rest dropped"."""

    def test_node_decorator_modifier_chain_has_no_early_return(self) -> None:
        """RULE 1. Every modifier-sugar branch in ``@node`` falls through.

        Regrowth here is SILENT: the node is built, assembly passes, and the
        dropped modifier simply is not there. Pre-Phase-7 this cost
        ``@node(map_over=..., interrupt_when=...)`` its Operator with no error.
        """
        body = _node_decorator_body((SRC_DIR / "decorators.py").read_text())
        assert body is not None, "could not locate decorators.node.decorator -- update this guard"

        offenders = _returns_inside_modifier_branches(body)
        assert not offenders, (
            "\ndecorators.py: return(s) inside a modifier-sugar branch of the @node "
            "modifier-application chain:\n"
            + "\n".join(f"  line {ln}: inside the `{kw}` branch" for ln, kw in offenders)
            + "\n\nEvery modifier sugar must FALL THROUGH to the ones after it "
            "(map_over -> ensemble_n -> interrupt_when -> loop_when -> portal), so a node "
            "carrying two of them gets BOTH. An early return silently drops every modifier "
            "downstream of it -- neograph-s7zt3.10 found exactly that, twice. If the "
            "combination is genuinely invalid, RAISE a ConstructError up in the kwarg "
            "validation block (as map_over+loop_when and portal+map_over already do); never "
            "return early and let the caller believe the modifier was applied."
        )

    def test_agent_spec_operator_postlude_is_not_inside_a_match_arm(self) -> None:
        """RULE 2. The export-side Operator composite stays a shared postlude.

        Before Phase 7 the BARE arm owned the only Operator handling and every
        other arm raised "no Agent Spec lowering yet". Re-inlining
        ``_lower_operator`` into an arm is that shape coming back.
        """
        offenders = _operator_postlude_is_inside_a_match_arm((SRC_DIR / "_agent_spec.py").read_text())
        assert not offenders, (
            f"\n_agent_spec.py: _lower_operator is called from inside a `match` arm of "
            f"_lower_construct_item at line(s) {offenders}.\n"
            "The Operator postlude is UNCONDITIONAL and orthogonal to the primary shape -- it "
            "belongs after the match, where every arm's bound _LoweredItem flows through it "
            "(mirroring compiler.py's _add_subgraph / _add_node_to_graph). An arm-local copy is "
            "how the pre-neograph-s7zt3.10 tree looked, and it is why four of the five fusion "
            "combos had no export at all."
        )


class TestModifierCompositionCompletenessMetaTests:
    """Positive + negative controls for both scanners, against synthetic sources.

    No regex-slip case: both scanners are pure ``ast``.
    """

    _CHAIN_TEMPLATE = '''
def node(**kw):
    def decorator(f):
        n = build(f)
        if map_over is not None and has_oracle_kwarg:
            n = n | Oracle()
            n = n | Each()
{fused_tail}
        elif map_over is not None:
            n = n | Each()
        if interrupt_when is not None:
            n = n | Operator()
        return n
    return decorator
'''

    def test_meta_scanner_catches_an_early_return_in_a_modifier_branch(self) -> None:
        """POSITIVE control: the exact pre-fix decorators.py shape is caught."""
        diseased = self._CHAIN_TEMPLATE.format(fused_tail="            return n")
        body = _node_decorator_body(diseased)
        assert body is not None
        assert _returns_inside_modifier_branches(body), "scanner MISSED an early return -- it is vacuous"

    def test_meta_scanner_accepts_the_fall_through_shape(self) -> None:
        """NEGATIVE control: the fixed shape, and the terminal `return n` that
        every version has, must NOT be reported."""
        healthy = self._CHAIN_TEMPLATE.format(fused_tail="            n = rebind(n)")
        body = _node_decorator_body(healthy)
        assert body is not None
        assert not _returns_inside_modifier_branches(body), (
            "scanner flagged the correct fall-through shape (or the terminal return) -- false positive"
        )

    _MATCH_TEMPLATE = '''
def _lower_construct_item(item):
    match decomp.primary:
        case PrimaryShape.EACH:
            arm = lower_each(item)
{arm_tail}
        case PrimaryShape.BARE:
            arm = lower_body(item)
{postlude}
'''

    def test_meta_scanner_catches_an_arm_local_operator_postlude(self) -> None:
        """POSITIVE control: _lower_operator re-inlined into an arm is caught."""
        diseased = self._MATCH_TEMPLATE.format(
            arm_tail="            check = _lower_operator(item, mods['operator'])",
            postlude="    return arm",
        )
        assert _operator_postlude_is_inside_a_match_arm(diseased), "scanner MISSED an arm-local postlude"

    def test_meta_scanner_accepts_the_shared_postlude_after_the_match(self) -> None:
        """NEGATIVE control: the shared postlude AFTER the match is not flagged."""
        healthy = self._MATCH_TEMPLATE.format(
            arm_tail="            pass",
            postlude="    check = _lower_operator(item, mods['operator'])\n    return arm",
        )
        assert not _operator_postlude_is_inside_a_match_arm(healthy), (
            "scanner flagged the correct shared-postlude shape -- false positive"
        )
