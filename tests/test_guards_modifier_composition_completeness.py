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
  ONE exit. RETARGETED for neograph-jtawq.4 Phase 2's ModifierCombo-keyed
  registry dispatch (the 6-branch if/elif chain this rule originally scanned
  no longer exists): three invariants now hold over ``decorator(f)`` -- (i)
  exactly one ``return`` (the terminal one); (ii) every membership-check block
  (one that calls a ``_build_*_node`` function) tests ``"<name>" in members``,
  never a re-derived raw-kwarg condition; (iii) the checked name set equals
  ``MODIFIER_KWARGS``' row names, so a modifier can never be silently
  unhandled. (i) generalizes the original "no return inside a modifier
  branch" rule to catch a second return ANYWHERE, not just one keyed on a
  raw-kwarg name that no longer appears in the source; (ii)+(iii) are new --
  they guard the registry-driven dispatch's OWN failure mode (a
  re-introduced raw-kwarg condition, or a dropped membership check) that the
  pre-registry scanner had no vocabulary to express.

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

from neograph._node_modifier_kwargs import MODIFIER_KWARGS
from tests.test_guards_node_kwarg_grid import _GridIO  # noqa: F401 - resolved via module globals, see usage below

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

#: RULE 1's expected membership-check name set, DERIVED from the registry
#: (neograph-jtawq.4) -- never hand-typed, so this guard cannot drift from
#: the table it exists to police.
MODIFIER_ROW_NAMES: frozenset[str] = frozenset(row.name for row in MODIFIER_KWARGS)

#: The 5 build-node functions a membership-check block is expected to call
#: (one per MODIFIER_KWARGS row).
_BUILD_NODE_FUNCTIONS: frozenset[str] = frozenset(
    {"_build_oracle_node", "_build_each_node", "_build_operator_node", "_build_loop_node", "_build_portal_node"}
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


def _walk_excluding_nested_defs(node: ast.AST):
    """Yield every descendant of ``node``, EXCLUDING the bodies of nested
    function/lambda definitions -- a closure's own return/if is not part of
    the enclosing function's control flow."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        stack.extend(ast.iter_child_nodes(child))


def _decorator_top_level_returns(body: list[ast.stmt]) -> list[int]:
    """Every ``return`` anywhere in ``decorator(f)``'s body (excluding nested
    defs). RULE 1(i): there must be exactly ONE -- the terminal ``return n``.
    A second return is an early-exit regrowth signature regardless of which
    condition guards it -- stronger than the pre-Phase-2 scanner, which only
    caught a return inside a raw-kwarg-named ``if``."""
    lines: list[int] = []
    for stmt in body:
        if isinstance(stmt, ast.Return):
            lines.append(stmt.lineno)
            continue
        for node in _walk_excluding_nested_defs(stmt):
            if isinstance(node, ast.Return):
                lines.append(node.lineno)
    return lines


def _membership_check_blocks(body: list[ast.stmt]) -> list[ast.If]:
    """Every top-level ``if`` in ``decorator(f)``'s body whose body calls one
    of the 5 ``_build_*_node`` functions -- the membership-check blocks the
    new registry-driven dispatch is built from."""
    blocks: list[ast.If] = []
    for stmt in body:
        if not isinstance(stmt, ast.If):
            continue
        calls_build_node = any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _BUILD_NODE_FUNCTIONS
            for n in ast.walk(stmt)
        )
        if calls_build_node:
            blocks.append(stmt)
    return blocks


def _membership_test_name(test: ast.expr) -> str | None:
    """If ``test`` is exactly ``"<name>" in members``, return ``<name>``;
    else ``None``. RULE 1(ii)'s regrowth signature is anything else -- a
    re-derived raw-kwarg condition (e.g. ``map_over is not None``)
    reappearing instead of reading the classified ``members`` set."""
    if (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.In)
        and isinstance(test.left, ast.Constant)
        and isinstance(test.left.value, str)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Name)
        and test.comparators[0].id == "members"
    ):
        return test.left.value
    return None


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
        """RULE 1 (neograph-jtawq.4 Phase 2 retarget). Three invariants over the
        ModifierCombo-keyed dispatch in ``decorator(f)``:

        (i) exactly one ``return`` (the terminal ``return n``) -- a second
            return is an early-exit regrowth signature regardless of which
            condition guards it. Pre-Phase-7 this cost
            ``@node(map_over=..., interrupt_when=...)`` its Operator with no
            error (neograph-s7zt3.10) -- the disease this whole guard file
            exists to catch, now pinned more generally than the original
            raw-kwarg-named scanner could.
        (ii) every membership-check block (a top-level ``if`` that calls one
             of the 5 ``_build_*_node`` functions) tests ``"<name>" in
             members`` -- never a re-derived raw-kwarg condition. A raw-kwarg
             test reappearing (e.g. ``if map_over is not None:``) is the
             regrowth signature for the ORIGINAL pre-Phase-2 disease: two
             independent sources of truth for "which modifiers apply" that
             can silently diverge.
        (iii) the checked name set equals MODIFIER_KWARGS' row names -- no
              modifier's membership check silently missing.
        """
        body = _node_decorator_body((SRC_DIR / "decorators.py").read_text())
        assert body is not None, "could not locate decorators.node.decorator -- update this guard"

        returns = _decorator_top_level_returns(body)
        assert len(returns) == 1, (
            f"\ndecorators.py: decorator(f) has {len(returns)} return(s) at line(s) {returns}, expected "
            "exactly 1 (the terminal `return n`). A second return is an early-exit regrowth signature -- "
            "neograph-s7zt3.10 found exactly that, twice."
        )

        blocks = _membership_check_blocks(body)
        checked_names: set[str] = set()
        malformed: list[int] = []
        for block in blocks:
            check_name = _membership_test_name(block.test)
            if check_name is None:
                malformed.append(block.lineno)
            else:
                checked_names.add(check_name)

        assert not malformed, (
            f"\ndecorators.py: membership-check block(s) at line(s) {malformed} call a _build_*_node "
            'function but do NOT test `"<name>" in members` -- a re-derived raw-kwarg condition is the '
            "regrowth signature this guard exists to catch."
        )
        assert checked_names == MODIFIER_ROW_NAMES, (
            f"\ndecorators.py: membership checks cover {sorted(checked_names)}, expected exactly "
            f"{sorted(MODIFIER_ROW_NAMES)} (MODIFIER_KWARGS' row names) -- a modifier with no membership "
            "check is silently unhandled."
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

    #: The healthy post-Phase-2 dispatch shape: derive combo -> 5 unconditional
    #: `"<name>" in members` checks -> one terminal register+return. `{extra}`
    #: lets a positive control inject a second return or malform one check.
    _REGISTRY_TEMPLATE = '''
def node(**kw):
    def decorator(f):
        n = build(f)
        combo = derive_combo(sugar_kwargs, node_label=node_label)
        members = modifier_names_for_combo(combo)
        if "oracle" in members:
            n = _build_oracle_node(n, node_label=node_label, f=f, kwargs=sugar_kwargs)
{extra}
        if "each" in members:
            n = _build_each_node(n, kwargs=sugar_kwargs)
        if "operator" in members:
            n = _build_operator_node(n, node_label=node_label, kwargs=sugar_kwargs)
        if "loop" in members:
            n = _build_loop_node(n, node_label=node_label, kwargs=sugar_kwargs)
{portal_check}
        _register_sidecar(n, f, param_names)
        return n
    return decorator
'''
    _HEALTHY_PORTAL_CHECK = '        if "portal" in members:\n            n = _build_portal_node(n, node_label=node_label, kwargs=sugar_kwargs)'

    def test_meta_scanner_catches_a_second_return(self) -> None:
        """POSITIVE control (i): a second return anywhere is caught, even
        outside a membership-check block."""
        diseased = self._REGISTRY_TEMPLATE.format(extra="            return n", portal_check=self._HEALTHY_PORTAL_CHECK)
        body = _node_decorator_body(diseased)
        assert body is not None
        assert len(_decorator_top_level_returns(body)) == 2, "scanner MISSED a second return -- it is vacuous"

    def test_meta_scanner_catches_a_raw_kwarg_condition_reappearing(self) -> None:
        """POSITIVE control (ii): a membership-check block whose test is a
        re-derived raw-kwarg condition (the ORIGINAL pre-Phase-2 disease
        shape) instead of `"oracle" in members` is flagged as malformed."""
        diseased = self._REGISTRY_TEMPLATE.replace(
            'if "oracle" in members:',
            "if ensemble_n is not None or merge_fn is not None:",
        ).format(extra="", portal_check=self._HEALTHY_PORTAL_CHECK)
        body = _node_decorator_body(diseased)
        assert body is not None
        blocks = _membership_check_blocks(body)
        malformed = [b for b in blocks if _membership_test_name(b.test) is None]
        assert malformed, "scanner MISSED a re-derived raw-kwarg condition -- it is vacuous"

    def test_meta_scanner_catches_a_missing_membership_check(self) -> None:
        """POSITIVE control (iii): dropping the portal check entirely (the
        exact neograph-s7zt3.10 shape -- a modifier silently unhandled)
        shrinks the checked-name set below MODIFIER_ROW_NAMES."""
        diseased = self._REGISTRY_TEMPLATE.format(extra="", portal_check="")
        body = _node_decorator_body(diseased)
        assert body is not None
        checked = {
            name
            for block in _membership_check_blocks(body)
            if (name := _membership_test_name(block.test)) is not None
        }
        assert checked != MODIFIER_ROW_NAMES, "scanner MISSED a dropped membership check -- it is vacuous"

    def test_meta_scanner_accepts_the_healthy_registry_dispatch_shape(self) -> None:
        """NEGATIVE control: the healthy shape -- one return, all 5 checks in
        `"<name>" in members` form -- passes all three invariants clean."""
        healthy = self._REGISTRY_TEMPLATE.format(extra="", portal_check=self._HEALTHY_PORTAL_CHECK)
        body = _node_decorator_body(healthy)
        assert body is not None

        assert len(_decorator_top_level_returns(body)) == 1, "false positive on the healthy shape's return count"

        blocks = _membership_check_blocks(body)
        checked_names: set[str] = set()
        for block in blocks:
            name = _membership_test_name(block.test)
            assert name is not None, f"false positive: healthy check at line {block.lineno} flagged as malformed"
            checked_names.add(name)
        assert checked_names == MODIFIER_ROW_NAMES, "false positive: healthy shape's checked names incomplete"

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


class TestModifierKwargsRegistryIntegrity:
    """neograph-jtawq.4 Phase 1: the ModifierCombo-keyed registry
    (``_node_modifier_kwargs.MODIFIER_KWARGS`` / ``IDENTITY_KWARGS``) that
    Phase 2 will make ``@node``'s dispatch authority. These guards fail
    BEFORE the registry exists (written first, per the refined plan's Phase 1
    -- bd show neograph-jtawq.4) and stay green forever after, so the
    registry cannot silently drift from ``_COMBO_MAP`` (the one composition/
    validity authority) or from ``node()``'s actual signature.
    """

    def test_combo_map_round_trips_through_modifier_names_for_combo(self) -> None:
        """Bijectivity: for every valid modifier-name set S, classifying S to
        a combo and asking that combo's membership back out returns S
        unchanged. If this ever fails, ``_COMBO_MAP`` and
        ``modifier_names_for_combo`` have silently diverged."""
        from neograph.modifiers import _COMBO_MAP, combo_for_modifier_names, modifier_names_for_combo

        for names in _COMBO_MAP:
            combo = combo_for_modifier_names(names)
            assert modifier_names_for_combo(combo) == names, (
                f"round trip broke for {sorted(names)}: combo={combo}, "
                f"back out={sorted(modifier_names_for_combo(combo))}"
            )

    def test_modifier_kwargs_row_names_equal_combo_map_modifier_universe(self) -> None:
        """MODIFIER_KWARGS declares exactly the modifier names _COMBO_MAP's
        keys are built from -- no row for a name _COMBO_MAP never mentions,
        no _COMBO_MAP name missing a row."""
        from neograph._node_modifier_kwargs import MODIFIER_KWARGS
        from neograph.modifiers import _COMBO_MAP

        combo_map_universe: set[str] = set()
        for names in _COMBO_MAP:
            combo_map_universe |= names

        row_names = {row.name for row in MODIFIER_KWARGS}
        assert row_names == combo_map_universe, (
            f"MODIFIER_KWARGS row names {sorted(row_names)} != _COMBO_MAP's modifier universe "
            f"{sorted(combo_map_universe)}"
        )

    def test_every_node_kwarg_is_identity_or_owned_by_at_least_one_row(self) -> None:
        """Anti-flat-explosion ratchet: every one of ``node()``'s kwargs (via
        live ``inspect.signature`` -- never a hand-copied list) appears in
        IDENTITY_KWARGS or in >=1 MODIFIER_KWARGS row's triggers/satellites.
        A kwarg added to ``node()`` without declaring its owning shape here
        fails this test -- the ratchet a 33rd kwarg cannot silently bypass.

        ``>=1``, not ``exactly 1``: ``on_exhaust`` is a documented SHARED
        satellite of both loop and portal (never both triggered in the same
        valid combo, but the registry still declares it on both rows) --
        the corrected wording from the architect review (doc Sec 6.3.3;
        'exactly one' is unsatisfiable and was a plan-wording bug, not a
        registry bug).
        """
        import inspect

        from neograph._node_modifier_kwargs import IDENTITY_KWARGS, MODIFIER_KWARGS
        from neograph.decorators import node

        sig = inspect.signature(node)
        node_kwargs = {name for name in sig.parameters if name != "fn"}

        owned: dict[str, int] = {}
        for row in MODIFIER_KWARGS:
            for kw in row.triggers | row.satellites:
                owned[kw] = owned.get(kw, 0) + 1

        unowned = node_kwargs - IDENTITY_KWARGS - owned.keys()
        assert not unowned, (
            f"node() kwargs with no declared owner: {sorted(unowned)} -- add them to "
            "IDENTITY_KWARGS or to a MODIFIER_KWARGS row's triggers/satellites"
        )

        # Every declared kwarg must be REAL (catches typos in the registry
        # tables pointing at a kwarg node() does not actually have).
        declared = IDENTITY_KWARGS | owned.keys()
        phantom = declared - node_kwargs
        assert not phantom, f"registry declares kwargs node() does not have: {sorted(phantom)}"

    def test_field_map_values_are_real_fields_on_the_row_modifier_class(self) -> None:
        """field_map honesty: every ``field_map`` value names a real field on
        the row's modifier class (catches a stale rename in either
        direction -- the modifier class changes a field name, or the
        registry typos one)."""
        from neograph._node_modifier_kwargs import MODIFIER_KWARGS
        from neograph.modifiers import Each, Loop, Operator, Oracle, Portal

        classes_by_name = {"each": Each, "oracle": Oracle, "operator": Operator, "loop": Loop, "portal": Portal}

        for row in MODIFIER_KWARGS:
            cls = classes_by_name[row.name]
            fields = cls.model_fields
            for kwarg, field in row.field_map.items():
                assert kwarg in (row.triggers | row.satellites), (
                    f"{row.name}: field_map key {kwarg!r} is not one of its own triggers/satellites"
                )
                assert field in fields, (
                    f"{row.name}: field_map[{kwarg!r}] = {field!r} is not a real field on "
                    f"{cls.__name__} (fields: {sorted(fields)})"
                )

    def test_valid_kwargs_derive_combo_agree_with_todays_live_dispatch(self) -> None:
        """Phase-1 audit (design Finding 1 resolution): for every VALID
        combo in the Phase-0 grid's 32 subsets, decorate a node through
        TODAY's live ``decorator(f)`` branches, then independently compute
        ``derive_combo`` on the SAME kwargs and assert it equals the built
        node's actual ``modifier_set.combo``.

        Proves the registry classifies identically to the current
        production dispatch -- from OUTSIDE, via the public @node surface --
        without decorator(f) importing or calling anything from the new
        registry yet (Phase 2 wires that). Runs from the test suite, not
        from inside decorator(f), so Phase 1 makes ZERO line-count change to
        decorators.py (it sits at its exact file-size ceiling).

        Reuses the grid's module-level ``_GridIO`` type (not a function-local
        class): a function-local Pydantic model needs ``caller_ns`` forward-
        ref resolution to work from inside a loop, which is incidental
        friction this audit does not need to re-solve.
        """
        from neograph import node
        from neograph._node_modifier_kwargs import derive_combo
        from neograph.modifiers import _COMBO_MAP
        from tests.test_guards_node_kwarg_grid import ALL_SUBSETS, _kwargs_for, _subset_id

        for names in ALL_SUBSETS:
            if names not in _COMBO_MAP:
                continue  # invalid combos are Phase-0's concern, not this audit's
            kwargs = _kwargs_for(names)

            @node(outputs=_GridIO, name=f"audit-{_subset_id(names)}", **kwargs)
            def _fn(seed: _GridIO) -> _GridIO: ...

            live_combo = _fn.modifier_set.combo
            registry_combo = derive_combo(kwargs)
            assert registry_combo is live_combo, (
                f"subset {sorted(names)}: registry derive_combo={registry_combo}, "
                f"live decorator(f) built combo={live_combo}"
            )
