"""Structural guard: the Each x Oracle FUSION test has exactly ONE authority.

neograph-c265k. The disease this guard exists to kill: a consumer that asks
"is this item the fused Each x Oracle node?" by HAND -- an ``"oracle"`` /
``"each"`` membership or ``.get(...)`` co-presence read, in either polarity, or
a ``ms.each is not None and ms.oracle is not None`` slot read -- instead of
calling the single named predicate ``is_each_oracle_fused`` in
``src/neograph/modifiers.py``.

Why the concept needs its own predicate at all: ``COMBO_DECOMPOSITION`` folds
``EACH_ORACLE`` / ``EACH_ORACLE_OPERATOR`` down to ``primary=EACH`` (the fusion
is a Node concern, and ``agent-spec-rewrite-2026-07-27.md`` states that as a
deliberate design decision), so a consumer standing in a ``PrimaryShape.EACH``
arm needs a SECOND, orthogonal test to split a fused node from a plain Each one.
Before neograph-c265k that second test was open-coded in six places.

Why the guard is context-aware rather than a spelling ban
--------------------------------------------------------
A naive "ban ``\"oracle\" in mods`` outside modifiers.py" rule is
UNSATISFIABLE: the tree holds EIGHT legitimate look-alike presence reads that
spell the same token and must never move (``_fan_agent.py``'s first-hit label
chain and its Oracle-over-fan-out / Oracle-multi-input gates, ``state.py``'s
construct-wide ``has_any_oracle`` / ``has_any_each`` scan, and
``testing/scaffold.py``'s serialized "has ANY modifier" filter). So the disease
is SYNTACTIC-WITH-CONTEXT and the scanners encode exactly that context.

Three rules
-----------
* **R1 -- expression-level co-presence.** An ``ast.BoolOp`` whose set of
  membership-tested / ``.get()``-ed string names is a SUBSET of
  ``{"each", "oracle"}``, CONTAINS ``"oracle"``, and which also contains either
  the ``"each"`` test or a ``PrimaryShape.EACH`` reference.
  The ``<= {"each","oracle"}`` subset clause is LOAD-BEARING: it is what spares
  ``testing/scaffold.py``'s four-name disjunction. The "also names each / EACH"
  clause is what spares ``_fan_agent.py``'s ``"oracle" in mods and
  item.fan_out_param is not None`` -- the sharpest false positive in the tree,
  a genuine co-presence conjunction whose SECOND operand is not the Each
  modifier.

* **R2 -- context-level (widened per R-RC2).** An ``"oracle"`` ``In``/``NotIn``
  compare **OR** an ``X.get("oracle")`` call, lexically inside an
  ``ast.match_case`` whose PATTERN names ``PrimaryShape.EACH`` (the pattern, the
  guard, and the body are all scanned). Inside such an arm the ``"each"`` half
  is implied by the arm itself, so the co-presence is contextual rather than
  expressed. Covering the ``.get`` spelling is deliberate: it is precisely the
  spelling this ticket removes from ``compiler.py``, so a future re-inline of
  ``mods.get("oracle") is not None`` inside an EACH arm must not walk through.

* **R3 -- slot-attribute spelling (pre-emptive, per R-RC3).** An ``ast.BoolOp``
  whose set of ``<expr>.<slot> is not None`` reads, restricted to the modifier
  vocabulary ``{each, oracle, loop, operator, portal}``, equals EXACTLY
  ``{each, oracle}``. R3 has ZERO hits on the tree today and is GREEN from the
  start -- it is a pre-emptive RATCHET, not a migration check, because a
  consumer holding a ``Node``/``ModifierSet`` rather than a ``mods`` dict
  (``loader.py``, ``_validation_inputs.py``, ``_param_classify.py`` all already
  hold one) would reach for that spelling naturally, and such a reader evades
  BOTH assertion (a)'s dict-shaped rules AND assertion (c) (a file that never
  imports the predicate is on neither side of the equality). Its meta-tests --
  not its offender count -- are what prove it is not a dead scanner. It spares
  ``ModifierSet.model_post_init``'s four pairwise excludes (each+loop,
  oracle+loop, portal+each, portal+oracle) and ``_validation_inputs.py``'s
  ``ms is not None and ms.each is not None`` by construction.

Four assertions
---------------
  (a) **No open-coded fusion test** anywhere under ``src/neograph`` EXCEPT the
      owner ``modifiers.py``. Offenders are reported as ``file:line  rule``.
  (b) **The owner IS flagged.** ``modifiers.py`` must produce exactly one hit,
      by R1 -- the predicate body itself. This is the ANTI-DEAD-SCANNER
      assertion: a scanner that matched nothing would satisfy (a) vacuously,
      forever, and nobody would notice.
  (c) **Consumer inventory (anti-tautology).** The hand-written
      ``FUSION_READERS`` literal must EQUAL the filesystem-derived set of files
      that import AND use ``is_each_oracle_fused``. A seventh consumer -- or a
      caller that drops the predicate and re-inlines the test -- fails loud.
      This equality is a RATCHET in BOTH directions: when a consumer
      legitimately disappears (e.g. neograph-s7zt3.10 / Phase 7 may delete
      ``_agent_spec.py``'s fusion check), SHRINK the literal in the same commit.
      Do not relax the equality into a subset test -- the same instruction the
      sibling guard's ``PENDING`` ratchet carries.
  (d) Scanner meta-tests, positive AND negative, for all three rules, including
      REAL-FILE negative controls (``_fan_agent.py``, ``testing/scaffold.py``,
      and ``state.py``'s ``has_any_oracle`` lines).

Deliberately NOT in scope: ``is_each_oracle_fused`` must NOT be added to
``TABLE_SYMBOLS`` in ``tests/test_guards_combo_decomposition_consumers.py``.
It is a modifier-PRESENCE predicate, not a decomposition-table symbol; adding it
would drag every future presence-only reader into that guard's combo-consumer
inventory (whose assertion (c) is also an equality).

Written in pure ``ast`` with no ``re``, so it is exempt by construction from
``tests/test_guards_meta.py``'s named-regex/slip-test discipline -- the same
move ``tests/test_guards_combo_decomposition_consumers.py`` makes.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# --- independent literals (hand-written; never derived from the scan) --------

#: The single definition site of the fusion predicate.
PREDICATE_OWNER = "modifiers.py"

#: The one name that answers "is this the fused Each x Oracle node?".
PREDICATE = "is_each_oracle_fused"

#: Every file that must CALL the predicate instead of open-coding the test.
#: Hand-written and sourced from the neograph-c265k census, independently of the
#: filesystem scan assertion (c) compares it against. RATCHET IN BOTH
#: DIRECTIONS: a new consumer must be added here; a consumer that legitimately
#: disappears must be REMOVED here in the same commit.
FUSION_READERS: frozenset[str] = frozenset(
    {
        "compiler.py",  # the pre-`match` fusion split (M x N Send topology)
        "state.py",  # dict-form fused arm + single-type collector
        "_state_write.py",  # Each key-wrapping suppression for the fusion
        "_subconstruct.py",  # NEGATED: EACH-shaped but not fused
        "_agent_spec.py",  # fused combos have no Agent Spec lowering
    }
)

#: The modifier slot vocabulary R3 restricts itself to.
MODIFIER_VOCAB: frozenset[str] = frozenset({"each", "oracle", "loop", "operator", "portal"})

#: The two names whose CO-PRESENCE is the fusion question.
FUSION_NAMES: frozenset[str] = frozenset({"each", "oracle"})


# --- scanners (pure ast) ----------------------------------------------------


def _primary_shape_bindings(tree: ast.Module) -> set[str]:
    """Local binding names for ``PrimaryShape`` in this module (alias-tolerant).

    Mirrors ``_combo_binding_names`` in the sibling Phase-3 guard, so
    ``from neograph.modifiers import PrimaryShape as _PS`` + ``_PS.EACH`` is
    still recognised as an EACH reference.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "PrimaryShape":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ClassDef) and node.name == "PrimaryShape":
            names.add(node.name)
    return names


def _tested_string_names(node: ast.AST) -> set[str]:
    """String names membership-tested (``"x" in d`` / ``"x" not in d``) or
    ``.get("x")``-ed anywhere inside ``node``."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in sub.ops):
            if isinstance(sub.left, ast.Constant) and isinstance(sub.left.value, str):
                names.add(sub.left.value)
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get":
            if sub.args and isinstance(sub.args[0], ast.Constant) and isinstance(sub.args[0].value, str):
                names.add(sub.args[0].value)
    return names


def _names_each_shape(node: ast.AST, shape_bindings: set[str]) -> bool:
    """True when ``node`` contains a ``PrimaryShape.EACH`` attribute reference."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Attribute)
            and sub.attr == "EACH"
            and isinstance(sub.value, ast.Name)
            and sub.value.id in shape_bindings
        ):
            return True
    return False


def _slot_is_not_none_reads(node: ast.AST) -> set[str]:
    """Attribute names read as ``<expr>.<attr> is not None`` inside ``node``,
    restricted to the modifier slot vocabulary."""
    slots: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Compare) or len(sub.ops) != 1:
            continue
        if not isinstance(sub.ops[0], ast.IsNot):
            continue
        comparator = sub.comparators[0]
        if not (isinstance(comparator, ast.Constant) and comparator.value is None):
            continue
        if isinstance(sub.left, ast.Attribute) and sub.left.attr in MODIFIER_VOCAB:
            slots.add(sub.left.attr)
    return slots


def _r1_sites(tree: ast.Module, shape_bindings: set[str]) -> set[int]:
    """R1: expression-level co-presence of ``each`` and ``oracle``."""
    hits: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp):
            continue
        tested = _tested_string_names(node)
        if "oracle" not in tested or not tested <= FUSION_NAMES:
            continue
        if "each" in tested or _names_each_shape(node, shape_bindings):
            hits.add(node.lineno)
    return hits


def _r2_sites(tree: ast.Module, shape_bindings: set[str]) -> set[int]:
    """R2: an ``"oracle"`` membership test OR ``.get("oracle")`` call lexically
    inside a ``match_case`` whose pattern names ``PrimaryShape.EACH``."""
    hits: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.match_case):
            continue
        if not _names_each_shape(node.pattern, shape_bindings):
            continue
        scanned: list[ast.AST] = [node.pattern, *node.body]
        if node.guard is not None:
            scanned.append(node.guard)
        for region in scanned:
            for sub in ast.walk(region):
                if isinstance(sub, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in sub.ops):
                    if isinstance(sub.left, ast.Constant) and sub.left.value == "oracle":
                        hits.add(sub.lineno)
                elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get":
                    if sub.args and isinstance(sub.args[0], ast.Constant) and sub.args[0].value == "oracle":
                        hits.add(sub.lineno)
    return hits


def _r3_sites(tree: ast.Module) -> set[int]:
    """R3: a ``BoolOp`` whose modifier-slot ``is not None`` reads are exactly
    ``{each, oracle}`` -- the ModifierSet-slot spelling of the same question."""
    hits: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and _slot_is_not_none_reads(node) == FUSION_NAMES:
            hits.add(node.lineno)
    return hits


def _fusion_test_sites(source: str) -> list[tuple[int, str]]:
    """Return sorted ``(lineno, rule)`` for every open-coded fusion test.

    A line caught by more than one rule is reported ONCE with the rules joined,
    so the offender count is a count of SITES, not of rule firings.
    """
    tree = ast.parse(source)
    shape_bindings = _primary_shape_bindings(tree)
    by_line: dict[int, list[str]] = {}
    for rule, linenos in (
        ("R1", _r1_sites(tree, shape_bindings)),
        ("R2", _r2_sites(tree, shape_bindings)),
        ("R3", _r3_sites(tree)),
    ):
        for lineno in linenos:
            by_line.setdefault(lineno, []).append(rule)
    return sorted((lineno, "+".join(sorted(rules))) for lineno, rules in by_line.items())


def _uses_predicate(source: str) -> bool:
    """True when the module imports ``is_each_oracle_fused`` from
    ``neograph.modifiers`` AND actually calls/uses the binding.

    A dead import must not satisfy assertion (c) -- the same R-L3 rule the
    sibling guard's ``_used_table_symbols`` applies.
    """
    tree = ast.parse(source)
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] == "modifiers":
            for alias in node.names:
                if alias.name == PREDICATE:
                    bound.add(alias.asname or alias.name)
    if not bound:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in bound:
            return True
    return False


def _package_files() -> list[pathlib.Path]:
    """Every .py under src/neograph (recursive -- a subpackage must not escape)."""
    return sorted(p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC_DIR).as_posix()


# --- the guard --------------------------------------------------------------


class TestEachOracleFusionPredicateMonopoly:
    """neograph-c265k: "is this the fused Each x Oracle node?" is answered by
    exactly ONE named predicate (``is_each_oracle_fused`` in modifiers.py), a
    modifier-PRESENCE read, and every consumer CALLS it -- in either polarity.
    Questions genuinely about single-modifier presence (``"oracle" in mods``
    on its own, ``has_any_oracle``, a first-hit label chain) are NOT fusion
    tests and are never flagged.
    """

    # --- (a) no open-coded fusion test outside the owner --------------------

    def test_no_open_coded_fusion_test_outside_the_owner(self):
        offenders: list[str] = []
        for path in _package_files():
            if path.name == PREDICATE_OWNER and path.parent == SRC_DIR:
                continue
            for lineno, rule in _fusion_test_sites(path.read_text()):
                offenders.append(f"{_rel(path)}:{lineno}\t{rule}")
        assert not offenders, (
            "Open-coded Each x Oracle fusion test(s) found outside modifiers.py.\n"
            f"Call the single predicate instead: `{PREDICATE}(mods)` "
            f"(or `not {PREDICATE}(mods)` for the negated polarity). Do NOT add a second "
            "name such as `is_plain_each`, and do NOT re-derive the question from "
            "ModifierCombo members -- it is a modifier-PRESENCE read (neograph-c265k).\n"
            "A genuinely single-modifier presence question (`has_any_oracle`, a first-hit "
            "label chain) is not this disease and is not matched by these rules.\n" + "\n".join(offenders)
        )

    # --- (b) the owner IS flagged (anti-dead-scanner) -----------------------

    def test_the_owner_is_flagged_exactly_once(self):
        """The predicate body itself must trip R1.

        Without this, a scanner that silently matched NOTHING would satisfy
        assertion (a) vacuously forever and the guard would be decorative.
        """
        sites = _fusion_test_sites((SRC_DIR / PREDICATE_OWNER).read_text())
        assert [rule for _lineno, rule in sites] == ["R1"], (
            f"{PREDICATE_OWNER} must contain EXACTLY ONE open-coded fusion test -- the body of "
            f"`{PREDICATE}` -- and it must be caught by R1. Anything else means either the "
            "predicate is missing/renamed (the scanner is now dead and assertion (a) passes "
            f"vacuously) or a SECOND spelling grew inside the owner. Found: {sites}"
        )

    # --- (c) consumer inventory / anti-tautology ----------------------------

    def test_predicate_consumers_are_exactly_the_declared_inventory(self):
        """Filesystem-derived census == the hand-written ``FUSION_READERS``.

        The two sides come from independent sources, so this cannot pass
        tautologically. RATCHET IN BOTH DIRECTIONS: a new consumer is ADDED to
        the literal; a consumer that legitimately disappears (Phase 7 may drop
        ``_agent_spec.py``'s check) is REMOVED from it in the same commit.
        """
        actual = {_rel(p) for p in _package_files() if _uses_predicate(p.read_text())}
        expected = set(FUSION_READERS)
        assert actual == expected, (
            f"The set of src/neograph files that import-and-use `{PREDICATE}` diverged "
            "from the declared FUSION_READERS inventory.\n"
            f"  undeclared (new consumer -- add it to FUSION_READERS): {sorted(actual - expected)}\n"
            "  declared but gone (either the caller re-inlined the test -- assertion (a) will "
            "also be red -- or the consumer legitimately disappeared, in which case SHRINK "
            f"the literal): {sorted(expected - actual)}"
        )

    def test_owner_is_not_listed_as_a_consumer(self):
        """modifiers.py DEFINES the predicate; it is not one of its readers."""
        assert PREDICATE_OWNER not in FUSION_READERS
        assert FUSION_READERS <= {p.name for p in _package_files()}


class TestR1CoPresenceScannerMetaTests:
    """Positive + negative meta-tests for R1 (expression-level co-presence)."""

    @staticmethod
    def _hits(src: str) -> set[int]:
        tree = ast.parse(src)
        return _r1_sites(tree, _primary_shape_bindings(tree))

    def test_meta_flags_get_is_not_none_two_half_form(self):
        """The ``compiler.py`` spelling: both halves, pre-``match``."""
        src = (
            "def f(mods):\n    if mods.get('each') is not None and mods.get('oracle') is not None:\n        return 1\n"
        )
        assert self._hits(src) == {2}

    def test_meta_flags_membership_two_half_form(self):
        src = "def f(mods):\n    return 'each' in mods and 'oracle' in mods\n"
        assert self._hits(src) == {2}

    def test_meta_flags_negated_polarity_against_each_shape(self):
        """The ``_subconstruct.py`` spelling: NEGATED, guarded by PrimaryShape.EACH."""
        src = (
            "from neograph.modifiers import PrimaryShape\n"
            "def f(sub_shape, sub_mods):\n"
            "    return sub_shape is PrimaryShape.EACH and 'oracle' not in sub_mods\n"
        )
        assert self._hits(src) == {3}

    def test_meta_flags_aliased_primary_shape_binding(self):
        src = (
            "from neograph.modifiers import PrimaryShape as _PS\n"
            "def f(shape, mods):\n"
            "    return shape is _PS.EACH and 'oracle' in mods\n"
        )
        assert self._hits(src) == {3}

    def test_meta_ignores_bare_single_modifier_presence_read(self):
        """Negative: ``"oracle" in mods`` alone is a presence read, not a fusion test."""
        src = "def f(mods):\n    if 'oracle' in mods:\n        return 1\n    return 'each' in mods\n"
        assert self._hits(src) == set()

    def test_meta_ignores_co_presence_with_a_non_each_second_operand(self):
        """Negative: ``_fan_agent.py``'s Oracle-over-fan-out gate -- a genuine
        co-presence conjunction whose second operand is ``fan_out_param``, not
        the Each modifier. The sharpest false positive in the tree."""
        src = "def f(mods, item):\n    return 'oracle' in mods and item.fan_out_param is not None\n"
        assert self._hits(src) == set()

    def test_meta_ignores_multi_modifier_has_any_disjunction(self):
        """Negative: ``testing/scaffold.py``'s serialized "has ANY modifier"
        filter. This is what the ``<= {each, oracle}`` subset clause exists for."""
        src = "def f(n):\n    return n.get('oracle') or n.get('each') or n.get('loop') or n.get('operator')\n"
        assert self._hits(src) == set()

    def test_meta_ignores_oracle_paired_with_input_arity(self):
        """Negative: ``_fan_agent.py``'s Oracle-multi-input capability check."""
        src = "def f(ni, mods):\n    return ni.is_dict_form and len(ni.by_name) > 1 and 'oracle' not in mods\n"
        assert self._hits(src) == set()

    def test_meta_ignores_each_without_oracle(self):
        src = "def f(mods, ni):\n    return 'each' in mods and ni.is_none\n"
        assert self._hits(src) == set()


class TestR2MatchCaseScannerMetaTests:
    """Positive + negative meta-tests for R2 (context-level, inside an EACH arm)."""

    @staticmethod
    def _hits(src: str) -> set[int]:
        tree = ast.parse(src)
        return _r2_sites(tree, _primary_shape_bindings(tree))

    def test_meta_flags_membership_test_in_the_case_guard(self):
        """The ``state.py`` dict-form spelling: the test IS the case guard."""
        src = (
            "from neograph.modifiers import COMBO_DECOMPOSITION, PrimaryShape\n"
            "def f(combo, mods):\n"
            "    match COMBO_DECOMPOSITION[combo].primary:\n"
            "        case PrimaryShape.EACH if 'oracle' in mods:\n"
            "            return 1\n"
            "        case _:\n"
            "            return 0\n"
        )
        assert self._hits(src) == {4}

    def test_meta_flags_membership_test_in_the_case_body(self):
        """The ``state.py`` single-type and ``_state_write.py`` spellings."""
        src = (
            "from neograph.modifiers import PrimaryShape\n"
            "def f(shape, mods):\n"
            "    match shape:\n"
            "        case PrimaryShape.EACH:\n"
            "            each_mod = None if 'oracle' in mods else mods['each']\n"
            "            return each_mod\n"
        )
        assert self._hits(src) == {5}

    def test_meta_flags_negated_membership_in_the_case_body(self):
        src = (
            "from neograph.modifiers import PrimaryShape\n"
            "def f(shape, mods):\n"
            "    match shape:\n"
            "        case PrimaryShape.EACH:\n"
            "            if 'oracle' not in mods:\n"
            "                return 1\n"
            "    return 0\n"
        )
        assert self._hits(src) == {5}

    def test_meta_flags_get_spelling_inside_an_each_arm(self):
        """R-RC2: the ``.get`` spelling is precisely what this ticket removes
        from ``compiler.py``, so a re-inline of it INSIDE an EACH arm must not
        walk through R2 just because R1's co-presence clause does not apply."""
        src = (
            "from neograph.modifiers import PrimaryShape\n"
            "def f(shape, mods):\n"
            "    match shape:\n"
            "        case PrimaryShape.EACH:\n"
            "            if mods.get('oracle') is not None:\n"
            "                return 1\n"
            "    return 0\n"
        )
        assert self._hits(src) == {5}

    def test_meta_flags_test_inside_a_raise_condition_disjunction(self):
        """The ``_agent_spec.py`` spelling: buried in a raise-condition ``or``.
        R1 cannot see it (the second operand is ``has_operator``), so R2 is the
        only rule that catches site 6."""
        src = (
            "from neograph.modifiers import PrimaryShape\n"
            "def f(shape, decomp, mods):\n"
            "    match shape:\n"
            "        case PrimaryShape.EACH:\n"
            "            if decomp.has_operator or 'oracle' in mods:\n"
            "                raise ValueError('no lowering')\n"
            "    return 0\n"
        )
        assert self._hits(src) == {5}

    def test_meta_flags_test_inside_an_or_pattern_naming_each(self):
        """A ``case A | PrimaryShape.EACH | B:`` arm still establishes the Each
        half for the branches that reach it."""
        src = (
            "from neograph.modifiers import PrimaryShape\n"
            "def f(shape, mods):\n"
            "    match shape:\n"
            "        case PrimaryShape.BARE | PrimaryShape.EACH:\n"
            "            if 'oracle' in mods:\n"
            "                return 1\n"
            "    return 0\n"
        )
        assert self._hits(src) == {5}

    def test_meta_ignores_oracle_test_in_a_non_each_arm(self):
        src = (
            "from neograph.modifiers import PrimaryShape\n"
            "def f(shape, mods):\n"
            "    match shape:\n"
            "        case PrimaryShape.ORACLE:\n"
            "            if 'oracle' in mods:\n"
            "                return 1\n"
            "    return 0\n"
        )
        assert self._hits(src) == set()

    def test_meta_ignores_oracle_test_outside_any_match(self):
        """Negative: ``state.py``'s construct-wide ``has_any_oracle`` scan -- a
        bare ``if`` in a plain ``for`` loop, no match statement in sight."""
        src = (
            "def f(items, classify):\n"
            "    has_any_oracle = False\n"
            "    for item in items:\n"
            "        _combo, item_mods = classify(item)\n"
            "        if 'oracle' in item_mods:\n"
            "            has_any_oracle = True\n"
            "    return has_any_oracle\n"
        )
        assert self._hits(src) == set()

    def test_meta_ignores_each_only_test_in_an_each_arm(self):
        """R2 keys on the ``oracle`` half; the ``each`` half is contextual."""
        src = (
            "from neograph.modifiers import PrimaryShape\n"
            "def f(shape, mods):\n"
            "    match shape:\n"
            "        case PrimaryShape.EACH:\n"
            "            return mods['each'] if 'each' in mods else None\n"
        )
        assert self._hits(src) == set()


class TestR3SlotAttributeScannerMetaTests:
    """Positive + negative meta-tests for R3 -- the pre-emptive slot-attribute
    ratchet. R3 has ZERO real hits today, so THESE tests are the only proof it
    is a live scanner rather than a decorative one."""

    @staticmethod
    def _hits(src: str) -> set[int]:
        return _r3_sites(ast.parse(src))

    def test_meta_flags_modifier_set_slot_co_presence(self):
        src = "def f(ms):\n    return ms.each is not None and ms.oracle is not None\n"
        assert self._hits(src) == {2}

    def test_meta_flags_slot_co_presence_in_reverse_order(self):
        src = "def f(node):\n    if node.oracle is not None and node.each is not None:\n        return 1\n"
        assert self._hits(src) == {2}

    def test_meta_flags_slot_co_presence_through_a_nested_attribute_chain(self):
        src = "def f(item):\n    return item.modifier_set.each is not None and item.modifier_set.oracle is not None\n"
        assert self._hits(src) == {2}

    def test_meta_ignores_each_plus_loop_exclusion(self):
        """Negative: ``ModifierSet.model_post_init``'s Each+Loop pairwise exclude."""
        src = "def f(self):\n    if self.each is not None and self.loop is not None:\n        raise ValueError('x')\n"
        assert self._hits(src) == set()

    def test_meta_ignores_portal_pairwise_excludes(self):
        src = (
            "def f(self):\n"
            "    if self.portal is not None and self.each is not None:\n"
            "        raise ValueError('a')\n"
            "    if self.portal is not None and self.oracle is not None:\n"
            "        raise ValueError('b')\n"
            "    if self.oracle is not None and self.loop is not None:\n"
            "        raise ValueError('c')\n"
        )
        assert self._hits(src) == set()

    def test_meta_ignores_none_guard_plus_single_slot_read(self):
        """Negative: ``_validation_inputs.py``'s ``has_each`` -- SAME variable
        name as the negated fusion site, but only ONE modifier slot is read."""
        src = "def f(ms):\n    has_each = ms is not None and ms.each is not None\n    return has_each\n"
        assert self._hits(src) == set()

    def test_meta_ignores_three_slot_conjunction(self):
        src = "def f(ms):\n    return ms.each is not None and ms.oracle is not None and ms.operator is not None\n"
        assert self._hits(src) == set()

    def test_meta_ignores_is_none_polarity(self):
        """R3 is specified over ``is not None`` slot reads; an ``is None``
        conjunction asks the opposite (and no real site spells it)."""
        src = "def f(ms):\n    return ms.each is None and ms.oracle is None\n"
        assert self._hits(src) == set()

    def test_r3_has_no_hits_anywhere_in_the_package_today(self):
        """R3 is a RATCHET at its end stop: zero hits, and it must stay zero.

        Distinct from assertion (a): (a) is currently red on R1/R2 sites, so it
        cannot testify that R3 specifically is at zero. This one can, and it
        covers ``modifiers.py`` too (which (a) skips).
        """
        offenders = [
            f"{_rel(p)}:{lineno}" for p in _package_files() for lineno in sorted(_r3_sites(ast.parse(p.read_text())))
        ]
        assert not offenders, (
            "The ModifierSet-slot spelling of the Each x Oracle fusion test appeared "
            f"(`ms.each is not None and ms.oracle is not None`). Call `{PREDICATE}` on the "
            "`classify_modifiers` dict instead -- a Node/ModifierSet holder can get one via "
            "`classify_modifiers(item)[1]` (neograph-c265k R-RC3).\n" + "\n".join(offenders)
        )


class TestRealFileNegativeControls:
    """Real-file negative controls: the look-alike presence readers must stay
    unflagged by ALL THREE rules, before AND after the migration."""

    def test_fan_agent_presence_reads_are_never_flagged(self):
        """``_fan_agent.py`` holds FOUR look-alikes, including the sharpest one
        (``"oracle" in mods and item.fan_out_param is not None``)."""
        assert _fusion_test_sites((SRC_DIR / "_fan_agent.py").read_text()) == []

    def test_scaffold_serialized_modifier_filter_is_never_flagged(self):
        """``testing/scaffold.py``'s four-name "has ANY modifier" disjunction."""
        assert _fusion_test_sites((SRC_DIR / "testing" / "scaffold.py").read_text()) == []

    def test_state_has_any_oracle_scan_is_never_flagged(self):
        """``state.py``'s construct-wide ``has_any_oracle``/``has_any_each``
        presence scan must not be swept up by the fusion rules.

        Asserted line-independently (the file's own fusion sites ARE flagged
        until the migration lands, so a whole-file zero is not available yet):
        no flagged line may sit inside the ``for item in all_items:`` presence
        loop that sets those two flags.
        """
        source = (SRC_DIR / "state.py").read_text()
        tree = ast.parse(source)
        loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and any(
                isinstance(sub, ast.Name) and sub.id in {"has_any_oracle", "has_any_each"} for sub in ast.walk(node)
            )
        ]
        assert loops, "state.py no longer holds the has_any_oracle/has_any_each presence loop"
        flagged = {lineno for lineno, _rule in _fusion_test_sites(source)}
        for loop in loops:
            span = range(loop.lineno, (loop.end_lineno or loop.lineno) + 1)
            assert not (flagged & set(span)), (
                "state.py's has_any_oracle/has_any_each PRESENCE scan was flagged as a fusion "
                f"test. It is deliberately independent of Each (a fused node must set BOTH "
                f"flags) and must stay a presence read. Flagged: {sorted(flagged & set(span))}"
            )
