"""Structural guard: the Each x Oracle FUSION test has exactly ONE authority.

neograph-c265k, re-anchored by neograph-jtawq.2. The disease this guard exists
to kill is unchanged: a consumer that asks "is this item the fused Each x Oracle
node?" by HAND -- an ``"oracle"`` / ``"each"`` membership or ``.get(...)``
co-presence read, in either polarity, or a
``ms.each is not None and ms.oracle is not None`` slot read.

What CHANGED is the authority it must call instead. c265k answered the question
with a free-floating predicate ``is_each_oracle_fused(mods)`` over modifier
INSTANCES; jtawq.2 replaced it with the ``fused`` column on
``COMBO_DECOMPOSITION`` (``src/neograph/modifiers.py``), derived at definition
from ``modifier_names_for_combo``. Fusion is a fact about the COMBO -- EACH_ORACLE
is fused whichever Each and Oracle instances are attached -- and every one of the
six consumers already holds the combo (four already index the table on the same
or an adjacent line), while the one consumer that held no instances
(``loader.py``) had to forge a ``dict.fromkeys(names, True)`` argument to reach
the instance-shaped predicate at all. The predicate is DELETED; this file keeps
its scanners and re-points its inventory. (The filename still says "predicate"
for history's sake -- the authority it enforces is the column.)

Why the concept needs a second question at all: ``COMBO_DECOMPOSITION`` folds
``EACH_ORACLE`` / ``EACH_ORACLE_OPERATOR`` down to ``primary=EACH`` (the fusion
is a Node-level topology concern, and ``agent-spec-rewrite-2026-07-27.md`` states
that as a deliberate design decision), so a consumer standing in a
``PrimaryShape.EACH`` arm needs a SECOND, orthogonal test to split a fused node
from a plain Each one. Before c265k that second test was open-coded in six
places; after jtawq.2 it is one field read on a table those same six files
already hold.

Why the guard is context-aware rather than a spelling ban
--------------------------------------------------------
A naive "ban ``\"oracle\" in mods``" rule is UNSATISFIABLE: the tree holds EIGHT
legitimate look-alike presence reads that spell the same token and must never
move (``_fan_agent.py``'s first-hit label chain and its Oracle-over-fan-out /
Oracle-multi-input gates, ``state.py``'s construct-wide ``has_any_oracle`` /
``has_any_each`` scan, and ``testing/scaffold.py``'s serialized "has ANY
modifier" filter). So the disease is SYNTACTIC-WITH-CONTEXT and the scanners
encode exactly that context.

Three rules (unchanged from c265k -- they were, and remain, the substance)
-------------------------------------------------------------------------
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
  expressed. Covering the ``.get`` spelling is deliberate: a re-inline of
  ``mods.get("oracle") is not None`` inside an EACH arm must not walk through.

* **R3 -- slot-attribute spelling (pre-emptive, per R-RC3).** An ``ast.BoolOp``
  whose set of ``<expr>.<slot> is not None`` reads, restricted to the modifier
  vocabulary ``{each, oracle, loop, operator, portal}``, equals EXACTLY
  ``{each, oracle}``. R3 has ZERO hits on the tree and must stay at zero -- it is
  a pre-emptive RATCHET, not a migration check, because a consumer holding a
  ``Node``/``ModifierSet`` rather than a ``mods`` dict (``loader.py``,
  ``_validation_inputs.py``, ``_param_classify.py`` all already hold one) would
  reach for that spelling naturally, and such a reader evades BOTH assertion
  (a)'s dict-shaped rules AND assertion (c) (a file that never reads the column
  is on neither side of the equality). Its meta-tests -- not its offender count
  -- are what prove it is not a dead scanner. It spares
  ``ModifierSet.model_post_init``'s four pairwise excludes (each+loop,
  oracle+loop, portal+each, portal+oracle) and ``_validation_inputs.py``'s
  ``ms is not None and ms.each is not None`` by construction.

Five assertions
---------------
  (a) **No open-coded fusion test** anywhere under ``src/neograph``, with NO
      owner exemption. c265k exempted ``modifiers.py`` because the predicate body
      it owned was itself an R1 hit; the column's derivation
      (``frozenset({"each","oracle"}) <= modifier_names_for_combo(combo)``) is an
      ``ast.Compare``, not a ``BoolOp``, so there is no longer anything to
      exempt. Offenders are reported as ``file:line  rule``.
  (b) **Every scanner is live (anti-dead-scanner).** c265k proved liveness by
      requiring the owner file to yield exactly one R1 hit -- an assertion that
      cannot survive the predicate's deletion. Liveness is now proved the way
      c265k's own docstring already justified it for R3: against fixtures this
      guard owns, run through the same public ``_fusion_test_sites`` entry point
      the tree scan uses. A scanner that silently matched nothing would satisfy
      (a) vacuously forever, and this is what stops that.
  (c) **Consumer inventory (anti-tautology).** The hand-written ``FUSED_READERS``
      literal must EQUAL the filesystem-derived set of files that import
      ``COMBO_DECOMPOSITION`` AND read a ``.fused`` field off it. A seventh
      consumer -- or a caller that drops the column read and re-inlines the test
      -- fails loud. This equality is a RATCHET in BOTH directions: when a
      consumer legitimately disappears, SHRINK the literal in the same commit.
      Do not relax the equality into a subset test -- the same instruction the
      sibling guard's ``PENDING`` ratchet carries.
  (d) **The deleted predicate does not come back.** ``is_each_oracle_fused`` must
      not be defined, imported, or referenced anywhere under ``src/neograph``. A
      thin table-backed wrapper is the one shape jtawq.2 explicitly rejected: it
      restores two authorities for one fact while evading R1 (an attribute read
      is not a ``BoolOp``), so assertion (a) alone would not catch it.
  (e) Scanner meta-tests, positive AND negative, for all three rules, including
      REAL-FILE negative controls (``_fan_agent.py``, ``testing/scaffold.py``,
      and ``state.py``'s ``has_any_oracle`` lines).

The ``fused`` field needs no entry in ``TABLE_SYMBOLS``
(``tests/test_guards_combo_decomposition_consumers.py``): it is a field on
``COMBO_DECOMPOSITION``, a symbol that census already tracks. The exhaustive
true/false partition of the column against an INDEPENDENT oracle (the enum name)
lives in ``tests/test_combo_decomposition.py`` (G-FUSE), which is the
table-contract home; this file governs only how consumers ASK the question.

Written in pure ``ast`` with no ``re``, so it is exempt by construction from
``tests/test_guards_meta.py``'s named-regex/slip-test discipline -- the same
move ``tests/test_guards_combo_decomposition_consumers.py`` makes.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# --- independent literals (hand-written; never derived from the scan) --------

#: The file that OWNS the fusion answer -- it defines the table and the column.
COLUMN_OWNER = "modifiers.py"

#: The decomposition table, and the field on it that answers the fusion question.
TABLE = "COMBO_DECOMPOSITION"
FUSED_FIELD = "fused"

#: The predicate jtawq.2 deleted. Named here ONLY so assertion (d) can keep it
#: out of the tree; nothing in src/neograph may define, import, or call it.
DELETED_PREDICATE = "is_each_oracle_fused"

#: Every file that must READ the ``fused`` column instead of open-coding the
#: test. Hand-written and sourced from the neograph-c265k census (carried over
#: unchanged by jtawq.2's migration -- the same six files, a different spelling),
#: independently of the filesystem scan assertion (c) compares it against.
#: RATCHET IN BOTH DIRECTIONS: a new consumer must be added here; a consumer that
#: legitimately disappears must be REMOVED here in the same commit.
FUSED_READERS: frozenset[str] = frozenset(
    {
        "compiler.py",  # the pre-`match` fusion split (M x N Send topology)
        "state.py",  # dict-form fused arm + single-type collector
        "_state_write.py",  # Each key-wrapping suppression for the fusion
        "_subconstruct.py",  # NEGATED: EACH-shaped but not fused
        # neograph-qtfof.13: RE-KEYED from _agent_spec.py -- the fused-column read
        # moved with _lower_construct_item, it did not spread to a new consumer.
        "_agent_spec_item_dispatch.py",  # EXPORT: the pre-`match` fusion split (MapNode over an Oracle subflow)
        "loader.py",  # IMPORT: the mirror fusion split
    }
)

#: The modifier slot vocabulary R3 restricts itself to.
MODIFIER_VOCAB: frozenset[str] = frozenset({"each", "oracle", "loop", "operator", "portal"})

#: The two names whose CO-PRESENCE is the fusion question.
FUSION_NAMES: frozenset[str] = frozenset({"each", "oracle"})

#: One synthetic module carrying exactly one instance of EACH rule, in the
#: spelling each rule was written for. Assertion (b) runs it through the public
#: scanner entry point so liveness is proved against fixtures this guard owns
#: rather than against a production body that must no longer exist.
LIVENESS_FIXTURE = (
    "from neograph.modifiers import PrimaryShape\n"
    "def r1(mods):\n"
    "    return mods.get('each') is not None and mods.get('oracle') is not None\n"
    "def r2(shape, mods):\n"
    "    match shape:\n"
    "        case PrimaryShape.EACH:\n"
    "            if 'oracle' in mods:\n"
    "                return 1\n"
    "    return 0\n"
    "def r3(ms):\n"
    "    return ms.each is not None and ms.oracle is not None\n"
)


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


def _reads_fused_column(source: str) -> bool:
    """True when the module imports ``COMBO_DECOMPOSITION`` from
    ``neograph.modifiers`` AND reads a ``.fused`` field.

    Both halves are required, for the reason the deleted ``_uses_predicate``
    demanded both: a dead import must not satisfy assertion (c), and a ``.fused``
    read in a module that never reaches the table is not a table read.
    """
    tree = ast.parse(source)
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] == "modifiers":
            for alias in node.names:
                if alias.name == TABLE:
                    bound.add(alias.asname or alias.name)
    if not bound:
        return False
    return any(
        isinstance(node, ast.Attribute) and node.attr == FUSED_FIELD and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )


def _references(source: str, name: str) -> bool:
    """True when ``name`` is defined, imported, or referenced as an identifier.

    Deliberately identifier-level, not textual: a docstring or comment may cite
    the deleted predicate as history, and doing so is not a resurrection.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(alias.name == name or alias.asname == name for alias in node.names):
                return True
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.Attribute) and node.attr == name:
            return True
    return False


def _package_files() -> list[pathlib.Path]:
    """Every .py under src/neograph (recursive -- a subpackage must not escape)."""
    return sorted(p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC_DIR).as_posix()


# --- the guard --------------------------------------------------------------


class TestEachOracleFusionColumnMonopoly:
    """neograph-jtawq.2: "is this the fused Each x Oracle node?" is answered by
    exactly ONE authority (the ``fused`` column on ``COMBO_DECOMPOSITION``), and
    every consumer READS it -- in either polarity. Questions genuinely about
    single-modifier presence (``"oracle" in mods`` on its own, ``has_any_oracle``,
    a first-hit label chain) are NOT fusion tests and are never flagged.
    """

    # --- (a) no open-coded fusion test anywhere -----------------------------

    def test_no_open_coded_fusion_test_anywhere_in_the_package(self):
        offenders: list[str] = []
        for path in _package_files():
            for lineno, rule in _fusion_test_sites(path.read_text()):
                offenders.append(f"{_rel(path)}:{lineno}\t{rule}")
        assert not offenders, (
            "Open-coded Each x Oracle fusion test(s) found under src/neograph.\n"
            f"Read the single authority instead: `{TABLE}[combo].{FUSED_FIELD}` "
            f"(or `not ...{FUSED_FIELD}` for the negated polarity) -- every consumer already "
            "holds the combo. Do NOT add a second name such as `is_plain_each`, do NOT "
            f"reintroduce a `{DELETED_PREDICATE}`-style predicate over modifier instances, and "
            "do NOT re-derive the question from ModifierCombo members (neograph-jtawq.2).\n"
            "There is NO owner exemption: the column's own derivation is a set compare, not a "
            "BoolOp, so modifiers.py is held to the same rule as every consumer.\n"
            "A genuinely single-modifier presence question (`has_any_oracle`, a first-hit "
            "label chain) is not this disease and is not matched by these rules.\n" + "\n".join(offenders)
        )

    # --- (b) every scanner is live (anti-dead-scanner) ----------------------

    def test_every_scanner_is_live_on_the_owned_fixture(self):
        """All three rules fire, through the same entry point the tree scan uses.

        Without this, a scanner that silently matched NOTHING would satisfy
        assertion (a) vacuously forever and the guard would be decorative. c265k
        proved this by requiring the predicate body to trip R1; with the
        predicate deleted the proof moves onto fixtures this guard owns -- the
        precedent c265k's own docstring set for R3, whose offender count was
        always zero.
        """
        sites = _fusion_test_sites(LIVENESS_FIXTURE)
        assert [rule for _lineno, rule in sites] == ["R1", "R2", "R3"], (
            "The liveness fixture must trip R1, R2 and R3 exactly once each, in source order. "
            "A missing rule means that scanner is dead and assertion (a) is passing vacuously "
            f"for it. Found: {sites}"
        )

    # --- (c) consumer inventory / anti-tautology ----------------------------

    def test_fused_column_readers_are_exactly_the_declared_inventory(self):
        """Filesystem-derived census == the hand-written ``FUSED_READERS``.

        The two sides come from independent sources, so this cannot pass
        tautologically. RATCHET IN BOTH DIRECTIONS: a new consumer is ADDED to
        the literal; a consumer that legitimately disappears is REMOVED from it
        in the same commit.
        """
        actual = {_rel(p) for p in _package_files() if _reads_fused_column(p.read_text())}
        expected = set(FUSED_READERS)
        assert actual == expected, (
            f"The set of src/neograph files that read `{TABLE}[...].{FUSED_FIELD}` diverged "
            "from the declared FUSED_READERS inventory.\n"
            f"  undeclared (new consumer -- add it to FUSED_READERS): {sorted(actual - expected)}\n"
            "  declared but gone (either the caller re-inlined the test -- assertion (a) will "
            "also be red -- or the consumer legitimately disappeared, in which case SHRINK "
            f"the literal): {sorted(expected - actual)}"
        )

    def test_the_column_owner_is_not_listed_as_a_reader(self):
        """modifiers.py DEFINES the column; it is not one of its readers.

        In particular ``SUB_CONSTRUCT_UNSUPPORTED_COMBOS`` stays hand-written and
        is NOT derived from ``fused`` -- "unsupported on a Construct item" and
        "fused" are different concepts that coincide today, pinned as an
        intentional coincidence in tests/test_combo_decomposition.py.
        """
        assert COLUMN_OWNER not in FUSED_READERS
        assert FUSED_READERS <= {p.name for p in _package_files()}

    # --- (d) the deleted predicate stays deleted ----------------------------

    def test_the_deleted_instance_level_predicate_does_not_come_back(self):
        """``is_each_oracle_fused`` is gone and must not be reintroduced.

        Including as a thin table-backed wrapper: that shape restores two
        authorities for one fact while evading R1 (an attribute read is not a
        ``BoolOp``), so assertion (a) alone would not catch it. It is also what
        forced ``loader.py`` to counterfeit a ``dict.fromkeys(names, True)``
        argument -- the evidence the instance-level signature was wrong.
        """
        offenders = [_rel(p) for p in _package_files() if _references(p.read_text(), DELETED_PREDICATE)]
        assert not offenders, (
            f"`{DELETED_PREDICATE}` reappeared under src/neograph. Fusion is a fact about the "
            f"COMBO, not about which modifier instances are attached -- read "
            f"`{TABLE}[combo].{FUSED_FIELD}`. A thin table-backed wrapper is NOT an acceptable "
            "compromise: it is the two-authorities shape neograph-jtawq.2 deleted.\n" + "\n".join(offenders)
        )


class TestR1CoPresenceScannerMetaTests:
    """Positive + negative meta-tests for R1 (expression-level co-presence)."""

    @staticmethod
    def _hits(src: str) -> set[int]:
        tree = ast.parse(src)
        return _r1_sites(tree, _primary_shape_bindings(tree))

    def test_meta_flags_get_is_not_none_two_half_form(self):
        """The pre-migration ``compiler.py`` spelling: both halves, pre-``match``."""
        src = (
            "def f(mods):\n    if mods.get('each') is not None and mods.get('oracle') is not None:\n        return 1\n"
        )
        assert self._hits(src) == {2}

    def test_meta_flags_membership_two_half_form(self):
        src = "def f(mods):\n    return 'each' in mods and 'oracle' in mods\n"
        assert self._hits(src) == {2}

    def test_meta_flags_negated_polarity_against_each_shape(self):
        """The pre-migration ``_subconstruct.py`` spelling: NEGATED, guarded by
        ``PrimaryShape.EACH``."""
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

    def test_meta_ignores_the_migrated_column_read(self):
        """Negative, and the point of the whole migration: the spelling every
        consumer moves TO must not be flagged by the rule it replaces."""
        src = (
            "from neograph.modifiers import COMBO_DECOMPOSITION, PrimaryShape\n"
            "def f(sub_shape, sub_combo):\n"
            "    return sub_shape is PrimaryShape.EACH and not COMBO_DECOMPOSITION[sub_combo].fused\n"
        )
        assert self._hits(src) == set()


class TestR2MatchCaseScannerMetaTests:
    """Positive + negative meta-tests for R2 (context-level, inside an EACH arm)."""

    @staticmethod
    def _hits(src: str) -> set[int]:
        tree = ast.parse(src)
        return _r2_sites(tree, _primary_shape_bindings(tree))

    def test_meta_flags_membership_test_in_the_case_guard(self):
        """The pre-migration ``state.py`` dict-form spelling: the test IS the case guard."""
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
        """The pre-migration ``state.py`` single-type and ``_state_write.py`` spellings."""
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
        """R-RC2: a re-inline of ``mods.get('oracle') is not None`` INSIDE an EACH
        arm must not walk through R2 just because R1's co-presence clause does
        not apply."""
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
        """The pre-migration ``_agent_spec.py`` spelling: buried in a raise-condition
        ``or``. R1 cannot see it (the second operand is ``has_operator``), so R2 is
        the only rule that catches that site."""
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

    def test_meta_ignores_the_migrated_column_read_in_a_case_guard(self):
        """Negative: the line-neutral spelling ``state.py`` migrates its case
        guard to. R2 keys on an ``"oracle"`` string test; a field read is not one."""
        src = (
            "from neograph.modifiers import COMBO_DECOMPOSITION, PrimaryShape\n"
            "def f(combo, mods):\n"
            "    match COMBO_DECOMPOSITION[combo].primary:\n"
            "        case PrimaryShape.EACH if COMBO_DECOMPOSITION[combo].fused:\n"
            "            return 1\n"
            "        case _:\n"
            "            return 0\n"
        )
        assert self._hits(src) == set()


class TestR3SlotAttributeScannerMetaTests:
    """Positive + negative meta-tests for R3 -- the pre-emptive slot-attribute
    ratchet. R3 has ZERO real hits, so THESE tests are the only proof it is a
    live scanner rather than a decorative one."""

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

        Kept as its own assertion even though (a) is now whole-tree: it names R3
        specifically, so it stays diagnostic when (a) is red for some other rule.
        """
        offenders = [
            f"{_rel(p)}:{lineno}" for p in _package_files() for lineno in sorted(_r3_sites(ast.parse(p.read_text())))
        ]
        assert not offenders, (
            "The ModifierSet-slot spelling of the Each x Oracle fusion test appeared "
            f"(`ms.each is not None and ms.oracle is not None`). Read `{TABLE}[combo]."
            f"{FUSED_FIELD}` instead -- a Node/ModifierSet holder can get the combo via "
            "`classify_modifiers(item)[0]` (neograph-c265k R-RC3, neograph-jtawq.2).\n" + "\n".join(offenders)
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

        Asserted line-independently rather than as a whole-file zero, so it stays
        a targeted control on THAT loop: it is deliberately independent of Each
        (a fused node must set BOTH flags) and must stay a presence read.
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
