"""Structural guard: `ModifierCombo` decomposition has exactly ONE authority.

neograph-s7zt3.7 (Phase 3 of the Agent Spec / Portal rebuild epic). The disease
this guard exists to kill: a build-time consumer that RE-DERIVES "what does this
ModifierCombo decompose into" with a hand-written enumeration of combo MEMBERS
(a ``match combo:`` over ``ModifierCombo.X`` patterns, an ``==``/``!=`` compare,
or an ``in``/``not in`` against a tuple/set/dict literal of members) instead of
READING the single-source table in ``src/neograph/modifiers.py``:
``COMBO_DECOMPOSITION`` / ``PrimaryShape`` / ``SUB_CONSTRUCT_UNSUPPORTED_COMBOS``
(and the ``primary_shape`` helper that fronts them).

That duplication is exactly the anti-pattern AGENTS.md records for the Portal
rollout: seven files each grew their own combo-membership check, so adding a
combo meant editing seven places or silently diverging.

Three assertions, per the ticket's step 2:

  (a) **No second enumeration.** No file in ``MIGRATED`` may name a
      ``ModifierCombo`` MEMBER in a DISPATCH context.
  (b) **Reads the table.** Each ``MIGRATED`` file imports at least one table
      symbol from ``neograph.modifiers`` AND actually uses the binding (a dead
      import must not satisfy the guard -- R-L3). (b) is a cheap companion to
      (a), not the load-bearing half: (a) is what forbids regrowth, (b) only
      confirms the file joined the table's readership.
  (c) **Completeness / anti-tautology.** The set of package files that touch the
      combo vocabulary at all -- derived from the FILESYSTEM -- must equal
      ``MIGRATED | PENDING | {"modifiers.py"}``, two hand-written literals
      sourced independently of the scan (the anti-tautology lesson from
      ``tests/test_guards_parity_ratchet.py``). A brand-new file growing combo
      dispatch fails loud here even if it is not in ``MIGRATED``.

      The census counts a file when it references ``ModifierCombo`` **or**
      imports a table symbol. That union is what makes (c) stable ACROSS the
      migration: pre-migration the seven MIGRATED files qualify via
      ``ModifierCombo``; post-migration their combo imports are gone (ruff F401)
      and they qualify via the table symbols instead. Accepted tradeoff: any
      FUTURE file that legitimately reads the table must be added to
      ``MIGRATED``. That is the intended ratchet -- this guard IS the consumer
      inventory -- not friction to route around.

Scope note (deliberate, do NOT read this guard as total): the scanners are pure
``ast`` and match *member-constant* dispatch. A combo question asked as a STRING
compare (``combo.name != "BARE"``, ``compiler.py:174-176`` -- observability, not
decomposition) is invisible to them and knowingly allowlisted, as is a
``modifiers.ModifierCombo.X`` fully-qualified attribute chain. Alias-tolerance IS
covered: local binding names are collected from the module's ``ImportFrom`` nodes,
so ``from neograph.modifiers import ModifierCombo as MC`` + ``MC.EACH`` is caught
(R-M4).

Written in pure ``ast`` with no ``re``, so it is exempt by construction from
``tests/test_guards_meta.py``'s named-regex/slip-test discipline -- the same move
``TestComboMapMonopoly`` (``tests/test_guards_helper_monopoly.py``) makes.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# --- independent literals (hand-written; never derived from the scan) --------

#: Files that must READ the decomposition table and hold ZERO hand-written
#: member dispatch. compiler.py is here because Phase 2 migrated only its two
#: `match` statements and left the compare-shaped mesh-entry site at :263-265
#: behind (amendment A1) -- so Phase 3 touches SEVEN files, not six.
MIGRATED: frozenset[str] = frozenset(
    {
        "compiler.py",
        "state.py",
        "_state_write.py",
        "_subconstruct.py",
        "_input_shape.py",
        # neograph-3ffdg.9: runner.py's combo-vocabulary use (the Portal-mesh hop
        # costing that reads primary_shape/PrimaryShape) moved wholesale into
        # _recursion_budget.py as a pure file split. runner.py is REMOVED because
        # it no longer imports or references any table symbol -- assertion (a)
        # requires live use, not a dead import -- and the new module is declared in
        # its place. The consumer inventory tracks where the vocabulary actually
        # lives, so it follows the code.
        "_recursion_budget.py",
        "_wiring.py",
        # neograph-tjpn4: MOVED here from PENDING, not merely dropped -- assertion
        # (c) is an EQUALITY and _agent_spec.py is still a combo-vocabulary
        # consumer via its table imports.
        "_agent_spec.py",
        # neograph-s7zt3.10 (Phase 7): loader.py joined the readership when its
        # recognition walk moved to recognize->classify -- it now maps a recognized
        # modifier-NAME frozenset through combo_for_modifier_names and dispatches on
        # COMBO_DECOMPOSITION[combo].primary/.has_operator, the exact import-side
        # mirror of _agent_spec.py's export dispatch. It joins MIGRATED directly and
        # never passed through PENDING, exactly as the PENDING docstring pre-declared.
        "loader.py",
        # neograph-jtawq.4 Phase 1: the @node decorator's ModifierCombo-keyed
        # kwarg registry. Reads _COMBO_MAP's classification via
        # combo_for_modifier_names (derive_combo) / modifier_names_for_combo
        # (valid_kwargs) -- zero hand-written member dispatch, the same shape
        # every other MIGRATED file has for the decomposition table.
        "_node_modifier_kwargs.py",
        # neograph-jtawq.4 Phase 2: decorator(f)'s dispatch reads
        # modifier_names_for_combo(combo) directly to get the `members` set
        # its 5 membership checks test against -- zero hand-written member
        # dispatch (RULE 1 in test_guards_modifier_composition_completeness.py
        # pins that shape independently).
        "decorators.py",
        # neograph-ftnxl.1: the Agent Spec conformance classifier's structural
        # walker reads classify_modifiers(item) -> modifier_names_for_combo(combo)
        # to detect Each/Loop/Operator presence -- zero hand-written combo dispatch,
        # joins MIGRATED directly (never PENDING).
        "_agent_spec_conformance.py",
        # neograph-qtfof.9: the EndNode boundary-wiring helper reads
        # primary_shape(item) to scope terminal-producer wiring to BARE/ORACLE
        # shapes (Each/Loop/Portal fall back, per its own documented scope
        # boundary) -- reads the canonical classifier directly, zero
        # hand-written member dispatch, joins MIGRATED directly (never PENDING).
        "_agent_spec_boundary.py",
        # neograph-ftnxl.19: the branch-arm modifier guard
        # (_check_no_modifier_in_branch_arm) reads
        # COMBO_DECOMPOSITION[combo].primary/.has_operator to decide whether an
        # arm item carries ANY modifier, and modifier_names_for_combo(combo) to
        # name the offenders in the error. Deriving the predicate from the table
        # is what makes the guard TOTAL over future PrimaryShape values -- a
        # hand-written member list would silently let a new shape become
        # arm-legal, the exact regrowth this inventory forbids. Joins MIGRATED
        # directly (never PENDING).
        "_validation_arms.py",
    }
)

#: Known-diseased files whose migration is sequenced LATER, each with a ticket.
#: This set is a RATCHET: it may only shrink, and it is now EMPTY -- neograph-tjpn4
#: migrated the last one (_agent_spec.py's flat `if combo == ModifierCombo.X:`
#: chain + PORTAL membership test), so every combo consumer in the tree reads the
#: decomposition table. Re-parking a file here is forbidden: new combo dispatch
#: must be written against the table, never deferred.
#: `loader.py` was pre-declared here as "joins MIGRATED (never PENDING) when the
#: s6 recognize->classify design lands" -- that landed in neograph-s7zt3.10 and it
#: is now in MIGRATED above. PENDING stays EMPTY.
PENDING: frozenset[str] = frozenset()

#: The single definition site. Scoped out of (a)/(b) by construction.
TABLE_OWNER = "modifiers.py"

#: The symbols that ARE the single source of truth for combo decomposition
#: AND classification. ``combo_for_modifier_names`` / ``modifier_names_for_combo``
#: (neograph-jtawq.4) are the sanctioned readers of ``_COMBO_MAP`` -- the sibling
#: single-source-of-truth for combo *classification* (name-set -> combo), the
#: complement of ``COMBO_DECOMPOSITION``'s *meaning* (combo -> primary shape).
#: They belong in this set for the same reason the decomposition symbols do: a
#: consumer that imports-and-uses one of them is reading the table, not
#: re-deriving it, and (b) below must recognize that as legitimate readership.
TABLE_SYMBOLS: frozenset[str] = frozenset(
    {
        "COMBO_DECOMPOSITION",
        "PrimaryShape",
        "SUB_CONSTRUCT_UNSUPPORTED_COMBOS",
        "primary_shape",
        "combo_for_modifier_names",
        "modifier_names_for_combo",
    }
)


# --- scanners (pure ast) ----------------------------------------------------


def _combo_binding_names(tree: ast.Module) -> set[str]:
    """Local binding names for ``ModifierCombo`` in this module.

    Alias-tolerant (R-M4): reads every ``ImportFrom`` alias, so
    ``import ModifierCombo as MC`` binds ``MC``. Also binds the name at the
    definition site (``class ModifierCombo(Enum)`` in modifiers.py).
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "ModifierCombo":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ClassDef) and node.name == "ModifierCombo":
            names.add(node.name)
    return names


def _combo_dispatch_sites(source: str) -> list[tuple[int, str, str]]:
    """Return ``(lineno, form, member)`` for every combo MEMBER access used to
    DISPATCH: inside a ``match``/``case`` pattern, or inside an ``ast.Compare``
    with ``Eq``/``NotEq``/``In``/``NotIn`` (which covers a member named inside a
    tuple/set/dict literal on either side of such a compare).

    This is the scan command recorded in neograph-s7zt3.7's disease scan, made
    alias-tolerant.
    """
    tree = ast.parse(source)
    bindings = _combo_binding_names(tree)
    if not bindings:
        return []

    def _members(node: ast.AST):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in bindings:
                yield sub

    hits: set[tuple[int, str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.match_case):
            hits.update((m.lineno, "match/case", m.attr) for m in _members(node.pattern))
        elif isinstance(node, ast.Compare) and any(
            isinstance(op, (ast.Eq, ast.NotEq, ast.In, ast.NotIn)) for op in node.ops
        ):
            hits.update((m.lineno, "compare", m.attr) for m in _members(node))
    return sorted(hits)


def _used_table_symbols(source: str) -> set[str]:
    """Table symbols imported from ``neograph.modifiers`` AND actually USED.

    R-L3: a dead import must not satisfy assertion (b). Usage means an
    ``ast.Name`` load of the local binding somewhere outside the import (an
    ``ast.Attribute`` access such as ``PrimaryShape.EACH`` contains that Name
    node, so attribute references count too).
    """
    tree = ast.parse(source)
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] == "modifiers":
            for alias in node.names:
                if alias.name in TABLE_SYMBOLS:
                    imported[alias.asname or alias.name] = alias.name
    if not imported:
        return set()

    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in imported:
            used.add(imported[node.id])
    return used


def _touches_combo_vocabulary(source: str) -> bool:
    """True when the module references ``ModifierCombo`` (import, definition, or
    a fully-qualified ``x.ModifierCombo`` chain) OR imports a table symbol."""
    tree = ast.parse(source)
    if _combo_binding_names(tree):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "ModifierCombo":
            return True
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] == "modifiers":
            if any(alias.name in TABLE_SYMBOLS for alias in node.names):
                return True
    return False


def _package_files() -> list[pathlib.Path]:
    """Every .py under src/neograph (recursive -- a subpackage must not escape)."""
    return sorted(p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC_DIR).as_posix()


# --- the guard --------------------------------------------------------------


class TestComboDecompositionConsumerMonopoly:
    """neograph-s7zt3.7: "what does this ModifierCombo decompose into" is
    answered in exactly ONE place (``COMBO_DECOMPOSITION``/``PrimaryShape``/
    ``SUB_CONSTRUCT_UNSUPPORTED_COMBOS`` in modifiers.py). Every build-time
    consumer READS that table; none re-derives the grouping with a hand-written
    member enumeration. Questions genuinely about modifier PRESENCE
    (``"oracle" in mods``) are NOT decomposition and stay presence reads --
    they name no ``ModifierCombo`` member, so this guard never flags them."""

    # --- (a) no second enumeration -----------------------------------------

    def test_no_hand_written_combo_member_dispatch_in_migrated_files(self):
        offenders: list[str] = []
        for name in sorted(MIGRATED):
            path = SRC_DIR / name
            assert path.is_file(), f"MIGRATED names a file that does not exist: {name}"
            for lineno, form, member in _combo_dispatch_sites(path.read_text()):
                offenders.append(f"{_rel(path)}:{lineno}\t{form}\tModifierCombo.{member}")
        assert not offenders, (
            "Hand-written ModifierCombo member dispatch found in a MIGRATED file.\n"
            "Read the single-source table instead: "
            "`match COMBO_DECOMPOSITION[combo].primary:` with `case PrimaryShape.X:` arms "
            "(+ assert_never), `primary_shape(item) is PrimaryShape.X` for a membership "
            "test, or `combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS` for the sub-construct "
            'gate. A modifier-PRESENCE question stays `"oracle" in mods` '
            "(neograph-s7zt3.7).\n" + "\n".join(offenders)
        )

    # --- (b) reads the table ------------------------------------------------

    def test_every_migrated_file_imports_and_uses_a_table_symbol(self):
        missing: list[str] = []
        for name in sorted(MIGRATED):
            path = SRC_DIR / name
            if not _used_table_symbols(path.read_text()):
                missing.append(_rel(path))
        assert not missing, (
            "MIGRATED file(s) do not import-and-use any of "
            f"{sorted(TABLE_SYMBOLS)} from neograph.modifiers. A dead import does not "
            "count -- the binding must actually be referenced (neograph-s7zt3.7 R-L3).\n" + "\n".join(missing)
        )

    # --- (c) completeness / anti-tautology ----------------------------------

    def test_combo_vocabulary_consumers_are_exactly_the_declared_inventory(self):
        """Filesystem-derived census == the two hand-written literals + the owner.

        The two sides come from independent sources, so this cannot pass
        tautologically. A new file that grows combo dispatch (or starts reading
        the table) must be declared here -- the guard IS the consumer inventory.
        """
        actual = {_rel(p) for p in _package_files() if _touches_combo_vocabulary(p.read_text())}
        expected = set(MIGRATED) | set(PENDING) | {TABLE_OWNER}
        assert actual == expected, (
            "The set of src/neograph files touching the ModifierCombo vocabulary "
            "diverged from the declared inventory.\n"
            f"  undeclared (new consumer -- add to MIGRATED, or migrate it): {sorted(actual - expected)}\n"
            f"  declared but gone (shrink the literal -- PENDING is a ratchet):"
            f" {sorted(expected - actual)}"
        )

    def test_pending_is_a_ratchet_and_disjoint_from_migrated(self):
        """PENDING may only shrink, and a file cannot be both migrated and pending."""
        assert not (MIGRATED & PENDING), f"A file is both MIGRATED and PENDING: {sorted(MIGRATED & PENDING)}"
        assert TABLE_OWNER not in MIGRATED and TABLE_OWNER not in PENDING
        assert PENDING == frozenset(), (
            "PENDING is EMPTY as of neograph-tjpn4 and must stay empty. It is a "
            "ratchet at its end stop -- new combo dispatch must be written against "
            "COMBO_DECOMPOSITION, never parked here for a later migration."
        )


class TestComboDispatchScannerMetaTests:
    """Positive + negative meta-tests for `_combo_dispatch_sites` -- the scanner
    that assertion (a) rests on. A guard whose scanner silently matches nothing
    is worse than no guard."""

    def test_meta_flags_match_case_over_combo_members(self):
        src = (
            "from neograph.modifiers import ModifierCombo\n"
            "def f(combo):\n"
            "    match combo:\n"
            "        case ModifierCombo.EACH:\n"
            "            return 1\n"
            "        case ModifierCombo.ORACLE | ModifierCombo.ORACLE_OPERATOR:\n"
            "            return 2\n"
        )
        hits = _combo_dispatch_sites(src)
        assert [h[1] for h in hits] == ["match/case"] * 3
        assert {h[2] for h in hits} == {"EACH", "ORACLE", "ORACLE_OPERATOR"}

    def test_meta_flags_membership_tuple_compare(self):
        src = (
            "from neograph.modifiers import ModifierCombo, classify_modifiers\n"
            "def f(item):\n"
            "    return classify_modifiers(item)[0] in (ModifierCombo.PORTAL, ModifierCombo.PORTAL_OPERATOR)\n"
        )
        hits = _combo_dispatch_sites(src)
        assert [h[1] for h in hits] == ["compare", "compare"]
        assert {h[2] for h in hits} == {"PORTAL", "PORTAL_OPERATOR"}

    def test_meta_flags_equality_compare(self):
        src = "from neograph.modifiers import ModifierCombo\ndef f(combo):\n    return combo == ModifierCombo.BARE\n"
        assert [h[2] for h in _combo_dispatch_sites(src)] == ["BARE"]

    def test_meta_flags_aliased_import_form(self):
        """R-M4: `import ModifierCombo as MC` + `MC.EACH` must NOT walk through."""
        src = (
            "from neograph.modifiers import ModifierCombo as MC\n"
            "def f(combo):\n"
            "    if combo in {MC.EACH, MC.EACH_ORACLE}:\n"
            "        return 1\n"
            "    match combo:\n"
            "        case MC.LOOP:\n"
            "            return 2\n"
        )
        hits = _combo_dispatch_sites(src)
        assert {h[2] for h in hits} == {"EACH", "EACH_ORACLE", "LOOP"}

    def test_meta_ignores_healthy_table_read(self):
        """Negative: the sanctioned `match COMBO_DECOMPOSITION[combo].primary:`
        form with PrimaryShape arms is not a combo-member enumeration."""
        src = (
            "from neograph.modifiers import COMBO_DECOMPOSITION, PrimaryShape\n"
            "def f(combo):\n"
            "    match COMBO_DECOMPOSITION[combo].primary:\n"
            "        case PrimaryShape.EACH:\n"
            "            return 1\n"
            "        case PrimaryShape.ORACLE:\n"
            "            return 2\n"
        )
        assert _combo_dispatch_sites(src) == []

    def test_meta_ignores_unsupported_combos_frozenset_gate(self):
        """Negative: `combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS` READS the table
        (the healthy exemplar at _agent_spec.py:945) -- never flagged."""
        src = (
            "from neograph.modifiers import SUB_CONSTRUCT_UNSUPPORTED_COMBOS\n"
            "def f(combo):\n"
            "    return combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS\n"
        )
        assert _combo_dispatch_sites(src) == []

    def test_meta_ignores_modifier_presence_read(self):
        """Negative: the presence idiom (_fan_agent.py / __main__.py style, and
        a co-presence test in this shape -- superseded by decomp.fused, but
        this scanner's job is only to ignore modifier-PRESENCE reads, not to
        police which one is canonical) is not dispatch."""
        src = (
            "from neograph.modifiers import classify_modifiers\n"
            "def f(item):\n"
            "    _combo, mods = classify_modifiers(item)\n"
            "    if 'oracle' in mods and 'each' in mods:\n"
            "        return 'fused'\n"
            "    return 'oracle' in mods\n"
        )
        assert _combo_dispatch_sites(src) == []

    def test_meta_ignores_non_dispatch_member_reference(self):
        """Negative: naming a member OUTSIDE a dispatch context (e.g. a plain
        assignment or a call argument) is not the disease this guard bans."""
        src = "from neograph.modifiers import ModifierCombo\ndef f():\n    return ModifierCombo.BARE\n"
        assert _combo_dispatch_sites(src) == []

    def test_meta_negative_controls_from_real_files_are_unflagged(self):
        """The two real presence-only readers must stay clean forever."""
        for name in ("_fan_agent.py", "__main__.py"):
            assert _combo_dispatch_sites((SRC_DIR / name).read_text()) == [], (
                f"{name} is a modifier-PRESENCE reader and must never be flagged"
            )

    def test_meta_healthy_gate_unflagged_even_when_scanner_is_live(self):
        """The healthy `combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS` gate stays
        unflagged in a module where the scanner is LIVE.

        Distinct from test_meta_ignores_unsupported_combos_frozenset_gate, which
        proves the gate is ignored in a module holding NO ModifierCombo binding at
        all -- there the scanner has nothing to match and passing is nearly free.
        Here a ModifierCombo binding IS in scope and a diseased arm IS present, so
        the scanner is actively producing hits; the assertion is that it
        discriminates WITHIN such a module rather than flagging the whole file.

        Previously this asserted against the real _agent_spec.py and required its
        dispatch chain to exist (`assert hits`). neograph-tjpn4 cured that chain,
        which would have turned a successful migration into a red guard -- so the
        case is now expressed on a synthetic snippet that cannot rot.
        """
        src = (
            "from neograph.modifiers import SUB_CONSTRUCT_UNSUPPORTED_COMBOS, ModifierCombo\n"
            "def f(item, combo):\n"
            "    if combo in SUB_CONSTRUCT_UNSUPPORTED_COMBOS:\n"  # healthy: line 3
            "        raise ValueError('unsupported')\n"
            "    if combo == ModifierCombo.EACH:\n"  # diseased: line 5
            "        return 1\n"
            "    return 0\n"
        )
        flagged = {lineno for lineno, _form, _member in _combo_dispatch_sites(src)}
        assert 5 in flagged, "the diseased `combo == ModifierCombo.EACH` arm must be flagged"
        assert 3 not in flagged, "the healthy SUB_CONSTRUCT_UNSUPPORTED_COMBOS gate must NOT be flagged"

    def test_agent_spec_now_holds_zero_combo_dispatch(self):
        """neograph-tjpn4 end state: the last PENDING file is clean.

        Redundant with assertion (a) by construction, but it keeps _agent_spec.py
        named in a meta-test so a future reader sees the file that used to be the
        exception is now the proof.
        """
        assert _combo_dispatch_sites((SRC_DIR / "_agent_spec.py").read_text()) == []


class TestTableSymbolUsageScannerMetaTests:
    """Positive + negative meta-tests for `_used_table_symbols` (assertion b)."""

    def test_meta_detects_imported_and_used_symbol(self):
        src = (
            "from neograph.modifiers import COMBO_DECOMPOSITION, PrimaryShape\n"
            "def f(combo):\n"
            "    return COMBO_DECOMPOSITION[combo].primary is PrimaryShape.LOOP\n"
        )
        assert _used_table_symbols(src) == {"COMBO_DECOMPOSITION", "PrimaryShape"}

    def test_meta_rejects_dead_import(self):
        """R-L3: an unused import must NOT satisfy assertion (b)."""
        src = "from neograph.modifiers import COMBO_DECOMPOSITION\ndef f():\n    return 1\n"
        assert _used_table_symbols(src) == set()

    def test_meta_detects_aliased_import_and_reports_canonical_name(self):
        src = (
            "from neograph.modifiers import primary_shape as _shape, PrimaryShape as _PS\n"
            "def f(item):\n"
            "    return _shape(item) is _PS.PORTAL\n"
        )
        assert _used_table_symbols(src) == {"primary_shape", "PrimaryShape"}

    def test_meta_ignores_same_named_symbol_from_another_module(self):
        src = "from somewhere.else_ import PrimaryShape\ndef f():\n    return PrimaryShape.EACH\n"
        assert _used_table_symbols(src) == set()

    def test_meta_ignores_combo_only_import(self):
        src = "from neograph.modifiers import ModifierCombo\ndef f(c):\n    return c == ModifierCombo.BARE\n"
        assert _used_table_symbols(src) == set()


class TestVocabularyCensusScannerMetaTests:
    """Positive + negative meta-tests for `_touches_combo_vocabulary` (assertion c)."""

    def test_meta_counts_combo_import(self):
        assert _touches_combo_vocabulary("from neograph.modifiers import ModifierCombo\n")

    def test_meta_counts_aliased_combo_import(self):
        assert _touches_combo_vocabulary("from neograph.modifiers import ModifierCombo as MC\n")

    def test_meta_counts_the_definition_site(self):
        assert _touches_combo_vocabulary("from enum import Enum\nclass ModifierCombo(Enum):\n    BARE = 1\n")

    def test_meta_counts_table_symbol_import(self):
        """Post-migration a MIGRATED file qualifies through the table, not the enum."""
        assert _touches_combo_vocabulary(
            "from neograph.modifiers import COMBO_DECOMPOSITION\ndef f(c):\n    return COMBO_DECOMPOSITION[c]\n"
        )

    def test_meta_counts_qualified_attribute_chain(self):
        assert _touches_combo_vocabulary("import neograph.modifiers as m\ndef f():\n    return m.ModifierCombo.BARE\n")

    def test_meta_ignores_presence_only_reader(self):
        src = (
            "from neograph.modifiers import classify_modifiers\n"
            "def f(item):\n"
            "    _c, mods = classify_modifiers(item)\n"
            "    return 'each' in mods\n"
        )
        assert not _touches_combo_vocabulary(src)

    def test_meta_ignores_unrelated_module(self):
        assert not _touches_combo_vocabulary("import os\ndef f():\n    return os.getcwd()\n")
