"""G-SLOT — modifier-pair legality is decided in exactly ONE gate function.

Guard for neograph-jtawq.3. Today `modifiers.py` encodes "which modifier pairs
are legal" independently in three places (``_SlotRule.excludes``, five
hand-coded ``model_post_init`` arms, and the Portal-dispatch dynamic rule
duplicated verbatim in ``model_post_init`` and ``with_modifier``) plus six more
hand-written per-slot enumerations. This file pins the collapsed shape:

* ``_validate_slot_set(slots)`` is the ONE gate; ``model_post_init`` and
  ``with_modifier`` are its two thin callers.
* ``_COMBO_MAP`` (legal pairs) and ``_CONFLICT_DIAGNOSTICS`` (illegal pairs,
  with the pair-specific message) partition every 2-subset of the roster.
* ``_DYNAMIC_RULES`` carries the one instance-dependent rule (Portal dispatch
  mode + Operator) as DATA, so its message literal stays inside a table.
* Every per-slot enumeration derives from ``_SLOT_RULES``.

Extends ``TestNoRedundantValidation`` (tests/test_guards_assembly.py): that
guard's "ModifierSet only" rule strengthens to "_validate_slot_set only".
"""

from __future__ import annotations

import ast
import itertools
import pathlib

import pydantic
import pytest

from neograph import modifiers as m
from neograph._node_modifier_kwargs import MODIFIER_KWARGS
from neograph._portal import Portal
from neograph.errors import ConstructError
from neograph.modifiers import _COMBO_MAP, _SLOT_RULES, Each, Loop, ModifierSet, Operator, Oracle
from neograph.node import Node
from tests.fakes import register_scripted

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

ROSTER_SLOTS = tuple(r.slot for r in _SLOT_RULES)

#: The two assignment targets that are allowed to hold a "Cannot combine"
#: string literal after jtawq.3 lands (refined plan item 2: the two former
#: module-level constants are inlined into _CONFLICT_DIAGNOSTICS and deleted).
CONFLICT_TABLE_NAMES = frozenset({"_CONFLICT_DIAGNOSTICS", "_DYNAMIC_RULES"})

CANNOT_COMBINE = "Cannot combine"


def _require(name: str):
    """Fetch a symbol jtawq.3 introduces, failing loud while it is absent."""
    value = getattr(m, name, None)
    assert value is not None, (
        f"modifiers.{name} does not exist yet.\n"
        "jtawq.3 collapses the three encodings of modifier-pair legality into "
        "one gate function (_validate_slot_set) reading _COMBO_MAP + "
        "_CONFLICT_DIAGNOSTICS + _DYNAMIC_RULES."
    )
    return value


def _each() -> Each:
    return Each(over="items", key="item")


def _oracle() -> Oracle:
    return Oracle(n=2, merge_fn="combine")


def _loop() -> Loop:
    return Loop(when="keep_going")


def _bare_node() -> Node:
    return Node(name="n", outputs=None)


class _DispatchOut(pydantic.BaseModel):
    spec: dict = {}
    payload: dict = {}


def _dispatch_portal() -> Portal:
    return Portal(route="decide", spec_field="spec", input_field="payload", output=_DispatchOut, max_depth=5)


# --------------------------------------------------------------------------
# G-SLOT (i) — totality: every 2-subset is legal-or-diagnosed, never both
# --------------------------------------------------------------------------


class TestSlotPairTotality:
    """Every pair of roster slots is either a legal _COMBO_MAP key or carries a
    pair-specific diagnostic. A pair in neither is the drift hole that makes
    ``ModifierSet.combo``'s raw ``_COMBO_MAP[...]`` index a latent KeyError."""

    def test_every_slot_pair_is_legal_or_diagnosed(self):
        diagnostics = _require("_CONFLICT_DIAGNOSTICS")
        pairs = [frozenset(p) for p in itertools.combinations(ROSTER_SLOTS, 2)]
        assert len(pairs) == 10, "C(5,2) — the roster grew; update this guard deliberately"
        unclassified = [sorted(p) for p in pairs if p not in _COMBO_MAP and p not in diagnostics]
        assert unclassified == [], (
            f"\n{len(unclassified)} modifier pair(s) are neither in _COMBO_MAP (legal) nor in "
            f"_CONFLICT_DIAGNOSTICS (illegal, with a pair-specific message):\n"
            + "\n".join(f"  {p}" for p in unclassified)
            + "\n\nAn unclassified pair reaches ModifierSet.combo's _COMBO_MAP[...] index as a raw KeyError."
        )

    def test_legal_and_diagnosed_pair_domains_are_disjoint(self):
        diagnostics = _require("_CONFLICT_DIAGNOSTICS")
        overlap = sorted(sorted(k) for k in set(diagnostics) & set(_COMBO_MAP))
        assert overlap == [], (
            f"\n{len(overlap)} pair(s) appear in BOTH _COMBO_MAP and _CONFLICT_DIAGNOSTICS: {overlap}\n"
            "A pair is legal or illegal, never both — two tables disagreeing is the disease jtawq.3 removes."
        )

    def test_conflict_diagnostics_holds_only_pairs_and_message_hint_tuples(self):
        diagnostics = _require("_CONFLICT_DIAGNOSTICS")
        roster = set(ROSTER_SLOTS)
        for key, value in diagnostics.items():
            assert isinstance(key, frozenset) and len(key) == 2, f"{key!r} is not a slot PAIR"
            assert key <= roster, f"{sorted(key)} names a slot outside _SLOT_RULES"
            assert isinstance(value, tuple) and len(value) == 2, f"{key!r} -> {value!r} is not (message, hint)"
            assert all(isinstance(s, str) for s in value), f"{key!r} -> {value!r} must be two strings"

    def test_dynamic_rules_is_a_data_table_not_raising_callables(self):
        rules = _require("_DYNAMIC_RULES")
        assert len(rules) >= 1, "the Portal(dispatch)+Operator rule must be a _DYNAMIC_RULES row"
        for row in rules:
            assert isinstance(row, tuple) and len(row) == 3, f"{row!r} is not a (predicate, message, hint) triple"
            predicate, message, hint = row
            assert callable(predicate), f"{row!r}: first element must be the predicate"
            assert isinstance(message, str) and isinstance(hint, str), (
                f"{row!r}: message and hint must be plain strings — if a message literal moves inside a "
                "function body, the G-SLOT literal ban needs a carve-out, which re-opens this hole."
            )


# --------------------------------------------------------------------------
# G-SLOT (ii) — the "Cannot combine" literal lives only in the two tables
# --------------------------------------------------------------------------


class TestCannotCombineLiteralMonopoly:
    """The canonical conflict phrasing may appear ONLY inside the
    ``_CONFLICT_DIAGNOSTICS`` and ``_DYNAMIC_RULES`` assignments. A literal
    anywhere else is a fourth hand-maintained encoding.

    Scans ``src/`` only — tests legitimately assert on the string. Scans exact
    ``Cannot combine``, so ``decorators.py``'s deliberately-different
    ``"... cannot be combined on the same node"`` phrasing (the @node surface's
    own kwarg-named pre-checks, jtawq.4 Finding 5) is not a false positive.
    """

    @staticmethod
    def _string_fragments(node: ast.AST):
        """Yield every literal string fragment, including f-string parts.

        Required: today's ``_km_conflict`` literal is an f-string
        (``f"Cannot combine {this_label} and Portal..."``) that a plain
        ``ast.Constant`` scan would miss.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield part.value

    @staticmethod
    def _docstring_node_ids(tree: ast.AST) -> set[int]:
        ids: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
        return ids

    @classmethod
    def _table_subtree_ids(cls, tree: ast.AST) -> set[int]:
        """``id()`` of every node inside a ``_CONFLICT_DIAGNOSTICS`` /
        ``_DYNAMIC_RULES`` assignment's value."""
        ids: set[int] = set()
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            if value is None:
                continue
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if names & CONFLICT_TABLE_NAMES:
                ids.update(id(child) for child in ast.walk(value))
        return ids

    @classmethod
    def _stray_literals(cls, source: str, filename: str = "<scan>") -> list[int]:
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:  # pragma: no cover - src/ always parses
            return []
        exempt = cls._table_subtree_ids(tree) | cls._docstring_node_ids(tree)
        hits: list[int] = []
        for node in ast.walk(tree):
            if id(node) in exempt:
                continue
            if any(CANNOT_COMBINE in frag for frag in cls._string_fragments(node)):
                hits.append(node.lineno)
        return sorted(set(hits))

    def test_cannot_combine_literal_only_inside_the_two_tables(self):
        violations: list[str] = []
        for py_file in sorted(SRC_DIR.rglob("*.py")):
            for lineno in self._stray_literals(py_file.read_text(), str(py_file)):
                violations.append(f"  {py_file.relative_to(SRC_DIR)}:{lineno}")
        assert violations == [], (
            f"\n{len(violations)} '{CANNOT_COMBINE}' literal(s) outside the "
            f"{sorted(CONFLICT_TABLE_NAMES)} assignments:\n"
            + "\n".join(violations)
            + "\n\nPair-specific conflict messages belong in _CONFLICT_DIAGNOSTICS; the one "
            "instance-dependent message belongs in _DYNAMIC_RULES. Anywhere else is a fourth encoding."
        )

    # --- meta-tests ---

    def test_detector_flags_a_stray_plain_literal(self):
        bad = 'def f():\n    raise ConstructError.build("Cannot combine Each and Loop on the same item")\n'
        assert self._stray_literals(bad) == [2]

    def test_detector_flags_a_stray_fstring_literal(self):
        """Slip check: today's _km_conflict literal is an f-string."""
        bad = 'def f(label):\n    return f"Cannot combine {label} and Portal on the same item"\n'
        assert self._stray_literals(bad) == [2]

    def test_detector_ignores_a_docstring_mention(self):
        ok = '"""Raises Cannot combine Each and Loop when both land."""\nX = 1\n'
        assert self._stray_literals(ok) == []

    def test_detector_ignores_literals_inside_the_sanctioned_tables(self):
        ok = (
            "_CONFLICT_DIAGNOSTICS = {\n"
            '    frozenset({"each", "loop"}): ("Cannot combine Each and Loop on the same item", "hint"),\n'
            "}\n"
            "_DYNAMIC_RULES = (\n"
            '    (lambda s: True, "Cannot combine Portal (dispatch mode) and Operator on the same item", "h"),\n'
            ")\n"
        )
        assert self._stray_literals(ok) == []


# --------------------------------------------------------------------------
# G-SLOT (iii) — the roster is the field set, on both axes
# --------------------------------------------------------------------------


class TestRosterIsTheFieldSet:
    def test_modifier_set_fields_match_the_roster(self):
        assert set(ModifierSet.model_fields) == set(ROSTER_SLOTS), (
            "ModifierSet's slots and _SLOT_RULES have drifted — a modifier with a field but no roster row "
            "(or vice versa) is invisible to _validate_slot_set."
        )

    def test_modifier_kwargs_names_match_the_roster(self):
        """Pins the seventh enumeration (_node_modifier_kwargs.MODIFIER_KWARGS)
        without merging it: it lives on a different axis (@node kwarg triggers),
        but it must cover exactly the same modifiers."""
        assert {row.name for row in MODIFIER_KWARGS} == set(ROSTER_SLOTS)

    def test_slot_rule_no_longer_carries_an_excludes_column(self):
        assert "excludes" not in type(_SLOT_RULES[0])._fields, (
            "_SlotRule.excludes is the FIRST of the three encodings of pair legality. "
            "It is deleted by jtawq.3 — _CONFLICT_DIAGNOSTICS is the one table."
        )


# --------------------------------------------------------------------------
# G-SLOT (iv) — every per-slot enumeration derives from the roster
# --------------------------------------------------------------------------


class TestPerSlotEnumerationsDeriveFromRoster:
    """The nine enumeration clusters the disease scan found all iterate
    ``_SLOT_RULES`` instead of hand-listing each/oracle/loop/operator/portal."""

    TARGETS = (
        "classify_modifiers",
        "combo",
        "to_list",
        "has_modifier",
        "get_modifier",
    )

    @staticmethod
    def _functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
        return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    @staticmethod
    def _references_roster(func: ast.AST) -> bool:
        return any(isinstance(n, ast.Name) and n.id == "_SLOT_RULES" for n in ast.walk(func))

    @staticmethod
    def _hand_enumerated_slots(func: ast.AST) -> set[str]:
        """Roster slot names appearing as a literal attribute access or string
        constant in the body — the shape of a hand-written per-slot chain."""
        seen: set[str] = set()
        roster = set(ROSTER_SLOTS)
        for node in ast.walk(func):
            if isinstance(node, ast.Attribute) and node.attr in roster:
                seen.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in roster:
                seen.add(node.value)
        return seen

    @pytest.fixture(scope="class")
    def funcs(self) -> dict[str, ast.FunctionDef]:
        return self._functions(ast.parse((SRC_DIR / "modifiers.py").read_text()))

    @pytest.mark.parametrize("name", TARGETS)
    def test_function_iterates_the_roster(self, funcs, name):
        func = funcs.get(name)
        assert func is not None, f"modifiers.{name} not found"
        assert self._references_roster(func), (
            f"modifiers.{name} does not reference _SLOT_RULES — it hand-enumerates the slots instead. "
            "Derive from the roster so modifier #6 needs ONE row, not six edits."
        )

    @pytest.mark.parametrize("name", TARGETS)
    def test_function_does_not_hand_enumerate_every_slot(self, funcs, name):
        func = funcs.get(name)
        assert func is not None, f"modifiers.{name} not found"
        enumerated = self._hand_enumerated_slots(func)
        assert enumerated != set(ROSTER_SLOTS), (
            f"modifiers.{name} names every roster slot ({sorted(enumerated)}) inline — "
            "a hand-written per-slot chain that silently omits modifier #6."
        )


# --------------------------------------------------------------------------
# The gate itself: one function, two thin callers
# --------------------------------------------------------------------------


class TestOneGateTwoCallers:
    @staticmethod
    def _method(class_name: str, method_name: str) -> ast.FunctionDef:
        tree = ast.parse((SRC_DIR / "modifiers.py").read_text())
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)

    @staticmethod
    def _calls(func: ast.AST, callee: str) -> bool:
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == callee for n in ast.walk(func)
        )

    def test_validate_slot_set_is_module_level(self):
        gate = _require("_validate_slot_set")
        assert callable(gate)

    @pytest.mark.parametrize("method", ["model_post_init", "with_modifier"])
    def test_both_construction_paths_call_the_gate(self, method):
        func = self._method("ModifierSet", method)
        assert self._calls(func, "_validate_slot_set"), (
            f"ModifierSet.{method} does not call _validate_slot_set. Both construction paths "
            "(direct ctor and the pipe path) must route through the ONE gate — model_copy skips "
            "model_post_init, which is exactly why the shared function exists."
        )

    def test_model_post_init_has_no_hand_coded_pair_arms(self):
        func = self._method("ModifierSet", "model_post_init")
        raises = [n for n in ast.walk(func) if isinstance(n, ast.Raise)]
        assert raises == [], (
            f"ModifierSet.model_post_init still raises directly ({len(raises)} site(s)) — "
            "the five hand-coded arms collapse to one _validate_slot_set(...) call."
        )


# --------------------------------------------------------------------------
# Behavior the collapse must produce (and the two exception shapes it must NOT change)
# --------------------------------------------------------------------------


class TestGateBehavior:
    @pytest.mark.parametrize(
        "second, expected",
        [
            (_each, "Cannot combine Portal and Each on the same item"),
            (_oracle, "Cannot combine Portal and Oracle on the same item"),
            (_loop, "Cannot combine Portal and Loop on the same item"),
        ],
    )
    def test_pipe_path_phrasing_is_canonical_regardless_of_order(self, second, expected):
        """Today the pipe path phrases the conflict by "whichever landed
        second", so ``| Portal() | Each()`` says "Each and Portal" while
        ``| Each() | Portal()`` says "Portal and Each". One gate reading one
        table canonicalizes both onto the already-pinned direct-construction
        order (tests/modifiers/test_portal.py:222-230)."""
        for mods in ([Portal(to=["peer"]), second()], [second(), Portal(to=["peer"])]):
            node = _bare_node()
            with pytest.raises(ConstructError) as exc:
                for mod in mods:
                    node = node | mod
            assert expected in str(exc.value), f"pipe order {[type(x).__name__ for x in mods]}"

    def test_each_loop_message_and_hint_survive_byte_for_byte(self):
        with pytest.raises(pydantic.ValidationError) as exc:
            ModifierSet(each=_each(), loop=_loop())
        text = str(exc.value)
        assert "Cannot combine Each and Loop on the same item" in text
        assert "Use a sub-construct with Loop inside an Each fan-out instead" in text

    def test_oracle_loop_message_survives_byte_for_byte(self):
        with pytest.raises(pydantic.ValidationError) as exc:
            ModifierSet(oracle=_oracle(), loop=_loop())
        assert "Cannot combine Oracle and Loop on the same item" in str(exc.value)

    def test_superset_reports_the_roster_first_conflicting_pair(self):
        """A 3-superset has TWO conflicting pairs ({each,loop} and
        {oracle,loop}). Which message it reports must be deterministic — a
        hash-ordered scan over the present-name frozenset would coin-flip it
        per process. Roster order reproduces today's arm precedence."""
        with pytest.raises(pydantic.ValidationError) as exc:
            ModifierSet(each=_each(), oracle=_oracle(), loop=_loop())
        text = str(exc.value)
        assert "Cannot combine Each and Loop on the same item" in text
        assert "Cannot combine Oracle and Loop on the same item" not in text

    def test_direct_construction_still_raises_validation_error(self):
        """Pure-refactor guard: the direct path stays Pydantic-wrapped."""
        with pytest.raises(pydantic.ValidationError) as exc:
            ModifierSet(each=_each(), loop=_loop())
        assert not isinstance(exc.value, ConstructError)

    def test_pipe_path_still_raises_bare_construct_error(self):
        """Pure-refactor guard: the pipe path stays a bare ConstructError.
        Routing with_modifier through the constructor for symmetry would flip
        this to ValidationError — a real behavior change, refused."""
        node = _bare_node() | _each()
        with pytest.raises(ConstructError):
            node | _loop()

    def test_portal_dispatch_plus_operator_rejected_from_both_paths(self):
        """The one instance-dependent rule (_DYNAMIC_RULES): it reads
        ``Portal.is_dispatch``, so it cannot be a static table entry — and it
        is currently duplicated verbatim in both construction paths."""
        register_scripted("slot_set_guard_cond", lambda d: True)

        with pytest.raises(pydantic.ValidationError) as ctor:
            ModifierSet(portal=_dispatch_portal(), operator=Operator(when="slot_set_guard_cond"))
        assert "dispatch mode" in str(ctor.value)

        with pytest.raises(ConstructError) as pipe:
            _bare_node() | _dispatch_portal() | Operator(when="slot_set_guard_cond")
        assert "dispatch mode" in str(pipe.value)

    def test_portal_peer_plus_operator_stays_legal(self):
        """The gate must not over-reject: PEER-mode Portal + Operator is a
        sanctioned approval gate (neograph-kdr1u, D4 lift)."""
        register_scripted("slot_set_guard_cond2", lambda d: True)
        node = _bare_node() | Portal(to=["peer"]) | Operator(when="slot_set_guard_cond2")
        assert node.modifier_set.portal is not None
        assert node.modifier_set.operator is not None

    def test_unknown_modifier_error_names_every_roster_modifier(self):
        """The stale expected= list omits Portal today; deriving it from the
        roster labels fixes it and keeps it fixed."""
        with pytest.raises(ConstructError) as exc:
            ModifierSet().with_modifier(object())  # type: ignore[arg-type]
        text = str(exc.value)
        missing = [r.label for r in _SLOT_RULES if r.label not in text]
        assert missing == [], f"'Unknown modifier type' expected= omits {missing}"

    def test_duplicate_slot_detection_is_not_simplified_away(self):
        """A name-set loses slot occupancy ({loop} u {loop} = {loop}, a legal
        set), so the duplicate arm cannot move into the gate."""
        node = _bare_node() | _loop()
        with pytest.raises(ConstructError) as exc:
            node | _loop()
        assert "Duplicate Loop modifier" in str(exc.value)
