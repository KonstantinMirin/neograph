"""Structural guard: the branch-arm BARE rule has exactly ONE enforcement site
(neograph-ftnxl.19).

neograph-ftnxl.12 added ``_check_no_portal_in_branch_arm`` for Portal only.
neograph-ftnxl.19 found the SAME inertness for Loop/Each/Oracle/Operator and,
rather than adding a second arm rule beside the first, FOLDED both into one
table-driven ``_check_no_modifier_in_branch_arm``. That fold is the invariant
this guard protects.

Why it matters: AGENTS.md records the Portal-rollout lesson verbatim -- a
``ModifierCombo`` question answered independently in seven consumers is the
duplicated-source-of-truth anti-pattern, and the way it regrows is somebody
adding "just one more" local check. An arm rule is exactly that shape: the next
person to notice a modifier misbehaving in an arm will reach for a second
``_check_no_<X>_in_branch_arm``. This guard makes that fail loud.

Detector (pure ``ast``, no ``re`` -- exempt by construction from
``tests/test_guards_meta.py``'s named-regex/slip-test discipline, the same move
``TestComboMapMonopoly`` makes): a function is an ARM-RULE ENFORCER when it both
(a) walks arm contents -- ``iter_with_arm_ids`` / ``iter_with_arms``, or a
``true_arm_nodes`` / ``false_arm_nodes`` attribute read -- and (b) raises a
neograph error class. The set of such functions must equal the declared
inventory.

This is the structural COMPANION to the behavioral pin
``TestBranchArmGuardIsTotalOverModifierCombo``
(``tests/test_branch_arm_modifier_validation.py``), which proves the ONE rule is
total over every ``ModifierCombo``. Together: one site (here), covering every
shape (there).
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

#: Names whose appearance means "this function walks branch-arm contents".
_ARM_WALK_NAMES = frozenset({"iter_with_arm_ids", "iter_with_arms"})
_ARM_ATTRS = frozenset({"true_arm_nodes", "false_arm_nodes"})

#: neograph error classes. A raise of one of these is what turns an arm WALK
#: into an arm RULE.
_ERROR_CLASSES = frozenset(
    {"ConstructError", "CompileError", "ConfigurationError", "ExecutionError", "NeographError"}
)

#: The declared inventory: (module, function) pairs that enforce an arm rule.
#: Hand-written, sourced independently of the scan, so the equality below cannot
#: pass tautologically.
DECLARED_ARM_RULE_ENFORCERS: frozenset[tuple[str, str]] = frozenset(
    {
        # THE arm-BARE rule (neograph-ftnxl.19, folding in ftnxl.12's
        # Portal-only predecessor). Reads COMBO_DECOMPOSITION -- no hand-written
        # member list -- so it is total over future PrimaryShape values. This is
        # the entry the guard exists to protect: a SECOND
        # _check_no_<modifier>_in_branch_arm beside it is the regrowth.
        ("_validation_arms.py", "_check_no_modifier_in_branch_arm"),
        # NOT a modifier rule: the main assembly walk. It iterates
        # iter_with_arm_ids to producer-register and type-check each arm item,
        # raising the ordinary per-item input/fan-in/context errors. Its subject
        # is dataflow, orthogonal to what modifier an item carries.
        ("_construct_validation.py", "_validate_node_chain"),
        # NOT a modifier rule: the node-name-collision check (neograph-ftnxl.13).
        # Keyed on the normalized state FIELD NAME, so it is modifier-independent
        # by construction -- a same-/cross-arm collision is equally silent
        # whatever the items carry.
        ("state.py", "compile_state_model"),
        # NOT a modifier rule, though it reads one: the Operator string-condition
        # registration scan. It still governs TOP-LEVEL items; its arm half became
        # unreachable when ftnxl.19 made an arm Operator unrepresentable. Left in
        # place because iter_with_arms is one walk serving both.
        ("compiler.py", "compile"),
        # Defense-in-depth backstop, deliberately retained (see its in-file
        # comment): unreachable now that assembly rejects every modifier-carrying
        # arm item strictly earlier. Kept so a weakened assembly guard fails loud
        # instead of silently dropping an Oracle modifier.
        ("_fan_agent_wrap.py", "wrap_fan_over_agents"),
    }
)


def _iterates_arms(expr: ast.AST) -> bool:
    """True when a ``for`` loop's iterable walks branch-arm contents.

    Matches BOTH forms: a call to an arm-descent primitive
    (``iter_with_arm_ids(c)`` / ``iter_with_arms(c)``) and a raw attribute read
    (``meta.true_arm_nodes + meta.false_arm_nodes``). A second arm rule
    therefore cannot dodge the detector by hand-rolling its own walk.
    """
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and n.id in _ARM_WALK_NAMES:
            return True
        if isinstance(n, ast.Attribute) and n.attr in _ARM_ATTRS:
            return True
    return False


def _raises_neograph_error(node: ast.AST) -> bool:
    """True when ``node``'s subtree raises one of neograph's error classes."""
    for n in ast.walk(node):
        if not isinstance(n, ast.Raise) or n.exc is None:
            continue
        exc = n.exc
        # `raise X.build(...)` and `raise X(...)` alike.
        if isinstance(exc, ast.Call):
            exc = exc.func
        if isinstance(exc, ast.Attribute) and isinstance(exc.value, ast.Name):
            if exc.value.id in _ERROR_CLASSES:
                return True
        if isinstance(exc, ast.Name) and exc.id in _ERROR_CLASSES:
            return True
    return False


def _collect_arm_rule_enforcers(root: pathlib.Path) -> set[tuple[str, str]]:
    """Every ``(module, function)`` containing a per-arm-item RULE.

    The rule shape is deliberately narrow -- a ``for`` loop that iterates arm
    contents AND raises a neograph error from inside its own body. Narrowness is
    the point: merely walking arms is not a rule (the primitives, ``iter_nodes``,
    the fingerprint walk), and merely raising is not one either (most of the
    package). Only the conjunction, lexically nested, means "this code decides
    what an arm item may be".
    """
    found: set[tuple[str, str]] = set()
    for py_file in sorted(root.rglob("*.py")):
        tree = ast.parse(py_file.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for loop in ast.walk(fn):
                if isinstance(loop, ast.For) and _iterates_arms(loop.iter) and _raises_neograph_error(loop):
                    found.add((py_file.name, fn.name))
                    break
    return found


class TestBranchArmRuleMonopoly:
    """The arm-BARE rule lives in exactly one place, and its Portal-only
    predecessor is gone rather than merely bypassed."""

    def test_arm_rule_enforcers_are_exactly_the_declared_inventory(self):
        actual = _collect_arm_rule_enforcers(SRC_DIR)
        assert actual == DECLARED_ARM_RULE_ENFORCERS, (
            "The set of functions enforcing a rule over branch-arm contents diverged "
            "from the declared inventory.\n"
            f"  undeclared (a SECOND arm rule? fold it into "
            f"_check_no_modifier_in_branch_arm, or declare it): {sorted(actual - DECLARED_ARM_RULE_ENFORCERS)}\n"
            f"  declared but gone (shrink the literal): "
            f"{sorted(DECLARED_ARM_RULE_ENFORCERS - actual)}"
        )

    def test_portal_only_arm_guard_is_deleted_not_merely_unused(self):
        """ftnxl.12's ``_check_no_portal_in_branch_arm`` must not survive as a
        stale duplicate: a second definition would drift from the general rule
        the moment either is edited."""
        defined_in = [
            p.name
            for p in sorted(SRC_DIR.rglob("*.py"))
            if any(
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name == "_check_no_portal_in_branch_arm"
                for n in ast.walk(ast.parse(p.read_text()))
            )
        ]
        assert defined_in == [], (
            "_check_no_portal_in_branch_arm was folded into "
            f"_check_no_modifier_in_branch_arm (neograph-ftnxl.19) but is still defined in {defined_in}"
        )

    # --- meta-tests: prove the detector actually discriminates ---

    def test_meta_detector_flags_a_second_arm_rule(self, tmp_path):
        """Negative meta-test: a newly planted parallel arm guard is caught."""
        (tmp_path / "_validation_sneaky.py").write_text(
            "def _check_no_loop_in_branch_arm(construct):\n"
            "    for item, arm_key in iter_with_arm_ids(construct):\n"
            "        if arm_key is not None:\n"
            "            raise ConstructError.build('nope')\n"
        )
        found = _collect_arm_rule_enforcers(tmp_path)
        assert ("_validation_sneaky.py", "_check_no_loop_in_branch_arm") in found

    def test_meta_detector_flags_the_attribute_read_form(self, tmp_path):
        """A second arm rule cannot dodge the detector by reading
        ``meta.true_arm_nodes`` directly instead of calling a primitive."""
        (tmp_path / "_validation_sneaky2.py").write_text(
            "def _check_arm_raw(construct):\n"
            "    for branch in construct.nodes:\n"
            "        meta = branch._neo_branch_meta\n"
            "        for item in meta.true_arm_nodes + meta.false_arm_nodes:\n"
            "            raise CompileError.build('nope')\n"
        )
        found = _collect_arm_rule_enforcers(tmp_path)
        assert ("_validation_sneaky2.py", "_check_arm_raw") in found

    def test_meta_detector_ignores_an_arm_walk_that_raises_nothing(self, tmp_path):
        """Positive meta-test: a pure arm WALK (the primitives, iter_nodes,
        the fingerprint walk) is not a rule and must not be flagged --
        otherwise the inventory would balloon and stop meaning anything."""
        (tmp_path / "_walker.py").write_text(
            "def iter_arms(construct):\n"
            "    for branch in construct.nodes:\n"
            "        meta = branch._neo_branch_meta\n"
            "        yield from meta.true_arm_nodes\n"
            "        yield from meta.false_arm_nodes\n"
        )
        assert _collect_arm_rule_enforcers(tmp_path) == set()

    def test_meta_detector_ignores_a_raise_with_no_arm_walk(self, tmp_path):
        """Positive meta-test: an ordinary validator that raises but never
        touches arm contents is not an arm rule."""
        (tmp_path / "_plain.py").write_text(
            "def _check_something(construct):\n"
            "    for item in construct.nodes:\n"
            "        if item is None:\n"
            "            raise ConstructError.build('nope')\n"
        )
        assert _collect_arm_rule_enforcers(tmp_path) == set()
