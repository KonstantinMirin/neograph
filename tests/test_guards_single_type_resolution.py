"""Guard: exactly ONE site may answer "which producer satisfies this single-type
``inputs=``", and no site may answer it by scanning the whole state bag.

neograph-t1nbp: the question had two implementations with OPPOSITE precedence --
``_extract_single_type`` walked ``state.keys()`` forward (earliest match wins) and
the Agent Spec export walked preceding items in reverse (latest wins) -- so a
green run and its exported artifact wired different edges. Measured on a 3-node
construct: the runtime fed the consumer from ``a``, the export wired ``b``.

Both now read ``Node.input_source_field``, resolved once at assembly by
``_ir_normalize.resolve_single_type_source``. This guard is what stops a third
answer appearing, the way ``TestSubConstructBoundaryEligibilityMonopoly`` does for
the output boundary.

Deliberately a STRUCTURAL guard, not a behavioural one: the failure it prevents is
a future author adding a second resolver, which no runtime assertion can observe.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "neograph"

# The last commit BEFORE neograph-t1nbp's fix, pinned rather than spelled ``HEAD``:
# HEAD becomes the post-fix tree the moment the fix lands, which would silently
# invert the real-source meta-test below from "the guard still fires" into "the
# guard fires on code that no longer exists". Resolved-but-absent (a shallow clone,
# rewritten history) is tolerated; a WRONG answer is not.
_PRE_FIX_COMMIT = "0b6a3d1"

# The bags a "which value satisfies this type" scan iterates. A leading-underscore
# framework key (``_neo_isolated_input``) is why membership is tested against the
# NAME rather than a ``neo_``-anchored regex -- see the would-be-missed meta-test.
_BAG_CALLS = frozenset({"keys", "values", "items"})

# Builtin targets that make an isinstance a KIND test ("is this a class at all")
# rather than a match against a DECLARED type. ``isinstance(annotation, type)``
# guarding a recursive descent is not a producer selection.
_KIND_TESTS = frozenset({"type", "str", "int", "float", "bool", "bytes", "dict", "list", "tuple", "set"})

# Sites allowed to run a whole-bag type scan, each with the reason it cannot read
# a resolved name instead. Shrink-only: adding a row needs the same justification.
_ALLOWED: dict[str, str] = {
    # The sub-construct PORT twin, still open as neograph-5suot unknown #5. It is
    # listed rather than fixed because it is a different boundary with its own
    # ticket -- not because a whole-state scan is acceptable there.
    # neograph-af8ro ROW DELETED by neograph-9axw6.6, not relaxed. The Loop
    # self-feedback destination is no longer picked by first type match: the three
    # sites that each answered it -- this lowering, the validator, and the runtime's
    # positional next(iter(...)) fallback -- now read one derivation
    # (_ir_fields.loop_carry_dest_key). The guard itself demanded this deletion
    # ("the exemption outlived its reason; delete the row"), which is the allowlist
    # shrinking because the architecture got right rather than because a rule was
    # relaxed.
    # ADDED by neograph-9axw6.10, and it is a genuine over-match rather than an
    # exemption. _key_for walks the ADDRESS TABLE the resolver already stamped and
    # returns the key holding a given Source VARIANT -- it projects a resolved answer
    # into one of the four derived views that replaced the four collapsed fields. It
    # matches this guard's AST shape (isinstance test in a loop, return on first hit)
    # while doing the opposite of what the guard is for: no user type is tested, no
    # candidate is chosen, and at most one key can hold each variant by construction.
    #
    # The alternative was to restructure the loop until the pattern stopped matching,
    # which would be gaming the guard rather than answering it.
    "_node_addresses.py": (
        "projects the stamped address table into a derived view; the isinstance test is "
        "over Source VARIANTS, not user types, and selects nothing"
    ),
    "_subconstruct.py": (
        "_scan_subgraph_input (neograph-5suot #5, open) and _scan_subgraph_output, "
        "whose eligible=None arm is the sanctioned Portal mode-(b) fallback: the "
        "flow is EMITTED AT RUNTIME so its item names cannot exist at assembly."
    ),
}


def _scan_source(source: str) -> list[str]:
    """Return a description per whole-bag type-scan found in ``source``.

    A violation is a loop over ``<bag>.keys()/.values()/.items()`` containing an
    ``if`` whose TEST is a type check and whose BODY leaves the loop with the
    match (``return``/``break``). The test and the exit must be the SAME ``if``:
    that conjunction is "first positional type match wins", which is the disease.

    Three shapes are deliberately NOT violations, because each was checked in the
    tree and is a different thing:

    - **collect-then-refuse** -- gather every match, then raise on more than one
      (``_di_classify``). No early exit, so no ``if``-test-and-return pair. This is
      the CORRECT shape and the guard must never push anyone away from it.
    - **sum-type dispatch** -- iterate a ``{class: value}`` table and test against
      the loop's OWN variable (``_agent_spec_types``). Excluded explicitly below:
      the tested type is loop-bound, so nothing is competing to be selected.
    - **existence / name search** -- a scan whose exit is gated on something other
      than the type test (``_validation_outputs`` returns a formatted name gated on
      a marker flag). Nothing type-selected flows out as a value.
    """
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        if not _iterates_a_bag(node.iter):
            continue
        bound = {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
        body = ast.Module(body=node.body, type_ignores=[])
        leaves_loop = any(isinstance(n, (ast.Return, ast.Break)) for n in ast.walk(body))
        for inner in ast.walk(body):
            if not isinstance(inner, ast.If):
                continue
            if not _type_test_against_free_type(inner.test, bound):
                continue
            positive = any(
                isinstance(n, (ast.Return, ast.Break))
                for n in ast.walk(ast.Module(body=inner.body, type_ignores=[]))
            )
            # The guard-clause spelling: ``if not <type-match>: continue`` further
            # up a loop that ends in ``break`` is logically the SAME "first match
            # wins", and it is how the Agent Spec export half of neograph-t1nbp was
            # written. A detector that only understood ``if <match>: return``
            # caught the runtime twin and missed the export -- which is exactly the
            # half-fix this ticket exists to prevent.
            skips = _is_negated(inner.test) and all(isinstance(st, ast.Continue) for st in inner.body)
            if positive or (skips and leaves_loop):
                found.append(
                    f"line {node.lineno}: type-matching scan over a bag that returns/breaks on first hit"
                )
                break
    return found


def _is_negated(test: ast.expr) -> bool:
    """Is ``test`` a negated condition -- ``not X``, or a comparison via ``not``?"""
    return isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)


def _type_test_against_free_type(test: ast.expr, loop_bound: set[str]) -> bool:
    """Is ``test`` a type check whose TARGET TYPE is not bound by the loop?

    ``isinstance(v, node.inputs)`` -- yes: the type comes from a DECLARATION and
    runtime values are being matched against it, which is the selection this guard
    bans. ``isinstance(prop, prop_cls)`` where ``prop_cls`` is the loop variable --
    no: that is dispatch over a table, and nothing competes.

    ``_isinstance_safe`` counts: it is the codebase's own isinstance wrapper, and
    the pre-fix ``_extract_single_type`` used exactly that spelling.
    """
    for call in ast.walk(test):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id not in ("isinstance", "issubclass", "_isinstance_safe"):
            continue
        if len(call.args) < 2:
            continue
        type_arg = call.args[1]
        names = {n.id for n in ast.walk(type_arg) if isinstance(n, ast.Name)}
        if names & loop_bound:
            continue  # dispatch over a {class: value} table -- nothing competes
        if names & _KIND_TESTS:
            continue  # "is this a class/str/dict at all", not a declared-type match
        if names and all(n[:1].isupper() for n in names):
            # A literal class reference (``isinstance(m, FromInput)``) is a marker
            # or kind test. The disease matches against a type that ARRIVED from a
            # declaration -- spelled as an attribute (``node.inputs``) or a
            # lowercase binding (``sub_output_type``), never a CapWords literal.
            # Heuristic, and deliberately documented as one: it keys on the naming
            # convention, so a class named in lowercase would slip. That trade is
            # worth it -- the alternative is resolving imports, and the miss mode
            # is a false NEGATIVE on unconventional naming, not a false green on
            # the shape this guard exists for.
            continue
        return True
    return False


def _iterates_a_bag(expr: ast.expr) -> bool:
    """Does ``expr`` iterate a mapping/state bag, including through ``reversed()``
    or ``list()``? Those wrappers are how the same scan is spelled in three places,
    so unwrapping them is what keeps the guard from being trivially side-stepped."""
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in ("reversed", "list", "sorted"):
        # ``reversed(...)`` of ANYTHING counts, not just of a bag call. The export
        # half of neograph-t1nbp was ``for upstream in reversed(ordered_items[:idx])``
        # -- a list slice, no ``.keys()`` in sight -- and a bag-call-only detector
        # missed it while catching its runtime twin. "Latest match wins" is the
        # other half of the disease and has to be in scope for the guard to be
        # worth having.
        return expr.func.id == "reversed" or any(_iterates_a_bag(a) for a in expr.args)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return expr.func.attr in _BAG_CALLS
    return False


class TestNoSecondSingleTypeResolver:
    """No module outside the allowlist resolves a value by scanning a whole bag."""

    def test_no_unallowlisted_whole_bag_type_scan_exists(self):
        violations: list[str] = []
        for py_file in sorted(SRC_DIR.rglob("*.py")):
            if py_file.name in _ALLOWED:
                continue
            for hit in _scan_source(py_file.read_text()):
                violations.append(f"  {py_file.relative_to(SRC_DIR)}: {hit}")

        assert violations == [], (
            "A value is being resolved by scanning a bag for the first type match. That is "
            "neograph-t1nbp's disease: the runtime and the Agent Spec export each ran such a "
            "scan, in OPPOSITE directions, so the exported artifact wired an edge the runtime "
            "did not take.\n"
            "Read the resolved name instead -- Node.input_source_field, written once by "
            "_ir_normalize.resolve_single_type_source -- or add an allowlist row saying why "
            "this site structurally cannot.\nViolations:\n" + "\n".join(violations)
        )

    def test_allowlist_rows_are_live_and_justified(self):
        """A stale allowlist is worse than none: it certifies a site nobody rechecked."""
        for name, reason in _ALLOWED.items():
            matches = list(SRC_DIR.rglob(name))
            assert matches, f"allowlist names {name!r}, which no longer exists -- delete the row"
            assert _scan_source(matches[0].read_text()), (
                f"allowlist row {name!r} no longer contains a whole-bag type scan -- the "
                "exemption outlived its reason; delete the row"
            )
            assert len(reason) > 40, f"allowlist row {name!r} needs a real reason, got {reason!r}"


class TestTheGuardActuallyDetects:
    """Meta-tests. A structural guard that cannot be shown to fire is decoration."""

    def test_positive_forward_scan_is_caught(self):
        assert _scan_source(
            "def f(state, node):\n"
            "    for k in state.keys():\n"
            "        v = state.get(k)\n"
            "        if isinstance(v, node.inputs):\n"
            "            return v\n"
        ), "the exact pre-fix _extract_single_type shape must be caught"

    def test_positive_reverse_scan_is_caught(self):
        assert _scan_source(
            "def f(sub_result, t):\n"
            "    for v in reversed(list(sub_result.values())):\n"
            "        if isinstance(v, t):\n"
            "            return v\n"
        ), "wrapping the bag in reversed()/list() must not side-step the guard"

    def test_would_be_missed_leading_underscore_framework_key_is_still_caught(self):
        """The concrete slip this guard is built to survive.

        ``_neo_isolated_input`` is a real framework key whose leading underscore
        defeats a ``^neo_``-anchored regex. This guard keys on the BAG being
        iterated, not on key spelling, so a scan that would match that key is
        caught the same as any other -- but only as long as nobody rewrites the
        detector to match key names. This test fails if someone does.
        """
        assert _scan_source(
            "def f(state, t):\n"
            "    for k in state.keys():\n"
            "        if k == '_neo_isolated_input' and isinstance(state[k], t):\n"
            "            return state[k]\n"
        ), "a scan touching a leading-underscore framework key must still be caught"

    def test_negative_named_read_is_not_caught(self):
        assert not _scan_source(
            "def f(state, node):\n"
            "    for field in _source_candidates(node):\n"
            "        v = state.get(field)\n"
            "        if v is not None and isinstance(v, node.inputs):\n"
            "            return v\n"
        ), "reading an explicit, resolved candidate list is the FIX, not the disease"

    def test_negative_existence_check_is_not_caught(self):
        """A type test that keeps no winner is a validator's existence check --
        legitimate, and the thing _agent_spec.py once wrongly cited as its
        authority for picking one."""
        assert not _scan_source(
            "def f(producers, t):\n"
            "    ok = False\n"
            "    for p in producers.values():\n"
            "        if isinstance(p, t):\n"
            "            ok = True\n"
            "    return ok\n"
        ), "an existence check selects nothing and must not be flagged"

    def test_both_real_pre_fix_sites_are_caught(self):
        """The strongest meta-test available: run the detector against the ACTUAL
        pre-fix source of both divergent scans, recovered from git.

        Synthetic snippets prove a detector fires on what its author imagined. This
        proves it fires on what actually shipped -- and it caught a real hole: the
        first version of this guard flagged the runtime scan but MISSED the export
        one, because the export was written guard-clause style
        (``if not <match>: continue`` ... ``break``) and iterated a list slice
        rather than a ``.keys()`` bag. A guard that catches one half of a
        two-implementation divergence would have let this exact bug recur.
        """
        import subprocess

        repo = SRC_DIR.parents[1]
        for rel, label in (
            ("src/neograph/_input_shape.py", "runtime forward scan"),
            ("src/neograph/_agent_spec.py", "export reverse scan"),
        ):
            pre_fix = subprocess.run(
                ["git", "show", f"{_PRE_FIX_COMMIT}:{rel}"], capture_output=True, text=True, cwd=repo
            ).stdout
            if not pre_fix:
                continue  # shallow clone or rewritten history -- do not fail CI on it
            assert _scan_source(pre_fix), (
                f"the guard no longer detects the {label} that neograph-t1nbp removed "
                f"({rel} at {_PRE_FIX_COMMIT}). Someone narrowed the detector past usefulness."
            )

    def test_negative_plain_iteration_is_not_caught(self):
        assert not _scan_source(
            "def f(d):\n    for k, v in d.items():\n        print(k, v)\n"
        ), "iterating a bag without type-selecting is ordinary code"
