"""Structural guard: Agent-Spec metadata marker keys are centralized named
constants, never re-inlined ``"neograph/*"`` magic-string literals.

Disease (neograph-aa5gq Step 0, disease scan 2026-07-23): the ``neograph/*``
metadata marker keys are duplicated as inline string literals across the
EXPORT side (``_agent_spec.py``) and the IMPORT side (``loader.py``) with no
shared constant. Because export and import each spell the key by hand, a typo
on one side silently routes a marker-bearing primitive to the fail-loud/foreign
path -- a SILENT DOWNGRADE, the exact failure aa5gq's Core Invariant forbids.

Two complementary checks, both FAILING NOW (TDD red -- the constants do not
exist yet and 23 inline literals are present):

1. STRUCTURAL: no double-quoted ``"neograph/`` literal appears in
   ``_agent_spec.py`` / ``loader.py`` except on a module-level constant
   assignment line (``_MARK_... = "neograph/..."``). ``loader.py`` must import
   the constants, so it carries no literal at all.

2. VALUE-PIN: a shared symbol does NOT catch a VALUE typo (both sides move
   together), and the constant values ARE the wire format that stored YAMLs /
   foreign tools literally expect -- so the exact string values are pinned
   here (refinement MEDIUM-1: behavioral round-trip coverage alone is
   insufficient).

3. TEST-SIDE (neograph-741nn): the same re-typed-literal disease lived under
   ``tests/``, which check 1 could not see -- it scanned a fixed list of SRC
   files only. A test that re-types ``"neograph/modifier"`` and then drifts
   produces a SILENT no-match (the marker is simply unread and the node treated
   as foreign), and dgbqv.2's metadata-aware mini-executor reads these markers,
   so the test side is directly exposed. Check 3 scans ``tests/`` too.

   It matches by EXACT WIRE VALUE, not by the ``"neograph/`` prefix, and that
   precision is what makes widening safe -- three classes of prefix-matching
   literal legitimately stay literal and must not be flagged:
     - deliberately-FOREIGN fixture keys with no constant and no right to one
       (``"neograph/source"``, ``"neograph/tool_marker"``), which exist to test
       passthrough of an UNKNOWN key;
     - prose in docstrings and assert messages (``"neograph/__init__.py's
       __all__ ..."`` is a filesystem path in a message);
     - the bare namespace PREFIX scan ``key.startswith("neograph/")``, which
       tests the namespace rather than any one marker.
   An exact-value rule excludes all three structurally, so no allowlist of
   "acceptable literals" is needed and the guard cannot rot into one.

This guard module itself is exempt from check 3: it must re-type the wire
values to pin them (check 2 IS that pin).
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "neograph"
TESTS = Path(__file__).resolve().parent

# The one module exempt from the test-side scan: it re-types every wire value on
# purpose, because pinning those exact values IS check 2. Any OTHER test module
# that needs a marker imports the constant.
_TEST_SCAN_EXEMPT = frozenset({"test_guards_agent_spec_markers.py"})
# neograph-3ffdg.3 moved the _MARK_* definitions to _agent_spec_markers.py and
# split the exporter into three further modules. This list is EXTENDED, not
# repointed: the guard catches a re-inlined "neograph/..." literal anywhere on
# either side of the export<->import contract, so every module that could carry
# one stays in scope. Dropping the moved-from file, or omitting the moved-to
# files, would leave it scanning a shrinking surface and passing vacuously.
SCANNED = [
    SRC / "_agent_spec.py",
    SRC / "_agent_spec_markers.py",
    SRC / "_agent_spec_placeholders.py",
    SRC / "_agent_spec_node_lowering.py",
    SRC / "_agent_spec_modifier_lowering.py",
    SRC / "_agent_spec_portal.py",
    SRC / "loader.py",
    SRC / "_agent_spec_swarm_import.py",
]

# A double-quoted marker literal (the disease shape -- all 23 current instances
# are double-quoted; single-quoted occurrences are docstring/prose references,
# not marker usage).
_LITERAL = '"neograph/'

# The ONE sanctioned form: a module-level constant assignment binding the
# literal to a named constant, e.g. `_MARK_AGENT_SPEC = "neograph/agent_spec"`.
_ALLOWED_ASSIGNMENT = re.compile(r'^_?[A-Z][A-Z0-9_]*\s*=\s*"neograph/[a-z_]+"\s*(#.*)?$')

# The wire-format marker values the export<->import contract (and any stored
# Agent Spec YAML / foreign tool) literally depends on. A typo in any of these
# constant VALUES is a silent-downgrade bug a shared symbol cannot catch.
_EXPECTED_MARKER_VALUES = {
    "neograph/mode",
    "neograph/agent_spec",
    "neograph/tool_spec",
    "neograph/modifier",
    "neograph/group_id",
    "neograph/variant",
    "neograph/oracle_spec",
    "neograph/each_spec",
    "neograph/loop_spec",
    "neograph/operator_spec",
    "neograph/branch",
    "neograph/portal_spec",
    "neograph/portal_operator_spec",
    "neograph/prompt_spec",
}


def test_slip_allowed_assignment():
    """Slip meta-test (PROC-2) for the ``_ALLOWED_ASSIGNMENT`` regex: pins the
    boundary a naiver regex slips at -- a bare module-level constant assignment
    is the ONE sanctioned form; a dict-literal/lookup use-site or a
    non-constant (lowercase) name is NOT, so a re-inlined marker at a use-site
    is still caught."""
    # The sanctioned form (with and without a trailing comment) matches.
    assert _ALLOWED_ASSIGNMENT.match('_MARK_MODE = "neograph/mode"')
    assert _ALLOWED_ASSIGNMENT.match('_MARK_AGENT_SPEC = "neograph/agent_spec"  # wire key')
    # Use-sites (dict key / metadata lookup) must NOT read as the assignment.
    assert not _ALLOWED_ASSIGNMENT.match('metadata={_MARK_MODE: "neograph/mode"}')
    assert not _ALLOWED_ASSIGNMENT.match('spec = merge_node.metadata["neograph/oracle_spec"]')
    # The boundary: a lowercase (non-constant) LHS is not the allowed form.
    assert not _ALLOWED_ASSIGNMENT.match('mark = "neograph/mode"')


def test_no_reinlined_marker_literals_outside_constant_block():
    """No ``"neograph/`` literal outside a named-constant assignment line."""
    offenders: list[str] = []
    for path in SCANNED:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _LITERAL not in line:
                continue
            if _ALLOWED_ASSIGNMENT.match(line.strip()):
                continue
            offenders.append(f"{path.relative_to(SRC.parent.parent)}:{i}: {line.strip()}")

    assert not offenders, (
        "Agent-Spec marker keys must reference the centralized named constants "
        "in _agent_spec.py (imported into loader.py), never a re-inlined "
        '"neograph/*" literal -- a typo on one side of the export<->import '
        "contract silently downgrades a marker-bearing primitive "
        "(neograph-aa5gq Step 0).\n" + "\n".join(offenders)
    )


def _retyped_marker_literals_under_tests() -> list[str]:
    """Every EXACT marker wire value re-typed as a literal under ``tests/``.

    Matched with the surrounding quotes (``"neograph/variant"``, not the bare
    prefix) so a longer string that merely STARTS with a marker value -- prose
    such as ``"neograph/variant marker dict (never None)"`` -- is not a hit.
    """
    offenders: list[str] = []
    literals = {f'"{value}"' for value in _EXPECTED_MARKER_VALUES}
    for path in sorted(TESTS.rglob("*.py")):
        if path.name in _TEST_SCAN_EXEMPT:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(lit in line for lit in literals):
                offenders.append(f"{path.relative_to(SRC.parent.parent)}:{i}: {line.strip()}")
    return offenders


def test_slip_exact_value_match_does_not_flag_prose_or_foreign_keys():
    """Slip meta-test for check 3's exact-value rule: pins the boundary a
    prefix-matching scan would slip at. The three legitimate literal classes
    must NOT read as re-typed markers, and a real re-typed marker must."""
    literals = {f'"{value}"' for value in _EXPECTED_MARKER_VALUES}

    def hits(line: str) -> bool:
        return any(lit in line for lit in literals)

    # A real re-typed marker IS a hit -- otherwise check 3 is vacuous.
    assert hits('assert node.metadata["neograph/modifier"] == "oracle"')
    # Deliberately-foreign fixture keys have no constant and stay literal.
    assert not hits('metadata={"neograph/source": "FLOW_LEVEL_MARKER"}')
    assert not hits('tool_metadata={"neograph/tool_marker": "present"}')
    # Prose that merely begins with a marker value is not marker usage.
    assert not hits('"neograph/variant marker dict (never None, never bare {})"')
    assert not hits('"neograph/__init__.py\'s __all__ (layer discipline: not a "')
    # A bare namespace-prefix scan tests the namespace, not one marker.
    assert not hits('if key.startswith("neograph/"):')


def test_no_retyped_marker_literals_under_tests():
    """No test module re-types an exact marker wire value (neograph-741nn).

    A drifted re-typed key is a SILENT no-match on both sides of the contract,
    not a loud failure -- the same reason check 1 exists for src.
    """
    offenders = _retyped_marker_literals_under_tests()
    assert not offenders, (
        "these test modules re-type an exact Agent-Spec marker wire value "
        "instead of importing the _MARK_* constant from neograph._agent_spec "
        "(the form 12+ test modules already use). A re-typed key that drifts is "
        "a silent no-match -- the marker goes unread and the node is treated as "
        "foreign (neograph-741nn).\n" + "\n".join(offenders)
    )


def test_marker_constants_pin_the_exact_wire_values():
    """The three aa5gq-named constants pin their exact wire strings."""
    import neograph._agent_spec as ags

    assert getattr(ags, "_MARK_MODE", None) == "neograph/mode"
    assert getattr(ags, "_MARK_AGENT_SPEC", None) == "neograph/agent_spec"
    assert getattr(ags, "_MARK_TOOL_SPEC", None) == "neograph/tool_spec"


def test_every_marker_wire_value_is_a_module_constant():
    """Every marker the contract depends on is bound to a module-level
    constant -- a value typo on ANY key (not just the four named above) fails
    here, since the constant value is the literal wire format."""
    import neograph._agent_spec as ags

    bound_values = {v for v in vars(ags).values() if isinstance(v, str) and v.startswith("neograph/")}
    missing = _EXPECTED_MARKER_VALUES - bound_values
    assert not missing, (
        f"these marker wire values are not bound to any module-level constant "
        f"in _agent_spec.py (a value typo would go uncaught): {sorted(missing)}"
    )
