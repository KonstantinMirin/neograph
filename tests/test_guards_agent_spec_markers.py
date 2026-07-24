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

This guard lives under ``tests/`` (not scanned) so it may name the marker
strings freely.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "neograph"
SCANNED = [SRC / "_agent_spec.py", SRC / "loader.py"]

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
    "neograph/remote_agent",
    "neograph/modifier",
    "neograph/group_id",
    "neograph/variant",
    "neograph/oracle_spec",
    "neograph/each_spec",
    "neograph/loop_spec",
    "neograph/operator_spec",
    "neograph/branch",
    "neograph/portal_spec",
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


def test_marker_constants_pin_the_exact_wire_values():
    """The four aa5gq-named constants pin their exact wire strings."""
    import neograph._agent_spec as ags

    assert getattr(ags, "_MARK_MODE", None) == "neograph/mode"
    assert getattr(ags, "_MARK_AGENT_SPEC", None) == "neograph/agent_spec"
    assert getattr(ags, "_MARK_TOOL_SPEC", None) == "neograph/tool_spec"
    assert getattr(ags, "_MARK_REMOTE_AGENT", None) == "neograph/remote_agent"


def test_every_marker_wire_value_is_a_module_constant():
    """Every marker the contract depends on is bound to a module-level
    constant -- a value typo on ANY key (not just the four named above) fails
    here, since the constant value is the literal wire format."""
    import neograph._agent_spec as ags

    bound_values = {
        v for v in vars(ags).values() if isinstance(v, str) and v.startswith("neograph/")
    }
    missing = _EXPECTED_MARKER_VALUES - bound_values
    assert not missing, (
        f"these marker wire values are not bound to any module-level constant "
        f"in _agent_spec.py (a value typo would go uncaught): {sorted(missing)}"
    )
