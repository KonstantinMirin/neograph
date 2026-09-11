"""Structural guard: the item WRITE-SET derivation has exactly one home.

neograph-yz69e. "Which state fields does this item write" was answered
independently by six functions, each with a different omission, and the drift was
silent at every site:

  - ``_construct_validation`` registered the Portal ``{node}_dispatch`` producer;
  - ``_ir_fields.declared_output_fields`` and ``_ir_normalize._producer_pairs``
    did not, so a single-type consumer VALIDATED green and got ``None`` at run time;
  - ``_ir_fields.item_field_names`` did not, so a sub-construct boundary the
    dispatch result should satisfy raised at run time instead;
  - ``_schema_fingerprint._fingerprint_item`` did not, so changing a Portal's
    ``output=`` opened the resume gate with nothing to attribute it to and the run
    resumed from the tip with a STALE dispatched result (neograph-p93qh).

The lesson is not "add the field in six places". It is that neograph had already
monopolised how to SPELL a field name (``output_field_name``) and how to READ one
back (``split_output_field``), and never monopolised how to ENUMERATE them. This
guard is that missing third monopoly.

Written FAILING-FIRST against the pre-fix tree, where it finds every offender.

Sibling of ``TestDeclaredOutputSelectorMonopoly`` (which owns the
``Node.outputs``/``Construct.output`` SELECTOR) and
``TestSubConstructBoundaryEligibilityMonopoly`` (which owns the boundary
eligibility SET). Same shape, same enforcement, one layer down.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# A file may build a per-output-key state-field name ONLY with a structural
# reason. Each row states one; a row may be REMOVED but never loosened, and a new
# row needs a reason of the same kind (see AGENTS.md on shrink-only ratchets).
ALLOWLIST: dict[str, str] = {
    # THE write-set authority this guard exists to create.
    "_ir_fields.py": "defines contributed_fields -- the enumeration authority",
    # The single-FIELD helper (which ONE of the written fields is primary). A
    # different question from the SET, and already monopolised on its own terms.
    "_normalize.py": "defines primary_output_field -- the single-field selector",
    # Pairs each name with a per-field TYPE and REDUCER. It is the authority for
    # what the channel IS, not a copy of which channels exist; a name list cannot
    # carry a reducer.
    "state.py": "attaches per-field type + reducer; iterates the shared names",
    # Pairs each name with a runtime VALUE pulled from the result dict.
    "_state_write.py": "pairs names with values during the state update",
    # Iterates keys PRESENT IN THE RUNTIME RESULT, not declared ones -- a fused
    # Each x Oracle merge may legitimately produce a subset.
    "_wiring_oracle_each.py": "iterates runtime-present keys, not declared ones",
    # Framework-key helpers keyed off a node's own producer field.
    "_state_keys.py": "defines the StateKeys framework-field helpers",
    # Resolves an address the AUTHOR NAMED (`output_from="node.key"`) to its
    # field. Answers "what is this named port called", not "what does this item
    # write" -- the set is not enumerated here, a single member+key is spelled.
    "_ir_source.py": "spells one author-NAMED port address; does not enumerate a set",
    # Takes (field_name, output_model) with NO item in scope, so it cannot call
    # the per-item enumeration without a signature change -- which would make a
    # delegation into a behaviour change. Tracked as neograph-g0r1m.
    "_oracle.py": "no item in scope; threading one in is a signature change -- neograph-g0r1m",
}


def _output_field_name_sites(tree: ast.AST) -> list[int]:
    """Line numbers of calls to ``output_field_name(...)`` in ``tree``."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "output_field_name"
    ]


def _iter_src_files() -> list[pathlib.Path]:
    return sorted(p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


class TestWriteSetDerivationMonopoly:
    """No module outside the allowlist may build a per-output-key state-field
    name for itself -- it calls the shared enumeration instead."""

    def test_no_unlisted_module_builds_an_output_field_name(self) -> None:
        offenders: dict[str, list[int]] = {}
        for py in _iter_src_files():
            key = py.relative_to(SRC_DIR).as_posix()
            short = py.name
            if short in ALLOWLIST or key in ALLOWLIST:
                continue
            sites = _output_field_name_sites(ast.parse(py.read_text()))
            if sites:
                offenders[key] = sites

        assert offenders == {}, (
            f"\n{len(offenders)} module(s) derive a per-output-key state-field name for themselves:\n"
            + "\n".join(f"  {f}: lines {lines}" for f, lines in sorted(offenders.items()))
            + "\n\nCall the shared write-set enumeration (contributed_fields in _ir_fields.py) and read "
            "the field names off it. Each independent derivation drifts silently -- that is neograph-yz69e."
        )

    def test_allowlist_has_no_stale_rows(self) -> None:
        """A row whose file no longer builds such a name must be DELETED, not left
        as silent headroom -- the same shrink-only discipline the file-size ratchet
        uses.
        """
        stale = []
        for name in ALLOWLIST:
            matches = [p for p in _iter_src_files() if p.name == name or p.relative_to(SRC_DIR).as_posix() == name]
            if not matches:
                stale.append(f"{name} (file does not exist)")
                continue
            if not any(_output_field_name_sites(ast.parse(p.read_text())) for p in matches):
                stale.append(f"{name} (no output_field_name call remains)")
        assert stale == [], "\nStale ALLOWLIST rows -- delete them:\n  " + "\n  ".join(stale)

    def test_every_allowlist_row_states_a_reason(self) -> None:
        unjustified = [f for f, why in ALLOWLIST.items() if len(why.strip()) < 20]
        assert unjustified == [], f"ALLOWLIST rows without a structural reason: {unjustified}"

    def test_meta_scanner_catches_the_call(self) -> None:
        assert _output_field_name_sites(ast.parse("x = output_field_name(base, key)\n")) == [1]

    def test_meta_scanner_ignores_other_calls(self) -> None:
        tree = ast.parse("x = field_name_for(n)\ny = split_output_field(f)\n")
        assert _output_field_name_sites(tree) == []
