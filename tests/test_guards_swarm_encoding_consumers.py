"""Structural guard (neograph-dgbqv.5): the Agent-Spec Swarm <-> PortalMemberClass
encoding has exactly ONE authority.

PortalMemberClass is the runtime taxonomy and sole authority (neograph-dgbqv.3,
P7); the Agent-Spec Swarm encoding must be ONE table that READS it in both
directions, so no consumer re-derives "Agent vs Flow", "trigger", or
"HandoffMode" locally. Two disease shapes, both fresh-swept (not copied from a
stale disposition table -- AGENTS.md's provisional-list lesson):

  (a) **No second HandoffMode enumeration.** A hand-written
      ``HandoffMode.OPTIONAL if any_tool else HandoffMode.NEVER``-shaped ternary
      (an ``ast.Attribute`` access on ``HandoffMode``) outside the table module.
      The sanctioned post-migration shape ``HandoffMode(<str from table>)`` is a
      CALL, not an attribute reference, so it passes by construction.
  (b) **No type-name member-class dispatch.** A ``type(x).__name__ == "Flow"``
      compare outside the table module. Precise by design: comparisons against
      ``"FlowNode"``/``"AgentNode"``/``"Swarm"`` are spec-PRIMITIVE dispatch (a
      different question P7 also refused to migrate) and stay untouched.

Three more assertions mirror ``test_guards_combo_decomposition_consumers.py``'s
shape exactly (the exemplar this ticket's design field names):

  (c) **Reads the table.** Each declared consumer imports at least one table
      symbol from ``neograph._agent_spec_swarm_encoding`` AND actually uses the
      binding (a dead import must not satisfy the guard -- R-L3).
  (d) **Completeness / anti-tautology.** The set of package files that touch
      either disease shape OR import a table symbol -- derived from the
      FILESYSTEM -- must equal ``MIGRATED | {TABLE_OWNER}``, a hand-written
      literal sourced independently of the scan. A brand-new file growing
      either idiom fails loud here even if it names no member.

Written in pure ``ast`` with no ``re``, so it is exempt by construction from
``tests/test_guards_meta.py``'s named-regex/slip-test discipline.
"""

from __future__ import annotations

import ast
import pathlib

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

#: The single definition site. Scoped out of (a)/(b)/(c) by construction.
TABLE_OWNER = "_agent_spec_swarm_encoding.py"

#: Fresh-swept 2026-08-06 (grep -rn 'HandoffMode\.' / '__name__ == "Flow"'
#: src/neograph/): exactly one real dispatch site per disease shape, both
#: outside the (not-yet-created) table module. Both files must additionally
#: read the table once step 4/5 of the implementation plan lands.
MIGRATED: frozenset[str] = frozenset(
    {
        # _agent_spec_portal.py:126's `HandoffMode.OPTIONAL if any_tool else
        # HandoffMode.NEVER` ternary -- becomes HandoffMode(mesh_handoff_mode(...)).
        "_agent_spec_portal.py",
        # _agent_spec_swarm_import.py:179's `type(agent).__name__ == "Flow"` --
        # becomes a lookup through the table's spec_class inverse.
        "_agent_spec_swarm_import.py",
    }
)

#: The symbols that ARE the single source of truth. Consumers reading one of
#: these are reading the table, not re-deriving it.
TABLE_SYMBOLS: frozenset[str] = frozenset(
    {
        "HANDOFF_MODE_TRIGGER",
        "SWARM_ENCODING",
        "handoff_mode_for_class",
        "mesh_handoff_mode",
    }
)


# --- scanners (pure ast) ------------------------------------------------------


def _handoff_mode_attr_sites(source: str) -> list[tuple[int, str]]:
    """Every ``HandoffMode.<ATTR>`` attribute access (real dispatch, not a
    ``HandoffMode(<value>)`` call -- the sanctioned post-migration shape)."""
    tree = ast.parse(source)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "HandoffMode":
            hits.append((node.lineno, node.attr))
    return hits


def _flow_type_name_sites(source: str) -> list[int]:
    """Every ``type(x).__name__ == "Flow"``-shaped compare (either operand
    order)."""
    tree = ast.parse(source)
    hits: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and any(isinstance(op, ast.Eq) for op in node.ops)):
            continue
        operands = [node.left, *node.comparators]
        literals = [o for o in operands if isinstance(o, ast.Constant) and o.value == "Flow"]
        dunder_name = [
            o
            for o in operands
            if isinstance(o, ast.Attribute) and o.attr == "__name__" and isinstance(o.value, ast.Call)
        ]
        if literals and dunder_name:
            hits.append(node.lineno)
    return hits


def _used_table_symbols(source: str) -> set[str]:
    """Table symbols imported from the encoding module AND actually USED
    (R-L3: a dead import does not satisfy assertion (c))."""
    tree = ast.parse(source)
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[-1] == "_agent_spec_swarm_encoding"
        ):
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


def _touches_swarm_encoding_vocabulary(source: str) -> bool:
    """True when the module has a HandoffMode-attribute site, a Flow-type-name
    site, imports a table symbol, or DEFINES one (the table module itself) --
    the union that keeps assertion (d) stable across the migration
    (pre-migration via the disease shapes, post-migration via the table
    import/definition)."""
    if _handoff_mode_attr_sites(source) or _flow_type_name_sites(source):
        return True
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".")[-1] == "_agent_spec_swarm_encoding"
        ):
            if any(alias.name in TABLE_SYMBOLS for alias in node.names):
                return True
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef)):
            target_names = (
                [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.target.id]
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                else [node.name]
                if isinstance(node, ast.FunctionDef)
                else []
            )
            if any(name in TABLE_SYMBOLS for name in target_names):
                return True
    return False


def _package_files() -> list[pathlib.Path]:
    return sorted(p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _rel(path: pathlib.Path) -> str:
    return path.relative_to(SRC_DIR).as_posix()


# --- the guard ------------------------------------------------------------


class TestSwarmEncodingConsumerMonopoly:
    def test_no_second_handoff_mode_enumeration(self) -> None:
        offenders: list[str] = []
        for path in _package_files():
            rel = _rel(path)
            if rel in (TABLE_OWNER,):
                continue
            for lineno, attr in _handoff_mode_attr_sites(path.read_text(encoding="utf-8")):
                offenders.append(f"{rel}:{lineno} HandoffMode.{attr}")
        assert not offenders, (
            "hand-written HandoffMode attribute dispatch outside the table module -- "
            f"{offenders}. Route through SWARM_ENCODING/HANDOFF_MODE_TRIGGER instead; "
            "the sanctioned shape is HandoffMode(<str from mesh_handoff_mode(...)>), a "
            "CALL, not an attribute reference."
        )

    def test_no_type_name_member_class_dispatch(self) -> None:
        offenders: list[str] = []
        for path in _package_files():
            rel = _rel(path)
            if rel in (TABLE_OWNER,):
                continue
            for lineno in _flow_type_name_sites(path.read_text(encoding="utf-8")):
                offenders.append(f"{rel}:{lineno}")
        assert not offenders, (
            f"hand-written type(x).__name__ == 'Flow' member-class dispatch outside the "
            f"table module -- {offenders}. Route through SWARM_ENCODING's spec_class "
            "inverse instead."
        )

    def test_every_migrated_file_imports_and_uses_a_table_symbol(self) -> None:
        missing: dict[str, set[str]] = {}
        for rel in sorted(MIGRATED):
            path = SRC_DIR / rel
            used = _used_table_symbols(path.read_text(encoding="utf-8"))
            if not used:
                missing[rel] = used
        assert not missing, (
            f"MIGRATED files with no live import of a swarm-encoding table symbol: "
            f"{sorted(missing)}. A file that merely stopped hand-dispatching without "
            "starting to read the table is not actually migrated."
        )

    def test_swarm_encoding_consumers_are_exactly_the_declared_inventory(self) -> None:
        actual = {
            _rel(p) for p in _package_files() if _touches_swarm_encoding_vocabulary(p.read_text(encoding="utf-8"))
        }
        expected = MIGRATED | {TABLE_OWNER}
        assert actual == expected, (
            "the set of files touching the Swarm-encoding vocabulary diverged from the "
            f"declared inventory.\n  undeclared (new consumer -- migrate it, or declare "
            f"it): {sorted(actual - expected)}\n  declared but absent from the census: "
            f"{sorted(expected - actual)}"
        )


class TestSwarmEncodingScannerMetaTests:
    """Positive + negative coverage, so a scanner that silently stops matching
    fails loud (the same discipline every sibling consumer guard carries)."""

    def test_handoff_mode_scanner_fires_on_a_synthetic_ternary(self) -> None:
        src = "x = HandoffMode.OPTIONAL if flag else HandoffMode.NEVER\n"
        assert _handoff_mode_attr_sites(src) == [(1, "OPTIONAL"), (1, "NEVER")]

    def test_handoff_mode_scanner_ignores_the_sanctioned_call_form(self) -> None:
        src = "x = HandoffMode(mesh_handoff_mode(classes))\n"
        assert not _handoff_mode_attr_sites(src)

    def test_flow_type_name_scanner_fires_on_a_synthetic_compare(self) -> None:
        src = 'if type(agent).__name__ == "Flow":\n    pass\n'
        assert _flow_type_name_sites(src) == [1]

    def test_flow_type_name_scanner_ignores_a_different_literal(self) -> None:
        src = 'if type(agent).__name__ == "FlowNode":\n    pass\n'
        assert not _flow_type_name_sites(src)

    def test_flow_type_name_scanner_fires_regardless_of_operand_order(self) -> None:
        src = 'if "Flow" == type(agent).__name__:\n    pass\n'
        assert _flow_type_name_sites(src) == [1]

    def test_table_symbol_scanner_rejects_a_dead_import(self) -> None:
        src = "from neograph._agent_spec_swarm_encoding import SWARM_ENCODING\n"
        assert not _used_table_symbols(src)

    def test_table_symbol_scanner_detects_live_use(self) -> None:
        src = "from neograph._agent_spec_swarm_encoding import SWARM_ENCODING\nx = SWARM_ENCODING[cls]\n"
        assert _used_table_symbols(src) == {"SWARM_ENCODING"}
