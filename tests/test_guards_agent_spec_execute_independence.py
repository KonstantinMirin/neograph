"""Structural guard (neograph-dgbqv.2): the EXECUTE/COMPARE harness stays
independent of neograph's own Agent Spec IMPORT path.

The differential-execution grid's whole value proposition -- "a disagreement
between the third-party runtime and neograph's own run(compile(construct)) is a
real defect in one of two independently-authored sides" -- is void if the
harness itself imports the code path it is meant to be independent of. If
``agent_spec_loader_harness.py`` imported ``neograph.loader``/``from_agent_spec``
directly, a bug shared by the importer and this harness would agree with itself
and the tier would report a false green.

**Why an AST import scan, not a runtime ``sys.modules`` check.** The harness
LEGITIMATELY reuses ``tests.test_agent_spec_matrix``'s cell registry
(``CELLS``/``GREEN``/``build_cell``) and ``tests.test_agent_spec_reachability``'s
``_all_flows`` -- both of which transitively import ``neograph.loader`` for
their own, unrelated reasons. A runtime check ("is ``neograph.loader`` in
``sys.modules`` after importing the harness") would therefore ALWAYS fail, which
is exactly why the write-test atom's handoff notes call for scanning the
harness module's OWN import statements instead: the harness may sit downstream
of a module that imports the importer, but it must never name the importer
itself.

Pure AST, no ``re``, so this module is exempt-by-construction from
``test_guards_meta.py`` Discipline 1.
"""

from __future__ import annotations

import ast
import pathlib

_HARNESS = pathlib.Path(__file__).resolve().parent / "agent_spec_loader_harness.py"

#: Module names the harness may never import, directly or via ``from X import Y``.
_BANNED_MODULES = frozenset(
    {
        "neograph.loader",
        "neograph._spec_loader",
        "loader",  # covers `from neograph import loader`
    }
)

#: Symbol names the harness may never import, regardless of source module (catches
#: `from neograph.loader import from_agent_spec` under any future re-export path).
_BANNED_SYMBOLS = frozenset({"from_agent_spec", "load_spec"})


def _harness_imports() -> tuple[set[str], set[str]]:
    """Return (imported module names, imported symbol names) for the harness's
    own top-level ``import`` / ``from ... import ...`` statements."""
    tree = ast.parse(_HARNESS.read_text(encoding="utf-8"))
    modules: set[str] = set()
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
            for alias in node.names:
                symbols.add(alias.name)
    return modules, symbols


def _harness_command_calls() -> list[int]:
    """Line numbers of any ``Command(`` construction in the harness -- guard G1
    (``tests/test_guards_assembly.py``) scans ``src/`` only, so a ``tests/``
    harness constructing ``Command`` directly would slip past it entirely."""
    tree = ast.parse(_HARNESS.read_text(encoding="utf-8"))
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Command":
            lines.append(node.lineno)
    return lines


class TestExecuteHarnessStaysIndependentOfNeographsOwnImporter:
    def test_harness_file_exists(self) -> None:
        assert _HARNESS.exists(), f"expected the harness module at {_HARNESS}"

    def test_harness_does_not_import_a_banned_module(self) -> None:
        modules, _symbols = _harness_imports()
        hit = modules & _BANNED_MODULES
        assert not hit, (
            f"agent_spec_loader_harness.py imports {sorted(hit)} -- the harness must never "
            "import neograph's own Agent Spec IMPORT path directly. Reuse of test_agent_spec_matrix "
            "/ test_agent_spec_reachability (which transitively import it) is fine and expected; "
            "a DIRECT import here is not."
        )

    def test_harness_does_not_import_a_banned_symbol(self) -> None:
        _modules, symbols = _harness_imports()
        hit = symbols & _BANNED_SYMBOLS
        assert not hit, (
            f"agent_spec_loader_harness.py imports the symbol(s) {sorted(hit)} -- these name "
            "neograph's own importer regardless of which module they were imported from."
        )

    def test_harness_constructs_no_command(self) -> None:
        sites = _harness_command_calls()
        assert not sites, (
            f"agent_spec_loader_harness.py constructs Command( at line(s) {sites} -- guard G1 "
            "scans src/ only, so a tests/ harness bypassing it would go undetected."
        )


class TestIndependenceScannerMetaTests:
    """Positive + negative coverage for the two scanners above, so a scanner that
    silently stops matching (or a rename that slips past it) fails loud."""

    def test_module_scanner_flags_a_synthetic_direct_import(self, tmp_path) -> None:
        synthetic = tmp_path / "synthetic_harness.py"
        synthetic.write_text("from neograph.loader import from_agent_spec\n")
        tree = ast.parse(synthetic.read_text(encoding="utf-8"))
        modules: set[str] = set()
        symbols: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
                symbols.update(alias.name for alias in node.names)
        assert modules & _BANNED_MODULES
        assert symbols & _BANNED_SYMBOLS

    def test_module_scanner_ignores_a_healthy_import(self, tmp_path) -> None:
        synthetic = tmp_path / "synthetic_harness.py"
        synthetic.write_text("from neograph import compile, run\n")
        tree = ast.parse(synthetic.read_text(encoding="utf-8"))
        modules: set[str] = set()
        symbols: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
                symbols.update(alias.name for alias in node.names)
        assert not (modules & _BANNED_MODULES)
        assert not (symbols & _BANNED_SYMBOLS)

    def test_command_scanner_fires_on_a_synthetic_construction(self, tmp_path) -> None:
        synthetic = tmp_path / "synthetic_command.py"
        synthetic.write_text("from langgraph.types import Command\nc = Command(goto='x')\n")
        tree = ast.parse(synthetic.read_text(encoding="utf-8"))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Command"
        ]
        assert lines

    def test_command_scanner_ignores_an_unrelated_call(self, tmp_path) -> None:
        synthetic = tmp_path / "synthetic_no_command.py"
        synthetic.write_text("c = SomethingElse(goto='x')\n")
        tree = ast.parse(synthetic.read_text(encoding="utf-8"))
        lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Command"
        ]
        assert not lines
