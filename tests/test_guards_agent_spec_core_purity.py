"""Structural guard: importing ``neograph`` must not drag in ``pyagentspec``
(neograph-x75es, review finding F2).

## Why this file exists

``src/neograph/_agent_spec.py`` and ``spec_types.py`` both document that core
"stays Agent-Spec-free by default" and back it with function-local, import-guarded
``import pyagentspec...`` blocks that raise a fail-loud ``ConfigurationError``.

Until neograph-x75es, that claim was proven only ACCIDENTALLY: ``pyagentspec``
lived in ``[project.optional-dependencies].agent-spec``, so it was simply absent
from every default environment and a stray module-level import would have blown
up on sight. neograph-x75es moves the dep into ``[dependency-groups].dev`` so the
271 ``importorskip``-gated Agent Spec tests actually run under the project's own
gate -- which makes ``pyagentspec`` always importable and DESTROYS that accidental
proof. A module-level ``import pyagentspec`` regression in ``src/neograph`` would
then pass every gate silently.

Trading a documented safety claim for ergonomics, and re-admitting a silent
failure in the process, is exactly what the north star forbids. So the proof has
to become explicit before the dep becomes ambient -- this guard is a PRECONDITION
of that dependency move, not a follow-up to it.

The MCP family already solved the identical problem; this is deliberately the same
shape as ``tests/test_mcp_battery.py::test_neograph_core_imports_with_zero_mcp_deps``
(subprocess + ``sys.modules`` inspection), so the two purity claims read alike.

NOT importorskip-gated, by design: the guard is most load-bearing precisely when
``pyagentspec`` is absent, so it must run in every environment.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "neograph"

# The import roots the core must never pull in at import time.
AGENT_SPEC_ROOTS = ("pyagentspec",)


class TestNeographCoreIsAgentSpecFree:
    """``import neograph`` must not put pyagentspec into ``sys.modules``."""

    def test_neograph_core_imports_with_zero_agent_spec_deps(self):
        """Runs in a SUBPROCESS so the assertion sees a clean interpreter -- this
        test process has pyagentspec installed (that is the point of x75es), so an
        in-process check would be vacuous."""
        code = (
            "import sys\n"
            "import neograph\n"
            "leaked = [m for m in sys.modules if m.split('.')[0] == 'pyagentspec']\n"
            "assert not leaked, f'neograph core dragged in Agent Spec deps: {leaked}'\n"
            "print('CORE_AGENT_SPEC_FREE_OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, (
            "core-is-agent-spec-free check failed:\n"
            f"STDOUT:{result.stdout}\nSTDERR:{result.stderr}\n"
            "A module-level `import pyagentspec` crept into src/neograph. Move it "
            "function-local behind the ConfigurationError import guard (see "
            "_agent_spec.py::_import_agent_spec_classes)."
        )
        assert "CORE_AGENT_SPEC_FREE_OK" in result.stdout

    def test_no_module_level_pyagentspec_import_in_src(self):
        """Static twin of the subprocess check: catches a module-level import even
        in a src module that ``import neograph`` does not happen to reach."""
        offenders: list[str] = []
        for path in sorted(SRC_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in tree.body:  # module level ONLY -- function-local is the sanctioned form
                roots: list[str] = []
                if isinstance(node, ast.Import):
                    roots = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    roots = [node.module.split(".")[0]]
                if any(r in AGENT_SPEC_ROOTS for r in roots):
                    offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}")
        assert not offenders, (
            "module-level pyagentspec import in src/neograph (must be function-local "
            f"behind the fail-loud import guard): {offenders}"
        )

    def test_meta_static_scanner_detects_a_module_level_import(self):
        """Non-vacuity: the static scanner really fires on the shape it bans."""
        tree = ast.parse("import pyagentspec.flows.flow as f\n")
        roots = [a.name.split(".")[0] for a in tree.body[0].names]  # type: ignore[attr-defined]
        assert any(r in AGENT_SPEC_ROOTS for r in roots)

    def test_meta_static_scanner_ignores_function_local_import(self):
        """Negative: the sanctioned function-local form is NOT flagged (the scanner
        walks ``tree.body`` only, never nested bodies)."""
        source = "def f():\n    import pyagentspec.flows.flow as m\n    return m\n"
        tree = ast.parse(source)
        module_level_imports = [n for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom)]
        assert module_level_imports == []
