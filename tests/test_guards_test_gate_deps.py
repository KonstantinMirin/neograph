"""Structural guard: every ``pytest.importorskip`` gate in the suite is backed
by a dependency the PROJECT'S OWN DEFAULT GATE installs (neograph-x75es).

## The disease

``pytest.importorskip("X")`` at module scope is a *silent* switch: when ``X`` is
absent the whole module yields ZERO tests, pytest reports green, and nothing
compares the collected count against anything. That is fine when the optionality
is deliberate and documented; it is an existential defect when it is an
accident, because the local gate is the ONLY gate this repo has
(``.github/workflows/`` contains ``publish.yml`` and no test workflow).

It bit for real: the entire Agent Spec proof surface (~271 tests across 14
files) was gated on ``pyagentspec``, which was declared ONLY in
``[project.optional-dependencies].agent-spec``. Neither ``make quality``
(bare ``uv run pytest``) nor the AGENTS.md-documented ``uv run --extra dev
pytest`` installs that extra, so on any clean checkout those 271 tests never
ran and the gate stayed green.

## The invariant this pins

For every dependency-conditional test gate under ``tests/``, the backing
distribution MUST be declared in ``[dependency-groups].dev`` -- the group uv
installs for the bare ``uv run pytest`` command -- UNLESS the module is listed
in :data:`GATED_IMPORT_EXEMPTIONS` with a structural reason AND the named
optional extra really declares it (non-vacuity, mirroring
``TestOptionalExtraImportRoots`` in ``test_guards_examples.py``).

Adding a new gate therefore forces an explicit choice: either the dep joins the
default gate, or you write down why it must not.

## Both spellings, one table

The disease has two forms in live use, and pinning only the first would leave
half of it open (review finding F4):

* ``pytest.importorskip("x")`` -- **SILENT**. The module yields ZERO tests and
  the summary line says nothing. This is the form that hid the Agent Spec suite.
* ``importlib.util.find_spec("x")`` feeding a ``pytest.mark.skipif(not _HAS_X)``
  -- **LOUD**. Tests still collect and the summary reports them as skipped.

Both route through the SAME :data:`GATED_IMPORT_EXEMPTIONS` table. Today every
``find_spec`` site happens to be MCP (already exempt), so a detector covering
only ``importorskip`` would be green by luck and the next non-MCP ``skipif``
gate would walk straight past it.

This module uses PURE AST + ``tomllib`` (no ``re`` at all), so it is exempt by
construction from the named-regex/slip-test discipline in ``test_guards_meta``.
"""

from __future__ import annotations

import ast
import pathlib
import tomllib

TESTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Modules whose importorskip gate may stay OUTSIDE the default dev group.
# Each entry: import root -> (optional extra that declares it, distribution
# name, structural reason). A reason is mandatory and non-vacuity is asserted:
# the exemption cannot outlive the declaration it names.
GATED_IMPORT_EXEMPTIONS: dict[str, tuple[str, str, str]] = {
    "mcp": (
        "mcp-examples",
        "mcp",
        "The no-session-ownership guard keeps src/neograph MCP-free, which "
        "depends on the MCP stack staying an optional extra. AGENTS.md documents "
        "the run command ('no-key != no extra'): "
        "uv run --extra mcp-examples pytest tests/test_mcp_examples_e2e.py",
    ),
    "langchain_mcp_adapters": (
        "mcp-examples",
        "langchain-mcp-adapters",
        "Same structural reason as 'mcp' above -- the adapters ship with the MCP "
        "stack that must stay out of the core install.",
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# Detector -- pure AST
# ════════════════════════════════════════════════════════════════════════════
# The two call forms that gate tests on a dependency's presence. Keyed by the
# callee's bare attribute/name so `pytest.importorskip` / `importorskip` and
# `importlib.util.find_spec` / `find_spec` all match.
_GATE_CALLS = frozenset({"importorskip", "find_spec"})


def _gated_import_roots(source: str) -> set[str]:
    """Return every module name that ``source`` gates a test on.

    Covers BOTH forms: ``pytest.importorskip("x")`` (silent) and
    ``importlib.util.find_spec("x")`` feeding a ``skipif`` flag (loud), in their
    dotted and bare-import spellings. Only string-literal first args count -- a
    computed module name is not a static gate.
    """
    roots: set[str] = set()
    for call in ast.walk(ast.parse(source)):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if name not in _GATE_CALLS or not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            roots.add(first.value)
    return roots


def _iter_test_sources() -> list[pathlib.Path]:
    """Every .py file under tests/ (including package subdirs and helpers)."""
    return sorted(p for p in TESTS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _declared_dists(specs: list[str]) -> set[str]:
    """Normalize PEP 508 requirement strings to bare distribution names."""
    return {spec.split(">")[0].split("<")[0].split("=")[0].split("[")[0].split(";")[0].strip() for spec in specs}


def _default_gate_dists() -> set[str]:
    """Distributions installed by the bare ``uv run pytest`` command.

    That is ``[dependency-groups].dev`` (uv's default group) plus the core
    ``[project].dependencies`` -- NOT ``[project.optional-dependencies]``,
    which requires an explicit ``--extra``.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    return _declared_dists(data["dependency-groups"]["dev"]) | _declared_dists(data["project"]["dependencies"])


def _dist_candidates(import_root: str) -> set[str]:
    """Plausible distribution names for an import root (PEP 503-ish)."""
    return {import_root, import_root.replace("_", "-"), import_root.replace("-", "_")}


# ════════════════════════════════════════════════════════════════════════════
# The guard
# ════════════════════════════════════════════════════════════════════════════
class TestImportOrSkipGatesRunUnderDefaultGate:
    """Every dependency gate is either default-installed or explicitly exempt."""

    def test_every_gated_dep_is_in_the_default_dev_group(self):
        installed = _default_gate_dists()
        offenders: dict[str, list[str]] = {}
        for path in _iter_test_sources():
            for root in _gated_import_roots(path.read_text()):
                if root in GATED_IMPORT_EXEMPTIONS:
                    continue
                if _dist_candidates(root) & installed:
                    continue
                offenders.setdefault(root, []).append(path.relative_to(REPO_ROOT).as_posix())
        assert not offenders, (
            "dependency-conditional test gates (importorskip / find_spec->skipif) "
            "whose dependency the DEFAULT gate (`uv run pytest` -> "
            "[dependency-groups].dev) does not install. Under importorskip these "
            "tests collect ZERO on a clean checkout while the gate still reports "
            "green. Either add the dist to [dependency-groups].dev, or add the "
            "import root to GATED_IMPORT_EXEMPTIONS with a structural reason: "
            + "; ".join(f"{root} <- {sorted(files)}" for root, files in sorted(offenders.items()))
        )

    def test_default_gate_owns_its_own_toolchain(self):
        """neograph-x75es rows 2-5: the gate's runner, async plugin, linter and
        type checker must be DECLARED in the group the gate installs -- not left
        to an extra (invisible) or to a transitive pull (accidental).

        pytest-asyncio's absence cost 51 async failures on a clean checkout;
        pytest and ruff only resolved because pytest-cov / pytest-examples happen
        to depend on them, and mypy resolved only via a PATH-binary fallback."""
        declared = _declared_dists(tomllib.loads(PYPROJECT.read_text())["dependency-groups"]["dev"])
        missing = {"pytest", "pytest-asyncio", "ruff", "mypy"} - declared
        assert not missing, (
            f"quality-gate toolchain not declared in [dependency-groups].dev: {sorted(missing)}. "
            "`make quality` runs pytest + ruff + mypy; each must be installed by the "
            "same command that runs the gate."
        )

    def test_there_is_no_second_dev_namespace(self):
        """Single source of truth: a `dev` EXTRA alongside the `dev` GROUP is the
        ambiguity that caused this bug -- two plausible homes for a dev dep, only
        one of which the gate reads. Keep exactly one."""
        extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
        assert "dev" not in extras, (
            "[project.optional-dependencies].dev is back. Dev dependencies belong in "
            "[dependency-groups].dev, which `uv run pytest` installs; an extra is "
            "invisible to the gate unless someone remembers `--extra dev`."
        )

    def test_exemptions_are_non_vacuous(self):
        """Each exemption names a real extra that really declares the dist, and
        carries a non-empty reason -- the exemption cannot outlive either."""
        extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
        for root, (extra_name, dist, reason) in GATED_IMPORT_EXEMPTIONS.items():
            assert extra_name in extras, f"exempt root {root!r} names missing extra {extra_name!r}"
            declared = _declared_dists(extras[extra_name])
            assert dist in declared, f"exempt root {root!r} -> dist {dist!r} not in extra {extra_name!r}: {declared}"
            assert reason.strip(), f"exempt root {root!r} has an empty reason"

    def test_exemptions_are_all_still_used(self):
        """No dead exemptions: every exempt root is actually gated somewhere."""
        gated: set[str] = set()
        for path in _iter_test_sources():
            gated |= _gated_import_roots(path.read_text())
        stale = set(GATED_IMPORT_EXEMPTIONS) - gated
        assert not stale, f"GATED_IMPORT_EXEMPTIONS entries with no dependency gate left: {sorted(stale)}"


class TestImportOrSkipDetector:
    """Meta-tests: the detector cannot be slipped by a spelling variant."""

    def test_meta_detects_attribute_call(self):
        assert _gated_import_roots('import pytest\npytest.importorskip("pyagentspec")\n') == {"pyagentspec"}

    def test_meta_detects_bare_call(self):
        source = 'from pytest import importorskip\nimportorskip("mcp")\n'
        assert _gated_import_roots(source) == {"mcp"}

    def test_meta_detects_call_with_reason_kwarg(self):
        source = 'import pytest\npytest.importorskip("mcp", reason="needs the extra")\n'
        assert _gated_import_roots(source) == {"mcp"}

    def test_meta_detects_find_spec_skipif_form(self):
        """F4: the LOUD spelling -- find_spec feeding a skipif flag -- is a gate too."""
        source = (
            "import importlib.util, pytest\n"
            '_HAS_X = bool(importlib.util.find_spec("some_dep"))\n'
            '_requires_x = pytest.mark.skipif(not _HAS_X, reason="needs it")\n'
        )
        assert _gated_import_roots(source) == {"some_dep"}

    def test_meta_detects_bare_find_spec(self):
        source = 'from importlib.util import find_spec\n_HAS_X = bool(find_spec("some_dep"))\n'
        assert _gated_import_roots(source) == {"some_dep"}

    def test_meta_ignores_docstring_mentions(self):
        """Prose about importorskip is not a gate -- text scanning would false-positive."""
        assert _gated_import_roots('"""Gated with pytest.importorskip("nope") in prose."""\n') == set()

    def test_meta_ignores_non_literal_module(self):
        source = 'import pytest\nmod = "x"\npytest.importorskip(mod)\n'
        assert _gated_import_roots(source) == set()

    def test_meta_ignores_unrelated_calls(self):
        """Negative: a same-named-ish call that is not a dependency gate is ignored."""
        assert _gated_import_roots('import importlib\nimportlib.import_module("json")\n') == set()

    def test_meta_would_flag_an_undeclared_gate(self):
        """Non-vacuity of the main guard: an undeclared root is not in the
        default gate's dists, so the offender branch is reachable."""
        assert not (_dist_candidates("totally_not_installed_pkg") & _default_gate_dists())
