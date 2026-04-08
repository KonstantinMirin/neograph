"""Compiler safety net — parametrized fixtures that verify neograph catches errors.

Inspired by rustc's compiletest: each fixture is a self-contained Python file
with a `# CHECK_ERROR: <regex>` comment for should_fail fixtures.

    tests/check_fixtures/
        should_fail/   — each file has a known defect, must raise during import or compile
        should_pass/   — each file is valid, must compile without errors

To add a new test case: create a .py file in the right directory.
The test harness discovers it automatically.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "check_fixtures"
SHOULD_FAIL = sorted(FIXTURES.glob("should_fail/*.py"))
SHOULD_PASS = sorted(FIXTURES.glob("should_pass/*.py"))


def _load_fixture(path: Path) -> tuple[object | None, Exception | None]:
    """Import a fixture module, return (module, None) or (None, exception)."""
    mod_name = f"_check_fixture_{path.stem}"
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Clean up any previous import
    sys.modules.pop(mod_name, None)

    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as exc:
        return None, exc


def _extract_error_pattern(path: Path) -> str | None:
    """Extract the CHECK_ERROR regex from the first comment line."""
    for line in path.read_text().splitlines():
        if line.startswith("# CHECK_ERROR:"):
            return line.split(":", 1)[1].strip()
    return None


def _try_compile(mod: object) -> Exception | None:
    """Find Constructs in the module and try to compile them."""
    from neograph.compiler import compile
    from neograph.construct import Construct

    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, Construct):
            try:
                compile(obj)
            except Exception as exc:
                return exc
    return None


# =============================================================================
# Parametrized tests
# =============================================================================


@pytest.mark.parametrize(
    "fixture_path",
    SHOULD_FAIL,
    ids=[p.stem for p in SHOULD_FAIL],
)
def test_should_fail(fixture_path: Path):
    """Fixture must raise during import or compile, matching CHECK_ERROR pattern."""
    pattern = _extract_error_pattern(fixture_path)

    mod, import_error = _load_fixture(fixture_path)

    if import_error is not None:
        # Error during import (e.g., ConstructError at assembly time)
        if pattern:
            assert re.search(pattern, str(import_error), re.IGNORECASE), (
                f"Import raised {type(import_error).__name__}: {import_error}\n"
                f"but didn't match pattern: {pattern}"
            )
        return  # error caught, test passes

    # Module imported OK — try compiling
    compile_error = _try_compile(mod)
    assert compile_error is not None, (
        f"Fixture {fixture_path.name} should have raised an error "
        f"during import or compile, but didn't."
    )

    if pattern:
        assert re.search(pattern, str(compile_error), re.IGNORECASE), (
            f"Compile raised {type(compile_error).__name__}: {compile_error}\n"
            f"but didn't match pattern: {pattern}"
        )


@pytest.mark.parametrize(
    "fixture_path",
    SHOULD_PASS,
    ids=[p.stem for p in SHOULD_PASS],
)
def test_should_pass(fixture_path: Path):
    """Fixture must import and compile without errors."""
    mod, import_error = _load_fixture(fixture_path)
    assert import_error is None, (
        f"Fixture {fixture_path.name} should import cleanly but raised: {import_error}"
    )

    compile_error = _try_compile(mod)
    assert compile_error is None, (
        f"Fixture {fixture_path.name} should compile cleanly but raised: {compile_error}"
    )
