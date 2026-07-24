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

import contextlib
import importlib
import re
import sys
from pathlib import Path

import pytest

from tests.fakes import build_test_compile_kwargs

FIXTURES = Path(__file__).parent / "check_fixtures"
SHOULD_FAIL = sorted(FIXTURES.glob("should_fail/*.py"))
SHOULD_PASS = sorted(FIXTURES.glob("should_pass/*.py"))


@contextlib.contextmanager
def _isolated_registries():
    """Execute a fixture's import+compile against a private registry snapshot.

    Fixture modules register into GLOBAL registries at *exec* time — scripted
    fns via ``register_scripted`` and, critically, ``@merge_fn`` shims into the
    process-wide ``_merge_fn_registry``. Two fixtures each defining a
    ``@merge_fn def combine`` at DIFFERENT def sites trip that registry's
    fail-loud different-site collision guard the moment both are present at once.

    The autouse ``_clean_registries`` conftest fixture clears these dicts only at
    each test's SETUP, so it does NOT prevent a leaked ``combine`` from a
    neighbor (registrations persist past a test — there is no teardown clear)
    from being live when this fixture execs. Under pytest-randomly ordering (and
    per-process hash-seed variation) that leak-forward intermittently ERRORs a
    check_fixture that is itself perfectly valid (neograph-cfp7).

    This wraps each fixture's load+compile in a snapshot: every touched registry
    is saved and CLEARED before exec (so the fixture never sees a neighbor's
    residue) and RESTORED on exit (so the fixture's own registrations never leak
    forward to poison a neighbor). Order-independent by construction — no
    ``-p no:randomly`` pin.
    """
    from neograph._runtime_registry import _decoration_registry
    from neograph._sidecar import _merge_fn_caller_ns, _merge_fn_registry
    from neograph.spec_types import _type_registry
    from tests.fakes import _TEST_CONDITIONS, _TEST_SCRIPTED, _TEST_TOOL_FACTORIES

    plain_dicts = (
        _merge_fn_registry,
        _merge_fn_caller_ns,
        _type_registry,
        _TEST_SCRIPTED,
        _TEST_CONDITIONS,
        _TEST_TOOL_FACTORIES,
    )
    saved = [dict(d) for d in plain_dicts]
    for d in plain_dicts:
        d.clear()
    # _decoration_registry holds the @merge_fn auto-registered scripted shim +
    # conditions/tool_factories; its own session() snapshots/clears/restores.
    with _decoration_registry.session():
        try:
            yield
        finally:
            for d, snap in zip(plain_dicts, saved, strict=True):
                d.clear()
                d.update(snap)


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


def _try_compile(mod: object, *, try_without_llm: bool = True) -> Exception | None:
    """Find Constructs in the module and try to compile them.

    Pass placeholder LLM kwargs so the §2 fail-loud check doesn't mask
    other expected failures (missing tool, unregistered merge_fn, etc.).
    Fixtures that test the LLM-kwargs-missing error itself supply their
    own pattern and don't need these placeholders to fire.

    ``try_without_llm`` (should_fail only): a second compile WITHOUT LLM kwargs,
    so a should_fail fixture whose expected error IS "LLM not configured" surfaces
    it (the first, placeholder-LLM compile would hide it). should_pass fixtures set
    this False — an LLM-mode node (e.g. an agent) legitimately requires runtime
    config, so it can only be expected to compile WITH the placeholder LLM.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from neograph.compiler import compile
    from neograph.construct import Construct

    placeholder_llm_kwargs = {
        "llm_factory": lambda tier: None,
        "prompt_compiler": lambda template, data, **kw: [],
        # A placeholder checkpointer so an Operator-carrying fixture (which
        # always requires one to compile, regardless of its condition) isn't
        # universally excluded from this harness -- harmless for fixtures
        # that don't need one.
        "checkpointer": MemorySaver(),
    }

    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, Construct):
            try:
                # First attempt: pass placeholder LLM kwargs so the LLM
                # check passes (most fixtures don't test that path).
                compile(obj, **placeholder_llm_kwargs, **build_test_compile_kwargs())
            except Exception as exc:
                return exc
            # Second attempt without LLM kwargs (in case the fixture's
            # expected error is "LLM not configured"). should_fail only.
            if try_without_llm:
                try:
                    compile(obj, **build_test_compile_kwargs())
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

    with _isolated_registries():
        mod, import_error = _load_fixture(fixture_path)

        if import_error is not None:
            # Error during import (e.g., ConstructError at assembly time)
            if pattern:
                assert re.search(pattern, str(import_error), re.IGNORECASE), (
                    f"Import raised {type(import_error).__name__}: {import_error}\nbut didn't match pattern: {pattern}"
                )
            return  # error caught, test passes

        # Module imported OK — try compiling
        compile_error = _try_compile(mod)
    assert compile_error is not None, (
        f"Fixture {fixture_path.name} should have raised an error during import or compile, but didn't."
    )

    if pattern:
        assert re.search(pattern, str(compile_error), re.IGNORECASE), (
            f"Compile raised {type(compile_error).__name__}: {compile_error}\nbut didn't match pattern: {pattern}"
        )


@pytest.mark.parametrize(
    "fixture_path",
    SHOULD_PASS,
    ids=[p.stem for p in SHOULD_PASS],
)
def test_should_pass(fixture_path: Path):
    """Fixture must import and compile without errors."""
    with _isolated_registries():
        mod, import_error = _load_fixture(fixture_path)
        assert import_error is None, f"Fixture {fixture_path.name} should import cleanly but raised: {import_error}"

        # should_pass fixtures compile once, WITH the placeholder LLM — an
        # LLM-mode node (agent/act/think) legitimately requires runtime config,
        # so the no-LLM second compile (a should_fail probe) must not gate
        # should_pass.
        compile_error = _try_compile(mod, try_without_llm=False)
    assert compile_error is None, f"Fixture {fixture_path.name} should compile cleanly but raised: {compile_error}"
