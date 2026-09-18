"""Compiler safety net — parametrized fixtures that verify neograph catches errors.

Inspired by rustc's compiletest: each fixture is a self-contained Python file
with a `# CHECK_ERROR: <regex>` comment for should_fail fixtures.

    tests/check_fixtures/
        should_fail/   — each file has a known defect, must raise during import or compile
        should_pass/   — each file is valid, must compile without errors; a file that
                         declares a module-level ``EXPECT`` is also RUN and its values
                         asserted

To add a new test case: create a .py file in the right directory.
The test harness discovers it automatically.

``EXPECT`` (neograph-36302) is the tier's second question. "It compiles" is an
absence-of-exception assertion, so a fixture could assemble the right graph,
deliver the WRONG VALUE at run time, and stay green. A fixture whose point is a
resolved value — which producer an input port read, which member is the boundary,
which peer a mesh handed off to — declares the state fields it must resolve to and
is executed. Opt-in rather than blanket: a fixture needing an LLM, credentials or a
driver this harness lacks declares nothing, so no skip is ever introduced.
"""

from __future__ import annotations

import contextlib
import importlib
import re
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from tests.fakes import build_test_compile_kwargs

FIXTURES = Path(__file__).parent / "check_fixtures"
SHOULD_FAIL = sorted(FIXTURES.glob("should_fail/*.py"))
SHOULD_PASS = sorted(FIXTURES.glob("should_pass/*.py"))

# The module-level name a should_pass fixture uses to declare the state values it
# must resolve to. Its presence is what opts the fixture into execution.
EXPECT_ATTR = "EXPECT"


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
    from neograph.compiler import compile
    from neograph.construct import Construct

    placeholder_llm_kwargs = _placeholder_llm_kwargs()

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


def _placeholder_llm_kwargs() -> dict:
    """Compile kwargs that let a fixture past the LLM fail-loud check.

    A placeholder checkpointer rides along so an Operator-carrying fixture
    (which always requires one to compile, regardless of its condition) isn't
    universally excluded from this harness -- harmless for fixtures that don't
    need one, though it does mean a RUN must supply a ``thread_id``.
    """
    from langgraph.checkpoint.memory import MemorySaver

    return {
        "llm_factory": lambda tier: None,
        "prompt_compiler": lambda template, data, **kw: [],
        "checkpointer": MemorySaver(),
    }


def _nested_construct_names(construct: object) -> set[str]:
    """Every Construct name reachable BELOW ``construct`` (excluding itself).

    Walks through branch arms via ``iter_with_arms`` -- the sanctioned
    arm-descent primitive -- so a sub-construct parked inside an arm is not
    mistaken for a second root.
    """
    from neograph._ir_branch import iter_with_arms
    from neograph.construct import Construct

    found: set[str] = set()
    stack = [construct]
    while stack:
        for item in iter_with_arms(stack.pop()):
            if isinstance(item, Construct):
                found.add(item.name)
                stack.append(item)
    return found


def _root_construct(mod: object, fixture_name: str):
    """The one module-level Construct that no other module-level Construct contains.

    Matched by NAME, not identity: ``sub | Each(...)`` pipes a ``model_copy``,
    so the parent holds a different object with the same name. Two roots is a
    LOUD failure rather than a guess about which one the fixture meant.
    """
    from neograph.construct import Construct

    module_level: list[Construct] = []
    seen: set[int] = set()
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, Construct) and id(obj) not in seen:
            seen.add(id(obj))
            module_level.append(obj)

    contained: set[str] = set()
    for candidate in module_level:
        contained |= _nested_construct_names(candidate)
    roots = [c for c in module_level if c.name not in contained]

    assert len(roots) == 1, (
        f"Fixture {fixture_name} declares {EXPECT_ATTR} but has {len(roots)} root constructs "
        f"({[c.name for c in roots]}). A fixture that declares expectations must have exactly one "
        f"top-level construct, so the harness knows which one to run."
    )
    return roots[0]


def _assert_declared_expectations(mod: object, fixture_path: Path) -> None:
    """Run a fixture that declares ``EXPECT`` and assert the values it claims.

    Opt-in by design (neograph-36302). A fixture whose point is a valid SHAPE
    declares nothing and is compiled exactly as before; a fixture whose point is
    a resolved VALUE -- which producer an input port read, which member is the
    boundary, which peer a mesh handed off to -- declares the state fields it
    expects and is executed. Opt-in rather than blanket, because some fixtures
    need an LLM, credentials or a driver this harness does not have, and forcing
    a run on those would reintroduce skips.
    """
    expect = getattr(mod, EXPECT_ATTR, None)
    if expect is None:
        return  # Shape-only fixture: compile is the whole question.

    from neograph.compiler import compile
    from neograph.runner import run

    assert isinstance(expect, dict) and expect, (
        f"Fixture {fixture_path.name}: {EXPECT_ATTR} must be a non-empty dict of "
        f"{{state_field: expected_value}}, got {expect!r}."
    )

    root = _root_construct(mod, fixture_path.name)
    graph = compile(root, **_placeholder_llm_kwargs(), **build_test_compile_kwargs())
    result = run(graph, input={}, config={"configurable": {"thread_id": f"check-fixture-{uuid4()}"}})

    for field, expected in expect.items():
        assert field in result, (
            f"Fixture {fixture_path.name}: {EXPECT_ATTR} names state field {field!r}, "
            f"which the run never produced. Produced: {sorted(result)}."
        )
        assert result[field] == expected, (
            f"Fixture {fixture_path.name}: state field {field!r} resolved to {result[field]!r}, "
            f"but the fixture declares {expected!r}. The graph assembled correctly and still "
            f"delivered the wrong value."
        )


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

        # And, for a fixture that declares what it should RESOLVE TO, run it.
        _assert_declared_expectations(mod, fixture_path)


# =============================================================================
# The harness's own regression tests (neograph-36302)
# =============================================================================


class TestShouldPassExecutesDeclaredExpectations:
    """A should_pass fixture that declares ``EXPECT`` is RUN and its values asserted.

    Regression for neograph-36302: ``test_should_pass`` asserted only
    absence-of-exception, so a fixture could assemble the right graph, deliver
    the WRONG VALUE at run time, and stay green. The tier could not distinguish
    "this pipeline is valid" from "this pipeline is valid and means what it says".
    """

    _FIXTURE_SRC = '''\
"""Synthetic should_pass fixture, written by the harness's own regression tests."""

from pathlib import Path

from pydantic import BaseModel

from neograph import Construct, Node
from neograph._runtime_registry import register_scripted


class Alpha(BaseModel, frozen=True):
    tag: str = "a"


def _source(_i, _c):
    Path({sentinel!r}).write_text("ran")
    return Alpha(tag="FIRST")


register_scripted("{prefix}_source", _source)

pipeline = Construct(
    "{prefix}-synthetic",
    nodes=[Node.scripted("source", fn="{prefix}_source", outputs=Alpha)],
)

EXPECT = {{"source": Alpha(tag={expected!r})}}
'''

    def _write_fixture(self, tmp_path: Path, *, prefix: str, expected: str, sentinel: Path) -> Path:
        path = tmp_path / f"{prefix}_synthetic.py"
        path.write_text(
            self._FIXTURE_SRC.format(prefix=prefix, expected=expected, sentinel=str(sentinel)),
        )
        return path

    def test_should_pass_fails_when_the_declared_expectation_is_wrong(self, tmp_path: Path):
        """The node produces FIRST; the fixture declares SECOND. That must be red."""
        sentinel = tmp_path / "ran.txt"
        path = self._write_fixture(tmp_path, prefix="wrongexp", expected="SECOND", sentinel=sentinel)

        with pytest.raises(AssertionError):
            test_should_pass(path)

    def test_should_pass_actually_executes_a_fixture_that_declares_expectations(self, tmp_path: Path):
        """Compiling is not running -- the node body must have been invoked."""
        sentinel = tmp_path / "ran.txt"
        path = self._write_fixture(tmp_path, prefix="rightexp", expected="FIRST", sentinel=sentinel)

        test_should_pass(path)

        assert sentinel.exists(), "Fixture declared EXPECT but its node body never ran -- the harness only compiled it."
