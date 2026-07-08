"""Test fixtures — registry isolation between tests."""

import asyncio
import hashlib
import subprocess
from pathlib import Path

import pytest
import structlog

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src" / "neograph"


def _src_tree_signature() -> str | None:
    """Hash the current uncommitted state of git-tracked ``src/neograph``.

    Combines ``git diff`` (tracked-content changes vs HEAD) and
    ``git status --porcelain`` (also surfaces new/deleted files) so any
    mutation of the source tree — modification, creation, or deletion —
    changes the digest. Returns ``None`` when git is unavailable so the
    guard degrades to a no-op instead of failing the suite.
    """
    try:
        diff = subprocess.run(
            ["git", "diff", "--", "src/neograph"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "src/neograph"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256((diff + "\x00" + status).encode("utf-8", "surrogatepass")).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def _src_tree_unmodified_by_suite():
    """Permanent alarm: the test run must not mutate git-tracked ``src/neograph``.

    A source-mutating guard test that skips its finally-restore once leaked a
    MUTATION-PROBE line into the real ``naming.py`` (TQ-03). This session-scoped
    fixture captures a signature of the source tree's uncommitted state at the
    START of the run and re-checks it at the END, failing loudly on any delta.

    It compares a session-start SNAPSHOT against session-end — NOT ``git diff``
    against HEAD — so the legitimate uncommitted changes already present in the
    working tree do not trip it; only a change introduced *by the test run*
    does. Degrades to a no-op when git is unavailable.
    """
    before = _src_tree_signature()
    yield
    if before is None:
        return
    after = _src_tree_signature()
    assert after == before, (
        "The test run modified git-tracked source under src/neograph/. A "
        "source-mutating guard/verification test likely skipped its "
        "finally-restore. Inspect `git status src/neograph` and ensure every "
        "test that writes real source files restores them in a finally block "
        "(or writes to tmp_path instead)."
    )


@pytest.fixture(autouse=True)
def _clean_registries():
    """Clear test-local registries before each test.

    The decoration-time shim registry (inline body-merge / interrupt_when /
    @merge_fn / @tool) lives in the leaf `_runtime_registry` and is reset via
    its own `reset()` — no hand-maintained `.clear()` block reaching into
    `decorators.py` internals (neograph-v3xx HIGH-01). The test-side helpers in
    `tests/fakes.py` (register_scripted etc.) write into test-local dicts that
    `reset_test_registry()` clears.
    """
    from neograph import _run_cache, _runtime_registry
    from neograph._sidecar import _merge_fn_registry
    from neograph.spec_types import _type_registry
    from tests.fakes import reset_test_registry

    reset_test_registry()
    _runtime_registry.reset()
    _merge_fn_registry.clear()
    _type_registry.clear()
    # Drop any per-run handle/resource cache entries (keyed on RUN_ID) so a test
    # session cannot accumulate them (neograph-m6d3.8 / neograph-43do).
    _run_cache.clear()
    # Reset structlog to defaults so tests that capture warnings via stdout
    # (capsys) are not affected by an earlier test's reconfigure (e.g.
    # tests that route structlog through stdlib logging).
    structlog.reset_defaults()
    yield


# ═══════════════════════════════════════════════════════════════════════════
# Dual-surface driver fixture (neograph-w74k.1, Phase 0 — SCAFFOLDING)
#
# Runs one behavioral test through both execution surfaces so a single test body
# asserts identical behavior under run() and (future) arun(). Parametrized on
# ["sync", "async"]; the "async" cell is INERT today and auto-activates when
# Phase 1 lands arun() — it is skip-guarded on hasattr(neograph, "arun"), NOT
# gated on arun semantics that do not exist yet (architecture-review M5).
#
# THE 6-CELL POLICY (architecture-review M1/M7). This is a 3-API-surface x
# sync/async grid, not a doubling:
#   * Full 6-cell (declarative / @node / programmatic  x  sync / async) is
#     MANDATORY only for IR-level behavioral changes (node.py,
#     _construct_validation.py, factory.py, state.py).
#   * A representative SUBSET suffices for execution-surface-agnostic logic
#     (rendering, validation, describe_type, lint).
#   * Target the async doubling at I/O-bearing (LLM / tool) tests. Pure/scripted
#     nodes are threadpooled by LangGraph under arun, giving near-zero async
#     signal — doubling them just burns wall-clock.
#
# PARITY IS NECESSARY BUT NOT SUFFICIENT (M1). This fixture proves plumbing:
# that a pipeline produces the same result on both surfaces. It does NOT prove
# concurrency / ordering / event-loop non-blocking / cancellation. Those need
# the dedicated primitives in tests/fakes.py (GatedAsyncFake,
# event_loop_lag_watchdog) and explicit `async def test_*` cases — NOT this
# per-call driver (real-concurrency means N runs on ONE loop; this fixture does
# one asyncio.run per call).
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(params=["sync", "async"])
def run_driver(request):
    """Yield a callable ``driver(graph, **run_kwargs)`` for the parametrized
    execution surface.

    "sync"  -> neograph.run(graph, **kwargs)
    "async" -> asyncio.run(neograph.arun(graph, **kwargs))   [skipped until Phase 1]

    Usage::

        def test_pipeline_result(run_driver):
            graph = compile(construct, **build_test_compile_kwargs(...))
            result = run_driver(graph, input={...})
            assert result[...] == ...

    The single test body then runs under both surfaces automatically.
    """
    import neograph

    surface = request.param
    if surface == "async":
        if not hasattr(neograph, "arun"):
            pytest.skip("arun() not implemented yet (Phase 1) — async cell inert")

        def _async_driver(graph, **run_kwargs):
            return asyncio.run(neograph.arun(graph, **run_kwargs))

        yield _async_driver
    else:

        def _sync_driver(graph, **run_kwargs):
            return neograph.run(graph, **run_kwargs)

        yield _sync_driver
