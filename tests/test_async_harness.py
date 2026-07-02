"""Harness self-tests for the Phase 0 dual-surface scaffolding (neograph-w74k.1).

Proves the scaffolding is wired and functional TODAY (with the async cell inert
until Phase 1):

  * ``run_driver`` (conftest) drives a real pipeline on the sync surface; the
    async cell auto-skips on ``hasattr(neograph, "arun")`` — NOT gated on arun
    semantics that do not exist yet (architecture-review M5).
  * ``GatedAsyncFake`` (fakes) parks N concurrent ``ainvoke`` coroutines on one
    loop and releases them — the primitive the Phase-1 real-concurrency and
    cancellation E2Es build on (parity cannot prove concurrency — M1).
  * ``event_loop_lag_watchdog`` (fakes) measures event-loop lag — the primitive
    the Phase-1 blocking-detector E2E builds on (parity cannot prove
    non-blocking — M1).

These EXERCISE the scaffolding so it is not dead code and so a future regression
in the harness itself is caught. Full run==arun parity and the concurrency /
blocking / cancellation E2Es land with Phase 1.
"""

from __future__ import annotations

import asyncio
import types

from neograph import compile, construct_from_module, node
from tests.fakes import (
    GatedAsyncFake,
    build_test_compile_kwargs,
    event_loop_lag_watchdog,
)
from tests.schemas import Claims, RawText


def _trivial_pipeline():
    """A minimal two-node scripted pipeline: fetch -> process."""
    mod = types.ModuleType("async_harness_mod")

    @node(mode="scripted", outputs=RawText)
    def fetch() -> RawText:
        return RawText(text="hello")

    @node(mode="scripted", outputs=Claims)
    def process(fetch: RawText) -> Claims:
        return Claims(items=[fetch.text.upper()])

    mod.fetch = fetch
    mod.process = process
    return construct_from_module(mod, name="async-harness")


class TestRunDriverFixture:
    """The parametrized dual-surface driver runs a real pipeline."""

    def test_driver_runs_pipeline_on_the_parametrized_surface(self, run_driver):
        """Single test body runs under both surfaces; the async cell skips until
        Phase 1, the sync cell produces the pipeline result."""
        graph = compile(_trivial_pipeline(), **build_test_compile_kwargs())
        result = run_driver(graph, input={"node_id": "async-harness-001"})
        assert result["process"] == Claims(items=["HELLO"])


class TestGatedAsyncFake:
    """The awaitable-gated primitive parks and releases concurrent coroutines."""

    async def test_n_concurrent_ainvokes_park_until_released(self):
        """N ainvoke coroutines all reach the gate (interleave on one loop) and
        none completes until release() — proving real concurrency is observable,
        which run==arun parity structurally cannot prove (M1)."""
        fake = GatedAsyncFake()
        tasks = [asyncio.create_task(fake.ainvoke([])) for _ in range(4)]

        # All four park at the gate before any completes.
        while fake.enter_count < 4:
            await fake.wait_entered()
            await asyncio.sleep(0)
        assert not any(t.done() for t in tasks)

        fake.release()
        results = await asyncio.gather(*tasks)
        assert len(results) == 4
        assert all(r.content == "done" for r in results)


class TestEventLoopLagWatchdog:
    """The watchdog measures event-loop lag from a blocking call."""

    async def test_watchdog_records_lag_when_loop_is_blocked(self):
        """A synchronous sleep inside the loop blocks the heartbeat; the watchdog
        records lag above its interval. This is the primitive the Phase-1
        blocking-detector E2E asserts on (parity cannot catch a blocked loop —
        LangGraph threadpools truly-sync nodes, masking it)."""
        import time

        async with event_loop_lag_watchdog(interval_s=0.01) as handle:
            time.sleep(0.15)  # BLOCK the loop synchronously
            await asyncio.sleep(0.02)  # let a heartbeat land late

        assert handle.max_lag > 0.05

    async def test_watchdog_stays_quiet_on_a_cooperative_loop(self):
        """With no blocking, lag stays small — the generous default threshold
        keeps CI non-flaky."""
        async with event_loop_lag_watchdog(interval_s=0.01) as handle:
            for _ in range(5):
                await asyncio.sleep(0.01)

        assert handle.max_lag < 0.5
