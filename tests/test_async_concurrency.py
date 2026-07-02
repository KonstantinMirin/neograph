"""Phase 1f M1 trio — CHARACTERIZATION for neograph-w74k.2.6.

The three properties ``run == arun`` parity structurally CANNOT prove, wired
around REAL ``neograph.arun``:

  * M1-a REAL CONCURRENCY: N independent runs interleave on ONE event loop (all
    park at their own LLM gate simultaneously, none completes early) and each
    returns its OWN isolated result.
  * M1-b EVENT-LOOP-LAG WATCHDOG: a node that BLOCKS the loop trips the watchdog;
    a cooperative node does not. This is the ONLY guard for the H2
    silently-blocking-twin failure mode — a twin that runs sync-blocking work on
    the loop returns the right result (passes parity) while starving the loop.
  * M1-c CANCELLATION E2E: a mid-flight ``arun`` task cancelled while parked at
    the LLM gate raises ``CancelledError`` cleanly and leaves the file-backed
    ``AsyncSqliteSaver`` reusable — re-``arun`` of the SAME ``thread_id``
    completes (proves checkpoint consistency + the saver connection was not torn
    down by the cancel).

CHARACTERIZATION, not TDD red — Phase 1a-1d already wired the async dispatch,
inline-on-loop scripted bodies, ``arun``, and the async checkpoint twins. These
PASS on first run and LOCK the concurrency/non-blocking/cancellation properties.
A FAIL is a real async runtime bug — report it loudly, do NOT paper over it.

WATCHDOG RATIONALE (refinement mybm.12, correcting the original note): post-1a
BOTH a plain sync ``time.sleep`` body and an ``async def`` body run INLINE on the
loop thread (``test_async_dual_path.py`` pins node_tid == loop_tid; LangGraph does
NOT threadpool @node scripted bodies here). The ``async def`` body is chosen for
EXPLICITNESS — it declares "this runs on the loop" — NOT to dodge threadpool
masking.

CANCELLATION RESUME (refinement mybm.12): the resume leg uses a FRESH non-gated
fake so it cannot re-park and hang; the ``AsyncSqliteSaver`` async-with is owned
by the TEST scope (not the cancelled task), so the cancel does not tear down the
connection.

NO MOCKS of the event loop. Reuses ``GatedAsyncFake`` /
``event_loop_lag_watchdog`` (tests/fakes.py) and file-backed ``AsyncSqliteSaver``
(user DECISION on this bead: real sqlite, not InMemory). Driver/runtime E2E — not
an IR-shape change, so the three-surface matrix does not apply.
"""

from __future__ import annotations

import asyncio
import time
import types as _types

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import neograph
from neograph import compile, construct_from_functions, construct_from_module, node
from tests.fakes import (
    GatedAsyncFake,
    StructuredFake,
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    event_loop_lag_watchdog,
)
from tests.schemas import Claims


def _think_graph(fake, *, checkpointer=None):
    """Compile a single-node THINK pipeline driven by ``fake``.

    Copies the wiring in ``test_async_llm_tool.py:146``. Each caller passes its
    OWN fake instance so ``enter_count`` / results stay isolated per graph."""

    @node(mode="think", outputs=Claims, model="fast", prompt="test/extract")
    def gen() -> Claims: ...

    extra = {"checkpointer": checkpointer} if checkpointer is not None else {}
    return compile(
        construct_from_functions("p", [gen]),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: fake),
        **extra,
    )


# ═══════════════════════════════════════════════════════════════════════════
# M1-a — REAL CONCURRENCY: N runs interleave on one loop, isolated results
# ═══════════════════════════════════════════════════════════════════════════


async def test_n_arun_interleave_on_one_loop_with_isolated_results():
    """N independent ``arun`` tasks each park at their OWN GatedAsyncFake gate on
    ONE loop (all ``enter_count == 1``, none done => genuine interleaving), and
    after release each returns its OWN distinct result (isolation)."""
    n = 4
    fakes = [GatedAsyncFake(lambda m, v=i: m(items=[f"g{v}"])) for i in range(n)]
    graphs = [_think_graph(fakes[i]) for i in range(n)]

    tasks = [
        asyncio.create_task(
            neograph.arun(graphs[i], input={"node_id": f"c{i}"})
        )
        for i in range(n)
    ]

    # Poll until every run has parked at its gate (all interleaved on the loop).
    for _ in range(300):
        if all(f.enter_count == 1 for f in fakes) or any(t.done() for t in tasks):
            break
        await asyncio.sleep(0.005)

    assert all(f.enter_count == 1 for f in fakes), (
        f"not all runs parked concurrently: enter_counts="
        f"{[f.enter_count for f in fakes]}"
    )
    assert not any(t.done() for t in tasks), (
        "a run completed before any gate was released — the awaits did not "
        "actually interleave on the loop"
    )

    for f in fakes:
        f.release()

    results = await asyncio.gather(*tasks)
    for i, r in enumerate(results):
        assert r["gen"] == Claims(items=[f"g{i}"]), (
            f"run {i} did not return its OWN result — concurrent runs "
            f"cross-contaminated: {r['gen']}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# M1-b — EVENT-LOOP-LAG WATCHDOG: blocking body trips it, cooperative does not
# ═══════════════════════════════════════════════════════════════════════════


def _one_scripted_body_graph(name: str, body):
    """Compile a single scripted @node whose async body is ``body`` (a
    zero-arg coroutine function returning a ``Claims``)."""
    mod = _types.ModuleType(f"test_async_wd_{name}_mod")
    decorated = node(mode="scripted", outputs=Claims)(body)
    setattr(mod, name, decorated)
    return compile(
        construct_from_module(mod, name=f"async-wd-{name}"),
        **build_test_compile_kwargs(),
    )


async def test_watchdog_trips_when_node_body_blocks_the_loop():
    """A scripted node whose (inline-on-loop) body calls ``time.sleep`` blocks the
    event loop; the watchdog records lag above threshold. Sole guard for the H2
    silently-blocking-twin mode (a blocking twin passes parity but starves the
    loop)."""

    async def block() -> Claims:
        time.sleep(0.15)  # BLOCKS the loop — runs inline on the loop thread post-1a
        return Claims(items=["blocked"])

    graph = _one_scripted_body_graph("block", block)

    async with event_loop_lag_watchdog(interval_s=0.01) as handle:
        result = await neograph.arun(graph, input={"node_id": "wd-block"})

    assert result["block"] == Claims(items=["blocked"])
    assert handle.max_lag > 0.05, (
        f"blocking node body did NOT trip the watchdog (max_lag={handle.max_lag:.4f}) "
        "— the block was masked (not running inline on the loop), so the H2 "
        "silently-blocking-twin mode would go undetected"
    )


async def test_watchdog_stays_quiet_when_node_body_is_cooperative():
    """A cooperative node body (``await asyncio.sleep``) yields the loop; lag
    stays well under the generous ceiling — the watchdog does not false-red."""

    async def coop() -> Claims:
        await asyncio.sleep(0.15)  # cooperative — yields the loop
        return Claims(items=["coop"])

    graph = _one_scripted_body_graph("coop", coop)

    async with event_loop_lag_watchdog(interval_s=0.01) as handle:
        result = await neograph.arun(graph, input={"node_id": "wd-coop"})

    assert result["coop"] == Claims(items=["coop"])
    assert handle.max_lag < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# M1-c — CANCELLATION E2E: cancel while parked, saver stays reusable
# ═══════════════════════════════════════════════════════════════════════════


async def test_cancel_parked_arun_leaves_saver_reusable(tmp_path):
    """Cancel an ``arun`` task while it is parked at the LLM gate: it raises
    ``CancelledError`` cleanly, and the SAME file-backed ``AsyncSqliteSaver``
    (owned by the TEST scope, not the cancelled task) is still usable — a fresh
    non-gated re-``arun`` of the SAME thread_id completes, proving the checkpoint
    is consistent and the saver connection was not torn down by the cancel."""
    db = str(tmp_path / "cancel.db")
    thread = {"configurable": {"thread_id": "cancel-1f"}}
    gated = GatedAsyncFake(lambda m: m(items=["gated"]))

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        graph = _think_graph(gated, checkpointer=saver)

        task = asyncio.create_task(
            neograph.arun(graph, input={"node_id": "cx"}, config=thread)
        )

        # Poll until the run parks at the gate mid-flight.
        for _ in range(300):
            if gated.enter_count == 1 or task.done():
                break
            await asyncio.sleep(0.005)

        assert gated.enter_count == 1 and not task.done(), (
            "arun did not park at the LLM gate before cancel — cannot test a "
            "mid-flight cancellation"
        )

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2.0)

        # The saver is still open in THIS scope: a fresh non-gated fake resumes
        # the SAME thread_id to completion (no re-park hang, no torn connection).
        resume_graph = _think_graph(
            StructuredFake(lambda m: m(items=["resumed"])), checkpointer=saver
        )
        completed = await asyncio.wait_for(
            neograph.arun(resume_graph, input={"node_id": "cx"}, config=thread),
            timeout=5.0,
        )

    assert completed["gen"] == Claims(items=["resumed"]), (
        "re-arun of the same thread_id after cancel did not complete cleanly — "
        "the cancel left the saver/checkpoint in an inconsistent state"
    )
