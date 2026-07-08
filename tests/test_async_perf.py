"""Phase 1f per-node async-overhead perf gate — CHARACTERIZATION for neograph-w74k.2.6.

CHARACTERIZATION, not TDD red. Asserts the async driver adds no meaningful
per-node overhead vs the sync driver over ONE shared pure core. PASSES on first
run; a FAIL means a real per-node async regression (e.g. accidental per-node loop
creation) — report it loudly.

FORMULATION (refinement mybm.12, replacing the original step-2 assertion after
the architect's #1-risk finding):
  * K-node (K=20) chained SCRIPTED pipeline; NO checkpointer (per-node sqlite I/O
    would inflate variance).
  * WARM UP both paths first.
  * Measure BOTH run() and arun() as MEDIAN-OF-M (M=9 >= 7) INSIDE ONE
    ``asyncio.run()`` that awaits arun M times in a loop — this amortizes
    loop-setup to ~0 AND gives outlier rejection on BOTH sides (a single
    unmedianed arun sample was the original flaw).
  * Assert a SINGLE generous absolute per-node ceiling:
        (async_median - sync_median) / K < 0.020  (20 ms / node)
    Ratio/max() terms DROPPED entirely (they flake when sync is sub-ms). No
    OR-escape added — a single absolute floor is the least-flaky gate.

Marked ``@pytest.mark.perf`` (registered in pyproject) so a loaded CI box can
deselect it; a flaky perf gate that randomly reds CI is worse than none.

PARTIAL COVERAGE (flag): within-run MCP session reuse (H1/§5) is Phase 3 — this
gate asserts only that the async node vertical adds no per-call overhead. It does
NOT cover MCP session-reuse amortization.

Driver/runtime E2E — not an IR-shape change, so the three-surface matrix does
not apply.
"""

from __future__ import annotations

import asyncio
import types as _types
from statistics import median
from time import perf_counter

import pytest

import neograph
from neograph import compile, construct_from_module, node, run
from tests.fakes import build_test_compile_kwargs
from tests.schemas import RawText

K = 20
M = 9  # median-of-M, M >= 7 per the refinement
_PER_NODE_CEILING_S = 0.020  # 20 ms / node — generous single absolute gate


def _chain_pipeline(k: int):
    """A k-node chained scripted pipeline: n0 -> n1 -> ... -> n{k-1}.

    Each node consumes the previous node's ``RawText`` and re-emits it, so the
    DAG is a straight chain of k real superstep hops — the unit the gate
    amortizes over."""
    mod = _types.ModuleType("test_async_perf_chain_mod")
    prev: str | None = None
    for i in range(k):
        name = f"n{i}"
        if prev is None:
            src = f"def {name}() -> RawText:\n    return RawText(text='x')\n"
        else:
            src = f"def {name}({prev}: RawText) -> RawText:\n    return RawText(text={prev}.text)\n"
        ns: dict = {"RawText": RawText}
        exec(src, ns)  # noqa: S102 — test-local codegen for named-param edges
        decorated = node(mode="scripted", outputs=RawText)(ns[name])
        setattr(mod, name, decorated)
        prev = name
    return construct_from_module(mod, name="async-perf-chain")


@pytest.mark.perf
def test_async_per_node_overhead_within_absolute_ceiling():
    """(async_median - sync_median) / K < 20 ms/node — the async driver adds no
    meaningful per-node overhead vs the sync driver, measured median-of-M on both
    sides inside one event loop (see module docstring for the formulation)."""
    graph = compile(_chain_pipeline(K), **build_test_compile_kwargs())

    async def _measure() -> tuple[float, float]:
        # Warm up BOTH paths (import/JIT/first-call costs excluded from timing).
        run(graph, input={"node_id": "warm-sync"})
        await neograph.arun(graph, input={"node_id": "warm-async"})

        sync_times: list[float] = []
        for i in range(M):
            t0 = perf_counter()
            run(graph, input={"node_id": f"s{i}"})
            sync_times.append(perf_counter() - t0)

        async_times: list[float] = []
        for i in range(M):
            t0 = perf_counter()
            await neograph.arun(graph, input={"node_id": f"a{i}"})
            async_times.append(perf_counter() - t0)

        return median(sync_times), median(async_times)

    sync_med, async_med = asyncio.run(_measure())
    per_node = (async_med - sync_med) / K

    assert per_node < _PER_NODE_CEILING_S, (
        f"async driver adds {per_node * 1000:.2f} ms/node over sync "
        f"(ceiling {_PER_NODE_CEILING_S * 1000:.0f} ms/node); "
        f"sync_median={sync_med * 1000:.2f}ms async_median={async_med * 1000:.2f}ms "
        f"over K={K} nodes, M={M} repeats — possible per-node async regression"
    )
