"""Example 3: Oracle Ensemble — N parallel generators + merge.

Scenario: Decompose a complex requirement from multiple angles. Three
generators independently break down the same input, producing different
claim lists. A merge function combines and deduplicates them into a
single consensus list.

This demonstrates the Oracle modifier via @node kwargs: run a node N
times in parallel, then merge results. The merge can be scripted
(deterministic function) or LLM-powered (judge prompt).

Run:
    python examples/03_oracle_ensemble.py
"""

from __future__ import annotations

import sys
import threading

from pydantic import BaseModel

from neograph import compile, construct_from_module, node, register_scripted, run


# ── Schemas ──────────────────────────────────────────────────────────────

class Topic(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]


# ── Generator: each Oracle instance produces a different decomposition ──
# A thread-safe counter rotates through perspectives so each of the 3
# parallel generators returns a different claim list.

_perspectives = [
    ["security: must authenticate", "security: must encrypt"],
    ["reliability: must handle failures", "reliability: must log errors"],
    ["performance: must respond in 200ms", "security: must authenticate"],
]

_gen_counter_lock = threading.Lock()
_gen_counter = [0]


# ── Merge: combine and deduplicate claims from all generators ────────────

def merge_claims(variants, config):
    """Merge N claim lists into one deduplicated list."""
    seen = set()
    merged = []
    for variant in variants:
        for claim in variant.items:
            if claim not in seen:
                seen.add(claim)
                merged.append(claim)
    return Claims(items=merged)

register_scripted("merge_claims", merge_claims)


# ── Build pipeline ───────────────────────────────────────────────────────
# @node with ensemble_n=3 and merge_fn= attaches the Oracle modifier.

@node(outputs=Claims, ensemble_n=3, merge_fn="merge_claims")
def decompose() -> Claims:
    with _gen_counter_lock:
        idx = _gen_counter[0] % len(_perspectives)
        _gen_counter[0] += 1
    return Claims(items=_perspectives[idx])


pipeline = construct_from_module(sys.modules[__name__], name="oracle-demo")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _gen_counter[0] = 0  # reset for clean run
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "REQ-001"})

    merged = result["decompose"]
    print(f"3 generators produced {len(merged.items)} unique claims:")
    for claim in merged.items:
        print(f"  - {claim}")
    # "security: must authenticate" appears in gen-0 and gen-2 but is deduplicated
