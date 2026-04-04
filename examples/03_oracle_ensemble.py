"""Example 3: Oracle Ensemble — N parallel generators + merge.

Scenario: Decompose a complex requirement from multiple angles. Three
generators independently break down the same input, producing different
claim lists. A merge function combines and deduplicates them into a
single consensus list.

This demonstrates the Oracle modifier: run a node N times in parallel,
then merge results. The merge can be scripted (deterministic function)
or LLM-powered (judge prompt).

Run:
    python examples/03_oracle_ensemble.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node, Oracle, compile, register_scripted, run


# ── Schemas ──────────────────────────────────────────────────────────────

class Topic(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]


# ── Generator: each instance produces a different decomposition ──────────
# The generator receives _generator_id via config["configurable"] so each
# parallel instance can vary its output.

variant_perspectives = {
    "gen-0": ["security: must authenticate", "security: must encrypt"],
    "gen-1": ["reliability: must handle failures", "reliability: must log errors"],
    "gen-2": ["performance: must respond in 200ms", "security: must authenticate"],
}

def generate_decomposition(input_data, config):
    gen_id = config.get("configurable", {}).get("_generator_id", "gen-0")
    claims = variant_perspectives.get(gen_id, ["unknown perspective"])
    return Claims(items=claims)

register_scripted("generate_decomposition", generate_decomposition)


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
# Oracle(n=3) runs the generator 3 times in parallel, merge_fn combines.

decompose = Node.scripted(
    "decompose", fn="generate_decomposition", output=Claims
) | Oracle(n=3, merge_fn="merge_claims")

pipeline = Construct("oracle-demo", nodes=[decompose])


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "REQ-001"})

    merged = result["decompose"]
    print(f"3 generators produced {len(merged.items)} unique claims:")
    for claim in merged.items:
        print(f"  - {claim}")
    # "security: must authenticate" appears in gen-0 and gen-2 but is deduplicated
