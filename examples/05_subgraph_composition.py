"""Example 5: Subgraph Composition — isolated sub-pipelines with modifiers.

Scenario: A requirements analysis system with three phases:
  1. Decompose: break requirement into claims
  2. Enrich: a sub-pipeline that runs in isolation — looks up context,
     then scores each claim (2 internal nodes, hidden from parent)
  3. Report: format the scored output

The enrich phase is a Construct with declared I/O boundary. It gets its
own state — internal nodes (lookup, score) don't appear in the parent's
result. Only the declared output type surfaces.

This also demonstrates Construct | Oracle — running the entire enrich
sub-pipeline 3 times and merging the results.

Run:
    python examples/05_subgraph_composition.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node, Oracle, compile, register_scripted, run


# ── Schemas ──────────────────────────────────────────────────────────────

class Claims(BaseModel, frozen=True):
    items: list[str]

class Context(BaseModel, frozen=True):
    references: list[str]

class ScoredClaims(BaseModel, frozen=True):
    scored: list[dict[str, str]]

class Report(BaseModel, frozen=True):
    text: str


# ── Parent pipeline functions ────────────────────────────────────────────

def decompose_req(input_data, config):
    return Claims(items=["shall authenticate", "shall log access", "shall encrypt data"])

def format_report(input_data, config):
    lines = [f"  {s['claim']}: {s['score']}" for s in input_data.scored]
    return Report(text="Coverage Report:\n" + "\n".join(lines))

register_scripted("decompose_req", decompose_req)
register_scripted("format_report", format_report)


# ── Sub-pipeline: enrich (lookup + score) ────────────────────────────────
# These functions are internal to the sub-pipeline. The parent never sees
# the Context intermediate — only the final ScoredClaims output.

def lookup_context(input_data, config):
    """Look up implementation references for each claim."""
    return Context(references=["auth.py:42", "logger.py:18", "crypto.py:7"])

def score_claims(input_data, config):
    """Score claims based on available context."""
    # input_data is Claims (from neo_subgraph_input)
    scores = {"authenticate": "high", "log access": "medium", "encrypt": "high"}
    scored = []
    for claim in input_data.items:
        score = "low"
        for keyword, s in scores.items():
            if keyword in claim:
                score = s
                break
        scored.append({"claim": claim, "score": score})
    return ScoredClaims(scored=scored)

register_scripted("lookup_context", lookup_context)
register_scripted("score_claims", score_claims)


# ── Build sub-pipeline with declared I/O ─────────────────────────────────

enrich = Construct(
    "enrich",
    input=Claims,
    output=ScoredClaims,
    nodes=[
        Node.scripted("lookup", fn="lookup_context", input=Claims, output=Context),
        Node.scripted("score", fn="score_claims", input=Claims, output=ScoredClaims),
    ],
)

# ── Build parent pipeline ────────────────────────────────────────────────

pipeline = Construct("req-analysis", nodes=[
    Node.scripted("decompose", fn="decompose_req", output=Claims),
    enrich,  # sub-pipeline: Claims → ScoredClaims (isolated state)
    Node.scripted("report", fn="format_report", input=ScoredClaims, output=Report),
])


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "REQ-100"})

    print(result["report"].text)
    print(f"\nResult keys: {list(result.keys())}")
    print("Note: 'lookup' and 'score' are NOT in the result — they're internal to enrich")


# ═══════════════════════════════════════════════════════════════════════════
# Variant: Construct | Oracle — ensemble the entire sub-pipeline
# ═══════════════════════════════════════════════════════════════════════════

def merge_scored(variants, config):
    """Merge multiple scoring runs — pick highest score per claim."""
    best = {}
    score_rank = {"high": 3, "medium": 2, "low": 1}
    for variant in variants:
        for item in variant.scored:
            claim = item["claim"]
            if claim not in best or score_rank.get(item["score"], 0) > score_rank.get(best[claim], 0):
                best[claim] = item["score"]
    return ScoredClaims(scored=[{"claim": c, "score": s} for c, s in best.items()])

register_scripted("merge_scored", merge_scored)

# Same sub-pipeline, but ensembled 3 times:
enrich_oracle = Construct(
    "enrich",
    input=Claims,
    output=ScoredClaims,
    nodes=[
        Node.scripted("lookup", fn="lookup_context", input=Claims, output=Context),
        Node.scripted("score", fn="score_claims", input=Claims, output=ScoredClaims),
    ],
) | Oracle(n=3, merge_fn="merge_scored")

pipeline_oracle = Construct("req-analysis-oracle", nodes=[
    Node.scripted("decompose", fn="decompose_req", output=Claims),
    enrich_oracle,
    Node.scripted("report", fn="format_report", input=ScoredClaims, output=Report),
])

# Uncomment to run the Oracle variant:
# if __name__ == "__main__":
#     graph = compile(pipeline_oracle)
#     result = run(graph, input={"node_id": "REQ-101"})
#     print("\n--- Oracle variant ---")
#     print(result["report"].text)
