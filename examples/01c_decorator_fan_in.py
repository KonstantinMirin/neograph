"""Example 1c: Fan-in with @node — the decorator's killer feature.

01 (declarative) and 01b (linear decorator) show a simple A->B->C chain.
That shape doesn't need @node — Construct(nodes=[...]) works fine.

This example shows what @node unlocks: fan-in without manual wiring.
Four producers feed one consumer whose *parameter names* resolve to upstream
nodes automatically. Nodes are declared in shuffled order —
construct_from_module topologically sorts the graph at import time.

Run:  python examples/01c_decorator_fan_in.py
"""
from __future__ import annotations
import sys
from pydantic import BaseModel
from neograph import compile, construct_from_module, node, run

# ── Schemas ──────────────────────────────────────────────────────────────
class Claims(BaseModel, frozen=True):
    items: list[str]
class Scores(BaseModel, frozen=True):
    ratings: dict[str, float]
class Verification(BaseModel, frozen=True):
    passed: list[str]
    failed: list[str]
class Metadata(BaseModel, frozen=True):
    source: str
    version: str
class Report(BaseModel, frozen=True):
    summary: str

# ── Nodes — deliberately out of order to prove topo-sort works ──────────
# report is defined FIRST, before any of its dependencies.

@node(mode="scripted", output=Report)
def report(
    fetch_claims: Claims, score_claims: Scores,
    verify_claims: Verification, gather_metadata: Metadata,
) -> Report:
    """Fan-in: four parameter names -> four upstream nodes, auto-wired."""
    passed = ", ".join(verify_claims.passed) or "none"
    failed = ", ".join(verify_claims.failed) or "none"
    avg = sum(score_claims.ratings.values()) / len(score_claims.ratings)
    return Report(summary="\n".join([
        f"Source: {gather_metadata.source} v{gather_metadata.version}",
        f"Claims analysed: {len(fetch_claims.items)}",
        f"Average score: {avg:.1f}",
        f"Passed: {passed}",
        f"Failed: {failed}",
    ]))

@node(mode="scripted", output=Verification)
def verify_claims(fetch_claims: Claims, score_claims: Scores) -> Verification:
    passed = [c for c in fetch_claims.items if score_claims.ratings.get(c, 0) >= 0.5]
    failed = [c for c in fetch_claims.items if score_claims.ratings.get(c, 0) < 0.5]
    return Verification(passed=passed, failed=failed)

@node(mode="scripted", output=Metadata)
def gather_metadata() -> Metadata:
    return Metadata(source="requirements-doc", version="2.1")

@node(mode="scripted", output=Scores)
def score_claims(fetch_claims: Claims) -> Scores:
    ratings = {c: (0.8 if "shall" in c.lower() else 0.3) for c in fetch_claims.items}
    return Scores(ratings=ratings)

@node(mode="scripted", output=Claims)
def fetch_claims() -> Claims:
    return Claims(items=[
        "The system shall log all access attempts",
        "The system shall validate input",
        "Nice to have: dark mode",
    ])

# ── Build pipeline — no nodes= list, no ordering required ───────────────
pipeline = construct_from_module(sys.modules[__name__], name="requirements-review")

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "review-001"})
    print("=== Requirements Review Report ===")
    print(result["report"].summary)
