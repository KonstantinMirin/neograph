"""Example 4: Each Fan-out — process a collection in parallel.

Scenario: A pipeline produces a set of requirement clusters (grouped by
domain). Each cluster needs independent verification — search for evidence,
score coverage. The Each modifier runs the verify node once per cluster,
collecting results as a dict keyed by cluster label.

This demonstrates: first node produces a collection, second node processes
each item in parallel, results merge into a dict.

Run:
    python examples/04_each_fanout.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Each, Node, compile, run
from neograph.factory import register_scripted


# ── Schemas ──────────────────────────────────────────────────────────────

class ClusterGroup(BaseModel, frozen=True):
    label: str
    claim_ids: list[str]

class Clusters(BaseModel, frozen=True):
    groups: list[ClusterGroup]

class VerifyResult(BaseModel, frozen=True):
    cluster_label: str
    coverage_pct: int
    gaps: list[str]


# ── Functions ────────────────────────────────────────────────────────────

def discover_clusters(input_data, config):
    """Simulate discovering requirement clusters from analysis."""
    return Clusters(groups=[
        ClusterGroup(label="authentication", claim_ids=["REQ-1", "REQ-2", "REQ-3"]),
        ClusterGroup(label="logging", claim_ids=["REQ-4", "REQ-5"]),
        ClusterGroup(label="performance", claim_ids=["REQ-6"]),
    ])

def verify_cluster(input_data, config):
    """Verify a single cluster — check coverage against codebase."""
    # input_data is a ClusterGroup (the specific item from the collection)
    coverage = {"authentication": 85, "logging": 60, "performance": 100}
    gaps_map = {
        "authentication": ["MFA not implemented"],
        "logging": ["no structured logging", "missing audit trail"],
        "performance": [],
    }
    return VerifyResult(
        cluster_label=input_data.label,
        coverage_pct=coverage.get(input_data.label, 0),
        gaps=gaps_map.get(input_data.label, ["unknown"]),
    )

register_scripted("discover_clusters", discover_clusters)
register_scripted("verify_cluster", verify_cluster)


# ── Build pipeline ───────────────────────────────────────────────────────
# Step 1: discover clusters
# Step 2: verify each cluster in parallel (Each over clusters.groups, keyed by label)

discover = Node.scripted("discover", fn="discover_clusters", output=Clusters)

verify = Node.scripted(
    "verify", fn="verify_cluster", input=ClusterGroup, output=VerifyResult
) | Each(over="discover.groups", key="label")

pipeline = Construct("cluster-verification", nodes=[discover, verify])


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "analysis-001"})

    verify_results = result["verify"]  # dict[str, VerifyResult]

    print(f"Verified {len(verify_results)} clusters:\n")
    for label, vr in verify_results.items():
        status = "PASS" if vr.coverage_pct >= 80 else "GAPS"
        print(f"  [{status}] {label}: {vr.coverage_pct}% coverage")
        for gap in vr.gaps:
            print(f"         - {gap}")
