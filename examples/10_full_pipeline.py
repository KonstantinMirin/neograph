"""Example 10: Full Pipeline — every NeoGraph feature in one realistic scenario.

Scenario: Requirement verification system for a software project.

Pipeline:
  1. decompose: break requirement into claims (produce, LLM, Oracle x3)
  2. enrich: sub-pipeline that looks up context and scores each claim
  3. verify-clusters: group claims into clusters, verify each (Each fan-out)
  4. validate: check if all clusters pass threshold (Operator: pause if not)
  5. report: format final output

This combines: produce, scripted, Oracle, Each, Operator, subgraph,
per-node LLM config, tool budgets, checkpointer, and observability.

The decompose node uses the @node decorator with Oracle kwargs for a
concise LLM-mode declaration. Scripted nodes use register_scripted +
Node.scripted because the pipeline includes a Construct subgraph that
requires manual assembly (construct_from_module cannot inline subgraphs).

Run (with fakes):
    python examples/10_full_pipeline.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from neograph import (
    Construct, Each, Node, Operator, Oracle, Tool,
    compile, configure_llm, node, run,
    register_condition, register_scripted, register_tool_factory,
)


# ══════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════

class Claims(BaseModel, frozen=True):
    items: list[str]

class Context(BaseModel, frozen=True):
    references: list[str]

class ScoredClaims(BaseModel, frozen=True):
    scored: list[dict[str, str]]  # [{claim, score}]

class ClusterGroup(BaseModel, frozen=True):
    label: str
    claim_ids: list[str]

class Clusters(BaseModel, frozen=True):
    groups: list[ClusterGroup]

class VerifyResult(BaseModel, frozen=True):
    cluster_label: str
    passed: bool
    gaps: list[str]

class ValidationResult(BaseModel, frozen=True):
    passed: bool
    issues: list[str]

class Report(BaseModel, frozen=True):
    text: str


# ══════════════════════════════════════════════════════════════════════════
# FAKE LLM + TOOLS (replace with real ones in production)
# ══════════════════════════════════════════════════════════════════════════

class FakeProduceLLM:
    def with_structured_output(self, model):
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        if self._model is Claims:
            return Claims(items=["shall authenticate", "shall log", "shall encrypt"])
        return self._model()


call_count = {"search": 0}


class FakeGatherLLM:
    def __init__(self):
        self._call = 0

    def bind_tools(self, tools):
        clone = FakeGatherLLM()
        clone._call = self._call
        clone._has_tools = len(tools) > 0
        return clone

    def invoke(self, messages, **kwargs):
        self._call += 1
        if getattr(self, '_has_tools', True) and self._call <= 2:
            msg = AIMessage(content="")
            msg.tool_calls = [{"name": "search_code", "args": {"q": "test"}, "id": f"c{self._call}"}]
            return msg
        return AIMessage(content="done")

    def with_structured_output(self, model):
        self._model = model
        return self


class FakeSearchTool:
    name = "search_code"

    def invoke(self, args):
        call_count["search"] += 1
        return f"Found reference #{call_count['search']}"


def llm_factory(tier, node_name=None, llm_config=None):
    if tier == "fast":
        return FakeProduceLLM()
    return FakeGatherLLM()


configure_llm(
    llm_factory=llm_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": "analyze"}],
)

register_tool_factory("search_code", lambda config, tool_config: FakeSearchTool())


# ══════════════════════════════════════════════════════════════════════════
# SCRIPTED FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def merge_claims(variants, config):
    """Oracle merge: deduplicate claims from N generators."""
    seen = set()
    merged = []
    for v in variants:
        for item in v.items:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return Claims(items=merged)

def lookup_context(input_data, config):
    return Context(references=["auth.py:42", "logger.py:18", "crypto.py:7"])

def score_claims(input_data, config):
    scores = {"authenticate": "high", "log": "medium", "encrypt": "high"}
    scored = []
    for claim in input_data.items:
        s = next((v for k, v in scores.items() if k in claim), "low")
        scored.append({"claim": claim, "score": s})
    return ScoredClaims(scored=scored)

def make_clusters(input_data, config):
    return Clusters(groups=[
        ClusterGroup(label="security", claim_ids=["authenticate", "encrypt"]),
        ClusterGroup(label="observability", claim_ids=["log"]),
    ])

def verify_cluster(input_data, config):
    passing = {"security": True, "observability": False}
    gaps = {"observability": ["missing structured logging"]}
    return VerifyResult(
        cluster_label=input_data.label,
        passed=passing.get(input_data.label, False),
        gaps=gaps.get(input_data.label, []),
    )

def check_all_passed(input_data, config):
    return ValidationResult(passed=False, issues=["observability cluster failed"])

def build_report(input_data, config):
    return Report(text="Verification complete. 1 cluster needs attention.")

register_scripted("merge_claims", merge_claims)
register_scripted("lookup_context", lookup_context)
register_scripted("score_claims", score_claims)
register_scripted("make_clusters", make_clusters)
register_scripted("verify_cluster", verify_cluster)
register_scripted("check_passed", check_all_passed)
register_scripted("build_report", build_report)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION (for Operator)
# ══════════════════════════════════════════════════════════════════════════

def needs_review(state):
    val = state.check_results
    if val and not val.passed:
        return {"issues": val.issues}
    return None

register_condition("needs_review", needs_review)


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════

# Step 1: Decompose — @node with Oracle kwargs (LLM mode, no register_scripted needed)
@node(outputs=Claims, prompt="decompose", model="fast",
      llm_config={"temperature": 0.8},
      ensemble_n=3, merge_fn="merge_claims")
def decompose() -> Claims: ...

# Step 2: Enrich — sub-pipeline with isolated state
enrich = Construct(
    "enrich",
    input=Claims,
    output=ScoredClaims,
    nodes=[
        Node.scripted("lookup", fn="lookup_context", inputs=Claims, outputs=Context),
        Node.scripted("score", fn="score_claims", inputs=Claims, outputs=ScoredClaims),
    ],
)

# Step 3: Cluster and verify — Each fan-out
cluster = Node.scripted("cluster", fn="make_clusters", outputs=Clusters)
verify = Node.scripted(
    "verify", fn="verify_cluster", inputs=ClusterGroup, outputs=VerifyResult
) | Each(over="cluster.groups", key="label")

# Step 4: Validate — Operator pauses if not all clusters pass
check_results = Node.scripted(
    "check-results", fn="check_passed", outputs=ValidationResult
) | Operator(when="needs_review")

# Step 5: Report
report = Node.scripted("report", fn="build_report", outputs=Report)

# Assemble
pipeline = Construct(
    "full-verification",
    description="End-to-end requirement verification with all NeoGraph features",
    nodes=[decompose, enrich, cluster, verify, check_results, report],
)


# ══════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Checkpointer required because Operator is in the pipeline
    graph = compile(pipeline, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "full-001"}}

    print("=== Running full pipeline ===\n")
    result = run(graph, input={"node_id": "REQ-FULL-001"}, config=config)

    print(f"Decompose: {result['decompose'].items}")
    print(f"Enrich: {[s['score'] for s in result['enrich'].scored]}")
    print(f"Verify clusters: {list(result['verify'].keys())}")
    print(f"Validation: passed={result['check_results'].passed}")
    print(f"Search tool calls: {call_count['search']}")

    if "__interrupt__" in result:
        print(f"\nPaused for review: {result['__interrupt__'][0].value}")
        print("\n=== Resuming with approval ===\n")
        result = run(graph, resume={"approved": True}, config=config)
        print(f"Report: {result['report'].text}")
