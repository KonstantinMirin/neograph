"""Example 19: Checkpoint Auto-Resume — preserve upstream work on schema changes.

Scenario: A 4-node linear pipeline (prepare -> enrich -> analyze -> report)
processes data. After the first run completes, the analyze node's output
model changes (a field is added). On the second run with the same thread_id,
neograph detects the schema divergence and automatically rewinds to the
checkpoint before the changed node. Nodes A and B are preserved from the
checkpoint; nodes C and D re-execute.

This demonstrates two behaviors:
  1. auto_resume=True (default) — automatic rewind and selective re-execution
  2. auto_resume=False — raises CheckpointSchemaError so you can handle it

Patterns demonstrated:
  - Scripted @node pipeline (no LLM, no API keys)
  - MemorySaver checkpointer for in-memory checkpoint persistence
  - construct_from_functions for pipeline assembly
  - Schema fingerprinting and per-node invalidation

Run:
    python examples/19_checkpoint_auto_resume.py
"""

from __future__ import annotations

from pydantic import BaseModel

from langgraph.checkpoint.memory import MemorySaver

from neograph import (
    CheckpointSchemaError,
    compile,
    construct_from_functions,
    node,
    run,
)


# -- Schemas ------------------------------------------------------------------
# Shared types that stay the same across both pipeline versions.

class Prepared(BaseModel):
    text: str
    token_count: int


class Enriched(BaseModel):
    text: str
    token_count: int
    entities: list[str]


class Report(BaseModel):
    title: str
    body: str


# -- Demo 1: auto_resume=True (default) — selective re-execution -------------

def demo_auto_resume():
    """Automatic rewind and selective re-execution on schema change."""
    print("=" * 60)
    print("DEMO 1: auto_resume=True (default)")
    print("=" * 60)

    # Analysis output types — v1 and v2 differ by one field.
    # Defined locally so the checkpoint serializer round-trips them as dicts,
    # allowing the new state model to absorb old checkpoint data.

    class AnalysisV1(BaseModel):
        summary: str
        entity_count: int

    class AnalysisV2(BaseModel):
        summary: str
        entity_count: int
        confidence: float = 0.0  # new field in v2

    # -- v1 nodes --

    @node(outputs=Prepared)
    def prepare() -> Prepared:
        print("  [prepare] running")
        text = "The system validates all user inputs and logs access attempts."
        return Prepared(text=text, token_count=len(text.split()))

    @node(outputs=Enriched)
    def enrich(prepare: Prepared) -> Enriched:
        print("  [enrich] running")
        entities = ["system", "user", "inputs", "access"]
        return Enriched(
            text=prepare.text,
            token_count=prepare.token_count,
            entities=entities,
        )

    @node(outputs=AnalysisV1, name="analyze")
    def analyze_v1(enrich: Enriched) -> AnalysisV1:
        print("  [analyze] running (v1)")
        return AnalysisV1(
            summary=f"Found {len(enrich.entities)} entities in {enrich.token_count} tokens",
            entity_count=len(enrich.entities),
        )

    @node(outputs=Report, name="report")
    def report_v1(analyze: AnalysisV1) -> Report:
        print("  [report] running (v1)")
        return Report(title="Analysis Report", body=analyze.summary)

    # -- v2 nodes (analyze output gains a field) --

    @node(outputs=AnalysisV2, name="analyze")
    def analyze_v2(enrich: Enriched) -> AnalysisV2:
        print("  [analyze] running (v2 -- new field)")
        return AnalysisV2(
            summary=f"Found {len(enrich.entities)} entities in {enrich.token_count} tokens",
            entity_count=len(enrich.entities),
            confidence=0.92,
        )

    @node(outputs=Report, name="report")
    def report_v2(analyze: AnalysisV2) -> Report:
        print("  [report] running (v2)")
        return Report(
            title="Analysis Report v2",
            body=f"{analyze.summary} (confidence={analyze.confidence})",
        )

    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "demo-auto-resume"}}

    # -- Run 1: full pipeline, all 4 nodes execute --
    print("\nRun 1: full pipeline (v1)")
    print("-" * 40)
    pipeline_v1 = construct_from_functions(
        "checkpoint-demo", [prepare, enrich, analyze_v1, report_v1],
    )
    graph_v1 = compile(pipeline_v1, checkpointer=checkpointer)
    result_v1 = run(graph_v1, input={"node_id": "demo"}, config=config)
    print(f"\n  Result: {result_v1['report'].body}")

    # -- Run 2: schema changed on analyze, only analyze+report re-execute --
    print("\nRun 2: analyze output changed (v1 -> v2)")
    print("-" * 40)
    print("  Expected: prepare and enrich preserved, analyze and report re-run")
    pipeline_v2 = construct_from_functions(
        "checkpoint-demo", [prepare, enrich, analyze_v2, report_v2],
    )
    graph_v2 = compile(pipeline_v2, checkpointer=checkpointer)
    result_v2 = run(graph_v2, input={"node_id": "demo"}, config=config)
    print(f"\n  Result: {result_v2['report'].body}")


# -- Demo 2: auto_resume=False — CheckpointSchemaError -----------------------

def demo_strict_mode():
    """Strict mode: raise CheckpointSchemaError on mismatch."""
    print("\n" + "=" * 60)
    print("DEMO 2: auto_resume=False (strict mode)")
    print("=" * 60)

    class AnalysisV1(BaseModel):
        summary: str
        entity_count: int

    class AnalysisV2(BaseModel):
        summary: str
        entity_count: int
        confidence: float = 0.0

    @node(outputs=Prepared, name="prepare")
    def prepare_s() -> Prepared:
        print("  [prepare] running")
        text = "The system validates all user inputs and logs access attempts."
        return Prepared(text=text, token_count=len(text.split()))

    @node(outputs=Enriched, name="enrich")
    def enrich_s(prepare: Prepared) -> Enriched:
        print("  [enrich] running")
        entities = ["system", "user", "inputs", "access"]
        return Enriched(
            text=prepare.text,
            token_count=prepare.token_count,
            entities=entities,
        )

    @node(outputs=AnalysisV1, name="analyze")
    def analyze_s_v1(enrich: Enriched) -> AnalysisV1:
        print("  [analyze] running (v1)")
        return AnalysisV1(
            summary=f"Found {len(enrich.entities)} entities",
            entity_count=len(enrich.entities),
        )

    @node(outputs=Report, name="report")
    def report_s_v1(analyze: AnalysisV1) -> Report:
        print("  [report] running")
        return Report(title="Report", body=analyze.summary)

    @node(outputs=AnalysisV2, name="analyze")
    def analyze_s_v2(enrich: Enriched) -> AnalysisV2:
        print("  [analyze] running (v2)")
        return AnalysisV2(
            summary=f"Found {len(enrich.entities)} entities",
            entity_count=len(enrich.entities),
            confidence=0.95,
        )

    @node(outputs=Report, name="report")
    def report_s_v2(analyze: AnalysisV2) -> Report:
        print("  [report] running (v2)")
        return Report(title="Report v2", body=analyze.summary)

    checkpointer = MemorySaver()
    config = {"configurable": {"thread_id": "demo-strict"}}

    # -- Run 1: full pipeline --
    print("\nRun 1: full pipeline (v1)")
    print("-" * 40)
    pipeline_v1 = construct_from_functions(
        "strict-demo", [prepare_s, enrich_s, analyze_s_v1, report_s_v1],
    )
    graph_v1 = compile(pipeline_v1, checkpointer=checkpointer)
    run(graph_v1, input={"node_id": "demo"}, config=config)

    # -- Run 2: schema changed, strict mode rejects --
    print("\nRun 2: schema changed, auto_resume=False")
    print("-" * 40)
    pipeline_v2 = construct_from_functions(
        "strict-demo", [prepare_s, enrich_s, analyze_s_v2, report_s_v2],
    )
    graph_v2 = compile(pipeline_v2, checkpointer=checkpointer)
    try:
        run(graph_v2, input={"node_id": "demo"}, config=config, auto_resume=False)
        print("  ERROR: expected CheckpointSchemaError but none was raised")
    except CheckpointSchemaError as e:
        print(f"  Caught CheckpointSchemaError (expected)")
        print(f"  Invalidated nodes: {sorted(e.invalidated_nodes)}")
        print(f"  Message: {e}")


if __name__ == "__main__":
    # Suppress framework logs so the example output is clear.
    import logging
    import structlog
    logging.getLogger().setLevel(logging.WARNING)
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

    demo_auto_resume()
    demo_strict_mode()
