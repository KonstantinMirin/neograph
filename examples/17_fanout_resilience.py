"""Example 17: Fan-out Resilience — resume failed items from checkpoint.

Real-world problem: a pipeline fans out over 15 items. Item 8 fails with
a 402 (insufficient credits) or 429 (rate limit). Without resilience, you
re-run all 15 items. With checkpointing, you fix the issue and resume —
only the failed items re-run. The 14 that succeeded are preserved.

This demonstrates:
  - compile() with checkpointer for fan-out resilience
  - Catching errors from run()
  - Resuming with the same config — only failed Send items re-execute
  - Successful items are preserved in the checkpoint

Run:
    python examples/17_fanout_resilience.py
"""

from __future__ import annotations

import sys

from pydantic import BaseModel

from neograph import Construct, Each, Node, compile, register_scripted, run


# ── Schemas ──────────────────────────────────────────────────────────────

class Document(BaseModel, frozen=True):
    doc_id: str
    text: str

class Batch(BaseModel, frozen=True):
    documents: list[Document]

class AnalysisResult(BaseModel, frozen=True):
    doc_id: str
    summary: str
    word_count: int


# ── Simulate a flaky API ────────────────────────────────────────────────

attempt_count = {}

def analyze_document(input_data, config):
    """Simulates analyzing a document. Fails on first attempt for doc-03."""
    doc = input_data
    doc_id = doc.doc_id

    attempt_count.setdefault(doc_id, 0)
    attempt_count[doc_id] += 1

    if doc_id == "doc-03" and attempt_count[doc_id] <= 1:
        # Simulate a transient 402 error on first attempt
        raise RuntimeError(f"402 Insufficient Credits — failed on {doc_id}")

    return AnalysisResult(
        doc_id=doc_id,
        summary=f"Analysis of '{doc.text}' (attempt {attempt_count[doc_id]})",
        word_count=len(doc.text.split()),
    )


# ── Pipeline ────────────────────────────────────────────────────────────

register_scripted("load_docs", lambda _in, _cfg: Batch(documents=[
    Document(doc_id="doc-01", text="Authentication flow for OAuth2 integration"),
    Document(doc_id="doc-02", text="Rate limiting middleware configuration"),
    Document(doc_id="doc-03", text="Database migration strategy for multi-tenant"),
    Document(doc_id="doc-04", text="Observability pipeline with OpenTelemetry"),
    Document(doc_id="doc-05", text="CI/CD workflow for canary deployments"),
]))

register_scripted("analyze", analyze_document)

pipeline = Construct("resilient-fanout", nodes=[
    Node.scripted("load", fn="load_docs", outputs=Batch),
    Node.scripted("analyze", fn="analyze", inputs=Document, outputs=AnalysisResult)
    | Each(over="load.documents", key="doc_id"),
])


# ── Run with resilience ─────────────────────────────────────────────────

def main():
    from langgraph.checkpoint.memory import MemorySaver

    # The key: compile with a checkpointer.
    # Without it, a failed item kills everything and you start over.
    checkpointer = MemorySaver()
    graph = compile(pipeline, checkpointer=checkpointer)

    # Same config for both run and resume — the thread_id links them.
    config = {"configurable": {"thread_id": "resilience-demo"}}

    # ── First run: doc-03 will fail ──────────────────────────────────
    print("=== Run 1: processing 5 documents ===")
    try:
        result = run(graph, input={"node_id": "demo"}, config=config)
        print(f"All succeeded: {list(result['analyze'].keys())}")
    except Exception as e:
        print(f"Failed: {e}")
        print(f"  Attempts so far: {attempt_count}")
        print(f"  Documents that succeeded are checkpointed.")
        print()

    # ── Simulate fixing the issue (top up credits, wait for rate limit) ──
    print("=== Fixing the issue (credits topped up) ===")
    print()

    # ── Resume: only the failed item re-runs ─────────────────────────
    print("=== Run 2: resuming from checkpoint ===")
    try:
        # invoke(None, config) resumes from the last checkpoint.
        # Succeeded items (doc-01, 02, 04, 05) are already stored.
        # Only doc-03 re-executes.
        result = graph.invoke(None, config=config)

        # Strip framework fields for clean output
        analyze = result.get("analyze", {})
        if isinstance(analyze, dict):
            print(f"All documents processed: {sorted(analyze.keys())}")
            for doc_id in sorted(analyze.keys()):
                r = analyze[doc_id]
                print(f"  {doc_id}: {r.summary} ({r.word_count} words)")
        print()
        print(f"Total attempts per document: {attempt_count}")
        print(f"  doc-03 needed 2 attempts, all others needed 1.")
    except Exception as e:
        print(f"Resume also failed: {e}")


if __name__ == "__main__":
    main()
