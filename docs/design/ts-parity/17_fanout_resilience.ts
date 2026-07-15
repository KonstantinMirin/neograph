// TS parity sketch of: examples/17_fanout_resilience.py
// Fan-out resilience — resume failed fan-out items from a checkpoint.
//
// This is a HYPOTHETICAL port against the PROPOSED API in
// docs/design/typescript-port.md. It is NOT meant to compile or run — it
// grounds a feature-parity analysis. `// PARITY:` comments flag every point
// where the TS DX diverges from the Python original.
//
// KEY OBSERVATION: this example lives ENTIRELY on the PROGRAMMATIC surface
// (Node.scripted + Construct + `| Each(...)`), which the design doc rates as
// the "TS-first surface / Direct". It uses NO @node decorator, so the AD-0
// compiler transformer (the doc's centerpiece) is IRRELEVANT here. The load-
// bearing feature is LangGraph's checkpoint-resume-of-a-partial-superstep,
// which is inherited wholesale from LangGraph.js.

import { z } from "zod";
import { MemorySaver } from "@langchain/langgraph";
import type { RunnableConfig } from "@langchain/core/runnables";
import { Construct, Node, Each, compile, run } from "@neograph/core";

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY: Python `class Document(BaseModel, frozen=True)` → Zod object.
//   `frozen=True` has NO direct Zod equivalent. Zod validates shape but the
//   returned value is a plain, MUTABLE object. To reproduce Python's runtime
//   immutability you'd `.transform(Object.freeze)` (shallow, runtime) or rely
//   on a `readonly` TS type (compile-time only, erased at runtime). Neither is
//   a faithful match for Pydantic frozen (deep, hashable, equality-by-value).
const Document = z.object({
  doc_id: z.string(),
  text: z.string(),
});
type Document = z.infer<typeof Document>;

const Batch = z.object({
  documents: z.array(Document),
});
type Batch = z.infer<typeof Batch>;

const AnalysisResult = z.object({
  doc_id: z.string(),
  summary: z.string(),
  word_count: z.number().int(),
});
type AnalysisResult = z.infer<typeof AnalysisResult>;

// ── Simulate a flaky API ────────────────────────────────────────────────
// Module-level mutable counter to fail doc-03 once. Ports directly.
const attemptCount: Record<string, number> = {};

// PARITY: Python raw scripted fn `def analyze_document(input_data, config)`.
//   On the programmatic surface the scripted body is UNTYPED — `input_data` is
//   whatever `_extract_input` hands it (here, the Each item = one Document).
//   TS lets us annotate `inputData: Document` for editor help, but this type is
//   NOT enforced by the transformer (this surface bypasses AD-0) — it is a bare
//   assertion the author makes. Same trust level as Python's untyped positional.
function analyzeDocument(inputData: Document, _config: RunnableConfig): AnalysisResult {
  const doc = inputData;
  const docId = doc.doc_id;

  attemptCount[docId] = (attemptCount[docId] ?? 0) + 1;

  if (docId === "doc-03" && attemptCount[docId] <= 1) {
    // Simulate a transient 402 error on the first attempt.
    throw new Error(`402 Insufficient Credits — failed on ${docId}`);
  }

  return {
    doc_id: docId,
    summary: `Analysis of '${doc.text}' (attempt ${attemptCount[docId]})`,
    word_count: doc.text.split(/\s+/).length,
  };
}

// PARITY: `load_docs(_in, _cfg)` ignores both args and seeds the batch.
function loadDocs(_inputData: unknown, _config: RunnableConfig): Batch {
  return {
    documents: [
      { doc_id: "doc-01", text: "Authentication flow for OAuth2 integration" },
      { doc_id: "doc-02", text: "Rate limiting middleware configuration" },
      { doc_id: "doc-03", text: "Database migration strategy for multi-tenant" },
      { doc_id: "doc-04", text: "Observability pipeline with OpenTelemetry" },
      { doc_id: "doc-05", text: "CI/CD workflow for canary deployments" },
    ],
  };
}

// ── Pipeline ────────────────────────────────────────────────────────────
// PARITY: Python `Node.scripted("analyze", fn="analyze", ...) | Each(...)`.
//   The `|` pipe becomes `.pipe(...)` (design doc: Redesign, but mechanical).
//   `Node.scripted` static factory ports 1:1. Node references its body by the
//   STRING key "analyze"; the actual fn is supplied to compile() below — the
//   runtime-assembly double-indirection that makes this the "LLM-driven" surface.
const pipeline = new Construct("resilient-fanout", {
  nodes: [
    Node.scripted("load", { fn: "load_docs", outputs: Batch }),
    Node.scripted("analyze", {
      fn: "analyze",
      // PARITY: Python `inputs=Document` single-type shorthand (skips fan-in
      //   validation, defers to a runtime shape scan). Maps to a single Zod
      //   schema rather than a `{name: schema}` fan-in dict. Direct.
      inputs: Document,
      outputs: AnalysisResult,
    }).pipe(
      // PARITY: `Each(over="load.documents", key="doc_id")` — both args are
      //   STRINGS, so nothing here needs reflection or the transformer.
      //   `over: "load.documents"` is a runtime dotted path (unchecked at
      //   compile time in BOTH languages). `key: "doc_id"` must name a field
      //   on Document — Python checks via `model_fields`; TS would check via
      //   `Document.shape` (Zod introspection). Direct.
      new Each({ over: "load.documents", key: "doc_id" }),
    ),
  ],
});

// ── Run with resilience ─────────────────────────────────────────────────
// PARITY: Python `main()` is SYNCHRONOUS (`graph.invoke`). LangGraph.js is
//   async-only, so the whole flow becomes `async` and every `run(...)` is
//   `await`ed. This is a pervasive, if shallow, divergence for every example.
async function main() {
  // The key: compile with a checkpointer. Without it, a failed fan-out item
  // kills the whole run and you restart from zero.
  const checkpointer = new MemorySaver();

  // PARITY: the `scripted` map (string key → fn) is supplied at compile time,
  //   exactly like Python's `scripted={"load_docs": ..., "analyze": ...}`.
  //   Module-level registry → plain object / Map. Direct.
  const graph = compile(pipeline, {
    checkpointer,
    scripted: { load_docs: loadDocs, analyze: analyzeDocument },
  });

  // Same config for run and resume — the thread_id links them.
  const config: RunnableConfig = { configurable: { thread_id: "resilience-demo" } };

  // ── First run: doc-03 will fail ──────────────────────────────────
  console.log("=== Run 1: processing 5 documents ===");
  try {
    // PARITY: `input={"node_id": "demo"}` — a seed dict. `load` ignores it;
    //   there are no FromInput params in this example. Direct.
    const result = await run(graph, { input: { node_id: "demo" }, config });
    console.log(`All succeeded: ${Object.keys(result.analyze)}`);
  } catch (e) {
    console.log(`Failed: ${(e as Error).message}`);
    console.log(`  Attempts so far: ${JSON.stringify(attemptCount)}`);
    console.log("  Documents that succeeded are checkpointed.\n");
  }

  console.log("=== Fixing the issue (credits topped up) ===\n");

  // ── Resume: only the failed item re-runs ─────────────────────────
  console.log("=== Run 2: resuming from checkpoint ===");
  try {
    // PARITY: `run(graph, config)` with NO input = resume. Python maps this to
    //   `graph.invoke(None, config)`; the TS wrapper maps it to
    //   `graph.invoke(null, config)`. LangGraph(.js) skips completed supersteps
    //   and continues from the failure point via pending-writes — this is the
    //   ACTUAL resilience mechanic, inherited from LangGraph, not neograph.
    //   The succeeded items (doc-01/02/04/05) are already in the checkpoint;
    //   only doc-03 re-executes.
    const result = await run(graph, { config });

    // PARITY: `result["analyze"]` is the Each dict-form output keyed by doc_id.
    //   At the LangGraph.js layer this is an Annotation with a merge reducer
    //   `(existing, update) => ({...existing, ...update})` (design-doc AD-4) —
    //   the reducer is what preserves the 4 successes across the failure.
    //   BUT: on checkpoint round-trip each value is a PLAIN JSON object, not an
    //   `AnalysisResult` CLASS instance. Python rehydrates Pydantic instances
    //   (via msgpack type registration); TS gets structurally-typed plain
    //   objects. Property access (`r.summary`, `r.word_count`) works
    //   identically, but any method/identity/`instanceof` on the model is lost.
    const analyze = (result.analyze ?? {}) as Record<string, AnalysisResult>;
    const ids = Object.keys(analyze).sort();
    console.log(`All documents processed: ${ids}`);
    for (const docId of ids) {
      const r = analyze[docId];
      console.log(`  ${docId}: ${r.summary} (${r.word_count} words)`);
    }
    console.log(`\nTotal attempts per document: ${JSON.stringify(attemptCount)}`);
    console.log("  doc-03 needed 2 attempts, all others needed 1.");
  } catch (e) {
    console.log(`Resume also failed: ${(e as Error).message}`);
  }
}

// PARITY: `if __name__ == "__main__": main()` → direct top-level invocation.
void main();
