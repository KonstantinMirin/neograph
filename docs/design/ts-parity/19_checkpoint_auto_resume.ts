// TS parity sketch of: examples/19_checkpoint_auto_resume.py
// NOT runnable. Hypothetical port against the PROPOSED API in
// docs/design/typescript-port.md (AD-0 transformer + .pipe + node() wrapper).
//
// What this example exercises (all keyless/scripted — no LLM, no DI, no Oracle/Each/Loop,
// no ForwardConstruct, no describe_type):
//   - scripted @node pipeline (4 linear nodes: prepare -> enrich -> analyze -> report)
//   - construct_from_functions assembly
//   - compile(..., checkpointer=MemorySaver())
//   - run(..., config={thread_id}, auto_resume=True|False)
//   - schema fingerprinting + per-node invalidation + auto-rewind on schema change
//   - CheckpointSchemaError.invalidated_nodes
//   - locally-scoped output models whose checkpoint serde round-trips as dicts
//
// The DX-hard part of THIS example is NOT node authoring — it is the checkpoint
// schema-fingerprint auto-rewind subsystem, which the parity matrix does not cover.

import { z } from "zod";
import { MemorySaver } from "@langchain/langgraph";
import {
  node,
  compile,
  run,
  constructFromFunctions,
  CheckpointSchemaError,
} from "@neograph/core";

// -- Schemas -----------------------------------------------------------------
// PARITY(direct): Pydantic BaseModel -> Zod object. Attribute access
// `result.report.body` in Python (a model INSTANCE) becomes property access on a
// Zod-parsed plain object — same call site. AD-1 (Zod) row.

const Prepared = z.object({ text: z.string(), tokenCount: z.number().int() });
type Prepared = z.infer<typeof Prepared>;

const Enriched = z.object({
  text: z.string(),
  tokenCount: z.number().int(),
  entities: z.array(z.string()),
});
type Enriched = z.infer<typeof Enriched>;

const Report = z.object({ title: z.string(), body: z.string() });
type Report = z.infer<typeof Report>;

// -- Demo 1: autoResume=true (default) — selective re-execution --------------

async function demoAutoResume() {
  console.log("=".repeat(60));
  console.log("DEMO 1: autoResume=true (default)");
  console.log("=".repeat(60));

  // Analysis output types — v1 and v2 differ by one field.
  //
  // PARITY(redesign): in Python these are defined *inside the function* so the
  // checkpoint serializer round-trips them as plain dicts and the new state model
  // can absorb old checkpoint data. In TS there are no runtime classes to begin
  // with — state holds plain objects and Zod .parse() rehydrates them. The
  // "absorb old checkpoint" behavior maps to Zod: the new v2 field carries
  // `.default(0)`, so an old checkpoint dict lacking `confidence` still parses.
  // (Python's `confidence: float = 0.0` default -> `z.number().default(0)`.)
  const AnalysisV1 = z.object({ summary: z.string(), entityCount: z.number().int() });
  const AnalysisV2 = z.object({
    summary: z.string(),
    entityCount: z.number().int(),
    confidence: z.number().default(0), // new field in v2
  });
  type AnalysisV1 = z.infer<typeof AnalysisV1>;
  type AnalysisV2 = z.infer<typeof AnalysisV2>;

  // -- v1 nodes --
  // PARITY(direct, via AD-0): the transformer reads the fn signature and emits
  // __neo_meta.inputs from parameter names + __neo_meta.output from the return
  // type — so `enrich(prepare: Prepared)` wires the prepare->enrich edge exactly
  // like Python's inspect.signature. `outputs:`/`name:` are explicit config.
  //
  // PARITY(gap): the AD-0 wrapper `node({...}, (prepare: Prepared): Enriched => ...)`
  // cannot recover the *return type* the way Python reads `-> Enriched`, because a
  // scripted body may `return` at several points. The transformer must trust the
  // arrow's declared return annotation. Where it can't, fall back to explicit
  // `outputs:` (shown here for robustness, since checkpoint fingerprinting below
  // depends on this schema being exact).

  const prepare = node({ outputs: Prepared }, (): Prepared => {
    console.log("  [prepare] running");
    const text = "The system validates all user inputs and logs access attempts.";
    return { text, tokenCount: text.split(" ").length };
  });

  const enrich = node({ outputs: Enriched }, (prepare: Prepared): Enriched => {
    console.log("  [enrich] running");
    return {
      text: prepare.text,
      tokenCount: prepare.tokenCount,
      entities: ["system", "user", "inputs", "access"],
    };
  });

  // PARITY(direct): `name: "analyze"` overrides the wrapper-inferred name so both
  // schema versions occupy the SAME node identity across runs. This is what makes
  // the fingerprint of state field `analyze` change (not add a new field).
  const analyzeV1 = node({ outputs: AnalysisV1, name: "analyze" }, (enrich: Enriched): AnalysisV1 => {
    console.log("  [analyze] running (v1)");
    return {
      summary: `Found ${enrich.entities.length} entities in ${enrich.tokenCount} tokens`,
      entityCount: enrich.entities.length,
    };
  });

  const reportV1 = node({ outputs: Report, name: "report" }, (analyze: AnalysisV1): Report => {
    console.log("  [report] running (v1)");
    return { title: "Analysis Report", body: analyze.summary };
  });

  // -- v2 nodes (analyze output gains a field) --
  const analyzeV2 = node({ outputs: AnalysisV2, name: "analyze" }, (enrich: Enriched): AnalysisV2 => {
    console.log("  [analyze] running (v2 -- new field)");
    return {
      summary: `Found ${enrich.entities.length} entities in ${enrich.tokenCount} tokens`,
      entityCount: enrich.entities.length,
      confidence: 0.92,
    };
  });

  const reportV2 = node({ outputs: Report, name: "report" }, (analyze: AnalysisV2): Report => {
    console.log("  [report] running (v2)");
    return {
      title: "Analysis Report v2",
      body: `${analyze.summary} (confidence=${analyze.confidence})`,
    };
  });

  const checkpointer = new MemorySaver();
  const config = { configurable: { thread_id: "demo-auto-resume" } };

  // -- Run 1: full pipeline, all 4 nodes execute --
  console.log("\nRun 1: full pipeline (v1)");
  // PARITY(direct): construct_from_functions -> constructFromFunctions. The
  // transformer-emitted __neo_meta on each node lets the assembler build the DAG
  // from param names, same as the Python builder walking sidecars.
  const pipelineV1 = constructFromFunctions("checkpoint-demo", [prepare, enrich, analyzeV1, reportV1]);
  const graphV1 = compile(pipelineV1, { checkpointer });
  const resultV1 = await run(graphV1, { input: { node_id: "demo" }, config });
  console.log(`\n  Result: ${(resultV1.report as Report).body}`);

  // -- Run 2: schema changed on analyze, only analyze+report re-execute --
  //
  // PARITY(GAP — the whole point of this example is unmapped): neograph rewinds
  // here because compile() stashed graph.schema_fingerprint (compute_schema_fingerprint,
  // state.py:401) + graph.node_fingerprints (compute_node_fingerprints, state.py:346),
  // and on resume runner.py diffs them, computes `invalidated = {analyze, report}`,
  // walks get_state_history() backwards for the oldest checkpoint whose `.next`
  // intersects `invalidated`, injects that checkpoint_id, and invoke(None). The
  // parity matrix has NO row for any of this — schema fingerprint, node fingerprint,
  // _type_signature, auto-rewind, get_state_history divergence walk. See report.
  console.log("\nRun 2: analyze output changed (v1 -> v2)");
  console.log("  Expected: prepare and enrich preserved, analyze and report re-run");
  const pipelineV2 = constructFromFunctions("checkpoint-demo", [prepare, enrich, analyzeV2, reportV2]);
  const graphV2 = compile(pipelineV2, { checkpointer });
  const resultV2 = await run(graphV2, { input: { node_id: "demo" }, config });
  console.log(`\n  Result: ${(resultV2.report as Report).body}`);
}

// -- Demo 2: autoResume=false — CheckpointSchemaError ------------------------

async function demoStrictMode() {
  console.log("\n" + "=".repeat(60));
  console.log("DEMO 2: autoResume=false (strict mode)");
  console.log("=".repeat(60));

  const AnalysisV1 = z.object({ summary: z.string(), entityCount: z.number().int() });
  const AnalysisV2 = z.object({
    summary: z.string(),
    entityCount: z.number().int(),
    confidence: z.number().default(0),
  });
  type AnalysisV1 = z.infer<typeof AnalysisV1>;
  type AnalysisV2 = z.infer<typeof AnalysisV2>;

  const prepare = node({ outputs: Prepared, name: "prepare" }, (): Prepared => {
    const text = "The system validates all user inputs and logs access attempts.";
    return { text, tokenCount: text.split(" ").length };
  });
  const enrich = node({ outputs: Enriched, name: "enrich" }, (prepare: Prepared): Enriched => ({
    text: prepare.text,
    tokenCount: prepare.tokenCount,
    entities: ["system", "user", "inputs", "access"],
  }));
  const analyzeV1 = node({ outputs: AnalysisV1, name: "analyze" }, (enrich: Enriched): AnalysisV1 => ({
    summary: `Found ${enrich.entities.length} entities`,
    entityCount: enrich.entities.length,
  }));
  const reportV1 = node({ outputs: Report, name: "report" }, (analyze: AnalysisV1): Report => ({
    title: "Report",
    body: analyze.summary,
  }));
  const analyzeV2 = node({ outputs: AnalysisV2, name: "analyze" }, (enrich: Enriched): AnalysisV2 => ({
    summary: `Found ${enrich.entities.length} entities`,
    entityCount: enrich.entities.length,
    confidence: 0.95,
  }));
  const reportV2 = node({ outputs: Report, name: "report" }, (analyze: AnalysisV2): Report => ({
    title: "Report v2",
    body: analyze.summary,
  }));

  const checkpointer = new MemorySaver();
  const config = { configurable: { thread_id: "demo-strict" } };

  console.log("\nRun 1: full pipeline (v1)");
  const pipelineV1 = constructFromFunctions("strict-demo", [prepare, enrich, analyzeV1, reportV1]);
  const graphV1 = compile(pipelineV1, { checkpointer });
  await run(graphV1, { input: { node_id: "demo" }, config });

  console.log("\nRun 2: schema changed, autoResume=false");
  const pipelineV2 = constructFromFunctions("strict-demo", [prepare, enrich, analyzeV2, reportV2]);
  const graphV2 = compile(pipelineV2, { checkpointer });
  try {
    // PARITY(direct, given the subsystem exists): the auto_resume flag is a plain
    // run() option. The redesign is entirely under the hood in what raises this.
    await run(graphV2, { input: { node_id: "demo" }, config, autoResume: false });
    console.log("  ERROR: expected CheckpointSchemaError but none was raised");
  } catch (e) {
    if (e instanceof CheckpointSchemaError) {
      // PARITY(direct): error class + `invalidatedNodes` field. Python exposes a
      // set; TS uses a Set<string>. Error hierarchy row = Direct.
      console.log("  Caught CheckpointSchemaError (expected)");
      console.log(`  Invalidated nodes: ${[...e.invalidatedNodes].sort()}`);
      console.log(`  Message: ${e.message}`);
    } else {
      throw e;
    }
  }
}

async function main() {
  await demoAutoResume();
  await demoStrictMode();
}

void main();
