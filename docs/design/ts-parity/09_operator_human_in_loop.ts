// TS parity sketch of: examples/09_operator_human_in_loop.py
// Feature: @node(interrupt_when=...) human-in-the-loop, checkpointer-backed
//          pause/resume via run(resume=...). Scripted nodes only (no LLM).
//
// Target: the PROPOSED @neograph/core TS API (docs/design/typescript-port.md).
// This is a HYPOTHETICAL sketch — not compilable, not runnable. It exists to
// surface where the DX degrades vs the Python original.
//
// AD-0 transformer form used for node signatures ("signature IS the DAG").
// Explicit Zod schemas stand in for the frozen Pydantic models.

import { z } from "zod";
import { MemorySaver } from "@langchain/langgraph";
import {
  node,
  compile,
  run,
  constructFromFunctions, // PARITY: construct_from_module has NO TS equivalent (see notes)
  Command,                 // PARITY: resume payload becomes a LangGraph.js Command
} from "@neograph/core";

// ── Schemas ────────────────────────────────────────────────────────────────
// PARITY[schema]: frozen Pydantic BaseModel -> Zod schema + z.infer type.
//   `frozen=True` immutability is not expressible in Zod; the closest is
//   `Readonly<T>` on the inferred type (compile-time only, no runtime freeze).

const AnalysisSchema = z.object({
  claims: z.array(z.string()),
  // PARITY[naming]: Python `coverage_pct` (snake_case) becomes camelCase by TS
  // convention — BUT this field name leaks into the interrupt_when closure and
  // into result-dict keys below, so the rename must be applied consistently or
  // parity breaks silently. See interruptWhen note.
  coveragePct: z.number().int(),
});
type Analysis = Readonly<z.infer<typeof AnalysisSchema>>;

const ValidationResultSchema = z.object({
  passed: z.boolean(),
  issues: z.array(z.string()),
});
type ValidationResult = Readonly<z.infer<typeof ValidationResultSchema>>;

const FinalReportSchema = z.object({
  text: z.string(),
});
type FinalReport = Readonly<z.infer<typeof FinalReportSchema>>;

// ── Pipeline (node() wrapper — the @node decorator has no standalone-fn form) ─
// PARITY[decorator]: Python `@node(mode="scripted", outputs=Analysis)` on a
//   module-level `def` becomes the wrapper `node({...}, fn)`. TS decorators
//   cannot decorate standalone functions, so the decorator syntax is lost.

const analyze = node(
  { mode: "scripted", outputs: AnalysisSchema },
  (): Analysis => ({ claims: ["auth", "logging", "encryption"], coveragePct: 55 })
);

const check = node(
  {
    mode: "scripted",
    outputs: ValidationResultSchema,
    // PARITY[interrupt_when — THE core friction of this example]:
    // Python passes a lambda that reads `state.check.issues` / `state.check.passed`
    // by ATTRIBUTE and returns EITHER a payload dict OR None. Two problems in TS:
    //
    //  (1) `state` is the compiled state model — a map of node-name -> that
    //      node's output. Its shape is only known AFTER construct assembly, so
    //      no transformer can hand this closure a typed `state`. It is `any`
    //      (shown here as an explicit interface we must hand-maintain).
    //  (2) The Feature Parity Matrix only documents `interruptWhen: "cond"`
    //      (a STRING condition -> bool). The RICH callable form used here
    //      returns a structured payload object, not a bool. That form is an
    //      API GAP — see report.
    interruptWhen: (state: StateView): InterruptPayload | null =>
      state.check && !state.check.passed
        ? { issues: state.check.issues, message: "Please review and approve" }
        : null,
  },
  // PARITY[auto-wiring]: `analyze: Analysis` param name = edge from the analyze
  //   node. Transformer must emit `inputs: { analyze: AnalysisSchema }`. Note
  //   the param name must match the node's *variable/emitted name* ("analyze"),
  //   which in TS is not guaranteed to equal the source fn name the way Python's
  //   `def analyze` guarantees it. Assembly must pin node names explicitly.
  (analyze: Analysis): ValidationResult =>
    analyze.coveragePct < 80
      ? { passed: false, issues: [`Coverage ${analyze.coveragePct}% is below 80% threshold`] }
      : { passed: true, issues: [] }
);

const report = node(
  { mode: "scripted", outputs: FinalReportSchema },
  (analyze: Analysis): FinalReport => ({
    text: `Report: ${analyze.claims}, coverage: ${analyze.coveragePct}%`,
  })
);

// PARITY[state typing]: this interface is HAND-WRITTEN to give interruptWhen a
// typed `state`. Python needs nothing here — the state proxy resolves node
// names dynamically. This is pure TS overhead and must be kept in sync with the
// node set by hand (no transformer reaches it — the shape depends on assembly).
interface StateView {
  analyze?: Analysis;
  check?: ValidationResult;
  report?: FinalReport;
}
interface InterruptPayload {
  issues: string[];
  message: string;
}

// PARITY[construct_from_module — BLOCKED]: Python introspects the module
//   (`sys.modules[__name__]`) to collect every @node. TS has no runtime module
//   introspection and the design doc explicitly excludes construct_from_module
//   from v0.1.0-ts. The nodes must be listed EXPLICITLY. This also means node
//   *names* must be passed/pinned here rather than inferred from `def` names.
const pipeline = constructFromFunctions("review-pipeline", [analyze, check, report]);

// ── Run with checkpointer (required for interrupt/resume) ────────────────────

async function main() {
  // PARITY[checkpointer requirement]: Direct — compile() enforcing a
  //   checkpointer when an Operator/interrupt node is present ports 1:1.
  //   MemorySaver exists in LangGraph.js.
  const graph = compile(pipeline, { checkpointer: new MemorySaver() });

  const config = { configurable: { thread_id: "review-001" } };

  console.log("=== First run: will pause for human review ===\n");
  // PARITY[run/result]: `run` returns a heterogeneous dict keyed by node name
  //   plus framework keys (`__interrupt__`). In Python that's a plain dict; in
  //   TS it is `Record<string, unknown>` and every access needs a cast. The
  //   nice typed access `result['analyze'].coverage_pct` is lost — see the
  //   casts below. An API GAP: run() has no typed-result overload.
  let result = await run(graph, { input: { node_id: "REQ-001" }, config });

  console.log(`Analysis: ${(result.analyze as Analysis).coveragePct}% coverage`);
  console.log(`Validation: passed=${(result.check as ValidationResult).passed}`);

  // PARITY[__interrupt__]: LangGraph.js surfaces interrupts on the result too,
  //   but the exact key/shape differs from Python's `result["__interrupt__"]`
  //   list of Interrupt(value=...). neograph-ts would need to normalize it back
  //   to the Python shape to keep the DX identical. Shown assuming it does.
  if ("__interrupt__" in result) {
    const interruptData = result.__interrupt__ as Array<{ value: unknown }>;
    console.log("\nGraph PAUSED. Interrupt payload:");
    for (const intr of interruptData) {
      console.log(`  ${JSON.stringify(intr.value)}`);
    }

    console.log("\n=== Human approves, resuming... ===\n");
    // PARITY[resume]: Python `run(graph, resume={...})` maps to passing a
    //   LangGraph.js `Command({ resume: {...} })`. neograph-ts can hide this
    //   behind a `resume:` option to keep parity; shown with the option form.
    result = await run(graph, {
      resume: { approved: true, reviewer: "alice" },
      config,
    });

    // PARITY[result.get]: Python `result.get('human_feedback')` -> optional key
    //   access. In TS, `result.human_feedback` on Record<string, unknown> is
    //   fine but untyped.
    console.log(`Human feedback recorded: ${result.human_feedback}`);
    console.log(`Report generated: ${(result.report as FinalReport).text}`);
  } else {
    console.log("Validation passed — no interrupt needed");
    console.log(`Report: ${(result.report as FinalReport).text}`);
  }
}

void main();
