// TS parity sketch of: examples/vs_langgraph/04_human_in_the_loop.py
// HYPOTHETICAL — targets the proposed API in docs/design/typescript-port.md.
// NOT meant to compile or run; it grounds a feature-parity analysis.
//
// This example uses the PROGRAMMATIC / DECLARATIVE surface (plain `Node(...)`,
// `| Operator(...)`, `Construct(...)`, `compile(...)`, `run(...)`) — NOT the
// `@node` decorator. Per typescript-port.md AD-2 this is the "TS-first surface"
// that "maps naturally", so most of the port is DIRECT. The friction here is
// NOT in the DAG-authoring (no transformer/DI/describe_type is exercised); it is
// in the *dynamically-typed state* that conditions and results read, and in the
// `prompt_compiler` **kwargs introspection seam.

import { z } from "zod";
import { MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { Construct, Node, Operator, compile, run } from "@neograph/core";

const MODEL = "openai/gpt-4o-mini";

const llm = new ChatOpenAI({
  model: MODEL,
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});

// PARITY (Pydantic -> Zod, DIRECT-with-friction): Python's `class Analysis(BaseModel)`
// is BOTH the static type AND the runtime schema. Zod splits them: a schema value
// (`AnalysisSchema`) plus a derived static type (`Analysis`). Every model referenced
// by the IR (inputs=/outputs=) must be passed as the *schema value*, while the
// condition/result callbacks want the *type*. This duality recurs below.
const AnalysisSchema = z.object({
  claims: z.array(z.string()),
  confidence: z.number(),
});
type Analysis = z.infer<typeof AnalysisSchema>;

const ReportSchema = z.object({ text: z.string() });
type Report = z.infer<typeof ReportSchema>;

// ─────────────────────────────────────────────────────────────────────────────
// NEOGRAPH-TS WAY — Node(...).pipe(Operator({when})), done.
// ─────────────────────────────────────────────────────────────────────────────

async function runNeographTs(): Promise<string> {
  // PARITY (Node model, DIRECT): declarative `Node(...)` maps 1:1. `mode: "think"`
  // means the node has no body — nothing for the AD-0 transformer to extract, so
  // this surface needs NO transformer at all. `outputs`/`inputs` take Zod schemas
  // where Python takes the Pydantic class (see duality note above).
  const analyze = Node({
    name: "analyze",
    mode: "think",
    outputs: AnalysisSchema,
    model: "fast",
    prompt: "analyze",
  });

  const report = Node({
    name: "report",
    mode: "think",
    inputs: AnalysisSchema, // PARITY: single-type `inputs=Analysis` shorthand -> single schema
    outputs: ReportSchema,
    model: "fast",
    prompt: "report",
  });

  // PARITY (pipe -> .pipe(), REDESIGN per matrix, trivial in practice): Python's
  // `analyze | Operator(when="low_confidence")` overloads `__or__`. TS has no
  // operator overloading, so `.pipe(Operator({...}))`. Mutual-exclusion guards
  // (Operator+Loop etc.) live inside `.pipe()` — same logic, DIRECT.
  const pipeline = Construct("review", {
    nodes: [analyze.pipe(Operator({ when: "low_confidence" })), report],
  });

  const graph = compile(pipeline, {
    checkpointer: new MemorySaver(),
    llmFactory: (_tier: string) => llm, // PARITY (DIRECT): `lambda tier: llm`

    // PARITY (prompt_compiler, REDESIGN — the sharpest friction in this example):
    // Python's compiler is `lambda template, data, **kw: [...]`. The `**kw` is
    // load-bearing framework machinery: neograph introspects the compiler's
    // signature (`_accepted_params`) to decide whether to pass `di_inputs`. TS has
    // NO runtime signature introspection — you cannot ask "does this closure accept
    // a `diInputs` param?". The proposed API must replace the introspection gate
    // with an explicit contract: always hand the compiler a single options object
    // and let it ignore fields it doesn't want. (This example never uses di_inputs,
    // so behaviourally it's fine, but the *seam* does not port as-is.)
    promptCompiler: ({ template, data }: { template: string; data: unknown }) => [
      {
        role: "user",
        content:
          template === "analyze"
            ? "Analyze 'microservice security'. List 3 claims and rate confidence 0-1."
            : `Write a brief report based on: ${JSON.stringify(data)}`,
      },
    ],

    // PARITY (conditions registry, DIRECT logic / GAP on the state type):
    // The condition returns an interrupt PAYLOAD (dict) or null; `Operator(when=...)`
    // turns a non-null return into a LangGraph interrupt. The closure logic ports
    // directly. What does NOT port is the STATE TYPE: in Python `state` is the
    // dynamically-generated Pydantic state model, so `state.analyze.confidence` is a
    // real attribute (field name == node name "analyze"). In TS the compiled state
    // model is generated at runtime via `Annotation.Root` (AD-4) and has no static
    // TS type, so `state.analyze` is untyped (`any`) — no compile-time guarantee the
    // "analyze" field exists or that it is an `Analysis`. Hand-annotating helps but
    // is unchecked against the actual graph.
    conditions: {
      low_confidence: (state: { analyze: Analysis | null }) =>
        state.analyze && state.analyze.confidence < 0.8
          ? {
              // PARITY (format spec, DIRECT-trivial): Python `:.0%` has no TS
              // equivalent; spell it out.
              message: `Confidence ${Math.round(
                state.analyze.confidence * 100
              )}% is low. Approve?`,
            }
          : null,
    },
  });

  const config = { configurable: { thread_id: "neo-ts-demo" } };

  // PARITY (run kwargs -> options object, DIRECT): Python `run(graph, input=..., config=...)`
  // becomes an options object. `input={"node_id": "demo"}` is run-input.
  let result: Record<string, any> = await run(graph, {
    input: { node_id: "demo" },
    config,
  });

  // PARITY (interrupt detection, DIRECT): `"__interrupt__" in result` works on a JS
  // object; the nested `.value.message` access ports 1:1. But `result` is the same
  // untyped state bag (GAP) — `result.__interrupt__[0].value.message` is `any`.
  if ("__interrupt__" in result) {
    console.log(`  Interrupted: ${result.__interrupt__[0].value.message}`);
    // PARITY (resume, DIRECT): two-phase invocation, `run(graph, {resume, config})`.
    result = await run(graph, { resume: { approved: true }, config });
  }

  // PARITY (result access, GAP): Python `result["report"].text`. In TS the "report"
  // field and its `.text` are untyped — the dynamically-generated state model gives
  // no static type back to the caller. A `Report` cast is manual and unchecked.
  return (result.report as Report).text;
}

// ─────────────────────────────────────────────────────────────────────────────

async function main() {
  console.log("=".repeat(60));
  console.log("NEOGRAPH-TS (Node.pipe(Operator({when})) + interrupt/resume):");
  console.log("=".repeat(60));
  console.log(await runNeographTs());
}

void main;
