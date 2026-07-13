// TS PARITY SKETCH — HYPOTHETICAL, NOT COMPILABLE.
// Python source: examples/vs_langgraph/01_sequential_pipeline.py (the run_neograph() half)
// Proposed API: docs/design/typescript-port.md
//
// This example uses the PROGRAMMATIC surface (Node(...) + Construct + compile + run),
// NOT the @node decorator. Therefore the AD-0 ts-patch transformer is IRRELEVANT here:
// the programmatic surface already carries explicit inputs=/outputs=, so there is no
// function signature to extract. This is the "TS-first surface" (matrix line 110) and
// it is the highest-parity path in the whole port. The friction that remains is entirely
// in the compile() seam (prompt_compiler) and the untyped run() result — see PARITY notes.

import { z } from "zod";
import { Node, Construct, compile, run } from "@neograph/core";
import { ChatOpenAI } from "@langchain/openai";

const MODEL = "openai/gpt-4o-mini";

const llm = new ChatOpenAI({
  model: MODEL,
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});

// ── Schemas ─────────────────────────────────────────────────────────────
// PARITY: DIRECT. Pydantic BaseModel -> Zod object. LangChain.js withStructuredOutput()
// consumes Zod natively (AD-1), so think-mode structured output is free.
// The one ergonomic tax: Python gets `class Claims(BaseModel)` as BOTH a runtime schema
// AND a static type in one declaration; TS needs the schema value + a separate
// `z.infer` type alias if you want the static type.
const Claims = z.object({ items: z.array(z.string()) });
const Classification = z.object({ claim: z.string(), category: z.string() });
const ClassifiedClaims = z.object({ classified: z.array(Classification) });
const Summary = z.object({ text: z.string() });

type Summary = z.infer<typeof Summary>;

// ── 3 nodes, ordered. No addEdge, no addNode, no StateGraph. ─────────────
// PARITY: DIRECT. Programmatic Node(...) maps 1:1 to a Node({...}) options object
// (matrix line 93/110). `mode: "think"` is a string-literal union member.
// `inputs: Claims` is the single-type (backward-compat) form — skips fan-in validation,
// defers to a runtime schema scan; ports unchanged.
const decompose = new Node({
  name: "decompose",
  mode: "think",
  outputs: Claims,
  model: "fast",
  prompt: "decompose",
});

const classify = new Node({
  name: "classify",
  mode: "think",
  inputs: Claims,
  outputs: ClassifiedClaims,
  model: "fast",
  prompt: "classify",
});

const summarize = new Node({
  name: "summarize",
  mode: "think",
  inputs: ClassifiedClaims,
  outputs: Summary,
  model: "fast",
  prompt: "summarize",
});

const pipeline = new Construct("analysis", { nodes: [decompose, classify, summarize] });

const graph = compile(pipeline, {
  llmFactory: (_tier: string) => llm,

  // PARITY: REDESIGN — this is the sharpest divergence in the example.
  //
  // Python signature:  prompt_compiler=lambda template, data, **kw: [...]
  //   - `**kw` is variadic KEYWORD args. neograph decides WHICH kwargs to pass by
  //     introspecting the compiler's signature (`_accepted_params` /
  //     `prompt_compiler_params` filter — the same `inspect.signature` seam that gates
  //     `di_inputs`, neograph-euyh). A compiler that declares `config` (or `**kw`) gets
  //     config; one that declares `di_inputs` opts into DI values; others don't.
  //   - The body reads `kw.get('config',{}).get('configurable',{}).get('topic','AI')`.
  //
  // TS has NO keyword args and erases parameter names at runtime, so:
  //   (a) `**kw` collapses to a single trailing options OBJECT, and
  //   (b) `_accepted_params` introspection is IMPOSSIBLE — you cannot ask a JS closure
  //       "do you accept a `diInputs` parameter?". So neograph-ts must ALWAYS pass the
  //       full opts bag and let the compiler cherry-pick. That silently changes the
  //       neograph-euyh opt-in semantics (di_inputs was demand-gated precisely so an
  //       unaware compiler never received it). See API GAP #1 in the report.
  //   (c) the `.get(...).get(...)` safe-nav chain becomes optional chaining `?.`.
  promptCompiler: (
    template: string,
    data: unknown,
    opts: { config?: { configurable?: Record<string, unknown> } } = {},
  ) => {
    const topic = opts.config?.configurable?.topic ?? "AI";
    const content =
      template === "decompose"
        ? `Break this topic into 3-5 factual claims: ${topic}`
        : template === "classify"
          ? `Classify each claim (security/reliability/performance): ${JSON.stringify(data)}`
          : `Summarize in one paragraph: ${JSON.stringify(data)}`;
    return [{ role: "user", content }];
  },
});

// PARITY: MOSTLY-DIRECT on the call, but the RETURN TYPE degrades.
// Python: `run(graph, input={"node_id": "demo", "topic": "..."})` returns a dict keyed
// by node name; `result["summarize"].text` is dynamic dict access (Python doesn't type it
// either). In TS the same result is `Record<string, unknown>` — there is NO type-level
// link between the node NAME string "summarize" and its declared output schema `Summary`.
// A TS user expects `result.summarize.text` to be typed; instead they must cast.
// See API GAP #2 in the report.
const result = run(graph, { input: { node_id: "demo", topic: "microservice authentication" } });
const summary = (result["summarize"] as Summary).text;

export { summary };
