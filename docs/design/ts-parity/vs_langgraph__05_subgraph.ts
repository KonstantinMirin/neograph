// TS PARITY SKETCH — HYPOTHETICAL, NOT COMPILABLE.
// Python source: examples/vs_langgraph/05_subgraph.py (the run_neograph() half)
// Proposed API: docs/design/typescript-port.md
//
// Feature under test: SUBGRAPH COMPOSITION — a sub-pipeline with a declared
// input=/output= boundary and isolated state, nested inline in a parent
// pipeline (no manual parent<->child state-mapping wrapper). This is the whole
// value prop the Python example sells against LangGraph (~12 lines vs ~45: the
// LangGraph half needs 2 StateGraphs, 2 TypedDicts, and a hand-written
// enrich_wrapper() that translates parent state <-> child state).
//
// Surface: PROGRAMMATIC/declarative (Node({...}) + Construct + compile + run),
// NOT the @node decorator. So the AD-0 ts-patch transformer is IRRELEVANT here:
// declarative nodes already carry explicit inputs=/outputs=, there is no fn
// signature to extract. Matrix line 110 rates this the "TS-first surface", and
// for the STRUCTURE that holds — the nesting ports Direct. The friction is at
// the compile() seam (prompt_compiler), the untyped run() result, and the
// nominal-vs-structural boundary-port matching. See // PARITY notes inline and
// the report.

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
// PARITY: DIRECT (with an ergonomic tax). Pydantic BaseModel -> Zod object.
// Because every node here is mode:"think", the outputs= schema is fed straight
// into LangChain.js withStructuredOutput() (AD-1) — so structured output is
// FREE and this is actually a smoother path than the scripted 05 variant.
// The tax: Python's `class ScoredClaim(BaseModel)` is BOTH a runtime schema AND
// a static type in one declaration; TS needs the schema value AND a separate
// `z.infer` alias if you want the static type (see Report below).
const Claims = z.object({ items: z.array(z.string()) });

// PARITY: `confidence: str  # high / medium / low` is a bare str in Python (a
// comment, not a constraint). Faithful port keeps z.string(); a TS author would
// be TEMPTED to tighten it to z.enum(["high","medium","low"]) — which is
// actually BETTER DX than the Python original, but would diverge from source.
const ScoredClaim = z.object({ claim: z.string(), confidence: z.string() });
const ScoredClaims = z.object({ scored: z.array(ScoredClaim) });

const Report = z.object({ text: z.string() });
type Report = z.infer<typeof Report>;

// ── Sub-pipeline: enrich — declared I/O boundary, isolated state ──────────
// PARITY: DIRECT. THE headline of this example.
// Python:  Construct("enrich", input=Claims, output=ScoredClaims, nodes=[...])
// TS:      new Construct("enrich", { input: Claims, output: ScoredClaims, nodes: [...] })
//
// `input`/`output` are the SINGULAR boundary ports (Construct.input /
// Construct.output — NOT Node.inputs/outputs plural). The sub-pipeline receives
// an isolated state seeded from `neo_subgraph_input` and only `output` surfaces
// back to the parent. Matrix rates "Subgraph compilation (Recursive compile)"
// and "Output boundary contract" both Direct — the compiler mechanic
// (make_subgraph_fn recursion) ports unchanged onto LangGraph.js subgraphs.
//
// PARITY CAVEAT (nominal -> structural): the boundary port `input: Claims` is a
// SINGLE-TYPE form. At runtime the compiler feeds the port by finding the
// upstream bus value that matches Claims (Python: an isinstance/type scan —
// NOMINAL). Zod is STRUCTURAL: Claims {items:string[]} is indistinguishable
// from any other {items:string[]} producer. One Claims producer here, so it's
// unambiguous — but the resolution MECHANISM leans on nominal identity that
// Zod doesn't provide. See API GAP #3.
const enrich = new Construct("enrich", {
  input: Claims,
  output: ScoredClaims,
  nodes: [
    new Node({
      name: "score",
      mode: "think",
      inputs: Claims,
      outputs: ScoredClaims,
      model: "fast",
      prompt: "score",
    }),
  ],
});

// ── Parent pipeline: decompose -> [enrich sub-pipeline] -> report ─────────
// PARITY: DIRECT. A Construct nested directly in another Construct's nodes[].
// The `enrich` sub-pipeline sits INLINE between decompose and report with NO
// wrapper function — exactly what the Python example sells vs LangGraph's
// hand-written enrich_wrapper(). TS nodes[] accepts (Node | Construct)[] just
// like Python's list[Node | Construct]. This is the parity win: the ~45->~12
// line collapse and the elimination of manual state mapping both survive.
const pipeline = new Construct("analysis", {
  nodes: [
    new Node({
      name: "decompose",
      mode: "think",
      outputs: Claims,
      model: "fast",
      prompt: "decompose",
    }),
    enrich, // ← sub-pipeline, isolated state, NO wrapper function
    new Node({
      name: "report",
      mode: "think",
      inputs: ScoredClaims, // reads the sub-construct's output boundary
      outputs: Report,
      model: "fast",
      prompt: "report",
    }),
  ],
});

// ── compile() ─────────────────────────────────────────────────────────────
const graph = compile(pipeline, {
  // PARITY: DIRECT. Python `llm_factory=lambda tier: llm`.
  llmFactory: (_tier: string) => llm,

  // PARITY: REDESIGN — the sharpest divergence, identical to vs_langgraph__01.
  //
  // Python signature:  prompt_compiler=lambda template, data, **kw: [{...}]
  //   - `**kw` is variadic KEYWORD args. At runtime neograph decides WHICH
  //     kwargs to actually pass by introspecting the compiler's declared
  //     parameters (`_accepted_params` -> `prompt_compiler_params`, the same
  //     inspect.signature seam that gates `di_inputs`, neograph-euyh; see
  //     src/neograph/_llm_render.py:211-215). A compiler that declares `config`
  //     or `di_inputs` opts in; one that only declares (template, data) never
  //     receives them.
  //   - In THIS example `**kw` is declared-but-unused (swallowed), so the
  //     gating is invisible here — but the MECHANISM is the port blocker.
  //
  // TS has NO keyword args and ERASES parameter names at runtime, so:
  //   (a) `**kw` collapses to a single trailing options OBJECT, and
  //   (b) `_accepted_params` introspection is IMPOSSIBLE — you cannot ask a JS
  //       closure "do you declare a `diInputs` parameter?". neograph-ts must
  //       therefore ALWAYS pass the full opts bag and let the compiler
  //       cherry-pick, which silently changes the neograph-euyh demand-gated
  //       opt-in semantics. See API GAP #1.
  //   (c) `data` is the rendered upstream value; JSON.stringify stands in for
  //       Python's f-string interpolation of the Pydantic model.
  promptCompiler: (
    template: string,
    data: unknown,
    _opts: { config?: { configurable?: Record<string, unknown> } } = {},
  ) => {
    const content =
      template === "decompose"
        ? "Break 'API rate limiting' into 3-5 factual claims."
        : template === "score"
          ? `Score each claim as high/medium/low: ${JSON.stringify(data)}`
          : `Write a brief report: ${JSON.stringify(data)}`;
    return [{ role: "user", content }];
  },
});

// ── run() ─────────────────────────────────────────────────────────────────
// PARITY: MOSTLY-DIRECT on the call; the RETURN TYPE degrades.
// Python: run(graph, input={"node_id": "demo"}) returns a dict keyed by node
// name; result["report"].text is dynamic dict access (Python doesn't type it
// either). In TS the same result is Record<string, unknown> — there is NO
// type-level link between the node NAME "report" and its declared output schema
// Report, because the state model is BUILT AT RUNTIME from the construct
// (LangGraph.js Annotation.Root state is not reflected back into run()'s return
// type). A TS user reasonably expects result.report.text to be typed; instead
// they must cast. Also note: "score" (inside enrich) is NOT a result key — the
// sub-construct's isolated state proves the boundary held. See API GAP #2.
async function main() {
  const result = await run(graph, { input: { node_id: "demo" } });
  const report = (result["report"] as Report).text;
  console.log(report);
}

void main;
export { pipeline };
