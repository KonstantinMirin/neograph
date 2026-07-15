// TS PARITY SKETCH — HYPOTHETICAL, NOT COMPILABLE.
// Python source: examples/vs_langgraph/03_map_reduce.py  (run_neograph path only)
// Target API: docs/design/typescript-port.md (AD-0 transformer; .pipe(); Zod schema layer).
//
// Feature focus of THIS example (neograph path):
//   - PROGRAMMATIC surface: Node({...}) | Oracle(...)  — NOT @node, NOT DI, NOT ForwardConstruct.
//   - Oracle(n=3, merge_prompt="pick-best")  — fan-out N think-mode generators + LLM merge.
//   - mode="think" structured output (outputs=Jokes).
//   - compile(llm_factory, prompt_compiler)  — the user-supplied prompt_compiler callback.
//   - run(graph, input).
//
// Because this is the PROGRAMMATIC surface, the AD-0 compiler transformer is IRRELEVANT here:
// there is no function signature to extract. Schemas are declared explicitly as Zod (this is
// the same shape the proposed API calls its "TS-first surface", matrix line 110). So most of
// typescript-port.md's *hard* problems (frame/AST reflection, __bool__ re-tracing, describe_type)
// do NOT surface in this example. What surfaces instead: the prompt_compiler contract, the
// runtime SHAPE of Oracle variants, and Pydantic Field-constraint -> Zod.

import { z } from "zod";
import { Node, Oracle, Construct, compile, run } from "@neograph/core";
import { ChatOpenAI } from "@langchain/openai";

const MODEL = "openai/gpt-4o-mini";

const llm = new ChatOpenAI({
  model: MODEL,
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY (direct): Pydantic BaseModel -> Zod object. AD-1 makes Zod the schema layer, and
// LangChain.js withStructuredOutput() consumes Zod natively, so no adapter.
const Jokes = z.object({
  items: z.array(z.string()),
});
type Jokes = z.infer<typeof Jokes>;

// PARITY (direct, but UNUSED in the neograph path — same as Python): BestJoke with Field(ge=0)
// is only referenced by the LangGraph comparison, never by run_neograph. Kept to mirror source.
// Field(description=..., ge=0)  ->  z.number().int().gte(0).describe(...). The `ge=0` constraint
// would flow to the LLM via describe_type; matrix line 220 claims describe_type is "already
// TS-native notation" (Direct). Not exercised here because run_neograph builds prompts by hand.
const BestJoke = z.object({
  id: z.number().int().gte(0).describe("Index of the best joke, starting with 0"),
});

// ── Pipeline ───────────────────────────────────────────────────────────────
// PARITY (Redesign, but mechanical here): Python `Node(...) | Oracle(...)` becomes
// `Node({...}).pipe(Oracle({...}))` (matrix line 101). Operator overloading is unavailable in
// TS, so `|` -> `.pipe()`. For a single modifier this is a pure syntax swap; the mutual-exclusion
// guards (Oracle+Each etc.) still live inside .pipe() (matrix line 102, Direct).
const generate = Node({
  name: "jokes",
  mode: "think",          // string literal, same as Python. Programmatic Node carries mode
                          // explicitly; no inference needed on this surface.
  outputs: Jokes,         // Zod schema stands in for the Pydantic class.
  model: "fast",
  prompt: "generate",
}).pipe(
  Oracle({ n: 3, mergePrompt: "pick-best" })
  // PARITY (API GAP): the Python Oracle also accepts merge_pre_process / merge_post_process /
  // merge_fallback hooks (source lines 116-118). typescript-port.md's Oracle row (line 96) lists
  // only {n, models, merge_fn/merge_prompt}. The three merge_* hook callbacks are NOT in the
  // proposed field set — an unlisted gap for the merge_prompt path.
);

const pipeline = Construct("joke-contest", { nodes: [generate] });

const graph = compile(pipeline, {
  // PARITY (direct): llm_factory(tier) -> (tier) => llm. The "fast" tier resolves here.
  llmFactory: (_tier: string) => llm,

  // PARITY (API GAP — the sharpest one this example exposes): the user-supplied prompt_compiler.
  // Python signature is `(template, data, **kw) -> list[{role, content}]`. typescript-port.md
  // documents DefaultPromptCompiler.build_vars internals (factory row, line 203) but NEVER
  // specifies the USER-FACING compile-time prompt_compiler callback signature — not its params,
  // not its return type, not the **kw introspection-gating seam (di_inputs etc.). This example
  // depends entirely on that callback, so the contract must be pinned in the TS API.
  //
  // Two sub-gaps inside `data`:
  //  (1) VARIANT SHAPE. On the merge call Python hands data = {"variants": [<Jokes instance>, ...]}
  //      and the body does `v.items[0]` via Pydantic ATTRIBUTE access. In TS the variants would be
  //      Zod-PARSED plain objects, so `v.items[0]` is property access — works, but the API must
  //      state that variants arrive as z.infer<> objects (not class instances, not raw JSON).
  //  (2) `**kw` catch-all. TS has no **kwargs; the introspection-gated di_inputs kwarg
  //      (neograph-euyh) that Python passes only when the compiler declares it has no TS analogue
  //      here. Would become an explicit optional options arg the runtime feature-detects — a
  //      Redesign of the gating mechanism, though this specific example never reads di_inputs.
  promptCompiler: (
    template: string,
    data: { variants?: Jokes[] } & Record<string, unknown>,
    _opts?: Record<string, unknown>,   // stands in for Python **kw
  ): Array<{ role: string; content: string }> => {
    if (template === "generate") {
      return [{ role: "user", content: "Write a short joke about programming languages." }];
    }
    // merge_prompt="pick-best" branch. data.variants is the list of Jokes produced by the 3
    // parallel generators (neograph-iu05: read the variant list, not the dict).
    const variants = data.variants ?? [];
    const body = variants
      .filter((v) => v.items.length > 0)
      .map((v) => `- ${v.items[0]}`)
      .join("\n");
    return [{
      role: "user",
      content: "Pick the best joke and return it as a single item list:\n" + body,
    }];
  },
});

// PARITY (direct): run(graph, input) -> await run(graph, {input}). LangGraph.js is async-first,
// so run() is a Promise; Python's sync run() has an implicit sync/async split the TS port collapses
// to always-async (matrix Runner row, Direct). input={"node_id": "demo"} passes through unused —
// the hand-written promptCompiler ignores it (same as Python).
const result = await run(graph, { input: { node_id: "demo" } });

// PARITY (direct): result["jokes"] is the merged Jokes object; typed as Jokes via z.infer.
const best = result.jokes.items.length > 0 ? result.jokes.items[0] : "no joke";
console.log(best);
