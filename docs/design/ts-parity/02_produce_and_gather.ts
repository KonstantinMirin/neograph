// Hypothetical TypeScript port of neograph example:
//   examples/02_produce_and_gather.py
//
// Feature-parity study ONLY. There is no @neograph TS package yet. This is
// what the code WOULD look like against the proposed API in
// docs/design/typescript-port.md (AD-0 transformer form + .pipe() + Zod).
// It is NOT meant to compile or run. `// PARITY:` marks every place the TS
// DX diverges from the Python original.
//
// Python features exercised by this example:
//   - @node(outputs=, model=, prompt=)                 -> think mode, no inputs
//   - @node(mode="agent", tools=[Tool(budget=2)], llm_config={announce_tool_budget})
//   - parameter-name wiring: research(decompose: Claims)
//   - construct_from_module(sys.modules[__name__])
//   - compile(llm_factory=, prompt_compiler=, tool_factories=)
//   - run(graph, input=)
//   - typed tool result (CodeReference) preserved in ToolInteraction.typed_result
//   - frozen Pydantic models

import { z } from "zod";
import {
  node,
  Tool,
  compile,
  run,
  constructFromFunctions, // PARITY: replaces construct_from_module (see below)
  type ToolInteraction,
} from "@neograph/core";

// ── Schemas ────────────────────────────────────────────────────────────────
// PARITY: Python `class Requirement(BaseModel, frozen=True)` is ONE declaration
// that is both the runtime validator AND the static type. In TS you write the
// Zod schema, then `z.infer` the type — two artifacts per model. `frozen=True`
// has no Zod equivalent: Zod validates shape, it does not freeze instances.
// You'd add `.readonly()` for the static type but nothing enforces immutability
// at runtime the way Pydantic frozen does.

const Requirement = z.object({ text: z.string() });
type Requirement = z.infer<typeof Requirement>;

const Claims = z.object({ items: z.array(z.string()) });
type Claims = z.infer<typeof Claims>;

const Finding = z.object({ claim: z.string(), evidence: z.string() });
type Finding = z.infer<typeof Finding>;

const ResearchResult = z.object({ findings: z.array(Finding) });
type ResearchResult = z.infer<typeof ResearchResult>;

// Typed tool result — what search_codebase returns. The framework preserves it
// in ToolInteraction.typedResult.
const CodeReference = z.object({
  query: z.string(),
  matches: z.number().int(),
  topFile: z.string(),
});
type CodeReference = z.infer<typeof CodeReference>;

// ── Pipeline nodes ──────────────────────────────────────────────────────────
// Step 1: decompose requirement into claims (single LLM call, think mode)
// Step 2: research claims with a search tool (budget: max 2 searches)

// PARITY: think-mode node with a DEAD body. In Python the body is a clean
// Ellipsis `...`; the LLM runs, the body never executes. TS has no equivalent
// "unimplemented but type-checks" expression. The callback still must satisfy
// its `: Claims` return type, so you cast an unreachable throw. This ceremony
// exists purely to feed the transformer a return annotation. See API-GAP #2.
//
// PARITY: `decompose()` takes NO parameters — it is a source node. The
// transformer emits `inputs: {}`; nothing to auto-wire. Direct, but note the
// node NAME must be captured. Python gets it free from `def decompose`. In TS
// `const decompose = node(...)` the variable name is not reflectively available
// at runtime; the AD-0 transformer must read the VariableDeclaration identifier
// to emit `name: "decompose"`, otherwise downstream param-name wiring (below)
// has nothing to match against. See API-GAP #1.
const decompose = node(
  { model: "fast", prompt: "req/decompose", outputs: Claims },
  (): Claims => {
    throw new Error("think-mode body never runs"); // PARITY: replaces Python `...`
  },
);

// PARITY: parameter-name wiring. Python `research(decompose: Claims)` binds to
// the upstream `decompose` node purely by the param name matching the node
// name. The transformer must extract BOTH the param name ("decompose") and its
// type (Claims schema). This is the "signature IS the DAG" claim — it survives
// here IF the transformer resolves the destructured param name back to the
// node identifier. Because the body is dead (agent mode), the param `decompose`
// is never actually read in the callback, so a naive transformer that only
// tracks USED params would drop the edge. It must key off the SIGNATURE, not
// usage.
const research = node(
  {
    mode: "agent",
    model: "reason",
    prompt: "req/research",
    outputs: ResearchResult,
    tools: [new Tool({ name: "search_codebase", budget: 2 })], // max 2 searches
    // Announce the budget up front so the model plans + batches. Off by
    // default; the announced count derives from the Tool budget above.
    llmConfig: { announceToolBudget: true },
  },
  // PARITY: with a real body you'd write `({ decompose }: { decompose: Claims })`
  // and the transformer maps the destructured key -> upstream node. But the
  // body is dead, so this signature is pure wiring metadata.
  (decompose: Claims): ResearchResult => {
    throw new Error("agent-mode body never runs"); // PARITY: replaces Python `...`
  },
);

// PARITY / API-GAP #1: Python does
//   construct_from_module(sys.modules[__name__], name="requirement-analysis")
// which walks the module namespace and collects every @node. TS has NO module
// introspection (typescript-port.md explicitly lists construct_from_module as
// "Not in v0.1.0-ts"). You must enumerate nodes by hand. The auto-assembly DX
// is gone; the node list is now a maintenance point that can silently drift
// from the declared nodes.
const pipeline = constructFromFunctions("requirement-analysis", [
  decompose,
  research,
]);

// ── Configure LLM layer + run ────────────────────────────────────────────────
// PARITY: llm_factory(tier), prompt_compiler, tool_factories all port DIRECT —
// same closures, camelCased keys. The fake LLMs from the Python file would
// become LangChain.js FakeChatModel / structured fakes; omitted here since this
// sketch is not runnable.

// PARITY: typed tool result. The Python FakeSearchTool returns a CodeReference
// *instance* and neograph keeps the typed object in ToolInteraction.typedResult.
// LangChain.js ToolMessage.content is fundamentally string-shaped, so preserving
// a typed object through the ReAct loop needs neograph to carry the object
// out-of-band alongside the stringified content. Expressible, but it fights the
// LangChain.js ToolMessage contract rather than riding on it. See friction.
function toolFactory(): { invoke(args: { query: string }): CodeReference } {
  return {
    invoke: ({ query }) => ({ query, matches: 3, topFile: "auth.py" }),
  };
}

async function main() {
  const graph = compile(pipeline, {
    llmFactory: (tier: string) => {
      // tier === "fast" -> decompose model; else -> research model
      throw new Error("wire LangChain.js fakes here");
    },
    promptCompiler: (_template: string, _data: unknown) => [
      { role: "user", content: "analyze" },
    ],
    toolFactories: { search_codebase: () => toolFactory() },
  });

  // PARITY: run() + input injection — DIRECT. Python `run(graph, input={...})`.
  const result = await run(graph, { input: { node_id: "REQ-042" } });

  // PARITY: result access. Python `result["decompose"].items`. In TS the state
  // bus is a Record keyed by node name; without a generated result type you get
  // `unknown` and must assert. The `outputs` schema is known at compile, but the
  // proposed API does not thread it into a typed `run()` return. See friction.
  const claims = result["decompose"] as Claims;
  console.log(`Decomposed into ${claims.items.length} claims:`);
  for (const claim of claims.items) console.log(`  - ${claim}`);

  console.log(`Research complete: ${result["research"] != null}`);

  // Inspect the typed tool interactions (Python: ToolInteraction.typed_result).
  const interactions = (result["research_tool_log"] ?? []) as ToolInteraction[];
  console.log(`Search tool called ${interactions.length} times (budget was 2)`);
}

void main();
