// TS parity sketch of: examples/13_gather_produce_subconstruct.py
// HYPOTHETICAL — written against the PROPOSED neograph-ts API in
// docs/design/typescript-port.md (AD-0 transformer form). NOT runnable; no TS
// implementation exists. `// PARITY:` notes flag where the TS DX diverges from
// the Python original.
//
// Python features this example exercises (the richest example in the set):
//   - @node(mode="agent", outputs={result, tool_log}, model=, prompt=, tools=[Tool(...)],
//           llm_config={"announce_tool_budget": True})   -> ReAct agent node, dict-form outputs
//   - @node(mode="think", outputs=ClaimVerdict)          -> think node consuming DICT-OUTPUT REFS
//       score(explore_result, explore_tool_log)          -> `{upstream}_{key}` param-name resolution
//   - construct_from_functions("verify-claim", [...], input=VerifyClaim, output=ClaimVerdict)
//       -> sub-construct boundary + PORT PARAM RESOLUTION (claim: VerifyClaim -> neo_subgraph_input)
//   - verify_claim.map("flatten_claims.claims", key="claim_id")  -> Each fan-out on a SUB-CONSTRUCT
//   - deterministic_merge(verify_claim: dict[str, ClaimVerdict]) -> merge-after-fan-out (dict consumer)
//   - Typed tool results: FakeEvidenceSearch returns EvidenceHit (Pydantic model, not str),
//     preserved on ToolInteraction.typed_result, rendered as JSON for the LLM
//   - construct_from_module(types.ModuleType(...))       -> dynamic module assembly (x2 demos)
//   - context=["build_catalog"]                          -> verbatim state injection
//   - compile(pipeline, llm_factory=, prompt_compiler=, tool_factories=) / run(graph, input=)
//   - Fake LangChain clients: bind_tools / with_structured_output / invoke, AIMessage + tool_calls,
//     ToolMessage history sniffing

import { z } from "zod";
import {
  node,
  constructFromFunctions,
  Tool,
  ToolInteraction,          // PARITY: framework class from @neograph/core
  compile,
  run,
  type LlmFactory,          // PARITY: proposed (tier: string) => BaseChatModel
  type PromptCompiler,      // PARITY: proposed (template, data, ...) => BaseMessage[]
  type ToolFactory,         // PARITY: proposed (config, toolConfig) => StructuredTool
} from "@neograph/core";
import { AIMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";

// ── Schemas ──────────────────────────────────────────────────────────────────
// PARITY (frozen=True): every model here is `BaseModel, frozen=True`. Zod has no
// frozen concept — `.parse()` returns a plain mutable object. `.readonly()` gives
// a compile-time `Readonly<T>` only. LOW severity (nothing mutates), but the
// Python models are hashable value objects and the TS ones are not — see the
// Each keying note far below where it *almost* matters.

const VerifyClaim = z.object({
  claimId: z.string(),
  text: z.string(),
}).readonly();
type VerifyClaim = z.infer<typeof VerifyClaim>;

const ClaimBatch = z.object({
  claims: z.array(VerifyClaim),
}).readonly();
type ClaimBatch = z.infer<typeof ClaimBatch>;

const ExplorationResult = z.object({
  evidence: z.array(z.string()),
  summary: z.string(),
}).readonly();
type ExplorationResult = z.infer<typeof ExplorationResult>;

const ClaimVerdict = z.object({
  claimId: z.string(),
  disposition: z.string(),
  reasoning: z.string(),
}).readonly();
type ClaimVerdict = z.infer<typeof ClaimVerdict>;

// Typed tool result — what search_evidence returns. The framework preserves this
// on ToolInteraction.typed_result and renders it as JSON for the LLM.
const EvidenceHit = z.object({
  sourceFile: z.string(),
  line: z.number().int(),
  snippet: z.string(),
  relevance: z.number(),
}).readonly();
type EvidenceHit = z.infer<typeof EvidenceHit>;

const FinalReport = z.object({
  summary: z.string(),
  claimCount: z.number().int(),
}).readonly();
type FinalReport = z.infer<typeof FinalReport>;

// ── Fake LLMs (replace with real ChatOpenAI/OpenRouter in production) ─────────
// PARITY (DIRECT-ish): the fakes mimic the LangChain client interface. LangChain.js
// has bindTools / withStructuredOutput / invoke, and AIMessage/ToolMessage with
// tool_calls, so this ports structurally. Two frictions:
//   (a) Python `.model_dump_json()` on a Pydantic instance -> in TS you JSON.stringify
//       a plain object; no schema-attached serializer. LOW.
//   (b) `with_structured_output(model)` returns a client bound to a Pydantic CLASS
//       usable as a constructor (`self._model(evidence=..., summary=...)`). In TS
//       the "model" is a Zod schema (a VALUE, not a constructor). The fake can't
//       `new schema(...)`; it returns a plain object literal that the runtime
//       validates via `schema.parse(...)`. The Python fake conflates class-as-
//       validator and class-as-constructor; TS must split them. LOW-MEDIUM.

const announced: { preamble: string | null } = { preamble: null };

class FakeExploreLLM {
  private structured = false;
  private model?: z.ZodType;

  bindTools(_tools: unknown[]): this {
    return this;
  }

  async invoke(messages: BaseMessage[]): Promise<AIMessage | ExplorationResult> {
    if (this.structured) {
      // PARITY: Python `self._model(evidence=..., summary=...)` constructs the
      // Pydantic instance. Here we return a validated plain object.
      return this.model!.parse({
        evidence: ["auth.py:42", "crypto.py:18"],
        summary: "found 2 references supporting the claim",
      }) as ExplorationResult;
    }
    if (announced.preamble === null) {
      const sys = messages.find((m) => m._getType() === "system");
      if (sys) announced.preamble = String(sys.content);
    }
    const searched = messages.some((m) => m instanceof ToolMessage);
    if (!searched) {
      // PARITY: AIMessage with tool_calls — DIRECT, LangChain.js supports this.
      return new AIMessage({
        content: "",
        tool_calls: [{ name: "search_evidence", args: { query: "verify claim" }, id: "call-1" }],
      });
    }
    return new AIMessage({
      content: JSON.stringify({
        evidence: ["auth.py:42", "crypto.py:18"],
        summary: "found 2 references supporting the claim",
      }),
    });
  }

  withStructuredOutput(model: z.ZodType): FakeExploreLLM {
    const clone = new FakeExploreLLM();
    clone.model = model;
    clone.structured = true;
    return clone;
  }
}

class FakeScoreLLM {
  private model?: z.ZodType;
  withStructuredOutput(model: z.ZodType): this {
    this.model = model;
    return this;
  }
  async invoke(_messages: BaseMessage[]): Promise<ClaimVerdict> {
    return this.model!.parse({
      claimId: "scored",
      disposition: "confirmed",
      reasoning: "evidence supports the claim based on 2 source references",
    }) as ClaimVerdict;
  }
}

// ── Fake tools (return TYPED objects, not strings) ────────────────────────────
// PARITY: Python's fake tool has `.name` + `.invoke(args, config) -> EvidenceHit`.
// LangChain.js StructuredTool has a Zod schema + a func returning any value; the
// "return a typed model not a string" behavior maps to returning a validated
// EvidenceHit object. The framework capturing it on ToolInteraction.typed_result
// is a runtime-dispatch behavior — see the typed_result GAP note at the demo.

class FakeEvidenceSearch {
  name = "search_evidence";
  async invoke(_args: unknown): Promise<EvidenceHit> {
    return EvidenceHit.parse({
      sourceFile: "auth.py",
      line: 42,
      snippet: "def authenticate(user, password): ...",
      relevance: 0.95,
    });
  }
}

class FakeLocalLookup {
  name = "local_lookup";
  async invoke(_args: unknown): Promise<EvidenceHit> {
    return EvidenceHit.parse({ sourceFile: "cache", line: 0, snippet: "", relevance: 0.0 });
  }
}

const TOOL_FACTORIES: Record<string, ToolFactory> = {
  search_evidence: (_config, _toolConfig) => new FakeEvidenceSearch() as unknown as never,
  local_lookup: (_config, _toolConfig) => new FakeLocalLookup() as unknown as never,
};

const llmFactory: LlmFactory = (tier) =>
  (tier === "research" ? new FakeExploreLLM() : new FakeScoreLLM()) as unknown as never;

// PARITY: prompt_compiler DIRECT — returns BaseMessage[]. Python returns a list of
// {role, content} dicts; LangChain.js prefers message objects but accepts the
// tuple/dict form too.
const promptCompiler: PromptCompiler = (_template, _data) => [
  { role: "user", content: "verify" } as unknown as BaseMessage,
];

// ── Pipeline nodes ────────────────────────────────────────────────────────────
// Phase 1: explore (agent mode, dict outputs: result + tool_log)
// Phase 2: score  (think mode, consumes explore_result + explore_tool_log)

// PARITY (dict-form outputs — the CENTRAL wrinkle of this example): Python
// `outputs={"result": ExplorationResult, "tool_log": list[ToolInteraction]}`.
// The proposed API can express the map as `outputs: { result: ..., tool_log: ... }`
// but note `list[ToolInteraction]` becomes `z.array(ToolInteraction)` where
// ToolInteraction is a FRAMEWORK type. Is there a Zod schema for the framework's
// own ToolInteraction? The doc never says — @neograph/core would have to ship
// `ToolInteractionSchema`. GAP #1.
//
// PARITY (announce_tool_budget / llm_config): DIRECT — an opaque config object.
// PARITY (Tool budget): DIRECT — `Tool({ name, budget })`. The budget=0 =
// "unlimited, omit from announced preamble" semantics is a runtime behavior the
// matrix rates Direct (ToolBudgetTracker).
//
// PARITY (dead body): Python `def explore(claim) -> ExplorationResult: ...` — the
// body is DEAD (LLM drives). In TS the wrapper form REQUIRES a callback value.
// You must write a throwing/`never`-returning stub. The dead-body UserWarning is
// "Skip" per matrix, but the inverse problem appears: TS FORCES a body you then
// have to neutralize. MEDIUM friction — the wrapper form has no `...`-equivalent.
const explore = node(
  {
    mode: "agent",
    outputs: { result: ExplorationResult, tool_log: z.array(ToolInteraction) }, // GAP #1: ToolInteraction schema
    model: "research",
    prompt: "verify/explore",
    tools: [Tool({ name: "search_evidence", budget: 3 }), Tool({ name: "local_lookup", budget: 0 })],
    llmConfig: { announceToolBudget: true },
  },
  // PARITY (port param): `claim: VerifyClaim` is the sub-construct's INPUT PORT
  // param, matched by TYPE to `input=VerifyClaim` below and wired to
  // neo_subgraph_input. The AD-0 transformer extracts `claim: VerifyClaim`, but
  // "does its type EQUAL the construct's input schema?" is a type→value identity
  // check the transformer must bridge (VerifyClaim-the-type vs VerifyClaim-the-Zod-
  // const). Python compares the annotation class to `input=`. See GAP #2.
  (_claim: VerifyClaim): ExplorationResult => {
    throw new Error("LLM-driven (agent mode); body never runs");
  },
);

// PARITY (dict-output REFERENCE params — the SHARPEST divergence): Python resolves
// `explore_result` and `explore_tool_log` by the `{upstream}_{key}` NAMING
// CONVENTION — `_resolve_dict_output_param` in construct_from_module splits
// "explore_result" into (node="explore", key="result"). The AD-0 transformer
// extracts the PARAM NAMES `explore_result` / `explore_tool_log` fine, but the
// convention-splitting logic that turns those names into (upstream, output-key)
// edges lives in construct_from_module — which the doc DROPS from v0.1.0-ts
// ("no module introspection"). construct_from_functions would have to absorb this
// resolution. The param-name→(node,key) string surgery is DIRECT logic, but the
// example's readability depends on it and the doc never mentions it. GAP #3.
const score = node(
  {
    mode: "think",
    outputs: ClaimVerdict,
    model: "judge",
    prompt: "verify/score",
  },
  (_exploreResult: ExplorationResult, _exploreToolLog: ToolInteraction[]): ClaimVerdict => {
    throw new Error("LLM-driven (think mode); body never runs");
  },
);
// PARITY (camelCase collision): Python param `explore_tool_log` maps 1:1 to key
// `explore` + `tool_log`. Idiomatic TS wants `exploreToolLog`, but the state field
// is `explore_tool_log` (snake, from the Python-compatible naming). The transformer
// must either forbid camelCase here or carry a snake↔camel map. The `{node}_{key}`
// convention is inherently snake-flavored; idiomatic TS camelCase fights it. MEDIUM.

// ── Sub-construct: explore -> score ───────────────────────────────────────────
// PARITY (construct_from_functions with input=/output=): DIRECT structurally.
// The port-param resolution (explore's `claim` reads neo_subgraph_input because its
// type == input=VerifyClaim) is GAP #2's type-identity bridge.
const verifyClaim = constructFromFunctions("verify-claim", [explore, score], {
  input: VerifyClaim,
  output: ClaimVerdict,
});

// ── Parent pipeline ────────────────────────────────────────────────────────────

// PARITY (source node, zero params): DIRECT. Return-type -> schema resolution.
const flattenClaims = node(
  { outputs: ClaimBatch },
  (): ClaimBatch => ({
    claims: [
      { claimId: "REQ-1", text: "system shall authenticate users via SSO" },
      { claimId: "REQ-2", text: "system shall encrypt data at rest" },
      { claimId: "REQ-3", text: "system shall log all access attempts" },
    ],
  }),
);

// PARITY (merge-after-fan-out, dict consumer): Python
// `deterministic_merge(verify_claim: dict[str, ClaimVerdict])`. The param name
// `verify_claim` == the sub-construct's name; its type `dict[str, ClaimVerdict]`
// tells the validator to consume the Each-merged dict. In TS
// `verifyClaim: Record<string, ClaimVerdict>` — the transformer extracts the
// Record type and the validator accepts Record<string,X> against an Each producer
// (matrix: list[X]/dict[str,X] compat Direct). DIRECT logic; the only wrinkle is
// the param-name↔node-name link crosses the camelCase boundary again
// (`verify_claim` node vs `verifyClaim` const/param).
const deterministicMerge = node(
  { outputs: FinalReport },
  (verifyClaim: Record<string, ClaimVerdict>): FinalReport => {
    const verdicts = Object.entries(verifyClaim)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${k}: ${v.disposition}`);
    return { summary: verdicts.join("\n"), claimCount: Object.keys(verifyClaim).length };
  },
);

// PARITY (.map on a SUB-CONSTRUCT — Each fan-out over a Construct): Python
// `verify_claim.map("flatten_claims.claims", key="claim_id")`. In TS `.map(...)` on
// a Construct returns an Each-modified construct. `mapOver` is the stringly-typed
// path "flatten_claims.claims" (unchecked at author time, same as Python — GAP,
// but no worse than Python). `key: "claimId"` — the Each key field. NOTE the
// snake/camel split: Python keys on `claim_id`; TS field is `claimId`, but the path
// segment "flatten_claims.claims" references the snake NODE/field names. The map
// key must match the ELEMENT schema field (`claimId`), while the over-path matches
// state field names. Two different naming worlds in one call. MEDIUM.
const pipeline = constructFromFunctions("verification-pipeline", [
  flattenClaims,
  verifyClaim.map("flattenClaims.claims", { key: "claimId" }),
  deterministicMerge,
]);

// ── Run ────────────────────────────────────────────────────────────────────────
// PARITY (compile with llm_factory/prompt_compiler/tool_factories): the doc's
// compile() row is `compile(construct, {checkpointer})` — it never lists
// llm_factory / prompt_compiler / tool_factories as options, yet EVERY LLM example
// needs them. The "configure_llm() global factory" row implies a global-config
// path (like Python's older configure_llm). This example passes them PER-COMPILE.
// The doc under-specifies which seam wins. GAP #4.
async function main() {
  const graph = compile(pipeline, {
    llmFactory,
    promptCompiler,
    toolFactories: TOOL_FACTORIES,
  });
  const result = await run(graph, { input: { nodeId: "VERIFY-001" } });

  // PARITY: `result["verify_claim"]` is dict[str, ClaimVerdict]. In TS
  // `result.verify_claim` is loosely typed (run() can't statically shape the
  // per-node output bag) -> cast. MEDIUM (same GAP as examples 04/05).
  const verdicts = result.verify_claim as Record<string, ClaimVerdict>;
  console.log(`Verified ${Object.keys(verdicts).length} claims:\n`);
  for (const [claimId, verdict] of Object.entries(verdicts).sort(([a], [b]) => a.localeCompare(b))) {
    console.log(`  ${claimId}: ${verdict.disposition}`);
    console.log(`    reasoning: ${verdict.reasoning}`);
  }

  const report = result.deterministic_merge as FinalReport;
  console.log(`\n${report.summary}`);
  console.log(`\nTotal claims: ${report.claimCount}`);
  console.log(`Result keys: ${Object.keys(result).sort()}`);

  console.log("\nBudget preamble the explore agent was told (announceToolBudget=true):");
  console.log(announced.preamble);
  // PARITY: search_evidence (budget 3) is announced; local_lookup (budget 0 =
  // unlimited) is omitted by design. Runtime behavior — Direct.
}

// ═══════════════════════════════════════════════════════════════════════════════
// Demo 1: typed tool results (standalone agent node)
// ═══════════════════════════════════════════════════════════════════════════════
// PARITY (construct_from_module — REDESIGN/BLOCKED as written): Python builds a
// throwaway `types.ModuleType("typed_tool_demo")`, attaches a @node to it, and
// runs `construct_from_module(mod)`. TS has NO dynamic-module + attribute
// introspection, and the doc explicitly drops construct_from_module from
// v0.1.0-ts. The clean rewrite is an explicit list via constructFromFunctions —
// so the FEATURE (build a one-node construct) ports, but the IDIOM (dynamic module
// as a namespace) is BLOCKED. This example uses the trick TWICE purely to scope
// demo nodes; in TS you'd just declare them and list them. MEDIUM.

const demoExplore = node(
  {
    mode: "agent",
    outputs: { result: ExplorationResult, tool_log: z.array(ToolInteraction) },
    model: "research",
    prompt: "verify/explore",
    tools: [Tool({ name: "search_evidence", budget: 1 })],
  },
  (): ExplorationResult => {
    throw new Error("LLM-driven; body never runs");
  },
);

async function demoTypedTools() {
  const demoGraph = compile(constructFromFunctions("typed-tool-demo", [demoExplore]), {
    llmFactory,
    promptCompiler,
    toolFactories: TOOL_FACTORIES,
  });
  const demoResult = await run(demoGraph, { input: { nodeId: "demo" } });

  // PARITY (ToolInteraction.typed_result — a REAL TS GAP): Python's
  // `typed_result` is `Any`, so `tool_log[0].typed_result.source_file` and
  // `.relevance` just resolve dynamically. In TS `typed_result` is `unknown`
  // (the framework can't know which model a given tool returned), so you MUST
  // narrow/cast to EvidenceHit before field access. The whole POINT of the demo
  // — "the typed model survives round-trip" — is exactly what TS's type system
  // can't preserve across the framework boundary without a cast. GAP #5, the
  // most example-specific friction here.
  const toolLog = demoResult.demo_explore_tool_log as ToolInteraction[];
  console.log("\n-- Typed tool results --");
  console.log(`  toolLog[0].toolName: ${toolLog[0].toolName}`);
  console.log(`  toolLog[0].result (rendered): ${String(toolLog[0].result).slice(0, 60)}...`);
  const typed = toolLog[0].typedResult as EvidenceHit; // PARITY: unavoidable cast
  console.log(`  typedResult.sourceFile: ${typed.sourceFile}`);
  console.log(`  typedResult.relevance: ${typed.relevance}`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Demo 2: context= verbatim state injection
// ═══════════════════════════════════════════════════════════════════════════════

const GraphCatalog = z.object({ content: z.string() }).readonly();
type GraphCatalog = z.infer<typeof GraphCatalog>;

const buildCatalog = node(
  { outputs: GraphCatalog },
  (): GraphCatalog => ({
    content:
      "=== Graph Catalog (BFS order) ===\n" +
      "UC-001: User Authentication [impl: auth.py]\n" +
      "UC-002: Data Encryption [impl: crypto.py]\n" +
      "BR-001: Password min 12 chars [traces: UC-001]\n",
  }),
);

// PARITY (context=): DIRECT per matrix (Context= validation = field-name check).
// `context: ["buildCatalog"]` injects the buildCatalog state field verbatim (NOT
// BAML-rendered) into the prompt compiler's `context` arg. The param
// `build_catalog: GraphCatalog` also reads that field. Note the same node NAME
// appears as (a) a context= string, (b) a param name, (c) an upstream producer —
// the snake/camel identity has to line up across all three. MEDIUM.
const ctxExplore = node(
  {
    mode: "agent",
    outputs: { result: ExplorationResult, tool_log: z.array(ToolInteraction) },
    model: "research",
    prompt: "verify/explore",
    tools: [Tool({ name: "search_evidence", budget: 1 })],
    context: ["buildCatalog"],
  },
  (_buildCatalog: GraphCatalog): ExplorationResult => {
    throw new Error("LLM-driven; body never runs");
  },
);

async function demoContext() {
  const ctxGraph = compile(constructFromFunctions("context-demo", [buildCatalog, ctxExplore]), {
    llmFactory,
    promptCompiler,
    toolFactories: TOOL_FACTORIES,
  });
  const ctxResult = await run(ctxGraph, { input: { nodeId: "ctx-demo" } });
  console.log("\n-- Context injection --");
  console.log(`  explore result: ${(ctxResult.ctx_explore_result as ExplorationResult).summary}`);
}

void main;
void demoTypedTools;
void demoContext;
