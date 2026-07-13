// TS parity sketch of examples/10_full_pipeline.py (neograph "Full Pipeline" example).
// HYPOTHETICAL — written against the PROPOSED API in docs/design/typescript-port.md.
// Not compilable/runnable; there is no TS implementation. Goal: ground a DX parity read.
//
// This example is dominated by the PROGRAMMATIC / DECLARATIVE surface:
//   Node.scripted(fn="string") + `| Modifier` pipe + Construct(nodes=[...]) + compile(scripted={...}).
// The AD-0 transformer ("signature IS the DAG") applies to exactly ONE node here
// (decompose), and even that has an EMPTY signature — so the transformer buys almost
// nothing for this example. See the PARITY notes and the friction report.

import { z } from "zod";
import {
  Node,
  Construct,
  Each,
  Operator,
  node,
  compile,
  run,
  type RunConfig,
} from "@neograph/core";
import { MemorySaver } from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";

// ════════════════════════════════════════════════════════════════════════════
// SCHEMAS
// PARITY: Pydantic `BaseModel, frozen=True` -> Zod object + `z.infer` type alias.
//   Zod has no `frozen=True` analog: instances are plain objects, not immutable.
//   The Python `frozen=True` (hashability, value-equality) is unused HERE except by
//   merge_claims' `set()` — which dedupes STRINGS, so the loss is inert in this example.
//   (If Oracle dedup were on model instances, JS `Set` uses reference equality -> FRICTION.)
// ════════════════════════════════════════════════════════════════════════════

const Claims = z.object({ items: z.array(z.string()) });
type Claims = z.infer<typeof Claims>;

const Context = z.object({ references: z.array(z.string()) });
type Context = z.infer<typeof Context>;

// PARITY: Python `list[dict[str, str]]` -> `z.array(z.record(z.string()))`. Direct.
const ScoredClaims = z.object({ scored: z.array(z.record(z.string())) });
type ScoredClaims = z.infer<typeof ScoredClaims>;

const ClusterGroup = z.object({ label: z.string(), claim_ids: z.array(z.string()) });
type ClusterGroup = z.infer<typeof ClusterGroup>;

const Clusters = z.object({ groups: z.array(ClusterGroup) });
type Clusters = z.infer<typeof Clusters>;

const VerifyResult = z.object({
  cluster_label: z.string(),
  passed: z.boolean(),
  gaps: z.array(z.string()),
});
type VerifyResult = z.infer<typeof VerifyResult>;

const ValidationResult = z.object({ passed: z.boolean(), issues: z.array(z.string()) });
type ValidationResult = z.infer<typeof ValidationResult>;

const Report = z.object({ text: z.string() });
type Report = z.infer<typeof Report>;

// ════════════════════════════════════════════════════════════════════════════
// FAKE LLM + TOOLS
// PARITY: the Python fakes duck-type LangChain's `with_structured_output` /
//   `bind_tools` / `invoke`. LangChain.js has withStructuredOutput/bindTools/invoke,
//   so the shape ports — BUT the Python fake mutates `msg.tool_calls = [...]` on an
//   already-constructed AIMessage. In LangChain.js AIMessage, tool_calls is a
//   constructor field, not a freely-reassignable attribute -> REDESIGN of the fake
//   (test infra only, not the neograph API).
// ════════════════════════════════════════════════════════════════════════════

class FakeProduceLLM {
  private model?: z.ZodTypeAny;
  withStructuredOutput(model: z.ZodTypeAny) {
    this.model = model;
    return this;
  }
  async invoke(_messages: unknown[]): Promise<Claims> {
    // PARITY: Python compared `self._model is Claims` (identity on the class).
    //   In TS the "model" is a Zod schema value; identity compare works the same way.
    return { items: ["shall authenticate", "shall log", "shall encrypt"] };
  }
}

const callCount = { search: 0 };

class FakeGatherLLM {
  private call = 0;
  private hasTools = true;
  bindTools(tools: unknown[]) {
    const clone = new FakeGatherLLM();
    clone.call = this.call;
    clone.hasTools = tools.length > 0;
    return clone;
  }
  async invoke(_messages: unknown[]): Promise<AIMessage> {
    this.call += 1;
    if (this.hasTools && this.call <= 2) {
      // REDESIGN (test infra): tool_calls passed via constructor, not post-assignment.
      return new AIMessage({
        content: "",
        tool_calls: [{ name: "search_code", args: { q: "test" }, id: `c${this.call}` }],
      });
    }
    return new AIMessage({ content: "done" });
  }
}

const fakeSearchTool = {
  name: "search_code",
  async invoke(_args: unknown): Promise<string> {
    callCount.search += 1;
    return `Found reference #${callCount.search}`;
  },
};

// PARITY: Python `llm_factory(tier, node_name=None, llm_config=None)` uses keyword args
//   with defaults. TS idiom is an options object OR positional-with-optional. Shown here
//   as an options bag to keep the extra params addressable. Minor syntactic FRICTION.
function llmFactory(tier: string, _opts?: { nodeName?: string; llmConfig?: unknown }) {
  return tier === "fast" ? new FakeProduceLLM() : new FakeGatherLLM();
}

// ════════════════════════════════════════════════════════════════════════════
// SCRIPTED FUNCTIONS
// PARITY: Python scripted fns take (input_data, config) positionally; input_data is the
//   upstream Pydantic instance. TS: same 2-arg callback, input_data is a plain object.
//   Attribute access (`v.items`, `input_data.label`) is identical. Direct.
// ════════════════════════════════════════════════════════════════════════════

// PARITY: Oracle merge_fn. Python signature `merge_claims(variants, config)` — variants is
//   list[Claims]. TS callback `(variants: Claims[], config) => Claims`. Direct.
//   NOTE: merge_claims is referenced BY STRING ("merge_claims") from the @node Oracle
//   kwargs and supplied via the compile `scripted` registry. See the GAP note at compile().
function mergeClaims(variants: Claims[], _config: RunConfig): Claims {
  const seen = new Set<string>();
  const merged: string[] = [];
  for (const v of variants) {
    for (const item of v.items) {
      if (!seen.has(item)) {
        seen.add(item);
        merged.push(item);
      }
    }
  }
  return { items: merged };
}

function lookupContext(_input: Claims, _config: RunConfig): Context {
  return { references: ["auth.py:42", "logger.py:18", "crypto.py:7"] };
}

function scoreClaims(input: Claims, _config: RunConfig): ScoredClaims {
  const scores: Record<string, string> = {
    authenticate: "high",
    log: "medium",
    encrypt: "high",
  };
  const scored = input.items.map((claim) => {
    const hit = Object.entries(scores).find(([k]) => claim.includes(k));
    return { claim, score: hit ? hit[1] : "low" };
  });
  return { scored };
}

function makeClusters(_input: unknown, _config: RunConfig): Clusters {
  return {
    groups: [
      { label: "security", claim_ids: ["authenticate", "encrypt"] },
      { label: "observability", claim_ids: ["log"] },
    ],
  };
}

function verifyCluster(input: ClusterGroup, _config: RunConfig): VerifyResult {
  const passing: Record<string, boolean> = { security: true, observability: false };
  const gaps: Record<string, string[]> = { observability: ["missing structured logging"] };
  return {
    cluster_label: input.label,
    passed: passing[input.label] ?? false,
    gaps: gaps[input.label] ?? [],
  };
}

function checkAllPassed(_input: unknown, _config: RunConfig): ValidationResult {
  return { passed: false, issues: ["observability cluster failed"] };
}

function buildReport(_input: unknown, _config: RunConfig): Report {
  return { text: "Verification complete. 1 cluster needs attention." };
}

// ════════════════════════════════════════════════════════════════════════════
// CONDITION (for Operator)
// PARITY: Python `needs_review(state)` reads `state.check_results` (attribute access on a
//   Pydantic-ish state) and returns a dict or None.
//   - TS: state is a plain object -> `state.check_results`. Direct.
//   - CAVEAT: the node is named "check-results" (hyphen) but its STATE FIELD is
//     `check_results` (underscore). neograph normalizes node name -> identifier for the
//     state key. TS must replicate that normalization, otherwise the natural access would
//     be `state["check-results"]`. Minor FRICTION (identifier normalization must port).
// ════════════════════════════════════════════════════════════════════════════

function needsReview(state: { check_results?: ValidationResult }): { issues: string[] } | null {
  const val = state.check_results;
  if (val && !val.passed) return { issues: val.issues };
  return null;
}

// ════════════════════════════════════════════════════════════════════════════
// PIPELINE ASSEMBLY
// ════════════════════════════════════════════════════════════════════════════

// Step 1: Decompose — @node LLM mode with Oracle-via-kwargs (ensemble_n + merge_fn).
// PARITY: Python `@node(outputs=Claims, prompt=, model=, llm_config=, ensemble_n=3,
//   merge_fn="merge_claims")` on `def decompose() -> Claims: ...`.
//   -> `node({...}, () => {...})`. The decorator becomes a wrapper (TS decorators don't
//   work on standalone fns — AD-0). ensemble_n -> ensembleN, merge_fn -> mergeFn.
//   THE TRANSFORMER BUYS NOTHING HERE: the signature is EMPTY (no params, so no edges to
//   extract) and the output type is given explicitly as `outputs`. This is a source node.
//   The body is `...` (dead in LLM mode) -> in TS an empty arrow `() => ({} as Claims)`;
//   the Python "dead-body AST warning" is dropped (doc: Skip / ESLint-plugin-later).
const decompose = node(
  {
    outputs: Claims,
    prompt: "decompose",
    model: "fast",
    llmConfig: { providerKwargs: { temperature: 0.8 } },
    ensembleN: 3,
    mergeFn: "merge_claims", // GAP: string ref resolved from a registry — see compile().
  },
  (): Claims => ({ items: [] }) // dead body in LLM mode
);

// Step 2: Enrich — sub-pipeline with an isolated state boundary (input=/output=).
// PARITY: `Construct(input=Claims, output=ScoredClaims, nodes=[...])`. The boundary
//   port (singular input/output) maps directly. Node.scripted with the SINGLE-TYPE
//   inputs shorthand (`inputs=Claims`, not a dict) ports 1:1.
const enrich = new Construct({
  name: "enrich",
  input: Claims,
  output: ScoredClaims,
  nodes: [
    Node.scripted({ name: "lookup", fn: "lookup_context", inputs: Claims, outputs: Context }),
    Node.scripted({ name: "score", fn: "score_claims", inputs: Claims, outputs: ScoredClaims }),
  ],
});

// Step 3: Cluster and verify — Each fan-out via the pipe.
// PARITY: Python `Node.scripted(...) | Each(over="cluster.groups", key="label")`.
//   `|` -> `.pipe()` (doc: Redesign at the operator level, mechanical at the call site).
//   `over` is a DOTTED STRING path into the cluster node's output; `key` names a field on
//   ClusterGroup. Both stay strings; Each.key existence is validated via Zod `.shape`.
const cluster = Node.scripted({ name: "cluster", fn: "make_clusters", outputs: Clusters });
const verify = Node.scripted({
  name: "verify",
  fn: "verify_cluster",
  inputs: ClusterGroup,
  outputs: VerifyResult,
}).pipe(Each({ over: "cluster.groups", key: "label" }));

// Step 4: Validate — Operator pauses (interrupt) if not all clusters pass.
// PARITY: `Node.scripted(...) | Operator(when="needs_review")`. Operator.when is a STRING
//   condition resolved from the conditions registry at compile time. Direct.
const checkResults = Node.scripted({
  name: "check-results",
  fn: "check_passed",
  outputs: ValidationResult,
}).pipe(Operator({ when: "needs_review" }));

// Step 5: Report
const report = Node.scripted({ name: "report", fn: "build_report", outputs: Report });

// Assemble top-level construct.
// PARITY: mixes a @node-built Node (decompose), a sub-Construct (enrich), and declarative
//   Node.scripted nodes in ONE `nodes: [...]` list. All produce the same IR — Direct.
const pipeline = new Construct({
  name: "full-verification",
  description: "End-to-end requirement verification with all NeoGraph features",
  nodes: [decompose, enrich, cluster, verify, checkResults, report],
});

// ════════════════════════════════════════════════════════════════════════════
// RUN
// ════════════════════════════════════════════════════════════════════════════

async function main() {
  // GAP (API surface): the proposed doc shows `compile(construct, {checkpointer})`.
  //   This example NEEDS a much richer options bag: llm_factory, prompt_compiler,
  //   scripted{}, conditions{}, tool_factories{}. The real Python compile() takes all of
  //   these as kwargs (compiler.py:63). The TS options type MUST include them, otherwise
  //   the config-driven surface (this whole example) cannot compile. Sketched below.
  // PARITY: string-name registries (scripted/conditions/tool_factories) are stringly-typed
  //   in BOTH languages — the transformer CANNOT link `fn="lookup_context"` to the function.
  //   So parity is Direct precisely because the weakness is symmetric. Maps -> plain objects.
  const graph = compile(pipeline, {
    checkpointer: new MemorySaver(), // required: Operator is present
    llmFactory,
    // PARITY: `lambda template, data: [...]` -> arrow fn returning a message list. Direct.
    promptCompiler: (_template: string, _data: unknown) => [{ role: "user", content: "analyze" }],
    scripted: {
      merge_claims: mergeClaims, // also serves the @node mergeFn string ref
      lookup_context: lookupContext,
      score_claims: scoreClaims,
      make_clusters: makeClusters,
      verify_cluster: verifyCluster,
      check_passed: checkAllPassed,
      build_report: buildReport,
    },
    conditions: { needs_review: needsReview },
    toolFactories: {
      // PARITY: `lambda config, tool_config: FakeSearchTool()` -> arrow. Direct.
      search_code: (_config: RunConfig, _toolConfig: unknown) => fakeSearchTool,
    },
  });

  const config: RunConfig = { configurable: { thread_id: "full-001" } };

  console.log("=== Running full pipeline ===\n");
  // PARITY: `run(graph, input={"node_id": ...})`. `node_id` is AMBIENT run input; no node
  //   consumes it via FromInput here (decompose takes no args), so no DI reflection is
  //   exercised. This example entirely SIDESTEPS the DI-classification redesign.
  let result = await run(graph, { input: { node_id: "REQ-FULL-001" }, config });

  console.log(`Decompose: ${result.decompose.items}`);
  console.log(`Enrich: ${result.enrich.scored.map((s: Record<string, string>) => s.score)}`);
  // PARITY: verify is Each-modified -> dict[str, VerifyResult]. `.keys()` -> Object.keys().
  console.log(`Verify clusters: ${Object.keys(result.verify)}`);
  console.log(`Validation: passed=${result.check_results.passed}`);
  console.log(`Search tool calls: ${callCount.search}`);

  // PARITY: interrupt / resume. `__interrupt__` payload + `run(graph, resume={...})`.
  //   LangGraph.js has interrupt()/Command; resume maps to `new Command({ resume })`.
  //   `result["__interrupt__"][0].value` -> `result.__interrupt__[0].value`. Direct.
  if ("__interrupt__" in result) {
    console.log(`\nPaused for review: ${result.__interrupt__[0].value}`);
    console.log("\n=== Resuming with approval ===\n");
    result = await run(graph, { resume: { approved: true }, config });
    console.log(`Report: ${result.report.text}`);
  }
}

void main();
