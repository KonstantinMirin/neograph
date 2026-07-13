// TS parity sketch of examples/14_context_injection.py
// HYPOTHETICAL — targets the proposed API in docs/design/typescript-port.md (AD-0
// transformer form). NOT compilable/runnable; illustrates DX + divergences only.
//
// Python source features exercised:
//   - scripted @node with NO params (build_catalog, make_claims)
//   - agent-mode @node with tools=[Tool(budget=)] + dict-form outputs
//     {"result": Verdict, "tool_log": list[ToolInteraction]}
//   - context=["build_catalog"]  <-- VERBATIM state injection into prompt compiler
//   - construct_from_functions(..., input=, output=) sub-construct port param
//   - .map("make_claims.claims", key="claim_id")  Each fan-out over a dotted path
//   - compile(llm_factory=, prompt_compiler=, tool_factories=) + run(input=)

import { z } from "zod";
import {
  node,
  Tool,
  ToolInteraction,          // framework-provided Zod schema
  constructFromFunctions,
  compile,
  run,
} from "@neograph/core";
import { AIMessage } from "@langchain/core/messages";

// -- Schemas ------------------------------------------------------------------
// PARITY: Pydantic `frozen=True` has no deep-immutable Zod equivalent. `.readonly()`
// only brands the TS type; runtime freeze is not enforced. Minor DX loss.

const GraphCatalog = z.object({ content: z.string() });
type GraphCatalog = z.infer<typeof GraphCatalog>;

const VerifyClaim = z.object({ claim_id: z.string(), text: z.string() });
type VerifyClaim = z.infer<typeof VerifyClaim>;

const ClaimBatch = z.object({ claims: z.array(VerifyClaim) });
type ClaimBatch = z.infer<typeof ClaimBatch>;

const EvidenceHit = z.object({
  source: z.string().describe("Source artifact"),
  line: z.number().int().describe("Line number"),
  relevance: z.number().describe("Relevance score 0-1"),
});
type EvidenceHit = z.infer<typeof EvidenceHit>;

const Verdict = z.object({
  claim_id: z.string(),
  disposition: z.string(),
  evidence_count: z.number().int(),
});
type Verdict = z.infer<typeof Verdict>;

// PARITY: In Python the transformer would map `class Foo(BaseModel)` -> schema by
// following the type. Here the developer maintains the Zod value AND the inferred
// type as two names. The AD-0 transformer resolves `VerifyClaim` (the *type*) back
// to `VerifyClaim` (the *schema value*) by convention; if they diverge, wiring breaks
// silently. Python has one source of truth (the class); TS has two.

// -- Fake LLM ----------------------------------------------------------------
// PARITY: LangChain.js fake. `msg.tool_calls = [...]` mutation and
// `with_structured_output` port to `bindTools`/`withStructuredOutput`. Direct but
// verbose; the Python `._model(...)` reflective structured-call becomes an explicit
// parse against the passed Zod schema.
class FakeAgentLLM {
  private callCount = 0;
  private structured = false;
  private model?: z.ZodTypeAny;

  bindTools(_tools: unknown[]): FakeAgentLLM {
    const clone = new FakeAgentLLM();
    clone.callCount = this.callCount;
    return clone;
  }

  invoke(_messages: unknown, _kwargs?: unknown): AIMessage {
    if (this.structured) {
      // Python returns a Verdict *instance* via self._model(...). Here we hand back
      // a JSON payload the structured path will parse against `this.model`.
      return new AIMessage({
        content: JSON.stringify({
          claim_id: "scored", disposition: "confirmed", evidence_count: 1,
        }),
      });
    }
    this.callCount += 1;
    if (this.callCount === 1) {
      return new AIMessage({
        content: "",
        tool_calls: [{ name: "search_evidence", args: { query: "verify" }, id: "call-1" }],
      });
    }
    return new AIMessage({
      content: JSON.stringify({
        claim_id: "scored", disposition: "confirmed", evidence_count: 1,
      }),
    });
  }

  withStructuredOutput(model: z.ZodTypeAny, _kwargs?: unknown): FakeAgentLLM {
    const clone = new FakeAgentLLM();
    clone.callCount = this.callCount;
    clone.model = model;
    clone.structured = true;
    return clone;
  }
}

// -- Fake tool ----------------------------------------------------------------
class FakeSearch {
  name = "search_evidence";
  invoke(_args: unknown, _config?: unknown): EvidenceHit {
    return { source: "auth.py", line: 42, relevance: 0.95 };
  }
}

// -- Tool factories + LLM layer ----------------------------------------------
const TOOL_FACTORIES = {
  search_evidence: (_cfg: unknown, _tc: unknown) => new FakeSearch(),
};

function llmFactory(_tier: string): FakeAgentLLM {
  return new FakeAgentLLM();
}

// PARITY (HIGH): Python `def prompt_compiler(template, data, **kw)` receives the
// verbatim catalog as `kw['context']`. The runtime decides whether to pass it by
// INTROSPECTING the compiler's parameter names (_llm_render.py:211-214, the same
// _accepted_params gate as di_inputs). TS functions expose no reliable runtime
// parameter-name list (types erased; `fn.toString()` parsing is fragile and breaks
// under minification). So there is NO clean equivalent of the "does this compiler
// accept `context`?" gate. Redesign below: always hand the compiler a single `opts`
// bag and let it read what it wants. This shifts a framework guarantee (only pass
// what's accepted, so an unaware compiler never sees stray kwargs) onto the author.
function promptCompiler(
  template: string,
  _data: unknown,
  opts: { context?: Record<string, string> | null; diInputs?: Record<string, unknown> },
): Array<{ role: string; content: string }> {
  const ctx = opts.context ?? "none";
  return [{ role: "user", content: `template=${template} context=${JSON.stringify(ctx)}` }];
}

// -- Pipeline -----------------------------------------------------------------

// Step 1: scripted node, NO params.
// PARITY: zero-parameter scripted node. The transformer extracts inputs={} and
// output=GraphCatalog from `(): GraphCatalog => ...`. Direct.
const buildCatalog = node({ outputs: GraphCatalog },
  (): GraphCatalog => ({
    content:
      "=== Graph Catalog (BFS from UC-001) ===\n" +
      "UC-001: User Authentication [auth.py:10-85]\n" +
      "  BR-001: Password min 12 chars [traces: UC-001]\n" +
      "  BR-002: Session timeout 30min [traces: UC-001]\n" +
      "UC-002: Data Encryption [crypto.py:1-120]\n" +
      "  BR-003: AES-256 at rest [traces: UC-002]\n",
  }),
);

// Step 2: scripted node producing claims.
const makeClaims = node({ outputs: ClaimBatch },
  (): ClaimBatch => ({
    claims: [
      { claim_id: "C1", text: "system authenticates via SSO" },
      { claim_id: "C2", text: "data encrypted at rest with AES-256" },
    ],
  }),
);

// Step 3: agent-mode node with tools, dict-form outputs, and verbatim context.
// PARITY (MEDIUM): dict-form outputs. The AD-0 transformer extracts `output=Verdict`
// from the arrow's `: Verdict` return, but the config also declares a 2-key
// `outputs` dict {result, tool_log}. The explicit `outputs` MUST win over the
// signature-extracted return, and the return annotation becomes redundant/ambiguous
// (it names only the primary key's type). Python tolerates this because `outputs=`
// is an explicit kwarg and the `-> Verdict` is just decoration; the wrapper needs a
// documented precedence rule. GAP: the matrix's "output inference from return
// annotation" (Direct) silently conflicts with explicit dict-form outputs here.
//
// PARITY (HIGH): `context: ["build_catalog"]` — verbatim state injection — is
// ABSENT from the Feature Parity Matrix entirely. It is a real Node field
// (node.py:214, `context: list[str] | None`) plumbed as a config side-channel to
// the prompt compiler. Structurally the field ports Direct (a string[]); the
// friction is the compiler-gating above, not the field.
//
// PARITY: LLM-mode dead body. Python writes `...`; TS has no ellipsis-body, so the
// arrow must still name param `claim: VerifyClaim` (for input wiring) and return
// `Verdict` (for the transformer) yet never actually run. A throwing stub reads
// oddly for a node whose body is intentionally dead.
const verify = node({
    mode: "agent",
    outputs: { result: Verdict, tool_log: z.array(ToolInteraction) },
    model: "research",
    prompt: "verify/explore",
    tools: [Tool("search_evidence", { budget: 2 })],
    context: ["build_catalog"],   // <-- catalog forwarded from parent state, verbatim
  },
  (_claim: VerifyClaim): Verdict => {
    throw new Error("llm-mode: body never runs");
  },
);

// PARITY: construct_from_functions with input=/output= port param. The transformer
// sees `claim: VerifyClaim`; the framework matches its type against `input:
// VerifyClaim` (schema-identity compare) to mark it a port param reading from the
// subgraph input instead of a peer node. Direct — but relies on the schema *value*
// identity noted above, not structural equality.
//
// PARITY: `.map("make_claims.claims", key="claim_id")` — string dotted-path form of
// Each fan-out. Direct. (Python also offers a refactor-safe lambda-proxy form,
// `.map(s => s.make_claims.claims, key=...)`, which ports cleanly to a JS Proxy
// get-trap — the ForwardConstruct pattern — but this example uses the string form.)
// `key: "claim_id"` requires Zod `.shape` introspection to confirm VerifyClaim has
// that field; matrix lists this Direct.
const verifyClaim = constructFromFunctions(
  "verify-claim", [verify],
  { input: VerifyClaim, output: Verdict },
).map("make_claims.claims", { key: "claim_id" });

// Step 4: assemble @node functions + sub-construct in one call.
const pipeline = constructFromFunctions(
  "verification",
  [buildCatalog, makeClaims, verifyClaim],
);

// -- Run ----------------------------------------------------------------------
async function main() {
  const graph = compile(pipeline, {
    llmFactory,
    promptCompiler,
    toolFactories: TOOL_FACTORIES,
  });
  // PARITY: `input={"node_id": ...}` is ambient run input; no node consumes it here
  // (no FromInput). Direct.
  const result = await run(graph, { input: { node_id: "VERIFY-001" } });

  console.log("=== Verification Results ===\n");

  // PARITY: Each fan-out yields a Record<string, Verdict> keyed by claim_id.
  // `sorted(verdicts.items())` -> sort the entries explicitly.
  const verdicts = result["verify_claim"] as Record<string, Verdict>;
  for (const [cid, v] of Object.entries(verdicts).sort(([a], [b]) => a.localeCompare(b))) {
    console.log(`  ${cid}: ${v.disposition} (evidence: ${v.evidence_count})`);
  }

  console.log(`\nResult keys: ${Object.keys(result).sort()}`);
  console.log("\nThe catalog was passed VERBATIM to the agent's prompt compiler");
  console.log("via context=['build_catalog'] -- not rendered, not modified.");
}

void main();
