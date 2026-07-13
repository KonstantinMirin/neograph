// HYPOTHETICAL TypeScript port of examples/25_mcp_single_tool_gateway.py
// Grounds a feature-parity analysis against docs/design/typescript-port.md (AD-0 transformer form).
// This is NOT meant to compile or run — it is what the TS DX WOULD look like, with
// '// PARITY:' notes wherever the TS diverges or the proposed API has no answer.
//
// Python source teaches: the SINGULAR neograph_mcp.mcp_tool_factory — bind ONE
// gateway-federated MCP tool with an OFFLINE build (zero network at compile()),
// namespaced->bare rename, and per-run identity via token_provider.

import { z } from "zod";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import { node, Tool, ToolInteraction, compile, run, constructFromFunctions } from "@neograph/core";

// PARITY(API GAP — headline): the ENTIRE neograph_mcp package is absent from the
// typescript-port.md Feature Parity Matrix. StdioServer / HttpServer /
// mcp_tool_factory (singular) / mcp_tool_factories (plural) / token_provider /
// output_model rehydration / gateway rename / mcp_resource_fetcher /
// mcp_run_context have NO proposed TS design. The imports below are invented
// (`@neograph/mcp`). @langchain/mcp-adapters exists in JS, so a port is BUILDABLE,
// but the whole abstraction layer this example demonstrates is undesigned.
import { StdioServer, mcpToolFactory, type ToolFactory } from "@neograph/mcp";

const DEMO_SERVER = new URL("./_mcp_demo_server.py", import.meta.url).pathname;
const CRM_GATEWAY = new StdioServer({ command: process.execPath, args: [DEMO_SERVER] });

// ── Schemas ──────────────────────────────────────────────────────────────────
// PARITY(Direct-ish): Python's `class X(BaseModel, frozen=True)` -> Zod (AD-1).
// `frozen=True` immutability has no runtime Zod analog; use `.readonly()` or rely
// on `as const`. The nominal class identity Python uses for producer-type matching
// becomes structural Zod-schema comparison (AD-3) — a known Redesign for the validator.
const ResearchNote = z.object({
  query: z.string(),
  finding: z.string(),
});
type ResearchNote = z.infer<typeof ResearchNote>;

// Client-side model for the gateway tool's structuredContent. Declaring it as
// output_model= makes typed_result THIS model, not a raw content-block list.
const PerplexityResult = z.object({
  query: z.string(),
  acting_as: z.string(), // the per-run identity the server echoed
});
type PerplexityResult = z.infer<typeof PerplexityResult>;

// ── Fake LLM (the ONLY fake — the MCP layer is real) ─────────────────────────
// One agent turn: call perplexity_research, then finalize from its result.
//
// PARITY(Redesign): Python duck-types the LLM (any object with bind_tools/
// with_structured_output/invoke/ainvoke). LangChain.js has no structural-typing
// escape hatch — a fake must satisfy the Runnable/BaseChatModel interface (or be
// cast `as unknown as BaseChatModel`). `with_structured_output(Model)` -> Zod
// `withStructuredOutput(schema)`. Python returns a Pydantic instance straight from
// `invoke`; JS returns a plain parsed object.
class ResearchFake {
  private static NOTE: ResearchNote = { query: "acme", finding: "Acme renewal research complete." };
  private structured = false;

  bindTools(_tools: unknown[]): ResearchFake {
    return this;
  }

  withStructuredOutput(_schema: z.ZodTypeAny): ResearchFake {
    const clone = new ResearchFake();
    clone.structured = true;
    return clone;
  }

  // PARITY(Simplification): JS is async-native, so Python's invoke/ainvoke split
  // collapses to ONE `invoke`. See the arun/run note below — a genuine TS win.
  async invoke(messages: unknown[]): Promise<unknown> {
    if (this.structured) return ResearchFake.NOTE;
    if (!messages.some((m) => m instanceof ToolMessage)) {
      // The model calls the BARE name — it never sees the gateway namespace.
      return new AIMessage({
        content: "",
        tool_calls: [{ name: "perplexity_research", args: { query: "acme" }, id: "p1" }],
      });
    }
    return new AIMessage({ content: JSON.stringify(ResearchFake.NOTE) });
  }
}

function llmFactory(_tier: string): ResearchFake {
  return new ResearchFake();
}

function promptCompiler(template: string, _data: unknown): Array<{ role: string; content: string }> {
  return [{ role: "user", content: template }];
}

// ── The node: written against the BARE tool name ─────────────────────────────
//
// PARITY(Redesign, but AD-0 buys NOTHING here): the doc's centerpiece — the
// signature-extraction transformer that preserves "signature IS the DAG" — is
// IRRELEVANT to this example. `research()` takes ZERO parameters (an agent leaf)
// and its body is DEAD (`...`), so there is no signature to extract. Everything
// load-bearing (mode, dict-form outputs, tools) is config-object data that ports
// identically WITH OR WITHOUT the transformer. This example is orthogonal to AD-0.
//
// PARITY(API GAP): dict-form `outputs={"result": X, "tool_log": list[ToolInteraction]}`
// with a framework-COLLECTED secondary key (tool_log) is not in the matrix. The
// demand-driven tool_log collection ("no consumer -> no overhead") must be
// re-specified. `list[ToolInteraction]` -> `z.array(ToolInteraction.schema)`.
const research = node(
  {
    mode: "agent",
    outputs: { result: ResearchNote, tool_log: z.array(ToolInteraction.schema) },
    model: "research",
    prompt: "research/note",
    tools: [new Tool({ name: "perplexity_research", budget: 1, idempotent: true })],
  },
  // PARITY: dead body must still be written as a callback; there is no `...`
  // stub form and no AST dead-body warning (matrix marks that Skip / ESLint-later).
  (): ResearchNote => {
    throw new Error("unreachable — the LLM drives via prompt= + tools=");
  },
);

// PARITY(Direct, ~2 days per matrix): construct_from_functions -> constructFromFunctions.
const pipeline = constructFromFunctions("gateway-research", [research]);

// ── Demos ─────────────────────────────────────────────────────────────────────

// Beat 1: construction connects NOWHERE — even an unreachable spec builds.
function demoOfflineBuild(): void {
  const unreachable = new StdioServer({ command: "/nonexistent/not-a-real-gateway", args: ["--nope"] });
  const factory: ToolFactory = mcpToolFactory("crm", unreachable, {
    toolName: "crm-perplexity_research",
    renameTo: "perplexity_research",
  });

  // PARITY(Redesign / no clean idiom — load-bearing for THIS example): Python
  // PROVES the offline-build guarantee with `asyncio.iscoroutinefunction(factory)`
  // — a first-class runtime predicate that the connect is deferred inside the
  // returned coroutine fn. JS has no honest equivalent: any function may return a
  // Promise, and `factory.constructor.name === "AsyncFunction"` is fragile (breaks
  // the moment the factory is wrapped/bound/transpiled to a generator by a
  // downlevel target). The teaching assertion of Beat 1 has no direct TS port.
  console.assert(factory.constructor.name === "AsyncFunction"); // fragile; see note

  console.log("Built a factory against an UNREACHABLE spec — no error, no connect.");
}

// Beats 2 + 3: namespaced->bare rename and per-run identity, end to end.
async function demoRenameAndIdentity(): Promise<void> {
  const factory: ToolFactory = mcpToolFactory("crm", CRM_GATEWAY, {
    toolName: "crm-perplexity_research", // the name the GATEWAY exposes
    renameTo: "perplexity_research", // the name the NODE binds
    // PARITY(Direct): a closure reading per-run config. Python's
    // `lambda configurable: configurable.get("mcp_auth", "anon")` ->
    // `(cfg) => cfg.mcp_auth ?? "anon"`. Clean.
    tokenProvider: (cfg: Record<string, unknown>) => (cfg.mcp_auth as string) ?? "anon",
    // PARITY(API GAP): output_model rehydration of the server's structuredContent
    // into a typed model. No proposed design. With Zod it would be a `.parse()`
    // at the tool boundary, but the whole seam (rehydrate MCP content-block JSON ->
    // typed instance so typed_result IS PerplexityResult) is unspecified.
    outputModel: PerplexityResult,
  });

  // PARITY(API GAP): the `tool_factories=` compile channel takes an ASYNC factory
  // with a `(config, tool_config) => Tool` signature (per-call identity + per-call
  // idempotency channel). The matrix lists "Registries (scripted, condition, tool)"
  // as Direct but says nothing about this deferred-async-factory contract.
  const graph = compile(pipeline, {
    llmFactory,
    promptCompiler,
    toolFactories: { perplexity_research: factory },
  });

  // PARITY(Simplification — TS WIN): Python splits run/arun; JS is async-native so
  // there is ONE `run` (awaitable). The matrix's Runner row doesn't note this, but
  // the arun/run duplication simply disappears in TS.
  const result = await run(graph, {
    input: { node_id: "GW-1" },
    config: { configurable: { mcp_auth: "operator-A" } },
  });

  // PARITY(Redesign): dict result keys are stringly-typed. Python reads
  // result["research_result"]; TS `result["research_result"]` is untyped unless
  // compile() threads an output type param. `{node}_{key}` state-field naming
  // must be reproduced.
  const note = result["research_result"] as ResearchNote;
  const toolLog = result["research_tool_log"] as ToolInteraction[];

  const call = toolLog[0];
  // PARITY(Redesign): Python `ToolInteraction.typed_result: Any`. In TS that field
  // is `unknown` (the "no-Any" ban has no TS analog beyond `unknown`), so the
  // caller MUST narrow/cast — there is no typed retrieval path in the proposed API.
  const payload = call.typed_result as PerplexityResult;

  console.log(`LLM-facing tool name : ${call.tool_name}`);
  console.log(`server-side identity : acting_as=${payload.acting_as}`);
  console.log(`final result         : ${note.finding}`);
  console.assert(call.tool_name === "perplexity_research"); // bare binding, not <peer>-<tool>
  console.assert(payload.acting_as === "operator-A"); // same token path as the plural
  console.assert(payload.query === "acme");
}

async function main(): Promise<void> {
  demoOfflineBuild();
  await demoRenameAndIdentity();
}

void main();
