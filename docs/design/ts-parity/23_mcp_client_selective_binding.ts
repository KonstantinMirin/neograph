// TS PARITY SKETCH — HYPOTHETICAL, NOT COMPILABLE.
// Python source: examples/23_mcp_client_selective_binding.py
// Proposed API: docs/design/typescript-port.md (AD-0 transformer, .pipe(), node({...}, fn)).
//
// This file sketches what the TS DX WOULD look like against the proposed API and
// flags every place the DX degrades with `// PARITY:` notes. It is deliberately
// NOT type-checkable — several constructs below have no proposed-API home yet
// (they are the API GAPS reported alongside this sketch).
//
// The example is an MCP-client CRM deal-review pipeline: selective per-node tool
// binding, per-run identity via a token_provider, typed structured results, a raw
// BaseTool passthrough + async-driver lint, and a HITL gate before a mutation.

import {
  node,
  compile,
  arun,
  lint,
  constructFromFunctions,
  Tool,
  type ToolInteraction,
} from "@neograph/core";
// PARITY: The MCP battery is a separate optional package, mirroring Python's
// `neograph_mcp` (installed via the `mcp-examples` extra). LangGraph.js has an
// MCP adapter (`@langchain/mcp-adapters`) so the transport layer exists, but the
// neograph battery `mcpToolFactories` + `StdioServer` do NOT exist yet — this is
// the single largest port surface in this example (see API GAPS).
import { StdioServer, mcpToolFactories } from "@neograph/mcp";
import { MemorySaver } from "@langchain/langgraph";
import { z } from "zod";

// ── The real MCP server: spawned as a stdio subprocess ───────────────────────
// PARITY: Direct. `process.execPath` replaces `sys.executable`; the demo server
// would be a Node/py child either way (the wire protocol is language-neutral).
const DEMO_SERVER = new URL("./_mcp_demo_server.py", import.meta.url).pathname;
const CRM = new StdioServer({ command: process.execPath, args: [DEMO_SERVER] });

// ── Per-run identity: the token_provider (beat 3) ────────────────────────────
// PARITY: Direct. A plain function of `configurable` → string. No decorator, no
// reflection — this ports as-is.
function tokenProvider(configurable: Record<string, unknown>): string {
  return (configurable.mcp_auth as string) ?? "anon";
}

// ── Schemas ──────────────────────────────────────────────────────────────────
// PARITY: REDESIGN. Python's `class Deal(BaseModel, frozen=True)` is BOTH the
// static type AND the runtime schema (Pydantic). In TS the runtime schema (Zod)
// and the static type are two artifacts: you write the Zod schema and derive the
// type with `z.infer`. `frozen=True` (hashable/immutable value object) has no Zod
// equivalent — you get `readonly` at the type level only, no runtime freeze, and
// no structural hashing. Every model below pays this double-declaration tax.
const Deal = z.object({ id: z.string(), name: z.string(), stage: z.string() });
type Deal = z.infer<typeof Deal>;

// Client-side result models (beat 5): the typed channel. output_models= makes the
// server's structuredContent rehydrate into THESE, so ToolInteraction.typed_result
// is a model instance and `.acting_as` / `.hits[0]` are plain attributes.
const CrmSearchResult = z.object({
  hits: z.array(Deal),
  acting_as: z.string(), // the per-run identity the server echoed
});
type CrmSearchResult = z.infer<typeof CrmSearchResult>;

const KbResult = z.object({ acting_as: z.string() });
type KbResult = z.infer<typeof KbResult>;

const ResearchBrief = z.object({
  deal_id: z.string(),
  summary: z.string(),
  playbook: z.string(),
});
type ResearchBrief = z.infer<typeof ResearchBrief>;

const ActionOutcome = z.object({
  deal_id: z.string(),
  status: z.string(), // "applied" | "blocked"
  note: z.string(),
});
type ActionOutcome = z.infer<typeof ActionOutcome>;

// ── Fake LLMs (the ONLY fakes — the MCP layer is real) ───────────────────────
// PARITY: REDESIGN (mechanical, but pervasive). The Python fakes duck-type a
// LangChain model: bind_tools / abind_tools / with_structured_output / invoke /
// ainvoke, returning `AIMessage` with `.tool_calls`. In TS these become classes
// implementing the LangChain.js `BaseChatModel`-shaped surface (`bindTools`,
// `withStructuredOutput`, `invoke`, and the async twins collapse — JS is
// async-native so there is no sync/async duplication). The history-driven logic
// (count ToolMessages, branch) ports 1:1. `model_dump_json()` → `JSON.stringify`.
class ResearchFake {
  private structured = false;
  private static BRIEF: ResearchBrief = {
    deal_id: "D1",
    summary: "Acme renewal is in negotiation; anchor on realized value.",
    playbook: "renewal-playbook",
  };
  bindTools(_tools: unknown[]): ResearchFake {
    return this;
  }
  withStructuredOutput(_schema: z.ZodTypeAny): ResearchFake {
    const clone = new ResearchFake();
    clone.structured = true;
    return clone;
  }
  async invoke(messages: any[]): Promise<any> {
    if (this.structured) return ResearchFake.BRIEF;
    const n = messages.filter((m) => m._getType?.() === "tool").length;
    if (n === 0)
      return aiMessage("", [{ name: "crm_search", args: { query: "Acme" }, id: "s1" }]);
    if (n === 1)
      return aiMessage("", [
        { name: "kb_lookup", args: { topic: "renewal-playbook" }, id: "k1" },
      ]);
    return aiMessage(JSON.stringify(ResearchFake.BRIEF));
  }
}

class ActionFake {
  private structured = false;
  withStructuredOutput(_schema: z.ZodTypeAny): ActionFake {
    const clone = new ActionFake();
    clone.structured = true;
    return clone;
  }
  bindTools(_tools: unknown[]): ActionFake {
    return this;
  }
  private static denied(messages: any[]): boolean {
    return messages.some(
      (m) => m._getType?.() === "tool" && String(m.content).toLowerCase().includes("denied"),
    );
  }
  async invoke(messages: any[]): Promise<any> {
    if (this.structured) {
      return ActionFake.denied(messages)
        ? { deal_id: "D1", status: "blocked", note: "update_deal was denied by the reviewer; deal left unchanged." }
        : { deal_id: "D1", status: "applied", note: "Advanced D1 to closed-won." };
    }
    const n = messages.filter((m) => m._getType?.() === "tool").length;
    if (n === 0)
      return aiMessage("", [
        { name: "update_deal", args: { deal_id: "D1", stage: "closed-won" }, id: "u1" },
      ]);
    return aiMessage(
      ActionFake.denied(messages)
        ? '{"deal_id":"D1","status":"blocked","note":"denied by reviewer"}'
        : '{"deal_id":"D1","status":"applied","note":"advanced D1"}',
    );
  }
}

// helper: build an AIMessage-shaped object with tool_calls (LangChain.js AIMessage)
declare function aiMessage(content: string, toolCalls?: any[]): any;

function llmFactory(tier: string): any {
  return tier === "action" ? new ActionFake() : new ResearchFake();
}

// PARITY: Direct. The trivial prompt compiler ports 1:1 (returns a message array).
function promptCompiler(template: string, _data: unknown): { role: string; content: string }[] {
  return [{ role: "user", content: template }];
}

// ── Beat 1: the battery, sliced per node for least-privilege binding ─────────
// PARITY: The DICT-returning battery ports conceptually, but `output_models=`
// is where the typed-channel promise strains. In Python, `output_model=Deal`
// carries a class that (a) validates structuredContent AND (b) is BOTH the value
// and its type. In TS you must pass the Zod schema for runtime rehydration AND
// separately annotate the static type, and the framework must thread `z.infer`
// so `typed_result` is typed. The proposed API's `ToolInteraction.typed_result`
// is `unknown` unless a generic parameter is plumbed through — see API GAPS.
const FACTORIES = mcpToolFactories(
  { crm: CRM },
  {
    tokenProvider,
    namespace: false, // single server, bare tool names (no "crm::" prefix)
    outputModels: { crm_search: CrmSearchResult, kb_lookup: KbResult },
  },
);

const READER_TOOLS = ["crm_search", "kb_lookup"] as const; // read-only slice
const MUTATOR_TOOLS = ["update_deal"] as const; // mutating slice

// ── Pipelines ─────────────────────────────────────────────────────────────────
// PARITY: REDESIGN. This is the core DX divergence of this example.
//
// The Python `@node` decorates an EMPTY-bodied function whose body is DEAD CODE
// (the LLM drives via prompt= + tools=). The signature exists ONLY to declare the
// output type via the return annotation (`def research() -> ResearchBrief: ...`).
// The AD-0 transformer is built to extract exactly that return type — but this
// node has NO parameters, so there is nothing for the "signature IS the DAG" story
// to wire; the value is purely the return-type extraction.
//
// TWO problems specific to LLM-mode nodes here:
//   1. The `outputs` DICT form `{result: ResearchBrief, tool_log: list[ToolInteraction]}`
//      cannot be expressed as a return annotation (a fn returns ONE type). So even
//      WITH the transformer you must fall back to an EXPLICIT `outputs` config —
//      the transformer buys nothing for multi-output LLM nodes. (typescript-port.md
//      lists dict-form outputs nowhere; this is an API GAP.)
//   2. `list[ToolInteraction]` as a secondary output is framework-collected; the
//      Zod schema for `ToolInteraction` must be shipped by @neograph/core.
//
// The empty-body idiom itself is uglier in TS: you cannot write `def f(): ...`.
// You must pass a stub arrow `() => { throw new Error("LLM-driven") }` or allow
// `node()` to accept NO function at all. The proposed API only shows the
// `node({...}, fn)` two-arg form — a body-less LLM node has no documented shape.
const research = node(
  {
    mode: "agent",
    // PARITY: dict-form outputs — NOT expressible via transformer return-type
    // extraction; must be explicit. `ToolInteraction` schema comes from core.
    outputs: { result: ResearchBrief, tool_log: z.array(ToolInteractionSchema) },
    model: "research",
    prompt: "research/brief",
    // SELECTIVE BINDING: only the read-only readers. idempotent=true marks them
    // replay-safe. PARITY: Direct — `Tool` fields map 1:1.
    tools: [
      new Tool("crm_search", { budget: 2, idempotent: true }),
      new Tool("kb_lookup", { budget: 1, idempotent: true }),
    ],
    llmConfig: { announceToolBudget: true },
  },
  // body unused — the LLM drives via prompt + tools. PARITY: no `def f(): ...`
  // in TS; a throwing stub stands in for Python's `...`.
  (): ResearchBrief => {
    throw new Error("LLM-driven node body is never executed");
  },
);

// PARITY: `construct_from_functions` → `constructFromFunctions`. Direct: an
// explicit array of nodes (TS has no `construct_from_module` — no module
// introspection — but this example never used that path, so no loss here).
const researchPipeline = constructFromFunctions("crm-research", [research]);

const applyAction = node(
  {
    mode: "act", // act mode = mutating tools
    outputs: ActionOutcome, // single-type form; transformer COULD infer via return type
    model: "action",
    prompt: "action/apply",
    tools: [new Tool("update_deal", { budget: 1 })],
    // GATED MUTATION: pause BEFORE update_deal and ask a human.
    // PARITY: Direct — `gate_tools_when` is a plain state→payload callback, no
    // reflection. Camel-cased to `gateToolsWhen`.
    gateToolsWhen: (_state: unknown) => ({
      pending_tool: "update_deal",
      reason: "Approve CRM mutation?",
    }),
  },
  (): ActionOutcome => {
    throw new Error("LLM-driven node body is never executed");
  },
);

const actionPipeline = constructFromFunctions("crm-action", [applyAction]);

// PARITY: `list[ToolInteraction]` needs a Zod schema shipped by core. Declared
// here as a placeholder — in the real port this is `import { ToolInteractionSchema }`.
declare const ToolInteractionSchema: z.ZodType<ToolInteraction>;

// ── Demos ─────────────────────────────────────────────────────────────────────

async function demoResearchIdentityAndTypedResults(): Promise<void> {
  console.log("=".repeat(66));
  console.log("BEAT 1/3/5: read-only binding, per-run identity, typed MCP results");

  const graph = compile(researchPipeline, {
    llmFactory,
    promptCompiler,
    // Least-privilege: this graph only knows the reader factories.
    // PARITY: Direct — Object.fromEntries slicing replaces the dict comprehension.
    toolFactories: Object.fromEntries(READER_TOOLS.map((k) => [k, FACTORIES[k]])),
  });

  for (const operator of ["operator-A", "operator-B"]) {
    const result = await arun(graph, {
      input: { node_id: "REVIEW-1" },
      config: { configurable: { mcp_auth: operator } },
    });
    // PARITY: State access is UNTYPED. Python reads `result["research_result"]`
    // and annotates `brief: ResearchBrief`. In TS the compiled state is a keyed
    // record whose value types the framework cannot statically know (the
    // `{node}_{outputkey}` field-name synthesis happens at runtime). So
    // `result.research_result` is `unknown` and needs a cast / parse. This is a
    // real DX regression vs Python's already-untyped-but-conventional dict.
    const brief = result.research_result as ResearchBrief;
    const toolLog = result.research_tool_log as ToolInteraction[];

    // typed_result IS a client model, so `.acting_as` is a plain attribute.
    // PARITY: `typed_result` is `unknown` in the proposed API (no generic on
    // ToolInteraction), so this read is a cast in TS, not a typed attribute.
    const who = Object.fromEntries(
      toolLog.map((i) => [i.tool_name, (i.typed_result as CrmSearchResult | KbResult).acting_as]),
    );
    console.log(`Run as ${operator}: ${brief.summary}`, who);
    console.assert(Object.values(who).every((v) => v === operator), "identity mismatch");

    const search = toolLog.find((i) => i.tool_name === "crm_search")!;
    const deal = (search.typed_result as CrmSearchResult).hits[0];
    console.assert(deal.id === "D1");
  }
}

async function demoRawPassthroughAndLint(): Promise<void> {
  console.log("BEAT 2: raw BaseTool passthrough + lint tool_requires_async_driver");

  // A battery factory yields a real MCP tool; hand it straight to a node.
  // PARITY: The factory is `(config, toolConfig) => Promise<Tool>` — Direct.
  const rawSearch = await FACTORIES.crm_search(
    { configurable: { mcp_auth: "service-account" } },
    null,
  );

  // PARITY: REDESIGN — the empty-body LLM node again. A raw tool object is passed
  // straight into `tools:` (bypasses the Tool()-by-name registry lookup). The
  // transformer's return-type extraction covers `outputs: ResearchBrief`, but the
  // body is still a throwing stub.
  const rawResearch = node(
    { mode: "agent", model: "research", prompt: "research/brief", tools: [rawSearch] },
    (): ResearchBrief => {
      throw new Error("LLM-driven");
    },
  );
  const rawPipeline = constructFromFunctions("raw-passthrough", [rawResearch]);

  // PARITY: The lint check `tool_requires_async_driver` is INTERESTING in TS.
  // In Python it exists because an MCP tool is coroutine-only and a SYNC `run()`
  // cannot drive it. JS has no sync/async split for graph execution — LangGraph.js
  // `invoke` is always Promise-returning. So this ENTIRE lint category, and the
  // `run()` vs `arun()` distinction it guards, arguably does NOT EXIST in TS.
  // Reported as a "vanishes" divergence, not a port. (Kept here to mirror the
  // Python beat; a real TS port would drop the beat or repurpose it.)
  const issues = lint(rawPipeline).filter((i) => i.kind === "tool_requires_async_driver");
  console.assert(issues.length > 0 || true, "async-driver lint is Python-only");

  const graph = compile(rawPipeline, { llmFactory, promptCompiler });
  const result = await arun(graph, {
    input: { node_id: "REVIEW-1" },
    config: { configurable: { mcp_auth: "service-account" } },
  });
  console.log((result.raw_research as ResearchBrief).summary);
}

async function demoGatedMutation(): Promise<void> {
  console.log("BEAT 4: gated mutation — pause before update_deal, then approve / deny");

  const graph = compile(actionPipeline, {
    llmFactory,
    promptCompiler,
    toolFactories: Object.fromEntries(MUTATOR_TOOLS.map((k) => [k, FACTORIES[k]])),
    checkpointer: new MemorySaver(), // required for interrupt/resume
  });

  for (const [leg, decision] of [
    ["APPROVE", true],
    ["DENY", false],
  ] as const) {
    const config = { configurable: { thread_id: `deal-${leg}`, mcp_auth: "operator-A" } };

    // First pass: the gate pauses BEFORE update_deal fires.
    // PARITY: Direct. LangGraph.js has `interrupt`/`Command(resume=...)` and the
    // `__interrupt__` sentinel in the returned state — same shape as Python.
    const paused = await arun(graph, { input: { node_id: "REVIEW-1" }, config });
    console.assert("__interrupt__" in paused, "gate did not pause");
    const payload = (paused as any).__interrupt__[0].value;
    console.log(`${leg} leg — paused at gate:`, payload);

    // Resume with the human decision.
    // PARITY: Direct — `resume` maps to LangGraph.js `new Command({ resume: {...} })`.
    const result = await arun(graph, { resume: { approved: decision }, config });
    const outcome = result.apply_action as ActionOutcome;
    console.log(`resumed approved=${decision} -> status=${outcome.status}: ${outcome.note}`);
    console.assert(outcome.status === (decision ? "applied" : "blocked"));
  }
}

async function main(): Promise<void> {
  await demoResearchIdentityAndTypedResults();
  await demoRawPassthroughAndLint();
  await demoGatedMutation();
}

// PARITY: Direct — top-level `await main()` replaces `asyncio.run(main())`.
void main();

// ─────────────────────────────────────────────────────────────────────────────
// APPENDICES A/B (hand-rolled factory; streamable-http httpx.Auth) OMITTED.
// PARITY: The Python appendices lean on `langchain_mcp_adapters` (Python) and
// `httpx.Auth` per-request auth_flow. The TS analogues are `@langchain/mcp-adapters`
// and an interceptor on the transport's `fetch`/`undici` client. The
// "factory OWNS its client, neograph core never holds a session" invariant ports
// cleanly (it is an ownership discipline, not a language feature). The
// per-request bearer-mint via `httpx.Auth.async_auth_flow` becomes a `fetch`
// wrapper or an undici `Dispatcher` — DIFFERENT primitive, same seam.
