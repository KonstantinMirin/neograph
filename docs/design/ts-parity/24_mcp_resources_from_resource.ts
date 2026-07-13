// PORT OF: examples/24_mcp_resources_from_resource.py
// Example 24 — "FromResource mechanics: the Typed Resource Manifest with Ephemeral Hydration."
//
// This is a HYPOTHETICAL sketch against the PROPOSED TS API in
// docs/design/typescript-port.md (AD-0 transformer form + .pipe() + branded DI).
// It is NOT meant to compile or run. Inline `// PARITY:` notes flag every place
// the TS DX diverges from, degrades against, or has NO path for, the Python.
//
// TOP-LINE FINDING: the parity matrix in typescript-port.md does not model the
// MCP feature family AT ALL. This example is ~80% MCP/FromResource surface —
// `resource_reader`, `FromResource(uri=/ref=)`, the manifest lift onto the
// `neo_resource_manifest_*` channel, the `(fetcher, replayer)` battery, self-heal
// replay, and the per-run fetch cache. None of these appear in the matrix's DI,
// Factory, or LLM rows. Most of the friction below is "unmodeled", not "hard".

import { z } from "zod";
import {
  node,
  compile,
  arun,
  Command,               // PARITY: Python passes `resume={...}` as an arun kwarg;
                          // LangGraph.js resumes via `new Command({ resume })`.
  Construct,             // PARITY: no construct_from_module in TS (doc §"Not in
                          // v0.1.0-ts"). Assemble nodes into an explicit list.
  fieldNameFor,          // port of neograph.naming.field_name_for
  FromInput,             // branded DI type (doc DI section)
  resourceReader,        // PARITY: UNMODELED in matrix — invented here as the
                          // port of neograph.tool.resource_reader.
  fromResource,          // PARITY: UNMODELED — see the big note at `research`.
} from "@neograph/core";

// PARITY: entire MCP battery (`neograph_mcp`) is unmodeled in the doc. These are
// hypothetical ports of StdioServer + mcp_resource_fetcher and would ride on
// @modelcontextprotocol/sdk + @langchain/mcp-adapters (both exist in JS, but the
// resource_link-preservation + replayer work is a whole unscoped TS package).
import { StdioServer, mcpResourceFetcher } from "@neograph/mcp";
import { AsyncSqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { StructuredTool } from "@langchain/core/tools";
import { AIMessage } from "@langchain/core/messages";
import { MultiServerMCPClient } from "@langchain/mcp-adapters";

import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const DEMO_SERVER = path.join(path.dirname(fileURLToPath(import.meta.url)), "_mcp_demo_server.py");
const DEAL_ID = "D1";
const STATE_MARKER = path.join(os.tmpdir(), "neograph_ex24_expiry.marker");

// ── Schemas ──────────────────────────────────────────────────────────────────
// PARITY: Python's `BaseModel, frozen=True` → Zod object. Zod has no `frozen`
// concept; immutability is by convention (readonly on the inferred type). The
// AD-0 transformer would read the *TS interface* — here we hand-write Zod because
// these types cross the LLM/state boundary and need a runtime schema regardless.

const DealSummary = z.object({
  deal_id: z.string(),
  headline: z.string(),
  recent_email_subjects: z.array(z.string()),
});
type DealSummary = z.infer<typeof DealSummary>;

const EmailPage = z.object({
  deal_id: z.string(),
  start: z.string(),
  end: z.string(),
  emails: z.array(z.record(z.unknown())),
});
type EmailPage = z.infer<typeof EmailPage>;

const ActivityHistory = z.object({
  deal_id: z.string(),
  events: z.array(z.record(z.unknown())),
});
type ActivityHistory = z.infer<typeof ActivityHistory>;

const EmailHistory = z.object({
  deal_id: z.string(),
  start: z.string().default(""),
  end: z.string().default(""),
  emails: z.array(z.record(z.unknown())).default([]),
});
type EmailHistory = z.infer<typeof EmailHistory>;

const GateAck = z.object({ reviewed_deal: z.string() });
type GateAck = z.infer<typeof GateAck>;

const Brief = z.object({ text: z.string() });
type Brief = z.infer<typeof Brief>;

// ── Beat-5 plumbing: counting fetcher ─────────────────────────────────────────
// PARITY: DIRECT. Closure-over-inner is idiomatic in both languages. The
// module-box `_CURRENT_RUN_ID` mirror is equally a hack in both.
const FETCH_LOG: Array<[string, string]> = [];
const CURRENT_RUN_ID = { value: "run-1" };

function countingFetcher(inner: (uri: string) => Promise<[unknown, string | null]>) {
  return async (uri: string): Promise<[unknown, string | null]> => {
    FETCH_LOG.push([CURRENT_RUN_ID.value, String(uri)]);
    return inner(uri);
  };
}

// ── The manifest-emitting tool ────────────────────────────────────────────────
// PARITY: MOSTLY-DIRECT *if* @langchain/mcp-adapters JS preserves resource_link
// blocks the way the Python raw session does. The Python comment explains the
// adapter rewrites resource_link → type:'file'; the JS adapter has the SAME
// limitation, so a TS port needs the identical raw-session workaround. The
// `metadata: { ng_idempotent: true }` idempotency marker on StructuredTool is
// carried by @langchain/core StructuredTool.metadata — DIRECT.
function stdioConnection(stateMarker: string) {
  return {
    command: process.execPath, // PARITY: demo server is a *Python* subprocess; a
                               // real TS port would ship a TS demo server. Cross-
                               // language stdio MCP is fine — protocol is neutral.
    args: [DEMO_SERVER],
    transport: "stdio" as const,
    env: { NEOGRAPH_MCP_DEMO_STATE: stateMarker },
  };
}

function makeGetDealTool(): StructuredTool {
  const getDeal = async ({ deal_id, token = "anon" }: { deal_id: string; token?: string }) => {
    const client = new MultiServerMCPClient({ crm: stdioConnection(STATE_MARKER) });
    const session = await client.session("crm");
    try {
      const result = await session.callTool("get_deal", { deal_id, token });
      const blocks: Array<Record<string, unknown>> = [];
      for (const block of result.content) {
        if (block.type === "resource_link") {
          blocks.push({ type: "resource_link", uri: String(block.uri), name: block.name, mimeType: block.mimeType ?? null });
        } else if (block.type === "text") {
          blocks.push({ type: "text", text: block.text });
        }
      }
      return blocks;
    } finally {
      await session.close();
    }
  };

  // PARITY: StructuredTool.from_function(coroutine=...) → new StructuredTool with a
  // Zod args schema. DIRECT, but note the JS tool needs an explicit Zod schema for
  // args where Python derived it from the fn signature — a small DX tax that
  // recurs everywhere a bare fn crossed a boundary in Python.
  return new StructuredTool({
    name: "get_deal",
    description: "Fetch a CRM deal; returns a summary plus resource_link refs to its corpora.",
    schema: z.object({ deal_id: z.string(), token: z.string().default("anon") }),
    func: getDeal,
    metadata: { ng_idempotent: true },
  });
}

// ── Beat 1: a typed domain reader ─────────────────────────────────────────────
// PARITY: UNMODELED-BUT-PORTABLE. resource_reader derives its args schema from the
// RFC-6570 vars in uri_template (`{deal_id}/{start}/{end}`) via pydantic
// create_model. In TS: parse the template vars and build a Zod object dynamically
// (matrix "Dynamic model generation" = Medium). The typed output_model → Zod.
// Idempotent flag → tool metadata. All doable; just entirely absent from the doc.
const readEmails = resourceReader({
  name: "read_emails",
  uriTemplate: "mcp://crm/deals/{deal_id}/emails/{start}/{end}",
  outputModel: EmailPage,
  description: "Read a date-range fraction of a deal's email history.",
  idempotent: true,
});

const getDealTool = makeGetDealTool();

// ── Fake LLMs ─────────────────────────────────────────────────────────────────
// PARITY: DIRECT (structural). LangChain.js chat models expose bindTools /
// withStructuredOutput / invoke, mirroring the Python duck-typed fake. The async
// ReAct driver is native in JS. One nit: `msg.tool_calls = [...]` mutation → JS
// AIMessage takes tool_calls in the constructor.
class ResearchAgentLLM {
  private turn = 0;
  private structured = false;
  private static ANSWER: DealSummary = {
    deal_id: DEAL_ID,
    headline: "Acme renewal in negotiation; recent pricing + renewal threads.",
    recent_email_subjects: ["Pricing questions", "Renewal terms"],
  };
  bindTools(_tools: unknown[]) { return this; }
  withStructuredOutput(_model: unknown) { const c = new ResearchAgentLLM(); (c as any).turn = this.turn; (c as any).structured = true; return c; }
  async invoke(_messages: unknown[]) {
    if (this.structured) return ResearchAgentLLM.ANSWER;
    this.turn += 1;
    if (this.turn === 1)
      return new AIMessage({ content: "", tool_calls: [{ name: "get_deal", args: { deal_id: DEAL_ID, token: "op-A" }, id: "t1" }] });
    if (this.turn === 2)
      return new AIMessage({ content: "", tool_calls: [{ name: "read_emails", args: { deal_id: DEAL_ID, start: "2024-04-01", end: "2024-12-31" }, id: "t2" }] });
    return new AIMessage({ content: JSON.stringify(ResearchAgentLLM.ANSWER) });
  }
}

class ConsumeThinkLLM {
  private structured = false;
  withStructuredOutput(_model: unknown) { const c = new ConsumeThinkLLM(); (c as any).structured = true; return c; }
  async invoke(messages: any[]) {
    const last = messages.at(-1) ?? {};
    const prompt = last?.content ?? "";
    const brief: Brief = { text: `Deal brief drafted from hydrated corpus. ${prompt}` };
    return this.structured ? brief : new AIMessage({ content: JSON.stringify(brief) });
  }
}

const llmFactory = (tier: string) => (tier === "consume" ? new ConsumeThinkLLM() : new ResearchAgentLLM());

// PARITY: DIRECT but note the `di_inputs` kwarg. The FromResource-hydrated value
// reaching the prompt compiler via `di_inputs` is the neograph-euyh feature — the
// matrix never mentions di_inputs. In Python the compiler opts in by *declaring*
// the `di_inputs` param (signature introspection gates it). TS has no runtime
// param-name introspection on this callback, so the opt-in must become an
// explicit flag or a named property on the compiler object. REDESIGN of the
// introspection-gated seam.
function promptCompiler(_template: unknown, _data: unknown, opts: { diInputs?: Record<string, any> } = {}) {
  const di = opts.diInputs ?? {};
  const history = di.history as EmailHistory | undefined;
  const dossier = di.dossier as ActivityHistory | undefined;
  let content: string;
  if (history) content = `Write a one-line brief. Recent email subjects: ${history.emails.map((e) => e.subject)}`;
  else if (dossier) content = `Research the deal. Activity so far: ${dossier.events.length} events.`;
  else content = "Write a one-line brief.";
  return [{ role: "user", content }];
}

// ── The pipeline nodes ────────────────────────────────────────────────────────
//
// PARITY — THE CENTRAL FRICTION OF THIS EXAMPLE (HIGH severity):
// The proposed DI form is a *branded type* on a parameter: `topic: FromInput<string>`.
// A branded type is purely type-level and carries NO runtime value. But
// `FromResource(uri="mcp://crm/deals/{deal_id}/activity")` and
// `FromResource(ref="email-history", max_bytes=200_000)` carry RUNTIME arguments —
// a URI template, a manifest kind, a byte cap. You cannot smuggle a runtime string
// into `dossier: FromResource<ActivityHistory>`; the transformer that reads the
// signature only sees the *type*, not the `uri=`/`ref=`/`max_bytes=` values.
//
// Consequence: FromResource CANNOT live in the signature the way Python's
// `Annotated[T, FromResource(uri=...)]` does. The binding must move OUT of the
// signature and INTO the node() config as a `resources` map keyed by param name.
// That splits one Python declaration into two TS sites (config key + fn param) and
// breaks the doc's headline promise that "signature IS the DAG" — for FromResource
// params the DAG edge is declared in config, not the signature. The transformer
// still supplies the *type* (ActivityHistory) but the *binding* is hand-written.

const research = node(
  {
    mode: "agent",
    outputs: DealSummary,
    model: "research",
    prompt: "research/scan",
    tools: [getDealTool, readEmails],
    // PARITY: FromResource relocated from the signature into config (see note above).
    // In Python this was `dossier: Annotated[ActivityHistory, FromResource(uri="...")]`.
    resources: {
      dossier: fromResource({ uri: "mcp://crm/deals/{deal_id}/activity" }),
    },
    // PARITY: `deal_id: Annotated[str, FromInput]` — FromInput is a branded type in
    // the doc and CAN stay in the signature. But mixing one DI kind in the signature
    // (deal_id) and another in config (dossier) is exactly the inconsistency the
    // relocation forces. Arguably ALL DI should move to config for uniformity —
    // which then abandons the branded-type DI story entirely for this example.
  },
  // Body is dead code in agent mode (LLM drives ReAct); params exist only so the
  // transformer records edges. deal_id via FromInput<string>; dossier is injected
  // by the runtime from the `resources` config, matched by param name.
  (deal_id: FromInput<string>, dossier: ActivityHistory): DealSummary => {
    return undefined as any; // `...` in Python
  },
);

// PARITY: `interrupt_when=lambda state: {...} if state.research else None`.
// DIRECT in shape, but `state.research.deal_id` is TYPED attribute access on a
// namespace object in Python. In LangGraph.js state is a plain object typed by the
// generated Annotation.Root — `state.research` is `DealSummary | null`, so the
// null-guard + `.deal_id` port cleanly ONLY if the generated state type is
// threaded into this callback. Untyped, it's `any` and you lose the guard's safety.
const gate = node(
  {
    outputs: GateAck,
    interruptWhen: (state: { research: DealSummary | null }) =>
      state.research
        ? { deal: state.research.deal_id, message: "Approve deal research before consuming corpora?" }
        : null,
  },
  (research: DealSummary): GateAck => ({ reviewed_deal: research.deal_id }),
);

// PARITY: `async def parse_activity` — DIRECT and actually CLEANER in TS.
// FromResource forces Python onto the async driver (`arun`); sync `run()` fails
// loud (di.py). LangGraph.js is async-native, so the sync/async split that
// FromResource creates in Python simply does not exist — every node body is a
// Promise. This is one of the few places TS is strictly nicer.
const parseActivity = node(
  {
    outputs: ActivityHistory,
    resources: {
      // PARITY: templated FromResource(uri=...) relocated to config again.
      activity: fromResource({ uri: "mcp://crm/deals/{deal_id}/activity" }),
    },
  },
  async (gate: GateAck, deal_id: FromInput<string>, activity: ActivityHistory): Promise<ActivityHistory> => {
    return activity;
  },
);

const consume = node(
  {
    mode: "think",
    outputs: Brief,
    model: "consume",
    prompt: "consume/brief",
    resources: {
      // PARITY: MANIFEST-mode FromResource(ref="email-history", max_bytes=...).
      // This `ref=` form hydrates a ResourceRef lifted from the get_deal tool
      // result onto the `neo_resource_manifest_*` channel. The whole manifest-lift
      // mechanism is UNMODELED in the matrix — see the getState beat below.
      history: fromResource({ ref: "email-history", maxBytes: 200_000 }),
    },
  },
  // Dead body (think mode). `history` reaches the prompt via the async di_inputs
  // path — the promptCompiler's `diInputs.history` above.
  async (parse_activity: ActivityHistory, history: EmailHistory): Promise<Brief> => {
    return undefined as any;
  },
);

// PARITY: construct_from_module is explicitly NOT in v0.1.0-ts ("no module
// introspection in TS; use explicit lists"). So the Python `_build_pipeline`
// that stuffs functions onto a throwaway module and calls construct_from_module
// becomes an explicit node list. Mechanically fine, slightly more verbose.
function buildPipeline(): Construct {
  return new Construct({ name: "deal-research", nodes: [research, gate, parseActivity, consume] });
}

// ── Driver ────────────────────────────────────────────────────────────────────

async function armExpiry(): Promise<void> {
  const client = new MultiServerMCPClient({ crm: stdioConnection(STATE_MARKER) });
  const session = await client.session("crm");
  try {
    await session.callTool("arm_email_expiry", {});
  } finally {
    await session.close();
  }
}

async function main(): Promise<void> {
  // (log-quieting omitted — structlog has no direct TS analogue; use pino level.)
  await import("node:fs/promises").then((fs) => fs.rm(STATE_MARKER, { force: true }));
  const dbPath = path.join(os.tmpdir(), `ex24_${Date.now()}_checkpoints.sqlite`);

  const server = new StdioServer({
    command: process.execPath,
    args: [DEMO_SERVER],
    env: { NEOGRAPH_MCP_DEMO_STATE: STATE_MARKER },
  });
  // PARITY: mcp_resource_fetcher → (fetcher, replayer) tuple. UNMODELED. The
  // replayer's job (re-invoke the idempotent producing call to re-derive an
  // expired resource_link) is a substantial neograph_mcp feature with zero matrix
  // coverage. The `-32002` expiry unwrap + resource_link preservation is
  // MCP-SDK-version-specific work that must be re-solved against the JS SDK.
  let [fetcher, replayer] = mcpResourceFetcher({ crm: server });
  fetcher = countingFetcher(fetcher);

  const pipeline = buildPipeline();

  const configBase = { mcp_resource_fetcher: fetcher, mcp_resource_replayer: replayer };

  // PARITY: AsyncSqliteSaver.from_conn_string context manager → JS saver has
  // fromConnString returning a saver you close in a finally. DIRECT; JS lacks
  // `async with`, so the try/finally is manual.
  const saver = await AsyncSqliteSaver.fromConnString(dbPath);
  try {
    const graph = compile(pipeline, { checkpointer: saver, llmFactory, promptCompiler });
    const thread = { configurable: { thread_id: "deal-D1", deal_id: DEAL_ID, ...configBase } };

    // ── Run 1 ────────────────────────────────────────────────────────────────
    CURRENT_RUN_ID.value = "run-1";
    let result: any = await arun(graph, { input: { node_id: "n1", deal_id: DEAL_ID }, config: thread });

    const summary: DealSummary = result.research;
    console.log(`[beat 1] headline: ${summary.headline}`);
    console.log(`[beat 1] subjects: ${summary.recent_email_subjects}`);

    // PARITY: `"__interrupt__" in result` → LangGraph.js surfaces interrupts under
    // the same `__interrupt__` key on the invoke result. DIRECT.
    if (!("__interrupt__" in result)) throw new Error("expected the human gate to pause");
    console.log(`[gate] paused: ${result.__interrupt__[0].value.message}`);

    // ── Beat 2: the manifest on the bus ───────────────────────────────────────
    // PARITY: graph.aget_state(thread) → graph.getState(thread). DIRECT call. BUT
    // the `neo_resource_manifest_{field}` CHANNEL and the ResourceRef shape it
    // holds (kind / uri / producing_call.tool_name / producer_idempotent) are the
    // manifest-lift feature — the factory/tool-loop scanning tool results for
    // resource_link blocks and lifting typed refs. UNMODELED in the matrix's
    // Factory + LLM rows. Reading it is easy; PRODUCING it is a whole subsystem.
    const state = await graph.getState(thread);
    const manifestField = `neo_resource_manifest_${fieldNameFor("research")}`;
    const manifest: Array<any> = state.values[manifestField] ?? [];
    for (const ref of manifest) {
      console.log(`  - kind=${ref.kind} uri=${ref.uri} idempotent=${ref.producing_call.producer_idempotent}`);
    }
    if (!manifest.some((r) => r.kind === "email-history")) throw new Error("email-history ref missing");

    // ── Beat 4 setup: link dies during the pause ──────────────────────────────
    await armExpiry();

    // ── Resume ────────────────────────────────────────────────────────────────
    // PARITY: Python `arun(graph, resume={"approved": True})` → LangGraph.js
    // `new Command({ resume: { approved: true } })` passed as input. Same two-call
    // pause/resume model; different resume vehicle. MOSTLY-DIRECT.
    CURRENT_RUN_ID.value = "run-2";
    result = await arun(graph, { input: new Command({ resume: { approved: true } }), config: thread });

    console.log(`[beat 3a] activity events: ${result.parse_activity.events.length}`);
    console.log(`[beat 3b/4] brief: ${result.consume.text}`);

    // ── Beat 5: per-run fetch cache ───────────────────────────────────────────
    // PARITY: the per-run fetch cache keyed on the framework-minted run id
    // (fetch-once-across-ReAct-supersteps, re-fetch-on-resume) is UNMODELED. The
    // assertion ports directly; the BEHAVIOR it checks depends on a caching seam
    // the matrix never describes.
    const r1 = FETCH_LOG.filter(([r, u]) => r === "run-1" && u.endsWith("/activity")).length;
    const r2 = FETCH_LOG.filter(([r, u]) => r === "run-2" && u.endsWith("/activity")).length;
    console.log(`[beat 5] run-1 activity reads: ${r1}, run-2 activity reads: ${r2}`);
    if (r1 !== 1) throw new Error(`expected dossier fetch cached across supersteps, got ${r1}`);
  } finally {
    await saver.close?.();
  }

  console.log("Done. Refs traveled on the bus; blobs fetched on demand and re-derived on expiry.");
}

main();
