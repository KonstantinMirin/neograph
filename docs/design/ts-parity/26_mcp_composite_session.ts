// TS parity sketch of: examples/26_mcp_composite_session.py
// "Composite over federated MCP primitives — mcp_session: ONE connection, N calls."
//
// This is a HYPOTHETICAL port against the PROPOSED API in
// docs/design/typescript-port.md (AD-0 transformer form). It is NOT meant to
// compile or run — it grounds a feature-parity analysis. Inline `// PARITY:`
// notes flag where the TS DX diverges from the Python original.
//
// The Python example is a `raw`-mode @node: a deterministic async body of the
// signature (state, config) -> state-update dict. It opens ONE mcp_session over
// a stdio subprocess, issues TWO federated primitive calls (crm_search typed +
// get_deal content), and assembles a typed DealReview. No LLM, no keys.

import { z } from "zod";
import { compile, arun, node, constructFromFunctions } from "@neograph/core";
// PARITY: the entire neograph_mcp battery must be re-implemented in TS as
// @neograph/mcp on top of @modelcontextprotocol/sdk + @langchain/mcp-adapters.
// This is a substantial NEW body of work not covered anywhere in the parity
// matrix (the matrix's MCP rows are absent — see report API GAP #1). The
// Python module leans on langchain-mcp-adapters 0.3.0 internals
// (_convert_mcp_content_to_lc_block) pinned by a parity test; the JS adapter
// package has a different surface, so the content-block conversion table
// (_convert_content_block: TextContent/ImageContent/ResourceLink/EmbeddedResource)
// must be re-derived against @langchain/mcp-adapters, not ported line-for-line.
import { mcpSession, StdioServer, type McpCallResult } from "@neograph/mcp";

import * as path from "node:path";
import * as url from "node:url";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const DEMO_SERVER = path.resolve(__dirname, "_mcp_demo_server.py");

// PARITY: StdioServer is a frozen dataclass in Python. In TS it is a plain
// class / readonly object literal — DIRECT. `process.execPath` stands in for
// `sys.executable`, but note the demo server is a PYTHON subprocess; the TS
// example would still shell out to `python examples/_mcp_demo_server.py`
// (there is no JS demo server), so `command` is a python interpreter path, not
// process.execPath. Cross-runtime, but the stdio transport doesn't care.
const CRM = new StdioServer({ command: "python", args: [DEMO_SERVER] });

// ── Per-run identity: the token_provider (beat 3) ────────────────────────────
// Python: `def token_provider(configurable: dict[str, Any]) -> str`.
// mcp_session calls this ONCE-per-call with config['configurable'].
//
// PARITY (DIRECT): a standalone function taking the configurable bag maps 1:1.
// TokenProvider in Python is `Callable[..., Awaitable[str] | str]` — it may
// take the configurable OR no args (the _declares_arg / _resolve_token vs
// _resolve_token_no_config fork inspects the fn's ARITY via inspect.signature).
// PARITY (REDESIGN): that arity fork has NO runtime equivalent in TS — a JS
// function's declared parameters are erasable and `fn.length` is unreliable
// (defaults/rest change it). The TS API must pick ONE provider shape (always
// pass the configurable) or make the caller flag it explicitly. See report.
function tokenProvider(configurable: Record<string, unknown>): string {
  return (configurable["mcp_auth"] as string) ?? "anon";
}

// ── Client-side result models ────────────────────────────────────────────────
// Python uses `class X(BaseModel, frozen=True)`. TS uses Zod schemas +
// inferred types. PARITY (Direct, per AD-1): the shape maps cleanly, but this
// example needs BOTH a runtime schema (for output_model rehydration) AND a
// static type (for the assembly code) — hence the schema+infer pairing on
// every model, which is more ceremony than one frozen Pydantic class.

const DealHit = z.object({
  id: z.string(),
  name: z.string(),
  stage: z.string(),
});
type DealHit = z.infer<typeof DealHit>;

const CrmSearchResult = z.object({
  query: z.string(),
  hits: z.array(DealHit),
  acting_as: z.string(), // the per-run identity the server echoed
});
type CrmSearchResult = z.infer<typeof CrmSearchResult>;

const DealReview = z.object({
  deal_id: z.string(),
  deal_name: z.string(),
  stage: z.string(),
  acting_as: z.string(),
  manifest_refs: z.number().int(), // resource_link blocks get_deal returned
});
type DealReview = z.infer<typeof DealReview>;

// ── The composite: ONE session, two primitives, assembled result ─────────────
//
// Python:
//   @node(mode="raw", outputs=DealReview)
//   async def deal_review(state, config): ...
//
// PARITY (REDESIGN): `mode="raw"` is the ONE mode whose contract is a fixed
// (state, config) signature — the transformer's "signature IS the DAG" premise
// does NOT apply (there are no typed upstream params to extract; state/config
// are framework ports, not edges). So raw mode falls back to the EXPLICIT
// wrapper form. `outputs` must be given as a Zod schema because there is no
// return annotation to extract (the body returns a partial state-update dict,
// not a DealReview). This is exactly the "fall back to explicit schemas where
// signature extraction can't reach" case.
//
// PARITY (REDESIGN): the Python decorator both DECLARES metadata and enforces
// the raw signature at decoration time (mode='raw' requires the (state, config)
// shape). The TS `node()` wrapper receives the fn as a value, so it can only
// duck-type the callback shape at runtime — no build-time enforcement of the
// two-parameter raw contract without transformer cooperation (which the matrix
// does not describe for raw mode).
const dealReview = node(
  { mode: "raw", outputs: DealReview },
  // PARITY: `state` is the LangGraph.js channels object; `config` is
  // RunnableConfig. Types are hand-written here — the transformer offers
  // nothing for the raw port params.
  async (state: Record<string, unknown>, config: any): Promise<Record<string, unknown>> => {
    const query = config.configurable.query as string;

    // Beat 1: ONE session. Python uses `async with mcp_session(...) as s`.
    // PARITY (REDESIGN): TS has no `async with`. The proposed API must expose
    // an explicit lifecycle. Two options, both worse than Python's context
    // manager for the nmb2 "enter+exit in the SAME task, never store it"
    // invariant:
    //   (a) `await using s = mcpSession(...)`  — TS 5.2 explicit resource mgmt
    //       via Symbol.asyncDispose. Closest analogue; shown below.
    //   (b) `await mcpSession(...).use(async (s) => { ... })` callback scope.
    // `await using` is the faithful choice but (i) requires TS>=5.2 +
    // downlevel helper, and (ii) does NOT structurally guarantee same-task
    // enter/exit the way Python's anyio cancel-scope does — the JS MCP SDK's
    // stdio transport has its own task model, so the RuntimeError-on-cross-task
    // guard the Python session raises has no clean JS equivalent (see report).
    await using s = mcpSession("crm", CRM, {
      tokenProvider,
      config,
      // stdioTokenArg defaults to "token"; timeout defaults to 30_000 ms.
    });

    // Beat 1+2: primitive #1 — typed search (rehydrated into CrmSearchResult).
    // Python: s.call("crm_search", {query}, output_model=CrmSearchResult).
    // PARITY (REDESIGN): Python's `call` is an @overload'd method whose return
    // type NARROWS on the presence of `output_model` (McpCallResult vs the
    // model instance). TS CAN reproduce this with conditional-type overloads on
    // `outputModel`, but rehydration itself (Python's `rehydrate(model, parse,
    // structuredContent)`) must re-validate the server's structuredContent with
    // Zod's `.parse()` instead of `Model(**structuredContent)`. Same missing-
    // structuredContent error path must be re-created by hand.
    const search = await s.call("crm_search", { query }, {
      outputModel: CrmSearchResult,
    });
    const top = search.hits[0];

    // Beat 1: primitive #2 — get_deal over the SAME session. Content-only, so
    // we read the McpCallResult, not a model.
    const deal: McpCallResult = await s.call("get_deal", { deal_id: top.id });

    // Python: detail = json.loads(deal.text or "{}").
    // PARITY (DIRECT): JSON.parse + the `.text` convenience getter port 1:1.
    const detail = JSON.parse(deal.text ?? "{}") as Record<string, unknown>;

    // Python: sum(1 for b in deal.content if b.get("type") == "file").
    // PARITY (DIRECT): the resource_link manifest arrives as file content
    // blocks; a filter+length maps cleanly. NOTE the block-type string ("file")
    // depends on @langchain/mcp-adapters' block shape matching the Python
    // langchain_core.messages.content.create_file_block output — a cross-package
    // contract the TS @neograph/mcp must pin with its own parity test.
    const manifestRefs = deal.content.filter((b) => b["type"] === "file").length;

    // `await using s` disposes here (end of scope), mirroring `async with` exit.

    // PARITY (Direct-ish): Python constructs `DealReview(...)`; TS validates the
    // literal through the Zod schema. The returned state-update dict key
    // ("deal_review") must match the node name — same brittle string-keying as
    // Python's raw mode; the transformer can't help because raw bodies return
    // opaque partial-state dicts.
    return {
      deal_review: DealReview.parse({
        deal_id: top.id,
        deal_name: top.name,
        stage: (detail["stage"] as string) ?? top.stage,
        acting_as: search.acting_as,
        manifest_refs: manifestRefs,
      }),
    };
  },
);

// Python: construct_from_functions("deal-review", [deal_review]).
// PARITY (DIRECT): a single-node construct from an explicit list. Note the
// parity matrix lists construct_from_MODULE as "Not in v0.1.0-ts" (no module
// introspection), but construct_from_FUNCTIONS takes an explicit list, so it
// ports — this example happens to use the supported one.
const pipeline = constructFromFunctions("deal-review", [dealReview]);

async function main(): Promise<void> {
  console.log("=".repeat(66));
  console.log("Composite over federated MCP primitives — one session, two calls");
  console.log("=".repeat(66));

  const graph = compile(pipeline);

  for (const operator of ["operator-A", "operator-B"]) {
    // Python: await neograph.arun(graph, input={}, config={configurable:{...}}).
    // PARITY (DIRECT): arun + config injection maps to LangGraph.js invoke.
    const result = await arun(graph, {
      input: {},
      config: { configurable: { query: "Acme", mcp_auth: operator } },
    });
    const review = result["deal_review"] as DealReview;
    console.log(`\nRun as ${operator}:`);
    console.log(`  deal        : ${review.deal_id} — ${review.deal_name} [${review.stage}]`);
    console.log(`  acting_as   : ${review.acting_as}  (identity minted once, rode both calls)`);
    console.log(`  manifest    : ${review.manifest_refs} resource_link refs from get_deal`);
    console.assert(review.deal_id === "D1");
    console.assert(review.acting_as === operator);
    console.assert(review.manifest_refs === 2);
  }
}

// PARITY (DIRECT): top-level await or an explicit main() invocation.
void main();
