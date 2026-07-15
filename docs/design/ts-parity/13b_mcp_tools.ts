// HYPOTHETICAL TypeScript port of examples/13b_mcp_tools.py
// -----------------------------------------------------------------------------
// Python source: /Users/konst/projects/neograph/examples/13b_mcp_tools.py
// Proposed API:  /Users/konst/projects/neograph/docs/design/typescript-port.md (AD-0 transformer form)
//
// This is a DX-faithful SKETCH against the PROPOSED neograph-ts API. It is NOT
// meant to compile or run — there is no TS implementation. `// PARITY:` notes
// mark every point where the TS DX diverges from the Python original.
//
// What the Python example demonstrates, feature by feature:
//   * @node(mode="agent") bound to a RAW LangChain BaseTool passed in tools=[...]
//     (no Tool(name) spec, no register_tool_factory — auto-normalized at compile)
//   * The tool is ASYNC-ONLY (StructuredTool.from_function(coroutine=...)), which
//     forces the graph to be driven by arun() rather than run()
//   * lint() flags kind="tool_requires_async_driver" at compile time
//   * The tool returns a TYPED Pydantic model preserved in ToolInteraction.typed_result
//   * construct_from_module(mod) assembles the pipeline from a module's @node fns
// -----------------------------------------------------------------------------

import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { AIMessage } from "@langchain/core/messages";
import {
  node,
  compile,
  run,               // PARITY: see the arun/run collapse note below
  lint,
  constructFromFunctions, // PARITY: replaces construct_from_module (see §5)
} from "@neograph/core";

// -- Schemas ------------------------------------------------------------------
// PARITY (DIRECT-ish): Python `class EvidenceHit(BaseModel, frozen=True)` becomes a
// Zod object. `frozen=True` has no analog — Zod-parsed values are plain objects,
// already effectively immutable, so `frozen` simply disappears. The bigger loss:
// a Zod schema is a *validator*, not a *class*. `EvidenceHit` in Python is a
// nominal type carried on the tool's typed result; in TS the parsed value is a
// structural `{source_file, line, snippet}` with no `instanceof` identity.
const EvidenceHit = z.object({
  source_file: z.string(),
  line: z.number().int(),
  snippet: z.string(),
});
type EvidenceHit = z.infer<typeof EvidenceHit>;

const ExplorationResult = z.object({
  evidence: z.array(z.string()),
  summary: z.string(),
});
type ExplorationResult = z.infer<typeof ExplorationResult>;

// -- The "MCP tool": a raw LangChain tool -------------------------------------
// In production this is one element of `await loadMcpTools(session)` from
// @langchain/mcp-adapters. Here we build an equivalent local tool.
//
// PARITY (BLOCKED — the example's central premise): Python distinguishes an
// async-only tool (StructuredTool.from_function(coroutine=fn), func=None) from a
// sync tool. That distinction is the WHOLE POINT of 13b. In JS there is no
// sync/async tool split: every LangChain.js tool `.invoke()` returns a Promise.
// So `coroutine=` vs `func=` has no TS analog — the async-only-ness is not a
// property you can even express, because ALL tools are async. Everything below
// that hangs off "this tool is async-only" (arun(), the lint guardrail, the
// ConfigurationError) loses its reason to exist. See §3 and §4.
function makeFakeMcpTool() {
  return tool(
    async ({ query }: { query: string }): Promise<EvidenceHit> => {
      // A real MCP tool would round-trip a tools/call over the session here.
      await Promise.resolve(); // yield to the loop, like a real network hop
      return {
        source_file: "auth.py",
        line: 42,
        snippet: "def authenticate(user, password): ...",
      };
    },
    {
      name: "search_evidence",
      description: "Search the codebase for evidence supporting a claim.",
      // PARITY (API GAP): Python's StructuredTool.from_function INFERRED the
      // `query: str` arg schema from the coroutine signature. The AD-0 transformer
      // is scoped to @node-decorated functions ONLY — it does NOT reach into
      // free-standing tool callbacks. So the arg schema must be hand-written Zod
      // here, duplicating what the TS parameter type already says. The
      // "signature IS the schema" DX does not extend to tools.
      schema: z.object({ query: z.string() }),
      // PARITY (REDESIGN): to preserve a TYPED result (EvidenceHit) the way Python
      // keeps ToolInteraction.typed_result, the tool must opt into the
      // content-and-artifact channel; a plain LangChain.js tool return is coerced
      // to string content. neograph-ts would have to read the artifact slot.
      responseFormat: "content_and_artifact",
    },
  );
}

const mcpTool = makeFakeMcpTool();

// -- Fake async LLM (replace with a real model in production) -----------------
// PARITY (DIRECT): the Python fake implements bind_tools / with_structured_output
// / ainvoke. LangChain.js chat models expose bindTools / withStructuredOutput /
// invoke (all Promise-returning). The async-only `ainvoke` collapses to `invoke`.
// PARITY (REDESIGN, minor): the final structured parse returns an ExplorationResult
// *instance* in Python; in TS withStructuredOutput(zodSchema) yields a plain
// object matching the schema — no class instance, no nominal identity.
class FakeAsyncExploreLLM {
  private calls = 0;
  private structured = false;
  private static readonly ANSWER: ExplorationResult = {
    evidence: ["auth.py:42"],
    summary: "found a supporting reference via the MCP tool",
  };

  bindTools(_tools: unknown[]): FakeAsyncExploreLLM {
    return this; // keep the call counter across rebinds
  }

  withStructuredOutput(_schema: unknown): FakeAsyncExploreLLM {
    const clone = new FakeAsyncExploreLLM();
    clone.calls = this.calls;
    clone.structured = true;
    return clone;
  }

  async invoke(_messages: unknown[]): Promise<AIMessage | ExplorationResult> {
    if (this.structured) {
      return FakeAsyncExploreLLM.ANSWER; // final-turn structured parse
    }
    this.calls += 1;
    if (this.calls === 1) {
      return new AIMessage({
        content: "",
        tool_calls: [
          { name: "search_evidence", args: { query: "verify claim" }, id: "c1" },
        ],
      });
    }
    // Final turn: carry the answer as JSON so the loop can exit.
    return new AIMessage({ content: JSON.stringify(FakeAsyncExploreLLM.ANSWER) });
  }
}

function llmFactory(_tier: string): FakeAsyncExploreLLM {
  return new FakeAsyncExploreLLM();
}

function promptCompiler(_template: unknown, _data: unknown): unknown[] {
  return [{ role: "user", content: "explore" }];
}

// -- Pipeline: one agent node bound to the raw MCP tool -----------------------
// PARITY (DIRECT with transformer): the AD-0 transformer extracts the empty input
// set and the `ExplorationResult` return type from the arrow signature, emitting
// __neo_meta = { inputs: {}, output: ExplorationResult }. `mode:"agent"` +
// `tools:[mcpTool]` pass through as config, exactly like the Python kwargs.
//
// PARITY (SKIP): Python's decoration-time dead-body UserWarning (AST inspection of
// `explore`'s body) is explicitly out of scope for TS. The no-op arrow body below
// is silently accepted; the LLM drives execution via prompt+tools.
const explore = node(
  {
    mode: "agent",
    model: "research",
    prompt: "verify/explore",
    tools: [mcpTool], // <-- raw LangChain tool, passed directly (auto-registered)
  },
  (): ExplorationResult => {
    // Body unused -- the LLM drives execution via prompt + tools
    return undefined as unknown as ExplorationResult;
  },
);

function buildPipeline() {
  // PARITY (REDESIGN / GAP): Python uses construct_from_module(mod) — it
  // introspects a module object and harvests every @node attribute. TS has no
  // module-object introspection (the design doc lists construct_from_module as
  // NOT shipping in v0.1.0-ts). The idiomatic replacement is an EXPLICIT list of
  // the node handles. Cheap here (one node); for a large module this shifts a
  // "just point at the module" ergonomic onto the author.
  return constructFromFunctions("mcp-explore", [explore]);
}

async function main(): Promise<void> {
  const pipeline = buildPipeline();

  // PARITY (BLOCKED / VESTIGIAL): in Python, lint() surfaces
  // kind="tool_requires_async_driver" to warn you to use arun() before you run.
  // In TS there is no sync driver to protect against — run() is already async —
  // so this lint kind has NOTHING to guard. Either it is dropped entirely
  // (honest: the hazard cannot occur) or it degrades to a no-op that always
  // returns zero issues. Faithfully porting the example means deleting its
  // headline guardrail. Shown here filtering an issue kind that can never fire.
  const issues = lint(pipeline);
  const asyncIssues = issues.filter((i) => i.kind === "tool_requires_async_driver");
  console.log("-- lint() MCP guardrail (vestigial in TS: no sync driver) --");
  for (const issue of asyncIssues) {
    console.log(`  [${issue.kind}] ${issue.message}`);
  }
  console.log();

  const graph = compile(pipeline, {
    llmFactory,
    promptCompiler,
    // No toolFactories for search_evidence: auto-registered from the raw tool.
  });

  // PARITY (REDESIGN): Python has TWO drivers — run() (sync) and arun() (async) —
  // and this example REQUIRES arun() because the tool is async-only. TS collapses
  // both into a single always-async run(). The `input` payload maps 1:1.
  const result = await run(graph, { input: { node_id: "MCP-001" } });

  console.log("-- run() result --");
  console.log(`  summary: ${result.explore.summary}`);
  console.log(`  evidence: ${result.explore.evidence}`);
  console.log();
  console.log("Note: 'search_evidence' had no registerToolFactory call. It was a");
  console.log("raw LangChain tool in tools=, auto-registered at compile().");
}

// PARITY (DIRECT): Python's `asyncio.run(main())` becomes a top-level awaited call.
void main();
