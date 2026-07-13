// TS parity sketch of: examples/vs_langgraph/02_tool_agent.py
// Comparison 2: Tool-Calling Agent — LLM decides which tools to call, with a budget.
//
// This is a HYPOTHETICAL port against the PROPOSED API in docs/design/typescript-port.md.
// It is NOT meant to compile or run — it grounds a feature-parity analysis.
//
// KEY SHAPE NOTE: the neograph half of this example uses the PROGRAMMATIC surface
// (`Node(...)` + `Construct` + `compile` + `run`), NOT the `@node` decorator. Per the
// parity matrix this is the "TS-first surface" that "maps naturally / Direct". So the
// AD-0 compiler transformer (signature-extraction) plays NO role in this example — there
// is no function signature to extract; the DAG is one hand-built Node. The friction here
// is entirely in the LLM-integration seams (prompt_compiler, tool factory), not the IR.

import { z } from "zod";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { HumanMessage } from "@langchain/core/messages";
import { Node, Construct, Tool, compile, run } from "@neograph/core";

const MODEL = "openai/gpt-4o-mini";

const baseLlm = new ChatOpenAI({
  model: MODEL,
  apiKey: process.env.OPENROUTER_API_KEY,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});

// ── Tool (shared by both approaches) ──────────────────────────────────────
const searchCalls: Record<string, number> = { langgraph: 0, neograph: 0 };

// PARITY (REDESIGN, LangChain-side): Python's `@tool` decorator infers the tool NAME
// from the fn name, the DESCRIPTION from the docstring, and the ARGS SCHEMA from the
// `query: str` type hint. JS has no docstrings and erases the param type at runtime, so
// LangChain.js `tool()` requires ALL of that passed EXPLICITLY: a Zod schema for args, a
// `name`, and a `description` string. The one-line Python decorator becomes a config object.
const searchWeb = tool(
  async ({ query }: { query: string }): Promise<string> => {
    searchCalls.current = (searchCalls.current ?? 0) + 1;
    return `Search result for '${query}': Found 3 relevant articles about ${query}.`;
  },
  {
    name: "search_web",
    description: "Search the web for information about a topic.",
    schema: z.object({ query: z.string() }),
  },
);

// PARITY (Direct, low friction): a Pydantic BaseModel becomes a Zod object. The class
// identity is lost (no `ResearchResult` type carrying methods), but the field/type shape
// that `outputs=` and describe_type care about is fully preserved. Derive the static type
// with z.infer where the value type is needed.
const ResearchResult = z.object({
  findings: z.array(z.string()),
  sources_consulted: z.number().int(),
});
type ResearchResult = z.infer<typeof ResearchResult>;

// ═══════════════════════════════════════════════════════════════════════════
// LANGGRAPH WAY — router, conditional edges, manual ReAct cycle
// (LangGraph.js mirrors the Python API almost 1:1 — this half is essentially Direct)
// ═══════════════════════════════════════════════════════════════════════════
async function runLanggraph(): Promise<[string, number]> {
  const { StateGraph, START, END, Annotation, messagesStateReducer } = await import("@langchain/langgraph");
  const { ToolNode } = await import("@langchain/langgraph/prebuilt");

  searchCalls.current = 0;

  // PARITY (Redesign, per matrix AD-4): Python's `Annotated[list, add_messages]` TypedDict
  // becomes LangGraph.js `Annotation.Root` with an explicit reducer. Same semantics, more ceremony.
  const AgentState = Annotation.Root({
    messages: Annotation<any[]>({ reducer: messagesStateReducer, default: () => [] }),
  });

  const tools = [searchWeb];
  const llmWithTools = baseLlm.bindTools(tools);
  const toolNode = new ToolNode(tools);

  const agent = async (state: typeof AgentState.State) => ({
    messages: [await llmWithTools.invoke(state.messages)],
  });

  // --- Router: IDENTICAL boilerplate in every ReAct agent ---
  const shouldContinue = (state: typeof AgentState.State) => {
    const last = state.messages[state.messages.length - 1];
    return last.tool_calls?.length ? "tools" : END;
  };

  const graph = new StateGraph(AgentState)
    .addNode("agent", agent)
    .addNode("tools", toolNode)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", shouldContinue, ["tools", END])
    .addEdge("tools", "agent"); // cycle back
  const app = graph.compile();

  const result = await app.invoke({
    messages: [new HumanMessage("Research microservice authentication patterns. Use the search tool.")],
  });
  const lastMsg = result.messages[result.messages.length - 1].content as string;
  return [lastMsg, searchCalls.current];
}

// ═══════════════════════════════════════════════════════════════════════════
// NEOGRAPH WAY — one Node, agent mode = ReAct loop, budget = max 3 searches
// ═══════════════════════════════════════════════════════════════════════════
async function runNeograph(): Promise<[ResearchResult, number]> {
  searchCalls.current = 0;

  // PARITY (Direct): the programmatic Node maps 1:1. `mode="agent"` string, `outputs` takes
  // the Zod schema instead of a Pydantic class, `tools=[Tool(...)]` is unchanged, budget=3
  // rides through to the same ReAct-loop budget tracker (matrix: "Tool budget tracking — Direct").
  const research = new Node({
    name: "research",
    mode: "agent",
    outputs: ResearchResult,
    model: "fast",
    prompt: "research",
    tools: [new Tool({ name: "search_web", budget: 3 })],
  });

  const pipeline = new Construct("research", { nodes: [research] });

  const graph = compile(pipeline, {
    // PARITY (Direct): tool factory closure — creates a tool instance per node. The
    // (config, toolConfig) params map straight across; it just returns the LangChain.js tool.
    toolFactories: { search_web: (_config, _toolConfig) => searchWeb },

    // PARITY (Direct): tier -> LLM. Trivial closure.
    llmFactory: (_tier) => baseLlm,

    // PARITY (HIGH friction — see report): Python's prompt_compiler signature is
    //   `lambda template, data, **kw: [...]`.
    // The `**kw` is load-bearing: neograph introspects the compiler's signature at runtime
    // (`_accepted_params` / `_ACCEPT_ALL`) and ONLY passes the gated `di_inputs` kwarg when the
    // compiler declares it (or **kwargs). JS cannot reliably introspect which named params a
    // function accepts — arrow fns erase names, `fn.length` counts only positional arity, and
    // `fn.toString()` parsing breaks on destructuring/minification. So the whole introspection-
    // gated opt-in seam (neograph-euyh) has no faithful TS port. Best TS shape is a fixed object
    // param `{ template, data, diInputs? }` where the framework ALWAYS passes diInputs and the
    // compiler ignores it if unused — losing the "compiler opts in by declaring the param" contract.
    promptCompiler: ({ template, data }: { template: string; data: unknown }) => [
      { role: "user", content: "Research microservice authentication patterns. Use the search tool." },
    ],
  });

  // PARITY (Direct): run() + config injection. `node_id` is ambient run input.
  const result = await run(graph, { input: { node_id: "demo" } });
  return [result.research as ResearchResult, searchCalls.current];
}

// ═══════════════════════════════════════════════════════════════════════════
async function main() {
  console.log("=".repeat(60));
  console.log("LANGGRAPH (no budget — LLM decides when to stop):");
  console.log("=".repeat(60));
  const [text, lgCalls] = await runLanggraph();
  console.log(`Search calls: ${lgCalls}`);
  console.log(`Result: ${text.slice(0, 200)}...`);

  console.log();
  console.log("=".repeat(60));
  console.log("NEOGRAPH (budget=3 — enforced, then forced to respond):");
  console.log("=".repeat(60));
  const [result, ngCalls] = await runNeograph();
  console.log(`Search calls: ${ngCalls}`);
  console.log(`Result:`, result);
}

void main();
