// TS parity sketch of: examples/07_llm_configuration.py
//
// HYPOTHETICAL — there is no neograph-ts implementation. This is written against
// the PROPOSED API in docs/design/typescript-port.md (AD-0 transformer form:
// `const decompose = node({model, prompt}, (topic: RawText): Claims => {...})`;
// pipe → `.pipe(...)`; Zod schemas per AD-1). Not meant to compile or run.
//
// Feature: per-node LLM configuration — model tier routing, temperature,
// max_tokens budgets, a user-supplied llm_factory + prompt_compiler, and
// think-mode nodes whose bodies never run.
//
// The Python source wires the DAG with `construct_from_module(sys.modules[__name__])`
// and lets each `@node(mode="think", model=..., llm_config={...})` flow its config
// through to a factory of shape `factory(tier, node_name=None, llm_config=None)`.

import { z } from "zod";
import {
  node,
  constructFromFunctions, // PARITY: NOT construct_from_module — see notes at bottom
  compile,
  run,
  type LlmFactory,
  type PromptCompiler,
} from "@neograph/core";

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY: Python `class Claims(BaseModel, frozen=True)` → Zod object.
// `frozen=True` has no Zod analogue at runtime; the closest is a `readonly`
// static type via `z.infer`. Zod validates shape but does not freeze instances.
const Claims = z.object({
  items: z.array(z.string()),
});
type Claims = Readonly<z.infer<typeof Claims>>;

const Classification = z.object({
  claim: z.string(),
  category: z.string(),
});
type Classification = Readonly<z.infer<typeof Classification>>;

const ClassifiedClaims = z.object({
  classified: z.array(Classification),
});
type ClassifiedClaims = Readonly<z.infer<typeof ClassifiedClaims>>;

// ══════════════════════════════════════════════════════════════════════════
// LLM FACTORY — the single place all LLM configuration happens.
// Python contract: factory(tier, node_name=None, llm_config=None). neograph
// introspects the factory signature (_accepted_params) and passes ONLY the
// kwargs the factory declares.
//
// PARITY (HIGH friction): TS has no runtime `inspect.signature`. The proposed
// design leans on the AD-0 transformer for @node signatures, but the
// llm_factory is a plain user function passed to compile() — the transformer
// does not necessarily rewrite it. Options:
//   (a) always pass the full context object (no introspective filtering), or
//   (b) require the factory to accept a single `ctx` object.
// This sketch uses (b): a single typed context object. This DIVERGES from the
// Python positional/keyword+defaults contract, and the introspection-gated
// "only pass what you declare" behavior is LOST.
// ══════════════════════════════════════════════════════════════════════════

const USE_FAKE = process.argv.includes("--fake");

// PARITY: LangChain.js runnable interface stands in for `ChatOpenAI`.
const realLlmFactory: LlmFactory = ({ tier, nodeName, llmConfig }) => {
  // PARITY: dotenv/env access is identical in spirit (process.env).
  const cfg = llmConfig ?? {};
  const models: Record<string, string> = {
    fast: "openai/gpt-4o-mini",
    reason: "openai/gpt-4o",
  };

  // PARITY: LangChain.js ChatOpenAI import; `provider_kwargs` flattens into
  // the factory-kwargs object (Python LlmConfig.as_factory_kwargs()).
  return new ChatOpenAI({
    model: models[tier] ?? models.fast,
    apiKey: process.env.OPENROUTER_API_KEY!,
    configuration: { baseURL: "https://openrouter.ai/api/v1" },
    temperature: (cfg.temperature as number) ?? 0,
    maxTokens: (cfg.max_tokens as number) ?? undefined,
  });
};

// Test factory: records what config each node received.
// PARITY (MEDIUM friction): Python's FakeLLM is duck-typed — any object with
// `with_structured_output(model)` + `invoke(messages)` works. In TS the fake
// must structurally satisfy the LangChain Runnable interface (or a neograph
// StructuredLLM interface). More boilerplate; no free duck typing.
//
// PARITY (MEDIUM): Python does identity checks `self._model is Claims` on the
// Pydantic CLASS. Here the "model" handed to withStructuredOutput is a Zod
// SCHEMA object. Identity (`=== Claims`) works by reference, but the semantics
// shift from "is this the class" to "is this the schema object" — and if the
// framework clones/wraps the schema before handing it over, `===` breaks.
const fakeLlmFactory: LlmFactory = ({ tier, nodeName, llmConfig }) => {
  const cfg = llmConfig ?? {};
  console.log(
    `  [factory] node=${nodeName}, tier=${tier}, ` +
      `temp=${cfg.temperature ?? "default"}, ` +
      `max_tokens=${cfg.max_tokens ?? "default"}`,
  );

  let boundSchema: z.ZodTypeAny | null = null;
  return {
    withStructuredOutput(schema: z.ZodTypeAny) {
      boundSchema = schema;
      return this;
    },
    invoke(_messages: unknown) {
      if (boundSchema === Claims) {
        return { items: ["claim-1", "claim-2"] } satisfies Claims;
      }
      if (boundSchema === ClassifiedClaims) {
        return {
          classified: [
            { claim: "claim-1", category: "security" },
            { claim: "claim-2", category: "reliability" },
          ],
        } satisfies ClassifiedClaims;
      }
      return {};
    },
  };
};

const llmFactory = USE_FAKE ? fakeLlmFactory : realLlmFactory;

// PARITY: prompt_compiler(template, data) → direct. Template literal body.
const promptCompiler: PromptCompiler = (_template, data) => [
  {
    role: "user",
    content: data
      ? `Process this: ${JSON.stringify(data)}`
      : "Generate claims about system security",
  },
];

// ══════════════════════════════════════════════════════════════════════════
// PIPELINE — each node has its own LLM configuration.
// ══════════════════════════════════════════════════════════════════════════

// Creative decomposition: high temperature, more tokens.
// PARITY: `@node(mode="think", outputs=Claims, model="reason", prompt=...)`
// → node({...}, fn). The AD-0 transformer extracts output type `Claims` from
// the return annotation, so `outputs:` need not be restated. mode="think" is
// inferred from prompt+model presence (matrix: "Mode inference — Direct").
const decompose = node(
  {
    mode: "think",
    model: "reason", // "reason" tier → more capable model
    prompt: "decompose",
    llmConfig: {
      providerKwargs: {
        temperature: 0.9, // creative — explore diverse decompositions
        maxTokens: 2000,
      },
    },
  },
  // PARITY (LOW friction but real): Python think-mode bodies are `...` (dead
  // code). TS has no `...` statement expression; a callback needs a body. The
  // idiomatic stand-in is a throwing stub — it documents "never runs" but is
  // uglier than Python's `...`. The Python "dead-body UserWarning" (AST check)
  // is explicitly SKIPPED in TS (matrix), so nothing warns if you accidentally
  // put real logic here.
  (): Claims => {
    throw new Error("unreachable: mode='think' — LLM executes via prompt");
  },
);

// Precise classification: zero temperature, generous token budget.
// PARITY: the parameter `decompose: Claims` — its NAME is the edge to the
// decompose node, its TYPE is the fan-in contract. The AD-0 transformer must
// still extract this param even though the body never runs (think mode), so
// the DAG edge survives.
const classify = node(
  {
    mode: "think",
    model: "fast", // "fast" tier → cheaper model
    prompt: "classify",
    llmConfig: {
      providerKwargs: {
        temperature: 0, // deterministic — consistent classification
        maxTokens: 2000, // must fit one structured row per decomposed claim
      },
    },
  },
  (decompose: Claims): ClassifiedClaims => {
    throw new Error("unreachable: mode='think' — LLM executes via prompt");
  },
);

// PARITY (HIGH friction — API GAP): Python does
//   construct_from_module(sys.modules[__name__], name="configured-pipeline")
// which introspects the module namespace, finds every @node, and wires the DAG
// by parameter names. TS has NO module-namespace introspection, and
// typescript-port.md explicitly lists construct_from_module as "Not in
// v0.1.0-ts". The port MUST enumerate nodes explicitly.
const pipeline = constructFromFunctions("configured-pipeline", [decompose, classify]);

// ── Run ────────────────────────────────────────────────────────────────────

async function main() {
  console.log("LLM factory calls:\n");

  // PARITY: compile(construct, {llm_factory, prompt_compiler}) → options object.
  const graph = compile(pipeline, {
    llmFactory,
    promptCompiler,
  });

  // PARITY: run(graph, input={...}). LangGraph.js invoke is async, so the whole
  // entry point becomes `async`/`await` — Python's `run()` is sync here.
  const result = await run(graph, { input: { node_id: "REQ-001" } });

  // PARITY (MEDIUM friction — API GAP): `result['decompose'].items`. Python
  // returns a dict keyed by node name; state is dynamically typed so `.items`
  // just works. In TS `result` is `Record<string, unknown>` (node names are
  // runtime strings), so `result.decompose` is `unknown` — you must cast or
  // the proposed API must generate a typed result shape from the construct.
  const decomposed = result["decompose"] as Claims;
  const classified = result["classify"] as ClassifiedClaims;

  console.log(`\nDecomposed: ${decomposed.items}`);
  console.log("Classified:");
  for (const c of classified.classified) {
    console.log(`  - ${c.claim} -> ${c.category}`);
  }
  console.log();
  console.log("Note: output_strategy is per-node — mix structured and json_mode");
  console.log("in the same pipeline when using different model providers.");
}

// PARITY: `if __name__ == "__main__"` → conventional entrypoint guard.
main();
