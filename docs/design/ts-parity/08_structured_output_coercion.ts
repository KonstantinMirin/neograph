// TS parity sketch of examples/08_structured_output_coercion.py
// HYPOTHETICAL — targets the PROPOSED neograph-ts API (docs/design/typescript-port.md).
// NOT compilable/runnable. Uses the AD-0 transformer wrapper form:
//   node({model, prompt}, (params): Out => {...})   // transformer extracts signature -> __neo_meta
// Inline `// PARITY:` notes mark every place the TS DX diverges from the Python original.
//
// What this example exercises (Python): @node think-mode, per-node llm_config output_strategy,
// name= override, construct_from_module auto-collection, custom llm_factory + prompt_compiler,
// compile()/run(). Three disconnected ROOT nodes (no params, no edges) that each emit Claims.

import { z } from "zod";
import { node, compile, run } from "@neograph/core";
import { AIMessage } from "@langchain/core/messages";

// ── Schema ───────────────────────────────────────────────────────────────
// PARITY: Python `class Claims(BaseModel, frozen=True)` -> a Zod schema PLUS a TS type.
// Two artifacts instead of one. `frozen=True` (Pydantic immutability) has no direct Zod
// equivalent — Zod validates but does not freeze; you'd add `Object.freeze` or `.readonly()`
// on the inferred type to approximate. Minor, but it is not the single-declaration Python has.
const Claims = z.object({
  items: z.array(z.string()),
});
type Claims = z.infer<typeof Claims>;

// ══════════════════════════════════════════════════════════════════════════
// FAKE LLMs — each simulates a different model behavior (test scaffolding, NOT neograph API)
// ══════════════════════════════════════════════════════════════════════════
// PARITY: these are LangChain-shaped fakes; they port to LangChain.js shapes 1:1.
// `with_structured_output(model)` -> `.withStructuredOutput(zodSchema)`. Note the Python
// fake returns a *constructed model instance* (`self._model(items=[...])`); the TS structured
// path returns a *parsed plain object* validated by Zod — no class instance. Behaviorally equal
// for this example, but a consumer who relied on Pydantic methods on the result would notice.

class CleanLLM {
  private _schema!: z.ZodTypeAny;
  withStructuredOutput(schema: z.ZodTypeAny) {
    this._schema = schema;
    return this;
  }
  async invoke(_messages: unknown): Promise<Claims> {
    return { items: ["claim-1", "claim-2"] };
  }
}

class FencedJsonLLM {
  async invoke(_messages: unknown) {
    return new AIMessage('```json\n{"items": ["claim-1", "claim-2"]}\n```');
  }
}

class VerboseTextLLM {
  async invoke(_messages: unknown) {
    return new AIMessage(
      "Let me analyze this requirement.\n\n" +
        "After careful consideration, here are the claims:\n" +
        '{"items": ["claim-1", "claim-2"]}\n\n' +
        "These claims cover the key aspects of the requirement."
    );
  }
}

// Pick LLM based on strategy — in production, this maps to real models.
// PARITY: Python `def demo_factory(tier, node_name=None, llm_config=None)` uses keyword args.
// TS has no kwargs — the factory must take an OPTIONS OBJECT for the extra fields. The proposed
// matrix only documents `llmFactory(tier)`; the real neograph factory contract passes node_name +
// llm_config, so the TS signature widens to `(tier, opts)`. Small friction, but the shape changes.
type FactoryOpts = { nodeName?: string; llmConfig?: Record<string, unknown> };
function demoFactory(_tier: string, opts: FactoryOpts = {}) {
  const strategy = (opts.llmConfig?.["output_strategy"] as string) ?? "structured";
  console.log(`  [${opts.nodeName}] strategy=${strategy}`);

  if (strategy === "structured") return new CleanLLM();
  if (strategy === "json_mode") return new FencedJsonLLM();
  return new VerboseTextLLM();
}

// ══════════════════════════════════════════════════════════════════════════
// THREE STRATEGIES — same schema, different models, same result
// ══════════════════════════════════════════════════════════════════════════

// Strategy 1: structured (default) — model supports withStructuredOutput.
// PARITY: Python `@node(outputs=Claims, model="fast", prompt="extract") def structured() -> Claims: ...`
// becomes the wrapper form. The body is DEAD in think mode (LLM executes via prompt=). In Python the
// dead `...` body triggers an AST-based dead-body UserWarning that reassures you the body is intentional.
// TS has no runtime AST warning (matrix: "Skip (or ESLint plugin)"), so the empty `=> {}` below is
// silent — the safety cue that tells a reader "this body is deliberately unused" is gone.
// PARITY: transformer extracts inputs={} (no params) and output=Claims from the annotated arrow.
// Because there are NO params, there are NO edges — this is a disconnected ROOT node. Fine, but the
// arrow's `(): Claims => {}` return type is the ONLY signal the transformer keys on; if a dev omits the
// `: Claims` return annotation the transformer cannot recover outputs (Python has the same requirement
// via the return annotation, so this is parity, not regression).
const structured = node(
  { model: "fast", prompt: "extract", outputs: Claims },
  (): Claims => {
    // body unused for think mode — no output_strategy => defaults to "structured"
    return undefined as unknown as Claims;
  }
);
// PARITY (redundancy): `outputs: Claims` is passed EXPLICITLY in config above even though the
// transformer also reads the `: Claims` return type. In Python the return annotation is the single
// source. Here we either (a) trust the transformer (drop `outputs`) or (b) pass it explicitly as a
// runtime-available Zod value. The transformer emits a *schema reference*, but the config object needs a
// real runtime Zod value for compile()/validation — so in practice you pass `outputs: Claims` anyway,
// and the "signature IS the DAG" promise is only half-kept for the OUTPUT of a param-less node.

// Strategy 2: json_mode — model returns JSON in fences, framework parses.
const jsonMode = node(
  {
    model: "fast",
    prompt: "extract",
    outputs: Claims,
    llmConfig: { output_strategy: "json_mode" }, // PARITY: llm_config -> llmConfig; value dict is Direct.
    name: "json-mode", // PARITY: name override is a Direct config field.
  },
  (): Claims => {
    return undefined as unknown as Claims;
  }
);

// Strategy 3: text — model returns prose with embedded JSON, framework extracts.
const textMode = node(
  {
    model: "fast",
    prompt: "extract",
    outputs: Claims,
    llmConfig: { output_strategy: "text" },
    name: "text-mode",
  },
  (): Claims => {
    return undefined as unknown as Claims;
  }
);

// ── Assemble ─────────────────────────────────────────────────────────────
// PARITY (REDESIGN — the headline gap for THIS example):
// Python: `construct_from_module(sys.modules[__name__], name="output-strategies")` introspects the
// MODULE and auto-collects every @node in it. TS has NO runtime module introspection; the design doc
// explicitly lists construct_from_module under "Not in v0.1.0-ts". So the three siblings must be
// ENUMERATED BY HAND. The entire narrative of example 08 — "declare three nodes, the module collects
// them for you" — degrades to a manual array. Forget to add a fourth node to the list and it silently
// vanishes from the pipeline (in Python it would be auto-discovered). This is the biggest DX loss here.
const pipeline = constructFromFunctions([structured, jsonMode, textMode], {
  name: "output-strategies",
});

// ══════════════════════════════════════════════════════════════════════════
// RUN — all three produce identical results
// ══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log("Output strategy comparison:\n");

  const graph = compile(pipeline, {
    llmFactory: demoFactory,
    // PARITY: Python `lambda template, data, **kw: [...]`. The `**kw` catch-all is load-bearing:
    // neograph introspects the compiler's SIGNATURE (`_accepted_params`) to decide whether to pass
    // gated kwargs like di_inputs. In TS, runtime function-parameter introspection is unreliable
    // (arrow funcs, minification, no `inspect.signature`), so the di_inputs OPT-IN mechanism can't be
    // signature-gated the Python way. TS would need an explicit opt-in flag or an options-bag param.
    // This example's compiler ignores kwargs, so it's moot HERE, but the seam is a real API gap.
    promptCompiler: (_template, _data, ..._kw) => [
      { role: "user", content: "analyze" },
    ],
  });

  // PARITY: Python `run(graph, input={"node_id": "test"})` — `node_id` is unconsumed seed state
  // (no FromInput binds it). Direct: pass the same seed dict.
  const result = await run(graph, { input: { node_id: "test" } });

  // PARITY: result is keyed by node STATE FIELD name. Note the two `name=`-overridden nodes are
  // "json-mode"/"text-mode" yet Python reads `result["json_mode"]`/`result["text_mode"]` — neograph
  // normalizes hyphen->underscore for the state field. TS must replicate that normalization exactly
  // or these lookups miss. Same-logic port, but an easy place to silently diverge.
  for (const field of ["structured", "json_mode", "text_mode"] as const) {
    console.log(`  Result: ${result[field].items}\n`);
  }

  console.log("All three strategies produce the same parsed output.");
  console.log("The consumer writes zero parsing code — NeoGraph handles it.");
}

main();
