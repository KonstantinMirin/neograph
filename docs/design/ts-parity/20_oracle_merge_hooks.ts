// TS parity sketch of: examples/20_oracle_merge_hooks.py
// Feature: Oracle merge hooks on a THINK-mode @node — LLM judge + self-healing.
//   ensemble_n=3 generators -> merge_prompt (LLM judge) -> pre/post/fallback hooks.
//
// This is a HYPOTHETICAL sketch against the PROPOSED API in
// docs/design/typescript-port.md (AD-0 transformer form). It is NOT meant to
// compile or run — it exists to expose where the TS DX diverges from Python.
// Every divergence is flagged inline with `// PARITY:`.
//
// This example differs from 03_oracle_ensemble in three load-bearing ways, so the
// friction it surfaces is NEW:
//   1. The node is THINK mode (prompt= + model=), body is `...` (dead code).
//   2. The merge hooks hang off the `@node(...)` config directly, not `.pipe(Oracle)`.
//   3. Python VALIDATES the hooks' arity + type annotations at assembly time
//      (_validation_modifiers._validate_merge_hooks). See the big note near the hooks.

import { z } from "zod";
import { node, constructFromFunctions, compile, run } from "@neograph/core";

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY [DIRECT-ish]: Python `BaseModel(frozen=True)` -> Zod object + inferred
// type. `Field(description=...)` -> `.describe(...)` (this DOES survive into the
// LLM structured-output schema, so it ports faithfully). `frozen=True` has no
// runtime Zod analogue — `.readonly()` only brands the inferred TS type; Python's
// frozen model raises on mutation at runtime. Also note the "declare twice"
// tax: Python has ONE `ProductDescription` symbol usable as both type and
// constructor; TS needs the schema VALUE and a separate `type` alias.
const ProductBrief = z
  .object({
    name: z.string(),
    features: z.array(z.string()),
  })
  .readonly();
type ProductBrief = z.infer<typeof ProductBrief>;

const ProductDescription = z
  .object({
    headline: z.string().describe("One-line attention grabber"),
    body: z.string().describe("2-3 sentence description"),
    cta: z.string().describe("Call to action"),
  })
  .readonly();
type ProductDescription = z.infer<typeof ProductDescription>;

// ── Fake LLM (replace with real models in production) ─────────────────────
// PARITY [REDESIGN]: Python's `with_structured_output(model)` stashes the Pydantic
// CLASS and later calls it as a constructor: `self._model(headline=..., ...)`.
// In TS the "model" is a Zod SCHEMA, not a constructor — you build a plain object
// and (optionally) `schema.parse(obj)`. So the fake returns object literals, not
// `new Model(...)`. Minor but pervasive: every place Python treats a model class
// as a callable factory becomes "plain object + schema.parse" in TS.
let _genCounter = 0;
const _HEADLINES = [
  "Transform Your Workflow",
  "Build Faster, Ship Safer",
  "The Future of Automation",
];

class FakeProductLLM {
  private model?: z.ZodType;
  constructor(private tier: string) {}

  withStructuredOutput(model: z.ZodType): this {
    this.model = model;
    return this;
  }

  invoke(_messages: unknown[]): ProductDescription {
    if (this.tier === "reason") {
      // Judge merge: pick the best variant.
      return {
        headline: "Build Faster, Ship Safer",
        body:
          "NeoGraph turns typed Python functions into production agents. " +
          "Declare the logic, we handle the wiring.",
        cta: "Get started in 5 minutes",
      };
    }
    // Generator: produce a variant. `idx % 2 === 0` gives some variants no CTA.
    const idx = _genCounter % _HEADLINES.length;
    _genCounter += 1;
    return {
      headline: _HEADLINES[idx],
      body: `Variant ${idx}: A great product for everyone.`,
      cta: idx % 2 === 0 ? "Try it now" : "",
    };
  }
}

// ── Hooks: the only domain logic you write ────────────────────────────────
//
// PARITY [REDESIGN + API GAP — the headline finding for this example]:
// These three functions are STANDALONE callbacks passed as config values. Two
// distinct things Python does with them have NO clean TS path:
//
// (a) RUNTIME CALL — DIRECT. Python calls them POSITIONALLY:
//       merge_pre_process(variants)             (_oracle.py:307)
//       merge_post_process(merged, variants)    (_oracle.py:319)
//       merge_fallback(variants, exc)           (_oracle.py:337)
//     Positional callbacks port 1:1. This part is fine in TS.
//
// (b) BUILD-TIME VALIDATION — BLOCKED. Python's `_validate_merge_hooks`
//     (_validation_modifiers.py:107-214) runs `inspect.signature(hook)` AND
//     `get_type_hints(hook)` on each bare function to check, at assembly time:
//       - arity (pre=1, post=2, fallback=2 required positional params),
//       - that the `variants` param is annotated `list[<gen_type>]`,
//       - that post/fallback RETURN the node's output type.
//     The AD-0 transformer only instruments `node(config, fn)` call sites — it
//     does NOT see functions passed as plain values inside a config object. So a
//     TS `mergeFallback` with the wrong arity or a `variants: string[]` typo is
//     caught by NEITHER the transformer NOR runtime-until-the-fallback-fires.
//     Python surfaces it at `compile()`. This is a genuine compile-time-safety
//     regression specific to hook-valued config fields.
//
// PARITY [DIRECT]: `-> dict` return is untyped in Python; TS `Record<string,unknown>`
// is arguably tighter. The returned key `numbered_variants` must string-match the
// `${numbered_variants}` template var below — NOTHING checks that linkage
// statically in either language (Python's `lint()` template-placeholder check
// COULD, but the TS design doesn't wire hook-produced template vars into lint).
function tagVariants(variants: ProductDescription[]): Record<string, unknown> {
  const tagged = variants.map(
    (v, i) =>
      `[Variant ${i + 1}]\n` +
      `  Headline: ${v.headline}\n` +
      `  Body: ${v.body}\n` +
      `  CTA: ${v.cta || "(missing)"}`,
  );
  return { numbered_variants: tagged.join("\n\n") };
}

// PARITY [DIRECT]: reconstruct-with-override. Python builds a fresh frozen
// `ProductDescription(...)`; TS uses object spread. Because the model is
// `.readonly()`, the spread is the idiomatic "copy with change".
function ensureCta(
  result: ProductDescription,
  variants: ProductDescription[],
): ProductDescription {
  if (!result.cta) {
    for (const v of variants) {
      if (v.cta) return { ...result, cta: v.cta };
    }
  }
  return result;
}

// PARITY [DIRECT]: `max(variants, key=lambda v: len(v.body))` -> reduce. Python
// `Exception` -> TS `Error`. The fallback signature ports 1:1.
function fallbackPickLongest(
  variants: ProductDescription[],
  _error: Error,
): ProductDescription {
  return variants.reduce((a, b) => (b.body.length > a.body.length ? b : a));
}

// ── The pipeline: 10 lines ────────────────────────────────────────────────
// Python:
//   @node(outputs=ProductDescription,
//         prompt="...", model="fast", ensemble_n=3,
//         merge_prompt="...${numbered_variants}",
//         merge_pre_process=tag_variants,
//         merge_post_process=ensure_cta,
//         merge_fallback=fallback_pick_longest)
//   def write_description() -> ProductDescription: ...
//
// PARITY [REDESIGN]: THINK-mode leaf node. Two frictions:
//   1. DEAD BODY. The Python body is `...` because prompt+model => the LLM runs,
//      not the body. The `node(config, fn)` wrapper form REQUIRES a callback, so
//      TS must pass a dummy `() => { throw ... }` that can never run. Python's `...`
//      is cleaner. (A pure-programmatic `Node({...})` form would avoid the dummy,
//      but then this leaf loses nothing since it has no inputs to auto-wire.)
//   2. NO SIGNATURE TO EXTRACT. `write_description()` takes no params and has a
//      `...` body, so the AD-0 transformer extracts `inputs: {}` and CANNOT infer
//      the output (no `return`). `outputs: ProductDescription` MUST be explicit —
//      the "signature IS the DAG" story gives nothing here; it degrades to the
//      fallback explicit-schema form for output.
//
// PARITY [API GAP — parity matrix omission]: the @node feature table in
// typescript-port.md lists `ensemble_n` (Ensemble: Direct) but lists NONE of
// `merge_prompt` / `merge_pre_process` / `merge_post_process` / `merge_fallback`.
// This example is BUILT on that quartet. They aren't hard to add (callbacks +
// a string), but the design doc has not committed to the field names, to how the
// callback fields coexist with an "immutable Zod-modeled Node" (function-valued
// fields can't live in a Zod schema — see 03's note), or to the build-time hook
// validation above. The matrix needs a row per hook.
const writeDescription = node(
  {
    outputs: ProductDescription,
    prompt: "Write a product description for a developer tool",
    model: "fast",
    ensembleN: 3,
    mergePrompt: "Pick the best product description:\n${numbered_variants}",
    mergePreProcess: tagVariants,
    mergePostProcess: ensureCta,
    mergeFallback: fallbackPickLongest,
  },
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  (): ProductDescription => {
    // PARITY: dead body — think mode never invokes it. Python writes `...`.
    throw new Error("think mode: body never runs");
  },
);

// PARITY [BLOCKED -> REDESIGN]: Python does
//   pipeline = construct_from_module(sys.modules[__name__], name="product-copy")
// which scans THIS module for every @node and wires them. TS has no module
// introspection (typescript-port.md: "Not in v0.1.0-ts: construct_from_module").
// You hand-enumerate. For a one-node module the cost is tiny, but the "declare
// functions, let the framework discover them" ergonomic is simply gone.
const pipeline = constructFromFunctions([writeDescription], {
  name: "product-copy",
});

// ── Run ───────────────────────────────────────────────────────────────────
async function main() {
  _genCounter = 0;

  // PARITY [REDESIGN]: `llm_factory=lambda tier: FakeProductLLM(tier)` ports as a
  // closure — DIRECT. BUT `prompt_compiler=lambda tmpl, data, **kw: [...]` uses
  // Python `**kw` to swallow framework-injected extras (e.g. di_inputs). TS has no
  // `**kwargs`; the variadic tail must become an explicit optional options param,
  // and the introspection-gated `di_inputs` opt-in (Python inspects the compiler's
  // signature to decide whether to pass it) has no TS analogue — TS can't inspect
  // a closure's parameter names at runtime. So the di_inputs seam either always-
  // passes (loses the gate) or needs an explicit `{ acceptsDiInputs: true }` flag.
  const graph = compile(pipeline, {
    llmFactory: (tier: string) => new FakeProductLLM(tier),
    promptCompiler: (tmpl: string, _data: unknown, _opts?: PromptCompilerOpts) => [
      { role: "user", content: tmpl },
    ],
  });

  // PARITY [DIRECT, but async]: run(graph, input={node_id}) -> run(graph, {input}).
  // LangGraph.js `invoke` returns a Promise, so the whole script is async; Python's
  // run() is synchronous.
  const result = await run(graph, { input: { node_id: "launch-v1" } });

  // PARITY [REDESIGN]: `result["write_description"]` is `unknown` in TS — the state
  // bus is a string-keyed bag and TS can't relate the runtime key to the node's
  // output type without a generated typed accessor. A cast is unavoidable. Python
  // is dynamically typed here anyway, so it's a wash at the source level, but TS
  // users FEEL the missing type where they'd expect the framework to carry it.
  const desc = result["write_description"] as ProductDescription;
  console.log("=== Oracle + LLM Judge + Hooks ===");
  console.log(`  Headline: ${desc.headline}`);
  console.log(`  Body:     ${desc.body}`);
  console.log(`  CTA:      ${desc.cta}`);
  console.log();
  console.log(
    "3 generators, 1 LLM judge, self-healing CTA, fallback on failure.",
  );
  console.log("Zero manual LLM calls. Zero retry logic. Zero schema parsing.");
}

// ── Ambient types referenced above (would come from @neograph/core) ───────
// PARITY: the prompt_compiler's Python `**kw` tail. See the compile() note.
interface PromptCompilerOpts {
  [k: string]: unknown;
}

void main();
