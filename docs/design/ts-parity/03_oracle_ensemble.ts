// TS parity sketch of: examples/03_oracle_ensemble.py
// Feature: Oracle modifier (ensemble) — N parallel generators + merge.
// Merge variants: (1) scripted merge_fn, (2) multi-model ensemble, (3) LLM merge_prompt + hooks.
//
// This is a HYPOTHETICAL sketch against the PROPOSED API in
// docs/design/typescript-port.md (AD-0 transformer form). It is NOT meant to
// compile or run — it exists to expose where the TS DX diverges from Python.
// Every divergence is flagged inline with `// PARITY:`.

import { z } from "zod";
import {
  node,
  mergeFn,
  Node,
  Oracle,
  Construct,
  compile,
  run,
} from "@neograph/core";

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY [DIRECT-ish]: Python `class Topic(BaseModel, frozen=True)` becomes a
// Zod schema + inferred type. The `frozen=True` immutability contract has no
// runtime Zod equivalent — `z.readonly()` only affects the *inferred TS type*,
// not runtime mutation. Python's frozen BaseModel actually raises on mutation.
const Topic = z.object({ text: z.string() }).readonly();
type Topic = z.infer<typeof Topic>;

const Claims = z.object({ items: z.array(z.string()) }).readonly();
type Claims = z.infer<typeof Claims>;

// ── Generator state: rotate perspectives so each parallel generator differs ──
const _perspectives: string[][] = [
  ["security: must authenticate", "security: must encrypt"],
  ["reliability: must handle failures", "reliability: must log errors"],
  ["performance: must respond in 200ms", "security: must authenticate"],
];

// PARITY [REDESIGN]: Python uses `threading.Lock` + a mutable `[0]` cell because
// the Oracle fan-out runs the 3 generators on real OS threads. LangGraph.js
// fans out with `Send` on a SINGLE-THREADED event loop, so there is no data
// race and the lock is meaningless — a plain closure counter suffices. BUT the
// determinism guarantee also weakens: Python's lock made each generator grab a
// distinct index atomically; in JS the interleaving of `await`-suspended
// generators still yields distinct indices ONLY because each call is synchronous
// up to its return. If a generator ever awaits before reading the counter, two
// could read the same index. The example's "each generator gets a different
// perspective" invariant is thread-luck in Python and event-loop-luck in JS —
// neither is actually guaranteed by the framework. (Same latent smell both ways.)
let _genCounter = 0;

// ── Merge: combine and deduplicate claims from all generators ────────────
// PARITY [REDESIGN]: Python's merge_fn is a bare module-level function that the
// Oracle references BY STRING NAME (`merge_fn="merge_claims"`) and the compiler
// resolves via `scripted={"merge_claims": merge_claims}`. In TS there is no
// module-symbol table to scan, so the merge fn must self-register its name. The
// `mergeFn(...)` wrapper is the proposed analogue of Python's `@merge_fn`
// decorator — but note THIS example never used `@merge_fn`; it used a plain
// function + the compile-time `scripted={}` dict. That plain-function + string
// path has NO clean TS equivalent (see API GAP note at bottom).
//
// PARITY [DIRECT]: the (variants, config) signature ports 1:1. `variants` is
// list[Claims]; config is the runtime config bag.
const mergeClaims = mergeFn("merge_claims",
  (variants: Claims[], _config: RunConfig): Claims => {
    const seen = new Set<string>();
    const merged: string[] = [];
    for (const variant of variants) {
      for (const claim of variant.items) {
        if (!seen.has(claim)) {
          seen.add(claim);
          merged.push(claim);
        }
      }
    }
    return { items: merged };
  },
);

// ── Build pipeline (surface 1: node() wrapper w/ Oracle kwargs) ──────────
// Python: @node(outputs=Claims, ensemble_n=3, merge_fn="merge_claims")
//         def decompose() -> Claims: ...
//
// PARITY [REDESIGN]: `decompose` takes NO parameters, so the AD-0 transformer
// extracts `inputs: {}` and `output: Claims` from the return annotation. That
// part is clean. BUT `merge_fn: "merge_claims"` is a STRING that must resolve to
// `mergeClaims` — the transformer cannot follow a string literal to a value, so
// the registration is a pure runtime concern (handled by `mergeFn(...)` above
// self-registering into a module Map). The ensembleN/mergeFn kwargs port 1:1
// per the parity matrix ("Ensemble (ensemble_n): Direct").
const decompose = node(
  { outputs: Claims, ensembleN: 3, mergeFn: "merge_claims" },
  (): Claims => {
    const idx = _genCounter % _perspectives.length;
    _genCounter += 1;
    return { items: _perspectives[idx] };
  },
);

// PARITY [BLOCKED → REDESIGN]: Python does
//   pipeline = construct_from_module(sys.modules[__name__], name="oracle-demo")
// which SCANS the module namespace for every @node-decorated function and wires
// them into a Construct. TS has NO module introspection (typescript-port.md,
// "Not in v0.1.0-ts: construct_from_module — no module introspection in TS; use
// explicit lists"). So the single most ergonomic assembly call in this example
// has no TS analogue — you MUST hand-enumerate the nodes. This is the biggest DX
// regression this example surfaces: the "define functions, let the framework
// discover them" story is gone.
const pipeline = new Construct("oracle-demo", { nodes: [decompose] });

// ── Run: same-model ensemble ─────────────────────────────────────────────
async function main() {
  _genCounter = 0; // reset for clean run

  // PARITY [REDESIGN]: `compile(pipeline, scripted={"merge_claims": merge_claims})`.
  // In Python `scripted=` is a name→fn dict passed at compile. Here the fn
  // already self-registered via `mergeFn("merge_claims", ...)`, so the explicit
  // `scripted` map is redundant/absent — the registry IS the transport. Shown
  // for fidelity but the name-keying moved from a compile arg to a module Map.
  const graph = compile(pipeline, { scripted: { merge_claims: mergeClaims } });

  // PARITY [DIRECT]: run(graph, input={node_id}) → run(graph, {input}). LangGraph.js
  // invoke is async, so the whole script is `async`. Python's is sync.
  const result = await run(graph, { input: { node_id: "REQ-001" } });

  const merged = result["decompose"] as Claims;
  console.log("=== Same-model ensemble (3 generators) ===");
  console.log(`${merged.items.length} unique claims:`);
  for (const claim of merged.items) console.log(`  - ${claim}`);
  console.log();

  // ── Multi-model ensemble (surface 2: programmatic Node | Oracle) ────────
  // Python:
  //   gen_node = Node.scripted("multi-gen", fn="multi_model_gen", outputs=Claims)
  //              | Oracle(models=["reason","fast","creative"], merge_fn="pick_best")
  //
  // PARITY [DIRECT]: this is the "TS-first surface" per the matrix. `|` → `.pipe()`.
  const seenModels: string[] = [];

  // PARITY [REDESIGN]: `multi_model_gen(input_data, config)` reads the per-generator
  // model tier from `config["configurable"]["_oracle_model"]` — a framework
  // side-channel key. This is a raw scripted fn (signature (input_data, config)),
  // NOT a signature-typed @node. The transformer gives us nothing here; it is a
  // hand-shaped (state, config) callback registered by name. Ports as a plain
  // registered fn — the `_oracle_model` magic string is identical in both langs.
  const multiModelGen = mergeFn.scripted("multi_model_gen",
    (_input: unknown, config: RunConfig): Claims => {
      const model = config?.configurable?._oracle_model ?? "default";
      seenModels.push(model);
      return { items: [`claim-from-${model}`] };
    },
  );

  const pickBest = mergeFn("pick_best",
    (variants: Claims[], _config: RunConfig): Claims => {
      const allItems: string[] = [];
      for (const v of variants) allItems.push(...v.items);
      return { items: allItems };
    },
  );

  // PARITY [DIRECT]: Node.scripted(...) → Node.scripted(...); Oracle(models=...) →
  // Oracle({models, mergeFn}). `.pipe()` replaces `|`. Mutual-exclusion guards
  // (Oracle+Each etc.) live in `.pipe()` per matrix — same logic.
  const genNode = Node.scripted("multi-gen", { fn: "multi_model_gen", outputs: Claims })
    .pipe(Oracle({ models: ["reason", "fast", "creative"], mergeFn: "pick_best" }));
  const multiPipeline = new Construct("multi-model", { nodes: [genNode] });
  const multiGraph = compile(multiPipeline, {
    scripted: { multi_model_gen: multiModelGen, pick_best: pickBest },
  });
  const multiResult = await run(multiGraph, { input: { node_id: "REQ-002" } });

  const multiMerged = multiResult["multi_gen"] as Claims;
  console.log(`=== Multi-model ensemble (models=[reason,fast,creative]) ===`);
  console.log(`Models used: ${JSON.stringify(seenModels)}`);
  console.log("Merged claims:");
  for (const claim of multiMerged.items) console.log(`  - ${claim}`);
  console.log();

  // ── Merge hooks: pre_process, post_process, fallback ────────────────────
  // PARITY [REDESIGN, mostly-direct]: the three hooks are passed as CALLABLES
  // directly on the Oracle (not string-named). Callbacks port cleanly to TS —
  // the matrix marks "Body-as-merge / callback IS the merge: Direct". The nuance:
  // Python's Oracle stores these via `arbitrary_types_allowed=True` because
  // Callables aren't Pydantic-validatable. In TS a Zod-modeled Oracle would need
  // `z.custom<Function>()` or to keep the callback fields OUT of the Zod schema
  // (plain class props) — the immutable-Zod-model story (matrix: "Oracle: Direct")
  // quietly breaks for function-valued fields.

  // Pre-process: tag each variant with a generator ID → dict for the prompt.
  // PARITY [REDESIGN]: Python returns a `dict` ({"tagged_claims": [...]}) that
  // becomes prompt input_data. TS has no positional-vs-keyword duality; a plain
  // Record works, but the untyped `list` param (`variants: list`) loses its
  // element type in Python too — TS would force `Claims[]` (arguably better).
  const tagVariants = (variants: Claims[]): Record<string, unknown> => {
    const tagged = variants.map((v, i) => ({ gen_id: `gen-${i}`, claims: v.items }));
    return { tagged_claims: tagged };
  };

  // Post-process: ensure every input claim appears in the output.
  const validateMerge = (result: Claims, variants: Claims[]): Claims => {
    const allInput = new Set(variants.flatMap((v) => v.items));
    const missing = [...allInput].filter((c) => !result.items.includes(c)).sort();
    if (missing.length) return { items: [...result.items, ...missing] };
    return result;
  };

  // Fallback: on LLM failure, deduplicate deterministically.
  // PARITY [DIRECT]: (variants, error) → (variants, error). Exception → Error.
  const deterministicFallback = (variants: Claims[], _error: Error): Claims => {
    const seen = new Set<string>();
    const merged: string[] = [];
    for (const v of variants) {
      for (const claim of v.items) {
        if (!seen.has(claim)) {
          seen.add(claim);
          merged.push(claim);
        }
      }
    }
    return { items: merged };
  };

  const hooksGen = mergeFn.scripted("hooks_gen",
    (_input: unknown, _config: RunConfig): Claims => {
      const idx = _genCounter % _perspectives.length;
      _genCounter += 1;
      return { items: _perspectives[idx] };
    },
  );

  _genCounter = 0; // reset
  const hooksNode = Node.scripted("decompose-hooks", { fn: "hooks_gen", outputs: Claims })
    .pipe(Oracle({
      n: 3,
      // PARITY [DIRECT]: inline `${var}` prompt substitution — matrix: Direct.
      mergePrompt: "Pick the best decomposition: ${tagged_claims}",
      mergePreProcess: tagVariants,
      mergePostProcess: validateMerge,
      mergeFallback: deterministicFallback,
    }));
  const hooksPipeline = new Construct("hooks-demo", { nodes: [hooksNode] });

  // PARITY [REDESIGN]: Python passes `llm_factory=lambda tier: None` and a
  // `prompt_compiler=lambda tmpl, data, **kw: [...]` to force the LLM path to
  // fail so the fallback fires. TS has no `**kwargs`; the prompt_compiler's
  // variadic tail must be modeled as an explicit options object / rest param.
  // `llm_factory(tier) -> None` returning null to signal "no LLM" is a sentinel
  // hack that's identical in both — but TS's stricter nullability makes the
  // downstream `.invoke()` on a null model a compile error unless the factory's
  // return type is `ChatModel | null`, which then infects every call site.
  const hooksGraph = compile(hooksPipeline, {
    llmFactory: (_tier: string) => null, // no real LLM → invoke fails → fallback
    promptCompiler: (tmpl: string, _data: unknown, _opts?: PromptCompilerOpts) =>
      [{ role: "user", content: tmpl }],
    scripted: { hooks_gen: hooksGen },
  });
  const hooksResult = await run(hooksGraph, { input: { node_id: "REQ-003" } });

  const hooksMerged = hooksResult["decompose_hooks"] as Claims;
  console.log("=== Merge hooks (fallback fires — no LLM configured) ===");
  console.log(`Fallback produced ${hooksMerged.items.length} claims:`);
  for (const claim of hooksMerged.items) console.log(`  - ${claim}`);
}

// ── Ambient types referenced above (would come from @neograph/core) ──────
// PARITY note: `RunConfig` models LangGraph's config bag. Python reads it as an
// untyped dict (`config.get("configurable",{}).get("_oracle_model")`); TS forces
// a shape, and the framework side-channel keys (`_oracle_model`) must be declared
// as optional-any to stay open. This is stricter but leakier.
interface RunConfig {
  configurable?: { _oracle_model?: string;[k: string]: unknown };
}
interface PromptCompilerOpts { [k: string]: unknown }

void main();
