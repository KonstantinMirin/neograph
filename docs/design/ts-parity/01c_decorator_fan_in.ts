// Port of: examples/01c_decorator_fan_in.py
// Feature under test: @node scripted fan-in + construct_from_module topo-sort.
//
// Python's headline DX here is TWO things at once:
//   (1) a consumer's PARAMETER NAMES resolve to upstream nodes (auto-wiring), and
//   (2) construct_from_module() collects every @node in the module and topo-sorts
//       them, so nodes can be DECLARED OUT OF ORDER with NO nodes=[...] list.
//
// This sketch targets the PROPOSED API in docs/design/typescript-port.md (AD-0
// transformer form). It is illustrative, NOT compilable. // PARITY: notes mark
// every place the TS DX diverges from the Python.

import { node, compile, run, constructFromFunctions } from "@neograph/core";
import { z } from "zod";

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY [mostly-direct]: Python `class Claims(BaseModel, frozen=True)` is a
// runtime class you both CONSTRUCT (`Claims(items=[...])`) and get attribute
// access on (`x.items`). Zod gives you a validator + an inferred *type*, but no
// constructor and no instance identity. You build plain object literals and,
// for the `frozen=True` guarantee, must opt into `.readonly()` (compile-time
// only) — there is no runtime Object.freeze unless you call it yourself.
const Claims = z.object({ items: z.array(z.string()) }).readonly();
type Claims = z.infer<typeof Claims>;

const Scores = z.object({ ratings: z.record(z.string(), z.number()) }).readonly();
type Scores = z.infer<typeof Scores>;

const Verification = z.object({
  passed: z.array(z.string()),
  failed: z.array(z.string()),
}).readonly();
type Verification = z.infer<typeof Verification>;

const Metadata = z.object({ source: z.string(), version: z.string() }).readonly();
type Metadata = z.infer<typeof Metadata>;

const Report = z.object({ summary: z.string() }).readonly();
type Report = z.infer<typeof Report>;

// ── Nodes — deliberately out of order to prove topo-sort works ──────────
// PARITY: `report` is defined FIRST, before its dependencies, exactly like the
// Python. Ordering within the list handed to constructFromFunctions() below is
// still topo-sorted, so this half of the example's point survives. The OTHER
// half — "no nodes=[...] list at all" — does NOT (see the pipeline build).

// PARITY [redesign — node NAME derivation]: in Python the node's name is
// `fn.__name__`, intrinsic to the function. Arrow functions are anonymous, so
// the AD-0 transformer must reach OUT of the `node(...)` call to the enclosing
// `const report =` binding to name the node "report". That name is what other
// nodes' parameters resolve against. Node identity is therefore coupled to the
// variable name via transformer magic, where Python gets it for free.
export const report = node({ mode: "scripted", outputs: Report },
  // PARITY [direct, transformer-dependent]: fan-in. The four positional param
  // NAMES — fetch_claims, score_claims, verify_claims, gather_metadata — are
  // the edges. The AD-0 transformer extracts (name, type) per positional param
  // and emits __neo_meta.inputs, reproducing `inspect.signature`. Without the
  // transformer you fall to the explicit-inputs form (see FALLBACK block).
  (fetch_claims: Claims, score_claims: Scores,
   verify_claims: Verification, gather_metadata: Metadata): Report => {
    // PARITY [direct]: body logic is a 1:1 syntax translation.
    const passed = verify_claims.passed.join(", ") || "none";
    const failed = verify_claims.failed.join(", ") || "none";
    const vals = Object.values(score_claims.ratings);
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    return {
      summary: [
        `Source: ${gather_metadata.source} v${gather_metadata.version}`,
        `Claims analysed: ${fetch_claims.items.length}`,
        `Average score: ${avg.toFixed(1)}`,
        `Passed: ${passed}`,
        `Failed: ${failed}`,
      ].join("\n"),
    };
    // PARITY [api-gap — scripted return validation]: Python validates the
    // returned Report against the Pydantic model on the way out of the node. A
    // TS callback can return any object literal that structurally satisfies
    // `Report`; nothing runs `Report.parse(...)` unless the factory does it.
    // withStructuredOutput() only guards LLM nodes, not scripted ones, so the
    // runtime output-validation guarantee is an open question for scripted @node.
  }
);

export const verify_claims = node({ mode: "scripted", outputs: Verification },
  (fetch_claims: Claims, score_claims: Scores): Verification => {
    const passed = fetch_claims.items.filter((c) => (score_claims.ratings[c] ?? 0) >= 0.5);
    const failed = fetch_claims.items.filter((c) => (score_claims.ratings[c] ?? 0) < 0.5);
    return { passed, failed };
  }
);

// PARITY [direct]: zero-parameter node = a source. No inputs to extract.
export const gather_metadata = node({ mode: "scripted", outputs: Metadata },
  (): Metadata => ({ source: "requirements-doc", version: "2.1" })
);

export const score_claims = node({ mode: "scripted", outputs: Scores },
  (fetch_claims: Claims): Scores => {
    const ratings: Record<string, number> = {};
    for (const c of fetch_claims.items) {
      ratings[c] = c.toLowerCase().includes("shall") ? 0.8 : 0.3;
    }
    return { ratings };
  }
);

export const fetch_claims = node({ mode: "scripted", outputs: Claims },
  (): Claims => ({
    items: [
      "The system shall log all access attempts",
      "The system shall validate input",
      "Nice to have: dark mode",
    ],
  })
);

// ── Build pipeline ──────────────────────────────────────────────────────
// PARITY [redesign — the example's headline is only HALF portable]:
// Python: `construct_from_module(sys.modules[__name__], name="...")` reflects
// over the module, discovers every @node, and needs NO explicit list.
// typescript-port.md lists construct_from_module under "Not in v0.1.0-ts"
// ("no module introspection in TS; use explicit lists"). So you must ENUMERATE
// the nodes. Topo-sort of that list survives (order below is still shuffled);
// the "no nodes=[...] list" DX does not.
const pipeline = constructFromFunctions("requirements-review", [
  report, verify_claims, gather_metadata, score_claims, fetch_claims,
]);

// PARITY [direct]: compile + run. `node_id` is run input; no node consumes it
// here, so it is passed through untouched exactly as in Python.
const graph = compile(pipeline);
const result = await run(graph, { input: { node_id: "review-001" } });
console.log("=== Requirements Review Report ===");
console.log(result.report.summary);

// ── FALLBACK (no AD-0 transformer) ──────────────────────────────────────
// Without the compiler transformer, param-name-as-edge is lost. You declare
// inputs explicitly and destructure. Note the DOUBLE authoring cost: the edge
// name "fetch_claims" now appears BOTH as an inputs key AND in the destructure,
// and the schema is repeated — the redundancy the @node decorator exists to
// remove.
//
// export const reportFallback = node(
//   {
//     mode: "scripted",
//     inputs: {
//       fetch_claims: Claims, score_claims: Scores,
//       verify_claims: Verification, gather_metadata: Metadata,
//     },
//     outputs: Report,
//   },
//   ({ fetch_claims, score_claims, verify_claims, gather_metadata }) => { ... }
// );
