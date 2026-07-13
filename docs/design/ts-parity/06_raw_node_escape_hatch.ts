// TS parity sketch of: examples/06_raw_node_escape_hatch.py
// HYPOTHETICAL — no TS implementation exists. Written against the PROPOSED API in
// docs/design/typescript-port.md (AD-0 transformer, .pipe(), node({...}, fn)).
// NOT meant to compile or run. `// PARITY:` marks where the TS DX diverges.
//
// This example's whole point is the mode='raw' escape hatch: one node written as a
// classic LangGraph (state, config) => update function while the framework wires
// edges/state/observability around it. The raw node also does STATE REFLECTION —
// it scans every state field looking for a value that is an instance of `Claims`.
// That reflection is the sharpest TS friction in the file (see PARITY-3).

import { z } from "zod";
import { node, compile, run } from "@neograph/core";
import type { RunnableConfig } from "@langchain/core/runnables";

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY-1 (Direct-ish): Pydantic `class Claims(BaseModel, frozen=True)` becomes a
// Zod schema + inferred type. `frozen=True` (deep immutability + hashability) has NO
// direct Zod equivalent — Zod validates shape but returns a plain, mutable object.
// Closest is `.readonly()` (compile-time only) or Object.freeze at the boundary.
// The frozen/hashable guarantee Python leans on (models usable as dict keys / set
// members) is simply absent; nothing in THIS example depends on it, so it degrades
// silently rather than breaking.
const Claims = z.object({
  items: z.array(z.string()),
});
type Claims = z.infer<typeof Claims>;

const FilteredClaims = z.object({
  kept: z.array(z.string()),
  dropped: z.array(z.string()),
  reason: z.string(),
});
type FilteredClaims = z.infer<typeof FilteredClaims>;

// ── Scripted: produce initial claims ─────────────────────────────────────
// PARITY-2 (Direct, with transformer): a scripted source node with NO parameters.
// The AD-0 transformer reads the return type `Claims` off the arrow fn and emits
// `{ inputs: {}, output: Claims }`. Mode inference sees no prompt/model → scripted,
// matching Python's `mode="scripted"`. This is the cleanest port in the file.
const extractClaims = node({ mode: "scripted", outputs: Claims }, (): Claims => ({
  items: [
    "system shall authenticate users via SSO",
    "system shall support dark mode",
    "system shall encrypt data at rest",
    "system shall have a cool logo",
    "system shall rate-limit API calls",
  ],
}));

// ── Raw node: custom filtering logic ─────────────────────────────────────
// The escape hatch. Python: `@node(mode="raw", inputs=Claims, outputs=FilteredClaims)`
// over `def filter_non_functional(state, config)`.
//
// PARITY-3 (BLOCKED as written — needs REDESIGN): the raw signature is (state, config),
// NOT (topic: RawText). The transformer CANNOT extract dataflow inputs/outputs from it —
// exactly why the Python example passes explicit `inputs=`/`outputs=`. So raw mode always
// falls back to the explicit-schema form. That part is fine.
//
// The real blocker is the BODY. Python scans the whole state model by TYPE:
//     for field_name in state.__class__.model_fields:
//         val = getattr(state, field_name, None)
//         if isinstance(val, Claims): claims = val; break
// TS has NO runtime equivalent:
//   * `state.__class__.model_fields` — LangGraph.js state is a plain object built by
//     Annotation.Root, not a Pydantic model. There is no model_fields registry to iterate.
//     Closest: `Object.keys(state)` (loses declared-but-unset fields & their types).
//   * `isinstance(val, Claims)` — Zod-inferred types ERASE at runtime. A `Claims` value is
//     an indistinguishable `{ items: string[] }` plain object; there is no class to test
//     against. The nearest analog is a structural probe `Claims.safeParse(val).success`,
//     but that's a FALSE-POSITIVE magnet: any `{ items: string[] }` (or a superset) passes,
//     so it can't distinguish Claims from another same-shaped model the way isinstance can.
//
// Below is the least-bad redesign: probe each field structurally with safeParse and take
// the first match. It changes the semantics (nominal → structural) and is noted loudly.
const filterNonFunctional = node(
  { mode: "raw", inputs: { claims: Claims }, outputs: FilteredClaims },
  // PARITY: (state, config) raw signature — Direct shape in LangGraph.js.
  (state: Record<string, unknown>, _config: RunnableConfig) => {
    // PARITY-3 cont.: structural scan replaces Python's isinstance type scan.
    let claims: Claims | null = null;
    for (const key of Object.keys(state)) {
      const parsed = Claims.safeParse(state[key]);
      if (parsed.success) {
        claims = parsed.data;
        break;
      }
    }

    if (claims === null) {
      // PARITY-4 (Direct): dict-keyed state write. Python returns
      // `{"filter_non_functional": FilteredClaims(...)}`; TS returns the same object
      // keyed by node name. But note the node-name string ("filter_non_functional") is
      // now DUPLICATED as a magic string here vs. the const identifier `filterNonFunctional`
      // — Python's snake_case fn name == state field; camelCase TS breaks that identity,
      // so the raw author must hand-write the exact field key the compiler will use.
      return {
        filter_non_functional: { kept: [], dropped: [], reason: "no claims found" },
      };
    }

    const functionalKeywords = [
      "authenticate", "encrypt", "rate-limit", "validate", "shall log", "authorize",
    ];

    const kept: string[] = [];
    const dropped: string[] = [];
    for (const claim of claims.items) {
      if (functionalKeywords.some((kw) => claim.toLowerCase().includes(kw))) {
        kept.push(claim);
      } else {
        dropped.push(claim);
      }
    }

    return {
      filter_non_functional: {
        kept,
        dropped,
        reason: `kept ${kept.length} functional, dropped ${dropped.length} cosmetic`,
      },
    };
  },
);

// ── Scripted: summarize what was kept ────────────────────────────────────
// PARITY-5 (Redesign / API GAP): Python auto-wires by PARAMETER NAME —
//     def summarize(filter_non_functional: FilteredClaims) -> Claims
// the param `filter_non_functional` names the upstream node, so the edge is inferred.
// With the AD-0 transformer the param NAME survives, so `(filterNonFunctional: ...)`
// COULD wire — BUT the upstream node's state field is `filter_non_functional`
// (derived from the raw node's dict key / Python fn name), and camelCase param
// `filterNonFunctional` no longer string-matches it. The transformer preserves the
// name but the name is now WRONG. Author must fall back to explicit inputs to bridge
// the snake_case-field ↔ camelCase-param gap. This is the escape hatch's edge tax.
const summarize = node(
  { mode: "scripted", inputs: { filter_non_functional: FilteredClaims }, outputs: Claims },
  ({ filter_non_functional }: { filter_non_functional: FilteredClaims }): Claims => ({
    items: filter_non_functional.kept,
  }),
);

// ── Build pipeline ───────────────────────────────────────────────────────
// PARITY-6 (BLOCKED — no TS equivalent): Python uses
//     construct_from_module(sys.modules[__name__], name="filter-pipeline")
// which introspects the MODULE, finds every @node, and topo-sorts them. typescript-port.md
// explicitly lists construct_from_module as NOT shipping ("no module introspection in TS;
// use explicit lists"). So the DAG must be assembled from an explicit array. Ordering is
// still inferred from the edges (param names / inputs), not from array order.
const pipeline = compile.constructFromFunctions("filter-pipeline", [
  extractClaims,
  filterNonFunctional,
  summarize,
]);

// ── Run ──────────────────────────────────────────────────────────────────
async function main() {
  const graph = compile(pipeline);
  // PARITY-7 (Direct): run input → config.configurable. `node_id` is unused here
  // (no FromInput consumer), so it's inert context — same as Python.
  const result = await run(graph, { input: { node_id: "REQ-007" } });

  // PARITY-8 (Direct-ish): result is a plain object keyed by node field. Python indexes
  // `result["filter_non_functional"]` and gets a typed FilteredClaims instance; TS gets a
  // plain object — no methods, structurally typed only. Casting/parse needed for full types.
  const filtered = result["filter_non_functional"] as FilteredClaims;
  console.log(`Filter result: ${filtered.reason}\n`);
  console.log("Kept (functional):");
  for (const claim of filtered.kept) console.log(`  + ${claim}`);
  console.log("\nDropped (cosmetic):");
  for (const claim of filtered.dropped) console.log(`  - ${claim}`);
  const summ = result["summarize"] as Claims;
  console.log(`\nFinal claims: ${summ.items.length}`);
}

void main();
