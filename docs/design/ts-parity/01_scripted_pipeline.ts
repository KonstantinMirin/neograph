// TS parity sketch of: examples/01_scripted_pipeline.py
// Scenario: deterministic doc-processing pipeline (extract -> split -> classify),
// pure scripted logic, no LLM, no API keys.
//
// Proposed API source: docs/design/typescript-port.md (AD-0 transformer form).
// This is a HYPOTHETICAL sketch against an API that does NOT exist yet.
// It is not meant to compile or run.

import { node, compile, run } from "@neograph/core";
import { z } from "zod";

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY: Python uses `class RawText(BaseModel, frozen=True)`. TS has no runtime
// type reflection, so schemas are Zod objects (AD-1). The `frozen=True` intent
// (immutable value objects) has no direct Zod equivalent -- Zod validates shape
// but does not freeze the parsed object. Would need `.transform(Object.freeze)`
// or `readonly` TS types layered on top. The transformer (AD-0) reads the *TS
// type*, so we still need BOTH the Zod schema (runtime) and the inferred type
// (compile-time). Python's single `BaseModel` class carries both roles.

const RawText = z.object({ text: z.string() });
type RawText = z.infer<typeof RawText>;

const Claims = z.object({ items: z.array(z.string()) });
type Claims = z.infer<typeof Claims>;

const ClassifiedClaims = z.object({
  // PARITY: Python `list[dict[str, str]]` -> z.array(z.record(...)). Direct.
  classified: z.array(z.record(z.string(), z.string())),
});
type ClassifiedClaims = z.infer<typeof ClassifiedClaims>;

// ── Nodes ────────────────────────────────────────────────────────────────
// PARITY: Python `@node(outputs=RawText)` decorator on a standalone `def`.
// TS decorators cannot decorate standalone functions, so the proposed API uses
// the `node({...}, fn)` wrapper form. The AD-0 transformer extracts the param
// NAMES and TS return type to synthesize `__neo_meta = { inputs, output }`.
//
// PARITY (mode inference): no `model`/`prompt` here => scripted mode. Same rule
// as Python ("neither => scripted"). Direct.
//
// PARITY (output inference): Python infers `outputs=RawText` from the return
// annotation `-> RawText`. Here the transformer must read `(): RawText` return
// type. We pass `{ outputs: RawText }` (the Zod value) explicitly anyway because
// the transformer emits the *type* name but the runtime needs the *Zod schema*
// value -- the transformer cannot conjure a runtime value from a type. This is a
// real gap: Python's `outputs=RawText` names one object that is both; TS needs
// the type (for wiring) AND the schema value (for validation/state).

const extract = node({ outputs: RawText }, (): RawText => {
  // Simulate extracting text from a document source.
  return { text: "The system shall log all access attempts. The system shall validate input." };
});

// PARITY (auto-wiring): Python infers the edge `split <- extract` from the
// PARAMETER NAME `extract`. In TS the parameter name only survives to runtime
// via the AD-0 transformer emitting `inputs: { extract: RawTextSchema }`.
// Without the transformer (fallback), you must hand-write `inputs: { extract }`.
// Critically: the wrapper's own callback arg is named `extract` -- the whole DX
// hinges on the transformer preserving that identifier. A minifier/bundler that
// renames params would silently break the DAG unless the transformer runs first.
const split = node({ outputs: Claims }, (extract: RawText): Claims => {
  const sentences = extract.text
    .split(".")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  return { items: sentences };
});

const classify = node({ outputs: ClassifiedClaims }, (split: Claims): ClassifiedClaims => {
  const classified = split.items.map((claim) => {
    const lc = claim.toLowerCase();
    const category = lc.includes("access") || lc.includes("validate") ? "security" : "general";
    return { claim, category };
  });
  return { classified };
});

// ── Build pipeline ───────────────────────────────────────────────────────
// PARITY (BLOCKED): Python does
//   pipeline = construct_from_module(sys.modules[__name__], name="doc-processor")
// which introspects the MODULE, finds every @node, and topo-sorts them. The
// design doc explicitly says construct_from_module is NOT in v0.1.0-ts:
// "no module introspection in TS; use explicit lists". So the zero-maintenance
// "no nodes=[...] list" DX -- the literal headline of this example -- is LOST.
// You must enumerate the nodes by hand:
const pipeline = fromNodes("doc-processor", [extract, split, classify]);
// PARITY: `fromNodes` is a stand-in for the "explicit list" replacement the doc
// implies but does not name. Ordering still comes from the inferred edges
// (transformer metadata), so at least you don't hand-maintain topological order
// -- only the membership list. Adding a 4th node means editing this array, which
// Python never required.

// ── Run ──────────────────────────────────────────────────────────────────
// PARITY: `compile` + `run` map directly (Direct in the matrix).
const graph = compile(pipeline);

// PARITY: Python `run(graph, input={"node_id": "doc-001"})`. The `node_id` input
// is not consumed by any node here (no FromInput param). Direct -- passes through
// config['configurable']. TS: `run(graph, { input: { node_id: "doc-001" } })`.
const result = await run(graph, { input: { node_id: "doc-001" } });

// PARITY: Python `result["classify"]` keys the state bus by node name. TS keeps
// the same string-keyed result. But the return type is untyped (`any`) unless the
// compiler threads per-node output types through -- Python's dict is also untyped
// at the type level, so this is parity-neutral, just not an improvement.
console.log("Claims found:", result["classify"].classified.length);
for (const item of result["classify"].classified) {
  console.log(`  [${item.category}] ${item.claim}`);
}
