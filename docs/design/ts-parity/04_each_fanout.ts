// TS parity sketch of: examples/04_each_fanout.py
// HYPOTHETICAL — written against the PROPOSED neograph-ts API in
// docs/design/typescript-port.md (AD-0 transformer form). NOT runnable; no TS
// implementation exists. `// PARITY:` notes flag where the TS DX diverges from
// the Python original.
//
// Python features this example exercises:
//   - @node(outputs=Clusters)                        -> source/scripted node, zero params
//   - @node(outputs=VerifyResult, map_over=..., map_key=...)  -> Each fan-out via decorator sugar
//   - construct_from_module(sys.modules[__name__])   -> module introspection assembly
//   - compile(pipeline) / run(graph, input={...})
//   - frozen Pydantic BaseModels
//   - result["verify"] : dict[str, VerifyResult]     -> Each merge-to-dict output

import { z } from "zod";
import { node, constructFromFunctions, compile, run } from "@neograph/core";

// ── Schemas ────────────────────────────────────────────────────────────────
// PARITY: Python's `class Clusters(BaseModel, frozen=True)` becomes a Zod schema
// + inferred type. `frozen=True` has NO direct Zod equivalent — Zod validates
// but does not freeze the parsed object. Closest is `z.readonly()` (type-level
// `Readonly<T>` only) or `Object.freeze` in a `.transform()`. The Python model
// is hashable *because* it's frozen (Each keys/dedups on it); TS loses the
// "hashable value object" guarantee. LOW severity here (we key by a string
// field, not by object identity), but it is a real semantic gap.

const ClusterGroup = z.object({
  label: z.string(),
  claimIds: z.array(z.string()),
}).readonly();
type ClusterGroup = z.infer<typeof ClusterGroup>;

const Clusters = z.object({
  groups: z.array(ClusterGroup),
}).readonly();
type Clusters = z.infer<typeof Clusters>;

const VerifyResult = z.object({
  clusterLabel: z.string(),
  coveragePct: z.number().int(),
  gaps: z.array(z.string()),
}).readonly();
type VerifyResult = z.infer<typeof VerifyResult>;

// ── Nodes ──────────────────────────────────────────────────────────────────

// PARITY: `@node(outputs=Clusters) def discover_clusters() -> Clusters` is a
// ZERO-parameter source node. The AD-0 transformer reads the return annotation
// (`: Clusters`) and emits `{ inputs: {}, output: Clusters }`. This ports
// DIRECT — the wrapper's callback is a plain `() => Clusters`.
//
// One wrinkle: the transformer must map the return *type* `Clusters` back to the
// runtime *schema value* `ClustersSchema`. In Python the annotation IS the class
// (usable at runtime). In TS the type and the Zod value are two separate
// symbols; the transformer has to resolve `Clusters` (type) -> `Clusters` (const)
// by name convention. If resolution fails, you fall back to explicit
// `node({ outputs: Clusters }, () => ...)`.
const discoverClusters = node(
  {}, // mode inferred: no prompt/model -> scripted
  (): Clusters => ({
    groups: [
      { label: "authentication", claimIds: ["REQ-1", "REQ-2", "REQ-3"] },
      { label: "logging", claimIds: ["REQ-4", "REQ-5"] },
      { label: "performance", claimIds: ["REQ-6"] },
    ],
  }),
);

// PARITY: the Each fan-out. Python infers three things the TS API must state or
// re-derive:
//   1. map_over="discover_clusters.groups"  -> WHICH producer field to fan over.
//      Kept as an opaque string path. DIRECT, but see API GAP #1: the string is
//      unchecked against `discoverClusters`'s output shape at author time — the
//      transformer sees a string literal, not a symbol, so a typo
//      ("discover_clusters.group") only fails at compile()/validation time, same
//      as Python. No stronger typing than Python here.
//   2. map_key="label"  -> which field of the ELEMENT keys the result dict.
//      DIRECT as a string; Each.key existence is validated via Zod `.shape`
//      introspection (matrix row "Each.key field existence": Direct).
//   3. The fan-out RECEIVER parameter. In Python, `cluster: ClusterGroup` is
//      recognized as the per-item receiver (reads neo_each_item, not an upstream
//      named "cluster") because map_over is set and it's the sole non-DI param
//      — `Node.fan_out_param`. See API GAP #2: with the transformer we can
//      replicate "sole non-DI param is the receiver", but the moment a fan-out
//      node ALSO consumes a peer upstream, TS has no annotation to disambiguate
//      which param is the item vs which is the upstream. Python has the same
//      ambiguity but resolves it by type-matching the map_over element type;
//      the TS transformer would need the element schema at build time to do the
//      same, which crosses the type/value boundary again.
const verify = node(
  {
    mapOver: "discoverClusters.groups", // PARITY: camelCased node name; path is still a stringly-typed reference
    mapKey: "label",
    outputs: VerifyResult, // PARITY: given explicitly because return-type->schema resolution for a fanned node is the least-safe transformer path; see note on discoverClusters
  },
  (cluster: ClusterGroup): VerifyResult => {
    const coverage: Record<string, number> = {
      authentication: 85,
      logging: 60,
      performance: 100,
    };
    const gapsMap: Record<string, string[]> = {
      authentication: ["MFA not implemented"],
      logging: ["no structured logging", "missing audit trail"],
      performance: [],
    };
    return {
      clusterLabel: cluster.label,
      coveragePct: coverage[cluster.label] ?? 0,
      gaps: gapsMap[cluster.label] ?? ["unknown"],
    };
  },
);

// ── Build pipeline ───────────────────────────────────────────────────────────

// PARITY (REDESIGN): Python's `construct_from_module(sys.modules[__name__])`
// auto-collects every @node in the module. TS has NO module-member
// introspection, and the design doc explicitly drops construct_from_module from
// v0.1.0-ts ("no module introspection in TS; use explicit lists"). So the
// one-liner becomes a hand-maintained array. Clean path, but the DX regresses:
// adding a node means also remembering to append it here (Python never had that
// footgun). MEDIUM severity — it removes a headline convenience of this example.
const pipeline = constructFromFunctions([discoverClusters, verify]);

// ── Run ──────────────────────────────────────────────────────────────────────

const graph = compile(pipeline);
const result = await run(graph, { input: { nodeId: "analysis-001" } });
// PARITY: Python `run(...)` is sync; TS run() is async (LangGraph.js invoke is
// Promise-based). Top-level await or an async main wrapper is required. LOW.

// PARITY: `result["verify"]` is `dict[str, VerifyResult]` in Python (Each merges
// to a keyed dict). In TS it's `Record<string, VerifyResult>`. But the run()
// return type is a loose bag keyed by node name — TS cannot statically type
// `result.verify` as `Record<string, VerifyResult>` without the transformer also
// synthesizing a per-pipeline output interface. Likely `result.verify` is typed
// `unknown`/`any` and you re-parse. GAP #3.
const verifyResults = result.verify as Record<string, VerifyResult>;

console.log(`Verified ${Object.keys(verifyResults).length} clusters:\n`);
for (const [label, vr] of Object.entries(verifyResults)) {
  const status = vr.coveragePct >= 80 ? "PASS" : "GAPS";
  console.log(`  [${status}] ${label}: ${vr.coveragePct}% coverage`);
  for (const gap of vr.gaps) {
    console.log(`         - ${gap}`);
  }
}
