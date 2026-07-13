// TS parity sketch of: examples/05_subgraph_composition.py
// HYPOTHETICAL — targets the PROPOSED @neograph/core TS API (docs/design/typescript-port.md).
// Not compilable/runnable; grounds a feature-parity analysis.
//
// This example lives ENTIRELY on the programmatic/declarative surface
// (Construct + Node.scripted + Oracle + `|` pipe). It uses ZERO @node
// decorators and ZERO signature inference — every node declares its
// inputs/outputs explicitly. So AD-0's ts-patch transformer ("signature IS
// the DAG") is IRRELEVANT to this file: there is no signature to extract.
// The doc rates this surface "TS-first ... maps naturally (Direct)", and that
// holds for structure. The friction is elsewhere: raw scripted-fn typing and
// the string-name + registry indirection. See // PARITY: notes inline.

import { z } from "zod";
import {
  Construct,
  Node,
  Oracle,
  compile,
  run,
  type ScriptedFn,     // PARITY: proposed signature (inputData, config) => output
  type MergeFn,        // PARITY: proposed signature (variants, config) => merged
} from "@neograph/core";

// ── Schemas ──────────────────────────────────────────────────────────────
// PARITY (frozen=True): Python's `BaseModel(frozen=True)` gives immutable,
// hashable value objects. Zod has no frozen concept — `.parse()` returns a
// plain mutable object. Closest is `z....` + `Readonly<T>` at the type level
// (compile-time only) or Object.freeze() at runtime. Neither is enforced by
// the schema the way Pydantic frozen is. LOW severity for this example
// (nothing mutates these), but it's a real semantic gap.

const Claims = z.object({ items: z.array(z.string()) });
type Claims = Readonly<z.infer<typeof Claims>>;

const Context = z.object({ references: z.array(z.string()) });
type Context = Readonly<z.infer<typeof Context>>;

// PARITY (dict[str, str]): Python `list[dict[str, str]]` is a loosely-typed
// bag. `z.record(z.string(), z.string())` is the faithful port, but note the
// element shape {claim, score} is NOT captured by the type — same looseness
// as Python. describe_type/validation stays coarse here (as in Python).
const ScoredClaims = z.object({
  scored: z.array(z.record(z.string(), z.string())),
});
type ScoredClaims = Readonly<z.infer<typeof ScoredClaims>>;

const Report = z.object({ text: z.string() });
type Report = Readonly<z.infer<typeof Report>>;

// ── Parent pipeline functions ────────────────────────────────────────────
// PARITY (raw scripted signature): these are the RAW `(input_data, config)`
// form — the SAME form example 05 uses in Python. The AD-0 transformer only
// targets the `node({...}, fn)` wrapper; it does NOT reach Node.scripted
// callbacks. So `inputData` has NO inferred type here. In Python it's duck-
// typed and "just works" at runtime (input_data.scored resolves dynamically).
// In TS you must pick a poison:
//   (a) `inputData: unknown` + manual cast (shown below), or
//   (b) thread an explicit generic ScriptedFn<TIn, TOut> and repeat the type.
// The Node.scripted `inputs`/`outputs` Zod schemas below DON'T flow into the
// callback param type — there's no transformer binding them. MEDIUM friction:
// the declarative surface loses the static-typing that the @node surface keeps.

const decomposeReq: ScriptedFn = (_inputData, _config): Claims => {
  return { items: ["shall authenticate", "shall log access", "shall encrypt data"] };
};

const formatReport: ScriptedFn = (inputData, _config): Report => {
  const scored = (inputData as ScoredClaims).scored; // PARITY: cast — see note above
  const lines = scored.map((s) => `  ${s.claim}: ${s.score}`);
  return { text: "Coverage Report:\n" + lines.join("\n") };
};

// ── Sub-pipeline: enrich (lookup + score) — internal to the sub-pipeline ──

const lookupContext: ScriptedFn = (_inputData, _config): Context => {
  return { references: ["auth.py:42", "logger.py:18", "crypto.py:7"] };
};

const scoreClaims: ScriptedFn = (inputData, _config): ScoredClaims => {
  // PARITY (neo_subgraph_input): here inputData is the sub-construct's boundary
  // INPUT (Claims), injected by _extract_input from neo_subgraph_input — NOT a
  // peer node's output. Matrix rates _extract_input + subgraph Direct, so the
  // runtime mechanic ports. But the TYPE of inputData is still unknowable
  // statically (same cast problem).
  const claims = inputData as Claims;
  const scores: Record<string, string> = {
    authenticate: "high",
    "log access": "medium",
    encrypt: "high",
  };
  const scored = claims.items.map((claim) => {
    let score = "low";
    for (const [keyword, s] of Object.entries(scores)) {
      if (claim.includes(keyword)) {
        score = s;
        break;
      }
    }
    return { claim, score };
  });
  return { scored };
};

// ── Build sub-pipeline with declared I/O boundary ────────────────────────
// PARITY (Construct input=/output= boundary ports): Direct. The TS `Construct`
// class takes `input`/`output` Zod schemas as the isolated boundary port.
// State isolation (lookup/score hidden from parent) is a compiler behavior;
// matrix rates "Subgraph compilation / Output boundary contract" Direct.

const enrich = new Construct("enrich", {
  input: Claims,
  output: ScoredClaims,
  nodes: [
    // PARITY (fn="string" indirection): Python passes fn as a STRING NAME and
    // resolves it via the `scripted={...}` dict at compile(). This name→fn
    // indirection is what lets an LLM/config assemble the graph without the fn
    // in scope (the programmatic-surface use case). The proposed TS API keeps
    // it (matrix: "Registries (scripted) → Module-level Maps, Direct"). But
    // the doc never SHOWS the declarative Node.scripted string form nor the
    // compile-time scripted-dict injection — see API-GAP note at compile().
    Node.scripted("lookup", { fn: "lookupContext", inputs: Claims, outputs: Context }),
    Node.scripted("score", { fn: "scoreClaims", inputs: Claims, outputs: ScoredClaims }),
  ],
});

// ── Build parent pipeline ────────────────────────────────────────────────
// PARITY: a Construct nested directly in another Construct's nodes[] — the
// enrich sub-pipeline sits inline between decompose and report. Direct; TS
// nodes[] accepts Node | Construct just like Python's list[Node | Construct].

const pipeline = new Construct("req-analysis", {
  nodes: [
    Node.scripted("decompose", { fn: "decomposeReq", outputs: Claims }),
    enrich, // sub-pipeline: Claims -> ScoredClaims (isolated state)
    Node.scripted("report", { fn: "formatReport", inputs: ScoredClaims, outputs: Report }),
  ],
});

// ── Run ──────────────────────────────────────────────────────────────────
// PARITY (scripted registry at compile): the string names above are bound to
// real fns HERE. Python: `compile(pipeline, scripted=_SCRIPTED)`.
const SCRIPTED: Record<string, ScriptedFn> = {
  decomposeReq,
  formatReport,
  lookupContext,
  scoreClaims,
};

async function main() {
  // API-GAP: `compile(construct, { scripted })` — the doc's compile() row is
  // `compile(construct, {checkpointer})`; it never lists a `scripted` registry
  // option, yet the declarative surface REQUIRES one to resolve fn="string"
  // refs. Every worked TS example in the doc uses inline `node({}, fn)`
  // closures, sidestepping this. The registry-injection seam for the
  // programmatic surface is unspecified.
  const graph = compile(pipeline, { scripted: SCRIPTED });

  // PARITY (run input): Direct. Python run(graph, input={"node_id": "REQ-100"}).
  const result = await run(graph, { input: { node_id: "REQ-100" } });

  // PARITY (result keys): result["report"] is the Report. In TS `result.report`
  // is `unknown`/loosely typed — the state model is built at runtime from the
  // construct, so the compiler can't give `result` a precise static shape
  // (LangGraph.js Annotation.Root state isn't reflected back into `run`'s
  // return type). Another spot where the declarative surface loses static
  // typing the @node surface could keep. MEDIUM.
  const report = result.report as Report;
  console.log(report.text);
  console.log(`\nResult keys: ${Object.keys(result)}`);
  console.log("Note: 'lookup' and 'score' are NOT in the result — internal to enrich");
}

// ═══════════════════════════════════════════════════════════════════════════
// Variant: Construct | Oracle — ensemble the entire sub-pipeline
// ═══════════════════════════════════════════════════════════════════════════

// PARITY (merge_fn signature + string ref): `(variants, config)`. Same raw
// untyped `variants` situation as scripted fns — `variants` is ScoredClaims[]
// but nothing binds that statically; cast required. Referenced by STRING NAME
// from Oracle, resolved via the compile registry (like @merge_fn's name-keyed
// registry in Python). Matrix rates make_oracle_merge_fn Direct.
const mergeScored: MergeFn = (variants, _config): ScoredClaims => {
  const best: Record<string, string> = {};
  const scoreRank: Record<string, number> = { high: 3, medium: 2, low: 1 };
  for (const variant of variants as ScoredClaims[]) {
    for (const item of variant.scored) {
      const claim = item.claim;
      const rank = scoreRank[item.score] ?? 0; // PARITY: dict.get(k,0) -> ?? 0
      if (!(claim in best) || rank > (scoreRank[best[claim]] ?? 0)) {
        best[claim] = item.score;
      }
    }
  }
  return {
    scored: Object.entries(best).map(([claim, score]) => ({ claim, score })),
  };
};

// PARITY (Construct | Oracle -> .pipe(Oracle(...))): the headline of this
// variant. Python `Construct(...) | Oracle(n=3, merge_fn="merge_scored")`
// becomes `.pipe(Oracle({ n: 3, mergeFn: "mergeScored" }))`. Matrix rates the
// pipe redesign Direct-ish; the Oracle-on-a-Construct (ensemble a whole
// sub-pipeline, not a single node) is the interesting case and the doc's
// "Oracle fan-out (Send) Direct" + "make_subgraph_fn Direct" cover it.
const enrichOracle = new Construct("enrich", {
  input: Claims,
  output: ScoredClaims,
  nodes: [
    Node.scripted("lookup", { fn: "lookupContext", inputs: Claims, outputs: Context }),
    Node.scripted("score", { fn: "scoreClaims", inputs: Claims, outputs: ScoredClaims }),
  ],
}).pipe(Oracle({ n: 3, mergeFn: "mergeScored" }));

const pipelineOracle = new Construct("req-analysis-oracle", {
  nodes: [
    Node.scripted("decompose", { fn: "decomposeReq", outputs: Claims }),
    enrichOracle,
    Node.scripted("report", { fn: "formatReport", inputs: ScoredClaims, outputs: Report }),
  ],
});

// Uncomment to run the Oracle variant:
// async function mainOracle() {
//   const graph = compile(pipelineOracle, {
//     scripted: { ...SCRIPTED, mergeScored }, // PARITY: merge_fn joins the registry
//   });
//   const result = await run(graph, { input: { node_id: "REQ-101" } });
//   console.log("\n--- Oracle variant ---");
//   console.log((result.report as Report).text);
// }

void main;
void pipelineOracle;
