// HYPOTHETICAL TypeScript port of examples/27_forward_agent_wiring.py
// Source: /Users/konst/projects/neograph/examples/27_forward_agent_wiring.py
// Target API: docs/design/typescript-port.md (AD-0 transformer, AD-5 fluent branch API).
//
// This file is NOT compilable — there is no TS implementation. It is a faithful
// sketch of what the DX WOULD look like against the PROPOSED API, annotated with
// `// PARITY:` where TS diverges from the Python original.
//
// This example is a ForwardConstruct showcase built on the PROGRAMMATIC surface
// (`Node.scripted(fn="name")` + `compile(scripted={...})`), NOT the @node decorator.
// So the AD-0 transformer buys us almost nothing here: signatures are Python
// closures resolved by string name at compile time, not typed node functions the
// transformer can read. Every node needs an explicit Zod schema regardless.

import { z } from "zod";
import {
  ForwardConstruct,
  Node,
  compile,
  run,
  MemorySaver, // PARITY: langgraph.checkpoint.memory.MemorySaver -> @langchain/langgraph MemorySaver
  type Proxy, // symbolic-tracing proxy handed to forward()
} from "@neograph/core";

// ── shared typed channels ────────────────────────────────────────────────────
// PARITY: Pydantic `class Triage(BaseModel, frozen=True)` -> Zod schema + inferred type.
//   - `frozen=True` has NO Zod equivalent: Zod validates plain objects, it does not
//     mint immutable class instances. Downstream code gets mutable POJOs.
//   - Field defaults (`score: float = 0.0`, `revision: int = 0`) become `.default(0)`,
//     but that only fills on PARSE, not on object-literal construction (see fns below).

const Triage = z.object({ confidence: z.number(), topic: z.string() });
type Triage = z.infer<typeof Triage>;

const Analysis = z.object({ summary: z.string(), depth: z.string() });
type Analysis = z.infer<typeof Analysis>;

const Draft = z.object({
  content: z.string(),
  score: z.number().default(0.0),
  revision: z.number().int().default(0),
});
type Draft = z.infer<typeof Draft>;

const Claim = z.object({ id: z.string(), text: z.string() });
type Claim = z.infer<typeof Claim>;

const ClaimBatch = z.object({ claims: z.array(Claim) });
type ClaimBatch = z.infer<typeof ClaimBatch>;

const Verdict = z.object({ id: z.string(), supported: z.boolean() });
type Verdict = z.infer<typeof Verdict>;

const Report = z.object({ supported: z.number().int(), total: z.number().int() });
type Report = z.infer<typeof Report>;

// =============================================================================
// Demo 1 — BRANCH: `if` compiles to a conditional edge
// =============================================================================
function demoBranch(): void {
  console.log("DEMO 1 — branch: if -> conditional edge (deep vs shallow)");

  // PARITY: Python scripted fns are `(inp, _cfg)`; keep the (inp, cfg) shape.
  //   Return an object literal — but note there is no `Triage(...)` constructor.
  //   `Triage.parse({...})` would apply defaults; here we hand-roll the POJO and
  //   trust the compiler's state-write validation.  DX regression vs `Triage(...)`.
  const fnTriage = (_inp: unknown, _cfg: unknown): Triage =>
    ({ confidence: 0.55, topic: "distributed consensus" });
  const fnDeep = (inp: Triage, _cfg: unknown): Analysis =>
    ({ summary: `deep dive on ${inp.topic}`, depth: "deep" });
  const fnShallow = (inp: Triage, _cfg: unknown): Analysis =>
    ({ summary: `quick scan of ${inp.topic}`, depth: "shallow" });

  class Router extends ForwardConstruct {
    // PARITY: class-attribute nodes port directly. `Node.scripted` static factory
    //   exists in the matrix (Direct). `fn="fn_triage"` string key -> resolved via
    //   compile({scripted}) Map.
    triage = Node.scripted("triage", { fn: "fn_triage", outputs: Triage });
    deep = Node.scripted("deep", { fn: "fn_deep", inputs: Triage, outputs: Analysis });
    shallow = Node.scripted("shallow", { fn: "fn_shallow", inputs: Triage, outputs: Analysis });

    // PARITY (REDESIGN — the sharpest divergence in this whole example):
    //   Python:
    //       checked = self.triage(topic)
    //       if checked.confidence > 0.8:        # proxy.__gt__ + __bool__ re-trace
    //           return self.shallow(checked)
    //       return self.deep(checked)
    //   JS cannot intercept `if (proxy)` (no __bool__), so per AD-5 the branch
    //   becomes a fluent expression. The natural Python EARLY-RETURN + FALL-THROUGH
    //   collapses into an explicit two-arm .then()/.else(). Both arms must be
    //   present as callbacks; there is no "fall through to the rest of forward()".
    forward(topic: Proxy) {
      const checked = this.triage(topic);
      return checked.confidence
        .gt(0.8) // PARITY: `checked.confidence > 0.8` -> `.gt(0.8)` on the proxy attr
        .then(() => this.shallow(checked))
        .else(() => this.deep(checked));
    }
  }

  const graph = compile(new Router(), {
    // PARITY GAP: matrix shows `compile(construct, {checkpointer})` only. The
    //   scripted-fn registry (essential for THIS surface) is not in the proposed
    //   compile signature. Adding `scripted` here is an inferred extension.
    scripted: { fn_triage: fnTriage, fn_deep: fnDeep, fn_shallow: fnShallow },
  });
  const result = run(graph, { input: { node_id: "d1" } });
  const picked = (result["deep"] ?? result["shallow"]) as Analysis;
  console.log(`routed to: ${picked.depth} (${picked.summary})`);
  console.assert(picked.depth === "deep");
}

// =============================================================================
// Demo 2 — SELF.LOOP: iterative refine cycle (real graph back-edge)
// =============================================================================
function demoLoop(): void {
  console.log("DEMO 2 — self.loop(): draft -> [review -> revise] until score >= 0.8");

  const rounds = { n: 0 };
  const fnDraft = (_inp: unknown, _cfg: unknown): Draft => ({ content: "v0", score: 0.0, revision: 0 });
  const fnReview = (inp: Draft, _cfg: unknown): Draft => {
    rounds.n += 1;
    return { content: inp.content, score: Math.min(0.3 * rounds.n, 1.0), revision: rounds.n };
  };
  const fnRevise = (inp: Draft, _cfg: unknown): Draft =>
    ({ content: `v${rounds.n}`, score: inp.score, revision: rounds.n });

  class Writer extends ForwardConstruct {
    draft = Node.scripted("draft", { fn: "fn_draft", outputs: Draft });
    review = Node.scripted("review", { fn: "fn_review", inputs: Draft, outputs: Draft });
    revise = Node.scripted("revise", { fn: "fn_revise", inputs: Draft, outputs: Draft });

    forward(topic: Proxy) {
      const d = this.draft(topic);
      // PARITY GAP (deferred-builder method NOT specified in the matrix): the AD-5
      //   section lists `.loop()` as the loop marker but never shows it taking a
      //   `body=[...]` array + a `when` predicate + `maxIterations`. Assumed shape:
      return this.loop({
        body: [this.review, this.revise],
        // PARITY (mostly-direct): `when=lambda d: d is None or d.score < 0.8` -> a JS
        //   arrow. Null-safety is idiomatic in both. `d` is the loop-carried value.
        when: (d: Draft | null) => d === null || d.score < 0.8,
        maxIterations: 10,
      })(d); // deferred builder applied to the carried proxy — Proxy call-trap.
    }
  }

  const graph = compile(new Writer(), {
    scripted: { fn_draft: fnDraft, fn_review: fnReview, fn_revise: fnRevise },
  });
  const result = run(graph, { input: { node_id: "d2" } });
  // PARITY: loop-body output accumulates as a list, same runtime behavior; the
  //   "hunt result.values() for the array" idiom ports 1:1 to Object.values().
  const history = (Object.values(result).find(
    (v) => Array.isArray(v) && v.length > 0 && "score" in (v[0] as object),
  ) ?? []) as Draft[];
  console.log(`looped ${rounds.n} rounds; scores: ${history.map((d) => d.score.toFixed(1))}`);
  console.assert(history.length > 0 && history[history.length - 1].score >= 0.8);
}

// =============================================================================
// Demo 3 — SELF.EACH: custom-key fan-out over a batch
// =============================================================================
function demoEachFanout(): void {
  console.log("DEMO 3 — self.each(): fan verify out over claims, keyed by id, then reduce");

  const fnExtract = (_inp: unknown, _cfg: unknown): ClaimBatch =>
    ({ claims: [0, 1, 2, 3].map((i) => ({ id: `c${i}`, text: `claim ${i}` })) });
  const fnVerify = (item: Claim, _cfg: unknown): Verdict =>
    ({ id: item.id, supported: Number(item.id.slice(1)) % 2 === 0 });
  const fnReport = (inp: { each_verify: Record<string, Verdict> }, _cfg: unknown): Report => {
    // PARITY: `inp["each_verify"].values()` -> Object.values(inp.each_verify). Direct.
    const verdicts = Object.values(inp.each_verify);
    return { supported: verdicts.filter((v) => v.supported).length, total: verdicts.length };
  };

  class Auditor extends ForwardConstruct {
    extract = Node.scripted("extract", { fn: "fn_extract", outputs: ClaimBatch });
    verify = Node.scripted("verify", { fn: "fn_verify", inputs: Claim, outputs: Verdict });
    // PARITY: dict-form inputs `{"each_verify": dict[str, Verdict]}` ->
    //   `{ each_verify: z.record(z.string(), Verdict) }`. Direct; the `each_` key
    //   convention is a runtime naming contract, unchanged.
    report = Node.scripted("report", {
      fn: "fn_report",
      inputs: { each_verify: z.record(z.string(), Verdict) },
      outputs: Report,
    });

    forward(topic: Proxy) {
      const batch = this.extract(topic);
      // PARITY GAP: `self.each(body=[...], key="id")(batch.claims)` — the each
      //   deferred-builder is not in the AD-5 matrix. `batch.claims` is a proxy
      //   attr read (Proxy get-trap) feeding the fan-out collection. Assumed shape:
      this.each({ body: [this.verify], key: "id" })(batch.claims);
      return this.report(batch);
    }
  }

  const graph = compile(new Auditor(), {
    scripted: { fn_extract: fnExtract, fn_verify: fnVerify, fn_report: fnReport },
  });
  const result = run(graph, { input: { node_id: "d3" } });
  const rep = result["report"] as Report;
  console.log(`verified ${rep.total} claims; ${rep.supported} supported`);
  console.assert(rep.total === 4);
}

// =============================================================================
// Demo 4 — CASCADE: fan-out INSIDE a loop (the e9zse flagship shape)
// =============================================================================
function demoCascade(): void {
  console.log("DEMO 4 — cascade: self.loop(body=[ get_claims, self.each(verify), collect ])");

  const fnIntake = (_inp: unknown, _cfg: unknown): ClaimBatch =>
    ({ claims: [0, 1, 2].map((i) => ({ id: `c${i}`, text: `claim ${i}` })) });
  const fnGetClaims = (inp: ClaimBatch, _cfg: unknown): ClaimBatch => inp;
  const fnVerify = (item: Claim, _cfg: unknown): Verdict => ({ id: item.id, supported: true });
  const fnCollect = (inp: { each_verify: Record<string, Verdict> }, _cfg: unknown): Report => {
    const verdicts = Object.values(inp.each_verify);
    return { supported: verdicts.filter((v) => v.supported).length, total: verdicts.length };
  };

  class Cascade extends ForwardConstruct {
    intake = Node.scripted("intake", { fn: "fn_intake", outputs: ClaimBatch });
    get_claims = Node.scripted("get_claims", { fn: "fn_get_claims", inputs: ClaimBatch, outputs: ClaimBatch });
    verify = Node.scripted("verify", { fn: "fn_verify", inputs: Claim, outputs: Verdict });
    collect = Node.scripted("collect", {
      fn: "fn_collect",
      inputs: { each_verify: z.record(z.string(), Verdict) },
      outputs: Report,
    });

    forward(topic: Proxy) {
      const batch = this.intake(topic);
      // PARITY GAP (compound, the biggest single gap this example exposes):
      //   NESTED deferred builders — a `this.each(...)` INSIDE a `this.loop(body=[...])`.
      //   A nested each is never called directly, so its collection is supplied at
      //   BUILD time via `over="get_claims.claims"` (a string path), not a proxy call.
      //   The matrix's `.loop()` says nothing about a `body` array holding a nested
      //   each with a build-time `over`. This whole cascade shape is unspecified in
      //   the proposed API.
      return this.loop({
        body: [
          this.get_claims,
          this.each({ body: [this.verify], key: "id", over: "get_claims.claims" }),
          this.collect,
        ],
        when: (r: Report | null) => r === null || r.total === 0,
        maxIterations: 2,
      })(batch);
    }
  }

  const graph = compile(new Cascade(), {
    scripted: { fn_intake: fnIntake, fn_get_claims: fnGetClaims, fn_verify: fnVerify, fn_collect: fnCollect },
  });
  const result = run(graph, { input: { node_id: "d4" } });
  const reports = (Object.values(result).find(
    (v) => Array.isArray(v) && v.length > 0 && "total" in (v[0] as object),
  ) ?? []) as Report[];
  const last = reports[reports.length - 1];
  console.log(`cascade ran; final: ${last.supported}/${last.total}`);
  console.assert(reports.length > 0 && last.total === 3);
}

// =============================================================================
// Demo 5 — SELF.ENSEMBLE: N parallel generators + judge-merge (Oracle)
// =============================================================================
function demoEnsemble(): void {
  console.log("DEMO 5 — self.ensemble(): 3 parallel drafts, merged to the best");

  const fnSeed = (_inp: unknown, _cfg: unknown): Draft => ({ content: "seed", score: 0.0, revision: 0 });
  const fnGenerate = (_inp: unknown, _cfg: unknown): Draft => ({ content: "candidate", score: 0.7, revision: 0 });
  // PARITY: merge_fn `(variants, _cfg)` — variants is the list of N generator
  //   outputs. Ports directly as a scripted fn resolved by string name.
  const fnMerge = (variants: Draft[], _cfg: unknown): Draft => {
    const best = variants.reduce((a, b) => (b.score > a.score ? b : a));
    return { content: `merged from ${variants.length} candidates`, score: best.score, revision: 0 };
  };

  class Ensembled extends ForwardConstruct {
    seed = Node.scripted("seed", { fn: "fn_seed", outputs: Draft });
    generate = Node.scripted("generate", { fn: "fn_generate", inputs: Draft, outputs: Draft });

    forward(topic: Proxy) {
      const s = this.seed(topic);
      // PARITY GAP: `self.ensemble(node, n=3, merge_fn="fn_merge")` deferred builder
      //   not in the matrix. `merge_fn` is a STRING name into the scripted registry
      //   (rides the same `scripted={}` map). Assumed shape:
      return this.ensemble(this.generate, { n: 3, mergeFn: "fn_merge" })(s);
    }
  }

  const graph = compile(new Ensembled(), {
    scripted: { fn_seed: fnSeed, fn_generate: fnGenerate, fn_merge: fnMerge },
  });
  const result = run(graph, { input: { node_id: "d5" } });
  const merged = result["generate"] as Draft;
  console.log(`ensemble merged -> ${merged.content} (score ${merged.score})`);
  console.assert(merged.content.includes("merged from 3"));
}

// =============================================================================
// Demo 6 — SELF.INTERRUPT: human-in-the-loop gate before a mutation
// =============================================================================
function demoInterrupt(): void {
  console.log("DEMO 6 — self.interrupt(): pause before apply for human approval");

  const fnPrecheck = (_inp: unknown, _cfg: unknown): Analysis =>
    ({ summary: "2 issues found", depth: "review-needed" });
  const fnApply = (_inp: unknown, _cfg: unknown): Report => ({ supported: 1, total: 1 });

  // PARITY: the condition receives the compiled state model. In Python it reads a
  //   Pydantic attr `getattr(state, "precheck", None)`. In TS the state is a plain
  //   object (LangGraph.js state), so `state.precheck`. Registered by string name.
  const needsReview = (state: { precheck?: Analysis }): boolean => {
    const val = state.precheck;
    return Boolean(val && val.depth === "review-needed");
  };

  class Gated extends ForwardConstruct {
    precheck = Node.scripted("precheck", { fn: "fn_precheck", outputs: Analysis });
    apply = Node.scripted("apply", { fn: "fn_apply", inputs: Analysis, outputs: Report });

    forward(topic: Proxy) {
      const checked = this.precheck(topic);
      // PARITY GAP: `self.interrupt(node, when="needs_review")` deferred builder not
      //   in the matrix. `when` is a STRING condition name into the conditions
      //   registry (contrast demo 2/4, where loop `when` is an inline lambda). The
      //   proposed API never reconciles the lambda-vs-string-name `when` duality.
      return this.interrupt(this.apply, { when: "needs_review" })(checked);
    }
  }

  const graph = compile(new Gated(), {
    scripted: { fn_precheck: fnPrecheck, fn_apply: fnApply },
    // PARITY GAP: `conditions={}` registry at compile — like `scripted={}`, absent
    //   from the proposed compile signature.
    conditions: { needs_review: needsReview },
    checkpointer: new MemorySaver(),
  });
  const cfg = { configurable: { thread_id: "d6" } };
  const paused = run(graph, { input: { node_id: "d6" }, config: cfg });
  console.log(`paused: __interrupt__ present = ${"__interrupt__" in paused}`);
  // PARITY: `run(graph, resume={...}, config=cfg)` HITL resume. The `resume` arg
  //   maps to LangGraph.js Command({resume}); shape assumed, not in the matrix.
  const resumed = run(graph, { resume: { approved: true }, config: cfg });
  console.log(`resumed -> apply ran: ${JSON.stringify(resumed["apply"])}`);
  console.assert("__interrupt__" in paused);
}

function main(): void {
  demoBranch();
  demoLoop();
  demoEachFanout();
  demoCascade();
  demoEnsemble();
  demoInterrupt();
}

main();
