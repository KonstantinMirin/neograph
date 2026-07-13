// TS PARITY SKETCH — hypothetical, NOT compilable/runnable.
// Python source: examples/15_loop_refinement.py
// Target API: docs/design/typescript-port.md (AD-0 transformer form + .pipe() + fluent ForwardConstruct)
//
// This file grounds a feature-parity analysis of neograph's Loop modifier across
// five patterns: self-loop, multi-node loop sub-construct, hidden-loop sub-construct,
// ForwardConstruct self.loop(), and produce+validate (input != output).
//
// Legend for inline notes:
//   // PARITY:DIRECT   — same logic, different syntax
//   // PARITY:REDESIGN — TS limitation forces a different approach
//   // PARITY:BLOCKED  — no clean TS path
//   // PARITY:GAP      — proposed TS API in typescript-port.md doesn't cover this

import { z } from "zod";
import {
  node,
  compile,
  run,
  Loop,
  constructFromFunctions,
  ForwardConstruct,
  Node,
} from "@neograph/core";

// -- Schemas ------------------------------------------------------------------
// PARITY:DIRECT — Pydantic BaseModel(frozen=True) -> Zod object.
// PARITY:GAP — frozen=True has no Zod analogue; z.parse() returns a plain mutable
//   object. Would need Object.freeze() wrapping or a readonly branded type to
//   match the Python immutability guarantee. The example never mutates a model,
//   so behavior is preserved, but the *contract* silently weakens.

const Draft = z.object({
  content: z.string(),
  iteration: z.number().int().default(0),
  score: z.number().default(0.0),
});
type Draft = z.infer<typeof Draft>;

const ReviewResult = z.object({
  score: z.number(),
  feedback: z.string(),
});
type ReviewResult = z.infer<typeof ReviewResult>;

const Essay = z.object({
  content: z.string(),
  final_score: z.number(),
  iterations: z.number().int(),
});
type Essay = z.infer<typeof Essay>;

// =============================================================================
// Pattern 1: Self-loop — single node refines its own output
// =============================================================================

function demoSelfLoop() {
  // PARITY:DIRECT (actually nicer) — Python boxes mutable closure state as
  // `call_count = [0]` because Python closures can't rebind an outer scalar.
  // TS `let` closures rebind natively, so no [0] boxing ceremony.
  let callCount = 0;

  // PARITY:REDESIGN — Python `@node(outputs=Draft)` decorator on a standalone
  // `def seed()` has no TS equivalent (TS decorators only attach to class members).
  // Becomes the node({config}, fn) wrapper. AD-0 transformer reads `(): Draft`
  // return annotation to emit outputs=Draft, so the schema stays implicit.
  // PARITY:GAP — a no-input node forces an empty config object `node({}, ...)`;
  // Python's bare `@node(outputs=Draft)` is cleaner for zero-param seeds.
  const seed = node({}, (): Draft => ({
    content: "Initial draft about microservices",
    iteration: 0,
    score: 0.0,
  }));

  const refine = node(
    {
      // PARITY:DIRECT — loop_when lambda -> loopWhen arrow fn.
      // PARITY:REDESIGN — the `d is None` first-iteration sentinel. Python passes
      //   None; LangGraph.js state defaults are typically `undefined` OR a
      //   default() factory. Which null-ish value the runtime feeds the predicate
      //   on iteration 0 must be pinned, else `d === null` misses an `undefined`.
      //   Mirrors the loop_condition_none_unsafe lint — but TS has TWO empties.
      loopWhen: (d: Draft | null) => d === null || d.score < 0.8,
      maxIterations: 5,
    },
    // AD-0 transformer extracts inputs={draft: Draft}, output=Draft from signature.
    (draft: Draft): Draft => {
      callCount += 1;
      const newScore = draft.score + 0.3;
      return {
        content: `v${callCount}: refined draft`,
        iteration: draft.iteration + 1,
        score: newScore,
      };
    },
  );

  // PARITY:DIRECT — construct_from_functions([...]) passes wrapper handles.
  const pipeline = constructFromFunctions("self-loop", [seed, refine]);
  const graph = compile(pipeline);
  const result = run(graph, { input: { node_id: "self-loop-demo" } });

  // PARITY:GAP — Loop's append-reducer makes result["refine"] a Draft[], but the
  //   NODE output type is Draft. run()'s return is Record<string, unknown>, so
  //   there is no typed accessor that reflects "this one field got list-ified by
  //   a modifier". Python is also untyped at result[...] but the language never
  //   promised more; TS users EXPECT typed results, so this reads as a regression.
  const history = result["refine"] as Draft[];
  // PARITY:DIRECT (minor) — Python `history[-1]` negative index -> length math.
  const final = history[history.length - 1];

  console.log("=== Pattern 1: Self-loop ===");
  console.log(`Iterations: ${history.length}`);
  history.forEach((draft, i) =>
    console.log(`  [${i + 1}] score=${draft.score.toFixed(1)} content=${JSON.stringify(draft.content)}`),
  );
  console.log(`Final: score=${final.score.toFixed(1)}, iteration=${final.iteration}`);
}

// =============================================================================
// Pattern 2: Multi-node loop body as sub-construct
// =============================================================================

function demoMultiNodeLoop() {
  let reviewCount = 0;

  const draft = node({}, (): Draft => ({
    content: "Initial essay about authentication",
    iteration: 0,
    score: 0.0,
  }));

  const review = node({}, (draft: Draft): ReviewResult => {
    reviewCount += 1;
    const score = 0.3 * reviewCount;
    return {
      score: Math.min(score, 1.0),
      feedback: `Iteration ${reviewCount}: ${score < 0.8 ? "needs work" : "approved"}`,
    };
  });

  // PARITY:DIRECT — fan-in: two upstream params (draft, review). AD-0 transformer
  // emits inputs={draft: Draft, review: ReviewResult}; param names ARE the edges.
  const revise = node({}, (draft: Draft, review: ReviewResult): Draft => ({
    content: `Revised: ${review.feedback}`,
    iteration: draft.iteration + 1,
    score: review.score,
  }));

  // PARITY:REDESIGN — the consumer param `refine` is named after the SUB-CONSTRUCT
  //   below, and it's typed Draft (single) while the state field holds Draft[]
  //   (loop append-list). neograph feeds downstream consumers the last element.
  //   The transformer will extract inputs={refine: Draft}; the list->element
  //   unwrap is a runtime convention the TS type system can't see. Fine at
  //   runtime, but `refine: Draft` here vs `Draft[]` in result[] is a type schism.
  const finalize = node({}, (refine: Draft): Essay => ({
    content: refine.content,
    final_score: refine.score,
    iterations: refine.iteration,
  }));

  // PARITY:REDESIGN — the pipe. Python `sub | Loop(...)` -> `.pipe(Loop({...}))`.
  //   Also: input=Draft / output=Draft are Pydantic CLASSES in Python but must be
  //   Zod SCHEMA VALUES here. These are runtime boundary ports the transformer
  //   can't infer from any signature — they're explicit args either way.
  const refine = constructFromFunctions(
    "refine",
    [review, revise],
    { input: Draft, output: Draft },
  ).pipe(
    Loop({ when: (d: Draft | null) => d === null || d.score < 0.8, maxIterations: 10 }),
  );

  const pipeline = constructFromFunctions("writer", [draft, refine, finalize]);
  const graph = compile(pipeline);
  const result = run(graph, { input: { node_id: "multi-loop-demo" } });

  const history = result["refine"] as Draft[];
  const essay = result["finalize"] as Essay;

  console.log("=== Pattern 2: Multi-node loop (sub-construct) ===");
  console.log(`Review iterations: ${reviewCount}`);
  history.forEach((d, i) =>
    console.log(`  [${i + 1}] score=${d.score.toFixed(1)} content=${JSON.stringify(d.content)}`),
  );
  console.log(`Final essay: score=${essay.final_score}, iterations=${essay.iterations}`);
}

// =============================================================================
// Pattern 3: Loop inside a sub-construct (hidden from parent)
// =============================================================================

function demoLoopInSubConstruct() {
  let improveCount = 0;

  // PARITY:REDESIGN — `write(topic: Draft)`: the param `topic` type matches the
  //   sub-construct's input=Draft, so it's a PORT param (reads neo_subgraph_input,
  //   not a peer node). Python resolves this by issubclass(annotation, input).
  //   TS must match extracted typeName "Draft" against the input schema by name/
  //   structure (AD-3 Zod comparison) — issubclass has no TS equivalent.
  const write = node({}, (topic: Draft): Draft => ({
    content: "First draft",
    iteration: 0,
    score: 0.5,
  }));

  const improve = node(
    {
      loopWhen: (d: Draft | null) => d === null || d.score < 0.8,
      maxIterations: 5,
    },
    (write: Draft): Draft => {
      improveCount += 1;
      return {
        content: `Improved v${improveCount}`,
        iteration: write.iteration + 1,
        score: write.score + 0.2,
      };
    },
  );

  const refineSub = constructFromFunctions(
    "refine",
    [write, improve],
    { input: Draft, output: Draft },
  );

  const seed = node({}, (): Draft => ({
    content: "topic: distributed systems",
    iteration: 0,
    score: 0.0,
  }));

  const finalize = node({}, (refine: Draft): Essay => ({
    content: refine.content,
    final_score: refine.score,
    iterations: refine.iteration,
  }));

  const pipeline = constructFromFunctions("writer", [seed, refineSub, finalize]);
  const graph = compile(pipeline);
  const result = run(graph, { input: { node_id: "sub-loop-demo" } });

  const essay = result["finalize"] as Essay;

  console.log("=== Pattern 3: Loop inside sub-construct ===");
  console.log(`Internal improve iterations: ${improveCount}`);
  console.log(`Final essay: score=${essay.final_score}, iterations=${essay.iterations}`);
  // PARITY:DIRECT — state-hygiene assertions (sub-construct internals don't leak).
  console.log(`Parent sees 'refine' output: ${"refine" in result}`);
  console.log(`Parent does NOT see 'write': ${!("write" in result)}`);
  console.log(`Parent does NOT see 'improve': ${!("improve" in result)}`);
}

// =============================================================================
// Pattern 4: ForwardConstruct with self.loop() — explicit cycle primitive
// =============================================================================

function demoForwardLoop() {
  let reviewCount = 0;

  // PARITY:DIRECT — raw scripted fns with (state, config) signature.
  const fcDraft = (_in: unknown, _cfg: unknown): Draft => ({
    content: "Initial essay about distributed systems",
    iteration: 0,
    score: 0.0,
  });
  const fcReview = (_in: unknown, _cfg: unknown): ReviewResult => {
    reviewCount += 1;
    const score = 0.3 * reviewCount;
    return {
      score: Math.min(score, 1.0),
      feedback: `Iteration ${reviewCount}: ${score < 0.8 ? "needs work" : "approved"}`,
    };
  };
  const fcRevise = (_in: unknown, _cfg: unknown): Draft => ({
    content: `Revised v${reviewCount}`,
    iteration: reviewCount,
    score: 0.3 * reviewCount,
  });

  // PARITY:REDESIGN — ForwardConstruct subclass. Node.scripted(...) as class
  //   FIELDS ports directly (class members CAN take TS decorators/initializers).
  //   forward() body is symbolically traced: `this.draft(topic)` returns a JS
  //   Proxy, recording the call. Python uses _Proxy.__getattr__; TS uses a
  //   Proxy get-trap. DIRECT per matrix.
  class Writer extends ForwardConstruct {
    draft = Node.scripted("draft", { fn: "fc_draft_ex15", outputs: Draft });
    review = Node.scripted("review", { fn: "fc_review_ex15", outputs: ReviewResult });
    revise = Node.scripted("revise", { fn: "fc_revise_ex15", outputs: Draft });

    forward(topic: any) {
      let d = this.draft(topic);
      // PARITY:REDESIGN — self.loop() is the EXPLICIT cycle primitive. This
      //   example NEVER uses `if (proxy)` branching, so it dodges the hardest
      //   ForwardConstruct port problem entirely: Python re-traces via
      //   `_Proxy.__bool__`, but JS `if (proxy)` always truthy and JS cannot
      //   intercept boolean coercion (no __bool__). Branch-heavy examples would
      //   need the .gt()/.then()/.else() fluent API; THIS one only needs .loop(),
      //   which is a plain method call — so the loop primitive is a clean port.
      d = this.loop({
        body: [this.review, this.revise],
        when: (r: ReviewResult | null) => r === null || r.score < 0.8,
        maxIterations: 10,
      })(d);
      return d;
    }
  }

  // PARITY:DIRECT — string-keyed scripted registry passed to compile.
  //   Python `scripted={...}` dict -> TS `{ scripted: {...} }` options object.
  //   The fn="fc_draft_ex15" string indirection ports 1:1 (module-level Map).
  const graph = compile(new Writer(), {
    scripted: {
      fc_draft_ex15: fcDraft,
      fc_review_ex15: fcReview,
      fc_revise_ex15: fcRevise,
    },
  });
  const result = run(graph, { input: { node_id: "forward-loop-demo" } });

  console.log("=== Pattern 4: ForwardConstruct with self.loop() ===");
  console.log(`Review iterations: ${reviewCount}`);
  console.log(`Draft output: ${JSON.stringify(result["draft"])}`);
  // PARITY:REDESIGN — Python duck-types the loop output via hasattr(val[0],
  //   "score") over result.values(). TS erases types at runtime, so "does this
  //   list element have a .score" needs a runtime shape check or the Zod schema.
  for (const val of Object.values(result)) {
    if (Array.isArray(val) && val.length && typeof (val[0] as any)?.score === "number") {
      (val as Draft[]).forEach((d, i) =>
        console.log(`  [${i + 1}] score=${d.score.toFixed(1)} content=${JSON.stringify(d.content)}`),
      );
    }
  }
}

// =============================================================================
// Pattern 5: Produce + Validate with Loop (input != output)
// =============================================================================

function demoProduceValidate() {
  // PARITY:DIRECT — nested schemas defined in function scope.
  const ProduceInput = z.object({ request_id: z.string(), payload: z.string() });
  type ProduceInput = z.infer<typeof ProduceInput>;

  // PARITY:DIRECT — `error: str | None = None` -> nullable+default.
  const ProduceOutput = z.object({
    request_id: z.string(),
    data: z.string(),
    error: z.string().nullable().default(null),
  });
  type ProduceOutput = z.infer<typeof ProduceOutput>;

  const Validated = z.object({
    request_id: z.string(),
    data: z.string(),
    errors: z.array(z.string()),
  });
  type Validated = z.infer<typeof Validated>;

  const Report = z.object({
    request_id: z.string(),
    data: z.string(),
    attempts: z.number().int(),
  });
  type Report = z.infer<typeof Report>;

  let produceCalls = 0;

  // PARITY:REDESIGN — `topic: ProduceInput` is a PORT param (type matches
  //   input=ProduceInput). Name+type identity match across the transformer/runtime
  //   seam replaces Python issubclass.
  const produce = node({}, (topic: ProduceInput): ProduceOutput => {
    produceCalls += 1;
    if (produceCalls === 1) {
      return { request_id: topic.request_id, data: "", error: "transient failure: service unavailable" };
    }
    return { request_id: topic.request_id, data: `generated data for '${topic.payload}'`, error: null };
  });

  const check = node({}, (produce: ProduceOutput): Validated => {
    const errors: string[] = [];
    if (produce.error) errors.push(produce.error);
    if (!produce.data) errors.push("empty data");
    return { request_id: produce.request_id, data: produce.data, errors };
  });

  // PARITY:REDESIGN — sub-construct with input != output (ProduceInput -> Validated).
  //   `.pipe(Loop(...))` on a boundary-ported sub-construct. The when predicate
  //   reads bool(v.errors); TS `v.errors.length > 0` (no truthy-list coercion).
  const produceValidate = constructFromFunctions(
    "produce_validate",
    [produce, check],
    { input: ProduceInput, output: Validated },
  ).pipe(
    Loop({
      when: (v: Validated | null) => v === null || v.errors.length > 0,
      maxIterations: 5,
    }),
  );

  const seed = node({}, (): ProduceInput => ({ request_id: "req-42", payload: "quarterly report" }));

  const report = node({}, (produce_validate: Validated): Report => ({
    request_id: produce_validate.request_id,
    data: produce_validate.data,
    attempts: produceCalls,
  }));

  const pipeline = constructFromFunctions("produce-pipeline", [seed, produceValidate, report]);
  const graph = compile(pipeline);
  const result = run(graph, { input: { node_id: "produce-validate-demo" } });

  const history = result["produce_validate"] as Validated[];
  const finalReport = result["report"] as Report;

  console.log("PRODUCE+VALIDATE LOOP (input != output)");
  console.log(`Producer calls: ${produceCalls}`);
  history.forEach((v, i) => {
    const status = v.errors.length === 0 ? "PASS" : `FAIL (${v.errors.join(", ")})`;
    console.log(`  [${i + 1}] ${status} data=${JSON.stringify(v.data)}`);
  });
  console.log(
    `Final report: request_id=${finalReport.request_id}, data=${JSON.stringify(finalReport.data)}, attempts=${finalReport.attempts}`,
  );
}

// PARITY:DIRECT — top-level runner.
demoSelfLoop();
demoMultiNodeLoop();
demoLoopInSubConstruct();
demoForwardLoop();
demoProduceValidate();
