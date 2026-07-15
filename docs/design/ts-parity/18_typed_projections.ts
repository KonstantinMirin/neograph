// TS PARITY SKETCH — HYPOTHETICAL, NON-COMPILING.
// Python source: examples/18_typed_projections.py
// Proposed API:  docs/design/typescript-port.md (AD-0 transformer, .pipe(), Zod schema layer)
//
// This example is a FIELD-VISIBILITY-CONTROL showcase, NOT a DAG/@node showcase. Its whole
// point is that a MODEL field can be shown/hidden independently on TWO axes:
//
//     axis A — INPUT RENDERING : what a downstream LLM SEES when it reads upstream data
//     axis B — OUTPUT SCHEMA   : what an LLM is ASKED to produce (via describe_type)
//
// Three Python mechanisms, each read at RUNTIME off Pydantic's per-field FieldInfo:
//   1. Field(exclude=True)               -> hidden on BOTH axes   (field_info.exclude)
//   2. Annotated[str, ExcludeFromOutput] -> visible on A, hidden on B (field_info.metadata)
//   3. render_for_prompt() -> BaseModel  -> full model swap for axis A only
//
// The proposed TS design doc marks "Renderers" and "describe_type" as Direct and never mentions
// exclude / ExcludeFromOutput / render_for_prompt AT ALL. That silence is exactly where this
// example breaks. The two load-bearing facts the design assumes away:
//   (i)  Pydantic FieldInfo carries per-field metadata slots (.exclude, .metadata, .description)
//        that renderer + describe_type read at runtime. Zod has NO per-field "exclude" slot and
//        no Annotated-metadata channel — you must invent one.
//   (ii) A Pydantic model is a CLASS with methods (render_for_prompt) AND a parsed value knows
//        its own fields (instance.__class__.model_fields). A Zod `z.infer<T>` value is a PLAIN
//        object: no methods, and no back-pointer to its schema. Both renderer.render(value) and
//        describe_value(value) rely on that back-pointer.

import { z } from "zod";
import {
  Node,
  Construct,
  compile,
  run,
  XmlRenderer,
  describeType,
  renderInput,
  // PARITY GAP: none of the following exist in typescript-port.md. They must be invented
  // for THIS example's visibility semantics to be expressible at all:
  neoField,            // Field(...) analogue: wraps a Zod type to carry {exclude, description}
  ExcludeFromOutput,   // marker; here a wrapper `excludeFromOutput(zodType)` (see note)
  type Renderable,     // interface a class implements to carry renderForPrompt()
} from "@neograph/core";

// ── Models ─────────────────────────────────────────────────────────────────

// Python: `class ResearchResult(BaseModel, frozen=True)` with two `Field(exclude=True)` fields.
//
// PARITY (REDESIGN): Zod's `z.object({...})` has no per-field `exclude`. `.describe()` is the
// ONLY native metadata slot, and it is a string, not a structured flag. So the exclude flag has
// to ride in an INVENTED wrapper (`neoField`) whose metadata both XmlRenderer and describeType
// must be taught to read. This is a NEW cross-cutting concept, not a syntax swap.
// frozen=True -> Readonly<> on the inferred type + Object.freeze at the boundary (Zod does not
// freeze parsed output the way Pydantic frozen models do).
const ResearchResult = z.object({
  content: z.string(),
  source_url: z.string(),
  score: z.number().default(0.0),

  // Pipeline-internal: hidden on BOTH axes.
  // Python: `Field(default=0, exclude=True)`.
  retrieval_latency_ms: neoField(z.number().default(0), { exclude: true }),
  cache_hit: neoField(z.boolean().default(false), { exclude: true }),
});
type ResearchResult = Readonly<z.infer<typeof ResearchResult>>;

// Python: `assigned_letter: Annotated[str, ExcludeFromOutput] = ""`.
//
// PARITY (REDESIGN — and this one specifically defeats AD-0). The design's transformer extracts
// NODE-FUNCTION parameter annotations (topic: FromInput<string>). ExcludeFromOutput is a MODEL
// FIELD annotation, on a schema, not on a function parameter — a surface the transformer never
// visits. So there is no "transformer detects it" path here; it MUST be an explicit wrapper.
// Semantics the wrapper must encode: keep the field in the value + in input rendering, but drop
// it from describe_type output. That is a THIRD state beyond present/excluded, so a boolean
// `exclude` is not enough — `neoField` needs an enum: "none" | "output" | "all".
const ExtensionCondition = z.object({
  text: z.string(),
  acceptance_criteria: z.array(z.string()).default([]),

  // Pipeline-set: visible when reading (axis A), hidden when producing (axis B).
  assigned_letter: ExcludeFromOutput(z.string().default("")),
  // equivalently: neoField(z.string().default(""), { exclude: "output" })
});
type ExtensionCondition = Readonly<z.infer<typeof ExtensionCondition>>;

// Python: HydratedResearch has a `render_for_prompt(self) -> ResearchPresentation` METHOD that
// returns a DIFFERENT, slimmer model (full projection: raw_html/metadata/internal_score dropped,
// content truncated, confidence rounded).
//
// PARITY (REDESIGN, verging on BLOCKED — the crux of this file). A Zod schema is a validator, not
// a class; `z.infer<T>` is a plain object with no methods, so a parsed value can NEVER carry
// `renderForPrompt`. Python dispatches on `hasattr(value, "render_for_prompt")` — a per-INSTANCE
// capability check that has no TS equivalent on plain data. Two escape hatches, both degrade DX:
//
//   (a) Use an actual CLASS (below) so instances carry methods. Cost: it is no longer the same
//       object the rest of the pipe validates/serializes, and a checkpointer (msgpack) round-trip
//       strips the prototype, so a resumed run silently loses renderForPrompt.
//   (b) A schema->projector side-channel registry. Cost: `renderInput(value)` — which Python calls
//       with the value ALONE — can no longer find the projector from a plain object; you must also
//       thread the schema in. Every renderInput call site diverges from Python.
//
// Sketching (a). The projection TARGET is itself a schema so describeType can render it.
const ResearchPresentation = z.object({
  content: z.string(),
  confidence: z.number().describe("0.0 to 1.0"),
});
type ResearchPresentation = Readonly<z.infer<typeof ResearchPresentation>>;

class HydratedResearch implements Renderable {
  static schema = z.object({
    raw_html: z.string(),
    metadata: z.record(z.string(), z.unknown()).default({}),
    internal_score: z.number().default(0.0),
  });
  constructor(
    readonly raw_html: string,
    readonly metadata: Record<string, unknown> = {},
    readonly internal_score: number = 0.0,
  ) {}

  // PARITY: must hand back {value, schema} — a bare object would give the renderer field NAMES
  // but no TYPES/descriptions, so BAML/describe_value of the projection would be untyped.
  renderForPrompt(): { value: ResearchPresentation; schema: typeof ResearchPresentation } {
    return {
      schema: ResearchPresentation,
      value: {
        content: this.raw_html.slice(0, 200).trim(),
        confidence: Math.round(this.internal_score * 100) / 100,
      },
    };
  }
}

const Analysis = z.object({ summary: z.string() });
type Analysis = Readonly<z.infer<typeof Analysis>>;

// ── Demo ─────────────────────────────────────────────────────────────────
async function main() {
  const renderer = new XmlRenderer();

  // ── 1. exclude=true: hidden from everything ────────────────────────────
  console.log("=".repeat(60));
  console.log("1. neoField(exclude:true) — hidden from BOTH schema and rendering");
  console.log("=".repeat(60));

  const research: ResearchResult = ResearchResult.parse({
    content: "Event sourcing provides auditability",
    source_url: "https://docs.corp/arch",
    score: 0.87,
    retrieval_latency_ms: 42,
    cache_hit: true,
  });

  // PARITY (Direct-ish, GATED on the neoField invention above): the two-pass hoist emitter ports,
  // but pass 1/pass 2 must skip fields whose neoField metadata says exclude !== "none".
  console.log("\nOutput schema (what LLM produces):");
  console.log(describeType(ResearchResult, { prefix: "" }));

  // PARITY (REDESIGN): renderer.render(research) reads model_fields off `research.__class__` in
  // Python to know which fields are `.exclude`. `research` here is a plain parsed object with NO
  // schema back-pointer, so XmlRenderer cannot recover the neoField exclude flags from the VALUE
  // alone. render() must widen to render(value, schema). Divergence from Python's render(value).
  console.log("\nInput rendering (what downstream LLM sees):");
  console.log(renderer.render(research, ResearchResult)); // PARITY: extra schema arg vs Python.

  console.log("\nNote: retrieval_latency_ms and cache_hit appear in NEITHER.");

  // ── 2. ExcludeFromOutput: visible in input, hidden from output ─────────
  console.log("\n" + "=".repeat(60));
  console.log("2. ExcludeFromOutput — visible when reading, hidden when producing");
  console.log("=".repeat(60));

  const condition: ExtensionCondition = ExtensionCondition.parse({
    text: "System shall log all access attempts",
    acceptance_criteria: ["Logs include timestamp", "Logs include user ID"],
    assigned_letter: "B", // set by pipeline, not LLM
  });

  // PARITY: the asymmetry — describeType drops assigned_letter (exclude:"output"|"all") ...
  console.log("\nOutput schema (brainstorm node — LLM produces this):");
  const schema = describeType(ExtensionCondition, { prefix: "" });
  console.log(schema);
  console.assert(!schema.includes("assigned_letter"), "LLM should not see assigned_letter in schema");

  // ... but XmlRenderer KEEPS it (exclude:"output" is not "all"). This split is the entire lesson,
  // and it hinges on the renderer + describeType reading the SAME neoField enum with OPPOSITE
  // gates — a coupling the design doc never anchors. If they drift, the example silently breaks.
  console.log("\nInput rendering (writer node — LLM reads this):");
  const rendered = renderer.render(condition, ExtensionCondition); // PARITY: extra schema arg.
  console.log(rendered);
  console.assert(rendered.includes("assigned_letter"), "Writer must see assigned_letter");
  console.assert(rendered.includes("B"));

  // ── 3. render_for_prompt: complete restructuring ───────────────────────
  console.log("\n" + "=".repeat(60));
  console.log("3. renderForPrompt() -> model — full projection control");
  console.log("=".repeat(60));

  const heavy = new HydratedResearch(
    "<h1>Architecture</h1><p>We chose event sourcing...</p>",
    { author: "team", version: 3 },
    0.92,
  );

  // PARITY (REDESIGN): "direct render" of the HEAVY model. In Python `renderer.render(heavy)`
  // renders ALL fields because render() takes precedence AFTER checking it is not a projection
  // call. With option (a), `heavy` is a class instance; XmlRenderer must read HydratedResearch.schema
  // (the static) rather than a value back-pointer. Yet another render(value, schema) shape.
  console.log("\nDirect render (all fields):");
  console.log(renderer.render(heavy, HydratedResearch.schema));

  // PARITY (REDESIGN): renderInput must (1) detect the renderForPrompt capability on the INSTANCE
  // and (2) consume the {value, schema} it returns. Python just calls `heavy.render_for_prompt()`
  // then renders the returned BaseModel. TS path only works because `heavy` is a class instance;
  // a plain parsed HydratedResearch value would have no method and this whole branch would be dead.
  console.log("\nProjected render (via renderForPrompt):");
  const projected = renderInput(heavy, { renderer });
  console.log(projected);
  console.assert(!projected.includes("raw_html"));
  console.assert(!projected.includes("metadata"));
  console.assert(projected.includes("confidence"));

  // ── 4. Pipeline execution ──────────────────────────────────────────────
  console.log("\n" + "=".repeat(60));
  console.log("4. Full pipeline with ExcludeFromOutput");
  console.log("=".repeat(60));

  // Python: scripted fns are `lambda _i, _c: ...` (input, config). NOTE this uses the PROGRAMMATIC
  // surface (Node.scripted), so AD-0's transformer is IRRELEVANT — no signature to extract.
  // PARITY (Direct): a name->closure map ports 1:1; `_i` is the extracted input, `_c` the config.
  const scripted = {
    make_condition: (_i: unknown, _c: unknown): ExtensionCondition =>
      ExtensionCondition.parse({
        text: "System shall validate inputs",
        acceptance_criteria: ["Reject empty strings"],
      }),
    assign_letter: (_i: ExtensionCondition, _c: unknown): ExtensionCondition =>
      ExtensionCondition.parse({
        text: _i.text,
        acceptance_criteria: _i.acceptance_criteria,
        assigned_letter: "C", // pipeline sets this
      }),
    write_extension: (_i: ExtensionCondition, _c: unknown): Analysis =>
      ({ summary: `Extension for condition ${_i.assigned_letter}: ${_i.text}` }),
  };

  // PARITY (Direct): Node.scripted static factory + Construct nodes list. Maps field-for-field.
  // `inputs`/`outputs` take the Zod SCHEMA object in place of the Pydantic class.
  const pipeline = new Construct("visibility-demo", {
    nodes: [
      Node.scripted("make-condition", { fn: "make_condition", outputs: ExtensionCondition }),
      Node.scripted("assign-letter", {
        fn: "assign_letter",
        inputs: ExtensionCondition,
        outputs: ExtensionCondition,
      }),
      Node.scripted("write-extension", {
        fn: "write_extension",
        inputs: ExtensionCondition,
        outputs: Analysis,
      }),
    ],
  });

  // PARITY (Direct): compile(construct, {scripted, llmFactory, promptCompiler}). The stub
  // llmFactory/promptCompiler (this pipeline is all-scripted) port as trivial arrow fns.
  const graph = compile(pipeline, {
    scripted,
    llmFactory: (_tier: string) => null,
    promptCompiler: (_tmpl: string, _data: unknown) => [],
  });

  // PARITY (Direct): run(graph, {input}). LangGraph.js invoke under the hood.
  const result = await run(graph, { input: { node_id: "example-18" } });
  console.log(`\n  Result: ${(result["write_extension"] as Analysis).summary}`);
  console.log();
  console.log("The writer node saw assigned_letter='C' and used it.");
  console.log("If this were an LLM node, describeType would NOT show assigned_letter.");
}

void main;
