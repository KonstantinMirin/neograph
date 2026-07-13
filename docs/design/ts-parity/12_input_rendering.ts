// TS PARITY SKETCH — HYPOTHETICAL, NON-COMPILING.
// Python source: examples/12_input_rendering.py
// Proposed API:  docs/design/typescript-port.md (AD-0 transformer, .pipe(), Zod schema layer)
//
// This example is a RENDERING-pipeline showcase, NOT a DAG/@node showcase. It exercises:
//   - the renderer dispatch hierarchy (Level 1 model-method > Level 2 node > Level 3 global)
//   - BAML default via describe_value()
//   - field flattening when render_for_prompt() returns a BaseModel
//   - inline (${var.field}) vs template-ref ({var}) prompt semantics
//   - the inspectors: render_input(), render_prompt(), build_rendered_input(), describe_type/value
//
// The proposed TS design doc's Feature Parity Matrix marks "Renderers (XML, Delimited, JSON)"
// and "describe_type" as Direct — but says NOTHING about ANY of the five bullets above.
// That silence is where this example lands most of its friction. See the // PARITY: notes.

import { z } from "zod";
import {
  Node,
  Construct,
  XmlRenderer,
  DelimitedRenderer,
  describeType,
  describeValue,
  renderInput,
  renderPrompt,
  buildRenderedInput,
  LlmRuntime,
  // PARITY GAP: none of the following are named anywhere in typescript-port.md.
  // They have to be invented for this example to even be expressible:
  type Renderable,          // interface a schema-bound class implements to carry render_for_prompt
  bindRenderer,             // side-channel: attach a render method to a Zod schema
} from "@neograph/core";

// -- Schemas ----------------------------------------------------------------
// Python: `class ReadContext(BaseModel, frozen=True)` with Field(description=...).
// PARITY (Direct-ish): Pydantic model -> Zod object. frozen=True -> readonly inferred type
// + Object.freeze at the data boundary (Zod does not freeze parsed output). Field(description=)
// -> .describe(); load-bearing because describe_type/XmlRenderer(include_field_info) emit it.

const ReadContext = z.object({
  title: z.string().describe("Document title"),
  raw: z.string().describe("Raw markdown content"),
  tags: z.array(z.string()).default([]).describe("Classification tags"),
});
type ReadContext = Readonly<z.infer<typeof ReadContext>>;

const Analysis = z.object({
  summary: z.string(),
  key_points: z.array(z.string()),
});
type Analysis = Readonly<z.infer<typeof Analysis>>;

// -- Level 1: Model with render_for_prompt() method -------------------------
// Python: a method ON the BaseModel — `CustomRendered(...).render_for_prompt()`.
//
// PARITY: **REDESIGN, verging on BLOCKED.** This is the crux of the example.
// A Zod schema is a validator, not a class; z.infer<T> yields a PLAIN object with no methods.
// So a parsed value can never carry `render_for_prompt`. Two escape hatches, both worse:
//
//   (a) class-based model (zod-class / a custom base) so instances carry methods —
//       but then it is NOT the same object the rest of the pipe validates/serializes,
//       and the checkpointer (msgpack) round-trip drops the prototype/method anyway.
//   (b) a schema->renderer side-channel registry (bindRenderer below) — decouples the
//       render override from the value, so `renderInput(value)` alone (as Python calls it)
//       can no longer find it; you must thread the schema in too.
//
// Python's dispatch is `hasattr(value, "render_for_prompt")` — a per-INSTANCE capability check.
// TS has no per-instance capability on a plain data object. Sketching option (a):

class CustomRendered implements Renderable {
  constructor(readonly name: string, readonly content: string) {}
  static schema = z.object({ name: z.string(), content: z.string() });
  renderForPrompt(): string {
    return `=== ${this.name} ===\n${this.content}`;
  }
}
// PARITY: the design doc's renderer row assumes renderers only ever consume schemas/plain
// values. The model-method LEVEL of the dispatch hierarchy has no mention and no home.

// -- Fake LLM + prompt compiler (no API keys) -------------------------------
// PARITY (Direct): withStructuredOutput exists in LangChain.js; the fake is trivial.
class FakeLLM {
  private model!: z.ZodTypeAny;
  withStructuredOutput(model: z.ZodTypeAny) {
    this.model = model;
    return this;
  }
  invoke(_messages: unknown): Analysis {
    return { summary: "Example summary", key_points: ["point-1"] };
  }
}

// Python: `def simple_compiler(template, data, **kwargs)`.
// PARITY (Direct): plain function; **kwargs -> a trailing `opts?: Record<string, unknown>`.
function simpleCompiler(template: string, data: unknown): Array<{ role: string; content: string }> {
  return [{ role: "user", content: `[${template}]\n${data}` }];
}

// Python: LlmRuntime.build(llm_factory=..., prompt_compiler=..., renderer=XmlRenderer())
// PARITY (Redesign-ish): LlmRuntime is NOT in the parity matrix (only configure_llm() is).
// The Level-3 resolution `node.renderer or runtime.renderer or None` must be specced somewhere;
// here it lives on a runtime object like Python.
const runtime = LlmRuntime.build({
  llmFactory: (_tier: string) => new FakeLLM(),
  promptCompiler: simpleCompiler,
  renderer: new XmlRenderer(), // Level 3: global default
});

// -- Pipeline ---------------------------------------------------------------
// Python: programmatic Node(...) — NOTE this example uses the PROGRAMMATIC surface, not @node,
// so AD-0's transformer is IRRELEVANT here. Node config maps field-for-field.
// PARITY (Direct): this is the "TS-first" surface per AD-2.
const analyze = new Node("analyze", {
  mode: "think",
  outputs: Analysis,
  model: "fast",
  prompt: "analyze",
  renderer: new DelimitedRenderer(), // Level 2: node-level override
});

const pipeline = new Construct("render-demo", { nodes: [analyze] });
void pipeline;

// -- Run --------------------------------------------------------------------
async function main() {
  const doc: ReadContext = ReadContext.parse({
    title: "Architecture Decision Record",
    raw: "We chose event sourcing for auditability.\nCQRS separates reads from writes.",
    tags: ["architecture", "decision"],
  });

  // Level 1: model method wins over any renderer.
  // PARITY (REDESIGN): renderInput must accept a method-carrying instance. With option (a)
  // above, `custom` is a CustomRendered INSTANCE (not a parsed Zod value), so this call site
  // diverges from every OTHER renderInput call (which take plain objects). Heterogeneous input.
  const custom = new CustomRendered("spec", "The system shall log all access.");
  console.log("Level 1 (model method):");
  console.log(renderInput(custom, { renderer: new XmlRenderer() }));
  console.log();

  // Level 2: node renderer (DelimitedRenderer on `analyze`).
  // PARITY (Direct): renderPrompt(node, data, {runtime}) — inspector maps cleanly.
  console.log("Level 2 (node renderer) via renderPrompt inspector:");
  console.log(renderPrompt(analyze, doc, { runtime }));
  console.log();

  // Level 3: global renderer (XmlRenderer via the runtime).
  const plainNode = new Node("plain", { mode: "think", outputs: Analysis, model: "fast", prompt: "p" });
  console.log("Level 3 (global renderer) via renderPrompt inspector:");
  console.log(renderPrompt(plainNode, doc, { runtime }));
  console.log();

  // describe_type for output schema.
  // PARITY (Direct, per doc — "already TS-native notation"). Caveat: describeType(Analysis)
  // takes the SCHEMA, and must read `.describe()` field docs + hoist repeated nested schemas
  // out of Zod's ._def. Doable but the two-pass hoist logic is non-trivial to port.
  console.log("Output schema (describeType):");
  console.log(describeType(Analysis));
  console.log();

  // ===================================================================
  // Section: BAML default rendering (no renderer configured)
  // ===================================================================
  console.log("=".repeat(60));
  console.log("BAML DEFAULT (no renderer)");
  console.log("=".repeat(60));

  // Python: describe_value(doc) — renders an INSTANCE as a typed value literal.
  // PARITY (API GAP / REDESIGN): describe_value's whole job is "value + its TYPE -> literal".
  // In Python the instance IS its type (pydantic carries model_fields). A TS plain object has
  // NO attached schema, so describeValue(doc) alone can't know field types/descriptions.
  // Signature MUST widen to describeValue(value, schema). Every call site below diverges.
  console.log("\ndescribeValue(doc) — raw BAML notation:");
  console.log(describeValue(doc, ReadContext)); // PARITY: extra `ReadContext` arg vs Python.
  console.log();

  console.log("renderInput(doc, {renderer: null}) — same BAML output:");
  console.log(renderInput(doc, { renderer: null, schema: ReadContext })); // PARITY: schema needed.
  console.log();

  console.log("renderInput(doc, {renderer: XmlRenderer}) — XML for comparison:");
  console.log(renderInput(doc, { renderer: new XmlRenderer(), schema: ReadContext }));
  console.log();

  // ===================================================================
  // Section: Field flattening via render_for_prompt() returning BaseModel
  // ===================================================================
  // Python: render_for_prompt() returns a SourceDoc BaseModel, whose fields
  // (headline/body/provenance) become individually addressable template vars.
  //
  // PARITY (REDESIGN, compounds the Level-1 problem): the returned value must ALSO be a
  // field-introspectable model so build_rendered_input can flatten it. In TS the method must
  // return { value, schema } (or a class instance with a static schema) so the framework can
  // read `.shape` to flatten. A bare returned object gives keys but no types.
  const SourceDoc = z.object({
    headline: z.string(),
    body: z.string(),
    provenance: z.string(),
  });

  class ResearchReport implements Renderable {
    constructor(
      readonly title: string,
      readonly raw_content: string,
      readonly source_url: string,
    ) {}
    renderForPrompt(): { value: z.infer<typeof SourceDoc>; schema: typeof SourceDoc } {
      // PARITY: must hand back the schema so the flattener can read field names/types.
      return {
        schema: SourceDoc,
        value: {
          headline: this.title.toUpperCase(),
          body: this.raw_content,
          provenance: `Source: ${this.source_url}`,
        },
      };
    }
  }

  const report = new ResearchReport(
    "Quarterly Review",
    "Revenue grew 12% year-over-year.",
    "https://example.com/q4",
  );

  console.log("=".repeat(60));
  console.log("FIELD FLATTENING (renderForPrompt returns a model)");
  console.log("=".repeat(60));

  // PARITY (Direct-ONCE the above redesign is accepted): RenderedInput itself — raw/rendered/
  // flattened/for_template_ref/available_keys_inline/available_keys_template — is a plain data
  // record and ports 1:1. But the WHOLE abstraction is absent from the parity matrix.
  const ri = buildRenderedInput({ report }, { renderer: null });
  console.log("\nRenderedInput.rendered (whole model, BAML):");
  console.log(ri.rendered);
  console.log("\nRenderedInput.flattened (individual fields from SourceDoc):");
  for (const [k, v] of Object.entries(ri.flattened)) console.log(`  ${k}: ${JSON.stringify(v)}`);
  console.log("\nRenderedInput.forTemplateRef (merged — what the prompt compiler sees):");
  for (const [k, v] of Object.entries(ri.forTemplateRef)) console.log(`  ${k}: ${JSON.stringify(v)}`);
  console.log("\navailableKeysTemplate:", [...ri.availableKeysTemplate].sort());
  console.log("availableKeysInline:  ", [...ri.availableKeysInline].sort());
  console.log();

  // ===================================================================
  // Section: Inline prompt vs template-ref prompt
  // ===================================================================
  console.log("=".repeat(60));
  console.log("INLINE vs TEMPLATE-REF prompt rendering");
  console.log("=".repeat(60));

  // Python: ri_baml.raw is the RAW model — inline ${var.field} does getattr chains on it.
  // PARITY (Direct, actually EASIER): a plain JS object supports `raw.title` natively; no
  // getattr machinery. The inline-vs-template string heuristic (has spaces / has ${...})
  // ports as a plain regex. This half of the section is the smoothest port in the file.
  console.log("\n--- Same data, two prompt styles ---\n");
  const riBaml = buildRenderedInput(doc, { renderer: null, schema: ReadContext });
  console.log("RenderedInput.raw (what inline ${var.field} sees):");
  console.log(`  type: ${typeof riBaml.raw}`);
  console.log(`  riBaml.raw.title = ${JSON.stringify((riBaml.raw as ReadContext).title)}`);
  console.log(`  riBaml.raw.tags  = ${JSON.stringify((riBaml.raw as ReadContext).tags)}`);
  console.log();

  console.log("RenderedInput.rendered (template-ref sees — BAML string):");
  console.log(riBaml.rendered);
  console.log();

  const riXml = buildRenderedInput(doc, { renderer: new XmlRenderer(), schema: ReadContext });
  console.log("RenderedInput.rendered with XmlRenderer (template-ref sees XML):");
  console.log(riXml.rendered);
  console.log();

  // Dict-form (fan-in) — two upstream values, both views.
  // PARITY (Direct on the container; each value STILL needs its schema for BAML). Python
  // infers each value's type from the instance; TS dict-form must carry a parallel schema map.
  const claimsModel: Analysis = { summary: "Strong growth", key_points: ["12% rev", "new market"] };
  const riDict = buildRenderedInput(
    { doc, prior: claimsModel },
    { renderer: null, schemas: { doc: ReadContext, prior: Analysis } }, // PARITY: extra schemas map.
  );
  console.log("Dict-form fan-in (two upstream values, BAML default):");
  console.log("  Inline keys (raw objects): ", [...riDict.availableKeysInline].sort());
  console.log("  Template keys (rendered):  ", [...riDict.availableKeysTemplate].sort());
  console.log();
  console.log("  riDict.raw['doc'].title =", (riDict.raw as Record<string, ReadContext>).doc.title);
}

void main;
