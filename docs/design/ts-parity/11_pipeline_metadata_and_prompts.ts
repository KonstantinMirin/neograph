// TS parity sketch of: examples/11_pipeline_metadata_and_prompts.py
// HYPOTHETICAL — targets the proposed API in docs/design/typescript-port.md.
// Not compilable/runnable; it exists to ground a feature-parity analysis.
//
// This example is the DECLARATIVE / PROGRAMMATIC surface (Node(...) + Node.scripted()
// + Construct(...) + compile()/run()), which the design doc calls the "TS-first surface"
// (Feature Parity Matrix, "Three API Surfaces" -> Programmatic = Direct). No @node, so the
// AD-0 compiler transformer is NOT exercised here. The friction that DOES surface is entirely
// in the compile()/run() SEAMS: the prompt_compiler's variadic signature, the untyped
// config["configurable"] bag, and output_model being a Pydantic *class* with methods.

import { z } from "zod";
import { Node, Construct, compile, run } from "@neograph/core";
import { zodToJsonSchema } from "zod-to-json-schema"; // PARITY: Zod has no model_json_schema(); needs this lib

// ── Schemas ──────────────────────────────────────────────────────────────
// Python: class Claims(BaseModel, frozen=True): items: list[str]
// PARITY (Direct-ish): Zod objects are the schema layer (AD-1). `frozen=True` has no
// Zod equivalent — Zod validates shape but does not produce frozen instances. You'd
// Object.freeze() the parse result manually if immutability matters. Low friction.
const Claims = z.object({ items: z.array(z.string()) });
type Claims = z.infer<typeof Claims>;

const Report = z.object({ text: z.string() });
type Report = z.infer<typeof Report>;

// ══════════════════════════════════════════════════════════════════════════
// SHARED RESOURCES — consumer's infrastructure passed through config
// ══════════════════════════════════════════════════════════════════════════
// Python: a plain class stashed into config["configurable"]["rate_limiter"].
// PARITY (Direct): a plain class/object threaded through config is idiomatic TS.
class RateLimiter {
  calls = 0;
  constructor(public maxRpm: number) {}
  call<T>(fn: () => T): T {
    this.calls += 1;
    return fn();
  }
}

// ══════════════════════════════════════════════════════════════════════════
// PROMPT COMPILER — receives full context from neograph
// ══════════════════════════════════════════════════════════════════════════
// Python signature (keyword-only, introspection-gated by the runtime):
//   (template, data, *, node_name=None, config=None, output_model=None, llm_config=None) -> messages
//
// PARITY (REDESIGN — the sharpest divergence in this example):
//   1. Python's `*,` keyword-only params become a single options object. Mechanical.
//   2. BLOCKED mechanism: neograph's runtime calls the compiler through `_accepted_params`
//      (inspect.signature) so a compiler that omits `output_model`/`di_inputs` simply never
//      receives them (the di_inputs opt-in seam, CLAUDE.md "di_inputs"). TS has NO reliable
//      runtime introspection of a function's declared parameter names (minifiers rename them;
//      no design:paramtypes — see AD-0 "Ruled out" table). So the proposed TS API CANNOT
//      replicate "only pass X if the compiler declared X". The only TS-idiomatic path is to
//      ALWAYS pass the full options bag and let the compiler ignore what it doesn't read.
//      That loses the gate's purpose (a fail-soft compiler that ignores a field it should
//      have consumed silently ships a literal placeholder — the "agent-stark" incident this
//      gate was built to prevent). Design doc's Parity Matrix does not mention the compiler
//      seam or _accepted_params at all — this is a GAP.
interface PromptCompilerCtx {
  nodeName?: string;
  config?: RunConfig;
  outputModel?: z.ZodTypeAny; // PARITY: a Zod *value*, not a class with .model_json_schema()
  llmConfig?: Record<string, unknown>;
}
type Message = { role: string; content: string };

function myPromptCompiler(
  template: string,
  data: unknown,
  ctx: PromptCompilerCtx = {},
): Message[] {
  // Python: configurable = (config or {}).get("configurable", {})
  const configurable = ctx.config?.configurable ?? {};

  // PARITY (low friction, TYPE EROSION): config["configurable"] is a heterogeneous bag
  // mixing run-input scalars (node_id, project_root) with resource objects (rate_limiter).
  // In Python it's dict[str, Any]; in TS the honest type is Record<string, unknown>, so every
  // read needs a cast. A TS user would *want* a typed config generic here, which the proposed
  // API does not offer (see api_gaps). Same untyped-bag ergonomics as Python, just louder.
  const nodeId = (configurable["node_id"] as string | undefined) ?? "unknown";
  const projectRoot = (configurable["project_root"] as string | undefined) ?? ".";
  const strategy = (ctx.llmConfig?.["output_strategy"] as string | undefined) ?? "structured";

  console.log(
    `  [prompt] template=${template}, node=${ctx.nodeName}, node_id=${nodeId}, strategy=${strategy}`,
  );

  let prompt: string;
  if (template === "decompose") {
    prompt =
      `Decompose requirement ${nodeId} from ${projectRoot} into claims. ` +
      `Previous analysis: ${JSON.stringify(data)}`;
  } else if (template === "summarize") {
    prompt = `Summarize: ${JSON.stringify(data)}`;
  } else {
    prompt = String(data);
  }

  const messages: Message[] = [{ role: "user", content: prompt }];

  // Python: schema = json.dumps(output_model.model_json_schema(), indent=2)
  // PARITY (REDESIGN): output_model is a Pydantic CLASS in Python — it both (a) exposes
  // .model_json_schema() and (b) is CALLABLE as a constructor (see FakeLLM below). A Zod
  // schema is a plain value: no method to call, and it PARSES rather than CONSTRUCTS. Schema
  // emission moves to the zod-to-json-schema lib. The "describe_type = Direct" row in the
  // Parity Matrix hides this: the seam contract (what type `output_model` IS) changes shape.
  if ((strategy === "json_mode" || strategy === "text") && ctx.outputModel) {
    const schema = JSON.stringify(zodToJsonSchema(ctx.outputModel), null, 2);
    messages.push({
      role: "user",
      content: `Return ONLY a valid JSON object matching this schema:\n${schema}`,
    });
  }

  return messages;
}

// ══════════════════════════════════════════════════════════════════════════
// LLM FACTORY — also receives node context
// ══════════════════════════════════════════════════════════════════════════
// Python FakeLLM: with_structured_output(model) stashes the class; invoke() calls it as a
//   constructor: self._model(items=["claim-1","claim-2"]).
// PARITY (REDESIGN): the "model" is a Zod schema, not a constructor. Producing a fake value
// means `schema.parse({ items: [...] })`, not `Model(items=[...])`. LangChain.js's real
// withStructuredOutput(zodSchema) returns a parsed object, so this maps cleanly to the real
// path — but the *fake* has to switch from call-the-class to parse-the-schema.
class FakeLLM {
  private model!: z.ZodTypeAny;
  withStructuredOutput(model: z.ZodTypeAny): this {
    this.model = model;
    return this;
  }
  invoke(_messages: Message[]): unknown {
    return this.model.parse({ items: ["claim-1", "claim-2"] });
  }
}

// Python: my_llm_factory(tier, node_name=None, llm_config=None)
// PARITY (Direct): tier-based factory maps 1:1 (Parity Matrix "Tier-based LLM creation").
// Optional trailing params become an options object again.
function myLlmFactory(
  tier: string,
  opts: { nodeName?: string; llmConfig?: Record<string, unknown> } = {},
): FakeLLM {
  console.log(`  [factory] tier=${tier}, node=${opts.nodeName}, config=${JSON.stringify(opts.llmConfig)}`);
  return new FakeLLM();
}

// ══════════════════════════════════════════════════════════════════════════
// SCRIPTED NODE — accesses pipeline metadata from config
// ══════════════════════════════════════════════════════════════════════════
// Python: def build_report(input_data, config): ... reads config["configurable"].
// PARITY (Direct): the scripted (state, config) signature ports 1:1 (Parity Matrix
// "Node.scripted() factory = Direct", "Scripted wrapper = Direct"). This node deliberately
// reads config MANUALLY rather than via DI — so none of the AD-0 DI-branded-type machinery
// is needed here. The only erosion is, again, the untyped configurable bag.
function buildReport(inputData: Claims, config: RunConfig): Report {
  const configurable = config.configurable ?? {};
  const nodeId = configurable["node_id"] as string | undefined;
  const projectRoot = configurable["project_root"] as string | undefined;

  const rateLimiter = configurable["rate_limiter"] as RateLimiter | undefined;
  if (rateLimiter) {
    console.log(`  [report] rate_limiter has made ${rateLimiter.calls} calls`);
  }

  return Report.parse({
    text: `Report for ${nodeId} at ${projectRoot}: ${JSON.stringify(inputData.items)}`,
  });
}

// ══════════════════════════════════════════════════════════════════════════
// PIPELINE
// ══════════════════════════════════════════════════════════════════════════
// Python:
//   decompose = Node(name="decompose", mode="think", outputs=Claims, model="fast", prompt="decompose")
//   report    = Node.scripted("report", fn="build_report", inputs=Claims, outputs=Report)
//   pipeline  = Construct("metadata-demo", nodes=[decompose, report])
//
// PARITY (Direct): the programmatic Node/Construct constructors map 1:1 (Parity Matrix marks
// the Programmatic surface Direct and the "TS-first surface"). `outputs=Claims` (a Pydantic
// class) becomes `outputs: Claims` (a Zod value) — same slot, value-vs-class is invisible here.
// String-keyed scripted reference `fn: "build_report"` + the compile(scripted={...}) registry
// is a plain object/Map in TS — Direct ("Registries = module-level Maps").
const decompose = new Node({
  name: "decompose",
  mode: "think",
  outputs: Claims,
  model: "fast",
  prompt: "decompose",
});

const report = Node.scripted("report", {
  fn: "build_report",
  inputs: Claims,
  outputs: Report,
});

const pipeline = new Construct("metadata-demo", { nodes: [decompose, report] });

// ══════════════════════════════════════════════════════════════════════════
// RUN — consumer passes everything through input + config
// ══════════════════════════════════════════════════════════════════════════
// PARITY: config type. Python threads dict[str, Any]; here it's an untyped bag by necessity.
interface RunConfig {
  configurable?: Record<string, unknown>;
}

async function main() {
  const limiter = new RateLimiter(60);

  // Python: compile(pipeline, llm_factory=..., prompt_compiler=..., scripted={"build_report": build_report})
  // PARITY (Direct): all three seams are just options on compile(). The scripted map is a
  // plain object keyed by the same string used in Node.scripted(fn=...).
  const graph = compile(pipeline, {
    llmFactory: myLlmFactory,
    promptCompiler: myPromptCompiler,
    scripted: { build_report: buildReport },
  });

  console.log("Running pipeline:\n");

  // Python: run(graph, input={node_id, project_root}, config={"configurable": {rate_limiter}})
  //   -> input fields land in BOTH state AND config["configurable"]; extra config merges in.
  // PARITY (Direct, runtime merge): this is a plain dict-merge in run()'s runner, no type
  // machinery — ports 1:1 ("Runner (run + config injection) = Direct"). The DX cost is that
  // `input` here carries pipeline METADATA (node_id/project_root), not typed upstream state,
  // so it's Record<string, unknown> on both sides — TS can't type the input->configurable
  // fan-out because the two consumers (state bus vs config bag) want different shapes.
  const result = await run(graph, {
    input: { node_id: "BR-RW-042", project_root: "/my/project" },
    config: { configurable: { rate_limiter: limiter } },
  });

  console.log(`\nResult: ${(result["report"] as Report).text}`);
  console.log("\nAll config accessible: node_id, project_root, rate_limiter — no boilerplate");
}

void main();
