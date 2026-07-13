// TS parity sketch of: examples/21_multimodal_vision.py
// Feature focus: ${image:field} multimodal content-block assembly, construct_from_module,
//   scripted seed node, think-mode structured output, programmatic Node/Construct surface,
//   custom llm_factory + prompt_compiler.
//
// This is a HYPOTHETICAL port against the PROPOSED API in docs/design/typescript-port.md
// (AD-0 ts-patch/typia transformer form). It is NOT meant to compile or run. Inline
// `// PARITY:` notes flag every place the TS DX diverges from the Python original.
//
// Key upfront finding: the design doc's Feature Parity Matrix lists
// "Inline prompt substitution ${var} — Direct" but NEVER mentions the `${image:...}`
// multimodal variant, which is the entire point of this example. See PARITY-IMAGE below.

import { z } from "zod";
import {
  node,
  construct,          // PARITY: replaces construct_from_module (see PARITY-MODULE)
  Node,
  Construct,
  compile,
  run,
  configureImage,     // PARITY: neograph.configure_image equivalent (proposed, not in matrix)
} from "@neograph/core";
import { readFileSync, writeFileSync } from "node:fs";     // PARITY-FS: Node-only
import { tmpdir } from "node:os";
import { join } from "node:path";

// ── Schemas ──────────────────────────────────────────────────────────────
// Python: class ProductPhoto(BaseModel, frozen=True) → Zod object.
// PARITY: `frozen=True` immutability has no direct Zod equivalent — Zod validates
//   shape, not mutability. Use TS `readonly` on the inferred type + Object.freeze if
//   you need runtime immutability. Advisory only; low friction.

const ProductPhoto = z.object({
  imageData: z.string(),     // PARITY: Python `image_data` (snake) → `imageData` (camel).
  productName: z.string(),   //   Field-name casing convention differs; the ${image:seed.image_data}
});                          //   dotted path in prompts must match whichever casing wins.
type ProductPhoto = z.infer<typeof ProductPhoto>;

const ProductMetadata = z.object({
  category: z.string().describe("Product category (e.g., electronics, furniture)"),
  qualityScore: z.number().describe("Image quality 0-1"),   // Python Field(description=...) → .describe()
  description: z.string().describe("One-sentence product description"),
});
type ProductMetadata = z.infer<typeof ProductMetadata>;

// ── Fake VLM ─────────────────────────────────────────────────────────────
// Python FakeVisionLLM.with_structured_output(model) + invoke(messages). In TS the
// structured-output contract is Zod-based (LangChain.js withStructuredOutput(zodSchema)).
// PARITY: Direct — the fake shape maps cleanly. The interesting part is `messages`:
//   for the ${image:...} path `content` is a content-block ARRAY, not a string. Same
//   shape as LangChain.js multimodal content, so runtime is portable.

class FakeVisionLLM {
  private model!: z.ZodTypeAny;
  constructor(private tier: string) {}

  withStructuredOutput(model: z.ZodTypeAny) {
    this.model = model;
    return this;
  }

  invoke(messages: Array<{ role: string; content: string | Array<Record<string, any>> }>) {
    const content = messages[0].content;
    let hasImage = false;
    let text = "";
    if (Array.isArray(content)) {
      hasImage = content.some((b) => b.type === "image_url");
      text = content.filter((b) => b.type === "text").map((b) => b.text).join(" ");
      console.log(`    VLM received: ${content.length} content blocks, has_image=${hasImage}, text_preview='${text.slice(0, 50)}...'`);
    } else {
      text = content;
      console.log("    VLM received: plain text, no image");
    }
    return {
      category: hasImage ? "electronics" : "unknown",
      qualityScore: hasImage ? 0.92 : 0.0,
      description: hasImage ? "High-quality product photo" : "No image provided",
    } as ProductMetadata;
  }
}

const llmFactory = (tier: string) => new FakeVisionLLM(tier);

// Python: def _prompt_compiler(t, d, **kw): return [{"role": "user", "content": t}]
// PARITY: `**kw` → a typed options object / rest. Direct.
// PARITY-COMPILER-BYPASS: In neograph, when a prompt contains ${image:...}, the multimodal
//   compiler (_compile_multimodal_prompt) builds the message list DIRECTLY and BYPASSES the
//   user prompt_compiler entirely. So this compiler only ever fires for the plain-text path.
//   The TS port must replicate that bypass, or the content-block array would be double-wrapped.
//   The matrix's "prompt_compiler — Direct" line hides this branch.
const promptCompiler = (t: string | Array<Record<string, any>>, _d: unknown, ..._kw: unknown[]) => [
  { role: "user", content: t },
];

// ── Pipeline (declarative @node surface) ───────────────────────────────────
// PARITY-WRAPPER: Python `@node(outputs=ProductPhoto) def load_photo() -> ProductPhoto`
//   becomes the wrapper form. Zero-param scripted node; the transformer (AD-0) reads the
//   return annotation `: ProductPhoto` as the output type. Body runs (scripted mode inferred:
//   no prompt/model). Direct with transformer.

const loadPhoto = node(
  { outputs: ProductMetadata /* PARITY: still needed? see note */ },
  (): ProductPhoto => {
    // PARITY-FS: Python uses tempfile.NamedTemporaryFile(suffix=".png"). The Node fs port
    //   works on server runtimes, but on edge/browser/Bun-serverless there is NO filesystem —
    //   file-path image input is silently non-portable. The design doc never scopes runtime.
    const tmp = join(tmpdir(), `neo-${Date.now()}.png`);
    const png = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
                               Buffer.from("fake-product-image-data".repeat(10))]);
    writeFileSync(tmp, png);
    return { imageData: tmp, productName: "Wireless Headphones" };
  }
);
// PARITY: the wrapper's first arg { outputs: ... } is redundant IF the transformer extracts
//   the return type — but for a zero-param arrow with a stub body the transformer CAN read the
//   annotation, so `outputs` should be omittable. Kept here to show the fallback-explicit form.

// PARITY-DEADBODY: Python think-node body is `...` (dead code, never runs). TS has no `...`
//   ellipsis statement — you must write a throwing stub. AND the transformer needs the explicit
//   `: ProductMetadata` return annotation because it cannot infer a return type from a body that
//   only throws. The Python dead-body UserWarning ("Skip" in the matrix) has no analogue; there
//   is no lint that this body is intentionally dead.
const classify = node(
  {
    outputs: ProductMetadata,
    model: "fast",
    // PARITY-IMAGE: THE marquee feature. `${image:load_photo.image_data}` is NOT covered by
    //   the matrix's "${var} — Direct" row. The image variant additionally requires:
    //     (1) a distinct grammar in the inline-prompt scanner (_IMAGE_RE) that the TS port must
    //         re-implement — it splits text/image segments and emits {type:'text'} / {type:'image_url'} blocks;
    //     (2) resolve_image(): file-path read + magic-byte MIME sniff (PNG/JPEG/GIF/WEBP/BMP/SVG)
    //         + base64 encode + data-URI wrap + size cap + allowed_dirs sandbox (configureImage);
    //     (3) fail-soft behavior (empty data-uri on missing file, never crash).
    //   None of this is a "different syntax, same logic" port — it is a whole subsystem the
    //   matrix omits. Classified BLOCKED/GAP below.
    prompt: "Analyze this product photo for ${load_photo.product_name}: ${image:load_photo.image_data}",
  },
  // PARITY-EDGE: param name `loadPhoto` IS the edge to the upstream node `loadPhoto`. The
  //   transformer must preserve the param identifier. But note the dotted prompt path is
  //   `${load_photo.product_name}` — snake_case in Python. In TS the node is `loadPhoto`, so
  //   the inline path must become `${loadPhoto.productName}`. Prompt strings are NOT type-checked,
  //   so a casing mismatch fails silently at runtime (fail-soft leaves the literal in the prompt).
  (loadPhoto: ProductPhoto): ProductMetadata => {
    throw new Error("dead body — think mode, VLM produces ProductMetadata");
  }
);

// PARITY-MODULE: Python `construct_from_module(sys.modules[__name__], name="product-catalog")`
//   auto-discovers every @node in the module via reflection. The design doc EXPLICITLY drops this
//   ("construct_from_module (no module introspection in TS; use explicit lists)"). So the TS DX
//   forces an explicit node list — a real regression for this example, whose ENTIRE main-pipeline
//   assembly is one module-introspection call.
const pipeline = construct("product-catalog", [loadPhoto, classify]);

// ── Run (declarative path) ─────────────────────────────────────────────────
async function main() {
  // PARITY: optional global image sandbox — proposed configureImage() mirrors configure_image().
  configureImage({ maxBytes: 10 * 1024 * 1024, allowedDirs: null, validateFormat: true });

  const graph = compile(pipeline, { llmFactory, promptCompiler });
  const result = await run(graph, { input: { node_id: "product-001" } });

  // PARITY-STATEKEY: Python reads result["classify"]; the state field is the node name.
  //   Direct — but see the hyphen-normalization subtlety in the b64 path below.
  const meta = result["classify"] as ProductMetadata;
  console.log("\n=== Product Metadata (from VLM) ===");
  console.log(`  Category:    ${meta.category}`);
  console.log(`  Quality:     ${meta.qualityScore}`);
  console.log(`  Description: ${meta.description}`);

  // ── Base64 input via the PROGRAMMATIC surface (part 2) ───────────────────
  // PARITY: this is the matrix's "TS-first" programmatic surface — maps most cleanly.
  console.log("\n=== Base64 Input (no file) ===");

  // Python: base64.b64encode(b"...").decode() → Buffer (Node-only again, PARITY-FS).
  const b64 = Buffer.from("\x89PNG\r\n\x1a\nbase64-product-photo", "binary").toString("base64");

  // Python raw scripted shim: def _b64_seed(i, c): return ProductPhoto(...)
  //   Signature is (input_data, config) — the low-level shim contract, not the @node signature.
  // PARITY: Direct. The scripted registry is a string-keyed Map passed to compile().
  const b64Seed = (_i: unknown, _c: unknown): ProductPhoto => ({
    imageData: b64,
    productName: "Smart Watch",
  });

  // Python: Node.scripted("seed", fn="_b64_seed", outputs=ProductPhoto)
  const seed = Node.scripted("seed", { fn: "_b64_seed", outputs: ProductPhoto });

  // Python: Node("classify-b64", mode="think", outputs=..., prompt=..., model=..., inputs={"seed": ProductPhoto})
  // PARITY: dict-form inputs → { seed: ProductPhoto } with Zod schemas. Direct.
  //   The prompt again uses ${image:seed.image_data} → BLOCKED/GAP per PARITY-IMAGE.
  const classifyNode = new Node("classify-b64", {
    mode: "think",
    outputs: ProductMetadata,
    prompt: "Classify this product: ${image:seed.imageData}",   // PARITY casing: image_data → imageData
    model: "fast",
    inputs: { seed: ProductPhoto },
  });

  const b64Pipeline = new Construct("b64-catalog", { nodes: [seed, classifyNode] });
  const graph2 = compile(b64Pipeline, {
    llmFactory,
    promptCompiler,
    // PARITY: Python `scripted={"_b64_seed": _b64_seed}` → a Map/record. Direct. The string
    //   `fn="_b64_seed"` on Node.scripted resolves against this at compile — string-keyed
    //   indirection ports fine (it is intentional runtime-assembly wiring, not reflection).
    scripted: { _b64_seed: b64Seed },
  });
  const result2 = await run(graph2, { input: { node_id: "product-002" } });

  // PARITY-NORMALIZE: node name "classify-b64" → state field "classify_b64" (hyphen→underscore
  //   normalization). Python does result2["classify_b64"]. The TS port must replicate the exact
  //   naming.normalize() rule or this key lookup silently returns undefined. Direct but load-bearing.
  const meta2 = result2["classify_b64"] as ProductMetadata;
  console.log(`  Category:    ${meta2.category}`);
  console.log(`  Quality:     ${meta2.qualityScore}`);
  console.log(`  Description: ${meta2.description}`);

  console.log("\nBoth file paths and base64 strings work with ${image:field}.");
}

main();
