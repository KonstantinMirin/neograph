// TS parity sketch of: examples/22_image_security.py
// HYPOTHETICAL — written against the PROPOSED @neograph/core TS API
// (docs/design/typescript-port.md). NOT compilable/runnable. It exists to
// surface where the TS DX diverges from Python for the image-security feature.
//
// This example exercises the PROGRAMMATIC surface (Node.scripted / Node(think)
// / Construct / compile / run) — the "TS-first surface" per AD-2 — plus the
// image subsystem: configure_image(), resolve_image(), magic-byte validation,
// allowed_dirs path containment, and the ${image:...} multimodal inline
// directive. NONE of the image subsystem appears in the proposed parity matrix,
// so most PARITY notes below flag GAPS, not just syntax deltas.

import { z } from "zod";
import {
  Node,
  Construct,
  compile,
  run,
  // PARITY-GAP: configureImage / resolveImage are NOT in the parity matrix
  // (typescript-port.md lists no image rows at all). Signatures below are
  // INVENTED to make the port concrete.
  configureImage,
  resolveImage,
} from "@neograph/core";
import type { ScriptedInput, RunConfig, PromptCompiler } from "@neograph/core";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ── Schemas ──────────────────────────────────────────────────────────────
// Python: frozen Pydantic models. TS: Zod objects (AD-1). `.readonly()` is the
// nearest analog to frozen=True but only affects the inferred TS type, not
// runtime immutability — a low-severity semantic loss.

const ImageInput = z.object({ photo: z.string() }).readonly();
type ImageInput = z.infer<typeof ImageInput>;

const Analysis = z.object({ result: z.string() }).readonly();
type Analysis = z.infer<typeof Analysis>;

// ── Fake LLM ─────────────────────────────────────────────────────────────
// Python duck-types with_structured_output/invoke. TS: LangChain.js expects a
// BaseChatModel; a fake must extend it or satisfy the withStructuredOutput
// contract. The Python fake inspects messages[0].content for an image_url block
// whose url !== the empty sentinel "data:image/png;base64,". That block shape
// comes straight from _compile_multimodal_prompt (see PARITY-MULTIMODAL below).
class FakeLLM {
  constructor(_tier: string) {}

  withStructuredOutput<T extends z.ZodTypeAny>(model: T) {
    return {
      invoke: async (messages: Array<{ role: string; content: unknown }>) => {
        const content = messages[0]?.content;
        const hasImage =
          Array.isArray(content) &&
          content.some(
            (b: any) =>
              b?.type === "image_url" &&
              b?.image_url?.url !== "data:image/png;base64,",
          );
        return model.parse({ result: hasImage ? "analyzed" : "no-image" });
      },
    };
  }
}

const llmFactory = (tier: string) => new FakeLLM(tier);

// Python: def _prompt_compiler(t, d, **kw) -> [{"role":"user","content":t}]
// PARITY-MULTIMODAL: this app compiler is a red herring for the image path.
// For an inline ${image:...} prompt, neograph builds the message list itself in
// _compile_multimodal_prompt (_llm_render.py:111) and NEVER calls this
// compiler. The compiler only matters for pure-text prompts. The TS port must
// replicate that framework-owned multimodal assembly, not lean on the compiler.
const promptCompiler: PromptCompiler = (t, _d) => [{ role: "user", content: t }];

// ── Set up a safe upload directory ───────────────────────────────────────
// Direct: node:fs/os mirror tempfile/os/pathlib 1:1.

const UPLOAD_DIR = fs.mkdtempSync(path.join(os.tmpdir(), "uploads_"));
console.log(`Upload directory: ${UPLOAD_DIR}`);

const legitImage = path.join(UPLOAD_DIR, "product.png");
fs.writeFileSync(
  legitImage,
  Buffer.concat([Buffer.from("\x89PNG\r\n\x1a\n", "binary"), Buffer.from("real-product-photo".repeat(5))]),
);

// Python used NamedTemporaryFile(delete=False). TS: write into os.tmpdir()
// directly (OUTSIDE UPLOAD_DIR) — that "outside allowed_dirs" placement is the
// whole point of case 2.
const secretFile = path.join(os.tmpdir(), `secret_${Date.now()}.png`);
fs.writeFileSync(secretFile, Buffer.from("\x89PNG\r\n\x1a\nthis-is-actually-sensitive-data", "binary"));

const configFile = path.join(UPLOAD_DIR, "oops.txt");
fs.writeFileSync(configFile, Buffer.from("DATABASE_URL=postgres://admin:password@localhost/prod"));

const bigImage = path.join(UPLOAD_DIR, "huge.png");
fs.writeFileSync(bigImage, Buffer.concat([Buffer.from("\x89PNG\r\n\x1a\n", "binary"), Buffer.alloc(5_000_000, "x")]));

// ── Configure image security ────────────────────────────────────────────
// PARITY-SECURITY (HIGH): configure_image is a global mutable module singleton
// (_image.py:_config). A module-level `let _config` in TS works — BUT ESM
// module identity is per-bundler; under Vite/webpack/Bun a duplicated copy of
// @neograph/core would get a SECOND _config, silently defeating the lockdown.
// Python's single interpreter has no such hazard. Needs a globalThis-symbol
// singleton to be safe. Not addressed anywhere in the parity matrix.
configureImage({
  allowedDirs: [UPLOAD_DIR], // only read from upload directory
  maxSizeBytes: 1_000_000, // 1MB max
  validateFormat: true, // reject non-image files
});

// ── Build pipeline ───────────────────────────────────────────────────────
// Python raw-scripted seed: def _img_seed(i, c) -> ImageInput reading
// c["configurable"]["photo_path"]. Programmatic Node.scripted references it by
// STRING NAME resolved through the compile(scripted={...}) registry — a
// serialization concession for LLM/YAML-driven assembly.
//
// PARITY: in TS the string-name + registry indirection is unnecessary at the
// call site (you can pass the closure directly), but it must stay SUPPORTED for
// the LLM-driven surface. Sketch keeps the registry to be faithful.
const imgSeed = (_input: ScriptedInput, cfg: RunConfig): ImageInput => ({
  photo: (cfg.configurable?.photo_path as string) ?? "",
});

const seed = Node.scripted({ name: "seed", fn: "_img_seed", outputs: ImageInput });

const analyze = Node({
  name: "analyze",
  mode: "think",
  outputs: Analysis,
  // PARITY-MULTIMODAL (MEDIUM): ${image:seed.photo} is a DISTINCT grammar from
  // ${var} (_llm_render.py:35 calls it out explicitly). The parity matrix rows
  // "Inline prompt substitution ${var}" (Direct) and "describe_type" say
  // nothing about the image variant. The TS port must: regex-split on
  // /\$\{image:([^}]+)\}/, dotted-resolve "seed.photo" against the RAW input
  // object (plain-object property walk — easy in TS), then call resolveImage
  // and emit {type:"image_url", image_url:{url}} blocks. Algorithm is Direct;
  // it's simply UNSPECIFIED in the design doc.
  prompt: "Analyze: ${image:seed.photo}",
  model: "fast",
  // think mode carries no fn signature, so inputs are declared explicitly on
  // the programmatic surface (AD-0 transformer can't reach a stringly-typed
  // Node() — it only fires on node({...}, fn) wrappers). Direct.
  inputs: { seed: ImageInput },
});

const pipeline = Construct({ name: "secure-vision", nodes: [seed, analyze] });

const graph = compile(pipeline, {
  llmFactory,
  promptCompiler,
  scripted: { _img_seed: imgSeed },
});

// ── Test cases ───────────────────────────────────────────────────────────
// PARITY-ASYNC (LOW): Python run() is sync. LangGraph.js .invoke() is
// Promise-based, so every run() is `await run(...)` inside an async main().
// Mechanical, but it colors the whole example — an IIFE async wrapper is
// required where Python is flat top-level script.
async function main() {
  console.log();
  console.log("=".repeat(60));

  // 1. Legitimate image in upload dir
  console.log("1. Legitimate image in upload dir:");
  const r1 = await run(graph, { input: { node_id: "t1", photo_path: legitImage } });
  console.log(`   Result: ${(r1.analyze as Analysis).result}`);
  console.assert((r1.analyze as Analysis).result === "analyzed", "Legit image should be analyzed");

  // 2. File outside upload dir (path traversal attempt)
  console.log("\n2. File outside upload dir (path traversal):");
  const r2 = await run(graph, { input: { node_id: "t2", photo_path: secretFile } });
  console.log(`   Result: ${(r2.analyze as Analysis).result}`);
  console.assert((r2.analyze as Analysis).result === "no-image", "File outside allowedDirs should be blocked");

  // 3. Non-image file in upload dir
  console.log("\n3. Non-image file (.txt) in upload dir:");
  const r3 = await run(graph, { input: { node_id: "t3", photo_path: configFile } });
  console.log(`   Result: ${(r3.analyze as Analysis).result}`);
  console.assert((r3.analyze as Analysis).result === "no-image", "Non-image file should be rejected");

  // 4. Oversized image in upload dir
  console.log("\n4. Oversized image (5MB > 1MB limit):");
  const r4 = await run(graph, { input: { node_id: "t4", photo_path: bigImage } });
  console.log(`   Result: ${(r4.analyze as Analysis).result}`);
  console.assert((r4.analyze as Analysis).result === "no-image", "Oversized file should be rejected");

  // 5. Direct resolveImage utility
  console.log("\n5. resolveImage utility (for template-ref consumers):");
  const uri = resolveImage(legitImage);
  console.log(`   URI prefix: ${uri.slice(0, 40)}...`);
  console.assert(uri.startsWith("data:image/png;base64,"));
  console.assert(uri !== "data:image/png;base64,"); // not empty

  console.log("\n" + "=".repeat(60));
  console.log("All 5 cases pass. configureImage locks down file access");
  console.log("while the pipeline degrades gracefully on blocked files.");

  // ── Cleanup ────────────────────────────────────────────────────────────
  fs.unlinkSync(legitImage);
  fs.unlinkSync(configFile);
  fs.unlinkSync(bigImage);
  fs.rmdirSync(UPLOAD_DIR);
  fs.unlinkSync(secretFile);

  configureImage(); // reset to defaults
}

void main();

// ────────────────────────────────────────────────────────────────────────
// Reference: the resolveImage internals this example depends on and how they
// would have to be re-implemented in TS (all UNSPEC'd in typescript-port.md):
//
// PARITY-SECURITY (HIGH) — path containment. Python (_image.py:154):
//     resolved = Path(path).resolve()            # resolves symlinks + ..
//     any(resolved.is_relative_to(Path(d).resolve()) for d in allowed_dirs)
//   Naive TS (path.resolve + startsWith) is UNSAFE:
//     (a) path.resolve() normalizes ".." but does NOT follow symlinks — a
//         symlink inside UPLOAD_DIR pointing at /etc/passwd would ESCAPE the
//         jail that Python's Path.resolve() (realpath) closes. TS must use
//         fs.realpathSync().
//     (b) is_relative_to is segment-aware; "/uploads".startsWith prefix match
//         lets "/uploads-evil/..." through. TS must use
//         path.relative(dir, p) and reject when it starts with ".." or is
//         absolute. This is the ONE row where a sloppy port is a live CVE, and
//         the parity matrix has no image/security row to warn about it.
//
// PARITY (LOW) — magic-byte sniff. Python _check_magic_bytes reads the header
//   and matches PNG/JPEG/GIF/WEBP/SVG signatures. TS: Buffer.subarray + compare.
//   Direct, but must be hand-ported; no library row in the matrix.
//
// PARITY (LOW) — fail-soft contract. Every block path returns the empty
//   sentinel "data:image/png;base64," (Python) rather than throwing. The FakeLLM
//   equality check depends on that EXACT string. TS must preserve the byte-exact
//   sentinel or case 2/3/4 assertions break. Direct but easy to drift.
