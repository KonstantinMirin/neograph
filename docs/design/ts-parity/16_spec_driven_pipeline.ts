// TS parity sketch of examples/16_spec_driven_pipeline.py
// HYPOTHETICAL — targets the proposed API in docs/design/typescript-port.md.
// Not meant to compile or run. `// PARITY:` notes mark where the TS DX diverges.
//
// What this example exercises (all four are spec-loader-surface features):
//   1. loadSpec(spec, {project}) — YAML pipeline + JSON-Schema project surface
//   2. runtime type generation from JSON Schema (Python: pydantic.create_model)
//   3. Loop modifier driven by a string condition expression
//   4. scripted map on compile() to plug functions into spec nodes by name
//
// KEY OBSERVATION: none of AD-0 (the ts-patch/typia signature transformer) is
// engaged here. The DAG comes from YAML, not from function signatures, and the
// node bodies are passed by string name in a `scripted` map — there is no
// decorator, no parameter-name-as-edge inference, no DI-marker classification.
// So this example sidesteps the single biggest TS-port risk entirely... and in
// exchange surfaces a *different* gap: the types are born at runtime, so the TS
// compiler never sees them.

import { loadSpec, compile, run, lookupType } from "@neograph/core";
import type { ScriptedFn } from "@neograph/core";
import { z } from "zod";

// -- Scripted node implementations -------------------------------------------
// These are what the spec's `scripted_fn:` names point at.
// Python signature: (input_data, config) -> output_model_instance
// Proposed TS signature: (inputData: unknown, config: RunConfig) => unknown
//
// PARITY (API GAP — untyped scripted bodies): In Python `lookup_type("ScanResult")`
// returns a real pydantic *class* you both construct and get field access on.
// The TS spec loader builds a `z.ZodType` from the JSON Schema at RUNTIME, so
// there is NO compile-time `ScanResult` type to annotate against. `lookupType`
// can only return `z.ZodType<unknown>` (or `z.ZodTypeAny`). Every body below is
// therefore statically untyped on both its input and its output. This is the
// core DX degradation for this example — see the report.

// PARITY (Redesign — duck typing -> explicit guards): Python leans on
// `hasattr(x, "findings")` / `isinstance(input_data, dict)` / `isinstance(x, list)`.
// TS has no attribute protocol; every one of those becomes an explicit `in`
// check, `Array.isArray`, or a Zod `.safeParse`. More ceremony, same logic.

const scanCodebase: ScriptedFn = (_inputData, _config) => {
  const ScanResult = lookupType("ScanResult"); // z.ZodType<unknown> — no static face
  // PARITY: Python calls `ScanResult(files_scanned=142, ...)` — a validating
  // constructor. Here we build a plain object and (optionally) validate it
  // through the runtime schema. The literal is NOT type-checked against the
  // schema by tsc; only `ScanResult.parse(...)` catches shape errors at runtime.
  return ScanResult.parse({
    files_scanned: 142,
    findings: [
      "SQL injection in user_handler.py:34",
      "Hardcoded secret in config.py:12",
      "Missing input validation in api/upload.py:87",
      "Insecure deserialization in data_loader.py:56",
      "Open redirect in auth/callback.py:23",
    ],
    severity_counts: "critical:2, high:2, medium:1",
  });
};

const initialAssessment: ScriptedFn = (inputData, _config) => {
  const AnalysisResult = lookupType("AnalysisResult");
  // Dict-form input: { scan: ScanResult }
  // PARITY: `input_data.get("scan")` -> guarded property access on `unknown`.
  const scan =
    inputData && typeof inputData === "object" && "scan" in inputData
      ? (inputData as Record<string, any>).scan
      : inputData;
  const findings: string[] =
    scan && typeof scan === "object" && "findings" in scan ? scan.findings : [];
  return AnalysisResult.parse({
    vulnerabilities: findings,
    risk_score: 0.85,
    iteration: 0,
    summary: `Initial assessment: ${findings.length} findings, risk=0.85`,
  });
};

// PARITY (Direct): Python's module-level `_refine_call = [0]` boxed counter ports
// to a module-scoped `let`. The loop body mutates ambient state to number passes.
// (Same reset-in-main caveat: see runMain below.)
let refineCall = 0;

const refineAnalysis: ScriptedFn = (inputData, _config) => {
  refineCall += 1;
  const AnalysisResult = lookupType("AnalysisResult");
  const d = inputData as Record<string, any>;
  const vulns: string[] =
    d && typeof d === "object" && "vulnerabilities" in d ? d.vulnerabilities : [];
  const prevRisk: number =
    d && typeof d === "object" && "risk_score" in d ? d.risk_score : 0.85;

  const risk = Math.max(prevRisk - 0.25, 0.1);
  const trimmed = vulns.slice(0, Math.max(vulns.length - 1, 2));

  return AnalysisResult.parse({
    vulnerabilities: trimmed,
    risk_score: Math.round(risk * 100) / 100,
    iteration: refineCall,
    summary: `Pass ${refineCall}: ${trimmed.length} confirmed, risk=${risk.toFixed(2)}`,
  });
};

const generateRecommendations: ScriptedFn = (inputData, _config) => {
  const Recommendation = lookupType("Recommendation");
  // Dict-form input { refine: AnalysisResult } OR a list (from the loop's append).
  // PARITY: Python does `isinstance(input_data, dict)` then `isinstance(analysis, list)`.
  let analysis: any = inputData;
  if (analysis && typeof analysis === "object" && !Array.isArray(analysis) && "refine" in analysis) {
    analysis = analysis.refine;
  }
  if (Array.isArray(analysis)) {
    analysis = analysis[analysis.length - 1]; // loop -> last entry
  }

  const vulns: string[] =
    analysis && typeof analysis === "object" && "vulnerabilities" in analysis
      ? analysis.vulnerabilities
      : [];
  const actions: string[] = [];
  for (const v of vulns) {
    const lower = v.toLowerCase();
    if (v.includes("SQL") || lower.includes("injection")) {
      actions.push(`CRITICAL: Parameterise queries -- ${v}`);
    } else if (lower.includes("secret") || lower.includes("hardcoded")) {
      actions.push(`CRITICAL: Move secrets to vault -- ${v}`);
    } else if (lower.includes("validation")) {
      actions.push(`HIGH: Add input validation -- ${v}`);
    } else if (lower.includes("deserialization")) {
      actions.push(`HIGH: Use safe deserializer -- ${v}`);
    } else {
      actions.push(`MEDIUM: Investigate -- ${v}`);
    }
  }
  const priority = actions.some((a) => a.includes("CRITICAL")) ? "critical" : "high";

  return Recommendation.parse({
    actions,
    priority,
    risk_score:
      analysis && typeof analysis === "object" && "risk_score" in analysis
        ? analysis.risk_score
        : 0.0,
  });
};

// -- Main --------------------------------------------------------------------

async function runMain() {
  refineCall = 0;

  // PARITY (Direct): file-path resolution. `import.meta.url` + node:path instead
  // of `Path(__file__).parent`. The two YAML files are SHARED ARTIFACTS —
  // byte-for-byte reusable across Python and TS (typescript-port.md §Shared).
  const here = new URL(".", import.meta.url).pathname;
  const specPath = here + "16_security_analysis.yaml";
  const projectPath = here + "16_project.yaml";

  // -- Load --
  // PARITY (Direct on the call; Medium under the hood): loadSpec parses YAML
  // (js-yaml) and the project surface (ajv). The project `types:` block is
  // JSON-Schema-ish (`type: integer|array|string|number`); the loader must run
  // json-schema -> Zod at runtime (Python: pydantic.create_model). Registered
  // into the same lookupType Map the scripted bodies read.
  console.log("=== Loading pipeline spec ===");
  const construct = await loadSpec(specPath, { project: projectPath });

  console.log(`\n  Pipeline: ${construct.name}`);
  console.log(`  Nodes:    ${construct.nodes.map((n) => n.name)}`);
  for (const n of construct.nodes) {
    const mods: string[] = [];
    // PARITY: `getattr(n, "loop", None)` -> optional field access.
    if (n.loop) mods.push(`Loop(max_iterations=${n.loop.max_iterations})`);
    const modStr = mods.length ? `  modifiers: ${mods.join(", ")}` : "";
    console.log(`    - ${n.name} (${n.mode})${modStr}`);
  }

  // -- Compile --
  // PARITY (Direct): `scripted` map name -> fn is identical in shape to Python's
  // `compile(construct, scripted={...})`.
  console.log("\n=== Compiling to LangGraph ===");
  const graph = compile(construct, {
    scripted: {
      scan_codebase: scanCodebase,
      initial_assessment: initialAssessment,
      refine_analysis: refineAnalysis,
      generate_recommendations: generateRecommendations,
    },
  });
  console.log("  Compiled successfully.");

  // -- Run --
  // PARITY (Direct): input dict -> object; run is async in LangGraph.js.
  console.log("\n=== Running pipeline ===");
  const result = await run(graph, { input: { node_id: "security-audit-001" } });

  // -- Results --
  // PARITY: `result["scan"]` -> `result.scan`. But `result` is
  // `Record<string, unknown>` (no per-node static typing — same root cause as
  // lookupType), so every access below is an untyped `any`/cast.
  const scan = result.scan as any;
  console.log(`\n[scan] Files scanned: ${scan.files_scanned}`);
  console.log(`[scan] Raw findings (${scan.findings.length}):`);
  for (const f of scan.findings) console.log(`    ${f}`);
  console.log(`[scan] Severity: ${scan.severity_counts}`);

  const assess = result.assess as any;
  console.log(`\n[assess] ${assess.summary}`);

  // Refine (loop produces an append-list)
  // PARITY (Direct): Loop's append reducer yields an array, exactly like Python.
  const refineHistory = result.refine as any;
  let finalAnalysis: any;
  if (Array.isArray(refineHistory)) {
    console.log(`\n[refine] Iterations: ${refineHistory.length}`);
    for (const entry of refineHistory) console.log(`    ${entry.summary}`);
    finalAnalysis = refineHistory[refineHistory.length - 1];
  } else {
    finalAnalysis = refineHistory;
    console.log(`\n[refine] ${finalAnalysis.summary}`);
  }
  console.log(`[refine] Final risk score: ${finalAnalysis.risk_score}`);
  console.log(`[refine] Confirmed vulnerabilities: ${finalAnalysis.vulnerabilities.length}`);

  const rec = result.recommend as any;
  console.log(`\n[recommend] Priority: ${rec.priority}`);
  console.log(`[recommend] Residual risk: ${rec.risk_score}`);
  console.log(`[recommend] Actions (${rec.actions.length}):`);
  for (const a of rec.actions) console.log(`    ${a}`);

  console.log("\n=== Done ===");
}

runMain();
