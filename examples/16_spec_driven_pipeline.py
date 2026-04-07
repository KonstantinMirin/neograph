"""Example 16: Spec-driven pipeline — YAML in, results out.

Scenario: An LLM generates a pipeline spec (YAML) describing a multi-step
security analysis. neograph loads the spec, compiles it to a LangGraph
graph, and runs it — no hand-written Node/Construct code required.

The pipeline has four stages:
  1. scan      — inventory files and flag raw findings
  2. assess    — convert raw findings into a scored analysis
  3. refine    — iteratively re-evaluate, eliminating false positives
                 (loops until risk_score < 0.3)
  4. recommend — produce prioritised remediation actions

All nodes are scripted (pure Python) so no API keys are needed.

Files:
  - 16_security_analysis.yaml  — the pipeline spec (as if LLM-generated)
  - 16_project.yaml            — the project surface (type definitions)

Patterns demonstrated:
  1. YAML spec loading with load_spec + project surface
  2. Type auto-generation from JSON Schema definitions
  3. Loop modifier driven by a condition expression
  4. register_scripted for plugging Python functions into spec nodes

Run:
    python examples/16_spec_driven_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

from neograph import compile, load_spec, register_scripted, run
from neograph.spec_types import lookup_type


# -- Scripted node implementations -------------------------------------------
# These functions are what the LLM's spec points at via scripted_fn names.
# Signature: (input_data, config) -> output_model_instance


def scan_codebase(input_data, config):
    """Simulate scanning a codebase for security issues."""
    ScanResult = lookup_type("ScanResult")
    return ScanResult(
        files_scanned=142,
        findings=[
            "SQL injection in user_handler.py:34",
            "Hardcoded secret in config.py:12",
            "Missing input validation in api/upload.py:87",
            "Insecure deserialization in data_loader.py:56",
            "Open redirect in auth/callback.py:23",
        ],
        severity_counts="critical:2, high:2, medium:1",
    )


def initial_assessment(input_data, config):
    """Convert raw scan findings into a scored AnalysisResult."""
    AnalysisResult = lookup_type("AnalysisResult")
    # Dict-form input: {scan: ScanResult}
    scan = input_data.get("scan") if isinstance(input_data, dict) else input_data
    findings = scan.findings if hasattr(scan, "findings") else []
    return AnalysisResult(
        vulnerabilities=findings,
        risk_score=0.85,
        iteration=0,
        summary=f"Initial assessment: {len(findings)} findings, risk=0.85",
    )


_refine_call = [0]


def refine_analysis(input_data, config):
    """Re-evaluate the analysis, eliminating false positives each pass."""
    _refine_call[0] += 1
    AnalysisResult = lookup_type("AnalysisResult")

    vulns = input_data.vulnerabilities if hasattr(input_data, "vulnerabilities") else []
    prev_risk = input_data.risk_score if hasattr(input_data, "risk_score") else 0.85

    # Each iteration drops risk by 0.25 and eliminates one false positive
    risk = max(prev_risk - 0.25, 0.1)
    trimmed = vulns[: max(len(vulns) - 1, 2)]

    return AnalysisResult(
        vulnerabilities=trimmed,
        risk_score=round(risk, 2),
        iteration=_refine_call[0],
        summary=f"Pass {_refine_call[0]}: {len(trimmed)} confirmed, risk={risk:.2f}",
    )


def generate_recommendations(input_data, config):
    """Produce prioritised remediation actions from the final analysis."""
    Recommendation = lookup_type("Recommendation")

    # Dict-form input: {refine: AnalysisResult} (or list from loop)
    if isinstance(input_data, dict):
        analysis = input_data.get("refine", input_data)
    else:
        analysis = input_data
    if isinstance(analysis, list):
        analysis = analysis[-1]

    vulns = analysis.vulnerabilities if hasattr(analysis, "vulnerabilities") else []
    actions = []
    for v in vulns:
        if "SQL" in v or "injection" in v.lower():
            actions.append(f"CRITICAL: Parameterise queries -- {v}")
        elif "secret" in v.lower() or "hardcoded" in v.lower():
            actions.append(f"CRITICAL: Move secrets to vault -- {v}")
        elif "validation" in v.lower():
            actions.append(f"HIGH: Add input validation -- {v}")
        elif "deserialization" in v.lower():
            actions.append(f"HIGH: Use safe deserializer -- {v}")
        else:
            actions.append(f"MEDIUM: Investigate -- {v}")

    priority = "critical" if any("CRITICAL" in a for a in actions) else "high"

    return Recommendation(
        actions=actions,
        priority=priority,
        risk_score=analysis.risk_score if hasattr(analysis, "risk_score") else 0.0,
    )


# -- Main --------------------------------------------------------------------


def main():
    _refine_call[0] = 0

    # Register scripted implementations so the spec can reference them
    register_scripted("scan_codebase", scan_codebase)
    register_scripted("initial_assessment", initial_assessment)
    register_scripted("refine_analysis", refine_analysis)
    register_scripted("generate_recommendations", generate_recommendations)

    here = Path(__file__).parent
    spec_path = str(here / "16_security_analysis.yaml")
    project_path = str(here / "16_project.yaml")

    # -- Load --
    print("=== Loading pipeline spec ===")
    print(f"  Spec:    16_security_analysis.yaml")
    print(f"  Project: 16_project.yaml")
    construct = load_spec(spec_path, project=project_path)

    print(f"\n  Pipeline: {construct.name}")
    print(f"  Nodes:    {[n.name for n in construct.nodes]}")
    for n in construct.nodes:
        modifiers = []
        if getattr(n, "loop", None):
            modifiers.append(f"Loop(max_iterations={n.loop.max_iterations})")
        mod_str = f"  modifiers: {', '.join(modifiers)}" if modifiers else ""
        print(f"    - {n.name} ({n.mode}){mod_str}")

    # -- Compile --
    print("\n=== Compiling to LangGraph ===")
    graph = compile(construct)
    print("  Compiled successfully.")

    # -- Run --
    print("\n=== Running pipeline ===")
    result = run(graph, input={"node_id": "security-audit-001"})

    # -- Results --
    print("\n=== Results ===")

    # Scan
    scan = result["scan"]
    print(f"\n[scan] Files scanned: {scan.files_scanned}")
    print(f"[scan] Raw findings ({len(scan.findings)}):")
    for f in scan.findings:
        print(f"    {f}")
    print(f"[scan] Severity: {scan.severity_counts}")

    # Assess
    assess = result["assess"]
    print(f"\n[assess] {assess.summary}")

    # Refine (loop produces append-list)
    refine_history = result["refine"]
    if isinstance(refine_history, list):
        print(f"\n[refine] Iterations: {len(refine_history)}")
        for entry in refine_history:
            print(f"    {entry.summary}")
        final_analysis = refine_history[-1]
    else:
        final_analysis = refine_history
        print(f"\n[refine] {final_analysis.summary}")

    print(f"[refine] Final risk score: {final_analysis.risk_score}")
    print(f"[refine] Confirmed vulnerabilities: {len(final_analysis.vulnerabilities)}")

    # Recommend
    rec = result["recommend"]
    print(f"\n[recommend] Priority: {rec.priority}")
    print(f"[recommend] Residual risk: {rec.risk_score}")
    print(f"[recommend] Actions ({len(rec.actions)}):")
    for a in rec.actions:
        print(f"    {a}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
