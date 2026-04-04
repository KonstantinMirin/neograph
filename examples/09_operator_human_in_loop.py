"""Example 9: Operator — human-in-the-loop interrupt and resume.

Scenario: A pipeline validates a requirement analysis. If validation fails,
the graph pauses and waits for human review. The human approves or provides
corrections, then the graph resumes from where it stopped.

This demonstrates:
  - Operator modifier: pause graph when a condition is truthy
  - Checkpointer: required for interrupt/resume (state must be persisted)
  - run() with resume: continue after human feedback

Run:
    python examples/09_operator_human_in_loop.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import Construct, Node, Operator, compile, run
from neograph.factory import register_condition, register_scripted


# ── Schemas ──────────────────────────────────────────────────────────────

class Analysis(BaseModel, frozen=True):
    claims: list[str]
    coverage_pct: int

class ValidationResult(BaseModel, frozen=True):
    passed: bool
    issues: list[str]

class FinalReport(BaseModel, frozen=True):
    text: str


# ── Functions ────────────────────────────────────────────────────────────

def analyze_requirement(input_data, config):
    return Analysis(claims=["auth", "logging", "encryption"], coverage_pct=55)

def validate_analysis(input_data, config):
    """Check if analysis meets quality bar."""
    if input_data.coverage_pct < 80:
        return ValidationResult(
            passed=False,
            issues=[f"Coverage {input_data.coverage_pct}% is below 80% threshold"],
        )
    return ValidationResult(passed=True, issues=[])

def build_report(input_data, config):
    return FinalReport(text=f"Report: {input_data.claims}, coverage: {input_data.coverage_pct}%")

register_scripted("analyze", analyze_requirement)
register_scripted("validate", validate_analysis)
register_scripted("report", build_report)


# ── Condition: when should the graph pause? ──────────────────────────────
# The condition function receives the full state. If it returns a truthy
# value, interrupt() is called with that value as the payload.
# If it returns None/falsy, the graph continues.

def check_validation(state):
    val = state.check  # field name matches the node name "check"
    if val and not val.passed:
        return {"issues": val.issues, "message": "Please review and approve"}
    return None

register_condition("needs_human_review", check_validation)


# ── Pipeline ─────────────────────────────────────────────────────────────

analyze = Node.scripted("analyze", fn="analyze", output=Analysis)

check = Node.scripted(
    "check", fn="validate", input=Analysis, output=ValidationResult
) | Operator(when="needs_human_review")  # ← pauses here if validation fails

report = Node.scripted("report", fn="report", input=Analysis, output=FinalReport)

pipeline = Construct("review-pipeline", nodes=[analyze, check, report])


# ── Run with checkpointer (required for interrupt/resume) ────────────────

if __name__ == "__main__":
    # Checkpointer is REQUIRED when using Operator — compile() enforces this
    graph = compile(pipeline, checkpointer=MemorySaver())

    # Thread ID identifies this execution for resume
    config = {"configurable": {"thread_id": "review-001"}}

    # First run — will pause at Operator
    print("=== First run: will pause for human review ===\n")
    result = run(graph, input={"node_id": "REQ-001"}, config=config)

    print(f"Analysis: {result['analyze'].coverage_pct}% coverage")
    print(f"Validation: passed={result['check'].passed}")

    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"]
        print(f"\nGraph PAUSED. Interrupt payload:")
        for interrupt in interrupt_data:
            print(f"  {interrupt.value}")

        # Human reviews and approves
        print("\n=== Human approves, resuming... ===\n")
        result = run(graph, resume={"approved": True, "reviewer": "alice"}, config=config)

        print(f"Human feedback recorded: {result.get('human_feedback')}")
        print(f"Report generated: {result['report'].text}")
    else:
        print("Validation passed — no interrupt needed")
        print(f"Report: {result['report'].text}")
