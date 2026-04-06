"""Example 9: Operator — human-in-the-loop interrupt and resume.

Scenario: A pipeline validates a requirement analysis. If validation fails,
the graph pauses and waits for human review. The human approves or provides
corrections, then the graph resumes from where it stopped.

This demonstrates:
  - @node(interrupt_when=...) — inline callable form for human-in-the-loop
  - Checkpointer: required for interrupt/resume (state must be persisted)
  - run() with resume: continue after human feedback

Run:
    python examples/09_operator_human_in_loop.py
"""

from __future__ import annotations

import sys

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import compile, construct_from_module, node, run


# ── Schemas ──────────────────────────────────────────────────────────────

class Analysis(BaseModel, frozen=True):
    claims: list[str]
    coverage_pct: int

class ValidationResult(BaseModel, frozen=True):
    passed: bool
    issues: list[str]

class FinalReport(BaseModel, frozen=True):
    text: str


# ── Pipeline (declarative @node) ────────────────────────────────────────

@node(mode="scripted", outputs=Analysis)
def analyze() -> Analysis:
    return Analysis(claims=["auth", "logging", "encryption"], coverage_pct=55)


@node(
    mode="scripted",
    outputs=ValidationResult,
    interrupt_when=lambda state: (
        {"issues": state.check.issues, "message": "Please review and approve"}
        if state.check and not state.check.passed
        else None
    ),
)
def check(analyze: Analysis) -> ValidationResult:
    """Check if analysis meets quality bar."""
    if analyze.coverage_pct < 80:
        return ValidationResult(
            passed=False,
            issues=[f"Coverage {analyze.coverage_pct}% is below 80% threshold"],
        )
    return ValidationResult(passed=True, issues=[])


@node(mode="scripted", outputs=FinalReport)
def report(analyze: Analysis) -> FinalReport:
    return FinalReport(text=f"Report: {analyze.claims}, coverage: {analyze.coverage_pct}%")


pipeline = construct_from_module(sys.modules[__name__], name="review-pipeline")


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
