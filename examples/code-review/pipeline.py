"""Code Review Agent -- neograph mini-project.

Takes a git diff and produces a structured code review with findings
categorized by severity across three dimensions: style, logic, security.

    OPENROUTER_API_KEY=sk-... python examples/code-review/pipeline.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from neograph import (
    Construct,
    Each,
    FromInput,
    Node,
    compile,
    configure_llm,
    construct_from_functions,
    node,
    run,
)

from schemas import (
    ChangedFile,
    ChangedFiles,
    FileReview,
    Finding,
    LogicFindings,
    ReviewReport,
    SecurityFindings,
    StyleFindings,
)


# =============================================================================
# Data + LLM setup
# =============================================================================

SKIP_PATTERNS = re.compile(
    r"(\.lock$|\.min\.|vendor/|node_modules/|__pycache__/|\.pyc$|\.png$|\.jpg$|\.gif$)"
)

LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".sql": "sql",
}


def _load_sample_diff() -> str:
    return (_HERE / "data" / "sample_diff.txt").read_text()


def _prompt(name: str) -> str:
    return (_HERE / "prompts" / f"{name}.md").read_text()


MODELS = {
    "reason": "anthropic/claude-sonnet-4",
    "fast": "google/gemini-2.0-flash-001",
}


def _llm_factory(tier: str, *, node_name: str = "", llm_config: dict | None = None):
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY to run this example.")
    return ChatOpenAI(
        model=MODELS.get(tier, MODELS["fast"]),
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=(llm_config or {}).get("temperature", 0.3),
        max_tokens=(llm_config or {}).get("max_tokens", 4000),
    )


configure_llm(
    llm_factory=_llm_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": template}],
)


# =============================================================================
# Phase 1: Parse the diff (scripted)
# =============================================================================

@node(outputs=ChangedFiles)
def parse_diff(diff_text: Annotated[str, FromInput]) -> ChangedFiles:
    """Parse a unified diff into structured ChangedFile objects."""
    files: list[ChangedFile] = []
    current_path = None
    current_hunks: list[str] = []

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            if current_path and current_hunks:
                ext = Path(current_path).suffix
                if not SKIP_PATTERNS.search(current_path):
                    files.append(ChangedFile(
                        path=current_path,
                        language=LANG_MAP.get(ext, ext.lstrip(".") or "unknown"),
                        hunks="\n".join(current_hunks),
                    ))
            parts = line.split(" b/")
            current_path = parts[-1] if len(parts) > 1 else "unknown"
            current_hunks = []
        elif current_path:
            current_hunks.append(line)

    if current_path and current_hunks:
        ext = Path(current_path).suffix
        if not SKIP_PATTERNS.search(current_path):
            files.append(ChangedFile(
                path=current_path,
                language=LANG_MAP.get(ext, ext.lstrip(".") or "unknown"),
                hunks="\n".join(current_hunks),
            ))

    return ChangedFiles(files=files)


# =============================================================================
# Phase 2: Per-file review sub-construct (3 dimensions in parallel)
# =============================================================================

@node(mode="think", outputs=StyleFindings, model="fast", prompt=_prompt("review_style"))
def review_style(file: ChangedFile) -> StyleFindings:
    ...


@node(mode="think", outputs=LogicFindings, model="fast", prompt=_prompt("review_logic"))
def review_logic(file: ChangedFile) -> LogicFindings:
    ...


@node(mode="think", outputs=SecurityFindings, model="reason", prompt=_prompt("review_security"))
def review_security(file: ChangedFile) -> SecurityFindings:
    ...


@node(outputs=FileReview)
def merge_file_findings(
    file: ChangedFile,
    review_style: StyleFindings,
    review_logic: LogicFindings,
    review_security: SecurityFindings,
) -> FileReview:
    """Merge findings from all three review dimensions into one FileReview."""
    all_findings = (
        review_style.findings
        + review_logic.findings
        + review_security.findings
    )
    return FileReview(
        path=file.path,
        findings=sorted(all_findings, key=lambda f: list(
            ("critical", "high", "medium", "low", "info")
        ).index(f.severity.value)),
    )


# Sub-construct: ChangedFile -> FileReview
# Three review nodes fan-in to merge_file_findings.
# `file` param on each node matches input=ChangedFile via port param resolution.
review_file = construct_from_functions(
    "review-file",
    [review_style, review_logic, review_security, merge_file_findings],
    input=ChangedFile,
    output=FileReview,
)


# =============================================================================
# Phase 3: Synthesize all file reviews
# =============================================================================

@node(mode="think", outputs=ReviewReport, model="reason", prompt=_prompt("synthesize"))
def synthesize(review_file: dict[str, FileReview]) -> ReviewReport:
    ...


# =============================================================================
# Pipeline assembly
# =============================================================================

pipeline = construct_from_functions(
    "code-review",
    [
        parse_diff,
        review_file.map("parse_diff.files", key="path"),
        synthesize,
    ],
)


# =============================================================================
# Run
# =============================================================================

def main():
    diff_text = _load_sample_diff()
    print(f"Loaded sample diff ({len(diff_text)} chars)")

    graph = compile(pipeline)
    result = run(graph, input={
        "node_id": "review-001",
        "diff_text": diff_text,
    })

    report: ReviewReport = result["synthesize"]
    print(f"\n{'=' * 60}")
    print("CODE REVIEW REPORT")
    print(f"{'=' * 60}\n")
    print(f"Summary: {report.summary}\n")
    print(f"Critical: {report.critical_count}  High: {report.high_count}  "
          f"Medium: {report.medium_count}  Low: {report.low_count}\n")

    if report.findings:
        print("Findings:")
        for f in report.findings:
            print(f"  [{f.severity.value.upper()}] {f.location}")
            print(f"    {f.description}")
            print(f"    Fix: {f.suggestion}\n")

    if report.positive_notes:
        print("What's good:")
        for note in report.positive_notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()
