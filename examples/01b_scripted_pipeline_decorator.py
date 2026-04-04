"""Example 1b: Scripted Pipeline with the @node decorator — Dagster-style.

Same behavior as 01_scripted_pipeline.py, but the pipeline is built from
@node-decorated functions with parameter-name dependency inference. No
explicit `nodes=[...]` list; `construct_from_module` walks this module and
topologically sorts the decorated functions into a Construct.

Run:
    python examples/01b_scripted_pipeline_decorator.py
"""

from __future__ import annotations

import sys

from pydantic import BaseModel

from neograph import compile, construct_from_module, node, run


# ── Schemas ──────────────────────────────────────────────────────────────

class RawText(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]

class ClassifiedClaims(BaseModel, frozen=True):
    classified: list[dict[str, str]]


# ── Nodes ────────────────────────────────────────────────────────────────
# Each @node-decorated function becomes a Node whose dependencies are
# inferred from its parameter names. Parameter name = upstream node name
# in Python-identifier form (hyphens are underscores).

@node(output=RawText)
def extract() -> RawText:
    """Simulate extracting text from a document source."""
    return RawText(text="The system shall log all access attempts. The system shall validate input.")


@node(output=Claims)
def split(extract: RawText) -> Claims:
    """Split raw text into individual claims by sentence."""
    sentences = [s.strip() for s in extract.text.split(".") if s.strip()]
    return Claims(items=sentences)


@node(output=ClassifiedClaims)
def classify(split: Claims) -> ClassifiedClaims:
    """Classify each claim by category based on keywords."""
    classified = []
    for claim in split.items:
        category = "security" if "access" in claim.lower() or "validate" in claim.lower() else "general"
        classified.append({"claim": claim, "category": category})
    return ClassifiedClaims(classified=classified)


# ── Build pipeline — no nodes=[...] list, no order maintenance ───────────

pipeline = construct_from_module(sys.modules[__name__], name="doc-processor")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "doc-001"})

    print("Claims found:", len(result["classify"].classified))
    for item in result["classify"].classified:
        print(f"  [{item['category']}] {item['claim']}")
