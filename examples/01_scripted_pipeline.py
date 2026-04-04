"""Example 1: Scripted Pipeline — deterministic data processing, no LLM.

Scenario: A document processing pipeline that extracts text from a source,
splits it into individual claims, then classifies each claim by category.
All logic is pure Python — no API keys needed.

This is the simplest NeoGraph pipeline: three scripted nodes connected
sequentially. Data flows through typed state: extract produces RawText,
split consumes RawText and produces Claims, classify consumes Claims
and produces ClassifiedClaims.

Run:
    python examples/01_scripted_pipeline.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node, compile, register_scripted, run


# ── Schemas ──────────────────────────────────────────────────────────────

class RawText(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]

class ClassifiedClaims(BaseModel, frozen=True):
    classified: list[dict[str, str]]


# ── Scripted functions ───────────────────────────────────────────────────
# Each function receives (input_data, config) where input_data is the
# typed output from a previous node, or None for the first node.

def extract_text(input_data, config):
    """Simulate extracting text from a document source."""
    return RawText(text="The system shall log all access attempts. The system shall validate input.")

def split_claims(input_data, config):
    """Split raw text into individual claims by sentence."""
    sentences = [s.strip() for s in input_data.text.split(".") if s.strip()]
    return Claims(items=sentences)

def classify_claims(input_data, config):
    """Classify each claim by category based on keywords."""
    classified = []
    for claim in input_data.items:
        category = "security" if "access" in claim.lower() or "validate" in claim.lower() else "general"
        classified.append({"claim": claim, "category": category})
    return ClassifiedClaims(classified=classified)


# ── Register functions ───────────────────────────────────────────────────

register_scripted("extract_text", extract_text)
register_scripted("split_claims", split_claims)
register_scripted("classify_claims", classify_claims)


# ── Build pipeline ───────────────────────────────────────────────────────

extract = Node.scripted("extract", fn="extract_text", output=RawText)
split = Node.scripted("split", fn="split_claims", input=RawText, output=Claims)
classify = Node.scripted("classify", fn="classify_claims", input=Claims, output=ClassifiedClaims)

pipeline = Construct("doc-processor", nodes=[extract, split, classify])


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "doc-001"})

    print("Claims found:", len(result["classify"].classified))
    for item in result["classify"].classified:
        print(f"  [{item['category']}] {item['claim']}")
