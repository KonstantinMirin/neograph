"""Example 6: Raw Node — escape hatch for custom LangGraph logic.

Scenario: A pipeline that mostly uses declarative nodes, but one step
needs custom logic that doesn't fit produce/gather/scripted modes.
The @raw_node decorator lets you write a classic LangGraph function
while NeoGraph handles edges, state wiring, and observability around it.

Use cases for raw nodes:
  - Complex conditional logic (if/else on state, not just data transforms)
  - Calling external APIs that aren't LangChain tools
  - State mutations that don't fit the typed input→output pattern
  - Prototyping before extracting into a proper mode

The raw node receives the full state object and returns a dict update,
exactly like a LangGraph node function.

Run:
    python examples/06_raw_node_escape_hatch.py
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Construct, Node, compile, raw_node, register_scripted, run


# ── Schemas ──────────────────────────────────────────────────────────────

class Claims(BaseModel, frozen=True):
    items: list[str]

class FilteredClaims(BaseModel, frozen=True):
    kept: list[str]
    dropped: list[str]
    reason: str


# ── Scripted: produce initial claims ─────────────────────────────────────

def extract_claims(input_data, config):
    return Claims(items=[
        "system shall authenticate users via SSO",
        "system shall support dark mode",
        "system shall encrypt data at rest",
        "system shall have a cool logo",
        "system shall rate-limit API calls",
    ])

register_scripted("extract_claims", extract_claims)


# ── Raw node: custom filtering logic ────────────────────────────────────
# This is the escape hatch. The function receives the full Pydantic state
# and returns a dict of field updates. NeoGraph wires the edges.

@raw_node(input=Claims, output=FilteredClaims)
def filter_non_functional(state, config):
    """Drop claims that aren't real requirements (e.g., cosmetic wishes).

    This is a raw node because the filtering logic is complex:
    - keyword-based heuristics
    - access to the full state for context
    - returns a structured diff (kept vs dropped)
    """
    # Find Claims in state
    claims = None
    for field_name in state.__class__.model_fields:
        val = getattr(state, field_name, None)
        if isinstance(val, Claims):
            claims = val
            break

    if claims is None:
        return {"filter_non_functional": FilteredClaims(kept=[], dropped=[], reason="no claims found")}

    # Heuristic: non-functional requirements contain these keywords
    functional_keywords = ["authenticate", "encrypt", "rate-limit", "validate", "shall log", "authorize"]

    kept = []
    dropped = []
    for claim in claims.items:
        if any(kw in claim.lower() for kw in functional_keywords):
            kept.append(claim)
        else:
            dropped.append(claim)

    return {"filter_non_functional": FilteredClaims(
        kept=kept,
        dropped=dropped,
        reason=f"kept {len(kept)} functional, dropped {len(dropped)} cosmetic",
    )}


# ── Build pipeline: scripted → raw ��� scripted ───────────────────────────

def summarize(input_data, config):
    """Summarize what was kept."""
    return Claims(items=input_data.kept)

register_scripted("summarize", summarize)

pipeline = Construct("filter-pipeline", nodes=[
    Node.scripted("extract", fn="extract_claims", output=Claims),
    filter_non_functional,  # raw node — mixed in with declarative nodes
    Node.scripted("summarize", fn="summarize", input=FilteredClaims, output=Claims),
])


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "REQ-007"})

    filtered = result["filter_non_functional"]
    print(f"Filter result: {filtered.reason}\n")
    print("Kept (functional):")
    for claim in filtered.kept:
        print(f"  + {claim}")
    print("\nDropped (cosmetic):")
    for claim in filtered.dropped:
        print(f"  - {claim}")
    print(f"\nFinal claims: {len(result['summarize'].items)}")
