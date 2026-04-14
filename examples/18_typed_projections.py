"""Example 18: Field Visibility Control — what the LLM sees vs what the pipeline keeps.

Like API serialization schemas (what the frontend sees vs what the server keeps),
NeoGraph lets you control which fields the LLM sees in two directions:

    INPUT RENDERING: what the LLM sees when reading data from upstream nodes
    OUTPUT SCHEMA: what the LLM is asked to produce

Three tools for this:

    1. Field(exclude=True)
       Hidden from BOTH input rendering and output schema.
       Use for: truly internal fields (IDs, timestamps, debug data).

    2. Annotated[T, ExcludeFromOutput]
       Visible in input rendering, hidden from output schema.
       Use for: pipeline-set fields the LLM should see but not produce.
       Example: assigned_letter set by a filter node, consumed by a writer node.

    3. render_for_prompt() -> BaseModel
       Complete control: return a different model entirely.
       Use for: heavy restructuring, computed fields, format changes.

Run:
    python examples/18_typed_projections.py
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from neograph import (
    Construct,
    ExcludeFromOutput,
    Node,
    XmlRenderer,
    compile,
    configure_llm,
    describe_type,
    render_input,
    run,
)
from neograph.factory import register_scripted


# ── Models ───────────────────────────────────────────────────────────────

class ResearchResult(BaseModel, frozen=True):
    """Full research data — everything the pipeline needs."""
    content: str
    source_url: str
    score: float = 0.0

    # Pipeline-internal: never shown to any LLM
    retrieval_latency_ms: int = Field(default=0, exclude=True)
    cache_hit: bool = Field(default=False, exclude=True)


class ExtensionCondition(BaseModel, frozen=True):
    """A condition to extend with new steps.

    assigned_letter is set by the pipeline (filter node), not by the LLM.
    The brainstorm node should NOT produce it (output schema hides it).
    The writer node MUST see it (input rendering shows it).
    """
    text: str
    acceptance_criteria: list[str] = Field(default_factory=list)

    # Pipeline-set: visible when reading, hidden when producing
    assigned_letter: Annotated[str, ExcludeFromOutput] = ""


class HydratedResearch(BaseModel, frozen=True):
    """Heavy model with render_for_prompt for complete restructuring."""
    raw_html: str
    metadata: dict = Field(default_factory=dict)
    internal_score: float = 0.0

    def render_for_prompt(self) -> "ResearchPresentation":
        """Return a slim typed projection for LLM consumption."""
        return ResearchPresentation(
            content=self.raw_html[:200].strip(),
            confidence=round(self.internal_score, 2),
        )


class ResearchPresentation(BaseModel, frozen=True):
    """What the LLM actually sees — clean, focused."""
    content: str
    confidence: float = Field(description="0.0 to 1.0")


class Analysis(BaseModel, frozen=True):
    summary: str


# ── Demo ─────────────────────────────────────────────────────────────────

def main():
    renderer = XmlRenderer()

    # ── 1. exclude=True: hidden from everything ──────────────────────────
    print("=" * 60)
    print("1. Field(exclude=True) — hidden from BOTH schema and rendering")
    print("=" * 60)

    research = ResearchResult(
        content="Event sourcing provides auditability",
        source_url="https://docs.corp/arch",
        score=0.87,
        retrieval_latency_ms=42,
        cache_hit=True,
    )

    print("\nOutput schema (what LLM produces):")
    print(describe_type(ResearchResult, prefix=""))

    print("\nInput rendering (what downstream LLM sees):")
    print(renderer.render(research))

    print("\nNote: retrieval_latency_ms and cache_hit appear in NEITHER.")

    # ── 2. ExcludeFromOutput: visible in input, hidden from output ───────
    print("\n" + "=" * 60)
    print("2. ExcludeFromOutput — visible when reading, hidden when producing")
    print("=" * 60)

    condition = ExtensionCondition(
        text="System shall log all access attempts",
        acceptance_criteria=["Logs include timestamp", "Logs include user ID"],
        assigned_letter="B",  # set by pipeline, not LLM
    )

    print("\nOutput schema (brainstorm node — LLM produces this):")
    schema = describe_type(ExtensionCondition, prefix="")
    print(schema)
    assert "assigned_letter" not in schema, "LLM should not see assigned_letter in schema"

    print("\nInput rendering (writer node — LLM reads this):")
    rendered = renderer.render(condition)
    print(rendered)
    assert "assigned_letter" in rendered, "Writer must see assigned_letter"
    assert "B" in rendered

    print("\nThe brainstorm LLM won't try to produce assigned_letter.")
    print("The writer LLM will see it and know to use letter B.")

    # ── 3. render_for_prompt: complete restructuring ─────────────────────
    print("\n" + "=" * 60)
    print("3. render_for_prompt() -> BaseModel — full projection control")
    print("=" * 60)

    heavy = HydratedResearch(
        raw_html="<h1>Architecture</h1><p>We chose event sourcing...</p>",
        metadata={"author": "team", "version": 3},
        internal_score=0.92,
    )

    print("\nDirect render (all fields):")
    print(renderer.render(heavy))

    print("\nProjected render (via render_for_prompt):")
    projected = render_input(heavy, renderer=renderer)
    print(projected)
    assert "raw_html" not in projected
    assert "metadata" not in projected
    assert "confidence" in projected

    # ── 4. Pipeline execution ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. Full pipeline with ExcludeFromOutput")
    print("=" * 60)

    register_scripted("make_condition", lambda _i, _c: ExtensionCondition(
        text="System shall validate inputs",
        acceptance_criteria=["Reject empty strings"],
    ))
    register_scripted("assign_letter", lambda _i, _c: ExtensionCondition(
        text=_i.text,
        acceptance_criteria=_i.acceptance_criteria,
        assigned_letter="C",  # pipeline sets this
    ))
    register_scripted("write_extension", lambda _i, _c: Analysis(
        summary=f"Extension for condition {_i.assigned_letter}: {_i.text}",
    ))

    configure_llm(
        llm_factory=lambda tier: None,
        prompt_compiler=lambda tmpl, data, **kw: [],
    )

    pipeline = Construct("visibility-demo", nodes=[
        Node.scripted("make-condition", fn="make_condition", outputs=ExtensionCondition),
        Node.scripted("assign-letter", fn="assign_letter",
                      inputs=ExtensionCondition, outputs=ExtensionCondition),
        Node.scripted("write-extension", fn="write_extension",
                      inputs=ExtensionCondition, outputs=Analysis),
    ])

    graph = compile(pipeline)
    result = run(graph, input={"node_id": "example-18"})
    print(f"\n  Result: {result['write_extension'].summary}")
    print()
    print("The writer node saw assigned_letter='C' and used it.")
    print("If this were an LLM node, describe_type would NOT show assigned_letter.")


if __name__ == "__main__":
    main()
