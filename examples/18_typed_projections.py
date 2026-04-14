"""Example 18: Typed Presentation Projections — render_for_prompt() returning BaseModel.

When a Pydantic model's render_for_prompt() returns another BaseModel (not a string),
NeoGraph automatically renders it through the active renderer (XML/JSON/Delimited).
This lets you define typed *presentation projections* — slim views of your data
optimized for the LLM prompt — without manual string formatting.

Pattern:
    class FullData(BaseModel):
        # all fields for pipeline logic
        raw_content: str
        internal_id: int
        score: float

        def render_for_prompt(self) -> SlimView:
            # return a BaseModel, not a string
            return SlimView(content=self.raw_content.upper(), score=self.score)

    class SlimView(BaseModel):
        content: str
        score: float

The LLM sees <content>...</content><score>...</score> (or JSON/delimited depending
on renderer) — never the internal_id or raw formatting details.

Run:
    python examples/18_typed_projections.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from neograph import (
    Node,
    Construct,
    XmlRenderer,
    compile,
    configure_llm,
    render_input,
    render_prompt,
    run,
)
from neograph.factory import register_scripted


# ── Schemas ──────────────────────────────────────────────────────────────

class ResearchData(BaseModel, frozen=True):
    """Full research payload — everything the pipeline needs."""
    raw_text: str
    source_url: str
    internal_score: float = 0.0
    retrieval_latency_ms: int = 0

    def render_for_prompt(self) -> "ResearchPresentation":
        """Typed projection: strip internals, format for LLM consumption."""
        return ResearchPresentation(
            content=self.raw_text.strip(),
            confidence=round(self.internal_score, 2),
        )


class ResearchPresentation(BaseModel, frozen=True):
    """Slim view the LLM actually sees — no URLs, no latency, no raw text."""
    content: str
    confidence: float = Field(description="0.0 to 1.0 confidence score")


class Analysis(BaseModel, frozen=True):
    """LLM output."""
    summary: str
    quality: str


# ── Demo ─────────────────────────────────────────────────────────────────

def main():
    # 1. Show the rendering pipeline
    data = ResearchData(
        raw_text="  The system uses event sourcing for full auditability.  ",
        source_url="https://internal.corp/docs/arch/123",
        internal_score=0.87,
        retrieval_latency_ms=42,
    )

    renderer = XmlRenderer()

    # Without typed projection: render_input would render ALL fields
    print("=== Direct XML render (all fields) ===")
    print(renderer.render(data))
    print()

    # With typed projection: render_for_prompt() returns ResearchPresentation,
    # which gets auto-rendered through the XML renderer
    print("=== Typed projection (slim view) ===")
    result = render_input(data, renderer=renderer)
    print(result)
    print()

    # Note: source_url, internal_score, retrieval_latency_ms are stripped.
    # The LLM only sees content + confidence.
    assert "source_url" not in result
    assert "retrieval_latency_ms" not in result
    assert "confidence" in result

    # 2. Show it working in a real pipeline
    register_scripted("research_source", lambda _in, _cfg: data)
    register_scripted("analyze", lambda _in, _cfg: Analysis(
        summary="Event sourcing provides audit trail",
        quality="high",
    ))

    # Configure a fake LLM (scripted nodes don't use it, but compile requires it)
    configure_llm(
        llm_factory=lambda tier: None,
        prompt_compiler=lambda tmpl, data, **kw: [],
    )

    print("=== Pipeline execution ===")
    pipeline = Construct("projection-demo", nodes=[
        Node.scripted("research-source", fn="research_source", outputs=ResearchData),
        Node.scripted("analyze", fn="analyze", inputs=ResearchData, outputs=Analysis),
    ])

    graph = compile(pipeline)
    result = run(graph, input={"node_id": "example-18"})
    print(f"  Result: {result['analyze']}")
    print()
    print("Done. The LLM never saw source_url or retrieval_latency_ms.")


if __name__ == "__main__":
    main()
