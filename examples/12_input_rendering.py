"""Example 12: Pluggable Input Rendering — dispatch levels, BAML default, field
flattening, and inline vs template-ref prompts.

Scenario: A produce node receives a Pydantic model with prose fields.
Instead of JSON-dumping it into the prompt (escaped newlines, token waste),
NeoGraph renders it as human-readable text via the renderer dispatch hierarchy:

    Level 1: Model method  — render_for_prompt() on the BaseModel
    Level 2: Node renderer — @node(renderer=XmlRenderer())
    Level 3: Construct/global — Construct(renderer=...) or configure_llm(renderer=...)

Additional sections demonstrate:

    BAML default        — when NO renderer is configured, BaseModel values are
                          rendered via describe_value() in BAML notation.
    Field flattening    — render_for_prompt() returning a BaseModel flattens its
                          fields into individually addressable template vars.
    Inline vs template  — inline prompts (${var.field}) get raw objects with
                          dotted access; template-ref prompts get BAML strings.

This example uses no real LLM. It demonstrates the rendering pipeline
and the render_prompt() / render_input() / build_rendered_input() inspectors.

Run:
    python examples/12_input_rendering.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from neograph import (
    Construct,
    Node,
    XmlRenderer,
    DelimitedRenderer,
    JsonRenderer,
    configure_llm,
    describe_type,
    describe_value,
    render_input,
    render_prompt,
)
from neograph.renderers import build_rendered_input


# -- Schemas ----------------------------------------------------------------

class ReadContext(BaseModel, frozen=True):
    """Document content with prose fields."""
    title: str = Field(description="Document title")
    raw: str = Field(description="Raw markdown content")
    tags: list[str] = Field(default_factory=list, description="Classification tags")


class Analysis(BaseModel, frozen=True):
    summary: str
    key_points: list[str]


# -- Level 1: Model with render_for_prompt() method ------------------------

class CustomRendered(BaseModel, frozen=True):
    """Model that controls its own rendering."""
    name: str
    content: str

    def render_for_prompt(self) -> str:
        return f"=== {self.name} ===\n{self.content}"


# -- Fake LLM + prompt compiler (no API keys) ------------------------------

class FakeLLM:
    def with_structured_output(self, model, **kwargs):
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        return self._model(summary="Example summary", key_points=["point-1"])


def simple_compiler(template, data, **kwargs):
    return [{"role": "user", "content": f"[{template}]\n{data}"}]


configure_llm(
    llm_factory=lambda tier: FakeLLM(),
    prompt_compiler=simple_compiler,
    renderer=XmlRenderer(),  # Level 3: global default
)


# -- Pipeline ---------------------------------------------------------------

analyze = Node(
    "analyze",
    mode="think",
    outputs=Analysis,
    model="fast",
    prompt="analyze",
    renderer=DelimitedRenderer(),  # Level 2: node-level override
)

pipeline = Construct("render-demo", nodes=[analyze])


# -- Run --------------------------------------------------------------------

if __name__ == "__main__":
    doc = ReadContext(
        title="Architecture Decision Record",
        raw="We chose event sourcing for auditability.\nCQRS separates reads from writes.",
        tags=["architecture", "decision"],
    )

    # Level 1: model method wins over any renderer
    custom = CustomRendered(name="spec", content="The system shall log all access.")
    print("Level 1 (model method):")
    print(render_input(custom, renderer=XmlRenderer()))
    print()

    # Level 2: node renderer (DelimitedRenderer on `analyze`)
    print("Level 2 (node renderer) via render_prompt inspector:")
    print(render_prompt(analyze, doc))
    print()

    # Level 3: global renderer (XmlRenderer via configure_llm)
    plain_node = Node("plain", mode="think", outputs=Analysis, model="fast", prompt="p")
    print("Level 3 (global renderer) via render_prompt inspector:")
    print(render_prompt(plain_node, doc))
    print()

    # describe_type for output schema
    print("Output schema (describe_type):")
    print(describe_type(Analysis))
    print()

    # ===================================================================
    # Section: BAML default rendering (no renderer configured)
    # ===================================================================
    # When no renderer is set at any level, BaseModel values are rendered
    # via describe_value() — the BAML notation, not XmlRenderer.

    print("=" * 60)
    print("BAML DEFAULT (no renderer)")
    print("=" * 60)

    # describe_value is the underlying function. render_input dispatches
    # to it when renderer=None and the value is a BaseModel.
    print("\ndescribe_value(doc) — raw BAML notation:")
    print(describe_value(doc))
    print()

    # render_input with renderer=None also produces BAML:
    print("render_input(doc, renderer=None) — same BAML output:")
    print(render_input(doc, renderer=None))
    print()

    # Compare to XmlRenderer to see the difference:
    print("render_input(doc, renderer=XmlRenderer()) — XML for comparison:")
    print(render_input(doc, renderer=XmlRenderer()))
    print()

    # ===================================================================
    # Section: Field flattening via render_for_prompt() returning BaseModel
    # ===================================================================
    # When render_for_prompt() returns a BaseModel (not a str), its fields
    # become individually addressable template vars in template-ref prompts.

    class SourceDoc(BaseModel, frozen=True):
        """The flattened model returned by render_for_prompt."""
        headline: str
        body: str
        provenance: str

    class ResearchReport(BaseModel, frozen=True):
        """Model that flattens itself for prompt consumption."""
        title: str
        raw_content: str
        source_url: str

        def render_for_prompt(self) -> SourceDoc:
            """Return a BaseModel whose fields become separate template vars."""
            return SourceDoc(
                headline=self.title.upper(),
                body=self.raw_content,
                provenance=f"Source: {self.source_url}",
            )

    report = ResearchReport(
        title="Quarterly Review",
        raw_content="Revenue grew 12% year-over-year.",
        source_url="https://example.com/q4",
    )

    print("=" * 60)
    print("FIELD FLATTENING (render_for_prompt returns BaseModel)")
    print("=" * 60)

    # build_rendered_input shows the full RenderedInput structure:
    ri = build_rendered_input({"report": report}, renderer=None)
    print("\nRenderedInput.rendered (whole model, BAML):")
    print(ri.rendered)
    print("\nRenderedInput.flattened (individual fields from SourceDoc):")
    for k, v in ri.flattened.items():
        print(f"  {k}: {v!r}")
    print("\nRenderedInput.for_template_ref (merged — what the prompt compiler sees):")
    tmpl_data = ri.for_template_ref
    for k, v in tmpl_data.items():
        print(f"  {k}: {v!r}")
    print("\navailable_keys_template:", sorted(ri.available_keys_template))
    print("available_keys_inline:  ", sorted(ri.available_keys_inline))
    print()

    # A template-ref prompt can now use {headline}, {body}, {provenance}
    # as first-class vars, not just {report}.

    # ===================================================================
    # Section: Inline prompt vs template-ref prompt
    # ===================================================================
    # Inline prompts contain ${var.field} — they receive raw objects so
    # dotted attribute access works. Template-ref prompts (no spaces, no
    # ${...}) receive BAML-rendered strings.

    print("=" * 60)
    print("INLINE vs TEMPLATE-REF prompt rendering")
    print("=" * 60)

    # Use a node with NO renderer (renderer=None) so BAML is the default.
    # Since the global configure_llm set XmlRenderer, we create a node
    # with explicit renderer=None to show BAML behavior.
    # However, render_prompt uses: effective_renderer = node.renderer or global
    # So we demonstrate inline vs template-ref via build_rendered_input directly.

    print("\n--- Same data, two prompt styles ---")
    print()

    # Build RenderedInput with no renderer (BAML default):
    ri_baml = build_rendered_input(doc, renderer=None)

    print("RenderedInput.raw (what inline ${var.field} sees):")
    print(f"  type: {type(ri_baml.raw).__name__}")
    print(f"  ri_baml.raw.title  = {ri_baml.raw.title!r}")
    print(f"  ri_baml.raw.tags   = {ri_baml.raw.tags!r}")
    print()

    print("RenderedInput.rendered (what template-ref sees — BAML string):")
    print(ri_baml.rendered)
    print()

    # With XmlRenderer for contrast:
    ri_xml = build_rendered_input(doc, renderer=XmlRenderer())

    print("RenderedInput.rendered with XmlRenderer (template-ref sees XML):")
    print(ri_xml.rendered)
    print()

    # Dict-form (fan-in) example showing both views:
    claims_model = Analysis(summary="Strong growth", key_points=["12% rev", "new market"])
    ri_dict = build_rendered_input({"doc": doc, "prior": claims_model}, renderer=None)

    print("Dict-form fan-in (two upstream values, BAML default):")
    print("  Inline keys (raw objects): ", sorted(ri_dict.available_keys_inline))
    print("  Template keys (rendered):  ", sorted(ri_dict.available_keys_template))
    print()
    print("  ri_dict.raw['doc'].title       =", ri_dict.raw["doc"].title)
    print("  ri_dict.rendered['doc']         =")
    for line in ri_dict.rendered["doc"].splitlines():
        print(f"    {line}")
    print("  ri_dict.rendered['prior']       =")
    for line in ri_dict.rendered["prior"].splitlines():
        print(f"    {line}")
