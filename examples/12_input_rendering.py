"""Example 12: Pluggable Input Rendering — three dispatch levels.

Scenario: A produce node receives a Pydantic model with prose fields.
Instead of JSON-dumping it into the prompt (escaped newlines, token waste),
NeoGraph renders it as human-readable XML/delimited text via the renderer
dispatch hierarchy:

    Level 1: Model method  — render_for_prompt() on the BaseModel
    Level 2: Node renderer — @node(renderer=XmlRenderer())
    Level 3: Construct/global — Construct(renderer=...) or configure_llm(renderer=...)

This example uses no real LLM. It demonstrates the rendering pipeline
and the render_prompt() inspector.

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
    render_prompt,
)


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
    from neograph.renderers import render_input
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
