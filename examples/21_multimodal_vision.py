"""Example 21: Multimodal Vision — send images to VLMs alongside text.

Scenario: A product catalog pipeline receives product photos and generates
structured metadata (category, quality score, description). The vision
model sees the actual image — not a text description of it.

NeoGraph's ``${image:field}`` syntax produces LangChain multimodal content
blocks (text + image_url). The rest of the pipeline — structured output,
retry, observability — works unchanged.

This example uses fakes. Replace with real VLM (Gemini, GPT-4o, Claude)
for production use.

Run:
    python examples/21_multimodal_vision.py
"""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from neograph import compile, construct_from_module, node, run

# ── Schemas ──────────────────────────────────────────────────────────────

class ProductPhoto(BaseModel, frozen=True):
    """Input: path or base64 of a product image."""
    image_data: str
    product_name: str

class ProductMetadata(BaseModel, frozen=True):
    """Output: structured metadata from the vision model."""
    category: str = Field(description="Product category (e.g., electronics, furniture)")
    quality_score: float = Field(description="Image quality 0-1")
    description: str = Field(description="One-sentence product description")


# ── Fake VLM ─────────────────────────────────────────────────────────────

class FakeVisionLLM:
    """Simulates a VLM that receives multimodal content blocks."""

    def __init__(self, tier: str):
        self._tier = tier

    def with_structured_output(self, model, **kw):
        self._model = model
        return self

    def invoke(self, messages, **kw):
        # In production, the VLM sees the actual image.
        # Here we verify the message format is correct.
        user_msg = messages[0]
        content = user_msg["content"] if isinstance(user_msg, dict) else user_msg.content

        if isinstance(content, list):
            # Multimodal: content blocks with text + image_url
            has_image = any(b.get("type") == "image_url" for b in content)
            text_parts = [b["text"] for b in content if b.get("type") == "text"]
            text = " ".join(text_parts)
            print(f"    VLM received: {len(content)} content blocks, "
                  f"has_image={has_image}, text_preview='{text[:50]}...'")
        else:
            text = content
            has_image = False
            print("    VLM received: plain text, no image")

        return self._model(
            category="electronics" if has_image else "unknown",
            quality_score=0.92 if has_image else 0.0,
            description="High-quality product photo" if has_image else "No image provided",
        )


def _llm_factory(tier):
    return FakeVisionLLM(tier)


def _prompt_compiler(t, d, **kw):
    return [{"role": "user", "content": t}]


# ── Pipeline ─────────────────────────────────────────────────────────────
# The ${image:field} syntax is the only new thing. Everything else is
# standard @node + construct_from_module.

@node(outputs=ProductPhoto)
def load_photo() -> ProductPhoto:
    """Load a product photo (simulated with a temp PNG file)."""
    # Create a fake PNG file to demonstrate file-path input
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(b"\x89PNG\r\n\x1a\n" + b"fake-product-image-data" * 10)
    tmp.close()
    return ProductPhoto(image_data=tmp.name, product_name="Wireless Headphones")


@node(outputs=ProductMetadata,
      prompt="Analyze this product photo for ${load_photo.product_name}: ${image:load_photo.image_data}",
      model="fast")
def classify(load_photo: ProductPhoto) -> ProductMetadata:
    """VLM classifies the product image into structured metadata."""
    ...


pipeline = construct_from_module(sys.modules[__name__], name="product-catalog")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(
        pipeline,
        llm_factory=_llm_factory,
        prompt_compiler=_prompt_compiler,
    )
    result = run(graph, input={"node_id": "product-001"})

    meta = result["classify"]
    print()
    print("=== Product Metadata (from VLM) ===")
    print(f"  Category:    {meta.category}")
    print(f"  Quality:     {meta.quality_score}")
    print(f"  Description: {meta.description}")

    # Clean up temp file
    photo = result["load_photo"]
    Path(photo.image_data).unlink(missing_ok=True)

    # ── Base64 input (no file on disk) ──────────────────────────────────
    print()
    print("=== Base64 Input (no file) ===")
    from neograph import Construct, Node

    b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nbase64-product-photo").decode()

    def _b64_seed(i, c):
        return ProductPhoto(image_data=b64, product_name="Smart Watch")

    seed = Node.scripted("seed", fn="_b64_seed", outputs=ProductPhoto)
    classify_node = Node(
        "classify-b64", mode="think", outputs=ProductMetadata,
        prompt="Classify this product: ${image:seed.image_data}",
        model="fast", inputs={"seed": ProductPhoto},
    )
    b64_pipeline = Construct("b64-catalog", nodes=[seed, classify_node])
    graph2 = compile(
        b64_pipeline,
        llm_factory=_llm_factory,
        prompt_compiler=_prompt_compiler,
        scripted={"_b64_seed": _b64_seed},
    )
    result2 = run(graph2, input={"node_id": "product-002"})

    meta2 = result2["classify_b64"]
    print(f"  Category:    {meta2.category}")
    print(f"  Quality:     {meta2.quality_score}")
    print(f"  Description: {meta2.description}")

    print()
    print("Both file paths and base64 strings work with ${image:field}.")
    print("NeoGraph handles encoding, MIME detection, and content block assembly.")
