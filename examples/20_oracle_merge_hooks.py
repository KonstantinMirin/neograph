"""Example 20: Oracle Merge Hooks — generate + evaluate in 10 lines.

Scenario: Write a product description from multiple angles, then use an
LLM judge to pick the best. Add self-healing: if the judge misses a
required section, post-processing fills the gap. If the judge LLM fails
entirely, a deterministic fallback merges the variants.

This is the pattern that replaces 60+ lines of manual @merge_fn code
with piarch_llm_call. NeoGraph handles retry-with-error-feedback, BAML
rendering, schema enforcement, cost tracking, and observability. Your
hooks handle only the domain logic.

Run:
    python examples/20_oracle_merge_hooks.py
"""

from __future__ import annotations

import sys

from pydantic import BaseModel, Field

from neograph import (
    Construct,
    Node,
    Oracle,
    compile,
    configure_llm,
    construct_from_module,
    node,
    run,
)


# ── Schemas ──────────────────────────────────────────────────────────────

class ProductBrief(BaseModel, frozen=True):
    name: str
    features: list[str]

class ProductDescription(BaseModel, frozen=True):
    headline: str = Field(description="One-line attention grabber")
    body: str = Field(description="2-3 sentence description")
    cta: str = Field(description="Call to action")


# ── Fake LLM (replace with real models in production) ─────────────────────

_gen_counter = [0]
_HEADLINES = [
    "Transform Your Workflow",
    "Build Faster, Ship Safer",
    "The Future of Automation",
]

class FakeProductLLM:
    """Each generator produces a different creative variant."""

    def __init__(self, tier: str):
        self._tier = tier

    def with_structured_output(self, model, **kw):
        self._model = model
        return self

    def invoke(self, messages, **kw):
        if self._tier == "reason":
            # Judge merge: pick the best variant
            return self._model(
                headline="Build Faster, Ship Safer",
                body="NeoGraph turns typed Python functions into production agents. "
                     "Declare the logic, we handle the wiring.",
                cta="Get started in 5 minutes",
            )
        # Generator: produce a variant
        idx = _gen_counter[0] % len(_HEADLINES)
        _gen_counter[0] += 1
        return self._model(
            headline=_HEADLINES[idx],
            body=f"Variant {idx}: A great product for everyone.",
            cta="Try it now" if idx % 2 == 0 else "",  # some variants miss the CTA
        )


configure_llm(
    llm_factory=lambda tier: FakeProductLLM(tier),
    prompt_compiler=lambda tmpl, data, **kw: [{"role": "user", "content": tmpl}],
)


# ── Hooks: the only domain logic you write ────────────────────────────────

def tag_variants(variants: list[ProductDescription]) -> dict:
    """Pre-process: number each variant so the judge can reference them."""
    tagged = [
        f"[Variant {i+1}]\n"
        f"  Headline: {v.headline}\n"
        f"  Body: {v.body}\n"
        f"  CTA: {v.cta or '(missing)'}"
        for i, v in enumerate(variants)
    ]
    return {"numbered_variants": "\n\n".join(tagged)}


def ensure_cta(result: ProductDescription, variants: list[ProductDescription]) -> ProductDescription:
    """Post-process: if the judge result has no CTA, pull one from the variants."""
    if not result.cta:
        for v in variants:
            if v.cta:
                return ProductDescription(
                    headline=result.headline,
                    body=result.body,
                    cta=v.cta,
                )
    return result


def fallback_pick_longest(variants: list[ProductDescription], error: Exception) -> ProductDescription:
    """Fallback: on LLM failure, pick the variant with the longest body."""
    return max(variants, key=lambda v: len(v.body))


# ── The pipeline: 10 lines ────────────────────────────────────────────────

@node(outputs=ProductDescription,
      prompt="Write a product description for a developer tool",
      model="fast",
      ensemble_n=3,
      merge_prompt="Pick the best product description:\n${numbered_variants}",
      merge_pre_process=tag_variants,
      merge_post_process=ensure_cta,
      merge_fallback=fallback_pick_longest)
def write_description() -> ProductDescription: ...


pipeline = construct_from_module(sys.modules[__name__], name="product-copy")


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _gen_counter[0] = 0
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "launch-v1"})

    desc = result["write_description"]
    print("=== Oracle + LLM Judge + Hooks ===")
    print(f"  Headline: {desc.headline}")
    print(f"  Body:     {desc.body}")
    print(f"  CTA:      {desc.cta}")
    print()
    print("3 generators, 1 LLM judge, self-healing CTA, fallback on failure.")
    print("Zero manual LLM calls. Zero retry logic. Zero schema parsing.")
