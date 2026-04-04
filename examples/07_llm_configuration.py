"""Example 7: LLM Configuration — per-node model routing, temperature, and budgets.

Scenario: A pipeline where different nodes need different LLM configurations.
The decompose node needs creative output (high temperature), the classify
node needs precise output (low temperature), and the verify node uses a
cheaper/faster model to save costs.

This shows how llm_config on each Node flows through to the LLM factory,
giving you per-node control over model, temperature, max tokens, and any
other parameter your LLM provider accepts.

Run (requires .env with OPENROUTER_API_KEY):
    python examples/07_llm_configuration.py

Or run with fakes (no API key):
    python examples/07_llm_configuration.py --fake
"""

from __future__ import annotations

import sys

from pydantic import BaseModel

from neograph import Construct, Node, compile, run
from neograph._llm import configure_llm


# ── Schemas ──────────────────────────────────────────────────────────────

class Claims(BaseModel, frozen=True):
    items: list[str]

class ClassifiedClaims(BaseModel, frozen=True):
    classified: list[dict[str, str]]


# ══════════════════════════════════════════════════════════════════════════
# LLM FACTORY — the single place where all LLM configuration happens.
#
# NeoGraph calls this with:
#   tier       — from Node(model="fast") or Node(model="reason")
#   node_name  — the node's name, for per-node routing
#   llm_config — from Node(llm_config={...}), any dict you want
#
# You control what these mean. NeoGraph just passes them through.
# ══════════════════════════════════════════════════════════════════════════

USE_FAKE = "--fake" in sys.argv


def real_llm_factory(tier, node_name=None, llm_config=None):
    """Production factory: OpenRouter with per-node configuration."""
    import os

    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv()

    llm_config = llm_config or {}

    # Model routing by tier
    models = {
        "fast": "google/gemini-2.0-flash-001",
        "reason": "anthropic/claude-sonnet-4",
    }

    return ChatOpenAI(
        model=models.get(tier, models["fast"]),
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        # Per-node settings from llm_config
        temperature=llm_config.get("temperature", 0),
        max_tokens=llm_config.get("max_tokens", None),
    )


def fake_llm_factory(tier, node_name=None, llm_config=None):
    """Test factory: records what config each node received."""
    llm_config = llm_config or {}
    print(f"  [factory] node={node_name}, tier={tier}, "
          f"temp={llm_config.get('temperature', 'default')}, "
          f"max_tokens={llm_config.get('max_tokens', 'default')}")

    class FakeLLM:
        def with_structured_output(self, model):
            self._model = model
            return self

        def invoke(self, messages, **kwargs):
            if self._model is Claims:
                return Claims(items=["claim-1", "claim-2"])
            if self._model is ClassifiedClaims:
                return ClassifiedClaims(classified=[
                    {"claim": "claim-1", "category": "security"},
                    {"claim": "claim-2", "category": "reliability"},
                ])
            return self._model()

    return FakeLLM()


configure_llm(
    llm_factory=fake_llm_factory if USE_FAKE else real_llm_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": (
        f"Process this: {data}" if data else "Generate claims about system security"
    )}],
)


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE — each node has its own LLM configuration
# ══════════════════════════════════════════════════════════════════════════

# Creative decomposition: high temperature, more tokens
decompose = Node(
    name="decompose",
    mode="produce",
    output=Claims,
    model="reason",          # uses the "reason" tier (more capable model)
    prompt="decompose",
    llm_config={
        "temperature": 0.9,  # creative — explore diverse decompositions
        "max_tokens": 2000,
    },
)

# Precise classification: zero temperature, fewer tokens
classify = Node(
    name="classify",
    mode="produce",
    input=Claims,
    output=ClassifiedClaims,
    model="fast",            # uses the "fast" tier (cheaper model)
    prompt="classify",
    llm_config={
        "temperature": 0,    # deterministic — consistent classification
        "max_tokens": 500,
    },
)

# Mixed models: reasoning model with json_mode (no structured output support)
# + fast model with native structured output
decompose_deepseek = Node(
    name="decompose-ds",
    mode="produce",
    output=Claims,
    model="reason",
    prompt="decompose",
    llm_config={
        "temperature": 0.9,
        "output_strategy": "json_mode",  # DeepSeek doesn't support with_structured_output
    },
)

classify_gemini = Node(
    name="classify-gm",
    mode="produce",
    input=Claims,
    output=ClassifiedClaims,
    model="fast",
    prompt="classify",
    llm_config={
        "temperature": 0,
        "output_strategy": "structured",  # Gemini supports native structured output
    },
)

pipeline = Construct("configured-pipeline", nodes=[decompose, classify])
pipeline_mixed = Construct("mixed-models", nodes=[decompose_deepseek, classify_gemini])


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LLM factory calls:\n")
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "REQ-001"})

    print(f"\nDecomposed: {result['decompose'].items}")
    print(f"Classified: {result['classify'].classified}")
    print()
    print("Note: output_strategy is per-node — mix structured and json_mode")
    print("in the same pipeline when using different model providers.")
