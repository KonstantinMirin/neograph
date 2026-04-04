"""Example 8: Structured Output Coercion — handling models that don't do JSON well.

Problem: Many models (especially reasoning/thinking models like DeepSeek-R1,
o1, QwQ) don't reliably produce structured JSON. They wrap output in
markdown code fences, add commentary, or return malformed JSON.

Solution: Handle this in the llm_factory. NeoGraph calls
`llm.with_structured_output(Model)` — your factory returns an LLM
that makes this work, however that needs to happen for your model.

This example shows three strategies:
  1. Model supports structured output natively (OpenAI, Gemini)
  2. Model needs JSON repair (strip fences, fix trailing commas)
  3. Model needs a two-pass approach (generate text, then parse)

Run:
    python examples/08_structured_output_coercion.py
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from neograph import Construct, Node, compile, run
from neograph._llm import configure_llm


# ── Schema ───────────────────────────────────────────────────────────────

class Claims(BaseModel, frozen=True):
    items: list[str]


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Native structured output (most models)
#
# This is the default. ChatOpenAI, ChatAnthropic, etc. handle
# with_structured_output natively. Nothing to configure.
#
#   def factory(tier, **kwargs):
#       return ChatOpenAI(model="gpt-4o")  # just works
#
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: JSON repair wrapper
#
# For models that return JSON wrapped in markdown fences or with minor
# formatting issues. Wrap the LLM in a repair layer.
# ══════════════════════════════════════════════════════════════════════════

class JsonRepairLLM:
    """Wraps an LLM that returns sloppy JSON — strips fences, repairs, parses."""

    def __init__(self, inner_llm):
        self._inner = inner_llm

    def with_structured_output(self, model):
        """Return a wrapper that parses the repaired JSON into the Pydantic model."""
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        # Get raw text from the inner LLM
        response = self._inner.invoke(messages, **kwargs)
        raw_text = response if isinstance(response, str) else getattr(response, 'content', str(response))

        # Repair: strip markdown fences
        cleaned = re.sub(r'^```(?:json)?\s*', '', raw_text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```\s*$', '', cleaned, flags=re.MULTILINE)

        # Repair: strip trailing commas before ] or }
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

        return self._model.model_validate_json(cleaned)

    def bind_tools(self, tools):
        return self._inner.bind_tools(tools)


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Two-pass approach
#
# For models that truly can't do structured output (e.g., some reasoning
# models). First call generates free-form text, second call extracts
# structure. Both calls happen inside the factory's returned LLM.
# ══════════════════════════════════════════════════════════════════════════

class TwoPassLLM:
    """First pass: free-form generation. Second pass: structured extraction."""

    def __init__(self, generator, extractor):
        self._generator = generator  # reasoning model (free-form)
        self._extractor = extractor  # fast model (structured output)

    def with_structured_output(self, model):
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        # Pass 1: reasoning model generates free-form analysis
        raw_response = self._generator.invoke(messages, **kwargs)

        # Pass 2: fast model extracts structured data
        extraction_messages = [
            {"role": "user", "content": (
                f"Extract structured data from this analysis. "
                f"Return valid JSON matching this schema: {self._model.model_json_schema()}\n\n"
                f"Analysis:\n{raw_response}"
            )},
        ]
        extractor = self._extractor.with_structured_output(self._model)
        return extractor.invoke(extraction_messages, **kwargs)


# ══════════════════════════════════════════════════════════════════════════
# DEMO: factory that picks strategy based on llm_config
# ══════════════════════════════════════════════════════════════════════════

class FakeInnerLLM:
    """Simulates a model that returns sloppy JSON."""
    def invoke(self, messages, **kwargs):
        return '```json\n{"items": ["claim-1", "claim-2",]}\n```'


class FakeCleanLLM:
    """Simulates a model with native structured output."""
    def with_structured_output(self, model):
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        return self._model(items=["claim-1", "claim-2"])


def demo_factory(tier, node_name=None, llm_config=None):
    """Factory that demonstrates strategy selection via llm_config."""
    llm_config = llm_config or {}
    strategy = llm_config.get("structured_output", "native")

    if strategy == "repair":
        # Wrap sloppy model in JSON repair
        print(f"  [{node_name}] Using JSON repair strategy")
        return JsonRepairLLM(FakeInnerLLM())

    if strategy == "two_pass":
        # Two-pass: reasoning model + extraction model
        print(f"  [{node_name}] Using two-pass strategy")
        return TwoPassLLM(FakeInnerLLM(), FakeCleanLLM())

    # Default: native structured output
    print(f"  [{node_name}] Using native structured output")
    return FakeCleanLLM()


configure_llm(
    llm_factory=demo_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": "analyze"}],
)


# ── Pipeline with different strategies per node ──────────────────────────

native_node = Node(
    name="native-extract",
    mode="produce",
    output=Claims,
    model="fast",
    prompt="extract",
    # No llm_config — uses native structured output
)

repair_node = Node(
    name="repair-extract",
    mode="produce",
    output=Claims,
    model="fast",
    prompt="extract",
    llm_config={"structured_output": "repair"},
)


# Run each independently to show the strategy in action
if __name__ == "__main__":
    print("Strategy selection:\n")

    # Native
    p1 = Construct("native", nodes=[native_node])
    g1 = compile(p1)
    r1 = run(g1, input={"node_id": "test"})
    print(f"  Result: {r1['native_extract'].items}\n")

    # Repair
    p2 = Construct("repair", nodes=[repair_node])
    g2 = compile(p2)
    r2 = run(g2, input={"node_id": "test"})
    print(f"  Result: {r2['repair_extract'].items}\n")

    print("Both strategies produced the same result — the consumer doesn't know which was used.")
