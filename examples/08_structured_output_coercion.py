"""Example 8: Output Strategies — handling models that don't do JSON well.

Problem: Many models (DeepSeek-R1, o1, QwQ, local models) don't support
with_structured_output. They return JSON wrapped in markdown fences,
embedded in commentary, or just malformed.

Solution: NeoGraph has three built-in output strategies, selected per-node
via llm_config["output_strategy"]:

  "structured" — llm.with_structured_output(model). Default. Works with
                 OpenAI, Anthropic, Gemini, any LangChain model with
                 native structured output support.

  "json_mode"  — LLM returns raw text. NeoGraph strips markdown fences,
                 extracts JSON, and parses it into the Pydantic model.
                 Works with DeepSeek, local models, any provider.

  "text"       — Same as json_mode. LLM returns plain text with JSON
                 somewhere in it. NeoGraph finds and parses the JSON.
                 Widest compatibility — works with literally anything.

The strategy is per-node, so you can mix models in the same pipeline:
fast structured calls for classification, json_mode for reasoning models.

Run:
    python examples/08_structured_output_coercion.py
"""

from pydantic import BaseModel

from neograph import Construct, Node, compile, configure_llm, run


# ── Schema ───────────────────────────────────────────────────────────────

class Claims(BaseModel, frozen=True):
    items: list[str]


# ══════════════════════════════════════════════════════════════════════════
# FAKE LLMs — each simulates a different model behavior
# ══════════════════════════════════════════════════════════════════════════

class CleanLLM:
    """Model with native structured output (OpenAI, Gemini)."""
    def with_structured_output(self, model, **kwargs):
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        return self._model(items=["claim-1", "claim-2"])


class FencedJsonLLM:
    """Model that returns JSON in markdown fences (DeepSeek, many local models)."""
    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content='```json\n{"items": ["claim-1", "claim-2"]}\n```')


class VerboseTextLLM:
    """Model that wraps JSON in commentary (reasoning models, o1-style)."""
    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=(
            "Let me analyze this requirement.\n\n"
            "After careful consideration, here are the claims:\n"
            '{"items": ["claim-1", "claim-2"]}\n\n'
            "These claims cover the key aspects of the requirement."
        ))


# Pick LLM based on strategy — in production, this maps to real models
def demo_factory(tier, node_name=None, llm_config=None):
    strategy = (llm_config or {}).get("output_strategy", "structured")
    print(f"  [{node_name}] strategy={strategy}")

    if strategy == "structured":
        return CleanLLM()
    # json_mode and text both get a non-structured LLM
    if strategy == "json_mode":
        return FencedJsonLLM()
    return VerboseTextLLM()


configure_llm(
    llm_factory=demo_factory,
    prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "analyze"}],
)


# ══════════════════════════════════════════════════════════════════════════
# THREE STRATEGIES — same schema, different models, same result
# ══════════════════════════════════════════════════════════════════════════

# Strategy 1: structured (default) — model supports with_structured_output
structured_node = Node(
    name="structured",
    mode="produce",
    output=Claims,
    model="fast",
    prompt="extract",
    # No output_strategy — defaults to "structured"
)

# Strategy 2: json_mode — model returns JSON in fences, framework parses
json_mode_node = Node(
    name="json-mode",
    mode="produce",
    output=Claims,
    model="fast",
    prompt="extract",
    llm_config={"output_strategy": "json_mode"},
)

# Strategy 3: text — model returns prose with embedded JSON, framework extracts
text_node = Node(
    name="text-mode",
    mode="produce",
    output=Claims,
    model="fast",
    prompt="extract",
    llm_config={"output_strategy": "text"},
)


# ══════════════════════════════════════════════════════════════════════════
# RUN — all three produce identical results
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Output strategy comparison:\n")

    for node in [structured_node, json_mode_node, text_node]:
        pipeline = Construct(f"test-{node.name}", nodes=[node])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "test"})
        field = node.name.replace("-", "_")
        print(f"  Result: {result[field].items}\n")

    print("All three strategies produce the same parsed output.")
    print("The consumer writes zero parsing code — NeoGraph handles it.")
    print()
    print("Production pattern:")
    print('  Node(name="decompose", mode="produce", output=Claims,')
    print('       model="reason", prompt="decompose",')
    print('       llm_config={"output_strategy": "json_mode"})  # <-- one line')
