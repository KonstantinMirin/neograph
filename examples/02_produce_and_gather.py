"""Example 2: Produce + Gather — LLM-powered pipeline with tool use.

Scenario: Analyze a software requirement. First, an LLM decomposes the
requirement into claims (think mode — single structured call). Then,
a gather node uses a search tool to research each claim, constrained
by a per-tool call budget. The tool budget prevents runaway API costs.

Budgets are enforced silently by default — the model only discovers a tool
is gone after it tries to call it. This example also opts into
`announce_tool_budget`, which prepends a framework-generated system message
telling the model its budget up front, so it can plan its calls and batch.
The announced numbers are computed at the same site that enforces them, so
they can never drift from the real budgets. Unlimited tools (`budget=0`) are
deliberately not announced. See `research` below and the printed preamble.

Built with the @node decorator + construct_from_module. Each function
becomes a Node; dependencies come from parameter names (e.g., `research`
takes `decompose: Claims`, which wires it to the upstream `decompose`
node). LLM-mode bodies are `...` — the LLM handles execution via the
`prompt=` template.

This example requires API keys (uses fakes here for demonstration).

Run:
    python examples/02_produce_and_gather.py
"""

from __future__ import annotations

import sys

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from neograph import Tool, compile, construct_from_module, node, run

# ── Schemas ──────────────────────────────────────────────────────────────

class Requirement(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]

class Finding(BaseModel, frozen=True):
    claim: str
    evidence: str

class ResearchResult(BaseModel, frozen=True):
    findings: list[Finding]

class CodeReference(BaseModel, frozen=True):
    """Typed tool result -- what search_codebase returns."""
    query: str
    matches: int
    top_file: str


# ── Fake LLM (replace with real OpenRouter/OpenAI in production) ─────────

class FakeDecomposeLLM:
    """Simulates an LLM that decomposes a requirement into claims."""
    def with_structured_output(self, model):
        self._model = model
        return self

    def invoke(self, messages, **kwargs):
        return self._model(items=[
            "system shall authenticate users",
            "system shall log failed attempts",
            "system shall rate-limit login",
        ])


# Captures the budget preamble the framework injects, so we can show the
# consumer exactly what the model is told. In production you never touch this —
# the framework prepends it and the model reads it.
announced = {"preamble": None}


def _first_system_content(messages):
    """Return the content of the first system message, dict- or object-shaped."""
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
        if role == "system":
            return m.get("content") if isinstance(m, dict) else m.content
    return None


class FakeResearchLLM:
    """Simulates an LLM that calls a search tool, then responds."""
    def __init__(self):
        self._call_count = 0

    def bind_tools(self, tools):
        clone = FakeResearchLLM()
        clone._call_count = self._call_count
        clone._has_tools = len(tools) > 0
        return clone

    def invoke(self, messages, **kwargs):
        self._call_count += 1
        if announced["preamble"] is None:
            announced["preamble"] = _first_system_content(messages)
        if getattr(self, '_has_tools', True) and self._call_count <= 3:
            msg = AIMessage(content="")
            msg.tool_calls = [{
                "name": "search_codebase",
                "args": {"query": f"claim-{self._call_count}"},
                "id": f"call-{self._call_count}",
            }]
            return msg
        # Agent mode parses the final ReAct turn as JSON (the model is told the
        # output schema up front), so emit a valid ResearchResult.
        return AIMessage(
            content=ResearchResult(
                findings=[Finding(claim="authentication", evidence="auth.py:42")]
            ).model_dump_json()
        )

    def with_structured_output(self, model):
        self._model = model
        return self


# ── Fake tool ────────────────────────────────────────────────────────────

search_count = {"n": 0}

class FakeSearchTool:
    """Returns a typed CodeReference model, not a string.
    The framework preserves it in ToolInteraction.typed_result."""
    name = "search_codebase"

    def invoke(self, args):
        search_count["n"] += 1
        query = args.get("query", "?")
        return CodeReference(query=query, matches=3, top_file="auth.py")


# ── Configure LLM layer ──────────────────────────────────────────────────

def llm_factory(tier):
    if tier == "fast":
        return FakeDecomposeLLM()
    return FakeResearchLLM()


# ── Pipeline nodes ───────────────────────────────────────────────────────
# Step 1: Decompose requirement into claims (single LLM call)
# Step 2: Research claims using search tool (budget: max 2 searches)

@node(outputs=Claims, model="fast", prompt="req/decompose")
def decompose() -> Claims:
    # body unused for mode='think' — LLM handles execution via prompt=
    ...


@node(
    mode="agent",
    outputs=ResearchResult,
    model="reason",
    prompt="req/research",
    tools=[Tool(name="search_codebase", budget=2)],  # max 2 searches
    # Announce the budget to the model up front so it plans + batches.
    # Off by default; the announced count is derived from the Tool budget above.
    llm_config={"announce_tool_budget": True},
)
def research(decompose: Claims) -> ResearchResult:
    # body unused for mode='agent' — LLM handles execution via prompt=
    ...


pipeline = construct_from_module(sys.modules[__name__], name="requirement-analysis")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(
        pipeline,
        llm_factory=llm_factory,
        prompt_compiler=lambda template, data: [{"role": "user", "content": "analyze"}],
        tool_factories={"search_codebase": lambda config, tool_config: FakeSearchTool()},
    )
    result = run(graph, input={"node_id": "REQ-042"})

    print(f"Decomposed into {len(result['decompose'].items)} claims:")
    for claim in result["decompose"].items:
        print(f"  - {claim}")

    print(f"\nSearch tool called {search_count['n']} times (budget was 2)")
    print(f"Research complete: {result['research'] is not None}")

    print("\nBudget preamble the model was told (announce_tool_budget=True):")
    print("-" * 60)
    print(announced["preamble"])
    print("-" * 60)
