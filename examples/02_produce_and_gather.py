"""Example 2: Produce + Gather — LLM-powered pipeline with tool use.

Scenario: Analyze a software requirement. First, an LLM decomposes the
requirement into claims (produce mode — single structured call). Then,
a gather node uses a search tool to research each claim, constrained
by a per-tool call budget. The tool budget prevents runaway API costs.

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

from neograph import Tool, compile, configure_llm, construct_from_module, node, register_tool_factory, run


# ── Schemas ──────────────────────────────────────────────────────────────

class Requirement(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]

class ResearchResult(BaseModel, frozen=True):
    findings: list[dict[str, str]]


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
        if getattr(self, '_has_tools', True) and self._call_count <= 3:
            msg = AIMessage(content="")
            msg.tool_calls = [{
                "name": "search_codebase",
                "args": {"query": f"claim-{self._call_count}"},
                "id": f"call-{self._call_count}",
            }]
            return msg
        return AIMessage(content="research complete")

    def with_structured_output(self, model):
        self._model = model
        return self


# ── Fake tool ────────────────────────────────────────────────────────────

search_count = {"n": 0}

class FakeSearchTool:
    name = "search_codebase"

    def invoke(self, args):
        search_count["n"] += 1
        return f"Found 3 references for: {args.get('query', '?')}"

register_tool_factory("search_codebase", lambda config, tool_config: FakeSearchTool())


# ── Configure LLM layer ──────────────────────────────────────────────────

def llm_factory(tier):
    if tier == "fast":
        return FakeDecomposeLLM()
    return FakeResearchLLM()

configure_llm(
    llm_factory=llm_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": "analyze"}],
)


# ── Pipeline nodes ───────────────────────────────────────────────────────
# Step 1: Decompose requirement into claims (single LLM call)
# Step 2: Research claims using search tool (budget: max 2 searches)

@node(mode="produce", output=Claims, model="fast", prompt="req/decompose")
def decompose() -> Claims:
    # body unused for mode='produce' — LLM handles execution via prompt=
    ...


@node(
    mode="gather",
    output=ResearchResult,
    model="reason",
    prompt="req/research",
    tools=[Tool(name="search_codebase", budget=2)],  # max 2 searches
)
def research(decompose: Claims) -> ResearchResult:
    # body unused for mode='gather' — LLM handles execution via prompt=
    ...


pipeline = construct_from_module(sys.modules[__name__], name="requirement-analysis")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "REQ-042"})

    print(f"Decomposed into {len(result['decompose'].items)} claims:")
    for claim in result["decompose"].items:
        print(f"  - {claim}")

    print(f"\nSearch tool called {search_count['n']} times (budget was 2)")
    print(f"Research complete: {result['research'] is not None}")
