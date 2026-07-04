"""Example 13b: MCP tools -- raw BaseTool in tools=, async-only, driven by arun().

A variant of example 13 that swaps the framework `Tool` spec + `tool_factories`
wiring for a raw LangChain `BaseTool` passed DIRECTLY in `tools=[...]`. This is
the ergonomics path for Model Context Protocol (MCP) tools: with
langchain-mcp-adapters you call `load_mcp_tools(session)` once, then hand the
resulting tools straight to a node. No `Tool(name)` spec, no
`register_tool_factory` -- neograph normalizes each raw BaseTool to a Tool spec
and auto-registers a factory returning it at compile time.

Two facts about MCP tools drive this example:

  1. They are async-only. langchain-mcp-adapters produces StructuredTool
     instances backed by a coroutine and no sync function. Calling `.invoke()`
     raises NotImplementedError. So the graph MUST be driven with `arun()` (the
     async driver), which runs the async tool loop (`await tool.ainvoke(...)`).
     Driving it with sync `run()` now raises a clear ConfigurationError telling
     you to use `arun()`, and `lint()` flags the same thing at compile time
     (issue kind `tool_requires_async_driver`).

  2. Their results are structured. The tool below returns a typed Pydantic
     model, which neograph preserves in `ToolInteraction.typed_result` -- MCP
     structured outputs map straight onto this.

This example runs end-to-end with NO real API keys and NO real MCP server: the
"MCP tool" is a local StructuredTool with an async implementation, and the LLM
is a fake. In production you would replace `_make_fake_mcp_tool()` with
`load_mcp_tools(session)` and the fake LLM with a real one.

Run:
    python examples/13b_mcp_tools.py
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from neograph import (
    arun,
    compile,
    construct_from_module,
    lint,
    node,
)

# -- Schemas ------------------------------------------------------------------

class EvidenceHit(BaseModel, frozen=True):
    """Typed tool result -- what the MCP tool returns. neograph preserves this
    in ToolInteraction.typed_result (MCP structured outputs map onto it)."""
    source_file: str
    line: int
    snippet: str


class ExplorationResult(BaseModel, frozen=True):
    evidence: list[str]
    summary: str


# -- The "MCP tool": a raw LangChain BaseTool, async-only --------------------
# In production this is one element of `load_mcp_tools(session)`. Here we build
# an equivalent StructuredTool from an async function so the example is
# self-contained. func=None + coroutine set == async-only (sync .invoke raises).

def _make_fake_mcp_tool() -> BaseTool:
    async def search_evidence(query: str) -> EvidenceHit:
        # A real MCP tool would round-trip a tools/call over the session here.
        await asyncio.sleep(0)  # yield to the loop, like a real network hop
        return EvidenceHit(
            source_file="auth.py",
            line=42,
            snippet="def authenticate(user, password): ...",
        )

    return StructuredTool.from_function(
        coroutine=search_evidence,
        name="search_evidence",
        description="Search the codebase for evidence supporting a claim.",
    )


# -- Fake async LLM (replace with a real model in production) -----------------

class FakeAsyncExploreLLM:
    """Async agent-mode LLM: one tool call, then a structured final answer.

    The async ReAct loop calls `await llm.ainvoke(...)`: the first turn asks for
    the tool, the second (no tool calls) is the final turn. Agent mode then
    parses the final answer through `with_structured_output(...)` -- exactly the
    method a real BaseChatModel exposes -- so the fake implements it too and
    returns the declared output model (a bare model is accepted by the
    structured-output classifier as Parsed).
    """

    _ANSWER = ExplorationResult(
        evidence=["auth.py:42"],
        summary="found a supporting reference via the MCP tool",
    )

    def __init__(self) -> None:
        self._calls = 0
        self._structured = False

    def bind_tools(self, tools: list) -> FakeAsyncExploreLLM:
        return self  # keep the call counter across rebinds

    def with_structured_output(self, model: type, **kwargs) -> FakeAsyncExploreLLM:
        clone = FakeAsyncExploreLLM()
        clone._calls = self._calls
        clone._structured = True
        return clone

    async def ainvoke(self, messages: list, **kwargs) -> AIMessage | ExplorationResult:
        if self._structured:
            # Final-turn structured parse: return the declared output model.
            return self._ANSWER
        self._calls += 1
        if self._calls == 1:
            msg = AIMessage(content="")
            msg.tool_calls = [
                {"name": "search_evidence", "args": {"query": "verify claim"}, "id": "c1"}
            ]
            return msg
        # Final turn (no tool calls): also carries the answer as JSON so the
        # loop can exit; the structured parse above produces the typed model.
        return AIMessage(content=self._ANSWER.model_dump_json())


def llm_factory(tier: str) -> FakeAsyncExploreLLM:
    return FakeAsyncExploreLLM()


def prompt_compiler(template, data, **kw):
    return [{"role": "user", "content": "explore"}]


# -- Pipeline: one agent node bound to the raw MCP tool -----------------------
# Note tools=[mcp_tool] -- a raw BaseTool, NOT a Tool(name) spec. No
# register_tool_factory call anywhere. neograph auto-registers it at compile().

mcp_tool = _make_fake_mcp_tool()


@node(
    mode="agent",
    outputs=ExplorationResult,
    model="research",
    prompt="verify/explore",
    tools=[mcp_tool],  # <-- raw LangChain BaseTool, passed directly
)
def explore() -> ExplorationResult:
    # Body unused -- the LLM drives execution via prompt= + tools=
    ...


def _build_pipeline():
    import types

    mod = types.ModuleType("mcp_example_mod")
    mod.explore = explore
    return construct_from_module(mod, name="mcp-explore")


async def main() -> None:
    pipeline = _build_pipeline()

    # lint() flags the async-only tool up front: it requires arun(). This is the
    # compile-time MCP guardrail -- no other MCP client validates bindings.
    issues = lint(pipeline)
    async_issues = [i for i in issues if i.kind == "tool_requires_async_driver"]
    print("-- lint() MCP guardrail --")
    for issue in async_issues:
        print(f"  [{issue.kind}] {issue.message}")
    print()

    graph = compile(
        pipeline,
        llm_factory=llm_factory,
        prompt_compiler=prompt_compiler,
        # No tool_factories= for search_evidence: it was auto-registered from
        # the raw BaseTool. (You could still pass tool_factories for other tools.)
    )

    # MUST use arun(): the tool is async-only. Sync run() would raise a clear
    # ConfigurationError pointing here.
    result = await arun(graph, input={"node_id": "MCP-001"})

    print("-- arun() result --")
    print(f"  summary: {result['explore'].summary}")
    print(f"  evidence: {result['explore'].evidence}")
    print()
    print("Note: 'search_evidence' had no register_tool_factory call. It was a")
    print("raw LangChain BaseTool in tools=, auto-registered at compile().")


if __name__ == "__main__":
    asyncio.run(main())
