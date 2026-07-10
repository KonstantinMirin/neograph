"""Example 25: binding ONE gateway-federated MCP tool — the singular ``mcp_tool_factory``.

Example 23 shows the plural ``mcp_tool_factories(...)``: connect at build time,
DISCOVER every tool the servers expose, slice the returned dict per node. This
example shows its counterpart for when you already KNOW the tool's name and want
the build path to stay OFFLINE:

    mcp_tool_factories(servers)          # plural  — asks the server (discovery,
                                         #           connects at build, fail-fast)
    mcp_tool_factory(key, spec,          # singular — tells the server (declared
                     tool_name=...)      #           name, ZERO network at build)

The mnemonic: the plural asks the server, the singular tells the server.

The scenario is the gateway-federated case that motivated the helper (filed by a
real consumer). An MCP gateway (e.g. IBM ContextForge) re-exposes a peer's tool
NAMESPACED as ``<peer>-<tool>`` — here the demo server exposes
``crm-perplexity_research``. The node, written long before the gateway config,
binds the bare name ``Tool("perplexity_research")``. The singular factory bridges
the two: it binds the declared namespaced name and renames the fetched tool back
to the bare binding, and nothing connects until the agent's first tool call.

Three beats:

  1. OFFLINE BUILD — constructing the factory (and compiling the graph) performs
     zero network I/O. We prove it by building a factory against a spec that
     CANNOT connect (a nonexistent command): construction still succeeds, because
     the connect lives inside the returned async factory, deferred to first await.
     This is what keeps a consumer's compile()-time and test paths offline.

  2. NAMESPACED -> BARE RENAME — ``tool_name="crm-perplexity_research",
     rename_to="perplexity_research"`` makes both the factory key and the
     LLM-facing ``tool.name`` match the node's fixed ``Tool`` spec.

  3. PER-RUN IDENTITY, SAME PATH — the singular takes the same ``token_provider``
     as the plural; over stdio the token rides as a tool argument the server
     echoes back (``acting_as``), exactly as in example 23 beat 3.

The MCP layer is real (the same stdio demo server as examples 23/24); the only
fake is the LLM, so the example is deterministic and keyless.

Run (needs the mcp-examples extra; no API keys, no network beyond the local
subprocess):
    uv run --extra dev --extra mcp-examples python examples/25_mcp_single_tool_gateway.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from neograph import Tool, ToolInteraction, arun, compile, construct_from_functions, node
from neograph_mcp import StdioServer, mcp_tool_factory

DEMO_SERVER = str((Path(__file__).parent / "_mcp_demo_server.py").resolve())

CRM_GATEWAY = StdioServer(command=sys.executable, args=[DEMO_SERVER])


# ── Schemas ──────────────────────────────────────────────────────────────────


class ResearchNote(BaseModel, frozen=True):
    query: str
    finding: str


class PerplexityResult(BaseModel, frozen=True):
    """Client-side model for the gateway tool's structuredContent — declaring it as
    output_model= makes typed_result the model, not a raw content-block list."""

    query: str
    acting_as: str  # the per-run identity the server echoed


# ── Fake LLM (the ONLY fake — the MCP layer is real) ─────────────────────────
# One agent turn: call perplexity_research, then finalize from its result.


class _ResearchFake:
    _NOTE = ResearchNote(query="acme", finding="Acme renewal research complete.")

    def __init__(self) -> None:
        self._structured = False

    def bind_tools(self, tools: list) -> _ResearchFake:
        return self

    def abind_tools(self, *a: Any, **k: Any) -> _ResearchFake:
        return self

    def with_structured_output(self, model: type[BaseModel], **k: Any) -> _ResearchFake:
        clone = _ResearchFake()
        clone._structured = True
        return clone

    def invoke(self, messages: list, **k: Any) -> Any:
        if self._structured:
            return self._NOTE
        if not any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content="")
            # The model calls the BARE name — it never sees the gateway namespace.
            msg.tool_calls = [{"name": "perplexity_research", "args": {"query": "acme"}, "id": "p1"}]
            return msg
        return AIMessage(content=self._NOTE.model_dump_json())

    async def ainvoke(self, *a: Any, **k: Any) -> Any:
        return self.invoke(*a, **k)


def llm_factory(tier: str) -> Any:
    return _ResearchFake()


def prompt_compiler(template: str, data: Any, **kw: Any) -> list[dict[str, str]]:
    return [{"role": "user", "content": template}]


# ── The node: written against the BARE tool name ─────────────────────────────


@node(
    mode="agent",
    outputs={"result": ResearchNote, "tool_log": list[ToolInteraction]},
    model="research",
    prompt="research/note",
    tools=[Tool("perplexity_research", budget=1, idempotent=True)],
)
def research() -> ResearchNote:  # body unused — the LLM drives via prompt= + tools=
    ...


pipeline = construct_from_functions("gateway-research", [research])


# ── Demos ─────────────────────────────────────────────────────────────────────


def demo_offline_build() -> None:
    """Beat 1: construction connects NOWHERE — even an unreachable spec builds."""
    print("=" * 66)
    print("BEAT 1: offline build — zero network at construction")
    print("=" * 66)

    unreachable = StdioServer(command="/nonexistent/not-a-real-gateway", args=["--nope"])
    factory = mcp_tool_factory(
        "crm",
        unreachable,
        tool_name="crm-perplexity_research",
        rename_to="perplexity_research",
    )
    # No subprocess spawned, no get_tools called. The connect lives inside the
    # returned async factory and fires only on its first await.
    assert asyncio.iscoroutinefunction(factory)
    print("\nBuilt a factory against an UNREACHABLE spec — no error, no connect.")
    print("Discovery is a connect, so the plural mcp_tool_factories() would have")
    print("failed here at build time. The singular defers: your compile() and")
    print("test paths stay offline; a wrong name fails loud at first tool call.")


async def demo_rename_and_identity() -> None:
    """Beats 2 + 3: namespaced->bare rename and per-run identity, end to end."""
    print("\n" + "=" * 66)
    print("BEAT 2/3: gateway rename (crm-perplexity_research -> perplexity_research)")
    print("=" * 66)

    factory = mcp_tool_factory(
        "crm",
        CRM_GATEWAY,
        tool_name="crm-perplexity_research",  # the name the GATEWAY exposes
        rename_to="perplexity_research",  # the name the NODE binds
        token_provider=lambda configurable: configurable.get("mcp_auth", "anon"),
        # TYPED RESULT: rehydrate the server's structuredContent into our model, so
        # typed_result IS a PerplexityResult — no hand-parsing of content blocks.
        output_model=PerplexityResult,
    )

    # Still offline: compiling binds the factory under the bare name; nothing has
    # connected yet.
    graph = compile(
        pipeline,
        llm_factory=llm_factory,
        prompt_compiler=prompt_compiler,
        tool_factories={"perplexity_research": factory},
    )
    print("\nCompiled the graph — still zero MCP connects.")

    # The first tool call inside the run is where the connect finally happens.
    result = await arun(
        graph,
        input={"node_id": "GW-1"},
        config={"configurable": {"mcp_auth": "operator-A"}},
    )
    note: ResearchNote = result["research_result"]
    tool_log: list[ToolInteraction] = result["research_tool_log"]

    call = tool_log[0]
    payload = call.typed_result  # a PerplexityResult model, not raw content blocks
    print(f"\nLLM-facing tool name : {call.tool_name}")
    print(f"server-side identity : acting_as={payload.acting_as!r}")
    print(f"final result         : {note.finding}")
    assert call.tool_name == "perplexity_research"  # the bare binding, not <peer>-<tool>
    assert payload.acting_as == "operator-A"  # same token path as the plural
    assert payload.query == "acme"


async def main() -> None:
    demo_offline_build()
    await demo_rename_and_identity()
    print("\n" + "=" * 66)
    print("Done. One declared tool, bound offline, renamed to the node's bare")
    print("Tool(name), carrying per-run identity — no discovery connect anywhere.")
    print("=" * 66)


if __name__ == "__main__":
    asyncio.run(main())
