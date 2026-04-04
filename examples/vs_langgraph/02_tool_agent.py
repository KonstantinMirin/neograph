"""Comparison 2: Tool-Calling Agent — LLM decides which tools to call.

Task: Research a topic using a search tool, with a budget limit.

LangGraph: ~40 lines (router function, conditional edges, tool loop cycle)
NeoGraph:  ~8 lines (one gather Node with tools + budget)

The ReAct pattern (call LLM → if tool_calls → execute → call LLM again)
is identical in every LangGraph agent. NeoGraph's gather mode handles
the entire loop, including per-tool budget enforcement.

Run (uses Gemini Flash via OpenRouter — ~$0.002):
    python examples/vs_langgraph/02_tool_agent.py
"""



import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

MODEL = "google/gemini-2.0-flash-001"

base_llm = ChatOpenAI(
    model=MODEL,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
)


# ── Tool (shared by both approaches) ────────────────────────────────────

search_calls = {"langgraph": 0, "neograph": 0}

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    search_calls["current"] = search_calls.get("current", 0) + 1
    return f"Search result for '{query}': Found 3 relevant articles about {query}."


class ResearchResult(BaseModel):
    findings: list[str]
    sources_consulted: int


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH WAY — ~40 lines: router, conditional edges, manual cycle
# ═══════════════════════════════════════════════════════════════════════════

def run_langgraph():
    from typing import Annotated, TypedDict
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode

    search_calls["current"] = 0

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    tools = [search_web]
    llm_with_tools = base_llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    # --- Node: call model ---
    def agent(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # --- Router: check if model wants to call tools ---
    # (This is IDENTICAL in every ReAct agent — pure boilerplate)
    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        return END

    # --- Wiring ---
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")  # cycle back
    app = graph.compile()

    result = app.invoke({"messages": [HumanMessage(
        content="Research microservice authentication patterns. Use the search tool."
    )]})
    last_msg = result["messages"][-1].content
    return last_msg, search_calls["current"]


# ═══════════════════════════════════════════════════════════════════════════
# NEOGRAPH WAY — ~8 lines: one Node, tools + budget, done
# ═══════════════════════════════════════════════════════════════════════════

def run_neograph():
    from neograph import Construct, Node, Tool, compile, configure_llm, register_tool_factory, run

    search_calls["current"] = 0

    # Tool factory: NeoGraph creates tool instances per-node
    register_tool_factory("search_web", lambda config, tool_config: search_web)

    configure_llm(
        llm_factory=lambda tier: base_llm,
        prompt_compiler=lambda template, data, **kw: [
            {"role": "user", "content": "Research microservice authentication patterns. Use the search tool."},
        ],
    )

    # One node. Gather mode = ReAct loop. Budget = max 3 searches.
    research = Node(
        name="research",
        mode="gather",
        output=ResearchResult,
        model="fast",
        prompt="research",
        tools=[Tool(name="search_web", budget=3)],
    )

    pipeline = Construct("research", nodes=[research])
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "demo"})
    return result["research"], search_calls["current"]


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("LANGGRAPH (no budget — LLM decides when to stop):")
    print("=" * 60)
    text, calls = run_langgraph()
    print(f"Search calls: {calls}")
    print(f"Result: {text[:200]}...")

    print()
    print("=" * 60)
    print("NEOGRAPH (budget=3 — enforced, then forced to respond):")
    print("=" * 60)
    result, calls = run_neograph()
    print(f"Search calls: {calls}")
    print(f"Result: {result}")
