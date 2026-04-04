"""Comparison 5: Subgraph Composition — nested pipeline with isolated state.

Task: Parent pipeline decomposes a topic into claims, a sub-pipeline
enriches them (lookup context + score), then parent generates a report.

LangGraph: ~45 lines (2 StateGraphs, 2 TypedDicts, manual state mapping
           in a wrapper function, 2x compile)
NeoGraph:  ~12 lines (Construct with input/output, nested in parent)

Run (uses Gemini Flash via OpenRouter — ~$0.003):
    python examples/vs_langgraph/05_subgraph.py
"""



import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

MODEL = "google/gemini-2.0-flash-001"

llm = ChatOpenAI(
    model=MODEL,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
)


class Claims(BaseModel):
    items: list[str]

class ScoredClaims(BaseModel):
    scored: list[dict[str, str]]

class Report(BaseModel):
    text: str


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH WAY — ~45 lines: 2 graphs, 2 TypedDicts, manual state mapping
# ═══════════════════════════════════════════════════════════════════════════

def run_langgraph():
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    # --- CHILD GRAPH (separate state schema) ---
    class EnrichState(TypedDict):
        claims: Claims | None
        scored: ScoredClaims | None

    def enrich_lookup(state: EnrichState):
        result = llm.with_structured_output(ScoredClaims).invoke(
            f"Score each claim as high/medium/low confidence: {state['claims']}")
        return {"scored": result}

    child_graph = StateGraph(EnrichState)
    child_graph.add_node("lookup", enrich_lookup)
    child_graph.add_edge(START, "lookup")
    child_graph.add_edge("lookup", END)
    child = child_graph.compile()

    # --- PARENT GRAPH (different state schema) ---
    class ParentState(TypedDict):
        topic: str
        claims: Claims | None
        enriched: ScoredClaims | None
        report: str

    def decompose(state: ParentState):
        result = llm.with_structured_output(Claims).invoke(
            f"Break '{state['topic']}' into 3-5 factual claims.")
        return {"claims": result}

    # MANUAL state mapping — user must translate parent ↔ child
    def enrich_wrapper(state: ParentState):
        child_result = child.invoke({"claims": state["claims"]})
        return {"enriched": child_result["scored"]}

    def report(state: ParentState):
        result = llm.with_structured_output(Report).invoke(
            f"Write a brief report from these scored claims: {state['enriched']}")
        return {"report": result.text}

    parent_graph = StateGraph(ParentState)
    parent_graph.add_node("decompose", decompose)
    parent_graph.add_node("enrich", enrich_wrapper)
    parent_graph.add_node("report", report)
    parent_graph.add_edge(START, "decompose")
    parent_graph.add_edge("decompose", "enrich")
    parent_graph.add_edge("enrich", "report")
    parent_graph.add_edge("report", END)
    app = parent_graph.compile()

    result = app.invoke({"topic": "API rate limiting"})
    return result["report"]


# ═══════════════════════════════════════════════════════════════════════════
# NEOGRAPH WAY — ~12 lines: Construct with input/output, nested
# ═══════════════════════════════════════════════════════════════════════════

def run_neograph():
    from neograph import Construct, Node, compile, configure_llm, run

    configure_llm(
        llm_factory=lambda tier: llm,
        prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": (
            f"Break 'API rate limiting' into 3-5 factual claims." if template == "decompose"
            else f"Score each claim as high/medium/low: {data}" if template == "score"
            else f"Write a brief report: {data}"
        )}],
    )

    # Sub-pipeline with declared I/O boundary — that's it
    enrich = Construct(
        "enrich",
        input=Claims,
        output=ScoredClaims,
        nodes=[Node(name="score", mode="produce", input=Claims, output=ScoredClaims, model="fast", prompt="score")],
    )

    pipeline = Construct("analysis", nodes=[
        Node(name="decompose", mode="produce", output=Claims, model="fast", prompt="decompose"),
        enrich,  # ← sub-pipeline, isolated state, no wrapper function
        Node(name="report", mode="produce", input=ScoredClaims, output=Report, model="fast", prompt="report"),
    ])

    graph = compile(pipeline)
    result = run(graph, input={"node_id": "demo"})
    return result["report"].text


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("LANGGRAPH (2 TypedDicts, 2 StateGraphs, manual state mapping):")
    print("=" * 60)
    print(run_langgraph())

    print()
    print("=" * 60)
    print("NEOGRAPH (Construct with input/output, nested in parent):")
    print("=" * 60)
    print(run_neograph())
