"""Comparison 1: Sequential Pipeline — 3 LLM steps in order.

Task: Take a topic → decompose into claims → classify each claim → summarize.

LangGraph: ~35 lines of wiring (StateGraph, add_node x3, add_edge x4, compile)
NeoGraph:  ~10 lines (3 Node declarations + Construct + compile + run)

Run (uses Gemini Flash via OpenRouter — ~$0.001):
    python examples/vs_langgraph/01_sequential_pipeline.py
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


# ── Schemas (shared by both approaches) ─────────────────────────────────

class Claims(BaseModel):
    items: list[str]

class ClassifiedClaims(BaseModel):
    classified: list[dict[str, str]]

class Summary(BaseModel):
    text: str


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH WAY — 35 lines of wiring
# ═══════════════════════════════════════════════════════════════════════════

def run_langgraph():
    from typing import Annotated, TypedDict
    from langgraph.graph import END, START, StateGraph

    # --- State schema (manual, must include every field) ---
    class PipelineState(TypedDict):
        topic: str
        claims: Claims | None
        classified: ClassifiedClaims | None
        summary: Summary | None

    # --- Node functions (each must read/write state dict) ---
    def decompose(state: PipelineState):
        result = llm.with_structured_output(Claims).invoke(
            f"Break this topic into 3-5 factual claims: {state['topic']}")
        return {"claims": result}

    def classify(state: PipelineState):
        result = llm.with_structured_output(ClassifiedClaims).invoke(
            f"Classify each claim by category (security/reliability/performance): {state['claims']}")
        return {"classified": result}

    def summarize(state: PipelineState):
        result = llm.with_structured_output(Summary).invoke(
            f"Summarize these classified claims in one paragraph: {state['classified']}")
        return {"summary": result}

    # --- Wiring (N add_node + N add_edge) ---
    graph = StateGraph(PipelineState)
    graph.add_node("decompose", decompose)
    graph.add_node("classify", classify)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "classify")
    graph.add_edge("classify", "summarize")
    graph.add_edge("summarize", END)
    app = graph.compile()

    result = app.invoke({"topic": "microservice authentication"})
    return result["summary"].text


# ═══════════════════════════════════════════════════════════════════════════
# NEOGRAPH WAY — 10 lines of wiring
# ═══════════════════════════════════════════════════════════════════════════

def run_neograph():
    from neograph import Construct, Node, compile, configure_llm, run

    configure_llm(
        llm_factory=lambda tier: llm,
        prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": (
            f"Break this topic into 3-5 factual claims: {kw.get('config', {}).get('configurable', {}).get('topic', 'AI')}"
            if template == "decompose"
            else f"Classify each claim (security/reliability/performance): {data}" if template == "classify"
            else f"Summarize in one paragraph: {data}"
        )}],
    )

    # 3 nodes, ordered. No add_edge, no add_node, no StateGraph.
    decompose = Node(name="decompose", mode="produce", outputs=Claims, model="fast", prompt="decompose")
    classify = Node(name="classify", mode="produce", inputs=Claims, outputs=ClassifiedClaims, model="fast", prompt="classify")
    summarize = Node(name="summarize", mode="produce", inputs=ClassifiedClaims, outputs=Summary, model="fast", prompt="summarize")

    pipeline = Construct("analysis", nodes=[decompose, classify, summarize])
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "demo", "topic": "microservice authentication"})
    return result["summarize"].text


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("LANGGRAPH:")
    print("=" * 60)
    print(run_langgraph())

    print()
    print("=" * 60)
    print("NEOGRAPH:")
    print("=" * 60)
    print(run_neograph())
