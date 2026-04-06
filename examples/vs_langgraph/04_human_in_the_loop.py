"""Comparison 4: Human-in-the-Loop — interrupt for approval, then resume.

Task: Analyze a topic, validate quality, pause for human review if
quality is below threshold, then generate final report.

LangGraph: ~50 lines (checkpointer, interrupt(), Command routing,
           two-phase invocation, separate router node)
NeoGraph:  ~12 lines (Operator modifier + condition function)

Run (uses Gemini Flash via OpenRouter — ~$0.002):
    python examples/vs_langgraph/04_human_in_the_loop.py
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


class Analysis(BaseModel):
    claims: list[str]
    confidence: float

class Report(BaseModel):
    text: str


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH WAY — ~50 lines: checkpointer, interrupt, Command, 2-phase
# ═══════════════════════════════════════════════════════════════════════════

def run_langgraph():
    from typing import TypedDict
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    class State(TypedDict):
        topic: str
        analysis: Analysis | None
        human_approved: bool
        report: str

    def analyze(state: State):
        result = llm.with_structured_output(Analysis).invoke(
            f"Analyze '{state['topic']}'. List 3 claims and rate your confidence 0-1.")
        return {"analysis": result}

    # Separate node JUST for the interrupt check
    def review_gate(state: State):
        if state["analysis"].confidence < 0.8:
            decision = interrupt({
                "message": f"Confidence {state['analysis'].confidence:.0%} is low. Approve?",
                "claims": state["analysis"].claims,
            })
            return {"human_approved": decision.get("approved", False)}
        return {"human_approved": True}

    # Another separate node for routing
    def route_after_review(state: State) -> Command:
        if state["human_approved"]:
            return Command(goto="report")
        return Command(goto=END)

    def report(state: State):
        result = llm.with_structured_output(Report).invoke(
            f"Write a brief report based on: {state['analysis']}")
        return {"report": result.text}

    # Wiring — 4 nodes, edges, conditional routing
    memory = MemorySaver()
    graph = StateGraph(State)
    graph.add_node("analyze", analyze)
    graph.add_node("review_gate", review_gate)
    graph.add_node("route", route_after_review)
    graph.add_node("report", report)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "review_gate")
    graph.add_edge("review_gate", "route")
    graph.add_edge("report", END)
    app = graph.compile(checkpointer=memory)

    # Two-phase invocation
    config = {"configurable": {"thread_id": "lg-demo"}}
    result = app.invoke({"topic": "microservice security"}, config)

    if "__interrupt__" in result:
        print(f"  Interrupted: {result['__interrupt__'][0].value['message']}")
        # Resume with approval
        result = app.invoke(Command(resume={"approved": True}), config)

    return result.get("report", "no report")


# ═══════════════════════════════════════════════════════════════════════════
# NEOGRAPH WAY — ~12 lines: Operator modifier, done
# ═══════════════════════════════════════════════════════════════════════════

def run_neograph():
    from langgraph.checkpoint.memory import MemorySaver

    from neograph import (Construct, Node, Operator, compile, configure_llm,
                          register_condition, run)

    configure_llm(
        llm_factory=lambda tier: llm,
        prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": (
            f"Analyze 'microservice security'. List 3 claims and rate confidence 0-1."
            if template == "analyze"
            else f"Write a brief report based on: {data}"
        )}],
    )

    # Condition: when should the graph pause?
    register_condition("low_confidence", lambda state: (
        {"message": f"Confidence {state.analyze.confidence:.0%} is low. Approve?"}
        if state.analyze and state.analyze.confidence < 0.8
        else None
    ))

    # Pipeline — Operator modifier handles the interrupt
    analyze = Node(name="analyze", mode="think", outputs=Analysis, model="fast", prompt="analyze")
    report = Node(name="report", mode="think", inputs=Analysis, outputs=Report, model="fast", prompt="report")

    pipeline = Construct("review", nodes=[
        analyze | Operator(when="low_confidence"),  # ← one pipe
        report,
    ])

    graph = compile(pipeline, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "neo-demo"}}

    result = run(graph, input={"node_id": "demo"}, config=config)

    if "__interrupt__" in result:
        print(f"  Interrupted: {result['__interrupt__'][0].value['message']}")
        result = run(graph, resume={"approved": True}, config=config)

    return result["report"].text


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("LANGGRAPH (checkpointer + interrupt + Command + router node):")
    print("=" * 60)
    print(run_langgraph())

    print()
    print("=" * 60)
    print("NEOGRAPH (Node | Operator(when=...)):")
    print("=" * 60)
    print(run_neograph())
