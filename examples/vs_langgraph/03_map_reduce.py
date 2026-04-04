"""Comparison 3: Map-Reduce / Fan-Out Fan-In — process items in parallel.

Task: Generate jokes about different subjects, then pick the best one.
(Classic LangGraph tutorial example.)

LangGraph: ~65 lines (2 state schemas, Send function, Annotated reducer,
           conditional_edges, separate worker state)
NeoGraph:  ~15 lines (Oracle modifier — N parallel generators + merge)

Run (uses Gemini Flash via OpenRouter — ~$0.003):
    python examples/vs_langgraph/03_map_reduce.py
"""



import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

MODEL = "google/gemini-2.0-flash-001"

llm = ChatOpenAI(
    model=MODEL,
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
)


# ── Schemas ──────────────────────────────────────────────────────────────

class Jokes(BaseModel):
    items: list[str]

class BestJoke(BaseModel):
    id: int = Field(description="Index of the best joke, starting with 0", ge=0)


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH WAY — ~65 lines: 2 state schemas, Send, reducers
# ═══════════════════════════════════════════════════════════════════════════

def run_langgraph():
    import operator
    from typing import Annotated, TypedDict
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    # TWO state schemas needed — parent and per-worker
    class OverallState(TypedDict):
        topic: str
        subjects: list[str]
        jokes: Annotated[list[str], operator.add]  # manual reducer
        best_joke: str

    class JokeState(TypedDict):
        subject: str

    # Node: generate subjects
    def generate_subjects(state: OverallState):
        result = llm.with_structured_output(Jokes).invoke(
            f"List 3 subtopics of: {state['topic']}")
        return {"subjects": result.items}

    # Node: generate one joke (per-worker)
    def generate_joke(state: JokeState):
        result = llm.invoke(f"Write a short joke about {state['subject']}")
        return {"jokes": [result.content]}

    # Fan-out function (manual Send boilerplate)
    def fan_out_jokes(state: OverallState):
        return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

    # Node: pick best
    def pick_best(state: OverallState):
        jokes_str = "\n".join(f"{i}: {j}" for i, j in enumerate(state["jokes"]))
        result = llm.with_structured_output(BestJoke).invoke(
            f"Pick the funniest joke (return its index):\n{jokes_str}")
        return {"best_joke": state["jokes"][result.id]}

    # Wiring
    graph = StateGraph(OverallState)
    graph.add_node("generate_subjects", generate_subjects)
    graph.add_node("generate_joke", generate_joke)
    graph.add_node("pick_best", pick_best)
    graph.add_edge(START, "generate_subjects")
    graph.add_conditional_edges("generate_subjects", fan_out_jokes, ["generate_joke"])
    graph.add_edge("generate_joke", "pick_best")
    graph.add_edge("pick_best", END)
    app = graph.compile()

    result = app.invoke({"topic": "programming languages"})
    return result["best_joke"]


# ═══════════════════════════════════════════════════════════════════════════
# NEOGRAPH WAY — ~15 lines: Oracle modifier does the fan-out + merge
# ═══════════════════════════════════════════════════════════════════════════

def run_neograph():
    from neograph import Construct, Node, Oracle, compile, configure_llm, run

    configure_llm(
        llm_factory=lambda tier: llm,
        prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": (
            "Write a short joke about programming languages."
            if template == "generate"
            else f"Pick the best joke and return it as a single item list:\n"
                 + "\n".join(f"- {j.items[0]}" for j in data if j.items)
        )}],
    )

    # Oracle: 3 parallel generators + LLM merge. One line.
    generate = Node(
        name="jokes",
        mode="produce",
        output=Jokes,
        model="fast",
        prompt="generate",
    ) | Oracle(n=3, merge_prompt="pick-best")

    pipeline = Construct("joke-contest", nodes=[generate])
    graph = compile(pipeline)
    result = run(graph, input={"node_id": "demo"})
    return result["jokes"].items[0] if result["jokes"].items else "no joke"


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("LANGGRAPH (2 state schemas, Send, reducer, conditional_edges):")
    print("=" * 60)
    print(run_langgraph())

    print()
    print("=" * 60)
    print("NEOGRAPH (Node | Oracle(n=3, merge_prompt=...)):")
    print("=" * 60)
    print(run_neograph())
