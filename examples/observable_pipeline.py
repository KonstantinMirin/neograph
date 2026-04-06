"""Runnable example: observable LLM pipeline with Langfuse tracing.

Requires .env with:
    OPENROUTER_API_KEY=sk-or-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_BASE_URL=https://cloud.langfuse.com

Run:
    python examples/observable_pipeline.py
"""

from __future__ import annotations

import os
import sys

import structlog
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel

from neograph import compile, configure_llm, construct_from_module, node, run

# ── Structlog: human-readable for this example ──────────────────────────

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)

# ── Schemas ──────────────────────────────────────────────────────────────


class Topic(BaseModel, frozen=True):
    text: str


class Claims(BaseModel, frozen=True):
    items: list[str]


# ── LLM factory: OpenRouter ─────────────────────────────────────────────

MODELS = {
    "fast": "google/gemini-2.0-flash-001",
    "reason": "google/gemini-2.0-flash-001",
}


def llm_factory(tier: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODELS.get(tier, MODELS["fast"]),
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
    )


def prompt_compiler(template: str, input_data) -> list[dict]:
    if template == "decompose":
        return [{"role": "user", "content": (
            "Break the following topic into 3-5 distinct factual claims. "
            "Return ONLY a JSON object with an 'items' field containing a list of strings.\n\n"
            f"Topic: {input_data.text if input_data else 'artificial intelligence'}"
        )}]
    if template == "merge-claims":
        # input_data is a list of Claims from Oracle generators
        all_claims = []
        for claims in input_data:
            all_claims.extend(claims.items)
        return [{"role": "user", "content": (
            "You received multiple decompositions of a topic. "
            "Deduplicate and synthesize into one definitive list of 3-5 claims. "
            "Return ONLY a JSON object with an 'items' field containing a list of strings.\n\n"
            "All claims:\n" + "\n".join(f"- {c}" for c in all_claims)
        )}]
    return [{"role": "user", "content": "Hello"}]


configure_llm(llm_factory=llm_factory, prompt_compiler=prompt_compiler)

# ── Pipeline ─────────────────────────────────────────────────────────────

# produce: LLM decomposes topic into claims (3 variants via Oracle, LLM merge)
@node(inputs=Topic, outputs=Claims, prompt="decompose", model="fast",
      ensemble_n=3, merge_prompt="merge-claims")
def decompose(topic: Topic) -> Claims: ...


# scripted: format the merged result
@node(outputs=Topic)
def report(decompose: Claims) -> Topic:
    return Topic(text="Final report:\n" + "\n".join(f"  - {c}" for c in decompose.items))


pipeline = construct_from_module(sys.modules[__name__], name="observable-demo")

# ── Run with Langfuse callback ───────────────────────────────────────────

if __name__ == "__main__":
    langfuse_handler = CallbackHandler()

    graph = compile(pipeline)
    result = run(
        graph,
        input={"node_id": "demo-001"},
        config={"callbacks": [langfuse_handler]},
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)

    merged = result.get("decompose")
    if merged:
        print(f"\n  Merged claims ({len(merged.items)}):")
        for claim in merged.items:
            print(f"    - {claim}")

    report_out = result.get("report")
    if report_out:
        print(f"\n  Report:\n{report_out.text}")

    # Flush traces to Langfuse
    from langfuse import get_client
    get_client().flush()
    print("\n\nTraces pushed to Langfuse.")
