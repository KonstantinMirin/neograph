"""Lead Outreach Email Sequence -- neograph mini-project.

Define your types. Define your prompts. Wire it up. Done.

    OPENROUTER_API_KEY=sk-... python examples/lead-outreach/pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from neograph import (
    Construct,
    Each,
    Node,
    compile,
    configure_llm,
    construct_from_functions,
    node,
    run,
)
from neograph import Loop, Oracle

from schemas import EmailState, EmailStates, KeyIdeas, LeadProfile


# =============================================================================
# Data + LLM setup
# =============================================================================

def _load_lead() -> LeadProfile:
    with open(_HERE / "data" / "lead_profile.json") as f:
        return LeadProfile(**json.load(f))

def _load_ideas() -> KeyIdeas:
    with open(_HERE / "data" / "key_ideas.yaml") as f:
        return KeyIdeas(**yaml.safe_load(f))

def _prompt(name: str) -> str:
    return (_HERE / "prompts" / f"{name}.md").read_text()

MODELS = {
    "reason": "anthropic/claude-sonnet-4",
    "fast": "google/gemini-2.0-flash-001",
    "creative": "openai/gpt-4o",
}

def _llm_factory(tier: str, *, node_name: str = "", llm_config: dict | None = None):
    from langchain_openai import ChatOpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY to run this example.")
    return ChatOpenAI(
        model=MODELS.get(tier, MODELS["fast"]),
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=(llm_config or {}).get("temperature", 0.7),
        max_tokens=(llm_config or {}).get("max_tokens", 4000),
    )

configure_llm(
    llm_factory=_llm_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": template}],
)


# =============================================================================
# Pipeline — this is the whole thing
# =============================================================================

@node(outputs=EmailStates)
def plan_sequence() -> EmailStates:
    lead, ideas = _load_lead(), _load_ideas()
    specs = [
        (1, 0, "cold_open", ideas.relevant_to_lead[2],
         "Reference their recent post. Ask a question. No pitch."),
        (2, 3, "value_drop", ideas.relevant_to_lead[1],
         "Share an insight connecting their hiring to type safety. Do not sell."),
        (3, 7, "soft_ask", ideas.angles[0],
         "Reference emails 1-2. One ask: 15 min demo. Easy to say yes."),
        (4, 14, "breakup", ideas.angles[1],
         "Final email. Direct. Social proof. Leave door open."),
    ]
    return EmailStates(items=[
        EmailState(email_number=n, send_day=d, intent=i, angle=a,
                   constraints=c, lead=lead, ideas=ideas)
        for n, d, i, a, c in specs
    ])


# draft (3 models, judge picks best) → feedback (LLM-as-prospect scores)
# loop until score >= 0.8
_draft_feedback = Construct(
    "draft-feedback",
    input=EmailState,
    output=EmailState,
    nodes=[
        Node(
            name="draft", mode="think",
            inputs=EmailState, outputs=EmailState,
            model="reason", prompt=_prompt("draft"),
            llm_config={"temperature": 0.8},
        ) | Oracle(
            models=["reason", "fast", "creative"],
            merge_prompt=_prompt("pick_best"),
        ),
        Node(
            name="feedback", mode="think",
            inputs=EmailState, outputs=EmailState,
            model="reason", prompt=_prompt("feedback"),
            llm_config={"temperature": 0.3},
        ),
    ],
) | Loop(when=lambda s: s is None or s.score < 0.8, max_iterations=3, on_exhaust="last")

# one instance per email, 4 in parallel
produce_email = Construct(
    "produce-email", input=EmailState, output=EmailState,
    nodes=[_draft_feedback],
) | Each(over="plan_sequence.items", key="intent")

pipeline = construct_from_functions("lead-outreach", [plan_sequence, produce_email])


# =============================================================================
# Run
# =============================================================================

def main():
    lead = _load_lead()
    print(f"Lead: {lead.first_name} {lead.last_name}, {lead.headline}")
    print(f"Models: {list(MODELS.values())}\n")

    graph = compile(pipeline)
    result = run(graph, input={"node_id": f"outreach-{lead.company.lower()}"})

    labels = {0: "Cold Open", 3: "Value Drop", 7: "Soft Ask", 14: "Breakup"}
    emails = result["produce_email"]

    for intent in ["cold_open", "value_drop", "soft_ask", "breakup"]:
        e = emails[intent]
        print(f"--- {labels[e.send_day]} (day {e.send_day}) | score {e.score} | {e.iteration} iterations ---")
        print(f"Subject: {e.subject}\n")
        for i in range(0, len(e.body), 70):
            print(f"  {e.body[i:i+70]}")
        print()

    print(json.dumps({k: v.model_dump() for k, v in emails.items()}, indent=2))


if __name__ == "__main__":
    main()
