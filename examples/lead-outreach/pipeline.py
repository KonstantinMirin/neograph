"""Lead Outreach Email Sequence -- neograph mini-project.

Produces a 4-email outreach sequence for a target lead. Each email goes
through ensemble drafting (3 models), LLM-as-judge evaluation, and
iterative revision until quality threshold is met.

Pipeline per email:
  draft (Oracle x3, picks best) → feedback (LLM-as-prospect scores it)
  └── Loop until score >= 0.8 ──┘

neograph features:
  - Each: 4 emails processed in parallel
  - Oracle models=: 3 model tiers draft independently, judge picks best
  - Loop: feedback drives revision -- draft gets feedback, rewrites, repeat
  - Sub-constructs: per-email pipeline isolated from parent state

Run:
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
from neograph.modifiers import Loop, Oracle

from schemas import (
    EmailResult,
    EmailState,
    EmailStateSet,
    KeyIdeas,
    LeadProfile,
    OutreachSequence,
)


# =============================================================================
# Data loading
# =============================================================================

def _load_lead() -> LeadProfile:
    with open(_HERE / "data" / "lead_profile.json") as f:
        return LeadProfile(**json.load(f))


def _load_ideas() -> KeyIdeas:
    with open(_HERE / "data" / "key_ideas.yaml") as f:
        return KeyIdeas(**yaml.safe_load(f))


def _load_prompt(name: str) -> str:
    return (_HERE / "prompts" / f"{name}.md").read_text()


# =============================================================================
# LLM configuration
# =============================================================================

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


PROMPT_DRAFT = _load_prompt("draft")
PROMPT_PICK_BEST = _load_prompt("pick_best")
PROMPT_FEEDBACK = _load_prompt("feedback")

configure_llm(
    llm_factory=_llm_factory,
    prompt_compiler=lambda template, data: [{"role": "user", "content": template}],
)


# =============================================================================
# plan_sequence: structure input data into email states
# =============================================================================

@node(outputs=EmailStateSet)
def plan_sequence() -> EmailStateSet:
    lead = _load_lead()
    ideas = _load_ideas()

    specs = [
        (1, 0, "cold_open", ideas.relevant_to_lead[2],
         "Reference their recent post about uptime. Ask a genuine "
         "question. No pitch. Pure curiosity and relevance."),
        (2, 3, "value_drop", ideas.relevant_to_lead[1],
         "Share a relevant insight connecting their Rust hiring to "
         "type safety in AI tooling. Do not sell. Provide genuine value."),
        (3, 7, "soft_ask", ideas.angles[0],
         "Reference continuity from emails 1-2. Make one specific "
         "ask: 15 min demo. Make it trivially easy to say yes."),
        (4, 14, "breakup", ideas.angles[1],
         "Final email. Be direct. Include one piece of social proof. "
         "Leave the door open gracefully. OK to use humor."),
    ]

    return EmailStateSet(items=[
        EmailState(
            email_number=num, send_day=day, intent=intent,
            angle=angle, constraints=constraints,
            lead=lead, ideas=ideas,
        )
        for num, day, intent, angle, constraints in specs
    ])


# =============================================================================
# Per-email pipeline: draft (Oracle x3) → feedback, looped
# =============================================================================

# Inner: draft → feedback, looped until score >= 0.8
_draft_feedback_loop = Construct(
    "draft-feedback",
    input=EmailState,
    output=EmailState,
    nodes=[
        Node(
            name="draft",
            mode="think",
            inputs=EmailState,
            outputs=EmailState,
            model="reason",
            prompt=PROMPT_DRAFT,
            llm_config={"temperature": 0.8},
        ) | Oracle(
            models=["reason", "fast", "creative"],
            merge_prompt=PROMPT_PICK_BEST,
        ),
        Node(
            name="feedback",
            mode="think",
            inputs=EmailState,
            outputs=EmailState,
            model="reason",
            prompt=PROMPT_FEEDBACK,
            llm_config={"temperature": 0.3},
        ),
    ],
) | Loop(
    when=lambda s: s is None or s.score < 0.8,
    max_iterations=3,
    on_exhaust="last",
)

# Outer: one instance per email brief, fanned out by Each
_produce_email = Construct(
    "produce-email",
    input=EmailState,
    output=EmailState,
    nodes=[_draft_feedback_loop],
)

produce_email = _produce_email | Each(
    over="plan_sequence.items",
    key="intent",
)


# =============================================================================
# Assemble
# =============================================================================

@node(outputs=OutreachSequence)
def assemble(produce_email: dict[str, EmailState]) -> OutreachSequence:
    lead = _load_lead()
    intent_order = {"cold_open": 0, "value_drop": 1, "soft_ask": 2, "breakup": 3}

    emails = []
    for _key, state in sorted(produce_email.items(), key=lambda kv: intent_order.get(kv[0], 99)):
        emails.append(EmailResult(
            email_number=state.email_number,
            send_day=state.send_day,
            subject=state.subject,
            body=state.body,
            evaluation={"overall": state.score, "feedback": state.feedback},
            iterations=state.iteration,
        ))

    return OutreachSequence(
        lead_name=f"{lead.first_name} {lead.last_name}",
        company=lead.company,
        emails=emails,
    )


# =============================================================================
# Pipeline
# =============================================================================

pipeline = construct_from_functions(
    "lead-outreach",
    [plan_sequence, produce_email, assemble],
)


# =============================================================================
# Runner
# =============================================================================

def main():
    lead = _load_lead()
    ideas = _load_ideas()
    print(f"Lead: {lead.first_name} {lead.last_name}, {lead.headline}")
    print(f"Product: {ideas.product}")
    print(f"Models: {list(MODELS.values())}")

    graph = compile(pipeline)
    result = run(graph, input={
        "node_id": f"outreach-{lead.first_name.lower()}-{lead.company.lower()}",
    })

    sequence = result["assemble"]
    print(f"\n{len(sequence.emails)} emails for {sequence.lead_name} at {sequence.company}:\n")

    labels = {0: "Cold Open", 3: "Value Drop", 7: "Soft Ask", 14: "Breakup"}
    for email in sequence.emails:
        print(f"--- Email {email.email_number}: {labels.get(email.send_day, f'Day {email.send_day}')} ---")
        print(f"Subject: {email.subject}")
        print(f"Score: {email.evaluation['overall']}  Iterations: {email.iterations}")
        print(f"Feedback: {email.evaluation['feedback']}")
        print()
        words = email.body.split()
        line, length = [], 0
        for w in words:
            if length + len(w) + 1 > 70 and line:
                print(f"  {' '.join(line)}")
                line, length = [w], len(w)
            else:
                line.append(w)
                length += len(w) + 1
        if line:
            print(f"  {' '.join(line)}")
        print()

    print(json.dumps(sequence.model_dump(), indent=2))


if __name__ == "__main__":
    main()
