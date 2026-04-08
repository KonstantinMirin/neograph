"""Lead Outreach Email Sequence -- neograph mini-project.

Produces a 4-email outreach sequence for a target lead. Each email goes
through ensemble drafting (3 models), LLM-as-judge evaluation, and
iterative revision until quality threshold is met.

neograph features demonstrated:
  - Each: fan-out over email briefs (4 emails processed in parallel)
  - Oracle models=: ensemble drafting (3 model tiers) with merge_prompt
  - Loop: LLM revision cycle until evaluation score >= 0.8
  - Sub-constructs: per-email pipeline isolated from parent state
  - Inline prompts loaded from markdown files

The only mocked data is the LinkedIn profile and the value proposition.
All drafting, evaluation, and revision is real LLM work.

Run (requires OPENROUTER_API_KEY):
    OPENROUTER_API_KEY=sk-... python examples/lead-outreach/pipeline.py

Run with fake LLM (no API key, for testing wiring):
    python examples/lead-outreach/pipeline.py --fake
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

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
from neograph.factory import register_scripted
from neograph.modifiers import Loop, Oracle

from schemas import (
    BriefSet,
    EmailBrief,
    EmailDraft,
    EmailResult,
    KeyIdeas,
    LeadProfile,
    OutreachSequence,
)


# =============================================================================
# Data loading (the only mocked part -- LinkedIn profile + value prop)
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

USE_FAKE = "--fake" in sys.argv

MODELS = {
    "reason": "anthropic/claude-sonnet-4",
    "fast": "google/gemini-2.0-flash-001",
    "creative": "openai/gpt-4o",
}


def _real_llm_factory(tier: str, *, node_name: str = "", llm_config: dict | None = None):
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or use --fake flag")
        sys.exit(1)

    return ChatOpenAI(
        model=MODELS.get(tier, MODELS["fast"]),
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=(llm_config or {}).get("temperature", 0.7),
        max_tokens=(llm_config or {}).get("max_tokens", 2000),
    )


def _fake_llm_factory(tier: str, *, node_name: str = "", llm_config: dict | None = None):
    """Returns a fake LLM that produces plausible EmailDraft outputs."""

    class _Fake:
        def __init__(self):
            self._model = None
            self._call_count = 0

        def with_structured_output(self, model):
            clone = _Fake()
            clone._model = model
            clone._call_count = self._call_count
            return clone

        def invoke(self, messages, **kwargs):
            self._call_count += 1
            if self._model is None:
                return None

            # For merge (pick_best): input is a list of variants, return the first
            content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

            if self._model is EmailDraft:
                # Drafting or refining -- produce a reasonable draft
                return EmailDraft(
                    subject="re: pipeline reliability",
                    body=(
                        "Sarah, your post about hitting 99.99% uptime on the payments "
                        "API caught my attention. That level of reliability on real-time "
                        "payment rails is genuinely rare. Most teams I talk to struggle "
                        "to maintain three nines once they cross 10k TPS. Having shipped "
                        "payments infra at Stripe, you probably have strong opinions on "
                        "what makes observability actually useful vs dashboard theater. "
                        "When something does go wrong at 3 AM, what is the first signal "
                        "your team trusts? Not selling anything here, genuinely curious."
                    ),
                    email_number=1,
                    send_day=0,
                    score=0.85,
                    eval_feedback="solid draft, good personalization",
                    iteration=1,
                )
            return self._model()

    return _Fake()


# Inline prompts loaded from files. neograph's built-in ${var} substitution
# resolves variables from the Pydantic input model. Structured output is
# automatic from the outputs= type — no manual JSON schema needed.
PROMPT_DRAFT = _load_prompt("draft")
PROMPT_PICK_BEST = _load_prompt("pick_best")
PROMPT_REFINE = _load_prompt("refine")

configure_llm(
    llm_factory=_fake_llm_factory if USE_FAKE else _real_llm_factory,
    # Prompt compiler unused — all prompts are inline (contain spaces / ${}).
    # neograph detects them as inline and handles substitution automatically.
    prompt_compiler=lambda template, data: [{"role": "user", "content": template}],
)


# =============================================================================
# Node: plan_sequence (scripted -- structures the input data into briefs)
# =============================================================================


@node(outputs=BriefSet)
def plan_sequence() -> BriefSet:
    """Generate email sequence plan from lead profile and key ideas."""
    lead = _load_lead()
    ideas = _load_ideas()

    # Pre-format context strings for prompt rendering
    exp_str = "; ".join(
        f"{e.title} at {e.company} ({e.duration})"
        for e in lead.experience
    )
    posts_str = "\n".join(f"- {p}" for p in lead.recent_posts)
    skills_str = ", ".join(lead.skills)
    angles_str = "\n".join(f"- {a}" for a in ideas.angles)
    relevance_str = "\n".join(f"- {r}" for r in ideas.relevant_to_lead)

    common = dict(
        lead_name=f"{lead.first_name} {lead.last_name}",
        lead_headline=lead.headline,
        lead_company=lead.company,
        lead_company_description=lead.company_description,
        lead_location=lead.location,
        lead_experience=exp_str,
        lead_recent_posts=posts_str,
        lead_skills=skills_str,
        product=ideas.product,
        positioning=ideas.positioning,
        seller_angles=angles_str,
        relevance=relevance_str,
    )

    briefs = [
        EmailBrief(
            email_number=1,
            send_day=0,
            intent="cold_open",
            angle=ideas.relevant_to_lead[2],
            constraints=(
                "Reference their recent post about uptime. Ask a genuine "
                "question. No pitch whatsoever. Pure curiosity and relevance."
            ),
            **common,
        ),
        EmailBrief(
            email_number=2,
            send_day=3,
            intent="value_drop",
            angle=ideas.relevant_to_lead[1],
            constraints=(
                "Share a relevant insight connecting their Rust hiring to "
                "type safety in AI tooling. Do not sell. Provide genuine value."
            ),
            **common,
        ),
        EmailBrief(
            email_number=3,
            send_day=7,
            intent="soft_ask",
            angle=ideas.angles[0],
            constraints=(
                "Reference continuity from emails 1-2. Make one specific "
                "ask: 15 min demo. Make it trivially easy to say yes."
            ),
            **common,
        ),
        EmailBrief(
            email_number=4,
            send_day=14,
            intent="breakup",
            angle=ideas.angles[1],
            constraints=(
                "Final email. Be direct. Include one piece of social proof. "
                "Leave the door open gracefully. OK to use humor."
            ),
            **common,
        ),
    ]

    return BriefSet(briefs=briefs)


# =============================================================================
# Per-email sub-construct: draft (Oracle x3) + refine (Loop)
# =============================================================================

# draft-email: LLM generates a draft. Oracle runs 3 model tiers in parallel.
# merge_prompt="pick_best" uses LLM judge to select the best variant.

_produce_email = Construct(
    "produce-email",
    input=EmailBrief,
    output=EmailDraft,
    nodes=[
        Node(
            name="draft-email",
            mode="think",
            inputs=EmailBrief,
            outputs=EmailDraft,
            model="reason",
            prompt=PROMPT_DRAFT,
            llm_config={"temperature": 0.8},
        ) | Oracle(
            models=["reason", "fast", "creative"],
            merge_prompt=PROMPT_PICK_BEST,
        ),
        Node(
            name="refine",
            mode="think",
            inputs=EmailDraft,
            outputs=EmailDraft,
            model="reason",
            prompt=PROMPT_REFINE,
            llm_config={"temperature": 0.4},
        ) | Loop(
            when=lambda d: d is None or d.score < 0.8,
            max_iterations=3,
            on_exhaust="last",
        ),
    ],
)

produce_email = _produce_email | Each(
    over="plan_sequence.briefs",
    key="intent",
)


# =============================================================================
# Assemble: collect Each results into final output
# =============================================================================


@node(outputs=OutreachSequence)
def assemble(produce_email: dict[str, EmailDraft]) -> OutreachSequence:
    """Collect all refined emails into the final outreach sequence."""
    lead = _load_lead()
    intent_order = {"cold_open": 0, "value_drop": 1, "soft_ask": 2, "breakup": 3}

    emails = []
    for _key, draft in sorted(produce_email.items(), key=lambda kv: intent_order.get(kv[0], 99)):
        emails.append(EmailResult(
            email_number=draft.email_number,
            send_day=draft.send_day,
            subject=draft.subject,
            body=draft.body,
            evaluation={
                "overall": draft.score,
                "feedback": draft.eval_feedback,
            },
            iterations=draft.iteration,
        ))

    return OutreachSequence(
        lead_name=f"{lead.first_name} {lead.last_name}",
        company=lead.company,
        emails=emails,
    )


# =============================================================================
# Pipeline assembly
# =============================================================================

pipeline = construct_from_functions(
    "lead-outreach",
    [plan_sequence, produce_email, assemble],
)


# =============================================================================
# Runner
# =============================================================================

def main():
    print("=" * 70)
    print("  Lead Outreach Email Sequence Generator")
    print("=" * 70)

    lead = _load_lead()
    ideas = _load_ideas()
    print(f"\nLead: {lead.first_name} {lead.last_name}, {lead.headline}")
    print(f"Product: {ideas.product} -- {ideas.positioning}")
    print(f"Mode: {'FAKE (no API calls)' if USE_FAKE else 'LIVE (OpenRouter)'}")
    if not USE_FAKE:
        print(f"Models: {list(MODELS.values())}")

    print("\n--- Compiling pipeline ---")
    graph = compile(pipeline)
    print("Compiled.")

    print("\n--- Running pipeline ---")
    result = run(graph, input={
        "node_id": f"outreach-{lead.first_name.lower()}-{lead.company.lower()}",
    })

    # Display results
    sequence = result["assemble"]
    print(f"\n{'=' * 70}")
    print(f"  Outreach Sequence for {sequence.lead_name} at {sequence.company}")
    print(f"  {len(sequence.emails)} emails generated")
    print(f"{'=' * 70}")

    intent_labels = {0: "Cold Open", 3: "Value Drop", 7: "Soft Ask", 14: "Breakup"}

    for email in sequence.emails:
        label = intent_labels.get(email.send_day, f"Day {email.send_day}")
        print(f"\n--- Email {email.email_number}: {label} (Day {email.send_day}) ---")
        print(f"Subject: {email.subject}")
        print(f"Score: {email.evaluation.get('overall', 'n/a')}")
        print(f"Iterations: {email.iterations}")
        if email.evaluation.get("feedback"):
            print(f"Feedback: {email.evaluation['feedback']}")
        print()
        # Word-wrap body at ~70 chars
        words = email.body.split()
        lines, current, length = [], [], 0
        for w in words:
            if length + len(w) + 1 > 70 and current:
                lines.append(" ".join(current))
                current, length = [w], len(w)
            else:
                current.append(w)
                length += len(w) + 1
        if current:
            lines.append(" ".join(current))
        for line in lines:
            print(f"  {line}")

    # JSON output
    print(f"\n{'=' * 70}")
    print("  JSON Output")
    print(f"{'=' * 70}")
    print(json.dumps(sequence.model_dump(), indent=2))


if __name__ == "__main__":
    main()
