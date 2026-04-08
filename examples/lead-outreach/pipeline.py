"""Lead Outreach Email Sequence -- neograph mini-project.

Produces a 4-email outreach sequence for a target lead. Each email goes
through ensemble drafting, deterministic quality checks, simulated LLM
evaluation, and iterative revision.

neograph features demonstrated:
  - Each: fan-out over email briefs (4 emails processed in parallel)
  - Oracle models=: ensemble drafting (3 model tiers) with merge_fn
  - Loop: revision cycle until quality score >= 0.8
  - Scripted nodes: quality gate (pure Python, no LLM)
  - Sub-constructs: per-email pipeline isolated from parent state

All nodes are scripted (no API keys needed). In production, swap the
drafting/evaluation nodes for LLM-powered equivalents by adding
prompt= and model= to the @node decorators.

Run:
    python examples/lead-outreach/pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

# Allow sibling imports when run as a script
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from neograph import (
    Construct,
    Each,
    Node,
    compile,
    construct_from_functions,
    node,
    run,
)
from neograph.factory import register_scripted
from neograph.modifiers import Loop, Oracle

from quality_gate import FORBIDDEN_PHRASES, run_quality_gate
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
# Data loading
# =============================================================================

def load_lead() -> LeadProfile:
    with open(_HERE / "data" / "lead_profile.json") as f:
        return LeadProfile(**json.load(f))


def load_ideas() -> KeyIdeas:
    with open(_HERE / "data" / "key_ideas.yaml") as f:
        return KeyIdeas(**yaml.safe_load(f))


# =============================================================================
# Simulated email content (what an LLM would generate in production)
# =============================================================================

# Per-model draft variants -- simulates different model "personalities"
_DRAFT_VARIANTS: dict[str, dict[int, dict[str, str]]] = {
    "reason": {
        1: {
            "subject": "four nines on payments",
            "body": (
                "Sarah, your post about hitting 99.99% uptime on the payments "
                "API caught my attention. That level of reliability on real-time "
                "payment rails is genuinely rare, especially at Finova's scale. "
                "Most teams I talk to struggle to maintain three nines once they "
                "cross 10k TPS. Having shipped payments infra at Stripe, you "
                "probably have strong opinions on what makes observability "
                "actually useful vs. dashboard theater. Curious: when something "
                "does go wrong at 3 AM, what is the first signal your team "
                "trusts? Is it structured logs, distributed traces, or "
                "something else entirely? Not selling anything here, just "
                "genuinely interested in how teams at your level think about "
                "this differently."
            ),
        },
        2: {
            "subject": "rust and reliable pipelines",
            "body": (
                "Sarah, saw you are hiring backend engineers with Rust "
                "experience at Finova. That is an interesting bet for a "
                "payments company, one that suggests you care about "
                "correctness guarantees at the language level, not just "
                "at the test level. I have been thinking about a similar "
                "problem in the AI tooling space: how do you get type safety "
                "and compile-time validation for LLM pipelines the way Rust "
                "gives you memory safety? Most agent frameworks punt on this "
                "entirely. The pipeline breaks at 3 AM, and the error message "
                "is 'NoneType has no attribute output'. Your Stripe background "
                "probably makes that kind of failure mode feel inexcusable. "
                "Would a 2-minute demo of typed pipeline compilation be "
                "interesting, or is this not on your radar right now?"
            ),
        },
        3: {
            "subject": "15 min on pipeline reliability",
            "body": (
                "Sarah, I have sent a couple of notes and I realize I might be "
                "solving a problem you have already solved. Finova clearly has "
                "engineering discipline, your uptime numbers prove that. But "
                "from the teams I have talked to at similar scale, the gap is "
                "usually not in the production services themselves but in the "
                "internal tooling and automation pipelines that support them. "
                "If you have 15 minutes this week, I would like to show you "
                "how neograph handles typed pipeline compilation. If it is not "
                "relevant, no worries, I will not follow up again. Either way, "
                "congrats on the four nines, that is real engineering. "
                "Does Thursday or Friday work?"
            ),
        },
        4: {
            "subject": "last note from me",
            "body": (
                "Sarah, this is my last email. I know VP Engineering inboxes "
                "are brutal, so I will keep it short. Three teams in payments "
                "infrastructure (similar scale to Finova) are using neograph "
                "to catch pipeline type errors at build time instead of "
                "at 3 AM. If that is a problem you recognize, the offer "
                "stands: 15 minutes, I will show you how it works, and you "
                "can tell me if I am wrong about the pain point. If not, "
                "genuinely wish you and the Finova team well. The distributed "
                "systems community is small and your recent posts on API "
                "reliability have been solid reads."
            ),
        },
    },
    "fast": {
        1: {
            "subject": "re: 99.99% uptime",
            "body": (
                "Sarah, four nines on a payments API is no joke. At Finova's "
                "growth stage, maintaining that while scaling the team takes "
                "real discipline. I have been building developer tools for "
                "teams in your position, where reliability is table stakes, "
                "not a nice-to-have. One pattern I keep seeing is that the "
                "production services are solid, but the internal pipelines "
                "(data, ML, automation) are held together with duct tape. "
                "The engineers who built Stripe-grade infra are stuck debugging "
                "Python scripts with no type safety. Having come from Stripe "
                "yourself, does that resonate, or has Finova managed to bring "
                "the same rigor to internal tooling? Not pitching, genuinely "
                "curious about how you think about this."
            ),
        },
        2: {
            "subject": "what rust teams want from AI tools",
            "body": (
                "Sarah, teams hiring Rust engineers tend to have a specific "
                "engineering philosophy: catch errors at compile time, not at "
                "runtime. I have been building something that applies that "
                "same principle to LLM pipelines. When you wire together AI "
                "agents today, type mismatches surface at runtime, usually "
                "at the worst possible moment. What if the pipeline compiler "
                "caught those before you deployed, the same way rustc catches "
                "lifetime errors before you ship? Given Finova is investing "
                "in Rust talent, I would guess this kind of correctness "
                "guarantee matters to your team. Worth a 5-minute look, or "
                "completely off base?"
            ),
        },
        3: {
            "subject": "quick ask",
            "body": (
                "Sarah, I have sent two notes about type-safe pipeline "
                "tooling and I want to be direct about why. Three payments "
                "companies at Finova's stage told me the same thing: their "
                "production APIs are rock solid, but their automation and "
                "data pipelines break in ways that are embarrassing for a "
                "team of their caliber. Neograph fixes that with compile-time "
                "validation. I am not asking for a long meeting, just 15 "
                "minutes to show you how it works. If it is not relevant, "
                "you will know in the first 3 minutes and I will not waste "
                "the other 12. Does this week work? If not, just say when "
                "or tell me to stop."
            ),
        },
        4: {
            "subject": "closing the loop",
            "body": (
                "Sarah, final note. I have been following Finova's engineering "
                "posts and it is clear you run a tight ship. If pipeline "
                "reliability is not a pain point for your team right now, I "
                "respect that. But if the day comes when someone asks why "
                "the internal data pipeline broke production, neograph is "
                "the tool that prevents that conversation. The offer stands "
                "indefinitely: 15 minutes, I will show you typed pipeline "
                "compilation, you tell me if it matters. Either way, "
                "appreciate you reading these. The distributed systems "
                "talks you have been giving are genuinely useful to "
                "the community."
            ),
        },
    },
    "creative": {
        1: {
            "subject": "the other four nines problem",
            "body": (
                "Sarah, you hit 99.99% uptime on payments. That puts "
                "Finova in rare company. But here is a question nobody "
                "is asking: what is the uptime on the internal tools "
                "your engineers use to build and maintain that API? In my "
                "experience, teams that achieve four nines on their product "
                "often run their internal pipelines at two nines on a good "
                "day. The data pipelines, the LLM automation, the test "
                "infrastructure, the things that are not customer-facing "
                "but absolutely mission-critical. Having built payments "
                "at Stripe, you know the difference between systems that "
                "are reliable by accident and systems that are reliable by "
                "design. Curious which category your internal tooling falls "
                "into? No agenda, just a question that has been on my mind."
            ),
        },
        2: {
            "subject": "a rust-shaped idea for AI pipelines",
            "body": (
                "Sarah, your job posting says Rust experience preferred. "
                "That tells me something about Finova's engineering values: "
                "you want correctness enforced by the compiler, not by hope "
                "and vigorous testing. I have been building something with "
                "the same philosophy for a different domain. Imagine if your "
                "LLM pipelines had the same compile-time guarantees as your "
                "Rust code: type mismatches caught before deployment, state "
                "transitions validated at build time, and observable execution "
                "with the same granularity as distributed tracing. That is "
                "neograph. Not asking for time today. Just wanted to plant "
                "the idea. If it resonates after your morning coffee, I am "
                "easy to find. Does this match a problem you are thinking "
                "about, or am I off base?"
            ),
        },
        3: {
            "subject": "one specific thing to show you",
            "body": (
                "Sarah, I have been going back and forth about whether to "
                "send this third email, so I will be honest about what "
                "I think would be valuable for you specifically. You run "
                "engineering at a payments company that values type safety "
                "(Rust hiring) and operational reliability (four nines). "
                "Neograph applies those same principles to AI and automation "
                "pipelines. I have one demo that takes 3 minutes: a pipeline "
                "that catches a type mismatch at compile time that would "
                "otherwise crash at 3 AM. If you have spent any time debugging "
                "Python-based automation that broke in production, you will "
                "immediately see why this matters. Would a 15-minute slot "
                "this week work, or should I try next week?"
            ),
        },
        4: {
            "subject": "appreciate your time",
            "body": (
                "Sarah, this is my last note. I wanted to share something "
                "useful regardless of whether we talk. Three patterns I have "
                "seen from payments teams running reliable LLM pipelines: "
                "typed state at every node boundary, append-only execution "
                "logs for debuggability, and compile-time graph validation. "
                "Finova might already do all three. If not, neograph handles "
                "them declaratively. No meeting required to try it, it is "
                "open source. And if you ever want to swap notes on building "
                "reliable systems at scale, I am always up for that "
                "conversation. Best of luck with the Rust hiring push, "
                "hope you find great people."
            ),
        },
    },
}


# =============================================================================
# Node implementations
# =============================================================================


@node(outputs=BriefSet)
def plan_sequence() -> BriefSet:
    """Generate email sequence plan from lead profile and key ideas."""
    lead = load_lead()
    ideas = load_ideas()

    briefs = [
        EmailBrief(
            email_number=1,
            send_day=0,
            intent="cold_open",
            angle=ideas.relevant_to_lead[2],  # 99.99% uptime reference
            constraints=(
                "Reference their recent post. Ask a question. "
                "No pitch. Under 50 char subject."
            ),
        ),
        EmailBrief(
            email_number=2,
            send_day=3,
            intent="value_drop",
            angle=ideas.relevant_to_lead[1],  # Stripe background / type safety
            constraints=(
                "Share a relevant insight connecting their hiring to "
                "the value prop. Still not selling."
            ),
        ),
        EmailBrief(
            email_number=3,
            send_day=7,
            intent="soft_ask",
            angle=ideas.angles[0],  # 70% wiring not logic
            constraints=(
                "Reference emails 1-2 continuity. One specific ask: "
                "15 min demo. Make it easy to say yes."
            ),
        ),
        EmailBrief(
            email_number=4,
            send_day=14,
            intent="breakup",
            angle=ideas.angles[1],  # Type mismatches at 3 AM
            constraints=(
                "Last email. Humor or directness. Social proof. "
                "Leave door open."
            ),
        ),
    ]

    return BriefSet(lead=lead, ideas=ideas, briefs=briefs)


# -- Oracle ensemble: 3 models draft independently, merge picks best ----------

def _draft_impl(input_data: Any, config: dict) -> EmailDraft:
    """Scripted draft generator. Reads _oracle_model to vary output."""
    model = config.get("configurable", {}).get("_oracle_model", "reason")
    brief = input_data  # EmailBrief from sub-construct input

    variants = _DRAFT_VARIANTS.get(model, _DRAFT_VARIANTS["reason"])
    content = variants.get(brief.email_number, variants[1])

    return EmailDraft(
        subject=content["subject"],
        body=content["body"],
        email_number=brief.email_number,
        send_day=brief.send_day,
    )


def _pick_best_draft(variants: list, config: dict) -> EmailDraft:
    """Merge function: pick the draft with the most personalization signals."""
    if not variants:
        return variants[0] if variants else None

    def personalization_score(draft: EmailDraft) -> int:
        lead = load_lead()
        body_lower = draft.body.lower()
        signals = [
            *lead.recent_posts,
            *lead.skills,
            *[exp.company for exp in lead.experience],
        ]
        return sum(
            1 for s in signals
            if len(s) > 3 and s.lower() in body_lower
        )

    # Pick variant with most personalization signals
    best = max(variants, key=personalization_score)
    return best


register_scripted("draft_email_impl", _draft_impl)
register_scripted("pick_best_draft", _pick_best_draft)


# -- Quality gate + evaluation + revision (self-loop) -------------------------

_EVAL_WEIGHTS = {
    "would_open": 0.2,
    "would_read": 0.15,
    "would_reply": 0.3,
    "feels_personal": 0.25,
    "not_annoying": 0.1,
}


def _simulate_eval_scores(draft: EmailDraft, lead: LeadProfile) -> dict:
    """Simulate LLM evaluation ensemble scoring."""
    body_lower = draft.body.lower()

    # would_open: short subject + curiosity-provoking
    would_open = 0.9 if len(draft.subject) <= 40 else 0.7
    if "?" in draft.subject or "re:" in draft.subject.lower():
        would_open = min(would_open + 0.1, 1.0)

    # would_read: first sentence hook
    first_sentence = draft.body.split(".")[0] if "." in draft.body else draft.body[:100]
    would_read = 0.8 if lead.first_name.lower() in first_sentence.lower() else 0.6

    # would_reply: has a question + relevant ask
    questions = draft.body.count("?")
    would_reply = min(0.5 + questions * 0.15, 0.95)

    # feels_personal: references specific details
    specifics = [*lead.recent_posts, *lead.skills, *[e.company for e in lead.experience]]
    ref_count = sum(1 for s in specifics if len(s) > 3 and s.lower() in body_lower)
    feels_personal = min(0.4 + ref_count * 0.15, 1.0)

    # not_annoying: no forbidden phrases, not too long
    annoyance_count = sum(1 for p in FORBIDDEN_PHRASES if p in body_lower)
    word_count = len(draft.body.split())
    not_annoying = 0.95 - annoyance_count * 0.2 - max(0, (word_count - 120) * 0.005)
    not_annoying = max(not_annoying, 0.1)

    scores = {
        "would_open": round(would_open, 2),
        "would_read": round(would_read, 2),
        "would_reply": round(would_reply, 2),
        "feels_personal": round(feels_personal, 2),
        "not_annoying": round(not_annoying, 2),
    }

    overall = sum(scores[k] * _EVAL_WEIGHTS[k] for k in _EVAL_WEIGHTS)
    scores["overall"] = round(overall, 2)

    # Generate feedback for dimensions below 0.7
    feedback_parts = []
    for dim, val in scores.items():
        if dim != "overall" and val < 0.7:
            feedback_parts.append(f"{dim} is low ({val})")
    scores["feedback"] = "; ".join(feedback_parts) if feedback_parts else "solid"

    return scores


def _revise_draft(draft: EmailDraft, violations: list[str], lead: LeadProfile) -> EmailDraft:
    """Apply deterministic fixes for common violations."""
    subject = draft.subject
    body = draft.body

    for v in violations:
        if "em-dash" in v.lower():
            body = body.replace("\u2014", ",")
            subject = subject.replace("\u2014", ",")
        if "exclamation" in v.lower():
            body = body.replace("!", ".")
            subject = subject.replace("!", ".")
        if "subject too long" in v.lower() and len(subject) > 50:
            subject = subject[:47] + "..."
        if "company name" in v.lower() and lead.company.lower() in subject.lower():
            subject = subject.lower().replace(lead.company.lower(), "").strip()
        if "no question" in v.lower() and "?" not in body:
            body = body.rstrip() + " What do you think?"

    return EmailDraft(
        subject=subject,
        body=body,
        email_number=draft.email_number,
        send_day=draft.send_day,
        score=draft.score,
        violations=[],
        eval_feedback=draft.eval_feedback,
        iteration=draft.iteration + 1,
    )


def _refine_impl(input_data: Any, config: dict) -> EmailDraft:
    """Quality gate + evaluation + revision. Self-loops until score >= 0.8.

    Each iteration:
      1. Run deterministic quality gate
      2. If violations, revise and return (next iteration re-checks)
      3. Run simulated evaluation scoring
      4. If score >= 0.8, return (loop exits)
      5. Otherwise, return with current score (loop decides)
    """
    draft = input_data
    lead = load_lead()

    # Step 1: Quality gate
    violations = run_quality_gate(draft, lead)
    if violations:
        revised = _revise_draft(draft, violations, lead)
        return revised.model_copy(update={
            "violations": violations,
            "score": 0.0,  # force loop to continue
        })

    # Step 2: Evaluation
    scores = _simulate_eval_scores(draft, lead)
    overall = scores["overall"]

    return draft.model_copy(update={
        "score": overall,
        "violations": [],
        "eval_feedback": scores.get("feedback", ""),
        "iteration": draft.iteration + 1,
    })


register_scripted("refine_email_impl", _refine_impl)


# -- Per-email sub-construct: draft (Oracle) + refine (Loop) ------------------

# Build the sub-construct that processes a single EmailBrief into EmailDraft.
# Oracle on draft-email generates 3 variants (one per model tier) and merges.
# Loop on refine iterates until quality score >= 0.8.

_produce_email = Construct(
    "produce-email",
    input=EmailBrief,
    output=EmailDraft,
    nodes=[
        Node.scripted(
            "draft-email",
            fn="draft_email_impl",
            inputs=EmailBrief,
            outputs=EmailDraft,
        ) | Oracle(
            models=["reason", "fast", "creative"],
            merge_fn="pick_best_draft",
        ),
        Node.scripted(
            "refine",
            fn="refine_email_impl",
            inputs=EmailDraft,
            outputs=EmailDraft,
        ) | Loop(
            when=lambda d: d is None or d.score < 0.8,
            max_iterations=3,
            on_exhaust="last",
        ),
    ],
)

# Apply Each: one sub-construct instance per email brief
produce_email = _produce_email | Each(
    over="plan_sequence.briefs",
    key="intent",
)


# -- Assemble: collect Each results into final output -------------------------

@node(outputs=OutreachSequence)
def assemble(produce_email: dict[str, EmailDraft]) -> OutreachSequence:
    """Collect all refined emails into the final outreach sequence."""
    lead = load_lead()

    intent_order = {"cold_open": 0, "value_drop": 1, "soft_ask": 2, "breakup": 3}
    emails = []
    for _key, draft in sorted(produce_email.items(), key=lambda kv: intent_order.get(kv[0], 99)):
        emails.append(EmailResult(
            email_number=draft.email_number,
            send_day=draft.send_day,
            subject=draft.subject,
            body=draft.body,
            quality_gate={"passed": len(draft.violations) == 0, "violations": draft.violations},
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

    lead = load_lead()
    ideas = load_ideas()
    print(f"\nLead: {lead.first_name} {lead.last_name}, {lead.headline}")
    print(f"Product: {ideas.product} -- {ideas.positioning}")

    print("\n--- Compiling pipeline ---")
    graph = compile(pipeline)
    print("Compiled.")

    print("\n--- Running pipeline ---")
    result = run(graph, input={"node_id": f"outreach-{lead.first_name.lower()}-{lead.company.lower()}"})

    # Display results
    sequence = result["assemble"]
    print(f"\n{'=' * 70}")
    print(f"  Outreach Sequence for {sequence.lead_name} at {sequence.company}")
    print(f"  {len(sequence.emails)} emails generated")
    print(f"{'=' * 70}")

    for email in sequence.emails:
        intent_labels = {
            0: "Cold Open",
            3: "Value Drop",
            7: "Soft Ask",
            14: "Breakup",
        }
        label = intent_labels.get(email.send_day, f"Day {email.send_day}")

        print(f"\n--- Email {email.email_number}: {label} (Day {email.send_day}) ---")
        print(f"Subject: {email.subject}")
        print(f"Score: {email.evaluation.get('overall', 'n/a')}")
        print(f"Iterations: {email.iterations}")
        if email.quality_gate.get("violations"):
            print(f"Violations: {email.quality_gate['violations']}")
        print()
        # Word-wrap the body at ~70 chars
        words = email.body.split()
        lines = []
        current = []
        length = 0
        for w in words:
            if length + len(w) + 1 > 70 and current:
                lines.append(" ".join(current))
                current = [w]
                length = len(w)
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
