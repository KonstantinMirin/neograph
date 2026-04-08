"""Deterministic quality gate for outreach emails.

Pure Python checks -- no LLM needed. Returns a list of violations.
These checks enforce the hard rules that make cold emails not feel
like cold emails.
"""

from __future__ import annotations

from schemas import EmailDraft, LeadProfile


FORBIDDEN_PHRASES = [
    "i hope this email finds you",
    "i wanted to reach out",
    "i came across your profile",
    "synergy",
    "circle back",
    "low-hanging fruit",
    "move the needle",
    "at the end of the day",
    "just following up",
    "per my last email",
    "i'd love to pick your brain",
]


def run_quality_gate(draft: EmailDraft, lead: LeadProfile) -> list[str]:
    """Run all deterministic checks. Returns list of violation strings."""
    violations = []

    # First name must appear in body
    if lead.first_name.lower() not in draft.body.lower():
        violations.append(
            f"Missing first name: body must contain '{lead.first_name}'"
        )

    # No overly formal addressing
    formal_patterns = [
        f"Dear {lead.last_name}",
        f"Dear Ms. {lead.last_name}",
        f"Dear Mr. {lead.last_name}",
        f"{lead.last_name},",
    ]
    for pat in formal_patterns:
        if pat.lower() in draft.body.lower():
            violations.append(f"Too formal: found '{pat}' -- use first name only")
            break

    # No em-dash
    if "\u2014" in draft.body or "\u2014" in draft.subject:
        violations.append("Contains em-dash (\u2014) -- use comma or period instead")

    # No exclamation marks
    excl_count = draft.body.count("!") + draft.subject.count("!")
    if excl_count > 0:
        violations.append(
            f"Contains {excl_count} exclamation mark(s) -- remove all"
        )

    # Subject length
    if len(draft.subject) > 50:
        violations.append(
            f"Subject too long: {len(draft.subject)} chars (max 50)"
        )

    # Body word count
    word_count = len(draft.body.split())
    if word_count < 50:
        violations.append(f"Body too short: {word_count} words (min 50)")
    if word_count > 150:
        violations.append(f"Body too long: {word_count} words (max 150)")

    # No forbidden phrases
    body_lower = draft.body.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in body_lower:
            violations.append(f"Forbidden phrase: '{phrase}'")

    # No company name in subject (screams cold email)
    if lead.company.lower() in draft.subject.lower():
        violations.append(
            f"Company name '{lead.company}' in subject -- remove it"
        )

    # Must contain a question (engagement hook)
    if "?" not in draft.body:
        violations.append("No question mark in body -- add an engagement hook")

    # Personalization: must reference something specific from the lead
    specifics = [
        *lead.recent_posts,
        *lead.skills,
        *[exp.company for exp in lead.experience],
        lead.headline,
    ]
    has_specific = any(
        detail.lower() in body_lower
        for detail in specifics
        if len(detail) > 3  # skip very short matches
    )
    if not has_specific:
        violations.append(
            "No personalization: body doesn't reference any specific "
            "detail from the lead's profile, posts, or experience"
        )

    return violations
