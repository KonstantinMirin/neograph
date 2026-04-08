"""Schemas for the lead outreach email sequence pipeline.

All types are frozen Pydantic models, which is idiomatic for neograph:
immutable state snapshots flowing between nodes.
"""

from __future__ import annotations

from pydantic import BaseModel


# -- Input data ---------------------------------------------------------------

class Experience(BaseModel, frozen=True):
    title: str
    company: str
    duration: str


class LeadProfile(BaseModel, frozen=True):
    first_name: str
    last_name: str
    headline: str
    company: str
    company_description: str
    location: str
    experience: list[Experience]
    recent_posts: list[str]
    skills: list[str]


class KeyIdeas(BaseModel, frozen=True):
    product: str
    positioning: str
    angles: list[str]
    relevant_to_lead: list[str]


# -- Intermediate types -------------------------------------------------------

class EmailBrief(BaseModel, frozen=True):
    """Planning output: what each email in the sequence should accomplish."""
    email_number: int
    send_day: int
    intent: str        # cold_open | value_drop | soft_ask | breakup
    angle: str         # the specific hook for this email
    constraints: str   # style/content requirements


class BriefSet(BaseModel, frozen=True):
    """Collection of email briefs -- the sequence plan."""
    lead: LeadProfile
    ideas: KeyIdeas
    briefs: list[EmailBrief]


class EmailDraft(BaseModel, frozen=True):
    """A single email draft flowing through the quality pipeline.

    Carries its own quality state so the Loop modifier's condition
    can inspect `draft.score` directly.
    """
    subject: str
    body: str
    email_number: int
    send_day: int
    score: float = 0.0
    violations: list[str] = []
    eval_feedback: str = ""
    iteration: int = 0


# -- Output -------------------------------------------------------------------

class EmailResult(BaseModel, frozen=True):
    """Final output per email after all quality passes."""
    email_number: int
    send_day: int
    subject: str
    body: str
    quality_gate: dict
    evaluation: dict
    iterations: int


class OutreachSequence(BaseModel, frozen=True):
    """The complete email sequence, ready to load into a sending tool."""
    lead_name: str
    company: str
    emails: list[EmailResult]
