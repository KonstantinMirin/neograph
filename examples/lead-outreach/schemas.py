"""Schemas for the lead outreach email sequence pipeline."""

from __future__ import annotations

from pydantic import BaseModel


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


class EmailState(BaseModel, frozen=True):
    """Full state for one email through the draft/feedback cycle.

    Carries the brief context (constant across iterations) plus the
    evolving draft and feedback. Both the draft and feedback LLM nodes
    output this same type — draft fills subject/body, feedback fills
    score/feedback. The Loop condition reads score.
    """
    # Context (constant, echoed through by the LLM)
    intent: str
    email_number: int = 0
    send_day: int = 0
    angle: str = ""
    constraints: str = ""
    lead: LeadProfile | None = None
    ideas: KeyIdeas | None = None
    # Draft (written by the draft node)
    subject: str = ""
    body: str = ""
    # Evaluation (written by the feedback node, fed back to draft)
    score: float = 0.0
    feedback: str = ""
    iteration: int = 0


class EmailStateSet(BaseModel, frozen=True):
    items: list[EmailState]


class EmailResult(BaseModel, frozen=True):
    email_number: int
    send_day: int
    subject: str
    body: str
    evaluation: dict
    iterations: int


class OutreachSequence(BaseModel, frozen=True):
    lead_name: str
    company: str
    emails: list[EmailResult]
