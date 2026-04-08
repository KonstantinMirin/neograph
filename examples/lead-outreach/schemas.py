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


class EmailBrief(BaseModel, frozen=True):
    """Self-contained brief for one email. Carries all context the LLM needs
    so the sub-construct input is complete without DI."""
    email_number: int
    send_day: int
    intent: str
    angle: str
    constraints: str
    # Lead context (pre-formatted for prompt rendering)
    lead_name: str
    lead_headline: str
    lead_company: str
    lead_company_description: str
    lead_location: str
    lead_experience: str
    lead_recent_posts: str
    lead_skills: str
    # Seller context
    product: str
    positioning: str
    seller_angles: str
    relevance: str


class BriefSet(BaseModel, frozen=True):
    briefs: list[EmailBrief]


class EmailDraft(BaseModel, frozen=True):
    """Email draft flowing through the pipeline. Carries quality state
    so the Loop condition can read draft.score."""
    subject: str
    body: str
    email_number: int = 0
    send_day: int = 0
    score: float = 0.0
    violations: list[str] = []
    eval_feedback: str = ""
    iteration: int = 0


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
