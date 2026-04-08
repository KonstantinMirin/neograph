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
    """One email in the sequence. Carries lead + ideas so the sub-construct
    input is self-contained — prompts reference ${lead.first_name} etc."""
    email_number: int
    send_day: int
    intent: str
    angle: str
    constraints: str
    lead: LeadProfile
    ideas: KeyIdeas


class BriefSet(BaseModel, frozen=True):
    briefs: list[EmailBrief]


class EmailDraft(BaseModel, frozen=True):
    """Email draft flowing through the pipeline. score drives the Loop exit."""
    subject: str
    body: str
    email_number: int = 0
    send_day: int = 0
    score: float = 0.0
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
