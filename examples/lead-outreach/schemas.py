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
    """The one type that flows through everything.

    Carries the brief context (constant) plus the evolving draft and
    feedback. draft node fills subject/body, feedback node fills
    score/feedback. Loop reads score.
    """
    # Brief context
    intent: str
    email_number: int = 0
    send_day: int = 0
    angle: str = ""
    constraints: str = ""
    lead: LeadProfile | None = None
    ideas: KeyIdeas | None = None
    # Draft
    subject: str = ""
    body: str = ""
    # Evaluation
    score: float = 0.0
    feedback: str = ""
    iteration: int = 0


class EmailStates(BaseModel, frozen=True):
    items: list[EmailState]
