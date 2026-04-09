"""Complex test schemas for model compatibility tests.

Exercises nested models, enums, lists, Optional fields, and defaults --
the kind of schema complexity that breaks across output strategies.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Experience(BaseModel, frozen=True):
    company: str
    role: str
    years: int
    current: bool = False


class ContactMethod(str, Enum):
    email = "email"
    phone = "phone"
    linkedin = "linkedin"


class LeadProfile(BaseModel, frozen=True):
    name: str
    title: str
    company: str
    experience: list[Experience]
    preferred_contact: ContactMethod
    tags: list[str] = []
    notes: str | None = None
