"""Schemas for the lead research pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class Lead(BaseModel, frozen=True):
    name: str
    company: str
    website: str
    role: str
    industry: str
    company_size: str


class LeadBatch(BaseModel, frozen=True):
    items: list[Lead]


class CompanyProfile(BaseModel, frozen=True):
    company_name: str
    website_url: str
    tagline: str = ""
    products: list[str] = []
    recent_updates: list[str] = []
    tech_stack_hints: list[str] = []


class NewsItem(BaseModel, frozen=True):
    headline: str
    source: str
    date: str
    summary: str


class NewsList(BaseModel, frozen=True):
    items: list[NewsItem]


class LeadReport(BaseModel, frozen=True):
    lead_name: str
    company: str
    role: str
    company_summary: str
    key_findings: list[str]
    opportunities: list[str]
    risks: list[str]
    recommended_angle: str


class QualifiedLead(BaseModel, frozen=True):
    lead_name: str
    company: str
    score: float
    tier: str
    reasoning: str
    recommended_next_step: str


class QualifiedLeads(BaseModel, frozen=True):
    leads: list[QualifiedLead]
