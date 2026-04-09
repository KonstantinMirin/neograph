"""Schemas for the code review pipeline."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class DiffInput(BaseModel, frozen=True):
    diff_text: str


class ChangedFile(BaseModel, frozen=True):
    path: str
    language: str
    hunks: str


class ChangedFiles(BaseModel, frozen=True):
    files: list[ChangedFile]


class Finding(BaseModel, frozen=True):
    severity: Severity
    location: str
    description: str
    suggestion: str


class StyleFindings(BaseModel, frozen=True):
    findings: list[Finding] = []


class LogicFindings(BaseModel, frozen=True):
    findings: list[Finding] = []


class SecurityFindings(BaseModel, frozen=True):
    findings: list[Finding] = []


class FileReview(BaseModel, frozen=True):
    path: str
    findings: list[Finding] = []


class ReviewReport(BaseModel, frozen=True):
    summary: str
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    findings: list[Finding] = []
    positive_notes: list[str] = []
