"""Shared test schemas and helpers — used across all test files."""

from __future__ import annotations

from pydantic import BaseModel

from neograph import Node

# ═══════════════════════════════════════════════════════════════════════════
# SHARED SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class RawText(BaseModel, frozen=True):
    text: str

class Claims(BaseModel, frozen=True):
    items: list[str]

class ClassifiedClaims(BaseModel, frozen=True):
    classified: list[dict[str, str]]

class ClusterGroup(BaseModel, frozen=True):
    label: str
    claim_ids: list[str]

class Clusters(BaseModel, frozen=True):
    groups: list[ClusterGroup]

class MatchResult(BaseModel, frozen=True):
    cluster_label: str
    matched: list[str]

class MergedResult(BaseModel, frozen=True):
    final_text: str

class ValidationResult(BaseModel, frozen=True):
    passed: bool
    issues: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS — shorthand for validation tests
# ═══════════════════════════════════════════════════════════════════════════

def _producer(name: str, out: type) -> Node:
    return Node.scripted(name, fn="f", outputs=out)


def _consumer(name: str, in_: type, out: type) -> Node:
    return Node.scripted(name, fn="f", inputs=in_, outputs=out)
