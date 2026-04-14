"""Shared schemas, helpers, and type pools for Hypothesis property tests."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════════════════════
# Test schemas — multiple distinct types to exercise type-matching paths
# ═══════════════════════════════════════════════════════════════════════════


class Alpha(BaseModel, frozen=True):
    value: str = "alpha"


class Beta(BaseModel, frozen=True):
    score: float = 1.0
    iteration: int = 0


class Gamma(BaseModel, frozen=True):
    tags: list[str] = []


class FanItem(BaseModel, frozen=True):
    item_id: str
    data: str = "x"


class FanCollection(BaseModel, frozen=True):
    items: list[FanItem]


# Dict-form output schemas (multi-output nodes)
class DictResult(BaseModel, frozen=True):
    text: str = "primary"

class DictLog(BaseModel, frozen=True):
    entries: list[str] = []


# Sub-construct boundary schemas
class SubInput(BaseModel, frozen=True):
    payload: str = "in"

class SubOutput(BaseModel, frozen=True):
    result: str = "out"


# Type pools for randomization
INTERMEDIATE_TYPES = [Alpha, Beta, Gamma]
TYPE_PAIRS = [
    (Alpha, Beta),
    (Beta, Gamma),
    (Alpha, Gamma),
    (Beta, Alpha),
    (Gamma, Beta),
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _make_fn(output_type: type[BaseModel]):
    """Create a scripted function that returns a default instance of output_type."""
    def fn(_input_data, _config):
        return output_type()
    return fn


def _make_transform_fn(input_type: type[BaseModel], output_type: type[BaseModel]):
    """Create a scripted function that asserts input type and returns output."""
    def fn(input_data, _config):
        assert isinstance(input_data, (input_type, dict, type(None))), (
            f"Expected {input_type.__name__} or dict, got {type(input_data).__name__}"
        )
        return output_type()
    return fn
