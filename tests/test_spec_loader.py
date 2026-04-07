"""Tests for load_spec: YAML/JSON spec -> Construct IR -> compile -> run.

End-to-end integration tests proving the full pipeline spec story.
Written BEFORE the loader exists (TDD red).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from neograph import compile, run
from neograph.errors import ConfigurationError
from neograph.factory import register_scripted


# -- Schemas for loader tests ------------------------------------------------

class Draft(BaseModel, frozen=True):
    content: str
    score: float = 0.0
    iteration: int = 0


class ReviewResult(BaseModel, frozen=True):
    score: float
    feedback: str


# -- Helpers -----------------------------------------------------------------

SIMPLE_PROJECT = {
    "types": {
        "Draft": {
            "properties": {
                "content": {"type": "string"},
                "score": {"type": "number"},
                "iteration": {"type": "integer"},
            }
        },
    }
}

MULTI_TYPE_PROJECT = {
    "types": {
        "Draft": {
            "properties": {
                "content": {"type": "string"},
                "score": {"type": "number"},
                "iteration": {"type": "integer"},
            }
        },
        "ReviewResult": {
            "properties": {
                "score": {"type": "number"},
                "feedback": {"type": "string"},
            }
        },
    }
}


# =============================================================================
# Test cases
# =============================================================================


class TestLoadSpecBasic:
    """Basic load_spec: YAML string -> Construct -> compile -> run."""

    def test_simple_scripted_pipeline_runs(self):
        """Minimal spec: one scripted node, load + compile + run."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        call_count = [0]

        def make_draft(input_data, config):
            call_count[0] += 1
            return Draft(content="hello", score=1.0)

        register_scripted("make_draft", make_draft)

        spec = {
            "name": "simple",
            "nodes": [
                {"name": "draft", "mode": "scripted", "scripted_fn": "make_draft", "outputs": "Draft"}
            ],
            "pipeline": {"nodes": ["draft"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        assert call_count[0] == 1
        assert result["draft"].content == "hello"

    def test_two_node_pipeline_wires_correctly(self):
        """Two nodes: seed -> refine. Refine reads seed's output."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        def seed_fn(input_data, config):
            return Draft(content="v0", score=0.0)

        def refine_fn(input_data, config):
            d = input_data if isinstance(input_data, Draft) else input_data.get("seed")
            return Draft(content="v1", score=d.score + 0.5, iteration=1)

        register_scripted("seed_fn", seed_fn)
        register_scripted("refine_fn", refine_fn)

        spec = {
            "name": "two-node",
            "nodes": [
                {"name": "seed", "mode": "scripted", "scripted_fn": "seed_fn", "outputs": "Draft"},
                {"name": "refine", "mode": "scripted", "scripted_fn": "refine_fn", "outputs": "Draft"},
            ],
            "pipeline": {"nodes": ["seed", "refine"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        assert result["refine"].score == 0.5

    def test_auto_generated_types_from_project(self):
        """Types defined in project surface are auto-generated and usable."""
        from neograph.loader import load_spec

        def gen_fn(input_data, config):
            # Import dynamically — the type is auto-generated
            from neograph.spec_types import lookup_type
            DraftType = lookup_type("Draft")
            return DraftType(content="generated", score=0.9, iteration=0)

        register_scripted("gen_fn", gen_fn)

        spec = {
            "name": "auto-types",
            "nodes": [
                {"name": "gen", "mode": "scripted", "scripted_fn": "gen_fn", "outputs": "Draft"}
            ],
            "pipeline": {"nodes": ["gen"]},
        }

        construct = load_spec(spec, project=SIMPLE_PROJECT)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        assert result["gen"].content == "generated"
        assert result["gen"].score == 0.9


class TestLoadSpecWithLoop:
    """Loop modifier from spec."""

    def test_self_loop_runs_until_condition_met(self):
        """Spec with loop: {when: 'score < 0.8'} compiles to a looping graph."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        def seed_fn(input_data, config):
            return Draft(content="v0", score=0.0)

        call_count = [0]

        def refine_fn(input_data, config):
            call_count[0] += 1
            d = input_data if isinstance(input_data, Draft) else next(iter(input_data.values()))
            return Draft(content=f"v{call_count[0]}", score=d.score + 0.3, iteration=d.iteration + 1)

        register_scripted("loop_seed", seed_fn)
        register_scripted("loop_refine", refine_fn)

        spec = {
            "name": "loop-test",
            "nodes": [
                {"name": "seed", "mode": "scripted", "scripted_fn": "loop_seed", "outputs": "Draft"},
                {
                    "name": "refine",
                    "mode": "scripted",
                    "scripted_fn": "loop_refine",
                    "outputs": "Draft",
                    "loop": {"when": "score < 0.8", "max_iterations": 10},
                },
            ],
            "pipeline": {"nodes": ["seed", "refine"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        assert call_count[0] == 3  # 0.0 -> 0.3 -> 0.6 -> 0.9
        assert result["refine"][-1].score >= 0.8


class TestLoadSpecWithConstruct:
    """Sub-construct from spec."""

    def test_construct_with_loop_compiles_and_runs(self):
        """Spec construct with loop modifier runs as a looping sub-construct.
        Uses a single-node sub-construct (self-loop inside) for simplicity."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        def draft_fn(input_data, config):
            return Draft(content="initial", score=0.0)

        call_count = [0]

        def improve_fn(input_data, config):
            call_count[0] += 1
            d = input_data if isinstance(input_data, Draft) else Draft(content="x", score=0.0)
            return Draft(content=f"v{call_count[0]}", score=d.score + 0.3, iteration=d.iteration + 1)

        register_scripted("cl_draft", draft_fn)
        register_scripted("cl_improve", improve_fn)

        spec = {
            "name": "construct-loop",
            "nodes": [
                {"name": "draft", "mode": "scripted", "scripted_fn": "cl_draft", "outputs": "Draft"},
                {"name": "improve", "mode": "scripted", "scripted_fn": "cl_improve", "outputs": "Draft"},
            ],
            "constructs": [
                {
                    "name": "refine",
                    "input": "Draft",
                    "output": "Draft",
                    "nodes": ["improve"],
                    "loop": {"when": "score < 0.8", "max_iterations": 10},
                }
            ],
            "pipeline": {"nodes": ["draft", "refine"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        assert call_count[0] >= 3  # 0.0 -> 0.3 -> 0.6 -> 0.9
        assert result["refine"][-1].score >= 0.8


class TestLoadSpecErrors:
    """Error paths."""

    def test_raises_when_type_not_registered(self):
        """Referencing an unregistered type raises ConfigurationError."""
        from neograph.loader import load_spec

        spec = {
            "name": "bad",
            "nodes": [
                {"name": "x", "mode": "scripted", "scripted_fn": "whatever", "outputs": "NonExistentType"}
            ],
            "pipeline": {"nodes": ["x"]},
        }

        with pytest.raises(ConfigurationError, match="not registered"):
            load_spec(spec)
