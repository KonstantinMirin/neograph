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


class TestLoadSpecOracle:
    """Oracle modifier from spec — multi-model ensemble."""

    def test_oracle_with_models_dispatches_to_each_model(self):
        """Oracle(models=["reason", "fast"]) sends each generator to a different model.
        The merge function receives all variants and combines them."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        seen_models = []

        def oracle_gen(input_data, config):
            model = config.get("configurable", {}).get("_oracle_model", "unknown")
            seen_models.append(model)
            return Draft(content=f"from-{model}", score=0.5)

        def oracle_merge(variants, config):
            contents = [v.content for v in variants]
            return Draft(content="|".join(sorted(contents)), score=1.0)

        register_scripted("oracle_gen", oracle_gen)
        register_scripted("oracle_merge", oracle_merge)

        spec = {
            "name": "oracle-test",
            "nodes": [
                {
                    "name": "generate",
                    "mode": "scripted",
                    "scripted_fn": "oracle_gen",
                    "outputs": "Draft",
                    "oracle": {
                        "models": ["reason", "fast"],
                        "merge_fn": "oracle_merge",
                    },
                }
            ],
            "pipeline": {"nodes": ["generate"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        # Both models were dispatched
        assert set(seen_models) == {"reason", "fast"}
        # Merge combined both
        merged = result["generate"]
        assert "from-fast" in merged.content
        assert "from-reason" in merged.content


class TestLoadSpecYamlString:
    """YAML string input (not just dict)."""

    def test_yaml_string_parses_and_runs(self):
        """load_spec accepts a YAML string, not just a dict."""
        import yaml as _yaml
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        def yaml_fn(input_data, config):
            return Draft(content="from-yaml", score=1.0)

        register_scripted("yaml_fn", yaml_fn)

        yaml_str = _yaml.dump({
            "name": "yaml-pipeline",
            "nodes": [
                {"name": "gen", "mode": "scripted", "scripted_fn": "yaml_fn", "outputs": "Draft"}
            ],
            "pipeline": {"nodes": ["gen"]},
        })

        construct = load_spec(yaml_str)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        assert result["gen"].content == "from-yaml"


class TestLoadSpecLoopHistory:
    """Loop iteration history is preserved in results."""

    def test_loop_preserves_all_iterations_in_append_list(self):
        """Each loop iteration's output is preserved in the append-list.
        result["node"] is a list with one entry per iteration."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        def seed_fn(input_data, config):
            return Draft(content="v0", score=0.0)

        iteration = [0]

        def refine_fn(input_data, config):
            iteration[0] += 1
            d = input_data if isinstance(input_data, Draft) else Draft(content="", score=0.0)
            return Draft(
                content=f"v{iteration[0]}",
                score=d.score + 0.25,
                iteration=iteration[0],
            )

        register_scripted("hist_seed", seed_fn)
        register_scripted("hist_refine", refine_fn)

        spec = {
            "name": "history-test",
            "nodes": [
                {"name": "seed", "mode": "scripted", "scripted_fn": "hist_seed", "outputs": "Draft"},
                {
                    "name": "refine",
                    "mode": "scripted",
                    "scripted_fn": "hist_refine",
                    "outputs": "Draft",
                    "loop": {"when": "score < 0.8", "max_iterations": 10},
                },
            ],
            "pipeline": {"nodes": ["seed", "refine"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        history = result["refine"]
        # Append-list preserves all iterations
        assert isinstance(history, list)
        assert len(history) == 4  # 0.0 -> 0.25 -> 0.50 -> 0.75 -> 1.0 (4 calls)
        # Each entry has increasing score
        scores = [d.score for d in history]
        assert scores == sorted(scores)
        assert scores[-1] >= 0.8
        # Iteration numbers are sequential
        iterations = [d.iteration for d in history]
        assert iterations == [1, 2, 3, 4]


class TestLoadSpecMultiNodeConstruct:
    """Multi-node sub-construct (review + revise) with Loop."""

    def test_multi_node_construct_loops_as_unit(self):
        """A sub-construct with two nodes (review + revise) loops until
        the output meets the condition. Both nodes re-run each iteration."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)
        register_type("ReviewResult", ReviewResult)

        review_count = [0]

        def mn_review(input_data, config):
            review_count[0] += 1
            score = 0.3 * review_count[0]
            return ReviewResult(score=min(score, 1.0), feedback=f"iter-{review_count[0]}")

        def mn_revise(input_data, config):
            # Inside sub-construct: reads review from state, draft from input
            return Draft(content="revised", score=0.3 * review_count[0], iteration=review_count[0])

        def mn_draft(input_data, config):
            return Draft(content="initial", score=0.0)

        register_scripted("mn_draft", mn_draft)
        register_scripted("mn_review", mn_review)
        register_scripted("mn_revise", mn_revise)

        spec = {
            "name": "multi-node-construct",
            "nodes": [
                {"name": "draft", "mode": "scripted", "scripted_fn": "mn_draft", "outputs": "Draft"},
                {"name": "review", "mode": "scripted", "scripted_fn": "mn_review", "outputs": "ReviewResult"},
                {"name": "revise", "mode": "scripted", "scripted_fn": "mn_revise", "outputs": "Draft"},
            ],
            "constructs": [
                {
                    "name": "refine",
                    "input": "Draft",
                    "output": "Draft",
                    "nodes": ["review", "revise"],
                    "loop": {"when": "score < 0.8", "max_iterations": 10},
                }
            ],
            "pipeline": {"nodes": ["draft", "refine"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        # review ran 3 times (0.3, 0.6, 0.9 — exits at 0.9 >= 0.8)
        assert review_count[0] == 3
        # Sub-construct result is an append-list
        assert isinstance(result["refine"], list)
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

    def test_raises_when_pipeline_refs_unknown_node(self):
        """Pipeline referencing a non-existent node raises ConfigurationError."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        spec = {
            "name": "bad-ref",
            "nodes": [
                {"name": "seed", "mode": "scripted", "scripted_fn": "x", "outputs": "Draft"}
            ],
            "pipeline": {"nodes": ["seed", "ghost"]},
        }

        with pytest.raises(ConfigurationError, match="ghost"):
            load_spec(spec)

    def test_raises_when_construct_refs_unknown_node(self):
        """Construct referencing a non-existent node raises ConfigurationError."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        spec = {
            "name": "bad-construct",
            "nodes": [
                {"name": "seed", "mode": "scripted", "scripted_fn": "x", "outputs": "Draft"}
            ],
            "constructs": [
                {"name": "sub", "input": "Draft", "output": "Draft", "nodes": ["nonexistent"]}
            ],
            "pipeline": {"nodes": ["seed", "sub"]},
        }

        with pytest.raises(ConfigurationError, match="nonexistent"):
            load_spec(spec)
