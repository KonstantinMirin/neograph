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
        """Spec with loop: {when: 'score < 0.8'} compiles to a looping graph.
        Honest: no forgiving fallbacks, strict type assertion on input."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        def seed_fn(input_data, config):
            return Draft(content="seed-marker", score=0.1)  # unique score to detect

        call_count = [0]
        received_inputs = []

        def refine_fn(input_data, config):
            call_count[0] += 1
            received_inputs.append(type(input_data).__name__)
            assert isinstance(input_data, Draft), (
                f"refine expected Draft, got {type(input_data).__name__}: {input_data!r}"
            )
            return Draft(content=f"v{call_count[0]}", score=input_data.score + 0.3, iteration=input_data.iteration + 1)

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

        # Seed starts at 0.1, each iteration adds 0.3: 0.1→0.4→0.7→1.0
        assert call_count[0] == 3
        assert result["refine"][-1].score >= 0.8
        # First iteration received seed output (score=0.1), not a fallback
        assert result["refine"][0].score == pytest.approx(0.4)  # 0.1 + 0.3


class TestLoadSpecWithConstruct:
    """Sub-construct from spec."""

    def test_construct_with_loop_compiles_and_runs(self):
        """Spec construct with loop modifier runs as a looping sub-construct.
        Honest: unique seed score, strict type assertion, no fallbacks."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        def draft_fn(input_data, config):
            return Draft(content="construct-seed", score=0.15)  # unique score

        call_count = [0]

        def improve_fn(input_data, config):
            call_count[0] += 1
            assert isinstance(input_data, Draft), (
                f"improve expected Draft, got {type(input_data).__name__}: {input_data!r}"
            )
            return Draft(content=f"v{call_count[0]}", score=input_data.score + 0.3, iteration=input_data.iteration + 1)

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

        # Seed at 0.15, +0.3 each: 0.15→0.45→0.75→1.05 (3 iterations)
        assert call_count[0] == 3
        assert result["refine"][-1].score >= 0.8
        # First iteration derived from seed (0.15 + 0.3 = 0.45)
        assert result["refine"][0].score == pytest.approx(0.45)


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
        Honest: unique seed score, strict type assertion, scores prove data flow."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        def seed_fn(input_data, config):
            return Draft(content="history-seed", score=0.05)  # unique non-zero score

        iteration = [0]

        def refine_fn(input_data, config):
            iteration[0] += 1
            assert isinstance(input_data, Draft), (
                f"refine expected Draft, got {type(input_data).__name__}: {input_data!r}"
            )
            return Draft(
                content=f"v{iteration[0]}",
                score=input_data.score + 0.25,
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
        assert isinstance(history, list)
        # Seed at 0.05, +0.25 each: 0.30, 0.55, 0.80 (3 iterations, exits at 0.80)
        assert len(history) == 3
        # Scores prove data flowed from seed (0.05), not from a fallback (0.0)
        scores = [round(d.score, 2) for d in history]
        assert scores == [0.30, 0.55, 0.80]
        # Iteration numbers are sequential
        iterations = [d.iteration for d in history]
        assert iterations == [1, 2, 3]


class TestLoadSpecMultiNodeConstruct:
    """Multi-node sub-construct (review + revise) with Loop."""

    def test_multi_node_construct_loops_as_unit(self):
        """A sub-construct with two nodes (review + revise) loops until
        the output meets the condition. Both nodes re-run each iteration.
        Honest: both functions derive output from input_data, not closures."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)
        register_type("ReviewResult", ReviewResult)

        review_received = []
        revise_received = []

        def mn_review(input_data, config):
            review_received.append(input_data)
            # Must read draft from input_data to derive score
            d = input_data if isinstance(input_data, Draft) else None
            base_score = d.score if d else 0.0
            new_score = base_score + 0.3
            return ReviewResult(score=min(new_score, 1.0), feedback=f"reviewed-{new_score:.1f}")

        def mn_revise(input_data, config):
            revise_received.append(input_data)
            # Must read review from input_data dict
            assert isinstance(input_data, dict), f"revise expected dict, got {type(input_data).__name__}"
            review = input_data.get("review")
            assert review is not None, "revise did not receive review output"
            assert isinstance(review, ReviewResult), f"review is {type(review).__name__}"
            return Draft(content=review.feedback, score=review.score, iteration=1)

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

        # review ran 3 times (0.0+0.3=0.3, 0.3+0.3=0.6, 0.6+0.3=0.9)
        assert len(review_received) == 3
        # revise actually received review output (not None, not just Draft)
        assert len(revise_received) == 3
        for r in revise_received:
            assert isinstance(r, dict), f"revise got {type(r).__name__}, expected dict"
            assert "review" in r, f"revise input missing 'review' key: {list(r.keys())}"
        # Final result reflects data that flowed through review → revise
        final = result["refine"][-1]
        assert final.score >= 0.8
        assert "reviewed-" in final.content, f"Content should contain review feedback: {final.content!r}"


class TestLoadSpecDataFlow:
    """Verify data flows correctly between spec-loaded nodes."""

    def test_second_node_receives_upstream_output_with_exact_values(self):
        """Scripted node receives upstream output with exact field values.
        Tests basic data flow, not variable substitution (that's in test_inline_prompts.py)."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        captured_input = [None]

        def capture_fn(input_data, config):
            captured_input[0] = input_data
            return Draft(content="captured", score=1.0)

        register_scripted("capture_fn", capture_fn)

        def seed_fn(input_data, config):
            return Draft(content="the real content", score=0.42, iteration=7)

        register_scripted("var_seed", seed_fn)

        spec = {
            "name": "flow-test",
            "nodes": [
                {"name": "seed", "mode": "scripted", "scripted_fn": "var_seed", "outputs": "Draft"},
                {"name": "process", "mode": "scripted", "scripted_fn": "capture_fn", "outputs": "Draft"},
            ],
            "pipeline": {"nodes": ["seed", "process"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        result = run(graph, input={"node_id": "test"})

        inp = captured_input[0]
        assert inp is not None, "process node received None — input wiring broken"
        assert isinstance(inp, Draft), f"Expected Draft, got {type(inp).__name__}"
        assert inp.score == 0.42, f"Expected score 0.42 from seed, got {inp.score}"
        assert inp.content == "the real content"


class TestLoadSpecWiringHonesty:
    """Tests that verify wiring actually works — no forgiving fallbacks."""

    def test_second_node_receives_first_nodes_exact_output(self):
        """Verify the second node receives exactly the first node's output
        object, not None, not a default, not a dict wrapper."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        received = [None]

        def seed_fn(input_data, config):
            return Draft(content="unique-marker-xyz", score=0.77, iteration=42)

        def consumer_fn(input_data, config):
            received[0] = input_data
            return Draft(content="consumed", score=1.0)

        register_scripted("wire_seed", seed_fn)
        register_scripted("wire_consumer", consumer_fn)

        spec = {
            "name": "wiring-test",
            "nodes": [
                {"name": "seed", "mode": "scripted", "scripted_fn": "wire_seed", "outputs": "Draft"},
                {"name": "consumer", "mode": "scripted", "scripted_fn": "wire_consumer", "outputs": "Draft"},
            ],
            "pipeline": {"nodes": ["seed", "consumer"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        run(graph, input={"node_id": "test"})

        r = received[0]
        assert r is not None, "Consumer received None — wiring is broken"
        assert isinstance(r, Draft), f"Consumer received {type(r).__name__}, not Draft"
        assert r.content == "unique-marker-xyz", f"Consumer got wrong content: {r.content!r}"
        assert r.score == 0.77
        assert r.iteration == 42

    def test_multi_node_construct_revise_actually_reads_review(self):
        """The revise node inside a multi-node construct must receive BOTH
        the draft (from input port) AND the review output. If revise only
        sees Draft but not ReviewResult, the sub-construct wiring is broken."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)
        register_type("ReviewResult", ReviewResult)

        revise_received = [None]

        def mn2_draft(input_data, config):
            return Draft(content="initial", score=0.0)

        def mn2_review(input_data, config):
            return ReviewResult(score=0.9, feedback="unique-feedback-marker")

        def mn2_revise(input_data, config):
            revise_received[0] = input_data
            # Must use review.feedback from input — NOT from closure
            if isinstance(input_data, dict) and "review" in input_data:
                review = input_data["review"]
                return Draft(content=review.feedback, score=review.score, iteration=1)
            # Fallback — proves the bug: revise didn't get review
            return Draft(content="NO_REVIEW_RECEIVED", score=0.9, iteration=1)

        register_scripted("mn2_draft", mn2_draft)
        register_scripted("mn2_review", mn2_review)
        register_scripted("mn2_revise", mn2_revise)

        spec = {
            "name": "honest-construct",
            "nodes": [
                {"name": "draft", "mode": "scripted", "scripted_fn": "mn2_draft", "outputs": "Draft"},
                {"name": "review", "mode": "scripted", "scripted_fn": "mn2_review", "outputs": "ReviewResult"},
                {"name": "revise", "mode": "scripted", "scripted_fn": "mn2_revise", "outputs": "Draft"},
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

        # The real assertion: revise must have received the review output
        final = result["refine"][-1]
        assert final.content == "unique-feedback-marker", (
            f"Revise did not receive review output. Got content={final.content!r}. "
            f"Revise received: {revise_received[0]!r}"
        )

    def test_loop_first_iteration_receives_upstream_not_none(self):
        """On the first loop iteration, the node must receive the upstream's
        output, not None. Tests that _extract_input with Loop works for
        spec-loaded nodes on the first call."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        first_input = [None]
        call_count = [0]

        def loop_seed(input_data, config):
            return Draft(content="seed-output", score=0.0, iteration=0)

        def loop_node(input_data, config):
            call_count[0] += 1
            if call_count[0] == 1:
                first_input[0] = input_data
            d = input_data if isinstance(input_data, Draft) else Draft(content="", score=0.0)
            return Draft(content="looped", score=d.score + 0.5, iteration=call_count[0])

        register_scripted("fi_seed", loop_seed)
        register_scripted("fi_loop", loop_node)

        spec = {
            "name": "first-iter",
            "nodes": [
                {"name": "seed", "mode": "scripted", "scripted_fn": "fi_seed", "outputs": "Draft"},
                {
                    "name": "loop-node",
                    "mode": "scripted",
                    "scripted_fn": "fi_loop",
                    "outputs": "Draft",
                    "loop": {"when": "score < 0.8", "max_iterations": 5},
                },
            ],
            "pipeline": {"nodes": ["seed", "loop-node"]},
        }

        construct = load_spec(spec)
        graph = compile(construct)
        run(graph, input={"node_id": "test"})

        fi = first_input[0]
        assert fi is not None, "Loop node received None on first iteration — upstream wiring broken"
        assert isinstance(fi, Draft), f"Expected Draft, got {type(fi).__name__}"
        assert fi.content == "seed-output", f"First iteration should receive seed output, got {fi.content!r}"


class TestLoadSpecSizeLimit:
    """YAML bomb DoS protection — neograph-cfrd."""

    def test_rejects_oversized_string_input(self):
        """A string spec exceeding MAX_SPEC_SIZE raises ValueError."""
        from neograph.loader import load_spec

        oversized = "a" * (1_048_576 + 1)
        with pytest.raises(ValueError, match="exceeds maximum size"):
            load_spec(oversized)

    def test_rejects_oversized_file_input(self, tmp_path):
        """A file spec exceeding MAX_SPEC_SIZE raises ValueError."""
        from neograph.loader import load_spec

        big_file = tmp_path / "bomb.yaml"
        big_file.write_text("a" * (1_048_576 + 1))
        with pytest.raises(ValueError, match="exceeds maximum size"):
            load_spec(str(big_file))

    def test_accepts_input_at_size_limit(self):
        """A string exactly at MAX_SPEC_SIZE is not rejected by the size check.
        (It will fail YAML parsing, but NOT the size guard.)"""
        from neograph.loader import load_spec

        at_limit = "a" * 1_048_576
        # Should not raise ValueError for size — will raise something else
        # (invalid YAML/spec), but not the size guard.
        with pytest.raises(Exception) as exc_info:
            load_spec(at_limit)
        assert "exceeds maximum size" not in str(exc_info.value)


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

    def test_raises_when_loop_condition_unparseable(self):
        """Malformed when expression raises ValueError at load time."""
        from neograph.loader import load_spec
        from neograph.spec_types import register_type

        register_type("Draft", Draft)

        spec = {
            "name": "bad-loop",
            "nodes": [
                {
                    "name": "x",
                    "mode": "scripted",
                    "scripted_fn": "whatever",
                    "outputs": "Draft",
                    "loop": {"when": "this is not a valid expression!!!"},
                }
            ],
            "pipeline": {"nodes": ["x"]},
        }

        with pytest.raises(ValueError, match="Invalid condition"):
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
