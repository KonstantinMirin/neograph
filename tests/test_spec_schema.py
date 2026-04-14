"""Tests for pipeline and project JSON Schema validation.

These tests validate that YAML/JSON specs conform to the neograph JSON Schemas.
Written BEFORE the schemas exist (TDD red).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SCHEMAS_DIR = Path(__file__).parent.parent / "src" / "neograph" / "schemas"


@pytest.fixture
def pipeline_schema():
    schema_path = SCHEMAS_DIR / "neograph-pipeline.schema.json"
    assert schema_path.exists(), f"Pipeline schema not found: {schema_path}"
    return json.loads(schema_path.read_text())


@pytest.fixture
def project_schema():
    schema_path = SCHEMAS_DIR / "project.neograph.schema.json"
    assert schema_path.exists(), f"Project schema not found: {schema_path}"
    return json.loads(schema_path.read_text())


def _validate(instance: dict, schema: dict) -> None:
    """Validate a dict against a JSON Schema. Raises on failure."""
    from jsonschema import validate
    validate(instance=instance, schema=schema)


# =============================================================================
# Pipeline schema tests
# =============================================================================


class TestPipelineSchema:
    """Validate pipeline specs against neograph-pipeline.schema.json."""

    def test_minimal_pipeline_validates(self, pipeline_schema):
        """Simplest valid pipeline: one scripted node."""
        spec = {
            "name": "hello",
            "nodes": [
                {"name": "greet", "mode": "scripted", "outputs": "Greeting"}
            ],
            "pipeline": {"nodes": ["greet"]},
        }
        _validate(spec, pipeline_schema)

    def test_full_pipeline_with_all_features(self, pipeline_schema):
        """Pipeline with think/agent modes, loop, oracle, constructs."""
        spec = {
            "name": "review-pipeline",
            "nodes": [
                {
                    "name": "draft",
                    "mode": "think",
                    "model": "fast",
                    "prompt": "Write a draft about ${topic}.",
                    "outputs": "Draft",
                },
                {
                    "name": "review",
                    "mode": "think",
                    "model": "reason",
                    "prompt": "Score: ${draft.content}",
                    "outputs": "ReviewResult",
                },
                {
                    "name": "revise",
                    "mode": "think",
                    "model": "fast",
                    "prompt": "Revise: ${review.feedback}",
                    "outputs": "Draft",
                },
                {
                    "name": "classify",
                    "mode": "think",
                    "model": "reason",
                    "prompt": "Classify findings",
                    "outputs": "Findings",
                    "oracle": {
                        "models": ["reason", "fast", "creative"],
                        "merge_fn": "pick_best",
                    },
                },
            ],
            "constructs": [
                {
                    "name": "refine",
                    "input": "Draft",
                    "output": "Draft",
                    "nodes": ["review", "revise"],
                    "loop": {"when": "score < 0.8", "max_iterations": 5},
                }
            ],
            "pipeline": {"nodes": ["draft", "refine", "classify"]},
        }
        _validate(spec, pipeline_schema)

    def test_rejects_node_without_name(self, pipeline_schema):
        """Nodes must have a name."""
        spec = {
            "name": "bad",
            "nodes": [{"mode": "scripted", "outputs": "X"}],
            "pipeline": {"nodes": ["unnamed"]},
        }
        from jsonschema import ValidationError
        with pytest.raises(ValidationError):
            _validate(spec, pipeline_schema)

    def test_rejects_pipeline_without_name(self, pipeline_schema):
        """Top-level name is required."""
        spec = {
            "nodes": [{"name": "a", "mode": "scripted", "outputs": "X"}],
            "pipeline": {"nodes": ["a"]},
        }
        from jsonschema import ValidationError
        with pytest.raises(ValidationError):
            _validate(spec, pipeline_schema)

    def test_node_with_tools_validates(self, pipeline_schema):
        """Agent mode node with tools list."""
        spec = {
            "name": "agent-pipe",
            "nodes": [
                {
                    "name": "search",
                    "mode": "agent",
                    "model": "reason",
                    "prompt": "Find info about ${topic}",
                    "outputs": "SearchResult",
                    "tools": ["web_search", "read_file"],
                }
            ],
            "pipeline": {"nodes": ["search"]},
        }
        _validate(spec, pipeline_schema)

    def test_node_with_each_validates(self, pipeline_schema):
        """Fan-out node with each modifier."""
        spec = {
            "name": "fanout",
            "nodes": [
                {
                    "name": "verify",
                    "mode": "think",
                    "model": "fast",
                    "prompt": "Verify claim",
                    "outputs": "Result",
                    "each": {"over": "claims.items", "key": "claim_id"},
                }
            ],
            "pipeline": {"nodes": ["verify"]},
        }
        _validate(spec, pipeline_schema)


# =============================================================================
# Project schema tests
# =============================================================================


class TestProjectSchema:
    """Validate project surface files against project.neograph.schema.json."""

    def test_minimal_project_validates(self, project_schema):
        """Project with just types."""
        spec = {
            "types": {
                "Draft": {
                    "properties": {
                        "content": {"type": "string"},
                        "score": {"type": "number"},
                    }
                }
            }
        }
        _validate(spec, project_schema)

    def test_full_project_validates(self, project_schema):
        """Project with types, tools, and models."""
        spec = {
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
            },
            "tools": {
                "web_search": {
                    "description": "Search the web",
                    "params": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
                "read_file": {
                    "description": "Read a file",
                },
            },
            "models": {
                "reason": {
                    "provider": "openrouter",
                    "model": "anthropic/claude-sonnet-4-20250514",
                },
                "fast": {
                    "provider": "openrouter",
                    "model": "anthropic/claude-haiku-4-5-20251001",
                    "config": {"temperature": 0.3},
                },
            },
        }
        _validate(spec, project_schema)

    def test_rejects_type_without_properties(self, project_schema):
        """Type definitions must have properties."""
        spec = {
            "types": {
                "Bad": {"description": "no properties"}
            }
        }
        from jsonschema import ValidationError
        with pytest.raises(ValidationError):
            _validate(spec, project_schema)

    def test_rejects_model_without_provider(self, project_schema):
        """Model definitions must have provider."""
        spec = {
            "models": {
                "bad": {"model": "gpt-4"}
            }
        }
        from jsonschema import ValidationError
        with pytest.raises(ValidationError):
            _validate(spec, project_schema)


# =============================================================================
# _validate_spec warning tests (neograph-htui)
# =============================================================================


class TestValidateSpecWarnings:
    """_validate_spec must log a warning when validation is skipped."""

    VALID_SPEC = {
        "name": "hello",
        "nodes": [
            {"name": "greet", "mode": "scripted", "outputs": "Greeting"}
        ],
        "pipeline": {"nodes": ["greet"]},
    }

    @staticmethod
    def _capture_structlog():
        """Set up structlog capture, return (captured_list, teardown_fn)."""
        import structlog

        captured: list[dict] = []

        def capture(logger, method_name, event_dict):
            captured.append(dict(event_dict))
            raise structlog.DropEvent

        structlog.configure(processors=[capture])
        return captured, structlog.reset_defaults

    def test_warns_when_schema_file_not_found(self, monkeypatch):
        """_validate_spec logs a warning when the schema file does not exist."""
        import neograph.loader as loader_mod
        from neograph.loader import _validate_spec

        captured, teardown = self._capture_structlog()
        try:
            # Point __file__ to a fake path so the schema dir doesn't exist
            monkeypatch.setattr(loader_mod, "__file__", "/fake/path/loader.py")
            _validate_spec(self.VALID_SPEC)

            warnings = [e for e in captured if e.get("event") == "spec_validation_skipped"]
            assert len(warnings) == 1
            assert warnings[0]["reason"] == "schema file not found"
        finally:
            teardown()

    def test_warns_when_jsonschema_not_installed(self):
        """_validate_spec logs a warning when jsonschema is not importable."""
        import builtins

        from neograph.loader import _validate_spec

        captured, teardown = self._capture_structlog()
        try:
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "jsonschema":
                    raise ImportError("mocked")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = fake_import
            try:
                _validate_spec(self.VALID_SPEC)
            finally:
                builtins.__import__ = real_import

            warnings = [e for e in captured if e.get("event") == "spec_validation_skipped"]
            assert len(warnings) == 1
            assert warnings[0]["reason"] == "jsonschema not installed"
        finally:
            teardown()

    def test_valid_spec_no_warning(self):
        """A valid spec with jsonschema installed produces no skip warning."""
        from neograph.loader import _validate_spec

        captured, teardown = self._capture_structlog()
        try:
            _validate_spec(self.VALID_SPEC)
            warnings = [e for e in captured if e.get("event") == "spec_validation_skipped"]
            assert len(warnings) == 0
        finally:
            teardown()

    def test_invalid_spec_raises_validation_error(self):
        """_validate_spec raises ValidationError for an invalid spec."""
        from jsonschema import ValidationError

        from neograph.loader import _validate_spec

        bad_spec = {"not": "a valid spec"}
        with pytest.raises(ValidationError):
            _validate_spec(bad_spec)
