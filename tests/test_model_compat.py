"""Model compatibility test suite -- parametrized across output strategies and model tiers.

Verifies schema round-trip: compile -> fake LLM response -> parse -> validate
for a moderately complex schema (nested models, enums, lists, Optional fields).
Also verifies renderers produce valid output for the complex schema.
"""

from __future__ import annotations

import json

import pytest

from neograph import Construct, Node, compile, run
from neograph.describe_type import describe_type
from neograph.renderers import JsonRenderer, XmlRenderer
from tests.fakes import StructuredFake, TextFake, configure_fake_llm
from tests.schemas_compat import ContactMethod, Experience, LeadProfile

# ---------------------------------------------------------------------------
# Fixtures: canonical test data
# ---------------------------------------------------------------------------

SAMPLE_PROFILE = LeadProfile(
    name="Alice Chen",
    title="VP Engineering",
    company="Acme Corp",
    experience=[
        Experience(company="Acme Corp", role="VP Engineering", years=3, current=True),
        Experience(company="Beta Inc", role="Staff Engineer", years=5),
    ],
    preferred_contact=ContactMethod.email,
    tags=["enterprise", "decision-maker"],
    notes="Met at re:Invent 2025",
)

SAMPLE_JSON = SAMPLE_PROFILE.model_dump_json()


def _make_profile_from_model(model):
    """StructuredFake respond callable -- returns a LeadProfile instance."""
    return SAMPLE_PROFILE


# ---------------------------------------------------------------------------
# Parametrized compile + run matrix
# ---------------------------------------------------------------------------

OUTPUT_STRATEGIES = ["structured", "json_mode", "text"]
MODEL_TIERS = ["reason", "fast", "creative"]


def _make_pipeline(strategy: str, model_tier: str) -> Construct:
    """Build a single-node think pipeline with the given strategy and model tier."""
    llm_config = {"output_strategy": strategy} if strategy != "structured" else {}
    node = Node(
        name="profile",
        mode="think",
        outputs=LeadProfile,
        model=model_tier,
        prompt="test/profile",
        llm_config=llm_config,
    )
    return Construct(f"compat-{strategy}-{model_tier}", nodes=[node])


class TestCompileSucceeds:
    """Every (strategy, model_tier) combination must compile without error."""

    @pytest.mark.parametrize("strategy", OUTPUT_STRATEGIES)
    @pytest.mark.parametrize("model_tier", MODEL_TIERS)
    def test_compile_succeeds(self, strategy, model_tier):
        # LLM nodes require configure_llm before compile
        configure_fake_llm(lambda tier: StructuredFake(_make_profile_from_model))

        pipeline = _make_pipeline(strategy, model_tier)
        compile(pipeline)  # succeeds = test passes; raises = test fails


class TestSchemaRoundTrip:
    """Full round-trip: compile -> fake LLM -> parse -> validate output type."""

    @pytest.mark.parametrize("model_tier", MODEL_TIERS)
    def test_structured_strategy_round_trip(self, model_tier):
        """structured: StructuredFake returns model instance directly."""
        configure_fake_llm(lambda tier: StructuredFake(_make_profile_from_model))

        pipeline = _make_pipeline("structured", model_tier)
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "compat-test"})

        profile = result["profile"]
        assert isinstance(profile, LeadProfile)
        assert profile.name == "Alice Chen"
        assert len(profile.experience) == 2
        assert profile.experience[0].current is True
        assert profile.preferred_contact == ContactMethod.email
        assert profile.tags == ["enterprise", "decision-maker"]
        assert profile.notes == "Met at re:Invent 2025"

    @pytest.mark.parametrize("model_tier", MODEL_TIERS)
    def test_json_mode_strategy_round_trip(self, model_tier):
        """json_mode: TextFake returns raw JSON, framework parses to model."""
        configure_fake_llm(lambda tier: TextFake(SAMPLE_JSON))

        pipeline = _make_pipeline("json_mode", model_tier)
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "compat-test"})

        profile = result["profile"]
        assert isinstance(profile, LeadProfile)
        assert profile.name == "Alice Chen"
        assert profile.preferred_contact == ContactMethod.email
        assert profile.experience[1].years == 5

    @pytest.mark.parametrize("model_tier", MODEL_TIERS)
    def test_text_strategy_round_trip(self, model_tier):
        """text: TextFake returns JSON embedded in prose, framework extracts."""
        prose_wrapped = f"Here is the lead profile:\n{SAMPLE_JSON}\nAnalysis complete."
        configure_fake_llm(lambda tier: TextFake(prose_wrapped))

        pipeline = _make_pipeline("text", model_tier)
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "compat-test"})

        profile = result["profile"]
        assert isinstance(profile, LeadProfile)
        assert profile.name == "Alice Chen"
        assert profile.preferred_contact == ContactMethod.email


# ---------------------------------------------------------------------------
# Renderer tests for complex schema
# ---------------------------------------------------------------------------

class TestXmlRendererComplex:
    """XmlRenderer handles nested models, enums, lists, and Optional fields."""

    def test_renders_nested_model_with_enum_and_list(self):
        renderer = XmlRenderer()
        output = renderer.render(SAMPLE_PROFILE)

        assert "<name>Alice Chen</name>" in output
        assert "<company>Acme Corp</company>" in output
        assert "<preferred_contact>email</preferred_contact>" in output
        assert "<experience>" in output
        assert "<item>" in output
        assert "<years>3</years>" in output
        assert "<current>True</current>" in output
        assert "<tags>" in output
        assert "enterprise" in output
        assert "Met at re:Invent 2025" in output

    def test_renders_none_field_as_text(self):
        """Optional field set to None renders as None text."""
        profile_no_notes = LeadProfile(
            name="Bob",
            title="CTO",
            company="Test",
            experience=[],
            preferred_contact=ContactMethod.phone,
        )
        renderer = XmlRenderer()
        output = renderer.render(profile_no_notes)

        assert "<name>Bob</name>" in output
        # notes defaults to None, tags defaults to []
        assert "<notes>None</notes>" in output


class TestJsonRendererComplex:
    """JsonRenderer produces valid JSON for the complex schema."""

    def test_produces_valid_json(self):
        renderer = JsonRenderer()
        output = renderer.render(SAMPLE_PROFILE)

        parsed = json.loads(output)
        assert parsed["name"] == "Alice Chen"
        assert len(parsed["experience"]) == 2
        assert parsed["preferred_contact"] == "email"
        assert parsed["tags"] == ["enterprise", "decision-maker"]
        assert parsed["notes"] == "Met at re:Invent 2025"

    def test_round_trips_through_json(self):
        """JSON output can be parsed back into the original model."""
        renderer = JsonRenderer()
        output = renderer.render(SAMPLE_PROFILE)
        reconstructed = LeadProfile.model_validate_json(output)

        assert reconstructed == SAMPLE_PROFILE


class TestDescribeTypeComplex:
    """describe_type produces valid schema notation for the complex schema."""

    def test_describes_all_fields(self):
        schema = describe_type(LeadProfile)

        assert "name: string" in schema
        assert "title: string" in schema
        assert "company: string" in schema
        assert "experience:" in schema
        assert "years: int" in schema
        assert "current: bool" in schema
        assert "tags:" in schema
        assert "notes:" in schema

    def test_hoists_enum(self):
        """ContactMethod enum should be hoisted as an enum declaration."""
        schema = describe_type(LeadProfile)

        assert "enum ContactMethod" in schema
        assert '"email"' in schema
        assert '"phone"' in schema
        assert '"linkedin"' in schema

    def test_hoists_experience_when_auto(self):
        """Experience appears in a list, so auto-hoisting may apply."""
        schema = describe_type(LeadProfile, hoist_classes="all")

        assert "type Experience" in schema

    def test_optional_field_marked(self):
        """Optional fields (with defaults) should show 'or null'."""
        schema = describe_type(LeadProfile)

        # notes is Optional[str] with default None -- should show null
        assert "null" in schema


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for the complex schema across strategies."""

    def test_empty_lists_round_trip_via_json_mode(self):
        """Model with empty lists and None optional parses correctly."""
        minimal = LeadProfile(
            name="Minimal",
            title="Intern",
            company="Startup",
            experience=[],
            preferred_contact=ContactMethod.linkedin,
        )
        minimal_json = minimal.model_dump_json()
        configure_fake_llm(lambda tier: TextFake(minimal_json))

        pipeline = _make_pipeline("json_mode", "fast")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "edge-test"})

        profile = result["profile"]
        assert isinstance(profile, LeadProfile)
        assert profile.experience == []
        assert profile.tags == []
        assert profile.notes is None

    def test_markdown_fenced_json_parsed_in_json_mode(self):
        """json_mode strips markdown fences before parsing."""
        fenced = f"```json\n{SAMPLE_JSON}\n```"
        configure_fake_llm(lambda tier: TextFake(fenced))

        pipeline = _make_pipeline("json_mode", "fast")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "fence-test"})

        assert isinstance(result["profile"], LeadProfile)
        assert result["profile"].name == "Alice Chen"
