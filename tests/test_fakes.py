"""Tests for StructuredFakeWithRaw."""

from __future__ import annotations

from pydantic import BaseModel

from tests.fakes import (
    StructuredFakeWithRaw,
    configure_fake_llm,
)


class Items(BaseModel):
    items: list[str]


class TestStructuredFakeWithRaw:
    """Tests for the StructuredFakeWithRaw fake LLM."""

    def test_include_raw_true_returns_dict(self):
        fake = StructuredFakeWithRaw(lambda model: model(items=["a", "b"]))
        structured = fake.with_structured_output(Items, include_raw=True)
        result = structured.invoke([])

        assert isinstance(result, dict)
        assert "parsed" in result
        assert "raw" in result
        assert isinstance(result["parsed"], Items)
        assert result["parsed"].items == ["a", "b"]

    def test_include_raw_true_has_usage_metadata(self):
        fake = StructuredFakeWithRaw(lambda model: model(items=["x"]))
        structured = fake.with_structured_output(Items, include_raw=True)
        result = structured.invoke([])

        raw_msg = result["raw"]
        assert raw_msg.usage_metadata is not None
        assert raw_msg.usage_metadata["input_tokens"] == 10
        assert raw_msg.usage_metadata["output_tokens"] == 20
        assert raw_msg.usage_metadata["total_tokens"] == 30

    def test_include_raw_false_returns_plain_model(self):
        fake = StructuredFakeWithRaw(lambda model: model(items=["z"]))
        structured = fake.with_structured_output(Items, include_raw=False)
        result = structured.invoke([])

        assert isinstance(result, Items)
        assert result.items == ["z"]

    def test_default_no_include_raw_returns_plain_model(self):
        fake = StructuredFakeWithRaw(lambda model: model(items=["q"]))
        structured = fake.with_structured_output(Items)
        result = structured.invoke([])

        assert isinstance(result, Items)
        assert result.items == ["q"]

    def test_custom_usage_tokens(self):
        usage = {"prompt_tokens": 50, "completion_tokens": 100}
        fake = StructuredFakeWithRaw(lambda model: model(items=["c"]), usage=usage)
        structured = fake.with_structured_output(Items, include_raw=True)
        result = structured.invoke([])

        raw_msg = result["raw"]
        assert raw_msg.usage_metadata["input_tokens"] == 50
        assert raw_msg.usage_metadata["output_tokens"] == 100
        assert raw_msg.usage_metadata["total_tokens"] == 150

    def test_call_structured_extracts_usage(self):
        """Test the _call_structured code path with StructuredFakeWithRaw."""
        from langchain_core.runnables import RunnableConfig

        from neograph._llm import _call_structured

        fake = StructuredFakeWithRaw(lambda model: model(items=["done"]))
        config = RunnableConfig(configurable={})

        result, usage = _call_structured(fake, [], Items, "structured", config)

        assert isinstance(result, Items)
        assert result.items == ["done"]
        assert usage is not None
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 20

    def test_invoke_structured_returns_correct_model_when_fake_with_raw(self):
        """invoke_structured returns the parsed model when using StructuredFakeWithRaw."""
        from neograph._llm import invoke_structured

        fake = StructuredFakeWithRaw(
            lambda model: model(items=["end"]),
            usage={"prompt_tokens": 5, "completion_tokens": 15},
        )
        configure_fake_llm(lambda tier: fake)

        result = invoke_structured(
            model_tier="default",
            prompt_template="test",
            input_data={},
            output_model=Items,
            config={"configurable": {}},
        )

        assert isinstance(result, Items)
        assert result.items == ["end"]
