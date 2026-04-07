"""Inline prompt support tests -- _compile_prompt inline detection,
variable substitution, and backward compatibility with file references.
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph._llm import (
    _compile_prompt,
    _is_inline_prompt,
    _resolve_var,
    _substitute_vars,
)
from tests.fakes import configure_fake_llm, StructuredFake


# ═══════════════════════════════════════════════════════════════════════════
# Detection heuristic
# ═══════════════════════════════════════════════════════════════════════════


class TestIsInlinePrompt:
    """_is_inline_prompt detects inline text vs file references."""

    def test_file_reference_without_space(self):
        assert _is_inline_prompt("rw/summarize") is False

    def test_file_reference_simple_name(self):
        assert _is_inline_prompt("summarize") is False

    def test_inline_with_space(self):
        assert _is_inline_prompt("Summarize the following claims") is True

    def test_inline_with_substitution_marker(self):
        assert _is_inline_prompt("Analyze${topic}") is True

    def test_inline_with_both_space_and_substitution(self):
        assert _is_inline_prompt("Summarize ${claims} for ${topic}") is True

    def test_empty_string_is_file_ref(self):
        assert _is_inline_prompt("") is False


# ═══════════════════════════════════════════════════════════════════════════
# Variable substitution
# ═══════════════════════════════════════════════════════════════════════════


class TestSubstituteVars:
    """_substitute_vars replaces ${} placeholders in inline text."""

    def test_simple_dict_lookup(self):
        result = _substitute_vars("Hello ${name}", {"name": "world"})
        assert result == "Hello world"

    def test_multiple_vars(self):
        result = _substitute_vars(
            "${greeting} ${name}!", {"greeting": "Hi", "name": "there"}
        )
        assert result == "Hi there!"

    def test_no_vars_passthrough(self):
        result = _substitute_vars("No variables here", {"key": "val"})
        assert result == "No variables here"

    def test_missing_key_returns_empty(self):
        result = _substitute_vars("Hello ${missing}", {"name": "world"})
        assert result == "Hello "

    def test_single_value_input(self):
        result = _substitute_vars("Value is ${data}", "hello")
        assert result == "Value is hello"

    def test_dotted_access_on_dict(self):
        class Inner(BaseModel):
            text: str

        result = _substitute_vars("Say ${claim.text}", {"claim": Inner(text="hi")})
        assert result == "Say hi"

    def test_dotted_access_on_single_value(self):
        class Outer(BaseModel):
            name: str

        obj = Outer(name="test")
        result = _substitute_vars("Name is ${x.name}", obj)
        assert result == "Name is test"

    def test_nonexistent_dotted_attr_returns_empty(self):
        class Obj(BaseModel):
            value: int

        result = _substitute_vars("Get ${x.missing}", {"x": Obj(value=1)})
        assert result == "Get "

    def test_value_coerced_to_str(self):
        result = _substitute_vars("Count: ${n}", {"n": 42})
        assert result == "Count: 42"


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_var unit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveVar:
    """_resolve_var resolves a single variable path against input_data."""

    def test_plain_key_from_dict(self):
        assert _resolve_var("name", {"name": "Alice"}) == "Alice"

    def test_dotted_key_from_dict(self):
        class Info(BaseModel):
            age: int

        assert _resolve_var("info.age", {"info": Info(age=30)}) == "30"

    def test_single_value_no_dots(self):
        assert _resolve_var("x", "hello") == "hello"

    def test_single_value_with_dots(self):
        class Val(BaseModel):
            field: str

        assert _resolve_var("x.field", Val(field="ok")) == "ok"


# ═══════════════════════════════════════════════════════════════════════════
# _compile_prompt integration: inline vs file ref
# ═══════════════════════════════════════════════════════════════════════════


class TestCompilePromptInline:
    """_compile_prompt handles inline text and delegates file refs."""

    def test_inline_prompt_returns_user_message(self):
        msgs = _compile_prompt("Summarize these ${topic} claims", {"topic": "science"})
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Summarize these science claims"

    def test_inline_prompt_without_vars(self):
        msgs = _compile_prompt("Just a plain instruction", {})
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Just a plain instruction"

    def test_file_ref_delegates_to_prompt_compiler(self):
        """File reference (no space, no ${}) goes through the consumer compiler."""
        captured = []

        def fake_compiler(template, data, **kw):
            captured.append(template)
            return [{"role": "system", "content": "compiled"}]

        configure_fake_llm(
            factory=lambda tier: StructuredFake(lambda m: m()),
            prompt_compiler=fake_compiler,
        )
        msgs = _compile_prompt("rw/summarize", {"data": "x"})
        assert captured == ["rw/summarize"]
        assert msgs[0]["content"] == "compiled"

    def test_inline_prompt_skips_prompt_compiler(self):
        """Inline prompt does NOT call the consumer prompt compiler."""
        called = []

        def spy_compiler(template, data, **kw):
            called.append(True)
            return [{"role": "user", "content": "from compiler"}]

        configure_fake_llm(
            factory=lambda tier: StructuredFake(lambda m: m()),
            prompt_compiler=spy_compiler,
        )
        msgs = _compile_prompt("Inline text here", {})
        assert called == []  # compiler was never called
        assert msgs[0]["content"] == "Inline text here"

    def test_inline_with_substitution_marker_no_space(self):
        """${} without spaces still detected as inline."""
        msgs = _compile_prompt("${greeting}", {"greeting": "Hello"})
        assert msgs[0]["content"] == "Hello"
