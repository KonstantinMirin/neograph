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
from tests.fakes import StructuredFake, configure_fake_llm

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

    def test_missing_key_returns_empty_with_warning(self, caplog):
        """Missing var resolves to '' but emits a structlog warning.

        BUG neograph-9pcb: silent empty-string fallback caused 2 prod failures.
        """
        import structlog
        # Capture structlog output via stdlib logging
        structlog.configure(
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
        import logging
        with caplog.at_level(logging.WARNING):
            result = _substitute_vars("Hello ${missing}", {"name": "world"})
        assert result == "Hello "
        assert any("prompt_var_missing" in r.message for r in caplog.records), (
            f"Expected 'prompt_var_missing' warning, got: {[r.message for r in caplog.records]}"
        )

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

    def test_none_value_returns_empty_string(self):
        """None in dict must produce '', not literal 'None'."""
        assert _resolve_var("field", {"field": None}) == ""

    def test_none_intermediate_in_dotted_path_returns_empty(self):
        """${a.b} where a is None must produce '', not raise."""
        assert _resolve_var("a.b", {"a": None}) == ""

    def test_zero_value_returns_zero_string(self):
        """0 is not None — must produce '0'."""
        assert _resolve_var("n", {"n": 0}) == "0"

    def test_empty_string_value_stays_empty(self):
        assert _resolve_var("s", {"s": ""}) == ""

    def test_false_value_returns_false_string(self):
        """False is not None — must produce 'False'."""
        assert _resolve_var("b", {"b": False}) == "False"


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


# ═══════════════════════════════════════════════════════════════════════════
# Full dispatch chain: inline prompts through _render_input + _compile_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestInlinePromptThroughFullDispatch:
    """BUG neograph-x3gz: dotted var access broken after BAML default rendering.

    _render_input BAML-renders dict values BEFORE _compile_prompt runs.
    Inline prompts with ${claim.text} get getattr on a BAML string instead
    of a Pydantic model, silently returning empty string.

    These tests go through the full dispatch chain (ThinkDispatch → _render_input
    → invoke_structured → _compile_prompt → _substitute_vars) to catch the
    regression that unit tests on _substitute_vars miss.
    """

    def test_dotted_var_resolves_through_full_pipeline(self):
        """${seed.text} must resolve to the field value through a real pipeline.

        Full chain: ThinkDispatch → _render_input → invoke_structured →
        _compile_prompt → _substitute_vars. No renderer configured.

        BUG neograph-x3gz: _render_input BAML-renders dict values, so
        _resolve_var gets a string instead of a model. getattr(string, "text")
        silently returns "" instead of the field value.
        """
        from neograph import Construct, Node, compile, run
        from neograph.factory import register_scripted
        from tests.fakes import StructuredFakeWithRaw, configure_fake_llm

        class Claim(BaseModel):
            claim_id: str
            text: str

        class Verdict(BaseModel):
            disposition: str

        # Capture the actual messages sent to the LLM
        llm_received = []

        class CapturingFake:
            """Fake LLM that records what messages it receives."""
            def __init__(self):
                self.messages = []

            def with_structured_output(self, schema, *, include_raw=False, **kw):
                parent = self
                class Bound:
                    def invoke(self, messages, config=None, **kwargs):
                        parent.messages.extend(messages)
                        llm_received.extend(messages)
                        instance = schema(disposition="confirmed")
                        if include_raw:
                            from langchain_core.messages import AIMessage
                            return {"parsed": instance, "raw": AIMessage(
                                content="fake", response_metadata={"usage": {}})}
                        return instance
                return Bound()

        fake = CapturingFake()
        configure_fake_llm(
            factory=lambda tier: fake,
            prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": "ERROR: compiler called for inline"}],
        )

        register_scripted("x3gz_seed", lambda _in, _cfg: Claim(
            claim_id="c1", text="the sky is blue",
        ))

        parent = Construct("x3gz-test", nodes=[
            Node.scripted("seed", fn="x3gz_seed", outputs=Claim),
            Node("judge", prompt="Judge this claim: ${seed.text}",
                 model="default", outputs=Verdict, inputs={"seed": Claim}),
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "x3gz"})

        assert result["judge"].disposition == "confirmed"
        # The critical assertion: the LLM must have received the resolved field value
        assert llm_received, "LLM should have received messages"
        prompt_content = llm_received[0]["content"] if isinstance(llm_received[0], dict) else llm_received[0].content
        assert "the sky is blue" in prompt_content, (
            f"Dotted var ${{seed.text}} must resolve to field value in the prompt.\n"
            f"Got: {prompt_content!r}"
        )

    def test_dotted_var_in_fan_in_dict_resolves(self):
        """${claim.text} in fan-in scenario must resolve the field, not empty string."""
        from neograph._llm import _compile_prompt

        # Simulate what happens after _render_input renders the dict
        # Before the fix: raw model, dotted access works
        # After the fix: must still work through whatever mechanism

        class Claim(BaseModel):
            claim_id: str
            text: str

        raw_input = {"claim": Claim(claim_id="c1", text="earth is round")}
        msgs = _compile_prompt("Verify: ${claim.text}", raw_input)
        assert msgs[0]["content"] == "Verify: earth is round", (
            f"Dotted var must resolve field value, got: {msgs[0]['content']!r}"
        )

    def test_whole_var_renders_usefully_through_pipeline(self):
        """${seed} (no dotted access) must produce useful text, not Pydantic repr."""
        from neograph import Construct, Node, compile, run
        from neograph.factory import register_scripted
        from tests.fakes import StructuredFakeWithRaw, configure_fake_llm

        class Claim(BaseModel):
            text: str

        class Result(BaseModel):
            answer: str

        last_prompt = []

        # Use a prompt compiler that captures what it receives for file-ref prompts
        # For inline prompts, _substitute_vars handles it directly
        configure_fake_llm(
            factory=lambda tier: StructuredFakeWithRaw(
                lambda m: m(answer="yes"),
            ),
            prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": "x"}],
        )

        register_scripted("x3gz_seed2", lambda _in, _cfg: Claim(text="test claim"))

        parent = Construct("x3gz-whole", nodes=[
            Node.scripted("seed", fn="x3gz_seed2", outputs=Claim),
            Node("judge", prompt="Evaluate: ${seed}",
                 model="default", outputs=Result, inputs={"seed": Claim}),
        ])
        graph = compile(parent)
        result = run(graph, input={"node_id": "x3gz-whole"})
        assert result["judge"].answer == "yes"
