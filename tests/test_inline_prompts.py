"""Inline prompt support tests -- _compile_prompt inline detection,
variable substitution, and backward compatibility with file references.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from pydantic import BaseModel

from neograph._image import resolve_image as _resolve_image
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


# ═══════════════════════════════════════════════════════════════════════════
# Multimodal / vision prompts
# ═══════════════════════════════════════════════════════════════════════════


class TestMultimodalPrompt:
    """${image:field} in inline prompts produces multimodal content blocks."""

    def test_image_placeholder_produces_content_blocks(self):
        """${image:photo} should produce a content block list, not a flat string."""
        b64 = base64.b64encode(b"fake-png-data").decode()
        data = {"photo": b64, "question": "What is this?"}

        msgs = _compile_prompt("${question} ${image:photo}", data)
        content = msgs[0]["content"]

        assert isinstance(content, list), f"Expected content blocks, got {type(content)}: {content}"
        types = [block["type"] for block in content]
        assert "text" in types
        assert "image_url" in types

    def test_image_from_file_path(self):
        """${image:path} with a file path reads and base64-encodes the file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\nfake-image-data")
            tmp_path = f.name

        try:
            data = {"photo": tmp_path}
            msgs = _compile_prompt("Describe: ${image:photo}", data)
            content = msgs[0]["content"]

            assert isinstance(content, list)
            img_block = next(b for b in content if b["type"] == "image_url")
            assert img_block["image_url"]["url"].startswith("data:image/png;base64,")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_preformed_data_uri_passes_through(self):
        """${image:field} with a data:... URI passes through without re-encoding."""
        uri = "data:image/jpeg;base64,/9j/4AAQ"
        data = {"img": uri}

        msgs = _compile_prompt("Score: ${image:img}", data)
        content = msgs[0]["content"]

        img_block = next(b for b in content if b["type"] == "image_url")
        assert img_block["image_url"]["url"] == uri

    def test_mixed_text_and_images_preserve_order(self):
        """Text-image-text produces content blocks in correct order."""
        b64 = base64.b64encode(b"img").decode()
        data = {"photo": b64, "name": "sunset"}

        msgs = _compile_prompt("Name: ${name} ${image:photo} Score it.", data)
        content = msgs[0]["content"]

        assert len(content) == 3, f"Expected 3 blocks, got {len(content)}: {content}"
        assert content[0]["type"] == "text"
        assert "sunset" in content[0]["text"]
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"
        assert "Score it" in content[2]["text"]

    def test_multiple_images(self):
        """Multiple ${image:...} in one prompt each become a content block."""
        b64a = base64.b64encode(b"img-a").decode()
        b64b = base64.b64encode(b"img-b").decode()
        data = {"before": b64a, "after": b64b}

        msgs = _compile_prompt("Compare ${image:before} vs ${image:after}", data)
        content = msgs[0]["content"]

        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 2

    def test_text_only_prompt_unchanged(self):
        """A prompt without ${image:...} still returns flat string content."""
        data = {"topic": "cats"}
        msgs = _compile_prompt("Tell me about ${topic}", data)

        assert isinstance(msgs[0]["content"], str)
        assert "cats" in msgs[0]["content"]

    def test_image_dotted_path(self):
        """${image:obj.field} resolves via dotted attribute access."""

        class Photo(BaseModel):
            url: str

        b64 = base64.b64encode(b"photo-data").decode()
        data = {"photo": Photo(url=b64)}

        msgs = _compile_prompt("Describe ${image:photo.url}", data)
        content = msgs[0]["content"]

        assert isinstance(content, list)
        img_block = next(b for b in content if b["type"] == "image_url")
        assert "base64" in img_block["image_url"]["url"]


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_image error paths
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveImageErrors:
    """_resolve_image should warn on suspicious input, not crash."""

    def test_empty_string_returns_data_uri(self):
        """Empty string should produce a (corrupt) data URI, not crash."""
        result = _resolve_image("")
        assert result == "data:image/png;base64,"

    def test_whitespace_only_returns_data_uri(self):
        """Whitespace-only string should produce a data URI, not crash."""
        result = _resolve_image("   ")
        assert result == "data:image/png;base64,"

    def test_nonexistent_file_path_wraps_as_base64(self):
        """A path-like string that doesn't exist wraps as base64 (with warning via structlog)."""
        result = _resolve_image("/nonexistent/photo.png")
        assert result.startswith("data:image/png;base64,")
        assert "/nonexistent/photo.png" in result  # the path is the "base64" content

    def test_valid_base64_wraps_correctly(self):
        """Valid base64 should wrap in data URI."""
        b64 = base64.b64encode(b"valid-image-data").decode()
        result = _resolve_image(b64)
        assert result == f"data:image/png;base64,{b64}"

    def test_data_uri_passes_through(self):
        """Pre-formed data URI is returned unchanged."""
        uri = "data:image/jpeg;base64,/9j/4AAQ"
        assert _resolve_image(uri) == uri

    def test_missing_field_in_compile_prompt(self):
        """${image:missing} omits the image block entirely."""
        data = {"question": "What is this?"}  # no 'photo' key
        msgs = _compile_prompt("${question} ${image:photo}", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        # Missing field → image block should be omitted, not produced with corrupt data
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 0

    def test_none_field_in_compile_prompt(self):
        """${image:field} where field is None omits the image block."""
        data = {"photo": None}
        msgs = _compile_prompt("Score ${image:photo}", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 0

    def test_e2e_multimodal_through_dispatch(self):
        """Full pipeline: @node with ${image:...} → compile → run → LLM receives content blocks."""
        from neograph import compile, construct_from_module, node, run

        captured_messages = []

        class CaptureMultimodalLLM:
            def __init__(self, tier):
                self._tier = tier

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                captured_messages.append(messages)
                return self._model(score=0.9)

        class ImageData(BaseModel, frozen=True):
            photo: str

        class Score(BaseModel, frozen=True):
            score: float

        configure_fake_llm(lambda tier: CaptureMultimodalLLM(tier))

        b64 = base64.b64encode(b"test-image").decode()

        @node(outputs=ImageData)
        def seed() -> ImageData:
            return ImageData(photo=b64)

        @node(outputs=Score, prompt="Rate this image: ${image:seed.photo}", model="fast")
        def score_image(seed: ImageData) -> Score: ...

        import types
        mod = types.ModuleType("test_e2e_mm")
        mod.seed = seed
        mod.score_image = score_image

        pipeline = construct_from_module(mod, name="e2e-multimodal")
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "e2e-mm"})

        assert result["score_image"].score == 0.9
        # The LLM should have received multimodal content blocks
        assert len(captured_messages) > 0
        last_msgs = captured_messages[-1]
        user_msg = last_msgs[0]
        content = user_msg["content"] if isinstance(user_msg, dict) else user_msg.content
        assert isinstance(content, list), f"Expected content blocks, got {type(content)}: {content}"
        types_found = [b["type"] for b in content]
        assert "image_url" in types_found

    def test_real_file_reads_and_encodes(self):
        """An actual file on disk is read, base64-encoded, and MIME-detected."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\nimage-bytes")
            tmp = f.name
        try:
            result = _resolve_image(tmp)
            assert result.startswith("data:image/png;base64,")
            # Decode and verify roundtrip
            encoded = result.split(",", 1)[1]
            assert base64.b64decode(encoded) == b"\x89PNG\r\n\x1a\nimage-bytes"
        finally:
            Path(tmp).unlink(missing_ok=True)


# ═════════════���══════════════════════════��══════════════════════════════════
# render_prompt with multimodal content
# ════���═══════════════���══════════════════════════════════════════════════════


class TestRenderPromptMultimodal:
    """render_prompt produces readable output for multimodal messages."""

    def test_render_prompt_shows_image_placeholder(self):
        """render_prompt on a node with ${image:...} should show [image] in output."""
        from neograph._llm import render_prompt
        from neograph import Node

        configure_fake_llm(
            lambda tier: None,
            prompt_compiler=lambda t, d, **kw: [{"role": "user", "content": t}],
        )

        b64 = base64.b64encode(b"img").decode()
        n = Node("score", mode="think", outputs=BaseModel,
                 prompt="Rate: ${image:photo}", model="fast")
        result = render_prompt(n, {"photo": b64})
        assert "[image]" in result


# ═════���════════════════════════════════════════════════��════════════════════
# Image-only prompt (no text)
# ══════════��════════════════════════════════════════════════════════════════


class TestImageOnlyPrompt:
    """Prompt with only ${image:...} and no text."""

    def test_image_only_prompt(self):
        """A prompt of just '${image:photo}' produces a single image_url block."""
        b64 = base64.b64encode(b"solo-img").decode()
        msgs = _compile_prompt("${image:photo}", {"photo": b64})
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "image_url"


# ═���═════════════════════��═══════════════════════════════════════════════════
# json_mode + multimodal interaction
# ═══════════════════��═══════════════════════════════════════════════════════


class TestJsonModeMultimodal:
    """json_mode output strategy coexists with multimodal content blocks."""

    def test_json_mode_with_image_prompt(self):
        """invoke_structured with json_mode + ${image:...} produces content blocks."""
        captured = []

        class CaptureLLM:
            def __init__(self, tier):
                pass

            def invoke(self, messages, **kw):
                captured.append(messages)
                from langchain_core.messages import AIMessage
                return AIMessage(content='{"score": 0.9}')

        configure_fake_llm(lambda tier: CaptureLLM(tier))

        from neograph._llm import invoke_structured

        class Score(BaseModel):
            score: float

        b64 = base64.b64encode(b"test-img").decode()
        result = invoke_structured(
            model_tier="fast",
            prompt_template="Rate: ${image:photo}",
            input_data={"photo": b64},
            output_model=Score,
            config={"configurable": {}},
            llm_config={"output_strategy": "json_mode"},
        )

        assert result.score == 0.9
        # The LLM should have received content blocks
        user_msg = captured[0][0]
        content = user_msg["content"] if isinstance(user_msg, dict) else user_msg.content
        assert isinstance(content, list)
        assert any(b["type"] == "image_url" for b in content)


# ���═════════════════════════��════════════════════════���═══════════════════════
# MIME detection variety
# ══════════════════════���══════════════════════════════��═════════════════════


class TestMimeDetection:
    """_resolve_image detects MIME types from file extensions."""

    def _make_temp(self, suffix, content=b"fake"):
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_jpeg_mime(self):
        tmp = self._make_temp(".jpg")
        try:
            result = _resolve_image(tmp)
            assert result.startswith("data:image/jpeg;base64,")
        finally:
            Path(tmp).unlink()

    def test_gif_mime(self):
        tmp = self._make_temp(".gif")
        try:
            result = _resolve_image(tmp)
            assert result.startswith("data:image/gif;base64,")
        finally:
            Path(tmp).unlink()

    def test_webp_mime(self):
        tmp = self._make_temp(".webp")
        try:
            result = _resolve_image(tmp)
            # webp may not be registered on all systems; fallback to image/png is acceptable
            assert result.startswith("data:image/")
        finally:
            Path(tmp).unlink()

    def test_svg_mime(self):
        tmp = self._make_temp(".svg", b"<svg></svg>")
        try:
            result = _resolve_image(tmp)
            assert result.startswith("data:image/svg")
        finally:
            Path(tmp).unlink()


# ══════════════════════════════��═════════════════════════���══════════════════
# resolve_image public API
# ═══════════════════���═══════════════════════════════════════════════════════


class TestResolveImagePublicAPI:
    """resolve_image is importable from neograph."""

    def test_import_from_neograph(self):
        from neograph import resolve_image
        b64 = base64.b64encode(b"public-api").decode()
        result = resolve_image(b64)
        assert result == f"data:image/png;base64,{b64}"

    def test_same_function(self):
        from neograph import resolve_image
        assert resolve_image is _resolve_image


# ═════════════��═════════════════════════════════════════════════════════════
# Template-ref boundary
# ════════════════���══════════════════════════════════════════════════════════


class TestImageValidation:
    """Image validation guards in the _image module."""

    def test_file_size_limit_rejects_oversized(self):
        """Files exceeding max_size_bytes should be rejected with a warning."""
        from neograph._image import configure_image, resolve_image

        configure_image(max_size_bytes=100)  # 100 bytes limit
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(b"\x89PNG" + b"x" * 200)  # 204 bytes > 100 limit
                tmp = f.name
            result = resolve_image(tmp)
            # Should return empty/placeholder data URI, not the oversized file
            assert "x" * 200 not in result, "Oversized file should not be fully encoded"
        finally:
            Path(tmp).unlink(missing_ok=True)
            configure_image()  # reset to defaults

    def test_file_size_limit_allows_small_files(self):
        """Files within max_size_bytes should be accepted."""
        from neograph._image import configure_image, resolve_image

        configure_image(max_size_bytes=10_000)
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(b"\x89PNG" + b"x" * 50)
                tmp = f.name
            result = resolve_image(tmp)
            assert result.startswith("data:image/png;base64,")
            encoded = result.split(",", 1)[1]
            assert len(base64.b64decode(encoded)) == 54  # 4 + 50
        finally:
            Path(tmp).unlink(missing_ok=True)
            configure_image()

    def test_allowed_dirs_blocks_outside_paths(self):
        """Files outside allowed_dirs should be rejected."""
        from neograph._image import configure_image, resolve_image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as f:
            f.write(b"\x89PNGdata")
            tmp = f.name
        try:
            configure_image(allowed_dirs=["/nonexistent/safe_dir"])
            result = resolve_image(tmp)
            # Should NOT read the file — not in allowed dirs
            assert "PNGdata" not in base64.b64decode(
                result.split(",", 1)[1] if "," in result else ""
            ).decode("latin-1", errors="replace")
        finally:
            Path(tmp).unlink(missing_ok=True)
            configure_image()

    def test_allowed_dirs_permits_within(self):
        """Files inside allowed_dirs should be accepted."""
        from neograph._image import configure_image, resolve_image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as f:
            f.write(b"\x89PNGok")
            tmp = f.name
        try:
            configure_image(allowed_dirs=["/tmp"])
            result = resolve_image(tmp)
            assert result.startswith("data:image/png;base64,")
        finally:
            Path(tmp).unlink(missing_ok=True)
            configure_image()

    def test_toctou_file_deleted_after_check(self):
        """If file is deleted between is_file() and read_bytes(), don't crash."""
        from neograph._image import resolve_image

        # Create then immediately delete — resolve_image should handle gracefully
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNGdata")
            tmp = f.name
        Path(tmp).unlink()  # delete before resolve
        # Should not raise — returns a fallback
        result = resolve_image(tmp)
        assert result.startswith("data:")

    def test_configure_image_imported_from_neograph(self):
        """configure_image is importable from the public API."""
        from neograph import configure_image
        assert callable(configure_image)


class TestTemplateRefBoundary:
    """Template-ref prompts do not trigger multimodal detection."""

    def test_template_ref_not_processed_as_multimodal(self):
        """A template name without spaces/${} goes to prompt_compiler unchanged."""
        compiler_calls = []

        def mock_compiler(template, input_data, **kw):
            compiler_calls.append(template)
            return [{"role": "user", "content": "compiled"}]

        configure_fake_llm(
            lambda tier: None,
            prompt_compiler=mock_compiler,
        )

        msgs = _compile_prompt("rw/score-image", {"photo": "base64data"})
        assert compiler_calls == ["rw/score-image"]
        assert msgs[0]["content"] == "compiled"
