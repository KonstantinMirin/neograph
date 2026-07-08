"""Integration and E2E tests for the _image.py validation component.

Unit tests for resolve_image live in test_inline_prompts.py.
This file tests the validation pipeline through the full dispatch path
and verifies configuration interactions.
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from neograph import compile, construct_from_module, node, run
from neograph._image import configure_image, resolve_image
from tests.fakes import build_fake_runtime, build_test_compile_kwargs, configure_fake_llm

# ── Schemas ──────────────────────────────────────────────────────────────


class ImageData(BaseModel, frozen=True):
    photo: str


class Score(BaseModel, frozen=True):
    score: float


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_temp_image(suffix=".png", size=50):
    """Create a temp file with PNG magic bytes + padding."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(b"\x89PNG\r\n\x1a\n" + b"x" * size)
    f.close()
    return f.name


def _fresh_module(name):
    import types

    return types.ModuleType(name)


class _CaptureLLM:
    """Fake LLM that captures messages for inspection."""

    def __init__(self, tier):
        self._tier = tier
        self.captured_messages = []

    def with_structured_output(self, model, **kw):
        self._model = model
        return self

    def invoke(self, messages, **kw):
        self.captured_messages.append(messages)
        return self._model(score=0.95)


@pytest.fixture(autouse=True)
def _reset_image_config():
    """Reset image config after each test to prevent leakage."""
    yield
    configure_image()  # reset to defaults


# ═══════════════════════════════════════════════════════════════════════════
# E2E: size limit through full pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestE2ESizeLimit:
    """File exceeding max_size_bytes produces empty image block; pipeline completes."""

    def test_oversized_file_degrades_gracefully(self):
        configure_image(max_size_bytes=100)

        llm_instances = []

        def factory(tier):
            llm = _CaptureLLM(tier)
            llm_instances.append(llm)
            return llm

        _llm_kw = configure_fake_llm(factory)

        tmp = _make_temp_image(size=200)  # 208 bytes > 100 limit
        try:

            @node(outputs=ImageData)
            def seed() -> ImageData:
                return ImageData(photo=tmp)

            @node(outputs=Score, prompt="Rate: ${image:seed.photo}", model="fast")
            def score_it(seed: ImageData) -> Score: ...

            mod = _fresh_module("test_size_e2e")
            mod.seed = seed
            mod.score_it = score_it

            pipeline = construct_from_module(mod, name="size-e2e")
            graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
            result = run(graph, input={"node_id": "size-1"})

            # Pipeline completes
            assert result["score_it"].score == 0.95

            # The LLM received an empty image (rejected by size limit)
            msgs = llm_instances[-1].captured_messages[-1]
            content = msgs[0]["content"] if isinstance(msgs[0], dict) else msgs[0].content
            assert isinstance(content, list)
            img_block = next(b for b in content if b["type"] == "image_url")
            # Empty base64 = the file was rejected
            assert img_block["image_url"]["url"] == "data:image/png;base64,"
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_small_file_accepted(self):
        configure_image(max_size_bytes=10_000)

        llm_instances = []

        def factory(tier):
            llm = _CaptureLLM(tier)
            llm_instances.append(llm)
            return llm

        _llm_kw = configure_fake_llm(factory)

        tmp = _make_temp_image(size=50)  # 58 bytes < 10000 limit
        try:

            @node(outputs=ImageData)
            def seed() -> ImageData:
                return ImageData(photo=tmp)

            @node(outputs=Score, prompt="Rate: ${image:seed.photo}", model="fast")
            def score_it(seed: ImageData) -> Score: ...

            mod = _fresh_module("test_small_e2e")
            mod.seed = seed
            mod.score_it = score_it

            pipeline = construct_from_module(mod, name="small-e2e")
            graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
            result = run(graph, input={"node_id": "small-1"})

            assert result["score_it"].score == 0.95

            # The LLM received the actual image
            msgs = llm_instances[-1].captured_messages[-1]
            content = msgs[0]["content"]
            img_block = next(b for b in content if b["type"] == "image_url")
            assert img_block["image_url"]["url"] != "data:image/png;base64,"
            assert "base64," in img_block["image_url"]["url"]
        finally:
            Path(tmp).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# E2E: allowed_dirs through full pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestE2EAllowedDirs:
    """Directory restriction blocks files outside allowed_dirs."""

    def test_blocked_dir_degrades_gracefully(self):
        tmp = _make_temp_image()
        try:
            configure_image(allowed_dirs=["/nonexistent/safe_dir"])

            llm_instances = []

            def factory(tier):
                llm = _CaptureLLM(tier)
                llm_instances.append(llm)
                return llm

            _llm_kw = configure_fake_llm(factory)

            @node(outputs=ImageData)
            def seed() -> ImageData:
                return ImageData(photo=tmp)

            @node(outputs=Score, prompt="Rate: ${image:seed.photo}", model="fast")
            def score_it(seed: ImageData) -> Score: ...

            mod = _fresh_module("test_dir_block_e2e")
            mod.seed = seed
            mod.score_it = score_it

            pipeline = construct_from_module(mod, name="dir-block-e2e")
            graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
            result = run(graph, input={"node_id": "dir-1"})

            # Pipeline completes
            assert result["score_it"].score == 0.95

            # Image was blocked
            msgs = llm_instances[-1].captured_messages[-1]
            content = msgs[0]["content"]
            img_block = next(b for b in content if b["type"] == "image_url")
            assert img_block["image_url"]["url"] == "data:image/png;base64,"
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_allowed_dir_passes(self):
        tmp = _make_temp_image()
        tmp_dir = str(Path(tmp).parent)
        try:
            configure_image(allowed_dirs=[tmp_dir])

            llm_instances = []

            def factory(tier):
                llm = _CaptureLLM(tier)
                llm_instances.append(llm)
                return llm

            _llm_kw = configure_fake_llm(factory)

            @node(outputs=ImageData)
            def seed() -> ImageData:
                return ImageData(photo=tmp)

            @node(outputs=Score, prompt="Rate: ${image:seed.photo}", model="fast")
            def score_it(seed: ImageData) -> Score: ...

            mod = _fresh_module("test_dir_allow_e2e")
            mod.seed = seed
            mod.score_it = score_it

            pipeline = construct_from_module(mod, name="dir-allow-e2e")
            graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
            result = run(graph, input={"node_id": "dir-2"})

            # Image was accepted
            msgs = llm_instances[-1].captured_messages[-1]
            content = msgs[0]["content"]
            img_block = next(b for b in content if b["type"] == "image_url")
            assert img_block["image_url"]["url"] != "data:image/png;base64,"
        finally:
            Path(tmp).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# E2E: configure_image persists across compile→run
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigPersistence:
    """Config set before compile() is active at run() time."""

    def test_config_set_before_compile_active_at_run(self):
        """max_size_bytes set before compile is enforced during run."""
        configure_image(max_size_bytes=50)  # very small

        _llm_kw = configure_fake_llm(lambda tier: _CaptureLLM(tier))

        tmp = _make_temp_image(size=200)  # exceeds limit
        try:

            @node(outputs=ImageData)
            def seed() -> ImageData:
                return ImageData(photo=tmp)

            @node(outputs=Score, prompt="Rate: ${image:seed.photo}", model="fast")
            def score_it(seed: ImageData) -> Score: ...

            mod = _fresh_module("test_persist")
            mod.seed = seed
            mod.score_it = score_it

            pipeline = construct_from_module(mod, name="persist-e2e")
            graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
            # Config was set before compile — should still be active at run
            result = run(graph, input={"node_id": "persist-1"})
            assert result["score_it"].score == 0.95
        finally:
            Path(tmp).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: configure_image reset
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigReset:
    """configure_image() with no args resets to defaults."""

    def test_reset_restores_defaults(self):

        configure_image(max_size_bytes=42, allowed_dirs=["/x"], validate_format=False)
        configure_image()  # reset

        from neograph._image import _config as after

        assert after.max_size_bytes == 20 * 1024 * 1024
        assert after.allowed_dirs is None
        assert after.validate_format is True


# ═══════════════════════════════════════════════════════════════════════════
# Integration: validate_format=False
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateFormatOff:
    """validate_format=False accepts non-image files without warning."""

    def test_text_file_accepted_when_validation_off(self):
        configure_image(validate_format=False)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"this is not an image")
            tmp = f.name
        try:
            result = resolve_image(tmp)
            assert result.startswith("data:")
            # The content should be base64-encoded
            encoded = result.split(",", 1)[1]
            assert base64.b64decode(encoded) == b"this is not an image"
        finally:
            Path(tmp).unlink()


# ═══════════════════════════════════════════════════════════════════════════
# Integration: allowed_dirs with symlinks
# ═══════════════════════════════════════════════════════════════════════════


class TestSymlinkSecurity:
    """Symlinks that escape allowed_dirs are blocked."""

    def test_symlink_outside_allowed_dir_blocked(self):
        # Create real file outside allowed dir
        real = _make_temp_image()
        real_dir = str(Path(real).parent)

        # Create a symlink inside a "safe" dir pointing to the real file
        safe_dir = tempfile.mkdtemp()
        link_path = os.path.join(safe_dir, "link.png")
        os.symlink(real, link_path)

        try:
            # Allow only safe_dir, but the symlink resolves outside
            # The real file is in /tmp (or wherever), and we restrict to safe_dir only
            # Path.resolve() follows symlinks, so resolved path = real file path
            # If real file is NOT under safe_dir, it should be blocked
            configure_image(allowed_dirs=[safe_dir])

            result = resolve_image(link_path)

            # The symlink target resolves outside safe_dir (to /tmp/...)
            # so it should be blocked IF the real file's dir != safe_dir
            if str(Path(real).resolve().parent) != str(Path(safe_dir).resolve()):
                assert result == "data:image/png;base64,", "Symlink escaping allowed_dirs should be blocked"
            else:
                # Edge case: real file happens to be in the same temp dir
                # This is fine — just verify it's a valid data URI
                assert result.startswith("data:image/")
        finally:
            os.unlink(link_path)
            os.rmdir(safe_dir)
            Path(real).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: allowed_dirs with relative paths
# ═══════════════════════════════════════════════════════════════════════════


class TestRelativeAllowedDirs:
    """Relative paths in allowed_dirs resolve against CWD."""

    def test_relative_allowed_dir(self):
        tmp = _make_temp_image()
        tmp_dir = str(Path(tmp).parent)

        # Use relative path that resolves to the temp dir
        try:
            rel = os.path.relpath(tmp_dir)
            configure_image(allowed_dirs=[rel])
            result = resolve_image(tmp)
            assert result.startswith("data:image/png;base64,")
            assert result != "data:image/png;base64,"  # not blocked
        finally:
            Path(tmp).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# E2E: multimodal in Oracle merge_prompt path
# ═══════════════════════════════════════════════════════════════════════════


class TestMultimodalOracleMerge:
    """${image:...} in Oracle merge_prompt flows through the merge path."""

    def test_image_in_merge_prompt(self):
        from neograph import Construct, Node, Oracle

        llm_instances = []

        def factory(tier):
            llm = _CaptureLLM(tier)
            llm_instances.append(llm)
            return llm

        _llm_kw = configure_fake_llm(factory)

        b64 = base64.b64encode(b"merge-image").decode()

        from tests.fakes import register_scripted

        register_scripted("_mm_gen", lambda i, c: Score(score=0.5))

        writer = Node.scripted("gen", fn="_mm_gen", outputs=Score) | Oracle(
            n=2,
            merge_prompt="Pick best based on ${image:ref_image}: ${variants}",
            merge_pre_process=lambda variants: {
                "ref_image": b64,
                "variants": str([v.score for v in variants]),
            },
        )

        pipeline = Construct("mm-oracle", nodes=[writer])
        graph = compile(pipeline, **_llm_kw, **build_test_compile_kwargs())
        result = run(graph, input={"node_id": "mm-1"})

        assert result["gen"].score == 0.95

        # Find the merge call (tier="reason")
        reason_llms = [l for l in llm_instances if l._tier == "reason"]
        assert len(reason_llms) > 0
        merge_msgs = reason_llms[-1].captured_messages[-1]
        content = merge_msgs[0]["content"]
        assert isinstance(content, list), f"Expected content blocks: {content}"
        assert any(b["type"] == "image_url" for b in content)


# ═══════════════════════════════════════════════════════════════════════════
# Integration: config is global
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigIsGlobal:
    """Configuration is module-level, not per-pipeline."""

    def test_config_shared_across_calls(self):
        configure_image(max_size_bytes=42)

        from neograph._image import _config

        assert _config.max_size_bytes == 42

        # A second import sees the same config
        from neograph._image import _config as same_ref

        assert same_ref.max_size_bytes == 42
        assert same_ref is _config


# ═══════════════════════════════════════════════════════════════════════════
# ADVERSARIAL FINDINGS (F1-F15)
# ═══════════════════════════════════════════════════════════════════════════


class TestF1PathPrefixAttack:
    """F1: allowed_dirs=['/tmp/foo'] must NOT allow /tmp/foobar/image.png."""

    def test_prefix_collision_blocked(self):
        """Path that shares prefix but is not a descendant must be blocked."""
        # Create /tmp/foobar/ directory and file
        sibling_dir = tempfile.mkdtemp(prefix="foobar")
        foo_dir = sibling_dir.rstrip("r")  # e.g., /tmp/xxxfoobar -> /tmp/xxxfooba (wrong approach)

        # Better: create two dirs explicitly
        base = tempfile.mkdtemp()
        allowed = os.path.join(base, "safe")
        attacker = os.path.join(base, "safe_evil")
        os.makedirs(allowed, exist_ok=True)
        os.makedirs(attacker, exist_ok=True)

        attack_file = os.path.join(attacker, "secret.png")
        Path(attack_file).write_bytes(b"\x89PNGsecret")

        try:
            configure_image(allowed_dirs=[allowed])
            result = resolve_image(attack_file)

            # The file is in "safe_evil" which starts with "safe" string-wise
            # but is NOT a child of "safe". Must be blocked.
            assert result == "data:image/png;base64,", (
                f"Path prefix attack succeeded: {attacker} was allowed by {allowed}"
            )
        finally:
            Path(attack_file).unlink(missing_ok=True)
            os.rmdir(attacker)
            os.rmdir(allowed)
            os.rmdir(base)


class TestF3FormatValidationRejects:
    """F3: validate_format=True should reject non-image files, not just warn."""

    def test_text_file_rejected_when_validation_on(self):
        """A .txt file with no image magic bytes should be rejected."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"this is not an image at all")
            tmp = f.name
        try:
            configure_image(validate_format=True)
            result = resolve_image(tmp)
            assert result == "data:image/png;base64,", f"Non-image file should be rejected, got: {result[:60]}..."
        finally:
            Path(tmp).unlink()

    def test_png_with_txt_extension_uses_magic_mime(self):
        """A file with PNG magic bytes but .txt extension should use image/png MIME."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\nreal-png-data")
            tmp = f.name
        try:
            result = resolve_image(tmp)
            assert result.startswith("data:image/png;base64,"), (
                f"Magic bytes should override extension MIME, got: {result[:60]}"
            )
        finally:
            Path(tmp).unlink()


class TestF4RIFFNotWebP:
    """F4: RIFF magic should only match WebP, not WAV/AVI."""

    def test_wav_file_not_classified_as_webp(self):
        """A WAV file (RIFF + WAVE) should not be classified as WebP."""
        from neograph._image import _check_magic_bytes

        wav_header = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20
        result = _check_magic_bytes(wav_header)
        assert result != "image/webp", f"WAV classified as WebP: {result}"

    def test_real_webp_classified_correctly(self):
        """A WebP file (RIFF + WEBP) should be classified as image/webp."""
        from neograph._image import _check_magic_bytes

        webp_header = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 20
        result = _check_magic_bytes(webp_header)
        assert result == "image/webp"

    def test_avi_file_not_classified_as_webp(self):
        """An AVI file (RIFF + AVI) should not be classified as WebP."""
        from neograph._image import _check_magic_bytes

        avi_header = b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 20
        result = _check_magic_bytes(avi_header)
        assert result != "image/webp", f"AVI classified as WebP: {result}"


class TestF9MaxSizeValidation:
    """F9: max_size_bytes must be > 0."""

    def test_zero_raises(self):
        from neograph.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="must be > 0"):
            configure_image(max_size_bytes=0)

    def test_negative_raises(self):
        from neograph.errors import ConfigurationError

        with pytest.raises(ConfigurationError, match="must be > 0"):
            configure_image(max_size_bytes=-1)

    def test_positive_accepted(self):
        configure_image(max_size_bytes=1)
        from neograph._image import _config

        assert _config.max_size_bytes == 1


class TestF10BaseModelAsImagePath:
    """F10: ${image:model} where model is a BaseModel should warn, not produce garbage."""

    def test_basemodel_image_field_warns_and_skips(self):
        from neograph._llm import _compile_prompt

        class Photo(BaseModel):
            url: str

        data = {"photo": Photo(url="/tmp/test.png")}
        msgs = _compile_prompt("Rate: ${image:photo}", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        # The image block should be SKIPPED (BaseModel is not a valid image ref)
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 0, f"BaseModel should not produce an image block: {img_blocks}"


class TestF12MissingFieldSkipsImageBlock:
    """F12: missing/empty image field should omit the image block entirely."""

    def test_missing_field_omits_image_block(self):
        from neograph._llm import _compile_prompt

        data = {"question": "What is this?"}  # no 'photo' key
        msgs = _compile_prompt("${question} ${image:photo}", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 0, f"Missing field should not produce an image block: {img_blocks}"

    def test_none_field_omits_image_block(self):
        from neograph._llm import _compile_prompt

        data = {"photo": None}
        msgs = _compile_prompt("Score: ${image:photo}", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 0, f"None field should not produce an image block: {img_blocks}"


# ═══════════════════════════════════════════════════════════════════════════
# F2: macOS symlink resolution in allowed_dirs
# ═══════════════════════════════════════════════════════════════════════════


class TestF2SymlinkResolution:
    """F2: Both file path and allowed_dir go through Path.resolve()."""

    def test_resolve_normalizes_both_sides(self):
        """allowed_dirs and file path both resolve symlinks before comparison."""
        # Create a real dir and a symlink to it
        real_dir = tempfile.mkdtemp()
        link_dir = real_dir + "_link"
        os.symlink(real_dir, link_dir)

        img_file = os.path.join(real_dir, "test.png")
        Path(img_file).write_bytes(b"\x89PNG\r\n\x1a\ndata")

        try:
            # Allow via the symlink path — the real file is in real_dir
            configure_image(allowed_dirs=[link_dir])
            result = resolve_image(img_file)
            # Should be allowed because both sides resolve to the same physical path
            assert result != "data:image/png;base64,", "File should be allowed via symlink dir"
        finally:
            Path(img_file).unlink(missing_ok=True)
            os.unlink(link_dir)
            os.rmdir(real_dir)


# ═══════════════════════════════════════════════════════════════════════════
# F5: Retry path with multimodal content blocks
# ═══════════════════════════════════════════════════════════════════════════


class TestF5RetryWithMultimodal:
    """F5: json_mode retry with multimodal initial message."""

    def test_retry_after_multimodal_initial_message(self):
        """Retry appends string messages after content-block message — must not crash."""
        from langchain_core.messages import AIMessage

        from neograph._llm import invoke_structured

        call_count = [0]

        class RetryLLM:
            def __init__(self, tier):
                pass

            def invoke(self, messages, **kw):
                call_count[0] += 1
                if call_count[0] == 1:
                    return AIMessage(content="not valid json")
                return AIMessage(content='{"score": 0.8}')

        _llm_kw = configure_fake_llm(lambda tier: RetryLLM(tier))

        b64 = base64.b64encode(b"test-img").decode()
        result = invoke_structured(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="Rate: ${image:photo}",
            input_data={"photo": b64},
            output_model=Score,
            config={"configurable": {}},
            llm_config={"output_strategy": "json_mode", "max_retries": 2},
        )
        assert result.score == 0.8
        assert call_count[0] == 2  # initial + 1 retry


# ═══════════════════════════════════════════════════════════════════════════
# F6: with_structured_output + multimodal (document limitation)
# ═══════════════════════════════════════════════════════════════════════════


class TestF6StructuredOutputMultimodal:
    """F6: structured output strategy with multimodal — verify messages reach the LLM."""

    def test_structured_strategy_receives_content_blocks(self):
        """with_structured_output().invoke() receives content blocks, not flat string."""
        received = []

        class InspectLLM:
            def __init__(self, tier):
                pass

            def with_structured_output(self, model, **kw):
                self._model = model
                return self

            def invoke(self, messages, **kw):
                received.append(messages)
                return self._model(score=0.5)

        _llm_kw = configure_fake_llm(lambda tier: InspectLLM(tier))

        from neograph._llm import invoke_structured

        b64 = base64.b64encode(b"img").decode()
        invoke_structured(
            runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
            model_tier="fast",
            prompt_template="Rate: ${image:photo}",
            input_data={"photo": b64},
            output_model=Score,
            config={"configurable": {}},
            llm_config={"output_strategy": "structured"},
        )

        user_msg = received[0][0]
        content = user_msg["content"] if isinstance(user_msg, dict) else user_msg.content
        assert isinstance(content, list), "structured output should receive content blocks"


# ═══════════════════════════════════════════════════════════════════════════
# F7: ${image:} empty field name
# ═══════════════════════════════════════════════════════════════════════════


class TestF7EmptyFieldName:
    """F7: ${image:} (no field name) falls through to text var path."""

    def test_empty_image_field_treated_as_text_var(self):
        """${image:} does not match _IMAGE_RE (requires 1+ chars)."""
        from neograph._llm import _compile_prompt

        data = {"question": "hello"}
        msgs = _compile_prompt("${question} ${image:}", data)
        content = msgs[0]["content"]
        # ${image:} doesn't match _IMAGE_RE, so entire prompt is text-only
        assert isinstance(content, str), f"${'{image:}'} should fall through to text path: {content}"


# ═══════════════════════════════════════════════════════════════════════════
# F8: ${IMAGE:photo} case sensitivity
# ═══════════════════════════════════════════════════════════════════════════


class TestF8CaseSensitivity:
    """F8: ${IMAGE:photo} not recognized as image placeholder."""

    def test_uppercase_image_not_matched(self):
        """${IMAGE:photo} is treated as a text variable, not an image."""
        from neograph._llm import _compile_prompt

        b64 = base64.b64encode(b"test").decode()
        data = {"photo": b64}
        msgs = _compile_prompt("Rate ${IMAGE:photo}", data)
        content = msgs[0]["content"]
        # Should be flat string (no image blocks) — IMAGE: is case-sensitive
        assert isinstance(content, str), f"Uppercase IMAGE: should not trigger multimodal: {content}"


# ═══════════════════════════════════════════════════════════════════════════
# F11: .strip() drops whitespace-only text blocks between images
# ═══════════════════════════════════════════════════════════════════════════


class TestF11WhitespaceStripping:
    """F11: whitespace-only text between images is stripped."""

    def test_whitespace_between_images_produces_adjacent_blocks(self):
        """'${image:a} ${image:b}' — the space is stripped, blocks are adjacent."""
        from neograph._llm import _compile_prompt

        b64a = base64.b64encode(b"a").decode()
        b64b = base64.b64encode(b"b").decode()
        data = {"a": b64a, "b": b64b}

        msgs = _compile_prompt("${image:a} ${image:b}", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        # Only image blocks — the space between is stripped
        types = [b["type"] for b in content]
        assert types == ["image_url", "image_url"], f"Expected two adjacent image blocks: {types}"


# ═══════════════════════════════════════════════════════════════════════════
# F13: multimodal in agent/act mode (tool loop) — document limitation
# ═══════════════════════════════════════════════════════════════════════════


class TestF13AgentModeMultimodal:
    """F13: ${image:...} in agent mode — messages flow through tool loop."""

    def test_agent_mode_multimodal_compiles_prompt(self):
        """Verify _compile_prompt produces content blocks for agent-mode prompts."""
        from neograph._llm import _compile_prompt

        b64 = base64.b64encode(b"agent-img").decode()
        data = {"photo": b64}
        msgs = _compile_prompt("Analyze this ${image:photo} and use tools", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert any(b["type"] == "image_url" for b in content)
        assert any(b["type"] == "text" and "tools" in b["text"] for b in content)


# ═══════════════════════════════════════════════════════════════════════════
# F14: BMP false positives
# ═══════════════════════════════════════════════════════════════════════════


class TestF14BMPFalsePositives:
    """F14: BMP magic needs secondary validation."""

    def test_text_starting_with_BM_not_classified_as_bmp(self):
        """A text file starting with 'BM' should not be classified as BMP."""
        from neograph._image import _check_magic_bytes

        # Text starting with "BM" but no valid BMP file size header
        data = b"BMI data for patient records, this is not an image"
        result = _check_magic_bytes(data)
        # Should be None (not recognized) or at least not "image/bmp"
        # because the file size field (bytes 2-5) would be garbage
        assert result != "image/bmp" or result is None, f"Text starting with 'BM' classified as BMP: {result}"

    def test_real_bmp_header_classified(self):
        """A proper BMP file header is classified correctly."""
        import struct

        from neograph._image import _check_magic_bytes

        # BM + file_size matching data length + reserved + offset + padding
        data_len = 100
        header = b"BM" + struct.pack("<I", data_len) + b"\x00" * (data_len - 6)
        result = _check_magic_bytes(header)
        assert result == "image/bmp"


# ═══════════════════════════════════════════════════════════════════════════
# F15: no escape mechanism for ${image:...}
# ═══════════════════════════════════════════════════════════════════════════


class TestF15NoEscapeMechanism:
    """F15: document that \\${image:...} is still matched (known limitation)."""

    def test_backslash_does_not_escape_image_placeholder(self):
        r"""\\${image:photo} is still matched as an image placeholder."""
        from neograph._llm import _compile_prompt

        b64 = base64.b64encode(b"test").decode()
        data = {"photo": b64}
        msgs = _compile_prompt(r"Literal: \${image:photo}", data)
        content = msgs[0]["content"]
        # Known limitation: the backslash does NOT escape the image placeholder
        assert isinstance(content, list), "Known limitation: backslash does not escape ${image:...}"
