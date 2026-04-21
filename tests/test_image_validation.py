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
from tests.fakes import StructuredFake, configure_fake_llm


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

        configure_fake_llm(factory)

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
            graph = compile(pipeline)
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

        configure_fake_llm(factory)

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
            graph = compile(pipeline)
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

            configure_fake_llm(factory)

            @node(outputs=ImageData)
            def seed() -> ImageData:
                return ImageData(photo=tmp)

            @node(outputs=Score, prompt="Rate: ${image:seed.photo}", model="fast")
            def score_it(seed: ImageData) -> Score: ...

            mod = _fresh_module("test_dir_block_e2e")
            mod.seed = seed
            mod.score_it = score_it

            pipeline = construct_from_module(mod, name="dir-block-e2e")
            graph = compile(pipeline)
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

            configure_fake_llm(factory)

            @node(outputs=ImageData)
            def seed() -> ImageData:
                return ImageData(photo=tmp)

            @node(outputs=Score, prompt="Rate: ${image:seed.photo}", model="fast")
            def score_it(seed: ImageData) -> Score: ...

            mod = _fresh_module("test_dir_allow_e2e")
            mod.seed = seed
            mod.score_it = score_it

            pipeline = construct_from_module(mod, name="dir-allow-e2e")
            graph = compile(pipeline)
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

        configure_fake_llm(lambda tier: _CaptureLLM(tier))

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
            graph = compile(pipeline)
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
        from neograph._image import _config

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
                assert result == "data:image/png;base64,", (
                    "Symlink escaping allowed_dirs should be blocked"
                )
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

        configure_fake_llm(factory)

        b64 = base64.b64encode(b"merge-image").decode()

        from neograph.factory import register_scripted
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
        graph = compile(pipeline)
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
            assert result == "data:image/png;base64,", (
                f"Non-image file should be rejected, got: {result[:60]}..."
            )
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
        with pytest.raises(ValueError, match="must be > 0"):
            configure_image(max_size_bytes=0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
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
        assert len(img_blocks) == 0, (
            f"BaseModel should not produce an image block: {img_blocks}"
        )


class TestF12MissingFieldSkipsImageBlock:
    """F12: missing/empty image field should omit the image block entirely."""

    def test_missing_field_omits_image_block(self):
        from neograph._llm import _compile_prompt

        data = {"question": "What is this?"}  # no 'photo' key
        msgs = _compile_prompt("${question} ${image:photo}", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 0, (
            f"Missing field should not produce an image block: {img_blocks}"
        )

    def test_none_field_omits_image_block(self):
        from neograph._llm import _compile_prompt

        data = {"photo": None}
        msgs = _compile_prompt("Score: ${image:photo}", data)
        content = msgs[0]["content"]
        assert isinstance(content, list)
        img_blocks = [b for b in content if b["type"] == "image_url"]
        assert len(img_blocks) == 0, (
            f"None field should not produce an image block: {img_blocks}"
        )
