"""Image resolution and validation for multimodal prompts.

Handles file paths, raw base64, and data URIs. Validates size, directory
restrictions, and image format before encoding. Never crashes the pipeline
-- warns and degrades gracefully.

Configuration via ``configure_image()``::

    from neograph import configure_image

    configure_image(
        max_size_bytes=10_000_000,           # 10MB
        allowed_dirs=["/data/images"],       # restrict file reads
    )

Security: when ``allowed_dirs`` is None (default), resolve_image reads any
file the process can access. Set ``allowed_dirs`` to restrict file reads
when image field values come from untrusted input.
"""

from __future__ import annotations

import base64
import dataclasses
import mimetypes
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

# ── Magic bytes for common image formats ─────────────────────────────────

_IMAGE_MAGIC: dict[bytes, str] = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
}

_IMAGE_MIME_PREFIXES = frozenset(["image/"])


def _check_magic_bytes(data: bytes) -> str | None:
    """Return MIME type from magic bytes, or None if unrecognized."""
    for magic, mime in _IMAGE_MAGIC.items():
        if data[:len(magic)] == magic:
            return mime
    # WebP: RIFF header + "WEBP" at bytes 8-12 (distinguishes from WAV/AVI)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # BMP: "BM" + file size field (bytes 2-5) must roughly match actual data length
    if data[:2] == b"BM" and len(data) >= 14:
        import struct
        file_size = struct.unpack_from("<I", data, 2)[0]
        if 14 <= file_size <= len(data) * 2:
            return "image/bmp"
    # SVG: look for <svg or <?xml in the first 512 bytes
    head = data[:512]
    if b"<svg" in head or (b"<?xml" in head and b"<svg" in data[:4096]):
        return "image/svg+xml"
    return None


# ── Configuration ────────────────────────────────────────────────────────

@dataclasses.dataclass
class ImageConfig:
    """Configuration for image resolution and validation."""

    max_size_bytes: int = 20 * 1024 * 1024  # 20MB
    """Maximum file size in bytes. Files exceeding this are rejected with a warning."""

    allowed_dirs: list[str] | None = None
    """Restrict file reads to these directories. None = allow all (default).
    Set this when image field values come from untrusted input."""

    validate_format: bool = True
    """Check that file content looks like a known image format (magic bytes)."""


_config = ImageConfig()


def configure_image(
    *,
    max_size_bytes: int = 20 * 1024 * 1024,
    allowed_dirs: list[str] | None = None,
    validate_format: bool = True,
) -> None:
    """Configure image resolution settings.

    Call with no arguments to reset to defaults.
    """
    if max_size_bytes <= 0:
        raise ValueError(f"max_size_bytes must be > 0, got {max_size_bytes}")
    global _config
    _config = ImageConfig(
        max_size_bytes=max_size_bytes,
        allowed_dirs=allowed_dirs,
        validate_format=validate_format,
    )


# ── Public API ───────────────────────────────────────────────────────────

def resolve_image(path_or_b64: str) -> str:
    """Convert a file path, raw base64, or data URI to a data-URI string.

    Applies configured validation (size limit, directory restriction,
    format check). Never raises -- warns and returns a placeholder or
    fallback data URI on failure.

    Three input forms:
    - ``data:image/...;base64,...`` — returned as-is
    - A file path that exists on disk — validated, read, base64-encoded
    - Anything else — assumed raw base64, wrapped as ``data:image/png;base64,...``
    """
    if not path_or_b64 or not path_or_b64.strip():
        log.warning("image_resolve_empty", hint="image field is empty or whitespace")
        return "data:image/png;base64,"

    if path_or_b64.startswith("data:"):
        return path_or_b64

    p = Path(path_or_b64)
    if p.is_file():
        return _read_and_encode_file(p)

    # Warn if the value looks like a file path but doesn't exist
    if "/" in path_or_b64 or "\\" in path_or_b64:
        log.warning("image_path_not_found", path=path_or_b64,
                     hint="value looks like a file path but file not found; treating as raw base64")

    # Assume raw base64
    return f"data:image/png;base64,{path_or_b64}"


def _read_and_encode_file(p: Path) -> str:
    """Read, validate, and base64-encode a file. Handles all failure modes."""

    # Directory restriction
    if _config.allowed_dirs is not None:
        resolved = p.resolve()
        allowed = any(
            resolved.is_relative_to(Path(d).resolve())
            for d in _config.allowed_dirs
        )
        if not allowed:
            log.warning("image_dir_blocked", path=str(p),
                        allowed_dirs=_config.allowed_dirs,
                        hint="file is outside allowed directories; skipping")
            return "data:image/png;base64,"

    # TOCTOU-safe read
    try:
        data = p.read_bytes()
    except (FileNotFoundError, PermissionError, OSError) as exc:
        log.warning("image_read_failed", path=str(p), error=str(exc))
        return "data:image/png;base64,"

    # Size check
    if len(data) > _config.max_size_bytes:
        log.warning("image_too_large", path=str(p),
                     size_bytes=len(data),
                     max_bytes=_config.max_size_bytes,
                     hint="file exceeds max_size_bytes; skipping")
        return "data:image/png;base64,"

    # Format validation + MIME detection
    detected_mime = _check_magic_bytes(data) if _config.validate_format else None
    extension_mime = mimetypes.guess_type(str(p))[0] or ""

    if _config.validate_format and detected_mime is None:
        if not any(extension_mime.startswith(pre) for pre in _IMAGE_MIME_PREFIXES):
            log.warning("image_format_rejected", path=str(p),
                        mime_guess=extension_mime,
                        hint="file does not look like a known image format; skipping")
            return "data:image/png;base64,"

    # Prefer magic-bytes MIME (accurate) over extension MIME (guesswork)
    mime = detected_mime or extension_mime or "image/png"

    encoded = base64.b64encode(data).decode()
    return f"data:{mime};base64,{encoded}"
