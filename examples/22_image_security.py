"""Example 22: Image Security — configure_image for production safety.

Scenario: A multi-tenant platform processes user-uploaded images through
a VLM pipeline. Users control the image path (it comes from an API request).
Without restrictions, a malicious user could read arbitrary files from the
server (e.g., /etc/passwd, .env, SSH keys).

``configure_image()`` locks down which files the pipeline can read:
- ``allowed_dirs`` — restrict to specific upload directories
- ``max_size_bytes`` — prevent memory exhaustion from large files
- ``validate_format`` — reject non-image files (executables, configs)

All validation is graceful: blocked files produce a warning and an empty
image block. The pipeline continues — the LLM simply doesn't see the image.

Run:
    python examples/22_image_security.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from neograph import compile, configure_image, run
from neograph import Construct, Node
from neograph._image import resolve_image


# ── Schemas ──────────────────────────────────────────────────────────────

class ImageInput(BaseModel, frozen=True):
    photo: str

class Analysis(BaseModel, frozen=True):
    result: str


# ── Fake LLM ─────────────────────────────────────────────────────────────

class FakeLLM:
    def __init__(self, tier):
        pass

    def with_structured_output(self, model, **kw):
        self._model = model
        return self

    def invoke(self, messages, **kw):
        content = messages[0]["content"] if isinstance(messages[0], dict) else ""
        if isinstance(content, list):
            has_image = any(
                b.get("type") == "image_url" and b["image_url"]["url"] != "data:image/png;base64,"
                for b in content
            )
        else:
            has_image = False
        return self._model(result="analyzed" if has_image else "no-image")


_llm_factory = lambda tier: FakeLLM(tier)
_prompt_compiler = lambda t, d, **kw: [{"role": "user", "content": t}]


# ── Set up a safe upload directory ───────────────────────────────────────

UPLOAD_DIR = tempfile.mkdtemp(prefix="uploads_")
print(f"Upload directory: {UPLOAD_DIR}")

# Create a legitimate image in the upload directory
legit_image = os.path.join(UPLOAD_DIR, "product.png")
Path(legit_image).write_bytes(b"\x89PNG\r\n\x1a\n" + b"real-product-photo" * 5)

# Create a "secret" file outside the upload directory
secret_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="secret_")
secret_file.write(b"\x89PNG\r\n\x1a\nthis-is-actually-sensitive-data")
secret_file.close()

# Create a non-image file in the upload directory
config_file = os.path.join(UPLOAD_DIR, "oops.txt")
Path(config_file).write_bytes(b"DATABASE_URL=postgres://admin:password@localhost/prod")

# Create an oversized image
big_image = os.path.join(UPLOAD_DIR, "huge.png")
Path(big_image).write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 5_000_000)


# ── Configure image security ────────────────────────────────────────────

configure_image(
    allowed_dirs=[UPLOAD_DIR],       # only read from upload directory
    max_size_bytes=1_000_000,        # 1MB max
    validate_format=True,            # reject non-image files
)


# ── Build pipeline ───────────────────────────────────────────────────────

def _img_seed(i, c):
    return ImageInput(photo=c.get("configurable", {}).get("photo_path", ""))


seed = Node.scripted("seed", fn="_img_seed", outputs=ImageInput)
analyze = Node(
    "analyze", mode="think", outputs=Analysis,
    prompt="Analyze: ${image:seed.photo}", model="fast",
    inputs={"seed": ImageInput},
)
pipeline = Construct("secure-vision", nodes=[seed, analyze])
graph = compile(
    pipeline,
    llm_factory=_llm_factory,
    prompt_compiler=_prompt_compiler,
    scripted={"_img_seed": _img_seed},
)


# ── Test cases ───────────────────────────────────────────────────────────

print()
print("=" * 60)

# 1. Legitimate image in upload dir
print("1. Legitimate image in upload dir:")
r1 = run(graph, input={"node_id": "t1", "photo_path": legit_image})
print(f"   Result: {r1['analyze'].result}")
assert r1["analyze"].result == "analyzed", "Legit image should be analyzed"

# 2. File outside upload dir (path traversal attempt)
print()
print("2. File outside upload dir (path traversal):")
r2 = run(graph, input={"node_id": "t2", "photo_path": secret_file.name})
print(f"   Result: {r2['analyze'].result}")
assert r2["analyze"].result == "no-image", "File outside allowed_dirs should be blocked"

# 3. Non-image file in upload dir
print()
print("3. Non-image file (.txt) in upload dir:")
r3 = run(graph, input={"node_id": "t3", "photo_path": config_file})
print(f"   Result: {r3['analyze'].result}")
assert r3["analyze"].result == "no-image", "Non-image file should be rejected"

# 4. Oversized image in upload dir
print()
print("4. Oversized image (5MB > 1MB limit):")
r4 = run(graph, input={"node_id": "t4", "photo_path": big_image})
print(f"   Result: {r4['analyze'].result}")
assert r4["analyze"].result == "no-image", "Oversized file should be rejected"

# 5. Direct resolve_image utility
print()
print("5. resolve_image utility (for template-ref consumers):")
uri = resolve_image(legit_image)
print(f"   URI prefix: {uri[:40]}...")
assert uri.startswith("data:image/png;base64,")
assert uri != "data:image/png;base64,"  # not empty

print()
print("=" * 60)
print("All 5 cases pass. configure_image locks down file access")
print("while the pipeline degrades gracefully on blocked files.")


# ── Cleanup ──────────────────────────────────────────────────────────────

Path(legit_image).unlink()
Path(config_file).unlink()
Path(big_image).unlink()
os.rmdir(UPLOAD_DIR)
Path(secret_file.name).unlink()

# Reset to defaults
configure_image()
