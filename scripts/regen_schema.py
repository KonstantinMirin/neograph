"""Regenerate src/neograph/schemas/neograph-pipeline.schema.json from Spec.

Run after modifying _spec_schema.py:

    uv run python scripts/regen_schema.py

A test (test_spec_schema.py::test_pipeline_schema_in_sync) asserts that
the on-disk JSON file equals Spec.model_json_schema(), so drift fails CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neograph._spec_schema import Spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "neograph" / "schemas" / "neograph-pipeline.schema.json"


def main() -> int:
    schema = Spec.model_json_schema()
    text = json.dumps(schema, indent=2) + "\n"
    TARGET.write_text(text)
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
