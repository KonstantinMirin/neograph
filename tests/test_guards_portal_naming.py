"""Structural guard: the dynamic-handoff construct is named Portal, never Keymaker.

Keymaker was the original (Matrix-themed) name for the dynamic-handoff /
runtime-flow-definition construct. It was renamed to ``Portal`` before it ever
shipped in a release (neograph-1t0zh, 2026-07-14): a functional name reads for
newcomers and is more accurate (a portal is an *entry port* to a declared
destination, spanning both peer-routing and dynamic-flow modes). This guard
pins the rename so the old name cannot creep back into the shipped package or
its public API manifest.

Scope note: only the CONSTRUCT-IDENTITY token (the old construct name) is
banned, and ONLY under ``src/neograph`` + the public manifest. The MECHANISM
word ``handoff`` (``handoff_param``, ``neo_handoff_*``, ``HANDOFF_END``) is a
different, accurate token that deliberately stays — it describes the runtime
mechanism, not the construct, and this guard does not touch it. This test file
lives under ``tests/`` (not scanned) so it may name the retired token freely.
"""

from __future__ import annotations

import json
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "neograph"
API_MANIFEST = (
    Path(__file__).resolve().parent.parent / "website" / "src" / "data" / "api-manifest.json"
)

# The retired construct name, assembled so this literal is the ONLY occurrence
# in the file and the ban is unambiguous (case-insensitive match below).
BANNED = "key" + "maker"


def test_no_retired_name_in_shipped_package():
    """No .py file under src/neograph carries the retired construct name."""
    offenders: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        if BANNED in text.lower():
            for i, line in enumerate(text.splitlines(), 1):
                if BANNED in line.lower():
                    offenders.append(f"{py.relative_to(SRC.parent.parent)}:{i}: {line.strip()}")
    assert not offenders, (
        f"retired construct name {BANNED!r} found in the shipped package "
        "(use 'Portal' — neograph-1t0zh). Note: the 'handoff' mechanism word is "
        "allowed and is a different token.\n" + "\n".join(offenders)
    )


def test_no_retired_name_in_public_api_manifest():
    """The introspection-generated public API manifest names Portal, not the old name."""
    if not API_MANIFEST.exists():
        return  # manifest optional in some checkouts; package scan is the hard gate
    raw = API_MANIFEST.read_text(encoding="utf-8")
    assert BANNED not in raw.lower(), (
        f"retired name {BANNED!r} found in website/src/data/api-manifest.json; "
        "regenerate via scripts/gen_api_manifest.py after the Portal rename (neograph-1t0zh)."
    )
    # sanity: the manifest is valid JSON
    json.loads(raw)
