"""Structural guard: no hand-parsing of ``ToolInteraction.typed_result`` as raw MCP
content blocks in production/demo code (neograph-wmmhc).

## Core Invariant

An MCP tool result is consumed through the TYPED channel: declare ``output_model=``
on the ``neograph_mcp`` factory and neograph rehydrates the server's
``structuredContent`` into a model, so ``ToolInteraction.typed_result`` IS that
model and downstream code reads ATTRIBUTES (``r.typed_result.hits[0]``). The
disease this guard bans is the pre-wmmhc shortcut: treating ``typed_result`` as an
indexable list of langchain content-block dicts and pulling a JSON text block out
of it by hand (``json.loads(r.typed_result[0]["text"])`` / a ``_parse_mcp_json``
helper). That shortcut ships the backwards-compat mirror text to the model instead
of a BAML-rendered model and re-implements what ``output_model=`` does for free.

Scope: ``examples/`` + ``src/`` (the demo + production surface, where declaring
``output_model=`` is the right pattern). ``tests/`` is intentionally OUT of scope —
a test that pins the *default* (no-``output_model``) content-block path legitimately
parses the RAW adapter ``ainvoke`` return (``json.loads(tool.ainvoke(...)[0]["text"])``),
which is a different object than ``ToolInteraction.typed_result``.

Non-vacuity is proven by ``test_scanner_flags_injected_hand_parse``, which feeds
synthetic source (not the real tree) through the SAME scanner and confirms it FLAGS
a planted violation, including a whitespace/split variant.
"""

from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCAN_DIRS = [_ROOT / "examples", _ROOT / "src"]

# Forms that treat ``typed_result`` as raw MCP content blocks instead of a model:
#   - a ``_parse_mcp_json``-style hand-parser (the deleted helper / any re-intro)
#   - subscripting ``typed_result`` directly (``typed_result[0]`` — content-block
#     index; attribute access ``typed_result.field`` is the CORRECT form and does
#     NOT match, because ``[`` must follow ``typed_result`` with only whitespace)
#   - ``json.loads(...)`` applied to a ``typed_result``-derived value on one line
_BANNED = re.compile(
    r"_parse_mcp_json"
    r"|typed_result\s*\["
    r"|json\.loads\([^\n]*typed_result",
    re.MULTILINE,
)


def _strip_comments(text: str) -> str:
    """Drop ``#`` line comments so the scan is about CODE, not prose. A comment may
    legitimately NAME the banned form (e.g. 'no _parse_mcp_json shim'); that is
    documentation, not a violation. Naive per-line strip — adequate here because no
    banned pattern spans a ``#`` (worst case it drops a ``#`` inside a string, which
    cannot hide a real hand-parse)."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _scan(text: str) -> list[str]:
    code = _strip_comments(text)
    return [m.group(0).strip() for m in _BANNED.finditer(code)]


class TestNoTypedResultHandParse:
    def test_examples_and_src_never_hand_parse_typed_result(self):
        offenders: dict[str, list[str]] = {}
        for base in _SCAN_DIRS:
            for path in base.rglob("*.py"):
                hits = _scan(path.read_text())
                if hits:
                    offenders[str(path.relative_to(_ROOT))] = hits
        assert offenders == {}, (
            "MCP tool results must be consumed through the typed channel "
            "(declare output_model= on the neograph_mcp factory and read "
            "ToolInteraction.typed_result as a MODEL), NOT hand-parsed as raw "
            "content blocks. Offending files: " + repr(offenders)
        )

    def test_slip_banned_flags_injected_hand_parse(self):
        """Slip meta-test for ``_BANNED`` (non-vacuity, PROC-2): the scanner must
        FLAG a planted violation, including a whitespace / split-line variant a
        naive regex would miss."""
        bad = (
            "payload = call.typed_result[0]['text']\n"          # direct subscript
            "data = json.loads(call.typed_result[0]['text'])\n"  # json.loads over it
            "spaced = other.typed_result [ 0 ]\n"                # whitespace variant
            "helper = _parse_mcp_json(i.typed_result)\n"         # the deleted helper
        )
        hits = _scan(bad)
        assert len(hits) >= 4, f"scanner missed a hand-parse variant: {hits}"

    def test_scanner_accepts_clean_typed_channel_source(self):
        """Attribute access on the rehydrated model is the CORRECT pattern and must
        NOT be flagged — including attribute-then-index on a model's list field."""
        clean = (
            "who = {i.tool_name: i.typed_result.acting_as for i in tool_log}\n"
            "deal = search.typed_result.hits[0]\n"       # index is on .hits, not typed_result
            "note = call.typed_result\n"
            "interaction = ToolInteraction(typed_result=result)\n"  # assignment, not subscript
        )
        assert _scan(clean) == [], f"clean typed-channel source was wrongly flagged: {_scan(clean)}"
