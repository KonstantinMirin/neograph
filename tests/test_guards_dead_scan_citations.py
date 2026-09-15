"""Guard: no source comment, docstring or user-facing text may cite the deleted
whole-state isinstance scan as a LIVE mechanism.

neograph-rfp5s: ``_extract_single_type`` once resolved a single-type ``inputs=``
by scanning every state key for the first ``isinstance`` match. neograph-t1nbp
deleted that scan and replaced it with a named read off the assembly-stamped
address table -- but four citations of the scan survived as JUSTIFICATIONS: the
validator kept tolerating a first-of-chain read "because the runtime scan will
find it", a ``DeprecationWarning`` told users the form "relies on an O(N)
isinstance scan", the spec loader minted an implicit read "for type-scan
extraction", and AGENTS.md said the single-type form "defers to runtime
isinstance scan". A tolerance whose reason no longer exists is a silent seam:
the body received ``None`` on a green run. This guard bans the citation so the
tolerance cannot grow back on its strength.

Deliberately a TEXT guard, not a behavioural one: the behaviour is pinned by
``tests/test_validation.py::TestFirstNodeSingleTypeInputsCannotBeFed``. What no
runtime assertion can observe is a future author writing "defers to the runtime
isinstance scan" above a new early return.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src" / "neograph"
DOCS = [ROOT / "AGENTS.md"]

#: Spellings of the dead mechanism as a live claim. Historical narration that
#: names the scan as REPLACED ("the scan this replaced", "there is no fallback
#: scan") does not use these spellings, which is what keeps the guard cheap.
_DEAD_CITATION = re.compile(
    r"isinstance[\s-]*scan|"  # "isinstance scan", "isinstance-scan", "isinstance scanning"
    r"type-scan extraction|"  # the loader's minting comment
    r"defers?\s+to\s+(?:the\s+)?runtime\s+isinstance|"  # the validator's docstring
    r"runtime-seeded|seeded_from_runtime|pre-seeded state",  # the test pins
    re.IGNORECASE,
)


def _citations(text: str) -> list[str]:
    return [m.group(0) for m in _DEAD_CITATION.finditer(text)]


def _scan_tree() -> list[str]:
    hits: list[str] = []
    files = [p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts] + DOCS
    for path in sorted(files):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _citations(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    return hits


class TestNoDeadScanCitation:
    def test_no_source_or_agents_line_cites_the_deleted_scan(self):
        hits = _scan_tree()
        assert hits == [], (
            "a comment, docstring or user-facing text cites the whole-state isinstance scan "
            "that neograph-t1nbp deleted. A tolerance justified by a mechanism that does not "
            "exist is how a declared read came to run with None on a green run (neograph-rfp5s). "
            "Delete the citation; do not reword it:\n  " + "\n  ".join(hits)
        )


class TestTheGuardActuallyDetects:
    """Non-vacuity: a regex that matches nothing passes the assertion above."""

    def test_positive_the_validator_wording_is_caught(self):
        assert _citations("Cases that defer to runtime isinstance-scanning:")

    def test_positive_the_warning_wording_is_caught(self):
        assert _citations("relies on O(N) isinstance scan at runtime.")

    def test_positive_the_loader_wording_is_caught(self):
        assert _citations("inputs = outputs  # single-type fallback for type-scan extraction")

    def test_positive_the_test_pin_wording_is_caught(self):
        assert _citations('"""First-of-chain with declared input is NOT flagged -- runtime-seeded."""')
        assert _citations('Each(over="seeded_from_runtime.groups", key="label")')

    def test_slip_dead_citation_spelling_variants_are_still_caught(self):
        """Hyphen, doubled whitespace, participle and case are the cheap dodges."""
        for variant in ("isinstance-scan", "isinstance  scan", "IsInstance Scanning", "defers to the runtime  isinstance"):
            assert _citations(variant), variant

    def test_negative_the_inverse_anchors_are_not_caught(self):
        """The docstrings that correctly say the scan is GONE must stay legal."""
        for text in (
            "there is deliberately no fallback scan here",
            "It replaced a forward scan over state.keys() that returned the first isinstance match",
            "The scan this replaces walked the whole state bag",
            "the whole-state scan is gone, neograph-t1nbp",
            "(runtime isinstance handles it)",
            "primitives (str, int, etc.) defer to runtime str(item) fallback.",
        ):
            assert not _citations(text), text

    def test_the_scanner_sees_the_real_tree(self):
        files = [p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in p.parts]
        assert any(p.name == "_validation_inputs.py" for p in files)
        assert all(p.exists() for p in DOCS)
