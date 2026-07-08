"""Structural guards: cross-module naming/casing consistency.

Currently pins the canonical wordmark casing (CON-08 from the Jun-2026 review):
the project wordmark is lowercase ``neograph`` everywhere (website SiteTitle,
CLAUDE.md, pyproject). Mixed-case ``NeoGraph`` had drifted into three source
docstrings. This guard bans the mixed-case substring in ``src/neograph/`` so a
future PR cannot reintroduce it.
"""

from __future__ import annotations

import pathlib

import pytest

SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph"

# The forbidden mixed-case wordmark. Matched as a *substring*, NOT a whole word:
# an embedded form like ``CompiledNeoGraph`` must also be caught (a word-boundary
# regex would slip it -- cf. MED-06, where ``_neo_isolated_input`` slipped a
# ``^neo_`` anchor).
FORBIDDEN_WORDMARK = "NeoGraph"


class TestNoMixedCaseWordmark:
    """CON-08: the wordmark is lowercase ``neograph`` everywhere in src prose.

    No mixed-case ``NeoGraph`` substring may appear in ``src/neograph/``, now or
    in any future PR. The scanner is substring-based (no word boundaries) so
    embedded PascalCase compounds cannot slip.

    Mutation-verified below with positive, negative, and regex-slip meta-tests.
    """

    @staticmethod
    def _find_mixed_case_wordmark(source: str) -> list[tuple[int, str]]:
        """Return ``(lineno, stripped_line)`` for every line containing the
        forbidden mixed-case wordmark substring."""
        offenders: list[tuple[int, str]] = []
        for lineno, line in enumerate(source.splitlines(), start=1):
            if FORBIDDEN_WORDMARK in line:
                offenders.append((lineno, line.strip()))
        return offenders

    def test_no_mixed_case_wordmark_in_src(self) -> None:
        """No src/neograph/ file may contain the mixed-case 'NeoGraph' wordmark."""
        offenders: list[str] = []
        for path in sorted(SRC_DIR.rglob("*.py")):
            for lineno, text in self._find_mixed_case_wordmark(path.read_text()):
                rel = path.relative_to(SRC_DIR.parent.parent)
                offenders.append(f"{rel}:{lineno}: {text}")
        assert not offenders, "Mixed-case 'NeoGraph' wordmark found (canonical is lowercase 'neograph'):\n" + "\n".join(
            offenders
        )

    # -- Mutation meta-tests: prove the scanner actually scans ----------------

    def test_meta_positive_flags_mixed_case(self) -> None:
        """Positive: a docstring with 'NeoGraph' is flagged."""
        hits = self._find_mixed_case_wordmark('"""NeoGraph compiler."""')
        assert hits == [(1, '"""NeoGraph compiler."""')]

    @pytest.mark.parametrize(
        "clean_source",
        [
            '"""neograph compiler."""',  # correct lowercase wordmark
            "class CompiledNeograph:  # correct PascalCase, lowercase wordmark",
            "from neograph.runner import run",  # package import path
        ],
    )
    def test_meta_negative_ignores_correct_casing(self, clean_source: str) -> None:
        """Negative: correctly-cased usages are NOT flagged."""
        assert self._find_mixed_case_wordmark(clean_source) == []

    def test_meta_regex_slip_catches_embedded_variant(self) -> None:
        """Regex-slip: an embedded 'CompiledNeoGraph' (no word boundary) is
        still caught -- a ``\\bNeoGraph\\b`` regex would miss it."""
        hits = self._find_mixed_case_wordmark("class CompiledNeoGraph: ...")
        assert hits == [(1, "class CompiledNeoGraph: ...")]
