"""neograph-wwhcw: repo-wide 500-line file-size ratchet for ``src/neograph``.

THE RULE. Every Python module shipped under ``src/neograph/`` (recursive, ``__pycache__``
skipped, so ``testing/`` is covered too) is either **under 500 lines** or carries an
explicit entry in ``ALLOWLIST`` whose value **exactly equals** the file's real current
line count. There is no tolerance band: a ceiling that merely *covers* the file is a
failure, because both pre-existing loose ceilings in this repo had already drifted
stale-loose before anyone noticed (compiler.py 775 declared vs 761 real; _llm_retry.py
665 declared vs 658 real). Exact equality is the self-check that whoever touched a
governed file actually re-derived its count instead of transcribing an old one.

THE RATCHET (shrink-only).
  * A file may shrink freely -- always welcome, no approval or ceremony -- and lowers its
    own ceiling in the SAME commit via the one-line dict edit.
  * A file may never grow past its ceiling. Growth is blocked and fixed IN-PR, never
    deferred; "raise the number" is not a remedy, splitting the file is.
  * A file NOT in the allowlist must stay under 500 outright. Crossing 500 requires a
    fresh, reviewed allowlist entry.
  * ANTI-DEAD-ENTRY: an entry whose file is missing, or whose file has dropped under 500,
    is DEAD and must be DELETED -- never lowered to a sub-500 number, which would quietly
    grant a private ceiling to a file the plain 500 rule should govern. This is what
    drives the allowlist toward empty as the refactor wave lands.

A RED GUARD IS OFTEN THE CORRECT SIGNAL. The 19 open ``neograph-3ffdg`` children and
``neograph-s7zt3.11`` each shrink a governed file; every one of them turns this gate red
by design. Lower (or DELETE) the entry in the same commit -- that is the ratchet working,
not a regression to route around. A ``ruff format`` pass likewise changes counts
mechanically and legitimately requires a dict update; the guard is not broken when it
fires for that reason.

FORMATTING IS LOAD-BEARING. ``ALLOWLIST`` is one entry per line, sorted by posix path, so
~19 parallel refactor branches edit disjoint lines and the merge-conflict surface stays
near zero. Keep it that way.

SCOPE, RECORDED HONESTLY. Two other per-file line caps survive elsewhere:
``TestConstructBuilderSplit.LINE_CAP = 330`` and the validation-cluster
``LINE_CAP = 400``, both in ``tests/test_guards_assembly.py``. They are kept ONLY because
their file sets are DISJOINT from this allowlist -- all 10 files they govern are under
500 (max 289 under the 330 cap, max 373 under the 400 cap), so this guard makes no claim
about any of them -- and because extending strict equality to 10 more files was never
scoped. They are NOT clean and NOT "complementary": both carry exactly the stale-loose
headroom this guard exists to eliminate (330 vs 289 real; 400 vs 373 real). Loose
semantics there are out of scope, not endorsed.
Alternative considered and deliberately REJECTED: consolidating all the surviving
mechanisms into one ceiling table -- purer per CLAUDE.md's Portal-rollout single-source
lesson, but it taxes 10 files the maintainer never scoped.

The refactor-down work itself is NOT this guard's job; see
``docs/design/file-size-refactor-master-proposal-2026-07-29.md``.

This module imports no ``re`` at all (pathlib + ``splitlines`` only), so it is
exempt-by-construction from ``test_guards_meta.py``'s Discipline 1 (named-regex constant
+ slip meta-test), whose pass-0 builds its alias set from ``re`` imports.
"""

from __future__ import annotations

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "neograph"

LIMIT = 500

# Anti-vacuity floor: the sweep saw 84 .py files on 2026-07-29. A count below this floor
# means the scan root is wrong (installed wheel, moved package), not that the repo shrank.
MIN_EXPECTED_FILES = 60

# Shrink-only per-file ceilings: posix path under src/neograph -> EXACT current line count.
# One entry per line, sorted by posix path. See the module docstring for the contract; no
# per-entry reason strings (all 21 would say the same thing).
ALLOWLIST: dict[str, int] = {
    "_agent_cycle.py": 776,
    "_agent_spec.py": 981,
    "_oracle.py": 509,
    "_tool_loop.py": 554,
    "_wiring.py": 817,
    "compiler.py": 689,
    "decorators.py": 671,
    "di.py": 506,
    "factory.py": 949,
    "forward.py": 1022,
    "lint.py": 633,
    "loader.py": 1181,
    "modifiers.py": 881,
    "node.py": 522,
    "runner.py": 685,
    "spec_types.py": 521,
    "testing/fakes.py": 729,
    "testing/scaffold.py": 664,
}


def _scan_counts(root: Path) -> dict[str, int]:
    """Every ``*.py`` under `root` (recursive, ``__pycache__`` skipped) -> line count."""
    return {
        path.relative_to(root).as_posix(): len(path.read_text().splitlines())
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _size_violations(root: Path, allowlist: dict[str, int], limit: int) -> list[str]:
    """Pure check: files at/over `limit` with no entry, plus entries whose ceiling is not
    EXACTLY the file's real count (too low = growth; too high = stale-loose)."""
    violations: list[str] = []
    for rel, n_lines in _scan_counts(root).items():
        if rel not in allowlist:
            if n_lines >= limit:
                violations.append(
                    f"  OVER-LIMIT: {rel} is {n_lines} lines (limit {limit}).\n"
                    f"      Remedy: split it into cohesive modules, or -- if the size is "
                    f"genuinely justified -- add a reviewed entry to ALLOWLIST in "
                    f'tests/test_guards_file_size.py: "{rel}": {n_lines},'
                )
            continue
        ceiling = allowlist[rel]
        if n_lines != ceiling:
            direction = "GREW PAST" if n_lines > ceiling else "shrank below"
            violations.append(
                f"  CEILING-MISMATCH: {rel} declares {ceiling} but is really {n_lines} "
                f"lines ({direction} its ceiling).\n"
                f"      Paste this exact line into ALLOWLIST:  "
                f'"{rel}": {n_lines},  # was {ceiling}\n'
                f"      If it GREW, do NOT just raise the number -- shrink the file "
                f"in-PR; the ceiling may only follow a real shrink."
            )
    return violations


def _dead_allowlist_entries(root: Path, allowlist: dict[str, int], limit: int) -> list[str]:
    """Pure check: entries whose file is MISSING, or whose file is now under `limit`.
    Both are dead and must be DELETED (never lowered to a sub-limit number)."""
    dead: list[str] = []
    for rel, ceiling in sorted(allowlist.items()):
        path = root / rel
        if not path.exists():
            dead.append(
                f"  DEAD (file missing): {rel} (declared ceiling {ceiling}) no longer "
                f"exists under {root.name}/.\n"
                f"      Remedy: DELETE the entry from ALLOWLIST."
            )
            continue
        n_lines = len(path.read_text().splitlines())
        if n_lines < limit:
            dead.append(
                f"  DEAD (now under {limit}): {rel} is {n_lines} lines (declared ceiling "
                f"{ceiling}).\n"
                f"      Remedy: DELETE the entry from ALLOWLIST -- do NOT lower it to "
                f"{n_lines}. The plain {limit}-line rule governs this file now; a "
                f"sub-{limit} entry would grant it a private ceiling."
            )
    return dead


class TestFileSizeRatchet:
    """src/neograph modules stay under 500 lines unless they carry an exact ceiling."""

    def test_scan_sees_a_plausible_number_of_source_files(self) -> None:
        """Anti-vacuity tripwire: the whole guard passes green against an empty or wrong
        directory, so pin that the sweep actually found the package."""
        counts = _scan_counts(SRC_DIR)
        assert len(counts) >= MIN_EXPECTED_FILES, (
            f"file-size sweep found only {len(counts)} .py file(s) under {SRC_DIR} "
            f"(expected at least {MIN_EXPECTED_FILES}; 84 on 2026-07-29). The SRC root "
            "path is wrong -- the guard is scanning the wrong directory (moved package, "
            "installed wheel, or a bad relative path), not passing legitimately."
        )

    def test_no_unallowlisted_file_over_limit(self) -> None:
        violations = [v for v in _size_violations(SRC_DIR, ALLOWLIST, LIMIT) if "OVER-LIMIT" in v]
        assert violations == [], (
            f"\n{len(violations)} src/neograph file(s) at or over the {LIMIT}-line limit "
            "with no ALLOWLIST entry:\n" + "\n".join(violations)
        )

    def test_allowlisted_files_are_at_their_exact_ceiling(self) -> None:
        """'Exactly right, not merely sufficient' -- blocks growth AND forbids a ceiling
        drifting stale-loose above the file it governs."""
        violations = [v for v in _size_violations(SRC_DIR, ALLOWLIST, LIMIT) if "CEILING-MISMATCH" in v]
        assert violations == [], (
            f"\n{len(violations)} ALLOWLIST ceiling(s) that do not EXACTLY match the "
            "file's real line count:\n" + "\n".join(violations)
        )

    def test_no_dead_allowlist_entries(self) -> None:
        dead = _dead_allowlist_entries(SRC_DIR, ALLOWLIST, LIMIT)
        assert dead == [], f"\n{len(dead)} DEAD ALLOWLIST entr(y/ies) -- the allowlist may only shrink:\n" + "\n".join(
            dead
        )

    # --- mutation meta-tests: prove the checkers actually catch each violation shape ---

    @staticmethod
    def _write_lines(path: Path, n_lines: int) -> None:
        path.write_text("".join(f"x = {i}\n" for i in range(n_lines)))

    def test_meta_flags_oversized_file_with_no_entry(self, tmp_path: Path) -> None:
        """(a) a 600-line file not in the allowlist IS flagged."""
        self._write_lines(tmp_path / "huge.py", 600)
        violations = _size_violations(tmp_path, {}, LIMIT)
        assert len(violations) == 1 and "OVER-LIMIT: huge.py is 600 lines" in violations[0], violations

    def test_meta_passes_small_file_with_no_entry(self, tmp_path: Path) -> None:
        """(b) a 400-line file is NOT flagged."""
        self._write_lines(tmp_path / "small.py", 400)
        assert _size_violations(tmp_path, {}, LIMIT) == []

    def test_meta_flags_ceiling_off_by_one_in_both_directions(self, tmp_path: Path) -> None:
        """(c) LOAD-BEARING: a real count differing from its ceiling by +1 AND by -1 is
        flagged BOTH ways. This is the only test distinguishing strict equality from a
        one-sided ``>`` (growth-only) implementation -- the -1 case is what a ``>``
        implementation silently lets through as stale-loose headroom."""
        self._write_lines(tmp_path / "governed.py", 600)

        grew = _size_violations(tmp_path, {"governed.py": 599}, LIMIT)
        assert len(grew) == 1 and "CEILING-MISMATCH" in grew[0], grew
        assert "declares 599 but is really 600" in grew[0] and "GREW PAST" in grew[0], grew

        shrank = _size_violations(tmp_path, {"governed.py": 601}, LIMIT)
        assert len(shrank) == 1 and "CEILING-MISMATCH" in shrank[0], shrank
        assert "declares 601 but is really 600" in shrank[0] and "shrank below" in shrank[0], shrank

        exact = _size_violations(tmp_path, {"governed.py": 600}, LIMIT)
        assert exact == [], exact

    def test_meta_flags_dead_entries_both_causes(self, tmp_path: Path) -> None:
        """(d) an entry for a now-under-limit file, and an entry for a missing file, are
        both reported by the dead-entry checker."""
        self._write_lines(tmp_path / "shrunk.py", 400)
        dead = _dead_allowlist_entries(tmp_path, {"shrunk.py": 400, "gone.py": 900}, LIMIT)
        assert len(dead) == 2, dead
        assert any("DEAD (file missing): gone.py" in d for d in dead), dead
        assert any("DEAD (now under 500): shrunk.py is 400 lines" in d for d in dead), dead
        assert all("DELETE the entry" in d for d in dead), dead

    def test_meta_dead_entry_checker_passes_a_live_governed_file(self, tmp_path: Path) -> None:
        """Negative control for (d): an over-limit file with an entry is not dead."""
        self._write_lines(tmp_path / "governed.py", 600)
        assert _dead_allowlist_entries(tmp_path, {"governed.py": 600}, LIMIT) == []
