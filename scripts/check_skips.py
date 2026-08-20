"""Fail when a test skips for a reason nobody has signed off on.

A skip is invisible in a pass count. "3211 passed, 86 skipped" reads as success
while an entire shipped surface silently stops being exercised -- which is how
0.7.4 shipped a flaky live assertion (the Langfuse tests had skipped for want of
credentials) and how 0.7.7 shipped with ``neograph[mcp]`` -- 74 tests, a whole
second top-level package -- contributing nothing to the gate.

``release-gate`` therefore runs the suite with every extra installed, where the
expected number of skips is ZERO, and this script turns any skip into a hard
failure unless its reason appears in ``tests/skip_allowlist.txt``.

The allowlist is a ratchet: it may only shrink. Adding a line means writing down
why a surface is knowingly unexercised, which is the thing that was missing.

Usage:
    python scripts/check_skips.py [pytest args...]
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
ALLOWLIST = REPO / "tests" / "skip_allowlist.txt"

# "SKIPPED [3] tests/foo.py:12: some reason"
_SKIP_RE = re.compile(r"^SKIPPED\s+\[\d+\]\s+(?P<loc>\S+?):\s*(?P<reason>.*)$")


def _allowed() -> list[str]:
    if not ALLOWLIST.exists():
        return []
    out = []
    for raw in ALLOWLIST.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main(argv: list[str]) -> int:
    cmd = ["uv", "run", "--extra", "mcp", "--extra", "mcp-examples", "pytest", "-q", "-rs", *argv]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    output = proc.stdout + proc.stderr

    allowed = _allowed()
    offenders: list[str] = []
    for line in output.splitlines():
        match = _SKIP_RE.match(line.strip())
        if not match:
            continue
        reason = match.group("reason").strip()
        if not any(entry in reason for entry in allowed):
            offenders.append(f"{match.group('loc')}: {reason}")

    if proc.returncode not in (0, 1):  # 1 == test failures, reported by pytest itself
        print(output[-4000:])
        print(f"\ncheck_skips: pytest exited {proc.returncode}", file=sys.stderr)
        return proc.returncode
    if proc.returncode == 1:
        print(output[-4000:])
        print("\ncheck_skips: the suite itself failed; fix that first.", file=sys.stderr)
        return 1

    if offenders:
        print("\nUNALLOWLISTED SKIPS -- a surface is silently unexercised:\n", file=sys.stderr)
        for entry in sorted(set(offenders)):
            print(f"  {entry}", file=sys.stderr)
        print(
            f"\n{len(set(offenders))} distinct skip reason(s) not in {ALLOWLIST.relative_to(REPO)}."
            "\nEither make the test run (install the extra, provide the credential, fix the import)"
            "\nor add the reason to the allowlist WITH a justification comment. The allowlist may"
            "\nonly shrink -- growing it is a decision, not a formality.",
            file=sys.stderr,
        )
        return 1

    print("check_skips: no unallowlisted skips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
