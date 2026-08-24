"""Fail when a test skips. Any test. For any reason.

A skip is invisible in a pass count. "3211 passed, 86 skipped" reads as success
while an entire shipped surface silently stops being exercised -- which is how
0.7.4 shipped a flaky live assertion (the Langfuse tests had skipped for want of
credentials) and how 0.7.7 shipped with ``neograph[mcp]`` -- 74 tests, a whole
second top-level package -- contributing nothing to the gate.

``release-gate`` runs the suite with every extra installed, where the expected
number of skips is ZERO, and this script turns any skip into a hard failure.

THERE IS NO ALLOWLIST, deliberately. This script shipped with one
(``tests/skip_allowlist.txt``, "empty by design, may only shrink") and it was
removed four days later without a single entry ever being added, because the
escape hatch cannot be justified: a test exists to verify a behaviour, so a test
that does not run is a defect with a cause, and writing its reason into a file
does not fix the cause -- it only stops anyone being told about it.

When this fails, there are exactly two honest answers:

  * Make the test run. Install the extra, provide the credential, fix the
    import, build the fixture.
  * If the behaviour is known-broken and the test cannot pass yet, mark it
    ``xfail(strict=True)`` -- which is reported distinctly from a pass AND
    turns red the moment the gap closes, so the exemption cannot outlive its
    reason. That is the property an allowlist can never have.

Usage:
    python scripts/check_skips.py [pytest args...]
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# "SKIPPED [3] tests/foo.py:12: some reason"
_SKIP_RE = re.compile(r"^SKIPPED\s+\[\d+\]\s+(?P<loc>\S+?):\s*(?P<reason>.*)$")


def main(argv: list[str]) -> int:
    cmd = ["uv", "run", "--extra", "mcp", "--extra", "mcp-examples", "pytest", "-q", "-rs", *argv]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    output = proc.stdout + proc.stderr

    offenders: list[str] = []
    for line in output.splitlines():
        match = _SKIP_RE.match(line.strip())
        if match:
            offenders.append(f"{match.group('loc')}: {match.group('reason').strip()}")

    if proc.returncode not in (0, 1):  # 1 == test failures, reported by pytest itself
        print(output[-4000:])
        print(f"\ncheck_skips: pytest exited {proc.returncode}", file=sys.stderr)
        return proc.returncode
    if proc.returncode == 1:
        print(output[-4000:])
        print("\ncheck_skips: the suite itself failed; fix that first.", file=sys.stderr)
        return 1

    if offenders:
        print("\nSKIPPED TESTS -- a behaviour is not being verified:\n", file=sys.stderr)
        for entry in sorted(set(offenders)):
            print(f"  {entry}", file=sys.stderr)
        print(
            f"\n{len(set(offenders))} distinct skip reason(s). There is no allowlist."
            "\nEither make the test run, or -- if the behaviour is known-broken -- mark it"
            "\nxfail(strict=True), which is reported distinctly and turns RED when the gap"
            "\ncloses, so the exemption cannot outlive its reason.",
            file=sys.stderr,
        )
        return 1

    print("check_skips: no skipped tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
