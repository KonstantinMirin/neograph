#!/usr/bin/env python3
"""Fail when a forward-port merge dropped documentation from the target branch.

The 0.7.8 forward-port carried every code change and silently discarded the
CHANGELOG sections for two shipped releases, the AGENTS.md lint documentation,
and a drafted upstream report -- conflicts resolved in favour of the target's
side without reading what the source side said. It was reported as verified on
the strength of a green test run, which cannot distinguish "the docs merged"
from "the docs were discarded".

This check answers the question the test suite structurally cannot:

    does the target branch still contain everything the source branch documented?

Two rules, both derived rather than listed:

1. Every ``## [X.Y.Z]`` release heading in the SOURCE CHANGELOG exists in the
   TARGET CHANGELOG. A release that shipped cannot lose its changelog.
2. Every ``docs/`` file that exists on the SOURCE exists on the TARGET. A design
   note or upstream draft is not deleted by moving forward.

Usage:
    python scripts/check_forward_port.py [--source origin/main] [--target HEAD]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

RELEASE_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)


def _show(ref: str, path: str) -> str | None:
    """File contents at *ref*, or None when the path does not exist there."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def _tracked_docs(ref: str) -> set[str]:
    """Every tracked path under docs/ at *ref*."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "docs/"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in result.stdout.splitlines() if line}


def check(source: str, target: str) -> list[str]:
    """Return one message per documentation loss; empty means the port is clean."""
    problems: list[str] = []

    source_changelog = _show(source, "CHANGELOG.md")
    target_changelog = _show(target, "CHANGELOG.md")
    if source_changelog is None or target_changelog is None:
        problems.append(f"CHANGELOG.md is missing on {source if source_changelog is None else target}")
    else:
        source_versions = set(RELEASE_HEADING.findall(source_changelog))
        target_versions = set(RELEASE_HEADING.findall(target_changelog))
        for version in sorted(source_versions - target_versions):
            problems.append(
                f"CHANGELOG.md on {target} has no [{version}] section, but {source} does.\n"
                f"    A shipped release cannot lose its changelog. Splice the section across."
            )

    for path in sorted(_tracked_docs(source) - _tracked_docs(target)):
        problems.append(
            f"{path} exists on {source} and not on {target}.\n"
            f"    Restore it, or state in the merge message why it was deliberately dropped."
        )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="origin/main", help="branch the port comes FROM")
    parser.add_argument("--target", default="HEAD", help="branch the port lands ON")
    args = parser.parse_args()

    problems = check(args.source, args.target)
    if not problems:
        print(f"forward-port check: {args.target} carries everything {args.source} documents.")
        return 0

    print(f"forward-port check FAILED: {len(problems)} documentation loss(es)\n")
    for problem in problems:
        print(f"  {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
