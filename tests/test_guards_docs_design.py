"""Structural guards for design-doc code snippets (neograph-1h02l).

docs/design/ pages are NOT scanned by verifiable-docs Stage B/D (those glob
only ``website/src/content/docs``), so API drift in their fenced snippets
ships untested. The neograph-fsoss trace-similar sweep found the decorator
kwarg ``output=`` (singular — never a valid ``@node`` kwarg; the decorator
accepts only ``outputs=``) in ``agent-spec-api/09-patterns.md``.

This guard pins the narrow disease: a ``@node(...)`` call whose argument
list contains a singular ``output=`` kwarg. It does NOT attempt full
snippet execution over design docs — those are point-in-time artifacts;
this guard only bans the one kwarg that reads as valid API but is not.

``Construct(output=...)`` and ``ForwardConstruct`` singular ``output`` are
legitimate API and never match: the regex requires the kwarg inside a
``@node(`` call's parenthesis span.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DESIGN_DOCS_ROOT = REPO_ROOT / "docs" / "design"

# A @node( opener followed (same line) by a singular `output=`/`input=` kwarg
# (the decorator accepts only the plural forms). `\boutput\s*=` cannot match
# `outputs=` (the `s` breaks the `=` adjacency); likewise for `input=`.
_NODE_SINGULAR_OUTPUT = re.compile(r"@node\([^)]*\b(?:out|in)put\s*=")


class TestDesignDocsNodeKwargDrift:
    """`@node(output=)` / `@node(input=)` (singular) is not real API — design docs must not teach it."""

    def test_no_singular_io_kwarg_in_node_calls_when_scanning_design_docs(self):
        offenders: list[str] = []
        for path in sorted(DESIGN_DOCS_ROOT.rglob("*.md")):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if _NODE_SINGULAR_OUTPUT.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
        assert not offenders, (
            "@node() has no singular `output=`/`input=` kwarg (the decorator accepts "
            "only the plural forms). Fix these design-doc snippets:\n  "
            + "\n  ".join(offenders)
        )

    def test_slip_node_singular_output(self):
        """Regex-slip: the plural forms must never match — `outputs=`/`inputs=`
        break the `=` adjacency the pattern requires — while both singular
        kwargs match anywhere in the @node argument list, whitespace included."""
        assert _NODE_SINGULAR_OUTPUT.search("@node(output=SwarmResult)")
        assert _NODE_SINGULAR_OUTPUT.search('@node(mode="agent", output = NextTarget)')
        assert _NODE_SINGULAR_OUTPUT.search("@node(input=Claims)")
        # Plural forms are the real API — must NOT match.
        assert not _NODE_SINGULAR_OUTPUT.search("@node(outputs=SwarmResult)")
        assert not _NODE_SINGULAR_OUTPUT.search('@node(mode="agent", inputs={"a": A})')
        # `output=` outside a @node call (e.g. Construct(output=...)) is legit API.
        assert not _NODE_SINGULAR_OUTPUT.search("Construct(output=Summary, nodes=[...])")
