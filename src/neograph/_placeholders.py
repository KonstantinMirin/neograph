"""The ONE placeholder-scanner core, shared by every template-var consumer.

Three consumers use this single scanner with three different resolver policies:

- ``_llm_render`` — inline ``${var}`` prompts, fail-SOFT (log + empty string).
- ``lint`` — ``${var}`` name collection for placeholder validation.
- ``prompt`` (public primitives) — ``substitute`` with a fail-LOUD strict policy.

Only the token grammar and the single-pass apply live here; the resolver (what a
matched name maps to, and what a miss does) is supplied per consumer. This is the
anti-duplication invariant for hjwv: one substitution rule, one scanner — never a
parallel second rendering path (the class of bug that shipped a literal ``{domain}``
to a production LLM when two apps hand-rolled divergent substitution seams).

Engine-free by construction: no langgraph, no pydantic — a neutral leaf importable
from every layer.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# ``${var}`` / ``${var.field}`` — the dollar-brace grammar neograph uses for
# inline prompts. Brace-safe: matches only ``${...}``, so literal ``{}`` in the
# template or in a substituted value is left untouched.
DOLLAR_RE = re.compile(r"\$\{([^}]+)\}")

# ``{var}`` / ``{var.field}`` — the brace grammar. Token-only: the name must start
# with an identifier character, so a bare ``{}`` and a JSON fragment like
# ``{ "k": 1 }`` (space/quote after the brace) do NOT match. That, together with
# single-pass application, is what lets an injected JSON schema render intact.
BRACE_RE = re.compile(r"\{([a-zA-Z_][\w.]*)\}")


def apply_scanner(
    template: str,
    pattern: re.Pattern[str],
    resolve: Callable[[str], str],
) -> str:
    """Single-pass token substitution: replace each match via *resolve*.

    ``re.sub`` walks the string once and never re-scans the text it just
    substituted, so a value that itself contains ``{...}`` (a JSON schema) is
    emitted verbatim rather than re-parsed. This single pass IS the brace-safety
    mechanism — ``str.format`` / ``.format_map`` reparse and would crash on it.
    """
    return pattern.sub(lambda m: resolve(m.group(1)), template)
