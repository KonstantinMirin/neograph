"""RFC 6570 (subset) URI-template interpolation — a neutral leaf.

Extracted from ``tool.py`` neograph-a5nh so BOTH the typed ``resource_reader``
tool (tool.py) AND the ``FromResource`` DI resolver (di.py) can interpolate
templated URIs without a tool<->di import cycle. Stdlib-only; imports nothing
from ``neograph``.

Supports the subset the resource layer emits:
  * simple ``{var}`` and reserved ``{+var}`` string expansion;
  * form-query ``{?a,b}`` / continuation ``{&a}`` expansion.
Missing values are omitted (a form-query with no present vars collapses away).
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

# group 1 = operator, group 2 = comma-separated var list; trailing '*' explode
# modifier is stripped.
_URI_VAR_RE = re.compile(r"\{([+#./;?&]?)([^{}]+)\}")


def _extract_uri_vars(uri_template: str) -> list[str]:
    """Ordered, de-duplicated variable names from an RFC 6570 uri template."""
    names: list[str] = []
    for _op, body in _URI_VAR_RE.findall(uri_template):
        for raw in body.split(","):
            nm = raw.strip().rstrip("*")
            if nm and nm not in names:
                names.append(nm)
    return names


def _expand_uri(uri_template: str, values: dict[str, Any]) -> str:
    """Interpolate an RFC 6570 (subset) uri template from ``values``.

    Supports simple ``{var}`` / reserved ``{+var}`` string expansion and
    form-query ``{?a,b}`` / ``{&a}`` expansion — enough for the static and
    templated resource URIs the resource layer emits. Missing values are omitted.
    """

    def _sub(match: re.Match[str]) -> str:
        op, body = match.group(1), match.group(2)
        varnames = [v.strip().rstrip("*") for v in body.split(",")]
        if op in ("?", "&"):
            pairs = [f"{vn}={quote(str(values[vn]), safe='')}" for vn in varnames if values.get(vn) is not None]
            return (op + "&".join(pairs)) if pairs else ""
        out = [
            str(values[vn]) if op == "+" else quote(str(values[vn]), safe="")
            for vn in varnames
            if values.get(vn) is not None
        ]
        return ",".join(out)

    return _URI_VAR_RE.sub(_sub, uri_template)
