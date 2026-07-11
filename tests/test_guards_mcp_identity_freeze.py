"""Structural guard: no static Authorization stamp in the MCP battery (qslrx).

## Core Invariant
http MCP identity is ALWAYS per-request: it rides the transport as an
``httpx.Auth`` whose flow re-resolves the credential on every request. A bearer
written into a connection/headers dict is frozen for the connection's lifetime —
and because tools/sessions are cached per RUN_ID (or held open across
supersteps), a connect-time stamp silently outlives the IdP's token lifespan and
surfaces as a spurious ACCESS_DENIED mid-run (the neograph-qslrx incident: an
Each fan-out spanning Keycloak's 300s lifespan hit a denial storm on round 2).

The ONE sanctioned Authorization write is inside a per-request auth flow:
``request.headers["Authorization"] = ...`` in ``_token_provider_auth``'s
``async_auth_flow`` / ``sync_auth_flow`` — that write happens per request by
construction, so it cannot freeze.

The scan is AST-based (not regex), so quote style, f-strings, ``.update({...})``
dict forms, and aliased variables cannot slip past on formatting.

Non-vacuity is proven by the meta-tests, which feed planted violations (and the
sanctioned form) through the SAME scanner.
"""

from __future__ import annotations

import ast
import pathlib

MCP_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "neograph_mcp"


def _is_request_headers(node: ast.expr) -> bool:
    """True for the sanctioned per-request target: ``request.headers``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "headers"
        and isinstance(node.value, ast.Name)
        and node.value.id == "request"
    )


def _scan_authorization_stamps(source: str, filename: str = "<mem>") -> list[str]:
    """Every static Authorization write in ``source``, as ``file:line`` strings.

    Flags (a) subscript assignment ``<anything>["Authorization"] = ...`` unless
    the subscripted object is ``request.headers``, and (b) any dict literal
    carrying an ``"Authorization"`` key (covers ``headers.update({...})`` and
    connection-dict construction — no sanctioned dict-literal form exists).
    """
    offenders: list[str] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "Authorization"
                    and not _is_request_headers(target.value)
                ):
                    offenders.append(f"{filename}:{node.lineno}")
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "Authorization":
                    offenders.append(f"{filename}:{node.lineno}")
    return offenders


class TestNoStaticAuthorizationStampInBattery:
    def test_battery_has_no_static_authorization_stamp(self):
        """The real tree: src/neograph_mcp contains NO Authorization write
        outside the per-request ``request.headers`` auth-flow form."""
        offenders: list[str] = []
        for path in sorted(MCP_SRC.rglob("*.py")):
            rel = str(path.relative_to(MCP_SRC))
            offenders.extend(_scan_authorization_stamps(path.read_text(), rel))
        assert offenders == [], (
            "static Authorization stamp found in src/neograph_mcp — a bearer "
            "written into a connection/headers dict freezes for the connection "
            "lifetime and goes stale mid-run under the RUN_ID cache "
            "(neograph-qslrx). Route identity through _http_identity / "
            f"_token_provider_auth instead. Offenders: {offenders}"
        )

    def test_scanner_flags_planted_static_header_stamp(self):
        """Positive meta-test: the pre-fix disease form is caught."""
        planted = (
            "def _connection(spec, token):\n"
            "    headers = dict(spec.headers or {})\n"
            '    headers["Authorization"] = f"Bearer {token}"\n'
            "    return headers\n"
        )
        assert _scan_authorization_stamps(planted), "scanner missed the pre-fix static-bearer stamp"

    def test_scanner_flags_dict_literal_and_update_forms(self):
        """Would-be-missed meta-test: a naive ``\"Authorization\"] =`` regex slips
        on single quotes and on the ``.update({...})`` / dict-literal forms; the
        AST scan must catch all of them."""
        update_form = "headers.update({'Authorization': 'Bearer ' + token})\n"
        literal_form = 'conn = {"transport": "http", "headers": {"Authorization": token}}\n'
        assert _scan_authorization_stamps(update_form), "scanner missed the .update({...}) form"
        assert _scan_authorization_stamps(literal_form), "scanner missed the dict-literal form"

    def test_scanner_accepts_per_request_auth_flow_stamp(self):
        """Negative meta-test: the sanctioned per-request write inside an
        httpx.Auth flow is NOT flagged."""
        sanctioned = (
            "async def async_auth_flow(self, request):\n"
            "    token = await _resolve_token(provider, config)\n"
            '    request.headers["Authorization"] = f"Bearer {token}"\n'
            "    yield request\n"
        )
        assert _scan_authorization_stamps(sanctioned) == [], (
            "scanner wrongly flagged the sanctioned per-request request.headers stamp"
        )
