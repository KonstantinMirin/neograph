"""Structural guard: every ``neograph_mcp`` transport-call exit boundary unwraps.

The bare-leaf invariant (neograph-2itlh): every transport-touching ``await``
(``get_tools`` / ``load_mcp_tools`` / ``call_tool`` / ``list_tools`` /
``read_resource`` / ``get_prompt``) must be inside a ``try`` whose ``except``
handler raises ``_unwrap_single(...)`` on exit — so a single-leaf transport
``ExceptionGroup`` surfaces as its bare leaf, not a grouped wrapper.

Two sanctioned handler forms, by boundary shape (neograph-lcrwd):
- in-place: ``raise _unwrap_single(exc) from None`` (held-session boundaries —
  no CM exit between the raise and the consumer);
- capture-then-reraise: ``leaf = _unwrap_single(exc)`` in the handler with the
  bare leaf re-raised AFTER the ``async with`` closes (consumer-owned-session
  boundaries — an in-scope raise would be RE-wrapped by the anyio teardown).

RATCHETING guard with a shrinking allowlist. The allowlist holds known-DEFERRED
sites: transport calls not yet unwrap-guarded. It may only SHRINK — wrapping a
deferred site MUST remove it here. A NEW unwrapped transport call anywhere in
the battery fails the guard (it is not in the allowlist); an allowlisted site
that gets wrapped also fails (the allowlist must shrink). As of neograph-lcrwd
the allowlist is EMPTY — every battery boundary unwraps.

Scope: the NAMED transport-method boundaries. The ``_resilient`` tool-call
wrapper (which wraps ``orig(**kwargs)``, not a named MCP method) is covered
BEHAVIOURALLY by ``tests/test_mcp_transport_resilience.py::
TestGroupedExceptionUnwrappedToBareLeaf``; this guard does not reach it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BATTERY_DIR = Path(__file__).resolve().parent.parent / "src" / "neograph_mcp"

# Transport-touching methods whose raise is wrapped by the transport's anyio
# TaskGroup — the boundaries that must unwrap on exit.
_TRANSPORT_METHODS = frozenset(
    {"get_tools", "load_mcp_tools", "call_tool", "list_tools", "read_resource", "get_prompt"}
)

# Known-DEFERRED sites: transport calls NOT yet unwrap-guarded, as
# (module_stem, enclosing_function, method). EMPTY since neograph-lcrwd wrapped
# the last four (_session call/_ensure_listing, _client fetcher/replayer) — the
# allowlist may only shrink, so it stays empty.
_DEFERRED_ALLOWLIST: frozenset[tuple[str, str, str]] = frozenset()


def _is_unwrap_call(expr: ast.expr | None) -> bool:
    return (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "_unwrap_single"
    )


def _has_unwrap_raise(node: ast.AST) -> bool:
    """True if ``node``'s subtree normalises via ``_unwrap_single`` — either the
    in-place ``raise _unwrap_single(...)`` or the capture-then-reraise form's
    ``leaf = _unwrap_single(...)`` (the leaf is re-raised after the ``async
    with`` closes, outside the handler)."""
    for n in ast.walk(node):
        if isinstance(n, ast.Raise) and _is_unwrap_call(n.exc):
            return True
        if isinstance(n, ast.Assign) and _is_unwrap_call(n.value):
            return True
    return False


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _enclosing_function(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(id(cur))
    return "<module>"


def _is_wrapped(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """True if ``node`` is inside a ``Try`` whose handler raises
    ``_unwrap_single``. Walks ALL ancestor tries (handles nesting)."""
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.Try):
            for handler in cur.handlers:
                if _has_unwrap_raise(handler):
                    return True
        cur = parents.get(id(cur))
    return False


def unwrap_violations(source: str) -> list[tuple[str, str, int]]:
    """Pure: return ``[(function, method, line)]`` for transport ``await`` calls
    NOT wrapped by a ``_unwrap_single`` exit handler. The caller applies the
    ratcheting allowlist so meta-tests can inspect the raw violation set."""
    tree = ast.parse(source)
    parents = _parent_map(tree)
    out: list[tuple[str, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Attribute):
            method = call.func.attr
        elif isinstance(call.func, ast.Name):
            method = call.func.id
        else:
            continue
        if method not in _TRANSPORT_METHODS:
            continue
        if _is_wrapped(node, parents):
            continue
        out.append((_enclosing_function(node, parents), method, node.lineno))
    return out


# ── the ratchet: every battery transport call unwraps or is deferred ─────────


@pytest.mark.parametrize("module", ["_client.py", "_session.py", "_prompt.py", "_run_context.py"])
def test_transport_calls_unwrap_or_are_allowlisted(module: str) -> None:
    raw = (_BATTERY_DIR / module).read_text()
    stem = module[:-3]
    violations = {(fn, method) for fn, method, _line in unwrap_violations(raw)}
    allowed = {(fn, method) for m, fn, method in _DEFERRED_ALLOWLIST if m == stem}

    unexpected = violations - allowed
    assert not unexpected, (
        f"{module}: NEW unwrapped transport calls (not in the neograph-lcrwd allowlist) — "
        "wrap them with `try: ... except BaseException as exc: raise _unwrap_single(exc) from None` "
        "or add a documented deferral row to _DEFERRED_ALLOWLIST:\n"
        + "\n".join(f"  {fn}::{method}" for fn, method in sorted(unexpected))
    )

    # Ratchet direction: an allowlisted site that is NOW wrapped must be REMOVED
    # from the allowlist (allowlists may only shrink).
    stale = allowed - violations
    assert not stale, (
        f"{module}: these allowlisted sites are now WRAPPED — remove them from "
        "_DEFERRED_ALLOWLIST (allowlists may only shrink):\n"
        + "\n".join(f"  {fn}::{method}" for fn, method in sorted(stale))
    )


# ── meta-tests: the detector catches the disease, including would-be-missed ──


def test_meta_wrapped_get_tools_is_not_a_violation() -> None:
    """Positive: a transport call inside a try whose except raises
    ``_unwrap_single`` is NOT a violation."""
    src = (
        "async def f(client):\n"
        "    try:\n"
        "        tools = await client.get_tools(server_name='x')\n"
        "    except BaseException as exc:\n"
        "        raise _unwrap_single(exc) from None\n"
        "    return tools\n"
    )
    assert unwrap_violations(src) == []


def test_meta_unwrapped_get_tools_is_a_violation() -> None:
    """Negative: a bare transport call with no surrounding unwrap try is a
    violation."""
    src = (
        "async def f(client):\n"
        "    tools = await client.get_tools(server_name='x')\n"
        "    return tools\n"
    )
    assert unwrap_violations(src) == [("f", "get_tools", 2)]


def test_meta_try_with_non_unwrap_except_is_still_a_violation() -> None:
    """Would-be-missed: a transport call inside a try whose except does NOT raise
    ``_unwrap_single`` (e.g. the fetcher's ``except McpError as exc: error = exc``)
    is STILL a violation — the guard checks the except UNWRAPS, not just that a
    try exists. A guard that only checked 'is inside a try' would miss this."""
    src = (
        "async def f(session):\n"
        "    try:\n"
        "        result = await session.read_resource(uri)\n"
        "    except McpError as exc:\n"
        "        error = exc\n"
        "    return error\n"
    )
    assert unwrap_violations(src) == [("f", "read_resource", 3)]


def test_meta_capture_then_reraise_is_not_a_violation() -> None:
    """Positive: the consumer-owned-session form — the handler CAPTURES the
    unwrapped leaf (``leaf = _unwrap_single(exc)``) and re-raises it after the
    ``async with`` closes (an in-scope raise would be re-wrapped by the anyio
    teardown) — satisfies the invariant."""
    src = (
        "async def f(client, uri):\n"
        "    error = None\n"
        "    async with client.session('x') as session:\n"
        "        try:\n"
        "            result = await session.read_resource(uri)\n"
        "        except BaseException as exc:\n"
        "            leaf = _unwrap_single(exc)\n"
        "            error = leaf\n"
        "    if error is not None:\n"
        "        raise error\n"
        "    return result\n"
    )
    assert unwrap_violations(src) == []


def test_meta_bare_function_transport_call_is_detected() -> None:
    """``load_mcp_tools(...)`` is a bare ``Name`` call (not ``obj.method``); the
    detector must still recognise it as a transport call."""
    src = (
        "async def f(held):\n"
        "    tools = await load_mcp_tools(held, server_name='x')\n"
        "    return tools\n"
    )
    assert unwrap_violations(src) == [("f", "load_mcp_tools", 2)]
