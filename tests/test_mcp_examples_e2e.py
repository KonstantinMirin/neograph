"""End-to-end integration harness for the MCP examples (neograph-g4q9).

Two things live here:

1. A **server-contract E2E** — spawns ``examples/_mcp_demo_server.py`` as a real
   stdio subprocess via ``MultiServerMCPClient`` and asserts the actual Model
   Context Protocol behaviors the examples depend on: tool discovery, the
   ``get_deal`` resource_link manifest, the path-segment email fraction read, the
   per-operator auth echo, and the one-shot expiry (plus the self-heal that
   follows). The server is PLAIN FastMCP — copy-paste-safe, no low-level
   handlers — so the expired resource surfaces as FastMCP's own code-0-wrapped
   error, not a hand-crafted ``-32002``; neograph heals code-agnostically. This
   proves the shared server end-to-end, offline.

2. A **reusable example runner** — ``run_example_subprocess(path)`` runs a full
   example file as a subprocess and asserts a clean exit. The parametrized
   ``test_mcp_example_runs_end_to_end`` discovers ``examples/2?_mcp_*.py`` and
   runs each. Examples 23/24 (neograph-qb7q / neograph-3m6g) drop in and are
   covered automatically; until then the discovery yields nothing and the test
   skips.

These tests need the ``mcp-examples`` extra (``mcp`` + ``langchain-mcp-adapters``).
They are keyless and offline but NOT dependency-light, so the whole module is
gated with ``pytest.importorskip`` — the core ``uv run --extra dev pytest`` suite
stays light and skips this file. Run the MCP E2Es with::

    uv run --extra dev --extra mcp-examples pytest tests/test_mcp_examples_e2e.py
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Gate the whole module on the extra — keeps the core suite dependency-light.
pytest.importorskip("mcp", reason="requires the mcp-examples extra")
pytest.importorskip("langchain_mcp_adapters", reason="requires the mcp-examples extra")

from pydantic import AnyUrl  # noqa: E402  (after importorskip by design)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
DEMO_SERVER = EXAMPLES_DIR / "_mcp_demo_server.py"


def _demo_connection(state_file: Path) -> dict:
    """Build the StdioConnection dict that spawns the demo server as a subprocess.

    Shared by every test (and mirrored by examples 23/24): the demo server is a
    child process, its expiry-marker path pinned via ``env`` so every spawn — the
    arming call and the later read live in DIFFERENT subprocesses — agrees on it.
    """
    return {
        "command": sys.executable,
        "args": [str(DEMO_SERVER)],
        "transport": "stdio",
        "env": {"NEOGRAPH_MCP_DEMO_STATE": str(state_file)},
    }


class TestDemoServerContract:
    """The shared FastMCP demo server exercised over real stdio MCP."""

    async def test_tools_are_discoverable(self, tmp_path):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({"crm": _demo_connection(tmp_path / "state.marker")})
        tools = await client.get_tools()
        names = {t.name for t in tools}
        assert {"crm_search", "kb_lookup", "get_deal", "update_deal", "arm_email_expiry"} <= names

    async def test_get_deal_returns_resource_link_manifest(self, tmp_path):
        """get_deal returns the deal text PLUS resource_link blocks — the manifest
        example 24 lifts (surfaced by langchain-mcp-adapters as 'file' blocks)."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({"crm": _demo_connection(tmp_path / "state.marker")})
        tools = await client.get_tools()
        get_deal = next(t for t in tools if t.name == "get_deal")
        blocks = await get_deal.ainvoke({"deal_id": "D1", "token": "operator-A"})

        assert isinstance(blocks, list)
        links = [b for b in blocks if isinstance(b, dict) and b.get("type") == "file"]
        urls = {b.get("url") for b in links}
        assert any(u.endswith("/D1/activity") for u in urls)
        assert any("/D1/emails/" in u for u in urls)

    async def test_read_email_fraction_via_path_template(self, tmp_path):
        """The path-segment ``emails/{start}/{end}`` fraction read returns only
        in-range emails — the 'query a fraction of a corpus with a typed result'
        beat. Path segments are what plain FastMCP's ``@mcp.resource`` supports
        natively (query templates are not expressible — see the server docstring)."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({"crm": _demo_connection(tmp_path / "state.marker")})
        async with client.session("crm") as session:
            result = await session.read_resource(AnyUrl("mcp://crm/deals/D1/emails/2024-04-01/2024-12-31"))
        import json

        payload = json.loads(result.contents[0].text)
        # D1 has emails on 02-10, 04-22, 06-30 — the start=2024-04-01 fraction drops the first.
        stamps = {e["ts"] for e in payload["emails"]}
        assert stamps == {"2024-04-22", "2024-06-30"}

    async def test_resource_templates_advertise_path_form(self, tmp_path):
        """Plain FastMCP auto-advertises the parameterized resources as templates —
        no custom ``list_resource_templates`` handler."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({"crm": _demo_connection(tmp_path / "state.marker")})
        async with client.session("crm") as session:
            listed = await session.list_resource_templates()
        forms = {t.uriTemplate for t in listed.resourceTemplates}
        assert "mcp://crm/deals/{deal_id}/emails/{start}/{end}" in forms
        assert "mcp://crm/deals/{deal_id}/activity" in forms

    async def test_auth_token_is_echoed_per_operator(self, tmp_path):
        """Two runs with different tokens observably carry their own identity —
        the per-operator beat (example 23)."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({"crm": _demo_connection(tmp_path / "state.marker")})
        tools = await client.get_tools()
        crm_search = next(t for t in tools if t.name == "crm_search")

        import json

        a = json.loads((await crm_search.ainvoke({"query": "acme", "token": "operator-A"}))[0]["text"])
        b = json.loads((await crm_search.ainvoke({"query": "acme", "token": "operator-B"}))[0]["text"])
        assert a["acting_as"] == "operator-A"
        assert b["acting_as"] == "operator-B"

    async def test_expiry_error_is_code_agnostic_then_self_heals(self, tmp_path):
        """Arm the one-shot expiry (server-side), read once -> error, read again
        -> success. This is example 24's self-healing hydration beat, and the
        PRIMARY expiry path now that the server is plain FastMCP.

        Plain FastMCP's ``@mcp.resource`` wrapper swallows JSON-RPC error codes: a
        handler-raised exception reaches the client as a generic ``code: 0``, NOT
        a hand-crafted ``-32002``. That is the point — neograph's replay-on-any-
        fetch-failure heal (di.py hydrate_resource_ref) is code-agnostic, so the
        demo does NOT need a low-level handler forging a specific code."""
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from mcp.shared.exceptions import McpError

        client = MultiServerMCPClient({"crm": _demo_connection(tmp_path / "state.marker")})

        async with client.session("crm") as session:
            await session.call_tool("arm_email_expiry", {})

        # Catch INSIDE the session: the session's anyio task-group teardown wraps
        # any exception escaping the `async with` in an ExceptionGroup, so the
        # McpError is only observable in-scope. This is exactly how example 24's
        # fetcher catches the expiry to trigger self-heal — on ANY fetch failure,
        # not a specific code.
        async with client.session("crm") as session:
            with pytest.raises(McpError) as excinfo:
                await session.read_resource(AnyUrl("mcp://crm/deals/D1/emails/2024-01-01/2024-12-31"))
        # code-0-wrapped by plain FastMCP (the swallowed-code SDK gap) — the heal
        # below does NOT depend on this code, proving code-agnostic robustness.
        assert excinfo.value.error.code == 0

        # One-shot: the very next read succeeds (the replay/self-heal path).
        async with client.session("crm") as session:
            healed = await session.read_resource(AnyUrl("mcp://crm/deals/D1/emails/2024-01-01/2024-12-31"))
        assert healed.contents[0].text  # non-empty typed payload


# ── http transport (neograph-jahj: bearer-header path, untested until now) ────


def _free_port() -> int:
    """Reserve an ephemeral localhost port for the http demo server.

    Bind-then-close: a small, accepted race (standard test-harness practice) —
    the OS won't hand the port back out before the subprocess below binds it.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_listening(host: str, port: int, timeout: float = 10.0) -> None:
    """Poll until the demo server's http listener accepts a TCP connection."""
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f"demo http server never listened on {host}:{port}") from last_error


@contextlib.contextmanager
def _demo_http_server(state_file: Path):
    """Launch the demo server in http mode on an ephemeral 127.0.0.1 port.

    Mirrors ``_demo_connection`` (stdio) for the http transport: still no real
    network, still CI-safe, just a different launch mode of the SAME server file.
    """
    port = _free_port()
    env = {
        **os.environ,
        "NEOGRAPH_MCP_DEMO_TRANSPORT": "http",
        "NEOGRAPH_MCP_DEMO_HTTP_PORT": str(port),
        "NEOGRAPH_MCP_DEMO_STATE": str(state_file),
    }
    proc = subprocess.Popen(
        [sys.executable, str(DEMO_SERVER)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(REPO_ROOT),
    )
    try:
        _wait_for_listening("127.0.0.1", port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


class TestDemoServerHttpContract:
    """The shared FastMCP demo server exercised over real streamable-http MCP.

    Mirrors ``TestDemoServerContract`` (stdio) for the transport that ships
    UNTESTED today: bearer auth via the ``Authorization`` header, instead of
    the stdio ``token`` tool argument.
    """

    async def test_tools_are_discoverable_over_http(self, tmp_path):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        with _demo_http_server(tmp_path / "state.marker") as url:
            client = MultiServerMCPClient({"crm": {"transport": "streamable_http", "url": url}})
            tools = await client.get_tools()
            names = {t.name for t in tools}
            assert {"crm_search", "kb_lookup", "get_deal", "update_deal", "arm_email_expiry"} <= names

    async def test_bearer_auth_is_echoed_per_operator(self, tmp_path):
        """Two clients with DIFFERENT bearer tokens observably carry their own
        identity through ``bearer_identity`` — the http counterpart of
        ``test_auth_token_is_echoed_per_operator`` (stdio's ``token`` argument
        beat). Uses two separate clients since the bearer header is set once
        per ``MultiServerMCPClient`` connection, not per call."""
        import json

        from langchain_mcp_adapters.client import MultiServerMCPClient

        with _demo_http_server(tmp_path / "state.marker") as url:
            client_a = MultiServerMCPClient(
                {"crm": {"transport": "streamable_http", "url": url, "headers": {"Authorization": "Bearer operator-A"}}}
            )
            client_b = MultiServerMCPClient(
                {"crm": {"transport": "streamable_http", "url": url, "headers": {"Authorization": "Bearer operator-B"}}}
            )
            tools_a = await client_a.get_tools()
            tools_b = await client_b.get_tools()
            crm_search_a = next(t for t in tools_a if t.name == "crm_search")
            crm_search_b = next(t for t in tools_b if t.name == "crm_search")

            a = json.loads((await crm_search_a.ainvoke({"query": "acme"}))[0]["text"])
            b = json.loads((await crm_search_b.ainvoke({"query": "acme"}))[0]["text"])

        assert a["bearer_identity"] == "operator-A"
        assert b["bearer_identity"] == "operator-B"
        # http carries no token tool-argument from the client above, so both
        # sides fall back to the tool's own default — proving identity rode the
        # header, not a tool argument the caller never sent.
        assert a["acting_as"] == "anon"
        assert b["acting_as"] == "anon"


# ── Reusable example runner (examples 23/24 plug in here) ─────────────────────


def run_example_subprocess(path: Path, *, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run an example FILE end-to-end as a subprocess and return the result.

    Uses the same interpreter (already has the extra installed when this module
    collected). Examples spawn the demo server as their OWN child, so this is a
    process tree: pytest -> example -> demo server.
    """
    return subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def _discover_mcp_examples() -> list[Path]:
    """MCP-featuring example files: examples/2?_mcp_*.py (23, 24, ...)."""
    return sorted(EXAMPLES_DIR.glob("2?_mcp_*.py"))


@pytest.mark.parametrize("example", _discover_mcp_examples(), ids=lambda p: p.name)
def test_mcp_example_runs_end_to_end(example: Path):
    """Every MCP example runs clean, offline, keyless. Skips until 23/24 land."""
    result = run_example_subprocess(example)
    assert result.returncode == 0, f"{example.name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def test_example_runner_reports_no_examples_when_none_present():
    """Non-vacuity: if discovery is empty the parametrized test above collects
    zero cases (a silent pass). This meta-test makes that state explicit so a
    reviewer knows coverage is pending the example tasks, not broken."""
    discovered = _discover_mcp_examples()
    if not discovered:
        pytest.skip("no examples/2?_mcp_*.py yet — coverage arrives with neograph-qb7q / neograph-3m6g")
    assert all(p.exists() for p in discovered)
