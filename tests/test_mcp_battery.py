"""Smoke tests for the optional ``neograph[mcp]`` client DX battery (neograph-xdt2).

The battery (``src/neograph_mcp``) graduates the consumer-side MCP client stitching
demonstrated by the g4q9 examples — MultiServerMCPClient + transport config +
token-provider seam -> ``tool_factories`` — into a shipped, overridable helper,
mirroring the ``PromptCompiler`` Protocol + ``DefaultPromptCompiler`` seam-plus-battery
precedent. It lives OUTSIDE ``src/neograph`` so the no-session-ownership guard
(which scans ``src/neograph`` only) stays green and core stays MCP-free.

Two tiers here:

1. **Always-on** (no extra): ``import neograph`` drags in zero MCP deps, and
   ``import neograph_mcp`` without the extra fails loud with an install hint. These
   run under the light ``uv run --extra dev pytest`` suite.
2. **Extra-gated** one-consumer smoke: bind ONE tool from a stdio server via
   ``mcp_tool_factories`` -> ``compile()`` passes -> ``lint()`` flags the async
   binding -> an ``arun()`` call drives the real per-run tool against the FastMCP
   demo server -> swapping the factory for a fake double keeps the fast tier
   deterministic. Gated with ``skipif`` so the light suite skips it cleanly.

Run the gated tier with::

    uv run --extra dev --extra mcp pytest tests/test_mcp_battery.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_SERVER = REPO_ROOT / "examples" / "_mcp_demo_server.py"

_HAS_MCP = bool(importlib.util.find_spec("mcp")) and bool(importlib.util.find_spec("langchain_mcp_adapters"))
requires_mcp = pytest.mark.skipif(not _HAS_MCP, reason="requires the mcp extra (mcp + langchain-mcp-adapters)")


# ── Always-on: core stays MCP-free, battery fails loud without the extra ──────


def test_neograph_core_imports_with_zero_mcp_deps():
    """Importing neograph core must not drag in mcp / langchain_mcp_adapters.

    Runs in a subprocess so the assertion sees a clean interpreter (this test
    process may itself have the extra installed). Proves the extra is genuinely
    optional and core is MCP-free — the whole packaging premise of this task.
    """
    code = (
        "import sys\n"
        "import neograph\n"
        "leaked = [m for m in sys.modules if m == 'mcp' or m.split('.')[0] in ('mcp', 'langchain_mcp_adapters')]\n"
        "assert not leaked, f'neograph core dragged in MCP deps: {leaked}'\n"
        "print('CORE_MCP_FREE_OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, f"core-is-mcp-free check failed:\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    assert "CORE_MCP_FREE_OK" in result.stdout


def test_importing_neograph_mcp_without_extra_fails_loud():
    """``import neograph_mcp`` with mcp/adapters unavailable must raise a clear
    ImportError naming the ``neograph[mcp]`` extra — the langfuse-observe fail-loud
    precedent. A meta-path blocker simulates the missing extra so this runs even
    when the extra IS installed in the test environment.
    """
    code = (
        "import sys, importlib.abc\n"
        "class _Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in ('mcp', 'langchain_mcp_adapters'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "try:\n"
        "    import neograph_mcp\n"
        "except ImportError as e:\n"
        "    assert 'neograph[mcp]' in str(e), f'hint missing from error: {e!r}'\n"
        "    print('FAILED_LOUD_OK')\n"
        "    sys.exit(0)\n"
        "sys.exit('import neograph_mcp did NOT fail loud without the extra')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, f"fail-loud check failed:\nSTDOUT:{result.stdout}\nSTDERR:{result.stderr}"
    assert "FAILED_LOUD_OK" in result.stdout


# ── Extra-gated one-consumer smoke ────────────────────────────────────────────


def _demo_stdio_server():
    """A StdioServer pointed at the shared FastMCP demo server (offline, keyless)."""
    from neograph_mcp import StdioServer

    return StdioServer(command=sys.executable, args=[str(DEMO_SERVER)])


def _build_agent_construct(tool_name: str):
    """A one-node agent construct binding exactly ``tool_name`` (least-privilege)."""
    from neograph import Construct, Node
    from neograph.tool import Tool, ToolInteraction
    from tests.schemas import Claims

    return Construct(
        "crm_pipeline",
        nodes=[
            Node(
                "scan",
                mode="agent",
                outputs={"result": Claims, "tool_log": list[ToolInteraction]},
                model="fast",
                prompt="test/scan",
                tools=[Tool(tool_name, budget=0)],
            ),
        ],
    )


def _react_fake(tool_name: str):
    from tests.fakes import ReActFake
    from tests.schemas import Claims

    return ReActFake(
        tool_calls=[
            [{"name": tool_name, "args": {"query": "acme"}, "id": "c1"}],
            [],
        ],
        final=lambda m: m(items=["done"]),
        output_model=Claims,
    )


@requires_mcp
class TestOneConsumerSmoke:
    """Bind one stdio tool via the battery and drive it end-to-end."""

    def test_tool_factories_enumerates_and_namespaces(self):
        """``mcp_tool_factories`` connects once, returns a DICT keyed per tool so a
        consumer can slice it per node. namespace=True keys as ``server::tool``."""
        from neograph_mcp import mcp_tool_factories

        factories = mcp_tool_factories({"crm": _demo_stdio_server()}, namespace=True)
        assert "crm::crm_search" in factories
        assert "crm::update_deal" in factories
        # Every value is an async factory (arun-driven per-run tool build).
        assert all(asyncio.iscoroutinefunction(f) for f in factories.values())

        flat = mcp_tool_factories({"crm": _demo_stdio_server()}, namespace=False)
        assert "crm_search" in flat and "crm::crm_search" not in flat

    def test_compile_lint_and_arun_through_the_battery(self):
        """The full one-consumer path: build factories -> slice ONE tool -> compile
        passes -> lint flags the async binding -> arun drives the REAL tool against
        the demo server carrying per-run identity."""
        import neograph
        from neograph import compile, lint
        from neograph_mcp import mcp_tool_factories
        from tests.fakes import build_test_compile_kwargs, configure_fake_llm
        from tests.schemas import Claims

        # token_provider mints per-run identity from config['configurable'].
        factories = mcp_tool_factories(
            {"crm": _demo_stdio_server()},
            token_provider=lambda configurable: configurable.get("op", "anon"),
            namespace=False,
        )
        # Least-privilege: bind ONLY crm_search, not the whole server.
        only_search = {"crm_search": factories["crm_search"]}

        construct = _build_agent_construct("crm_search")
        llm_kw = configure_fake_llm(lambda tier: _react_fake("crm_search"))
        graph = compile(construct, tool_factories=only_search, **build_test_compile_kwargs(), **llm_kw)

        # lint validates the binding: an async factory needs the arun() driver.
        issues = lint(construct, tool_factories=only_search)
        assert any(i.kind == "tool_requires_async_driver" and "crm_search" in i.param for i in issues), (
            f"lint did not flag the async MCP tool binding: {[i.kind for i in issues]}"
        )

        cfg = {"configurable": {"op": "operator-A"}}
        result = asyncio.run(neograph.arun(graph, input={"query": "acme"}, config=cfg))

        assert result["scan_result"] == Claims(items=["done"])
        tool_log = result["scan_tool_log"]
        assert tool_log, "expected the real MCP tool to have been called"
        # The demo echoes the per-run token under acting_as — proves per-run
        # identity rode through the battery to the real server as a stdio tool arg.
        assert "operator-A" in repr(tool_log[0].typed_result) or "operator-A" in tool_log[0].result

    def test_fake_double_swaps_in_for_deterministic_fast_tier(self):
        """The seam is overridable: a fake async factory drops in for the real one
        and keeps the fast tier deterministic (no subprocess, no protocol)."""
        from langchain_core.tools import StructuredTool

        import neograph
        from neograph import compile
        from tests.fakes import build_test_compile_kwargs, configure_fake_llm
        from tests.schemas import Claims

        calls: list[dict] = []

        async def _fake_factory(config, tool_config):
            def _run(query: str) -> str:
                calls.append({"query": query})
                return "acting_as=operator-A hits=[]"

            return StructuredTool.from_function(func=_run, name="crm_search", description="fake crm search")

        construct = _build_agent_construct("crm_search")
        llm_kw = configure_fake_llm(lambda tier: _react_fake("crm_search"))
        graph = compile(
            construct,
            tool_factories={"crm_search": _fake_factory},
            **build_test_compile_kwargs(),
            **llm_kw,
        )
        result = asyncio.run(neograph.arun(graph, input={"query": "acme"}))
        assert result["scan_result"] == Claims(items=["done"])
        assert calls == [{"query": "acme"}], "fake double was not driven deterministically"


@requires_mcp
def test_resource_fetcher_builder_returns_fetcher_and_replayer():
    """``mcp_resource_fetcher`` returns the (fetcher, replayer) callables a consumer
    drops into config so FromResource hydration + layered-expiry replay work out of
    the box. Shape-only smoke (the resource URIs are being reworked concurrently by
    the 8wbr teammate — this stays on the stable builder contract, not URI specifics).
    """
    from neograph_mcp import mcp_resource_fetcher

    fetcher, replayer = mcp_resource_fetcher({"crm": _demo_stdio_server()})
    assert asyncio.iscoroutinefunction(fetcher)
    assert asyncio.iscoroutinefunction(replayer)
