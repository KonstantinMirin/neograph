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
import contextlib
import importlib.util
import os
import socket
import subprocess
import sys
import time
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


# ── http transport (neograph-jahj: HttpServer/bearer path, untested until now) ─


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
    """Launch the demo server in http mode on an ephemeral 127.0.0.1 port — the
    HttpServer counterpart of ``_demo_stdio_server``, same server file."""
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


def _demo_http_server_spec(url: str):
    from neograph_mcp import HttpServer

    return HttpServer(url=url)


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
class TestLazySingleToolFactory:
    """The public singular ``mcp_tool_factory`` helper (neograph-g2jg).

    A thin lazy wrapper over the private ``_make_tool_factory``: a consumer binding
    ONE gateway-federated tool into a node with a fixed bare ``Tool(name)`` binding
    gets a single async ToolFactory that (a) connects NOWHERE at construction and
    (b) renames the discovered (gateway-namespaced ``<peer>-<tool>``) name back to
    the bare Tool binding. Mirrors ``TestOneConsumerSmoke`` — ``@requires_mcp``,
    the real ``_demo_stdio_server()``, real ``await``.
    """

    def test_construction_is_zero_network_and_returns_a_coroutine_function(self):
        """PRIMARY nmb2/deferred-connect proof: building the factory performs ZERO
        network I/O — the connect is deferred into the async factory body. Building
        against an UNREACHABLE/bogus spec must still return without raising, and the
        result must be a coroutine-function (the deferred ``_factory`` closure)."""
        from neograph_mcp import StdioServer, mcp_tool_factory

        bogus = StdioServer(command="/nonexistent/definitely-not-a-real-mcp-server", args=["--nope"])
        factory = mcp_tool_factory(
            "crm",
            bogus,
            tool_name="crm-perplexity_research",
            rename_to="perplexity_research",
        )
        # No subprocess spawned, no get_tools called: construction returned a plain
        # coroutine-function against a spec that would fail if it had connected.
        assert asyncio.iscoroutinefunction(factory)

    async def test_awaited_tool_renames_namespaced_to_bare_and_invokes(self):
        """Awaiting the factory connects, renames the gateway-namespaced
        ``crm-perplexity_research`` back to the bare ``perplexity_research`` Tool
        binding, and the tool invokes correctly against the real demo server."""
        import json

        from neograph_mcp import mcp_tool_factory

        factory = mcp_tool_factory(
            "crm",
            _demo_stdio_server(),
            tool_name="crm-perplexity_research",
            rename_to="perplexity_research",
        )
        tool = await factory({"configurable": {}}, None)
        assert tool.name == "perplexity_research"

        result = json.loads((await tool.ainvoke({"query": "acme"}))[0]["text"])
        assert result["query"] == "acme"

    async def test_stdio_token_provider_injects_through_the_singular_path(self):
        """Value regression pin: a stdio ``token_provider`` still injects
        ``stdio_token_arg`` through the singular helper (echoed under ``acting_as``)
        AND the tool carries the ``rename_to`` name. (A value pin only — it does not
        prove inject-introspects-pre-rename ordering, since rename touches only
        ``.name``, never ``.args``.)"""
        import json

        from neograph_mcp import mcp_tool_factory

        factory = mcp_tool_factory(
            "crm",
            _demo_stdio_server(),
            tool_name="crm-perplexity_research",
            rename_to="perplexity_research",
            token_provider=lambda configurable: configurable.get("op", "anon"),
            stdio_token_arg="token",
        )
        tool = await factory({"configurable": {"op": "operator-A"}}, None)
        assert tool.name == "perplexity_research"

        result = json.loads((await tool.ainvoke({"query": "acme"}))[0]["text"])
        assert result["acting_as"] == "operator-A"


# ── Typed MCP tool results: output_model= on the battery factories (wmmhc) ─────
#
# Client-side Pydantic models, defined INDEPENDENTLY of the demo server's classes
# (CrmSearchResult/DealHit). The whole point of the typed channel is that a
# consumer rehydrates the server's structuredContent into ITS OWN model — not that
# the two share a class. Fields use snake_case matching the JSON keys; Pydantic's
# default extra='ignore' drops bearer_identity, proving a narrower client model
# validates the server payload.


def _client_crm_models():
    from pydantic import BaseModel

    class ClientDealHit(BaseModel):
        id: str
        name: str
        stage: str

    class ClientCrmResult(BaseModel):
        query: str
        hits: list[ClientDealHit]
        acting_as: str

    return ClientCrmResult, ClientDealHit


@requires_mcp
class TestTypedOutputModel:
    """``output_model=`` on ``mcp_tool_factory`` / ``mcp_tool_factories`` (neograph-wmmhc).

    Declaring ``output_model=`` makes the wrapped adapter tool return the rehydrated
    CLIENT Pydantic model as the tool result (instead of the raw content-block list),
    so the typed channel — ``ToolInteraction.typed_result`` + the BAML ToolMessage
    render — carries a model with zero core changes. Mirrors ``TestLazySingleToolFactory``:
    ``@requires_mcp``, the real ``_demo_stdio_server()``, real ``await``, no MCP mocking.
    """

    async def test_awaited_tool_with_output_model_returns_model_instance(self):
        """(a) A factory built with ``output_model=`` produces a tool whose
        ``ainvoke`` returns the rehydrated CLIENT model INSTANCE — not the raw
        content-block list. crm_search emits structuredContent (CrmSearchResult
        server-side); the wrapper validates it into the client's own model."""
        from neograph_mcp import mcp_tool_factory

        ClientCrmResult, _ = _client_crm_models()

        factory = mcp_tool_factory(
            "crm",
            _demo_stdio_server(),
            tool_name="crm_search",
            output_model=ClientCrmResult,
        )
        tool = await factory({"configurable": {}}, None)
        result = await tool.ainvoke({"query": "acme"})

        assert isinstance(result, ClientCrmResult), (
            f"expected the rehydrated client model, got {type(result).__name__}: {result!r}"
        )
        assert result.acting_as == "anon"
        assert result.hits and result.hits[0].name == "Acme renewal"

    async def test_missing_structured_content_with_output_model_raises_typed_error(self):
        """(b) ``get_deal`` is annotated ``-> list`` and emits NO structuredContent
        (artifact is None). With ``output_model=`` declared, the wrapper raises a
        TYPED ValueError naming the tool AND pointing at the server-annotation fix —
        not a generic AttributeError/KeyError on a None artifact."""
        from neograph_mcp import mcp_tool_factory

        ClientCrmResult, _ = _client_crm_models()

        factory = mcp_tool_factory(
            "crm",
            _demo_stdio_server(),
            tool_name="get_deal",
            output_model=ClientCrmResult,
        )
        tool = await factory({"configurable": {}}, None)

        with pytest.raises(ValueError) as excinfo:
            await tool.ainvoke({"deal_id": "D1"})

        msg = str(excinfo.value)
        assert "get_deal" in msg, f"typed error must name the tool: {msg!r}"
        assert "structuredContent" in msg, f"typed error must mention structuredContent: {msg!r}"
        # points the consumer at the server-side annotation fix, not a bare failure.
        assert "annotation" in msg.lower() or "-> dict" in msg or "Pydantic" in msg, (
            f"typed error must point at the server-annotation fix: {msg!r}"
        )

    async def test_wrong_shape_output_model_propagates_validation_error(self):
        """(c) A client ``output_model`` requiring a field the server payload lacks
        must let Pydantic's ``ValidationError`` PROPAGATE untouched — it IS the
        type-mismatch signal, not something the wrapper swallows into content."""
        from pydantic import BaseModel, ValidationError

        from neograph_mcp import mcp_tool_factory

        class WrongShape(BaseModel):
            query: str
            missing_required: str  # crm_search never emits this key

        factory = mcp_tool_factory(
            "crm",
            _demo_stdio_server(),
            tool_name="crm_search",
            output_model=WrongShape,
        )
        tool = await factory({"configurable": {}}, None)

        with pytest.raises(ValidationError):
            await tool.ainvoke({"query": "acme"})

    async def test_iserror_with_output_model_surfaces_native_error_not_missing_sc(self):
        """(d) isError native path: a call the server/adapter errors on (crm_search
        with the required ``query`` arg omitted) on a tool WITH ``output_model=`` set
        must surface the adapter-native error path (handle_tool_error, preserved
        across the ``model_copy``, returns an error content block) — NOT the
        missing-structuredContent ValueError. Proves the wrapper lets the raise
        PROPAGATE rather than masking it as a schema-annotation miss."""
        from neograph_mcp import mcp_tool_factory

        ClientCrmResult, _ = _client_crm_models()

        factory = mcp_tool_factory(
            "crm",
            _demo_stdio_server(),
            tool_name="crm_search",
            output_model=ClientCrmResult,
        )
        tool = await factory({"configurable": {}}, None)

        # handle_tool_error catches the _MCPToolExecutionError raised inside the
        # wrapped coroutine and returns the native error content — no raise here.
        out = await tool.ainvoke({})  # missing required 'query'

        assert not isinstance(out, ClientCrmResult), (
            "a server error must NOT be rehydrated into the output model"
        )
        text = str(out)
        assert "Error executing tool crm_search" in text, (
            f"expected the adapter-native error, got: {text!r}"
        )
        # our missing-structuredContent ValueError must NOT have masked the native error.
        assert "output_model" not in text and "structuredContent" not in text, (
            f"wrapper masked the native isError as a missing-structuredContent error: {text!r}"
        )

    async def test_factories_output_models_dict_returns_typed_model(self):
        """(e) The plural ``mcp_tool_factories(..., output_models={...})`` dict form:
        a factory sliced from the returned dict returns the typed client model,
        keyed like the factory dict (bare key under ``namespace=False``)."""
        from neograph_mcp import mcp_tool_factories

        ClientCrmResult, _ = _client_crm_models()

        factories = mcp_tool_factories(
            {"crm": _demo_stdio_server()},
            output_models={"crm_search": ClientCrmResult},
            namespace=False,
        )
        tool = await factories["crm_search"]({"configurable": {}}, None)
        result = await tool.ainvoke({"query": "acme"})

        assert isinstance(result, ClientCrmResult), (
            f"expected the rehydrated client model via the plural factory, got {type(result).__name__}"
        )
        assert result.hits and result.hits[0].name == "Acme renewal"

    def test_e2e_typed_result_is_model_and_toolmessage_is_baml_render(self):
        """(E2E) Full agent run through ``compile()`` + ``arun()``: the tool bound
        with ``output_model=`` makes ``ToolInteraction.typed_result`` the CLIENT
        model instance, AND the ToolMessage content in message history is the
        ``describe_value`` (BAML) rendering — NOT ``str([{'type': 'text', ...}])``."""
        import neograph
        from neograph import compile
        from neograph_mcp import mcp_tool_factories
        from tests.fakes import build_test_compile_kwargs, configure_fake_llm
        from tests.schemas import Claims

        ClientCrmResult, _ = _client_crm_models()

        factories = mcp_tool_factories(
            {"crm": _demo_stdio_server()},
            output_models={"crm_search": ClientCrmResult},
            namespace=False,
        )
        only_search = {"crm_search": factories["crm_search"]}

        construct = _build_agent_construct("crm_search")
        llm_kw = configure_fake_llm(lambda tier: _react_fake("crm_search"))
        graph = compile(construct, tool_factories=only_search, **build_test_compile_kwargs(), **llm_kw)

        result = asyncio.run(neograph.arun(graph, input={"query": "acme"}))
        assert result["scan_result"] == Claims(items=["done"])

        tool_log = result["scan_tool_log"]
        assert tool_log, "expected the real MCP tool to have been called"
        ti = tool_log[0]
        assert isinstance(ti.typed_result, ClientCrmResult), (
            f"typed_result must be the client model, got {type(ti.typed_result).__name__}"
        )
        # The ToolMessage content fed to the next ReAct turn is the BAML render,
        # not a repr of raw content blocks.
        assert "'type': 'text'" not in ti.result, (
            f"ToolMessage carried raw content blocks instead of the BAML render: {ti.result!r}"
        )
        assert "Tool result:" in ti.result or "acme" in ti.result


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


# ── HttpServer / bearer transport (neograph-jahj) ──────────────────────────────
#
# Every test above drives StdioServer. HttpServer's bearer-via-Authorization-
# header path (_client.py:88-93) shipped with zero coverage — validated only by
# reading the isinstance check, never by observing a bearer token actually
# arrive at a server. These tests close that gap against the SAME demo server,
# launched in its http mode (examples/_mcp_demo_server.py).


@requires_mcp
class TestHttpServerSmoke:
    """Bind a tool over HttpServer via the battery and observe the bearer path."""

    async def test_bearer_token_arrives_via_tool_factory(self, tmp_path):
        """``mcp_tool_factories`` with ``HttpServer`` + ``token_provider`` mints a
        bearer header per factory call, and the demo server observably receives
        it (echoed back under ``bearer_identity``) — the http counterpart of the
        stdio ``acting_as`` smoke in ``TestOneConsumerSmoke``."""
        import json

        from neograph_mcp import mcp_tool_factories

        with _demo_http_server(tmp_path / "state.marker") as url:
            factories = mcp_tool_factories(
                {"crm": _demo_http_server_spec(url)},
                token_provider=lambda configurable: configurable.get("op", "anon"),
                namespace=False,
            )
            config = {"configurable": {"op": "operator-A"}}
            tool = await factories["crm_search"](config, None)
            result = json.loads((await tool.ainvoke({"query": "acme"}))[0]["text"])

        assert result["bearer_identity"] == "operator-A"

    async def test_stdio_token_arg_not_applied_over_http(self, tmp_path):
        """``stdio_token_arg`` is a stdio-only mechanism: ``_client.py`` only calls
        ``_inject_stdio_token`` for ``isinstance(spec, StdioServer)``. Over
        ``HttpServer`` the returned tool must be the RAW tool — an explicit
        ``token`` kwarg a caller passes must flow straight through untouched,
        even though a ``token_provider`` is configured and would (on stdio)
        override it. This is the behavioral proof; before this test the claim
        was pinned only by reading the isinstance check."""
        import json

        from neograph_mcp import mcp_tool_factories

        with _demo_http_server(tmp_path / "state.marker") as url:
            factories = mcp_tool_factories(
                {"crm": _demo_http_server_spec(url)},
                token_provider=lambda configurable: configurable.get("op", "anon"),
                stdio_token_arg="token",
                namespace=False,
            )
            config = {"configurable": {"op": "operator-A"}}
            tool = await factories["crm_search"](config, None)
            result = json.loads((await tool.ainvoke({"query": "acme", "token": "explicit-caller-value"}))[0]["text"])

        # The caller's own "token" argument survives unmodified...
        assert result["acting_as"] == "explicit-caller-value"
        # ...while the framework's per-run identity arrived via the bearer
        # header instead, proving the two channels are genuinely independent.
        assert result["bearer_identity"] == "operator-A"

    async def test_resource_fetcher_reads_real_content_over_http(self, tmp_path):
        """``mcp_resource_fetcher`` over ``HttpServer`` actually reads a resource
        end-to-end — beyond the shape-only stdio smoke above (which only checks
        the returned callables are coroutines)."""
        import json

        from neograph_mcp import mcp_resource_fetcher

        with _demo_http_server(tmp_path / "state.marker") as url:
            fetcher, _replayer = mcp_resource_fetcher({"crm": _demo_http_server_spec(url)})
            content, mime = await fetcher("mcp://crm/deals/D1/activity")

        payload = json.loads(content)
        assert payload["deal_id"] == "D1"
        assert payload["events"], "expected the real activity history, not a shape-only stub"
        assert mime == "application/json"


# ── Public session API: mcp_session / McpSession / McpCallResult (neograph-2plcl) ─
#
# TDD RED (neograph-5hwbb.6). The public `mcp_session` session API does NOT exist
# yet — `neograph_mcp.__all__` exports only StdioServer/HttpServer/ToolFactory/
# mcp_tool_factory(ies)/mcp_resource_fetcher. Every test below imports one of
# `mcp_session`/`McpSession`/`McpCallResult`/`McpToolCallError` INSIDE its body, so
# collection still succeeds (the file's other classes stay green) and each session
# test fails at runtime with an ImportError from `neograph_mcp` — the correct red
# for a not-yet-implemented public surface (task neograph-5hwbb.6). No MCP mocking:
# these mirror TestLazySingleToolFactory / TestHttpServerSmoke / TestTypedOutputModel
# against the real `_demo_stdio_server()` / `_demo_http_server()` with real `await`.
#
# The API under test (design docs/design/mcp-session-api-2plcl-2026-07-10.md §4/§4bis/§7):
#   mcp_session(server_key, spec, *, token_provider=None, config=None,
#               stdio_token_arg="token", timeout=30.0) -> McpSession   # async CM
#   await s.call(name, args, *, output_model=None, parse=None) -> McpCallResult | BaseModel
#   await s.tool_names() -> list[str]
#   McpCallResult(content: list[dict], structured: dict | None), frozen, with .text
#   McpToolCallError(Exception) raised when CallToolResult.isError is True.


def _hung_stdio_server():
    """A StdioServer whose subprocess NEVER answers `initialize` — it just sleeps.

    Exercises the connect-time hang class (§4.2): `ClientSession` has no default
    request timeout, so without `mcp_session`'s `asyncio.timeout(timeout)` on
    `__aenter__` this would block indefinitely. The one-line sleep command is the
    §7.4(ii) fixture the design and the architect review both prescribe.
    """
    from neograph_mcp import StdioServer

    return StdioServer(command=sys.executable, args=["-c", "import time; time.sleep(60)"])


def _refused_http_spec():
    """An HttpServer pointed at a free/unbound localhost port — connection REFUSED.

    `_free_port()` binds-then-closes, so nothing is listening: `__aenter__` gets an
    immediate ECONNREFUSED. Pins the §7.4(i) prompt-typed-error path (the natural
    fast failure, recursively unwrapped out of its single-leaf ExceptionGroup).
    """
    from neograph_mcp import HttpServer

    return HttpServer(url=f"http://127.0.0.1:{_free_port()}/mcp")


@requires_mcp
class TestMcpSession:
    """The public `mcp_session` session API (neograph-2plcl): one MCP connection,
    N tool calls, consumer-owned, offline-at-build.

    Mirrors TestLazySingleToolFactory (offline-at-build), TestHttpServerSmoke
    (bearer leg), and TestTypedOutputModel (typed rehydration / fail-loud) — real
    demo server, real `await`, NO MCP mocking. Every method fails RED today with an
    ImportError because `mcp_session` is not yet in `neograph_mcp.__all__`.
    """

    def test_construction_is_zero_network(self):
        """(a) OFFLINE-AT-BUILD: `mcp_session(...)` against a BOGUS/unreachable spec
        constructs with ZERO network — no subprocess spawn, no connect, no raise.
        The connect is deferred to `async with` entry. Mirrors
        TestLazySingleToolFactory.test_construction_is_zero_network..."""
        from neograph_mcp import McpSession, StdioServer, mcp_session

        bogus = StdioServer(command="/nonexistent/definitely-not-a-real-mcp-server", args=["--nope"])
        # Construction must not connect: building against a spec that would fail if
        # it had spawned/connected returns cleanly.
        session = mcp_session(
            "crm",
            bogus,
            token_provider=lambda configurable: configurable.get("op", "anon"),
        )
        assert isinstance(session, McpSession)
        # It is an async context manager (the connect fires at __aenter__, not now).
        assert hasattr(session, "__aenter__") and hasattr(session, "__aexit__")

    async def test_two_primitives_over_one_session_echo_same_minted_token(self):
        """(b) TWO PRIMITIVES over ONE session: inside a single `async with`,
        call `crm_search` then `get_deal`; both return `McpCallResult`, and the
        demo server echoes the SAME `acting_as` token on BOTH — proving the stdio
        token-arg was injected through the session and identity was minted ONCE.
        `get_deal` returns ResourceLink blocks, so this also pins the content
        table's file-block arm (§7.2)."""
        import json

        from neograph_mcp import McpCallResult, mcp_session

        config = {"configurable": {"op": "operator-A"}}
        async with mcp_session(
            "crm",
            _demo_stdio_server(),
            token_provider=lambda configurable: configurable.get("op", "anon"),
            config=config,
        ) as s:
            crm_result = await s.call("crm_search", {"query": "acme"})
            deal_result = await s.call("get_deal", {"deal_id": "D1"})

        assert isinstance(crm_result, McpCallResult)
        assert isinstance(deal_result, McpCallResult)

        # crm_search returns structuredContent (CrmSearchResult) — acting_as rides there.
        assert crm_result.structured is not None
        assert crm_result.structured["acting_as"] == "operator-A"

        # get_deal is content-only (a text mirror + 2 resource_link blocks): no
        # structuredContent, acting_as rides in the text block's JSON.
        assert deal_result.structured is None
        assert json.loads(deal_result.text)["acting_as"] == "operator-A"
        # The resource_link blocks convert to file blocks (adapter-parity table).
        assert any(block.get("type") == "file" for block in deal_result.content), (
            f"expected a file block from get_deal's resource_links: {deal_result.content!r}"
        )

    async def test_http_bearer_identity_rides_and_stdio_token_arg_not_injected(self, tmp_path):
        """(c) HttpServer bearer leg: over `_demo_http_server()` identity rides as
        the bearer Authorization header on `call()` (echoed under `bearer_identity`),
        and `stdio_token_arg` is NOT injected over http (so `acting_as` stays the
        server default `anon`). The http counterpart of (b); mirrors
        TestHttpServerSmoke.test_stdio_token_arg_not_applied_over_http."""
        from neograph_mcp import mcp_session

        config = {"configurable": {"op": "operator-A"}}
        with _demo_http_server(tmp_path / "state.marker") as url:
            async with mcp_session(
                "crm",
                _demo_http_server_spec(url),
                token_provider=lambda configurable: configurable.get("op", "anon"),
                stdio_token_arg="token",
                config=config,
            ) as s:
                result = await s.call("crm_search", {"query": "acme"})

        assert result.structured is not None
        # Identity arrived on the bearer header...
        assert result.structured["bearer_identity"] == "operator-A"
        # ...and the stdio token arg was NOT injected over http (server default).
        assert result.structured["acting_as"] == "anon"

    async def test_call_with_output_model_returns_rehydrated_client_model(self):
        """(d.1) TYPED RESULT: `call(..., output_model=ClientModel)` validates the
        server's structuredContent into the consumer's OWN independently-defined
        model and returns the INSTANCE (not `McpCallResult`). Mirrors
        TestTypedOutputModel with a client-side `_client_crm_models()` model."""
        from neograph_mcp import mcp_session

        ClientCrmResult, _ = _client_crm_models()

        async with mcp_session("crm", _demo_stdio_server()) as s:
            result = await s.call("crm_search", {"query": "acme"}, output_model=ClientCrmResult)

        assert isinstance(result, ClientCrmResult), (
            f"expected the rehydrated client model, got {type(result).__name__}: {result!r}"
        )
        assert result.acting_as == "anon"
        assert result.hits and result.hits[0].name == "Acme renewal"

    async def test_call_output_model_on_content_only_tool_raises_typed_value_error(self):
        """(d.2) `get_deal` is annotated `-> list` and emits NO structuredContent.
        With `output_model=` declared, `call()` raises a TYPED `ValueError` naming
        the tool AND pointing at the server-annotation fix — NOT a bare
        AttributeError/KeyError, and NOT `McpToolCallError` (the call succeeds; it
        just has no structured payload)."""
        from neograph_mcp import mcp_session

        ClientCrmResult, _ = _client_crm_models()

        async with mcp_session("crm", _demo_stdio_server()) as s:
            with pytest.raises(ValueError) as excinfo:
                await s.call("get_deal", {"deal_id": "D1"}, output_model=ClientCrmResult)

        msg = str(excinfo.value)
        assert "get_deal" in msg, f"typed error must name the tool: {msg!r}"
        assert "structuredContent" in msg, f"typed error must mention structuredContent: {msg!r}"
        assert "annotation" in msg.lower() or "-> dict" in msg or "Pydantic" in msg, (
            f"typed error must point at the server-annotation fix: {msg!r}"
        )

    async def test_call_wrong_shape_output_model_propagates_validation_error(self):
        """(d.3) A client `output_model` requiring a key the server payload lacks
        lets Pydantic's `ValidationError` PROPAGATE untouched — it IS the
        type-mismatch signal, not something `call()` swallows into content."""
        from pydantic import BaseModel, ValidationError

        from neograph_mcp import mcp_session

        class WrongShape(BaseModel):
            query: str
            missing_required: str  # crm_search never emits this key

        async with mcp_session("crm", _demo_stdio_server()) as s:
            with pytest.raises(ValidationError):
                await s.call("crm_search", {"query": "acme"}, output_model=WrongShape)

    async def test_server_side_iserror_raises_mcp_tool_call_error_not_value_error(self):
        """(e) isError -> `McpToolCallError`: a call the server errors on (crm_search
        with the required `query` arg omitted -> FastMCP CallToolResult.isError=True)
        raises `McpToolCallError`, NOT the missing-structuredContent `ValueError`
        and NOT a hang. `output_model=` is set to prove isError BEATS the
        output-model branch (§4.2 'isError beats output')."""
        from neograph_mcp import McpToolCallError, mcp_session

        ClientCrmResult, _ = _client_crm_models()

        async with mcp_session("crm", _demo_stdio_server()) as s:
            with pytest.raises(McpToolCallError) as excinfo:
                await s.call("crm_search", {}, output_model=ClientCrmResult)  # missing required 'query'

        # It must be the typed tool-call error, not the missing-structuredContent ValueError.
        assert not isinstance(excinfo.value, ValueError), (
            f"isError must surface as McpToolCallError, not a masked ValueError: {excinfo.value!r}"
        )

    async def test_refused_port_surfaces_typed_error_not_exceptiongroup(self):
        """(f) TYPED-ERROR / no-hang: connecting to a REFUSED localhost port surfaces
        a typed transport exception (`httpx.ConnectError`) PROMPTLY from `__aenter__`
        — recursively unwrapped out of its single-leaf ExceptionGroup, so NOT a
        `BaseExceptionGroup`, and NOT a hang (§4.2 / §7.4(i))."""
        import httpx

        from neograph_mcp import mcp_session

        start = time.monotonic()
        with pytest.raises(BaseException) as excinfo:  # noqa: PT011 — narrowed by asserts below
            async with mcp_session("crm", _refused_http_spec(), timeout=10.0) as s:
                await s.tool_names()
        elapsed = time.monotonic() - start

        exc = excinfo.value
        assert not isinstance(exc, BaseExceptionGroup), (
            f"connect error leaked as an ExceptionGroup instead of a single typed error: {exc!r}"
        )
        assert isinstance(exc, httpx.ConnectError), (
            f"expected a typed httpx.ConnectError from __aenter__, got {type(exc).__name__}: {exc!r}"
        )
        # Prompt, not a stalled read (the 300s HTTP read-stall class must not apply here).
        assert elapsed < 8.0, f"refused-port connect did not fail promptly: {elapsed:.1f}s"

    async def test_hung_stdio_server_raises_timeout_error_within_generous_ceiling(self):
        """(g) HUNG-SERVER TIMEOUT: a stdio subprocess that never answers `initialize`
        (a sleeping process) with a SHORT `timeout` raises `TimeoutError` at
        `__aenter__`. Asserted DEFENSIVELY per the architect review: the EXCEPTION
        TYPE plus a GENEROUS wall-clock ceiling (< 3x timeout) — never a tight
        `elapsed ~= timeout` bound (stdio subprocess teardown overshoots ~2x)."""
        from neograph_mcp import mcp_session

        timeout = 2.0
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            async with mcp_session("crm", _hung_stdio_server(), timeout=timeout) as s:
                await s.tool_names()
        elapsed = time.monotonic() - start

        # Generous ceiling only — subprocess-teardown overshoot is expected, an
        # equality-ish bound would be flaky.
        assert elapsed < 3 * timeout, (
            f"timeout overshot the generous ceiling: {elapsed:.1f}s for timeout={timeout}s"
        )

    async def test_tool_names_lists_the_demo_server_tools(self):
        """(h) `await s.tool_names()` returns the demo server's tool names (the
        paginated tools/list over the same session)."""
        from neograph_mcp import mcp_session

        async with mcp_session("crm", _demo_stdio_server()) as s:
            names = await s.tool_names()

        assert "crm_search" in names
        assert "get_deal" in names
        assert "update_deal" in names

    def test_session_content_table_matches_adapter(self):
        """PARITY GUARD: the session's local content-block conversion must match
        langchain-mcp-adapters' `_convert_mcp_content_to_lc_block` block-for-block.
        The session replicates that table (against public constructors) rather than
        importing the private symbol; this asserts parity so a future adapter bump
        that changes the table fails HERE and prompts a re-diff. No network."""
        from langchain_mcp_adapters.tools import _convert_mcp_content_to_lc_block
        from mcp.types import (
            BlobResourceContents,
            EmbeddedResource,
            ImageContent,
            ResourceLink,
            TextContent,
            TextResourceContents,
        )
        from pydantic import AnyUrl

        from neograph_mcp._session import _convert_content_block

        blocks = [
            TextContent(type="text", text="hello"),
            ImageContent(type="image", data="Zm9v", mimeType="image/png"),
            ResourceLink(type="resource_link", uri=AnyUrl("mcp://crm/deals/D1"), name="d1", mimeType="application/json"),
            ResourceLink(type="resource_link", uri=AnyUrl("mcp://crm/img/1"), name="img", mimeType="image/png"),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(uri=AnyUrl("mcp://crm/note/1"), text="note", mimeType="text/plain"),
            ),
            EmbeddedResource(
                type="resource",
                resource=BlobResourceContents(uri=AnyUrl("mcp://crm/blob/1"), blob="Zm9v", mimeType="application/pdf"),
            ),
        ]
        def _no_id(d: dict) -> dict:
            # The langchain constructors stamp a random per-call `id` uuid; compare
            # the meaningful payload (type/text/url/base64/mime_type), not the id.
            return {k: v for k, v in dict(d).items() if k != "id"}

        for block in blocks:
            assert _no_id(_convert_content_block(block)) == _no_id(_convert_mcp_content_to_lc_block(block)), (
                f"session content conversion drifted from the adapter for {type(block).__name__}"
            )

    def test_e2e_composite_session_assembles_typed_output_through_arun(self):
        """(i) E2E COMPOSITE (§8): a raw-mode node opens ONE session, calls two
        primitives (crm_search + get_deal), and assembles a typed output — driven
        through `compile()` + `arun()`, asserting the ASSEMBLED OUTPUT (not
        internals). The blessed 'scripted composite over federated primitives'
        pattern stark-8ok asked for."""
        from pydantic import BaseModel

        import neograph
        from neograph import compile, construct_from_functions, node
        from neograph_mcp import mcp_session
        from tests.fakes import build_test_compile_kwargs

        class DealSummary(BaseModel, frozen=True):
            query: str
            deal_count: int
            acting_as: str

        @node(mode="raw", outputs=DealSummary)
        async def compose(state, config):
            async with mcp_session(
                "crm",
                _demo_stdio_server(),
                token_provider=lambda configurable: configurable.get("op", "anon"),
                config=config,
            ) as s:
                found = await s.call("crm_search", {"query": "acme"})
                await s.call("get_deal", {"deal_id": "D1"})  # 2nd primitive over the same session
            payload = found.structured
            return {
                "compose": DealSummary(
                    query=payload["query"],
                    deal_count=len(payload["hits"]),
                    acting_as=payload["acting_as"],
                )
            }

        graph = compile(construct_from_functions("crm_compose", [compose]), **build_test_compile_kwargs())
        cfg = {"configurable": {"op": "operator-A"}}
        result = asyncio.run(neograph.arun(graph, input={}, config=cfg))

        assert result["compose"] == DealSummary(query="acme", deal_count=1, acting_as="operator-A")
