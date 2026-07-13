"""MCP tool ergonomics — raw BaseTool acceptance + async-only sync-driver error.

Covers the neograph-w74k.3 barrier-removal work:

1. A raw LangChain ``BaseTool`` may be passed directly in ``Node(tools=[...])``
   (and ``@node(tools=[...])``). Assembly normalizes it to a ``Tool`` spec and
   auto-registers a factory returning the tool — no ``register_tool_factory``
   boilerplate. Tested through all three API surfaces.
2. Driving an async-only tool (e.g. an MCP tool) through the *sync* tool loop
   raises a clear ``ConfigurationError`` naming the tool and pointing at
   ``arun()``, instead of leaking a bare ``NotImplementedError``.
"""

from __future__ import annotations

import types as _types

import pytest
from langchain_core.tools import BaseTool, StructuredTool

from neograph import Tool, compile, construct_from_module, node, run
from tests.fakes import (
    ReActFake,
    build_fake_runtime,
    build_fake_tool_lookup,
    build_test_compile_kwargs,
    configure_fake_llm,
    register_tool_factory,
)
from tests.schemas import Claims


def _make_async_only_tool(name: str = "mcp_echo") -> StructuredTool:
    """A StructuredTool with only a coroutine — models a langchain-mcp-adapters tool.

    Sync ``.invoke`` raises NotImplementedError; ``.ainvoke`` works.
    """

    async def _run(text: str) -> str:
        return f"echo:{text}"

    return StructuredTool.from_function(coroutine=_run, name=name, description="async-only MCP-style tool")


def _make_sync_base_tool(name: str = "sync_search") -> StructuredTool:
    """A plain sync StructuredTool — a raw LangChain BaseTool for tools= tests."""

    def _run(query: str) -> str:
        return f"result:{query}"

    return StructuredTool.from_function(func=_run, name=name, description="sync search tool")


class TestSyncDriverRejectsAsyncOnlyTool:
    """The sync ReAct loop must fail loud (ConfigurationError) when a tool is
    async-only, pointing the user at arun()."""

    def test_sync_loop_raises_clear_error_for_async_only_tool(self):
        from neograph.errors import ConfigurationError
        from neograph.tool import ToolBudgetTracker
        from tests.fakes import drive_agent_via_cycle as invoke_with_tools

        async_tool = _make_async_only_tool("mcp_echo")
        register_tool_factory("mcp_echo", lambda cfg, tc: async_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "mcp_echo", "args": {"text": "hi"}, "id": "c1"}],
                [],
            ],
            final=lambda m: m(items=["done"]),
            output_model=Claims,
        )
        _llm_kw = configure_fake_llm(lambda tier: fake)
        tools = [Tool("mcp_echo", budget=0)]
        tracker = ToolBudgetTracker(tools)

        with pytest.raises(ConfigurationError) as excinfo:
            invoke_with_tools(
                runtime=build_fake_runtime(_llm_kw["llm_factory"], _llm_kw["prompt_compiler"]),
                model_tier="fast",
                prompt_template="test",
                input_data="test",
                output_model=Claims,
                tools=tools,
                budget_tracker=tracker,
                config={"configurable": {}},
                node_name="research",
                tool_factory_lookup=build_fake_tool_lookup(),
            )

        msg = str(excinfo.value)
        assert "mcp_echo" in msg, msg
        assert "arun()" in msg, msg


class TestRawBaseToolAcceptance:
    """A raw LangChain BaseTool passed in tools= is normalized to a Tool spec
    with the tool carried on the private _bound_tool attribute, across all three
    API surfaces."""

    def test_declarative_node_normalizes_raw_base_tool(self):
        raw = _make_sync_base_tool("sync_search")
        n = Node_agent(tools=[raw])

        assert len(n.tools) == 1
        assert isinstance(n.tools[0], Tool)
        assert not isinstance(n.tools[0], BaseTool)
        assert n.tools[0].name == "sync_search"
        assert n.tools[0]._bound_tool is raw

    def test_node_decorator_normalizes_raw_base_tool(self):
        raw = _make_sync_base_tool("dec_search")

        @node(
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test/scan",
            tools=[raw],
        )
        def scan() -> Claims: ...

        spec = scan.tools[0]
        assert isinstance(spec, Tool)
        assert spec.name == "dec_search"
        assert spec._bound_tool is raw

    def test_programmatic_pipe_preserves_bound_tool(self):
        from neograph import Each

        raw = _make_sync_base_tool("pipe_search")
        n = Node_agent(tools=[raw]) | Each(over="items", key="id")

        assert isinstance(n.tools[0], Tool)
        assert n.tools[0]._bound_tool is raw

    def test_compile_auto_registers_factory_for_raw_base_tool(self):
        raw = _make_sync_base_tool("auto_search")

        mod = _types.ModuleType("test_mcp_autoreg_mod")

        @node(
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test/scan",
            tools=[raw],
        )
        def scan() -> Claims: ...

        mod.scan = scan
        pipeline = construct_from_module(mod, name="test-mcp-autoreg")

        fake = ReActFake(tool_calls=[[]], final=lambda m: m(items=["done"]), output_model=Claims)
        _llm_kw = configure_fake_llm(lambda tier: fake)

        # No register_tool_factory call for "auto_search": compile must
        # auto-register from the bound tool.
        graph = compile(pipeline, **build_test_compile_kwargs(), **_llm_kw)

        assert "auto_search" in graph.tool_factories
        produced = graph.tool_factories["auto_search"]({"configurable": {}}, {})
        assert produced is raw

    def test_raw_base_tool_runs_end_to_end_without_manual_registration(self):
        calls: list[tuple[dict[str, object], str]] = []

        def _run_wrapped(query: str, *, config=None) -> str:
            result_inner = _run(query, config=config)
            calls.append(({"query": query}, result_inner))
            return result_inner

        raw = _make_sync_base_tool("run_search")
        # Wrap the tool's _run to record calls (LOAD-BEARING: catches "tool turn skipped")
        _run = raw._run
        raw._run = _run_wrapped  # type: ignore[method-assign]

        mod = _types.ModuleType("test_mcp_e2e_mod")

        @node(
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test/scan",
            tools=[raw],
        )
        def scan() -> Claims: ...

        mod.scan = scan
        pipeline = construct_from_module(mod, name="test-mcp-e2e")

        fake = ReActFake(
            tool_calls=[
                [{"name": "run_search", "args": {"query": "q"}, "id": "c1"}],
                [],
            ],
            final=lambda m: m(items=["done"]),
            output_model=Claims,
        )
        _llm_kw = configure_fake_llm(lambda tier: fake)
        graph = compile(pipeline, **build_test_compile_kwargs(), **_llm_kw)
        result = run(graph, input={"node_id": "n1"})
        # Call-recording assertion (LOAD-BEARING for "tool turn skipped" regression)
        assert calls, "Tool 'run_search' was never called (tool turn skipped)"
        assert calls == [({"query": "q"}, "result:q")], f"Tool call mismatch: {calls}"
        # Output value assertion (pins final-output shape)
        assert result["scan"] == Claims(items=["done"])


def Node_agent(**kwargs):
    """Helper: a declarative agent Node with the LLM plumbing filled in."""
    from neograph import Node

    return Node(
        "research",
        mode="agent",
        outputs=Claims,
        model="fast",
        prompt="test/scan",
        **kwargs,
    )
