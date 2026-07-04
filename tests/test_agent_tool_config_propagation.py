"""Tool-config propagation in the agent/ReAct loop (neograph-zmfx).

The agent cycle's ``{node}__tools`` superstep executes each requested tool via
``tool_fn.invoke(args)`` / ``await tool_fn.ainvoke(args)``. Historically it
passed NO ``config=`` — so the run's ``RunnableConfig`` (which carries the
LangChain callback managers, including the ``astream_events`` tracer) was not
threaded EXPLICITLY into the tool. LangChain can still recover the ambient
config from a contextvar, but that propagation is fragile (it breaks across
thread / executor / context boundaries), so the documented-correct behavior is
to pass the node's ``config`` through explicitly. This is what agent-stark-style
consumers rely on for a live ``astream_events`` activity feed.

The core regression (RED without the fix) is a config-capturing tool that records
the ``config`` it receives at BOTH the sync (``tools_body``) and async
(``atools_body``) call sites: without ``config=config`` the tool is invoked with
``config=None``; with the fix it receives the node's ``RunnableConfig``.

An additional end-to-end test drives ``astream_events(version='v2')`` with a real
``StructuredTool`` and asserts ``on_tool_start`` / ``on_tool_end`` are visible —
the observable payoff of threading the config through.
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import StructuredTool

from neograph import Tool, compile, construct_from_functions, node, run
from tests.fakes import (
    ReActFake,
    build_fake_llm_kwargs,
    build_test_compile_kwargs,
    register_tool_factory,
)
from tests.schemas import Claims

_CFG = {"configurable": {}}
_INPUT = {"node_id": "t"}


class _ConfigCapturingTool:
    """Minimal tool stand-in that records the ``config`` passed at its call site.

    Mirrors the ``.invoke(args)`` / ``.ainvoke(args)`` surface the agent cycle
    drives. ``config`` defaults to ``None`` so a call site that omits ``config=``
    is directly observable (the recorded value is ``None``)."""

    def __init__(self, name: str, response: str = "found"):
        self.name = name
        self.response = response
        self.sync_configs: list = []
        self.async_configs: list = []

    def invoke(self, args: dict, config=None) -> str:
        self.sync_configs.append(config)
        return self.response

    async def ainvoke(self, args: dict, config=None) -> str:
        self.async_configs.append(config)
        return self.response


def _agent_node():
    @node(
        mode="agent",
        outputs=Claims,
        model="reason",
        prompt="test/explore",
        tools=[Tool(name="search", budget=2)],
    )
    def explore() -> Claims: ...

    return explore


def _react_fake():
    return ReActFake(
        tool_calls=[
            [{"name": "search", "args": {"q": "test"}, "id": "c1"}],
            [],  # final structured turn
        ],
        final=lambda m: m(items=["done"]),
    )


def _compile_agent(fake):
    return compile(
        construct_from_functions("p", [_agent_node()]),
        **build_test_compile_kwargs(),
        **build_fake_llm_kwargs(lambda tier: fake),
    )


class TestAgentToolConfigPropagation:
    """The node ``config`` must be threaded EXPLICITLY into the tool at both the
    sync and async call sites (neograph-zmfx)."""

    def test_sync_tool_call_receives_node_config(self):
        tool = _ConfigCapturingTool("search")
        register_tool_factory("search", lambda config, tool_config: tool)
        graph = _compile_agent(_react_fake())

        result = run(graph, input=dict(_INPUT), config=dict(_CFG))

        assert result["explore"] == Claims(items=["done"])
        assert tool.sync_configs, "tool was never invoked on the sync path"
        # RED without the fix: tools_body calls tool.invoke(args) with no config=,
        # so the captured config is None.
        assert tool.sync_configs[0] is not None, (
            "sync tool call did not receive the node RunnableConfig"
        )

    def test_async_tool_call_receives_node_config(self):
        tool = _ConfigCapturingTool("search")
        register_tool_factory("search", lambda config, tool_config: tool)
        graph = _compile_agent(_react_fake())

        result = asyncio.run(graph.graph.ainvoke(dict(_INPUT), dict(_CFG)))

        assert result["explore"] == Claims(items=["done"])
        assert tool.async_configs, "tool was never invoked on the async path"
        # RED without the fix: atools_body awaits tool.ainvoke(args) with no
        # config=, so the captured config is None.
        assert tool.async_configs[0] is not None, (
            "async tool call did not receive the node RunnableConfig"
        )


class TestAgentToolEventsVisible:
    """End-to-end payoff: with the config threaded through, a real LangChain tool
    fires on_tool_start / on_tool_end in the compiled graph's astream_events."""

    def test_tool_events_visible_in_astream_events(self):
        register_tool_factory(
            "search",
            lambda config, tool_config: StructuredTool.from_function(
                func=lambda q: f"found: {q}", name="search", description="search things"
            ),
        )
        graph = _compile_agent(_react_fake())

        async def _collect():
            return [
                ev
                async for ev in graph.astream_events(dict(_INPUT), dict(_CFG), version="v2")
            ]

        events = asyncio.run(_collect())
        starts = [e for e in events if e.get("event") == "on_tool_start"]
        ends = [e for e in events if e.get("event") == "on_tool_end"]

        assert any(e.get("name") == "search" for e in starts), [
            e.get("name") for e in starts
        ]
        assert ends, "on_tool_end did not fire for the tool call"
