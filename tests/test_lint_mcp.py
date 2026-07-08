"""lint() rule: async-only (MCP) tool under a run()-driven graph.

The differentiator for neograph as an MCP client: bindings are validated at
compile/lint time. An MCP tool is async-only (StructuredTool with a coroutine
and no sync func). lint() cannot know the driver (run vs arun) statically, so it
flags any agent/act node bound to an async-only tool with a
``tool_requires_async_driver`` issue pointing the user at arun().
"""

from __future__ import annotations

import types as _types
import warnings

from langchain_core.tools import StructuredTool

from neograph import Tool, construct_from_module, lint, node
from tests.schemas import Claims

_ISSUE_KIND = "tool_requires_async_driver"


def _make_async_only_tool(name: str = "mcp_echo") -> StructuredTool:
    async def _run(text: str) -> str:
        return f"echo:{text}"

    return StructuredTool.from_function(coroutine=_run, name=name, description="async-only MCP-style tool")


def _make_sync_tool(name: str = "sync_search") -> StructuredTool:
    def _run(query: str) -> str:
        return f"result:{query}"

    return StructuredTool.from_function(func=_run, name=name, description="sync search tool")


def _agent_construct(tools, name="lint-mcp"):
    mod = _types.ModuleType(f"test_{name.replace('-', '_')}_mod")

    @node(mode="agent", outputs=Claims, model="fast", prompt="test/scan", tools=tools)
    def scan() -> Claims: ...

    mod.scan = scan
    return construct_from_module(mod, name=name)


class TestAsyncOnlyToolLintRule:
    def test_flags_raw_async_only_base_tool_on_agent_node(self):
        async_tool = _make_async_only_tool("mcp_echo")
        construct = _agent_construct([async_tool], name="lint-mcp-raw")

        issues = lint(construct)

        mcp_issues = [i for i in issues if i.kind == _ISSUE_KIND]
        assert len(mcp_issues) == 1, [i.kind for i in issues]
        assert "mcp_echo" in mcp_issues[0].message
        assert "arun()" in mcp_issues[0].message

    def test_no_issue_for_sync_tool(self):
        sync_tool = _make_sync_tool("sync_search")
        construct = _agent_construct([sync_tool], name="lint-mcp-sync")

        issues = lint(construct)

        assert [i for i in issues if i.kind == _ISSUE_KIND] == []

    def test_flags_async_only_tool_resolved_via_tool_factories(self):
        async_tool = _make_async_only_tool("mcp_remote")
        construct = _agent_construct([Tool("mcp_remote", budget=0)], name="lint-mcp-factory")

        issues = lint(
            construct,
            tool_factories={"mcp_remote": lambda config, tool_config: async_tool},
        )

        mcp_issues = [i for i in issues if i.kind == _ISSUE_KIND]
        assert len(mcp_issues) == 1, [i.kind for i in issues]
        assert "mcp_remote" in mcp_issues[0].message

    def test_no_issue_when_factory_absent(self):
        # Tool spec with no bound tool and no factory to resolve — lint cannot
        # introspect, so it must not false-positive.
        construct = _agent_construct([Tool("unknown_tool", budget=0)], name="lint-mcp-nofac")

        issues = lint(construct)

        assert [i for i in issues if i.kind == _ISSUE_KIND] == []


class TestAsyncToolFactoryLintRule:
    """An async (coroutine) tool factory requires arun(). lint must classify it
    WITHOUT calling it (w74k.3.1 item 2) — calling would create an un-awaited
    coroutine (RuntimeWarning) and misintrospect the coroutine object as a tool.
    """

    def test_flags_async_factory_without_calling_it(self):
        called = []

        async def _async_factory(config, tool_config):
            called.append(True)  # must NEVER run
            return _make_sync_tool("mcp_remote")

        construct = _agent_construct([Tool("mcp_remote", budget=0)], name="lint-mcp-async-factory")

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # 'coroutine never awaited' -> error
            issues = lint(construct, tool_factories={"mcp_remote": _async_factory})

        mcp_issues = [i for i in issues if i.kind == _ISSUE_KIND]
        assert len(mcp_issues) == 1, [i.kind for i in issues]
        assert "mcp_remote" in mcp_issues[0].message
        assert "arun()" in mcp_issues[0].message
        assert called == [], "lint must NOT invoke an async tool factory"

    def test_no_issue_for_sync_factory_returning_sync_tool(self):
        construct = _agent_construct([Tool("sync_search", budget=0)], name="lint-mcp-sync-factory")

        issues = lint(
            construct,
            tool_factories={"sync_search": lambda config, tc: _make_sync_tool("sync_search")},
        )

        assert [i for i in issues if i.kind == _ISSUE_KIND] == []
