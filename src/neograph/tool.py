"""Tool — LLM-callable tool with per-tool budget.

Two ways to define a tool:

    # 1. Declarative: Tool class + register_tool_factory
    search = Tool("search_code", budget=5)
    register_tool_factory("search_code", lambda config, tool_config: MySearchTool())

    # 2. Decorator: @tool wraps a function, auto-registers the factory
    @tool(budget=5)
    def search_code(query: str) -> list[str]:
        '''Search the codebase for the given query.'''
        return _do_search(query)

    # In both cases, pass to a Node via tools=[...]
    research = Node("research", mode="gather", tools=[search_code], ...)
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field


class Tool(BaseModel, frozen=True):
    """A tool the LLM can call, with a per-tool call budget.

    Usage:
        search = Tool("search_nodes", budget=5)
        read   = Tool("read_artifact", budget=10, config={"max_chars": 6000})
    """

    name: str
    budget: int = 0  # max calls for this tool (0 = unlimited)
    config: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, name_: str | None = None, /, **kwargs):
        """Tool accepts name positionally or as a keyword argument."""
        if name_ is not None:
            kwargs["name"] = name_
        super().__init__(**kwargs)


class ToolInteraction(BaseModel, frozen=True):
    """Record of a single tool call during a ReAct loop.

    Collected by invoke_with_tools and exposed as a secondary output
    when the gather/execute node declares it in dict-form outputs.
    """

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    duration_ms: int = 0


class ToolBudgetTracker:
    """Tracks per-tool call counts and enforces budgets at runtime.

    Created by the node factory for gather/execute mode nodes.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._budgets: dict[str, int] = {}
        self._counts: dict[str, int] = {}
        for tool in tools:
            self._budgets[tool.name] = tool.budget
            self._counts[tool.name] = 0

    def can_call(self, tool_name: str) -> bool:
        """Check if a tool still has budget remaining."""
        budget = self._budgets.get(tool_name, 0)
        if budget == 0:
            return True  # unlimited
        return self._counts.get(tool_name, 0) < budget

    def record_call(self, tool_name: str) -> None:
        """Record a tool call."""
        self._counts[tool_name] = self._counts.get(tool_name, 0) + 1

    def exhausted_tools(self) -> set[str]:
        """Return names of tools that have hit their budget."""
        return {
            name
            for name, budget in self._budgets.items()
            if budget > 0 and self._counts.get(name, 0) >= budget
        }

    def all_exhausted(self) -> bool:
        """True if every budgeted tool is spent. Unlimited tools (budget=0) never exhaust."""
        if not self._budgets:
            return False  # no tools → nothing to exhaust
        for name, budget in self._budgets.items():
            if budget == 0:
                return False
            if self._counts.get(name, 0) < budget:
                return False
        return True


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    budget: int = 0,
    config: dict[str, Any] | None = None,
) -> Any:
    """Decorator that turns a function into a Tool and auto-registers its factory.

    The tool's name defaults to the function name. The description comes from
    the docstring. Arguments are introspected from the function signature.

    Usage:

        @tool(budget=5)
        def search_code(query: str) -> list[str]:
            '''Search the codebase for the given query.'''
            return _do_search(query)

        research = Node("research", mode="gather", tools=[search_code], ...)

    The decorated function IS a Tool instance (with the original function
    attached as .fn), so it can be passed directly to Node's tools= list.
    The factory is auto-registered under the tool's name.
    """
    def decorator(f: Callable) -> Tool:
        # Avoid circular import
        from neograph.factory import register_tool_factory

        tool_name = name or f.__name__

        # Build a LangChain-compatible tool from the function
        from langchain_core.tools import tool as lc_tool
        lc_tool_instance = lc_tool(f)

        # Register the factory so the ReAct loop can instantiate it
        register_tool_factory(tool_name, lambda config, tool_config: lc_tool_instance)

        # Return a Tool spec that Node accepts in tools=[...]
        return Tool(tool_name, budget=budget, config=config or {})

    # Support both @tool and @tool(budget=5)
    if fn is not None:
        return decorator(fn)
    return decorator

