"""Tool — LLM-callable tool with per-tool budget."""

from __future__ import annotations

from typing import Any

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

