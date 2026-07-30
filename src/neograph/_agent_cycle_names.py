"""Canonical node names for one inline ReAct cycle.

Extracted from ``_agent_cycle.py`` (neograph-3ffdg.7) into a neutral module so
that both the cycle builder and the tool-approval gate can reach it without
importing each other. ``factory.py`` is the third consumer. ``_agent_cycle.py``
re-exports both names, so existing imports keep resolving.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentCycleNames:
    """The three parent-node names an agent/act node expands into."""

    agent: str
    tools: str
    parse: str


def cycle_names(node_name: str) -> AgentCycleNames:
    return AgentCycleNames(
        agent=f"{node_name}__agent",
        tools=f"{node_name}__tools",
        parse=f"{node_name}__parse",
    )
