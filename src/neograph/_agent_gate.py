"""Tool-approval gate bodies for act-mode agents.

Extracted from ``_agent_cycle.py`` (neograph-3ffdg.7) as a pure file split — the
functions below are unchanged, only their home moved. ``_agent_cycle.py``
re-exports them, so existing imports keep resolving.

The gate is the human-in-the-loop seam: an act-mode node whose tools mutate can
pause via ``interrupt()`` and resume on approval. Self-contained apart from
``cycle_names``, which it shares with the cycle builder and with ``factory.py``;
that lives in ``_agent_cycle_names.py`` so neither this module nor the cycle has
to import the other.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel

from neograph._agent_cycle_names import cycle_names
from neograph._state_bus import adapt_state
from neograph._state_keys import StateKeys
from neograph.naming import field_name_for
from neograph.node import Node


def _gate_approved(human_input: Any) -> bool:
    """Fail-closed approval test for a tool-gate resume value; see neograph-whq0.

    Only an explicit ``{"approved": True}`` approves. A missing key, a non-dict
    payload, or any other value is treated as a DENY — a safety control must not
    execute a side-effecting tool on an ambiguous resume.
    """
    return isinstance(human_input, dict) and human_input.get("approved") is True


def make_tool_gate_bodies(node: Node, gate_condition: Callable) -> dict[str, Any]:
    """Build the tool-gate node body + decision router for ``gate_tools_when``.

    The gate sits on the agent cycle's tools arm (see ``_wiring._add_agent_cycle``).
    It pauses BEFORE the ``{node}__tools`` superstep so a human approves before the
    tool's side effects run, then HONORS the decision per neograph-whq0:

    - condition falsy (gate does not fire) → leave the pending tool_calls
      unanswered → router routes to ``{node}__tools`` (proceed).
    - approve → leave the pending tool_calls unanswered → router routes to
      ``{node}__tools`` (the tool runs).
    - deny (fail-closed default) → append one denial ``ToolMessage`` per pending
      tool_call to the messages channel so the LLM sees why → router routes back
      to ``{node}__agent`` so the loop continues reasoning to a final answer.

    The decision is encoded in the message channel itself (deny answers the
    pending tool_calls; approve/no-fire leaves them pending), so the router reads
    a FRESH per-visit signal — never a persistent channel that can go stale across
    turns. The routing decision lives in the Layer-1 conditional edge; only the
    denial-message synthesis (agent-cycle domain, where ``msgs_key`` lives) sits
    here — no in-body if-approved check inside the tools node.
    """
    field = field_name_for(node.name)
    msgs_key = StateKeys.agent_messages(field)
    names = cycle_names(node.name)

    def _pending_tool_calls(channel_msgs: list) -> list:
        last = channel_msgs[-1] if channel_msgs else None
        return list(getattr(last, "tool_calls", None) or [])

    def _denial_messages(channel_msgs: list, human_input: Any) -> list:
        reason = human_input.get("reason") if isinstance(human_input, dict) else None
        detail = f" Reason: {reason}" if reason else ""
        return [
            ToolMessage(
                content=(
                    f"Tool call '{tc['name']}' was denied by a human reviewer and did "
                    f"not run.{detail} Do not retry it; continue with the information "
                    f"you have and provide your final answer."
                ),
                tool_call_id=tc["id"],
            )
            for tc in _pending_tool_calls(channel_msgs)
        ]

    def gate_body(state: BaseModel, config: RunnableConfig) -> dict[str, Any]:
        payload = gate_condition(state)
        if not payload:
            return {}  # gate did not fire → proceed to tools
        human_input = interrupt(payload)
        if _gate_approved(human_input):
            return {StateKeys.HUMAN_FEEDBACK: human_input}
        # Deny (fail-closed): answer the pending tool_calls with denials so the
        # agent sees the rejection, and record the decision for observability.
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        return {
            msgs_key: _denial_messages(channel_msgs, human_input),
            StateKeys.HUMAN_FEEDBACK: human_input,
        }

    def gate_router(state: BaseModel) -> str:
        # Deny appended ToolMessages answering the pending tool_calls → the last
        # message is a ToolMessage → route back to the agent. Approve / no-fire
        # left the AIMessage's tool_calls pending → route to tools.
        bus = adapt_state(state)
        channel_msgs = bus.get(msgs_key) or []
        last = channel_msgs[-1] if channel_msgs else None
        if isinstance(last, ToolMessage):
            return names.agent
        return names.tools

    return {"gate": gate_body, "router": gate_router}
