"""Observability helpers — verbatim context extraction for LLM nodes."""

from __future__ import annotations

from typing import cast

from neograph._state_bus import StateBus
from neograph.naming import field_name_for
from neograph.node import Node


def _extract_context(state: StateBus, node: Node) -> dict[str, str] | None:
    """Extract verbatim context fields from state for LLM nodes.

    Returns a dict of {context_name: state_value} if the node declares
    context fields, or None if no context is configured. Context values
    are user-declared string fields rendered verbatim into prompts.
    """
    if not node.context:
        return None
    return {
        name: cast(str, state.get(field_name_for(name)))
        for name in node.context
    }
