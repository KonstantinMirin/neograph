"""Framework-generated agent output-schema instruction.

In agent/act mode the ReAct loop's FINAL turn IS the node's structured answer —
it is parsed directly from ``messages[-1]`` (no separate re-generation over the
message history). For that parse to succeed the model must be told, up front,
the exact JSON shape to emit when it finishes calling tools. This helper renders
that instruction as framework system content, injected into the agent loop
alongside the (opt-in) tool-budget preamble.

Pure logic: ``type[BaseModel] -> str``. No I/O, no state. Mirrors
``_tool_budget_preamble.py`` (repo convention for framework-injected system
content). The schema string itself is produced by ``describe_type`` — the same
renderer the single-shot ``json_mode`` path uses (``_llm.py``) — so there is one
source of truth for schema notation.

Import graph: ``_agent_output_schema_preamble -> describe_type`` (one-way).
"""

from __future__ import annotations

from pydantic import BaseModel

from neograph.describe_type import describe_type

# Phrased conditionally ("When finished with tools ...") so a well-behaved model
# still calls its tools first and only emits JSON as its final answer — not
# prematurely in place of a tool call.
_PREFIX = (
    "When finished with tools, respond with ONLY a JSON object matching this "
    "schema (no prose, no markup):"
)


def render_output_schema_instruction(output_model: type[BaseModel]) -> str:
    """Render the final-answer JSON-schema instruction for an agent/act node.

    A thin passthrough to ``describe_type`` with an agent-specific prefix — it
    does NOT hand-concatenate around ``describe_type``'s default prefix.
    """
    return describe_type(output_model, prefix=_PREFIX)
