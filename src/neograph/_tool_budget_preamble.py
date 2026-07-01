"""Framework-generated tool-budget preamble.

The ONE canonical place per-tool call budgets are turned into prose for the
model. Rendered inside ``invoke_with_tools`` — the only site that holds the
``Tool`` specs and ``cfg.max_iterations`` together — so the numbers the model
is TOLD are the same numbers that are ENFORCED, by construction. No caller ever
hand-writes a budget number (which drifts from enforcement).

Pure logic: ``(tools, max_iterations) -> str``. No I/O, no state.

Import graph: ``_tool_budget_preamble -> tool`` (one-way; ``tool.py`` imports
nothing from the loop, so there is no cycle).
"""

from __future__ import annotations

from neograph.tool import Tool

# Locked directive: plan-ahead + batch. Phrased in terms of
# reasoning steps/turns, NOT a total tool-call ceiling — max_iterations bounds
# ReAct turns, and a single turn may batch several tool calls.
_DIRECTIVE = (
    "Plan your calls before invoking; batch related calls where you can; "
    "you need not use every call. Stop and answer once you have enough."
)


def render_tool_budget_preamble(tools: list[Tool], max_iterations: int) -> str:
    """Render the accurate tool-budget preamble for the agent's system message.

    Only tools with a FINITE budget (``budget > 0``) are announced. Unlimited
    tools (``budget == 0``) are omitted entirely: "unlimited" is only safe when
    paired with context-status evaluation + compaction, which is out of scope
    for neograph — and omitting a tool already communicates "no explicit cap".

    Always renders the ``max_iterations`` step cap and the plan-ahead/batch
    directive, even when the per-tool list is empty (all-unlimited or no tools).
    Never raises.
    """
    lines = ["You have a limited tool-call budget. Plan before you call."]

    finite = [t for t in tools if t.budget > 0]
    for tool in finite:
        noun = "call" if tool.budget == 1 else "calls"
        lines.append(f"  - {tool.name}: {tool.budget} {noun}")

    lines.append(
        f"Overall step cap: {max_iterations} reasoning steps (turns); "
        "a single step may batch several tool calls."
    )
    lines.append(_DIRECTIVE)
    return "\n".join(lines)
