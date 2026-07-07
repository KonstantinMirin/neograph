"""Fan (Each/Oracle/Loop) over an agent/act node — support classification.

Why this module exists (neograph-m6d3.6; full rationale in
``docs/design/fan-over-agent-node-2026-07-07.md``): an agent/act node compiles to
a multi-node inline ReAct cycle (``{node}__agent`` / ``{node}__tools`` /
``{node}__parse``) whose per-turn state lives in SHARED reducer channels
(``neo_agent_messages_*`` = ``add_messages``, ``neo_agent_budget_*``,
``neo_agent_tool_log_*``). A fan modifier lowers to ``Send`` fan-out, and
LangGraph ``Send`` isolates only the FIRST superstep's input payload — every
later superstep reads the shared channel. So an inline fan over the ReAct region
collapses N>1 branches into one tangled channel. Subgraphs are the ONLY isolation
mechanism for parallel multi-superstep state.

The isolation-correct fix is to AUTO-WRAP the agent's ReAct cycle in an isolated
single-node sub-construct and fan over THAT via the existing subgraph fan path
(``_fan_agent_wrap.wrap_fan_over_agents`` builds the wrapper; ``_add_subgraph`` +
``_wire_oracle``/``_wire_each`` do the rest). This module is the single source of
truth for WHICH fan-over-agent shapes that wrap supports.

Supported today (the proven base case): **Oracle over a self-contained agent/act
node** — no upstream ``inputs``, no ``fan_out_param``, single-type ``outputs``.
Everything else is fail-loud (assembly-time ``ConstructError`` naming the exact
unsupported combination + node), with a filed follow-up bead:

- Each over an agent/act node                          -> neograph-1h8c
- Oracle/Each over an agent/act node WITH inputs       -> neograph-qot6
- Loop over an agent/act node                          -> neograph-gk3e

This module imports only leaf IR types (Node, modifiers, _normalize, errors) — it
must NOT import ``construct``/``compiler`` so ``_construct_validation`` can import
it at assembly time without a cycle. The wrapper-building side (which needs
``Construct``) lives in ``_fan_agent_wrap`` instead.
"""

from __future__ import annotations

from typing import Any, TypeGuard

from neograph._normalize import normalize_outputs
from neograph.errors import ConstructError
from neograph.modifiers import classify_modifiers
from neograph.node import Node

_AGENT_MODES = ("agent", "act")


def _fan_modifier_label(mods: dict[str, Any]) -> str | None:
    """The user-facing label of the fan modifier on this item, if any.

    Loop is checked first, then Each, then Oracle — order only affects which
    single label a stacked combo reports; each stacked case is unsupported
    anyway (only bare Oracle is supported), so the first hit is the right one
    to name in the error.
    """
    if "loop" in mods:
        return "Loop"
    if "each" in mods:
        return "Each"
    if "oracle" in mods:
        return "Oracle"
    return None


def _unsupported_reason(item: Node, mods: dict[str, Any]) -> str | None:
    """Return None when this fan-over-agent shape is supported (Oracle over a
    self-contained agent), else the human-facing reason string for the
    ConstructError. ``item`` is known to be an agent/act Node with a fan modifier.
    """
    fan = _fan_modifier_label(mods)
    if fan is None:
        return None

    if "loop" in mods:
        return (
            "Loop over an agent/act node is not supported. The auto-wrap isolates "
            "the ReAct cycle in a sub-construct, but the loop condition reading the "
            "wrapped subgraph output is an open design question; tracking neograph-gk3e. "
            "Wrap the agent in an explicit sub-construct and Loop over that, or move "
            "the Loop to a scripted/think node."
        )
    if "each" in mods:
        return (
            "Each over an agent/act node is not supported. Delivering the fanned "
            "item across the subgraph boundary to the wrapped agent is an open "
            "design question; tracking neograph-1h8c. Wrap the agent in an explicit "
            "sub-construct and Each over that, or move the Each to a scripted/think "
            "node."
        )
    # Oracle — the one supported family, but only for a self-contained agent.
    if item.inputs:
        return (
            "Oracle over an agent/act node with upstream inputs is not supported. "
            "The auto-wrap synthesizes an isolated sub-construct with an empty input "
            "port; synthesizing a real port from the agent's upstream inputs is an "
            "open design question; tracking neograph-qot6. Wrap the agent in an explicit "
            "sub-construct with input=/output= and Oracle over that."
        )
    if item.fan_out_param is not None:
        return (
            "Oracle over an agent/act node that consumes a fan-out item is not "
            "supported; tracking neograph-qot6. Wrap the agent in an explicit "
            "sub-construct and fan over that."
        )
    if normalize_outputs(item.outputs).is_dict_form:
        return (
            "Oracle over an agent/act node with multi-output (dict-form) outputs is "
            "not supported. The isolating sub-construct has a single output boundary "
            "port; tracking neograph-qot6. Use single-type outputs, or wrap the agent "
            "explicitly."
        )
    return None


def is_supported_fan_over_agent(item: Any) -> TypeGuard[Node]:
    """True iff ``item`` is the ONE supported fan-over-agent shape: Oracle over a
    self-contained agent/act node. This is the predicate the compiler pre-pass
    uses to decide what to auto-wrap."""
    if not isinstance(item, Node) or item.mode not in _AGENT_MODES:
        return False
    _combo, mods = classify_modifiers(item)
    if _fan_modifier_label(mods) is None:
        return False
    return _unsupported_reason(item, mods) is None


def raise_if_unsupported_fan_over_agent(item: Any) -> None:
    """Assembly-time fail-loud: raise a ConstructError for any fan-over-agent
    shape that the auto-wrap does not support. Supported shapes (and non-agent /
    non-fan items) return silently. Called from ``_validate_node_chain`` so the
    error fires when the ``Construct`` is built, never a silent broken graph."""
    if not isinstance(item, Node) or item.mode not in _AGENT_MODES:
        return
    _combo, mods = classify_modifiers(item)
    reason = _unsupported_reason(item, mods)
    if reason is not None:
        raise ConstructError.build(reason, node=item.name)
