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

Supported today: **Oracle, Each, or Loop over an agent/act node** that is either
self-contained OR consumes a SINGLE upstream producer — single-type ``inputs``
(port is that type) or a single-key dict-form fan-in (port is that key's type),
plus any DI (``FromInput``/``FromConfig``) params which ride the forwarded config
and never enter ``inputs``. Input-port synthesis (see neograph-qot6) builds the
sub-construct's ``input=`` boundary from that single value; for Each the fanned
``neo_each_item`` IS delivered as that single port value (see neograph-1h8c); Loop
reads the agent's typed output surfaced onto the parent field (see neograph-gk3e).
**Oracle** additionally supports MULTIPLE dict-form producers via packer-port
synthesis (see neograph-qzrv). Everything else is fail-loud (assembly-time
``ConstructError`` naming the exact unsupported combination + node):

- Each/Loop over an agent with MULTIPLE upstream inputs   -> packer wired for Oracle only
- Oracle/Each/Loop over an agent with dict-form OUTPUTS   -> single output boundary port
- Each over a SELF-CONTAINED agent                        -> no port for the fanned item

This module imports only leaf IR types (Node, modifiers, _normalize, errors) — it
must NOT import ``construct``/``compiler`` so ``_construct_validation`` can import
it at assembly time without a cycle. The wrapper-building side (which needs
``Construct``) lives in ``_fan_agent_wrap`` instead.
"""

from __future__ import annotations

from typing import Any, TypeGuard

from neograph._normalize import normalize_inputs, normalize_outputs
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
    """Return None when this fan-over-agent shape is supported (Oracle/Each/Loop
    over an agent with zero or one upstream producer, or Oracle over an agent with
    multiple dict-form producers via packer synthesis), else the human-facing
    reason string for the ConstructError. ``item`` is known to be an agent/act Node
    with a fan modifier.
    """
    fan = _fan_modifier_label(mods)
    if fan is None:
        return None

    # Oracle, Each, and Loop share ONE input-shape support gate: a self-contained
    # agent (empty port) OR a SINGLE upstream producer maps cleanly onto the
    # sub-construct's single-value input boundary (``neo_subgraph_input``) via
    # input-port synthesis (see neograph-qot6): single-type ``inputs`` -> port is that
    # type; a single-key dict-form fan-in -> port is that key's type. What stays
    # fail-loud for ALL THREE:
    #   - MULTIPLE distinct dict-form producers: the single-value boundary can't
    #     carry N values without a synthesized packer (tracking neograph-qzrv).
    #   - dict-form (multi-output) OUTPUTS: the sub-construct has one output port
    #     (tracking neograph-qzrv).
    # Differences by fan family:
    #   - Each DELIVERS the fanned ``neo_each_item`` as that single port value
    #     (see neograph-1h8c), so a fan-out receiver (``fan_out_param``) is the expected
    #     shape; Each over a self-contained agent has no port for the item and stays
    #     fail-loud (guarded below).
    #   - Loop reads the agent's typed output surfaced onto the parent field and
    #     wires a conditional back-edge over the isolated sub-construct via the
    #     existing subgraph-loop path (see neograph-gk3e).
    #   - Oracle never fans a per-item value, so a fan-out receiver on an
    #     Oracle-over-agent is a user error (guarded below).
    if "oracle" in mods and item.fan_out_param is not None:
        return (
            "Oracle over an agent/act node that consumes a fan-out item is not "
            "supported (Oracle does not fan a per-item value); use Each to fan over "
            "a collection, or wrap the agent in an explicit sub-construct."
        )
    # `fan` (from _fan_modifier_label above) is the single label site — do not
    # re-derive it here (review DRY-L2). Stacked combos are unsupported anyway,
    # so the first-hit label is the right one to name in these errors.
    ni = normalize_inputs(item.inputs)
    if "each" in mods and ni.is_none:
        # Design (2), neograph-1h8c: the fanned item is delivered AS the
        # sub-construct's single-value input port. A self-contained agent declares
        # no input, so the per-item value has nowhere to land — every isolated
        # cycle would run on empty input and produce identical results keyed by
        # distinct Each keys (a silent broken fan). Require a consumed input.
        return (
            "Each over a self-contained agent/act node is not supported: the fanned "
            "item has no input port to be delivered to. Declare inputs= on the agent "
            "(single-type or a single-key dict-form) so the Each item can be mapped "
            "onto the wrapped sub-construct's input boundary."
        )
    if ni.is_dict_form and len(ni.by_name) > 1 and "oracle" not in mods:
        # Oracle multi-input is now supported via packer-port synthesis
        # (see neograph-qzrv): a synthesized parent packer bundles the N upstreams into
        # one port model + inner per-key unpackers re-expose the original keys.
        # Each/Loop multi-input is NOT wired (Each also delivers a fanned item;
        # Loop also feeds back its output — both interact with the same single-value
        # boundary the packer occupies), so they stay fail-loud with a precise error.
        return (
            f"{fan} over an agent/act node with multiple upstream inputs is not "
            "supported. The isolating sub-construct has a single-value input "
            "boundary (neo_subgraph_input); packer-port synthesis is wired for Oracle "
            f"only (tracking neograph-qzrv). Wrap the agent in an explicit "
            f"sub-construct with input=/output= and {fan} over that."
        )
    if normalize_outputs(item.outputs).is_dict_form:
        return (
            f"{fan} over an agent/act node with multi-output (dict-form) outputs is "
            "not supported. The isolating sub-construct has a single output boundary "
            f"port; tracking neograph-qzrv. Use single-type outputs, or wrap the agent "
            "explicitly."
        )
    return None


def is_supported_fan_over_agent(item: Any) -> TypeGuard[Node]:
    """True iff ``item`` is a supported fan-over-agent shape: Oracle/Each/Loop over
    an agent/act node that is self-contained or has a single upstream producer, or
    Oracle over an agent with multiple dict-form producers (packer synthesis). This
    is the predicate the compiler pre-pass uses to decide what to auto-wrap."""
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
