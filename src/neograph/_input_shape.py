"""Read-side: classifies a node's input shape against state and extracts the typed input."""

from __future__ import annotations

from enum import Enum
from typing import Any, assert_never

from neograph._ir_source import EachItem, LastPresent, LoopCarry
from neograph._normalize import normalize_inputs, primary_output_field
from neograph._state_bus import StateBus
from neograph._state_keys import StateKeys
from neograph.describe_type import _admits_none
from neograph.di import _isinstance_safe, _unwrap_each_dict, _unwrap_loop_value, read_upstream
from neograph.errors import ExecutionError
from neograph.naming import field_name_for
from neograph.node import Node


class InputShape(Enum):
    """Classification of how a node reads its input from state."""

    NONE = "none"
    LOOP_REENTRY = "loop_reentry"
    EACH_ITEM = "each_item"
    FAN_IN_DICT = "fan_in_dict"
    SINGLE_TYPE = "single_type"


def _classify_input_shape(state: StateBus, node: Node) -> InputShape:
    """Determine which extraction strategy applies. Priority order matters."""
    if node.inputs is None:
        return InputShape.NONE

    # A Loop read is stamped with BOTH its arrivals, seed-then-carry, so the question
    # here is whether the CARRY RUNG is present -- not whether this node happens to
    # have a non-empty list somewhere. Presence is per rung type, and a LoopCarry's
    # append-list is present exactly when it is non-empty (iteration 1+).
    if _has_carry_rung(node):
        own_field = primary_output_field(field_name_for(node.name), node.outputs)
        # StateBus.get optional: this IS the presence test for the carry rung --
        # absence means iteration 0, where the seed rung answers instead.
        own_val = state.get(own_field)
        if isinstance(own_val, list) and own_val:
            return InputShape.LOOP_REENTRY

    # The ADDRESS decides, not the state. This was a presence check plus an
    # isinstance probe against the declared type -- so any node that happened to see
    # `neo_each_item` with a matching type would have read the item, and an Each node
    # whose item failed the probe would have fallen through to read a PEER instead.
    # Both are decisions the assembly-time stamp already made correctly.
    if isinstance((node.input_sources or {}).get(StateKeys.SINGLE_INPUT), EachItem):
        return InputShape.EACH_ITEM

    if normalize_inputs(node.inputs).is_dict_form:
        return InputShape.FAN_IN_DICT

    return InputShape.SINGLE_TYPE


def _has_carry_rung(node: Node) -> bool:
    """True when this node's read is stamped with a ``LoopCarry`` rung.

    Single-type reads carry it inside a ``LastPresent`` under the sentinel; dict-form
    reads carry it on the destination KEY. Either way the address says the carry is
    one of this read's arrivals, which is what used to be inferred from
    ``classify_modifiers`` plus a probe of state.
    """
    table = node.input_sources or {}
    single = table.get(StateKeys.SINGLE_INPUT)
    if isinstance(single, LastPresent) and any(isinstance(rung, LoopCarry) for rung in single.rungs):
        return True
    return node.carry_param is not None


def _extract_loop_reentry(state: StateBus, node: Node) -> Any:
    """Read from the node's own append-list on loop iteration 1+."""
    own_field = primary_output_field(field_name_for(node.name), node.outputs)
    # REQUIRED: _classify_input_shape already confirmed own_val is non-empty list.
    own_val = state.get_required(own_field, node_label=node.name)
    latest = own_val[-1]

    ni = normalize_inputs(node.inputs)
    if not ni.is_dict_form:
        return latest

    by_name = ni.by_name
    # Single-key dict: always self-reference
    if len(by_name) == 1:
        first_key = next(iter(by_name))
        return {first_key: latest}

    # Multi-key dict: the carry's destination is the key STAMPED with it. The stamp
    # is written once, by the normalizer, with the predicate validation uses -- so
    # the slot the run binds is the slot validation approved. This used to recompute
    # the destination here with a LOOSER default predicate, place
    # `latest` into whichever sibling key read None, and fall back to
    # next(iter(by_name)) -- a positional guess -- when no destination resolved.
    dest = node.carry_param
    if dest is None:
        raise ExecutionError.build(
            f"loop node '{node.name}' re-entered with no stamped carry destination",
            expected="a dict-form input key stamped LoopCarry by the normalizer",
            found=f"input_sources={node.input_sources!r}",
            hint="assembly refuses a dict-form Loop whose output fits no input slot, so this is a bug in neograph",
            node=node.name,
        )
    result = {}
    for key, expected_type in by_name.items():
        if key == dest:
            result[key] = latest
            continue
        # REQUIRED: a sibling key is an ordinary declared Peer read, and its producer
        # runs before the loop, so its field is present on every iteration. Reading it
        # optionally and substituting the CARRY on absence put a different value --
        # of a type the slot need not accept -- where the sibling's own belonged.
        result[key] = read_upstream(state, key, expected_type, required=True, node_label=node.name)
    return result


def _extract_each_item(state: StateBus, node: Node) -> Any:
    """Read the fan-out item from neo_each_item."""
    # REQUIRED: dispatched only after classification confirmed EACH_ITEM presence.
    return state.get_required(StateKeys.EACH_ITEM, node_label=node.name)


def _extract_fan_in_dict(state: StateBus, node: Node) -> dict[str, Any]:
    """Read each named upstream from state by key.

    ``node.fan_out_param`` is set once at Construct construction (see
    ``neograph._ir_normalize.normalize_ir``) so all three API surfaces —
    declarative, ``@node``, programmatic/YAML — produce identical IR by
    the time the runtime sees the node.

    ``node.handoff_param`` (the reserved ``"handoff"`` inputs key on a Portal
    mesh member) reads the shared mesh channel instead of a peer field, because
    a member entered from ANY caller cannot read a specific upstream's field
    (design §3.3). The entry-keyed channel field name lives on
    ``node.handoff_channel`` — a node-self-contained IR field stamped by the
    normalizer (decision D10), read here WITHOUT any signature threading, exactly
    like ``fan_out_param`` reads the fixed ``EACH_ITEM`` slot. Read is OPTIONAL:
    a member reached via a hop always has the channel populated by the previous
    hop's ``Command`` update, but an entry declaring a ``handoff`` param on its
    FIRST activation legitimately sees ``None`` (the channel default).
    """
    ni = normalize_inputs(node.inputs)
    assert ni.is_dict_form
    result: dict[str, Any] = {}
    for input_name, expected_type in ni.by_name.items():
        if node.handoff_param is not None and input_name == node.handoff_param:
            # The DECLARED TYPE decides whether absence is a value. A non-entry member
            # only ever arrives by hop, so its payload is always there and a missing one
            # is a defect; the mesh ENTRY's first activation is linear, so it reads
            # nothing there and types the key `payload | None` to say so (validated at
            # assembly, _validation_portal). Reading it optionally for everyone made the
            # non-entry case silent -- absence is a value only where the type admits one.
            if node.handoff_channel is None:
                value = None
            elif _admits_none(expected_type):
                # StateBus.get optional: the entry's first activation, declared Optional.
                value = state.get(node.handoff_channel)
            else:
                value = state.get_required(node.handoff_channel, node_label=node.name)
        elif input_name == node.fan_out_param:
            # REQUIRED: node IS the fan-out target; EACH_ITEM is the dispatched value.
            value = state.get_required(StateKeys.EACH_ITEM, node_label=node.name)
        else:
            # REQUIRED: fan-in upstreams guaranteed by _validate_node_chain.
            value = read_upstream(state, input_name, expected_type, required=True, node_label=node.name)
        result[input_name] = value
    return result


def _extract_single_type(state: StateBus, node: Node) -> Any:
    """Read the ONE state field that satisfies the node's single-type ``inputs=``.

    ``node.input_source_field`` is resolved at ASSEMBLY by
    ``_ir_normalize.resolve_single_type_source``, so this is a
    named read, not a search. It replaced a forward scan over ``state.keys()``
    that returned the first ``isinstance`` match -- which meant the whole state
    bag competed to be this node's input (framework bookkeeping included), and
    which disagreed with the Agent Spec export's own reverse scan, so a green run
    and its exported artifact wired different edges.

    ONE field, and no fallback. A list of framework port keys used to be consulted
    after the stamp, for two shapes whose address was wrong rather than missing: a
    node copied into the fan-agent wrapper kept an address into its parent's state,
    and ``run_isolated`` had no construct to resolve one at all. Both are stamped
    now -- to the port, which is where their value actually arrives -- and the
    fallback is deleted with them. While it existed no test could tell "resolved
    correctly" from "resolved wrongly and rescued", which is the property that
    matters more than the two shapes it served.

    ``None`` means there is nothing to resolve, NOT that resolution was
    ambiguous: two eligible producers raise at assembly.
    """
    field = node.input_source_field
    if field is None:
        return None
    # StateBus.get optional: the resolved field may be absent on this superstep (a
    # Loop's iteration-0 read, an unreached branch arm's producer). Step 9 is where
    # absence stops being a value.
    val = _unwrap_each_dict(_unwrap_loop_value(state.get(field), node.inputs), node.inputs)
    if val is not None and _isinstance_safe(val, node.inputs):
        return val
    return None


def _extract_input(state: StateBus, node: Node) -> Any:
    """Extract typed input from state — pure dispatch to shape helpers.

    A Portal member's reserved ``"handoff"`` input reads its entry-keyed mesh
    channel from ``node.handoff_channel`` (a normalizer-stamped IR field, decision
    D10) inside ``_extract_fan_in_dict`` — no signature threading needed.
    """
    shape = _classify_input_shape(state, node)
    match shape:
        case InputShape.NONE:
            return None
        case InputShape.LOOP_REENTRY:
            return _extract_loop_reentry(state, node)
        case InputShape.EACH_ITEM:
            return _extract_each_item(state, node)
        case InputShape.FAN_IN_DICT:
            return _extract_fan_in_dict(state, node)
        case InputShape.SINGLE_TYPE:
            return _extract_single_type(state, node)
    assert_never(shape)


def _extract_context(state: StateBus, node: Node) -> dict[str, Any] | None:
    """Extract the node's declared context fields from state for LLM nodes.

    Returns ``{context_name: state_value}`` if the node declares context fields,
    or None if none is configured.

    Read-side input shaping (sibling of ``_extract_input``); lives here so both
    node-body executors — the straight-line ``_execute`` lifecycle and the
    inline agent cycle (``_agent_cycle``) — reuse ONE implementation. It was
    parked in ``_execute`` only while ``_execute_node`` was its sole caller.

    The return type used to be ``dict[str, str]``, produced by a ``cast(str, ...)``
    that nothing backed: ``state.py`` types context fields ``Any`` and the
    validator only checks that SOME upstream produces the field, never its type.
    A cast is erased at runtime, so all it did was tell the next reader something
    untrue — and it was untrue: live Pydantic models flowed down a channel
    annotated as text. Deleting it is neograph-ufqr7; the channel becomes text
    for real, one layer up, where ``_llm_render`` renders it through the one
    ladder.

    Reads go through ``read_upstream`` like every other peer-field read
    see neograph-13k4i. ``expected_type=str`` is what the channel wants:
    the Loop unwrap fires (a context field naming a looping node means its LATEST
    value, not its whole history), and the Each unwrap correctly no-ops, since a
    fan-out dict has no single latest element to pick.
    """
    if not node.context:
        return None
    # REQUIRED: context fields are validator-guaranteed (see
    # _construct_validation.py); missing → wiring bug, fail loud rather than
    # render the literal string "None" into the LLM prompt.
    return {name: read_upstream(state, name, str, required=True, node_label=node.name) for name in node.context}
