"""The runtime interpreter of a stamped ``Source``: one case per variant.

The only place a ``Source`` meets state. ``_ir_source.source_channel_kind`` is its
assembly-time sibling -- naming the channel is the part that does not need a bus --
and this is the read that sibling's docstring says it grows into.

A LEAF module, and that is what it is for rather than an accident of tidiness: two
layers read addresses. ``_input_shape`` reads a Node's inputs, ``_subconstruct``
reads a placed child's port, and the assembly-cluster import DAG forbids the second
from importing the first. One interpreter reachable from both is the only
arrangement in which "one reader" is structurally true instead of asserted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, assert_never

from neograph._ir_source import (
    Accumulated,
    EachItem,
    HandoffChannel,
    LastPresent,
    LoopCarry,
    Peer,
    Port,
    Source,
)
from neograph._state_keys import StateKeys
from neograph.di import _unwrap_loop_value

if TYPE_CHECKING:
    from neograph._state_bus import StateBus

__all__ = ["read_source"]


def read_source(bus: StateBus, source: Source, expected: Any, *, label: str) -> Any:
    """Read the ONE value a stamped ``Source`` addresses, or report its absence.

    The runtime interpreter of the address table: one case per variant, and the only
    place a ``Source`` meets state. ``source_channel_kind`` is its assembly-time
    sibling -- naming the channel is the part that does not need a bus.

    Returns ``None`` only for a ``LastPresent`` whose rungs are ALL absent, which its
    caller decides about: a mesh entry's first activation legitimately has nothing on
    the channel and falls through to its feeder, while an omitted port is a defect.
    Every single-rung read is REQUIRED, because a stamped address that holds nothing
    is exactly the silent seam this epic removes.

    Not yet the reader for a Node's single-type ``inputs=`` -- that is step 9, which
    also turns the remaining absences into loud failures.
    """
    match source:
        case Peer(ref=ref):
            # The Loop unwrap belongs here: a Loop-modified producer's field holds an
            # APPEND-LIST and a port declaring T wants the latest element.
            return _unwrap_loop_value(bus.get(ref.field), object)
        case Port():
            return bus.get(StateKeys.SUBGRAPH_INPUT)
        case EachItem():
            return bus.get(StateKeys.EACH_ITEM)
        case HandoffChannel(channel=channel):
            return bus.get(channel)
        case LoopCarry():
            return _unwrap_loop_value(bus.get(label), object)
        case LastPresent(rungs=rungs):
            # LAST present wins, so the rungs are walked in reverse: the stamp orders
            # them lowest-precedence-first, which is the ladder's order inverted.
            for rung in reversed(rungs):
                value = read_source(bus, rung, expected, label=label)
                if value is not None:
                    return value
            return None
        case Accumulated(channel=channel):
            return bus.get(channel)
    assert_never(source)
