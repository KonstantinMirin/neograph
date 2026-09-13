"""``Accumulate[T]`` -- the marker that names an ACCUMULATOR CHANNEL.

A channel is declared where it is WRITTEN, as a dict-form ``Node.outputs`` value::

    Node(..., outputs={"result": Verdict, "readings": Accumulate[Reading]})

The KEY is the channel name. The field on the state bus is that key, UNPREFIXED
-- ``readings``, never ``{node}_readings`` -- and it is shared by every node that
declares it. Many nodes append (every branch of an ``Each``); one downstream node
reads the union with ``inputs={"readings": list[Reading]}``. The merge is the
existing ``_concat_reducer`` (``_state_reducers``), the same operator the agent
``tool_log`` and Oracle collectors already use: a per-write ``list`` extends, a
scalar appends.

This is a MARKER, not a container. ``Accumulate[Reading]`` is a plain
``typing`` generic alias -- it passes ``_validate_type_spec`` because it has an
``__origin__`` -- and it is stripped at ``normalize_outputs``, the single
discriminator of ``Node.outputs``, into ``list[Reading]`` plus an entry in
``NormalizedOutputs.accumulator_keys``. Nothing downstream of the discriminator
ever sees the marker; every reader that asks "which fields does this item write"
goes through ``contributed_fields``, which emits the channel as an unprefixed
``Producer(is_accumulator=True)``.

ORDERING, defined: the union is ordered WITHIN a branch and UNORDERED ACROSS
branches -- ``Each`` collects ``Send()`` results in arrival order, the rule the
``list[X]``-consumer-of-Each caveat already documents. A reader wanting
determinism sorts on a stable key it placed in the element.

Why a marker in the outputs value and not a construct-level declaration: the
alternative (``Construct(channels=...)`` projected onto a per-node IR field by a
new normalizer) is the Portal-scale shape, and this is not a new mechanism -- it
is LangGraph's own ``Annotated[list, add]`` surfaced. Declaring the channel at
the write site keeps ``contributed_fields`` per-item and adds no IR field, no
normalizer and no builder kwarg. See GH #16 / neograph-iq4a3.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, get_args, get_origin

from neograph.errors import ConfigurationError

T = TypeVar("T")


class Accumulate(Generic[T]):
    """Marker: this dict-form output key is an accumulator channel of ``T``.

    Never instantiated. Subscript it: ``Accumulate[Reading]``.
    """

    __slots__ = ()

    def __init__(self) -> None:  # pragma: no cover -- defensive
        raise ConfigurationError.build(
            "Accumulate[T] is a marker for Node.outputs, not a value",
            expected='outputs={"readings": Accumulate[Reading]}',
            found="Accumulate(...) instantiated",
            hint="subscript the class; never call it",
        )


def accumulate_element(spec: Any) -> Any | None:
    """The element type ``T`` when ``spec`` is ``Accumulate[T]``, else ``None``.

    The ONE test for "is this outputs value a channel marker". ``normalize_outputs``
    calls it; nothing else needs to, because after normalization the marker is
    gone.
    """
    if get_origin(spec) is Accumulate:
        args = get_args(spec)
        if len(args) == 1:
            return args[0]
    return None
