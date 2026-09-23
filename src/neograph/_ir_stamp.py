"""The stamping WALK: visit every declared read once, in the validator's order.

Split from ``_ir_normalize``, which owns the RESOLVERS and
is the only module allowed to construct a ``Source``. This module never builds one:
it receives ``Resolved(source)`` and attaches the source it was handed. The split
is therefore not a line-count move -- it is what let the walk grow an arm-scoped
producer set without widening ``SRC_CONSTRUCTION_ALLOWED``, which ``AGENTS.md``'s
first refusal forbids.

ONE WALK, ONE CANDIDATE SET. The walk reuses ``_validation_arms.ArmScopedProducers``
-- the VALIDATOR's own visibility policy -- rather than accumulating its own list.
Two walks with two policies is the defect this step exists to retire: the validator
hid branch-arm producers after a join and accepted the node above the branch, while
the normalizer's own list contained BOTH arms and stamped the false arm, so a
true-arm run handed the reader ``None`` on a green run. The
resolver cannot be called from the validator instead -- ``_ir_normalize`` imports
``_construct_validation``, so the reverse edge is an import cycle -- so the answer
is computed ONCE here, before validation, and validation RENDERS it.

What the walk returns is the refusals, not the stamps: a resolved read is visible
on the item itself (``Node.input_sources`` / ``Construct.port_source``), and an
UNRESOLVED read has nowhere to live, which is exactly why it used to be nothing at
all -- no stamp, no error, and ``None`` at run time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from neograph._construct_validation import ArmScopedProducers
from neograph._ir_branch import iter_item_slots_with_arm_ids
from neograph._ir_consume import with_source
from neograph._ir_fields import Producer, contributed_fields
from neograph._ir_protocols import ConstructItem
from neograph._ir_source import Resolution, Resolved, Unresolved
from neograph._state_keys import StateKeys
from neograph._type_spec import TypeSpecStatic
from neograph.node import Node

if TYPE_CHECKING:
    from neograph.construct import Construct

__all__ = ["read_refusals", "stamp_declared_reads"]

#: Where a construct's unresolved reads are parked between normalization and
#: validation. A private attribute rather than a field: it is derived, it is
#: consumed within one ``Construct.__init__``, and a serialisable "why this did
#: not resolve" record would be a second, weaker spelling of the refusal itself.
_REFUSALS_ATTR = "_neo_read_refusals"

ReadResolver = Callable[[Node, Sequence[Producer], Sequence[Producer]], Resolution | None]
PortResolver = Callable[[TypeSpecStatic | None, Sequence[Producer], Sequence[Producer]], Resolution | None]


def read_refusals(construct: Any) -> dict[str, Unresolved]:
    """The member-name -> ``Unresolved`` table the walk left on ``construct``.

    Read by ``_validation_inputs`` to RENDER a refusal the resolver already
    decided. Empty for a construct built before this ran, or one with nothing
    unresolved.
    """
    return getattr(construct, _REFUSALS_ATTR, None) or {}


def _sub_construct_input(item: ConstructItem) -> TypeSpecStatic | None:
    """``item.input`` when ``item`` is a placed sub-construct, else ``None``.

    ``isinstance(item, Construct)`` is not available here: importing ``construct``
    would close a cycle, and ``_ir_normalize`` documents the same constraint. The
    structural test is exact over what ``construct.nodes`` can hold -- a
    ``_BranchNode`` is never yielded by the slot walk, so a non-``Node`` item with
    a ``nodes`` attribute is a Construct.
    """
    if isinstance(item, Node) or getattr(item, "nodes", None) is None:
        return None
    return getattr(item, "input", None)


def stamp_declared_reads(
    construct: Construct,
    *,
    resolve_read: ReadResolver,
    resolve_port: PortResolver,
) -> None:
    """Resolve every declared read at this construct level, once, and stamp it.

    The resolvers are INJECTED rather than imported, which is what keeps this
    module free of ``Source`` construction (see the module docstring). Both are
    ``_ir_normalize``'s.

    Reads visited, in declaration order, arm-scoped:

    - a ``Node``'s single-type ``inputs=X`` -> ``input_sources[SINGLE_INPUT]``
    - a placed sub-construct's ``input=`` port -> ``Construct.port_source``

    The node stamp is NOT overwritten when already set; the PORT stamp is always
    recomputed. That asymmetry is deliberate and temporary: a never-overwritten
    ``port_source`` means a Construct placed in two parents keeps the FIRST
    parent's answer, and at run time the second parent's port is silently omitted.
    The node side has the same latent defect for a different
    reason (a stale stamp copied across the fan-agent wrapper boundary) and is
    retired by step 7, which makes both sides recompute.
    """
    arms = ArmScopedProducers(declared_output=None)
    arms.seed_subgraph_input(getattr(construct, "input", None), construct.name)
    refusals: dict[str, Unresolved] = {}

    for container, idx, arm_key in iter_item_slots_with_arm_ids(construct):
        item = container[idx]
        visible_map = arms.visible_for(arm_key)
        visible = list(visible_map.values())
        # Producers a BRANCH ARM registered that this item cannot see. Only a
        # TOP-LEVEL item has any: an arm item's hidden set is its SIBLING arm's
        # producers, which are mutually exclusive with its own path and so are
        # simply unreachable -- not shadowing. After a join both arms are on the
        # path that ran, which is what makes resolving by type wrong there.
        shadowed = [p for name, p in arms.all_producers.items() if name not in visible_map] if arm_key is None else []

        if isinstance(item, Node):
            resolution = resolve_read(item, visible, shadowed)
            if isinstance(resolution, Resolved) and item.input_source_field is None:
                container[idx] = item.model_copy(
                    update={"input_sources": with_source(item, StateKeys.SINGLE_INPUT, resolution.source)}
                )
        else:
            resolution = resolve_port(_sub_construct_input(item), visible, shadowed)
            if isinstance(resolution, Resolved):
                container[idx] = item.model_copy(update={"port_source": resolution.source})

        if isinstance(resolution, Unresolved):
            refusals[item.name] = resolution

        for producer in contributed_fields(container[idx]):
            arms.register(producer.field_name, producer, arm_key)

    arms.finalize()
    object.__setattr__(construct, _REFUSALS_ATTR, refusals)
