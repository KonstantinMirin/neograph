"""Assembly-time type validation for Construct chains.

Extracted from construct.py to keep the declaration module lean. Consumed
only by Construct.__init__ (which imports _validate_node_chain + ConstructError).

Validation rule: walk the node list in order, accumulating producers (earlier
nodes and sub-constructs, plus the Construct's own input port if declared).
For each item with a declared input, verify some upstream producer supplies
a compatible value — directly, or through an `Each` modifier whose `over`
path resolves to `list[input_type]`.

Defers to runtime isinstance-scanning when evidence is insufficient:
  - No upstream producers (first-of-chain — input comes from run(input=...))
  - dict / non-class input types (multi-field or raw extraction)
  - Each modifier whose root segment doesn't match a known producer

This module is the ORCHESTRATOR + the single public seam for the validation
cluster, per neograph-gig0. The per-rule logic lives in flat peer modules:
  - ``_validation_types``     — type-compat primitives + shared vocabulary
  - ``_validation_inputs``    — fan-in + Each fan-out consumer-input checks
  - ``_validation_modifiers`` — Loop self-edge/construct + Oracle merge hooks
The cluster is package-private; external callers import the names re-exported
below from ``neograph._construct_validation``, never from a ``_validation_*``
sub-module.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum

from neograph._ir_protocols import ConstructLike
from neograph._state_keys import StateKeys
from neograph._validation_inputs import _check_item_input
from neograph._validation_modifiers import (
    _validate_merge_hooks,
    validate_loop_construct,
    validate_loop_self_edge,
)
from neograph._validation_types import (
    NodeItem,
    Producer,
    ProducerMap,
    _declared_output,
    _fmt_type,
    _is_construct_like,
    _source_location,
    _types_compatible,
    effective_producer_type,
    effective_producer_type_for,
)
from neograph.di import DIKind as _DIKind
from neograph.errors import ConstructError
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node

# Re-export seam per neograph-gig0: these names form the validation cluster's
# public surface. construct.py imports _validate_node_chain + ConstructError;
# _construct_graph imports _types_compatible + effective_producer_type;
# modifiers.py imports validate_loop_self_edge + validate_loop_construct;
# tests import _types_compatible + effective_producer_type. Listed in __all__
# so the re-export is explicit (no star-import) and ruff treats it as used.
__all__ = [
    "ConstructError",
    "NodeItem",
    "Producer",
    "ProducerMap",
    "ValidationMode",
    "_types_compatible",
    "_validate_node_chain",
    "effective_producer_type",
    "effective_producer_type_for",
    "validate_loop_construct",
    "validate_loop_self_edge",
]


class ValidationMode(Enum):
    """Why a construct is being walked — names the state the old
    ``context_checkable`` boolean encoded implicitly.

    STANDALONE: top-level/standalone walk; no parent producer set is known.
    A sub-construct walked standalone defers its inner ``context=`` checks
    until the parent re-walks it IN_CONTEXT.

    IN_CONTEXT: walked as a nested sub-construct with the parent's producers
    supplied as ``ambient`` — inner ``context=`` references are checkable now.
    """

    STANDALONE = "standalone"
    IN_CONTEXT = "in_context"


def _validate_node_chain(
    construct: ConstructLike,
    *,
    ambient_producers: ProducerMap | None = None,
) -> None:
    """Walk the node list, verifying each input has a compatible producer.

    ``ambient_producers`` carries the PARENT's producers when this construct is
    being validated as a nested sub-construct. Inner nodes' ``context``
    references are checked against the union of ambient + locally collected
    producers, so context typos are caught at arbitrary depth.

    When called standalone (``ambient_producers is None`` →
    ``ValidationMode.STANDALONE``) on a sub-construct
    (``construct.input is not None``), inner-node context checks are DEFERRED
    because the parent's producer set is unknown; the parent's own validation
    pass will re-walk this sub-construct ``IN_CONTEXT`` with ambient supplied.
    """
    mode = (
        ValidationMode.STANDALONE
        if ambient_producers is None
        else ValidationMode.IN_CONTEXT
    )
    producers: ProducerMap = OrderedDict()

    # The Construct's own input port is the first producer, if declared —
    # used by inner nodes that read from `neo_subgraph_input`.
    if construct.input is not None:
        producers[StateKeys.SUBGRAPH_INPUT] = Producer(
            field_name=StateKeys.SUBGRAPH_INPUT,
            effective_type=construct.input,
            label=f"construct '{construct.name}' input port",
        )

    for item in construct.nodes:
        # Node has `inputs` (plural) since neograph-kqd.1. Construct still
        # uses `input` (singular) as its sub-construct boundary port —
        # that's handled above via construct.input, not here. The
        # isinstance check makes the two-field contract explicit.
        if isinstance(item, Node):
            input_type = getattr(item, "inputs", None)
        else:
            # Construct items use .input (singular boundary port).
            input_type = getattr(item, "input", None)
        if input_type is not None:
            # Warn on single-type inputs (isinstance scan) — dict-form is safer.
            if (
                isinstance(item, Node)
                and not isinstance(input_type, dict)
                and producers  # not the first node
            ):
                import warnings
                warnings.warn(
                    f"Node '{item.name}': single-type inputs={input_type.__name__ if hasattr(input_type, '__name__') else input_type} "
                    f"relies on O(N) isinstance scan at runtime. "
                    f"Use dict-form inputs={{'{field_name_for(item.name)}': {input_type.__name__ if hasattr(input_type, '__name__') else input_type}}} "
                    f"for explicit named resolution.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            _check_item_input(construct, item, input_type, producers)

        # Validate context= references against ambient (parent) + local producers.
        # When walked STANDALONE on a sub-construct (ambient is None and
        # construct.input is not None), inner-node context is DEFERRED: the
        # parent's recursive call will re-validate IN_CONTEXT with ambient.
        context_checkable = (
            mode is ValidationMode.IN_CONTEXT or construct.input is None
        )
        if isinstance(item, Node) and item.context and context_checkable:
            known_fields = set(ambient_producers or ()) | set(producers)
            for ctx_name in item.context:
                ctx_field = field_name_for(ctx_name)
                if ctx_field not in known_fields:
                    raise ConstructError.build(
                        f"references context='{ctx_name}' but no upstream node "
                        f"produces a field with that name",
                        found=f"known upstream fields: {sorted(known_fields) or '(none)'}",
                        node=item.name,
                        location=_source_location(),
                    )

        # Sub-construct items: recurse with the current producer union as the
        # ambient set so inner-node context checks fire at arbitrary depth.
        # The TypeGuard narrows `item` to ConstructLike — no untyped cast.
        if _is_construct_like(item) and context_checkable:
            ambient_for_recursion: ProducerMap = OrderedDict(ambient_producers or {})
            ambient_for_recursion.update(producers)
            _validate_node_chain(
                item,
                ambient_producers=ambient_for_recursion,
            )

        output_type = _declared_output(item)
        name = getattr(item, "name", None)
        if output_type is not None and name is not None:
            field_name = field_name_for(name)

            # Dict-form outputs (neograph-1bp.4): register one producer per
            # output key, with modifier wrapping applied independently per key.
            if isinstance(item, Node) and isinstance(output_type, dict):
                for output_key, key_type in output_type.items():
                    key_field = output_field_name(field_name, output_key)
                    key_label = f"node '{name}' output '{output_key}'"
                    # Per-key modifier rule via the single source of truth —
                    # the same helper the whole-node producer path uses.
                    producer_type = effective_producer_type_for(
                        key_type, item.modifier_set
                    )
                    producers[key_field] = Producer(
                        field_name=key_field,
                        effective_type=producer_type,
                        label=key_label,
                    )
            else:
                label = (
                    f"node '{name}'"
                    if isinstance(item, Node)
                    else f"sub-construct '{name}'"
                )
                # Shared helper decides the modifier-adjusted state-bus type.
                producers[field_name] = Producer(
                    field_name=field_name,
                    effective_type=effective_producer_type(item),
                    label=label,
                )

        # Loop + skip_when without skip_value is surprising.
        # The counter still increments so the loop exits, but re-entry
        # reads stale state. Warn rather than reject — the behavior is
        # valid, just easy to misuse.
        if isinstance(item, Node) and item.modifier_set.loop is not None:
            if item.skip_when is not None and item.skip_value is None:
                import structlog
                structlog.get_logger(__name__).error(
                    "loop_skip_when_no_skip_value",
                    node=item.name,
                    msg=(
                        f"Node '{item.name}' has Loop + skip_when but no skip_value. "
                        f"When skip_when fires inside a Loop, skip_value is recommended "
                        f"to provide the output for that iteration."
                    ),
                )

        # (Loop reenter validation removed — Loop.reenter no longer exists.
        # Multi-node loops use Loop on Construct instead.)

        # Validate @merge_fn state params.
        # When a merge_fn has from_state params, verify each references a
        # known upstream producer and the type is compatible.
        if isinstance(item, Node):
            oracle = item.modifier_set.oracle
            if oracle is not None and isinstance(getattr(oracle, 'merge_fn', None), str):
                # Function-local + leaf source: get_merge_fn_metadata lives in
                # _sidecar; a module-level import here cycles via
                # _sidecar -> _di_classify -> _construct_validation (ConstructError).
                from neograph._sidecar import get_merge_fn_metadata
                assert oracle.merge_fn is not None
                meta = get_merge_fn_metadata(oracle.merge_fn)
                if meta is not None and meta[1]:
                    _, merge_param_res = meta
                    item_field = field_name_for(item.name)
                    known_producers = {
                        p.field_name: (p.effective_type, p.label)
                        for p in producers.values()
                    }
                    for pname, binding in merge_param_res.items():
                        if binding.kind != _DIKind.FROM_STATE:
                            continue
                        expected_type = binding.inner_type
                        # Self-reference: merge runs before node output is written
                        if pname == item_field:
                            raise ConstructError.build(
                                f"merge_fn '{oracle.merge_fn}' param '{pname}' "
                                f"references the node's own output",
                                found=f"self-reference to '{pname}'",
                                hint="the merge barrier runs before the node's "
                                     "output is written — this can never resolve",
                                node=item.name,
                                construct=construct.name,
                                location=_source_location(),
                            )
                        # Unknown producer
                        if pname not in known_producers:
                            raise ConstructError.build(
                                f"merge_fn '{oracle.merge_fn}' param '{pname}' "
                                f"does not match any upstream node",
                                found=f"known producers: {sorted(known_producers.keys())}",
                                node=item.name,
                                construct=construct.name,
                                location=_source_location(),
                            )
                        # Type mismatch
                        prod_type, prod_label = known_producers[pname]
                        if (prod_type is not None
                                and isinstance(expected_type, type)
                                and not _types_compatible(prod_type, expected_type)):
                            raise ConstructError.build(
                                f"merge_fn '{oracle.merge_fn}' param '{pname}' "
                                f"type mismatch with {prod_label}",
                                expected=_fmt_type(expected_type),
                                found=_fmt_type(prod_type),
                                node=item.name,
                                construct=construct.name,
                                location=_source_location(),
                            )

        # Validate merge hook signatures (merge_pre_process, merge_post_process,
        # merge_fallback) when Oracle has merge_prompt set.
        if isinstance(item, Node):
            oracle = item.modifier_set.oracle
            if oracle is not None and oracle.merge_prompt is not None:
                _validate_merge_hooks(oracle, item, construct.name)

    # Sub-construct output boundary contract: if construct.output is declared,
    # at least one internal node must produce a compatible type.
    # Exclude neo_subgraph_input — the input port is NOT a valid producer
    # for the output contract.
    if construct.output is not None and producers:
        declared_output = construct.output
        internal_producers = [
            p for p in producers.values() if p.field_name != StateKeys.SUBGRAPH_INPUT
        ]
        for p in internal_producers:
            if p.effective_type is not None and _types_compatible(p.effective_type, declared_output):
                break
        else:
            producer_summary = "\n".join(
                f"    - {p.label}: {_fmt_type(p.effective_type)}"
                for p in internal_producers
                if p.effective_type is not None
            )
            raise ConstructError.build(
                f"declares output={_fmt_type(declared_output)} but no "
                f"internal node produces a compatible type",
                expected=_fmt_type(declared_output),
                found=f"internal producers:\n{producer_summary}",
                construct=construct.name,
                location=_source_location(),
            )
