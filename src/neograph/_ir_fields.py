"""What an item CONTRIBUTES to the state bus, and what it could CONSUME from it.

``contributed_fields`` is THE enumeration of an item's write-set -- which state
fields it writes, in declaration order, each with its declared and its
modifier-adjusted type. ``declared_output_fields`` and ``item_field_names`` are
projections of it; the validator registers what it returns; the schema
fingerprint hashes it. Before neograph-yz69e six functions each answered this
question for themselves and disagreed about the Portal ``{node}_dispatch`` field,
silently, at every one of them.

``Producer`` and ``effective_producer_type``/``effective_producer_type_for`` live
here for the same reason: a producer RECORD is a fact about what an item
contributes, not a fact about validation, so the module that enumerates the
fields owns the record describing one and the rule giving it a type. They moved
down from ``_validation_types``, which imports and re-exports them, so the
validation cluster's public seam is unchanged for every caller.

The CONSUME-side rules -- which upstream field or channel satisfies a given input --
live in ``_ir_consume``. They never shared a helper with this side in either
direction; the split in neograph-yz69e made an existing seam visible rather than
creating one.

Why they live in a LEAF rather than in ``_ir_normalize`` (neograph-9axw6.2).
``_ir_normalize`` imports ``_construct_validation``, so a validation-cluster module
wanting either rule could not import it at module level: ``_validation_inputs``
reached ``fan_out_candidates`` through a FUNCTION-LOCAL import carrying an
allowlist row in ``tests/test_guards_sidecar_imports.py``, whose own comment said
it "retires when" the cycle goes. Moving the rules to a leaf retires it -- the
allowlist SHRANK by one row rather than being re-keyed, because the architecture
stopped needing the exemption.

This module imports only leaves and near-leaves (``naming``, ``_normalize``,
``_ir_protocols``, ``node``), none of which reach the validation cluster, so it is
importable at module level from both sides of the old cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from neograph._ir_branch import iter_with_arms
from neograph._ir_protocols import ConstructItem
from neograph._normalize import _declared_output, normalize_outputs
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph._state_keys import StateKeys
from neograph._type_spec import TypeSpecStatic
from neograph.errors import NeographError
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node
from neograph.spec_types import lookup_type

__all__ = [
    "Producer",
    "boundary_member_name",
    "contributed_fields",
    "declared_output_fields",
    "effective_producer_type",
    "effective_producer_type_for",
    "item_field_names",
]


def contributed_fields(item: ConstructItem) -> list[Producer]:
    """Every state field ``item`` writes, in declaration order, with both its
    declared and its modifier-adjusted type.

    **The single enumeration of the item write-set**, per neograph-yz69e. neograph
    already monopolised how to SPELL a field name (``output_field_name``) and how
    to READ one back (``split_output_field``); this is the third monopoly, the one
    that was missing while six functions each enumerated the set for themselves:

    - dict-form ``Node.outputs`` -> one ``{base}_{key}`` per key (NO bare base)
    - single-type ``Node.outputs`` -> the bare ``base``
    - ``Node.outputs is None`` -> nothing
    - sub-construct (non-Node) -> the bare ``base``
    - **Portal ``route="decide"`` -> ADDITIONALLY ``{base}_dispatch``**, the field
      the dispatched flow's typed result lands on

    That last rule is the one every copy but the validator's was missing, and its
    absence was silent at each: a single-type consumer validated green and got
    ``None``; a sub-construct boundary the result should satisfy raised at run
    time; and a changed ``Portal(output=)`` opened the resume gate with nothing to
    attribute it to, so the run resumed from the tip with a stale result.

    **Per-ITEM, and deliberately never walks a construct.** Three callers need
    three different arm policies -- ``normalize_ir``'s peer set is top-level only,
    ``_stamp_single_type_sources`` is arm-SCOPED (an arm must never see its
    sibling arm's producers, which is the cross-arm read validation refuses), and
    ``item_field_names`` is arm-inclusive and flat. A version that walked
    internally could serve at most one of them and would silently destroy the arm
    scoping. The walk, and the arm policy, stay with the caller.

    **Order is load-bearing**, so this returns a list: ``item_field_names`` reads
    it last-declared-first for boundary precedence, and dict-form keys are
    contributed in key order. The name SET is the lossy projection, never the
    reverse.

    Types come from ``effective_producer_type``/``effective_producer_type_for``,
    which stay the modifier-aware type authority -- this calls them, it does not
    re-derive them.
    """
    name = getattr(item, "name", None)
    if name is None:
        return []
    base = field_name_for(name)

    if not isinstance(item, Node):
        return [
            Producer(
                field_name=base,
                effective_type=effective_producer_type(item),
                declared_type=_declared_output(item),
                label=f"sub-construct '{name}'",
            )
        ]

    label = f"node '{name}'"
    is_loop = item.modifier_set is not None and item.modifier_set.loop is not None
    no = normalize_outputs(item.outputs)

    out: list[Producer] = []
    if no.is_dict_form:
        out.extend(
            Producer(
                field_name=output_field_name(base, key),
                effective_type=effective_producer_type_for(key_type, item.modifier_set),
                declared_type=key_type,
                # Per-key label, rendered verbatim in validation errors.
                label=f"node '{name}' output '{key}'",
                is_loop=is_loop,
            )
            for key, key_type in no.all_keys.items()
        )
    elif not no.is_none:
        out.append(
            Producer(
                field_name=base,
                effective_type=effective_producer_type(item),
                declared_type=no.primary,
                label=label,
                is_loop=is_loop,
            )
        )

    dispatch = _dispatch_producer(item, base, name)
    if dispatch is not None:
        out.append(dispatch)
    return out


def _dispatch_producer(node: Node, base: str, name: str) -> Producer | None:
    """The ``{base}_dispatch`` producer of a Portal ``route="decide"`` member.

    ``Portal.output`` may be a type NAME, which ``lookup_type`` resolves. The
    resolution is LENIENT here on purpose: this runs inside ``normalize_ir``,
    which ``Construct.__init__`` calls BEFORE ``_validate_node_chain``, so raising
    on an unregistered name would move that failure earlier and change which error
    a user sees. An unresolvable name keeps its raw spec, matches no consumer, and
    is reported by the layer that already reports it.
    """
    portal = node.modifier_set.portal if node.modifier_set is not None else None
    if portal is None or portal.output is None:
        return None
    if portal_member_class(node) is not PortalMemberClass.DISPATCH:
        return None

    resolved: Any = portal.output
    if isinstance(resolved, str):
        try:
            resolved = lookup_type(resolved)
        except NeographError:
            pass
    return Producer(
        field_name=StateKeys.dispatch(base),
        effective_type=resolved,
        declared_type=resolved,
        label=f"node '{name}' dispatch result",
    )


def declared_output_fields(item: ConstructItem) -> set[str]:
    """The state-field NAMES ``item`` contributes as a producer.

    The lossy projection of :func:`contributed_fields` -- order and types
    discarded. It used to be an independent derivation asserting it was
    "IDENTICAL to the validator's producer field-name set"; it was not, and the
    citation was deleted rather than reworded (design 7.5: parity by CALLING,
    never by asserting).
    """
    return {p.field_name for p in contributed_fields(item)}








def _subclass_either_way(produced: object, declared: object) -> bool:
    """Bidirectional subclass test -- the default carry-compatibility predicate.

    Lives here so no caller has to reach across a layer for one. The runtime
    (``_input_shape``) must not import the validation cluster, and the Agent Spec
    lowering must not be imported by the runtime, so a shared default is the only
    arrangement in which all three read ONE derivation.

    Validation passes its richer ``_types_compatible`` instead, which understands
    generics and unions; the two agree on the plain-class case that a loop carry is.
    """
    return (
        isinstance(declared, type)
        and isinstance(produced, type)
        and (issubclass(produced, declared) or issubclass(declared, produced))
    )




def item_field_names(construct: Any) -> list[str]:
    """State-field names the construct's OWN DECLARED ITEMS write, in declaration order.

    The eligibility half of the shared boundary rule, per GH #17. A
    sub-construct's final state holds more than what the sub-construct COMPUTED:
    forwarded ``context=`` fields, ``neo_subgraph_input``, framework keys. Those
    are values the child was HANDED, and letting them compete to BE its output is
    the whole of GH #17 -- a branch that declared ``context=['read']`` had its
    ``output=Case`` silently re-pointed at the injected case, five readings and
    zero claims, with a green run.

    ``_scan_subgraph_input`` (neograph-5suot unknown #5) can adopt this same
    eligibility set.

    The claim that followed -- that this "agrees with
    ``_agent_spec_boundary.resolve_end_node_sources`` ... One rule, not a fifth
    answer" -- was FALSE when written and is deleted rather than reworded (design
    7.5). They disagreed on two axes: that function read ``construct.nodes[-1]``
    positionally and never looked at ``output_from`` at all, while this side honoured
    it; and this side type-filters in declaration order, so it can select an item
    that is NOT the last. The consequence was measured: an exported Flow wired one
    member to the EndNode while the run returned another's output.
    ``neograph-9axw6.3`` pointed that function at the declared port, so the two now
    agree on the NAMED case by both calling ``resolve_output_from``. The unnamed case
    is still two derivations -- positional there, type-filtered here -- so this notes
    what is true instead of asserting a parity that is not.
    """
    # The ARM-INCLUSIVE, FLAT projection of the shared write-set, neograph-yz69e.
    # The walk and its arm policy stay HERE -- `contributed_fields` is per-item
    # precisely so this caller, `normalize_ir` (top-level only) and
    # `_stamp_single_type_sources` (arm-SCOPED) can each keep their own.
    #
    # Declaration order is preserved because the reader depends on it: the
    # boundary picks the LAST eligible item. The dict-form per-key rule and the
    # Portal dispatch field now arrive from the shared enumeration rather than
    # being re-derived here -- omitting the latter is what made a sub-construct
    # whose output is satisfied only by a dispatched result raise at run time.
    return [p.field_name for item in iter_with_arms(construct) for p in contributed_fields(item)]




def boundary_member_name(
    construct: Any, compatible: Callable[[object, object], bool] = _subclass_either_way
) -> str | None:
    """The member whose declared output satisfies ``construct.output``, last first.

    The DECLARATION-level twin of the runtime's boundary pick. The runtime scans
    ``item_field_names`` last-declared-first and type-checks VALUES; an exporter has
    no values, so it asks the same question of the declarations and gets the same
    answer for the same reason -- last declared eligible member wins.

    Lives here, beside ``port_source_field`` and ``item_field_names``, because the
    alternative was a fresh reversed type-match loop inside the exporter, which is
    the shape this epic removes and which a guard duly objected to when it was
    written there.

    Only for the UNNAMED case: ``output_from`` is resolved by the normalizer and
    read directly, and a named port never consults this.

    This asks the single-field PRIMARY question, so it does NOT read
    ``contributed_fields`` and is not covered by the write-set monopoly guard. The
    asymmetry that creates is real and deliberately left: the runtime side
    (``item_field_names``) now sees a dispatch member's ``{node}_dispatch``, while
    this returns ``None`` for the same construct. It is unreachable rather than
    latent -- dispatch-mode Portal export fails loud before boundary resolution
    ("no Agent Spec lowering", a permanent scope boundary pinned by
    ``TestDispatchModePortalFailsLoud``) -- so closing it would be writing code for
    a path that raises. Measured, not assumed. If that fail-loud is ever lifted,
    this is the site that has to move with it.
    """
    declared = _declared_output(construct)
    if not isinstance(declared, type):
        return None
    for item in reversed(list(iter_with_arms(construct))):
        primary = normalize_outputs(getattr(item, "outputs", None)).primary
        if compatible(primary, declared):
            return getattr(item, "name", None)
    return None


@dataclass(frozen=True)
class Producer:
    """A producer registered during construct validation.

    effective_type is user-declared and therefore opaque from neograph's
    perspective — see docs/design/architecture-decisions.md §5 for the
    boundary rationale. label is rendered verbatim in error messages.

    is_loop marks a Loop-modified producer (see neograph-ftnxl.6): unlike
    Each, Loop does NOT change the declared/effective type (state.py keeps
    the append-list reducer opaque to the type system — the state field is
    Annotated[list[output_type], _append_loop_result], but ``effective_type``
    here intentionally stays the bare ``output_type`` because a plain-T
    consumer sees the unwrapped latest value, not the list). A list[T]
    consumer wants the FULL history instead (di.py's ``_unwrap_loop_value``
    already passes it through unchanged at runtime) — a producer-shape fact
    ``effective_type`` alone can't express since the SAME producer satisfies
    two different consumer shapes. ``_loop_aware_compatible`` is the read
    side of this flag.

    declared_type carries the type as the author WROTE it, before any modifier
    adjustment. Both are needed, and confusing them breaks live checkpoints:
    validation type-checks against ``effective_type`` (an Each producer writes
    ``dict[str, X]``), while ``_schema_fingerprint`` hashes the DECLARED type --
    fingerprinting ``dict[str, X]`` instead of ``X`` would change every Each
    node's fingerprint and invalidate every existing checkpoint on the release
    that landed it. Each reader names which one it means. See neograph-yz69e.
    """

    field_name: str
    effective_type: TypeSpecStatic
    label: str
    is_loop: bool = False
    declared_type: TypeSpecStatic | None = None


def effective_producer_type(item: ConstructItem) -> TypeSpecStatic:
    """Return the type this producer writes to the state bus, accounting
    for modifiers.

    This is the **single source of truth** for the "producer side" of
    type compatibility. The sole validator walker
    (``_validate_node_chain``) consults it, so a new modifier that
    reshapes state only needs to teach this one function about the new
    rule — the walker picks up the change automatically.

    Current rules:
      - ``Each`` modifier → ``dict[str, raw_output]`` (aggregated fan-out
        results land as a dict keyed by ``each.key``; see
        ``state.py:_add_output_field`` for the state builder side of
        this rule).
      - Everything else → the item's declared output (Node ``.outputs``,
        Construct ``.output``) unchanged.

    Returns ``None`` when the item has no declared output.
    """
    output = _declared_output(item)
    if output is None:
        return None
    return effective_producer_type_for(output, getattr(item, "modifier_set", None))


def effective_producer_type_for(declared_type: TypeSpecStatic, modifier_set: object | None) -> TypeSpecStatic:
    """Apply the modifier-to-bus rule to a SINGLE declared output type.

    This is the per-key core extracted from :func:`effective_producer_type`.
    Both producer-registration paths share it so the Each→dict[str, X] rule
    has exactly one implementation:

      - whole-node / single-type path → :func:`effective_producer_type`
        delegates here with the node's sole declared output;
      - dict-form multi-output path (``_construct_validation`` registers one
        producer per output key) → delegates here per key, so each key's type
        is wrapped independently.

    ``modifier_set`` is duck-typed (``.each``) rather than imported, keeping
    this validation-cluster leaf module free of a ``modifiers`` dependency.

    Current rules:
      - ``Each`` modifier → ``dict[str, declared_type]``
      - Everything else → ``declared_type`` unchanged.
    """
    if modifier_set is not None and getattr(modifier_set, "each", None) is not None:
        return dict[str, declared_type]  # type: ignore[valid-type]
    return declared_type
