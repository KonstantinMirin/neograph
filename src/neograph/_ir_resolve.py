"""WHERE A VALUE COMES FROM: the resolvers for every type-addressed read.

Two reads in the IR are addressed by TYPE rather than by name -- a Node's
single-type ``inputs=X`` and a placed sub-construct's ``input=`` port -- plus one
addressed by an AUTHORED NAME, ``Node.input_from``. This module answers all three,
and it is the only module that answers them.

It was one function inside ``_ir_normalize`` with a sibling of its own in
``_ir_consume``, and the two disagreed about the type predicate, about branch-arm
scoping and about whether the enclosing port is a candidate. Each disagreement was
a silent ``None`` at run time. The split is not a line-count move: ``_ir_normalize``
infers MODIFIER-IMPLIED fields (which key receives a fanned item, which mesh channel
a member reads), which is a different question from "which producer satisfies this
declared type", and keeping both in one file is what let a second answer to the
second question grow unnoticed.

A resolver returns a ``Resolution`` -- ``Resolved(Source)`` or ``Unresolved(Candidate,
...)`` -- and never raises. Refusal is the VALIDATOR's to render, because
normalization runs first and an error raised here would move every existing message
and refusal order. ``_ir_stamp`` walks the construct and attaches what these return.
"""

from __future__ import annotations

from collections.abc import Sequence

from neograph._construct_validation import _loop_aware_compatible
from neograph._ir_consume import single_type_candidates
from neograph._ir_fields import Producer
from neograph._ir_source import (
    Candidate,
    EachItem,
    Peer,
    Port,
    PortRef,
    Resolution,
    Resolved,
    Source,
    Unresolved,
)
from neograph._normalize import normalize_inputs
from neograph._state_keys import StateKeys
from neograph.node import Node, TypeSpecStatic

__all__ = [
    "resolve_port_source",
    "resolve_single_type_source",
    "single_type_demand",
]


def single_type_demand(node: Node) -> TypeSpecStatic | None:
    """The type a node's single-type ``inputs=`` demands, or ``None`` for no demand.

    Split out from the resolver so "is there a read here" and "what feeds it" are
    separate questions: the stamp walk, the validator's render and the instrument
    all need the first one, and only the resolver answers the second.
    """
    ni = normalize_inputs(node.inputs)
    if ni.is_dict_form or ni.is_none:
        return None
    return ni.single_type


def _type_name(spec: TypeSpecStatic) -> str:
    """A declared type as an author would recognise it, for a Candidate reason."""
    return getattr(spec, "__name__", None) or repr(spec)


def _candidate(producer: Producer, reason: str) -> Candidate:
    """One near-miss, with the reason COMPUTED HERE so diagnostics render it."""
    return Candidate(ref=PortRef(producer.field_name), reason=reason)


def _resolve_by_type(
    input_type: TypeSpecStatic,
    visible: Sequence[Producer],
    shadowed: Sequence[Producer],
) -> Resolution:
    """The ONE answer to "which declared producer satisfies this type-addressed read".

    Serves both type-addressed reads neograph has -- a Node's single-type
    ``inputs=X`` and a placed sub-construct's ``input=`` port. They were two
    derivations (``resolve_single_type_source`` and ``port_source_field``) that
    disagreed about the predicate, about arm scoping and about whether the
    enclosing port is a candidate; each disagreement was a separate silent
    ``None`` (neograph-la3a4, neograph-chunx, neograph-yi9t5).

    ``visible`` is the arm-scoped producer sequence in declaration order, WITH the
    enclosing construct's own port seeded first when it has one -- so "a
    type-compatible peer outranks the port" falls out of last-wins rather than
    being a separate rule that could drift from it.

    ``shadowed`` is the producers a BRANCH ARM registered that ``visible``
    deliberately hides -- non-empty only for a read placed after a join. When one
    of them satisfies the type, the read is REFUSED rather than resolved to
    whatever sits above the branch: which arm ran is a runtime fact, so resolving
    by type there hands the reader a stale value on every path. The legitimate
    every-arm-produces-it form -- stamping the arms as one ordered read -- is
    filed and deliberately not smuggled in here.
    """
    arm_matches = [p for p in shadowed if p.effective_type is not None and _loop_aware_compatible(p, input_type)]
    if arm_matches:
        return Unresolved(
            tuple(
                _candidate(
                    p, "produced on a branch arm; which arm runs is a runtime fact, so it cannot be resolved by type"
                )
                for p in arm_matches
            )
        )
    matches = single_type_candidates(visible, input_type, _loop_aware_compatible)
    if matches:
        # LAST compatible producer wins: the node's IMMEDIATE upstream, which is
        # what an author reading a pipeline top to bottom means by "the Claims"
        # -- and, not incidentally, the answer the Agent Spec export was already
        # giving. The runtime's forward scan was the side that was wrong.
        #
        # Several eligible is NOT refused: measured at 47 failures, nearly all
        # ordinary same-typed chains -- neograph-5fvsu.
        winner = matches[-1]
        source: Source = Port() if winner.field_name == StateKeys.SUBGRAPH_INPUT else Peer(PortRef(winner.field_name))
        return Resolved(source)
    return Unresolved(
        tuple(
            _candidate(p, f"produces {_type_name(p.effective_type)}, which does not satisfy the declared type")
            for p in visible
        )
    )


def _resolve_named_port(
    spelling: str,
    input_type: TypeSpecStatic,
    visible: Sequence[Producer],
    shadowed: Sequence[Producer],
) -> Resolution:
    """Resolve an AUTHORED ``input_from`` name: it must exist, and it must fit.

    The name was taken on TRUST until now, on the strength of a type check in the
    validation cluster that did not exist -- so a misspelling stamped a field
    nobody writes and silently DISPLACED a compatible producer, handing the body
    ``None`` on a green run. ``output_from`` had its checker from the day it
    shipped (``check_output_from``); this is the input-side twin, and with it every
    authored reference in the IR is verified at assembly.

    A BRANCH ARM producer is nameable, and that is deliberate. A post-join read is
    refused when an arm shadows it, because which arm ran is a runtime fact -- and
    the refusal tells the author to name the producer instead. If this resolved
    only against the visible set, that advice would be impossible to follow: the
    arm producer would be unnameable, and a refusal that advises the impossible is
    the defect class this function closes. Naming an arm is the author overriding
    an inference the resolver declines to make. When the named arm does not run its
    field is absent, which the runtime read is to report loudly rather than read as
    a value.
    """
    ref = PortRef.parse(spelling)
    named = next((p for p in (*visible, *shadowed) if p.field_name == ref.field), None)
    if named is None:
        available = sorted(p.field_name for p in (*visible, *shadowed))
        return Unresolved(
            (
                Candidate(
                    ref=ref,
                    reason=f"names no producer visible here; available: {available or '(none)'}",
                ),
            )
        )
    if not _loop_aware_compatible(named, input_type):
        return Unresolved(
            (
                Candidate(
                    ref=ref,
                    reason=f"produces {_type_name(named.effective_type)}, which does not satisfy the declared type",
                ),
            )
        )
    return Resolved(Peer(ref))


def resolve_single_type_source(
    node: Node,
    visible: Sequence[Producer],
    shadowed: Sequence[Producer] = (),
) -> Resolution | None:
    """Resolve a Node's single-type ``inputs=X``; ``None`` when it declares none.

    ``input_from`` is the author NAMING the port, so it replaces the search rather
    than biasing it -- but it is CHECKED, not trusted (see ``_resolve_named_port``).
    """
    input_type = single_type_demand(node)
    if input_type is None:
        return None
    if node.modifier_set.each is not None:
        # The fanned item, BEFORE any search: an Each-modified node's value arrives
        # on the fan-out channel, so there is no producer to look for. The search ran
        # anyway, and when a compatible peer happened to precede the node it was
        # stamped -- harmless at run time, where the item was read by presence, and
        # visible in the EXPORT, which read the stamp and drew an edge from a producer
        # no run reads. Stamping the channel makes the peer answer unrepresentable
        # rather than merely unused.
        if node.input_from is not None:
            # A contradiction, refused rather than silently dropped: the author named
            # a producer for a value that is not produced by one. Ignoring it is what
            # the runtime did, which is how a declaration comes to mean nothing.
            return Unresolved(
                (
                    Candidate(
                        ref=PortRef.parse(node.input_from),
                        reason=(
                            "cannot feed an Each-modified node: its input is the fanned item, "
                            "which arrives on the fan-out channel rather than from a producer"
                        ),
                    ),
                )
            )
        return Resolved(EachItem())
    if node.input_from is not None:
        return _resolve_named_port(node.input_from, input_type, visible, shadowed)
    return _resolve_by_type(input_type, visible, shadowed)


def resolve_port_source(
    sub_input: TypeSpecStatic | None,
    visible: Sequence[Producer],
    shadowed: Sequence[Producer] = (),
) -> Resolution | None:
    """Resolve a placed sub-construct's ``input=`` port; ``None`` when it has none.

    The PARENT is the only place this is answerable: a sub-construct normalises
    during its own ``__init__``, before it is placed, so it cannot see the
    producers that will feed it. Same derivation as a Node's read, which is what
    gives the port the enclosing-port fallback and the arm scoping it never had.
    """
    if sub_input is None:
        return None
    return _resolve_by_type(sub_input, visible, shadowed)
