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
from typing import cast

from neograph._construct_validation import _loop_aware_compatible, _types_compatible
from neograph._ir_consume import single_type_candidates
from neograph._ir_fields import Producer
from neograph._ir_protocols import ConstructItem
from neograph._ir_source import (
    Candidate,
    EachItem,
    HandoffChannel,
    LastPresent,
    LoopCarry,
    Peer,
    Port,
    PortRef,
    Resolution,
    Resolved,
    Rung,
    Source,
    Unresolved,
)
from neograph._normalize import _declared_output, normalize_inputs
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph._state_keys import StateKeys
from neograph.naming import field_name_for
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


def _shadowing_arm_matches(input_type: TypeSpecStatic, shadowed: Sequence[Producer]) -> list[Producer]:
    """The hidden BRANCH-ARM producers that could satisfy ``input_type``.

    Non-empty only for a read placed after a join, and it is what makes that read a
    REFUSAL rather than a resolution: which arm ran is a runtime fact. Shared by the
    two resolvers so a MODIFIED child cannot quietly resolve to one of its own
    channels while a shadowed feeder goes unreported -- which is what happened when
    the port resolver treated "feeder unresolved" as "the feeder is simply not one of
    this port's arrivals".
    """
    return [p for p in shadowed if p.effective_type is not None and _loop_aware_compatible(p, input_type)]


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
    arm_matches = _shadowing_arm_matches(input_type, shadowed)
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
    if portal_member_class(node) not in (None, PortalMemberClass.DISPATCH):
        # A MESH MEMBER's value arrives by hop, on the entry-keyed mesh channel, so a
        # single-type read declares something no producer feeds. The resolver used to
        # search anyway and stamp whichever SIBLING member produced the same type --
        # a field written only if that member happened to run. Refused, with the
        # spelling that IS fed: the reserved dict-form `handoff` key, which is stamped
        # HandoffChannel and read from the channel itself.
        #
        # portal_member_class, not `portal is not None`: a route="decide" Portal is a
        # standalone linear node rather than a member, so it gets no exemption here
        # and falls under the ordinary rules.
        return Unresolved(
            (
                Candidate(
                    ref=PortRef(field_name_for(node.name)),
                    reason=(
                        "is a Portal mesh member, whose value arrives on the mesh channel rather than "
                        "from a producer: declare inputs={'handoff': <payload>} to read it"
                    ),
                ),
            )
        )
    if node.modifier_set.loop is not None:
        # A Loop read has TWO arrivals and they are ordered: the SEED on iteration 0,
        # the node's own CARRY on 1+. That is `carry-before-seed`, LastPresent's
        # documented precedence rule, so the read is stamped with both rungs instead
        # of the runtime deciding between them by probing whether the carry list is
        # non-empty. The seed rung is resolved exactly as any other read would be,
        # which is what keeps `input_source_field` (and so the Agent Spec export)
        # answering with the seed.
        seed = _resolve_by_type(input_type, visible, shadowed)
        if isinstance(seed, Unresolved):
            return seed
        return Resolved(LastPresent((cast("Rung", seed.source), LoopCarry())))
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
    item: ConstructItem,
    visible: Sequence[Producer],
    shadowed: Sequence[Producer] = (),
) -> Resolution | None:
    """Resolve a placed sub-construct's ``input=`` port; ``None`` when it has none.

    The PARENT is the only place this is answerable: a sub-construct normalises
    during its own ``__init__``, before it is placed, so it cannot see the
    producers that will feed it. Same derivation as a Node's read, which is what
    gives the port the enclosing-port fallback and the arm scoping it never had.

    A MODIFIED child has more arrivals than its feeder, and they are ordered -- the
    runtime used to try them as a five-rung ladder of presence probes, then omit the
    port key entirely when every rung missed. They are now the rungs of one
    ``LastPresent``, in the ladder's own precedence order REVERSED (it took the first
    present, ``LastPresent`` takes the last), so the order is preserved rather than
    reinvented:

    * the resolved feeder (``Peer``/``Port``) -- the lowest precedence, tried last
    * the mesh channel, for a Portal member: a hop's payload outranks the feeder
    * the fanned item, for an ``Each`` child: THIS branch's value outranks both
    * the child's own carry, for a ``Loop``: an iteration's own output outranks all

    The carry rung is included only when the child's OUTPUT can satisfy its own
    ``input=``. That was a runtime ``isinstance(latest, sub.input)`` probe -- the
    produce-and-validate shape, where feeding the output back would be wrong -- and it
    is a question about two DECLARATIONS, so it is answered here.
    """
    sub_input = getattr(item, "input", None)
    if sub_input is None:
        return None
    if _shadowing_arm_matches(sub_input, shadowed):
        # A shadowed feeder is a REFUSAL even for a modified child, and it has to be
        # checked before the rungs: a Loop-on-Construct would otherwise resolve to its
        # own carry and the shadowed producer would go unreported -- swallowing step
        # 1's guarantee for exactly the `out = self.a(c) if cond else self.b(c)` shape
        # that motivated it.
        return _resolve_by_type(sub_input, visible, shadowed)
    feeder = _resolve_by_type(sub_input, visible, shadowed)

    # A MODIFIED child's value need not come from a producer at all: a fanned child
    # reads the item, a mesh member reads the hop. So an unresolved FEEDER is only a
    # refusal when no other rung can supply the port -- otherwise the feeder is simply
    # not one of this port's arrivals. (A plain child has no other rung, so the
    # refusal still lands there, which is xejyn's second half.)
    modifiers = getattr(item, "modifier_set", None)
    rungs: list[Rung] = [] if isinstance(feeder, Unresolved) else [cast("Rung", feeder.source)]
    portal = getattr(modifiers, "portal", None) if modifiers is not None else None
    if portal is not None and portal_member_class(item) not in (None, PortalMemberClass.DISPATCH):
        channel = getattr(item, "handoff_channel", None)
        if channel is not None:
            rungs.append(HandoffChannel(channel))
    if modifiers is not None and modifiers.each is not None:
        rungs.append(EachItem())
    if modifiers is not None and modifiers.loop is not None:
        declared_output = _declared_output(item)
        if declared_output is not None and _types_compatible(declared_output, sub_input):
            rungs.append(LoopCarry())
    if not rungs:
        return feeder  # Unresolved: nothing can feed this port.
    if len(rungs) == 1:
        return Resolved(rungs[0])
    return Resolved(LastPresent(tuple(rungs)))
