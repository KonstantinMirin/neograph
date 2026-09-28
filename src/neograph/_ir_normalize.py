"""Single-site IR-level field normalization (neograph-20xq, EPIC 1).

Every API surface — declarative ``@node`` decoration, programmatic
``Node() | Modifier()``, and the YAML loader — produces a ``Construct``.
Some Node IR fields are *inferred* rather than written by the user:

- ``fan_out_param`` — which dict-form input key receives the fanned-out item
  (Each modifier).
- ``oracle_gen_type`` — the per-generator output type, read from the Oracle
  ``merge_fn`` signature.

Before this module these inferences lived in two places: the ``@node`` assembly
path (``_construct_builder._cleanup_inputs_and_register``) and a pair of
``Construct._normalize_*`` methods. That parallel inference was the recurring
three-surface-parity drift class (neograph-8k3, neograph-ayq, vgc1, aqau).

``normalize_ir(construct)`` is the single site. ``Construct.__init__`` calls it
exactly once, before validation, regardless of which surface built the
Construct. Each inference is a registered :class:`IrNormalizer` — the GRASP
Strategy pattern, the same shape the codebase uses for ``StateBus`` and the
test fakes. Adding a new IR-level inference is a new ~15-line normalizer
appended to ``_NORMALIZERS`` — it touches nothing else.

Idempotency contract: every normalizer's ``applies_to`` gates on the field
being unset. The ``@node`` builder may pre-populate ``fan_out_param`` from
richer signature information before ``__init__`` runs; the matching normalizer
sees the field already set and no-ops, so the higher-fidelity value is
preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import BaseModel

from neograph._ir_branch import _BranchNode, iter_item_slots
from neograph._ir_consume import fan_out_candidates, with_source
from neograph._ir_fields import contributed_fields, declared_output_fields
from neograph._ir_protocols import ConstructItem, ConstructLike
from neograph._ir_resolve import resolve_port_source, resolve_single_type_source, single_type_demand
from neograph._ir_source import (
    Accumulated,
    EachItem,
    HandoffChannel,
    Port,
    PortRef,
)
from neograph._ir_stamp import stamp_declared_reads
from neograph._normalize import normalize_inputs
from neograph._portal_member import PortalMemberClass, portal_member_class
from neograph._sidecar import infer_oracle_gen_type
from neograph._state_keys import StateKeys
from neograph.modifiers import _group_portal_members
from neograph.naming import field_name_for
from neograph.node import Node

if TYPE_CHECKING:
    from neograph.construct import Construct


class IrNormalizer(Protocol):
    """Infers a single IR-level field on a Node post-construction.

    Stateless. Idempotent. Applied to every Node in a Construct exactly once
    per ``Construct.__init__``, regardless of which API surface built it.

    Splitting the predicate (``applies_to``) from the inference (``apply``)
    is the GRASP Pure Fabrication split: both are independently testable and
    the call site reads as "for each normalizer that applies, collect its
    updates".
    """

    def applies_to(self, node: Node) -> bool:
        """Whether this normalizer has work to do for ``node``."""
        ...

    def apply(self, node: Node, peer_field_names: set[str]) -> dict[str, Any]:
        """Return the ``model_copy`` update dict for ``node`` (possibly empty)."""
        ...


def resolve_output_from(construct: ConstructLike) -> PortRef | None:
    """Resolve ``Construct.output_from`` to a PORT address, or ``None`` when unset.

    The reader VALIDATION uses, and the seam step 2 stamps. Not yet the only one,
    and saying otherwise here would be the false-parity citation this epic exists to
    remove -- written by it.

    Four sites hand-rolled ``getattr(construct, "output_from", None)`` plus their own
    member->field hop -- the form AGENTS.md bans for ``output``. Validation, the
    runtime and lint now call this; ``_agent_spec_markers`` still reads the attribute
    directly, for the reason stated there.

    Spelling is parsed by ``PortRef.parse``. Resolution only; every refusal is raised
    by ``_validation_outputs.check_output_from``, which calls this. The split is
    forced: ``PortRef`` construction is banned outside this module, while a
    ``model_copy``-set ``output_from`` is caught only by the parent's
    ``_validate_node_chain`` recursion.
    """
    # construct.output_from read directly, not via getattr: this is known to be a
    # Construct, and TestDeclaredOutputSelectorMonopoly bans the hand-rolled getattr
    # selector form in this module. check_output_from reads construct.output the same
    # way, for the same reason.
    port = construct.output_from
    if port is None:
        return None
    return PortRef.parse(port)


class _FanOutParamNormalizer:
    """Set ``node.fan_out_param`` for an Each + dict-form node whose fan-out
    receiver hasn't been resolved yet.

    The receiver is the single dict-form input key that names neither a peer
    producer nor the node itself (see :func:`fan_out_candidates`). When exactly
    one such key exists it is the fan-out receiver; zero or many is
    deliberately left for the validator (an ambiguous fan-out is a user error,
    not an inference).
    """

    def applies_to(self, node: Node) -> bool:
        if node.fan_out_param is not None:
            return False
        if node.modifier_set.each is None:
            return False
        return normalize_inputs(node.inputs).is_dict_form

    def apply(self, node: Node, peer_field_names: set[str]) -> dict[str, Any]:
        candidates = fan_out_candidates(node, peer_field_names)
        if len(candidates) == 1:
            return {"input_sources": with_source(node, candidates[0], EachItem())}
        return {}


class _HandoffParamNormalizer:
    """Set ``node.handoff_param`` for a Portal + dict-form node that declares
    the reserved ``"handoff"`` inputs key (design §3.3).

    Unlike the Each fan-out receiver (inferred by candidate-elimination), the
    handoff receiver is a NAMED reserved key — so ``apply`` returns the literal
    ``"handoff"`` with no inference. This is the SOLE writer of
    ``node.handoff_param`` (review H2 / neograph-k7bg): all three API surfaces
    carry the ``"handoff"`` inputs key explicitly and converge here, so no
    assembly path (decorator, builder, loader) writes the field — writing it in
    an assembly path would re-create the neograph-ts7 three-surface parity bug.
    """

    def applies_to(self, node: Node) -> bool:
        if node.handoff_param is not None:
            return False
        if node.modifier_set.portal is None:
            return False
        ni = normalize_inputs(node.inputs)
        return ni.is_dict_form and "handoff" in ni.by_name

    def apply(self, node: Node, peer_field_names: set[str]) -> dict[str, Any]:
        # The KEY is known here; normalize_ir fills the channel once the mesh entry
        # is known. Both live in ONE address, so they cannot name different things.
        return {"input_sources": with_source(node, "handoff", HandoffChannel(""))}


def oracle_gen_type_for(node: Node) -> type[BaseModel] | None:
    """The per-generator output type inferred from a node's Oracle ``merge_fn``.

    The merge_fn's first parameter is ``list[T]`` where ``T`` is the type each
    generator should produce. Returns ``T`` when it differs from the node's
    declared ``outputs`` (the merged type); otherwise ``None`` (no override
    needed, no Oracle merge_fn, or inference failed).

    This is the single home for the oracle_gen_type inference rule. Both the
    assembly-time :class:`_OracleGenTypeNormalizer` and the @node decoration
    path (``decorators.py``, which sets the field eagerly on the bare Node so
    it is visible before the Node is placed in a Construct) call it, so the
    rule is expressed exactly once.
    """
    oracle = node.modifier_set.oracle
    if oracle is None or oracle.merge_fn is None:
        return None
    gen_type = infer_oracle_gen_type(oracle.merge_fn)
    if gen_type is not None and gen_type is not node.outputs:
        # infer_oracle_gen_type returns the introspected ``list[T]`` element,
        # which is intended to be a generator output model. Not statically
        # provable, hence the cast at this boundary.
        return cast("type[BaseModel]", gen_type)
    return None


class _OracleGenTypeNormalizer:
    """Set ``node.oracle_gen_type`` from the Oracle ``merge_fn`` signature.

    Fires for surfaces that did not already resolve it (e.g. the YAML loader,
    or a @node whose ``merge_fn`` was registered only after decoration ran).
    """

    def applies_to(self, node: Node) -> bool:
        if node.oracle_gen_type is not None:
            return False
        oracle = node.modifier_set.oracle
        # This oracle/merge_fn guard is intentionally duplicated with the one
        # inside oracle_gen_type_for: the GRASP predicate/inference split keeps
        # applies_to cheap (no inference) while apply does the real work. Do
        # NOT collapse it by calling oracle_gen_type_for here — that would run
        # the full inference twice per node.
        return oracle is not None and oracle.merge_fn is not None

    def apply(self, node: Node, peer_field_names: set[str]) -> dict[str, Any]:
        gen_type = oracle_gen_type_for(node)
        if gen_type is not None:
            return {"oracle_gen_type": gen_type}
        return {}


# Registered implementations — one per IR-level field. To add an inference,
# append a normalizer here; nothing else changes. The list (not a dict)
# preserves the typed Protocol contract: a bare callable cannot be inserted
# without losing applies_to/apply.
_NORMALIZERS: list[IrNormalizer] = [
    _FanOutParamNormalizer(),
    _OracleGenTypeNormalizer(),
    _HandoffParamNormalizer(),
]


def normalize_ir(construct: Construct) -> None:
    """Apply every registered IR normalizer to every Node in ``construct``.

    The single site that writes IR-level inferred fields. Walks
    ``construct.nodes`` once, collects each applicable normalizer's updates,
    and writes them via a single ``model_copy`` per node. Idempotent:
    normalizers whose field is already set return ``{}`` and are no-ops.
    """
    # Built from the shared declared_output_fields helper, so multi-output nodes
    # contribute {base}_{key} fields rather than the bare base neograph-bcct.
    #
    # Peer set stays TOP-LEVEL only (construct.nodes), not arm-inclusive: a
    # branch-arm Each node that reads an arm-SIBLING producer would misinfer its
    # fan_out_param receiver against this set. No consumer needs arm-sibling
    # fan-in today, so the limitation is documented rather than closed here.
    # See neograph-vn5f (site 2).
    peer_field_names: set[str] = set()
    # Accumulator channels at this level: the pre-registered `Accumulated`
    # Source variant goes live here and nowhere else (construction ban). Read
    # off the shared write-set enumeration, never re-derived from outputs.
    channel_names: set[str] = set()
    portal_members: list[ConstructItem] = []
    for item in construct.nodes:
        peer_field_names |= declared_output_fields(item)
        channel_names |= {p.field_name for p in contributed_fields(item) if p.is_accumulator}
        # Portal mesh members at THIS level (top-level siblings, D-MESH-LEVEL).
        # Collected in the existing allowlisted `.nodes` walk so no new raw walk
        # is introduced (arm-blind-walk guard). A member — including the mesh
        # ENTRY — may be a sub-Construct (neograph-s7zt3.5): excluding it here
        # would feed _group_portal_members the wrong entry, so the entry-keyed
        # channel stamped onto the Node PEERS below points at an empty channel
        # (handoff resolves to None at runtime). `not isinstance(_BranchNode)`
        # is the runtime-safe equivalent of `isinstance((Node, Construct))` over
        # construct.nodes (whose items are exactly Node | Construct | _BranchNode):
        # Construct cannot be imported here without growing the never-grow
        # function-local-import allowlist, and only a _BranchNode is neither. The
        # write-back loop below stays Node-only (a Construct has no
        # handoff_channel field); only DETECTION of which entry the peers belong
        # to needs the Construct-inclusive walk.
        # portal_member_class, not `portal is not None`: a route="decide" Portal is a
        # STANDALONE linear node, never a mesh member. Collecting it here made it
        # members[0] when it preceded a mesh, keying the channel off a non-member --
        # while _wiring and _validation_portal both skip DISPATCH, so the runtime
        # WROTE one key and every member READ another (neograph-dgbqv.12).
        if not isinstance(item, _BranchNode) and portal_member_class(item) not in (
            None,
            PortalMemberClass.DISPATCH,
        ):
            portal_members.append(item)

    # The mesh channel is keyed off each NAMED GROUP's own ENTRY (first member
    # of that group in node order — neograph-fefar extends design §3.1 from
    # one mesh per level to one mesh per (level, name) pair). Computed ONCE
    # here (the only place with the construct-level view) via the SAME shared
    # grouping helper the validator and compiler mesh collector use
    # (_group_portal_members) — never a re-derived inline grouping — and
    # stamped onto each member below, so _extract_input reads the channel
    # self-contained (decision D10, the fan_out_param precedent).
    handoff_channels: dict[str | None, str] = {
        group_name: StateKeys.handoff_payload(field_name_for(members[0].name))
        for group_name, members in _group_portal_members(portal_members).items()
    }
    # iter_item_slots descends into _BranchNode arms and yields each arm node's
    # OWN storage slot (meta.true_arm_nodes[j] / false_arm_nodes[j]), so the
    # model_copy write-back lands where the compiler reads it — not in a
    # detached copy. See neograph-vn5f (site 2).
    for container, idx in iter_item_slots(construct):
        item = container[idx]
        if not isinstance(item, Node):
            continue
        updates: dict[str, Any] = {}
        for normalizer in _NORMALIZERS:
            if normalizer.applies_to(item):
                updates.update(normalizer.apply(item, peer_field_names))
        # Stamp the entry-keyed mesh channel onto every Portal member (decision
        # D10). This module is the SOLE writer of handoff_channel — the same
        # single-writer ownership as handoff_param (review H2 / neograph-k7bg) —
        # because the entry-keyed key is a construct-level fact no assembly path
        # can compute per-node. Idempotent: skip if already set. Keyed by the
        # member's OWN mesh group so each named mesh gets its
        # own channel, never one shared across disjoint named meshes.
        # Same member test as the collection above, for the same reason: a
        # route="decide" Portal is not a mesh member, so it must not be STAMPED
        # with a mesh channel either -- otherwise _input_shape would feed a
        # standalone dispatch node a handoff payload addressed to the mesh.
        member_portal = item.modifier_set.portal
        if (
            # `is not None` is redundant with the classifier check (a member always
            # has a Portal) -- it is here to narrow Portal | None for mypy, which
            # cannot see through the classifier call. The classifier remains the
            # membership authority; this is not a second member test.
            member_portal is not None
            and portal_member_class(item) not in (None, PortalMemberClass.DISPATCH)
            and item.handoff_channel is None
        ):
            group_channel = handoff_channels.get(member_portal.name)
            if group_channel is not None:
                key = item.handoff_param or "handoff"
                updates["input_sources"] = with_source(item, key, HandoffChannel(group_channel))
        # A dict-form input key that names a channel reads the UNION off that
        # unprefixed field. Stamped as its own Source variant so the address
        # table says what it is; the runtime read is the same state[key] as a
        # peer field -- the difference is what the value MEANS, not where it is.
        ni = normalize_inputs(item.inputs)
        if ni.is_dict_form:
            for key in ni.by_name:
                if key in channel_names and key not in (item.input_sources or {}):
                    updates["input_sources"] = with_source(
                        item if "input_sources" not in updates else item.model_copy(update=updates),
                        key,
                        Accumulated(key),
                    )
        if updates:
            container[idx] = item.model_copy(update=updates)

    # Runs LAST: the pass above may replace node objects, and the declared-read
    # walk resolves against the final IR. ONE walk now answers both type-addressed
    # reads -- a Node's single-type ``inputs=`` and a sub-construct's ``input=``
    # port -- against the VALIDATOR's arm-scoped producer set, so the two cannot
    # disagree the way two walks with two candidate sets did (neograph-z2ayo,
    # neograph-chunx). The resolvers are handed over rather than imported, which
    # keeps Source construction in this module alone.
    resolve_and_stamp_reads(construct)


def stamp_isolated_port(node: Node) -> Node:
    """A copy of ``node`` whose single-type read is addressed to the CALLER.

    ``Node.run_isolated`` builds no construct, so nothing can resolve which producer
    feeds the read -- because none does: the caller hands the value in. That is a
    port, and saying so is what lets the runtime read through the same one address
    every other node does.

    Before this the value was seeded under a second framework key and picked up by a
    fallback list consulted after the stamp. The fallback is what made a WRONG stamp
    survivable anywhere in the graph, so retiring its last user is the point; this
    helper is not a convenience.

    Lives here because the address table has ONE writer, and a public entry point in
    ``node.py`` cannot import this module at the top level (the pass imports the
    validation cluster, which reaches back to ``node``). Returns the node unchanged
    when it declares no single-type read.
    """
    if single_type_demand(node) is None:
        return node
    return node.model_copy(update={"input_sources": with_source(node, StateKeys.SINGLE_INPUT, Port())})


def resolve_and_stamp_reads(construct: Construct) -> None:
    """Resolve and stamp every declared read of ``construct``, with THIS module's
    resolvers.

    The one place the walk is bound to its resolvers. ``_fan_agent_wrap`` calls it
    too: ``model_copy`` skips ``__init__``, so the sub-constructs it synthesizes
    have never been normalised in a parent context, and their port would otherwise
    hold no source at all.
    """
    stamp_declared_reads(construct, resolve_read=resolve_single_type_source, resolve_port=resolve_port_source)
