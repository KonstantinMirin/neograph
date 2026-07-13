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

from neograph._ir_branch import iter_item_slots
from neograph._ir_protocols import ConstructItem
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._sidecar import infer_oracle_gen_type
from neograph._state_keys import StateKeys
from neograph.naming import field_name_for, output_field_name
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


def declared_output_fields(item: ConstructItem) -> set[str]:
    """The state-field names a node/sub-construct contributes as a producer.

    Mirrors the validator's producer registration (``_construct_validation``):
    - dict-form ``Node.outputs`` → one ``{base}_{key}`` field per output key
      (NO bare base — matching the validator, which registers per-key only)
    - single-type ``Node.outputs`` → the bare ``base`` field
    - ``Node.outputs is None`` → no producer (empty set)
    - sub-construct (non-Node) → the bare ``base`` field

    Used by :func:`normalize_ir` to build the peer-field set so it is IDENTICAL
    to the validator's producer field-name set. See neograph-bcct. Field-name
    only (type-independent), so Each-wrapping of producer types does not affect
    it.
    """
    name = getattr(item, "name", None)
    if name is None:
        return set()
    base = field_name_for(name)
    if isinstance(item, Node):
        no = normalize_outputs(item.outputs)
        if no.is_none:
            return set()
        if no.is_dict_form:
            return {output_field_name(base, key) for key in no.all_keys}
        return {base}
    return {base}


def fan_out_candidates(node: Node, known_field_names: set[str]) -> list[str]:
    """The dict-form input keys of ``node`` that could be an Each fan-out
    receiver: those whose field name is neither a known producer/peer field
    nor the node's own field.

    Single definition of "fan-out candidate", shared by the two consumers that
    each supply their own ``known_field_names`` (they run at different pipeline
    stages with different information):

    - :class:`_FanOutParamNormalizer` (writer) — runs in ``Construct.__init__``
      before producers exist, so it passes the *peer node* field set.
    - ``_construct_validation._check_fan_in_inputs`` (tolerator) — runs after,
      so it passes the full *producer* field set (incl. per-output-key names).

    Returns ``[]`` for non-dict-form inputs. Order follows the inputs dict
    (insertion order). The policy on the result — write when exactly one
    (normalizer), tolerate one + error on extras (validator) — stays with each
    caller; only the candidate computation is shared.
    """
    ni = normalize_inputs(node.inputs)
    if not ni.is_dict_form:
        return []
    self_field = field_name_for(node.name)
    return [
        key for key in ni.by_name if field_name_for(key) not in known_field_names and field_name_for(key) != self_field
    ]


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
            return {"fan_out_param": candidates[0]}
        return {}


class _HandoffParamNormalizer:
    """Set ``node.handoff_param`` for a Keymaker + dict-form node that declares
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
        if node.modifier_set.keymaker is None:
            return False
        ni = normalize_inputs(node.inputs)
        return ni.is_dict_form and "handoff" in ni.by_name

    def apply(self, node: Node, peer_field_names: set[str]) -> dict[str, Any]:
        return {"handoff_param": "handoff"}


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
    # Peer-field set IDENTICAL to the validator's producer field-name set:
    # multi-output nodes contribute per-output-key fields ({base}_{key}), not
    # the bare base. Built from the shared declared_output_fields helper so the
    # two views cannot drift. See neograph-bcct.
    #
    # Peer set stays TOP-LEVEL only (construct.nodes), not arm-inclusive: a
    # branch-arm Each node that reads an arm-SIBLING producer would misinfer its
    # fan_out_param receiver against this set. No consumer needs arm-sibling
    # fan-in today, so the limitation is documented rather than closed here.
    # See neograph-vn5f (site 2).
    peer_field_names: set[str] = set()
    keymaker_members: list[Node] = []
    for item in construct.nodes:
        peer_field_names |= declared_output_fields(item)
        # Keymaker mesh members at THIS level (top-level siblings, D-MESH-LEVEL).
        # Collected in the existing allowlisted `.nodes` walk so no new raw walk
        # is introduced (arm-blind-walk guard).
        if isinstance(item, Node) and item.modifier_set.keymaker is not None:
            keymaker_members.append(item)

    # The mesh channel is keyed off the ENTRY (first member in node order — one
    # mesh per level, design §3.1). Compute it ONCE here (the only place with the
    # construct-level view) and stamp it onto each member below, so _extract_input
    # reads the channel self-contained (decision D10, the fan_out_param precedent).
    handoff_channel: str | None = None
    if keymaker_members:
        handoff_channel = StateKeys.handoff_payload(field_name_for(keymaker_members[0].name))
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
        # Stamp the entry-keyed mesh channel onto every Keymaker member (decision
        # D10). This module is the SOLE writer of handoff_channel — the same
        # single-writer ownership as handoff_param (review H2 / neograph-k7bg) —
        # because the entry-keyed key is a construct-level fact no assembly path
        # can compute per-node. Idempotent: skip if already set.
        if handoff_channel is not None and item.modifier_set.keymaker is not None and item.handoff_channel is None:
            updates["handoff_channel"] = handoff_channel
        if updates:
            container[idx] = item.model_copy(update=updates)
