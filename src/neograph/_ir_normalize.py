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

from neograph._normalize import normalize_inputs
from neograph._sidecar import infer_oracle_gen_type
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


class _FanOutParamNormalizer:
    """Set ``node.fan_out_param`` for an Each + dict-form node whose fan-out
    receiver hasn't been resolved yet.

    The receiver is the single dict-form input key that names neither a peer
    producer nor the node itself. When exactly one such key exists it is the
    fan-out receiver; zero or many is deliberately left for the validator
    (an ambiguous fan-out is a user error, not an inference).
    """

    def applies_to(self, node: Node) -> bool:
        if node.fan_out_param is not None:
            return False
        if node.modifier_set.each is None:
            return False
        return normalize_inputs(node.inputs).is_dict_form

    def apply(self, node: Node, peer_field_names: set[str]) -> dict[str, Any]:
        ni = normalize_inputs(node.inputs)
        self_field = field_name_for(node.name)
        unknown = [
            key for key in ni.by_name
            if field_name_for(key) not in peer_field_names
            and field_name_for(key) != self_field
        ]
        if len(unknown) == 1:
            return {"fan_out_param": unknown[0]}
        return {}


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
]


def normalize_ir(construct: Construct) -> None:
    """Apply every registered IR normalizer to every Node in ``construct``.

    The single site that writes IR-level inferred fields. Walks
    ``construct.nodes`` once, collects each applicable normalizer's updates,
    and writes them via a single ``model_copy`` per node. Idempotent:
    normalizers whose field is already set return ``{}`` and are no-ops.
    """
    peer_field_names = {
        field_name_for(item.name)
        for item in construct.nodes
        if getattr(item, "name", None) is not None
    }
    for i, item in enumerate(construct.nodes):
        if not isinstance(item, Node):
            continue
        updates: dict[str, Any] = {}
        for normalizer in _NORMALIZERS:
            if normalizer.applies_to(item):
                updates.update(normalizer.apply(item, peer_field_names))
        if updates:
            construct.nodes[i] = item.model_copy(update=updates)
