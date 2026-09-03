"""State-field-name derivation: what an item CONTRIBUTES, and what it could CONSUME.

Two field-name rules that several layers need and none should re-derive:
``declared_output_fields`` (which state fields an item writes as a producer) and
``fan_out_candidates`` (which dict-form input keys could be an Each fan-out
receiver). Both are field-name only and type-independent.

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

from neograph._ir_protocols import ConstructItem
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node

__all__ = ["declared_output_fields", "fan_out_candidates", "port_source_field", "single_type_candidates"]


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


def port_source_field(
    candidates: list[tuple[str, object, object]],
    sub_input: type | None,
    compatible: Callable[[object, type], bool],
) -> str | None:
    """Which PARENT field feeds a sub-construct's input port.

    The assembly-time answer to a question the runtime used to ask by scanning.
    ``_scan_subgraph_input`` reverse-iterated the ENTIRE parent state bag and
    returned the first value that passed ``isinstance`` against the declared
    ``input=`` -- so framework bookkeeping, forwarded ``context=`` fields and every
    unrelated producer all competed to be the port's value, and which one won
    depended on dict ordering at run time.

    Same PRECEDENCE, computed once from declarations instead of values: the LAST
    declared producer whose effective type can satisfy the port. Reverse iteration
    was the scan's own rule -- later pipeline nodes take precedence over earlier
    ones, e.g. a loop's output over its seed -- so preserving it is what keeps this
    a relocation of the answer rather than a change to it.

    ``candidates`` are ``(field_name, effective_type, item)`` triples in declaration
    order, and ``compatible`` is the caller's type predicate: this module is a leaf
    and must not reach into the validation cluster for one.

    Returns ``None`` when nothing can satisfy the port, which is not an error here
    -- the runtime's remaining ladder rungs (loop carry, fanned item, mesh channel)
    may still supply a value, and a genuinely unsatisfiable port is the validator's
    to refuse.
    """
    if sub_input is None:
        return None
    for field, effective, _item in reversed(candidates):
        if effective is not None and compatible(effective, sub_input):
            return field
    return None


def single_type_candidates(
    preceding: list[tuple[str, object, object]],
    input_type: type,
    compatible: Callable[[object, type], bool],
) -> list[str]:
    """Every declared producer field whose type can satisfy a single-type ``inputs=``.

    ONE derivation with two readers: the normalizer takes the last of these as the
    resolved source, and validation refuses when there is more than one. Before this
    they would have been two walks over the same producer list, which is how the
    runtime and the exporter came to disagree in the first place.

    Order is declaration order, so ``[-1]`` is the node's immediate upstream -- what
    an author reading a pipeline top to bottom means by "the Claims".
    """
    return [
        field for field, prod_type, _producer in preceding if prod_type is not None and compatible(prod_type, input_type)
    ]
