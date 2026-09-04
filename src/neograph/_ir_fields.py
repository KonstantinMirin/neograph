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

from collections.abc import Callable, Sequence
from typing import Any

from neograph._ir_branch import iter_with_arms
from neograph._ir_protocols import ConstructItem
from neograph._normalize import _declared_output, normalize_inputs, normalize_outputs
from neograph._type_spec import TypeSpecStatic
from neograph.naming import field_name_for, output_field_name
from neograph.node import Node

__all__ = ["declared_output_fields", "fan_out_candidates", "boundary_member_name", "item_field_names", "with_source", "loop_carry_dest_key", "port_source_field", "single_type_candidates"]


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
    candidates: Sequence[tuple[str, TypeSpecStatic, object]],
    sub_input: type | None,
    compatible: Callable[[TypeSpecStatic, TypeSpecStatic], bool],
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
    preceding: Sequence[tuple[str, TypeSpecStatic, object]],
    input_type: TypeSpecStatic,
    compatible: Callable[[TypeSpecStatic, TypeSpecStatic], bool],
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


def loop_carry_dest_key(
    node: Node,
    compatible: Callable[[TypeSpecStatic, TypeSpecStatic], bool] = _subclass_either_way,
) -> str | None:
    """Which dict-form input key receives a Loop's own fed-back output.

    ONE derivation of the loop carry's DESTINATION. Three sites answered this
    differently and each believed one of the others owned it:

    * the validator proved SOME slot was type-compatible and discarded which,
      under a comment saying "the compiler wires the specific slot";
    * the compiler does not -- the runtime picks at execution time by probing
      which siblings are present and falling back to ``next(iter(by_name))``,
      a POSITIONAL guess;
    * the Agent Spec lowering took the first ``issubclass`` match with a ``break``,
      under a comment claiming it mirrored the upstream-resolution scan.

    Three answers to one question, so validation could pass on one slot while the
    run bound another and the export drew a third.

    The rule, which is the one the exporter and the runtime already shared before
    diverging: the node's OWN field name if it appears among the input keys --
    a self-reference is named, not guessed -- otherwise the first key whose
    declared type can hold the fed-back output. ``None`` when the inputs are not
    dict-form, where there is no key to choose and the single value IS the carry.
    """
    ni = normalize_inputs(node.inputs)
    if not ni.is_dict_form:
        return None
    self_field = field_name_for(node.name)
    if self_field in ni.by_name:
        return self_field
    no = normalize_outputs(node.outputs)
    if no.is_none:
        return None
    # ``primary`` for dict-form outputs too: a Loop feeds back the PRIMARY output,
    # which is the value the carry list holds (primary_output_field states the same
    # rule for the field name). Treating dict-form outputs as having no destination
    # was over-strict -- it refused three working dict-form-output loops.
    for key, declared in ni.by_name.items():
        if compatible(no.primary, declared):
            return key
    return None


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
    fields: list[str] = []
    for item in iter_with_arms(construct):
        name = getattr(item, "name", None)
        if not name:
            continue
        base = field_name_for(name)
        declared = _declared_output(item)
        if isinstance(declared, dict):
            # Dict-form outputs write ONE state field per key ({node}_{key}), and the
            # bare {node} field does not exist. Missing these made every dict-form
            # sub-construct boundary unresolvable -- caught by
            # TestGatherProduceSubConstruct, not by the boundary tests.
            fields.extend(output_field_name(base, key) for key in declared)
        else:
            fields.append(base)
    return fields


def with_source(node: Any, key: str, source: Any) -> dict[str, Any]:
    """``node``'s address table with ``key`` bound to ``source``.

    Copy-not-mutate, so a normalizer pass that runs twice is idempotent and a Node
    shared between two constructs cannot have its table edited underneath it -- the
    same discipline the four collapsed fields each carried, now written once.
    """
    return {**(getattr(node, "input_sources", None) or {}), key: source}


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
    """
    declared = _declared_output(construct)
    if not isinstance(declared, type):
        return None
    for item in reversed(list(iter_with_arms(construct))):
        primary = normalize_outputs(getattr(item, "outputs", None)).primary
        if compatible(primary, declared):
            return getattr(item, "name", None)
    return None
