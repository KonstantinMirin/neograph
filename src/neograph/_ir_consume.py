"""What a CONSUMER binds to: which upstream field or channel satisfies an input.

The consume-side twin of ``_ir_fields``, which answers the producer-side question
(what an item CONTRIBUTES). The two clusters shared a file until neograph-yz69e
grew the contribute side -- they have never shared a helper in either direction,
so the seam was already there and only the line count made it visible.

Each rule here answers "which upstream thing feeds this binding":

- ``fan_out_candidates``  -- which dict-form input key could be an Each fan-out receiver
- ``single_type_candidates`` -- which declared producers satisfy a single-type ``inputs=X``
  (also the PARENT-side answer for a sub-construct's ``input=`` port: one question,
  one derivation -- ``port_source_field`` was the second one, and it disagreed about
  the loop-aware predicate, arm scoping and the port fallback all three)
- ``loop_carry_dest_key`` -- which input key a Loop's fed-back output lands on
- ``with_source``         -- the copy-not-mutate write into a node's address table

``_subclass_either_way`` stays in ``_ir_fields`` beside ``boundary_member_name``,
its other caller, and is imported here -- one direction only, consume -> contribute,
which is the direction the dependency naturally runs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from neograph._ir_fields import Producer
from neograph._normalize import normalize_inputs, normalize_outputs
from neograph._type_spec import TypeSpecStatic
from neograph.naming import field_name_for
from neograph.node import Node

__all__ = [
    "fan_out_candidates",
    "loop_carry_dest_key",
    "single_type_candidates",
    "with_source",
]


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


def single_type_candidates(
    preceding: Sequence[Producer],
    input_type: TypeSpecStatic,
    compatible: Callable[[Producer, TypeSpecStatic], bool],
) -> list[Producer]:
    """Every declared producer that can satisfy a single-type ``inputs=`` or an
    ``input=`` port, in declaration order.

    ONE derivation with one reader: ``_ir_normalize``'s resolver, which takes the
    LAST of these. Validation no longer walks the producers itself -- it RENDERS
    the resolver's ``Resolution`` (neograph-4cvx8 step 1) -- so "the last compatible
    producer" cannot be computed twice and come out differently, which is how the
    normalizer stamped a branch arm while the validator had accepted the node above
    it.

    Takes ``Producer`` records rather than ``(field, type)`` pairs: the pair form
    dropped ``is_loop``, so a ``list[T]`` read after a ``Loop`` producer of ``T``
    type-checked green in validation -- which IS loop-aware -- and resolved to
    nothing here. ``compatible`` is the caller's PRODUCER-level
    predicate for the same reason; this module is a leaf and must not reach into
    the validation cluster for one.

    Order is declaration order, so ``[-1]`` is the node's immediate upstream -- what
    an author reading a pipeline top to bottom means by "the Claims".
    """
    return [
        producer for producer in preceding if producer.effective_type is not None and compatible(producer, input_type)
    ]


def loop_carry_dest_key(
    node: Node,
    compatible: Callable[[TypeSpecStatic, TypeSpecStatic], bool],
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

    ``compatible`` has NO DEFAULT, and that is the fix for the second half of the
    same defect. One derivation was still two answers, because the validator passed
    the strict predicate and the runtime and the export took the lenient default --
    so a ``Base`` output was approved into slot ``b`` and delivered into slot ``a``,
    declared ``Derived``, on a green run. There is now ONE caller:
    the normalizer, which STAMPS the destination it picks, so the runtime and the
    export read an answer instead of recomputing one.
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


def with_source(node: Any, key: str, source: Any) -> dict[str, Any]:
    """``node``'s address table with ``key`` bound to ``source``.

    Copy-not-mutate, so a normalizer pass that runs twice is idempotent and a Node
    shared between two constructs cannot have its table edited underneath it -- the
    same discipline the four collapsed fields each carried, now written once.
    """
    return {**(getattr(node, "input_sources", None) or {}), key: source}
