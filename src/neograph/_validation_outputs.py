"""Assembly-time validation for output-field markers (neograph-ftnxl.4).

Joins the validation cluster (``VALIDATION_CLUSTER`` in
``tests/test_guards_assembly.py``) as the sibling of ``_validation_arms.py`` --
same shape, different concern: this owns the SCOPE FENCE and the depth-0-only
restriction for ``Carried``/``ExcludeFromOutput`` markers, wired into
``_validate_node_chain`` via one delegating call, the exact
``_check_portal_mesh`` precedent.

Four rejections, all fail-loud ``ConstructError`` at assembly time (never a
silent pass-through to a runtime surprise):

1. A ``Carried`` path whose root is not a name the node itself declares (a
   ``node.inputs`` key or a ``_param_res`` DI param name) -- the SCOPE FENCE.
   Where the root is not statically resolvable (single-form ``inputs=``, an
   opaque ``FromConfig`` type), this defers to the runtime check honestly --
   it does not overclaim compile-time coverage.
2. A ``Carried``/``ExcludeFromOutput`` marker on a NESTED (depth > 0) model
   field -- ``project_output_model`` only strips depth 0; a nested marker
   would strip from the rendered text while staying demanded by the
   validating schema with no splice to fill it, one level down.
3. A ``Carried`` marker on an agent/act node's output model -- that seam
   strips text (``_agent_output_schema_preamble.py``) with no splice
   mechanism at all.
4. A ``Carried`` root whose producer is Each/Loop-modified -- the producer's
   actual runtime value shape (``dict[str, X]`` / an append-list) is not what
   a flat splice expects; read via ``effective_producer_type`` (this
   project's single source of truth for modifier-aware type effects), never a
   second hand-rolled modifier check.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from neograph._ir_branch import iter_with_arms
from neograph._ir_source import PortRef

# The port resolver, INJECTED rather than imported. ``_ir_normalize`` owns minting a
# PortRef (Source construction is banned elsewhere), but it imports
# ``_construct_validation``, which imports this module -- so importing it back here is a
# cycle. ``construct.py`` holds both and passes the function down, which is AGENTS.md's
# sanctioned "inject as a parameter" rung and the ``resolve_condition`` precedent.
PortResolver = Callable[["ConstructLike"], "PortRef | None"]
from neograph._normalize import _declared_output, normalize_outputs
from neograph._output_classify import output_markers
from neograph._validation_types import (
    Producer,
    _fmt_type,
    _loop_aware_compatible,
    _source_location,
    effective_producer_type,
    effective_producer_type_for,
)
from neograph.errors import ConstructError
from neograph.naming import field_name_for
from neograph.node import Node

if TYPE_CHECKING:
    from neograph._ir_protocols import ConstructItem, ConstructLike


def _output_model_of(item: Any) -> type[Any] | None:
    output = _declared_output(item)
    if isinstance(output, dict):
        return None  # dict-form outputs: each key's type is checked independently below
    if isinstance(output, type) and hasattr(output, "model_fields"):
        return output
    return None


def _iter_output_models(item: Any) -> list[type[Any]]:
    output = _declared_output(item)
    if isinstance(output, dict):
        return [t for t in output.values() if isinstance(t, type) and hasattr(t, "model_fields")]
    model = _output_model_of(item)
    return [model] if model is not None else []


def _check_carried_paths(construct: ConstructLike) -> None:
    """The ONE walker for output-marker assembly checks. See module docstring."""
    name_to_item: dict[str, Any] = {}
    for item in iter_with_arms(construct):
        name = getattr(item, "name", None)
        if name is not None:
            name_to_item[field_name_for(name)] = item

    for item in iter_with_arms(construct):
        for model in _iter_output_models(item):
            _check_model_markers(construct, item, model, name_to_item)


def _find_nested_marker(model: type[Any], _seen: set[type[Any]] | None = None) -> str | None:
    """Depth>0 scan: True if ANY field reachable through a nested BaseModel
    (never the top level itself) carries an output marker. Cycle-safe via
    ``_seen`` (self-referential models are legal Pydantic)."""
    seen = _seen if _seen is not None else set()
    if model in seen:
        return None
    seen.add(model)
    for field_name, field_info in model.model_fields.items():
        strip, _carried = output_markers(field_info)
        if strip:
            return f"{model.__name__}.{field_name}"
        annotation = field_info.annotation
        if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
            found = _find_nested_marker(annotation, seen)
            if found is not None:
                return found
    return None


def _check_model_markers(
    construct: ConstructLike,
    item: Any,
    model: type[Any],
    name_to_item: dict[str, Any],
) -> None:
    is_agent_act = isinstance(item, Node) and item.mode in ("agent", "act")
    declared_inputs = getattr(item, "inputs", None)
    declared_input_names = set(declared_inputs) if isinstance(declared_inputs, dict) else set()
    param_res = getattr(item, "_param_res", None) or {}
    di_names = set(param_res)

    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
            nested = _find_nested_marker(annotation)
            if nested is not None:
                raise ConstructError.build(
                    f"node {item.name!r}: output field {field_name!r} ({model.__name__}.{field_name}) "
                    f"is a nested model carrying an output marker at {nested!r}, which is not supported",
                    found=f"marker on {nested}, reachable through {model.__name__}.{field_name}",
                    hint="output markers (ExcludeFromOutput / Carried) are top-level-only -- move the "
                    "marked field to the top-level output model",
                    node=item.name,
                    construct=getattr(construct, "name", None),
                    location=None,
                )

        strip, carried = output_markers(field_info, field_label=f"{model.__name__}.{field_name}")
        if not strip:
            continue

        if carried is None:
            # ExcludeFromOutput (never spliced): a default is its ONLY value
            # source. Carried (always spliced) deliberately has NO such
            # requirement -- a default there would only manufacture the
            # missed-splice-masking surface this ticket eliminates.
            if field_info.is_required():
                raise ConstructError.build(
                    f"node {item.name!r}: output field {field_name!r} carries ExcludeFromOutput "
                    "but has no default value",
                    hint="an ExcludeFromOutput field is never produced by the LLM -- give it a "
                    "Pydantic default, or use Carried if the framework should supply the value",
                    node=item.name,
                    construct=getattr(construct, "name", None),
                    location=None,
                )
            continue

        if is_agent_act:
            raise ConstructError.build(
                f"node {item.name!r}: Carried is not supported on agent/act nodes yet",
                found=f"Carried({carried.path!r}) on an agent/act-mode node's output model",
                hint="the agent/act preamble strips text with no splice mechanism -- use "
                "ExcludeFromOutput, or move this field off the output model, until Carried "
                "gains agent/act support",
                node=item.name,
                construct=getattr(construct, "name", None),
                location=None,
            )

        root = carried.segments[0]
        if root not in declared_input_names and root not in di_names:
            raise ConstructError.build(
                f"node {item.name!r}: Carried({carried.path!r}) root {root!r} is not a name this node declares",
                expected="a node.inputs key or a FromInput/FromConfig DI param name",
                found=f"declared inputs: {sorted(declared_input_names)}; DI params: {sorted(di_names)}",
                hint="Carried may only root at a name the node itself declares -- referencing "
                "another node's output by an undeclared path would add a validator-invisible "
                "dataflow edge",
                node=item.name,
                construct=getattr(construct, "name", None),
                location=None,
            )

        producer = name_to_item.get(root)
        if producer is not None and isinstance(producer, Node):
            ms = producer.modifier_set
            if ms.each is not None or ms.loop is not None:
                raise ConstructError.build(
                    f"node {item.name!r}: Carried({carried.path!r}) roots at {root!r}, whose "
                    "producer is Each/Loop-modified",
                    found=f"producer {root!r} effective type: {effective_producer_type(producer)!r}",
                    hint="an Each producer's value is dict[str, X]; a Loop producer's is an "
                    "append-list -- neither is what a flat Carried splice expects",
                    node=item.name,
                    construct=getattr(construct, "name", None),
                    location=None,
                )


def check_output_from(construct: ConstructLike, resolve_port: PortResolver) -> None:
    """Validate ``Construct.output_from`` names exactly one declared item.

    GH #17. ``output_from`` says WHICH item's output is the boundary; ``output=``
    still says WHAT type it is. A name that matches no item -- or matches more than
    one -- refuses HERE, at assembly, like any other wiring mistake.

    Refusing a name that resolves to a forwarded ``context=`` field matters more
    than it looks: that field is exactly the value the reverse type-scan used to
    return silently, so accepting it would re-spell the original bug through the
    very field added to prevent it.
    """
    ref = resolve_port(construct)
    if ref is None:
        return
    port = construct.output_from
    # Direct attribute read, not _declared_output: the selector exists to abstract
    # the Node(.outputs)-vs-Construct(.output) split, and this is known to be a
    # Construct. The orchestrator reads construct.output the same way.
    if construct.output is None:
        raise ConstructError.build(
            f"declares output_from={port!r} but no output= type",
            expected="output= set alongside output_from=",
            found="output=None",
            hint="output_from says WHICH producer is the boundary; output= says WHAT type it is. Both are needed.",
            construct=construct.name,
            location=_source_location(),
        )
    items = [item for item in iter_with_arms(construct) if getattr(item, "name", None)]
    names = [item.name for item in items]
    matches = [item for item in items if item.name == ref.member]
    if len(matches) == 1:
        _check_named_port_satisfies_boundary(construct, ref, matches[0])
        return
    problem = "matches no item" if not matches else f"is ambiguous -- {len(matches)} items share it"
    raise ConstructError.build(
        f"declares output_from={port!r}, which {problem}",
        expected=f"the name of exactly one item of construct {construct.name!r}",
        found=f"declared items: {names}" if names else "this construct declares no items",
        hint=(
            "output_from must name an item THIS construct declares. A forwarded context= field or a "
            "framework key is not an item -- it is a value the construct was handed, and letting such a "
            "value be the boundary is the bug output_from exists to prevent."
        ),
        construct=construct.name,
        location=_source_location(),
    )


def _modifier_set_is_loop(item: ConstructItem) -> bool:
    """Loop-modified? Derived exactly as ``_construct_validation`` derives it at
    producer registration (``is_loop=item.modifier_set.loop is not None``), so the
    two cannot answer differently."""
    ms = getattr(item, "modifier_set", None)
    return ms is not None and getattr(ms, "loop", None) is not None


def _check_named_port_satisfies_boundary(construct: ConstructLike, ref: PortRef, item: ConstructItem) -> None:
    """The two refusals that make a NAMED port trustworthy (design 6.1, rows 3-4).

    ``check_output_from`` proves the name resolves to one declared item and used to
    STOP there. Everything downstream then ignored the name: the boundary-satisfaction
    check at ``_construct_validation`` asks whether ANY internal producer matches
    ``output=`` and discards WHICH, so a named member of the wrong type was accepted
    whenever some peer happened to match. The author's explicit answer lost to a guess,
    and the run died later at ``_subconstruct``'s ``eligible=[the named field]`` with
    "no internal node produces a compatible output value" -- a message describing the
    wrong cause, since the node you NAMED produces something, just not this.

    1. A member with SEVERAL outputs, named without a port key, is not an address
       neograph-kgndo. ``settle`` writes ``settle_result`` and ``settle_tool_log``;
       the name says which MEMBER, never which VALUE.
    2. A named port whose type cannot satisfy ``output=`` refuses HERE
       neograph-x8i3s, naming the port, its type and the expected type.

    The type compared is the EFFECTIVE producer type, not the declared one, and the
    reason is the runtime rather than convenience: ``effective_producer_type_for`` is
    what the state FIELD HOLDS, and the boundary scan isinstance-checks that value. An
    ``Each``-modified member declaring ``Case`` holds ``dict[str, Case]``, so naming it
    for ``output=Case`` is refused -- correctly, because the run would fail the
    isinstance check. Reaching for the raw declared type here to make such a case pass
    would restore accept-at-assembly-then-die-at-runtime, which is the defect.

    ``_loop_aware_compatible`` rather than bare ``_types_compatible``: a Loop-modified
    producer's effective type stays the bare element type while the field holds an
    append-list, so ``output=list[Case]`` against a Loop-named port is legitimate and
    the bare predicate would refuse it. AGENTS.md records that this widening belongs at
    the CALL SITE, never inlined into ``_types_compatible``; this is such a call site.

    NARROWING, intended and recorded: when the named member lives inside a branch arm
    and its type mismatches, this refuses even though ``output_reachable_on_every_arm``
    could satisfy the boundary through OTHER nodes. Naming a port and having the arm
    scan quietly rescue a different producer is the same defect one layer down. A
    correctly-typed arm-named port is unaffected. Whether a port may address INTO an arm
    at all is a separate, open question: neograph-7siep.
    """
    declared = _declared_output(item)
    no = normalize_outputs(getattr(item, "outputs", None))
    if ref.output is None and isinstance(item, Node) and no.is_dict_form and len(no.all_keys) > 1:
        raise ConstructError.build(
            f"declares output_from={construct.output_from!r}, which names a member with several outputs",
            expected=f"a port address naming one output, e.g. {ref.member}.{sorted(no.all_keys)[0]!r}",
            found=f"member {ref.member!r} declares outputs: {sorted(no.all_keys)}",
            hint=(
                "A member name says WHICH MEMBER, never WHICH VALUE: a dict-form outputs= "
                "writes one state field per key. Name the port you mean."
            ),
            construct=construct.name,
            location=_source_location(),
        )
    if ref.output is not None:
        if not (isinstance(item, Node) and no.is_dict_form and ref.output in no.all_keys):
            available = sorted(no.all_keys) if (isinstance(item, Node) and no.is_dict_form) else []
            raise ConstructError.build(
                f"declares output_from={construct.output_from!r}, whose port {ref.output!r} is not an output of {ref.member!r}",
                expected=f"one of {available}" if available else f"member {ref.member!r} to declare dict-form outputs",
                found=f"member {ref.member!r} declares: {_fmt_type(declared)}",
                hint="A dotted address names an output KEY of a dict-form outputs= declaration.",
                construct=construct.name,
                location=_source_location(),
            )
        declared = no.all_keys[ref.output]
    effective = effective_producer_type_for(declared, getattr(item, "modifier_set", None))
    if effective is None or construct.output is None:
        return
    producer = Producer(field_name="", label=ref.member, effective_type=effective, is_loop=_modifier_set_is_loop(item))
    if _loop_aware_compatible(producer, construct.output):
        return
    raise ConstructError.build(
        f"declares output_from={construct.output_from!r}, whose type cannot satisfy output=",
        expected=_fmt_type(construct.output),
        found=f"port {construct.output_from!r} produces {_fmt_type(effective)}",
        hint=(
            "You NAMED this port, so it is not a candidate among several -- it is the answer. "
            "A named port whose type does not match is a mistake to fix, not a reason to fall "
            "back to scanning the other producers."
        ),
        construct=construct.name,
        location=_source_location(),
    )
