"""Assembly-time type validation for Construct chains.

Extracted from construct.py to keep the declaration module lean. Consumed
only by Construct.__init__ (which imports _validate_node_chain + ConstructError).

Validation rule: walk the node list in order, accumulating producers (earlier
nodes and sub-constructs, plus the Construct's own input port if declared).
For each item with a declared input, verify some upstream producer supplies
a compatible value — directly, or through an `Each` modifier whose `over`
path resolves to `list[input_type]`.

Defers to runtime isinstance-scanning when evidence is insufficient:
  - No upstream producers (first-of-chain — input comes from run(input=...))
  - dict / non-class input types (multi-field or raw extraction)
  - Each modifier whose root segment doesn't match a known producer
"""

from __future__ import annotations

import os
import sys
import types
from typing import TYPE_CHECKING, Any, ForwardRef, Union, get_args, get_origin, get_type_hints

from neograph.di import DIKind as _DIKind
from neograph.errors import ConstructError, NeographError
from neograph.modifiers import Each, Modifiable, split_each_path
from neograph.naming import field_name_for
from neograph.node import Node

if TYPE_CHECKING:
    from neograph.construct import Construct

# Type alias for items that appear in Construct.nodes — avoids bare Any.
# Cannot be a runtime Union due to circular imports (Construct imports us).
NodeItem = Node | Modifiable  # Node, _BranchNode (via Modifiable), or Construct (subtype of Modifiable)


def effective_producer_type(item: NodeItem) -> Any:
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
    # Node uses .outputs (plural); Construct uses .output (singular).
    output = item.outputs if isinstance(item, Node) else getattr(item, "output", None)
    if output is None:
        return None
    ms = getattr(item, "modifier_set", None)
    if ms is None:
        return output
    if ms.each is not None:
        return dict[str, output]  # type: ignore[valid-type]
    return output


def validate_loop_self_edge(node: Node) -> None:
    """Check that a self-loop's output type is compatible with its input type.

    Called at ``|`` time (from ``Modifiable.__or__``) when a ``Loop`` modifier
    is applied to a Node. Loop on Node always means self-loop.
    """
    loop = node.modifier_set.loop
    if loop is None:
        return

    output_type = node.outputs
    input_type = node.inputs
    if output_type is None or input_type is None:
        return

    # Dict-form outputs: loop feeds back the PRIMARY key only (first key).
    # Secondary keys (e.g. tool_log) are per-iteration metadata.
    if isinstance(output_type, dict):
        output_type = next(iter(output_type.values()))

    # Dict-form inputs: the back-edge feeds the node's output back as one
    # of its own inputs.  Check if the output is compatible with ANY input
    # value type — the compiler wires the specific slot.
    if isinstance(input_type, dict):
        for _key, expected in input_type.items():
            if _types_compatible(output_type, expected):
                return
        # No compatible input slot found.
        type_list = ", ".join(
            f"{k}={_fmt_type(v)}" for k, v in input_type.items()
        )
        raise ConstructError.build(
            f"loop back-edge: output type {_fmt_type(output_type)} not "
            f"compatible with any reentry input type",
            expected=f"one of ({type_list})",
            found=_fmt_type(output_type),
            hint="the loop's last node output must match the reentry node's input type",
            node=node.name,
            location=_source_location(),
        )
        return

    # Single-type inputs.
    if not _types_compatible(output_type, input_type):
        raise ConstructError.build(
            "loop back-edge: output type not compatible with reentry input type",
            expected=_fmt_type(input_type),
            found=_fmt_type(output_type),
            hint="the loop's last node output must match the reentry node's input type",
            node=node.name,
            location=_source_location(),
        )


def _validate_node_chain(construct: Construct) -> None:
    """Walk the node list, verifying each input has a compatible producer."""
    # Producers: (state_field_name, output_type, human_label)
    producers: list[tuple[str, Any, str]] = []

    # The Construct's own input port is the first producer, if declared —
    # used by inner nodes that read from `neo_subgraph_input`.
    if construct.input is not None:
        producers.append((
            "neo_subgraph_input",
            construct.input,
            f"construct '{construct.name}' input port",
        ))

    for item in construct.nodes:
        # Node has `inputs` (plural) since neograph-kqd.1. Construct still
        # uses `input` (singular) as its sub-construct boundary port —
        # that's handled above via construct.input, not here. The
        # isinstance check makes the two-field contract explicit.
        if isinstance(item, Node):
            input_type = getattr(item, "inputs", None)
        else:
            # Construct items use .input (singular boundary port).
            input_type = getattr(item, "input", None)
        if input_type is not None:
            # Warn on single-type inputs (isinstance scan) — dict-form is safer.
            if (
                isinstance(item, Node)
                and not isinstance(input_type, dict)
                and producers  # not the first node
            ):
                import warnings
                warnings.warn(
                    f"Node '{item.name}': single-type inputs={input_type.__name__ if hasattr(input_type, '__name__') else input_type} "
                    f"relies on O(N) isinstance scan at runtime. "
                    f"Use dict-form inputs={{'{field_name_for(item.name)}': {input_type.__name__ if hasattr(input_type, '__name__') else input_type}}} "
                    f"for explicit named resolution.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            _check_item_input(construct, item, input_type, producers)

        # Validate context= references.
        # Skip for sub-constructs (input is set) — context comes from parent.
        if isinstance(item, Node) and item.context and construct.input is None:
            known_fields = {fn for fn, _, _ in producers}
            for ctx_name in item.context:
                if ctx_name not in known_fields:
                    raise ConstructError.build(
                        f"references context='{ctx_name}' but no upstream node "
                        f"produces a field with that name",
                        found=f"known upstream fields: {sorted(known_fields) or '(none)'}",
                        node=item.name,
                        location=_source_location(),
                    )

        # Node uses .outputs (plural); Construct / _BranchNode use .output (singular).
        output_type = item.outputs if isinstance(item, Node) else getattr(item, "output", None)
        name = getattr(item, "name", None)
        if output_type is not None and name is not None:
            field_name = field_name_for(name)

            # Dict-form outputs (neograph-1bp.4): register one producer per
            # output key, with modifier wrapping applied independently per key.
            if isinstance(item, Node) and isinstance(output_type, dict):
                has_each = item.modifier_set.each is not None
                for output_key, key_type in output_type.items():
                    key_field = f"{field_name}_{output_key}"
                    key_label = f"node '{name}' output '{output_key}'"
                    producer_type = dict[str, key_type] if has_each else key_type  # type: ignore[valid-type]
                    producers.append((key_field, producer_type, key_label))
            else:
                label = (
                    f"node '{name}'"
                    if isinstance(item, Node)
                    else f"sub-construct '{name}'"
                )
                # Shared helper decides the modifier-adjusted state-bus type.
                producers.append((field_name, effective_producer_type(item), label))

        # Loop + skip_when without skip_value is surprising.
        # The counter still increments so the loop exits, but re-entry
        # reads stale state. Warn rather than reject — the behavior is
        # valid, just easy to misuse.
        if isinstance(item, Node) and item.modifier_set.loop is not None:
            if item.skip_when is not None and item.skip_value is None:
                import structlog
                structlog.get_logger(__name__).error(
                    "loop_skip_when_no_skip_value",
                    node=item.name,
                    msg=(
                        f"Node '{item.name}' has Loop + skip_when but no skip_value. "
                        f"When skip_when fires inside a Loop, skip_value is recommended "
                        f"to provide the output for that iteration."
                    ),
                )

        # (Loop reenter validation removed — Loop.reenter no longer exists.
        # Multi-node loops use Loop on Construct instead.)

        # Validate @merge_fn state params.
        # When a merge_fn has from_state params, verify each references a
        # known upstream producer and the type is compatible.
        if isinstance(item, Node):
            oracle = item.modifier_set.oracle
            if oracle is not None and isinstance(getattr(oracle, 'merge_fn', None), str):
                from neograph.decorators import get_merge_fn_metadata
                assert oracle.merge_fn is not None
                meta = get_merge_fn_metadata(oracle.merge_fn)
                if meta is not None and meta[1]:
                    _, merge_param_res = meta
                    item_field = field_name_for(item.name)
                    known_producers = {fn: (pt, lbl) for fn, pt, lbl in producers}
                    for pname, binding in merge_param_res.items():
                        if binding.kind != _DIKind.FROM_STATE:
                            continue
                        expected_type = binding.inner_type
                        # Self-reference: merge runs before node output is written
                        if pname == item_field:
                            raise ConstructError.build(
                                f"merge_fn '{oracle.merge_fn}' param '{pname}' "
                                f"references the node's own output",
                                found=f"self-reference to '{pname}'",
                                hint="the merge barrier runs before the node's "
                                     "output is written — this can never resolve",
                                node=item.name,
                                construct=construct.name,
                                location=_source_location(),
                            )
                        # Unknown producer
                        if pname not in known_producers:
                            raise ConstructError.build(
                                f"merge_fn '{oracle.merge_fn}' param '{pname}' "
                                f"does not match any upstream node",
                                found=f"known producers: {sorted(known_producers.keys())}",
                                node=item.name,
                                construct=construct.name,
                                location=_source_location(),
                            )
                        # Type mismatch
                        prod_type, prod_label = known_producers[pname]
                        if (prod_type is not None
                                and isinstance(expected_type, type)
                                and not _types_compatible(prod_type, expected_type)):
                            raise ConstructError.build(
                                f"merge_fn '{oracle.merge_fn}' param '{pname}' "
                                f"type mismatch with {prod_label}",
                                expected=_fmt_type(expected_type),
                                found=_fmt_type(prod_type),
                                node=item.name,
                                construct=construct.name,
                                location=_source_location(),
                            )

    # Sub-construct output boundary contract: if construct.output is declared,
    # at least one internal node must produce a compatible type.
    # Exclude neo_subgraph_input — the input port is NOT a valid producer
    # for the output contract.
    if construct.output is not None and producers:
        declared_output = construct.output
        internal_producers = [
            (fn, pt, lbl) for fn, pt, lbl in producers
            if fn != "neo_subgraph_input"
        ]
        for _, producer_type, _ in internal_producers:
            if producer_type is not None and _types_compatible(producer_type, declared_output):
                break
        else:
            producer_summary = "\n".join(
                f"    - {label}: {_fmt_type(t)}"
                for _, t, label in internal_producers
                if t is not None
            )
            raise ConstructError.build(
                f"declares output={_fmt_type(declared_output)} but no "
                f"internal node produces a compatible type",
                expected=_fmt_type(declared_output),
                found=f"internal producers:\n{producer_summary}",
                construct=construct.name,
                location=_source_location(),
            )


def validate_loop_construct(construct: Construct) -> None:
    """Validate Loop on a Construct: output must be compatible with input.

    Called at ``|`` time from ``Modifiable.__or__`` when a Loop modifier
    is applied to a Construct.
    """
    if construct.output is None or construct.input is None:
        raise ConstructError.build(
            "Loop requires both input= and output= declared",
            found=f"input={'declared' if construct.input is not None else 'None'}, "
                  f"output={'declared' if construct.output is not None else 'None'}",
            hint="declare both input= and output= on the construct for Loop to wire the back-edge",
            construct=construct.name,
            location=_source_location(),
        )
    if not _types_compatible(construct.output, construct.input):
        raise ConstructError.build(
            "Loop output type not compatible with input type for back-edge",
            expected=_fmt_type(construct.input),
            found=_fmt_type(construct.output),
            hint="the loop's output must match the construct's input type for the back-edge",
            construct=construct.name,
            location=_source_location(),
        )


def _check_item_input(
    construct: Construct,
    item: NodeItem,
    input_type: Any,
    producers: list[tuple[str, Any, str]],
) -> None:
    """Validate that `item.inputs` is satisfied by some upstream producer.

    Static validation only fires when there's enough evidence to make a
    definite call. Cases that defer to runtime isinstance-scanning:
      - No upstream producers at all (first-of-chain).
      - `dict` / non-class input types (multi-field or raw extraction).
      - Each modifier whose root segment doesn't match a known producer.
    """
    if not producers:
        # Even with no producers, reject self-referencing dict-form inputs
        # (a node naming itself as upstream without Loop).
        if isinstance(input_type, dict):
            item_field = field_name_for(item.name)
            ms = getattr(item, "modifier_set", None)
            has_loop = ms is not None and ms.loop is not None
            if not has_loop and item_field in input_type:
                raise ConstructError.build(
                    f"declares inputs['{item_field}'] referencing itself without a Loop modifier",
                    hint="self-referencing inputs require Loop(when=...) to create a feedback cycle",
                    node=item.name,
                    construct=construct.name,
                    location=_source_location(),
                )
        return
    # Fan-in dict instance: inputs={"a": A, "b": B, ...} — validate each
    # (upstream_name, expected_type) pair against the upstream named by the
    # key (neograph-kqd.2). This was a bypass pre-kqd; it is now a positive
    # check.
    if isinstance(input_type, dict):
        _check_fan_in_inputs(construct, item, input_type, producers)
        return
    # Raw dict class: inputs=dict — multi-field isinstance extraction,
    # defers to runtime.
    if input_type is dict:
        return
    # Parameterized generic dict[str, X]: validate against producers if any
    # upstream has a parameterized dict output, otherwise defer to runtime.
    if get_origin(input_type) is dict:
        has_dict_producer = any(
            get_origin(pt) is dict for _, pt, _ in producers
        )
        if not has_dict_producer:
            return
        # Fall through to plain-input validation below — _types_compatible
        # handles parameterized generic comparison.
    if not isinstance(input_type, type) and get_origin(input_type) is None:
        return

    # Each modifier rewires the effective input type via the `over` path.
    ms = getattr(item, "modifier_set", None)
    each = ms.each if ms is not None else None
    if each is not None:
        _check_each_path(construct, item, input_type, each, producers)
        return

    # Plain input: any producer whose output is assignable to input_type wins.
    for _, producer_type, _ in producers:
        if _types_compatible(producer_type, input_type):
            return

    raise _build_no_producer_error(construct, item, input_type, producers)


def _check_fan_in_inputs(
    construct: Construct,
    item: NodeItem,
    inputs_dict: dict[str, Any],
    producers: list[tuple[str, Any, str]],
) -> None:
    """Validate a fan-in ``inputs={'name': Type, ...}`` spec against the
    producer list by upstream name (neograph-kqd.2).

    For each (upstream_name, expected_type) pair:
      1. Look up a producer whose state-field name matches ``upstream_name``.
         No match → ConstructError (unknown upstream).
      2. The producer's effective state-bus type was already computed
         by the caller (via ``effective_producer_type``) — do NOT
         re-compute or inline modifier rules here.
      3. Check compatibility via ``_types_compatible``. Mismatch →
         ConstructError.
    """
    # Determine which keys are fan-out receivers (not upstream producers).
    # Two sources:
    #   1. fan_out_param set by @node decoration (via _build_construct_from_decorated)
    #   2. Each modifier present + key not in producers (programmatic API)
    # Both must be handled so the programmatic `Node(inputs=...) | Each(...)`
    # path works without requiring fan_out_param to be set.
    fan_out_key = getattr(item, "fan_out_param", None)
    ms = getattr(item, "modifier_set", None)
    has_each = ms is not None and ms.each is not None
    producer_by_name: dict[str, tuple[Any, str]] = {
        field_name: (producer_type, label)
        for field_name, producer_type, label in producers
    }
    # Track if the Each fan-out receiver slot was consumed.
    # If fan_out_param is already set (from @node), the slot is pre-consumed.
    _each_skip_used = (fan_out_key is not None)
    for upstream_name, expected_type in inputs_dict.items():
        if upstream_name == fan_out_key:
            continue
        if upstream_name not in producer_by_name:
            # If the node has an Each modifier, ONE unmatched key is the
            # fan-out item receiver — skip it. Additional unmatched keys
            # are real errors (typos) and must be rejected.
            if has_each and not _each_skip_used:
                _each_skip_used = True
                continue
            # Loop self-reference: key matching the node's own name reads
            # from the node's own output on re-entry.
            if ms is not None and ms.loop is not None:
                item_field = field_name_for(item.name)
                if upstream_name == item_field:
                    continue
            raise ConstructError.build(
                f"declares inputs['{upstream_name}']={_fmt_type(expected_type)} "
                f"but no upstream node named '{upstream_name}' exists",
                found=f"available upstreams: {sorted(producer_by_name.keys())}",
                node=item.name,
                construct=construct.name,
                location=_source_location(),
            )
        producer_type, _label = producer_by_name[upstream_name]
        if not _types_compatible(producer_type, expected_type):
            raise ConstructError.build(
                f"declares inputs['{upstream_name}']={_fmt_type(expected_type)} "
                f"but upstream '{upstream_name}' produces "
                f"{_fmt_type(producer_type)}",
                expected=_fmt_type(expected_type),
                found=_fmt_type(producer_type),
                node=item.name,
                construct=construct.name,
                location=_source_location(),
            )


def _check_each_path(
    construct: Construct,
    item: NodeItem,
    input_type: Any,
    each: Each,
    producers: list[tuple[str, Any, str]],
) -> None:
    """Resolve each.over against producers; verify it lands on list[input_type]."""
    root, segments = split_each_path(each.over)

    root_type: Any = None
    for field_name, producer_type, _ in producers:
        if field_name == root:
            root_type = producer_type
            break

    if root_type is None:
        # Allow framework-injected roots (sub-construct port param).
        if root == "neo_subgraph_input":
            return
        known_roots = {fn for fn, _, _ in producers}
        raise ConstructError.build(
            f"Each(over='{each.over}') root '{root}' does not match any upstream node",
            found=f"known producers: {sorted(known_roots)}" if known_roots else "no producers available",
            node=item.name,
            construct=construct.name,
            hint="the first segment of 'over' must name an upstream node (e.g., 'source.items')",
            location=_source_location(),
        )

    # Walk remaining segments through Pydantic model_fields.
    current_type: Any = root_type
    walked = [root]
    for segment in segments:
        walked.append(segment)
        resolved = _resolve_field_annotation(current_type, segment)
        if resolved is _MISSING:
            raise ConstructError.build(
                f"Each(over='{each.over}') path '{'.'.join(walked)}' does not resolve",
                found=f"{_fmt_type(current_type)} has no field '{segment}'",
                node=item.name,
                construct=construct.name,
                location=_source_location(),
            )
        current_type = resolved

    element_type = _extract_list_element(current_type)
    if element_type is None:
        raise ConstructError.build(
            f"Each(over='{each.over}') terminal field is not a list",
            expected="list[...]",
            found=_fmt_type(current_type),
            hint="Each fans out over a collection; the terminal field must be a list",
            node=item.name,
            construct=construct.name,
            location=_source_location(),
        )

    if not _types_compatible(element_type, input_type):
        raise ConstructError.build(
            f"Each(over='{each.over}') element type mismatch",
            expected=_fmt_type(input_type),
            found=f"list[{_fmt_type(element_type)}]",
            node=item.name,
            construct=construct.name,
            location=_source_location(),
        )

    # Verify each.key names a valid field on the element type.
    # Only check when the element type has model_fields (Pydantic model);
    # primitives (str, int, etc.) defer to runtime str(item) fallback.
    element_fields = getattr(element_type, "model_fields", None)
    if element_fields is not None and each.key not in element_fields:
        raise ConstructError.build(
            f"Each(over='{each.over}', key='{each.key}') — "
            f"{_fmt_type(element_type)} has no field '{each.key}'",
            found=f"available fields: {sorted(element_fields.keys())}",
            node=item.name,
            construct=construct.name,
            location=_source_location(),
        )


# Sentinel distinguishing "field absent" from "field present but None-valued" —
# needed because a Pydantic field annotation could legitimately be None, and
# we want a returnable value that never collides with user types.
_MISSING = object()


def _resolve_field_annotation(model_class: Any, field_name: str) -> Any:
    """Return the fully-resolved annotation for a field, or _MISSING if absent.

    Tries `typing.get_type_hints` first to unwrap ForwardRefs and string
    annotations introduced by `from __future__ import annotations`, then
    falls back to `model_fields[name].annotation`. If the final result is
    still unresolved (ForwardRef or bare string), returns `_MISSING` rather
    than leaking it to callers — otherwise `_extract_list_element` silently
    returns None on an unresolved annotation and the validation appears to
    pass when it should have flagged a resolution failure.
    """
    model_fields = getattr(model_class, "model_fields", None) or {}
    if field_name not in model_fields:
        return _MISSING
    try:
        hints = get_type_hints(model_class)
    except (NameError, AttributeError, TypeError):
        hints = {}
    ann = hints.get(field_name, model_fields[field_name].annotation)
    if ann is None or isinstance(ann, (str, ForwardRef)):
        return _MISSING
    return ann


def _types_compatible(producer: Any, target: Any) -> bool:
    """True if a value of type `producer` can satisfy a consumer of `target`.

    Handles parameterized generics (e.g. dict[str, X]) as well as plain classes.
    """
    if producer is target:
        return True
    producer_origin = get_origin(producer)
    target_origin = get_origin(target)
    # Unwrap Union/Optional types.
    # typing.Optional[X] = Union[X, None], PEP 604 X | None = UnionType.
    if producer_origin is Union or producer_origin is types.UnionType:
        prod_args = [a for a in get_args(producer) if a is not type(None)]
        return bool(prod_args) and all(_types_compatible(a, target) for a in prod_args)
    if target_origin is Union or target_origin is types.UnionType:
        target_args = [a for a in get_args(target) if a is not type(None)]
        return bool(target_args) and any(_types_compatible(producer, a) for a in target_args)
    # Parameterized generic producer (e.g. dict[str, X]):
    # compatible with raw origin class (dict) or exact parameterized match.
    if producer_origin is not None:
        # dict[str, X] vs dict → compatible (runtime isinstance handles it)
        if isinstance(target, type) and issubclass(producer_origin, target):
            return True
        # dict[str, X] vs dict[str, Y] → compare origin + args recursively
        if target_origin is not None and producer_origin is target_origin:
            p_args, t_args = get_args(producer), get_args(target)
            if len(p_args) != len(t_args):
                return False
            return all(_types_compatible(p, t) for p, t in zip(p_args, t_args, strict=True))
        # dict[str, X] producer ↔ list[Y] consumer — merge-after-fanout
        # (neograph-kqd.2). A downstream node consuming an Each-fanned-out
        # result as list[Y] gets the runtime unwrap via dict.values() in
        # step 5 (factory._extract_input). Element-type compatibility is
        # checked recursively so subclass rules apply consistently.
        if producer_origin is dict and target_origin is list:
            dict_args = get_args(producer)     # (str, X)
            list_args = get_args(target)       # (Y,)
            if len(dict_args) == 2 and len(list_args) == 1:
                return _types_compatible(dict_args[1], list_args[0])
        return False
    if not (isinstance(producer, type) and isinstance(target, type)):
        return False
    try:
        return issubclass(producer, target)
    except TypeError:
        return False


def _extract_list_element(tp: Any) -> Any:
    """If tp is list[X], Optional[list[X]], or list[X] | None, return X."""
    origin = get_origin(tp)
    if origin is list:
        args = get_args(tp)
        return args[0] if args else None
    # Handle Union / Optional / X | None. `requires-python >= 3.11` guarantees
    # types.UnionType exists, so no hasattr guard needed.
    if origin is Union or origin is types.UnionType:
        for arg in get_args(tp):
            if arg is type(None):
                continue
            element = _extract_list_element(arg)
            if element is not None:
                return element
    return None


def _fmt_type(tp: Any) -> str:
    if tp is None:
        return "None"
    if hasattr(tp, "__name__"):
        return tp.__name__
    return repr(tp)


def _build_no_producer_error(
    construct: Construct,
    item: NodeItem,
    input_type: Any,
    producers: list[tuple[str, Any, str]],
) -> NeographError:
    if producers:
        producer_summary = "\n".join(
            f"    - {label}: {_fmt_type(t)}"
            for _, t, label in producers
        )
    else:
        producer_summary = "    (no upstream producers)"

    return ConstructError.build(
        f"declares "
        f"{'inputs' if isinstance(item, Node) else 'input'}="
        f"{_fmt_type(input_type)} but no upstream produces a "
        f"compatible value",
        found=f"upstream producers:\n{producer_summary}",
        hint=_suggest_hint(input_type, producers),
        node=item.name,
        construct=construct.name,
        location=_source_location(),
    )


def _suggest_hint(
    input_type: Any,
    producers: list[tuple[str, Any, str]],
) -> str | None:
    """Scan producer outputs for actionable suggestions."""
    # Check for Each dict[str, X] → raw X mismatch first.
    for _field_name, producer_type, _ in producers:
        p_origin = get_origin(producer_type)
        if p_origin is dict:
            p_args = get_args(producer_type)
            if p_args and len(p_args) == 2:
                element_type = p_args[1]
                if isinstance(input_type, type) and isinstance(element_type, type):
                    try:
                        match = issubclass(element_type, input_type) or issubclass(input_type, element_type)
                    except TypeError:
                        match = False
                    if match:
                        return (
                            f"upstream produces dict[str, {_fmt_type(element_type)}] "
                            f"via Each — consume the whole dict with input=dict "
                            f"or input=dict[str, {_fmt_type(element_type)}]"
                        )

    # Fallback: scan for list[input_type] fields and suggest .map().
    for field_name, producer_type, _ in producers:
        model_fields = getattr(producer_type, "model_fields", None) or {}
        for fname in model_fields:
            resolved = _resolve_field_annotation(producer_type, fname)
            if resolved is _MISSING:
                continue
            element = _extract_list_element(resolved)
            if element is not None and _types_compatible(element, input_type):
                return (
                    f"did you forget to fan out? try "
                    f".map(lambda s: s.{field_name}.{fname}, key='...')"
                )
    return None


def _location_suffix() -> str:
    loc = _source_location()
    return f"  at {loc}" if loc else ""


def _source_location() -> str | None:
    """Return 'file.py:line' for the user-code frame that assembled this Construct.

    Walks frames via `sys._getframe` past neograph and pydantic internals to
    find the first user frame — typically the `Construct(...)` call site.
    `sys._getframe` is ~50× cheaper than `inspect.stack()` because it doesn't
    materialize source context for every frame.

    Filters by module name (`frame.f_globals['__name__']`) rather than path
    substring, because user tests/examples often live under a `neograph/`
    directory that would otherwise get mis-filtered.
    """
    try:
        frame: types.FrameType | None = sys._getframe(1)
        while frame is not None:
            module_name = frame.f_globals.get("__name__", "")
            if not (
                module_name == "neograph"
                or module_name.startswith("neograph.")
                or module_name.startswith("pydantic")
            ):
                fname = frame.f_code.co_filename
                if fname and not fname.startswith("<"):
                    return f"{os.path.basename(fname)}:{frame.f_lineno}"
            frame = frame.f_back
    except (AttributeError, TypeError, ValueError):  # noqa: bare-except — frame walk best-effort
        return None
    return None
