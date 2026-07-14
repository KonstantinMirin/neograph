"""Consumer-input satisfaction checks: fan-in and Each fan-out.

Owns the rules that verify a node's declared ``inputs`` are supplied by some
upstream producer. ``_check_item_input`` is the dispatcher the orchestrator
calls; it routes to ``_check_fan_in_inputs`` (dict-form fan-in) or
``_check_each_path`` (Each fan-out path resolution), and builds the
no-producer error (``_build_no_producer_error`` / ``_suggest_hint``) when no
upstream satisfies a plain single-type input.

Imports the type-compat primitives + shared vocabulary from
``_validation_types``; imported only from within the validation cluster.
"""

from __future__ import annotations

from typing import cast, get_args, get_origin

from neograph._ir_protocols import ConstructLike
from neograph._state_keys import StateKeys
from neograph._validation_types import (
    _MISSING,
    NodeItem,
    ProducerMap,
    _extract_list_element,
    _fmt_type,
    _resolve_field_annotation,
    _source_location,
    _types_compatible,
)
from neograph.errors import ConstructError, NeographError
from neograph.modifiers import Each, split_each_path
from neograph.naming import field_name_for
from neograph.node import Node, TypeSpecStatic


def _check_item_input(
    construct: ConstructLike,
    item: NodeItem,
    input_type: TypeSpecStatic,
    producers: ProducerMap,
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
        # dict is invariant; the narrowed input_type is dict[str, type] (a
        # subtype of dict[str, TypeSpecStatic] semantically but not by mypy
        # rules).
        _check_fan_in_inputs(construct, item, cast(dict[str, TypeSpecStatic], input_type), producers)
        return
    # Raw dict class: inputs=dict — multi-field isinstance extraction,
    # defers to runtime.
    if input_type is dict:
        return
    # Parameterized generic dict[str, X]: validate against producers if any
    # upstream has a parameterized dict output, otherwise defer to runtime.
    if get_origin(input_type) is dict:
        has_dict_producer = any(get_origin(p.effective_type) is dict for p in producers.values())
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
    for p in producers.values():
        if _types_compatible(p.effective_type, input_type):
            return

    error = _build_no_producer_error(construct, item, input_type, producers)
    raise error


def _check_fan_in_inputs(
    construct: ConstructLike,
    item: NodeItem,
    inputs_dict: dict[str, TypeSpecStatic],
    producers: ProducerMap,
) -> None:
    """Validate a fan-in ``inputs={'name': Type, ...}`` spec against the
    producer map by upstream name (neograph-kqd.2).

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
    # The reserved Portal "handoff" input reads the mesh channel, not a named
    # upstream producer (design §3.3). It is set ONLY by the normalizer, so a
    # NON-mesh node that declares a "handoff" input has handoff_param=None and
    # still fails the "no upstream node named 'handoff'" check below — the
    # desired rejection. Legal mesh members skip it here (the counterpart to the
    # fan_out_key skip). review MEDIUM-1.
    handoff_key = getattr(item, "handoff_param", None)
    ms = getattr(item, "modifier_set", None)
    has_each = ms is not None and ms.each is not None
    # ``producers`` is already keyed by field_name, so the
    # lookup is O(1) per upstream — no per-call rebuild of a name→producer map.
    # The Each fan-out receiver candidate(s) are computed by the single shared
    # rule from neograph-k7bg — the same one the normalizer used to set
    # fan_out_param. We pass our producer field set as known_field_names. This
    # validator owns only the POLICY below (tolerate one, error on extras);
    # the candidate identity is not re-derived here. Imported function-locally
    # to break the _validation_inputs -> _ir_normalize -> _sidecar ->
    # _di_classify -> _construct_validation import cycle.
    from neograph._ir_normalize import fan_out_candidates

    # item has dict-form inputs here (the fan-in path), so it is a Node.
    _fan_out_candidates = set(fan_out_candidates(cast(Node, item), set(producers)))
    # Track if the Each fan-out receiver slot was consumed.
    # If fan_out_param is already set (from @node), the slot is pre-consumed.
    _each_skip_used = fan_out_key is not None
    for upstream_name, expected_type in inputs_dict.items():
        if upstream_name == fan_out_key:
            continue
        if upstream_name == handoff_key:
            continue
        if upstream_name not in producers:
            # If the node has an Each modifier, ONE unmatched key is the
            # fan-out item receiver — skip it. Additional unmatched keys
            # are real errors (typos) and must be rejected.
            if has_each and not _each_skip_used and upstream_name in _fan_out_candidates:
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
                found=f"available upstreams: {sorted(producers)}",
                node=item.name,
                construct=construct.name,
                location=_source_location(),
            )
        producer_type = producers[upstream_name].effective_type
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
    construct: ConstructLike,
    item: NodeItem,
    input_type: TypeSpecStatic,
    each: Each,
    producers: ProducerMap,
) -> None:
    """Resolve each.over against producers; verify it lands on list[input_type]."""
    root, segments = split_each_path(each.over)

    matched = producers.get(root)
    root_type: TypeSpecStatic = matched.effective_type if matched is not None else None

    if root_type is None:
        # Allow framework-injected roots (sub-construct port param).
        if root == StateKeys.SUBGRAPH_INPUT:
            return
        known_roots = set(producers)
        raise ConstructError.build(
            f"Each(over='{each.over}') root '{root}' does not match any upstream node",
            found=f"known producers: {sorted(known_roots)}" if known_roots else "no producers available",
            node=item.name,
            construct=construct.name,
            hint="the first segment of 'over' must name an upstream node (e.g., 'source.items')",
            location=_source_location(),
        )

    # Walk remaining segments through Pydantic model_fields.
    current_type: TypeSpecStatic = root_type
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
            f"Each(over='{each.over}', key='{each.key}') — {_fmt_type(element_type)} has no field '{each.key}'",
            found=f"available fields: {sorted(element_fields.keys())}",
            node=item.name,
            construct=construct.name,
            location=_source_location(),
        )


def _build_no_producer_error(
    construct: ConstructLike,
    item: NodeItem,
    input_type: TypeSpecStatic,
    producers: ProducerMap,
) -> NeographError:
    if producers:
        producer_summary = "\n".join(f"    - {p.label}: {_fmt_type(p.effective_type)}" for p in producers.values())
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
    input_type: TypeSpecStatic,
    producers: ProducerMap,
) -> str | None:
    """Scan producer outputs for actionable suggestions."""
    # Check for Each dict[str, X] → raw X mismatch first.
    for p in producers.values():
        p_origin = get_origin(p.effective_type)
        if p_origin is dict:
            p_args = get_args(p.effective_type)
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
    for p in producers.values():
        model_fields = getattr(p.effective_type, "model_fields", None) or {}
        for fname in model_fields:
            resolved = _resolve_field_annotation(p.effective_type, fname)
            if resolved is _MISSING:
                continue
            element = _extract_list_element(resolved)
            if element is not None and _types_compatible(element, input_type):
                return f"did you forget to fan out? try .map(lambda s: s.{p.field_name}.{fname}, key='...')"
    return None
