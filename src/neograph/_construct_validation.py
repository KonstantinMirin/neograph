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
from typing import Any, ForwardRef, Union, get_args, get_origin, get_type_hints

from neograph.errors import ConstructError
from neograph.modifiers import Each, Loop, Oracle, split_each_path
from neograph.node import Node


def effective_producer_type(item: Any) -> Any:
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
    get_mod = getattr(item, "has_modifier", None)
    if get_mod is None:
        return output
    if get_mod(Each):
        return dict[str, output]
    return output


def validate_loop_self_edge(node: Node) -> None:
    """Check that a self-loop's output type is compatible with its input type.

    Called at ``|`` time (from ``Modifiable.__or__``) when a ``Loop`` modifier
    is applied to a Node. Loop on Node always means self-loop.
    """
    loop = node.get_modifier(Loop)
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
        msg = (
            f"loop on node '{node.name}' back-edge to '{node.name}': "
            f"output type {_fmt_type(output_type)} not compatible with "
            f"reentry input types ({type_list}).\n"
            f"  hint: the loop's last node output must match the "
            f"reentry node's input type."
        )
        raise ConstructError(msg)
        return

    # Single-type inputs.
    if not _types_compatible(output_type, input_type):
        msg = (
            f"loop on node '{node.name}' back-edge to '{node.name}': "
            f"output type {_fmt_type(output_type)} not compatible with "
            f"reentry input type {_fmt_type(input_type)}.\n"
            f"  hint: the loop's last node output must match the "
            f"reentry node's input type."
        )
        raise ConstructError(msg)


def _validate_node_chain(construct: Any) -> None:
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
            _check_item_input(construct, item, input_type, producers)

        # Node uses .outputs (plural); Construct / _BranchNode use .output (singular).
        output_type = item.outputs if isinstance(item, Node) else getattr(item, "output", None)
        name = getattr(item, "name", None)
        if output_type is not None and name is not None:
            field_name = name.replace("-", "_")

            # Dict-form outputs (neograph-1bp.4): register one producer per
            # output key, with modifier wrapping applied independently per key.
            if isinstance(item, Node) and isinstance(output_type, dict):
                has_each = item.has_modifier(Each)
                for output_key, key_type in output_type.items():
                    key_field = f"{field_name}_{output_key}"
                    key_label = f"node '{name}' output '{output_key}'"
                    producer_type = dict[str, key_type] if has_each else key_type
                    producers.append((key_field, producer_type, key_label))
            else:
                label = (
                    f"node '{name}'"
                    if isinstance(item, Node)
                    else f"sub-construct '{name}'"
                )
                # Shared helper decides the modifier-adjusted state-bus type.
                producers.append((field_name, effective_producer_type(item), label))

        # Each + Loop mutual exclusion (belt-and-suspenders: also checked
        # at | time in Modifiable.__or__, but the programmatic API can bypass
        # pipe syntax by constructing modifiers lists directly).
        if isinstance(item, Node):
            if item.has_modifier(Each) and item.has_modifier(Loop):
                msg = (
                    f"Node '{item.name}' has both Each and Loop modifiers. "
                    f"These cannot be combined on a single node. "
                    f"Use a sub-construct with Loop inside an Each fan-out instead."
                )
                raise ConstructError(msg)

        # Oracle + Loop mutual exclusion (belt-and-suspenders).
        if isinstance(item, Node):
            if item.has_modifier(Oracle) and item.has_modifier(Loop):
                msg = (
                    f"Node '{item.name}' has both Oracle and Loop modifiers. "
                    f"These cannot be combined on a single node. "
                    f"Use a sub-construct: nest the Loop body inside an Oracle "
                    f"ensemble, or vice versa."
                )
                raise ConstructError(msg)

        # (Loop reenter validation removed — Loop.reenter no longer exists.
        # Multi-node loops use Loop on Construct instead.)

    # Sub-construct output boundary contract: if construct.output is declared,
    # at least one internal node must produce a compatible type.
    # Exclude neo_subgraph_input — the input port is NOT a valid producer
    # for the output contract (neograph-luzc).
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
            msg = (
                f"Construct '{construct.name}' declares "
                f"output={_fmt_type(declared_output)} but no internal node "
                f"produces a compatible type.\n"
                f"  internal producers:\n{producer_summary}\n"
                f"{_location_suffix()}"
            )
            raise ConstructError(msg)


def validate_loop_construct(construct: Any) -> None:
    """Validate Loop on a Construct: output must be compatible with input.

    Called at ``|`` time from ``Modifiable.__or__`` when a Loop modifier
    is applied to a Construct.
    """
    if construct.output is None or construct.input is None:
        msg = (
            f"Loop on construct '{construct.name}' requires both input= "
            f"and output= declared."
        )
        raise ConstructError(msg)
    if not _types_compatible(construct.output, construct.input):
        msg = (
            f"Loop on construct '{construct.name}': output type "
            f"{_fmt_type(construct.output)} not compatible with input type "
            f"{_fmt_type(construct.input)}.\n"
            f"  hint: the loop's output must match the construct's "
            f"input type for the back-edge."
        )
        raise ConstructError(msg)


def _check_item_input(
    construct: Any,
    item: Any,
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
    get_mod = getattr(item, "get_modifier", None)
    each = get_mod(Each) if get_mod else None
    if each is not None:
        _check_each_path(construct, item, input_type, each, producers)
        return

    # Plain input: any producer whose output is assignable to input_type wins.
    for _, producer_type, _ in producers:
        if _types_compatible(producer_type, input_type):
            return

    msg = _format_no_producer_error(construct, item, input_type, producers)
    raise ConstructError(msg)


def _check_fan_in_inputs(
    construct: Any,
    item: Any,
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
    # path works without requiring fan_out_param to be set (neograph-ts7).
    fan_out_key = getattr(item, "fan_out_param", None)
    has_each = False
    get_mod = getattr(item, "has_modifier", None)
    if get_mod is not None:
        has_each = get_mod(Each)
    producer_by_name: dict[str, tuple[Any, str]] = {
        field_name: (producer_type, label)
        for field_name, producer_type, label in producers
    }
    for upstream_name, expected_type in inputs_dict.items():
        if upstream_name == fan_out_key:
            continue
        if upstream_name not in producer_by_name:
            # If the node has an Each modifier, an unmatched key is the
            # fan-out item receiver — skip it rather than rejecting.
            # This handles the programmatic API where fan_out_param isn't set.
            if has_each:
                continue
            msg = (
                f"Node '{item.name}' in construct '{construct.name}' "
                f"declares inputs['{upstream_name}']={_fmt_type(expected_type)} "
                f"but no upstream node named '{upstream_name}' exists.\n"
                f"  available upstreams: {sorted(producer_by_name.keys())}\n"
                f"{_location_suffix()}"
            )
            raise ConstructError(msg)
        producer_type, _label = producer_by_name[upstream_name]
        if not _types_compatible(producer_type, expected_type):
            msg = (
                f"Node '{item.name}' in construct '{construct.name}' "
                f"declares inputs['{upstream_name}']={_fmt_type(expected_type)} "
                f"but upstream '{upstream_name}' produces "
                f"{_fmt_type(producer_type)}.\n"
                f"{_location_suffix()}"
            )
            raise ConstructError(msg)


def _check_each_path(
    construct: Any,
    item: Any,
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
        # Root doesn't name any upstream producer. The collection is likely
        # pre-seeded by runtime state (e.g. a top-level Each with a state
        # field supplied via run(input=...)); defer to runtime.
        return

    # Walk remaining segments through Pydantic model_fields.
    current_type: Any = root_type
    walked = [root]
    for segment in segments:
        walked.append(segment)
        resolved = _resolve_field_annotation(current_type, segment)
        if resolved is _MISSING:
            msg = (
                f"Node '{item.name}' in construct '{construct.name}' has "
                f"Each(over='{each.over}') but '{'.'.join(walked)}' does not "
                f"resolve: {_fmt_type(current_type)} has no field '{segment}'.\n"
                f"{_location_suffix()}"
            )
            raise ConstructError(msg)
        current_type = resolved

    element_type = _extract_list_element(current_type)
    if element_type is None:
        msg = (
            f"Node '{item.name}' in construct '{construct.name}' has "
            f"Each(over='{each.over}'), but the path resolves to "
            f"{_fmt_type(current_type)}, not a list.\n"
            f"  hint: Each fans out over a collection; the terminal field "
            f"must be a list.\n"
            f"{_location_suffix()}"
        )
        raise ConstructError(msg)

    if not _types_compatible(element_type, input_type):
        msg = (
            f"Node '{item.name}' in construct '{construct.name}' declares "
            f"{'inputs' if isinstance(item, Node) else 'input'}="
            f"{_fmt_type(input_type)} with Each(over='{each.over}'), "
            f"but the path resolves to list[{_fmt_type(element_type)}].\n"
            f"{_location_suffix()}"
        )
        raise ConstructError(msg)

    # Verify each.key names a valid field on the element type.
    # Only check when the element type has model_fields (Pydantic model);
    # primitives (str, int, etc.) defer to runtime str(item) fallback.
    element_fields = getattr(element_type, "model_fields", None)
    if element_fields is not None and each.key not in element_fields:
        msg = (
            f"Node '{item.name}' in construct '{construct.name}' has "
            f"Each(over='{each.over}', key='{each.key}') but "
            f"{_fmt_type(element_type)} has no field '{each.key}'.\n"
            f"  available fields: {sorted(element_fields.keys())}\n"
            f"{_location_suffix()}"
        )
        raise ConstructError(msg)


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
    except Exception:
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
    # Parameterized generic producer (e.g. dict[str, X]):
    # compatible with raw origin class (dict) or exact parameterized match.
    producer_origin = get_origin(producer)
    target_origin = get_origin(target)
    if producer_origin is not None:
        # dict[str, X] vs dict → compatible (runtime isinstance handles it)
        if isinstance(target, type) and issubclass(producer_origin, target):
            return True
        # dict[str, X] vs dict[str, X] → compare origin + args
        if target_origin is not None and producer_origin is target_origin:
            return get_args(producer) == get_args(target)
        # dict[str, X] producer ↔ list[Y] consumer — merge-after-fanout
        # (neograph-kqd.2). A downstream node consuming an Each-fanned-out
        # result as list[Y] gets the runtime unwrap via dict.values() in
        # step 5 (factory._extract_input). Element-type compatibility is
        # checked recursively so subclass rules apply consistently.
        if producer_origin is dict and target_origin is list:
            prod_args = get_args(producer)     # (str, X)
            target_args = get_args(target)     # (Y,)
            if len(prod_args) == 2 and len(target_args) == 1:
                return _types_compatible(prod_args[1], target_args[0])
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


def _format_no_producer_error(
    construct: Any,
    item: Any,
    input_type: Any,
    producers: list[tuple[str, Any, str]],
) -> str:
    # ConstructError messages deliberately diverge from the package-wide
    # one-line error shape (see state.py, compiler.py, factory.py). Assembly-
    # time type mismatches benefit from structured help — a producer inventory,
    # a `.map()` hint when applicable, and a source-location pointer — that
    # a terse "Node 'X' has wrong input." would lose. Other errors in the
    # package either stay single-line by design (missing output type is a
    # one-fix mistake) or can adopt this richer format deliberately.
    if producers:
        producer_summary = "\n".join(
            f"    • {label}: {_fmt_type(t)}"
            for _, t, label in producers
        )
    else:
        producer_summary = "    (no upstream producers)"

    hint = _suggest_hint(input_type, producers)
    hint_line = f"  hint: {hint}\n" if hint else ""

    return (
        f"Node '{item.name}' in construct '{construct.name}' declares "
        f"{'inputs' if isinstance(item, Node) else 'input'}="
        f"{_fmt_type(input_type)} but no upstream produces a "
        f"compatible value.\n"
        f"  upstream producers:\n{producer_summary}\n"
        f"{hint_line}"
        f"{_location_suffix()}"
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
        frame = sys._getframe(1)
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
    except Exception:
        # Best-effort: location lookup must never crash Construct assembly.
        return None
    return None
