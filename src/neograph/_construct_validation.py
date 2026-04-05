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

from neograph.modifiers import Each, split_each_path
from neograph.node import Node


class ConstructError(ValueError):
    """Raised when Construct assembly fails type/topology validation.

    Subclasses ValueError so existing `pytest.raises(ValueError)` patterns
    still catch it, while allowing callers that want finer-grained handling
    to catch the specific type.
    """


def effective_producer_type(item: Any) -> Any:
    """Return the type this producer writes to the state bus, accounting
    for modifiers.

    This is the **single source of truth** for the "producer side" of
    type compatibility. Both validator walkers (``_validate_node_chain``
    here and ``_validate_fan_in_types`` in ``decorators.py``) consult it,
    so a new modifier that reshapes state only needs to teach this one
    function about the new rule — both walkers pick up the change.

    Current rules:
      - ``Each`` modifier → ``dict[str, raw_output]`` (aggregated fan-out
        results land as a dict keyed by ``each.key``; see
        ``state.py:_add_output_field`` for the state builder side of
        this rule).
      - Everything else → the item's declared ``.output`` unchanged.

    Returns ``None`` when the item has no declared output.
    """
    output = getattr(item, "output", None)
    if output is None:
        return None
    get_mod = getattr(item, "has_modifier", None)
    if get_mod is None:
        return output
    if get_mod(Each):
        return dict[str, output]
    return output


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
        # Node has `inputs` (plural) since neograph-kqd.1. Construct still has
        # `input` (sub-construct boundary port) — that's handled above via
        # construct.input, not here.
        input_type = getattr(item, "inputs", None)
        if input_type is None:
            input_type = getattr(item, "input", None)
        if input_type is not None:
            _check_item_input(construct, item, input_type, producers)

        output_type = getattr(item, "output", None)
        name = getattr(item, "name", None)
        if output_type is not None and name is not None:
            field_name = name.replace("-", "_")
            label = (
                f"node '{name}'"
                if isinstance(item, Node)
                else f"sub-construct '{name}'"
            )
            # Shared helper decides the modifier-adjusted state-bus type.
            # Never compute this inline here or in the @node walker —
            # both walkers must stay in sync through this single call.
            producers.append((field_name, effective_producer_type(item), label))


def _check_item_input(
    construct: Any,
    item: Any,
    input_type: Any,
    producers: list[tuple[str, Any, str]],
) -> None:
    """Validate that `item.input` is satisfied by some upstream producer.

    Static validation only fires when there's enough evidence to make a
    definite call. Cases that defer to runtime isinstance-scanning:
      - No upstream producers at all (first-of-chain).
      - `dict` / non-class input types (multi-field or raw extraction).
      - Each modifier whose root segment doesn't match a known producer.
    """
    if not producers:
        return
    # dict-shaped inputs defer to runtime isinstance scanning
    # (factory._extract_input / _is_instance_safe) except parameterized
    # generics (dict[str, X]) which can be validated against upstream
    # Each-modified producers.
    #   - dict instance: multi-field extraction, e.g. input={"a": X, "b": Y}
    #   - raw dict class: input=dict (isinstance match on any dict state field)
    if isinstance(input_type, dict):
        return
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
            f"input={_fmt_type(input_type)} with Each(over='{each.over}'), "
            f"but the path resolves to list[{_fmt_type(element_type)}].\n"
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
        f"input={_fmt_type(input_type)} but no upstream produces a "
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
