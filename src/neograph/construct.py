"""Construct — ordered composition of Nodes. The pipeline blueprint.

    # Top-level pipeline
    rw_pipeline = Construct(
        "rw-ingestion",
        nodes=[read_node, decompose, classify, ...],
    )

    # Sub-construct with declared I/O boundary
    enrich = Construct(
        "enrich",
        input=Claims,
        output=ScoredClaims,
        nodes=[lookup, verify, score],
    )

    # Compose: sub-construct in a parent pipeline
    main = Construct("main", nodes=[decompose, enrich, report])
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, ForwardRef, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from neograph.modifiers import Each, Modifiable, Modifier, split_each_path
from neograph.node import Node


class ConstructError(ValueError):
    """Raised when Construct assembly fails type/topology validation.

    Subclasses ValueError so existing `pytest.raises(ValueError)` patterns
    still catch it, while allowing callers that want finer-grained handling
    to catch the specific type.
    """


class Construct(Modifiable, BaseModel):
    """An ordered composition of Nodes that compiles to a LangGraph StateGraph.

    Nodes execute in sequence. Modifiers (Oracle, Each, Operator) on
    individual nodes modify the topology — the Construct itself is a flat list.
    The compiler handles fan-out, barriers, and interrupts.

    When used as a sub-construct inside another Construct, declare input/output
    to define the state boundary. The sub-construct gets its own isolated state.

    Modifiers can be applied to Constructs via pipe:
        sub | Oracle(n=3, merge_fn="merge")   — ensemble the entire sub-pipeline
        sub | Each(over="items", key="label") — run sub-pipeline per item
        sub | Operator(when="check")          — interrupt after sub-pipeline

    Type safety: input/output compatibility across the node chain is validated
    at assembly time (when `Construct(nodes=[...])` is called), not at
    `compile()` time. Mismatches raise `ConstructError` with a pointer to the
    user source line and a suggestion (e.g. "did you forget .map()?").
    """

    name: str
    description: str = ""
    nodes: list[Any] = []  # list[Node | Construct] — Any avoids circular ref issues

    # I/O boundary — required when used as a sub-construct
    input: type[BaseModel] | None = None
    output: type[BaseModel] | None = None

    # Modifiers applied via | operator
    modifiers: list[Modifier] = []

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, name_: str | None = None, /, **kwargs):
        if name_ is not None:
            kwargs["name"] = name_
        super().__init__(**kwargs)
        # Validate after pydantic finishes so ConstructError escapes cleanly
        # rather than being wrapped in a pydantic ValidationError. Nested
        # constructs self-validate during their own __init__.
        _validate_node_chain(self)

    # has_modifier, get_modifier, __or__, map inherited from Modifiable


# ═══════════════════════════════════════════════════════════════════════════
# Assembly-time validation
# ═══════════════════════════════════════════════════════════════════════════


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
            producers.append((field_name, output_type, label))


def _check_item_input(
    construct: Construct,
    item: Any,
    input_type: Any,
    producers: list[tuple[str, Any, str]],
) -> None:
    """Validate that `item.input` is satisfied by some upstream producer.

    Static validation only fires when there's enough evidence to make a
    definite call. Cases that defer to runtime isinstance-scanning:
      - No upstream producers at all (first-of-chain — input comes from
        `run(input=...)` kwargs).
      - `dict` / non-class input types (multi-field or raw extraction).
      - Each modifier whose root segment doesn't match a known producer
        (the collection is pre-seeded by runtime state).
    """
    # Can't know what run(input=...) kwargs will seed the state with.
    if not producers:
        return
    # dict inputs (multi-field extraction) are too permissive to validate.
    if isinstance(input_type, dict):
        return
    # Anything that isn't a class — skip (defensive; shouldn't happen).
    if not isinstance(input_type, type):
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
    construct: Construct,
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
    """True if a value of type `producer` can satisfy a consumer of `target`."""
    if producer is target:
        return True
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
    # Handle Union / Optional / X | None
    is_union = origin is Union or (
        hasattr(types, "UnionType") and origin is types.UnionType
    )
    if is_union:
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
    construct: Construct,
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
    """Scan producer outputs for a list[input_type] field; suggest .map()."""
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
