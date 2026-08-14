"""TypeScript-style schema emitter for Pydantic models.

Two-pass recursive walker: pass 1 counts class occurrences (for auto-hoisting),
pass 2 renders the notation. Produces compact output that LLMs parse more
reliably than JSON Schema.
"""

from __future__ import annotations

import enum
import types
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

# --- extracted cluster (neograph-3ffdg.17), re-exported so existing
# --- `from neograph.describe_type import describe_value` keeps resolving.
# --- _field_comment is imported BACK: it moved with the value renderer but
# --- _render_model_body here still uses it.
from neograph._describe_value import (  # noqa: E402,F401
    _field_comment,
    _render_dict_value,
    _render_instance,
    _render_list_value,
    _render_value,
    describe_value,
)

# ExcludeFromOutput moved to _output_classify.py (neograph-ftnxl.4), which is
# now the single owner of the output-field marker family (ExcludeFromOutput +
# Carried) and the shared output_markers() predicate both this renderer and
# the structured-output schema projector (project_output_model) consume.
# Re-exported here so `from neograph.describe_type import ExcludeFromOutput`
# keeps resolving.
from neograph._output_classify import Carried, ExcludeFromOutput, output_markers  # noqa: E402,F401

if TYPE_CHECKING:
    from neograph.node import TypeSpecStatic


def type_display_name(t: TypeSpecStatic) -> str:
    """Render a ``TypeSpec`` to a short, human-readable display string.

    Single source of truth for the short type-name rendering used in logs and
    ``ConstructError``/``DeprecationWarning`` messages — as opposed to
    :func:`describe_type`, which emits the full TS-notation schema.

      - ``None``                -> ``"None"``
      - dict-form (``{key: type}``) -> ``"{key: TypeName, ...}"``
      - everything else         -> ``t.__name__`` if present, else ``str(t)``
    """
    if t is None:
        return "None"
    if isinstance(t, dict):
        parts = ", ".join(f"{k}: {getattr(v, '__name__', str(v))}" for k, v in t.items())
        return "{" + parts + "}"
    return getattr(t, "__name__", str(t))


_PRIMITIVE_MAP: dict[type, str] = {
    str: "string",
    int: "int",
    float: "float",
    bool: "bool",
    type(None): "null",
}


def describe_type(
    model: type[BaseModel],
    *,
    prefix: str = "Answer in JSON matching this schema:",
    hoist_classes: Literal["auto", "all"] | list[str] = "auto",
    always_hoist_enums: bool = True,
    or_splitter: str = " or ",
    indent: str = "  ",
) -> str:
    """Render a Pydantic model as TypeScript-style schema notation.

    Parameters
    ----------
    model:
        The Pydantic BaseModel class to describe.
    prefix:
        Text line prepended before the schema block. Empty string to omit.
    hoist_classes:
        'auto' hoists classes appearing 2+ times as ``type Foo = { ... }``
        declarations. 'all' hoists every nested BaseModel. A list of class
        names hoists only those.
    always_hoist_enums:
        When True, Enum classes are always hoisted as ``enum Foo { ... }``.
    or_splitter:
        Separator for Union types. Defaults to ' or ' (LLMs parse this better
        than '|').
    indent:
        Indentation unit.

    Returns
    -------
    str
        The rendered schema string.
    """
    # Pass 1: count class occurrences to decide what to hoist.
    class_counts: dict[type, int] = {}
    enum_classes: set[type] = set()
    recursive: set[type] = set()
    _count_classes(model, class_counts, enum_classes, recursive, visited=set(), path=set())

    # Determine which classes to hoist.
    hoisted: set[type] = set()
    if hoist_classes == "auto":
        hoisted = {cls for cls, count in class_counts.items() if count >= 2}
    elif hoist_classes == "all":
        hoisted = set(class_counts.keys())
    else:
        name_set = set(hoist_classes)
        hoisted = {cls for cls in class_counts if cls.__name__ in name_set}

    # Recursive models MUST be hoisted regardless of the hoist policy: the
    # back-edge renders a bare `Name` reference, so without a `type Name = {...}`
    # declaration the LLM sees a dangling type name.
    hoisted |= recursive

    if always_hoist_enums:
        hoisted |= enum_classes

    # Pass 2: render hoisted declarations, then the main type.
    lines: list[str] = []
    rendered_hoisted: set[type] = set()

    if prefix:
        lines.append(prefix)

    # Render hoisted declarations in a stable order.
    for cls in _stable_sort(hoisted):
        if cls in rendered_hoisted:  # pragma: no cover — dedup guard
            continue
        rendered_hoisted.add(cls)
        if cls in enum_classes:
            lines.append(_render_enum_declaration(cls, indent))
        else:
            body = _render_model_body(
                cls,
                indent=indent,
                depth=0,
                or_splitter=or_splitter,
                hoisted=hoisted,
                visited=set(),
            )
            lines.append(f"type {cls.__name__} = {body}")
        lines.append("")

    # Render the main model body.
    body = _render_model_body(
        model,
        indent=indent,
        depth=0,
        or_splitter=or_splitter,
        hoisted=hoisted,
        visited=set(),
    )
    lines.append(body)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pass 1: count class occurrences
# ---------------------------------------------------------------------------


def _count_classes(
    model: type[BaseModel],
    counts: dict[type, int],
    enum_classes: set[type],
    recursive: set[type],
    visited: set[type],
    path: set[type],
) -> None:
    """Recursively count how many times each nested BaseModel/Enum appears.

    ``visited`` is a global dedup set (each model's subtree is walked once, so
    counts stay accurate for repeated siblings). ``path`` is the current
    ancestor chain, used to detect back-edges — a model reachable from itself is
    recorded in ``recursive`` and later force-hoisted.
    """
    if model in visited:
        return
    visited.add(model)
    path.add(model)

    for _name, field_info in model.model_fields.items():
        if field_info.exclude or output_markers(field_info)[0]:
            continue
        _count_annotation(field_info.annotation, counts, enum_classes, recursive, visited, path)

    path.discard(model)


def _count_annotation(
    annotation: Any,
    counts: dict[type, int],
    enum_classes: set[type],
    recursive: set[type],
    visited: set[type],
    path: set[type],
) -> None:
    """Count occurrences within a single type annotation."""
    if annotation is None or annotation is type(None):
        return

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union or origin is types.UnionType:
        for arg in args:
            _count_annotation(arg, counts, enum_classes, recursive, visited, path)
        return

    if origin is Literal:
        return

    if origin in (list, tuple, frozenset, set):
        for arg in args:
            _count_annotation(arg, counts, enum_classes, recursive, visited, path)
        return

    if origin is dict:
        for arg in args:
            _count_annotation(arg, counts, enum_classes, recursive, visited, path)
        return

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        enum_classes.add(annotation)
        counts[annotation] = counts.get(annotation, 0) + 1
        return

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        counts[annotation] = counts.get(annotation, 0) + 1
        # A back-edge to an ancestor currently on the path is a recursion cycle.
        if annotation in path:
            recursive.add(annotation)
        _count_classes(annotation, counts, enum_classes, recursive, visited, path)
        return


# ---------------------------------------------------------------------------
# Pass 2: render
# ---------------------------------------------------------------------------


def _render_model_body(
    model: type[BaseModel],
    *,
    indent: str,
    depth: int,
    or_splitter: str,
    hoisted: set[type],
    visited: set[type],
) -> str:
    """Render a BaseModel as a ``{ field: type }`` block."""
    if model in visited:
        return model.__name__

    visited.add(model)
    pad = indent * depth
    inner_pad = indent * (depth + 1)
    field_lines: list[str] = []

    for field_name, field_info in model.model_fields.items():
        if field_info.exclude or output_markers(field_info)[0]:
            continue
        type_str = _render_type(
            field_info.annotation,
            indent=indent,
            depth=depth + 1,
            or_splitter=or_splitter,
            hoisted=hoisted,
            visited=visited,
        )
        # Check if field is optional (has a default).
        if not field_info.is_required():
            type_str = f"{type_str} or null"

        comment = _field_comment(field_info)
        line = f"{inner_pad}{field_name}: {type_str}"
        if comment:
            line = f"{line}  // {comment}"
        field_lines.append(line)

    if not field_lines:
        return "{}"

    return "{\n" + "\n".join(field_lines) + "\n" + pad + "}"


def _render_type(
    annotation: Any,
    *,
    indent: str,
    depth: int,
    or_splitter: str,
    hoisted: set[type],
    visited: set[type],
) -> str:
    """Render a single type annotation to schema notation."""
    if annotation is None or annotation is type(None):
        return "null"

    # Primitives.
    if annotation in _PRIMITIVE_MAP:
        return _PRIMITIVE_MAP[annotation]

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional.
    if origin is Union or origin is types.UnionType:
        parts = [
            _render_type(
                arg,
                indent=indent,
                depth=depth,
                or_splitter=or_splitter,
                hoisted=hoisted,
                visited=visited,
            )
            for arg in args
        ]
        return or_splitter.join(parts)

    # Literal.
    if origin is Literal:
        parts = [f'"{v}"' if isinstance(v, str) else str(v) for v in args]
        return or_splitter.join(parts)

    # Heterogeneous tuple: tuple[A, B, ...] keeps every positional member.
    # Variadic tuple[X, ...] and bare tuple fall through to the array arm below.
    if origin is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
        parts = [
            _render_type(
                arg,
                indent=indent,
                depth=depth,
                or_splitter=or_splitter,
                hoisted=hoisted,
                visited=visited,
            )
            for arg in args
        ]
        return "[" + ", ".join(parts) + "]"

    # list / tuple / set / frozenset — single-element array notation. For a
    # variadic tuple[X, ...] the second member is the Ellipsis sentinel, so
    # args[0] is the element type.
    if origin in (list, tuple, frozenset, set):
        if args:
            inner = _render_type(
                args[0],
                indent=indent,
                depth=depth,
                or_splitter=or_splitter,
                hoisted=hoisted,
                visited=visited,
            )
        else:
            inner = "any"
        return f"[{inner}]"

    # dict.
    if origin is dict:
        if args and len(args) == 2:
            key = _render_type(
                args[0],
                indent=indent,
                depth=depth,
                or_splitter=or_splitter,
                hoisted=hoisted,
                visited=visited,
            )
            val = _render_type(
                args[1],
                indent=indent,
                depth=depth,
                or_splitter=or_splitter,
                hoisted=hoisted,
                visited=visited,
            )
            return f"object<{key}, {val}>"
        return "object"

    # Enum.
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        if annotation in hoisted:
            return annotation.__name__
        members = [f'"{m.value}"' if isinstance(m.value, str) else str(m.value) for m in annotation]
        return or_splitter.join(members)

    # BaseModel.
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in hoisted:
            return annotation.__name__
        return _render_model_body(
            annotation,
            indent=indent,
            depth=depth,
            or_splitter=or_splitter,
            hoisted=hoisted,
            visited=visited,
        )

    # Any / unknown.
    if annotation is Any:
        return "any"

    return str(annotation)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_enum_declaration(cls: type, indent: str) -> str:
    """Render an Enum class as ``enum Foo { A, B, C }``."""
    members = [f'"{m.value}"' if isinstance(m.value, str) else str(m.value) for m in cls]  # type: ignore[attr-defined]
    return f"enum {cls.__name__} {{ {', '.join(members)} }}"


def _stable_sort(classes: set[type]) -> list[type]:
    """Sort classes by name for deterministic output."""
    return sorted(classes, key=lambda c: c.__name__)


# ---------------------------------------------------------------------------
# describe_value: BAML-style instance renderer
# ---------------------------------------------------------------------------
