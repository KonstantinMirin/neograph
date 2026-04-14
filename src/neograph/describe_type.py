"""TypeScript-style schema emitter for Pydantic models.

Two-pass recursive walker: pass 1 counts class occurrences (for auto-hoisting),
pass 2 renders the notation. Produces compact output that LLMs parse more
reliably than JSON Schema.
"""

from __future__ import annotations

import enum
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

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
    _count_classes(model, class_counts, enum_classes, visited=set())

    # Determine which classes to hoist.
    hoisted: set[type] = set()
    if hoist_classes == "auto":
        hoisted = {cls for cls, count in class_counts.items() if count >= 2}
    elif hoist_classes == "all":
        hoisted = set(class_counts.keys())
    else:
        name_set = set(hoist_classes)
        hoisted = {cls for cls in class_counts if cls.__name__ in name_set}

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
                cls, indent=indent, depth=0, or_splitter=or_splitter,
                hoisted=hoisted, visited=set(),
            )
            lines.append(f"type {cls.__name__} = {body}")
        lines.append("")

    # Render the main model body.
    body = _render_model_body(
        model, indent=indent, depth=0, or_splitter=or_splitter,
        hoisted=hoisted, visited=set(),
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
    visited: set[type],
) -> None:
    """Recursively count how many times each nested BaseModel/Enum appears."""
    if model in visited:
        return
    visited.add(model)

    for _name, field_info in model.model_fields.items():
        if field_info.exclude:
            continue
        _count_annotation(field_info.annotation, counts, enum_classes, visited)


def _count_annotation(
    annotation: Any,
    counts: dict[type, int],
    enum_classes: set[type],
    visited: set[type],
) -> None:
    """Count occurrences within a single type annotation."""
    if annotation is None or annotation is type(None):
        return

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union or origin is types.UnionType:
        for arg in args:
            _count_annotation(arg, counts, enum_classes, visited)
        return

    if origin is Literal:
        return

    if origin in (list, tuple, frozenset, set):
        for arg in args:
            _count_annotation(arg, counts, enum_classes, visited)
        return

    if origin is dict:
        for arg in args:
            _count_annotation(arg, counts, enum_classes, visited)
        return

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        enum_classes.add(annotation)
        counts[annotation] = counts.get(annotation, 0) + 1
        return

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        counts[annotation] = counts.get(annotation, 0) + 1
        _count_classes(annotation, counts, enum_classes, visited)
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
        if field_info.exclude:
            continue
        type_str = _render_type(
            field_info.annotation,
            indent=indent, depth=depth + 1, or_splitter=or_splitter,
            hoisted=hoisted, visited=visited,
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
                arg, indent=indent, depth=depth, or_splitter=or_splitter,
                hoisted=hoisted, visited=visited,
            )
            for arg in args
        ]
        return or_splitter.join(parts)

    # Literal.
    if origin is Literal:
        parts = [f'"{v}"' if isinstance(v, str) else str(v) for v in args]
        return or_splitter.join(parts)

    # list / tuple / set / frozenset.
    if origin in (list, tuple, frozenset, set):
        if args:
            inner = _render_type(
                args[0], indent=indent, depth=depth, or_splitter=or_splitter,
                hoisted=hoisted, visited=visited,
            )
        else:
            inner = "any"
        return f"[{inner}]"

    # dict.
    if origin is dict:
        if args and len(args) == 2:
            key = _render_type(
                args[0], indent=indent, depth=depth, or_splitter=or_splitter,
                hoisted=hoisted, visited=visited,
            )
            val = _render_type(
                args[1], indent=indent, depth=depth, or_splitter=or_splitter,
                hoisted=hoisted, visited=visited,
            )
            return f"object<{key}, {val}>"
        return "object"

    # Enum.
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        if annotation in hoisted:
            return annotation.__name__
        members = [f'"{m.value}"' if isinstance(m.value, str) else str(m.value)
                    for m in annotation]
        return or_splitter.join(members)

    # BaseModel.
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in hoisted:
            return annotation.__name__
        return _render_model_body(
            annotation, indent=indent, depth=depth, or_splitter=or_splitter,
            hoisted=hoisted, visited=visited,
        )

    # Any / unknown.
    if annotation is Any:
        return "any"

    return str(annotation)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _field_comment(field_info: FieldInfo) -> str:
    """Extract description from a Pydantic FieldInfo for inline comment."""
    if field_info.description:
        return field_info.description
    return ""


def _render_enum_declaration(cls: type, indent: str) -> str:
    """Render an Enum class as ``enum Foo { A, B, C }``."""
    members = [f'"{m.value}"' if isinstance(m.value, str) else str(m.value)
               for m in cls]  # type: ignore[attr-defined]
    return f"enum {cls.__name__} {{ {', '.join(members)} }}"


def _stable_sort(classes: set[type]) -> list[type]:
    """Sort classes by name for deterministic output."""
    return sorted(classes, key=lambda c: c.__name__)


# ---------------------------------------------------------------------------
# describe_value: BAML-style instance renderer
# ---------------------------------------------------------------------------


def describe_value(
    value: Any,
    *,
    prefix: str = "",
    indent: str = "  ",
) -> str:
    """Render a Pydantic model instance in TypeScript-style notation with values.

    Same format as ``describe_type`` but with actual values instead of type
    names. Field descriptions appear as ``//`` inline comments.

    Handles: BaseModel instances, lists of BaseModel instances, primitives.

    Parameters
    ----------
    value:
        A Pydantic BaseModel instance, a list of instances, or a primitive.
    prefix:
        Text line prepended before the rendered block.
    indent:
        Indentation unit.
    """
    lines: list[str] = []
    if prefix:
        lines.append(prefix)

    if isinstance(value, BaseModel):
        lines.append(_render_instance(value, indent=indent, depth=0))
    elif isinstance(value, list):
        lines.append(_render_list_value(value, indent=indent, depth=0))
    else:
        lines.append(repr(value))

    return "\n".join(lines)


def _render_instance(
    instance: BaseModel,
    *,
    indent: str,
    depth: int,
) -> str:
    """Render a BaseModel instance as ``{ field: value // description }``."""
    pad = indent * depth
    inner_pad = indent * (depth + 1)
    field_lines: list[str] = []

    for field_name, field_info in instance.__class__.model_fields.items():
        if field_info.exclude:
            continue
        val = getattr(instance, field_name)
        val_str = _render_value(val, indent=indent, depth=depth + 1)
        comment = _field_comment(field_info)
        line = f"{inner_pad}{field_name}: {val_str}"
        if comment:
            line = f"{line}  // {comment}"
        field_lines.append(line)

    if not field_lines:
        return "{}"

    return "{\n" + "\n".join(field_lines) + "\n" + pad + "}"


def _render_value(value: Any, *, indent: str, depth: int) -> str:
    """Render a single value in BAML notation."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, BaseModel):
        return _render_instance(value, indent=indent, depth=depth)
    if isinstance(value, list):
        return _render_list_value(value, indent=indent, depth=depth)
    if isinstance(value, dict):
        return _render_dict_value(value, indent=indent, depth=depth)
    return repr(value)


def _render_list_value(lst: list, *, indent: str, depth: int) -> str:
    """Render a list in BAML notation."""
    if not lst:
        return "[]"
    pad = indent * depth
    inner_pad = indent * (depth + 1)
    items = [
        f"{inner_pad}{_render_value(item, indent=indent, depth=depth + 1)}"
        for item in lst
    ]
    return "[\n" + ",\n".join(items) + "\n" + pad + "]"


def _render_dict_value(d: dict, *, indent: str, depth: int) -> str:
    """Render a dict in BAML notation."""
    if not d:
        return "{}"
    pad = indent * depth
    inner_pad = indent * (depth + 1)
    entries = [
        f"{inner_pad}{_render_value(k, indent=indent, depth=depth + 1)}: "
        f"{_render_value(v, indent=indent, depth=depth + 1)}"
        for k, v in d.items()
    ]
    return "{\n" + "\n".join(entries) + "\n" + pad + "}"
