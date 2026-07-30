"""``describe_value`` — render a model INSTANCE as a typed value literal.

Extracted from ``describe_type.py`` (neograph-3ffdg.17) as a pure file split —
the functions below are unchanged, only their home moved. ``describe_type.py``
re-exports them, so existing imports keep resolving.

This is the few-shot-example half of the renderer: ``describe_type`` renders a
model CLASS as a schema, this renders an INSTANCE as a value literal an LLM can
copy. ``_field_comment`` moved here too and ``describe_type.py`` imports it back
-- both halves use it, and the dependency has to point one way.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def _field_comment(field_info: FieldInfo) -> str:
    """Extract description from a Pydantic FieldInfo for inline comment."""
    if field_info.description:
        return field_info.description
    return ""


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
    items = [f"{inner_pad}{_render_value(item, indent=indent, depth=depth + 1)}" for item in lst]
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
