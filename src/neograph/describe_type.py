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
from pydantic.fields import FieldInfo

from neograph.errors import ConfigurationError

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
        parts = ", ".join(f"{k}: {type_display_name(v)}" for k, v in t.items())
        return "{" + parts + "}"

    # A union must render its MEMBERS. Python 3.14 gave `types.UnionType` a
    # `__name__` of "Union", so trusting `__name__` degrades every error message
    # that names a union: a fan-in mismatch reports the producer as "Union"
    # rather than "Claims | str", which is the one detail the reader needs. On
    # 3.12 `__name__` was absent and the `str(t)` fallback happened to be right,
    # though it leaks the module prefix.
    if get_origin(t) is Union or isinstance(t, types.UnionType):
        return " | ".join(type_display_name(arg) for arg in get_args(t))

    if t is type(None):
        return "None"

    # A subscripted generic must render its PARAMETER. `__name__` is the bare
    # origin -- "list" for list[Claims] -- so trusting it produced messages that
    # said `expected: list, found: list`, telling the author that list differs
    # from list. The parameter is the entire information the message carries.
    # Recurses, so list[dict[str, Claims]] renders in full, and reuses the union
    # branch above for a member like Claims | None.
    origin = get_origin(t)
    if origin is not None:
        args = get_args(t)
        if args:
            rendered = ", ".join("..." if a is Ellipsis else type_display_name(a) for a in args)
            return f"{type_display_name(origin)}[{rendered}]"

    return getattr(t, "__name__", str(t))


class ExcludeFromOutput:
    """Marker: field is visible in input rendering but excluded from output schema.

    Use with Annotated on a Pydantic field::

        source_url: Annotated[str, ExcludeFromOutput] = ""

    The field will:
    - Be rendered when the model is shown as input (XmlRenderer, DelimitedRenderer)
    - Be EXCLUDED from describe_type() output schema (LLM won't try to produce it)
    - Must have a default value (since the LLM won't provide it)
    """

    pass


def _is_output_excluded(field_info: FieldInfo) -> bool:
    """True if the field carries an ExcludeFromOutput marker in Annotated metadata."""
    return any(m is ExcludeFromOutput or isinstance(m, ExcludeFromOutput) for m in field_info.metadata)


_PRIMITIVE_MAP: dict[type, str] = {
    str: "string",
    int: "int",
    float: "float",
    bool: "bool",
    type(None): "null",
}


def describe_type(
    model: Any,
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
        The annotation to describe. A Pydantic ``BaseModel`` subclass renders
        as a ``{ field: type }`` block; a container or union over models
        (``list[M]``, ``dict[str, M]``, ``M | None``) renders through the same
        dispatch a nested field would use, hoisting its members as usual. An
        annotation this module has no notation for raises ``TypeError`` rather
        than rendering its ``repr`` -- see Raises.
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

    Raises
    ------
    ConfigurationError
        When *model* is neither a ``BaseModel`` subclass nor a shape
        :func:`_render_type` has notation for. Rendering such an annotation
        would emit its ``repr`` into an LLM prompt as though it were a schema,
        so the boundary refuses instead (neograph-vduhp / GH issue #8).
    """
    is_model = isinstance(model, type) and issubclass(model, BaseModel)

    # Pass 1: count class occurrences to decide what to hoist. A BaseModel is
    # walked field-by-field; anything else is walked as a bare annotation, which
    # recurses into generic args so a `list[M]` still hoists M's dependencies.
    # These stay SEPARATE on purpose: `_count_annotation` would also count the
    # top model itself, pushing a model that appears nested too from 1 to 2 and
    # silently starting to hoist it under `hoist_classes='auto'`.
    class_counts: dict[type, int] = {}
    enum_classes: set[type] = set()
    recursive: set[type] = set()
    if is_model:
        _count_classes(model, class_counts, enum_classes, recursive, visited=set(), path=set())
    else:
        _count_annotation(model, class_counts, enum_classes, recursive, set(), set())

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

    # Render the main type. `strict=True` turns _render_type's lenient
    # `str(annotation)` fallthrough into a refusal at THIS boundary only --
    # nested fields keep rendering leniently.
    if is_model:
        body = _render_model_body(
            model,
            indent=indent,
            depth=0,
            or_splitter=or_splitter,
            hoisted=hoisted,
            visited=set(),
        )
    else:
        body = _render_type(
            model,
            indent=indent,
            depth=0,
            or_splitter=or_splitter,
            hoisted=hoisted,
            visited=set(),
            strict=True,
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
        if field_info.exclude or _is_output_excluded(field_info):
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


def _admits_none(annotation: Any) -> bool:
    """True when *annotation* can hold ``None``.

    Mirrors the ``NoneType`` handling in :func:`_render_type` -- the same
    question that function answers when it emits a ``null`` union member --
    so the emitter and this guard cannot drift apart.
    """
    if annotation is None or annotation is type(None):
        return True
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        return any(arg is type(None) for arg in get_args(annotation))
    return False


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
        if field_info.exclude or _is_output_excluded(field_info):
            continue
        type_str = _render_type(
            field_info.annotation,
            indent=indent,
            depth=depth + 1,
            or_splitter=or_splitter,
            hoisted=hoisted,
            visited=visited,
        )
        # A nullable annotation already contributed its own ``null`` union
        # member in ``_render_type``; appending a second one for the same fact
        # (the field also being non-required) renders ``T or null or null``.
        # Guard on the ANNOTATION, not on ``type_str``: PEP-604 unions keep
        # author order, so ``None | str`` renders ``null or string`` and a
        # trailing-text check would miss it (neograph-g21jc / GH issue #7).
        if not field_info.is_required() and not _admits_none(field_info.annotation):
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
    strict: bool = False,
) -> str:
    """Render a single type annotation to schema notation.

    ``strict`` is consulted only at the final fallthrough, and only by the
    top-level :func:`describe_type` boundary: the recursive calls below
    deliberately do not pass it, so a field with an exotic annotation keeps
    rendering as its ``repr`` rather than breaking an existing pipeline.
    """
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

    if strict:
        raise ConfigurationError.build(
            "describe_type() has no schema notation for this annotation",
            found=repr(annotation),
            expected="a Pydantic BaseModel subclass, or a container/union over one",
            hint="pass Model, list[Model], dict[str, Model], or Model | None.",
        )
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
    members = [f'"{m.value}"' if isinstance(m.value, str) else str(m.value) for m in cls]  # type: ignore[attr-defined]
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
