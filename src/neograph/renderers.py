"""Pluggable prompt-input renderers.

Three built-in renderers convert Pydantic models to LLM-friendly text:

    XmlRenderer      — XML elements per field (Pydantic AI format_as_xml style)
    DelimitedRenderer — DSPy-style [[ ## field ## ]] headers
    JsonRenderer      — model_dump_json backward compat

Usage:

    renderer = XmlRenderer(include_field_info='once')
    text = renderer.render(my_model)

Dispatch helper for the factory layer:

    rendered = render_input(input_data, renderer=renderer)
"""

from __future__ import annotations

import json
from html import escape as _xml_escape
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from neograph.describe_type import describe_value


@runtime_checkable
class Renderer(Protocol):
    """Protocol for prompt-input renderers."""

    def render(self, value: Any) -> str: ...


class XmlRenderer:
    """Render Pydantic models as XML elements.

    Ported from Pydantic AI's format_as_xml pattern. Fields become XML
    elements; nested BaseModels become nested XML; lists become repeated
    <item> elements. Multi-line prose is NOT JSON-escaped.

    include_field_info controls field description emission:
        'once'   — description as attribute on first occurrence
        'always' — description as attribute on every occurrence
        'never'  — no descriptions
    """

    def __init__(
        self,
        *,
        include_field_info: Literal["once", "always", "never"] = "once",
    ) -> None:
        self.include_field_info = include_field_info

    def render(self, value: Any) -> str:
        seen: set[str] = set()
        return self._render_value(value, seen=seen)

    def _render_value(self, value: Any, *, tag: str | None = None, seen: set[str]) -> str:
        if isinstance(value, BaseModel):
            return self._render_model(value, tag=tag, seen=seen)
        if isinstance(value, list):
            return self._render_list(value, tag=tag, seen=seen)
        if isinstance(value, dict):
            return self._render_dict(value, tag=tag, seen=seen)
        # Scalar — render as text with XML escaping for special chars
        text = _xml_escape(str(value), quote=False)
        if tag is not None:
            return f"<{tag}>{text}</{tag}>"
        return text

    def _render_model(self, model: BaseModel, *, tag: str | None = None, seen: set[str]) -> str:
        lines: list[str] = []
        if tag is not None:
            lines.append(f"<{tag}>")

        fields = model.__class__.model_fields
        for field_name, field_info in fields.items():
            if field_info.exclude:
                continue
            field_value = getattr(model, field_name)
            desc = self._get_description(field_name, field_info, seen)
            if desc:
                attr = f' description="{_xml_escape(desc, quote=True)}"'
            else:
                attr = ""

            if isinstance(field_value, BaseModel):
                inner = self._render_model(field_value, tag=None, seen=seen)
                lines.append(f"<{field_name}{attr}>")
                lines.append(inner)
                lines.append(f"</{field_name}>")
            elif isinstance(field_value, list):
                lines.append(f"<{field_name}{attr}>")
                for item in field_value:
                    item_str = self._render_value(item, tag="item", seen=seen)
                    lines.append(item_str)
                lines.append(f"</{field_name}>")
            elif isinstance(field_value, dict):
                inner = self._render_dict(field_value, tag=None, seen=seen)
                lines.append(f"<{field_name}{attr}>")
                lines.append(inner)
                lines.append(f"</{field_name}>")
            else:
                # Scalar — render with XML escaping
                lines.append(f"<{field_name}{attr}>{_xml_escape(str(field_value), quote=False)}</{field_name}>")

        if tag is not None:
            lines.append(f"</{tag}>")
        return "\n".join(lines)

    def _render_list(self, lst: list, *, tag: str | None = None, seen: set[str]) -> str:
        lines: list[str] = []
        if tag is not None:
            lines.append(f"<{tag}>")
        for item in lst:
            lines.append(self._render_value(item, tag="item", seen=seen))
        if tag is not None:
            lines.append(f"</{tag}>")
        return "\n".join(lines)

    def _render_dict(self, d: dict, *, tag: str | None = None, seen: set[str]) -> str:
        lines: list[str] = []
        if tag is not None:
            lines.append(f"<{tag}>")
        for k, v in d.items():
            lines.append(self._render_value(v, tag=str(k), seen=seen))
        if tag is not None:
            lines.append(f"</{tag}>")
        return "\n".join(lines)

    def _get_description(
        self, field_name: str, field_info: Any, seen: set[str]
    ) -> str | None:
        if self.include_field_info == "never":
            return None
        desc = field_info.description
        if desc is None:
            return None
        if self.include_field_info == "once":
            if field_name in seen:
                return None
            seen.add(field_name)
        return desc


class DelimitedRenderer:
    """Render Pydantic models with DSPy-style delimited field headers.

    Fields get [[ ## field_name ## ]] headers. Nested models recurse.
    Lists use bullet points with '- ' prefix.
    """

    def render(self, value: Any) -> str:
        if isinstance(value, BaseModel):
            return self._render_model(value)
        if isinstance(value, list):
            return self._render_list(value)
        return str(value)

    def _render_model(self, model: BaseModel, *, prefix: str = "") -> str:
        lines: list[str] = []
        for field_name, field_info in model.__class__.model_fields.items():
            if field_info.exclude:
                continue
            field_value = getattr(model, field_name)
            header = f"{prefix}{field_name}" if prefix else field_name
            if isinstance(field_value, BaseModel):
                inner = self._render_model(field_value, prefix=f"{header}.")
                lines.append(inner)
            elif isinstance(field_value, list):
                lines.append(f"[[ ## {header} ## ]]")
                for item in field_value:
                    if isinstance(item, BaseModel):
                        inner = self._render_model(item, prefix=f"{header}.")
                        lines.append(f"- {inner}")
                    else:
                        lines.append(f"- {item}")
            else:
                lines.append(f"[[ ## {header} ## ]]")
                lines.append(str(field_value))
        return "\n".join(lines)

    def _render_list(self, lst: list) -> str:
        lines: list[str] = []
        for item in lst:
            if isinstance(item, BaseModel):
                lines.append(f"- {self._render_model(item)}")
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)


class JsonRenderer:
    """Render values as JSON — explicit backward-compat opt-in.

    Uses model_dump_json for BaseModel instances, json.dumps for others.
    """

    def __init__(self, *, indent: int = 2) -> None:
        self.indent = indent

    def render(self, value: Any) -> str:
        if isinstance(value, BaseModel):
            return value.model_dump_json(indent=self.indent)
        return json.dumps(value, indent=self.indent, default=str)


def render_input(input_data: Any, *, renderer: Renderer | None) -> Any:
    """Dispatch helper: render input data for prompt insertion.

    Default rendering is BAML via ``describe_value()`` — the same format used
    for tool results, so input and tool-result rendering are symmetric.

    Dispatch precedence per value:
      1. ``model.render_for_prompt()`` wins (str verbatim, BaseModel
         re-rendered with field flattening for dict-form inputs).
      2. Explicit renderer (XmlRenderer / DelimitedRenderer / JsonRenderer).
      3. BAML via ``describe_value()`` for Pydantic models / list[BaseModel].
      4. Primitives / non-Pydantic values pass through unchanged.

    For dict-form (fan-in): each value is rendered independently. When
    ``render_for_prompt()`` returns a BaseModel, its fields are flattened
    into the parent dict as individually addressable template vars.
    """
    if isinstance(input_data, dict):
        result: dict[str, Any] = {}
        for k, v in input_data.items():
            rendered, extra = _render_with_flattening(v, renderer)
            result[k] = rendered
            for fname, fval in extra.items():
                if fname not in result:
                    result[fname] = fval
        return result

    return _render_single(input_data, renderer)


def _render_with_flattening(
    value: Any, renderer: Renderer | None,
) -> tuple[Any, dict[str, Any]]:
    """Render a value and extract flattened fields if render_for_prompt returns a model.

    Returns (rendered_value, extra_fields). extra_fields is empty unless
    render_for_prompt() returned a BaseModel, in which case it maps each
    non-excluded field name to its individually rendered value.
    """
    if hasattr(value, "render_for_prompt") and callable(value.render_for_prompt):
        result = value.render_for_prompt()
        if isinstance(result, BaseModel):
            whole = renderer.render(result) if renderer else describe_value(result)
            fields: dict[str, Any] = {}
            for fname, finfo in result.__class__.model_fields.items():
                if finfo.exclude:
                    continue
                fval = getattr(result, fname)
                # Preserve BaseModel and list[BaseModel] children for dotted
                # template access ({var.attr} and {items[0].source})
                if isinstance(fval, BaseModel):
                    fields[fname] = fval
                elif isinstance(fval, list) and fval and isinstance(fval[0], BaseModel):
                    fields[fname] = fval
                else:
                    fields[fname] = _render_single(fval, renderer)
            return whole, fields
        if isinstance(result, str):
            return result, {}
        # list[BaseModel] or other non-str/non-BaseModel returns: render via _render_single
        if isinstance(result, list) and result and isinstance(result[0], BaseModel):
            rendered = renderer.render(result) if renderer else describe_value(result)
            return rendered, {}
        return result, {}
    return _render_single(value, renderer), {}


def _render_single(value: Any, renderer: Renderer | None) -> Any:
    """Render a single value, checking for model-level override first.

    When renderer is None, Pydantic models are BAML-rendered via
    describe_value() — symmetric with tool-result rendering.
    """
    # 1. render_for_prompt() always wins, regardless of renderer config
    if hasattr(value, "render_for_prompt") and callable(value.render_for_prompt):
        result = value.render_for_prompt()
        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            if renderer is not None:
                return renderer.render(result)
            return describe_value(result)
        if isinstance(result, list) and result and isinstance(result[0], BaseModel):
            if renderer is not None:
                return renderer.render(result)
            return describe_value(result)
        return result

    # 2. Explicit renderer
    if renderer is not None:
        return renderer.render(value)

    # 3. BAML default for Pydantic models and lists of models
    if isinstance(value, BaseModel):
        return describe_value(value)
    if isinstance(value, list) and value and isinstance(value[0], BaseModel):
        return describe_value(value)

    # 4. Primitives pass through
    return value
