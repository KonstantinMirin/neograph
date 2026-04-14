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
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel


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
        # Scalar — render as text (no JSON escaping)
        text = str(value)
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
                attr = f' description="{desc}"'
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
                # Scalar — render directly, no JSON escaping
                lines.append(f"<{field_name}{attr}>{field_value}</{field_name}>")

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

    - If renderer is None: return raw value unchanged (backward compat).
    - For dict-form (fan-in): render each value independently.
    - Checks hasattr(value, 'render_for_prompt') first (model method wins).
    - For single value: render directly.
    """
    if renderer is None:
        return input_data

    if isinstance(input_data, dict):
        return {k: _render_single(v, renderer) for k, v in input_data.items()}

    return _render_single(input_data, renderer)


def _render_single(value: Any, renderer: Renderer) -> Any:
    """Render a single value, checking for model-level override first.

    If render_for_prompt() returns a BaseModel, re-render it through the
    active renderer (BAML/XML/JSON). This lets models define typed
    presentation projections without doing string formatting themselves.
    """
    if hasattr(value, "render_for_prompt") and callable(value.render_for_prompt):
        result = value.render_for_prompt()
        if isinstance(result, BaseModel):
            return renderer.render(result)
        return result
    return renderer.render(value)
