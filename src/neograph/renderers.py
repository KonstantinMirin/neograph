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
from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape as _xml_escape
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from neograph._rendered import Rendered
from neograph._state_keys import StateKeys
from neograph.describe_type import describe_value
from neograph.tool import ToolInteraction


@dataclass
class RenderedInput:
    """Unified representation of input data for prompt insertion.

    Carries both raw (for inline ``${var.field}`` dotted access) and rendered
    (for template-ref prompt_compiler) views of the same data. Built once via
    ``build_rendered_input()``, consumed by dispatch and lint.
    """

    raw: dict[str, Any] | Any
    """Raw Pydantic models — for inline prompt var substitution."""

    rendered: dict[str, Any] | Any
    """BAML/rendered strings — for template-ref prompt_compiler."""

    flattened: dict[str, Any] = field(default_factory=dict)
    """Extra fields from render_for_prompt BaseModel returns (template-ref only)."""

    available_keys_inline: set[str] = field(default_factory=set)
    """Keys valid for inline ${var} — raw dict keys only."""

    available_keys_template: set[str] = field(default_factory=set)
    """Keys valid for template-ref {var} — raw + flattened."""

    @property
    def for_template_ref(self) -> dict[str, Any] | Any:
        """What the prompt_compiler receives: rendered values + flattened fields.

        A non-dict input is KEYED here, by the raw value's type name, rather than
        travelling on as a bare scalar. It has to happen at this point because
        this is the last place the original type is still known -- once the value
        is rendered its type name is ``Rendered``, and the name is gone. That is
        also why the key cannot be assigned at the compiler seam.

        The same convention ``_alias_subgraph_input_port`` uses for a
        sub-construct port, and it closes the seam where a bare value used to be
        silently dropped to ``{}`` by ``render_inputs``.
        """
        if isinstance(self.rendered, dict):
            merged = dict(self.rendered)
            for k, v in self.flattened.items():
                if k not in merged:
                    merged[k] = v
            return merged
        if self.rendered is None or self.raw is None:
            return {}
        return {type(self.raw).__name__: self.rendered}


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

    def _get_description(self, field_name: str, field_info: Any, seen: set[str]) -> str | None:
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


# Separator between folded tool-interaction results in a research packet. A
# blank-line-fenced rule reads as a section break to the model without inventing
# a heading grammar the consumer must learn.
_TOOL_PACKET_SEP = "\n\n---\n\n"
# Separator between rendered values of a dict[str, BaseModel] fan-in. A blank
# line keeps each describe_value block visually distinct.
_MODEL_DICT_SEP = "\n\n"


def _is_model_dict(value: Any) -> bool:
    """True when *value* is a non-empty dict whose values are ALL BaseModels.

    This is the shape an ``Each`` fan-out produces when its results merge back
    onto the bus (``dict[str, output]`` per state.py:_add_output_field). A dict
    with any non-model value is left untouched (existing passthrough).
    """
    return isinstance(value, dict) and len(value) > 0 and all(isinstance(v, BaseModel) for v in value.values())


def _is_tool_interaction_list(value: Any) -> bool:
    """True when *value* is a non-empty list of ``ToolInteraction`` records."""
    if not isinstance(value, list) or not value:
        return False
    return all(isinstance(v, ToolInteraction) for v in value)


def _render_tool_interactions(value: list) -> str:
    """Fold a ``list[ToolInteraction]`` into a research packet: ``tool_name`` header (provenance,
    test_composition's gather->produce chain asserts it) + a body, joined by ``_TOOL_PACKET_SEP``.
    Body is ``.result`` when non-empty, else ``describe_value(.typed_result)`` -- a caller-built
    interaction commonly sets only ``typed_result``, and the default-empty ``result`` would else
    silently drop the only field with data (neograph-k7pjj, GH #19). ``args``/``duration_ms`` omitted."""

    def body(ti: ToolInteraction) -> str:
        return ti.result or (describe_value(ti.typed_result) if ti.typed_result is not None else "")

    return _TOOL_PACKET_SEP.join(f"{ti.tool_name}:\n{body(ti)}" for ti in value)


def _render_model_dict(value: dict) -> str:
    """Render a ``dict[str, BaseModel]`` as its values' ``describe_value`` blocks
    joined by ``_MODEL_DICT_SEP``.

    Keys are dropped: an ``Each`` barrier collects results in arrival order under
    synthesized keys (see the ordering caveat in AGENTS.md), so the keys carry no
    semantic meaning — the dict is an unordered bag of results.
    """
    return _MODEL_DICT_SEP.join(describe_value(v) for v in value.values())


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
        # Delegate rather than re-walk. This function used to carry its own copy
        # of the dict loop with two silent divergences from the real one: extras
        # merged into the same dict (so a flattened field could shadow a LATER
        # producer key) and no sub-construct port alias. It is public and
        # documented, so a divergent copy here is a standing invitation to a
        # fourth implementation. See neograph-l2a7w.
        return build_rendered_input(input_data, renderer).for_template_ref

    return to_rendered(input_data, renderer)


def build_rendered_input(
    input_data: Any,
    renderer: Renderer | None = None,
) -> RenderedInput:
    """Build a RenderedInput carrying both raw and rendered views.

    Single entry point for all rendering. The caller picks the appropriate
    view based on prompt type (inline vs template-ref).
    """
    if isinstance(input_data, dict):
        raw_dict = dict(input_data)
        rendered_dict: dict[str, Any] = {}
        flattened: dict[str, Any] = {}

        for k, v in input_data.items():
            rendered_val, extra = _render_with_flattening(v, renderer)
            rendered_dict[k] = rendered_val
            for fname, fval in extra.items():
                if fname not in flattened and fname not in rendered_dict:
                    flattened[fname] = fval

        _alias_subgraph_input_port(input_data, rendered_dict, flattened)

        inline_keys = set(raw_dict.keys())
        template_keys = inline_keys | set(flattened.keys())

        return RenderedInput(
            raw=raw_dict,
            rendered=rendered_dict,
            flattened=flattened,
            available_keys_inline=inline_keys,
            available_keys_template=template_keys,
        )

    # Single value (non-dict)
    rendered = to_rendered(input_data, renderer)
    return RenderedInput(
        raw=input_data,
        rendered=rendered,
        available_keys_inline=set(),
        available_keys_template=set(),
    )


def _alias_subgraph_input_port(
    input_data: dict[str, Any],
    rendered_dict: dict[str, Any],
    flattened: dict[str, Any],
) -> None:
    """Expose a sub-construct's port value under a friendly, template-ref-only
    alias -- the port's declared type name -- alongside the framework-internal
    ``neo_subgraph_input`` key (neograph-bluv, F3.4).

    Three IR-level mechanisms rewrite a node's ``inputs`` dict-form key to the
    literal ``neo_subgraph_input`` (the ``@node`` port rewrite in
    ``_construct_builder.py``, a hand-authored declarative port key, and the
    qot6 fan-over-agent auto-wrap in ``_fan_agent_wrap.py``). All three converge
    on this one ``{neo_subgraph_input: <value>}`` dict shape before rendering,
    so one alias here covers all three -- no rewrite site needs to change.

    ``neo_subgraph_input`` is NEVER removed (purely additive; back-compat for
    existing consumer code that reads it directly, e.g.
    ``_unwrap(input_data, "neo_subgraph_input", ...)``). The alias is skipped
    if it would collide with an existing ``rendered_dict``/``flattened`` key --
    mutates *flattened* only, so a real producer field of the same name wins
    over the synthesized alias (the same shadowing rule ``di_inputs`` uses).
    """
    port_value = input_data.get(StateKeys.SUBGRAPH_INPUT)
    if port_value is None:
        return
    alias = type(port_value).__name__
    if alias in rendered_dict or alias in flattened:
        return
    flattened[alias] = rendered_dict[StateKeys.SUBGRAPH_INPUT]


def to_prompt_input(value: Any, *, renderer: Renderer | None = None) -> dict[str, Rendered]:
    """Normalize anything into the mapping a prompt_compiler receives.

    The ONE place the compiler-facing shape is decided see neograph-l2a7w. Every
    value comes out ``Rendered``; every mapping comes out keyed. Idempotent,
    because ``to_rendered``'s rung 0 is — so a call site that already rendered
    with the node's ``renderer=`` loses nothing when the seam normalizes again.

    Bare, unkeyed values get a key here rather than at any call site. Four
    different producers hand a bare value to the compiler -- a single-type node
    input, a ``merge_pre_process`` return, ``compile_prompt(input_data=Model)``
    and ``render_prompt`` -- so naming them at the seam covers all four at once
    and keeps ``_oracle.py`` free of shape decisions (design section 4.4 / A2).
    The key is the value's type name, the same convention
    ``_alias_subgraph_input_port`` already uses for a sub-construct port.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): to_rendered(v, renderer) for k, v in value.items()}
    return {type(value).__name__: to_rendered(value, renderer)}


def to_raw_inputs(value: Any) -> dict[str, Any]:
    """``to_prompt_input``'s keying WITHOUT the rendering — the opt-in
    ``raw_inputs`` channel a prompt_compiler receives only when it declares one.

    Same keys as ``to_prompt_input`` so a compiler can read the two side by side:
    ``input_data[k]`` is always text, ``raw_inputs[k]`` is the object behind it.
    This is the home for the work rendering cannot express -- a per-NODE template
    loop over a model's children, which ``render_for_prompt()`` cannot serve
    because it is per-MODEL (design section 4.6).
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {type(value).__name__: value}


def _render_prompt_result(result: Any, renderer: Renderer | None) -> Any:
    """Render the return value of a ``render_for_prompt()`` override.

    Decodes the render_for_prompt return-type contract in ONE place (DRY-01): a
    str is verbatim; a BaseModel or non-empty list[BaseModel] goes through the
    explicit renderer or the BAML ``describe_value`` default; anything else
    passes through unchanged. Both ``to_rendered`` (inline/raw) and
    ``_render_with_flattening`` (template-ref) wrap this so the two rendering
    tails can never silently diverge on the contract.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return renderer.render(result) if renderer is not None else describe_value(result)
    if isinstance(result, list) and result and isinstance(result[0], BaseModel):
        return renderer.render(result) if renderer is not None else describe_value(result)
    return result


def _render_with_flattening(
    value: Any,
    renderer: Renderer | None,
) -> tuple[Any, dict[str, Any]]:
    """Render a value and extract flattened fields if render_for_prompt returns a model.

    Returns (rendered_value, extra_fields). extra_fields is empty unless
    render_for_prompt() returned a BaseModel, in which case it maps each
    non-excluded field name to its individually rendered value.
    """
    # Rung 0, before any probe. `hasattr` below would trip a Rendered value's
    # loud __getattr__, and this function is reached by every consumer compiler
    # that calls the public `render_inputs` -- so idempotence has to live at the
    # probe, not only inside `to_rendered`.
    if isinstance(value, Rendered):
        return value, {}
    if hasattr(value, "render_for_prompt") and callable(value.render_for_prompt):
        result = value.render_for_prompt()
        whole = _render_prompt_result(result, renderer)
        if isinstance(result, BaseModel):
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
                    fields[fname] = to_rendered(fval, renderer)
            return whole, fields
        return whole, {}
    return to_rendered(value, renderer), {}


def to_rendered(value: Any, renderer: Renderer | None = None, *, prefix: str | None = None) -> Rendered:
    """Render a single value to prompt-ready text — THE one rendering rule.

    Every channel that turns a pipeline value into LLM-facing text goes through
    here: node inputs, di_inputs, Oracle merge payloads, inline ``${var}`` leaves,
    and tool results. Three partial re-implementations of this ladder used to
    exist and disagreed pairwise; collapsing them is neograph-l2a7w.

    The ladder:

    0. already ``Rendered`` -> returned unchanged. This runs FIRST, before any
       ``hasattr`` probe, which makes the function IDEMPOTENT: rendering twice
       equals rendering once. That is what lets a call site render where the node
       (and its ``renderer=``) is in scope AND the seam render again, without the
       unenforceable "nobody may render twice" rule.
    1. ``render_for_prompt()`` — the user's presenter, wins over everything.
    2. an explicit ``renderer=`` (Xml / Delimited / Json).
    3. BAML ``describe_value`` for models and containers of models, including the
       two framework container shapes.
    4. ``str(value)``; ``None`` -> ``""``.

    ``prefix`` is threaded to ``describe_value`` for the tool-result channel,
    whose only genuine difference from the others was a ``"Tool result:"`` header.
    """
    # 0. Idempotence — already text, and probing it would trip its loud __getattr__.
    if isinstance(value, Rendered):
        return value

    # 1. render_for_prompt() always wins, regardless of renderer config
    if hasattr(value, "render_for_prompt") and callable(value.render_for_prompt):
        return Rendered(_render_prompt_result(value.render_for_prompt(), renderer))

    # 2. Explicit renderer — the renderer owns its own container handling
    # (XmlRenderer/DelimitedRenderer already walk dicts and lists), so the
    # framework-container rules below apply only to the BAML default.
    if renderer is not None:
        return Rendered(renderer.render(value))

    # 3. BAML default. None renders empty (not the literal 'None'); the two
    # framework-produced container shapes render before the generic
    # list[BaseModel] rule so their specialized folding wins.
    if value is None:
        return Rendered("")
    if isinstance(value, BaseModel):
        return Rendered(_describe(value, prefix))
    if _is_tool_interaction_list(value):
        return Rendered(_render_tool_interactions(value))
    if isinstance(value, list) and value and isinstance(value[0], BaseModel):
        return Rendered(_describe(value, prefix))
    if _is_model_dict(value):
        return Rendered(_render_model_dict(value))

    # 4. Coerce. Primitives, plain dicts and non-model lists become text HERE
    # rather than travelling on as live objects — the anomaly that forced every
    # downstream consumer to branch on type.
    return Rendered(str(value))


def _describe(value: Any, prefix: str | None) -> str:
    """``describe_value`` with the channel's optional header (tool results)."""
    return describe_value(value) if prefix is None else describe_value(value, prefix=prefix)
