"""Each fan-out receiver shaping + conditional DataFlowEdge for the Agent Spec
export (neograph-qtfof.7).

Extracted from ``_agent_spec_modifier_lowering.py``/``_agent_spec.py`` (both at
their file-size ceiling) as a focused peer module, mirroring the
``_agent_spec_loop_predicate.py`` split-by-concern precedent (neograph-qtfof.6).
Package-private: imported only by those two modules.

The bug this closes: ``_lower_each``'s StartNode declared the fan-out receiver
as FLATTENED per-field Properties (``item.v``), so pyagentspec's MapNode infers
``iterated_item:v`` instead of one ``iterated_item`` Property a real collection
edge can target -- confirmed empirically against the installed pyagentspec
adapter (see neograph-qtfof.7's research notes). ``each_item_property`` fixes
the SHAPE (one ObjectProperty per item, titled with the ``fan_out_param`` key).

Fixing the shape is necessary but not sufficient: pyagentspec's own converter
(``_langgraphconverter.py``) only honours a DataFlowEdge into a MapNode's
``iterated_*`` input when ``json_schemas_have_same_type(source, ListProperty(
item_type=receiver))`` -- otherwise it is silently DROPPED from
``inputs_to_iterate`` and the whole collection is bound as one item (a silent
broadcast, worse than no edge, per the Core Invariant). ``each_edge_is_safe``
pre-checks that SAME predicate at export time so neograph only emits an edge
pyagentspec will actually honour; ``resolve_each_source_property`` re-derives
the JSON-Schema shape of ``Each.over``'s dotted path (assembly-time validation,
``_check_each_path``, already proved the path resolves to
``list[compatible-element-type]`` -- this does not re-validate that, only
re-derives the SCHEMA for the compatibility check).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, create_model

from neograph._agent_spec_placeholders import _item_outputs
from neograph._construct_validation import _MISSING, _resolve_field_annotation
from neograph._each import split_each_path
from neograph.spec_types import _annotation_to_property


def each_item_property(item_type: type[BaseModel], title: str) -> Any:
    """ONE ObjectProperty for the fan-out receiver's WHOLE per-item model,
    titled ``title`` (the ``fan_out_param`` key) -- not the flattened
    per-field form ``_properties_for`` would produce. Reuses Pydantic's own
    ``model_json_schema()`` (Core Invariant: one JSON-Schema walker), never a
    hand-rolled field walk.
    """
    schema = item_type.model_json_schema()
    return _annotation_to_property(None, schema, schema.get("$defs", {}), title=title)


def _type_to_property(tp: Any, title: str) -> Any:
    """A Property for an ARBITRARY resolved type (``list[X]``, a nested model,
    ...) via a throwaway single-field wrapper model -- Pydantic already knows
    how to schema-ize any type it accepts as a field annotation, so this
    reuses that instead of hand-walking typing generics."""
    wrapper = create_model("_EachFanOutSourceWrapper", value=(tp, ...))
    full = wrapper.model_json_schema()
    return _annotation_to_property(None, full["properties"]["value"], full.get("$defs", {}), title=title)


def resolve_each_source_property(over: str, producer_types: dict[str, Any], title: str) -> Any | None:
    """Resolve ``Each.over``'s dotted path against ``producer_types`` (upstream
    node name -> declared output type) to the Property representing the FULL
    collection the fan-out reads (e.g. ``list[Elem]``). Returns ``None`` if the
    root or a path segment can't be resolved -- should not happen post-
    assembly-validation, but export must not crash on it; the caller falls
    back to emitting no edge (anti-band-aid: no edge beats a wrong edge).
    """
    root, segments = split_each_path(over)
    current = producer_types.get(root)
    if current is None:
        return None
    for segment in segments:
        current = _resolve_field_annotation(current, segment)
        if current is _MISSING:
            return None
    return _type_to_property(current, title)


def each_receiver_properties(item_inputs: dict[str, Any], fan_out_param: str | None, flat_fn: Any) -> list[Any]:
    """The StartNode's declared input Properties for an Each sub-flow: the
    ``fan_out_param`` key gets ONE ``each_item_property`` (fixing the shape
    bug), every other key stays flattened via the caller's ``flat_fn``
    (normally ``_properties_for``) unchanged -- a non-fan-out context input
    (e.g. ``verify(source: RawText, cluster: Elem)``) is an ordinary
    DataFlowEdge target, not an iteration receiver."""
    if not isinstance(item_inputs, dict) or fan_out_param not in item_inputs:
        return list(flat_fn(item_inputs))
    others = {k: v for k, v in item_inputs.items() if k != fan_out_param}
    return [each_item_property(item_inputs[fan_out_param], fan_out_param), *flat_fn(others)]


def each_edge_is_safe(source_property: Any, receiver_item_property: Any) -> bool:
    """``True`` iff wiring ``source_property`` into a MapNode's
    ``iterated_{title}`` input is an edge pyagentspec's OWN loader
    (``_langgraphconverter.py``) will actually honour -- pre-checks the exact
    predicate that loader applies (``json_schemas_have_same_type`` against
    ``ListProperty(item_type=receiver_item_property)``) so neograph never
    emits an edge the loader would silently drop (treating the whole
    collection as one broadcast item)."""
    from pyagentspec.property import ListProperty, json_schemas_have_same_type

    return json_schemas_have_same_type(source_property.json_schema, ListProperty(item_type=receiver_item_property).json_schema)


def each_fanout_edge_source(
    over: str,
    fan_out_key: str,
    fan_out_type: type[BaseModel],
    item_by_name: dict[str, Any],
    data_node_by_item_name: dict[str, Any],
    properties_for_fn: Any,
) -> tuple[Any, str] | None:
    """Resolve ``Each.over`` to ``(source_node, source_property_title)`` for a
    real DataFlowEdge into the MapNode's ``iterated_{fan_out_key}`` input, or
    ``None`` when this slice's scope boundary applies: a MULTI-segment path
    (only single-segment -- ``upstream.field`` -- is supported; deeper paths
    stay metadata-only, an explicit scope choice not a bug) or an unsafe edge
    (``each_edge_is_safe`` is False -- anti-band-aid: no edge beats a wrong
    edge). ``item_by_name``/``data_node_by_item_name`` are ``to_agent_spec``'s
    own bookkeeping dicts, passed through rather than re-derived.
    """
    root, segments = split_each_path(over)
    if len(segments) != 1:
        return None
    upstream_item = item_by_name.get(root)
    source_node = data_node_by_item_name.get(root)
    if upstream_item is None or source_node is None:
        return None
    upstream_outputs = _item_outputs(upstream_item)
    source_prop = next((p for p in properties_for_fn(upstream_outputs) if p.title == segments[0]), None)
    if source_prop is None:
        return None
    receiver_prop = each_item_property(fan_out_type, fan_out_key)
    if not each_edge_is_safe(source_prop, receiver_prop):
        return None
    return source_node, source_prop.title
