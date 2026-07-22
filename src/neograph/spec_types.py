"""Type registry — maps string names to Pydantic models.

Supports two registration paths:

1. Explicit: ``register_type('Draft', Draft)`` for pre-existing Python types.
2. Auto-generated: ``load_project_types(config)`` reads a ``types`` dict of
   JSON Schema definitions and builds Pydantic models via ``create_model``.

The registry is consumed by ``load_spec`` (and future spec loaders) to
resolve string type references in YAML/JSON pipeline definitions into
concrete Pydantic classes.

Also bridges Agent Spec ``Property`` (JSON Schema) objects to/from this same
registry via ``model_to_agent_spec_properties`` / ``agent_spec_properties_to_types``
— both reuse ``_resolve_field_type`` in place rather than a parallel walker,
per the Core Invariant: exactly one JSON-Schema-dict-to-Pydantic walker in
the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, create_model

from neograph.errors import ConfigurationError

if TYPE_CHECKING:
    from pyagentspec.property import Property

log = structlog.get_logger()

_type_registry: dict[str, type[BaseModel]] = {}

# JSON Schema primitive type → Python type
_JSON_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}


def _fields_match(a: type[BaseModel], b: type[BaseModel]) -> bool:
    """Return True if both models have the same field names."""
    return set(a.model_fields.keys()) == set(b.model_fields.keys())


def register_type(name: str, cls: type[BaseModel]) -> None:
    """Register a Pydantic model under *name* for spec-based lookup.

    Idempotent: if *name* already maps to a model with the same fields,
    the call is a no-op.  If the fields differ, the registry is updated
    and a warning is logged.
    """
    existing = _type_registry.get(name)
    if existing is not None:
        if _fields_match(existing, cls):
            return  # same schema — skip silently
        log.warning(
            "register_type: overwriting type with different schema",
            type_name=name,
            old_fields=sorted(existing.model_fields.keys()),
            new_fields=sorted(cls.model_fields.keys()),
        )
    _type_registry[name] = cls


def lookup_type(name: str) -> type[BaseModel]:
    """Return the model registered under *name*.

    Raises ``ConfigurationError`` if *name* is not found.
    """
    try:
        return _type_registry[name]
    except KeyError:
        raise ConfigurationError.build(
            f"type {name!r} is not registered",
            hint="use register_type() or include it in project types",
        ) from None


_REF_POINTER_PREFIX = "#/$defs/"


def _no_repr_check(schema: dict[str, Any]) -> None:
    """Fail loud on JSON Schema shapes with no neograph type representation.

    tuple[...] (``prefixItems``), ``Literal[...]`` (``const``/single-value
    ``enum``), and ``Enum`` (``enum``) have no round-trippable neograph
    target — silently falling through to ``Any`` would be a silent-degrade,
    which violates the fail-loud-over-fail-soft North Star.
    """
    if "prefixItems" in schema:
        raise ConfigurationError.build(
            "tuple-shaped JSON Schema (prefixItems) has no neograph type representation",
            expected="a DIRECT-tier shape (primitive, list, dict, object, anyOf/Optional)",
            found=f"prefixItems schema: {schema!r}",
            hint="tuple types are NO-REPR for Agent Spec round-trip — use list[T] instead",
        )
    if "const" in schema or "enum" in schema:
        raise ConfigurationError.build(
            "Literal/Enum-shaped JSON Schema (const/enum) has no neograph type representation",
            expected="a DIRECT-tier shape (primitive, list, dict, object, anyOf/Optional)",
            found=f"const/enum schema: {schema!r}",
            hint="Literal/Enum types are NO-REPR for Agent Spec round-trip",
        )


def _resolve_field_type(schema: dict[str, Any], defs: dict[str, Any] | None = None) -> Any:
    """Map a single JSON Schema field definition to a Python type annotation.

    ``defs`` is the ``$defs`` map from a full ``model_json_schema()`` output
    (or an equivalent map of named object schemas), used to resolve
    ``#/$defs/Name`` JSON-pointer refs emitted by Pydantic's own schema
    generator. Threaded through every recursive call so nested refs resolve
    at any depth.
    """
    defs = defs or {}

    ref = schema.get("$ref")
    if ref:
        if ref.startswith(_REF_POINTER_PREFIX):
            def_name = ref.removeprefix(_REF_POINTER_PREFIX)
            if def_name not in defs:
                raise ConfigurationError.build(
                    f"JSON pointer {ref!r} has no matching entry in $defs",
                    expected=f"one of {sorted(defs.keys())}",
                    found=def_name,
                    hint="pass the full model_json_schema()['$defs'] map as defs=",
                )
            return _resolve_field_type(defs[def_name], defs=defs)
        # Bare type name (not a JSON pointer) — existing registry lookup.
        return lookup_type(ref)

    any_of = schema.get("anyOf")
    if any_of is not None:
        members = [_resolve_field_type(member, defs=defs) for member in any_of]
        non_null = [m for m in members if m is not type(None)]
        has_null = len(non_null) != len(members)
        if len(non_null) == 1:
            inner = non_null[0]
            return (inner | None) if has_null else inner  # type: ignore[operator]
        union = non_null[0]
        for member in non_null[1:]:
            union = union | member  # type: ignore[operator]
        return (union | None) if has_null else union  # type: ignore[operator]

    json_type = schema.get("type")

    if json_type == "null":
        return type(None)

    if json_type == "array":
        items = schema.get("items", {})
        inner = _resolve_field_type(items, defs=defs)
        return list[inner]  # type: ignore[valid-type]

    if json_type == "object":
        _no_repr_check(schema)
        properties = schema.get("properties")
        if properties:
            required_fields = set(schema.get("required", []))
            fields: dict[str, Any] = {}
            for field_name, field_schema in properties.items():
                field_type = _resolve_field_type(field_schema, defs=defs)
                if field_name in required_fields:
                    fields[field_name] = (field_type, ...)
                else:
                    fields[field_name] = (field_type, None)
            model_name = schema.get("title") or "AnonymousNestedModel"
            return create_model(model_name, __base__=BaseModel, **fields)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            value_type = _resolve_field_type(additional, defs=defs)
            return dict[str, value_type]  # type: ignore[valid-type]
        return dict[str, Any]

    if json_type in _JSON_SCHEMA_TYPE_MAP:
        return _JSON_SCHEMA_TYPE_MAP[json_type]

    _no_repr_check(schema)

    # Fallback for unrecognised schemas
    return Any


def load_project_types(project_config: dict[str, Any]) -> None:
    """Generate and register Pydantic models from the ``types`` section.

    ``project_config["types"]`` is a dict mapping type names to JSON Schema
    object definitions.  Each definition is converted to a Pydantic model
    via ``create_model`` and registered under its name.

    Types are processed in definition order. A type may reference another
    type that appears *earlier* in the dict via ``{"$ref": "TypeName"}``.
    """
    types_section: dict[str, Any] = project_config.get("types", {})

    for type_name, type_def in types_section.items():
        fields: dict[str, Any] = {}
        properties = type_def.get("properties", {})
        required_fields = set(type_def.get("required", []))
        unknown_required = required_fields - set(properties.keys())
        if unknown_required:
            raise ConfigurationError.build(
                f"type '{type_name}' has unknown required fields",
                expected=f"fields from {sorted(properties.keys())}",
                found=f"required fields {sorted(unknown_required)} not in properties",
                hint="check that every required field is also defined in properties",
            )

        for field_name, field_schema in properties.items():
            field_type = _resolve_field_type(field_schema)
            if field_name in required_fields:
                fields[field_name] = (field_type, ...)
            else:
                fields[field_name] = (field_type, None)

        model = create_model(type_name, __base__=BaseModel, **fields)
        register_type(type_name, model)


def _import_agent_spec_property_classes() -> Any:
    """Function-local import of pyagentspec's Property classes.

    Import-guarded so ``src/neograph`` core stays Agent-Spec-free by
    default -- only calling one of the two bridge functions below pulls in
    the optional ``[agent-spec]`` extra.
    """
    try:
        import pyagentspec.property as pyagentspec_property
    except ImportError as exc:
        raise ConfigurationError.build(
            "pyagentspec is not installed",
            expected="the [agent-spec] optional extra",
            found="ImportError on pyagentspec.property",
            hint="install with: uv sync --extra agent-spec (or pip install neograph[agent-spec])",
        ) from exc
    return pyagentspec_property


def _structural_type_name(props: list[Property]) -> str:
    """Derive a registry name purely from a Property list's STRUCTURE
    (title + type, sorted), not from any node/model name.

    A reconstructed Agent Spec import has no back-reference to the original
    Pydantic class name (Property carries only per-field shape) -- so two
    DIFFERENT sites reconstructing the SAME structural shape (e.g. a
    self-loop's own output feeding back as one of its own inputs, or a
    nested object type appearing both top-level and inside a list) would
    otherwise synthesize DIFFERENT, incompatible classes and fail
    construct-validation type-compatibility checks even though the data is
    identical. Naming the registration purely by structure makes
    ``register_type``'s existing content-match idempotency
    (``_fields_match``) automatically reuse ONE class for every
    structurally-identical Property list -- the single canonical helper
    both the top-level bridge (``agent_spec_properties_to_types``) and the
    nested-object branch (``_property_to_field_type``) use.
    """
    import hashlib

    sig = tuple(sorted((p.title, str(getattr(p, "type", None))) for p in props))
    digest = hashlib.sha256(repr(sig).encode()).hexdigest()[:16]
    return f"AgentSpecType_{digest}"


def _property_to_field_type(prop: Property) -> tuple[Any, Any]:
    """Map a single Agent Spec ``Property`` to a (type, default) field spec.

    Reuses the SAME primitive-type map (``_JSON_SCHEMA_TYPE_MAP``) and the
    same NO-REPR fail-loud discipline as ``_resolve_field_type`` -- this is
    the Property-object-shaped twin of that function, not a parallel
    walker: both ultimately produce a type via the same rules, just reading
    from a live ``Property`` tree instead of a raw JSON-Schema dict.
    """
    pas = _import_agent_spec_property_classes()

    if isinstance(prop, pas.UnionProperty):
        non_null_members = [m for m in prop.any_of if not isinstance(m, pas.NullProperty)]
        has_null = len(non_null_members) != len(prop.any_of)
        non_null = [_property_to_field_type(m)[0] for m in non_null_members]
        if len(non_null) == 1:
            inner = non_null[0]
            field_type = (inner | None) if has_null else inner
        else:
            union = non_null[0]
            for member in non_null[1:]:
                union = union | member
            field_type = (union | None) if has_null else union
        default = None if has_null else ...
        return field_type, default

    if isinstance(prop, pas.ListProperty):
        inner, _ = _property_to_field_type(prop.item_type)
        return list[inner], ...  # type: ignore[valid-type]

    if isinstance(prop, pas.DictProperty):
        value_type, _ = _property_to_field_type(prop.value_type)
        return dict[str, value_type], ...  # type: ignore[valid-type]

    if isinstance(prop, pas.ObjectProperty):
        fields: dict[str, Any] = {}
        for field_name, field_prop in prop.properties.items():
            field_type, field_default = _property_to_field_type(field_prop)
            fields[field_name] = (field_type, field_default)
        # Structural dedup (see _structural_type_name): a nested object
        # appearing in two different places (e.g. top-level AND inside a
        # list) must reconstruct to the SAME class both times, or type
        # compatibility checks between them fail even though the data is
        # identical. register_type's content-match idempotency does the
        # actual reuse; this is the canonical (register + lookup) path, not
        # a second ad-hoc create_model call.
        model_name = _structural_type_name(list(prop.properties.values()))
        # from_attributes: a reconstructed Agent Spec import has no back-reference
        # to the ORIGINAL Pydantic class name (Property only carries per-field
        # shape, never a model-level identity) -- the model built here is
        # structurally equivalent, not identical. Runtime state passed between
        # dispatched nodes (e.g. Portal mode (b)) is a REAL instance of the
        # original class; from_attributes lets Pydantic validate it into this
        # reconstructed model by matching attribute names, rather than requiring
        # exact class identity LangGraph's state coercion would otherwise demand.
        model = create_model(model_name, __base__=BaseModel, __config__=ConfigDict(from_attributes=True), **fields)
        register_type(model_name, model)
        return lookup_type(model_name), ...

    if isinstance(prop, pas.NullProperty):
        return type(None), None

    primitive_map: dict[type, type] = {
        pas.StringProperty: str,
        pas.IntegerProperty: int,
        pas.NumberProperty: float,
        pas.FloatProperty: float,
        pas.BooleanProperty: bool,
    }
    for prop_cls, py_type in primitive_map.items():
        if isinstance(prop, prop_cls):
            return py_type, ...

    # Verified pyagentspec round-trip gap: a Property serialized via
    # to_dict() (Flow.to_dict()/AgentSpecSerializer) does not carry a
    # component_type discriminator, so AgentSpecDeserializer.from_dict()
    # cannot resolve the concrete subclass (StringProperty etc.) and hands
    # back a bare `Property` instead. `prop.json_schema` DOES survive
    # (it's the original JSON-Schema dict) -- reuse `_resolve_field_type`
    # (the JSON-Schema-dict twin of this exact function) rather than a
    # second walker, per the Core Invariant.
    if type(prop) is pas.Property and prop.json_schema:
        return _resolve_field_type(prop.json_schema), ...

    raise ConfigurationError.build(
        f"Agent Spec Property type {type(prop).__name__} has no neograph type representation",
        expected="one of StringProperty/IntegerProperty/NumberProperty/BooleanProperty/"
        "ListProperty/DictProperty/ObjectProperty/UnionProperty/NullProperty",
        found=type(prop).__name__,
        hint="this Property subclass is NO-REPR for the current bridge -- extend "
        "_property_to_field_type in spec_types.py",
    )


def agent_spec_properties_to_types(properties: list[Property], name: str) -> None:
    """Register a Pydantic model built from a list of Agent Spec ``Property`` objects.

    Import direction of the neograph-nkjv9 bridge: each ``Property``'s
    ``.title`` becomes a field name; the property's own shape (primitive,
    list, dict, nested object, union) is walked via ``_property_to_field_type``
    (the Property-object twin of ``_resolve_field_type``) and the resulting
    model is registered under *name* via the same ``register_type`` every
    other registration path uses.
    """
    fields: dict[str, Any] = {}
    for prop in properties:
        field_type, field_default = _property_to_field_type(prop)
        fields[prop.title] = (field_type, field_default)

    # from_attributes: see the identical rationale in _property_to_field_type's
    # ObjectProperty branch -- this top-level model has no identity link back
    # to the original class either, and runtime state crossing a dispatched
    # node boundary is a real instance of that original class.
    model = create_model(name, __base__=BaseModel, __config__=ConfigDict(from_attributes=True), **fields)
    register_type(name, model)


def _annotation_to_property(annotation: Any, schema: dict[str, Any], defs: dict[str, Any], title: str) -> Property:
    """Build a single Agent Spec ``Property`` from a Pydantic field's JSON Schema.

    Export-side twin of ``_property_to_field_type``: reuses Pydantic's own
    ``model_json_schema()`` output (never a hand-rolled annotation walker,
    per the Core Invariant) and adapts it into the corresponding ``Property``
    subclass, resolving ``$ref``/``$defs`` pointers and ``anyOf``/Optional
    the same way ``_resolve_field_type`` does on the import side.
    """
    del annotation  # the JSON-Schema dict (already resolved by Pydantic) drives the mapping
    pas = _import_agent_spec_property_classes()

    ref = schema.get("$ref")
    if ref and ref.startswith(_REF_POINTER_PREFIX):
        def_name = ref.removeprefix(_REF_POINTER_PREFIX)
        return _annotation_to_property(None, defs[def_name], defs, title=defs[def_name].get("title", title))

    any_of = schema.get("anyOf")
    if any_of is not None:
        members = [
            _annotation_to_property(None, member, defs, title=title)
            for member in any_of
        ]
        return pas.UnionProperty(any_of=members, title=title)

    json_type = schema.get("type")

    if json_type == "null":
        return pas.NullProperty(title=title)

    if json_type == "array":
        item_schema = schema.get("items", {})
        item_property = _annotation_to_property(None, item_schema, defs, title=title)
        return pas.ListProperty(item_type=item_property, title=title)

    if json_type == "object":
        _no_repr_check(schema)
        properties = schema.get("properties")
        if properties:
            child_properties = {
                field_name: _annotation_to_property(None, field_schema, defs, title=field_name)
                for field_name, field_schema in properties.items()
            }
            return pas.ObjectProperty(properties=child_properties, title=title)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            value_property = _annotation_to_property(None, additional, defs, title=title)
            return pas.DictProperty(value_type=value_property, title=title)
        return pas.ObjectProperty(properties={}, title=title)

    if json_type == "string":
        return pas.StringProperty(title=title)
    if json_type == "integer":
        return pas.IntegerProperty(title=title)
    if json_type == "number":
        return pas.NumberProperty(title=title)
    if json_type == "boolean":
        return pas.BooleanProperty(title=title)

    _no_repr_check(schema)
    raise ConfigurationError.build(
        f"JSON Schema shape for field {title!r} has no Agent Spec Property equivalent",
        expected="a DIRECT-tier shape (primitive, list, dict, object, anyOf/Optional)",
        found=f"schema: {schema!r}",
        hint="this field is NO-REPR for Agent Spec export -- fail loud rather than "
        "silently downgrading (deferred: metadata['neograph/original_type'] marker, neograph-i3zsh)",
    )


def model_to_agent_spec_properties(model: type[BaseModel]) -> list[Property]:
    """Export a Pydantic model's fields as a list of Agent Spec ``Property`` objects.

    Export direction of the neograph-nkjv9 bridge: reuses Pydantic's own
    ``model_json_schema()`` (never a hand-walked ``model_fields`` traversal,
    per the Core Invariant) to get a JSON-Schema dict + ``$defs``, then adapts
    it into ``Property`` subclasses via ``_annotation_to_property``. NO-REPR
    fields (tuple/Literal/Enum) fail loud rather than silently downgrading;
    full downgrade-with-marker machinery is deferred to the dedicated
    export epic.
    """
    full_schema = model.model_json_schema()
    defs = full_schema.get("$defs", {})
    properties = full_schema.get("properties", {})

    return [
        _annotation_to_property(None, field_schema, defs, title=field_name)
        for field_name, field_schema in properties.items()
    ]
