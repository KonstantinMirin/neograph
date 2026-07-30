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
from pydantic import BaseModel, create_model

from neograph.errors import ConfigurationError

if TYPE_CHECKING:
    pass

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
    """Return True if both models have the same field names AND field types.

    Comparing only field NAMES (the pre-neograph-wqb5t behaviour) treated two
    structurally DIFFERENT models sharing a name set — e.g.
    ``Bag(items: list[Tagged])`` vs ``Claims(items: list[str])`` — as the same
    schema, so ``register_type`` silently no-oped on the second call under a
    reused EXPLICIT name (loader's caller-supplied path) and kept the first,
    wrong-shaped class. Folding the field annotation (via ``str`` — stable for
    the deterministically-named synthesized types) into the comparison makes a
    genuine type mismatch overwrite/warn instead of silently collapsing."""

    def _sig(m: type[BaseModel]) -> dict[str, str]:
        return {name: str(field.annotation) for name, field in m.model_fields.items()}

    return _sig(a) == _sig(b)


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


# Re-exports: the Agent Spec Property bridge moved to _agent_spec_types.py
# (neograph-s7zt3.16). Kept here so this module's import surface is unchanged --
# loader.py and _agent_spec_placeholders.py import these from neograph.spec_types.
# noqa F401 is REQUIRED: without it ruff --fix strips them as unused and breaks
# test collection.
from neograph._agent_spec_types import (  # noqa: E402,F401
    _annotation_to_property,
    _import_agent_spec_property_classes,
    _normalize_erased_property,
    _property_to_field_type,
    _property_type_signature,
    _structural_type_name,
    agent_spec_properties_to_types,
    model_to_agent_spec_properties,
)
