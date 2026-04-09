"""Type registry — maps string names to Pydantic models.

Supports two registration paths:

1. Explicit: ``register_type('Draft', Draft)`` for pre-existing Python types.
2. Auto-generated: ``load_project_types(config)`` reads a ``types`` dict of
   JSON Schema definitions and builds Pydantic models via ``create_model``.

The registry is consumed by ``load_spec`` (and future spec loaders) to
resolve string type references in YAML/JSON pipeline definitions into
concrete Pydantic classes.
"""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import BaseModel, create_model

from neograph.errors import ConfigurationError

log = structlog.get_logger(__name__)

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
        raise ConfigurationError(
            f"Type {name!r} is not registered. "
            f"Use register_type() or include it in project types."
        ) from None


def _resolve_field_type(schema: dict[str, Any]) -> Any:
    """Map a single JSON Schema field definition to a Python type annotation."""
    json_type = schema.get("type")

    if json_type == "array":
        items = schema.get("items", {})
        inner = _resolve_field_type(items)
        return list[inner]

    if json_type in _JSON_SCHEMA_TYPE_MAP:
        return _JSON_SCHEMA_TYPE_MAP[json_type]

    # No explicit type — check for a $ref-style name (nested type reference)
    ref = schema.get("$ref")
    if ref:
        # Expect a bare type name (not a JSON pointer)
        return lookup_type(ref)

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
            raise ConfigurationError(
                f"Type '{type_name}': required fields {sorted(unknown_required)} "
                f"not found in properties. Available: {sorted(properties.keys())}"
            )

        for field_name, field_schema in properties.items():
            field_type = _resolve_field_type(field_schema)
            if field_name in required_fields:
                fields[field_name] = (field_type, ...)
            else:
                fields[field_name] = (field_type, None)

        model = create_model(type_name, __base__=BaseModel, **fields)
        register_type(type_name, model)
