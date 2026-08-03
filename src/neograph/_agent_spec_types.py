"""Agent Spec ``Property`` <-> neograph type bridge.

Extracted from ``spec_types.py`` (neograph-s7zt3.16): that module had reached
its 500-line ceiling holding two clusters that are conceptually separate but
were implementationally entangled -- the JSON-Schema type REGISTRY (which
``load_spec`` and the compiler use, and which has no Agent Spec involvement),
and this pyagentspec BRIDGE (which only exists when the optional
``[agent-spec]`` extra is installed).

The seam is strictly one-way and was verified from the AST, not by eye: the
registry half references nothing here, and this half needs exactly four names
from it (``_no_repr_check``, ``_REF_POINTER_PREFIX``, ``lookup_type``,
``register_type``).

That seam only became clean as part of s7zt3.16. Before it, this half also
called ``_resolve_field_type`` -- the registry's JSON-Schema walker -- through
an erased-``Property`` fallback. Removing that fallback (it was minting
unregistered ad-hoc classes and causing the very identity mismatch the ticket
fixed) severed the last tie, which is why the split is possible now and was not
before.

``src/neograph`` stays Agent-Spec-free at import time: pyagentspec is pulled in
only by ``_import_agent_spec_property_classes()``, a function-local import, and
the ``Property`` annotation below is ``TYPE_CHECKING``-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, create_model

from neograph.errors import ConfigurationError
from neograph.spec_types import _REF_POINTER_PREFIX, _no_repr_check, lookup_type, register_type

if TYPE_CHECKING:
    from pyagentspec.property import Property


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


def _normalize_erased_property(prop: Property) -> Property:
    """Re-inflate a ``Property`` whose SUBCLASS a serialization round trip erased.

    Verified pyagentspec round-trip gap: a ``Property`` serialized via
    ``to_dict()`` (``Flow.to_dict()`` / ``AgentSpecSerializer``) carries no
    ``component_type`` discriminator, so ``AgentSpecDeserializer.from_dict()``
    cannot resolve the concrete subclass and hands back a BARE ``Property``.
    ``json_schema`` DOES survive verbatim -- only the class is lost.

    Both bridge dispatchers below branch on the CLASS, so an erased property
    misses every branch. This is the ONE seam that repairs that, called at the
    head of each dispatcher: re-inflate from the surviving schema, then let the
    existing class dispatch run completely unchanged (neograph-s7zt3.16).

    Reuses ``_annotation_to_property`` -- already schema-driven, its
    ``annotation`` parameter is vestigial -- rather than adding a second
    schema-to-Property walker. Two walkers for one job is what caused the bug:
    the removed fallback resolved erased properties through
    ``_resolve_field_type``, whose object branch mints an ad-hoc
    ``create_model(title)`` class instead of the canonical
    ``_structural_type_name`` + ``register_type`` one, so the same shape
    reconstructed to two NON-IDENTICAL classes depending on whether the Flow
    had been through JSON.

    Passes through unchanged (both are legitimate, neither is a silent skip):
    a property that still has its subclass, and a bare property with no
    ``json_schema`` -- an ``Each`` node's ``iterated_*`` inputs are bare with
    ``type=None`` even in memory, and those must still reach
    ``_property_to_field_type``'s fail-loud raise rather than be guessed at.
    """
    pas = _import_agent_spec_property_classes()

    if type(prop) is not pas.Property:
        return prop
    schema = getattr(prop, "json_schema", None)
    if not schema:
        return prop
    # defs={} is correct, not a shortcut: pyagentspec emits FULLY INLINED
    # json_schema on a Property -- verified against a repeated-nested-model
    # shape (the one Pydantic hoists into $defs) and against union/Optional.
    # A $ref therefore cannot appear here.
    return _annotation_to_property(None, schema, {}, title=prop.title)


def _property_type_signature(prop: Property) -> Any:
    """Recursive structural signature for a single Property's TYPE.

    ``Property.type`` carries only the bare top-level JSON-schema keyword
    (e.g. ``'array'``), never the nested item/value/field schema -- so two
    structurally DIFFERENT properties (``list[str]`` vs ``list[SomeModel]``)
    both signature to the same bare ``'array'`` if the recursion stops
    there. This walks into ``ListProperty.item_type``,
    ``DictProperty.value_type``, ``ObjectProperty.properties``, and
    ``UnionProperty.any_of`` so the signature is genuinely structural at
    every depth, not just the top level. Used exclusively by
    ``_structural_type_name`` -- keep the two in lockstep.

    Normalizes first: a serialization round trip erases the subclass this
    function dispatches on, and an erased property would otherwise signature to
    its bare top-level keyword ('array'), discarding the element schema
    entirely (neograph-s7zt3.16).
    """
    pas = _import_agent_spec_property_classes()
    prop = _normalize_erased_property(prop)

    if isinstance(prop, pas.ListProperty):
        return ("array", _property_type_signature(prop.item_type))
    if isinstance(prop, pas.DictProperty):
        return ("object_map", _property_type_signature(prop.value_type))
    if isinstance(prop, pas.ObjectProperty):
        return (
            "object",
            tuple(sorted((name, _property_type_signature(p)) for name, p in prop.properties.items())),
        )
    if isinstance(prop, pas.UnionProperty):
        # key=repr, not a bare sorted(): member signatures are heterogeneous --
        # a primitive signatures to a str, a container to a tuple -- and Python
        # cannot order str against tuple, so a bare sorted() raises TypeError on
        # any union mixing the two. `Tagged | None` (ObjectProperty + NullProperty)
        # is the common case and crashed here even WITHOUT serialization. The sort
        # exists only to make member ORDER irrelevant to the hash, so any total
        # order will do (neograph-s7zt3.16).
        return ("union", tuple(sorted((_property_type_signature(m) for m in prop.any_of), key=repr)))
    return str(getattr(prop, "type", None))


def _structural_type_name(props: list[Property]) -> str:
    """Derive a registry name purely from a Property list's STRUCTURE
    (title + recursive type signature, sorted), not from any node/model
    name.

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

    The type half of the signature recurses via ``_property_type_signature``
    -- a bare top-level type keyword (e.g. ``'array'``) is not enough to
    distinguish ``list[str]`` from ``list[SomeModel]`` when both share a
    field name (neograph-qtfof.4).
    """
    import hashlib

    sig = tuple(sorted((p.title, _property_type_signature(p)) for p in props))
    digest = hashlib.sha256(repr(sig).encode()).hexdigest()[:16]
    return f"AgentSpecType_{digest}"


def _property_to_field_type(prop: Property) -> tuple[Any, Any]:
    """Map a single Agent Spec ``Property`` to a (type, default) field spec.

    Reuses the SAME primitive-type map (``_JSON_SCHEMA_TYPE_MAP``) and the
    same NO-REPR fail-loud discipline as ``_resolve_field_type`` -- this is
    the Property-object-shaped twin of that function, not a parallel
    walker: both ultimately produce a type via the same rules, just reading
    from a live ``Property`` tree instead of a raw JSON-Schema dict.

    Normalizes first, so an erased property is re-inflated and then dispatched
    like any other rather than falling through to a second walker
    (neograph-s7zt3.16).
    """
    pas = _import_agent_spec_property_classes()
    prop = _normalize_erased_property(prop)

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

    # NOTE: the erased-Property case is handled by _normalize_erased_property at
    # the head of this function, NOT by a fallback here. The fallback that used
    # to sit at this spot resolved erased properties through _resolve_field_type,
    # whose object branch mints an unregistered create_model(title) class -- so
    # the same shape reconstructed to two NON-IDENTICAL classes depending on
    # whether the Flow had been through JSON, and identity checks failed
    # (neograph-s7zt3.16). Do not reintroduce it; a bare Property reaching here
    # now genuinely has no representation and must fail loud.
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
        # The definition supplies the SHAPE only; the title stays the FIELD's.
        # Preferring the definition's title collapsed two fields of the same
        # nested type into one -- both exported as the model's name, and the
        # import side uses Property.title as the field name.
        def_name = ref.removeprefix(_REF_POINTER_PREFIX)
        return _annotation_to_property(None, defs[def_name], defs, title=title)

    any_of = schema.get("anyOf")
    if any_of is not None:
        members = [_annotation_to_property(None, member, defs, title=title) for member in any_of]
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
