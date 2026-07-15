"""Regression tests for the Agent Spec ``Property`` (JSON Schema) <-> neograph
``spec_types`` registry bidirectional bridge (neograph-nkjv9).

Gated on ``pyagentspec`` via ``pytest.importorskip`` -- the ``[agent-spec]``
optional extra keeps ``src/neograph`` core dependency-light by default. Run
with::

    uv run --extra dev --extra agent-spec pytest tests/test_agent_spec_types.py

## $ref decision this test pins (design step 3, option (a))

The existing ``_resolve_field_type`` ``$ref`` branch (``spec_types.py:88-91``)
resolves ONLY bare type names via ``lookup_type``. Pydantic's
``model_json_schema()`` emits ``#/$defs/Name`` JSON-pointer refs for nested
``BaseModel`` fields, and per the ratified design doc pyagentspec
``Property.json_schema`` carries the same JSON Schema shape. This test pins
**option (a)**: ``_resolve_field_type`` grows an optional ``defs`` parameter
and resolves ``#/$defs/Name`` pointers against that map directly, rather than
requiring the export adapter to fully inline every nested model before it
ever reaches the import walker. This is the choice consistent with the Core
Invariant (extend the ONE existing walker in place) -- reusing Pydantic's own
``model_json_schema()`` for export (step 6) naturally produces ``$defs`` +
pointer refs for nested models, and hand-inlining them in the exporter would
mean re-implementing Pydantic's own schema walker, which step 6 explicitly
forbids ("reuse Pydantic's own schema generator, do not hand-walk
model_fields").

## Function names assumed by these tests (not yet fixed by the design)

Design step 5/6 leave exact naming to the implementer. These tests assume:

- ``neograph.spec_types.model_to_agent_spec_properties(model) -> list[Property]``
  (export: Pydantic BaseModel -> Agent Spec Property list)
- ``neograph.spec_types.agent_spec_properties_to_types(properties, name) -> None``
  (import: Agent Spec Property list -> registers a Pydantic model via
  ``register_type``, mirroring the step-5 contract's return-None side effect)
- ``neograph.spec_types._resolve_field_type(schema, defs=None)`` (extended
  signature -- current signature is ``_resolve_field_type(schema)`` with no
  ``defs`` param)

If the implementer picks different names, update the imports below -- the
*behavior* pinned (round-trip fidelity for nested/list/dict/Optional fields,
and pointer-ref resolution against a defs map) is what must survive.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyagentspec")

from pydantic import BaseModel  # noqa: E402

from neograph.spec_types import lookup_type, register_type  # noqa: E402

# -- throwaway nested models used across both directions ---------------------


class _Address(BaseModel):
    city: str
    zip_code: str | None = None


class _Contact(BaseModel):
    name: str
    tags: list[str]
    addresses: list[_Address]
    metadata: dict[str, str]
    nickname: str | None = None


class TestPydanticToAgentSpecPropertyRoundTrip:
    """Pydantic BaseModel -> Agent Spec Property list -> back to Pydantic."""

    def test_nested_list_dict_optional_model_round_trips_through_properties(self):
        from neograph.spec_types import (
            agent_spec_properties_to_types,
            model_to_agent_spec_properties,
        )

        properties = model_to_agent_spec_properties(_Contact)
        agent_spec_properties_to_types(properties, name="ContactRebuilt")
        rebuilt = lookup_type("ContactRebuilt")

        instance = rebuilt(
            name="Ada",
            tags=["eng", "lead"],
            addresses=[{"city": "London", "zip_code": "E1"}],
            metadata={"team": "core"},
            nickname=None,
        )

        assert instance.name == "Ada"
        assert instance.tags == ["eng", "lead"]
        assert instance.addresses[0].city == "London"
        assert instance.addresses[0].zip_code == "E1"
        assert instance.metadata == {"team": "core"}
        assert instance.nickname is None

        # required-ness + Optional-ness preserved through the round trip
        assert rebuilt.model_fields["name"].is_required() is True
        assert rebuilt.model_fields["nickname"].is_required() is False


class TestAgentSpecPropertyJsonRoundTripsThroughRegisteredType:
    """A Property list -> registered Pydantic model -> re-exported Property list.

    Diffed for structural equivalence, simulating receiving a Property list
    from an external Agent Spec source (not one we exported ourselves).

    ## Test-bug correction (verified against real pyagentspec 26.1.2)

    The original version of this test imported ``pyagentspec.serialization
    .Deserializer`` and called ``Property.to_dict()`` to simulate an
    external producer round-tripping Property JSON. Neither exists on the
    real installed package -- verified by reading
    ``pyagentspec/serialization/__init__.py`` (exports
    ``AgentSpecDeserializer``, a *Component*-level deserializer for
    Flow/Node graphs, not bare Property lists) and ``pyagentspec/property.py``
    (``Property`` has NO ``to_dict``/``from_dict``; its ``@model_serializer``
    override makes ``model_dump()`` return ONLY ``self.json_schema``,
    discarding which Property subclass produced it -- so a generic
    "deserialize an arbitrary JSON-Schema dict back into the correct
    Property subclass" operation does not exist in the package at all;
    subclass selection is a design choice for a Property-producer, not
    something pyagentspec resolves for you).

    This is exactly the same category of doc-vs-installed-package drift
    neograph-03djs found and corrected elsewhere -- fixing it here rather
    than leaving an untestable premise in place, per the "document why"
    sacred rule.

    Corrected simulation: an external producer hands over live ``Property``
    objects (that is what pyagentspec actually deals in -- e.g. as parsed
    out of a ``Flow``'s node inputs/outputs), not a bare JSON dict requiring
    a generic deserializer. So "receiving from an external source" is
    simulated by constructing an independent Property tree with the SAME
    shape neograph would produce (not reusing neograph's own export
    function), and diffing on the populated ``.json_schema`` attribute
    (already a live dict on every Property instance -- no serialization
    step needed) rather than a nonexistent ``.to_dict()``.
    """

    def test_property_json_diff_is_structurally_equivalent_after_re_export(self):
        from pyagentspec.property import ListProperty, ObjectProperty, StringProperty

        from neograph.spec_types import (
            agent_spec_properties_to_types,
            model_to_agent_spec_properties,
        )

        # Independently hand-built Property list mirroring _Contact's shape
        # -- simulates an external Agent Spec producer, not neograph's own
        # exporter (which is exactly what's under test).
        externally_received_properties = [
            StringProperty(title="name"),
            ListProperty(item_type=StringProperty(title="tags"), title="tags"),
        ]

        agent_spec_properties_to_types(externally_received_properties, name="SimpleRoundTrip")

        rebuilt_model = lookup_type("SimpleRoundTrip")
        register_type("SimpleRoundTrip", rebuilt_model)  # idempotency check

        re_exported_properties = model_to_agent_spec_properties(rebuilt_model)

        def _structural_keys(properties):
            return {
                (p.title, p.json_schema.get("type"), p.json_schema.get("items", {}).get("type"))
                for p in properties
            }

        original_keys = {
            (p.title, p.json_schema.get("type"), p.json_schema.get("items", {}).get("type"))
            for p in externally_received_properties
        }
        assert _structural_keys(re_exported_properties) == original_keys

        # Also exercise the nested-object shape through the same round trip,
        # confirming ObjectProperty survives independently of the simple case.
        nested_original = [
            ObjectProperty(
                properties={"city": StringProperty(title="city")},
                title="address",
            ),
        ]
        agent_spec_properties_to_types(nested_original, name="NestedRoundTrip")
        nested_rebuilt = lookup_type("NestedRoundTrip")
        nested_re_exported = model_to_agent_spec_properties(nested_rebuilt)

        assert len(nested_re_exported) == 1
        assert nested_re_exported[0].json_schema.get("type") == "object"
        assert set(nested_re_exported[0].json_schema.get("properties", {}).keys()) == {"city"}


class TestRefPointerResolution:
    """Pins the $ref decision: #/$defs/Name pointers resolve via a defs map."""

    def test_resolve_field_type_resolves_json_pointer_ref_against_defs_map(self):
        from neograph.spec_types import _resolve_field_type

        full_schema = _Contact.model_json_schema()
        defs = full_schema.get("$defs", {})
        assert "_Address" in defs, (
            "expected Pydantic to emit a #/$defs/_Address entry for the "
            "nested Address model -- this test's premise (pointer refs "
            "reach _resolve_field_type) depends on it"
        )

        pointer_schema = {"$ref": "#/$defs/_Address"}

        resolved = _resolve_field_type(pointer_schema, defs=defs)

        instance = resolved(city="Paris", zip_code=None)
        assert instance.city == "Paris"
        assert instance.zip_code is None
