"""Tests for the type registry (spec_types module)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph.errors import ConfigurationError
from neograph.spec_types import (
    _type_registry,
    load_project_types,
    lookup_type,
    register_type,
)


# -- register + lookup --------------------------------------------------------


class TestRegisterAndLookup:
    """register_type stores models; lookup_type retrieves them."""

    def test_register_and_lookup_round_trip(self):
        class Draft(BaseModel):
            text: str

        register_type("Draft", Draft)
        assert lookup_type("Draft") is Draft

    def test_lookup_raises_configuration_error_when_not_found(self):
        with pytest.raises(ConfigurationError, match="NoSuchType"):
            lookup_type("NoSuchType")

    def test_register_overwrites_previous(self):
        class V1(BaseModel):
            x: int

        class V2(BaseModel):
            y: str

        register_type("Thing", V1)
        register_type("Thing", V2)
        assert lookup_type("Thing") is V2


# -- load_project_types: auto-generate from JSON Schema -----------------------


class TestLoadProjectTypes:
    """load_project_types builds Pydantic models from JSON Schema defs."""

    def test_generates_model_with_primitive_fields(self):
        config = {
            "types": {
                "Claim": {
                    "type": "object",
                    "required": ["text", "confidence"],
                    "properties": {
                        "text": {"type": "string"},
                        "confidence": {"type": "number"},
                        "priority": {"type": "integer"},
                        "verified": {"type": "boolean"},
                    },
                }
            }
        }
        load_project_types(config)
        Claim = lookup_type("Claim")

        assert issubclass(Claim, BaseModel)
        instance = Claim(text="hello", confidence=0.9, priority=1, verified=True)
        assert instance.text == "hello"
        assert instance.confidence == 0.9
        assert instance.priority == 1
        assert instance.verified is True

    def test_optional_fields_default_to_none(self):
        config = {
            "types": {
                "Note": {
                    "type": "object",
                    "required": ["body"],
                    "properties": {
                        "body": {"type": "string"},
                        "tag": {"type": "string"},
                    },
                }
            }
        }
        load_project_types(config)
        Note = lookup_type("Note")
        instance = Note(body="content")
        assert instance.tag is None

    def test_array_field(self):
        config = {
            "types": {
                "TagList": {
                    "type": "object",
                    "required": ["tags"],
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                }
            }
        }
        load_project_types(config)
        TagList = lookup_type("TagList")
        instance = TagList(tags=["a", "b"])
        assert instance.tags == ["a", "b"]

    def test_nested_type_via_ref(self):
        config = {
            "types": {
                "Author": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                    },
                },
                "Article": {
                    "type": "object",
                    "required": ["title", "author"],
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"$ref": "Author"},
                    },
                },
            }
        }
        load_project_types(config)
        Author = lookup_type("Author")
        Article = lookup_type("Article")

        author = Author(name="Alice")
        article = Article(title="Test", author=author)
        assert article.author.name == "Alice"

    def test_empty_types_section_is_noop(self):
        load_project_types({"types": {}})
        # No error, registry unchanged

    def test_missing_types_key_is_noop(self):
        load_project_types({})
        # No error, registry unchanged

    def test_nested_ref_not_found_raises(self):
        config = {
            "types": {
                "Bad": {
                    "type": "object",
                    "required": ["ref"],
                    "properties": {
                        "ref": {"$ref": "DoesNotExist"},
                    },
                }
            }
        }
        with pytest.raises(ConfigurationError, match="DoesNotExist"):
            load_project_types(config)

    def test_array_of_nested_type(self):
        config = {
            "types": {
                "Item": {
                    "type": "object",
                    "required": ["value"],
                    "properties": {
                        "value": {"type": "integer"},
                    },
                },
                "Basket": {
                    "type": "object",
                    "required": ["items"],
                    "properties": {
                        "items": {"type": "array", "items": {"$ref": "Item"}},
                    },
                },
            }
        }
        load_project_types(config)
        Item = lookup_type("Item")
        Basket = lookup_type("Basket")

        basket = Basket(items=[Item(value=1), Item(value=2)])
        assert len(basket.items) == 2
        assert basket.items[0].value == 1
