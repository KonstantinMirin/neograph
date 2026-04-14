"""Tests for the type registry (spec_types module)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph.errors import ConfigurationError
from neograph.spec_types import (
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

    def test_register_skips_when_same_fields(self):
        class V1(BaseModel):
            x: int

        class V1Copy(BaseModel):
            x: int

        register_type("Idem", V1)
        register_type("Idem", V1Copy)
        # First registration wins — no overwrite
        assert lookup_type("Idem") is V1

    def test_register_warns_and_overwrites_when_fields_differ(self, capsys):
        class V1(BaseModel):
            x: int

        class V2(BaseModel):
            y: str

        register_type("Thing", V1)
        register_type("Thing", V2)

        assert lookup_type("Thing") is V2
        captured = capsys.readouterr()
        assert "overwriting type" in captured.out


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

    def test_required_field_in_properties_works(self):
        config = {
            "types": {
                "Simple": {
                    "type": "object",
                    "required": ["x"],
                    "properties": {
                        "x": {"type": "string"},
                    },
                }
            }
        }
        load_project_types(config)
        Simple = lookup_type("Simple")
        assert Simple(x="hi").x == "hi"

    def test_required_field_not_in_properties_raises(self):
        config = {
            "types": {
                "Bad": {
                    "type": "object",
                    "required": ["x", "y"],
                    "properties": {
                        "x": {"type": "string"},
                    },
                }
            }
        }
        with pytest.raises(ConfigurationError, match="y"):
            load_project_types(config)

    def test_empty_required_list_works(self):
        config = {
            "types": {
                "AllOptional": {
                    "type": "object",
                    "required": [],
                    "properties": {
                        "x": {"type": "string"},
                    },
                }
            }
        }
        load_project_types(config)
        AllOptional = lookup_type("AllOptional")
        assert AllOptional().x is None

    def test_no_required_key_works(self):
        config = {
            "types": {
                "NoReq": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "string"},
                    },
                }
            }
        }
        load_project_types(config)
        NoReq = lookup_type("NoReq")
        assert NoReq().x is None

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

    def test_load_project_types_twice_same_spec_is_idempotent(self, capsys):
        config = {
            "types": {
                "Ping": {
                    "type": "object",
                    "required": ["msg"],
                    "properties": {"msg": {"type": "string"}},
                }
            }
        }
        load_project_types(config)
        load_project_types(config)

        captured = capsys.readouterr()
        # No warning emitted — schemas match
        assert "overwriting" not in captured.out

    def test_load_project_types_twice_different_spec_warns(self, capsys):
        config_v1 = {
            "types": {
                "Pong": {
                    "type": "object",
                    "required": ["a"],
                    "properties": {"a": {"type": "string"}},
                }
            }
        }
        config_v2 = {
            "types": {
                "Pong": {
                    "type": "object",
                    "required": ["b"],
                    "properties": {"b": {"type": "integer"}},
                }
            }
        }
        load_project_types(config_v1)
        load_project_types(config_v2)

        captured = capsys.readouterr()
        assert "overwriting" in captured.out
        # New schema is active
        Pong = lookup_type("Pong")
        assert "b" in Pong.model_fields


class TestResolveFieldTypeUnknownSchema:
    """Line 94: unknown JSON schema type falls back to Any."""

    def test_unknown_json_schema_returns_any(self):
        """A field with no recognized type/ref falls back to Any."""
        from typing import Any

        from neograph.spec_types import _resolve_field_type

        result = _resolve_field_type({"description": "some field"})
        assert result is Any

    def test_ref_field_type_resolves(self):
        """A field with $ref resolves to a registered type."""
        from neograph.spec_types import _resolve_field_type

        class Inner(BaseModel):
            x: str

        register_type("Inner", Inner)
        result = _resolve_field_type({"$ref": "Inner"})
        assert result is Inner
