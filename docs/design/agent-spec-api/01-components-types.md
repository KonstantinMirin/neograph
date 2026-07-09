# Agent Spec → neograph mapping: Components & IO Properties

## Source

- Agent Spec Components: https://oracle.github.io/agent-spec/26.1.2/api/components.html
- Agent Spec Development Docs: https://oracle.github.io/agent-spec/development/agentspec/index.html
- PyAgentSpec Property Classes (DeepWiki): https://deepwiki.com/search/what-are-the-property-classes_b6e9f01e-eaae-4814-bcf8-d212dde94aba

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|---|---|---|---|---|---|
| `Component` | Base class for all components with id, name, description, metadata | N/A | N/A (structural base) | All | RECONSTRUCT |
| `ComponentWithIO` | Extends Component with inputs/outputs (List[Property]) | N/A | N/A (structural base) | All | RECONSTRUCT |
| `AgenticComponent` | Extends ComponentWithIO for interactive components | N/A | N/A (structural base) | All | RECONSTRUCT |
| `Property` | JSON Schema container (title, type, default, description, json_schema) | TypedDict field | Pydantic BaseModel field | All | DIRECT |
| `StringProperty` | String type | TypedDict: str | Pydantic field with str annotation | All | DIRECT |
| `IntegerProperty` | Integer type | TypedDict: int | Pydantic field with int annotation | All | DIRECT |
| `FloatProperty` | Float/number type (inherits NumberProperty) | TypedDict: float | Pydantic field with float annotation | All | DIRECT |
| `BooleanProperty` | Boolean type | TypedDict: bool | Pydantic field with bool annotation | All | DIRECT |
| `NullProperty` | Null type | TypedDict: None | Optional[T] / T \| None | All | RECONSTRUCT |
| `UnionProperty` | anyOf union of Property types | TypedDict: Union (not natively JSON Schema) | Union[A, B] / A \| B | All | RECONSTRUCT |
| `ListProperty` | Array with item_type Property | TypedDict: list | list[T] | All | DIRECT |
| `DictProperty` | Object with additionalProperties (single value_type) | TypedDict: dict | dict[str, T] (homogeneous) | All | DIRECT |
| `ObjectProperty` | Object with named properties dict | TypedDict: dict | BaseModel / dict[str, T] (heterogeneous) | All | LOWER |

## Status legend used

- **DIRECT** — 1:1 mapping, no transformation needed
- **LOWER** — neograph concept is more expressive; export lowers to Agent Spec primitive
- **RECONSTRUCT** — Import reconstructs neograph concept from Agent Spec primitive with potential fidelity loss
- **GAP-AS** — Agent Spec has it, neograph lacks it (addressable via metadata marker)
- **GAP-NG** — neograph has it, Agent Spec lacks it (no representation, needs metadata marker)
- **NO-REPR** — Cannot be represented in Agent Spec format (blocker, requires metadata marker)

## Serialization notes

### Agent Spec side

All Property classes serialize cleanly to JSON/YAML via `to_dict()`/`to_yaml()`/`to_json()`. The `json_schema` field carries the JSON Schema definition, and standard Pydantic serialization handles the rest. Properties are nested in `ComponentWithIO.inputs` and `ComponentWithIO.outputs` lists.

### neograph side

**Current state (import-only):**
- `loader.py` reads YAML/JSON into `Spec` (Pydantic)
- `spec_types.py` provides `_type_registry` (name → Pydantic model)
- `_resolve_field_type()` converts JSON Schema to Python types (handles array, $ref, primitives)
- No export path exists — this is greenfield

**Type representation gaps:**
- `describe_type.py` renders Pydantic as TypeScript-style notation (LLM-facing), not JSON Schema
- Pydantic models carry full field metadata (descriptions, constraints) that flatten to JSON Schema on export
- `tuple[T, ...]` (variadic) and heterogeneous `tuple[A, B, C]` have no Agent Spec Property equivalent → NO-REPR
- `Optional[T]` maps to UnionProperty(anyOf=[TypeProperty, NullProperty]) on export; reconstructible as `T | None`
- `Literal` values have no direct Property equivalent → NO-REPR (requires metadata marker or downgrade to string)
- Constrained fields (`Field(gt=0, ge=10)`) serialize into JSON Schema constraints, Agent Spec Property can represent via `json_schema` extensions
- `Enum` classes have no Property equivalent → NO-REPR (downgrade to string or embed literal values in metadata)
- `Annotated[T, ...]` metadata (beyond type) requires metadata marker to preserve

### Round-trip considerations

**What's preserved on round-trip:**
- Primitive types (str, int, float, bool) — perfect fidelity
- list[T] ↔ ListProperty — perfect fidelity
- dict[str, T] ↔ DictProperty — perfect fidelity
- BaseModel ↔ ObjectProperty — perfect fidelity (field order may vary, semantically identical)
- Optional[T] ↔ UnionProperty(anyOf=[T, null]) — reconstructible

**What's lost on neograph → Agent Spec export:**
- `tuple` types (both heterogeneous and variadic) → NO-REPR, must error or downgrade to list
- `Literal` types → NO-REPR, must error or downgrade to string
- `Enum` types → NO-REPR, must error or downgrade to string
- `Annotated` metadata beyond ExcludeFromOutput → NO-REPR unless carried in metadata marker
- Field-level Pydantic constraints serialize to JSON Schema but Property class may not expose all keywords

**What's lost on Agent Spec → neograph import:**
- `Property.default` values survive JSON Schema round-trip but neograph must reconstruct field defaults
- `Property.description` survives but maps to Pydantic Field description
- Heterogeneous dict types (not `dict[str, T]`) → Agent Spec can't express, import yields dict[str, Any]
- Agent Spec `NumberProperty` (generic number) → neograph must choose float or int (defaults to float)

## Export lowering

When exporting a neograph Pydantic BaseModel to Agent Spec Properties:

```python
# Single primitive field
class Foo(BaseModel):
    name: str  # → StringProperty(title="name", type="string")
    count: int  # → IntegerProperty(title="count", type="integer")

# List field
items: list[Item]  # → ListProperty(title="items", item_type=ObjectProperty(...))

# Dict field
tags: dict[str, str]  # → DictProperty(title="tags", value_type=StringProperty())

# BaseModel field
nested: NestedModel  # → ObjectProperty(title="nested", properties={...})

# Optional field
optional: str | None  # → UnionProperty(any_of=[StringProperty(), NullProperty()])
```

**Generator contract:**
```python
def pydantic_to_property(model: type[BaseModel]) -> List[Property]:
    """Convert a Pydantic model to Agent Spec Property list."""
    props: List[Property] = []
    for field_name, field_info in model.model_fields.items():
        prop = _annotation_to_property(field_name, field_info.annotation, field_info)
        props.append(prop)
    return props
```

**Type conversion table:**

| neograph annotation | Agent Spec Property |
|---|---|
| `str` | `StringProperty` |
| `int` | `IntegerProperty` |
| `float` | `FloatProperty` |
| `bool` | `BooleanProperty` |
| `list[T]` | `ListProperty(item_type=...)` |
| `dict[str, T]` | `DictProperty(value_type=...)` |
| `BaseModel` | `ObjectProperty(properties={...})` |
| `T \| None` | `UnionProperty(any_of=[..., NullProperty()])` |
| `A \| B` | `UnionProperty(any_of=[...])` |
| `tuple[T, ...]` | NO-REPR (error) |
| `tuple[A, B, ...]` | NO-REPR (error) |
| `Literal[...]` | NO-REPR (error) |
| `Enum` | NO-REPR (error) |

## Import reconstruction

When importing Agent Spec Properties into neograph:

**Generator contract:**
```python
def properties_to_pydantic(props: List[Property], name: str) -> type[BaseModel]:
    """Convert Agent Spec Properties to a Pydantic model."""
    fields: Dict[str, Any] = {}
    for prop in props:
        field_type = _property_to_annotation(prop)
        # Determine if optional based on UnionProperty containing NullProperty
        if isinstance(prop, UnionProperty) and _has_null(prop.any_of):
            if field_type is not None:
                field_type = Optional[field_type]
        fields[prop.title] = (field_type, ...)  # Required; default handling omitted
    return create_model(name, __base__=BaseModel, **fields)
```

**Property conversion table:**

| Agent Spec Property | neograph annotation | Notes |
|---|---|---|
| `StringProperty` | `str` | Direct |
| `IntegerProperty` | `int` | Direct |
| `FloatProperty` / `NumberProperty` | `float` | Generic number → float |
| `BooleanProperty` | `bool` | Direct |
| `NullProperty` | `None` | Rare standalone; usually inside UnionProperty |
| `ListProperty` | `list[T]` | Recurse on `item_type` |
| `DictProperty` | `dict[str, T]` | Recurse on `value_type` |
| `ObjectProperty` | `BaseModel` | Recurse on `properties` |
| `UnionProperty(any_of=[...])` | `Union[*args]` or `T \| None` | Recurse on each `any_of` member |

**Reconstruction losses:**
- Field defaults lost unless explicitly stored in Property.default
- Field constraints beyond type (gt, lt, pattern) may not survive round-trip through Property json_schema
- Heterogeneous BaseModel → ObjectProperty preserves structure; back to BaseModel is faithful

## Verdict for interop

**Fidelity impact:** HIGH for core types (primitives, lists, dicts, BaseModels). The Property ↔ Pydantic mapping is the backbone of the entire round-trip and is well-supported. The primary gaps are tuple types, Literal types, and Enum types, which are neograph expressible but Agent Spec cannot represent without metadata markers.

**Single biggest risk:** **NO-REPR types (tuple, Literal, Enum)**. These are valid neograph I/O annotations that cannot be expressed in Agent Spec Property classes. Export must either reject (fail-loud) or downgrade with metadata marker preservation. The downgrade path (tuple → list, Literal → string, Enum → string) loses semantic information that cannot be recovered on import without the marker. This is a blocker for pipelines that use these types and need full round-trip fidelity.

**Mitigation:** Use `metadata["neograph/original_type"]` on exported Properties to store the full type signature (e.g., `"tuple[str, int, float]"`, `"Literal['a', 'b']"`, `"MyEnum"`) and reconstruct via `eval()` against a safe namespace on import. The `neograph/source` embed (§6a) is the stronger guarantee — downgrade individual Properties only when not using full-source embed.
