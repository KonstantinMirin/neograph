# Agent Spec → neograph mapping: Serialization / Deserialization

## Source

Oracle Open Agent Spec v26.1.2 API documentation: "Serialization / Deserialization"
https://oracle.github.io/agent-spec/26.1.2/api/serialization.html

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|---|---|---|---|---|---|
| `AgentSpecSerializer` | Main entry point for serializing Components to dict/JSON/YAML | N/A | **No equivalent yet** — export is greenfield. `loader.py` is import-only. | N/A | **GAP** — to_agent_spec() will be the neograph analog |
| `AgentSpecDeserializer` | Main entry point for deserializing dict/JSON/YAML to Components | N/A | `loader.load_spec()` — YAML/dict → `Spec` → `_build_construct` → `Construct` | N/A | **PARTIAL** — neograph deserializes to its own `Spec`, not Agent Spec Components |
| `SerializationContext` | Interface for field-level serialization (dump_field) | N/A | No equivalent — neograph's neograph `Spec` models serialize via Pydantic `.model_dump()` | N/A | N/A — Agent Spec detail |
| `DeserializationContext` | Interface for field-level deserialization (load_field) | N/A | No equivalent — neograph's `Spec` models deserialize via Pydantic `.model_validate()` | N/A | N/A — Agent Spec detail |
| `ComponentSerializationPlugin` | Plugin base for custom component type serialization | N/A | **Potential Oracle plugin** — emit `ParallelFlowNode` + merge with metadata markers | N/A | **FUTURE** — could handle modifier lowering |
| `PydanticComponentSerializationPlugin` | Pydantic model serialization plugin | N/A | Not needed — neograph's `Spec` already Pydantic | N/A | N/A — neograph models are Pydantic-native |
| `ComponentDeserializationPlugin` | Plugin base for custom component type deserialization | N/A | **Potential Oracle plugin** — read `ParallelFlowNode` group + reconstruct `Oracle` | N/A | **FUTURE** — could handle modifier reconstruction from markers |
| `PydanticComponentDeserializationPlugin` | Pydantic model deserialization plugin | N/A | Not needed — neograph's `Spec` already Pydantic | N/A | N/A — neograph models are Pydantic-native |
| `PyAgentSpecErrorDetails` | Validation error details with type/msg/loc | N/A | `ConfigurationError` (raised from `loader._validate_spec`) | N/A | **DIFFERENT** — neograph uses structured errors, not error-details lists |

## Status legend used

- **GAP**: No neograph equivalent; must be built for interop
- **PARTIAL**: Concept exists but shape/format differs
- **FUTURE**: Optional enhancement for better fidelity
- N/A: Implementation detail of Agent Spec, not applicable to neograph

## Serialization notes

### Agent Spec Serializer pattern

`AgentSpecSerializer` has three output modes:
- `to_dict()` → returns `ComponentAsDictT` (Python dict)
- `to_json()` → returns JSON string
- `to_yaml()` → returns YAML string

All three support **disaggregation** (`disaggregated_components`, `export_disaggregated_components`):
- Extracts components by ID into a separate `$referenced_components` dict
- Main spec references components by ID string
- Enables component reuse and cleaner diffs

### Agent Spec Deserializer pattern

`AgentSpecDeserializer` has matching input modes:
- `from_dict()` → accepts `ComponentAsDictT`
- `from_json()` → accepts JSON string
- `from_yaml()` → accepts YAML string

All three support **component registry** (`components_registry`, `import_only_referenced_components`):
- Load disaggregated components first into a registry
- Pass registry when deserializing main component
- Mirrors the serialize-side disaggregation

### Plugin system extensibility

The `ComponentSerializationPlugin` / `ComponentDeserializationPlugin` system is how Agent Spec extends (de)serialization for custom component types:
- Plugin declares `supported_component_types()` (list of string type names)
- Plugin implements `serialize()` / `deserialize()` with Context
- `PydanticComponentSerializationPlugin` is the built-in for Pydantic models
- **This is the mechanism for handling neograph modifiers** (Oracle, Each, Loop, Operator)

## Export lowering

### neograph → Agent Spec (NEW)

`to_agent_spec(construct: Construct) -> Flow` (free function in new `_agent_spec.py`):

1. **Walk the IR** via `iter_nodes()` / `iter_with_arms()` from `construct.py`
2. **Lower modifiers to primitives**:
   - `node | Each(...)` → `MapNode` / `ParallelMapNode` wrapping extracted sub-Flow
   - `node | Oracle(...)` → `ParallelFlowNode` of N `LlmNode` + merge `LlmNode`/`ToolNode`
   - `node | Loop(...)` → `BranchingNode` with cyclic edges
   - `Operator(...)` → `BranchingNode` or gated edge (TBD)
3. **Stamp metadata markers** (`metadata["neograph/..."]`) on lowered groups
4. **Embed original neograph Spec** in `Flow.metadata["neograph/source"]` (§6a lossless round-trip)
5. **Reject/placeholder callable fields**: `raw_fn`, `skip_when`, `skip_value`, merge hooks, `renderer`
6. **Handle `_BranchNode` sentinels** from `ForwardConstruct` → `BranchingNode`

### Callable field handling

Per design §6, callable-valued IR fields do NOT serialize:
- `raw_fn` → reject (cannot round-trip through name registry like `scripted_fn`)
- `skip_when` / `skip_value` → reject or placeholder
- `Oracle.merge_pre_process` / `merge_post_process` / `merge_fallback` → reject
- `Loop.when` (as callable) → reject (string conditions already serialize)
- `renderer` → reject (renderer selection is runtime)

### Disaggregation opportunity

The Agent Spec disaggregation mechanism could be useful for:
- Shared `LlmConfig` objects (per-node vs construct-level)
- Tool definitions (`ToolBox` / `MCPToolBox` references)
- Type schemas (Property definitions in a `$types` section)

## Import reconstruction

### Agent Spec → neograph (NEW)

`from_agent_spec(flow: Flow | dict) -> Construct` (in `loader.py`):

1. **Deserialize Agent Spec** via `AgentSpecDeserializer.from_dict()`
2. **Check for `neograph/source` metadata** → if present, deserialize embedded neograph Spec
3. **Verify primitive-to-marker alignment**:
   - If `neograph/source` present → re-lower it, diff against actual Flow
   - Match → use rich source (lossless)
   - Diverged → fall back to primitive-level import, warn about broken modifier groups
4. **Reconstruct from markers** (only for aligned groups):
   - `metadata["neograph/modifier"]="oracle"` + `group_id` → reconstruct `Oracle`
   - `metadata["neograph/modifier"]="each"` → reconstruct `Each`
   - `metadata["neograph/modifier"]="loop"` → reconstruct `Loop`
5. **Build primitive-level Construct** via `_build_construct` patterns
6. **Resolve types** via `spec_types.py` registry (Property JSON Schema → Pydantic)

### Primitive-only import baseline

Even without metadata markers, `from_agent_spec()` must succeed:
- Agent Spec `LlmNode` → neograph `Node(mode="think")`
- Agent Spec `Agent` / `AgentNode` → neograph `Node(mode="agent")` + tools
- Agent Spec `ToolNode` / `ApiNode` → neograph `Node(mode="scripted")` (behavior name-bound)
- Agent Spec `MapNode` / `ParallelMapNode` → reconstruct as plain nodes (Each lost)
- Agent Spec `ParallelFlowNode` → reconstruct as plain sub-construct (Oracle lost)
- Agent Spec `BranchingNode` → reconstruct as plain node (Loop/Operator lost)

This is the **guaranteed baseline** — import never fails, just loses sugar.

## Verdict for interop

**Fidelity impact**: Export flattens modifiers → behavior preserved, abstraction lost. Import primitives-only → same. Metadata markers enable lossless round-trip for neograph-authored specs.

**Single biggest risk**: The **metadata round-trip guarantee**. The design §6a lossless-embed strategy assumes Agent Spec's `Component.metadata` survives serialize→deserialize. The Agent Spec docs confirm `metadata` is an optional `dict[str, Any]` on every Component, and the docs show metadata is threaded through `to_dict` / `from_dict`. **BUT we must verify this** — is there any code path that drops metadata? Do disaggregation / reaggregation preserve metadata? The epic gate task (neograph-swcy1) must include a test that asserts `metadata["neograph/source"]` survives a full `AgentSpecSerializer.to_dict` → `AgentSpecDeserializer.from_dict` round-trip.

**Secondary risk**: Plugin system complexity. Implementing Oracle/Each plugins for modifier reconstruction may be over-engineering. A simpler approach: reconstruct modifiers by pattern-matching on the primitive structure (N parallel LlmNodes + merge = Oracle) rather than building a full plugin system. The metadata markers provide enough signal; plugins are unnecessary indirection.
