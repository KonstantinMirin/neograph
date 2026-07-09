# Agent Spec → neograph mapping: Tools

## Source

- **Agent Spec v26.1.2**: [oracle/agent-spec GitHub](https://github.com/oracle/agent-spec) — Python SDK `pyagentspec`
- **Agent Spec Docs**: [oracle.github.io/agent-spec](https://oracle.github.io/agent-spec/)
- **DeepWiki on Tools**: Tool class hierarchy (Tool, ClientTool, RemoteTool, ServerTool, MCPTool)
- **neograph source**:
  - `src/neograph/tool.py` — Tool model + ToolInteraction + resource_reader
  - `src/neograph/_spec_schema.py` — ToolSpec (name, budget, config)
  - `src/neograph/loader.py` — _resolve_tool normalization
  - `src/neograph/node.py` — Node(tools=...) + tool_factories binding
  - `src/neograph/_tool_loop.py` — ReAct tool loop + per-tool budget tracking
  - `src/neograph/modifiers.py` — Each/Oracle tool_budget field

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|------------------|--------------|-------------------|-----------------|-----------|--------|
| **Tool** (base) | Abstract base; inherits `ComponentWithIO` with `inputs`/`outputs` as JSON Schema lists | `BaseTool` (langchain_core) | `Tool` model with `name`/`budget`/`config`/`idempotent`; runtime binds via `tool_factories` | LangChain | **PARTIAL** |
| **ClientTool** | Client executes tool; returns results to agent | Client-side `@tool` or scripted `@node` | Scripted `@tool` decorator / `Node.scripted()` / `@node(mode='scripted')` | Decorator layer | **NATIVE** |
| **RemoteTool** | HTTP API integration via url, http_method, data, query_params, headers with `{{placeholder}}` templating | Scripted `@node` with HTTP calls (e.g. `httpx`) | Scripted `@node` with explicit HTTP client; no dedicated RemoteTool primitive | LangChain `StructuredTool` over HTTP | **GAP-AS** |
| **ServerTool** | Tool that is registered to and executed by the orchestrator (orchestrator-side execution) | N/A — LangGraph has no orchestrator-side tool execution | GAP-AS. neograph has NO orchestrator-side execution. All tools are client-resolved functions bound via `tool_factories`. The graph compiler NEVER executes tools directly. | N/A | **GAP** |
| **BuiltinTool** | Tool that is built into and executed by the orchestrator (tool_type, configuration, executor_name, tool_version) | N/A | GAP-AS. Same as ServerTool — neograph lacks orchestrator-side tool execution. No built-in tool registry. | N/A | **GAP** |
| **MCPTool** | MCP server tool with `client_transport` (Stdio/SSE/StreamableHTTP) | LangChain MCP adapters (`langchain-mcp-adapters`) | `resource_reader()` + MCP client in DI; typed async `StructuredTool` with coroutine | MCP SDK + LangChain | **NATIVE** |
| **ToolBox** | Component that exposes one or more tools to agentic components | `Node.tools` list | `Node(tools=[...])` list; no explicit ToolBox grouping primitive | — | **GAP-AS** |
| **MCPToolBox** | Dynamically exposes tools from MCP server with client_transport, tool_filter | LangChain MCP adapters | `MCPToolBox` in `neograph_mcp` (package-level, not core neograph) | MCP SDK + LangChain | **FULL (in neograph_mcp)** |

### Field-by-field: Tool

| Agent Spec field | Type | LangGraph | neograph Tool | Status |
|------------------|------|-----------|---------------|--------|
| `name` | string | `BaseTool.name` | `Tool.name` (required) | **NATIVE** |
| `inputs` | `list[JSONSchemaValue]` | `args_schema` | Inferred from factory's callable; NOT in Tool spec | **GAP-AS** |
| `outputs` | `list[JSONSchemaValue]` | Return type | Inferred from factory's callable; NOT in Tool spec | **GAP-AS** |
| `description` | string | `BaseTool.description` | From factory (docstring or `StructuredTool` description); NOT in Tool spec | **GAP-AS** |
| `budget` / call limits | per-tool integer | — | `Tool.budget` (int, 0=unlimited); enforced by `ToolBudgetTracker` | **NATIVE** |
| `config` | dict | — | `Tool.config` (passed to factory) | **NATIVE** |
| `idempotent` | bool | — | `Tool.idempotent` (replay-safe flag; neograph-lhc6) | **NATIVE** |
| `requires_confirmation` | bool | — | GAP-AS. No confirmation flag in neograph; use `gate_tools_when` on node for tool-gating HITL | **GAP-AS** |

### Field-by-field: RemoteTool (extends Tool)

| Agent Spec field | Type | LangGraph | neograph | Status |
|------------------|------|-----------|-----------|--------|
| `url` | string with `{{placeholder}}` | — | Not a distinct type; use scripted `@node` with URL templating | **GAP-AS** |
| `http_method` | string with `{{placeholder}}` | — | Not a distinct type; use scripted `@node` | **GAP-AS** |
| `api_spec_uri` | string (optional) | — | Not represented | **GAP-AS** |
| `data` | dict with `{{placeholder}}` values | — | Not represented; use scripted `@node` body | **GAP-AS** |
| `query_params` | dict with `{{placeholder}}` values | — | Not represented; use scripted `@node` | **GAP-AS** |
| `headers` | dict with `{{placeholder}}` values | — | Not represented; use scripted `@node` | **GAP-AS** |
| `sensitive_headers` | dict (for auth tokens, excluded from export) | — | Not represented | **GAP-AS** |
| `url_allow_list` | list of allowed URL patterns | — | Not represented | **GAP-AS** |
| `retry_policy` | RetryPolicy object | — | Not represented; use scripted `@node` with retry logic | **GAP-AS** |

### Field-by-field: MCPTool (extends Tool)

| Agent Spec field | Type | LangGraph | neograph | Status |
|------------------|------|-----------|-----------|--------|
| `client_transport` | transport enum | Stdio/SSE/StreamableHTTP | `resource_reader()` uses consumer-owned async fetcher from DI; NOT in Tool spec | **GAP-AS** |

### ToolBox / tool grouping

| Agent Spec concept | LangGraph | neograph | Status |
|--------------------|-----------|-----------|--------|
| `ToolBox` (exposes tools) | — | `Node.tools` list (no named grouping) | **GAP-AS** |
| `MCPToolBox` (MCP tools) | LangChain MCP adapters | `MCPToolBox` in `neograph_mcp` package with client_transport, tool_filter | **FULL (in neograph_mcp)** |

## Status legend used

- **NATIVE**: Direct 1:1 correspondence; neograph already has this
- **PARTIAL**: Concept exists but field coverage is incomplete
- **GAP-AS**: Agent Spec has a primitive neograph lacks (must lower to scripted node)
- **GAP**: True architectural gap — neograph cannot represent this concept (e.g., ServerTool orchestrator-side execution)
- **N/A**: Not applicable or not a distinct concept in Agent Spec
- **FULL (in neograph_mcp)**: Full support in the separate `neograph_mcp` package (not core neograph)

## Serialization notes

### neograph `Tool` spec shape

From `_spec_schema.py`:
```python
class ToolSpec(BaseModel):
    name: str
    budget: int = 0
    config: dict[str, Any] = Field(default_factory=dict)
```

**Carries in Agent Spec export**: `name`, `budget`, `config` map directly.

**Does NOT serialize**:
- `idempotent`: neograph-specific replay flag (not in Agent Spec)
- Input/output schemas: inferred from factory at runtime, not in spec
- Description: from factory, not in spec

### Agent Spec Tool shape

From DeepWiki:
- Base `Tool` extends `ComponentWithIO` → `inputs: list[JSONSchemaValue]`, `outputs: list[JSONSchemaValue]`
- `RemoteTool` adds: `url`, `http_method`, `api_spec_uri`, `data`, `query_params`, `headers`
- `MCPTool` adds: `client_transport`

**Carries in neograph import**: Only `name` is required; `budget` maps to `Tool.budget`; `config` maps to `Tool.config`.

**Does NOT import**:
- `inputs`/`outputs` schemas: neograph infers from factory callable, not from spec
- RemoteTool HTTP fields: no primitive to map to; import as scripted node
- `description`: from factory, not from spec

## Export lowering

When exporting a neograph Construct to Agent Spec:

### Tool spec → Agent Spec Tool

| neograph | Agent Spec target | Lowering |
|----------|-------------------|----------|
| `Tool(name, budget, config)` | `Tool` with `name`, metadata for budget | Name-only export; budget in `metadata['neograph']`; factory not serialized |
| `Tool.idempotent` | `metadata['neograph/idempotent']` | Stash in metadata (Agent Spec has no idempotent field) |
| Scripted `@tool` function | `ClientTool` or `ServerTool` | Export as name; factory binds at runtime |
| Raw `BaseTool` passed to `Node(tools=[...])` | `ServerTool` | Export as name; auto-registered factory (compile seam) |
| `resource_reader()` tool | `MCPTool` | Export as `MCPTool` with `client_transport` inferred from DI |

### Tool budget → Agent Spec

- **neograph**: `Tool.budget` (per-tool call limit) + `ToolBudgetTracker` enforcement
- **Agent Spec**: No direct field; stash in `metadata['neograph/budget']`
- **Oracle/Each**: `tool_budget` field (modifier-level budget per tool)
  - Export to `metadata['neograph/tool_budget']` on the Oracle/Map node

### RemoteTool gap

**No neograph primitive.** Agent Spec `RemoteTool` must lower to a scripted `@node`:

```yaml
# Agent Spec RemoteTool
RemoteTool:
  name: update_profile
  url: "http://api.example.com/users/{{username}}"
  http_method: "PUT"
  data: {"email": "{{email}}"}

# Lowers to neograph scripted node
Node:
  name: update_profile
  mode: scripted
  scripted_fn: "update_profile"  # user-implemented
  inputs: {username: str, email: str}
```

The lowering is **lossy**: the URL/method/data shape is encoded in the function body, not as declarative fields.

## Import reconstruction

When importing Agent Spec → neograph:

### Tool → neograph Tool

| Agent Spec | neograph target | Reconstruction |
|------------|-----------------|----------------|
| `Tool` (name only) | `Tool(name=...)` | Name-only; factory registered separately via `tool_factories=` |
| `Tool` with inputs/outputs | `Tool(name=...)` | Input/output schemas IGNORED; inferred from factory instead |
| `ClientTool` | Scripted `@node` or `@tool` | Interpreted as name; factory must be provided |
| `ServerTool` | `Tool(name=...)` | Same as Tool; factory binds runtime impl |
| `RemoteTool` | Scripted `@node(mode='scripted')` | Lowered to node; HTTP fields become implementation detail |
| `MCPTool` | `resource_reader()` or MCP tool | Name-only; MCP client bound at runtime via DI |

### Tool budget reconstruction

- If `metadata['neograph/budget']` present → set `Tool.budget`
- If `metadata['neograph/tool_budget']` present (Oracle/Each) → warn; neograph has no per-modifier tool budget
- Agent Spec tool budgets (if any) → stash in `Tool.config` for future support

### ToolBox reconstruction

- Agent Spec `ToolBox` → neograph `Node.tools` list
- Named tool groups → lost; import as flat tool list
- `MCPToolBox` → list of MCP tools; each imported as name-only with MCP client in DI

## Verdict for interop

**Fidelity impact**: Tool name and budget round-trip cleanly. Input/output schemas do NOT serialize (neograph infers from factory; Agent Spec declares explicitly). This is a **fidelity gap** for export/import: an Agent Spec Tool's `inputs`/`outputs` are lost on import, and neograph's factory-inferred schemas are not represented in export.

**Biggest risk**: The **ServerTool/Builtintool GAP** is a TRUE architectural mismatch — Agent Spec assumes orchestrator-side tool execution, but neograph has NO orchestrator-side execution. All neograph tools are client-resolved functions bound via `tool_factories`. Importing a ServerTool or BuiltinTool would require synthesizing a client-side factory or failing outright.

**Secondary risk**: The **RemoteTool gap**. Agent Spec's declarative HTTP tool has no neograph primitive; it lowers to a scripted node, losing the declarative URL/method/data shape. Export→import of RemoteTool produces a generic "call this scripted node" with no record of the HTTP structure. ToolBox grouping is also lost (neograph has no named tool groups).

**Mitigation**: Stash RemoteTool/ServerTool fields in `metadata['neograph/remote_tool']` or `metadata['neograph/server_tool']` for lossless round-trip, but recognize that cross-runtime fidelity (WayFlow/AutoGen) will not see a reconstructed tool—only a scripted node with metadata.

**Export strategy**:
- Name-only tools → clean export
- Budget/config → metadata
- RemoteTool → lower to scripted node + stash full spec in metadata
- MCPTool → export as `MCPTool` with inferred `client_transport`
- ServerTool/Builtintool → FAIL or emit GAP warning (no orchestrator-side execution)

**Import strategy**:
- Name-only tools → reconstruct as `Tool(name=...)`
- RemoteTool → reconstruct as scripted node; emit warning if HTTP fields present
- ServerTool/Builtintool → FAIL with clear error (orchestrator-side execution not supported)
- MCPTool → reconstruct as `resource_reader()` or MCP tool name
- ToolBox → flatten to `Node.tools` list
