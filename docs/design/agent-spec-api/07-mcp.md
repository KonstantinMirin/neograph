# Agent Spec → neograph mapping: MCP

## Source

**Agent Spec v26.1.2 MCP API:**
- [API Reference: MCP](https://oracle.github.io/agent-spec/26.1.2/api/mcp.html) — Client Transports (SessionParameters, StdioTransport, SSETransport, SSEmTLSTransport, StreamableHTTPTransport, StreamableHTTPmTLSTransport)
- **Agent Spec source:** `pyagentspec.mcp.clienttransport`, `pyagentspec.mcp.tools` (MCPTool, MCPToolBox)
- How-to: "Connect MCP tools to assistants" (unavailable via web fetch at time of writing)

**neograph MCP surface:**
- `src/neograph_mcp/_client.py` — mcp_tool_factory, mcp_tool_factories, StdioServer, HttpServer, mcp_resource_fetcher
- `src/neograph/_tool_loop.py` — ReAct tool loop invocation
- `src/neograph/node.py` — Node(tools=..., mode=agent|act)
- `src/neograph/tool.py` — Tool spec, ToolInteraction, resource_reader

**Key design reference:**
- `docs/design/agent-spec-interop-2026-07-09.md` §8 (MCP typed-binding seam — neograph's defensible niche)
- `docs/design/wayflow-competitive-analysis-2026-07-09.md` §4a (mTLS variants = ~50-line httpx additions, GAP-AS)

---

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|---|---|---|---|---|---|
| `SessionParameters` | Session config (read_timeout_seconds) | — | Implicit in `mcp_tool_factories` (token_provider + per-run client creation) | `mcp` SDK session params | **DIRECT** |
| `ClientTransport` (base) | Abstract MCP client transport | — | — (base class only) | `mcp` SDK | **N/A** |
| `RemoteTransport` (base) | Base for remote transports (url, auth, headers, retry_policy) | — | — (partial mapping via HttpServer) | `mcp` SDK | **PARTIAL** |
| `StdioTransport` | Subprocess stdio transport (command, args, env, cwd) | — | `StdioServer(command, args, env)` — cwd not yet supported | `mcp` SDK | **DIRECT** |
| `SSETransport` | Server-Sent Events transport (url, auth, headers, sensitive_headers, retry_policy) | — | `HttpServer(url, headers)` — unified for SSE/StreamableHTTP; auth/retry_policy not yet supported | `mcp` SDK | **PARTIAL** |
| `StreamableHTTPTransport` | Streamable HTTP transport (same params as SSETransport) | — | `HttpServer(url, headers)` — unified; auth/retry_policy not yet supported | `mcp` SDK | **PARTIAL** |
| `SSEmTLSTransport` | SSE with mTLS (adds key_file, cert_file, ca_file) | — | **GAP-AS** — ~50-line httpx client addition per competitive analysis | WayFlow-only | **GAP-AS** |
| `StreamableHTTPmTLSTransport` | Streamable HTTP with mTLS (same additions) | — | **GAP-AS** — ~50-line httpx client addition per competitive analysis | WayFlow-only | **GAP-AS** |
| `MCPTool` | Tool specification (name, budget, config) bound at runtime via tool_registry | `ToolNode` | `Tool(name, budget, config)` bound via `tool_factories` at compile time | Name-vs-factory split shared | **DIRECT** |
| `MCPToolBox` | Collection of MCP tools (tool_name → MCPTool map) | — | `dict[str, ToolFactory]` from `mcp_tool_factories` (namespace=`server::tool` by default) | langchain-mcp-adapters | **DIRECT** |

**Legend:**
- **DIRECT**: Full semantic and param mapping
- **PARTIAL**: Core concept maps; some Agent Spec params not yet supported
- **GAP-AS**: Addable surface; competitive analysis shows ~50-line implementation
- **N/A**: Base class with no neograph counterpart

---

## Status legend used

| Status | Meaning |
|---|---|
| **DIRECT** | Complete semantic and parameter mapping; interop trivial |
| **PARTIAL** | Core concept maps; documented gaps (auth, retry_policy, cwd) |
| **GAP-AS** | Addable surface; WayFlow has ~50-line implementation (mTLS via httpx) |
| **N/A** | Base class or structural artifact with no direct neograph equivalent |

---

## Serialization notes

**SessionParameters** (`read_timeout_seconds`):
- Agent Spec serializes as part of Transport config
- neograph: implicit — per-run timeout handled by adapter, not surfaced in Spec

**Transport fields**:
- `StdioTransport.cwd` — neograph's `StdioServer` does not yet support cwd; GAP
- `RemoteTransport.auth` — neograph does not yet support AuthConfig; GAP
- `RemoteTransport.retry_policy` — neograph does not yet support RetryPolicy; GAP
- `RemoteTransport.sensitive_headers` — not surfaced in neograph (exclusion from export)
- mTLS fields (`key_file`, `cert_file`, `ca_file`) — GAP-AS

**MCPTool**:
- Agent Spec: `{name, budget, config}` (name-only reference)
- neograph `Tool`: `{name, budget, config, idempotent, _bound_tool}`
- Export: lower to `{name, budget, config}`; `idempotent` and `_bound_tool` are neograph-only
- Import: reconstruct as `Tool(name, budget, config)`; `idempotent` defaults False (conservative)

**MCPToolBox**:
- Agent Spec: map of `{tool_name: MCPTool}` or server-grouped collection
- neograph: `dict[str, ToolFactory]` keyed by `server::tool` when `namespace=True`
- Export: serialize as `{tool_name: {name, budget, config}}` map
- Import: reconstruct as `ToolFactory` binding (name-based lookup)

---

## Export lowering

How neograph's typed MCP binding lowers to Agent Spec MCPTool/MCPToolBox + metadata marker:

```
neograph IR (Node.tools=[Tool(...)]) → Agent Spec (MCPTool/MCPToolBox)
```

**Export path:**

1. **Tool spec lowering:**
   - `Tool(name, budget, config, idempotent, _bound_tool)` → `{name, budget, config}`
   - `idempotent` and `_bound_tool` are neograph-only; dropped from Agent Spec export
   - Budget and config serialize directly

2. **MCPToolBox lowering:**
   - Group tools by server (from factory key prefix before `::`)
   - Emit `{server_name: {tool_name: {name, budget, config}}}` structure
   - Namespace convention (`server::tool`) is neograph's default; explicit grouping on export

3. **Transport lowering:**
   - `StdioServer(command, args, env)` → `StdioTransport(command, args, env)` (cwd omitted)
   - `HttpServer(url, headers)` → `SSETransport(url, headers)` OR `StreamableHTTPTransport(url, headers)`
   - Choice of SSE vs StreamableHTTP must be explicit (neograph unifies; Agent Spec separates)
   - mTLS fields are NOT yet exported (GAP-AS)

4. **Metadata marker:**
   - Stamp exported `MCPTool` with `metadata["neograph/source"]` containing full neograph `Tool` spec (including `idempotent`, `_bound_tool`)
   - Enables lossless round-trip via verbatim embed (§6a of interop design)

**Losses on export:**
- `idempotent` flag (replay-safety gate) — not represented in Agent Spec
- `_bound_tool` (LangChain BaseTool instance) — runtime-only, never serializes
- `cwd` (stdio working dir) — not yet supported
- `auth`, `retry_policy`, `sensitive_headers` — not yet supported
- mTLS fields — GAP-AS

---

## Import reconstruction

How an imported MCPTool/MCPToolBox rebinds via tool_factories/mcp_tool_factory:

```
Agent Spec (MCPTool/MCPToolBox) → neograph Tool spec + tool_factories binding
```

**Import path:**

1. **MCPTool reconstruction:**
   - Read `{name, budget, config}` from Agent Spec
   - Reconstruct as `Tool(name, budget, config)`
   - `idempotent` defaults to `False` (conservative; requires manual override or metadata marker)

2. **MCPToolBox reconstruction:**
   - Read grouped tools map or flat tool list
   - For each `{name, budget, config}`:
     - Register `Tool(name, budget, config)`
     - Bind via `tool_factories[name]` → factory returning runtime tool instance

3. **Transport reconstruction:**
   - `StdioTransport(command, args, env)` → `StdioServer(command, args, env)`
   - `SSETransport(url, headers)` → `HttpServer(url, headers)` (assume SSE)
   - `StreamableHTTPTransport(url, headers)` → `HttpServer(url, headers)` (assume streamable HTTP)
   - mTLS transports → ERROR (not yet supported; GAP-AS)

4. **Factory binding (runtime):**
   - Import constructs `tool_factories` dict from imported MCPTool list
   - Each `tool_factories[name]` is a `(config, tool_config) -> tool` async factory
   - Bind at compile time via `compile(tool_factories=...)`

5. **Metadata marker recovery (lossless path):**
   - If `metadata["neograph/source"]` present → deserialize full neograph `Tool`
   - Reconstruct `idempotent` and `_bound_tool` (if a factory reference can be materialized)
   - Verify primitive matches metadata (fail-loud if diverged per §6a)

---

## Verdict for interop

**Fidelity impact:** Good for tool declaration (name, budget, config); losses on replay-safety (`idempotent`) and advanced transport features (auth, retry_policy, mTLS). Round-trip fidelity achievable via metadata marker for core tool shape; advanced features require GAP-AS work.

**Single biggest risk/gap:** mTLS transports (`SSEmTLSTransport`, `StreamableHTTPmTLSTransport`) are a ~50-line GAP-AS per competitive analysis — WayFlow implements them via httpx client cert_file/key_file/ca_file params, and neograph does not yet support this. Enterprise deployments requiring mTLS MCP will fail import unless this gap is closed. Secondary risk: `idempotent` flag loss means replay-safety gates (neograph-a5nh, layered expiry) cannot round-trip — conservative default may block valid replays on import.

**Key differentiator preserved:** neograph's MCP typed-binding seam (type channels, compile-time validation, DI, self-heal via resource replay) lives ABOVE the Agent Spec tool serialization — Agent Spec stops at "name + budget + config"; neograph adds the typed graph validation layer. This is the defensible niche per interop design §8.

---

## Sources

- Agent Spec v26.1.2 API: [MCP](https://oracle.github.io/agent-spec/26.1.2/api/mcp.html)
- neograph MCP surface: `src/neograph_mcp/_client.py`
- Design: `docs/design/agent-spec-interop-2026-07-09.md` §8
- Competitive analysis: `docs/design/wayflow-competitive-analysis-2026-07-09.md` §4a
