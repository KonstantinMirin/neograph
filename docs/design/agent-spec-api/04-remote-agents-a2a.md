# Agent Spec → neograph mapping: Remote Agents & A2A

## Source

**Agent Spec v26.1.2** — `RemoteAgent` is an abstract `AgenticComponent` for agents executed outside the current process, typically via a remote endpoint (REST/RPC). Serves as orchestration surface for SaaS or microservice-based agents.

**WayFlow** — Concrete `OciAgent` (OCI Generative AI Agents) with `agent_endpoint_id` and `OciClientConfig`. A2A concepts relate to MCP client transports (`RemoteBaseTransport`, `SessionParameters`) rather than a dedicated `A2AAgent` class.

**Note:** `A2AAgent`, `A2AConnectionConfig`, and `A2ASessionParameters` as named classes **do not exist** in the Agent Spec or WayFlow codebases (v26.1.2). A2A is referenced as the Google Agent2Agent protocol standard — a future consideration, not an implemented component.

---

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|---|---|---|---|---|---|
| `RemoteAgent` (abstract) | Agent logic executed in another runtime via network endpoint (REST/RPC) | `RemoteGraph` client OR raw HTTP call | `Node.scripted(name="remote_x", fn="call_http")` or `@node(mode='raw')` with `httpx`/`requests` | HTTP client libraries | **GAP-AS** (import as name-bound scripted node) |
| `OciAgent` (extends RemoteAgent) | OCI Generative AI Agents endpoint; carries `agent_endpoint_id: str` + `client_config: OciClientConfig` (instance principal/API key auth) | Same as RemoteAgent (HTTP to OCI endpoint) | Same scripted/raw node pattern; DI injects `agent_endpoint_id` + OCI creds via `FromConfig` | OCI SDK, Oracle Cloud | **GAP-AS** (import as name-bound scripted node) |
| `A2AAgent` | *(Does not exist in v26.1.2)* A2A protocol for agent-to-agent communication | N/A (unimplemented) | N/A | Google Agent2Agent Protocol | **N/A** (future/external) |
| `A2AConnectionConfig` | *(Does not exist)* Connection parameters for A2A protocol | N/A | N/A | A2A Protocol spec | **N/A** (future/external) |
| `A2ASessionParameters` | *(Does not exist)* A2A session parameters | N/A | N/A | A2A Protocol spec | **N/A** (future/external) |
| `SessionParameters` (MCP, not A2A) | Arguments for MCP client session (`read_timeout_seconds`, etc.) | MCP session init | Handled by `mcp` SDK when binding MCP tools via `mcp_tool_factory` | `mcp` Python SDK | **-covered-** (via MCP integration) |
| `RemoteBaseTransport` (MCP) | Base for transports connecting to remote MCP servers (`url`, `headers`, `timeout`, `auth`, `session_parameters`) | MCP transport | Not surfaced at node level; MCP tools are name-bound factories | `mcp` Python SDK | **covered-** (via MCP integration) |
| `OciClientConfig` | OCI authentication config (instance principal, API key) | HTTP headers/auth | DI `FromConfig` to inject into scripted/raw node | OCI SDK | **DI-able** (config layer) |

---

## Status legend used

| Status | Meaning |
|---|---|
| **GAP-AS** | Genuine structural gap; import as fallback primitive (scripted/raw node) with metadata marker preserving original spec. Round-trip risk localized. |
| **covered-** | Fully covered via existing neograph surface (MCP tools, DI). |
| **DI-able** | Mappable to DI layer (`FromConfig`/`FromInput`). |
| **N/A** | Does not exist in Agent Spec v26.1.2; future consideration or external standard. |

---

## Serialization notes

**RemoteAgent/OciAgent** on import:
- No Agent Spec primitive maps 1:1. Import as a name-bound `Node.scripted(name="{remote_agent_name}", fn="call_remote_endpoint")`.
- Output type declared in Agent Spec becomes `outputs=` on the scripted Node.
- Input descriptor(s) become `inputs=` (single type or dict-form).
- OCI-specific fields (`agent_endpoint_id`, `client_config`) map to DI: `endpoint_id: Annotated[str, FromConfig]`, `oci_config: Annotated[OciClientConfig, FromConfig]`.

**A2AAgent/A2AConnectionConfig/A2ASessionParameters**:
- Not present in Agent Spec v26.1.2. No serialization path needed until Agent Spec adds them.
- If encountered in a future version, treat same as RemoteAgent: scripted node with DI for connection/session params.

**Export** (neograph → Agent Spec):
- A scripted/raw node calling HTTP to a remote agent can be annotated with `metadata["neograph/remote_agent"]` to preserve semantic intent.
- Export as `ToolNode` (behavior name-bound at runtime, per §5) — the Agent Spec runtime binds the name to an actual remote-call implementation.
- Lowering is lossy: the "this is a remote agent" semantic becomes "this is a named tool."

---

## Export lowering

No first-class `RemoteAgent` in Agent Spec primitives. The closest mapping:

| neograph | Agent Spec export | Cost |
|---|---|---|
| Scripted node calling HTTP REST/RPC | `ToolNode` with `tool_name` bound at runtime to an HTTP client factory | Semantic loss; "remote agent" becomes "tool" |
| DI `FromConfig` for endpoint/creds | Stored in separate config; not in Flow YAML | Requires config file coordination |

**Do NOT expand** to multi-node inline subgraphs (e.g., HTTP call + response parsing). Agent Spec runtimes expect a single `ToolNode` name they can bind. Implementation lives in the runtime's tool registry, not the spec.

---

## Import reconstruction

When importing a Flow containing a node that represents a remote agent:

1. **No `RemoteAgent` type in Agent Spec** — the Flow will have a generic node (likely `ToolNode` or `ApiNode`) whose `tool_name` or implementation is bound at runtime in the WayFlow loader.
2. Import as `Node.scripted(name=node_name, fn=tool_name)` where `tool_name` is the Agent Spec node's name (convention: "call this tool").
3. If the Agent Spec node carries metadata indicating it was a remote agent (e.g., from a prior neograph export), stamp `metadata["neograph/remote_agent"]` on the imported Node.
4. OCI-specific connection configs map to DI parameters in the scripted node's signature, resolved via `FromConfig`.

**Cannot reconstruct** the original "this is a remote agent" abstraction from a primitive Agent Spec Flow — only a name-bound tool placeholder. The importer emits a script that a runtime tool factory must implement.

---

## Verdict for interop

**RemoteAgent/OciAgent** represent a genuine structural gap: neograph is single-runtime, while RemoteAgent explicitly crosses runtime boundaries via network calls. The mapping is **lower-fidelity** — remote agents become name-bound scripted/raw nodes, and the semantic "remote agent" abstraction survives only via metadata markers. Round-trip risk is localized to this component type (per §4a of the competitive analysis).

**A2AAgent/A2AConnectionConfig/A2ASessionParameters** do not exist in Agent Spec v26.1.2. A2A is an external protocol standard (Google) referenced in docs but not implemented. When Agent Spec adds these, treat same as RemoteAgent: scripted node + DI for connection/session params. Do NOT chase now — "file, don't chase" per the competitive analysis verdict.

**The biggest interop risk:** Agent Spec Flows that rely on runtime tool registries to bind names to remote-call implementations. Import produces a Node that references a function name (`scripted_fn`) with no implementation guarantee — the caller must provide a matching `scripted=` dict at `compile()` or `run()` time. This is **already the contract for scripted nodes**, so no new gap, but a point of friction for cross-ecosystem round-trips.
