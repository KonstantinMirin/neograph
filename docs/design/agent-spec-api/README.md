# Agent Spec → neograph section mapping (fan-out research)

Date: 2026-07-09
Companion to: `../agent-spec-interop-2026-07-09.md` (interop design) and
`../wayflow-competitive-analysis-2026-07-09.md`.

## Purpose

One file per section of the Open Agent Spec **v26.1.2 API reference**
(https://oracle.github.io/agent-spec/26.1.2/api/index.html). Each maps every
class in that section to:

- the **LangGraph primitive** it corresponds to,
- the **neograph concept** (IR `Node`/mode/modifier, or a gap),
- the **ecosystem** (LangChain / `mcp` SDK / Langfuse / etc.).

This feeds the interop gate task (`neograph-swcy1`) and the `from_agent_spec` /
`to_agent_spec` design. The emitter must know what types exist and how things
connect — this is the grounded map of exactly that.

## Status legend (used across files)

| Code | Meaning |
|---|---|
| DIRECT | clean 1:1 mapping exists |
| LOWER | a neograph concept lowers INTO this primitive on export (one-directional) |
| RECONSTRUCT | importing this primitive can reconstruct a neograph concept |
| GAP-AS | Agent Spec has it, neograph lacks it (candidate feature, or "compose as `@node`") |
| GAP-NG | neograph has the stronger concept; the Agent Spec primitive is a downgrade |
| NO-REPR | cannot round-trip (e.g. callable-valued fields) |

## Files

| # | File | Section |
|---|---|---|
| 01 | `01-components-types.md` | Components + IO Properties (the type/`Property` system) |
| 02 | `02-flows-nodes.md` | Flows & Nodes (core: edges, all node types) |
| 03 | `03-agents-specialization.md` | Agents + Agent Specialization |
| 04 | `04-remote-agents-a2a.md` | Remote Agents + A2A |
| 05 | `05-llms.md` | LLMs (configs + generation config) |
| 06 | `06-tools.md` | Tools + ToolBox |
| 07 | `07-mcp.md` | MCP (transports, `MCPTool`, `MCPToolBox`) |
| 08 | `08-serialization.md` | Serialization / Deserialization (the adapter blueprint) |
| 09 | `09-patterns.md` | Agentic Patterns (Swarm, ManagerWorkers) |
| 10 | `10-connectivity.md` | Connectivity & Resilience (Auth, OAuth, RetryPolicy) |
| 11 | `11-datastores.md` | Datastores |
| 12 | `12-transforms.md` | Transforms (the declarative context-management example) |
| 13 | `13-tracing.md` | Tracing |
| 14 | `14-adapters.md` | Adapters + Version (the cross-framework round-trip blueprint) |
| 15 | `15-evaluation.md` | Evaluation |

A synthesized master index is produced after the fan-out completes.
