# Agent Spec → neograph: master synthesis (API fan-out)

Date: 2026-07-09
Inputs: sibling files `01`–`15` (one per API section of Open Agent Spec v26.1.2).
Companion: `../agent-spec-interop-2026-07-09.md` (design) · `../wayflow-competitive-analysis-2026-07-09.md`.
Feeds: gate task `neograph-swcy1` (fidelity audit + interop-vs-replacement ratification).

The per-class tables live in the section files. This is the decision-grade roll-up.

---

## 1. Round-trip status at a glance

| # | Section | DIRECT | LOWER | GAP-AS | NO-REPR | One-line verdict |
|---|---|---|---|---|---|---|
| 01 | Components + IO Properties | most | — | — | tuple/Literal/Enum | Type backbone is solid; 3 neograph types have no Property repr |
| 02 | Flows & Nodes | 12 of 18 | Oracle/Each/Loop/Operator | Input/OutputMessageNode | callable fields | Core maps cleanly; the modifier lowerings are export-only |
| 03 | Agents + Specialization | Agent | SpecializedAgent | SpecializedAgent | mode agent vs act | ReAct 1:1; "specialize existing agent" is a real DX gap |
| 04 | Remote Agents + A2A | — | — | RemoteAgent/OciAgent/A2AAgent | — | Single-runtime gap; fake as scripted HTTP node + marker |
| 05 | LLMs | LlmGenerationConfig | — | OCI/Gemini auth tree | llm_factory | Auth collapses to `provider_kwargs`; stash factory in metadata |
| 06 | Tools | Tool name+budget | — | ServerTool/BuiltinTool, RemoteTool, ToolBox | inferred schemas | Name-vs-factory split shared; orchestrator-side tools = FAIL on import |
| 07 | MCP | tool name+budget+config | — | mTLS transports, idempotent | typed-binding layer | The defensible seam lives ABOVE serialization; mTLS ~50 lines |
| 08 | Serialization | blueprint | — | — | — | Adapter pattern + plugin system = our template; metadata gate unverified |
| 09 | Agentic Patterns | ManagerWorkers | — | Swarm | — | Swarm = addable combinator via Command(goto); faithful round-trip risky |
| 10 | Connectivity & Resilience | RetryPolicy.max_retries | — | OAuth/PKCE/Scope family | richer retry | OAuth entirely outside IR; provider-layer concern |
| 11 | Datastores | — | — | all (import-only shim) | — | One-way import; export yields opaque nodes. Correct trade. |
| 12 | Transforms | — | @node→marker→Transform | (no neograph primitive) | — | KEEP as composed @node; do NOT add a Transform modifier |
| 13 | Tracing | — | — | (SwarmExecutionSpan) | — | OUT OF wire scope; runtime callbacks only |
| 14 | Adapters + Version | blueprint | — | — | — | LangGraph adapter = template; depend on pyagentspec behind an extra |
| 15 | Evaluation | — | Each+@node+Oracle demo | — | — | OUT OF wire scope; harness concern |

---

## 2. What round-trips cleanly (DIRECT)

- **Graph skeleton:** `Flow`↔`Construct`; `StartNode`/`EndNode`↔boundary ports; `FlowNode`↔sub-`Construct`; `ControlFlowEdge`↔inferred order.
- **Node modes:** `LlmNode`↔`Node(mode=think)`; `AgentNode`/`Agent`↔`Node(mode=agent/act)`; `ToolNode`/`ApiNode`↔`Node(mode=scripted)`; `CatchExceptionNode`↔ForwardConstruct `try`.
- **Routing:** `BranchingNode`↔`Operator`/`Loop`/`_BranchNode`.
- **Types:** primitives/list/dict/object/null/union/`BaseModel` ↔ `Property` (the backbone).
- **Tools:** name + budget ↔ `tool_factories` (the name-vs-factory split is identical on both sides).
- **RetryPolicy:** `max_retries` (integer only).
- **ManagerWorkers:** a recipe over router + sub-constructs + Loop — reconstructable losslessly.

The single biggest *structural* mapping fact, repeated across files: **Agent Spec's explicit `DataFlowEdge`/`ControlFlowEdge` are what neograph INFERS from typed signatures.** neograph is Agent Spec's implicit name-based data-flow mode promoted to the only mode, with assembly-time validation the spec lacks (design §2/§3, confirmed in `02`).

---

## 3. What lowers on EXPORT only (neograph value-add, NOT reversible on import)

These are the modifier lowerings — the whole point of neograph sitting above the flat node library:

| neograph | Agent Spec expansion | Reversible? |
|---|---|---|
| `Oracle(models=[…], merge_fn=…)` | `ParallelFlowNode` of N `LlmNode`s + merge `LlmNode`/`ToolNode` + fan-out/fan-in edges | **NO** — flagship gap |
| `Each(over=…)` | `MapNode`/`ParallelMapNode` wrapping an extracted sub-Flow | no (the `over` dotted path isn't recoverable) |
| `Loop(when=…)` | `BranchingNode` + cyclic `ControlFlowEdge` + self `DataFlowEdge` | no (branch-with-back-edge ≠ loop without a marker) |
| `Operator(when=…)` | conditional `BranchingNode` / gated edge | no |
| `SpecializedAgent` (base + specialization) | flattened to one `Agent`/`@node` with a fixed prompt | no (specialization boundary lost) |

Without a per-group `metadata["neograph/modifier"]` marker **or** the verbatim `neograph/source` embed (§6a), import cannot reconstruct any of these — the graph still *runs*, but the authoring intent is gone. **Oracle is the clearest demo of why neograph is an authoring layer above Agent Spec.**

---

## 4. The genuine gaps (GAP-AS) — file, don't chase

Ranked by how much they actually matter:

| Gap | Section | Disposition |
|---|---|---|
| **RemoteAgent / OciAgent / A2AAgent** | 04 | Fake as name-bound scripted/raw HTTP node + DI for connection params + `metadata`. A2A isn't even in v26.1.2 core. Round-trip risk localized here. |
| **Swarm** | 09 | Genuinely dynamic (agents pick next at runtime from `relationships`). Addable as a new combinator over LangGraph `Command(goto)`/`Send` — **not a substrate limit**. Faithful round-trip is risky (reconstruction forces a "router in the middle", not peer-to-peer). |
| **SpecializedAgent** | 03 | No "specialize an existing agent" concept. Closest = `@node` with fixed prompt. Real DX difference (template-style instantiation). |
| **ServerTool / BuiltinTool** | 06 | Orchestrator-side execution; neograph is client-only. **FAIL on import** with a clear error (don't synthesize). |
| **RemoteTool / ToolBox** | 06 | RemoteTool → scripted node (loses declarative HTTP shape); ToolBox grouping has no neograph equivalent (flatten to `Node.tools`). Stash in metadata. |
| **OAuth / PKCE / Scope family** | 10 | Entirely outside neograph IR. Handle at the provider/LangChain-client layer; orthogonal to the wire format. |
| **mTLS MCP transports** | 07 | `SSEmTLSTransport`/`StreamableHTTPmTLSTransport` = WayFlow's own ~50-line httpx additions. Tiny, matchable (competitive §4a). |
| **Datastores** | 11 | One-way **import-only** shim (name-bound scripted `@node`). Export yields opaque `ToolNode`/`ApiNode`. Correct trade — neograph composes data access as `@node`s over LangChain. |
| **InputMessageNode / OutputMessageNode** | 02 | `interrupt()` not first-class in neograph (exposed via LangGraph, no node); conversation/message state not modeled. |
| **tuple / Literal / Enum types** | 01 | NO-REPR — valid neograph I/O with no `Property` form. Downgrade (tuple→list, Literal→string, Enum→string) + `metadata["neograph/original_type"]`, or rely on the `neograph/source` embed. |

---

## 5. Out of wire-format scope (runtime-only constructs)

- **Tracing (13):** `Trace`/`Span`/`Event`/`SpanProcessor` are runtime callback constructs, never serialized. neograph has **no first-party span model** — it delegates to the LangChain/LangGraph callback protocol + optional `langfuse` extra (`_trace.named()` stamps `neograph:node`/`neograph_mode` metadata on callback spans). Zero fidelity impact on export/import. (The one genuine tracing gap, `SwarmExecutionSpan`, is a *capability* gap, not a serialization gap.)
- **Evaluation (15):** `Dataset`/`Evaluator`/`Metric`/`Aggregator` are test-harness constructs, not pipeline components. Import/export ignores them. Notably, the Agent Spec eval pattern *is* expressible in neograph as `Each(over=dataset)` + per-sample `@node` metric + `Oracle`/`merge_fn` aggregation — a demonstration, not a gap.

---

## 6. The four decisions the gate task must ratify

### 6.1 Metadata round-trip gate — UNVERIFIED (blocking)
The §6a lossless-embed strategy assumes `Component.metadata` survives `AgentSpecSerializer.to_dict → AgentSpecDeserializer.from_dict`. `08` confirms `metadata` is a `dict[str,Any]` threaded through (de)serialization, **but whether disaggregation/reaggregation preserves it is unconfirmed**. **`neograph-swcy1` must include a test asserting `metadata["neograph/source"]` survives a full round-trip including disaggregation.** If it doesn't, the whole marker/embed strategy needs a different carrier.

### 6.2 Transforms — do NOT add a modifier (12)
**Decision: keep Transforms as a composed scripted `@node` + a metadata kind tag; do not add a neograph Transform modifier.** Rationale (layer discipline): transforms are *message-level* concerns (LangChain's `trim_messages`/`summarize_messages`), not *graph-level* concerns (neograph's typed DAG). The export layer recognizes `node.metadata["agent_spec_kind"]="message_summarization_transform"` and emits the proper Agent Spec Transform; import inverts it. Adding a modifier would bloat the IR (`_construct_validation.py`, `state.py`, factory dispatch) for something outside neograph's core competency.

### 6.3 Dependency — depend on `pyagentspec`, behind an extra (14)
**Decision: depend on `pyagentspec>=26.1.0,<27` via an optional `[agent-spec]` extra; do NOT vendor; do NOT depend on `pyagentspec.adapters.langgraph`.** The plugin architecture (`ComponentSerializationPlugin`) and disaggregation (two-phase `components_registry`) are non-trivial and battle-tested in `pyagentspec`; vendoring reimplements them and freezes neograph to one spec version. neograph depends on `pyagentspec.serialization` (core Serializer/Deserializer) only, then builds neograph-specific conversion. Core `loader.py` stays YAML-only; `from_agent_spec`/`to_agent_spec` live behind the extra. (Resolves design §9.4; ties to license-spike `neograph-dmqe` — confirm Apache-2.0 on the *package*, not just the repos.)

### 6.4 Oracle irreversibility — per-group marker required (02/06)
N parallel `LlmNode`s + a merge node is indistinguishable from hand-authored parallel flows. The per-group `metadata["neograph/modifier"]="oracle"` marker (linking the variant nodes + merge via a `group_id`) is what lets import reconstruct `Oracle`. **The source-embed (§6a Layer B) is the guaranteed lossless path; per-group markers (Layer A) are the granular fallback when the Flow was hand-edited.** This is the flagship demonstration that neograph is an authoring layer above Agent Spec — advertise it in the example/docs.

---

## 7. Type-system backbone (01)

`Property` ↔ Pydantic via `spec_types.py` is the round-trip backbone and is well-supported for primitives, `list`, `dict`, `object`, `null`, `union`, and nested `BaseModel`. The registry is the single point of truth — the importer/exporter must register every type it touches. **NO-REPR types: `tuple`, `Literal`, `Enum`** — downgrade + `metadata["neograph/original_type"]`, or rely on the source embed. This is a blocker for pipelines using these types that need full fidelity (the source-embed sidesteps it entirely for neograph-authored specs).

---

## 8. Provider / auth fidelity (05, 10)

`llm_factory` is runtime-flexible (closure-captured at compile) but serialization-weak: Agent Spec's typed auth hierarchy (`OciClientConfigWithApiKey`, `GeminiVertexAIAuthConfig`, etc.) collapses to neograph's untyped `provider_kwargs`. **Recommendation:** on export emit a portable `OpenAiCompatibleConfig` (model_id + url) **and** stash the original `llm_factory` descriptor in `metadata["neograph/llm_factory"]` for lossless round-trip (approach B from `05`, consistent with §6a). Oracle per-variant model selection (`Oracle._oracle_model`) has **no** Agent Spec representation — N hand-instantiated `LlmNode`s is the only lowering. OAuth/PKCE/Scope is a provider-layer concern, not an IR concern.

---

## 9. The MCP seam — confirmed defensible (07)

Agent Spec's MCP stops at **name + budget + config + transport**. neograph's value lives *above* that: MCP tools bound into **type channels** with compile-time validation, DI, and self-heal (resource replay). That layer is exactly what Agent Spec's serialization doesn't capture — and it's the defensible niche (positioning memory + design §8). Concrete: `MCPTool`/`MCPToolBox` ↔ `mcp_tool_factory` output; transports (Stdio/SSE/StreamableHTTP) come from the shared `mcp` SDK; mTLS is the one ~50-line real add; `idempotent` (replay-safety) must ride in metadata to round-trip.

---

## 10. Open questions to resolve in `neograph-swcy1`

1. **Metadata-via-disaggregation test** (6.1) — the make-or-break for the embed strategy.
2. **`InputMessageNode` ↔ LangGraph `interrupt()`** — does neograph expose interrupt first-class, or is it a GAP we document? (`02` flagged it GAP-AS.)
3. **Oracle reconstruction policy** (6.4) — per-group marker vs source-embed-only vs both.
4. **Swarm scope** (09) — add the combinator now (it's a nice differentiator and uses Command/Send neograph already uses internally for Each), or metadata-marker-only import + defer? Recommend: defer the combinator, marker-only import, because faithful peer-to-peer round-trip is structurally hard.
5. **`SpecializedAgent`** (03) — is "specialize an existing agent" worth a neograph concept, or is `@node`-with-prompt + a convention enough? Recommend: convention + metadata marker for now.

---

## Bottom line

The mapping is **comprehensive and mostly clean**. 12 of 18 flow-node types are DIRECT; the 4 modifiers are the export-only value-add; genuine gaps collapse to **RemoteAgent/A2A** (single-runtime), **Swarm** (addable), and a handful of minor ones (ServerTool, OAuth, mTLS, NO-REPR types) — exactly matching the competitive analysis's "two real unknowns + plumbing" verdict. The interop is buildable as designed (design §4–§7), gated only by the metadata round-trip test (6.1) and the license-spike (`neograph-dmqe`). Tracing and Evaluation are out of scope, as expected.
