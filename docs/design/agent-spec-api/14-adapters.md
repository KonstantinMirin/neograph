# Agent Spec → neograph mapping: Adapters & Version

## Source

- Agent Spec API v26.1.2: [Adapters](https://oracle.github.io/agent-spec/26.1.2/api/adapters.html), [Version](https://oracle.github.io/agent-spec/26.1.2/api/version.html)
- neograph source: `src/neograph/loader.py`, `src/neograph/_spec_schema.py`, `src/neograph/spec_types.py`, `src/neograph/construct.py`
- Design context: `docs/design/agent-spec-interop-2026-07-09.md`, `docs/design/wayflow-competitive-analysis-2026-07-09.md`

---

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|---|---|---|---|---|---|
| `AgentSpecVersionEnum` | Enum of supported Agent Spec versions (v25.3.0–v26.2.0) | N/A | Import gating: reject unsupported spec versions | versioning fence | **NEW** — add `version: Literal["1"]` check to `Spec` loader (already exists in `_spec_schema.py:134`), extend for Agent Spec imports |
| `AdapterAgnosticAgentSpecLoader` | Base class for all loaders; plugin-aware deserialization; disaggregated component support | N/A | `from_agent_spec()` in `loader.py` (import home) | shared deserialization logic | neograph's `load_spec` is simpler (no plugins, no disaggregation), but will inherit Agent Spec's plugin pattern for import |
| `AgentSpecToLangGraphConverter` | Converts Agent Spec `Flow` → LangGraph `StateGraph` | `StateGraph` + compiled graph | `compile()` output is already LangGraph; neograph **IS** a converter from its own IR to LangGraph | direct peer | **template for neograph export** — Agent Spec is the target format, not the source |
| `LangGraphToAgentSpecConverter` | Converts LangGraph `StateGraph` → Agent Spec `Flow` | `StateGraph` | `Construct` (neograph's IR) | inverse direction | **primary reference** for neograph's `to_agent_spec()` export path |
| `AgentSpecExporter` (LangGraph) | Helper class: LangGraph → Agent Spec dict/JSON/YAML | `StateGraph` → `Flow` dict | `to_agent_spec()` free function in new `_agent_spec.py` | export surface | mirrors neograph's `load_spec` (import-only) symmetry |
| `AgentSpecLoader` (LangGraph) | Helper class: Agent Spec dict/JSON/YAML → LangGraph | `Flow` → `StateGraph` | `from_agent_spec()` in `loader.py` | import surface | **direct template** — same loader pattern neograph already uses |
| `AgentSpecExporter` (CrewAI) | CrewAI → Agent Spec | CrewAI components | N/A (CrewAI not in ecosystem) | **IGNORE** — no mapping needed |
| `AgentSpecLoader` (CrewAI) | Agent Spec → CrewAI | CrewAI components | N/A | **IGNORE** |
| `AgentSpecExporter` (AutoGen) | AutoGen → Agent Spec | AutoGen components | N/A | **IGNORE** |
| `AgentSpecLoader` (AutoGen) | Agent Spec → AutoGen | AutoGen components | N/A | **IGNORE** |
| `AgentSpecExporter` (WayFlow) | WayFlow → Agent Spec | WayFlow `Component` | N/A (competitor, not target) | **IGNORE** |
| `AgentSpecLoader` (WayFlow) | Agent Spec → WayFlow | WayFlow `Component` | N/A | **IGNORE** — WayFlow's own adapter is not neograph's concern |
| `AgentSpecExporter` (agent_framework) | Microsoft Agent Framework → Agent Spec | MS agent components | N/A | **IGNORE** |
| `AgentSpecLoader` (agent_framework) | Agent Spec → Microsoft Agent Framework | MS agent components | N/A | **IGNORE** |

---

## Status legend used

- **NEW**: feature to add in neograph
- **template**: use as reference for implementation
- **IGNORE**: out of scope (different ecosystem/competitor)

---

## Serialization notes

### Agent Spec serialization model
- **Components** carry `metadata: dict[str, Any]` — preserved through round-trip, ignored by runtimes
- **Disaggregated components**: split out reusable parts (LLM configs, tools) into separate JSON/YAML files, loaded via `components_registry` dict
- **Plugins**: extensible deserialization for custom component types
- **Two-phase loading**: `import_only_referenced_components=True` returns a dict of component IDs → runtime objects for use as registry

### neograph's current serialization (ground truth)
From `loader.py` and `_spec_schema.py`:
- **YAML/JSON → `Spec`** → `Construct` (one-way, import-only)
- `Spec.version: Literal["1"]` (forward-compat gate; line 134 of `_spec_schema.py`)
- `Spec.nodes: list[NodeSpec]` (flat pool, not nested)
- `Spec.constructs: list[ConstructSpec]` (sub-pipelines referencing nodes by name)
- `Spec.pipeline.nodes: list[str]` (ordered references into node/construct pool)
- `NodeSpec` / `ConstructSpec` carry **modifier slots** (`oracle`, `each`, `loop`, `operator`) — high-level sugar, Agent Spec has no equivalent
- **No export path exists** — no `to_dict`, `to_spec`, `Construct.to_spec()`

### Version gating strategy
- neograph's `Spec` already has `version: Literal["1"]` for its own format
- For Agent Spec imports: check `Flow.spec_version` against supported list (current: 26.1.2, latest: 26.2.0)
- Unsupported version → `ConfigurationError` with hint: "Agent Spec version X.Y.Z not supported; highest supported is A.B.C"

---

## The LangGraph adapter as neograph's template

### What the langgraph `AgentSpecExporter`/`AgentSpecLoader` do

**Loader (Agent Spec → LangGraph):**
1. Parse Agent Spec JSON/YAML into Pydantic `Component` models
2. Resolve tool references via `tool_registry` dict (name → callable/`StructuredTool`)
3. Convert `Flow` → `StateGraph` via `AgentSpecToLangGraphConverter`:
   - `StartNode` → graph entry point
   - `EndNode` → END sentinel
   - `LlmNode` → LangGraph node with LLM call
   - `ToolNode` → LangGraph `ToolNode` or scripted node
   - `AgentNode` → nested ReAct loop
   - `FlowNode` → compiled subgraph
   - `MapNode`/`ParallelMapNode` → fan-out with collection
   - `BranchingNode` → conditional edges
   - `ControlFlowEdge` → `add_edge` / `add_conditional_edges`
   - `DataFlowEdge` → state field routing (source → destination)
4. Wire in checkpointer, config, and optional `components_registry` for disaggregated parts

**Exporter (LangGraph → Agent Spec):**
1. Walk `StateGraph` structure via introspection
2. Emit `Flow` with `nodes`, `control_flow_connections`, `data_flow_connections`
3. Serialize each node to its Agent Spec equivalent
4. Optionally disaggregate components (split out LLM configs, tools) into separate JSON/YAML

### What they LOSE in both directions

**Fidelity ceiling (Agent Spec → LangGraph → Agent Spec):**
- **Graph structure is preserved**, but node-level detail may be lost:
  - Custom node functions → generic `ToolNode` with behavior bound by name
  - LangGraph-specific features (interrupts, custom reducers) have no Agent Spec representation
  - State schema → inferred from `DataFlowEdge`s only; full `TypedDict` not preserved
- **Tools**: only NAME survives; runtime binding happens via `tool_registry`
- **Metadata**: Agent Spec `metadata` survives round-trip, but LangGraph node introspection cannot recover everything

**Why this matters for neograph:**
- neograph's `Construct` IR is **richer than Agent Spec** (modifiers, typed edges, skip predicates, Oracle merge hooks)
- Export **must lower** modifiers to Agent Spec primitives (see §5 of interop design doc)
- Import **cannot reconstruct** modifiers from primitives — round-trip is lossy without `metadata` markers

### How neograph's typed IR can do BETTER

**Advantages over LangGraph adapter:**

1. **Edge inference from types**: LangGraph adapter requires explicit `DataFlowEdge` declarations (source_output → destination_input). neograph **infers** edges from typed `Node.inputs` signatures at assembly time. When importing Agent Spec, neograph can:
   - Read `DataFlowEdge`s
   - Build `Node.inputs` dict entries
   - Validate type compatibility at import time (catch mismatches before compile)

2. **Modifier-aware export**: LangGraph adapter deals with flat node graphs only. neograph's `to_agent_spec()` must **lower** modifiers:
   - `Each(over=...)` → `MapNode` with extracted sub-Flow
   - `Oracle(n=3, models=[...])` → `ParallelFlowNode` of 3 `LlmNode`s + merge node + fan-in/out edges
   - `Loop(when=...)` → `BranchingNode` + back-edge
   - This lowering is **neograph's value-add** — Agent Spec has no modifier concept

3. **Type registry round-trip**: neograph's `spec_types.py` (`_type_registry`, `register_type`, `lookup_type`) handles:
   - Export: Pydantic model → JSON Schema `Property` in Agent Spec
   - Import: `Property` JSON Schema → Pydantic model via `create_model`
   - **Verifies type compatibility** at import boundary (Agent Spec uses looser JSON Schema; neograph enforces Pydantic strictness)

4. **Validation at import vs runtime**: LangGraph adapter defers validation to LangGraph execution. neograph validates at `from_agent_spec()` time via `_validate_node_chain`, catching type mismatches **before** `compile()`.

**Single biggest advantage:** neograph validates **during import** what LangGraph adapter only discovers at runtime (or never). A malformed Agent Spec with mismatched `DataFlowEdge` types fails immediately in neograph, runs (possibly corruptly) in LangGraph adapter.

---

## Export lowering

**Modifiers → Agent Spec primitives** (this is neograph's compilation, already half-implemented for LangGraph):

| neograph modifier | Agent Spec lowering | Cost (what neograph hides) |
|---|---|---|
| `node \| Each(over="items", key="label")` | `MapNode` wrapping extracted sub-Flow + collection edges back to parent | Body extraction + `Send()` fan-out + state contract for dict[str, item] |
| `node \| Oracle(n=3, models=[...], merge_fn="combine")` | `ParallelFlowNode` of 3 `LlmNode` flows + separate merge `ToolNode` + fan-out/fan-in `DataFlowEdge`s | ~4-6 nodes + ~8 edges → 1 modifier line |
| Oracle per-variant model selection (`_oracle_model`) | **No representation** — N hand-instantiated `LlmNode`s with different `llm_config` | Static `LlmConfig` per node, dynamic per-variant choice lost |
| `node \| Loop(when="continue")` | `BranchingNode` with condition → `{continue: back-edge, done: next}` + cyclic `ControlFlowEdge` + self `DataFlowEdge` | 227-line pattern (Oracle's own how-to) compressed to 1 line |
| `node \| Operator(when="check")` | `BranchingNode(mapping={cond: PAUSE_BRANCH})` + `ControlFlowEdge` → `InputMessageNode` (pause branch) / reconverge (default branch) — verified against pyagentspec 26.1.2 (see 02-flows-nodes.md) | Interrupt semantics + boolean→string-key coercion for `should_pause` |

**Callable-valued fields do NOT serialize:**
- `raw_fn`, `skip_when`, `skip_value` (callables → reject or emit name/placeholder)
- Loop `when` as callable → Agent Spec only supports string conditions (registered in condition registry)
- Oracle merge hooks (`merge_pre_process`/`merge_post_process`/`merge_fallback`) → reject with error

---

## Import reconstruction

**Agent Spec → `Construct` primitives** (no modifier recovery):

| Agent Spec primitive | neograph import produces | Modifier recovery? |
|---|---|---|
| `Flow` | `Construct` (flat `nodes` list) | N/A — top-level container |
| `FlowNode` (nested) | sub-`Construct` in parent's `nodes` | N/A |
| `StartNode` | absorbed into `Construct.input` type | N/A |
| `EndNode` | absorbed into `Construct.output` type | N/A |
| `LlmNode` | `Node(mode="think")` | N/A |
| `AgentNode` (tools) | `Node(mode="agent" or "act")` with `tools=[]` | N/A |
| `ToolNode` / `ApiNode` | `Node(mode="scripted" or "raw")` | N/A |
| `MapNode` / `ParallelMapNode` | `Node` with inferred `Each`? **NO** — cannot infer `over=` or `key=` from primitive alone | **NO** — import as flat fan-out node |
| `BranchingNode` | `_BranchNode` sentinel in `nodes` | **NO** — cannot infer `Loop` or `Operator` from primitive alone |
| `ControlFlowEdge` | execution order in `nodes` list (inferred DAG) | N/A — order is order |
| `DataFlowEdge` | `Node.inputs` dict entries (upstream_name → type) | N/A — typed edges from explicit data flow |

**Cannot reconstruct modifiers from primitives:**
- `MapNode` → no way to know it was `Each(over="items")` vs hand-written fan-out
- `BranchingNode` with back-edge → no way to know it was `Loop(when="continue")` vs hand-written cycle
- `ParallelFlowNode` of N same-type nodes → no way to know it was `Oracle(n=N)` vs N hand-written nodes

**Import discipline (from §6a of interop design):**
1. If `neograph/source` metadata present → deserialize embedded Spec, re-run lowering, **diff** against actual flattened Flow
2. Match → use rich source (lossless)
3. Diverged → fall back to primitive import, reconstruct only groups whose per-group markers still match, **warn** about touched modifiers

---

## Dep recommendation

**USE `pyagentspec` as a dependency. Do NOT vendor.**

**Reasons (updated based on v26.1.2 API review):**

1. **Plugin architecture is non-trivial:** `AdapterAgnosticAgentSpecLoader` uses `ComponentSerializationPlugin`/`ComponentDeserializationPlugin` for extensibility. Vendoring means reimplementing this plugin registry, resolver chain, and builtin plugins. `pyagentspec` already has this battle-tested.

2. **Disaggregation is complex:** Two-phase loading with `import_only_referenced_components`, `components_registry` round-trips, and custom ID mapping `(component, "custom_id")`. Vendoring risks subtle bugs in reference resolution.

3. **Version-specific builtin plugins:** `AdapterAgnosticAgentSpecLoader` auto-selects "builtin plugins compatible with the latest supported Agent Spec version" when `plugins=None`. Vendoring freezes neograph to one version or requires manual plugin updates.

4. **Mitigation for dep size concerns:** If `pyagentspec` proves heavy, make it an **optional extra** like `mcp-examples`, not a core dependency. Core `loader.py` uses YAML parsing only; `from_agent_spec()` with full plugin support lives behind `[agent-spec]` extra.

5. **DO NOT depend on `pyagentspec.adapters.langgraph`:** That adapter is for LangGraph users. neograph should depend on `pyagentspec.serialization` (core Serializer/Deserializer) only, then build neograph-specific conversion logic.

**Implementation path:**
- Add optional extra: `agent-spec = ["pyagentspec>=26.1.0,<27"]` to `pyproject.toml`
- `from_agent_spec()` in `loader.py`: uses `AgentSpecDeserializer` → `Component` objects → neograph `Construct` conversion
- `to_agent_spec()` in new `_agent_spec.py`: walks neograph IR → `Component` objects → `AgentSpecSerializer` output
- Version gating: check `Flow.spec_version` against supported list at import time

---

## Verdict for interop

**Fidelity impact:** Import from Agent Spec will be primitive-level (no modifier recovery). Export lowers modifiers to primitives; behavior preserved, abstraction lost. Round-trip is lossy without `metadata` markers. LangGraph adapter has the same ceiling — it's an intrinsic limit of Agent Spec's flat node model, not an adapter bug.

**Single biggest risk:** **Oracle expansion** — Agent Spec is at v26.1.2 with rapid cadence (~quarterly). New node types (e.g., future `SwarmNode`, `RemoteAgentNode`) may not map cleanly to neograph concepts. The `Swarm`/`RemoteAgent` gap is already real (see wayflow-competitive-analysis §4a). Mitigation: version-gate imports, reject unknown components with clear errors, and track Agent Spec releases in neograph's own issue tracker.
