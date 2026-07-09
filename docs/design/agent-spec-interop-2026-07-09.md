# neograph ↔ Open Agent Spec interop

Date: 2026-07-09
Status: design / decision record (pre-epic)
Author: Constantine Mirin (with agent-assisted analysis)

---

## 1. Context and decision

Oracle shipped **Open Agent Spec** (`oracle/agent-spec`, v26.1.x) — a portable
declarative IR for agentic systems (JSON/YAML), plus **WayFlow**
(`oracle/wayflow` / `wayflowcore`), its reference authoring library + runtime.
Oracle's own framing: "ONNX for agents" — define once, run on
LangGraph / AutoGen / CrewAI / WayFlow via runtime adapters.

**Decision:** neograph will **import from and export to Agent Spec** as an
*interop layer*, rather than invent (or keep expanding) a private serialization
dialect for cross-ecosystem exchange. This gives the LLM-driven
runtime-construction use case a wire format the ecosystem already speaks, and
positions neograph as "the typed, inference-based authoring layer and validating
compiler that emits the standard" instead of "another LangGraph wrapper."

**Non-goal:** standard-setting. neograph's value is developer experience + the
**type-channels** paradigm (typed nodes, edges inferred from signatures,
compile-time "if it compiles it runs" validation). We speak the standard; we do
not try to own it. Distribution is not the success metric (sole downstream =
piarch + maintainer).

---

## 2. Landscape: three layers, and where neograph sits

| Layer | Oracle piece | Analogy | neograph counterpart |
|---|---|---|---|
| Interchange format | Agent Spec (YAML/JSON IR) | ONNX | this feature's target |
| Authoring lib + runtime | WayFlow (`wayflowcore`) | PyTorch | **neograph itself** |
| Backend | LangGraph / AutoGen / CrewAI adapters | CUDA | LangGraph (compile target) |

**WayFlow — not Agent Spec — is the layer-peer competitor.** The core divergence
is the entire neograph thesis: Agent Spec / WayFlow default to **explicit
hand-declared edges** (`ControlFlowEdge` + `DataFlowEdge`; their conditional-branch
how-to is 227 lines); neograph **infers** edges from typed signatures. Convergent
evolution, opposite core bet.

Notably, Agent Spec's name-based data-flow mode (`data_flow_connections = None`
→ components read/write a shared variable space by name) **is neograph's state
bus.** neograph ≈ Agent Spec's implicit name-based mode promoted to the only
mode, with edges inferred from types + assembly-time validation the spec lacks.

### Agent Spec component model (target surface)

- Agentic components: `Agent` (ReAct), `Swarm`, `ManagerWorkers`, `RemoteAgent`,
  `A2AAgent`, `SpecializedAgent`.
- `Flow` = `start_node`, `nodes[]`, `control_flow_connections[]`,
  `data_flow_connections[]`.
- Node library: `StartNode`, `EndNode`, `LlmNode`, `ToolNode`, `AgentNode`,
  `FlowNode` (nested), `MapNode` / `ParallelMapNode`, `ParallelFlowNode`,
  `BranchingNode`, `ApiNode`, `CatchExceptionNode`, `Input/OutputMessageNode`.
- Types: `Property` carrying JSON Schema (`title`, `type`, `default`, `description`).
- Edges: `ControlFlowEdge(from_node, from_branch, to_node)`,
  `DataFlowEdge(source_node, source_output, destination_node, destination_input)`.
- Tools: `ClientTool` / `ServerTool` / `RemoteTool` / `MCPTool` / `BuiltinTool`,
  `ToolBox` / `MCPToolBox`. Spec names the tool; runtime binds the impl (WayFlow's
  `AgentSpecLoader(tool_registry={...})`) — **exactly neograph's name-vs-factory split.**

---

## 3. Current neograph serialization surface (grounded)

- **Loader** (import-only): `src/neograph/loader.py` (`load_spec`). One-way:
  dict/YAML → `Spec` → `_build_construct` → `Construct`.
- **Typed schema:** `src/neograph/_spec_schema.py`. Generated JSON Schema at
  `src/neograph/schemas/neograph-pipeline.schema.json` (regen via
  `scripts/regen_schema.py`).
- **Type registry:** `src/neograph/spec_types.py` — module-global
  `_type_registry`; `register_type` / `lookup_type` / `load_project_types`
  (builds Pydantic models from JSON-Schema-ish `types` dicts, supports `$ref`).
- **No export path exists.** No `to_dict` / `to_spec` / `Construct.to_spec()`.
  Export is greenfield.

### Critical property: neograph `Spec` is HIGH-LEVEL and modifier-aware

`NodeSpec` and `ConstructSpec` both carry first-class modifier slots
`oracle` / `each` / `loop` / `operator` (`OracleSpec`, `EachSpec`, `LoopSpec`,
`OperatorSpec`). The neograph YAML keeps `Oracle` as **one line**. Agent Spec has
**no modifier concept** — it is flat primitives.

The neograph `Spec` is also a **subset** of the full IR: it cannot express
`renderer`, `skip_when`/`skip_value`, `gate_tools_when`, `fan_out_param`,
`raw_fn`, `oracle_gen_type`, dict-form `Node.outputs`, Oracle merge hooks, Each
`on_error`, Loop `history`, `Tool.idempotent`, and all callable-valued fields.

---

## 4. DECISION: interop layer, not replacement

**Do NOT retarget the neograph `Spec` at Agent Spec.** Doing so would discard the
modifier-aware authoring DX (one-line `oracle:`/`each:`/`loop:`) that is a
neograph value-add. Instead:

| Format | Role | Level | Keep/Add |
|---|---|---|---|
| neograph `Spec` (`_spec_schema.py`) | authoring YAML (humans + LLMs building neograph pipelines) | high (modifier-first) | **keep** |
| Agent Spec | cross-ecosystem interchange (export target + import source) | low (flat primitives) | **add** |

Export **lowers** modifiers into Agent Spec primitives (same lowering neograph
already does when compiling to LangGraph). Import produces **primitive-level**
constructs (no modifier recovery).

---

## 5. IR ↔ Agent Spec mapping

### Structural (clean-ish)

| neograph IR | Agent Spec |
|---|---|
| `Construct` | `Flow` |
| `Construct.input` / `.output` (boundary ports) | `StartNode` inputs / `EndNode` outputs |
| `Node` mode `think` | `LlmNode` (+ `llm_config`) |
| `Node` mode `agent`/`act` (tools) | `Agent` / `AgentNode` (+ tools) |
| `Node` mode `scripted`/`raw` | `ToolNode` / `ApiNode` (behavior name-bound at runtime) |
| sub-`Construct` in `nodes` | `FlowNode` (nested) |
| `Node.inputs` dict (inferred) | explicit `DataFlowEdge`s (source_output → destination_input) |
| execution order (inferred DAG) | explicit `ControlFlowEdge`s |
| typed I/O (Pydantic) | `Property` (JSON Schema) via type registry |
| `Tool(name, budget, config)` | `Tool` / `MCPTool` (name-only; factory binds at runtime) |

### Helper lowerings (export-only; the neograph value-add)

| neograph helper | Agent Spec lowering | Cost neograph hides |
|---|---|---|
| `node \| Each(over=…)` | `MapNode` / `ParallelMapNode` wrapping an **extracted sub-Flow** + collection edges | body extraction + collection contract |
| `node \| Oracle(models=[…], merge_fn=…)` | **no node** — `ParallelFlowNode` of N single-`LlmNode` flows (N `LlmConfig`s) + a merge `LlmNode`/`ToolNode` + fan-out/fan-in edges | ~4-6 nodes + ~8 edges → 1 modifier |
| Oracle per-variant model selection (`_oracle_model`) | **no representation** — N hand-instantiated `LlmNode`s | static `LlmConfig` per node |
| `… \| Loop(when=…)` | `BranchingNode` (condition → `{continue: back-edge, done: next}`) + cyclic `ControlFlowEdge`s + self `DataFlowEdge`s | the whole 227-line pattern |
| `Operator(when=…)` | condition → `BranchingNode` / gated edge (exact lowering TBD in impl) | — |

**`Oracle` is the flagship gap:** ensemble / best-of-N + per-variant model
selection has NO Agent Spec node. It only survives export by expansion. Advertise
it — it's the clearest demo of why neograph sits above the flat node library.

---

## 6. Fidelity analysis

- **Export (IR → Agent Spec): faithful but flattening.** Modifiers lower to
  primitives; behavior preserved, abstraction lost. This is normal compilation.
- **Import (Agent Spec → IR): primitives, not sugar.** Cannot re-infer "3 parallel
  LlmNodes + merge = Oracle." Fine for running an external Agent Spec.
- **Callable-valued IR fields do not serialize:** `raw_fn`, `skip_when`,
  `skip_value`, Loop `when`-as-callable, Oracle merge hooks
  (`merge_pre_process`/`merge_post_process`/`merge_fallback`), `renderer`. The
  exporter must reject with a clear error OR emit a name/placeholder (mirrors how
  `scripted_fn` / `merge_fn` / `operator.when` are already name-only).
- **Types must round-trip** through the string registry (`spec_types.py`):
  exported `inputs`/`outputs` need registered names or an inline JSON-Schema
  `types` section; imported Agent Spec `Property` JSON Schema must register into
  the same registry.
- **`_BranchNode` sentinels** (from `ForwardConstruct`) can appear in
  `Construct.nodes` — the exporter must handle (→ `BranchingNode`) or reject.
- **Checkpoint metadata** (schema/node fingerprints) is neograph-only; not
  represented in Agent Spec. Out of scope for the wire format.

---

## 7. Ownership and layering

Per CLAUDE.md layer discipline, `node.py` / `construct.py` /
`_construct_validation.py` / `factory.py` / `modifiers.py` are **off-limits** for
new surface features. So:

- **`from_agent_spec()` (import)** → next to `load_spec` in `loader.py`
  (the import home). Translates Agent Spec → `Construct` via `_build_construct`
  patterns.
- **`to_agent_spec()` (export)** → **free function**, NOT an IR method. Add
  `src/neograph/_agent_spec.py` (advisory-underscore), exposed via
  `neograph/__init__.py.__all__` (the real public contract). Walk the IR via the
  existing `iter_nodes` / `iter_with_arms` helpers (`construct.py`); read fields
  per §5.
- **MCP tool binding stays runtime-only.** Spec carries tool NAME (+ budget/config)
  only; `mcp_tool_factory` (`src/neograph_mcp/_client.py`) binds at compile time.
  This mirrors Agent Spec's own name-vs-`tool_registry` split — a natural fit.

---

## 8. Strategic follow-on: the MCP typed-binding seam

Bet on the **seam**, not the gateway. Agent Spec explicitly positions MCP as
*complementary* — the tool layer is white space it does not claim. neograph's
defensible niche = "MCP tools bound into type channels with compile-time
validation + DI + self-heal," the intersection where WayFlow (trivial in-process
`tool_registry`) and MCP gateways (provisioning, no typed node graph) both stop.
Durable value = typed binding, NOT papering over `mcp` SDK gaps (those close as
the SDK matures). Mapping `Tool`/`MCPTool`/`ToolBox` ↔ neograph's tool surface is
a natural extension of this feature and reinforces the positioning.

---

## 9. Open risks / to verify

1. **License / governance** of `oracle/agent-spec` + `wayflowcore` before any hard
   dependency (UPL? Apache? Oracle CLA?). Blocks making it a required dep.
2. **Round-trip fidelity gate:** confirm the §5/§6 mapping against the real
   adapters (Agent Spec's own LangGraph adapter) — does export→import→compile
   preserve behavior on an `Each` + `Oracle` + `Loop` pipeline?
3. **`Operator` exact lowering** — pin its semantics and Agent Spec target.
4. Whether to depend on `pyagentspec` for the schema/models or vendor a minimal
   subset (avoid a heavy Oracle dep in core; possibly an optional extra like
   `mcp-examples`).

---

## 10. Proposed epic shape

Epic: **neograph ↔ Open Agent Spec interop (import + export)**. Children:

1. **[GATE] Fidelity mapping audit + interop-vs-replacement ratification** —
   ratify §4/§5/§6, resolve `Operator` lowering + `pyagentspec`-vs-vendor,
   independent review before any serializer code.
2. **License/governance spike** — verify §9.1; gate on making it a dependency.
3. **Type layer** — `Property`(JSON Schema) ↔ `spec_types` registry round-trip.
4. **`from_agent_spec()` import** — Agent Spec `Flow`/`Agent` → `Construct`
   (primitive-level), in `loader.py`.
5. **`to_agent_spec()` export** — IR → Agent Spec `Flow`; lower modifiers;
   reject/placeholder callable fields; `_BranchNode` handling; new
   `_agent_spec.py` free function, wired through `__init__.__all__`.
6. **Tool/MCP mapping** — `Tool`/`MCPTool` name-only serialization; factory
   binding stays runtime (§7, §8).
7. **Round-trip tests + fixtures** — golden Agent Spec docs; export→import→compile
   equivalence; three-surface parity; `should_pass`/`should_fail` fixtures.
8. **Docs + example** — website page + runnable example showing `Each`+`Oracle`+
   `Loop` exported to Agent Spec **side-by-side** (fidelity spec AND marketing
   asset).
