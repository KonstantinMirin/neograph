# Agent Spec → neograph mapping: Flows & Nodes

## Source

- **Agent Spec v26.1.2**: Oracle Open Agent Specification, the portable declarative IR for agentic systems ([GitHub](https://github.com/oracle/agent-spec), [Documentation](https://oracle.github.io/agent-spec/development/agentspec/index.html))
- **neograph source**: `src/neograph/node.py`, `src/neograph/construct.py`, `src/neograph/modifiers.py`, `src/neograph/compiler.py`, `src/neograph/_wiring.py`, `src/neograph/_spec_schema.py`
- **WayFlow competitive analysis**: `docs/design/wayflow-competitive-analysis-2026-07-09.md`

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|------------------|--------------|---------------------|-----------------|-----------|--------|
| **Flow** | Container with start_node, nodes[], control_flow_connections[], data_flow_connections[] | StateGraph | Construct (nodes list + inferred DAG) | LangGraph compiles Construct→StateGraph | DIRECT |
| **DataFlowEdge** | Explicit data wiring: source_node.source_output → destination_node.destination_input | N/A (inferred from state bus) | Inferred from Node.inputs dict (type-channel edges) | StateBus reads upstream by field_name | GAP-AS (explicit→implicit) |
| **ControlFlowEdge** | Explicit execution order: from_node[.from_branch] → to_node | add_edge / add_conditional_edges | Inferred from Construct.nodes order (sequential execution) | Compiler walks nodes list | GAP-AS (explicit→implicit) |
| **StartNode** | Flow entry point, defines inputs | START | Construct.input (boundary port) | Boundary port creates isolated state | DIRECT |
| **EndNode** | Flow exit point, defines outputs | END | Construct.output (boundary port) | Boundary port surfaces from sub-construct | DIRECT |
| **LlmNode** | Single LLM call with prompt template + LlmConfig | Node | Node(mode="think") with model=, prompt= | think mode dispatch via ThinkDispatch | DIRECT |
| **AgentNode** | Run an Agent (ReAct or other agentic component) | Node + ToolNode | Node(mode="agent") with tools= OR Node(mode="act") | agent/act modes compile to ReAct cycle (agent/tools/parse nodes) | DIRECT |
| **ToolNode** | Execute a tool (ClientTool/ServerTool/MCPTool/BuiltinTool) | ToolNode | Node.scripted() OR Tool spec with mcp_tool_factory | scripted nodes; MCP tools bound via mcp_tool_factory | DIRECT |
| **ApiNode** | Perform API call with config, provide parts of response as outputs | Node | Node.scripted() (scripted fn does HTTP) | Scripted function uses httpx/aiohttp | DIRECT |
| **BranchingNode** | Conditional branching: input value → branch mapping | add_conditional_edges | Operator(when=...) OR ForwardConstruct if/elif/else | Operator compiles to check node + conditional edge | DIRECT (Operator) |
| **MapNode** | Map-reduce: apply Flow to each collection element, reduce outputs | Send() + barrier | Each(over=..., key=...) modifier | Each compiles to router + Send() per item + barrier | DIRECT |
| **ParallelMapNode** | Concurrent map-style workflows | Send() + barrier (same as MapNode) | Each(over=...) modifier (no separate primitive) | Each already parallel; no distinction needed | DIRECT |
| **ParallelFlowNode** | Concurrent workflow execution (N independent flows in parallel) | Send() (one per branch) | Oracle(n=N, models=[...]) WITHOUT merge_fn | Oracle fans out N variants; merge_fn makes it ensemble | LOWER (Oracle) |
| **FlowNode** | Run sub-flow inline (nesting) | compiled subgraph as node | sub-Construct in Construct.nodes | make_subgraph_fn compiles child Construct | DIRECT |
| **InputMessageNode** | Interrupt flow, retrieve user input, append as user message | interrupt() | N/A (LangGraph interrupt exposed, no node) | LangGraph interrupt() available but not wrapped as node | GAP-AS (no node type) |
| **OutputMessageNode** | Append agent message to conversation | Node | N/A (conversation not a neograph concern) | neograph doesn't model conversation; state bus only | GAP-AS (out of scope) |
| **CatchExceptionNode** | Explicit exception-handling branches (try/catch pattern) | add_conditional_edges (error path) | try/except in ForwardConstruct body | ForwardConstruct supports try blocks | DIRECT (try/except) |
| **RemoteAgent** / **A2AAgent** | Cross-runtime agent via REST/RPC (A2A protocol) | Node | Node.scripted() (HTTP call to remote) | No first-class remote agent; scripted fn can call A2A endpoint | GAP-AS (no first-class support) |
| **Swarm** | Dynamic decentralized agent handoff (agents choose next at runtime) | Command(goto) / Send() | N/A (no Swarm combinator) | LangGraph Command/Send support it; neograph doesn't expose | GAP-AS (addable combinator) |

## Status legend used

- **DIRECT**: Agent Spec class maps directly to a neograph concept with equivalent semantics. Round-trip preserves structure.
- **LOWER**: Agent Spec class has no direct neograph equivalent; requires lowering from neograph's higher-level modifiers (Each, Oracle, Loop, Operator) into Agent Spec primitives. Import cannot reconstruct the modifier.
- **GAP-AS**: Agent Spec class has no neograph equivalent (or vice versa) due to fundamental design differences. Out of scope for interop or requires user-side shim.

## Serialization notes

### neograph Spec is modifier-aware; Agent Spec is flat

The neograph `Spec` (`_spec_schema.py`) carries first-class modifier slots:
- `NodeSpec.oracle: OracleSpec | None`
- `NodeSpec.each: EachSpec | None`
- `NodeSpec.loop: LoopSpec | None`
- `NodeSpec.operator: OperatorSpec | None`

Agent Spec has **no modifier concept** — it is flat primitives (Flow, nodes, edges). Export **lowers** modifiers into Agent Spec primitives (MapNode, ParallelFlowNode, BranchingNode, etc.). Import produces **primitive-level** constructs; modifiers cannot be reconstructed.

### Type representation

- **neograph**: Typed Pydantic models (inputs/outputs as `type[BaseModel]` or dict[str, type])
- **Agent Spec**: JSON Schema via `Property` (title, type, default, description)
- **Round-trip**: Type registry (`spec_types.py`) converts both ways; exported types must be registered names or inline JSON-Schema `types` sections.

### Callable fields don't serialize

These neograph IR fields are **name-only** in spec (callable itself never serializes):
- `Node.scripted_fn: str | None`
- `Node.raw_fn: RawNodeFn | None` — name/placeholder only
- `Loop.when: str | Callable[[Any], bool]` — string name when serialized
- `Oracle.merge_fn: str | None`
- `Oracle.merge_pre_process`, `merge_post_process`, `merge_fallback` — names only

Export must reject with clear error OR emit a name/placeholder (same as current spec behavior).

### Metadata markers enable lossless round-trip

Every Agent Spec `Component` carries `metadata: dict[str, Any]` — preserved through serialize→deserialize. Use `neograph/` namespace:
- **Layer A (per-group)**: stamp lowered groups with `"neograph/modifier": "oracle"`, `"each"`, `"loop"`, full modifier spec + `group_id`
- **Layer B (embed)**: `Flow.metadata["neograph/source"]` = original neograph Spec verbatim

Import **verifies, doesn't trust**: diff embedded Spec against flattened Flow; fall back to primitive-level import on divergence (§6a in design doc).

## Export lowering

For each neograph modifier and mode, the concrete Agent Spec expansion it lowers to:

### Each modifier → MapNode

```
node | Each(over="clusters.clusters", key="label")
```

Lowers to:
- **Extracted sub-Flow** with one `LlmNode` (or the original node's compiled form)
- `MapNode` wrapping the sub-Flow
- Data edges connecting upstream collection → MapNode input
- MapNode output → downstream consumer

**Cost neograph hides**: body extraction + collection contract (`iterated_`/`collected_` naming).

### Oracle (no merge_fn) → ParallelFlowNode

```
node | Oracle(n=3, models=["reason", "fast", "large"], merge_prompt="judge")
```

Lowers to:
- **N separate `LlmNode` flows** (one per model, or N copies of same model)
- `ParallelFlowNode` containing N flows
- Merge `LlmNode` (for merge_prompt) or `ToolNode` (for merge_fn)
- Fan-out ControlFlowEdges: source → each variant flow
- Fan-in DataFlowEdges: each variant output → merge node inputs

**Cost neograph hides**: ~4-6 nodes + ~8 edges → 1 modifier line. No representation for per-variant model selection in Agent Spec.

### Oracle (with merge_fn) → ParallelFlowNode + scripted merge

Same as above, but merge node is a `ToolNode` (scripted function) instead of `LlmNode`.

### Loop → BranchingNode + cyclic edges

```
node | Loop(when="score_low", max_iterations=5)
```

Lowers to:
- `BranchingNode` with condition → `{continue: back-edge, done: next}`
- Cyclic `ControlFlowEdge`: last node → back to target (self-loop or sub-construct start)
- Self `DataFlowEdge`: output → input (for Node-level Loop)

**Cost neograph hides**: the full 227-line pattern (explicit branch + cycle wiring vs one modifier).

### Operator → BranchingNode + gated edge

```
node | Operator(when="needs_approval")
```

Lowers to:
- Check node (condition → interrupt or bypass)
- `BranchingNode` routing
- LangGraph `interrupt()` on true branch

**Status**: exact lowering still TBD in implementation (gated edge vs explicit check node).

### Mode mapping

| neograph mode | Agent Spec node | Notes |
|----------------|-----------------|-------|
| `think` | `LlmNode` | Direct: model + prompt + output |
| `agent` | `AgentNode` | Agent with read-only tools |
| `act` | `AgentNode` | Agent with mutation tools (side effects) |
| `scripted` | `ToolNode` / `ApiNode` | Behavior name-bound at runtime |
| `raw` | `ToolNode` | LangGraph escape hatch (state, config) → dict |

### dict-form outputs → multiple DataFlowEdges

```
Node(outputs={"result": Claims, "tool_log": list[ToolInteraction]})
```

Lowers to:
- Two output fields: `{node}_result`, `{node}_tool_log`
- Two DataFlowEdges per consumer referencing each field

## Import reconstruction

For each Agent Spec node, the neograph Construct/Node/modifier it imports to:

| Agent Spec node | Imported neograph form | Reconstructability |
|-----------------|------------------------|-------------------|
| **Flow** | Construct(nodes=[...]) | Full reconstruction |
| **StartNode** | Construct.input (boundary port) | Direct: type from StartNode inputs |
| **EndNode** | Construct.output (boundary port) | Direct: type from EndNode outputs |
| **LlmNode** | Node(mode="think", model=, prompt=) | Direct: primitive → primitive |
| **AgentNode** | Node(mode="agent" or "act", tools=) | Direct: tools list → Tool specs |
| **ToolNode** | Node.scripted(scripted_fn=) | Direct: name → scripted fn |
| **ApiNode** | Node.scripted(scripted_fn=) | Direct: name → scripted fn (HTTP call) |
| **BranchingNode** | Operator(when=) OR _BranchNode | Partial: condition reconstructed, but not Loop's cyclic edge |
| **MapNode** | Each(over=, key=) modifier | **CANNOT** be inferred as modifier without per-group metadata |
| **ParallelFlowNode** | Oracle(n=N, models=[...]) | **CANNOT** be inferred as modifier; looks like N parallel flows |
| **ParallelMapNode** | Each(over=, key=) modifier | Same as MapNode |
| **FlowNode** (nested) | sub-Construct in Construct.nodes | Direct: nested Flow → sub-Construct |
| **CatchExceptionNode** | try/except in ForwardConstruct | **NOT** reconstructable as explicit node; requires ForwardConstruct parsing |
| **InputMessageNode** | N/A (gap) | Out of scope |
| **OutputMessageNode** | N/A (gap) | Out of scope |
| **Swarm** | N/A (gap, addable) | No neograph equivalent; would need new combinator |
| **RemoteAgent/A2AAgent** | Node.scripted() | Not first-class; scripted fn calls A2A endpoint |

### Groupings that CANNOT be re-inferred as modifiers

Even with `metadata` markers, these patterns **cannot** be reconstructed without the embedded neograph source:

1. **N parallel LlmNodes + merge → Oracle**: Without `neograph/modifier="oracle"`, this looks like N independent LLM nodes + a scripted merge. The "ensemble" relationship is lost.
2. **MapNode → Each**: MapNode is distinguishable from a generic LlmNode, but the Each modifier syntax (`over` dotted path, `key` dispatch) is not derivable from the iterated_`/`collected_` naming convention alone.
3. **BranchingNode + cycle → Loop**: A BranchingNode with a back-edge ControlFlowEdge could be a Loop OR just a conditional branch. Only `metadata["neograph/modifier"]="loop"` disambiguates.

### Import discipline with metadata markers

1. **If `neograph/source` present**: deserialize embedded Spec, re-run lowering, diff against actual Flow. Match → use rich source (lossless).
2. **If diverged** (hand-edited): actual Flow wins; reconstruct only groups whose per-group markers still match primitives; warn about broken modifiers.
3. **If no `neograph/source`**: import as primitives (Construct with scripted Nodes, no modifiers).

## Verdict for interop

**Fidelity impact**: Export is faithful but flattening (modifiers lower to primitives; abstraction lost). Import produces primitives, not sugar. Lossless round-trip is possible **only** with metadata markers (`neograph/modifier` per-group + `neograph/source` embed). Without markers, modifiers cannot be reconstructed — the graph runs correctly, but the high-level authoring intent is lost.

**Single biggest risk/gap**: **Oracle → ParallelFlowNode expansion is not reversible**. N parallel LlmNodes + merge node cannot be distinguished from "hand-authored parallel flows" without metadata. This is the flagship gap that demonstrates why neograph sits above Agent Spec as an authoring layer. Secondary gap: **Swarm / RemoteAgent** are real Agent Spec components with no neograph equivalent (addable via combinators or scripted shims, but not first-class).

**Sources**: [Agent Spec GitHub](https://github.com/oracle/agent-spec), [Agent Spec Documentation](https://oracle.github.io/agent-spec/development/agentspec/index.html), [A2A Protocol Specification](https://github.com/a2aproject/A2A/blob/main/docs/specification.md)
