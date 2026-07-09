# Agent Spec → neograph mapping: Tracing

## Source

**Oracle Open Agent Spec Tracing** (`pyagentspec.tracing`, v26.1.2): https://oracle.github.io/agent-spec/development/agentspec/tracing.html

Defines runtime observability primitives: `Event` (point-in-time facts), `Span` (time-bounded execution contexts that group events), `SpanProcessor` (hooks to consume spans/events), `Trace` (root grouping). Standard span types: `LlmGenerationSpan`, `ToolExecutionSpan`, `AgentExecutionSpan`, `FlowExecutionSpan`, `NodeExecutionSpan`, `SwarmExecutionSpan`, `ManagerWorkersExecutionSpan`. Standard event types cover LLM, tool, agentic component, flow, conversation/state/control.

## neograph observability survey (what neograph emits today)

neograph has **no first-party span model**. Observability is entirely delegated to the LangChain/LangGraph callback ecosystem:

1. **`_trace.py`** (`named()` function): Binds `run_name`, `tags`, `metadata` to LangGraph node `Runnable`s via `.with_config()`. This decorates LangChain callback spans with neograph-specific attributes (`neograph:node` tag, `neograph_mode`, `neograph_output_type`, `neograph_node_id` metadata). The binding is static and late — no runtime branching, no tracer dependency, no neograph-emitted OTEL spans.

2. **`_execute.py`**: Emits structlog logs (`node_start`, `node_complete`) with run_id correlation (`_run_id_binds` extracts `RUN_ID` from config for trace correlation). These are separate from callback spans.

3. **`factory.py`**: Wraps node functions in `RunnableLambda` and applies the `named()` binding before returning to LangGraph.

4. **Optional langfuse extra**: `pyproject.toml` declares `langfuse>=3.0` as an optional dependency. neograph reaches Langfuse/OTel backends **via LangChain/LangGraph callbacks**, not via a first-party span emitter.

**No `Trace`/`Span`/`Event`/`SpanProcessor` classes exist in neograph.** The framework relies entirely on the LangChain callback protocol (`on_chain_start`/`on_chain_end`) for span trees and any attached Langfuse/OTel/SDK processors.

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|------------------|--------------|---------------------|------------------|-----------|--------|
| `Trace` | Root grouping of spans for one assistant execution; holds SpanProcessors; calls startup/shutdown | `Runnable.invoke()` / `.ainvoke()` with `config["callbacks"]` | `run()` / `arun()` driver in `runner.py` | LangChain/LangGraph callback channel | GAP-AS (structural) / NOT-A-GAP (functional) |
| `Span` | Time-bounded execution context; start/end; contains events; nestable | LangChain `Runnable` callback span (`on_chain_start`/`on_chain_end`) | Node execution wrapped in `RunnableLambda` via `factory.make_node_fn()` | LangChain Core `RunnableConfig.callbacks` | GAP-AS (structural) / NOT-A-GAP (functional) |
| `SpanProcessor` | Hook interface (`on_start`, `on_end`, `on_event`, `startup`, `shutdown`) to consume spans/events | LangChain `BaseCallbackHandler` / `RunManager` | Callbacks attached via `config["callbacks"]` | LangChain `langchain_core.callbacks` | GAP-AS (structural) / NOT-A-GAP (functional) |
| `Event` | Point-in-time fact within a span (timestamped metadata) | LangChain callback events (e.g., `on_llm_start`, `on_tool_start`, `on_llm_end`) | LLM/tool events emitted by LangChain internals; neograph does not emit custom events | LangChain callback system | GAP-AS (structural) / NOT-A-GAP (functional) |
| `FlowExecutionSpan` | Covers entire Flow execution (StartNode → EndNode) | Top-level `StateGraph.invoke()` span | Graph compilation + execution in `compile()` + `runner.run()` | LangGraph `StateGraph` | GAP-AS (structural) / NOT-A-GAP (functional) |
| `NodeExecutionSpan` | Covers single Node execution within a Flow | Per-node callback span (one per LangGraph node) | `factory.make_node_fn()` wraps each `Node` in a span via `named()` | LangGraph node callback | GAP-AS (structural) / NOT-A-GAP (functional) |
| `LlmGenerationSpan` | LLM generation request → response | LangChain `on_llm_start`/`on_llm_end` callbacks | Emitted by LangChain LLM integrations; neograph does not wrap | LangChain LLM callbacks | GAP-AS (structural) / NOT-A-GAP (functional) |
| `ToolExecutionSpan` | Tool execution request → response | LangChain `on_tool_start`/`on_tool_end` callbacks | Emitted by LangChain `ToolNode` / tool wrappers | LangChain tool callbacks | GAP-AS (structural) / NOT-A-GAP (functional) |
| `AgentExecutionSpan` | Agent execution (may be nested for sub-agents) | Agent node callback span | Agent/act nodes (`ReAct` cycle) generate spans via LangGraph | LangGraph agent patterns | GAP-AS (structural) / NOT-A-GAP (functional) |
| `SwarmExecutionSpan` | Swarm component execution (dynamic decentralized handoff) | N/A (LangGraph `Command(goto)`/`Send` can emulate) | **No neograph equivalent today**; Swarm is a REAL WayFlow gap (§4a in competitive analysis) | LangGraph `langgraph-swarm` pattern | **REAL GAP** (not in mapping) |
| `ManagerWorkersExecutionSpan` | Manager-Workers pattern execution | N/A (can be expressed as Flow with BranchingNode) | **No neograph primitive**; expressible as router node + worker sub-constructs + loop | Neograph combinator pattern | **ADDABLE PRIMITIVE** (not a gap) |
| `StateSnapshotEmitted` event | Point-in-time state snapshot for resumability/debugging | LangGraph `get_state_history()` / `StateSnapshot` | **Checkpoint auto-rewind** (`schema_fingerprint`, `node_fingerprints`) emits snapshots implicitly | LangGraph checkpointer | **NOT-A-GAP** (different mechanism, same end) |

## Status legend used

- **GAP-AS (structural) / NOT-A-GAP (functional)**: Agent Spec defines first-party span/event classes; neograph delegates to LangChain/LangGraph callbacks. The **semantic surface is covered** (spans fire, events propagate), but the **API surface differs** (no `Trace`/`Span`/`Event` classes, no `SpanProcessor` interface in neograph). Functionally equivalent — tracing works via Langfuse/OTel because the LangChain callback channel is shared.
- **REAL GAP**: Agent Spec concept has no neograph counterpart (e.g., `SwarmExecutionSpan` — dynamic decentralized handoff is not a neograph primitive).
- **ADDABLE PRIMITIVE**: Can be expressed via existing neograph constructs (e.g., `ManagerWorkersExecutionSpan` → router + workers + loop), but not a named primitive.
- **NOT-A-GAP**: Covered by a different mechanism with the same end result (e.g., `StateSnapshotEmitted` → neograph's checkpoint fingerprint + auto-rewind).

## Serialization notes

**Tracing is RUNTIME-ONLY.** The `Trace`/`Span`/`Event`/`SpanProcessor` classes are **not serialized** in Agent Spec. They are emitted during execution by the runtime adapter (`pyagentspec.adapters.langgraph`) and consumed by backends via `SpanProcessor` hooks. They have **no representation in the Agent Spec YAML/JSON wire format** (which declares only `Flow`/`Agent`/`Node` components).

**Checkpoint metadata is similarly neograph-only** (design §6: "Checkpoint metadata (schema/node fingerprints) is neograph-only; not represented in Agent Spec. Out of scope for the wire format.").

## Export lowering

**Not applicable** — tracing does not exist in the Agent Spec wire format, so export/import never encounters it. Lowering is irrelevant for `Trace`/`Span`/`Event`/`SpanProcessor`.

## Import reconstruction

**Not applicable** — Agent Spec YAML/JSON carries no tracing constructs. An imported `Flow` contains only `nodes`, `control_flow_connections`, `data_flow_connections`. Tracing spans are emitted by the **runtime adapter** during execution, not reconstructed from the spec.

## Verdict for interop

**Tracing is OUT OF SCOPE for the Agent Spec wire format.** The `Trace`/`Span`/`Event`/`SpanProcessor` classes are runtime-only constructs emitted by adapters, not serialized pipeline components. This is explicitly stated in the design document (§6: "Checkpoint metadata...is neograph-only; not represented in Agent Spec. Out of scope for the wire format"). Tracing has **zero fidelity impact** on export/import because it never crosses the serialization boundary.

**Functional parity via shared callback channel.** neograph and Agent Spec runtimes both emit traces into the **LangChain/LangGraph callback ecosystem**, which means both reach Langfuse/OTel via the same mechanism. The single biggest interop "risk" is that neograph's callback binding (`_trace.named`) stamps different metadata (`neograph:node`, `neograph_mode`, etc.) than `pyagentspec.adapters.langgraph` would stamp, but this is purely cosmetic — the span tree structure and timing are identical.

**SwarmExecutionSpan is the one genuine tracing gap** (dynamic decentralized handoff has no neograph primitive today), but this is a **runtime capability gap**, not a serialization/interop gap. If neograph adds Swarm-as-combinator via LangGraph `Command(goto)`, the tracing span would automatically fire via the callback channel.
