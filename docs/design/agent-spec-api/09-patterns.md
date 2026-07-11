# Agent Spec → neograph mapping: Agentic Patterns

## Source (URL fetched)
- API Index: https://oracle.github.io/agent-spec/26.1.2/api/index.html
- DeepWiki (Swarm): https://deepwiki.com/search/what-does-the-swarm-class-do-w_622ddbed-2747-4357-b88a-b8f6dff48c99
- WayFlow analysis: `/Users/konst/projects/neograph/docs/design/wayflow-competitive-analysis-2026-07-09.md` §4a

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|------------------|--------------|-------------------|-----------------|-----------|--------|
| `pyagentspec.swarm.Swarm` | Dynamic decentralized multi-agent handoff. Agents choose next recipient at runtime from a predefined relationship graph (`first_agent` + `relationships` list of (caller, recipient) tuples). Beta in Agent Spec 26.1.2. | `Command(goto)` / `Send` API — LangGraph's runtime routing primitives. `langgraph-swarm` demonstrates Swarm-style handoff. | **GAP-AS** (no first-class Swarm combinator). Expressible via: router node + `Send()` fan-out (neograph already uses `Send` for Each modifier in `_wiring.py`) + relationship-as-config. | langgraph-swarm (reference implementation) | **ADDABLE** — Swarm is a candidate for a new neograph combinator, not a substrate limitation. Would expose LangGraph's existing `Command(goto)`/`Send` runtime handoff capability (already used internally for Each). |
| `pyagentspec.managerworkers.ManagerWorkers` | Hierarchical multi-agent pattern: manager orchestrates a set of worker agents based on problem topic. Named recipe, not a primitive. | `add_conditional_edges` + BranchingNode → AgentNodes. | Router @node + worker sub-constructs + Loop. Lowerable from neograph: an Operator-style LLM router node consuming upstream, outputting `{target_worker: str}`, followed by a fan-out (Each or manual Send) to N worker sub-constructs, then a Loop for multi-turn orchestration. | Orchestrator-Workers pattern (WayFlow docs) | **NOT A GAP** — ManagerWorkers is a syntactic recipe over primitives neograph already has. Import would reconstruct as a router node + N sub-constructs. |

## Status legend used
- **GAP-AS**: Genuine Agent-Specific gap — no direct neograph equivalent today. May be addable via existing LangGraph primitives.
- **NOT A GAP**: Agent Spec feature is a recipe/naming over primitives neograph already expresses.

## Serialization notes
- **Swarm**: `first_agent: str | Agent` (agent name or instance), `relationships: list[tuple[caller, recipient]]` (allowed caller→recipient pairs). Both fields serializable as JSON.
- **ManagerWorkers**: Not independently serialized in Agent Spec — expressed as Flow with BranchingNode + AgentNodes.

## Export lowering
**neograph → Swarm/ManagerWorkers:** No direct lowering. neograph does not emit Swarm or ManagerWorkers. Swarm could be surfaced as an export option if we add the combinator; ManagerWorkers is just a pattern users could author.

## Import reconstruction

**Swarm → neograph shape:**
```python
# Agent Spec Swarm
Swarm(first_agent="agent_a", relationships=[("agent_a", "agent_b"), ("agent_a", "agent_c")])

# Reconstructed as neograph router + Send fan-out
@node(mode="agent", outputs=NextTarget)
def router(swarm_input: SwarmInput, relationships: Annotated[RelationshipsConfig, FromConfig]) -> NextTarget:
    # LLM chooses next recipient from relationships

@node(outputs=SwarmResult)
def swarm_entry(first_agent_call: SomeInput) -> SwarmResult:
    pass  # Initial agent invocation

# Each agent becomes a sub-construct or @node
agent_a = Node(..., name="agent_a")
agent_b = Node(..., name="agent_b")
agent_c = Node(..., name="agent_c")

# Runtime routing via Send (already used in _wiring.py for Each)
# relationships config injected via FromConfig
```
**Unrecoverable bits:** Swarm's `first_agent` field can map to neograph's initial node, but the dynamic handoff graph (agents choosing next at runtime from relationships) is fundamentally different from neograph's static DAG. The reconstruction is an approximation — Swarm allows N agents to talk in any order allowed by `relationships`, while neograph's DAG is topologically ordered at compile time.

**ManagerWorkers → neograph shape:**
```python
# Agent Spec ManagerWorkers (conceptually)
ManagerWorkers(manager=AgentNode, workers=[AgentNode, ...])

# Reconstructed as neograph router + worker sub-constructs + Loop
@node(mode="agent", outputs=WorkerAssignment)
def manager_router(task: Task) -> WorkerAssignment:  # Which worker?
    pass

@node(outputs=WorkResult)
def worker_a(task: Task) -> WorkResult:
    pass

# ... more workers

# Loop for multi-turn manager orchestration
manager_workflow = manager_router | Loop(when="has_more_work")
```
**Unrecoverable bits:** None. ManagerWorkers is just a pattern; the reconstruction is faithful.

## Verdict for interop
**Swarm** is the only genuine Agentic Patterns mapping unknown. It maps cleanly to LangGraph's `Command(goto)`/`Send` primitives (which neograph already uses internally for Each fan-out), but neograph lacks a first-class Swarm combinator. Import would approximate Swarm as a router node + Send-based fan-out with relationships injected via FromConfig. The dynamic handoff graph (agents choosing next at runtime from allowed relationships) is fundamentally different from neograph's static DAG, so round-trip fidelity has risk — the reconstructed neograph graph would be topologically ordered, not truly decentralized. **Swarm is addable via a new combinator** that wraps a router node and uses Send for runtime handoff, not a substrate limitation.

**ManagerWorkers** is not a gap — it's a named recipe over primitives neograph already has (router + sub-constructs + Loop). Import reconstruction is straightforward and lossless.

**Biggest risk:** Swarm's runtime-decentralized handoff (agents choose next recipient at will within `relationships`) vs neograph's compile-time-static DAG. The reconstruction would force a "router in the middle" shape that's not faithful to Swarm's peer-to-peer model. Metadata markers could preserve the original `relationships` for re-export, but the runtime behavior would differ.
