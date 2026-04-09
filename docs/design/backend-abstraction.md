# Backend Abstraction -- Multi-Backend Strategy

## Direction

neograph should eventually support multiple execution backends, not just LangGraph. The spec format and IR are already backend-agnostic -- only `compile()` is coupled to LangGraph.

## Current architecture

```
Spec (YAML/JSON) --> IR (Construct + Nodes + Modifiers) --> compile() --> LangGraph StateGraph
                                                              ^
                                                         adapter layer
                                                      (only coupled part)
```

Everything before `compile()` is backend-agnostic. The IR (Construct with nodes/modifiers) is a pure data structure that describes the pipeline topology, types, and execution modes.

## Candidate backends

| Backend | Language | When | Notes |
|---------|----------|------|-------|
| LangGraph (Python) | Python | Now | Current default. Mature, checkpointing, human-in-loop. |
| LangGraph.js | TypeScript | Near-term | TS port target. Same API patterns. See [TS port design](typescript-port.md). |
| Anthropic SDK | Python/TS | Medium-term | Direct API calls, no framework overhead. For simple pipelines. |
| Google ADK | Python | Medium-term | If Google's agent framework gains traction. |
| PydanticAI | Python | Medium-term | Pydantic-native, good type story. |
| Raw asyncio | Python | Anytime | No-dependency option for production. |

## Design rules for multi-backend

1. **New features go in the IR layer** (Construct, Node, modifiers) -- not in the compiler layer. The compiler is the adapter.
2. **The spec format never references LangGraph concepts.** No `StateGraph`, `Send`, `interrupt` in YAML specs. These are compile-time translations.
3. **The TypeScript version shares specs and JSON Schemas** -- just has a different compiler backend.
4. **Validation lives in `_construct_validation.py`** (pre-compile) -- not in `compiler.py` (compile-time). Pre-compile validation is backend-agnostic.
5. **Factory dispatch is the only runtime coupling.** `make_node_fn`, `make_subgraph_fn`, etc. produce LangGraph-compatible closures. A different backend would need different factory implementations, same interfaces.

## What a backend adapter looks like

```python
# Current: LangGraph backend
from neograph.compiler import compile  # -> StateGraph

# Future: abstract
from neograph import compile
graph = compile(construct, backend="langgraph")  # default
graph = compile(construct, backend="asyncio")    # no-dependency
graph = compile(construct, backend="anthropic")  # direct API
```

Each backend implements:
- State model generation (how to track pipeline state)
- Node wiring (how to connect nodes in execution order)
- Fan-out mechanics (how to parallelize Each/Oracle)
- Checkpointing (how to persist state for resume)
- Tool dispatch (how to call tools in ReAct loops)

## Not a task yet

This is a direction for architecture decisions, not an implementation plan. The first concrete step is the TypeScript port (which proves the IR is backend-agnostic by implementing a second backend).
