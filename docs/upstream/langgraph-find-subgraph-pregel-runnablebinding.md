# Upstream report draft: `find_subgraph_pregel` cannot see through a `RunnableBinding`

Status: **DRAFT — not filed.** Written for the maintainer to file at
https://github.com/langchain-ai/langgraph/issues. Nothing here has been posted.

Tracked locally by `neograph-xunot` (neograph GH issue #6).

---

## Title

`get_subgraphs()` / `xray` / `to_json()` miss a subgraph when the node runnable has config bound to it

## Summary

`langgraph.pregel._utils.find_subgraph_pregel` walks a node's runnable looking for a
nested `Pregel`. It handles `RunnableSequence`/`RunnableSeq` (`.steps`),
`RunnableLambda` (`.deps`) and `RunnableCallable` (function nonlocals) — but it has no
branch for `RunnableBinding` (`.bound`).

`Runnable.with_config(...)` is the only way LangChain offers to attach `tags` /
`run_name` to a runnable, and it returns exactly a `RunnableBinding`. So **any node whose
runnable has been given a run name or tags becomes invisible to subgraph discovery**,
even though the nested graph is one attribute away.

## Impact

Everything that discovers nesting through this function silently degrades:
`get_subgraphs()`, `get_graph(xray=True)`, `Graph.to_json()`, `draw_mermaid()`,
LangGraph Studio, and Langfuse's agent-graph view. The graph still executes
correctly, so nothing fails loudly — the topology just renders as one opaque box.

This is not framework-specific: it hits any library that wraps a subgraph-holding
node to improve its trace spans, which is a natural thing to do.

## Reproduction

```python
from langchain_core.runnables import RunnableLambda
from langgraph.pregel._utils import find_subgraph_pregel

child = build_some_compiled_graph()          # a Pregel

def node(state):
    return child.invoke(state)               # closes over it

holder = RunnableLambda(node)

find_subgraph_pregel(holder)                       # -> the child Pregel  ✅
find_subgraph_pregel(holder.with_config(tags=["t"]))  # -> None           ❌
find_subgraph_pregel(holder.with_config(tags=["t"]).bound)  # -> the child ✅
```

The only difference between the working and broken case is the wrapper.

Observed with `langgraph` 1.2.7.

## Suggested fix

Add the missing branch alongside the three that already exist, in
`langgraph/pregel/_utils.py::find_subgraph_pregel`:

```python
elif isinstance(c, RunnableBinding):
    candidates.append(c.bound)
```

## Why there is no clean workaround on the caller's side

Of the three things `with_config` attaches, only two have a non-wrapping route:

| attachment | non-wrapping route |
|---|---|
| `run_name` | `RunnableLambda(name=...)` |
| `metadata`  | `StateGraph.add_node(..., metadata=...)` |
| `tags`      | **none** — `add_node` accepts no `tags=`, and `DeprecatedKwargs` is an empty `TypedDict` |

So a caller can keep the run name and the metadata by avoiding the binding, but must
give up tags entirely. That is the workaround neograph shipped
(`neograph-xunot`); the tags are a documented known gap that this upstream fix
would close.
