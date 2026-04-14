# Layering Review

**Scope**: Full project at `/Users/konst/projects/neograph/src/neograph/` -- all 9 source modules (node.py, tool.py, modifiers.py, construct.py, state.py, factory.py, _llm.py, compiler.py, runner.py)
**Date**: 2026-04-04

## Layer Map (as designed)

| Layer | Module(s) | Responsibility |
|---|---|---|
| L1 Data Models | tool.py, modifiers.py, node.py, construct.py | Pydantic declarations |
| L2 State Bus | state.py | Generate state model from Node I/O |
| L3 Node Factory | factory.py | Create LangGraph node functions |
| L4 LLM Layer | _llm.py | LLM invocation (structured output, ReAct) |
| L5 Graph Compiler | compiler.py | Construct -> LangGraph StateGraph topology |
| L6 Runner | runner.py | Execution, config injection, result cleanup |

Expected dependency direction: L1 <- L2 <- L3 <- L5 (compiler), L3 -> L4, L5 -> L6 (runner utilities).

## Findings

### LR-01: Compiler directly calls LLM layer -- merge logic belongs in factory
- **Severity**: High
- **Violation**: L5 (Compiler) -> L4 (LLM Layer) -- bypasses L3 (Factory)
- **Files**: `/Users/konst/projects/neograph/src/neograph/compiler.py:211`, `/Users/konst/projects/neograph/src/neograph/compiler.py:388`
- **Description**: The compiler creates inline `merge_fn` closures that directly call `invoke_structured()` from `_llm.py`. This happens in two places: `_add_oracle_subgraph` (line 211) and `_add_oracle_nodes` (line 388). These merge functions are runtime node functions -- they receive state and config, call the LLM, and return state updates. This is exactly what `factory.py` exists to do. The compiler's job is topology (wiring nodes and edges), not building runtime node functions that invoke LLMs. This means the compiler has two responsibilities: graph topology AND node function creation for Oracle merges.
- **Reproduction**: `grep -n "invoke_structured" /Users/konst/projects/neograph/src/neograph/compiler.py`
- **Recommended fix**: Create a `make_oracle_merge_fn(oracle, output_model, field_name, collector_field)` function in `factory.py` that builds the merge node function. The compiler should call this factory function instead of inlining the LLM call. This keeps all "node function creation" in one place and all "topology wiring" in another.

### LR-02: Compiler accesses factory's private registries directly
- **Severity**: Medium
- **Violation**: L5 (Compiler) -> L3 (Factory) internal state
- **File**: `/Users/konst/projects/neograph/src/neograph/compiler.py:18`
- **Description**: The compiler imports `_condition_registry` and `_scripted_registry` (underscore-prefixed, indicating private) from `factory.py` at the top level. It reads `_condition_registry` in `_add_operator_check` (line 470) and `_scripted_registry` in `_add_oracle_nodes` (line 401) and `_add_oracle_subgraph` (line 224). The compiler should not need to know about the factory's internal registry implementation. This couples the compiler to the factory's storage mechanism.
- **Reproduction**: `grep -n "_condition_registry\|_scripted_registry" /Users/konst/projects/neograph/src/neograph/compiler.py`
- **Recommended fix**: Add public lookup functions in `factory.py` (e.g., `get_condition(name)` and `get_scripted(name)`) that raise clear errors on missing entries. The compiler calls these instead of reaching into private dicts. This also centralizes the error messages that are currently duplicated.

### LR-03: LLM layer imports factory's private registry -- circular dependency
- **Severity**: High
- **Violation**: L4 (LLM) -> L3 (Factory) -- upward dependency creating a cycle
- **File**: `/Users/konst/projects/neograph/src/neograph/_llm.py:226`
- **Description**: `_llm.py:invoke_with_tools()` does a deferred import of `_tool_factory_registry` from `factory.py`. The dependency chain is: factory.py -> _llm.py (factory calls `invoke_structured`, `invoke_with_tools`) AND _llm.py -> factory.py (LLM layer reads `_tool_factory_registry`). This is a circular dependency, currently working only because the import is deferred. The LLM layer should not know about the tool factory registry; it should receive resolved tool instances, not resolve them itself.
- **Reproduction**: `grep -n "_tool_factory_registry" /Users/konst/projects/neograph/src/neograph/_llm.py`
- **Recommended fix**: Have `factory.py` resolve tool instances (by calling tool factories and passing the resulting callables) BEFORE calling `invoke_with_tools()`. Change `invoke_with_tools()` to accept `tool_instances: dict[str, Callable]` instead of `tools: list[Tool]`. This removes the upward dependency entirely and makes the LLM layer a pure execution layer with no knowledge of registration mechanics.

### LR-04: Compiler imports runner utility -- upward dependency
- **Severity**: Medium
- **Violation**: L5 (Compiler) -> L6 (Runner) -- lower layer importing from higher layer
- **File**: `/Users/konst/projects/neograph/src/neograph/compiler.py:86`
- **Description**: `_add_subgraph()` imports `_strip_internals` from `runner.py`. The runner sits above the compiler in the dependency hierarchy (runner calls compiled graphs). This creates a coupling where the compiler depends on the runner's utility function. The function itself is a simple dict filter that strips `neo_*` keys -- it is not runner-specific logic.
- **Reproduction**: `grep -n "_strip_internals" /Users/konst/projects/neograph/src/neograph/compiler.py`
- **Recommended fix**: Move `_strip_internals()` to a shared utility location. Options: (a) a `_utils.py` module both can import, or (b) inline in `state.py` since it relates to framework state field conventions (the `neo_*` prefix is defined by state.py). Both compiler and runner would import from this shared location.

### LR-05: Compiler duplicates input extraction logic from factory
- **Severity**: Medium
- **Violation**: L5 (Compiler) duplicates L3 (Factory) responsibility
- **File**: `/Users/konst/projects/neograph/src/neograph/compiler.py:100-116`
- **Description**: The `_make_subgraph_fn()` closure in `_add_subgraph` contains an inline input extraction pattern (iterating state fields by type to find matching input) that duplicates the logic in `factory.py:_extract_input()`. The factory has a well-defined `_extract_input(state, node)` function that handles dict vs. model states, Each items, dict-typed inputs, and type-based matching. The compiler reimplements a subset of this (type-based matching only). If the extraction logic changes (e.g., a new input resolution strategy), it must be updated in two places.
- **Reproduction**: `grep -n "isinstance(val, sub.input)" /Users/konst/projects/neograph/src/neograph/compiler.py`
- **Recommended fix**: Refactor the subgraph node function to use or delegate to the same input extraction logic in factory.py. Since the subgraph wraps a compiled graph (not a single node), this may require a thin adapter, but the type-matching logic should live in one place.

### LR-06: Factory handles Each modifier output wrapping -- compiler concern
- **Severity**: Low
- **Violation**: L3 (Factory) contains L5 (Compiler) topology awareness
- **File**: `/Users/konst/projects/neograph/src/neograph/factory.py:105-109`
- **Description**: In `_make_scripted_wrapper()`, the factory checks for the `Each` modifier and wraps the result in a dict keyed by the item's key field. This means the factory knows about the fan-out topology pattern that the compiler implements. The comment at line 111 ("Oracle redirection handled by compiler wrapper, not here") shows awareness that Oracle output wrapping is handled differently (in the compiler), but Each wrapping is split between the two layers. This inconsistency means modifier output wiring is partially in the factory and partially in the compiler.
- **Recommended fix**: Choose one layer to own all modifier output wrapping. Since Oracle wrapping is already in the compiler (via wrapper closures), move Each output wrapping there too for consistency. The factory would always write to `{field_name}`, and the compiler wrapper would handle dict-keying for Each, just as it handles collector-field redirection for Oracle.

### LR-07: Compiler builds runtime node functions inline -- factory concern
- **Severity**: Medium
- **Violation**: L5 (Compiler) doing L3 (Factory) work
- **Files**: `/Users/konst/projects/neograph/src/neograph/compiler.py:100-136` (subgraph_fn), `/Users/konst/projects/neograph/src/neograph/compiler.py:183-187` (oracle_subgraph_fn), `/Users/konst/projects/neograph/src/neograph/compiler.py:254-266` (each_subgraph_fn), `/Users/konst/projects/neograph/src/neograph/compiler.py:289-290` (barrier_fn), `/Users/konst/projects/neograph/src/neograph/compiler.py:355-359` (oracle node_fn wrapper), `/Users/konst/projects/neograph/src/neograph/compiler.py:453-454` (barrier_fn)
- **Description**: The compiler creates at least 6 different inline node function closures. These functions are runtime behaviors -- they receive state and config, perform computation, and return state updates. The factory exists specifically to create these functions. The compiler should only wire topology (add_node, add_edge, conditional_edges). By building node functions inline, the compiler has become responsible for both graph structure AND runtime behavior, which makes it the largest and most complex module (~486 lines vs factory's ~272 lines).
- **Reproduction**: `grep -n "def.*state.*config" /Users/konst/projects/neograph/src/neograph/compiler.py`
- **Recommended fix**: Extract all inline node functions into factory.py as named factory functions: `make_subgraph_fn()`, `make_oracle_wrapper_fn()`, `make_each_wrapper_fn()`, `make_barrier_fn()`, `make_oracle_merge_fn()`. The compiler calls these factories and only handles wiring the returned functions into the graph.

### LR-08: ToolBudgetTracker exposes private state
- **Severity**: Low
- **Violation**: L1 (Data Model) encapsulation break
- **File**: `/Users/konst/projects/neograph/src/neograph/_llm.py:273`
- **Description**: `invoke_with_tools()` accesses `budget_tracker._budgets[tool_name]` (private attribute with underscore prefix) to build the "budget exhausted" message. This bypasses the public API of `ToolBudgetTracker`.
- **Reproduction**: `grep -n "_budgets" /Users/konst/projects/neograph/src/neograph/_llm.py`
- **Recommended fix**: Add a public method to `ToolBudgetTracker` (e.g., `get_budget(tool_name) -> int`) and use it in the message construction.

## Summary

- Critical: 0
- High: 2
- Medium: 3
- Low: 2

## Dependency Graph (actual)

```
modifiers.py  <--  node.py  <--  construct.py
    ^               ^    ^           ^
    |               |    |           |
tool.py        factory.py |     state.py
    ^            ^    |   |       ^
    |            |    v   |       |
    +--- _llm.py-+    |  +--compiler.py
         ^   |        |       |    ^
         |   +--------+       |    |
         |     (circular)      v    |
         |                runner.py-+
         |                  (upward)
```

The two high-severity findings (LR-01 and LR-03) share a root cause: the compiler and LLM layer both bypass the factory to do each other's work. LR-03 is the more structurally dangerous one because it creates a circular dependency that only works due to deferred imports -- a fragile arrangement that will break if someone converts the deferred import to a top-level import during a refactor.

The medium findings (LR-02, LR-04, LR-05) represent coupling that increases maintenance burden but does not cause architectural fragility. The compiler reaching into private registries and duplicating input extraction means changes to the factory's internals require synchronized edits in the compiler.

The overall pattern is that `compiler.py` has absorbed responsibilities that belong in `factory.py`. The compiler should be a pure topology builder; all runtime node function creation should flow through the factory.
