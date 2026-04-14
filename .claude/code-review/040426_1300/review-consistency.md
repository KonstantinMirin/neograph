# Consistency Review — NeoGraph

## High

### CON-01: Generated graph node name separator inconsistent
`compiler.py` uses three different separators for synthetic names:
- `merge_{name}` (underscore) — subgraph oracle
- `assemble_{name}` (underscore) — subgraph each  
- `assemble-{name}` (HYPHEN) — node each
- `{name}__operator` (double underscore) — operator

Line 426 uses hyphen, line 251 uses underscore for the same concept.

### CON-02: "replicate" vs "each" naming split
Node-level Each uses old name `_add_replicate_nodes`, `replicate_router`, variable `replicate`.
Subgraph-level uses `_add_each_subgraph`, `each_router`, variable `each`.
Incomplete rename.

## Medium

### CON-03: Error message missing hint suffix
`compiler.py:226` — `"Merge function '{oracle.merge_fn}' not registered."` (missing `Use register_scripted().`)
All other registration errors include the hint.

### CON-04: Node function signature types diverge
`factory.py`: `(state: BaseModel, config: RunnableConfig) -> dict[str, Any]`
`compiler.py`: `(state: Any, config: RunnableConfig) -> dict`
Two compiler functions use `config=None` default.

### CON-05: Config typed as `dict` in _llm.py but `RunnableConfig` elsewhere

### CON-06: RuntimeError outlier
`_llm.py:83` is the only `RuntimeError` in the codebase. All other config errors use `ValueError`.

### CON-07: node_start log fields inconsistent across modes
- scripted: logs `input_type`, `output_type`
- produce: logs only `output_type`
- gather/execute: logs `tools`, `budgets` (no types)

## Low

### CON-08: Duplicate oracle/each code (same as DRY-02/DRY-03)
### CON-09: Unused `RunnableConfig as RC` import at compiler.py:259
### CON-10: "must declare" validation phrasing inconsistent
