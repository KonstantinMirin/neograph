# DRY Review

**Scope**: Node.inputs refactor feature (commits 43f39cd..f58f9b9, diff base 41d910a). Focused on `src/neograph/decorators.py`, `src/neograph/factory.py`, `src/neograph/_construct_validation.py`.
**Date**: 2026-04-06

## Duplication Map

| Pattern | Occurrences | Files | Extractable? |
|---------|------------|-------|-------------|
| Frame-walking namespace builder for `get_type_hints` | 2 | `decorators.py` | Yes -> shared helper |
| `_make_gather_fn` / `_make_execute_fn` identical bodies | 2 | `factory.py` | Yes -> single `_make_tool_fn(node, mode)` |
| DI resolution loop (`param_res` iteration + `_resolve_di_value`) | 2 | `decorators.py`, `factory.py` | Partial -> shared `_resolve_all_di(param_res, config)` helper |
| State access closure (`dict.get` / `getattr` polymorphism) | 2 | `factory.py` | Yes -> module-level helper |
| Sidecar re-registration after `| modifier` | 4 | `decorators.py` | Marginal -> 3-line block, documented pattern |
| Name collision check in `construct_from_*` | 2 | `decorators.py` | Yes -> push into `_build_construct_from_decorated` |

## Findings

### DRY-01: Frame-walking namespace builder duplicated between `_classify_di_params` and `@node` inputs inference

- **Severity**: High
- **Category**: Auth (DI resolution)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/decorators.py:194-223` -- `_classify_di_params` variant (parameterized `frame_depth`, `include_extras=True`)
  - `src/neograph/decorators.py:521-552` -- inline variant in `decorator(f)` (hardcoded `sys._getframe(1)`, `include_extras=False`)
- **Description**: Both blocks build an identical namespace dict (seeding `FromInput`, `FromConfig`, `Annotated`; merging closure vars; walking 8 frames of caller locals) and then call `typing.get_type_hints(f, localns=..., include_extras=...)`. The only semantic differences are (a) the starting frame offset and (b) whether `include_extras` is True or False. Variable names differ (`extra_locals` vs `extra_ns`, `cv` vs `_cv`, `caller` vs `caller_frame`) -- classic AI-generated semantic duplication with cosmetic variation.
- **Risk**: A bug fix to the frame-walking logic (e.g., changing the 8-hop limit, adjusting the `_`-prefix filter, adding new seed symbols) must be applied in both places or they silently diverge. The inputs inference block was added in the kqd refactor and appears to have been written independently rather than calling the existing helper.
- **Proposed extraction**: Add a shared helper, e.g. `_resolve_type_hints(f: Callable, frame_depth: int, include_extras: bool) -> dict[str, Any]` in `decorators.py` at module level. Both `_classify_di_params` and the inputs inference block call it. The `frame_depth` parameter already exists in `_classify_di_params`; the inputs inference block just needs to pass the correct offset. `include_extras` is the only divergent flag.

### DRY-02: `_make_gather_fn` and `_make_execute_fn` are identical

- **Severity**: High
- **Category**: Transport (factory wrappers)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/factory.py:210-248` -- `_make_gather_fn`
  - `src/neograph/factory.py:251-289` -- `_make_execute_fn`
- **Description**: These two factory functions produce byte-for-byte identical inner closures. Both call `invoke_with_tools` with the same arguments, build the same `ToolBudgetTracker`, extract input the same way, and wire the output identically. The only difference is the string `"gather"` vs `"execute"` passed to `log.bind(mode=...)`. The mode distinction (gather = read-only tools, execute = mutation tools) is enforced elsewhere (tool definitions, not the wrapper). The wrapper code is 100% duplicated.
- **Risk**: Any change to the tool-invocation wrapper (error handling, logging format, metric emission) must be applied to both. Since both call the same `invoke_with_tools`, there is no functional reason for two separate wrapper factories.
- **Proposed extraction**: Merge into `_make_tool_fn(node: Node) -> Callable` that reads `node.mode` for the log label. Delete `_make_gather_fn` and `_make_execute_fn`. Update `make_node_fn` dispatch to call `_make_tool_fn` for both `"gather"` and `"execute"`.

### DRY-03: DI resolution loop duplicated between `legacy_shim` and `make_oracle_merge_fn`

- **Severity**: Medium
- **Category**: Auth (DI resolution)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/decorators.py:720-726` -- `legacy_shim` inside `@merge_fn`
  - `src/neograph/factory.py:415-420` -- DI-aware merge path inside `make_oracle_merge_fn`
- **Description**: Both iterate `param_res.items()`, call `_resolve_di_value(kind, payload, pname, config)`, append to an args list, and call the user function with `*args`. The `legacy_shim` returns the raw result; the merge path wraps it in `{field_name: ...}`. The DI iteration loop is semantically identical.
- **Risk**: If `_resolve_di_value`'s contract changes (e.g., new resolution kinds, error handling), both call sites must be updated. More importantly, the `legacy_shim` in `@merge_fn` is described as "rarely invoked" in the docstring because the factory always checks `_merge_fn_registry` first. Having the same logic in two places increases the chance that one drifts. The CLAUDE.md comment on `make_oracle_merge_fn` confirms the factory "detects the decorator's metadata" -- meaning the `legacy_shim` is a dead-code fallback that duplicates live logic.
- **Proposed extraction**: Extract `_call_with_di(fn, first_arg, param_res, config) -> Any` in `decorators.py`. Both `legacy_shim` and the merge path call it. The merge path wraps the return in `{field_name: ...}`.

### DRY-04: State access polymorphism (`dict.get` / `getattr`) defined twice in `factory.py`

- **Severity**: Medium
- **Category**: Query (state access)
- **Occurrences**: 2 places (closures), plus ad-hoc inline uses
- **Files**:
  - `src/neograph/factory.py:141-144` -- `_state_get(key)` closure inside `_make_scripted_wrapper`
  - `src/neograph/factory.py:312-315` -- `_get(key)` closure inside `_extract_input`
  - `src/neograph/factory.py:460` -- inline `state.get(...) if isinstance(state, dict) else getattr(...)` in `make_subgraph_fn`
  - `src/neograph/factory.py:488` -- inline `state.get(...) if isinstance(state, dict) else getattr(...)` in `make_each_redirect_fn`
- **Description**: The helper `if isinstance(state, dict): return state.get(key); return getattr(state, key, None)` is defined as a closure twice (under different names) and written inline twice more. All four serve the same purpose: polymorphic state field access over dict or Pydantic model state.
- **Risk**: If the state access convention changes (e.g., supporting a third state form), all four locations must be updated. The closures exist because they capture `state` in scope, but a module-level `_state_get(state, key)` taking state as a parameter eliminates the duplication without changing behavior.
- **Proposed extraction**: Module-level `def _state_get(state: Any, key: str, default: Any = None) -> Any` in `factory.py`. Replace all four sites.

### DRY-05: Sidecar re-registration block repeated after each modifier application

- **Severity**: Low
- **Category**: Response (decorator construction)
- **Occurrences**: 4 places (Each, base, Oracle, Operator paths)
- **Files**:
  - `src/neograph/decorators.py:583-585` -- after `| Each`
  - `src/neograph/decorators.py:588-590` -- base path (no modifier)
  - `src/neograph/decorators.py:610-612` -- after `| Oracle`
  - `src/neograph/decorators.py:631-633` -- after `| Operator`
- **Description**: The pattern `_register_sidecar(n, f, param_names); if param_res: _register_param_resolutions(n, param_res)` is repeated 4 times. Each occurrence is identical (same args, same guard). The `Each` path also has an early return, making it 5 total sidecar registrations counting the `n_mapped` variant.
- **Risk**: Low. Adding a new modifier kwarg requires copying the pattern, but CLAUDE.md documents this explicitly ("Any new modifier kwarg you add must follow the same pattern"). The block is 3 lines and each call site has different control flow around it (early return for Each, conditional for Oracle/Operator). Extracting would obscure the control flow.
- **Proposed extraction**: Optional. A `_register_node(n, f, param_names, param_res)` helper would reduce each site to 1 line. Whether this improves readability is debatable given the documented convention.

### DRY-06: Name collision check duplicated between `construct_from_module` and `construct_from_functions`

- **Severity**: Low
- **Category**: Error (validation)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/decorators.py:769-777` -- in `construct_from_module`
  - `src/neograph/decorators.py:823-832` -- in `construct_from_functions`
- **Description**: Both functions check `if field_name in decorated:` and raise `ConstructError` with a structurally identical error message (differing only in `source_label`). Both then call the shared `_build_construct_from_decorated`. The collision check could live inside the shared builder, which already receives `source_label`.
- **Risk**: Low. The check is simple and unlikely to change. But if a third entry point (e.g., `construct_from_config`) is added, it would need to copy the check.
- **Proposed extraction**: Move the collision check into `_build_construct_from_decorated`. The builder already receives the `decorated` dict and `source_label`, so the check is a natural fit. Each caller would just pass `(field_name, Node)` pairs without pre-checking.

## Cleared Areas (No Duplication Found)

1. **`list[X]` unwrap logic**: The old copy in `_attach_scripted_raw_fn` is fully removed. The unwrap now lives only in `factory._extract_input:341-346`. Clean.

2. **`_register_node_scripted` DI resolution**: Uses the shared `_resolve_di_value` helper (line 1024). Does NOT duplicate the resolution logic. Clean.

3. **`_check_fan_in_inputs` vs `_check_item_input`**: `_check_item_input` dispatches to `_check_fan_in_inputs` for dict-form inputs (line 127-129). They are complementary handlers (single-type vs dict-form), not overlapping paths. Clean.

4. **`effective_producer_type`**: Single source of truth in `_construct_validation.py:38-65`. Both validator walkers call it. No inline copies of modifier rules found. Clean.

## Summary

- Critical: 0
- High: 2 (DRY-01, DRY-02)
- Medium: 2 (DRY-03, DRY-04)
- Low: 2 (DRY-05, DRY-06)
- Total duplicated logic blocks: 6
- Estimated lines removable by extraction: ~65
  - DRY-01: ~25 lines (inline namespace builder replaced by helper call)
  - DRY-02: ~35 lines (entire `_make_execute_fn` body replaced by `_make_tool_fn` reuse)
  - DRY-03: ~5 lines (iteration loop replaced by helper call)
  - DRY-04: not lines saved, but 4 sites unified
  - DRY-05, DRY-06: marginal
