# Layering Review

**Scope**: Full `src/neograph/` codebase — all modules. Focus on whether logic lives in the correct architectural layer.
**Date**: 2026-04-10

## Layer Definitions Applied

```
DX layer:              decorators.py, forward.py
IR layer:              node.py, construct.py, _construct_validation.py, modifiers.py
Compiler layer:        compiler.py, state.py
Runtime dispatch:      factory.py
LLM integration:      _llm.py
Utilities:             errors.py, tool.py, renderers.py, describe_type.py, _dev_warnings.py
Loader:                loader.py, spec_types.py, conditions.py
Runner:                runner.py
```

## Findings

### LR-01: `factory.py` imports from `decorators.py` (runtime dispatch imports DX layer)

- **Severity**: Medium
- **Violation**: Runtime dispatch layer (`factory.py`) --> DX layer (`decorators.py`)
- **Files**: `src/neograph/factory.py:807`, `src/neograph/compiler.py:505`
- **Description**: `make_oracle_merge_fn` (factory.py:807) and `_merge_one_group` (compiler.py:505) contain deferred imports of `get_merge_fn_metadata` and `_resolve_merge_args` from `decorators.py`. This is a lower-layer module importing from a higher-layer module. The import is deferred (inside function bodies) to avoid circular imports, which signals the dependency direction is inverted.
- **Impact**: The `@merge_fn` decorator's registry and DI resolution logic live in `decorators.py`, but the runtime dispatch needs to consult them. This couples factory.py to the DX layer's internal data structures.
- **Pre-existing**: Yes, introduced with `@merge_fn` (not new).
- **Recommended fix**: Extract `_resolve_di_value`, `_resolve_merge_args`, `_resolve_di_args`, `_merge_fn_registry`, `get_merge_fn_metadata`, `ParamResolution`, and `_param_resolutions` into a shared module (e.g., `_di.py` or `_di_resolution.py`). Both `decorators.py` and `factory.py` would import from this module without a layer violation. The DI classifier (`_classify_di_params`) stays in `decorators.py` since it uses frame walking and `inspect.Signature`, which are DX-layer concerns.

### LR-02: `_construct_validation.py` imports from `decorators.py` (IR imports DX layer)

- **Severity**: Medium
- **Violation**: IR layer (`_construct_validation.py`) --> DX layer (`decorators.py`)
- **File**: `src/neograph/_construct_validation.py:233`
- **Description**: `_validate_node_chain` at line 233 does a deferred `from neograph.decorators import get_merge_fn_metadata` to validate `@merge_fn` state params against known producers. The IR-layer validation module is reaching up into the DX layer to access the merge function registry.
- **Impact**: The validator depends on the `@merge_fn` decorator's metadata registry. If the merge function was registered via `register_scripted` (non-`@merge_fn` path), this validation is skipped entirely because `get_merge_fn_metadata` returns None. This creates asymmetric validation: `@merge_fn`-decorated functions get state param validation; plain scripted merge functions do not.
- **Pre-existing**: Yes, introduced with `@merge_fn` state-param validation (neograph-jg2g).
- **Recommended fix**: Same as LR-01 — moving the registry to a shared `_di.py` module would eliminate the upward dependency. Alternatively, the merge_fn metadata could be stored on the Node or Oracle modifier so the validator accesses it through IR-layer objects.

### LR-03: `_llm.py` imports from `factory.py` (LLM integration imports runtime dispatch)

- **Severity**: Low
- **Violation**: LLM integration layer (`_llm.py`) --> Runtime dispatch layer (`factory.py`)
- **File**: `src/neograph/_llm.py:539`
- **Description**: `invoke_with_tools` does a deferred `from neograph.factory import _tool_factory_registry` at line 539 to instantiate tool factories. The LLM layer reads a private registry from the factory layer.
- **Impact**: The `_tool_factory_registry` is a module-level dict in `factory.py` that `_llm.py` reads. The tool factory registry logically belongs to the factory layer (tools are instantiated per-node at runtime), but the ReAct loop in `_llm.py` needs to create tool instances.
- **Pre-existing**: Yes.
- **Recommended fix**: Pass `tool_instances` (already-instantiated tools) to `invoke_with_tools` instead of having it instantiate them from the registry. `factory._make_tool_fn` already creates tool instances (lines 562-568) before calling `invoke_with_tools` — it currently duplicates tool creation. Passing the instances directly would eliminate both the import and the duplication. However, this is actually NOT a duplication: `_make_tool_fn` passes `tools` (Tool specs) to `invoke_with_tools`, which creates the instances itself. The fix would be to move tool instantiation to `_make_tool_fn` and pass `tool_instances` to `invoke_with_tools`.

### LR-04: `node.py` imports from `factory.py` (IR imports runtime dispatch)

- **Severity**: Low
- **Violation**: IR layer (`node.py`) --> Runtime dispatch layer (`factory.py`)
- **File**: `src/neograph/node.py:156`
- **Description**: `Node.run_isolated()` contains a deferred `from neograph.factory import make_node_fn`. This is the IR layer importing from the runtime dispatch layer. The method exists solely for testing convenience.
- **Pre-existing**: Yes.
- **Recommended fix**: Move `run_isolated` to a standalone function (e.g., in a testing utilities module or in `runner.py`). Low priority — the deferred import keeps the dependency from affecting module-level imports.

### LR-05: `fan_out_param` on IR-layer `Node` for DX-only use

- **Severity**: Low
- **Violation**: DX layer (`decorators.py`) --> IR layer (`node.py`) leak of @node-specific concern
- **File**: `src/neograph/node.py:64-68`
- **Description**: `Node.fan_out_param` is only ever set from `decorators.py:1518` (`n.fan_out_param = fan_out_name`). The programmatic Node API never sets it except indirectly. The field was introduced as a documented trade-off (see AGENTS.md) to communicate @node decoration-time topology from DX to runtime without coupling those layers directly.
- **Pre-existing**: Yes, documented in AGENTS.md.
- **Mitigating factors**: The validator in `_construct_validation.py:405-416` also handles the programmatic path (Each modifier present + key not in producers) independently of `fan_out_param`, so the programmatic API works without this field. The field is a convenience for the @node path to pre-identify the fan-out param, avoiding a second inference at validation time.

### LR-06: `decorators.py` imports `_unwrap_loop_value` from `factory.py` (DX imports runtime dispatch)

- **Severity**: Low
- **Violation**: DX layer (`decorators.py`) --> Runtime dispatch layer (`factory.py`)
- **File**: `src/neograph/decorators.py:420`
- **Description**: `_resolve_merge_args` in `decorators.py` does a deferred `from neograph.factory import _unwrap_loop_value` to unwrap Loop append-lists when resolving `from_state` params. This is the DX layer reaching down into the runtime dispatch layer for a utility function.
- **Impact**: `_unwrap_loop_value` is a pure helper that unwraps `list[-1]` for Loop nodes. Its logic is not runtime-specific — it is a shared concern between any code that reads Loop state.
- **Recommended fix**: Move `_unwrap_loop_value` (and `_unwrap_each_dict`) to a shared utility module (e.g., `_state_utils.py` or the proposed `_di.py`). Both `factory.py` and `decorators.py` would import from the shared module.

### LR-07: `compiler.py` imports from `decorators.py` (compiler imports DX layer)

- **Severity**: Medium
- **Violation**: Compiler layer (`compiler.py`) --> DX layer (`decorators.py`)
- **File**: `src/neograph/compiler.py:505-506`
- **Description**: `_merge_one_group` in `compiler.py` imports `get_merge_fn_metadata` and `_resolve_merge_args` from `decorators.py`. The compiler layer is reaching up into the DX layer to resolve `@merge_fn` DI params during Each×Oracle fusion.
- **Impact**: The Each×Oracle fusion path in the compiler duplicates the merge dispatch logic from `factory.make_oracle_merge_fn`. Both call `get_merge_fn_metadata` and `_resolve_merge_args`. The compiler shouldn't need to know about DI resolution.
- **Pre-existing**: Yes, introduced with Each×Oracle fusion (neograph-tpgi).
- **Recommended fix**: (a) Same DI extraction as LR-01/LR-02 — shared `_di.py`. (b) Alternatively, refactor `_merge_one_group` to call `make_oracle_merge_fn` from factory.py (already imported by compiler.py) to produce the merge function, then call it. This would keep merge dispatch in the factory where it belongs.

### LR-08: `lint.py` imports from `decorators.py` (utility imports DX layer)

- **Severity**: Low
- **Violation**: Utility module (`lint.py`) --> DX layer (`decorators.py`)
- **File**: `src/neograph/lint.py:14`
- **Description**: `lint.py` imports `_get_param_resolutions` and `get_merge_fn_metadata` from `decorators.py` at module level. The linter needs to inspect DI bindings, which are stored in the DX layer's sidecar dicts.
- **Impact**: `lint()` works only on `@node`-decorated nodes (it checks the `_param_resolutions` sidecar). Programmatic Node instances without sidecars are silently skipped. This is acceptable because programmatic nodes don't use the DI surface.
- **Pre-existing**: Yes.
- **Recommended fix**: Would benefit from the same `_di.py` extraction proposed in LR-01.

### LR-09: `modifiers.py` imports from `_construct_validation.py` (IR imports validation)

- **Severity**: Low (positive finding — within IR layer)
- **Violation**: None — both are IR layer
- **File**: `src/neograph/modifiers.py:216,226`
- **Description**: `Modifiable.__or__` imports `validate_loop_self_edge` and `validate_loop_construct` from `_construct_validation.py`. These are deferred imports inside the `__or__` method, gated behind `isinstance(modifier, Loop)`.
- **Assessment**: Both modules are in the IR layer. The imports are deferred to avoid circular imports (modifiers.py is imported by construct.py which imports _construct_validation.py). The deferred imports are appropriate — validation at `|` time is a valid IR-layer concern.

### LR-10: `modifiers.py` imports from `_dev_warnings.py` (IR imports utility)

- **Severity**: Low (non-issue)
- **Violation**: None
- **File**: `src/neograph/modifiers.py:187`
- **Description**: `Modifiable.__or__` imports `dev_warn` from `_dev_warnings.py`. This is a utility import, not a layer violation. Dev warnings are a cross-cutting concern.

### LR-11: `compiler.py` has runtime logic inline (loop routers, group merge)

- **Severity**: Medium
- **Violation**: Compiler layer contains runtime dispatch logic
- **Files**: `src/neograph/compiler.py:468-494` (`group_merge_barrier`), `src/neograph/compiler.py:427-458` (`flat_router`), `src/neograph/compiler.py:570-610` (`loop_router`)
- **Description**: Several closures defined inside `compiler.py` contain runtime dispatch logic — they execute during graph invocation, not at compile time. The `group_merge_barrier` (lines 468-494) calls `_merge_one_group` which does DI resolution; `loop_router` (lines 575-610) accesses state fields and evaluates conditions; `flat_router` (lines 427-458) navigates dotted paths and dispatches Send calls. These are runtime functions that happen to be defined in the compiler because they're closures that capture compile-time state.
- **Impact**: The compiler module grows to 938 lines with significant runtime logic embedded. The Each×Oracle fusion path is the most complex, containing `flat_router`, `group_merge_barrier`, and `_merge_one_group` — all runtime functions. By contrast, the simple Node/Construct paths correctly delegate runtime logic to factory.py.
- **Mitigating factors**: Closures that capture compile-time state (like `oracle.n`, `each.over`, `node.outputs`) are inherently tied to the compile-time context. Moving them to factory.py would require passing many parameters or creating intermediate data structures.
- **Recommended fix**: Consider extracting `_merge_one_group` and the Each×Oracle-specific closures into factory.py as parameterized factory functions (similar to `make_oracle_merge_fn`, `make_oracle_redirect_fn`). The compiler would call `make_eachoracle_merge_fn(oracle, node, each)` and get back the closure. This would keep the compiler focused on topology and delegate runtime behavior to the factory.

### LR-12: `tool.py` `@tool` decorator imports from `factory.py` (utility imports runtime dispatch)

- **Severity**: Low
- **Violation**: Utility module (`tool.py`) --> Runtime dispatch layer (`factory.py`)
- **File**: `src/neograph/tool.py:135`
- **Description**: The `@tool` decorator does a deferred `from neograph.factory import register_tool_factory` to auto-register the decorated function. This is an upward dependency — the tool definition module reaching into the registration layer.
- **Impact**: Minimal. The deferred import avoids circular imports and the dependency is narrow (one function call).
- **Pre-existing**: Yes.
- **Recommended fix**: Move `register_tool_factory` (and the `_tool_factory_registry`) to `tool.py` where the Tool class lives. Factory.py would import from tool.py, which matches the layer hierarchy (higher imports lower). However, `_tool_factory_registry` is also read by `_llm.py:539` and `compiler.py:133`, so this would need those reads to import from `tool.py` as well.

## Cross-Layer Import Map (Full Codebase)

| Source (layer) | Target (layer) | Import | Deferred? |
|---|---|---|---|
| `decorators.py` (DX) | `_construct_validation.py` (IR) | `ConstructError` | No (module-level) |
| `decorators.py` (DX) | `_construct_validation.py` (IR) | `effective_producer_type`, `_types_compatible` | Yes (in `_resolve_loop_self_param`) |
| `decorators.py` (DX) | `factory.py` (Runtime) | `register_scripted`, `register_condition`, `lookup_scripted`, `_unwrap_loop_value` | Yes (7 call sites) |
| `factory.py` (Runtime) | `decorators.py` (DX) | `get_merge_fn_metadata`, `_resolve_merge_args` | Yes (in `make_oracle_merge_fn`) |
| `factory.py` (Runtime) | `_llm.py` (LLM) | `invoke_structured`, `invoke_with_tools`, `_get_global_renderer` | Yes |
| `factory.py` (Runtime) | `runner.py` | `_strip_internals` | Yes (in `make_subgraph_fn`) |
| `compiler.py` (Compiler) | `factory.py` (Runtime) | 8 factory functions | No (module-level) |
| `compiler.py` (Compiler) | `decorators.py` (DX) | `get_merge_fn_metadata`, `_resolve_merge_args` | Yes (in `_merge_one_group`) |
| `compiler.py` (Compiler) | `_llm.py` (LLM) | `_llm_factory`, `_prompt_compiler` | Yes (in `compile`) |
| `_construct_validation.py` (IR) | `decorators.py` (DX) | `get_merge_fn_metadata` | Yes (in `_validate_node_chain`) |
| `_llm.py` (LLM) | `factory.py` (Runtime) | `_tool_factory_registry` | Yes (in `invoke_with_tools`) |
| `node.py` (IR) | `factory.py` (Runtime) | `make_node_fn` | Yes (in `run_isolated`) |
| `tool.py` (Utility) | `factory.py` (Runtime) | `register_tool_factory` | Yes (in `@tool`) |
| `lint.py` (Utility) | `decorators.py` (DX) | `_get_param_resolutions`, `get_merge_fn_metadata` | No (module-level) |
| `modifiers.py` (IR) | `_construct_validation.py` (IR) | `validate_loop_self_edge`, `validate_loop_construct` | Yes (in `__or__`) |

## Dependency Direction Summary

Expected layer hierarchy (higher imports lower):
```
DX (decorators, forward) ──> IR (node, construct, modifiers, _construct_validation)
                         ──> Runtime (factory)
                         ──> Compiler (compiler, state)
Compiler                 ──> IR
                         ──> Runtime (factory)
Runtime (factory)        ──> IR
                         ──> LLM (_llm)
LLM (_llm)              ──> Utility (tool, renderers, describe_type)
```

Violations (lower imports higher):
```
factory.py (Runtime)          ──> decorators.py (DX)         [LR-01]
_construct_validation.py (IR) ──> decorators.py (DX)         [LR-02]
_llm.py (LLM)                ──> factory.py (Runtime)        [LR-03]
node.py (IR)                  ──> factory.py (Runtime)        [LR-04]
compiler.py (Compiler)        ──> decorators.py (DX)         [LR-07]
```

## Systemic Pattern: `@merge_fn` Registry as Root Cause

The `_merge_fn_registry` in `decorators.py` is the root cause of three findings (LR-01, LR-02, LR-07). The registry stores `(user_fn, ParamResolution)` tuples and is consulted by:

1. `factory.make_oracle_merge_fn` — to detect DI-aware merge functions
2. `_construct_validation._validate_node_chain` — to validate state param references
3. `compiler._merge_one_group` — to resolve DI during Each×Oracle fusion

All three are lower layers reaching up into the DX layer. Extracting the DI infrastructure (`ParamResolution`, `_resolve_di_value`, `_resolve_merge_args`, `_merge_fn_registry`, `_param_resolutions`) into a shared `_di.py` module would eliminate all three violations in one move.

## Summary

- Critical: 0
- High: 0
- Medium: 3 (LR-01, LR-02, LR-07 all stem from the `@merge_fn` registry placement; LR-11 is compiler-runtime mixing)
- Low: 7 (LR-03, LR-04, LR-05, LR-06, LR-08, LR-09, LR-10, LR-12)

**Overall assessment**: The codebase generally respects the documented layer boundaries. The dominant pattern is that all three medium-severity violations originate from the same root cause: the `@merge_fn` DI registry living in `decorators.py` while being consumed by lower layers. This is a single architectural debt that would be resolved by extracting DI infrastructure into a shared module. The remaining low-severity findings are either documented trade-offs (fan_out_param) or testing conveniences (run_isolated) with deferred imports that keep module-level dependencies clean.

Compared to the previous review (2026-04-06), the layering posture is stable — no new violations have been introduced. The pre-existing `factory.py -> decorators.py` dependency (LR-05 in the previous review) is unchanged. The scope expansion reveals two additional medium-severity instances of the same pattern (LR-02 in _construct_validation.py and LR-07 in compiler.py) that weren't visible in the narrower diff-scoped review.
