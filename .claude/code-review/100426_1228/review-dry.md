# DRY Review

**Scope**: All of `src/neograph/` — logic duplication including semantically equivalent code expressed differently.
**Date**: 2026-04-10

## Duplication Map

| Pattern | Occurrences | Files | Extractable? |
|---------|------------|-------|-------------|
| Oracle config injection (gen_id + model override into config) | 3 | `factory.py` | Yes -> shared helper |
| Node wrapper boilerplate (log, t0, extract, skip, update, elapsed, log) | 3 | `factory.py` | Partial -> shared preamble/postamble |
| Loop router closures (`_add_loop_back_edge` vs `_add_subgraph_loop`) | 2 | `compiler.py` | Yes -> shared `_wire_loop` |
| DI lint checking for scalar vs bundled model params | 4 | `lint.py` | Yes -> shared `_check_di_binding` |
| Branch arm node addition to graph | 2 | `compiler.py` | Yes -> extract loop body |
| Condition spec building from `_ConditionProxy` or `_Proxy` | 2 | `forward.py` | Yes -> shared helper |
| Sidecar re-registration block after modifier application | 5 | `decorators.py` | Marginal -> documented pattern |
| Body-as-merge Oracle registration | 2 | `decorators.py` | Yes -> shared helper |
| `_collect_oracle_results` and `_append_tagged` reducers | 2 | `state.py` | Marginal -> semantically distinct |
| Each×Oracle + dict-form output field creation | 2 | `state.py` | Yes -> inline into `_add_single_output_field` |

## Findings

### DRY-01: Oracle config injection block repeated in all three LLM wrapper factories

- **Severity**: High
- **Category**: Transport (factory wrappers)
- **Occurrences**: 3 places
- **Files**:
  - `src/neograph/factory.py:338-345` -- `_make_scripted_wrapper`
  - `src/neograph/factory.py:380-387` -- `_make_produce_fn`
  - `src/neograph/factory.py:450-457` -- `_make_tool_fn`
- **Description**: All three LLM-mode wrapper factories contain an identical 8-line block that reads `neo_oracle_gen_id` from state, builds an `extra` dict with `_generator_id` and optionally `_oracle_model`, and merges it into `config["configurable"]`. The logic is byte-for-byte identical across all three closures. The block handles Oracle generator dispatch — when a node runs inside an Oracle fan-out, the per-generator ID and model override must be injected into config so the LLM layer and prompt compiler can access them.
- **Risk**: Any change to Oracle config injection (e.g., adding a new per-generator field, changing the key prefix, handling missing configurable) must be applied in three places. The scripted wrapper was added later during the kqd refactor and appears to have been copy-pasted from the produce wrapper.
- **Proposed extraction**: Module-level helper `_inject_oracle_config(state, config) -> RunnableConfig` in `factory.py`. Each wrapper calls it as `config = _inject_oracle_config(state, config)` — one line replaces eight in each closure.

### DRY-02: Node wrapper preamble/postamble repeated across `_make_scripted_wrapper`, `_make_produce_fn`, `_make_tool_fn`

- **Severity**: Medium
- **Category**: Transport (factory wrappers)
- **Occurrences**: 3 places
- **Files**:
  - `src/neograph/factory.py:326-364` -- `_make_scripted_wrapper`
  - `src/neograph/factory.py:367-431` -- `_make_produce_fn`
  - `src/neograph/factory.py:434-521` -- `_make_tool_fn`
- **Description**: All three wrappers share the same structural skeleton: (1) bind structlog with node name/mode, (2) log `node_start`, (3) `t0 = time.monotonic()`, (4) inject Oracle config, (5) `_extract_input`, (6) `_apply_skip_when`, (7) mode-specific body, (8) `_build_state_update`, (9) `elapsed = time.monotonic() - t0`, (10) log `node_complete`, (11) return update. Steps 1-6 and 8-11 are structurally identical; the only divergent part is the mode-specific body (step 7). The produce and tool wrappers additionally share: `_render_input`, `_resolve_primary_output`, context extraction, and Oracle model override — all in the same order with the same code.
- **Risk**: Medium. The shared skeleton means every cross-cutting concern (new logging field, new skip behavior, new state-update rule) must be patched in three places. The `_make_produce_fn` and `_make_tool_fn` bodies differ primarily in the LLM call (`invoke_structured` vs `invoke_with_tools`) and tool_log handling. This is a classic "template method" candidate but the closures make extraction non-trivial without changing the architecture.
- **Proposed extraction**: A `_make_llm_wrapper(node, invoke_fn)` higher-order function that handles the shared preamble/postamble and delegates the mode-specific call to `invoke_fn(input_data, output_model, ...)`. The scripted wrapper is different enough (no render, no output resolution) that it could stay separate, but the produce/tool pair could merge. Alternatively, extract just the shared preamble as `_wrapper_preamble(state, config, node) -> (input_data, config, t0, node_log, skip_result)` and postamble as `_wrapper_postamble(node, field_name, result, state, t0, node_log) -> dict`.

### DRY-03: Loop router closures duplicated between `_add_loop_back_edge` and `_add_subgraph_loop`

- **Severity**: High
- **Category**: Topology (compiler wiring)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/compiler.py:537-616` -- `_add_loop_back_edge` (Node loop)
  - `src/neograph/compiler.py:619-680` -- `_add_subgraph_loop` (Construct loop)
- **Description**: Both functions follow the exact same pattern: (1) compute `field_name` and `count_field`, (2) add the node to the graph, (3) wire `prev_node` or `START` edge, (4) resolve condition (string vs callable), (5) create `exit_name` and a pass-through `loop_exit` node, (6) create `loop_router` closure that checks `count >= max_iterations`, evaluates the condition, and returns either the re-entry target or `exit_name`, (7) add conditional edges. The loop_router closures are semantically identical — both check `max_iterations`, handle `on_exhaust`, catch `(AttributeError, TypeError)`, and return the same shape. The only differences are: (a) Node loop unwraps the latest value from an append-list and handles dict-form outputs, while Construct loop calls `_unwrap_loop_value`; (b) the re-entry target name differs (`node_name` vs `sub.name`).
- **Risk**: High. The `_add_subgraph_loop` function's docstring explicitly says "Same pattern as `_add_loop_back_edge` but for sub-constructs." Any bug fix to the loop router logic (e.g., the `on_exhaust` behavior, the exception handling, the max_iterations check) must be applied to both. The value unwrap difference is the only genuine divergence and could be parameterized.
- **Proposed extraction**: Merge into `_wire_loop(graph, name, fn, loop, prev_node, unwrap_fn, retry_policy=None) -> str` that takes an `unwrap_fn` parameter for the value extraction. Node loops pass a closure that handles dict-form outputs and append-list unwrap; Construct loops pass `_unwrap_loop_value`. Everything else is identical.

### DRY-04: DI binding check logic repeated 4 times in `lint.py`

- **Severity**: Medium
- **Category**: Validation (lint)
- **Occurrences**: 4 structurally identical blocks
- **Files**:
  - `src/neograph/lint.py:69-97` -- node scalar DI check
  - `src/neograph/lint.py:98-127` -- node bundled model DI check
  - `src/neograph/lint.py:137-164` -- merge_fn scalar DI check
  - `src/neograph/lint.py:166-196` -- merge_fn bundled model DI check
- **Description**: The lint walker has four near-identical blocks that check DI bindings: (1) node-level scalar params (`from_input`/`from_config`), (2) node-level bundled model params (`from_input_model`/`from_config_model`), (3) merge_fn scalar params, (4) merge_fn bundled model params. Each block follows the same pattern: if config is provided, check key exists, else if required, flag as missing. The only difference between node vs merge_fn blocks is the `node_name` label (node name vs merge label) and the error message prefix. The scalar vs model blocks differ in whether they iterate `model_cls.model_fields` or check a single key.
- **Risk**: Medium. Adding a new DI kind (e.g., `from_state`) or changing the lint message format requires touching four places. The merge_fn lint check (neograph-f70z) was clearly copy-pasted from the node lint check with minimal adaptation.
- **Proposed extraction**: A shared `_check_di_param(label, pname, kind, payload, config, issues)` helper that handles both scalar and bundled forms. The lint walker calls it once per param, passing the appropriate label. This reduces the four blocks to two loops (node params + merge_fn params), each calling the shared helper.

### DRY-05: Branch arm node addition duplicated in `_add_branch_to_graph`

- **Severity**: Low
- **Category**: Topology (compiler wiring)
- **Occurrences**: 2 places (loop bodies within one function)
- **Files**:
  - `src/neograph/compiler.py:827-834` -- true arm node addition
  - `src/neograph/compiler.py:836-843` -- false arm node addition
- **Description**: The two `for item in ..._nodes` loops that add branch arm nodes to the graph are byte-for-byte identical (same `isinstance(item, Construct)` check, same `compile()` + `make_subgraph_fn()` for Constructs, same `make_node_fn()` for Nodes, same `graph.add_node()`). Similarly, the sequential edge wiring loops (lines 846-849) are identical. And the "shared nodes" check within `_merge_single_branch` and `_merge_sequential_branches` in `forward.py` (lines 779-780 vs 837-838) duplicates the same true-only/false-only set difference pattern.
- **Risk**: Low. The duplication is local to one function and unlikely to diverge. But if branch arm handling gains complexity (e.g., retry_policy threading, modifier support on branch arms), both loops must be updated.
- **Proposed extraction**: Extract `_add_arm_nodes(graph, arm_nodes)` that handles both Construct and Node items. Both loops become one-liners.

### DRY-06: Condition spec building duplicated between `_merge_single_branch` and `_merge_sequential_branches`

- **Severity**: Low
- **Category**: Tracing (ForwardConstruct)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/forward.py:783-794` -- in `_merge_single_branch`
  - `src/neograph/forward.py:842-851` -- in `_merge_sequential_branches`
- **Description**: Both branch merge functions contain the same 10-line block: check if `condition` is a `_ConditionProxy`, call `_build_runtime_condition()` if so, else build a `_ConditionSpec` with `op_module.truth` for plain proxy truthiness. The fallback `_ConditionSpec` construction is identical.
- **Risk**: Low. Adding a new condition proxy type or changing the fallback behavior requires updating both. The code is short but the duplication is clear.
- **Proposed extraction**: `_build_condition_spec(condition) -> _ConditionSpec` function in `forward.py`. Both merge functions call it.

### DRY-07: Body-as-merge Oracle registration duplicated in `@node` decorator

- **Severity**: Medium
- **Category**: DX (decorator construction)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/decorators.py:726-737` -- Each×Oracle fused path
  - `src/neograph/decorators.py:786-801` -- Oracle-only path
- **Description**: When `models=` is set without `merge_fn` or `merge_prompt`, the `@node` decorator treats the function body as the merge function. Both the Each×Oracle fused path and the Oracle-only path contain the same pattern: synthesize a `body_merge_name`, create a `_make_body_merge` (or `_make_body_merge_fused`) closure factory that wraps `(variants: list, config) -> user_fn(variants)`, register it via `_reg_scripted`, and set `effective_merge_fn`. The inner closure is functionally identical in both paths — the "fused" variant has the same signature and behavior. Variable names differ slightly (`_make_body_merge_fused` vs `_make_body_merge`).
- **Risk**: Medium. Adding new behavior to body-as-merge (e.g., passing config to the user function, injecting DI params) requires updating both paths. The fused path was added later and appears to have been copy-pasted with minor renaming.
- **Proposed extraction**: A shared `_register_body_as_merge(f, node_label) -> str` helper in `decorators.py` that synthesizes the name, builds the closure, registers it, and returns the `effective_merge_fn` name. Both paths call it.

### DRY-08: Sidecar re-registration block repeated after each modifier path

- **Severity**: Low
- **Category**: DX (decorator construction)
- **Occurrences**: 5 places
- **Files**:
  - `src/neograph/decorators.py:756-758` -- after Each×Oracle fused
  - `src/neograph/decorators.py:771-773` -- after Each-only
  - `src/neograph/decorators.py:776-778` -- base path (no modifier)
  - `src/neograph/decorators.py:827-829` -- after Oracle-only
  - `src/neograph/decorators.py:853-855` -- after Operator
  - `src/neograph/decorators.py:868-870` -- after Loop
- **Description**: The pattern `_register_sidecar(n, f, param_names); if param_res: _register_param_resolutions(n, param_res)` is repeated 6 times across different modifier application paths. Each occurrence is identical. This is documented in AGENTS.md ("Any new modifier kwarg you add must follow the same pattern").
- **Risk**: Low. The block is 2-3 lines and documented. However, with 6 occurrences across branching control flow, it's easy to miss one when adding a new modifier kwarg. The Loop path was the most recent addition and follows the established pattern.
- **Proposed extraction**: Optional. A `_finalize_node(n, f, param_names, param_res)` helper would reduce each site to 1 line. The `return` statements that follow some registration blocks make the control flow slightly tricky but not prohibitive.

### DRY-09: Each×Oracle dict-form output field creation duplicated in `_add_output_field` and `_add_single_output_field`

- **Severity**: Low
- **Category**: State (state model generation)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/state.py:232-246` -- dict-form Each×Oracle in `_add_output_field`
  - `src/neograph/state.py:275-287` -- single-type Each×Oracle in `_add_single_output_field`
- **Description**: `_add_output_field` handles dict-form outputs by iterating keys and creating per-key fields with `_append_tagged` collector and `_merge_dicts` reducer. `_add_single_output_field` handles the same Each×Oracle combination for single-type outputs with the same reducer pair. The collector creation pattern (`Annotated[list, _append_tagged]`, `Annotated[field_type, _merge_dicts]`) is identical. The difference is only whether it loops over output keys or handles a single field.
- **Risk**: Low. The reducer assignments are unlikely to change independently. But if a new Each×Oracle state shape is needed, both paths must be updated.
- **Proposed extraction**: Marginal. The dict-form path in `_add_output_field` could delegate per-key to `_add_single_output_field`, but the collector field naming differs (`neo_eachoracle_{field_name}` vs `neo_eachoracle_{field_name}_{key_field}`), making delegation awkward without parameter changes.

### DRY-10: Context field extraction duplicated between `_make_produce_fn` and `_make_tool_fn`

- **Severity**: Medium
- **Category**: Transport (factory wrappers)
- **Occurrences**: 2 places
- **Files**:
  - `src/neograph/factory.py:400-404` -- in `_make_produce_fn`
  - `src/neograph/factory.py:478-483` -- in `_make_tool_fn`
- **Description**: Both LLM wrapper factories contain the same 4-line block: `context_data = None; if node.context: context_data = {name: _state_get(state, name.replace("-", "_")) for name in node.context}`. This is the verbatim context field extraction that injects pre-formatted state fields into the prompt. Identical code, identical variable names, identical dict comprehension.
- **Risk**: If context extraction semantics change (e.g., filtering None values, applying transformers, supporting nested paths), both must be updated. This is a subset of the DRY-02 shared skeleton problem but is independently extractable.
- **Proposed extraction**: `_extract_context(node, state) -> dict | None` in `factory.py`. Both wrappers call it.

## Cleared Areas (Previously Flagged, Now Clean)

1. **Frame-walking namespace builder** (prior DRY-01): Fixed. `_build_annotation_namespace` is now a shared helper called by `_classify_di_params`, `@node` inputs inference, `@node` output inference, `infer_oracle_gen_type`, and `@merge_fn` state param inference. All 5 call sites use the same function with different `frame_depth` values. Clean.

2. **`_make_gather_fn` / `_make_execute_fn` identical bodies** (prior DRY-02): Fixed. Merged into `_make_tool_fn(node)` that reads `node.mode` for the log label. Clean.

3. **`_state_get` polymorphism** (prior DRY-04): Fixed. Module-level `_state_get(state, key)` in `factory.py:72-76` is used by all state access sites. No inline duplicates remain. Clean.

4. **Name collision check** (prior DRY-06): Fixed. The collision check now lives inside `_build_construct_from_decorated` (line 1266) — both `construct_from_module` and `construct_from_functions` delegate to it without pre-checking. Clean.

5. **`effective_producer_type`**: Single source of truth in `_construct_validation.py:31-60`. No inline copies of modifier rules found anywhere. Clean.

6. **`_resolve_di_value`**: Single shared helper in `decorators.py:321-389`. Called by `_resolve_di_args`, `_resolve_merge_args`, and `_register_node_scripted`. No duplicated resolution logic. Clean.

7. **`_unwrap_loop_value` / `_unwrap_each_dict`**: Single helpers in `factory.py:32-69`. Called by `_extract_input`, `_resolve_merge_args`, and `loop_router`. No inline unwrap logic. Clean.

## Summary

- Critical: 0
- High: 2 (DRY-01, DRY-03)
- Medium: 4 (DRY-02, DRY-04, DRY-07, DRY-10)
- Low: 4 (DRY-05, DRY-06, DRY-08, DRY-09)
- Total duplicated logic blocks: 10
- Estimated lines removable by extraction: ~95
  - DRY-01: ~16 lines (3x8 replaced by 3x1 helper call)
  - DRY-02: ~30 lines (shared preamble/postamble; partial — mode-specific body stays)
  - DRY-03: ~40 lines (entire `_add_subgraph_loop` body replaced by `_wire_loop` reuse)
  - DRY-04: ~50 lines (4 blocks replaced by shared `_check_di_param` helper)
  - DRY-05, DRY-06, DRY-07, DRY-08, DRY-09, DRY-10: marginal per-item, ~30 lines total

### Progress Since Prior Review (060426)

- 4 of 6 prior findings addressed (DRY-01 frame-walking, DRY-02 gather/execute merge, DRY-04 `_state_get`, DRY-06 name collision). The codebase is measurably cleaner.
- DRY-03 (DI resolution loop) from the prior review is now clean — `_resolve_di_args` and `_resolve_merge_args` are shared helpers.
- DRY-05 (sidecar re-registration) persists as this review's DRY-08 with one additional occurrence (Loop path added since last review).
- New duplication introduced since last review: Oracle config injection (DRY-01), loop router closures (DRY-03), body-as-merge (DRY-07), context extraction (DRY-10). These are primarily from the Loop-on-Construct and Each×Oracle fusion features.
