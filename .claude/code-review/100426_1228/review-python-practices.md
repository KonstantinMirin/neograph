# Python Practices Review

**Scope**: All of `src/neograph/` (22 files). Focus on Pythonic idioms, Pydantic best practices, async correctness, and error handling.
**Date**: 2026-04-09

## Findings

### PP-01: `ExecutionError` used but not imported in `make_subgraph_fn` scope
- **Severity**: High
- **Category**: Errors / NameError bug
- **File**: `src/neograph/factory.py:912-914`
- **Description**: Inside `make_subgraph_fn`, the inner `subgraph_node` function references `ExecutionError` at line 914 (`raise ExecutionError(...)`) but `ExecutionError` is only imported at the module level (line 24). This works fine — the module-level import IS in scope. However, the comment on line 912 says "Note: ExecutionError is not imported at module level — latent NameError bug." This comment is **wrong**: `ExecutionError` IS imported at line 24 (`from neograph.errors import ConfigurationError, ExecutionError`). The comment is stale/misleading and may cause someone to "fix" something that is not broken, or worse, avoid this code path believing it is buggy. The runtime defense itself is correct.
- **Recommended fix**: Remove or update the misleading comment on line 912. Change "Note: ExecutionError is not imported at module level — latent NameError bug." to something like "Runtime defense: fail loud if no internal node produced the output."

### PP-02: Unreachable code after `raise` in `validate_loop_self_edge`
- **Severity**: Medium
- **Category**: Dead code
- **File**: `src/neograph/_construct_validation.py:101-102`
- **Description**: In `validate_loop_self_edge`, inside the `for _key, expected in input_type.items()` loop, if no compatible input slot is found, a `ConstructError` is raised at line 101. Line 102 has `return` immediately after the `raise`, which is unreachable dead code. This was likely left from a control-flow refactor where the `raise` replaced an earlier `break` + post-loop check.
- **Reproduction**: Read lines 91-102 of `_construct_validation.py`.
- **Recommended fix**: Remove the `return` on line 102.

### PP-03: Silent `except Exception` swallowing in `_resolve_di_value` hides Pydantic validation errors
- **Severity**: Medium
- **Category**: Errors
- **File**: `src/neograph/decorators.py:369-386`
- **Description**: When `_resolve_di_value` attempts to construct a Pydantic model via `model_cls(**field_values)` (line 370), a `ValidationError` from Pydantic would be caught by `except Exception:` on line 371. For non-required models, the code logs a warning and returns `None` — but for the non-required path, the log message doesn't include the actual exception, making debugging harder. For the required path (line 372-378), it raises `_ExecutionError` with a generic message that doesn't include the original exception details. This was flagged in the prior review (PP-01 of 060426_0124) but the fix only partially addressed it: the required path now re-raises, but the non-required path still swallows without including `exc` in the log. A developer debugging a missing DI value has to guess what validation failed.
- **Recommended fix**: Capture the exception (`except Exception as exc:`) and include `str(exc)` in both the warning log and the ExecutionError message. This surfaces field-level Pydantic validation details.

### PP-04: Duplicate `import re` inside function body when already imported at module level
- **Severity**: Low
- **Category**: Resources / Imports
- **File**: `src/neograph/_llm.py:295`
- **Description**: `_parse_json_response` does `import re` at line 295, but `re` is already imported at the module level (line 16). The inner import is redundant. While Python caches imports after the first load, the import machinery still acquires the import lock and does a dict lookup in `sys.modules` on every call. `_parse_json_response` is called on every LLM response, making this a hot path.
- **Recommended fix**: Remove the inner `import re` at line 295.

### PP-05: `_build_annotation_namespace` frame-walking silently swallows all exceptions
- **Severity**: Medium
- **Category**: Errors
- **File**: `src/neograph/decorators.py:218-228`
- **Description**: The frame-walking block in `_build_annotation_namespace` catches `except Exception: pass`. This means any error during frame introspection — including `RecursionError`, `SystemError`, or bugs in the walking logic itself — is silently swallowed. The function returns a partial namespace, which may cause `get_type_hints()` to fail later with an inscrutable `NameError` ("name 'MyClass' is not defined") that points the developer to the wrong place. The `except (TypeError, ValueError): pass` for `inspect.getclosurevars` at line 216 is appropriately narrow. The frame-walking block should follow the same pattern.
- **Recommended fix**: Narrow the exception to `except (AttributeError, TypeError, ValueError): pass` — these are the actual failure modes of frame attribute access. Let unexpected exceptions propagate.

### PP-06: `classify_modifiers` rebuilds the combo_map dict on every call
- **Severity**: Low
- **Category**: Resources / Performance
- **File**: `src/neograph/modifiers.py:70-81`
- **Description**: `classify_modifiers` constructs a `combo_map` dict of 10 frozenset entries on every invocation. This function is called from the compiler, state builder, and factory for every node — potentially hundreds of times during a single `compile()`. The map is static and never changes.
- **Recommended fix**: Promote `combo_map` to a module-level constant: `_COMBO_MAP: dict[frozenset[str], ModifierCombo] = { ... }`. The `frozenset` keys are immutable and hashable, so this is safe.

### PP-07: `classify_modifiers` imports `ConstructError` inside the function body
- **Severity**: Low
- **Category**: Imports
- **File**: `src/neograph/modifiers.py:47`
- **Description**: `classify_modifiers` does `from neograph.errors import ConstructError` inside the function body. This deferred import was likely added to avoid circular imports, but `neograph.errors` has no imports from `neograph.modifiers` — the cycle doesn't exist. The `Modifiable.__or__` method at line 141 also imports `ConstructError` from `neograph.errors` inline, and `Oracle.model_post_init` imports `ConfigurationError` at the module level (line 18). The function-level import adds per-call overhead for no benefit.
- **Recommended fix**: Move to module-level import.

### PP-08: `Node.tools` default value uses mutable `list` literal
- **Severity**: Medium
- **Category**: Pydantic best practices
- **File**: `src/neograph/node.py:56`
- **Description**: `tools: list[Tool] = []` uses a mutable default. Pydantic v2 handles this safely (it copies the default for each instance), so this is not a runtime bug. However, it is a Pydantic best practice violation — the canonical pattern is `tools: list[Tool] = Field(default_factory=list)`, which makes the intent explicit and avoids reliance on Pydantic's implicit copy behavior. The same file uses `Field(default_factory=dict)` for `llm_config` (line 53), making the inconsistency visible.
- **Recommended fix**: Change to `tools: list[Tool] = Field(default_factory=list)` for consistency with the `llm_config` field two lines earlier.

### PP-09: `_extract_input` has complex multi-path logic with deeply nested conditionals
- **Severity**: Medium
- **Category**: Pythonic idioms / Readability
- **File**: `src/neograph/factory.py:538-631`
- **Description**: `_extract_input` is 93 lines with 6+ levels of nesting and 4 distinct code paths (Loop re-entry with dict-form, Loop re-entry with single-type, Each fan-out, fan-in dict, single-type scan). Each path has its own unwrap/dispatch logic. The multi-key dict loop re-entry path (lines 567-591) is particularly dense: it iterates over `node.inputs.items()`, checks `_state_get`, compares `state_key != own_field`, unwraps lists, and has a `placed_latest` fallback. While the code is correct (heavily tested), the complexity makes it fragile for future changes — a new modifier or input form requires understanding all 4 paths.
- **Recommended fix**: Extract the loop re-entry paths into a helper `_extract_loop_reentry(state, node) -> Any | None`. Return early if non-None. This cuts the nesting depth and makes each path independently readable.

### PP-10: `_is_instance_safe` re-imports `get_origin` and `get_args` already available at module scope
- **Severity**: Low
- **Category**: Imports
- **File**: `src/neograph/factory.py:526`
- **Description**: `_is_instance_safe` does `from typing import get_origin, get_args` inside the function body. Both `get_origin` (imported as `_get_origin` at line 18) and the typing module are already available at module scope. This inner import runs on every invocation of `_is_instance_safe`, which is called for every state field scan in `_extract_input`.
- **Recommended fix**: Use the already-imported `_get_origin` at module level. Remove the inner import.

### PP-11: Oracle code duplication across produce/scripted/tool wrappers
- **Severity**: Low
- **Category**: DRY / Idioms
- **File**: `src/neograph/factory.py:337-345, 378-388, 449-458`
- **Description**: The "inject oracle generator ID + model override into config" block is copied verbatim across `_make_scripted_wrapper`, `_make_produce_fn`, and `_make_tool_fn`. Each block does the same 8 lines: read `neo_oracle_gen_id` from state, build an `extra` dict, read `neo_oracle_model`, merge into config. This violates DRY and means a bug fix (or new oracle-related config field) must be applied in three places.
- **Recommended fix**: Extract a helper `_inject_oracle_config(state, config) -> RunnableConfig` and call it in each wrapper.

### PP-12: `Modifiable.__or__` has duplicated mutual-exclusion error messages
- **Severity**: Low
- **Category**: DRY / Idioms
- **File**: `src/neograph/modifiers.py:154-181`
- **Description**: The Each+Loop and Oracle+Loop mutual exclusion checks each have two branches that produce identical error messages (one for "applying Loop when Each exists" and the reverse). This results in 4 nearly-identical blocks with 28 lines total for 2 logical checks. Using a helper or collapsing the direction check would halve the code.
- **Recommended fix**: For each pair, combine into a single check: `if {type(modifier), Each} <= {type(modifier)} | {type(m) for m in self.modifiers}: raise ...`. Or simpler: check after the isinstance chain and produce the message once.

### PP-13: `runner.run` mutates the caller's `config` dict via `setdefault`
- **Severity**: High
- **Category**: Side effects
- **File**: `src/neograph/runner.py:77`
- **Description**: `run()` does `configurable = config.setdefault("configurable", {})` followed by `configurable["_neo_input"] = input`. This mutates the caller's original `config` dict by adding `_neo_input` to `configurable`. If the caller reuses the same config dict for multiple `run()` calls (common in tests and orchestration layers), the `_neo_input` key from the first call leaks into subsequent calls, potentially causing stale data bugs. The `_inject_input_to_config` helper (line 37-42) correctly creates a new dict with `{**config, "configurable": merged}`, but the mutation at line 77-78 happens BEFORE that helper is called. The mutation is intentional (the comment says "stash input in the CALLER'S config so resume can re-inject"), but the side effect is surprising and undocumented.
- **Recommended fix**: Copy config before mutation: `config = {**config}; config["configurable"] = {**config.get("configurable", {}), "_neo_input": input}`. This preserves the resume-re-inject behavior without mutating the caller's dict.

### PP-14: `loader._build_sub_construct` mutates Node instances via direct field assignment
- **Severity**: Medium
- **Category**: Pydantic best practices
- **File**: `src/neograph/loader.py:208-215`
- **Description**: `_build_sub_construct` mutates Node instances' `.inputs` field directly (`node.inputs = input_type`, `node.inputs = inputs_dict`). These Node instances come from `all_nodes`, which is the shared `node_defs` dict built from the spec's node list. If the same node appears in multiple sub-constructs (or in both a sub-construct and the pipeline), the mutation from the first sub-construct affects the second. Pydantic v2 models allow attribute assignment by default, but mutating shared instances is a structural bug waiting to happen. The `@node` decorator path uses `model_copy(update=...)` to avoid this pattern.
- **Recommended fix**: Use `node = node.model_copy(update={"inputs": ...})` instead of direct assignment, matching the pattern used everywhere else in the codebase.

### PP-15: `_llm.py` module-level mutable state with no thread safety
- **Severity**: Medium
- **Category**: Thread safety
- **File**: `src/neograph/_llm.py:30-41`
- **Description**: `_llm_factory`, `_prompt_compiler`, `_global_renderer`, and `_cost_callback` are module-level mutable globals mutated by `configure_llm()`. In concurrent scenarios (e.g., running two pipelines with different LLM configs in the same process), calling `configure_llm()` in one thread replaces the globals for all threads. LangGraph supports concurrent graph invocations, so this is a plausible real-world scenario. The `global` statement at line 131 with 6 variables in one line is also a style concern (though suppressed by `noqa: PLW0603`).
- **Recommended fix**: For the single-user, single-pipeline use case this is fine. If concurrent pipeline support is ever needed, wrap the globals in a `contextvars.ContextVar` or pass the LLM config through the `RunnableConfig` pipeline (which LangGraph already threads through). For now, document the single-pipeline constraint.

## Summary

- Critical: 0
- High: 2
- Medium: 7
- Low: 6

## Automated Check Results (no findings)

The following automated checks from the Python practices checklist returned clean results for all in-scope files:

- **Pydantic v1 patterns**: No `@validator`, `@root_validator`, `.dict()`, or `.parse_obj()` calls
- **Async/Sync mixing**: No `asyncio.run()`, `run_async_in_sync`, or mixed async/sync calls
- **f-string logging**: No `logger.info(f"...")` patterns (project uses structlog with keyword args throughout)
- **Mutable function defaults**: No `def f(x=[])` or `def f(x={})` patterns in function signatures
- **Bare `except:`**: No bare except clauses (all use `except Exception` or narrower)
- **File handles**: No `open()` calls without context managers (the project doesn't directly open files; `loader.py` uses `Path.read_text()`)
- **SQLAlchemy**: Not applicable to this project
- **FastAPI imports in business logic**: Not applicable to this project
