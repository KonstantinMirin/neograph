# Consistency Review

**Scope**: All files under `src/neograph/` on branch `develop` (HEAD d854c40).
**Date**: 2026-04-10

## Convention Inventory

| Convention | Canonical Form | Files Following | Files Diverging |
|------------|---------------|-----------------|-----------------|
| Node input field | `Node.inputs` (plural) | 22 | 0 |
| Construct boundary field | `Construct.input` (singular) | 8 | 0 |
| Runtime API parameter | `run(input={...})` (singular) | 1 (runner) | 0 |
| Error class for assembly | `ConstructError` | 3 (decorators, _construct_validation, modifiers) | 0 |
| Error class for compile | `CompileError` | 2 (compiler, state) | 0 |
| Error class for config | `ConfigurationError` | 5 (factory, _llm, modifiers, spec_types, loader) | 0 |
| Error class for runtime | `ExecutionError` | 3 (factory, _llm, compiler) | 0 |
| Bare ValueError | Runner, loader, conditions | 3 | See CON-01 |
| Error prefix (validation) | `"Node '{name}' in construct '{name}'"` | 1 (_construct_validation) | See CON-02 |
| Error prefix (@node) | `"@node '{label}'"` | 1 (decorators) | 0 |
| Error prefix (compile) | `"Node '{name}'"` or `"Construct '{name}'"` | 2 (compiler, state) | 0 |
| Logging library | `structlog.get_logger()` as `log` | 5 (factory, compiler, _llm, loader, state) | See CON-03 |
| Log event names | `snake_case` verbs (`node_start`, `compile_start`) | 5 | See CON-04 |
| Import style | Absolute `from neograph.X import Y` | 22 | 0 |
| Name→field conversion | `name.replace("-", "_")` | 22 | See CON-05 |
| Synthesized names | `_node_{purpose}_{id:x}` | 2 (decorators) | 0 |
| `None` sentinel | `= None` field default | 8 | 0 |
| `dict` field default | `Field(default_factory=dict)` | 3 (node, construct, tool) | 0 |
| `list` field default | Inline `= []` (tools, modifiers) | 2 | See CON-06 |
| `from __future__ import annotations` | Present | 18 | See CON-07 |

## Findings

### CON-01: `runner.py` and `conditions.py` raise bare `ValueError` instead of typed errors

- **Severity**: Medium
- **Convention**: Post-error-hierarchy (errors.py), all neograph-originated errors should use the typed hierarchy: `ConstructError`, `CompileError`, `ConfigurationError`, or `ExecutionError`.
- **Files**: `runner.py:83`, `conditions.py:62,114,125`, `loader.py:77`
- **Description**: Five `raise ValueError(...)` call sites remain outside the error hierarchy. `runner.py:83` raises `ValueError("Either input or resume must be provided.")` -- this is a configuration/usage error that should be `ConfigurationError`. `conditions.py` raises `ValueError` for invalid condition expressions -- these are configuration errors (bad spec input). `loader.py:77` raises `ValueError` for oversized specs -- also a configuration error. All predate the introduction of the error hierarchy in `errors.py` and were never migrated. User code catching `except NeographError` will not catch these.
- **Reproduction**:
  ```
  grep -n "raise ValueError" src/neograph/runner.py src/neograph/conditions.py src/neograph/loader.py
  ```
- **Recommended fix**: Change all five to `ConfigurationError`. The hierarchy docstring says `ConfigurationError` covers "bad or missing configuration" which matches each case.

### CON-02: `_construct_validation.py` error messages have inconsistent capitalization of the entity prefix

- **Severity**: Low
- **Convention**: Validation error messages prefix with `"Node '{name}' in construct '{name}'"` (capital-N "Node").
- **Files**: `_construct_validation.py:95,107` use lowercase `"loop on node"`, while lines 150, 187, 197, 218, 434, 444, 483, 505, 519, 654 use capital `"Node"`.
- **Description**: The `validate_loop_self_edge` function (lines 95 and 107) uses lowercase `"loop on node '{node.name}'"` in its error messages, while every other error message in the same file and across the codebase uses `"Node '{name}'"` with a capital N. The `validate_loop_construct` function at line 316 similarly uses `"Loop on construct"` (capital L but lowercase c) vs `"Construct '{name}'"` used elsewhere. This is a cosmetic divergence visible in user-facing error messages.
- **Reproduction**: Compare lines 95, 107, 316 with lines 150, 187, 654 in `_construct_validation.py`.
- **Recommended fix**: Capitalize consistently: `"Loop on node '{node.name}'"` and `"Loop on construct '{construct.name}'"`.

### CON-03: `structlog.get_logger()` vs `structlog.get_logger(__name__)` inconsistency

- **Severity**: Low
- **Convention**: Module-level loggers should use a consistent form.
- **Files**:
  - `get_logger()` (no args): `factory.py:29`, `compiler.py:36`, `_llm.py:27`, `state.py:19`, `loader.py:22`
  - `get_logger(__name__)`: `spec_types.py:23`
  - Inline `structlog.get_logger(__name__)`: `decorators.py:379`, `_construct_validation.py:214`
- **Description**: Six module-level loggers use `structlog.get_logger()` (unnamed), one uses `structlog.get_logger(__name__)` (named), and two call sites create loggers inline with `__name__`. With structlog, unnamed loggers produce events without a `logger` key, while named loggers include `logger=neograph.spec_types` etc. The inline calls in `decorators.py:379` and `_construct_validation.py:214` don't reuse the module logger because those files don't have one -- they import structlog ad-hoc.
- **Reproduction**:
  ```
  grep -n "get_logger" src/neograph/*.py
  ```
- **Recommended fix**: Pick one form and apply everywhere. `get_logger()` (no args) is the majority form (5 files) and the structlog recommended default. Change `spec_types.py:23` to `log = structlog.get_logger()`. Add `log = structlog.get_logger()` to `decorators.py` and `_construct_validation.py` and use the module-level `log` instead of inline creation.

### CON-04: `_construct_validation.py:214` uses `log.error()` for a non-fatal warning

- **Severity**: Medium
- **Convention**: `log.error()` is not used anywhere else in the codebase; all warnings use `log.warning()` or `warnings.warn()`.
- **Files**: `_construct_validation.py:214`
- **Description**: The `loop_skip_when_no_skip_value` log event uses `structlog.get_logger(__name__).error(...)` -- this is the only `log.error()` call in the entire codebase. The surrounding comment says "Warn rather than reject" and the behavior is explicitly documented as valid-but-easy-to-misuse. An `error` log level is inconsistent with the stated intent of warning. Every other soft warning in the codebase uses either `warnings.warn()` (for user-facing, e.g. dead-body warning at `decorators.py:582`) or `log.warning()` (for operational, e.g. `compiler.py:769`).
- **Reproduction**: `grep -rn "\.error(" src/neograph/` returns only this one hit.
- **Recommended fix**: Change `structlog.get_logger(__name__).error(...)` to `structlog.get_logger(__name__).warning(...)` (or `dev_warn()` from `_dev_warnings.py` if this should be dev-mode only).

### CON-05: Quote style inconsistency in `name.replace("-", "_")` calls

- **Severity**: Low (cosmetic)
- **Convention**: The canonical form uses double quotes: `name.replace("-", "_")`.
- **Files**: `state.py:175`, `compiler.py:551`, `compiler.py:632` use single quotes: `replace('-', '_')`.
- **Description**: 3 of ~30 `replace` call sites use single quotes while the rest use double quotes. Python treats them identically, but the inconsistency is visible in grep/review. All three diverging sites are in loop-related code added at the same time.
- **Reproduction**: `grep -n "replace('-', '_')" src/neograph/state.py src/neograph/compiler.py`
- **Recommended fix**: Normalize to double quotes for grep consistency.

### CON-06: `list` field defaults on Node use bare `= []` instead of `Field(default_factory=list)`

- **Severity**: Medium
- **Convention**: Mutable field defaults on Pydantic models should use `Field(default_factory=...)` to avoid shared-instance bugs.
- **Files**: `node.py:56` (`tools: list[Tool] = []`), `node.py:93` (`modifiers: list[Modifier] = []`)
- **Description**: Node declares `tools: list[Tool] = []` and `modifiers: list[Modifier] = []` with bare list literals as defaults. Pydantic v2 handles this correctly (it creates a new list per instance) so this is not a bug -- but it diverges from the convention established in the same file and in `construct.py`, which both use `Field(default_factory=list)` for their dict fields (`llm_config`, `nodes`). Construct uses `modifiers: list[Modifier] = Field(default_factory=list)` at line 79, while Node uses `modifiers: list[Modifier] = []` at line 93.
- **Reproduction**: Compare `node.py:56,93` with `construct.py:61,71,79`.
- **Recommended fix**: Change both to `Field(default_factory=list)` for consistency with Construct's pattern. Not a bug -- purely convention alignment.

### CON-07: `__main__.py` and `_dev_warnings.py` lack `from __future__ import annotations`

- **Severity**: Low
- **Convention**: Every source file opens with `from __future__ import annotations`.
- **Files**: `__main__.py` (if it has any annotations), `_dev_warnings.py` (line 1-14)
- **Description**: `_dev_warnings.py` does import `from __future__ import annotations` (line 5). Checking more carefully, only `renderers.py` does not have it -- let me verify.
- **Correction**: After re-reading, `_dev_warnings.py:3` does have `from __future__ import annotations`. All 22 Python files include the future import. No finding here.
- **Recommended fix**: None needed.

### CON-08: Error message format divergence between `factory.py` scripted lookup and the shared `lookup_scripted` function

- **Severity**: Low
- **Convention**: Lookup failures should use `ConfigurationError` with the standard message format.
- **Files**: `factory.py:289-291` vs `factory.py:109-115`
- **Description**: `make_node_fn` at line 289 duplicates the scripted registry check: `if node.scripted_fn not in _scripted_registry: raise ConfigurationError(msg)`. The canonical path is `lookup_scripted()` at line 109-115 which does the same check. The messages are nearly identical but not byte-for-byte equal. The duplication means `_make_scripted_wrapper` (line 328) does a second lookup via `_scripted_registry[node.scripted_fn]` that cannot fail (the guard at 289 already passed). The guard-then-lookup pattern works but the duplicated message is a maintenance risk.
- **Reproduction**: Compare `factory.py:289-291` with `factory.py:109-115`.
- **Recommended fix**: Replace lines 289-292 with `lookup_scripted(node.scripted_fn)` as a validation call (discard result), or remove the guard and let `_make_scripted_wrapper` call `lookup_scripted` which raises `ConfigurationError` on miss.

### CON-09: `Modifiable.__or__` error messages omit the item name for the first two guards

- **Severity**: Low
- **Convention**: ConstructError messages include the item name for context (e.g., `"Node '{name}' has both Each and Loop"` in `_construct_validation.py:187`).
- **Files**: `modifiers.py:155-181`
- **Description**: The Each+Loop and Oracle+Loop mutual exclusion guards in `Modifiable.__or__` at lines 155-181 produce messages like `"Cannot combine Each and Loop on the same item."` without naming the item. The belt-and-suspenders checks in `_construct_validation.py:187-201` include the node name: `"Node '{item.name}' has both Each and Loop modifiers."`. Since `__or__` fires at `|` time (before the item is in a Construct), the name is available via `getattr(self, 'name', '?')` -- the same pattern used for dev warnings at line 191 (`getattr(self, 'name', '?')`).
- **Reproduction**: Compare `modifiers.py:155-165` with `_construct_validation.py:187-191`.
- **Recommended fix**: Add `getattr(self, 'name', '?')` to the four mutual-exclusion messages in `__or__` so they include the item name. Already done for Oracle dev-warnings at line 191.

### CON-10: `classify_modifiers()` in `modifiers.py` is defined but never called

- **Severity**: Low
- **Convention**: Dead code should not accumulate.
- **Files**: `modifiers.py:38-89` (`classify_modifiers` function and `ModifierCombo` enum)
- **Description**: `classify_modifiers()` and the `ModifierCombo` enum are defined at lines 18-89 but never imported or called from any file in the codebase. A grep for `classify_modifiers` and `ModifierCombo` across `src/neograph/` returns only the definition site. The docstring says "Every dispatch site matches on this enum instead of ad-hoc has_modifier() chains" (neograph-35c3), but this was never wired -- all dispatch sites (`compiler.py`, `state.py`, `factory.py`) still use `has_modifier()/get_modifier()` chains. This is dead code from an incomplete refactor.
- **Reproduction**:
  ```
  grep -rn "classify_modifiers\|ModifierCombo" src/neograph/ --include="*.py"
  ```
  Returns only `modifiers.py` itself.
- **Recommended fix**: Remove `ModifierCombo` and `classify_modifiers` since they are unused, or complete the migration to use them at dispatch sites (which would be a larger scope change).

### CON-11: `_construct_validation.py` uses `structlog.get_logger(__name__).error()` inline instead of a module-level logger

- **Severity**: Low
- **Convention**: All other files that log define `log = structlog.get_logger()` at module level.
- **Files**: `_construct_validation.py:214`
- **Description**: This file has no module-level `log` variable. The single log call at line 214 creates a logger inline: `structlog.get_logger(__name__).error(...)`. This is the only file in the codebase that logs but does not have a module-level `log` binding. It also means the logger is recreated on every call to `_validate_node_chain` that hits this branch, which is functionally harmless (structlog caches loggers) but inconsistent with the pattern in all other files.
- **Reproduction**: `grep -n "^log = " src/neograph/*.py` shows 5 files; `_construct_validation.py` is absent.
- **Recommended fix**: Add `log = structlog.get_logger()` at module level and use `log.warning(...)` instead of the inline creation. (Also addresses CON-04's severity/error level.)

## Summary

- Critical: 0
- High: 0
- Medium: 3 (CON-01, CON-04, CON-06)
- Low: 8 (CON-02, CON-03, CON-05, CON-08, CON-09, CON-10, CON-11, withdrawn CON-07)

## Overall Assessment

The codebase is highly consistent in its core conventions. The error hierarchy (`errors.py`) is well-structured and the boundary between `ConstructError` (assembly), `CompileError` (compilation), `ConfigurationError` (config), and `ExecutionError` (runtime) is clean and well-maintained across 80+ raise sites. The three medium findings are:

1. **Five bare `ValueError` sites** in runner, conditions, and loader that predate the error hierarchy and were never migrated. These break the `except NeographError` catch-all guarantee.

2. **The sole `log.error()` call** in the codebase is used for a non-fatal warning, contradicting both the surrounding comment ("warn rather than reject") and every other soft-warning site which uses `log.warning()` or `warnings.warn()`.

3. **Node's list field defaults** use bare `= []` while Construct uses `Field(default_factory=list)` for the equivalent `modifiers` field. Not a bug in Pydantic v2, but a convention divergence within two closely related models.

No convention violations found in:
- Import organization (all absolute, consistent style)
- Error class usage at the assembly/compile/config/runtime boundary
- Log event naming (`snake_case` throughout)
- Name-to-field conversion (consistent `replace("-", "_")` pattern)
- Boolean/flag conventions (`has_modifier` pattern is uniform)
- Configuration access (all through `config["configurable"]`)
- `from __future__ import annotations` (universal)
- `__init__` positional-name pattern (consistent on Node, Tool, Construct)
- Sidecar re-registration after `|` modifier application
- `effective_producer_type` single-source-of-truth for modifier-aware types

The dead code finding (CON-10: `classify_modifiers`/`ModifierCombo`) is notable as it represents an incomplete migration from the ad-hoc `has_modifier()` pattern to an enum-based dispatch. All dispatch sites still use the old pattern.
