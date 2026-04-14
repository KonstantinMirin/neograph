# Code Review Synthesis — 2026-04-10

**Scope**: Best practices violation patterns across all of `src/neograph/` and `tests/`
**Agents**: 6 ran, 6 produced findings
**Date**: 2026-04-10

## Validation Summary

| Agent | Raw Findings | High+ | Medium | Low |
|-------|-------------|-------|--------|-----|
| python-practices | 10 | 1 | 5 | 4 |
| consistency | 10 | 0 | 3 | 7 |
| dry | 10 | 2 | 4 | 4 |
| layering | 10 | 0 | 3 | 7 |
| testing | 10 | 3 | 4 | 3 |
| security | 4 | 0 | 2 | 2 |
| **Total** | **54** | **6** | **21** | **27** |

## High Findings

### HIGH-01: Oracle config injection duplicated across 3 factory wrappers
- **Source**: review-dry (DRY-01)
- **Files**: `factory.py` lines 298-305, 339-346, 407-414
- **Pattern**: The `neo_oracle_gen_id` + `neo_oracle_model` config injection block is copy-pasted in `_make_scripted_wrapper`, `_make_produce_fn`, and `_make_tool_fn`
- **Cross-ref**: review-python-practices also flagged this as duplicated imperative code
- **Action**: Extract `_inject_oracle_config(state, config) -> config` helper

### HIGH-02: Loop router closures structurally identical
- **Source**: review-dry (DRY-03)
- **Files**: `compiler.py` `_add_loop_back_edge` and `_add_subgraph_loop`
- **Pattern**: Both functions build identical loop_router closures with the same count/exhaust/condition logic. Only the value-unwrap path differs.
- **Action**: Extract shared `_make_loop_router(field_name, loop, condition, unwrap_fn)` factory

### HIGH-03: Empty test classes masquerading as coverage
- **Source**: review-testing
- **Files**: `test_obligation_r1r2.py`, `test_coverage_gaps.py`
- **Pattern**: Some test classes have `pass` bodies or were generated with placeholder tests that don't assert behavior
- **Action**: Audit and either fill with real assertions or remove

### HIGH-04: Forgiving scripted functions in tests mask wiring bugs
- **Source**: review-testing
- **Pattern**: Test scripted functions use `isinstance(input_data, dict)` + fallback that accepts any input shape. If the factory wires wrong input, the test still passes.
- **Action**: Make test functions strict — assert expected input type, fail on wrong shape

### HIGH-05: Coverage-gap tests with weak assertions
- **Source**: review-testing
- **Pattern**: Some tests from the coverage push assert `is not None` or `isinstance(result, X)` without verifying the actual value. The line executes but the behavior isn't validated.
- **Action**: Strengthen assertions to check specific values

### HIGH-06: Bare `except Exception` in _resolve_di_value
- **Source**: review-python-practices (PP-01)
- **File**: `decorators.py:369-386`
- **Pattern**: Bundled model construction failure caught with bare `except Exception`. If required=False, logs warning and returns None. The broad catch hides unexpected errors (e.g., TypeError from wrong field types).
- **Action**: Catch `(ValidationError, TypeError)` specifically

## Medium Findings (top patterns)

### Pattern A: Inconsistent error message formatting (3 agents flagged)
- consistency, python-practices, layering all noted that error messages mix f-string styles, some include `_location_suffix()` and some don't, some use `ConstructError` and some use `CompileError` for similar checks.

### Pattern B: DI binding check 4x duplication in lint.py
- review-dry (DRY-04): node scalar, node model, merge_fn scalar, merge_fn model — four copies of the same check-config-key-present logic. Extract `_check_di_binding(node_name, param, kind, payload, config, issues)`.

### Pattern C: Body-as-merge Oracle registration duplicated
- review-dry (DRY-07): The `_make_body_merge` closure + `register_scripted` block appears in both the Each×Oracle fused path and the Oracle-only path in decorators.py. Extract shared helper.

### Pattern D: Context field extraction duplicated
- review-dry (DRY-10): `context_data = {name: _state_get(...) for name in node.context}` appears in both `_make_produce_fn` and `_make_tool_fn`.

### Pattern E: Manual monkeypatching in test_cli.py
- review-testing: Uses `try/finally` to swap module attributes instead of pytest's `monkeypatch` fixture. Fragile and doesn't auto-restore on test failure.

## Low Findings (summary)

| Count | Pattern |
|-------|---------|
| 7 | Naming inconsistency (mix of `_neo_` prefix conventions, field_name vs node_name) |
| 4 | Missing type annotations on internal helper return types |
| 3 | Log level inconsistency (some guards log at warning, some at debug, some at error) |
| 3 | Layering: some utility functions in wrong module (e.g., `_type_name` in factory.py) |
| 2 | Security: YAML size limit is per-string only, not per-file when reading from Path |
| 2 | Cosmetic: comment style inconsistency (some use ═══ banners, some don't) |

## Patterns Observed (Cross-Agent)

1. **Factory wrapper duplication** — flagged by dry, python-practices, and layering. The 3 LLM wrappers (`_make_scripted_wrapper`, `_make_produce_fn`, `_make_tool_fn`) share a common preamble (log, extract, skip_when, oracle config) and postamble (build_state_update, log complete) with mode-specific logic in the middle. A template method or decorator could eliminate ~60 lines.

2. **Coverage-driven test weakness** — flagged by testing. The recent push to 99% coverage introduced tests that execute lines without validating behavior. The obligation analysis methodology is better than line coverage, but some tests from the coverage agents are "execute, don't assert."

3. **Error message inconsistency** — flagged by consistency, python-practices. Error messages across 5 source files use different patterns for location hints, upstream listings, and fix suggestions. A shared error builder would standardize.

## Security Assessment

Low risk overall. The main surface is:
- YAML parsing uses `safe_load` + 1MB size limit (good)
- `parse_condition` restricts to safe ops (good)
- Prompt compilation delegates to user code — no injection risk in neograph itself
- Tool execution is sandboxed per LangChain's tool protocol

No critical or high security findings.
