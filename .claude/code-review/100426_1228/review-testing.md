# Testing Best Practices Review

**Scope**: All files in `tests/` (1076 tests across 20+ files)
**Focus**: Behavior vs mock testing, assertion quality, coverage honesty, mock correctness

---

## Executive Summary

The test suite is **strong overall** — tests predominantly verify real behavior through compile-and-run integration paths rather than mocking internals. The fake infrastructure (`fakes.py`) is well-designed and reusable. However, there are specific patterns that weaken coverage honesty or test value.

**Severity scale**: CRITICAL = likely masking real bugs; HIGH = meaningful gap in test value; MEDIUM = quality concern; LOW = style/maintenance issue.

---

## Findings

### 1. [HIGH] Empty test classes masquerading as coverage

**File**: `tests/test_obligation_r1r2.py:417-427`

```python
class TestTokenDoubleCountEdgeCases:
    """Probe: strategy detection and usage accounting edge cases.
    ...
    These paths are covered by existing integration tests. The probe here
    documents the expected behavior without duplicating coverage.
    """
    pass
```

An empty test class with a `pass` body and a docstring claiming coverage exists elsewhere. This is misleading — it occupies mental space in test discovery, inflates the apparent scope of the test file, and makes it look like token double-counting is tested when nothing here actually runs. Either write the test or delete the class.

**Also at**: `test_coverage_gaps.py:384-406` — `test_loop_router_handles_non_list_state_value` has a `pass` body with an extensive comment explaining why the test is hard to write. This is a documented known gap, which is better than silence, but a `pytest.skip("defensive branch — requires state manipulation")` would be more honest in test output.

**Also at**: `test_coverage_gaps.py:295-299` — `test_eachoracle_as_first_node_is_impractical` has `pass` with a comment "Covered by pragma: no cover in compiler.py." This is a test about *not* testing — it should not exist as a test function.

---

### 2. [HIGH] Coverage-gap tests that verify wiring, not behavior

**File**: `tests/test_coverage_gaps.py` (entire file)

The file header says "minimal tests written to achieve 100% line coverage." Many tests in this file exercise code paths but make weak assertions:

- `test_operator_added_after_loop_on_sub_construct` (line 257): asserts only `result["inner"] is not None` — does not verify the operator actually fired or the loop completed correctly.
- `test_branch_with_construct_in_true_arm` (line 555): asserts `result.get("br_sub") is not None` — does not verify the branch actually took the true arm based on the condition, or that the sub-construct produced the expected output.
- `test_subgraph_loop_starts_from_start_when_first_node` (line 440): asserts `result["inner"] is not None` — a `None` check on a non-None-returning pipeline is nearly tautological.

These tests provide line coverage but not behavioral confidence. If the operator silently broke and the pipeline still completed, these assertions would still pass.

**Recommendation**: Strengthen assertions to verify the specific behavior: check output values, verify call counts, assert on intermediate state.

---

### 3. [MEDIUM] Manual monkeypatching instead of pytest fixtures

**File**: `tests/test_cli.py:275-373`

`test_compile_error_via_file` and `test_lint_issues_displayed` do manual monkeypatching with try/finally blocks:

```python
cli_mod._import_module = patched_import
try:
    neograph.compiler.compile = bad_compile
    try:
        result = cmd_check(args)
    finally:
        neograph.compiler.compile = orig_compile
finally:
    cli_mod._import_module = original_import
```

This is fragile — if an assertion fails between the patch and the restore, cleanup still runs (via finally), but the pattern is error-prone and harder to read than `monkeypatch.setattr()` which is already available via the pytest fixture. The `monkeypatch` fixture is already used elsewhere in this same file (e.g., `test_no_command_exits_0`).

---

### 4. [MEDIUM] Fake LLMs in test_cli.py test are testing its own mock logic

**File**: `tests/test_cli.py:236-273`

`test_compile_error_shows_fail` manually constructs errors and formats them, then asserts on the format:

```python
try:
    raise CompileError("test compile error")
except (CompileError, ConstructError) as exc:
    errors.append(f"compile: {exc}")
...
assert len(errors) == 3
assert "compile:" in errors[0]
```

This test never calls `cmd_check` — it reimplements part of cmd_check's logic inline and then asserts on its own reimplementation. It tests nothing about the actual code. The `test_compile_error_via_file` test immediately below *does* test the real code path, making this one redundant.

---

### 5. [MEDIUM] Assertion on string containment without boundary checking

**File**: Multiple locations across test suite

Many error-path tests use loose string containment checks:

- `test_validation.py:54`: `assert "declares inputs=Claims" in msg` — would match `"declares inputs=ClaimsSummary"`.
- `test_validation.py:55`: `assert "node 'a': RawText" in msg` — reasonable but fragile if message format changes.
- `test_node_decorator.py:1208-1209`: Asserts `"test_node_decorator.py:" in msg` AND `"/tests/test_node_decorator.py:" not in msg` — good, this is the right pattern for boundary checking.

The string containment pattern is acceptable for error messages but becomes a problem when assertions are too loose. The `pytest.raises(match=...)` pattern used elsewhere is preferable.

---

### 6. [MEDIUM] `conftest.py` registry cleanup is comprehensive but brittle

**File**: `tests/conftest.py`

The autouse fixture directly clears internal module-level registries:

```python
factory._scripted_registry.clear()
factory._condition_registry.clear()
factory._tool_factory_registry.clear()
_llm._llm_factory = None
_llm._llm_factory_params = set()
_llm._prompt_compiler = None
_llm._prompt_compiler_params = set()
_llm._global_renderer = None
_merge_fn_registry.clear()
_type_registry.clear()
```

This works but is tightly coupled to internal implementation. If a new registry is added and not included here, tests will leak state across runs with no obvious failure message. Consider a `reset_all()` function in the library itself that the fixture calls, keeping the cleanup logic co-located with the registries.

---

### 7. [LOW] Duplicate section headers in test files

**File**: `tests/test_node_decorator.py`

Several section comment blocks are duplicated verbatim:

- Lines 549-563 and 556-563: `@node decorator: Oracle ensemble kwargs` header appears twice
- Lines 666-683 and 676-683: `@node(mode='raw')` header appears twice
- Lines 809-822 and 817-822: `@node interrupt_when` header appears twice
- Lines 976-995 and 987-995: `@node scalar parameters` header appears twice

This is a cosmetic issue but suggests copy-paste assembly that may have introduced subtle duplication in test logic too.

---

### 8. [LOW] `test_fakes.py` tests the test infrastructure

**File**: `tests/test_fakes.py`

This file tests `StructuredFakeWithRaw` — a test helper. While testing test infrastructure is fine for complex helpers, 7 tests on a fake LLM is disproportionate. The `test_call_structured_extracts_usage` and `test_invoke_structured_returns_correct_model_when_fake_with_raw` tests cross the line into testing the real `_call_structured` and `invoke_structured` code paths through the fake, which is valuable, but they should live in a test file for `_llm.py`, not in `test_fakes.py`.

---

### 9. [HIGH] Forgiving scripted functions mask wiring bugs

**File**: `tests/test_obligation_r1r2.py` and `tests/test_spec_loader.py`

Several scripted test functions use forgiving input handling:

```python
def refine_fn(input_data, config):
    if isinstance(input_data, dict):
        ctx = input_data.get("context")
        draft = input_data.get("my_refiner")
    else:
        ctx = None
        draft = input_data
```

This `isinstance(input_data, dict)` + `.get()` pattern means the test will silently pass even if the framework delivers the wrong type. The test at `test_spec_loader.py:662` (`test_loop_first_iteration_receives_upstream_not_none`) does this explicitly:

```python
d = input_data if isinstance(input_data, Draft) else Draft(content="", score=0.0)
```

The fallback `Draft(content="", score=0.0)` means if wiring is broken, the test still runs (just with default values). This is acknowledged in some test docstrings ("Honest: no forgiving fallbacks") but not enforced consistently. The obligation tests in `test_obligation_r1r2.py` are particularly good about strict assertions (`assert isinstance(ctx, Alpha)`), which is the right pattern.

**Recommendation**: Replace all forgiving input handlers with strict `assert isinstance(...)` checks. The test should fail loudly when wiring is wrong.

---

### 10. [MEDIUM] Check fixture suite has 0 known_gaps entries

**File**: `tests/check_fixtures/known_gaps/` (empty)

The known_gaps directory is empty. Per CLAUDE.md, "known_gaps IS the backlog for validation improvements." Either all validation gaps have been fixed (unlikely given the `pass`-body tests in test_coverage_gaps.py), or gaps aren't being filed here as they should be. The empty directory undermines the fixture suite's value as a living backlog.

---

## What Works Well

1. **Behavioral integration tests dominate**. The majority of tests (test_pipeline_modes, test_spec_loader, test_loop, test_composition, test_modifiers) compile real Constructs and run real graphs. They test the framework end-to-end, not just individual functions.

2. **Fake infrastructure is well-designed**. `fakes.py` provides three fake LLM patterns (StructuredFake, ReActFake, TextFake) that map cleanly to the three LLM interaction modes. `GuardFake` and `StubbornFake` test specific safety behaviors. All are documented with usage examples.

3. **The check_fixtures pattern is excellent**. Rustc-style should_fail/should_pass fixtures (50+ fixtures) give high confidence in the validator. Each fixture is minimal (~15 lines), self-contained, and auto-discovered. The `# CHECK_ERROR:` regex convention is clean.

4. **Error message quality is tested**. Many tests verify not just that errors are raised, but that the error message contains the right node names, type names, and source locations. This is rare and valuable.

5. **Three-surface parity is practiced**. Tests like `test_programmatic_each_with_dict_inputs_passes_when_fan_out_key` (neograph-ts7 regression) explicitly test the programmatic API alongside the `@node` decorator path.

6. **TDD evidence is visible**. Test docstrings reference specific bugs (neograph-ts7, neograph-6jd, neograph-cfrd) and the `test_spec_loader.py` header explicitly states "Written BEFORE the loader exists (TDD red)."

7. **Security tests in conditions**. `test_conditions.py` includes injection rejection tests (import injection, eval injection, semicolon injection, dunder field access) — good defense-in-depth.

---

## Summary of Actionable Items

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | HIGH | test_obligation_r1r2.py:417 | Empty test class with `pass` — delete or implement |
| 2 | HIGH | test_coverage_gaps.py (multiple) | Weak assertions (`is not None`) on coverage-gap tests |
| 3 | MEDIUM | test_cli.py:275-373 | Manual monkeypatching instead of `monkeypatch.setattr()` |
| 4 | MEDIUM | test_cli.py:236-273 | Test reimplements code logic instead of calling it |
| 5 | MEDIUM | multiple | Loose string containment assertions on error messages |
| 6 | MEDIUM | conftest.py | Registry cleanup tightly coupled to internals |
| 7 | LOW | test_node_decorator.py | Duplicate section headers (copy-paste artifact) |
| 8 | LOW | test_fakes.py | Tests for test infrastructure in wrong file |
| 9 | HIGH | multiple | Forgiving input handlers mask wiring bugs in test functions |
| 10 | MEDIUM | check_fixtures/known_gaps/ | Empty known_gaps directory undermines fixture-as-backlog |
