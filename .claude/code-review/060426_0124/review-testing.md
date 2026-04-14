# Testing Quality Review

**Scope**: 28 new tests across 7 test classes added for the Node.inputs refactor epic (neograph-kqd), all in `tests/test_e2e_piarch_ready.py` (lines 4997-5684).
**Date**: 2026-04-06

## Test Suite Shape

| Category | Count | Notes |
|----------|-------|-------|
| Field/IR-level tests | 8 | TestNodeInputsFieldRename (5), TestNodeInputsEpicAcceptance (3 non-e2e) |
| Validation-level tests | 7 | TestFanInValidation (3), TestTypesCompatibleListOverDict (4) |
| Factory/runtime tests | 3 | TestExtractInputListUnwrap (3) |
| Decorator metadata tests | 4 | TestNodeDecoratorDictInputs (4 non-e2e) |
| End-to-end (compile+run) | 8 | TestListOverEachEndToEnd (3), TestNodeInputsEpicAcceptance (4), TestNodeDecoratorDictInputs (1) |
| **Total** | **30** | 2 tests more than the 28 stated in the brief (possibly miscounted) |
| Assertion-free tests | 0 | All tests have explicit assertions |
| Mock/patch usage | 0 | Zero mocks across all 7 classes |
| `return_value`/`side_effect` | 0 | No mock echo possible |

## Findings

### TQ-01: Trivial assertion on happy-path validation tests
- **Severity**: Low
- **Anti-pattern**: Near-assertion-free (implicit "no exception" testing)
- **File**: `tests/test_e2e_piarch_ready.py:5041`
- **Test**: `TestFanInValidation::test_fan_in_dict_matching_upstreams_passes`
- **Description**: The test's assertion is `assert len(pipeline.nodes) == 4` — which is trivially true (it's the count of nodes passed to the constructor). The actual behavior being tested is that `Construct()` does not raise `ConstructError`. The assertion doesn't verify any interesting property of the validation result.
- **Evidence**: Replace the four `_producer`/`Node.scripted` calls with any types and the assertion still passes as long as Construct doesn't raise. The `len` check adds no discrimination.
- **Recommended fix**: This is a stylistic concern, not a correctness gap. The companion error-path tests (`test_fan_in_dict_unknown_upstream_rejected`, `test_fan_in_dict_type_mismatch_rejected`) are the real value. If you want to strengthen the happy path, assert on the consumer's resolved inputs dict or on a property of the validated construct, not just node count. Alternatively, accept that "no exception" is the intended assertion and remove the trivial `len` check to avoid implying more is verified than actually is.
- **Same pattern in**: `TestNodeInputsEpicAcceptance::test_zero_upstream_node_inputs_none` (line 5582) — `assert len(pipeline.nodes) == 1` is trivially true; `assert pipeline.nodes[0].inputs is None` restates the constructor argument.

### TQ-02: Unused `caplog` fixture parameter
- **Severity**: Low
- **Anti-pattern**: Testing the framework (fixture requested but not used)
- **File**: `tests/test_e2e_piarch_ready.py:5278`
- **Test**: `TestNodeDecoratorDictInputs::test_scripted_fan_in_log_mode_is_scripted`
- **Description**: The test signature requests `caplog` (pytest's log capture fixture), but the test body uses a custom structlog processor to capture events instead. `caplog` is never referenced in the body.
- **Evidence**: Removing `caplog` from the signature has no effect on the test.
- **Recommended fix**: Remove the unused `caplog` parameter from the test signature.

### TQ-03: Global registry pollution without cleanup
- **Severity**: Low
- **Anti-pattern**: Test isolation concern
- **File**: `tests/test_e2e_piarch_ready.py` (lines 5338, 5432, 5483, 5515, 5658)
- **Tests**: All tests that call `register_scripted()` in `TestListOverEachEndToEnd` and `TestNodeInputsEpicAcceptance`
- **Description**: `register_scripted()` writes to a global `_scripted_registry` dict in `factory.py`. The tests register functions with unique names (`l5_seed_claims`, `l7_seed_claims`, `make_clusters_l5`, etc.) but never clean up. The registry grows monotonically across the test suite.
- **Evidence**: Not a correctness issue today (names don't collide). But if a future test reuses a name like `l7_a`, it would silently get the wrong function. The test execution order matters.
- **Recommended fix**: Either add a pytest fixture that snapshots and restores `_scripted_registry` after each test, or note this as an accepted convention. The unique naming scheme is a sufficient guard for now but doesn't scale.

### TQ-04: Missing edge case — empty dict inputs
- **Severity**: Medium
- **Anti-pattern**: Happy path only
- **File**: Not present in any test
- **Test**: (missing)
- **Description**: No test exercises `inputs={}` (empty dict). The production code in `_check_fan_in_inputs` iterates over the dict items, so an empty dict would silently pass validation with zero checks. Whether this is valid or should be rejected is a design decision, but it's untested.
- **Evidence**: `grep -rn 'inputs={}' tests/` returns no results. The `_check_fan_in_inputs` function at `_construct_validation.py:163` would iterate zero times and return without error.
- **Recommended fix**: Add a test that documents the expected behavior for `inputs={}`. Either it should be treated as `inputs=None` (no upstream needed), or it should raise a validation error ("empty inputs dict is meaningless").

### TQ-05: Missing edge case — duplicate upstream names in dict-form inputs
- **Severity**: Low
- **Anti-pattern**: Happy path only
- **File**: Not present in any test
- **Test**: (missing)
- **Description**: Python dict literals can't have duplicate keys, but a programmatic builder (like the LLM-driven spec) could theoretically construct a dict with overwritten keys. This is a Python language guarantee (last key wins), not a neograph concern, so this is informational.
- **Evidence**: N/A — Python dicts prevent this by construction.
- **Recommended fix**: No action needed. Documenting for completeness.

### TQ-06: Error message assertions check substrings, not structure
- **Severity**: Low
- **Anti-pattern**: Brittle assertions (but in the right direction)
- **File**: `tests/test_e2e_piarch_ready.py:5064`, `5076`, `5258`, `5546`
- **Tests**: `test_fan_in_dict_unknown_upstream_rejected`, `test_fan_in_dict_type_mismatch_rejected`, `test_node_decorator_fan_in_type_mismatch_caught_by_validator`, `test_llm_driven_spec_type_mismatch_rejected`
- **Description**: Error-path tests assert substrings in exception messages (e.g., `assert "'nonexistent'" in msg`, `assert "no upstream node" in msg`). This is the right approach for verifying error quality, but it means tests could pass with a completely different error that happens to contain the same words.
- **Evidence**: If `Construct.__init__` raised a different ConstructError (e.g., from a different validation path) that happened to mention "nonexistent", the test would still pass.
- **Recommended fix**: This is acceptable and idiomatic for Python. The alternative (matching the full message) would be over-specified. No change needed.

## Positive Examples

Tests that exemplify good testing practices in this codebase:

- **`TestListOverEachEndToEnd::test_declarative_each_to_list_consumer`** (line 5332) — Full compile+run e2e with real data flowing through three nodes, runtime assertion inside the scripted function (`assert isinstance(verify_list, list)`), and a final output value check. Would catch breakage in any layer: validator, state builder, factory unwrap, or runtime wiring. This is the gold standard for this test suite.

- **`TestListOverEachEndToEnd::test_decorator_each_to_list_consumer`** (line 5378) — Same pattern through the `@node` decorator surface. Proves the decorator produces the same runtime behavior as the declarative API.

- **`TestNodeInputsEpicAcceptance::test_node_decorator_mixed_upstream_and_fanout_e2e`** (line 5589) — Tests the critical path where a single node has BOTH upstream params AND a fan-out param (Each). Verifies dict keying, value correctness for each key, and cross-node data flow. This is the highest-risk code path and it has the strongest test.

- **`TestNodeInputsEpicAcceptance::test_llm_driven_spec_fan_in_roundtrip`** (line 5479) — Simulates the LLM-driven pipeline construction use case end-to-end: JSON spec, type registry resolution, Node construction, Construct assembly, compile, run, output verification. Tests the primary downstream use case (piarch) directly.

- **`TestFanInValidation::test_fan_in_dict_unknown_upstream_rejected`** (line 5055) and **`test_fan_in_dict_type_mismatch_rejected`** (line 5070) — Clean error-path tests that verify both the exception type and message quality. Together with the happy-path test, they form a complete trio for dict-form fan-in validation.

- **`TestExtractInputListUnwrap`** (line 5125) — Tests the factory runtime directly with real Pydantic model instances and verifies the exact unwrap behavior (dict-to-list conversion, dict passthrough, None handling). Three focused tests covering the three code paths in `_extract_input`'s dict-form handler.

- **`TestTypesCompatibleListOverDict`** (line 5088) — Tests the internal `_types_compatible` function directly with four cases: matching element, wrong element, subclass, and non-dict producer. Clean, focused unit tests on a pure function.

## Summary

- Critical: 0
- High: 0
- Medium: 1 (missing empty-dict-inputs edge case)
- Low: 5 (trivial assertions, unused fixture, registry pollution, missing edge case, substring assertions)
- Overall test quality assessment: **Strong.** These 28 tests are well above the quality bar. Zero mocks, zero patches, 8 genuine end-to-end tests that compile and run real pipelines with real data. The test structure follows a deliberate layering: field-level IR checks, validator unit tests, factory runtime tests, decorator metadata tests, and full e2e integration. Error paths are tested with message quality assertions. The only gap is the missing empty-dict-inputs edge case, which is a design question more than a bug. The e2e tests for the mixed upstream+fanout path and the LLM-driven spec roundtrip are particularly strong — they test the exact use cases that motivated the refactor.
