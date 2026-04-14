# Testing Quality Review

**Scope**: test_spec_loader.py, test_spec_schema.py, test_spec_types.py, test_conditions.py, test_inline_prompts.py, test_loop.py
**Date**: 2026-04-07

## Test Suite Shape

| Category | Count | Notes |
|----------|-------|-------|
| Unit tests (conditions, types, inline prompts) | 67 | Pure function tests |
| Integration tests (spec_loader, loop) | 26 | compile + run E2E |
| Schema validation tests | 10 | JSON Schema contract |
| Assertion-free tests | 1 | test_loop.py:410 |
| Forgiving-fallback tests | 5 | test_spec_loader.py |
| Closure-echo tests | 2 | test_spec_loader.py |

## Findings

### TQ-01: mn_revise ignores input_data -- closure echo chamber
- **Severity**: Critical
- **Anti-pattern**: Mock Echo / Closure Echo
- **File**: `tests/test_spec_loader.py:423-425`
- **Test**: `TestLoadSpecMultiNodeConstruct::test_multi_node_construct_loops_as_unit`
- **Description**: The `mn_revise` function completely ignores its `input_data` parameter. It computes score and iteration from `review_count[0]`, a closure variable shared with `mn_review`. The test asserts `result["refine"][-1].score >= 0.8`, but this score is `0.3 * review_count[0]` from the closure, not from anything received via `input_data`. If the sub-construct wiring were completely broken and `mn_revise` never received the review output, the test would still pass identically.
- **Evidence**: Replace `mn_revise` body with `return Draft(content="revised", score=0.3 * review_count[0], iteration=review_count[0])` and delete the comment about reading from state -- the test passes. The function already IS that exact code. It does not read `input_data` at all.
- **Recommended fix**: `mn_revise` should read `input_data["review"]` and derive its output from the review's fields. Assert that `final.content` contains the review's feedback string. The existing `TestLoadSpecWiringHonesty::test_multi_node_construct_revise_actually_reads_review` at line 553 does exactly this -- it is the correct version of this test. The test at line 407 provides false confidence.

### TQ-02: mn_review ignores input_data -- closure echo chamber
- **Severity**: High
- **Anti-pattern**: Mock Echo / Closure Echo
- **File**: `tests/test_spec_loader.py:418-421`
- **Test**: `TestLoadSpecMultiNodeConstruct::test_multi_node_construct_loops_as_unit`
- **Description**: The `mn_review` function ignores `input_data` entirely. It returns `ReviewResult(score=0.3 * review_count[0], ...)` from a closure counter. The function does not read the draft it receives. If review received `None` as input, the test would still pass because the score is entirely counter-driven.
- **Evidence**: The function signature takes `input_data` but never references it. The score sequence 0.3, 0.6, 0.9 is generated entirely by the counter, not by any data flowing from the pipeline.
- **Recommended fix**: `mn_review` should read the draft from `input_data` and incorporate something from it (e.g., check `input_data.content` and include it in feedback). Assert that the review's feedback contains content from the actual draft.

### TQ-03: ForwardConstruct loop test has assertion-free-equivalent assertion
- **Severity**: Critical
- **Anti-pattern**: Assertion-Free / Trivially-True Assertion
- **File**: `tests/test_loop.py:410`
- **Test**: `TestForwardConstructLoop::test_for_loop_compiles_to_cycle`
- **Description**: The only assertion is `assert result["draft"] is not None or result.get("revise") is not None`. This is a disjunction where either branch being truthy makes it pass. Since `result["draft"]` is populated by `fc_draft` (which always returns a Draft), the left branch is always True. The assertion literally cannot fail as long as the `draft` node ran at all. It tells us nothing about whether the loop cycled, whether `review` ran, or whether `revise` ran. Additionally, `fc_revise` ignores its input (returns hardcoded values from a lambda), so even the pipeline execution is untested.
- **Evidence**: Remove the `for` loop from `forward()`, replace with `return self.draft(topic)`. The assertion still passes because `result["draft"]` is not None.
- **Recommended fix**: Assert `_review_count[0] >= 2` (proving the loop ran multiple times). Assert that the revise node actually produced output visible in the result. Assert the final output's score or content reflects the iterative refinement.

### TQ-04: Forgiving fallback in test_loop_preserves_all_iterations_in_append_list
- **Severity**: Medium
- **Anti-pattern**: Forgiving Fallback
- **File**: `tests/test_spec_loader.py:362`
- **Test**: `TestLoadSpecLoopHistory::test_loop_preserves_all_iterations_in_append_list`
- **Description**: The `refine_fn` has `d = input_data if isinstance(input_data, Draft) else Draft(content="", score=0.0)`. The fallback creates a Draft with `score=0.0`, which is the same score the seed produces. On the first iteration, if wiring is broken and `input_data` is `None`, the fallback fires and the score accumulation starts from 0.0 anyway, producing the exact same progression (0.25, 0.50, 0.75, 1.0). The test cannot distinguish between "received seed's output correctly" and "fell back to default because wiring was broken."
- **Evidence**: The loop's score accumulation works identically starting from the fallback `Draft(score=0.0)` as from the seed's output `Draft(score=0.0)`. The test would only catch a wiring failure if the seed produced a non-zero score that the fallback doesn't match.
- **Recommended fix**: Either (a) have the seed produce `Draft(score=0.1, ...)` and assert the first element's score is 0.35 (0.1 + 0.25), proving data flowed from seed; or (b) assert on a distinctive field like `content` that proves the first iteration received the seed's specific output.

### TQ-05: Forgiving fallback in test_construct_with_loop_compiles_and_runs
- **Severity**: Medium
- **Anti-pattern**: Forgiving Fallback
- **File**: `tests/test_spec_loader.py:227`
- **Test**: `TestLoadSpecWithConstruct::test_construct_with_loop_compiles_and_runs`
- **Description**: `improve_fn` has `d = input_data if isinstance(input_data, Draft) else Draft(content="x", score=0.0)`. If the sub-construct's input wiring delivers a dict instead of a Draft (which is the case when the sub-construct has dict-form inputs), the fallback fires silently. The test asserts `call_count[0] >= 3` and `result["refine"][-1].score >= 0.8`, both of which pass because the counter-driven score accumulation works from `score=0.0` regardless of the fallback path taken.
- **Evidence**: The `elif` branch creates `Draft(score=0.0)` which is the same starting score as the actual upstream. The accumulation `d.score + 0.3` yields the same sequence from either path.
- **Recommended fix**: Same as TQ-04 -- make the seed produce a distinctive score and assert the first iteration's accumulation reflects it.

### TQ-06: Forgiving fallback in test_self_loop_runs_until_condition_met
- **Severity**: Medium
- **Anti-pattern**: Forgiving Fallback
- **File**: `tests/test_spec_loader.py:180`
- **Test**: `TestLoadSpecWithLoop::test_self_loop_runs_until_condition_met`
- **Description**: `refine_fn` has `d = input_data if isinstance(input_data, Draft) else next(iter(input_data.values()))`. The `else` branch calls `next(iter(input_data.values()))` which silently unwraps a dict to its first value. This masks the actual shape of `input_data` -- whether it's a Draft directly or wrapped in a dict. If the framework changed how it delivers input to Loop nodes (dict vs raw), this test would silently accept either form.
- **Evidence**: The function accepts both `Draft` and `dict[str, Draft]` and produces the same result from both. This is a weaker finding than the others because the fallback is smarter (it extracts the right value from a dict), but it still means the test can't catch a regression in input delivery shape.
- **Recommended fix**: Capture `input_data` in a list and assert its exact type in the test assertions, so a shape change is caught explicitly.

### TQ-07: Forgiving fallback in test_loop_first_iteration_receives_upstream_not_none
- **Severity**: Low
- **Anti-pattern**: Forgiving Fallback (partially mitigated)
- **File**: `tests/test_spec_loader.py:633`
- **Test**: `TestLoadSpecWiringHonesty::test_loop_first_iteration_receives_upstream_not_none`
- **Description**: `loop_node` has `d = input_data if isinstance(input_data, Draft) else Draft(content="", score=0.0)`. However, this test IS in the WiringHonesty class and captures `first_input[0] = input_data` on the first call, then asserts `fi.content == "seed-output"` at line 661. This means the test DOES catch first-iteration wiring failures via the capture mechanism. The forgiving fallback in the function body is only used for subsequent iterations (where the Loop modifier feeds back the own output). This is mostly fine -- the fallback only affects iterations 2+, and the explicit assertion on `first_input` catches the critical path.
- **Evidence**: The assertion at line 661 is strong: `assert fi.content == "seed-output"`. The fallback pattern in the function is not ideal but does not undermine the test's primary claim.
- **Recommended fix**: Minor cleanup -- remove the `else` fallback on iteration 2+ and instead assert `isinstance(input_data, Draft)` to catch unexpected input shapes on all iterations.

### TQ-08: test_variable_substitution test does not test variable substitution
- **Severity**: High
- **Anti-pattern**: Testing the Wrong Thing
- **File**: `tests/test_spec_loader.py:467-508`
- **Test**: `TestLoadSpecVariableSubstitution::test_inline_prompt_with_variable_substitution_renders`
- **Description**: Despite the class name and test name mentioning "variable substitution", this test does NOT test `${node.field}` substitution at all. The spec defines two scripted nodes (not think-mode LLM nodes), so no inline prompt is ever compiled. The test's actual assertion (line 503-508) verifies that the second scripted node receives the first node's output as `input_data` -- which is just basic wiring. The `captured_input` mechanism proves wiring works but has nothing to do with `${variable}` substitution. The test name creates false confidence that prompt variable substitution is covered via this E2E path.
- **Evidence**: Neither node has a `prompt` field. `capture_fn` is a scripted function that just captures `input_data`. No `_compile_prompt` or `_substitute_vars` is invoked during this test. The inline prompt feature is tested in `test_inline_prompts.py` at the unit level, but this E2E "integration" test of it is a misnomer.
- **Recommended fix**: Either (a) rename to something like `test_second_node_receives_first_nodes_output` (but that duplicates `TestLoadSpecWiringHonesty`), or (b) rewrite to actually test inline prompt substitution by using a think-mode node with `prompt: "Score is ${seed.score}"` and a fake LLM that captures the prompt messages.

### TQ-09: Missing error path -- invalid loop `on_exhaust` value in spec
- **Severity**: Low
- **Anti-pattern**: Happy Path Only
- **File**: `tests/test_spec_loader.py`
- **Test**: (missing)
- **Description**: `TestLoadSpecErrors` tests unknown type, unknown node ref, and unknown construct ref. But it does not test what happens when a loop spec has an invalid `on_exhaust` value (e.g., `"crash"` instead of `"error"` or `"last"`), or when `max_iterations` is 0 or negative, or when `when` is an invalid expression.
- **Recommended fix**: Add tests for malformed loop specs: invalid `on_exhaust`, `max_iterations < 1`, unparseable `when` expression.

### TQ-10: Missing test -- condition evaluator edge case with negative numbers
- **Severity**: Low
- **Anti-pattern**: Happy Path Only
- **File**: `tests/test_conditions.py`
- **Test**: (missing)
- **Description**: `TestNumericComparisons` covers positive floats and integers but never tests negative number literals (e.g., `score > -0.5`). Looking at the regex `_EXPR_RE`, the literal group is `(?P<literal>.+)` and `_parse_literal` calls `float(raw)` which accepts negative numbers. However, the sign `-` could in theory conflict with a minus in the field name or operator. The code likely works but there is zero test coverage for this edge case.
- **Recommended fix**: Add `test_less_than_negative_literal` with expression `"value < -0.5"`.

### TQ-11: Missing test -- condition evaluator with integer field against float literal
- **Severity**: Low
- **Anti-pattern**: Happy Path Only
- **File**: `tests/test_conditions.py`
- **Test**: (missing)
- **Description**: The tests never exercise cross-type comparison (e.g., an integer field compared against a float literal like `count > 2.5`). This exercises the Python comparison semantics through the `operator.gt` dispatch.
- **Recommended fix**: Add one test for int-field vs float-literal comparison.

### TQ-12: Missing test -- spec_types `load_project_types` with no `required` key
- **Severity**: Low
- **Anti-pattern**: Happy Path Only
- **File**: `tests/test_spec_types.py`
- **Test**: (missing)
- **Description**: `test_generates_model_with_primitive_fields` always includes `"required"` in the config. `test_optional_fields_default_to_none` omits `priority` from required. But no test exercises a type definition that has NO `required` key at all (which makes all fields optional). The production code handles this via `required_fields = set(type_def.get("required", []))`, but it is untested.
- **Recommended fix**: Add a test with a type definition that has `properties` but no `required` key at all, and verify all fields default to None.


## Positive Examples

Tests that exemplify good testing practices in this codebase:

- `tests/test_spec_loader.py::TestLoadSpecWiringHonesty::test_second_node_receives_first_nodes_exact_output` (line 513) -- Captures `input_data` directly, then asserts exact field values including a unique marker string `"unique-marker-xyz"`. This is how all wiring tests should work: unique markers prove data flowed from A to B.

- `tests/test_spec_loader.py::TestLoadSpecWiringHonesty::test_multi_node_construct_revise_actually_reads_review` (line 553) -- The revise function reads `input_data["review"]` and derives output from it. The assertion checks that `final.content == "unique-feedback-marker"`, proving the review's output actually arrived at the revise node. This is the correct version of TQ-01's broken test.

- `tests/test_loop.py::TestSelfLoop::test_self_loop_exits_when_condition_met` (line 76) -- Uses `@node` decorator with typed parameters (`seed: Draft`). The function body `seed.score + 0.3` MUST receive a valid Draft -- if wiring is broken, `.score` raises AttributeError. Assertions check call count, list structure, final score, iteration count, and list length. Comprehensive.

- `tests/test_conditions.py` as a whole -- Pure function tests with both True and False cases for every operator. Both error paths and injection attacks are covered. Clear Given/When/Then structure. No mocks needed because the SUT is a pure function.

- `tests/test_inline_prompts.py::TestCompilePromptInline::test_file_ref_delegates_to_prompt_compiler` (line 148) -- Uses a spy (captured list) to verify the prompt compiler was called with the correct template. Combined with `test_inline_prompt_skips_prompt_compiler` (line 164), these two tests prove the routing decision is correct in both directions.


## Summary

- Critical: 2 (TQ-01 closure echo in multi-node construct revise, TQ-03 assertion-free ForwardConstruct loop test)
- High: 2 (TQ-02 closure echo in multi-node construct review, TQ-08 test name claims variable substitution but tests wiring)
- Medium: 3 (TQ-04/TQ-05/TQ-06 forgiving fallbacks that mask first-iteration wiring failures)
- Low: 4 (TQ-07 minor fallback, TQ-09/TQ-10/TQ-11/TQ-12 missing edge case coverage)
- Overall test quality assessment: The codebase has a split personality. The `test_conditions.py`, `test_inline_prompts.py`, `test_spec_types.py`, and `test_spec_schema.py` files are solid -- pure function tests with strong assertions. The `test_loop.py` `@node` tests are also good because typed parameters enforce wiring. The problems concentrate in `test_spec_loader.py` (scripted functions that use closures instead of reading `input_data`) and the single ForwardConstruct test in `test_loop.py`. The `TestLoadSpecWiringHonesty` class in `test_spec_loader.py` shows the author recognized the closure-echo problem and wrote correct tests -- but the earlier tests in the same file still have the anti-pattern. The ForwardConstruct loop test is the weakest test in the entire set reviewed, providing effectively zero confidence about loop compilation.
