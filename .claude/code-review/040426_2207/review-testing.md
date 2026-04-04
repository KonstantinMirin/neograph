# Testing Quality Review

**Scope**: New DX feature tests in `tests/test_e2e_piarch_ready.py` —
`TestNodeMap` (9 tests, `.map()` sugar over `Each`) and
`TestConstructValidation` (14 tests, `ConstructError` assembly-time checks).
Supporting code under review: `src/neograph/modifiers.py` (`_PathRecorder`,
`Modifiable.map`) and `src/neograph/construct.py` (`ConstructError`,
`_validate_node_chain` and helpers).

**Date**: 2026-04-04

## Test Suite Shape

| Category | Count | Notes |
|----------|-------|-------|
| New unit tests — `.map()` introspection / shape | 7 | lambda path → Each, string path, error branches |
| New integration test — `.map()` end-to-end | 1 | `test_map_end_to_end_fanout` (compile + run) |
| New unit test — `.map()` on `Construct` | 1 | `test_map_on_construct` |
| New assembly-validation tests | 14 | all exercise real `Construct(...)` call, no mocks |
| Mocked / `patch()`-heavy tests added | 0 | Both new classes are mock-free |
| Assertion-free new tests | 0 | Every test has at least one meaningful assertion |
| Full suite count after changes | 99 passing | confirmed 9 + 14 = 23 new, 76 pre-existing |

Coverage of the new code (ran with `--cov=neograph.construct
--cov=neograph.modifiers` against the full file):

- `src/neograph/modifiers.py`: **100%** (all new `_PathRecorder` / `.map()`
  branches reached; the 6 lines reported uncovered when running only the two
  new classes are Oracle's `model_post_init`, out of scope).
- `src/neograph/construct.py`: **85%** (23 / 155 lines uncovered). Of the
  uncovered lines, most are defensive dead code (see TQ-06), with one
  meaningful gap: line 189, the Each-root-not-in-producers defer branch, is
  never exercised via `Construct(...)` — see TQ-04.

Overall the new tests are solidly mock-free and exercise the real code path.
Issues below are confined to a few weak assertions and one coverage gap in
`construct.py`.

## Findings

### TQ-01: `test_map_end_to_end_fanout` assertion passes for a degenerate single-item result
- **Severity**: High
- **Anti-pattern**: Happy Path Only / Assertion Hole
- **File**: `tests/test_e2e_piarch_ready.py:386`
- **Test**: `TestNodeMap::test_map_end_to_end_fanout`
- **Description**: This is the test the review brief specifically calls out.
  `make_clusters` produces two groups (`alpha`, `beta`); the expected fan-out
  result is a dict with two keys. The assertion is:
  ```python
  assert "alpha" in verify_results or len(verify_results) == 2
  ```
  Because of the `or`, a result of `{"alpha": ...}` (fan-out fired for only
  the first group and the second silently lost) passes. Even worse, if
  `.map()` degenerated to a no-op that left `verify_results` as a single
  scalar dict keyed by `"alpha"` for any reason, the test is green. The
  test's docstring says "drives the same fan-out/collect behavior" but the
  assertion does not verify fan-out cardinality.
- **Evidence**: Running the real pipeline returns
  `{"alpha": MatchResult(...), "beta": MatchResult(...)}`. Mutating the
  expected map to drop one element (`{"alpha": ...}`) still satisfies the
  predicate: `"alpha" in d` is True, so the `or` short-circuits. Test would
  not catch a regression that broke the beta dispatch.
- **Recommended fix**: Replace with a hard equality on the set of produced
  keys and on the inner payload:
  ```python
  assert set(verify_results.keys()) == {"alpha", "beta"}
  assert verify_results["alpha"].cluster_label == "alpha"
  assert verify_results["beta"].cluster_label == "beta"
  ```
  This is the only integration test for `.map()` — the redundant-or pattern
  was copied from the older `TestEach::test_fanout_over_collection:319` and
  should be tightened there too, but fixing at least the new test is the
  priority.

### TQ-02: `test_mismatch_hint_suggests_map` does not verify hint content, only that the phrase appears
- **Severity**: Medium
- **Anti-pattern**: Shallow String Match
- **File**: `tests/test_e2e_piarch_ready.py:1762`
- **Test**: `TestConstructValidation::test_mismatch_hint_suggests_map`
- **Description**: The test asserts only `match="did you forget to fan out"`.
  `_suggest_hint` (`construct.py:317-334`) scans every field of every producer
  and emits the first `list[T]` that is assignable to the needed input,
  rendering `try .map(lambda s: s.{field_name}.{fname}, key='...')`. The test
  never pins which field the hint points at, so a bug that suggested the
  wrong field (e.g. iterating `producer_type` fields in reverse order, or
  picking a compatible field on a different producer) would pass. For the
  specific fixture used (`a: output=Clusters`, `b: input=ClusterGroup`), the
  correct suggestion is `s.a.groups` — this is the actionable part of the
  error message and the whole selling point of the feature.
- **Evidence**: Replace line 332 `f"try .map(lambda s: s.{field_name}.{fname}, key='...')"` with a hardcoded
  `"try .map(lambda s: s.bogus.field, key='...')"` and the test still passes.
- **Recommended fix**:
  ```python
  with pytest.raises(ConstructError) as exc_info:
      Construct("bad-fanout", nodes=[a, b])
  msg = str(exc_info.value)
  assert "did you forget to fan out" in msg
  assert "s.a.groups" in msg  # pin the concrete suggestion
  ```

### TQ-03: `test_plain_input_mismatch_raises` only pins the error header, not the producer listing
- **Severity**: Medium
- **Anti-pattern**: Shallow String Match
- **File**: `tests/test_e2e_piarch_ready.py:1753`
- **Test**: `TestConstructValidation::test_plain_input_mismatch_raises`
- **Description**: Matches on `"declares input=Claims"`, which is in the
  first line of `_format_no_producer_error`. The function also renders a
  `upstream producers:` block (lines 296-302) listing every prior producer
  with its type — this is user-facing debugging copy and it is not tested.
  `_format_no_producer_error` has a dead-else branch at line 302 for the
  empty-producers case that `_check_item_input` already guards against
  (returns at line 142 if there are no producers), so the current test only
  touches the populated branch without asserting the output.
- **Recommended fix**: Assert the producer list is rendered:
  ```python
  msg = str(exc_info.value)
  assert "declares input=Claims" in msg
  assert "node 'a': RawText" in msg  # the one producer we expect listed
  ```

### TQ-04: No test exercises `_check_each_path` root-not-in-producers deferral via `Construct(...)`
- **Severity**: Medium
- **Anti-pattern**: Happy Path Only (boundary case missed)
- **File**: `tests/test_e2e_piarch_ready.py` (TestConstructValidation) / `src/neograph/construct.py:185-189`
- **Description**: When an `Each` is used as the first node in a construct
  (the user plans to seed the collection via `run(input=...)`), the validator
  must defer — `_check_each_path` returns early at line 189. `TestConstructValidation`
  has `test_first_item_with_input_deferred_to_runtime` for the *plain* input
  case but no equivalent for the *Each* case. `TestModifierAsFirstNode::test_each_at_start`
  constructs the node by hand via `_add_node_to_graph` and explicitly bypasses
  `_validate_node_chain`, so it does not cover this path either. Full-suite
  coverage run confirms line 189 is never hit.
- **Evidence**: `pytest tests/test_e2e_piarch_ready.py
  --cov=neograph.construct --cov-report=term-missing` reports `Missing: 189`
  with all 99 tests passing. An implementation bug that raised
  `ConstructError` for a legitimate top-level `Each` (a plausible regression
  when refactoring the producer-lookup loop) would slip through CI.
- **Recommended fix**: Add one positive test:
  ```python
  def test_top_level_each_deferred_to_runtime(self):
      """Each on the first node (root not in producers) validates OK."""
      process = Node.scripted(
          "process", fn="f", input=ClusterGroup, output=MatchResult,
      ) | Each(over="seeded_from_runtime.groups", key="label")
      Construct("top-each", nodes=[process])  # no error
  ```

### TQ-05: `test_sub_construct_input_mismatch_in_parent` regex is loose
- **Severity**: Low
- **Anti-pattern**: Brittle/Underspecified Mock Assertion (regex version)
- **File**: `tests/test_e2e_piarch_ready.py:1837`
- **Test**: `TestConstructValidation::test_sub_construct_input_mismatch_in_parent`
- **Description**: `match="sub.*declares input=Claims"` will also match e.g.
  `"Node 'subsystem' ... declares input=Claims"` or any error emitted from
  an inner construct whose own validation fails first. If a refactor made
  the nested construct raise *before* the parent's chain walk even reached
  `sub`, the test would still pass against the wrong error. This is the only
  test that exercises the sub-construct-as-consumer path and it should be
  tight.
- **Recommended fix**: Anchor on both the construct name and the item:
  ```python
  with pytest.raises(ConstructError) as exc_info:
      Construct("parent", nodes=[upstream, sub])
  msg = str(exc_info.value)
  assert "sub-construct 'sub'" in msg
  assert "declares input=Claims" in msg
  ```

### TQ-06: `test_each_correct_path_passes` is the only happy-path assertion for Each chains — risks silent regression
- **Severity**: Low
- **Anti-pattern**: Assertion-by-absence
- **File**: `tests/test_e2e_piarch_ready.py:1772`
- **Test**: `TestConstructValidation::test_each_correct_path_passes`
- **Description**: Pure "did not raise" — no assertion on construct state
  after assembly. If `_check_each_path` silently passed through without
  validating anything (e.g., a refactor made `_resolve_field_annotation`
  return `_MISSING` for valid fields, and the `element_type = None` branch
  was reordered), this test would still pass because the eventual return is
  silent. The test serves as a coverage marker, not a behavior check.
  "No exception" is the default state of any passing test.
- **Recommended fix**: Assert the resulting construct surface — node count,
  the Each modifier is attached with the expected `over`/`key` — so that a
  missed-assignment regression in the happy path is caught:
  ```python
  c = Construct("good-each", nodes=[a, b])
  assert len(c.nodes) == 2
  each = c.nodes[1].get_modifier(Each)
  assert each.over == "a.groups" and each.key == "label"
  ```
  The same observation applies to `test_valid_chain_passes:1747`,
  `test_first_item_with_input_deferred_to_runtime:1813`,
  `test_sub_construct_input_port_satisfies_inner_node:1819`,
  `test_sub_construct_chained_in_parent:1828`, and
  `test_dict_input_skipped:1857`. Treat this as one class-wide fix, not six.

## Positive Examples

Tests in the reviewed set that demonstrate good practice:

- `tests/test_e2e_piarch_ready.py::TestNodeMap::test_map_equivalent_to_pipe_each`
  (line 352) — compares `modifiers` lists via value equality on frozen
  Pydantic `Modifier` models. This is the right way to assert "sugar is
  equivalent to explicit form"; the assertion would fail if `.map()` added an
  extra modifier, used a different `over` encoding, or lost the `key`.
- `tests/test_e2e_piarch_ready.py::TestNodeMap::test_map_lambda_that_errors_raises_typeerror`
  (line 400) — exercises the real `except Exception` branch in `modifiers.py:96`
  by calling `[0]` on the recorder (verified: `_PathRecorder` is not
  subscriptable → raises `TypeError`, which the `.map()` handler wraps into a
  clean `TypeError` with the "pure attribute-access chain" message). The
  regex match on the wrapper message pins the specific error path.
- `tests/test_e2e_piarch_ready.py::TestConstructValidation::test_construct_error_is_valueerror`
  (line 1840) — a one-line test, but it pins the documented subclass
  relationship (`ConstructError(ValueError)`) that the docstring promises for
  backward compatibility. Cheap, specific, high-value.
- `tests/test_e2e_piarch_ready.py::TestConstructValidation::test_mismatch_error_includes_source_location`
  (line 1765) — correctly asserts the `file:line` contract via regex on the
  real basename. This is the kind of integration assertion that would catch
  a `_source_location` refactor that accidentally returned the full absolute
  path or `None`.
- `tests/test_e2e_piarch_ready.py::TestConstructValidation::test_each_missing_field_raises`,
  `test_each_terminal_not_list_raises`, `test_each_list_wrong_element_raises`
  (lines 1780-1807) — three distinct Each failure modes, each pinned to its
  specific error phrase (`"has no field 'nonexistent'"`, `"not a list"`,
  `"list[str]"`). If any pair of these branches were collapsed or reordered,
  at least one test would fail. Good branch discrimination.

## Summary

- Critical: 0
- High: 1 (TQ-01)
- Medium: 3 (TQ-02, TQ-03, TQ-04)
- Low: 2 (TQ-05, TQ-06)

The new test additions are mock-free, exercise the real production code, and
achieve 100% branch coverage of `modifiers.py` and 85% of `construct.py`
(remaining gaps are mostly defensive/dead-code paths plus one real gap
flagged in TQ-04). The suite is in better shape than typical AI-assisted test
additions — there are zero mock-echo or assertion-free cases. The issues
found are all about **tightening existing assertions**, not replacing tests
wholesale. TQ-01 is the only finding that represents genuine false
confidence: the single end-to-end fan-out assertion cannot distinguish
working fan-out from a degenerate single-item dispatch. Fix TQ-01 first;
TQ-02, TQ-03, and TQ-04 together would raise the validation suite from "good
enough" to "actually enforces its promises."

## Files reviewed

- `/Users/konst/projects/neograph/src/neograph/modifiers.py`
- `/Users/konst/projects/neograph/src/neograph/construct.py`
- `/Users/konst/projects/neograph/src/neograph/__init__.py`
- `/Users/konst/projects/neograph/tests/test_e2e_piarch_ready.py` (new
  sections only: lines 331-422 `TestNodeMap`; lines 1738-1857
  `TestConstructValidation`; line 14 import)
