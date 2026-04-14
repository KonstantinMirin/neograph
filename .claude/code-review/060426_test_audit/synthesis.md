# Test Audit Synthesis — 2026-04-06

**Scope**: Full test suite audit — inventory, coverage gaps, architecture issues, migration strategy
**Agents**: 4 ran (inventory, gaps, arch-issues, strategy)
**Current state**: 336 tests in 86 classes across 2 files (one monolith at 6800 lines)

---

## The Situation

neograph has 336 tests. They pass. The framework just passed production tests. But the test suite has structural problems that will compound with every feature:

1. **One monolith, no signal**: 86 test classes in one file. When a test fails, you grep to find it. When a feature ships, you can't tell if it's covered.
2. **Three-surface parity is a myth**: Only 2 of 86 classes test all three API surfaces. The ts7 bug (programmatic Each + dict inputs) was exactly this gap — @node worked, programmatic didn't. We added the parity rule to AGENTS.md but don't enforce it.
3. **Layer coverage is lopsided**: factory-dispatch and decorator tests are rich (~126 tests). State compilation has 6. LLM integration has 4. Runner has 7.
4. **Modifier combinations are sparse**: Each+Operator, Oracle+Operator, skip_when+any_modifier, dict-outputs+Each/Oracle — all untested.

---

## P0 Fixes (Applied This Session)

- `_merge_fn_registry` not cleared between tests (conftest.py)
- Bare `assert sidecar is not None` replaced with diagnostic ConstructError (decorators.py)

---

## Architecture Issues to File

These are follow-up tasks, not blockers:

### A1. Error hierarchy (P1)
ValueError used for 5+ distinct error categories across 7 modules. Tests can't assert specific failure modes. Fix: `NeographError` base + `CompileError`, `ConfigurationError`, `ExecutionError` subclasses.

### A2. Duplicated skip_when/renderer logic (P1)
`_make_produce_fn` and `_make_tool_fn` have identical blocks for skip_when, renderer dispatch, and dict-output wrapping. Tests must cover both. Fix: extract `_apply_skip_when()`, `_get_effective_renderer()`.

### A3. StructuredFake can't test include_raw path (P1)
`_call_structured()` passes `include_raw=True` by default. StructuredFake ignores it, so the `{"parsed": ..., "raw": ...}` code path + usage metadata extraction are untestable. Fix: add `StructuredFakeWithRaw` variant.

### A4. _merge_dicts duplicate key in Each — no guard in factory (P2)
The state reducer `_merge_dicts` raises on duplicate keys, but the factory `_build_state_update` doesn't check for duplicates when building keyed dicts for Each. If two fan-out items produce the same key, the error comes from deep in LangGraph's state management, not from neograph.

---

## Migration Strategy

### Target layout (9 files)

```
tests/
  conftest.py              # shared schemas + fixtures + registry cleanup
  fakes.py                 # unchanged
  test_pipeline_modes.py   # compile+run through all 5 modes          (~40 tests)
  test_modifiers.py        # Oracle, Each, Operator — all 3 surfaces  (~80 tests)
  test_validation.py       # assembly-time type checking               (~80 tests)
  test_node_decorator.py   # @node DX: inference, DI, error locations  (~70 tests)
  test_forward.py          # ForwardConstruct (merge existing file)    (~50 tests)
  test_composition.py      # sub-constructs, nesting, multi-output     (~30 tests)
  test_renderers.py        # renderers + describe_type                 (~60 tests)
```

### Migration phases (each = one commit, tests must pass)

| Phase | What moves | Risk |
|-------|-----------|------|
| 1 | Extract `test_renderers.py` (7 classes, ~44 tests) | Low — no deps on other classes |
| 2 | Merge forward files → `test_forward.py` (10 classes, ~50 tests) | Low — already separate |
| 3 | Extract `test_validation.py` (8 classes, ~47 tests) | Low — pure validation, no e2e |
| 4 | Extract `test_node_decorator.py` (14 classes, ~64 tests) | Med — some share schemas |
| 5 | Extract `test_modifiers.py` + `test_composition.py` | Med — interleaved |
| 6 | Remainder → `test_pipeline_modes.py` | Low — what's left |
| 7 | Shared schemas → conftest.py | Low — mechanical |

### Tests to delete

- `TestNodeInputsFieldRename` (5 tests) — migration artifact; any regression caught by 40+ tests that use `inputs`
- `TestNodeOutputsRename` (7 tests) — same pattern
- `TestExtractInputListUnwrap` (3 tests) — tests internal function; behavior covered by TestListOverEachEndToEnd

### Three-surface parity pattern

```python
@pytest.mark.parametrize("build", [
    _each_via_declarative,
    _each_via_decorator,
], ids=["declarative", "decorator"])
def test_each_produces_dict_result(build):
    graph = compile(build())
    result = run(graph, input={})
    assert isinstance(result["verify"], dict)
```

ForwardConstruct gets separate tests when topology differs (branching, loops).

---

## New Integration Tests (Priority Order)

### High (6 tests — real-world patterns with zero coverage)

| # | What | Why it matters |
|---|------|---------------|
| 1 | Dict-form outputs + Each modifier e2e | Unique code path in factory + state. Untested. |
| 2 | Each + Operator (fan-out with human review) | Common real-world pattern. Completely untested. |
| 3 | skip_when on gather/execute nodes | Same code as produce but never exercised on tool nodes. |
| 4 | Operator interrupt->resume via @node | Only tested via declarative. @node path untested. |
| 5 | DI params (FromInput/FromConfig) inside Each fan-out | Does neo_each_item override DI resolution? Unknown. |
| 6 | Each + Oracle combo via @node | Only tested via declarative in TestDeepCompositions. |

### Medium (6 tests — error paths and edge cases)

| # | What | Why |
|---|------|-----|
| 7 | `_check_each_path` validation errors (3 paths) | Each.over with bad dotted path, non-list terminal, type mismatch — all silent pass. |
| 8 | `_call_structured` TypeError fallback | Production LLMs may not support include_raw. |
| 9 | `_extract_json` regex edge cases | json_mode depends on this parser. No focused tests. |
| 10 | Dict-form outputs + Oracle modifier | State model emits per-key fields but no e2e test. |
| 11 | `Modifiable.map()` error paths (4 branches) | User-facing validation for .map() sugar. |
| 12 | Tool not registered error in invoke_with_tools | User-facing error never tested. |

### Modifier combination matrix (completeness)

| | Each | Oracle | Operator | skip_when |
|---|---|---|---|---|
| Each | -- | tested | **MISSING** | **MISSING** |
| Oracle | tested | -- | **MISSING** | **MISSING** |
| Operator | **MISSING** | **MISSING** | -- | **MISSING** |
| skip_when | **MISSING** | **MISSING** | **MISSING** | -- |

5 of 6 modifier pair combinations are untested.

---

## Execution Recommendation

**Phase A** (this session or next): File beads issues for A1-A4 architecture fixes + the 12 missing integration tests. Don't execute yet — the user reviews.

**Phase B** (dedicated session): Migration phases 1-7. Mechanical moves, one commit per phase. No test logic changes.

**Phase C** (after migration): Write the 6 high-priority integration tests in their target files. Add three-surface parity to modifier tests.

**Phase D** (ongoing): Architecture fixes (error hierarchy, dedup factory logic). These are code changes, not test changes.
