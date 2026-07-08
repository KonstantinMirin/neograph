# Testing Quality Review — neograph

**Scope:** `src/neograph/` (76 files) + `tests/` (210 files, 2750 tests collected).
**Date:** 2026-07-08
**Question framing:** Would a senior Python engineer call this "elegantly engineered"? Is the coverage real or illusory?

---

## Headline

**The test suite is, on the whole, genuinely strong and would survive senior scrutiny.** The dominant pattern is end-to-end: tests build a real `Construct`, `compile()` it, and `run()` it, with fakes standing in *only* at the LLM boundary. Mock usage is remarkably low (4 files touch `unittest.mock` at all; only one leans on it). The structural-guard layer is unusually sophisticated — several guards are pure-AST with their own positive/negative mutation meta-tests. This is not a suite that "asserts on mocks."

The findings below are mostly refinements, not foundations. There is **no illusory-coverage crisis**, but there are a handful of real soft spots: a couple of weak positive assertions, loose fixture error-regexes, a low Hypothesis exploration budget, narrow adoption of the dual-surface parity machinery, and no enforced coverage floor. None individually undercut the "elegant" claim; collectively they're the honest asterisks.

---

## Findings

### MEDIUM — Weak positive assertion in the flagship three-surface parity test
`tests/test_fanin_validation.py:667-670` — `test_fan_in_assembles_when_types_correct` is the canonical "template pattern for future parity tests" (its own docstring, line 656). The positive assertion is:
```python
pipeline = build()
assert len(pipeline.nodes) == 3
```
This verifies the construct was *built* with three nodes but proves nothing about fan-in validation actually running and passing. It would stay green if fan-in type-checking were deleted entirely. The negative twin (`test_fan_in_rejects_when_type_mismatches`, line 681) is meaningful (asserts `ConstructError` with the offending key + type in the message), but the positive case — held up as the template others should copy — anchors on node count. Recommend asserting on a compiled/validated artifact (e.g. `compile(pipeline, ...)` succeeds, or the producer/consumer edge resolved to the expected type).

### MEDIUM — "Three-surface parity" is real but the declarative surface tests a different code path
In the same parity block, `_fan_in_valid_declarative` / `_fan_in_mismatch_declarative` use **single-type** `inputs` (the isinstance-scan path), while the decorator and programmatic builders use **dict-form** fan-in. Per CLAUDE.md the single-type form "skips fan-in validation and defers to runtime isinstance scan." So the "declarative" cell of the parity matrix does not exercise the same dict-form validation as the other two cells — the parity is asserted across surfaces that aren't running identical logic. This is closer to "three surfaces each do something reasonable" than "one validator, three surfaces, identical behavior." Worth either using dict-form inputs in the declarative builder or documenting the asymmetry in the test.

### MEDIUM — Low Hypothesis exploration budget on the expensive topology/invariant tests
`tests/hypothesis/test_topologies.py` and `test_invariants.py` run many properties at `max_examples=10–20` (e.g. `test_topologies.py:317,352,735,804,848,933,970` at 10). The invariants themselves are strong and real — they `compile()` + `run()` generated pipelines and assert structural facts (both dict-output fields present with correct types `test_topologies.py:255-260`; sub-construct internals don't leak `:288-289`; empty-Each yields `{}` without deadlock `:325-326`). But at 10 examples per property the effective exploration is a handful of shapes, not a broad sweep. This is a defensible latency tradeoff (each example is a full compile+run), but it means these are "randomized spot-checks with good invariants" more than deep property exploration. Flagging so the tradeoff is a conscious one. (See the Hypothesis-focused findings appended below for strategy-diversity detail.)

### LOW/MEDIUM — Loose `CHECK_ERROR` regexes let a fixture pass on the wrong error
`tests/check_fixtures/should_fail/*.py` — several expected-error patterns are broad enough to match unintended failure paths: `type.*compatible|produces`, `no field`, `shadow|ambig|conflict|upstream`, `type.*compatible|no upstream produces` (this last appears on ~9 fixtures). Because the harness (`test_check_fixtures.py:129-132`) only checks that *some* raised error matches the regex, a fixture that regresses to a *different* defect emitting one of these common substrings ("produces", "upstream", "no field" all appear widely in neograph error text) would still pass — a false sense that the specific rule is guarded. Tighten to name the specific rule/type where practical.

### LOW/MEDIUM — `should_pass` fixtures can pass vacuously (no Construct to compile)
`tests/test_check_fixtures.py:140-149` — `test_should_pass` calls `_try_compile`, which iterates module-level `Construct` instances and returns `None` if there are none; the assertion `compile_error is None` then passes without anything compiling. Two current fixtures have no module-level Construct at all — `should_pass/outputs_inferred_from_annotation.py` and `outputs_explicit_matches_annotation.py` — they self-assert decorator behavior at import (`assert extract.outputs is Claims`) and never exercise the compile safety net they live in. Today the inline `assert` does real work, so it's not broken; but the harness would happily green a *future* should_pass fixture whose Construct is built inside a function (never compiled). Recommend: `test_should_pass` should assert it found ≥1 Construct (or that the module self-asserts), so "compiles cleanly" can't degrade to "compiled nothing."

### LOW — Dual-surface (sync/async) parity machinery is built but barely adopted
`tests/conftest.py:34-99` defines a `run_driver` fixture parametrized on `["sync","async"]` to prove identical behavior across `run()`/`arun()`, and the comment block lays out a mandatory "6-cell policy." Only **2** test files actually use it (`test_async_dual_path.py`, `test_async_harness.py`). The async execution path is otherwise covered by dedicated `test_async_*.py` files, so async isn't unguarded — but the "identical behavior on both surfaces" guarantee the fixture was built to enforce is spot-checked, not systematic.

### LOW — Stale scaffolding comments (doc drift, not a test hole)
- `tests/conftest.py:44-45,75,91` still describe `arun()` as "not implemented yet (Phase 1)" and the async cell as "INERT today." `arun` is in fact landed and exported (`hasattr(neograph, "arun")` is True), so the async cell is live and these comments mislead a reader about what's actually running.
- CLAUDE.md:428-434 documents a `tests/check_fixtures/known_gaps/` directory as the validation backlog ("move it there and file a bug"). The directory **does not exist**, and `test_check_fixtures.py` only globs `should_fail/` and `should_pass/` — so even if someone followed the doc and created `known_gaps/`, nothing would scan it. Either the mechanism was retired (update the doc) or it should be wired into the harness.

### LOW — Hypothesis runtime type-assertion helper is permissive
`tests/hypothesis/conftest.py` `_make_transform_fn` asserts `isinstance(input_data, (input_type, dict, type(None)))`. Accepting `dict` and `None` alongside the expected model means a wrong-typed input arriving as a bare dict would satisfy the assertion — weakening the runtime type-flow check that these generated pipelines are supposed to exercise.

---

## What is genuinely strong (evidence for "elegantly engineered")

- **Behavioral, not mock-based.** `tests/modes/test_llm_internals.py:47-93` (`test_context_passed_to_prompt_compiler_when_declared`) is representative: it builds a real module, compiles, `run()`s it, captures what the prompt compiler actually received, and asserts on the *verbatim raw model* flowing through state (`ctx_val.text == "<catalog>...`). The fake is only at the LLM seam. This pattern recurs across the modes suite.
- **The meta-guard is exemplary.** `tests/test_guards_meta.py` guards the guards with pure AST (no `re`, so it can't slip its own discipline), is receiver-agnostic about how `re` is imported (`import re as r`, `from re import search`, `pattern=` kwargs all detected — lines 257-274), and ships positive *and* negative mutation meta-tests plus an allowlist-staleness check (`test_allowlist_entries_are_actually_near_duplicates`, line 495). This is above-median engineering discipline.
- **Real fixtures, auto-discovered.** The rustc-style `check_fixtures` harness (`test_check_fixtures.py`) discovers fixtures by glob, imports and compiles each, and matches error text — a genuine validator-testing-the-validator layer, derived from real consumer (piarch) patterns.
- **Low skip/xfail (1 total), no trivial `assert True`.** 2612 `def test_` across the suite; the only skip is the async-cell guard, which is now live.

---

## Reproduction commands

```bash
# Fast, high-signal suites referenced above
uv run --extra dev pytest tests/test_check_fixtures.py tests/test_fanin_validation.py -q
uv run --extra dev pytest tests/test_guards_meta.py -q
uv run --extra dev pytest tests/hypothesis/ -q

# Confirm the doc-drift items
ls tests/check_fixtures/known_gaps            # -> No such file or directory
python -c "import neograph; print(hasattr(neograph,'arun'))"   # -> True (conftest says "not implemented")

# Survey loose fixture regexes
grep -rh '# CHECK_ERROR:' tests/check_fixtures/should_fail/*.py | sort | uniq -c | sort -rn
```

---
