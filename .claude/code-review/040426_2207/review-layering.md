# Layering Review

**Scope**: Recent additions to neograph — `Node.map()` on `Modifiable`, assembly-time `Construct` validation, `ConstructError`, and new tests in `test_e2e_piarch_ready.py`. Commits `6a7c8d6` (map sugar) and `bed01e2` (validation).
**Date**: 2026-04-04

## Layer Map (recap from prior review)

| Layer | Module(s) | Responsibility |
|---|---|---|
| L1 Data Models | `tool.py`, `modifiers.py`, `node.py`, `construct.py` | Pydantic declarations; constructor-time static checks |
| L2 State Bus | `state.py` | Generate state model from Node I/O |
| L3 Node Factory | `factory.py` | Create LangGraph node functions (incl. `_extract_input`) |
| L4 LLM Layer | `_llm.py` | LLM invocation |
| L5 Graph Compiler | `compiler.py` | `Construct` → `StateGraph` topology; owns the runtime dotted-path walker (`each_router`) |
| L6 Runner | `runner.py` | Execution + cleanup |

Expected direction: declarations flow down to the compiler. The new validation logic lives at L1, which is the correct home for declarative/static checks — but the new code introduces a parallel walker and some smaller responsibility smears noted below.

## Findings

### LR-01: Runtime `each.over` walker duplicated between `construct.py` and `compiler.py`
- **Severity**: Medium
- **Violation**: L1 (data model validation) re-implements a traversal that L5 (compiler) also owns
- **Files**:
  - `/Users/konst/projects/neograph/src/neograph/construct.py:176` (`_check_each_path` / `_resolve_field_annotation`)
  - `/Users/konst/projects/neograph/src/neograph/compiler.py:262-275` (`each_router` closure)
- **Description**: Both files split `each.over` on `"."` and walk segments, but against different targets: `construct.py` walks Pydantic `model_fields` to resolve *types*, `compiler.py` walks state instances to resolve *values*. The two walkers agree on path syntax (dot-separated, left-associative attribute access) by convention only — there is no shared helper or grammar definition. This is exactly the kind of "split brain" that invites drift: if `Each.over` ever grows to support indexing (`groups[0].label`), unions, or wildcards, both walkers must be updated in lock-step with no compiler check to enforce it. The split itself (types vs. values) is justified — type resolution must use `typing.get_type_hints` to unwrap `ForwardRef`s, which the runtime walker can't use. But the **path-parsing layer** should be shared.
- **Reproduction**: `grep -n 'each.over.split' /Users/konst/projects/neograph/src/neograph/construct.py /Users/konst/projects/neograph/src/neograph/compiler.py`
- **Recommended fix**: Extract a tiny `_parse_each_path(over: str) -> tuple[str, ...]` helper (e.g. on `Each` itself in `modifiers.py`, or a `_path.py` utility) and have both walkers consume it. This makes the path grammar a single point of truth even though the two traversals (by-type vs. by-value) stay independent. Low-lift and prevents a future "we added `.` escaping in one place but not the other" bug.

### LR-02: Inline 250-LOC validation module stuffed into `construct.py`
- **Severity**: Low
- **Violation**: Single-responsibility smear inside L1 (not a cross-layer leak)
- **File**: `/Users/konst/projects/neograph/src/neograph/construct.py:89-368`
- **Description**: `construct.py` is now 369 lines, of which ~280 are validation helpers (`_validate_node_chain`, `_check_item_input`, `_check_each_path`, `_resolve_field_annotation`, `_types_compatible`, `_extract_list_element`, `_type_name`, `_format_no_producer_error`, `_suggest_hint`, `_location_suffix`, `_source_location`, `_MISSING`). The `Construct` class body is 46 lines. The module's docstring and name both suggest "the pipeline blueprint declaration" — the validation subsystem is a second, larger concern that happens to share a file. This is not a layering *violation* (validation is correctly at L1 and the helpers are module-private), but the file now mixes "what a Construct is" with "how we type-check a chain of Constructs." That's the classic prelude to a validation rule growing a corner case, then a second one, then becoming impossible to review.
- **Reproduction**: `wc -l /Users/konst/projects/neograph/src/neograph/construct.py` and compare private-helper block (lines 89-368) to the class body (lines 43-87).
- **Recommended fix**: Move the validation block to `src/neograph/_construct_validation.py` (underscore-prefixed to mark internal). `construct.py` imports `_validate_node_chain` and `ConstructError` stays alongside it. The tests don't need to change — they already import `ConstructError` from `neograph` (re-exported via `__init__.py`). This is a pure file-split, no API change.

### LR-03: `_source_location()` runs `inspect.stack()` eagerly for every `Construct(...)` call
- **Severity**: Low
- **Violation**: Performance/layering smell — diagnostic cost paid on the happy path
- **File**: `/Users/konst/projects/neograph/src/neograph/construct.py:337-368`
- **Description**: `_location_suffix()` is called **unconditionally** inside the error-message builders (`_format_no_producer_error`, `_check_each_path` raises). That part is fine — it only fires when an error is being raised. However, the larger concern is architectural: the function walks `inspect.stack()`, filters by `f_globals['__name__']` against module names, and basenames paths. This is expensive introspection (frame walking materializes the whole Python stack) and it lives on the constructor path of a data model. On the happy path `_source_location` is never invoked, so the hot path is safe. But: (a) if a future refactor ever moves `_location_suffix()` out of the raise-branch (e.g. embedding it in the Construct as metadata), the cost becomes per-instance. (b) The frame filter is a private coupling to module names (`neograph`, `pydantic`) that is fragile under renaming/vendoring. (c) `construct.py` gains a `os` + `inspect` import purely for a diagnostic — dependencies that don't belong to the declaration layer.
- **Reproduction**: `grep -n '_source_location\|_location_suffix' /Users/konst/projects/neograph/src/neograph/construct.py`
- **Recommended fix**: Lazy-capture only the frame you need, at the one place you need it — the top of `_validate_node_chain`. Take one `sys._getframe(2)` walk (or capture via a `traceback.extract_stack(limit=N)` with a small N), store `(filename, lineno)` as a local variable, and pass it into the error builders. That way the cost scales with `limit`, not the full stack depth, and the module-name filter becomes a single string compare. Alternatively: move this to the suggested `_construct_validation.py` submodule so `construct.py` stays free of the `inspect` import. This is cosmetic but tightens the layer.

### LR-04: `.map()` on `Modifiable` carries a hidden forward reference to `Each`
- **Severity**: Low
- **Violation**: L1 (mixin) reaches down to a concrete modifier defined later in the same file
- **File**: `/Users/konst/projects/neograph/src/neograph/modifiers.py:68-124`
- **Description**: `Modifiable.map()` is defined at line 68, but its return statement `return self | Each(over=over, key=key)` depends on `Each` which is defined at line 159. This works because the method body runs at call time (not at class definition time), so by the time anyone calls `.map()` the whole module is loaded. It is, however, a subtle layering inversion: the mixin base class is coupled to one specific concrete modifier out of the five+ types that could inherit from `Modifier`. If a sixth modifier ever wanted to hijack `.map()` semantics (imagine `| Batch(...)`), it couldn't — the base class has hard-coded `Each`. The placement on `Modifiable` (so both `Node` and `Construct` inherit) is right; it's the hard reference to `Each` that makes this feel like L1/L1 coupling that should be parameterized.
- **Reproduction**: `grep -n 'class Each\|def map\|from neograph.modifiers' /Users/konst/projects/neograph/src/neograph/modifiers.py`
- **Recommended fix**: Two cheap options, both non-breaking:
  1. Keep `.map()` on `Modifiable` but import `Each` locally inside the method (`from neograph.modifiers import Each` — no-op since already in the module, but documents intent) to make the forward reference explicit.
  2. Move `.map()` closer to `Each` in the file (below the `Each` class) as a free function, and attach it to `Modifiable` via `Modifiable.map = _map`. Ugly but eliminates the forward-ref.
  Neither is urgent — the current code works and tests pass. Flagging it as layering-hygiene, not a bug.

### LR-05: Validation runs *after* `super().__init__`, bypassing Pydantic lifecycle hooks
- **Severity**: Low
- **Violation**: Framework-native lifecycle smell; not a layer leak
- **File**: `/Users/konst/projects/neograph/src/neograph/construct.py:77-84`
- **Description**: The validation is deliberately placed after `super().__init__(**kwargs)` inside `__init__` rather than in `model_post_init`. The comment explains the reason: `ConstructError` would otherwise be wrapped in a Pydantic `ValidationError`. This is a legitimate escape hatch — `model_post_init` exceptions are wrapped by Pydantic's constructor plumbing, so raising there produces an unhelpful stack. However, the current arrangement means `Construct` has **two** validation phases: Pydantic's field validators (invoked by `super().__init__`) and then NeoGraph's type/topology validator (invoked after). A subclass that overrode `__init__` and forgot to call `_validate_node_chain` would silently lose the check. This is a layering concern in the soft sense: NeoGraph's validation is coexisting with Pydantic's rather than integrated into it.
- **Reproduction**: `grep -n 'super().__init__\|_validate_node_chain' /Users/konst/projects/neograph/src/neograph/construct.py`
- **Recommended fix**: Keep the current approach — it's the cleanest way to preserve a bare `ConstructError` for users. But add a comment-level or docstring note that `Construct` is `final` (not meant to be subclassed) and that any subclass must chain to this validator. Alternatively, move the call into `model_post_init` and re-raise the inner `ConstructError` by catching Pydantic's wrapper — messier, not worth it. Flagging for awareness, not a required change.

### LR-06: `ConstructError(ValueError)` — intentional leaky abstraction
- **Severity**: Low
- **Violation**: Hierarchy design tradeoff, not a layer violation
- **File**: `/Users/konst/projects/neograph/src/neograph/construct.py:34-40`
- **Description**: `ConstructError` subclasses `ValueError` so existing `pytest.raises(ValueError)` sites in the test suite still catch it. This is explicitly documented in the class docstring. The test `test_construct_error_is_valueerror` (line 1840 in the test file) pins the behavior. This is a conscious tradeoff, and from a pure layering standpoint there's nothing wrong — `ValueError` is the right superclass for "bad value was passed." The weaker side is that `pytest.raises(ValueError)` in the test suite is now ambiguous: a test that meant to catch a Pydantic `ValidationError` (which is also a `ValueError`) could also catch a `ConstructError` by accident. For the current codebase this isn't a problem (reviewed the test file — no such case), but it's a risk going forward.
- **Reproduction**: `grep -n 'pytest.raises(ValueError)' /Users/konst/projects/neograph/tests/test_e2e_piarch_ready.py`
- **Recommended fix**: Keep `ValueError` as the base. The alternative (separate hierarchy) would force every existing test to migrate, and the risk is low. If tighter catching is wanted later, tests that care about the specific error should catch `ConstructError` directly.

### LR-07: `TestConstructValidation` + `TestNodeMap` inside the 2,542-line monolithic test file
- **Severity**: Low
- **Violation**: Test organization convention, not a layering issue
- **File**: `/Users/konst/projects/neograph/tests/test_e2e_piarch_ready.py` (only test file in the project)
- **Description**: The project has exactly one test file, `test_e2e_piarch_ready.py`, which now contains 30+ test classes across 2,542 lines. The new `TestNodeMap` (line 331) and `TestConstructValidation` (line 1738) follow the existing convention: every test lives in this file. Splitting them out now would break that convention. However, the file is over the healthy limit where you can still reason about it as a unit. The validation tests are particularly well-suited to extraction because they have no LLM-fake dependencies — they're pure data-model tests that could run in milliseconds in their own file.
- **Reproduction**: `wc -l /Users/konst/projects/neograph/tests/test_e2e_piarch_ready.py` and `find /Users/konst/projects/neograph/tests -name 'test_*.py'`
- **Recommended fix**: Don't split in this PR — the existing convention is "one file, many classes." If a future PR introduces per-module test files (`tests/test_construct_validation.py`, `tests/test_node_map.py`, `tests/test_compiler.py`, etc.), move them then. The current placement is consistent with the convention even though the convention itself is worth revisiting.

## Explicit answers to the review questions

1. **Is `Construct.__init__` the right layer for type validation?** Yes — validation of a declarative data model at construction time is the textbook L1 job and no lower layer (state, factory, compiler) has the information needed to check "does node N's input have an upstream producer." Inlining vs. submodule: see LR-02 — current inline layout is a style smell, not a layer violation, but a 280-LOC helper block in a 369-LOC file is the right moment to split. Low-lift recommendation.

2. **Does validation logic duplicate `state.py`/`compiler.py`?** It does duplicate the *path-parsing grammar* used by `compiler.each_router` (LR-01). The type-vs-value split is justified — they need different resolvers. The *splitter* (`.over.split(".")`) and the concept of "walk segments left-to-right" should be one shared helper. State builder (`state.py`) does not overlap — it just creates Pydantic fields from node outputs, no path walking.

3. **Should `.map()` live on `Modifiable` or on `Node`/`Construct`?** `Modifiable` is correct — both `Node` and `Construct` inherit it, and the test `test_map_on_construct` (line 412) exercises the `Construct` path. The weakness (LR-04) is that the mixin references `Each` specifically; if that ever becomes a problem, extract to a free function or a protocol. Not urgent.

4. **Is `inspect.stack()` in `_source_location` the right layer?** See LR-03. It's safe on the happy path (only invoked when formatting an error) but the `os` + `inspect` imports leak into the declaration module. Two fixes: move to `_construct_validation.py` (cleanest) or replace with bounded `sys._getframe` walk. Low-severity.

5. **Is `ConstructError(ValueError)` leaky?** See LR-06. It is intentionally leaky for back-compat with existing `pytest.raises(ValueError)` sites. Acceptable, documented tradeoff.

6. **Does validation-in-`__init__` violate Pydantic lifecycle?** See LR-05. It violates the spirit (there's a reason `model_post_init` exists), but for a good reason: `ConstructError` escapes cleanly instead of being wrapped in Pydantic's `ValidationError`. Keep as-is; add a docstring note that subclasses must chain to `_validate_node_chain`.

7. **Test placement.** See LR-07. Monolithic file is the existing convention; don't break it in this PR. Flag as a future refactor.

## Summary

- Critical: 0
- High: 0
- Medium: 1 (LR-01 — duplicate path walker)
- Low: 6 (LR-02..LR-07)

## Verdict

The new validation code is **well-placed at the correct layer**. The `Construct` data model is the right home for chain-compatibility checks, `.map()` is correctly on the shared mixin, and `ConstructError` is re-exported at the package level as users expect. There are no critical or high-severity layering violations introduced by these commits.

The one finding worth addressing before forgetting about it is **LR-01**: the path-parsing grammar for `Each.over` now lives in two places (type-side walker in `construct.py`, value-side walker in `compiler.py`). Extracting a one-line helper prevents a class of future bugs.

The rest of the findings are hygiene — file-size growth (LR-02), cheap `inspect` machinery in a data-model module (LR-03), a subtle forward reference inside the mixin (LR-04), a Pydantic-lifecycle note (LR-05), a documented-intentional hierarchy leak (LR-06), and test-file organization (LR-07). None of these block the current work, and I'd ship this as-is and file a small follow-up for LR-01 + LR-02.

## Relevant files

- `/Users/konst/projects/neograph/src/neograph/construct.py`
- `/Users/konst/projects/neograph/src/neograph/modifiers.py`
- `/Users/konst/projects/neograph/src/neograph/compiler.py`
- `/Users/konst/projects/neograph/src/neograph/node.py`
- `/Users/konst/projects/neograph/src/neograph/state.py`
- `/Users/konst/projects/neograph/src/neograph/__init__.py`
- `/Users/konst/projects/neograph/tests/test_e2e_piarch_ready.py`
- `/Users/konst/projects/neograph/.claude/code-review/040426_1300/review-layering.md` (prior review for baseline layer map)
