# Migration Plan v2 -- From Current State to Architecture v2

This document defines the phased migration for each architectural principle
in `architecture-v2.md`. Phases are ordered by bug-prevention ROI first,
lowest disruption first within the same ROI tier.

---

## Migration order (ROI-ranked)

1. **Factory wrapper unification** (HIGH-01, HIGH-02, DRY-01..10) --
   highest bug count, lowest disruption (internal refactor, no API change)
2. **DI resolution unification** (12 historical + HIGH-06) --
   high bug count, moderate disruption (new module, internal rewiring)
3. **Error builder pattern** (Pattern A) --
   medium value, lowest disruption (additive, no breaking changes)
4. **Type-safe modifier composition** (22 historical + neograph-35c3) --
   highest bug count, highest disruption (changes `Node` model, touches
   every dispatch site)
5. **Test behavioral contract** (HIGH-03..05) --
   continuous, no disruption (process change, not code change)

---

## 1. Factory Wrapper Unification

### Phase 1 -- Extract shared helpers (DONE / quick wins)

**What exists today:**

- `_apply_skip_when()` is already extracted (shared by all 3 wrappers)
- `_build_state_update()` is already extracted
- `_extract_input()` is already extracted
- `_render_input()` is already extracted
- `_resolve_primary_output()` is already extracted
- `_unwrap_loop_value()` and `_unwrap_each_dict()` are extracted

**Quick wins (not yet done):**

- [ ] Extract `_inject_oracle_config(state, config) -> config` from the
  identical block at `factory.py:337-345`, `factory.py:379-387`,
  `factory.py:449-457`. One function, 3 call sites.

  ```python
  def _inject_oracle_config(state: Any, config: RunnableConfig) -> RunnableConfig:
      oracle_gen_id = _state_get(state, "neo_oracle_gen_id")
      if oracle_gen_id is None:
          return config
      configurable = config.get("configurable", {})
      extra = {"_generator_id": oracle_gen_id}
      oracle_model = _state_get(state, "neo_oracle_model")
      if oracle_model is not None:
          extra["_oracle_model"] = oracle_model
      return {**config, "configurable": {**configurable, **extra}}
  ```

- [ ] Extract `_extract_context(state, node) -> dict | None` from the
  identical block at `factory.py:400-404` and `factory.py:478-483`.

  ```python
  def _extract_context(state: Any, node: Node) -> dict[str, Any] | None:
      if not node.context:
          return None
      return {
          name: _state_get(state, name.replace("-", "_"))
          for name in node.context
      }
  ```

- [ ] Extract `_resolve_effective_model(config, node) -> str` from the
  identical line at `factory.py:407` and `factory.py:486`.

**Verification:** Each extraction is a pure refactor. Run `pytest` -- all
tests must pass with identical behavior. The 3 wrapper functions shrink
but still exist as separate functions.

### Phase 2 -- Unified execution path (incremental, backward-compatible)

- [ ] Create `NodeInput` and `NodeOutput` typed containers as described
  in `architecture-v2.md` section 1 — no `Any` at dispatch boundaries.
- [ ] Create `_execute_node(node, state, config, dispatch)` using typed I/O.
- [ ] Create `ModeDispatch` protocol with typed `execute(NodeInput) -> NodeOutput`
  and three implementations: `ScriptedDispatch`, `ThinkDispatch`, `ToolDispatch`.
- [ ] Create `_dispatch_for_mode(node) -> ModeDispatch` that returns the
  right dispatch based on `node.mode`.
- [ ] Rewrite `make_node_fn` to call `_execute_node` for all non-raw modes.
- [ ] Keep `_make_raw_wrapper` unchanged (raw mode is an escape hatch with
  no preamble/postamble contract).
- [ ] Delete `_make_scripted_wrapper`, `_make_produce_fn`, `_make_tool_fn`.

**Files touched:** `factory.py` only.

**Verification:**
- All existing tests pass unchanged.
- Add a new test that verifies a custom preamble hook (e.g., timing
  callback) fires for scripted, think, and tool modes -- proving the
  single path serves all three.
- Measure: the Oracle config injection block appears exactly once in
  `factory.py` (grep count = 1).

### Phase 2b -- Loop router unification

- [ ] Create `_make_loop_router(item_name, field_name, count_field, loop,
  condition, unwrap_fn)` in `compiler.py`.
- [ ] `_add_loop_back_edge` calls it with a Node-aware unwrap function.
- [ ] `_add_subgraph_loop` calls it with a Construct-aware unwrap function.
- [ ] Delete the duplicated `loop_router` closures from both functions.

**Files touched:** `compiler.py` only.

**Verification:** All loop tests pass. The `loop_router` closure appears
exactly once (inside `_make_loop_router`).

### Phase 3 -- N/A

No breaking changes required. This is a pure internal refactor.

### Risk assessment

- **Low risk.** All changes are internal to `factory.py` and `compiler.py`.
  No public API surfaces change. No Node model changes. No Construct
  changes. The only risk is a subtle behavioral difference in the unified
  path vs. the original paths.
- **Mitigation:** Phase 1 extractions are trivially verifiable (inline
  the helper and diff). Phase 2 gets a dedicated integration test that
  exercises all 5 modes through `_execute_node`.

---

## 2. DI Resolution Unification

### Phase 1 -- Consolidate unwrap helpers (DONE)

**What exists today:**

- `_unwrap_loop_value()` in `factory.py` -- single source of truth for
  Loop unwrap, used by `_extract_input`, `_resolve_merge_args`, and
  `loop_router`.
- `_unwrap_each_dict()` in `factory.py` -- single source of truth for
  Each dict-to-list unwrap.
- `_resolve_di_value()` in `decorators.py` -- shared resolver for @node
  and @merge_fn.
- `_classify_di_params()` in `decorators.py` -- shared classifier.

### Phase 2 -- Extract `di.py` module, introduce `DIBinding` (incremental)

- [ ] Create `src/neograph/di.py` with:
  - `DIKind` enum: `FROM_INPUT`, `FROM_CONFIG`, `FROM_INPUT_MODEL`,
    `FROM_CONFIG_MODEL`, `FROM_STATE`, `CONSTANT`, `UPSTREAM`
  - `DIBinding` dataclass: `name`, `kind`, `inner_type`, `required`,
    `payload`
  - `DIBinding.resolve(config, state)` -- the ONE resolution function
  - `classify_params(fn, sig) -> list[DIBinding]` -- moved from
    `_classify_di_params`
  - `resolve_all(bindings, config, state, modifier_set) -> dict` -- the
    bouncer function

- [ ] Change `_classify_di_params` in `decorators.py` to delegate to
  `di.classify_params` and convert the result to `ParamResolution` format
  (backward compat shim).

- [ ] Change `_resolve_di_value` in `decorators.py` to delegate to
  `DIBinding.resolve` (backward compat shim).

- [ ] Change `_resolve_merge_args` to use `resolve_all` from `di.py`.

- [ ] Fix HIGH-06: in `DIBinding.resolve`, catch
  `(ValidationError, TypeError, ValueError)` instead of bare
  `except Exception`.

- [ ] Unify `lint.py` -- replace the 4 copy-pasted blocks with a single
  `_check_binding(binding, config, issues)` function that operates on
  `DIBinding` instances.

**Files touched:** New `di.py`, `decorators.py` (shim), `factory.py`
(import change), `lint.py` (simplification).

**Verification:**
- All existing DI tests pass unchanged (shim preserves `ParamResolution`
  format during transition).
- New test: `test_di_resolver_loop_unwrap` -- verify that `resolve_all`
  applies Loop unwrap for from_state params.
- New test: `test_di_resolver_each_unwrap` -- verify Each dict-to-list
  unwrap.
- New test: `test_di_binding_required_validation_error` -- verify that
  `(ValidationError, TypeError)` are caught but `RuntimeError` propagates.
- Measure: grep for `_unwrap_loop_value` call sites -- should be exactly
  2 (inside `di.py:resolve_all` and `factory.py:_extract_input` for
  upstream params, which are not DI).

### Phase 3 -- Full `DIResolver` adoption (minor breaking change)

- [ ] Replace `ParamResolution` type alias with `list[DIBinding]` in the
  `_node_sidecar` and `_param_resolutions` dicts.
- [ ] Remove the backward-compat shim in `decorators.py`.
- [ ] `_register_node_scripted` (the shim builder) uses `DIResolver`
  directly instead of manually iterating `param_res`.
- [ ] `make_oracle_merge_fn` in `factory.py` uses `DIResolver` for
  merge_fn DI resolution.

**Breaking change:** `ParamResolution` type alias is no longer exported.
Since the only consumer is neograph itself (the user never touches
`ParamResolution`), this is safe.

### Risk assessment

- **Medium risk.** The DI system is the most subtle part of neograph --
  bundled model resolution, frame-walking for local classes, the
  required/optional distinction. The shim layer in Phase 2 minimizes risk
  by keeping the old format alive during transition.
- **Mitigation:** Phase 2 is 100% backward compatible -- old tests pass
  through the shim. Phase 3 only happens after Phase 2 soaks for at least
  one release cycle (0.2.0.dev → 0.2.0).

---

## 3. Error Builder Pattern

### Phase 1 -- Add the builder (quick win)

- [ ] Add `NeographError.build()` class method to `errors.py` as described
  in `architecture-v2.md` section 5.
- [ ] All subclasses inherit it (`ConstructError.build()`,
  `CompileError.build()`, etc.).

**Files touched:** `errors.py` only.

**Verification:** New test: `test_error_builder_format` -- verify the
structured output format for each combination of parameters.

### Phase 2 -- Migrate existing error sites (incremental, backward-compatible)

Priority order (by frequency of user-facing errors):

1. [ ] `_construct_validation.py` -- highest volume. Convert
   `_format_no_producer_error`, `_check_fan_in_inputs`, `_check_each_path`
   to use `ConstructError.build()`.
2. [ ] `compiler.py` -- convert `CompileError(msg)` calls.
3. [ ] `factory.py` -- convert `ExecutionError(msg)` and
   `ConfigurationError(msg)` calls.
4. [ ] `modifiers.py` -- convert `ConfigurationError(msg)` calls in
   `model_post_init`.
5. [ ] `decorators.py` -- convert `ConstructError(msg)` calls.

Each file migration is an independent PR. Error messages change format
but not content. Downstream `pytest.raises(ConstructError, match="...")` tests
may need regex updates if they match on the old format.

**Verification per file:**
- Existing tests pass (possibly with updated `match=` patterns).
- New snapshot test: capture error messages from key paths and assert
  they match the structured format.

### Phase 3 -- Lint rule enforcement

- [ ] Add a `ruff` custom rule (or a grep-based CI check) that rejects
  direct `raise ConstructError(f"...")` in favor of
  `raise ConstructError.build(...)`.
- [ ] Or simpler: `ConstructError.__init__` logs a deprecation warning
  when called with a plain string instead of through `.build()`.

### Risk assessment

- **Low risk.** Additive change. Old-style errors still work. Migration
  is file-by-file, each independently verifiable.
- **Only risk:** downstream `match=` patterns in tests break if the
  message format changes. Mitigation: update test patterns in the same PR
  that migrates the source file.

---

## 4. Type-Safe Modifier Composition

### Phase 1 -- `ModifierCombo` enum + `classify_modifiers()` (DONE)

**What exists today:**

- `ModifierCombo` enum with 10 variants in `modifiers.py`
- `classify_modifiers(item)` function that classifies a node's modifier
  list into a `ModifierCombo` + modifier dict
- `Modifiable.__or__` with runtime guards for illegal combos
- Belt-and-suspenders checks in `_construct_validation.py` for programmatic
  API bypass

### Phase 2 -- `ModifierSet` model (incremental, backward-compatible)

- [ ] Create `ModifierSet(BaseModel, frozen=True)` in `modifiers.py` with
  typed slots (`each`, `oracle`, `loop`, `operator`).
- [ ] Add `ModifierSet.model_post_init` with illegal combo rejection.
- [ ] Add `ModifierSet.combo` property that returns `ModifierCombo`.
- [ ] Add `ModifierSet.with_modifier(mod)` method.
- [ ] Add `Node.modifier_set` as a computed property that constructs a
  `ModifierSet` from the existing `modifiers` list:

  ```python
  @property
  def modifier_set(self) -> ModifierSet:
      return ModifierSet(
          each=self.get_modifier(Each),
          oracle=self.get_modifier(Oracle),
          loop=self.get_modifier(Loop),
          operator=self.get_modifier(Operator),
      )
  ```

  This is a backward-compatible bridge: existing code still reads
  `modifiers`, new code reads `modifier_set`.

- [ ] Migrate ONE dispatch site (e.g., `_add_node_to_graph` in
  `compiler.py`) to use `match node.modifier_set.combo:` instead of
  `has_modifier()` chains. Verify all tests pass.
- [ ] Migrate remaining dispatch sites one at a time:
  - `compiler.py:_add_subgraph`
  - `factory.py:_build_state_update`
  - `state.py:compile_state_model` (the `_add_output_field` helper)
  - `_construct_validation.py:_validate_node_chain`

**Files touched:** `modifiers.py`, `node.py` (property), `compiler.py`,
`factory.py`, `state.py`, `_construct_validation.py`.

**Verification per dispatch site:**
- All existing tests pass.
- Add a test that constructs a Node with each valid `ModifierCombo` and
  verifies `modifier_set.combo` matches.
- Add a test that constructs an invalid combo via the programmatic API
  and verifies `ModifierSet.model_post_init` rejects it.

### Phase 3 -- Replace `list[Modifier]` with `ModifierSet` (breaking change)

- [ ] Change `Node` and `Construct` to carry `modifier_set: ModifierSet`
  instead of `modifiers: list[Modifier]`.
- [ ] `Modifiable.__or__` delegates to `modifier_set.with_modifier()`.
- [ ] Remove `has_modifier()` and `get_modifier()` -- access modifiers
  via `modifier_set.each`, `modifier_set.oracle`, etc.
- [ ] Remove `classify_modifiers()` -- the `combo` property replaces it.
- [ ] Remove belt-and-suspenders checks in `_construct_validation.py` --
  illegal combos are now unrepresentable.
- [ ] Version bump: 0.3.0 (or next minor).

**Breaking change:** `Node(modifiers=[...])` constructor form no longer
works. Programmatic API uses:

```python
node = Node(name="x", outputs=X)
node = node | Oracle(n=3, merge_fn="combine")
node = node | Each(over="y.items", key="id")
```

Or equivalently:

```python
node = Node(
    name="x",
    outputs=X,
    modifier_set=ModifierSet(
        oracle=Oracle(n=3, merge_fn="combine"),
        each=Each(over="y.items", key="id"),
    ),
)
```

Both go through `ModifierSet.model_post_init`.

### Risk assessment

- **High risk for Phase 3.** Every dispatch site changes. The `@node`
  decorator's sidecar re-registration after `model_copy` must be verified
  to work with the new field name.
- **Mitigation:** Phase 2 is fully backward-compatible and provides the
  bridge property. Phase 3 only happens after Phase 2 has been exercised
  in piarch for at least one release.
- **The sidecar problem:** when `__or__` calls `model_copy`, the new Node
  has a different `id()`. The sidecar re-registration pattern in
  `decorators.py` must work with `modifier_set` instead of `modifiers`.
  This is the most fragile part -- test with the neograph-jyw repro case.

---

## 5. Test Behavioral Contract

### Phase 1 -- Assertion audit (continuous, in progress)

- [ ] Run the test audit skill on all test files. Classify each assertion
  as "behavioral" (traces to setup) or "structural" (existence/type check).
- [ ] File beads issues for each structural-only assertion that should be
  behavioral.
- [ ] Remove or fill empty test classes (HIGH-03). Grep:
  `class Test.*:\n    pass` across test files.

**Verification:** The audit produces a report. Track structural assertion
count over time -- it should trend to zero.

### Phase 2 -- Strict test fixtures (incremental)

- [ ] Create `tests/strict_helpers.py` with strict scripted function
  factories that assert input types:

  ```python
  def strict_scripted(expected_type: type, result_factory: Callable) -> Callable:
      def fn(input_data, config):
          assert isinstance(input_data, expected_type), (
              f"Expected {expected_type.__name__}, got {type(input_data).__name__}: {input_data!r}"
          )
          return result_factory(input_data)
      return fn
  ```

- [ ] Migrate existing test scripted functions to use the strict factory,
  one test file at a time.
- [ ] Replace `monkeypatch` workarounds in `test_cli.py` with proper
  `monkeypatch` fixture usage (Pattern E).

**Verification:** Tests that previously silently accepted wrong input now
fail loudly. Track the count of `isinstance(input_data, dict)` fallback
patterns in test files -- should trend to zero.

### Phase 3 -- Obligation workflow enforcement (process change)

- [ ] All new tests must go through the `/obligation-test` workflow.
- [ ] Legacy tests that fail the assertion audit are tracked in beads
  and remediated via `/dev-practices:remediate`.
- [ ] CI check: a pytest plugin that counts `assert ... is not None`
  and `assert isinstance(...)` without a follow-up value assertion.
  Warn on introduction of new instances (ratchet -- count can only
  go down).

### Risk assessment

- **No code risk.** This is a process and tooling change.
- **Culture risk:** the strictness of test fixtures may slow down initial
  test writing. Mitigation: the strict factory helpers make it easier to
  write correct tests, not harder -- `strict_scripted(Claims, lambda c: Result(...))`
  is one line.

---

## Timeline Summary

| Phase | Work | Est. effort | Disruption | Depends on |
|-------|------|-------------|------------|------------|
| 1.1 | Extract `_inject_oracle_config`, `_extract_context`, `_resolve_effective_model` | 1 hour | None | -- |
| 1.2 | Unified `_execute_node` + `ModeDispatch` | 3 hours | None | 1.1 |
| 1.2b | Loop router unification | 1 hour | None | -- |
| 2.2 | Extract `di.py`, `DIBinding`, lint unification | 4 hours | None | -- |
| 3.1 | `NeographError.build()` | 30 min | None | -- |
| 3.2 | Migrate error sites (5 files) | 2 hours | Low (test match patterns) | 3.1 |
| 4.2 | `ModifierSet` model + bridge property | 4 hours | None | -- |
| 4.2b | Migrate dispatch sites (5 files) | 3 hours | None | 4.2 |
| 2.3 | Full `DIResolver` adoption | 2 hours | Low | 2.2 |
| 4.3 | Replace `list[Modifier]` with `ModifierSet` | 4 hours | High (API change) | 4.2b |
| 5.* | Test behavioral contract (continuous) | Ongoing | None | -- |

Total estimated effort for Phase 2 (the non-breaking migration): ~18 hours.
Phase 3 (breaking changes): ~6 hours, gated on piarch validation.

---

## Verification Checklist (per-phase completion criteria)

### Phase 1 complete when:

- [ ] `grep -c "neo_oracle_gen_id" src/neograph/factory.py` returns 1
  (not 3)
- [ ] `grep -c "context_data = {" src/neograph/factory.py` returns 1
  (not 2)
- [ ] All existing tests pass
- [ ] No new test files needed

### Phase 2 complete when:

- [ ] `factory.py` has exactly one `def .*wrapper` (the raw wrapper) plus
  `_execute_node`
- [ ] `compiler.py` has exactly one `loop_router` closure (inside
  `_make_loop_router`)
- [ ] `di.py` exists with `DIBinding.resolve` as the single resolution path
- [ ] `lint.py` has no copy-pasted DI check blocks (one `_check_binding`)
- [ ] `errors.py` has `NeographError.build()`
- [ ] 5+ source files use `.build()` for error construction
- [ ] `ModifierSet` model exists with `combo` property
- [ ] At least 2 dispatch sites use `match modifier_set.combo:`
- [ ] All existing tests pass
- [ ] Coverage does not decrease

### Phase 3 complete when:

- [ ] `Node` and `Construct` carry `modifier_set: ModifierSet`, not
  `modifiers: list[Modifier]`
- [ ] `has_modifier()` and `get_modifier()` are removed
- [ ] `classify_modifiers()` is removed
- [ ] Belt-and-suspenders checks in `_construct_validation.py` are removed
- [ ] `ParamResolution` type alias is removed
- [ ] piarch runs clean against the new API
- [ ] Version bumped to 0.3.0
