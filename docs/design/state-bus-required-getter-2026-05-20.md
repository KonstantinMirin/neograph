# neograph-izo1: StateBus.get_required() — best-path analysis

**Date**: 2026-05-20
**Ticket**: neograph-izo1
**Status**: read-only audit (no code changes)
**Spec source**: `docs/design/architecture-decisions.md` §7
**Predecessor**: neograph-jo35 (obsolete after Batch 2 / neograph-036p deleted `_state_get`)

---

## 0. Scope

§7 says:

> - The default state read raises `NeographError` on missing fields.
> - An explicitly-optional read returns `None` silently. Used only where the missing case is meaningful (e.g., reading a not-yet-populated collector field during fan-out, or reading an output a downstream node has not produced yet within a partial run).
> - Silent-`None` reads of required fields are a bug.

Today the `StateBus` Protocol exposes only `get(key, default=None) -> Any`, which silently returns the default on missing — the **optional** semantics, applied universally. This is the inversion of §7's intent: the *default* read should raise.

This document audits every call to `.get(...)` on a `StateBus`-typed expression in `src/neograph/`, classifies each as REQUIRED / OPTIONAL-* / AMBIGUOUS, proposes the `get_required` API, and lists the migration steps.

**Out of scope**: `.get(...)` calls on plain dicts (e.g., `config.get("configurable", {})`, `result.get(field_name)`), on `dict` reducer output (e.g., `mods.get("loop")`), or on registry dicts (`per_compile.get(...)`). Those are not state-bus reads.

---

## 1. Per-call-site classification

Twenty-five (25) `.get(...)` call sites on `StateBus` expressions. Convention used below:

- **state** / **bus** is always a `StateBus` instance at the call site.
- Where the file's `state` parameter is union-typed `StateBus | None`, the inner `.get(...)` only runs when `state is not None`.

| # | Call site (file:line) | Caller | Key being read | Classification | Rationale |
|---|---|---|---|---|---|
| 1 | `_input_shape.py:40` | `_classify_input_shape` | `own_field` (`field_name_for(node.name)` or dict-form `{field}_{primary_key}`) | **OPTIONAL-loop-bootstrap** | Read happens on **first** loop-router pass before the loop body has produced any output. Missing here signals "iteration 0, no self-output yet" and the function correctly falls through to `EACH_ITEM`/`FAN_IN_DICT` classification. Silent-None is meaningful. |
| 2 | `_input_shape.py:44` | `_classify_input_shape` | `StateKeys.EACH_ITEM` (`neo_each_item`) | **OPTIONAL-framework** | Probes whether the current dispatch came from an `Each` fan-out. For non-fan-out nodes `EACH_ITEM` is never set; absence is the signal. Silent-None drives classification. |
| 3 | `_input_shape.py:60` | `_extract_loop_reentry` | `own_field` (loop-reentry self-output) | **REQUIRED** | This branch is reached only after `_classify_input_shape` confirmed `own_val` is a non-empty list. By the time we reach line 60 the field MUST exist. If it disappeared between classification and extraction, that's a real bug. Fail loud. |
| 4 | `_input_shape.py:79` | `_extract_loop_reentry` | `state_key = field_name_for(key)` (sibling upstream in multi-key loop re-entry) | **OPTIONAL-loop-bootstrap** | On loop iter 1+, sibling keys may or may not have been re-produced this iteration. The branch at line 80 explicitly handles `upstream_val is not None`. Silent-None is the documented sentinel for "use latest self-value instead." |
| 5 | `_input_shape.py:96` | `_extract_each_item` | `StateKeys.EACH_ITEM` | **REQUIRED** | Called only after `_classify_input_shape` returned `EACH_ITEM`, which checks the same key. By construction the key is present. (Aside: this is a dispatch-confirmed read; could even use `get_required` and discard a None check, since classification already gated.) |
| 6 | `_input_shape.py:106` | `_extract_fan_in_dict` | `StateKeys.EACH_ITEM` | **REQUIRED** | Branch taken only when `input_name == node.fan_out_param` — the node IS the fan-out target, so `EACH_ITEM` is the dispatched value. Silent-None would propagate `None` to the user's callable, masking a wiring bug. |
| 7 | `_input_shape.py:109` | `_extract_fan_in_dict` | `state_key = field_name_for(input_name)` (named upstream) | **REQUIRED** | Fan-in inputs come from upstream nodes already validated by `_validate_node_chain` to exist before this consumer runs. Missing upstream = a real wiring bug. The current silent-None lets `None` flow into user callables and produces a confusing `AttributeError` rather than a structured `NeographError`. |
| 8 | `_input_shape.py:124` | `_extract_single_type` | `attr_name` (every key in `state.keys()`) | **REQUIRED** | The loop iterates `state.keys()`, so by definition every key exists. Silent-None here is harmless but technically wrong — switching to `get_required` makes the intent explicit. |
| 9 | `_execute.py:46` | `_extract_context` | `field_name_for(name)` for each declared `node.context` field | **REQUIRED** | `node.context` is a user-declared list of upstream string-context fields. The validator should guarantee they exist (or be extended to do so — see open question §8). `cast(str, ...)` already assumes non-None; today a missing field silently gives `None` cast to `str`, which then renders as the literal string `"None"` in the prompt. Definitively a §7 violation. |
| 10 | `_wiring.py:395` | `loop_router` (inside `_make_loop_router`) | `count_field = StateKeys.loop_count(field_name)` | **OPTIONAL-loop-counter** | The loop counter is incremented by `_build_state_update` AFTER the first body execution. On the very first router pass after the first body run, the counter exists; but the call site uses `bus.get(count_field, 0)` with an explicit `0` default which is the documented bootstrap value. Acceptable; keep with comment. |
| 11 | `_wiring.py:436` | `unwrap` (inside `_node_loop_unwrap`) | `state_field` (the loop body's output field, possibly dict-form `{field}_{primary_key}`) | **OPTIONAL-loop-bootstrap** | Same situation as #1: on the first router pass the field may be a not-yet-populated `[]` (from the append-reducer). The code at lines 437-444 explicitly handles `isinstance(own_val, list)` empty vs non-empty AND the `else` falls through to `pragma: no cover` "Loop reducer always produces a list". Silent-None is meaningful: signals "no output yet" and the loop condition lambda is expected to handle it (see lint check `loop_condition_none_unsafe`). |
| 12 | `_wiring.py:454` | `_construct_loop_unwrap` | `field_name` (sub-construct output) | **OPTIONAL-loop-bootstrap** | Sub-construct version of #11. First-pass bootstrap; user condition expected to handle `None`. |
| 13 | `_oracle.py:34` | `_inject_oracle_config` | `StateKeys.ORACLE_GEN_ID` | **OPTIONAL-framework** | Only set when running inside an Oracle fan-out via `Send`. Non-oracle dispatches have it absent; the function returns the original config unchanged in that case. The next line (`if oracle_gen_id is None: return config`) explicitly relies on silent-None. |
| 14 | `_oracle.py:39` | `_inject_oracle_config` | `StateKeys.ORACLE_MODEL` | **OPTIONAL-framework** | Set only when Oracle was configured with a `models=` list. May legitimately be absent even inside an Oracle fan-out. Silent-None drives the conditional below. |
| 15 | `_oracle.py:84` | `make_eachoracle_redirect_fn`'s inner closure | `StateKeys.EACH_ITEM` | **REQUIRED** | This closure runs as the generator inside an Each×Oracle fusion. The flat router (in `_wiring.py:_add_each_oracle_fused`) always populates `EACH_ITEM` in the `Send` payload. Absence = wiring bug. (Line 85 falls back to `"unknown"` if `item is None`, which masks the bug.) |
| 16 | `_oracle.py:316` | `make_each_redirect_fn`'s inner closure | `StateKeys.EACH_ITEM` | **REQUIRED** | Same as #15 for the plain Each path. Each router always populates `EACH_ITEM` in the `Send` payload. |
| 17 | `_subconstruct.py:34` | `_scan_subgraph_input` | `attr_name` (every key in `state.keys()`) | **REQUIRED** | Same as #8 — iterating known keys. |
| 18 | `_subconstruct.py:84` | `subgraph_node` (inside `make_subgraph_fn`) | `field_name` (sub-construct's own field for loop re-entry) | **OPTIONAL-loop-bootstrap** | Same situation as #1 / #11 — first iteration has no own output yet. The `isinstance(own_val, list) and own_val` guard handles missing/empty. |
| 19 | `_subconstruct.py:96` | `subgraph_node` | `"node_id"` (with explicit `""` default) | **OPTIONAL-framework** | `node_id` is a DI-style context-keyed input that may or may not be present. Empty-string default propagates through to the sub-graph. Acceptable but the magic-string `""` should be made explicit with a comment. |
| 20 | `_subconstruct.py:105` | `subgraph_node` | `ctx_field = field_name_for(ctx_name)` (context forwarding) | **OPTIONAL-context** | Iterates over each declared sub-node's `context` and forwards available context to the sub-construct. Missing context is acceptable in the parent (sub-node will get `None`); the `if val is not None` guard at line 106 makes the optional semantics explicit. Same root cause as #9 — context fields aren't validated to exist. Treat consistently with #9: if we lock down #9, this loops becomes REQUIRED too. **See open question §8.** |
| 21 | `_subconstruct.py:133` | `subgraph_node` | `count_field = StateKeys.loop_count(field_name)` | **OPTIONAL-loop-counter** | Same as #10; same justification. |
| 22 | `_state_write.py:53` | `_build_state_update` | `StateKeys.EACH_ITEM` | **OPTIONAL-framework** | Same as #2. The state-write path is invoked for both Each and non-Each nodes; absence of `EACH_ITEM` is the "not inside a fan-out" signal. The `each_mod and each_item is not None` guards at lines 64-71 make the optional semantics explicit. |
| 23 | `_state_write.py:81` | `_build_state_update` (loop counter increment branch) | `count_field = StateKeys.loop_count(field_name)` | **OPTIONAL-loop-counter** | Same as #10 / #21. |
| 24 | `_state_write.py:136` | `_apply_skip_when` (loop counter increment on skip-no-value branch) | `count_field = StateKeys.loop_count(field_name)` | **OPTIONAL-loop-counter** | Same as #10 / #21 / #23. |
| 25 | *(reserved — no further call sites found)* | — | — | — | — |

---

## 2. Aggregate counts + clusters

### Distribution

| Classification | Count |
|---|---|
| REQUIRED | 8 |
| OPTIONAL-loop-bootstrap | 5 |
| OPTIONAL-framework | 4 |
| OPTIONAL-loop-counter | 4 |
| OPTIONAL-context | 1 (paired with #9 — see §8) |
| AMBIGUOUS | 0 (all #9 / #20 ambiguity is surfaced as an open question, not as ambiguous classification) |
| Total audited | 22 unique behavior categories, 24 call sites (excluding #25 placeholder) |

Wait — recounting strictly: 24 call sites (rows #1–#24). All classified, no AMBIGUOUS rows.

### Clusters

- **Loop iteration bootstrap** (5: #1, #4, #11, #12, #18) — every loop-router and loop-aware extractor has the "first pass before body output exists" problem. All five are legitimately optional. The cluster suggests these may want a dedicated helper (`bus.get_loop_bootstrap(key)`) but that's a future refactor, not part of izo1.
- **Loop counter** (4: #10, #21, #23, #24) — every loop counter read pairs with `or 0`. These could move to a dedicated `bus.get_counter(key) -> int` helper that internalizes the `or 0` idiom and skips going through the general silent-None path.
- **Framework neo_* signals** (4: #2, #13, #14, #22) — `EACH_ITEM` / `ORACLE_GEN_ID` / `ORACLE_MODEL` are framework-internal signals whose absence is the documented "not in this regime" sentinel. All legitimately optional.
- **Wiring-validated upstreams** (7: #3, #5, #6, #7, #8, #15, #16, #17) — by the time we read these, the dispatch path / validator has guaranteed presence. Switching to `get_required` makes the assumption explicit and converts wiring bugs from confusing `AttributeError`s into structured `NeographError`s.
- **User-declared context** (1: #9, paired with #20) — `node.context` is user-declared; the validator should guarantee context fields are produced upstream. Open question whether they're actually enforced today; see §8.

### Discard-pattern check

No call site reads-and-discards. Every `.get(...)` result is either passed to a downstream check, stored in a dict, indexed, or compared against `None`. So there are no "read just to assert presence" candidates that would become `get_required(...)` with discarded value.

---

## 3. API proposal: `StateBus.get_required`

### Protocol extension

```python
# src/neograph/_state_bus.py

@runtime_checkable
class StateBus(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def get_required(self, key: str, *, node_label: str | None = None) -> Any: ...
    def keys(self) -> list[str]: ...
```

### Semantics

- `get_required(key, *, node_label=None)` returns the bound value when `key` is present, including `None` if `None` was explicitly stored. (Mirrors `getattr`/`dict.get` "key present" semantics.)
- Raises `NeographError` (via `NeographError.build(...)`) when `key` is **absent** (not bound).

### Implementation sketch

```python
class _DictStateBus:
    __slots__ = ("_state",)

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def get_required(self, key: str, *, node_label: str | None = None) -> Any:
        if key not in self._state:
            raise NeographError.build(
                f"required state field '{key}' missing",
                expected=f"state to contain '{key}' bound to a value",
                found="key not in state",
                hint=(
                    "Upstream node did not write this field, or the dispatch "
                    "path did not populate it. This is a wiring bug — required "
                    "reads must see a bound value."
                ),
                node=node_label,
            )
        return self._state[key]

    def keys(self) -> list[str]:
        return list(self._state.keys())


class _ModelStateBus:
    __slots__ = ("_state",)

    def __init__(self, state: BaseModel) -> None:
        self._state = state

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self._state, key, default)

    def get_required(self, key: str, *, node_label: str | None = None) -> Any:
        _SENTINEL = object()
        val = getattr(self._state, key, _SENTINEL)
        if val is _SENTINEL:
            raise NeographError.build(
                f"required state field '{key}' missing",
                expected=f"state to contain '{key}' bound to a value",
                found=f"attribute not on state model {type(self._state).__name__}",
                hint=(
                    "Upstream node did not write this field, or the state "
                    "schema does not declare it. Required reads must see a "
                    "bound value."
                ),
                node=node_label,
            )
        return val

    def keys(self) -> list[str]:
        return list(self._state.__class__.model_fields.keys())
```

### Notes

- **Sentinel vs `hasattr`**: `_ModelStateBus` uses a sentinel rather than `hasattr` because Pydantic `BaseModel` instances with `extra="allow"` may not raise on missing attrs the same way bare classes do. Sentinel is safer.
- **Why `node_label` is keyword-only**: forces explicit labeling at call sites; encourages plumbing `node.name` through helpers.
- **`NeographError`, not a new subclass**: §7 names `NeographError` explicitly. A `StateMissingError(NeographError)` subclass is gratuitous unless tests need to distinguish. (Open question §8.)
- **No change to `get(default=...)`**: optional reads keep their existing signature. The migration is additive.

---

## 4. Migration plan

### REQUIRED → switch to `get_required`

Total: **8 call sites**, grouped by file.

#### `src/neograph/_input_shape.py` (5)

The helpers don't take `node_label` today; they take a `Node` object as second arg so we can pass `node_label=node.name`.

```python
# Line 60 — BEFORE:
own_val = state.get(own_field)
# AFTER:
own_val = state.get_required(own_field, node_label=node.name)
# Justification: only reached after _classify_input_shape confirmed list-bound.
```

```python
# Line 96 — BEFORE:
return state.get(StateKeys.EACH_ITEM)
# AFTER:
return state.get_required(StateKeys.EACH_ITEM, node_label=node.name)
# Justification: classify returned EACH_ITEM after probing the same key.
```

```python
# Line 106 — BEFORE:
value = state.get(StateKeys.EACH_ITEM)
# AFTER:
value = state.get_required(StateKeys.EACH_ITEM, node_label=node.name)
# Justification: branch taken only when input_name == node.fan_out_param.
```

```python
# Line 109 — BEFORE:
value = state.get(state_key)
# AFTER:
value = state.get_required(state_key, node_label=node.name)
# Justification: fan-in upstream validated by _validate_node_chain.
```

```python
# Line 124 — BEFORE (inside `for attr_name in state.keys():`):
val = state.get(attr_name)
# AFTER:
val = state.get_required(attr_name, node_label=node.name)
# Justification: iterating known keys; absence here = StateBus invariant bug.
```

#### `src/neograph/_execute.py` (1)

```python
# Line 46 — BEFORE:
return {
    name: cast(str, state.get(field_name_for(name)))
    for name in node.context
}
# AFTER:
return {
    name: cast(str, state.get_required(field_name_for(name), node_label=node.name))
    for name in node.context
}
# Justification: user-declared node.context — see §8 about validator coverage.
# If validator does NOT yet enforce context-field production, this raise is
# what surfaces the gap. Test fix-out (§7) reflects this.
```

#### `src/neograph/_oracle.py` (2)

These are inside closures that don't have `node_label` in scope. Pass it in from the caller (`make_eachoracle_redirect_fn` / `make_each_redirect_fn` already receive `field_name` — use that).

```python
# Line 84 — BEFORE:
item = adapt_state(state).get(StateKeys.EACH_ITEM)
# AFTER:
item = adapt_state(state).get_required(StateKeys.EACH_ITEM, node_label=field_name)
# Justification: Each×Oracle fusion router always populates EACH_ITEM.
```

```python
# Line 316 — BEFORE:
each_item = adapt_state(state).get(StateKeys.EACH_ITEM)
# AFTER:
each_item = adapt_state(state).get_required(StateKeys.EACH_ITEM, node_label=field_name)
# Justification: Each router always populates EACH_ITEM.
# NOTE: existing `if val is not None and each_item is not None:` guard at
# line 321 becomes redundant for each_item — only val needs guarding.
# Simplify the guard accordingly in the same PR.
```

#### `src/neograph/_subconstruct.py` (1)

```python
# Line 34 — BEFORE (inside `for attr_name in reversed(state.keys()):`):
val = state.get(attr_name)
# AFTER:
val = state.get_required(attr_name)
# Justification: iterating known keys; no node_label available (this is a
# sub-construct boundary helper, not a per-node helper).
```

### OPTIONAL-* → keep `.get(...)`, add comment

Total: **16 call sites**. Each gets a one-line comment with the rationale. Grouped by file.

#### `src/neograph/_input_shape.py` (3)

```python
# Line 40 (in _classify_input_shape):
# StateBus.get optional: first loop-router pass; self-output not yet bound.
own_val = state.get(own_field)
```

```python
# Line 44 (in _classify_input_shape):
# StateBus.get optional: framework signal — absent for non-Each nodes.
replicate_item = state.get(StateKeys.EACH_ITEM)
```

```python
# Line 79 (in _extract_loop_reentry, multi-key branch):
# StateBus.get optional: sibling may not have been re-produced this iteration.
upstream_val = state.get(state_key)
```

#### `src/neograph/_wiring.py` (3)

```python
# Line 395 (loop_router):
# StateBus.get optional: loop counter — 0 on first pass (pre-body).
count = bus.get(count_field, 0)
```

```python
# Line 436 (unwrap in _node_loop_unwrap):
# StateBus.get optional: loop body output not yet bound on first router pass.
own_val = state.get(state_field)
```

```python
# Line 454 (_construct_loop_unwrap):
# StateBus.get optional: sub-construct output not yet bound on first pass.
val = state.get(field_name)
```

#### `src/neograph/_oracle.py` (2)

```python
# Line 34 (_inject_oracle_config):
# StateBus.get optional: framework signal — set only inside Oracle fan-out.
oracle_gen_id = state.get(StateKeys.ORACLE_GEN_ID)
```

```python
# Line 39 (_inject_oracle_config):
# StateBus.get optional: framework signal — set only when models= configured.
oracle_model = state.get(StateKeys.ORACLE_MODEL)
```

#### `src/neograph/_subconstruct.py` (4)

```python
# Line 84 (subgraph_node, loop reentry probe):
# StateBus.get optional: sub-construct's own output not bound on first iter.
own_val = bus.get(field_name)
```

```python
# Line 96 (subgraph_node, node_id default):
# StateBus.get optional: node_id is a contextual DI field; empty default OK.
sub_input: dict[str, Any] = {"node_id": bus.get("node_id", "")}
```

```python
# Line 105 (subgraph_node, context forwarding):
# StateBus.get optional: forwarding only when parent already populated context.
val = bus.get(ctx_field)
```

```python
# Line 133 (subgraph_node, loop counter):
# StateBus.get optional: loop counter — 0 on first iter.
current = bus.get(count_field) or 0
```

#### `src/neograph/_state_write.py` (4)

```python
# Line 53 (_build_state_update):
# StateBus.get optional: framework signal — absent outside Each fan-out.
each_item = state.get(StateKeys.EACH_ITEM) if state is not None else None
```

```python
# Line 81 (_build_state_update, loop counter):
# StateBus.get optional: loop counter — 0 on first iter.
current_count = (state.get(count_field) if state is not None else None) or 0
```

```python
# Line 136 (_apply_skip_when, loop counter):
# StateBus.get optional: loop counter — 0 on first iter.
current_count = (state.get(count_field) if state is not None else None) or 0
```

---

## 5. Test additions

### 5.1 Unit tests — new file `tests/test_state_bus_required.py`

```text
TestDictStateBusGetRequired:
  test_returns_value_when_key_present_dict
  test_returns_none_when_value_is_explicitly_none_dict
  test_raises_neograph_error_when_key_absent_dict
  test_error_message_names_key
  test_error_message_includes_node_label_when_provided
  test_error_message_omits_node_label_when_not_provided

TestModelStateBusGetRequired:
  test_returns_value_when_attribute_present_model
  test_returns_none_when_attribute_is_explicitly_none_model
  test_raises_neograph_error_when_attribute_absent_model
  test_error_message_names_key
  test_error_message_includes_node_label
  test_dynamic_basemodel_with_extra_allow

TestProtocolSurface:
  test_state_bus_protocol_requires_get_required  # isinstance(_, StateBus) checks
```

### 5.2 Behavioral tests — new file `tests/test_state_bus_required.py` (same file)

```text
TestRequiredReadsRaiseOnMissingFanIn:
  test_missing_fan_in_upstream_raises_with_node_context
    # Build a Construct where a fan-in input has its state field stripped
    # (e.g., synthetically-empty state passed to _extract_fan_in_dict).
    # Pre-fix: returns None, propagates AttributeError in user callable.
    # Post-fix: raises NeographError with the missing key and node name.

  test_missing_loop_reentry_self_output_raises
    # _extract_loop_reentry called with state lacking own_field.
    # Should raise (this is the post-classification REQUIRED read).

  test_missing_each_item_in_each_dispatch_raises
    # _extract_each_item called without EACH_ITEM in state.
    # Should raise; today silently returns None.

TestOptionalReadsRemainSilent:
  test_classify_input_shape_first_loop_pass_no_raise
    # Pre-loop dispatch with no own_field in state — classify must NOT raise.
  test_inject_oracle_config_non_oracle_dispatch_no_raise
    # Non-Oracle node — ORACLE_GEN_ID absent — must NOT raise.
  test_loop_counter_bootstrap_returns_zero
    # First loop iter — count_field absent — `get(_, 0)` returns 0.

TestThreeSurfaceParity:
  test_node_decorator_surface_required_read_raises
  test_declarative_node_surface_required_read_raises
  test_forward_construct_surface_required_read_raises
    # The three-surface parity rule: a required read failing on a missing
    # upstream must raise consistently regardless of which API built the IR.
```

### 5.3 Structural guard — extend `tests/test_structural_guards.py`

```text
class TestStateBusGetUsesRequiredByDefault:
    def test_state_bus_get_call_sites_annotated(self):
        # AST-walk every src/neograph/*.py file.
        # For every Call node whose .func is `<expr>.get` where <expr> is
        # name-bound to a StateBus instance (annotation 'StateBus' or
        # result of adapt_state(...)), assert one of:
        #   (a) the call site is on a line with a preceding comment line
        #       matching r'#\s*StateBus\.get optional:'
        #   (b) the method name is `get_required` (no annotation needed)
        #   (c) the call has 2+ positional args AND a comment on the same
        #       line matching r'#.*default'  (explicit-default optional)
        # All other state-bus .get(...) calls fail the guard.

    def test_mutation_violation_detected(self, tmp_path):
        # Materialize a tmp src/neograph/_demo.py with:
        #   def f(state: StateBus): return state.get("missing")
        # Run the same scan against tmp_path; assert it would flag.
```

### 5.4 Equivalence tests update — `tests/hypothesis/test_state_bus_equivalence.py`

Existing Hypothesis tests pin `.get(...)` against legacy `_state_get`. They stay unchanged — `.get(...)` semantics are preserved. Add a parallel property set:

```text
TestGetRequiredProperties:
  @given(state=STATES, present_key=...)
  def test_get_required_present_matches_get(state, present_key):
      # When key is present, .get_required and .get agree on value.

  @given(state=STATES)
  def test_get_required_absent_raises(state):
      # When key is guaranteed-absent, .get_required raises NeographError.

  @given(state=STATES)
  def test_get_required_none_value_returns_none_not_raise(state):
      # explicit_none key bound to None → returns None, does NOT raise.
```

---

## 6. Risks

### 6.1 Three-surface parity

The eight REQUIRED reads (#3, #5, #6, #7, #8, #9, #15, #16, #17) live in helpers shared by all three surfaces (`@node`, declarative `Node`, `ForwardConstruct`). They run inside the LangGraph dispatch path, which is identical across surfaces. Risk of asymmetric breakage: **low**.

The one place to double-check is `_extract_fan_in_dict` (#6, #7). The validator `_validate_node_chain` already requires fan-in upstreams to exist at assembly time. But there's a gap: validator runs on the IR; runtime dispatch passes a freshly-constructed state model. If any surface has a code path that constructs state without all fan-in fields populated for the dispatched node (e.g., a branch arm where the upstream's branch didn't fire), the new raise will surface that. **This is desirable** — it's the §7 intent — but it may need an exception path for "branch did not produce upstream" (currently produces silent `None`). Treat as a downstream finding from running the test suite, not a blocker for izo1.

### 6.2 LangGraph internals

LangGraph dispatches into the wrapper functions in `_execute.py`, `_oracle.py`, `_subconstruct.py`, `_wiring.py`. None of them mutate or pre-populate state in ways the wrappers don't already account for. The `Send(...)` payload semantics are: keys included in `Send` are present in dispatched state; keys not included may or may not be present (depend on reducer behavior).

The risk surface for #15, #16 (EACH_ITEM inside Oracle and Each redirect): the `Send` payload in `_wire_each` line 155 always includes `EACH_ITEM`, and in `_add_each_oracle_fused` line 246 always includes `EACH_ITEM`. So the required-read assumption holds. **Low risk.**

### 6.3 Test exposure (existing tests that may start failing)

Estimating by category:

- **REQUIRED reads on validator-guaranteed fields** (#3, #5, #6, #7, #8, #15, #16, #17): tests that exercise valid pipelines should not start failing. Tests that synthetically construct partial state (likely in `tests/test_coverage_gaps.py`, `tests/modes/test_execution.py::TestStateGet`) might start failing because they bypass validation. Estimate: **3–8 tests in the coverage-gap / direct-helper exercises will need their state setup augmented**.
- **REQUIRED read on `node.context`** (#9): if the validator does NOT today enforce that context fields are produced upstream, every test that exercises a `node.context` declaration with a "synthetic" (validator-bypassing) state may start raising. Estimate: **0–5 tests** depending on whether validator already enforces this. **Open question §8.**
- **Three-surface parity tests**: should still pass; the new test additions in §5 are net-new, not modifications.
- **`tests/hypothesis/test_state_bus_equivalence.py`**: unchanged. The `.get(...)` equivalence semantics are preserved.

Worst-case estimate: **~12 existing tests** may need updates, all by augmenting state setup. None are conceptually wrong; they're masking the very bug §7 says we should catch.

### 6.4 Downstream consumer (piarch)

Piarch pulls neograph from `develop`. If a piarch pipeline relies on silent-None for what is actually a REQUIRED field (e.g., a node declares fan-in inputs that aren't always produced upstream because of a branch), it will start raising `NeographError` on first run. This is **good** — that's a bug — but it may require coordinated cleanup. Risk surface: **moderate**, but contained to one downstream.

---

## 7. Estimated effort

| Phase | Hours |
|---|---|
| Implement `get_required` on Protocol + `_DictStateBus` + `_ModelStateBus` | 0.5 |
| Migrate 8 REQUIRED call sites (mechanical edit + plumb `node_label` through closures in `_oracle.py`) | 1.5 |
| Annotate 16 OPTIONAL call sites with `# StateBus.get optional:` comments | 0.5 |
| Write unit tests + behavioral tests + three-surface parity tests (~15 new tests) | 2.0 |
| Write structural guard `TestStateBusGetUsesRequiredByDefault` + mutation case | 1.5 |
| Extend hypothesis equivalence tests with `get_required` properties | 0.5 |
| Run full test suite + fix-out of 3–12 surprise failures | 2.0 |
| Run examples 01, 01c, 02, 03, 04, 05, 06, 08, 09, 10 (skip 07, 11) | 0.5 |
| Update CHANGELOG / commit hygiene | 0.25 |

**Total: ~9 hours** to land izo1 to a fully passing state, including three-surface parity tests and structural guard. Mid-range estimate assumes fix-out costs in the lower-to-middle band; if the validator turns out to NOT enforce `node.context` (open question §8), add another ~2 hours for that gap.

---

## 8. Open questions for maintainer

### Q1. Should `get_required` raise `NeographError` directly, or a new `StateMissingError(NeographError)` subclass?

§7 names `NeographError` literally. The ticket description says "specific subclass like `StateMissingError`" is acceptable. A subclass is useful only if tests need to distinguish; the test additions in §5 don't strictly need it. **Recommendation**: start with bare `NeographError.build(...)`; add subclass only if a concrete need surfaces.

### Q2. Does `_validate_node_chain` enforce that `node.context` fields are produced upstream?

Audit call site #9 (`_execute.py:46`) and #20 (`_subconstruct.py:105`) both read user-declared context fields. If the validator already requires upstream production of every context field, switching #9 to `get_required` is purely a fail-loud improvement with zero test fallout. If it doesn't, the new raise will surface real construction bugs in existing tests/examples.

**Action needed**: maintainer (or follow-up grep) confirms whether `_validate_node_chain` walks `node.context` and verifies each name maps to a producer. If not, this is a **second gap worth filing as a follow-up beads ticket** (validator must guarantee context-field producers exist).

### Q3. For #20 (context forwarding in subgraph), should `bus.get(ctx_field)` become `get_required(...)` too?

Consistent with #9: if context fields are validator-guaranteed, then yes — the sub-construct's parent must have populated every context field its children declare. But subgraph context forwarding is a "best-effort" pass-through today, and silently skipping unbound context is documented behavior. Two valid positions:

- **(a) Tighten both**: subgraph forwarding requires the parent to have populated every declared context field. Symmetric with #9.
- **(b) Keep #20 optional**: subgraph forwarding is a forwarding helper; missing context is forwarded as `None`, the receiving sub-node's #9 read enforces.

Recommendation: **(b)** — keep #20 OPTIONAL with a comment. The required check happens at the receiving sub-node side via #9. Avoids double-validation and keeps the boundary symmetric with `node_id` (#19) which is conventionally optional.

### Q4. Should `loop_count` reads have a dedicated `bus.get_counter(key) -> int` helper?

Four call sites (#10, #21, #23, #24) all follow the pattern `(state.get(count_field) if ... ) or 0`. A `get_counter` helper would internalize that idiom. **Not in scope for izo1** (it's a future micro-refactor) but worth filing as a tiny follow-up beads ticket if maintainer agrees.

### Q5. The `_apply_skip_when` and `_build_state_update` `state: StateBus | None` union — keep or tighten?

Both functions take `state: StateBus | None = None`. The `None` branch is used by call sites that don't have a state in scope (e.g., synthetic test invocations or programmatic invocations from `Node.run_isolated`). The `or None` guard preserves their behavior. Worth checking whether the `None` branch is still reachable post-Batch-2; if not, tighten the type to `StateBus` and remove the `if state is not None` guards. **Out of scope for izo1.**

### Q6. After landing izo1, should we file a sub-ticket to consolidate the loop-bootstrap reads (#1, #4, #11, #12, #18) behind a dedicated helper?

Five call sites share the same "first iteration, no own output yet" semantic. A `bus.get_loop_self(key) -> list | None` would encapsulate that. Not blocking; file as follow-up if there's appetite for further refactoring after the §7 work lands.

---

## 9. Summary

- **24 StateBus `.get(...)` call sites** across 6 files in `src/neograph/`.
- **8 REQUIRED** (must migrate to `.get_required(...)`).
- **16 OPTIONAL** (keep `.get(...)` with one-line `# StateBus.get optional:` comment).
- **0 AMBIGUOUS** — every site has a clear classification, though three (#9, #20, sub-construct context) hinge on whether the validator already enforces context-field production. **Filed as Q2.**
- Estimated total effort: **~9 hours**.
- **6 open questions** for maintainer (Q1–Q6), of which Q2 is the only one that gates implementation effort.

End of analysis.
