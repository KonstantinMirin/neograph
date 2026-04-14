# Architecture v2 -- Making Review Findings Structurally Impossible

This document defines the architectural principles that eliminate each finding
cluster from the 2026-04-10 code review synthesis. The goal is not "better
tests" or "more linters" but structural impossibility: code that violates
the invariant should not compile, not type-check, or not be expressible in
the first place.

---

## 1. Factory Wrapper Unification

### Findings eliminated

HIGH-01 (Oracle config injection duplication), HIGH-02 (loop router closure
duplication), DRY-01 through DRY-10 (various factory/compiler duplication),
Pattern D (context field extraction), Pattern C (body-as-merge Oracle
registration).

### The problem today

`factory.py` had three non-raw wrappers: `_make_scripted_wrapper`,
`_make_produce_fn`, `_make_tool_fn` (plus `_make_raw_wrapper` which stays
separate as an escape hatch). The three non-raw wrappers shared an
identical preamble:

1. Bind logger
2. Log `node_start`
3. Inject Oracle config (`neo_oracle_gen_id`, `neo_oracle_model`)
4. Call `_extract_input`
5. Call `_apply_skip_when`

And an identical postamble:

6. Call `_build_state_update`
7. Log `node_complete` with elapsed time

The mode-specific logic sits between steps 5 and 6:

- **scripted**: `result = fn(input_data, config)`
- **think**: render input, resolve output model, extract context, call
  `invoke_structured`
- **tool**: render input, resolve output model, create budget tracker,
  extract context, call `invoke_with_tools`, wire tool_log

A fix in the preamble (e.g., new Oracle config key) must be applied in 3
places. Same for any postamble change.

Similarly in `compiler.py`, `_add_loop_back_edge` and `_add_subgraph_loop`
build nearly identical `loop_router` closures -- same counter check, same
exhaust policy, same condition invocation -- differing only in value unwrap.

### Invariant

**There is exactly one execution path through the factory for all node modes.**
Mode-specific behavior is injected, not branched.

### Mechanism: Template Method via a dataclass pipeline

Replace the three non-raw wrapper functions with a single `_execute_node`
function that accepts a `ModeDispatch` protocol:

```python
# factory.py

class NodeInput:
    """Typed container for extracted node input — replaces bare Any."""
    single: Any | None = None       # single-type input
    fan_in: dict[str, Any] | None = None  # dict-form fan-in input

    @property
    def value(self) -> Any | dict[str, Any]:
        """The input value in the shape the mode dispatch expects."""
        return self.fan_in if self.fan_in is not None else self.single


class NodeOutput:
    """Typed container for node output — replaces bare Any."""
    single: Any | None = None       # single-type output
    multi: dict[str, Any] | None = None  # dict-form multi-output

    @property
    def value(self) -> Any | dict[str, Any]:
        return self.multi if self.multi is not None else self.single


class ModeDispatch(Protocol):
    """Mode-specific execution logic, injected into the unified wrapper.

    Input and output are typed containers wrapping Any values (real node
    I/O includes str, int, list — not just BaseModel). The containers
    provide shape discrimination (single vs fan_in/multi) without type
    erasure at the boundary.
    """
    def execute(
        self,
        node: Node,
        input_data: NodeInput,
        config: RunnableConfig,
        context_data: dict[str, Any] | None,
    ) -> NodeOutput:
        """Run mode-specific logic. Returns typed output (before state update)."""
        ...


def _execute_node(node: Node, state: BaseModel, config: RunnableConfig, dispatch: ModeDispatch) -> dict[str, Any]:
    """Single execution path for all non-raw node modes.

    Preamble → dispatch.execute(NodeInput) → NodeOutput → postamble.

    **Dual-path contract**: _apply_skip_when has state-writing side effects.
    If it returns non-None, we return immediately — BEFORE the dispatch and
    BEFORE the postamble's _build_state_update. Any future change to
    _build_state_update must account for this: skip_when's internal state
    write is the ONLY write on the skip path.

    **Input wrapping**: skip_when receives the raw extracted input (dict or
    BaseModel); dispatch.execute receives NodeInput (typed container). This
    is intentional — skip_when predicates operate on the user's data shape.
    """
    field_name = node.name.replace("-", "_")
    node_log = log.bind(node=node.name, mode=node.mode)
    node_log.info("node_start", input_type=_type_name(node.inputs), output_type=_type_name(node.outputs))

    t0 = time.monotonic()

    # Preamble: Oracle config injection
    config = _inject_oracle_config(state, config)

    # Preamble: input extraction (raw form — not yet wrapped in NodeInput)
    raw_input = _extract_input(state, node)

    # Preamble: skip_when — early return with side effects (writes state internally)
    skip_result = _apply_skip_when(node, raw_input, field_name, t0, node_log, state)
    if skip_result is not None:
        return skip_result

    # Preamble: context extraction — only for LLM modes, not scripted
    context_data = _extract_context(state, node) if node.mode != "scripted" else None

    # Wrap raw input in typed container for dispatch
    node_input = NodeInput(fan_in=raw_input) if isinstance(raw_input, dict) and isinstance(node.inputs, dict) else NodeInput(single=raw_input)

    # Mode dispatch
    result = dispatch.execute(node, node_input, config, context_data)

    # Postamble: state update (uses classify_modifiers + match + assert_never)
    update = _build_state_update(node, field_name, result.value, state)

    elapsed = time.monotonic() - t0
    node_log.info("node_complete", duration_s=round(elapsed, 3))
    return update
```

Mode dispatch implementations:

```python
class ScriptedDispatch:
    def __init__(self, fn: Callable): ...
    def execute(self, node, input_data, config, context_data):
        return self.fn(input_data, config)

class ThinkDispatch:
    def execute(self, node, input_data, config, context_data):
        input_data = _render_input(node, input_data)
        output_model, primary_key = _resolve_primary_output(node)
        effective_model = config.get("configurable", {}).get("_oracle_model", node.model)
        result = invoke_structured(...)
        if primary_key is not None and result is not None:
            result = {primary_key: result}
        return result

class ToolDispatch:
    def execute(self, node, input_data, config, context_data):
        input_data = _render_input(node, input_data)
        output_model, primary_key = _resolve_primary_output(node)
        # Budget tracker: tracks tool call counts/costs per tool
        budget_tracker = ToolBudgetTracker(node.tools)
        effective_model = config.get("configurable", {}).get("_oracle_model", node.model)
        # Renderer resolution: node-level renderer takes priority, then global
        renderer = node.renderer if node.renderer is not None else _get_global_renderer()
        # Resolve oracle_gen_type for dict-form outputs with tools
        oracle_gen_type = output_model
        if isinstance(node.outputs, dict) and primary_key is not None:
            oracle_gen_type = node.outputs[primary_key]
        result, tool_interactions = invoke_with_tools(
            model=effective_model,
            rendered_input=input_data,
            output_type=oracle_gen_type,
            tools=node.tools,
            budget_tracker=budget_tracker,
            renderer=renderer,
            context=context_data,
            config=config,
        )
        # Wire tool_log into result dict
        if primary_key is not None and result is not None:
            result_dict = {primary_key: result}
            if isinstance(node.outputs, dict) and "tool_log" in node.outputs:
                result_dict["tool_log"] = tool_interactions
            result = result_dict
        return result
```

**Registration validation asymmetry**: `make_node_fn` validates scripted
function registration early (at graph construction time). A missing
`register_scripted()` call fails immediately. LLM modes (think/agent/act)
cannot be validated early because LLM configuration (`configure_llm()`)
happens at runtime — a missing LLM config fails at execution time inside
`ThinkDispatch.execute` or `ToolDispatch.execute`. This asymmetry is by
design, not a gap.

`make_node_fn` becomes:

```python
def make_node_fn(node: Node) -> Callable:
    if node.raw_fn is not None:
        return _make_raw_wrapper(node)  # raw stays separate: no preamble/postamble
    dispatch = _dispatch_for_mode(node)
    def node_wrapper(state, config):
        return _execute_node(node, state, config, dispatch)
    node_wrapper.__name__ = node.name.replace("-", "_")
    return node_wrapper
```

### What this eliminates

- HIGH-01: Oracle config injection exists in one place (`_inject_oracle_config`),
  called once inside `_execute_node`.
  **DONE**: `_inject_oracle_config` already extracted (neograph-bldf, 42123e6).
- DRY-01 through DRY-10: all preamble/postamble duplication gone.
- Pattern D: context extraction in one place (`_extract_context`).
  **DONE**: `_extract_context` already extracted (neograph-yudg, 42123e6).
- Any future preamble/postamble addition (new modifier config, new
  observability hook) is automatically applied to all modes.
- **Note**: `_build_state_update` in the postamble already uses
  `classify_modifiers` + `match` + `assert_never` (Section 2 mechanism).
  This arrived as part of the factory unification but is technically a
  Section 2 adoption in the factory layer.

### Loop router unification (HIGH-02) -- DONE

**Already implemented** in compiler.py:537 as `_make_loop_router` with
`_node_loop_unwrap` and `_construct_loop_unwrap` callbacks. The design
below matches the existing code:


```python
def _make_loop_router(
    item_name: str,
    field_name: str,
    count_field: str,
    loop: Loop,
    condition: Any,
    exit_name: str,
    reenter_target: str,
    unwrap_fn: Any,  # (state, field_name) -> value
) -> Any:
    """Single loop_router factory for both Node and Construct loops.

    ``exit_name`` and ``reenter_target`` are parameters (not computed
    internally) because Node loops re-enter the same node while Construct
    loops may re-enter a different graph node.
    """

    def loop_router(state: Any) -> str:
        count = getattr(state, count_field, 0)
        if count >= loop.max_iterations:
            if loop.on_exhaust == 'error':
                raise ExecutionError(...)
            return exit_name
        val = unwrap_fn(state, field_name)
        try:
            should_continue = condition(val)
        except (AttributeError, TypeError) as exc:
            raise ExecutionError(...) from exc
        return reenter_target if should_continue else exit_name

    return loop_router
```

Node loop passes an `unwrap_fn(state, field_name)` that reads
`state.{field}`, picks the latest from the append-list, and handles
dict-form (primary key lookup). Construct loop passes one that reads
`state.{field}` and delegates to `_unwrap_loop_value`. One factory, two
unwrap strategies.

---

## 2. Type-Safe Modifier Composition

### Findings eliminated

22 historical modifier composition bugs (Pattern 1 in weak-point-map.md),
neograph-35c3, and future modifier-combo bugs.

### The problem today

`Node.modifiers` is `list[Modifier]`. Any modifier can be appended to any
node. The `__or__` method in `Modifiable` has runtime checks for illegal
combos (Each+Loop, Oracle+Loop), but:

1. The programmatic API can bypass `__or__` entirely by constructing
   `modifiers=[Each(...), Loop(...)]` in the `Node()` constructor.
2. The `__or__` guards are if/elif chains that must be updated for every
   new modifier. Missing a guard = silent bug.
3. `ModifierCombo` enum + `classify_modifiers()` exist as Phase 1 but are
   not yet the enforcement mechanism -- they classify after the fact.

### Invariant

**The set of applied modifiers is always a member of `ModifierCombo`.
Illegal combinations are unrepresentable in the type system.**

### Mechanism: Replace `list[Modifier]` with a validated `ModifierSet`

Phase 1 already has `ModifierCombo` (10 variants). The full end state:

```python
class ModifierSet(BaseModel, frozen=True):
    """Validated, typed modifier configuration.

    Cannot be constructed with an invalid combination -- pydantic
    model_post_init rejects it. Replaces list[Modifier] everywhere.
    """
    each: Each | None = None
    oracle: Oracle | None = None
    loop: Loop | None = None
    operator: Operator | None = None

    @property
    def combo(self) -> ModifierCombo:
        """Classify this set into a ModifierCombo enum value."""
        ...

    def model_post_init(self, __context: Any) -> None:
        # Reject illegal combos at construction time.
        # Each + Loop: rejected
        if self.each is not None and self.loop is not None:
            raise ConstructError("Cannot combine Each and Loop...")
        # Oracle + Loop: rejected
        if self.oracle is not None and self.loop is not None:
            raise ConstructError("Cannot combine Oracle and Loop...")
        # Duplicate detection is impossible -- each slot is typed, not a list.

    def with_modifier(self, mod: Modifier) -> "ModifierSet":
        """Return a new ModifierSet with the given modifier added."""
        if isinstance(mod, Each):
            if self.each is not None:
                raise ConstructError("Duplicate Each...")
            return self.model_copy(update={"each": mod})
        ...
```

`Node` and `Construct` carry `modifier_set: ModifierSet` instead of
`modifiers: list[Modifier]`. `__or__` delegates to
`ModifierSet.with_modifier()`.

Dispatch sites use `match modifier_set.combo:` (structural pattern
matching) instead of `has_modifier()` chains.

**`_BranchNode` compat note**: `_BranchNode` (in `forward.py`) is a
duck-typed sentinel that currently satisfies Construct validation by
carrying `modifiers = []` and stub `has_modifier()` / `get_modifier()`
methods. When `Node.modifiers` migrates to `ModifierSet`, `_BranchNode`
must carry `modifier_set = ModifierSet()` (the empty default) instead of
a bare list, and its stub methods must match the `ModifierSet` accessor
API. Otherwise any code that does `item.modifier_set.combo` on a
construct's node list will fail on branch sentinels.

**HARD REQUIREMENT**: Python's `match` does NOT enforce exhaustiveness at
the type level. A new `ModifierCombo` variant will silently fall through
if no `case` handles it. Every `match combo:` block MUST end with:

```python
from typing import assert_never  # Python 3.11+

match ms.combo:
    case ModifierCombo.EACH_ORACLE:
        ...
    case ModifierCombo.ORACLE:
        ...
    # ... all other cases ...
    case _ as unreachable:
        assert_never(unreachable)
```

`assert_never` makes the exhaustiveness check a type-checker obligation
(mypy/pyright will error if a variant is unhandled) AND a runtime guard
(raises `AssertionError` if reached). This is not optional.

### Elimination guarantees

- **Duplicate modifiers**: impossible. Each slot is a single optional value,
  not a list.
- **Illegal combos (Each+Loop, Oracle+Loop)**: rejected in
  `ModifierSet.model_post_init`.
- **Bypassing `__or__`**: the programmatic `Node(modifier_set=...)` path
  goes through the same `model_post_init` validation.
- **New modifier added without dispatch**: `ModifierCombo` gains a variant;
  every `match combo:` in compiler/factory/state is incomplete until the
  new case is handled.

**Defense-in-depth (intentional triple validation)**: Illegal combos are
checked in three places: (1) `ModifierSet.model_post_init` at construction
time, (2) `Modifiable.__or__` via `with_modifier()` at pipe time, (3)
`_validate_node_chain` in `_construct_validation.py` at assembly time.
This redundancy is *intentional*. Each layer catches a different entry
path: `model_post_init` catches programmatic construction,
`with_modifier()` catches `|` operator use, and `_validate_node_chain`
catches any node that somehow bypassed both (e.g., deserialized from
config). The cost of the redundant checks is negligible (runs once at
assembly, not per invocation). Do not remove any of the three layers.

### Accessor migration

Current code:

```python
oracle = node.get_modifier(Oracle)
each = node.get_modifier(Each)
if oracle and each: ...
```

v2 code:

```python
ms = node.modifier_set
match ms.combo:
    case ModifierCombo.EACH_ORACLE:
        _add_each_oracle_fused(graph, node, ms.each, ms.oracle, ...)
    case ModifierCombo.ORACLE:
        _add_oracle_nodes(graph, node, ms.oracle, ...)
    ...
    case _ as unreachable:
        assert_never(unreachable)
```

No more `has_modifier()` → no more missing-modifier bugs.

**Performance note on `.modifiers` bridge**: The backward-compat
`modifiers` property on `Modifiable` returns `self.modifier_set.to_list()`,
creating a new list on every access. All hot paths (compiler, factory,
state) use `modifier_set.combo` directly (O(1) property). The `.modifiers`
bridge exists only for backward-compat read access. Do not use it in
performance-critical loops.

### `.map()` implementation

`.map(source, *, key: str)` delegates to `__or__`, which calls
`modifier_set.with_modifier()` internally. `key` is required (no default):

```python
def map(self, source: Any, *, key: str) -> Self:
    # source can be a string ("upstream.field") or a lambda.
    # Lambda form: introspected via _PathRecorder proxy at definition time.
    if isinstance(source, str):
        over = source
    elif callable(source):
        recorder = _PathRecorder()
        result = source(recorder)  # records attribute-access chain
        over = ".".join(result._neo_path)
    else:
        raise TypeError(...)
    return self | Each(over=over, key=key)
    # __or__ calls self.modifier_set.with_modifier(Each(...))
    # then model_copy(update={"modifier_set": new_ms})
```

The `_PathRecorder` proxy enables refactor-safe lambda introspection:
`verify.map(lambda s: s.make_clusters.groups, key="label")` — Pyright
catches typos in the attribute chain. The recorder rejects indexing,
arithmetic, and underscore-prefixed attributes with clear `TypeError`s.

This was kept as `__or__` delegation rather than calling `with_each()`
directly — `__or__` already handles sidecar re-registration and
validation, so `.map()` reuses that path.

### Sidecar Interaction

`_node_sidecar` and `_param_resolutions` are keyed by `id(Node)`. Every
`model_copy` (from `__or__`, `.map()`, or `ModifierSet.with_modifier()`)
produces a new Node instance with a new `id()`. The sidecar entries for
the old instance are NOT inherited. The caller MUST re-register sidecars
on the copy:

```python
n = n.model_copy(update={"modifier_set": new_ms})
_register_sidecar(n, fn, param_names, fan_out_param)
_register_param_resolutions(n, param_res)
```

This invariant is unchanged from the current `list[Modifier]` regime --
`model_copy` already triggers re-registration in every `__or__` call
path. ModifierSet changes the shape of what is copied but not the
`id()`-invalidation behavior. Every code path that produces a modified
Node must re-register or the sidecar lookup will silently return `None`.

---

## 3. Unified DI Resolution

### Findings eliminated

12 historical DI bugs (Pattern 2 in weak-point-map.md), neograph-26ih,
Pattern B (lint.py 4x duplication), HIGH-06 (bare except in
`_resolve_di_value`).

### The problem today

DI resolution happens in four places:

1. `_resolve_di_value` in `decorators.py` -- the primary resolver for @node
   and @merge_fn
2. `_extract_input` in `factory.py` -- calls `_unwrap_loop_value` and
   `_unwrap_each_dict` inline
3. `_resolve_merge_args` in `decorators.py` -- calls `_unwrap_loop_value`
   for from_state params
4. `lint.py` `_walk` -- 4 near-identical blocks for scalar/model x
   node/merge_fn DI binding checks

Each resolution path handles Loop unwrap independently. When Loop behavior
changes (e.g., new unwrap rule), all four paths must be updated.

### Invariant

**There is exactly one function that resolves a DI parameter value from
runtime state + config. All callers go through it.**

### Mechanism: The Bouncer pattern

Two distinct resolution scopes:

1. **`DIResolver`** -- config-based resolution only. Handles `from_input`,
   `from_config`, `from_input_model`, `from_config_model`, `constant`.
   All read from `config['configurable']`. This is what `_resolve_di_value`
   does today.
2. **Upstream resolution** -- stays in `_extract_input` in the factory.
   Reads from graph state by field name. NOT part of DIResolver. The
   scripted shim interleaves DI-resolved values with upstream-resolved
   values at call time.
3. **`from_state`** -- merge_fn-only. Resolved from graph state by param
   name, not from config. Handled by `_resolve_merge_args` which calls
   DIResolver for config-based params and does direct state lookup for
   `from_state` params.

`DIKind` enum has 6 values (no `upstream`):

```python
class DIKind(Enum):
    FROM_INPUT = "from_input"
    FROM_CONFIG = "from_config"
    FROM_INPUT_MODEL = "from_input_model"
    FROM_CONFIG_MODEL = "from_config_model"
    FROM_STATE = "from_state"  # merge_fn only — resolved from graph state by param name, not from config
    CONSTANT = "constant"
```

```python
# di.py (extracted from decorators.py)

@dataclass
class DIBinding:
    """A fully resolved DI parameter binding."""
    name: str
    kind: DIKind  # 6 kinds — upstream is NOT a DI kind
    inner_type: type
    required: bool
    payload: Any  # constant value, (model_cls, required), or expected type

    def resolve(self, config: Any, *, state: Any = None) -> Any:
        """The ONE resolution path for DI parameters.

        For FROM_STATE (merge_fn only), `state` must be provided.
        For all other kinds, reads from config['configurable'].
        """
        ...
```

**Deferred: modifier-aware DI unwrap.** The original design showed a
`DIResolver` wrapper class with `modifier_set` and `_apply_modifier_unwrap`
that would centralize Loop/Each unwrap in the DI layer. This is NOT
implemented. Modifier-aware unwrap currently stays in `_extract_input`
(factory layer) and `_resolve_merge_args` (decorators layer). All call
sites use `DIBinding.resolve()` directly — no `DIResolver` intermediary.
This is acceptable because the unwrap helpers (`_unwrap_loop_value`,
`_unwrap_each_dict`) are already centralized in `di.py` as single-source
functions; they just aren't called from inside a resolver object.

The lint module becomes:

```python
# lint.py

def _check_binding(
    node_label: str,
    param: str,
    kind: str,
    payload: Any,
    config: dict | None,
    issues: list[LintIssue],
):
    """One function for all DI lint checks.

    `node_label` is pre-formatted by the caller — node and merge_fn paths
    use different naming conventions (e.g., "node 'X'" vs "merge_fn 'Y' on Oracle 'Z'"),
    so the caller supplies the label, not this helper.
    """
    if kind in (DIKind.FROM_INPUT, DIKind.FROM_CONFIG):
        if config is not None and param not in config:
            issues.append(LintIssue(f"{node_label}: DI param '{param}' not in config"))
        elif config is None:
            issues.append(LintIssue(f"{node_label}: DI param '{param}' requires config"))
    elif kind in (DIKind.FROM_INPUT_MODEL, DIKind.FROM_CONFIG_MODEL):
        for fname in payload.model_fields:
            if config is not None and fname not in config:
                issues.append(LintIssue(f"{node_label}: model field '{fname}' not in config"))
            elif config is None:
                issues.append(LintIssue(f"{node_label}: model field '{fname}' requires config"))
```

Four copies in current `lint.py` collapse to one call to `_check_binding`
per binding. The same `DIBinding` objects that lint checks are the same ones
that runtime resolves -- no divergence possible.

### HIGH-06 fix (bare except)

Inside `DIBinding.resolve`, the bundled model construction catch becomes:

```python
from pydantic import ValidationError  # must be imported explicitly

try:
    return model_cls(**field_values)
except (ValidationError, TypeError, ValueError) as exc:
    if self.required:
        raise ExecutionError(...) from exc
    log.warning(...)
    return None
```

Note: `ValidationError` is from `pydantic`, not a stdlib type. The import
must be present in `di.py` (or wherever `DIBinding.resolve` lands).

Specific exceptions, not bare `except Exception`. This is enforced by the
single resolution path -- there is no second place where someone could
add a lazier catch.

---

## 4. Test Behavioral Discipline

### Findings eliminated

HIGH-03 (empty test classes), HIGH-04 (forgiving scripted functions mask
wiring bugs), HIGH-05 (weak assertions), Pattern E (manual monkeypatching).

### The problem today

Coverage-driven testing produced tests that execute code paths without
validating behavior. The obligation-testing framework exists but some
tests predate it and use patterns like:

```python
assert result is not None  # HIGH-05: asserts existence, not correctness
```

```python
# HIGH-04: test function accepts any input shape
def my_scripted(input_data, config):
    if isinstance(input_data, dict):
        return input_data.get("claims", Claims(...))
    return Claims(...)
```

### Invariant

**Every test assertion references a specific expected value or structural
property derived from the test setup. No `is not None`, no `isinstance`
without a corresponding value check.**

### Mechanism: Assertion audit rule + strict test fixtures

This is not a type-system fix -- it is a process and tooling fix.

**Rule 1: The assertion must be traceable to the setup.**

Every `assert` in a test must reference either:
- A literal value constructed in the test setup
- A structural property (field name, list length, dict keys) that is a
  direct consequence of the test input

Bad: `assert result is not None`
Good: `assert result.score == 0.85`
Good: `assert len(result.claims) == 3`
Good: `assert set(result.keys()) == {"alpha", "beta"}`

**Rule 2: Test scripted functions must be strict receivers.**

Every scripted function used in tests must assert the type of its input
and fail on unexpected shapes:

```python
def strict_transform(input_data: dict, config: RunnableConfig) -> Result:
    assert isinstance(input_data, dict), f"Expected dict, got {type(input_data)}"
    assert "claims" in input_data, f"Missing 'claims' key, got {input_data.keys()}"
    claims = input_data["claims"]
    assert isinstance(claims, Claims), f"Expected Claims, got {type(claims)}"
    return Result(summary=f"Processed {len(claims.items)} claims")
```

**Rule 3: No test class with `pass` body.**

A `ruff` rule or a custom pytest plugin that scans for test classes
containing only `pass` or `...`:

```python
# conftest.py or pytest plugin
def pytest_collect_modifyitems(items):
    """Reject empty test classes at collection time."""
    ...
```

**Enforcement**: this is process discipline, not a type-system guarantee.
Unlike Principles 1-3, violations are not structurally impossible -- they
require active enforcement:

1. **Obligation workflow**: new tests MUST go through `/test-obligations`,
   which produces per-obligation tests with specific assertions.
2. **CI gate (concrete)**: a pytest plugin that flags tests whose only
   assertions are `assert X is not None` or `assert isinstance(X, T)`
   with no subsequent value check. Fires at collection time as a warning;
   promoted to error after legacy cleanup.
3. **Review gate**: the obligation analysis workflow catches weak assertions
   during code review. This is the primary enforcement today.
4. **Legacy backlog**: existing tests that fail the assertion audit are
   tracked in beads. Not blocking, but must not grow.

---

## 5. Error Builder Pattern

### Findings eliminated

Pattern A (inconsistent error message formatting across 5 files).

### The problem today

Error messages in neograph use at least 4 different styles:

1. `_construct_validation.py`: multi-line with `\n`, producer summaries,
   `.map()` hints, `_location_suffix()` trailing.
2. `compiler.py`: single-line f-strings, no location.
3. `factory.py`: single-line f-strings with `f"Node '{name}'"` prefix.
4. `modifiers.py`: single-line from `model_post_init`.

There is no shared vocabulary for "which node", "what was expected", "what
was found", "what to do about it".

### Invariant

**Every neograph error message is produced by the error builder. The builder
enforces a consistent structure: what failed, what was expected, what was
found, and what to do.**

### Mechanism: `NeographError.build()` class method

```python
# errors.py

class NeographError(Exception):
    """Base for all neograph-originated errors."""

    @classmethod
    def build(
        cls,
        what: str,
        *,
        expected: str | None = None,
        found: str | None = None,
        hint: str | None = None,
        location: str | None = None,
        node: str | None = None,
        construct: str | None = None,
    ) -> "NeographError":
        """Structured error builder. All neograph errors go through here.

        Format:
            [Node 'X' in Construct 'Y'] what
              expected: ...
              found: ...
              hint: ...
              at file.py:42
        """
        parts = []
        if node and construct:
            parts.append(f"[Node '{node}' in '{construct}']")
        elif node:
            parts.append(f"[Node '{node}']")
        elif construct:
            parts.append(f"[Construct '{construct}']")
        parts.append(what)

        msg = " ".join(parts)
        if expected:
            msg += f"\n  expected: {expected}"
        if found:
            msg += f"\n  found: {found}"
        if hint:
            msg += f"\n  hint: {hint}"
        if location:
            msg += f"\n  at {location}"

        return cls(msg)
```

Each error subclass inherits the builder. Note: `ConstructError` inherits
from both `NeographError` and `ValueError`; `.build()` returns `cls(msg)`
which calls `__init__(msg)` and works for both. `ExecutionError` has a
custom `__init__` accepting an optional `validation_errors` kwarg --
`.build()` handles this via an extra parameter:

```python
# Standard case -- ConstructError, NeographError
raise ConstructError.build(
    "output type not compatible with reentry input type",
    node=node.name,
    expected=_fmt_type(input_type),
    found=_fmt_type(output_type),
    hint="the loop's last node output must match the reentry node's input type",
    location=_source_location(),
)

# ExecutionError case -- accepts validation_errors kwarg
raise ExecutionError.build(
    "DI resolution failed",
    node=node.name,
    found="field X missing from config",
    validation_errors="field X missing",  # passed to ExecutionError.__init__
)
```

### What this eliminates

- Inconsistent message structure: the builder enforces it.
- Missing location hints: `location=` is always available; callers pass
  it when they have it. Location is best-effort: for `@node`, use
  `_get_node_source()` (inspects the decorated function). For declarative
  `Node()`/`Construct()`, use `_caller_location()` (already in
  `_construct_validation.py`, walks the call stack). Location may be
  absent for the programmatic API (runtime-constructed nodes have no
  source file).
- Missing "what to do" guidance: `hint=` is a named parameter that shows
  up in code review if omitted.
- Style drift across files: all files call `.build()`, not f-strings.

### Adoption status

**Partially adopted.** `NeographError.build()` and `ExecutionError.build()`
are implemented in `errors.py`. Adopted in `_construct_validation.py`
(~5 sites use `ConstructError.build(...)` with `expected=`, `found=`,
`hint=`, `location=_source_location()`).

**NOT yet adopted** in `compiler.py`, `factory.py`, `decorators.py`, or
`modifiers.py` (~57 remaining sites still use f-string error construction).
Tracked as neograph-mt2y.

### `_source_location()` limitation

`_source_location()` walks the call stack past neograph/pydantic frames
to find the user's call site. This works for `_construct_validation.py`
(assembly-time errors where the user's `Construct(...)` or `@node` call
is on the stack). It does NOT work for `compiler.py` or `factory.py`
errors, which fire from graph construction or execution contexts where
the user's call site is not on the stack. For those files, location
information would need a different strategy (e.g., storing source info
on the Node at creation time).

---

---

## Architectural Boundary: Execution vs Wiring

The architecture deliberately separates two concerns:

**Factory layer** (Principles 1-3): How nodes EXECUTE. Unified via
ModeDispatch. Covers preamble, DI resolution, skip_when, mode-specific
logic, state update. Every node — regardless of which API surface created
it — goes through this single path.

**Compiler layer**: How nodes are WIRED. Topology decisions: sequential
edges, conditional edges (branches), Send() fan-out, barriers, Loop
back-edges. This layer maps directly to LangGraph's API.

ForwardConstruct's `if`/`for` control flow compiles to **conditional
edges** — these are wiring decisions, not execution logic. The branch
router is a LangGraph routing function, not a node. The nodes INSIDE
branches go through the unified factory path like any other node.

This is intentional. The clean 1:1 mapping to LangGraph:
- `if` → `add_conditional_edges()`
- `for` / Each → `Send()` per item
- Loop → conditional back-edge

...means `describe_graph()` shows the real topology. Adding synthetic
"decision nodes" to force unification would pollute the graph with nodes
that do no real work, break the mental model ("my `if` IS a conditional
edge"), and add overhead for zero user benefit.

**The rule**: if it decides WHAT runs → compiler layer. If it decides
HOW something runs → factory layer. Branch routing is "what runs."
Node execution is "how it runs."

---

## Scope of `_execute_node` (what it covers and what it doesn't)

Covered by the unified execution path:
- `_make_scripted_wrapper` → `ScriptedDispatch`
- `_make_produce_fn` → `ThinkDispatch`
- `_make_tool_fn` → `ToolDispatch`

Explicitly outside:
- `_make_raw_wrapper` — escape hatch, no preamble contract (has logging only)
- `make_subgraph_fn` — composition wrapper, runs a compiled sub-graph
- `make_oracle_redirect_fn` / `make_eachoracle_redirect_fn` — composition
  decorators that wrap the result of `make_node_fn`, not replacements for it
- Branch routers — topology, not execution (compiler layer)

`make_subgraph_fn` is a separate concern (sub-graph invocation) that shares
SOME helpers (`_inject_oracle_config`, `_unwrap_loop_value`) but has its own
input/output extraction logic specific to sub-graph boundaries. It should be
documented as a separate path, not forced into ModeDispatch.

---

## Summary: Invariant → Mechanism → Findings Matrix

| # | Invariant | Mechanism | Status | Findings eliminated |
|---|-----------|-----------|--------|---------------------|
| 1 | One execution path for all modes | `_execute_node` + `ModeDispatch` protocol | **DONE** | HIGH-01, HIGH-02, DRY-01..10, Pattern C, D |
| 2 | Modifier set is always a valid ModifierCombo | `ModifierSet` model with validated slots | **DONE** | 22 historical, neograph-35c3, future combos |
| 3 | One DI resolution function | `DIBinding.resolve()` (single path) | **DONE** | 12 historical, neograph-26ih, Pattern B, HIGH-06 |
| 4 | Every assertion traces to test setup | Obligation workflow + review gate + CI plugin | **PARTIAL** (neograph-nesk) | HIGH-03, HIGH-04, HIGH-05, Pattern E |
| 5 | Every error through the builder | `NeographError.build()` class method | **PARTIAL** (~8%, neograph-mt2y) | Pattern A |
