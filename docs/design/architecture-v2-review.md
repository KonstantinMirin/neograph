# Architecture v2 Review -- Gap Analysis

Reviewed against: `src/neograph/factory.py`, `compiler.py`, `decorators.py`,
`modifiers.py`, `node.py`, `construct.py`, `forward.py`, `state.py`,
`_construct_validation.py`, `lint.py`, and the test suite.

---

## 1. Confirmed Sound

These aspects of the design check out against the actual code:

**1a. Preamble/postamble duplication is real and exactly as described.**
`_make_scripted_wrapper`, `_make_produce_fn`, and `_make_tool_fn` all share the
same 7-step sequence (bind logger, log start, inject oracle config, extract
input, apply skip_when, ... build state update, log complete). The design's
`_execute_node` template correctly captures this structure.

**1b. Loop router unification is correct and already partially done.** The
codebase already has `_make_loop_router` (compiler.py:537) with separate
`_node_loop_unwrap` and `_construct_loop_unwrap` callbacks. The design's
proposal matches what was already implemented -- this section of the design
is describing existing code, not a proposal. The design should acknowledge
this is done rather than framing it as future work.

**1c. The lint.py 4x duplication is real.** Lines 69--127 handle
scalar/model x node bindings, then lines 137--196 repeat nearly identical
logic for merge_fn bindings. The `DIBinding.resolve()` + `_check_binding()`
approach would genuinely collapse this.

**1d. ModifierCombo + classify_modifiers already exist.** The design proposes
`ModifierSet` as the enforcement layer on top of `ModifierCombo`. The enum
and classifier (modifiers.py:19--89) are already in place, so Phase 1 of
the modifier work is done. The design's Phase 2 (`ModifierSet` replacing
`list[Modifier]`) is the actual new work.

**1e. The `except Exception` in `_resolve_di_value` is real (HIGH-06).**
decorators.py:371 catches bare `Exception` on `model_cls(**field_values)`.
The design's proposal to narrow to `(ValidationError, TypeError, ValueError)`
is correct and safe.

---

## 2. Gaps

### 2a. ModeDispatch does not cover the @node scripted shim path

The design proposes `ScriptedDispatch.execute(node, input_data, config,
context_data)` where `self.fn(input_data, config)` is the mode-specific call.
This matches `_make_scripted_wrapper`'s dispatch: `fn(input_data, config)`.

But there is a second scripted path: `_register_node_scripted` in
decorators.py:1555. This builds a shim closure that receives
`(input_data, config)` from the factory and interleaves DI-resolved args
with upstream-resolved args before calling the user function with positional
args. The shim is registered via `register_scripted` and then dispatched
through the same `_make_scripted_wrapper`.

This is not a gap in the dispatch path itself (the shim IS the `fn` that
ScriptedDispatch would call), but the design does not mention that the
`input_data` flowing into `ScriptedDispatch.fn` is consumed differently
depending on whether the node was created via `@node` (shim does
positional-arg mapping) vs `Node.scripted()` (user fn receives raw
`(input_data, config)`). The `NodeInput` typed container must be compatible
with both consumption patterns. Specifically:

- The shim calls `input_data.get(lookup_key)` -- it expects a dict.
- A plain `Node.scripted()` user fn receives whatever `_extract_input`
  returns, which could be `None`, a `BaseModel`, a `dict`, or a raw value.

If `NodeInput.value` returns a `NodeInput` instance instead of the raw
value, the shim's `.get()` call breaks.

**Recommendation**: `ScriptedDispatch.execute` must return the raw
`_extract_input` result to the fn, not a `NodeInput` wrapper. Or `NodeInput`
must implement `__getitem__`/`.get()` to be dict-compatible.

### 2b. NodeInput/NodeOutput type constraints are too narrow

The design declares:

```python
class NodeInput:
    single: BaseModel | None = None
    fan_in: dict[str, BaseModel] | None = None
```

But `_extract_input` (factory.py:534) returns:

1. `None` -- when `node.inputs is None`
2. A bare `BaseModel` instance -- single-type match (line 623)
3. A `dict[str, Any]` -- fan-in dict (line 615), where values can be
   `BaseModel`, `None`, `list`, `dict`, `str`, `int`, or any type
4. A raw value from `neo_each_item` -- could be any type (line 592)
5. A raw value from Loop re-entry (line 587) -- could be a `BaseModel` or
   a dict with mixed value types

The design's `BaseModel` constraint is wrong. Fan-in dict values are not
necessarily `BaseModel` -- they can be `str`, `int`, `list[X]`, `dict[str, X]`,
or any type the upstream node produces. The `neo_each_item` value can be
any element type from a collection.

Similarly for `NodeOutput`:
```python
class NodeOutput:
    single: BaseModel | None = None
    multi: dict[str, BaseModel] | None = None
```

Node outputs can be any type (str, int, list, dict), not just BaseModel.
`Node.outputs` is typed as `Any` (node.py:48).

**Recommendation**: Use `Any` for value types in NodeInput/NodeOutput, or
don't use typed containers at all. The type safety gain from wrapping
`Any` in a dataclass with `Any` fields is minimal.

### 2c. ForwardConstruct execution paths are entirely unaddressed

The design focuses on factory.py and compiler.py but never mentions
ForwardConstruct. ForwardConstruct introduces:

1. **`_BranchNode` sentinel** -- a non-Node object in the node list that
   the compiler dispatches via `_add_branch_to_graph`. It has `self.modifiers = []`
   (forward.py:354) set as a plain list on an ordinary object, not a
   Pydantic field. The `ModifierSet` migration must handle this sentinel
   or `_BranchNode` will break.

2. **`_LoopCall`** -- builds sub-constructs with Loop modifiers during
   forward() tracing. These sub-constructs go through `_add_subgraph_loop`
   in the compiler. The design's unified `_execute_node` does not
   participate here -- subgraph execution is entirely separate.

3. **Branch arm nodes** -- nodes that appear only in the true or false arm
   of a branch. These are added to the graph inside `_add_branch_to_graph`,
   not in the main compile loop. The design's "single execution path"
   claim does not account for these nodes.

The design should explicitly state that ForwardConstruct's
`_BranchNode`-specific paths are out of scope, or explain how the unified
path handles them.

### 2d. Redirect wrapper functions are not part of the unified path

The design's `_execute_node` replaces `_make_scripted_wrapper`,
`_make_produce_fn`, and `_make_tool_fn`. But the factory also has five
wrapper-like functions that intercept or modify node output:

- `make_oracle_redirect_fn` (factory.py:632)
- `make_eachoracle_redirect_fn` (factory.py:658)
- `make_each_redirect_fn` (factory.py:929)
- `make_oracle_merge_fn` (factory.py:774)
- `make_subgraph_fn` (factory.py:826)

These functions wrap the output of `make_node_fn`. Under the design, they
would wrap the closure returned by `_execute_node`. This is fine
structurally, but the design never mentions them. Since these wrappers also
contain duplicated patterns (oracle config injection in `make_subgraph_fn`
at line 886-889, state extraction logic duplicated from `_extract_input`
at lines 853-868), the design should address whether they are brought
into the unified path or remain as output decorators.

`make_subgraph_fn` in particular is a significant execution path with its
own input extraction, oracle config forwarding, context forwarding, and
loop counter management -- none of which flow through `_execute_node`.

### 2e. DIResolver design conflates upstream resolution with DI resolution

The design proposes `DIKind` enum with 7 values including `upstream`.
But upstream parameters are NOT resolved by DI -- they are resolved by
`_extract_input` in the factory, which reads from graph state by field
name. The `_register_node_scripted` shim (decorators.py:1599-1621) shows
the split clearly:

```python
for pname in param_names:
    resolution = param_res.get(pname)
    if resolution is not None:
        # DI resolution
        args.append(_resolve_di_value(kind, payload, pname, config))
    else:
        # Upstream resolution -- from input_data dict
        args.append(input_data.get(lookup_key) ...)
```

The design's `DIResolver.resolve_all(config, state)` would need to handle
both DI params (from config) and upstream params (from the input_data dict
that `_extract_input` already computed). But `_extract_input` reads from
raw graph state, not from a DIBinding. Mixing these two resolution mechanisms
in one resolver creates confusion about what `state` means: is it the raw
LangGraph state, or the already-extracted input_data dict?

**Recommendation**: Keep upstream resolution separate from DI resolution.
The design's `DIResolver` should only handle the 6 non-upstream DI kinds.
The interleaving happens in the scripted shim, which is already correct.

### 2f. `from_state` DI kind is merge_fn-only, not general

The design's `DIKind` enum lists `from_state` as one of 7 kinds. But
`from_state` only exists in `@merge_fn` parameter resolution
(decorators.py:1032-1034). It reads from graph state by attribute name
and applies Loop unwrap. It is never used by `@node` parameters.

The design does not distinguish this. If `DIResolver` tries to resolve
`from_state` params using the same `config['configurable']` lookup as
`from_input`/`from_config`, it will silently fail. The resolver needs
access to graph state (the raw Pydantic model), not just config.

The current `_resolve_merge_args` (decorators.py:405-425) handles this
correctly by receiving `state` as a separate argument. The design's
`DIBinding.resolve(config, state)` signature does accept `state`, but
the doc text focuses on config-based resolution and may mislead
implementers.

### 2g. No mention of `_apply_skip_when` returning a state update dict

The design shows `_apply_skip_when` as a preamble step that returns early
with a `skip_result`. But `_apply_skip_when` (factory.py:162-208) has
significant logic:

- It calls `_build_state_update` when `skip_value` is set (producing
  a full state update dict with modifier wrapping)
- It increments the Loop counter even when no skip_value is set
- It unwraps single-key dicts to pass a typed value to the predicate

This is more than a "preamble check" -- it is a conditional execution
path with state-writing side effects. The design's `_execute_node`
pseudocode shows `skip_result = _apply_skip_when(...)` with an early
return, which is correct, but the design should note that this function
depends on `_build_state_update` and Loop modifier state, which ties
it to the postamble.

---

## 3. Contradictions

### 3a. Design says "3 dispatches" but code has 4 non-raw wrapper functions

The design says: "Replace the four wrapper functions with a single
`_execute_node` function" and proposes 3 dispatches: Scripted, Think, Tool.

The code has `_make_scripted_wrapper`, `_make_produce_fn`, and
`_make_tool_fn` -- that is 3 non-raw wrappers, not 4. The design counts
`_make_raw_wrapper` as a fourth, then says "raw stays separate: no
preamble/postamble." This is consistent, but the "four wrappers" framing
in the problem statement is misleading since only 3 are being unified.

### 3b. Design's ScriptedDispatch signature vs actual scripted fn signature

The design shows:
```python
class ScriptedDispatch:
    def execute(self, node, input_data, config, context_data):
        return self.fn(input_data, config)
```

But `context_data` is never passed to scripted functions. Scripted functions
receive `(input_data, config)` -- context is only used by LLM modes. The
dispatch signature forces ScriptedDispatch to accept and ignore
`context_data`, which is a code smell. The protocol should make
`context_data` optional or move context extraction into the LLM dispatches
where it is actually used.

### 3c. Design claims `_extract_context` is duplicated ("Pattern D")

The design says context extraction is duplicated across wrappers. In
reality, `_extract_context` (factory.py:97-108) is already a single
extracted function called from `_make_produce_fn` (line 414) and
`_make_tool_fn` (line 479). It is called in 2 places, not 3 or 4.
`_make_scripted_wrapper` does not call it. This is correct duplication
(2 LLM modes share one helper), not the "Pattern D" the design implies.

### 3d. ModifierSet.model_post_init vs existing _validate_node_chain checks

The design says:
> "Bypassing `__or__`: the programmatic `Node(modifier_set=...)` path goes
> through the same `model_post_init` validation."

But the codebase already has belt-and-suspenders validation in
`_validate_node_chain` (_construct_validation.py:181-202) that catches
`Each+Loop` and `Oracle+Loop` on the `modifiers=[]` programmatic path.
The test fixture `mod_oracle_loop_programmatic.py` confirms this works.

The design does not mention removing or replacing this validation. With
`ModifierSet`, the `model_post_init` check fires at Node construction time,
and `_validate_node_chain` fires at Construct assembly time. Both would
reject the same combinations. The design should state whether the
`_validate_node_chain` checks are removed (since ModifierSet makes them
redundant) or kept as defense-in-depth.

---

## 4. Risks

### 4a. `modifiers: list[Modifier]` is a public API surface

The field `Node.modifiers` (node.py:93) and `Construct.modifiers`
(construct.py:79) are `list[Modifier]`. Replacing with `modifier_set:
ModifierSet` changes:

1. The field name (migration from `.modifiers` to `.modifier_set`)
2. The field type (list to ModifierSet)
3. All programmatic construction: `Node(..., modifiers=[Oracle(), Each()])`
   becomes `Node(..., modifier_set=ModifierSet(oracle=..., each=...))`

**One test directly asserts on `.modifiers`**: test_modifiers.py:229
(`assert via_map.modifiers == via_pipe.modifiers`). This will break.

**22 uses of `has_modifier()` and `get_modifier()` in src/**: These use
the Modifiable mixin methods. The design proposes replacing these with
`match ms.combo:` but does not address whether `has_modifier` /
`get_modifier` survive as compatibility methods on ModifierSet.

If they are removed, 10+ call sites in compiler.py, state.py, and
_construct_validation.py must be rewritten simultaneously.

**`_BranchNode.modifiers = []`** (forward.py:354): This is a plain list
assignment on a non-Pydantic object. ModifierSet migration must handle
this sentinel specially.

### 4b. `modifiers=[]` in fixtures and examples

The check_fixtures `should_fail/mod_oracle_loop_programmatic.py` and
`should_fail/mod_each_loop_programmatic.py` use `modifiers=[Oracle(), Loop()]`
syntax. These must be rewritten to use `modifier_set=ModifierSet(...)`.

### 4c. Compile log uses `n.modifiers` as a list

compiler.py:85-87:
```python
modifiers={n.name: [type(m).__name__ for m in n.modifiers]
           for n in construct.nodes
           if isinstance(n, Node) and n.modifiers}
```

This iterates `.modifiers` as a list. With ModifierSet, this needs
to be rewritten to iterate the non-None slots.

### 4d. Performance overhead of protocol dispatch

The current path is: `make_node_fn` returns a closure. Each node
invocation is a single function call into that closure.

The proposed path adds:
1. `_execute_node` function call overhead
2. `dispatch.execute()` method call (protocol dispatch)
3. `NodeInput` / `NodeOutput` construction (dataclass allocation)

For a pipeline with 100+ nodes on a hot path (e.g., Each fan-out with
1000 items), the extra function call + dataclass allocation per node
invocation adds up. The current closure-based approach has zero
allocation overhead.

In practice this is likely negligible compared to LLM API latency,
but for scripted pipelines (no LLM calls), the overhead is proportionally
larger. The design should acknowledge this and recommend benchmarking
after implementation.

### 4e. No tests directly import the wrapper functions, but several reference them

Grepping the test suite shows no direct `from factory import
_make_scripted_wrapper` imports. Tests reference these functions in
comments and docstrings (test_composition.py:792, test_modifiers.py:2167)
but don't import them. This means Phase 2 deletion of the wrapper functions
should not cause import failures.

However, the tests test these paths indirectly through `compile()` and
`run()`. If the unified path has a subtle behavioral difference (e.g.,
different argument ordering, different error message text), the tests
will catch it. This is good.

### 4f. `_BranchNode` has a duck-typed `modifiers` attribute

forward.py:354 sets `self.modifiers = []` on `_BranchNode` to satisfy
Construct validation. The compiler checks `isinstance(item, _BranchNode)`
before accessing modifiers. If ModifierSet replaces `list[Modifier]`,
`_BranchNode` must either:
- Carry a `ModifierSet()` (empty), or
- Be checked before any modifier access (already done in compiler.py:163)

The compiler already handles this correctly via isinstance dispatch, but
any new code that accesses `.modifier_set` without checking for
`_BranchNode` first will fail.

### 4g. DIResolver creates a new module (`di.py`) that crosses the layer boundary

The design proposes extracting DI logic into `di.py`. Currently, DI
classification (`_classify_di_params`) and resolution (`_resolve_di_value`)
live in `decorators.py` -- the DX layer. Moving them to a new `di.py`
module is fine architecturally, but the design does not address:

- `_resolve_merge_args` also lives in decorators.py and is imported by
  factory.py (make_oracle_merge_fn at line 803) and compiler.py
  (_merge_one_group at line 505). These cross-layer imports exist today
  and would move to `di.py`, cleaning up the dependency graph.

- The `_build_annotation_namespace` frame-walking logic is tightly coupled
  to the decorator call stack. It uses `frame_depth` relative to the
  decorator's `__init__` call. Moving it to `di.py` changes the frame
  depth calculation.

---

## 5. Recommendations

### R1. Drop NodeInput/NodeOutput or make them dict-compatible

The typed containers add no safety because the value types must be `Any`.
Options:
- Remove them entirely and keep `input_data: Any` / `result: Any` in the
  unified path.
- If you keep them, implement `__getitem__` and `.get()` on NodeInput
  so the scripted shim's `input_data.get(key)` calls work transparently.

### R2. Address ForwardConstruct explicitly

Add a section to the design: "ForwardConstruct paths are out of scope for
Phase 2. `_BranchNode`, `_LoopCall`, and branch-arm node addition remain
separate code paths. ModifierSet migration must handle `_BranchNode`'s
duck-typed `modifiers = []` by [specific mechanism]."

### R3. Keep upstream resolution separate from DIResolver

The `DIResolver` should handle 6 kinds: `from_input`, `from_config`,
`from_input_model`, `from_config_model`, `from_state`, `constant`. Do not
add `upstream` as a DIKind. Upstream resolution stays in `_extract_input` /
the scripted shim. The design's 7-kind enum should be 6.

### R4. Provide backward-compat accessors on ModifierSet

During migration, keep `has_modifier(type)` and `get_modifier(type)` as
methods on ModifierSet that delegate to slot access. This allows
incremental migration: call sites switch from `match combo:` at their
own pace. Kill the methods only after all call sites are migrated.

### R5. Plan the _BranchNode migration explicitly

Either:
- Give `_BranchNode` a `modifier_set = ModifierSet()` attribute, or
- Add it to the isinstance checks in any code that accesses modifier_set

The second is already the pattern (compiler.py:163 checks `isinstance(item,
_BranchNode)` before the modifier access block).

### R6. Acknowledge redirect wrappers in the design

The 5 redirect/merge/subgraph factory functions contain their own
duplication (oracle config forwarding, input extraction, loop counter
management). The design should either:
- Include them in the unified path (harder but higher payoff), or
- Explicitly scope them out with a note that their duplication is a
  separate cleanup.

### R7. Address the `except Exception` instances beyond HIGH-06

decorators.py has 7 `except Exception` catches beyond the one the design
flags. Most are in annotation-resolution code where any exception should
be swallowed (defensive frame-walking). The design should distinguish:
- HIGH-06: `_resolve_di_value` bare except -- must narrow (design covers this)
- LOW: annotation-resolution bare excepts -- acceptable (frame-walking
  can fail for many reasons, swallowing is intentional)

### R8. Loop router: acknowledge it is already unified

The design's Section 1 includes "Loop router unification (HIGH-02)" with
a proposed `_make_loop_router`. This function already exists in the
codebase (compiler.py:537-577). The design should mark this as "done"
rather than "proposed".

### R9. Benchmark scripted-only pipelines before/after

The current closure-based dispatch has zero per-invocation overhead. The
protocol dispatch + dataclass construction path should be benchmarked on
a 1000-node Each fan-out with scripted functions to verify the overhead
is acceptable.
