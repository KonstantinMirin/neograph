# Consistency Review

**Scope**: Node.inputs refactor (commits 43f39cd..f58f9b9, diff base 41d910a). All files under `src/neograph/`, `tests/test_e2e_piarch_ready.py`, and `examples/`.
**Date**: 2026-04-06

## Convention Inventory

Before listing findings, document the conventions observed:

| Convention | Canonical Form | Files Following | Files Diverging |
|------------|---------------|-----------------|-----------------|
| Node input field naming | `Node.inputs` (plural) | 10 | 0 |
| Construct boundary field | `Construct.input` (singular) | 5 | 0 |
| Runtime API parameter | `run(input={...})` (singular) | 2 | 0 |
| Error class for assembly | `ConstructError` | 2 (decorators, _construct_validation) | 0 |
| Error class for runtime/config | `ValueError` | 6 (factory, compiler, state, runner, _llm, modifiers) | 0 |
| Error message prefix (validation) | `"Node '{name}' in construct '{name}'"` | 1 (_construct_validation) | 0 |
| Error message prefix (decorator) | `"@node '{label}'"` | 1 (decorators) | 0 |
| Logging library | `structlog.get_logger()` as `log` | 3 (factory, compiler, _llm) | 0 |
| Log event names | `node_start` / `node_complete` | 1 (factory) | 0 |
| Log `input_type` field source | `_type_name(node.inputs)` | 1 (factory) | 0 |
| Import style | Absolute `from neograph.X import Y` | 13 | 0 |
| Synthesized names | `_node_{name}_{id:x}` | 2 (decorators) | 0 (minor variant, see CON-04) |
| `None` sentinel for absent value | `= None` field default | 6 | 0 |
| `dict` field default | `Field(default_factory=dict)` | 3 (node, construct, tool) | 0 |

## Findings

### CON-01: Error messages in `_construct_validation.py` say `input=` where the field is now `inputs`
- **Severity**: Medium
- **Convention**: Error messages should reference field names users see in API (`Node.inputs`, not `Node.input`)
- **Files**: `_construct_validation.py:265` uses `input=`, `_construct_validation.py:391` uses `input=`; `_construct_validation.py:194,204` correctly uses `inputs[...]`
- **Description**: Two error message formatters in `_construct_validation.py` still emit `input=SomeType` in their user-facing string, while the Node field was renamed to `inputs` in neograph-kqd.1. The `_check_fan_in_inputs` helper at lines 194/204 correctly says `inputs['...']`, making the divergence within the same file.
- **Reproduction**:
  ```
  grep -n "input=" src/neograph/_construct_validation.py
  ```
  Lines 265 and 391 say `input=`, lines 194 and 204 say `inputs[...]`.
- **Recommended fix**: Change `input={_fmt_type(input_type)}` to `inputs={_fmt_type(input_type)}` on lines 265 and 391 of `_construct_validation.py`.

### CON-02: Docstring in `_check_item_input` references `item.input` (singular)
- **Severity**: Low
- **Convention**: Docstrings should reference the canonical field name
- **Files**: `_construct_validation.py:113` says `item.input`, but the code at line 86 reads `getattr(item, "inputs", None)` first
- **Description**: The docstring says "Validate that `item.input` is satisfied by some upstream producer" but the code tries `item.inputs` first (Node) then falls back to `item.input` (Construct). The comment at lines 83-85 of `_validate_node_chain` correctly documents the distinction, but the `_check_item_input` docstring is stale.
- **Reproduction**:
  ```
  grep -n "item.input" src/neograph/_construct_validation.py
  ```
- **Recommended fix**: Update docstring to "Validate that `item.inputs` (or `item.input` for Construct) is satisfied by some upstream producer."

### CON-03: Logging `input_type` on dict-form inputs produces unstructured output
- **Severity**: Low
- **Convention**: `node_start` log events use `input_type=_type_name(node.inputs)` across all 5 wrapper factories
- **Files**: `factory.py:71-75` (`_type_name`), called at lines 116, 136, 184, 219, 260
- **Description**: `_type_name()` returns `getattr(t, '__name__', str(t))`. When `node.inputs` is a dict like `{'claims': Claims, 'clusters': Clusters}`, the logged `input_type` becomes a raw dict repr string containing class repr objects. All five wrapper functions pass this field identically, so the pattern is consistent -- but the dict-form case produces noisy logs. Pre-kqd, `node.input` was always a single type, so `_type_name` always returned a clean name. Now that dict-form inputs are canonical, the helper doesn't format them well.
- **Reproduction**: Any `@node` with fan-in parameters; check structlog output for `input_type` field.
- **Recommended fix**: Teach `_type_name` to handle dict-of-types: e.g., `if isinstance(t, dict): return {k: _type_name(v) for k, v in t.items()}`. Or accept the current behavior as "good enough for debug logs."

### CON-04: Synthesized name patterns diverge between condition names and scripted shim names
- **Severity**: Low
- **Convention**: Synthesized names for auto-registered functions use `_node_{something}_{hex_id}` pattern
- **Files**: `decorators.py:621` uses `f"_node_interrupt_{node_label}_{id(f):x}"`, `decorators.py:1015` uses `f"_node_{n.name}_{id(n):x}"`
- **Description**: Both patterns produce collision-free names, but they hash different objects: the interrupt version hashes `id(f)` (the user function), while the scripted shim hashes `id(n)` (the Node instance). Since `f` and `n` have different lifetimes, the uniqueness guarantee is the same, but greppability diverges. Also, one uses `node_label` (already a dash-form name) while the other uses `n.name` (also dash-form), so the actual name segment is equivalent.
- **Reproduction**:
  ```
  grep -n "_node_.*_" src/neograph/decorators.py | grep -E "(interrupt|_node_\{)"
  ```
- **Recommended fix**: Consider normalizing both to `_node_{purpose}_{n.name}_{id(n):x}` for uniformity, but this is cosmetic.

### CON-05: `log.bind()` fields vary across wrapper factories without clear pattern
- **Severity**: Low
- **Convention**: `_make_*_wrapper` functions in `factory.py` bind contextual fields to `node_log`
- **Files**: `factory.py:115` (raw: `node, mode`), `factory.py:135` (scripted: `node, mode, fn`), `factory.py:183` (produce: `node, mode, model, prompt`), `factory.py:217` (gather: `node, mode, model, prompt`), `factory.py:258` (execute: `node, mode, model, prompt`)
- **Description**: Each wrapper binds fields relevant to its mode, which is intentional (raw has no model/prompt, scripted has no model but has fn, LLM modes have model+prompt). The pattern is consistent within its design -- fields vary by mode semantics. However, `gather` and `execute` log extra fields (`tools`, `budgets`) in `node_start` that `produce` does not, because produce has no tools. This is correct behavior, not a divergence. All five use `input_type` and `output_type` consistently.
- **Reproduction**:
  ```
  grep -n "log.bind" src/neograph/factory.py
  ```
- **Recommended fix**: No fix needed. Variation is semantic, not accidental.

### CON-06: `ConstructError` vs `ValueError` boundary is clear but undocumented
- **Severity**: Low
- **Convention**: `ConstructError` for assembly-time/topology errors; `ValueError` for runtime/config errors
- **Files**: `_construct_validation.py` and `decorators.py` use `ConstructError`; `factory.py`, `compiler.py`, `state.py`, `runner.py`, `_llm.py`, `modifiers.py` use `ValueError`
- **Description**: The split is clean and consistent. `ConstructError(ValueError)` is used for assembly-time type validation and decorator-time constraint checking (17 call sites in decorators.py, 6 in _construct_validation.py). Plain `ValueError` is used for runtime failures (missing registry entries, missing config, invalid state). `TypeError` is used for incorrect usage patterns (bad lambda in `.map()`, bad proxy context). No file mixes the two. The comment at `_construct_validation.py:371-377` explicitly documents why validation errors use a richer multi-line format while other errors stay single-line.
- **Reproduction**:
  ```
  grep -rn "raise ConstructError\|raise ValueError\|raise TypeError" src/neograph/ --include="*.py"
  ```
- **Recommended fix**: No fix needed. The boundary is well-maintained. A docstring in `_construct_validation.py` at the `ConstructError` class already explains the subclass relationship.

### CON-07: `_check_fan_in_inputs` does not use `effective_producer_type` for type comparison
- **Severity**: Medium
- **Convention**: "Both validator walkers call `effective_producer_type(item)` [...] That helper is the single source of truth for modifier-aware type effects" (CLAUDE.md)
- **Files**: `_construct_validation.py:184-186` builds `producer_by_name` from the `producers` list which was populated by `effective_producer_type` at line 104. `_check_fan_in_inputs:200` reads directly from `producer_by_name`.
- **Description**: This is actually correct. `_check_fan_in_inputs` consumes the `producers` list which was already built using `effective_producer_type` at line 104 of `_validate_node_chain`. The function does NOT re-compute producer types inline. The CLAUDE.md invariant ("do NOT re-inline modifier checks in either walker") is honored. No finding here -- listing for completeness of the audit.
- **Recommended fix**: None needed.

## Summary

- Critical: 0
- High: 0
- Medium: 2 (CON-01, CON-03)
- Low: 4 (CON-02, CON-04, CON-05, CON-06)

## Overall Assessment

The Node.inputs refactor is highly consistent. The field rename from `Node.input` to `Node.inputs` has been applied correctly across all source files, factory wrappers, examples, and tests. The `Construct.input` (singular) vs `Node.inputs` (plural) distinction is well-documented and correctly maintained throughout.

The two medium findings are residual artifacts of the rename:
1. Two error message strings in `_construct_validation.py` still say `input=` (singular) instead of `inputs=` -- this affects user-facing error messages but not runtime behavior.
2. `_type_name()` doesn't handle the new dict-form inputs gracefully in log output -- this is a pre-existing limitation that became more visible with the refactor.

No convention violations were found in:
- Import organization (all absolute, consistent style)
- Error class usage (ConstructError vs ValueError boundary is clean)
- Logging patterns (structlog usage is uniform, event names are consistent)
- Boolean/flag conventions (has_modifier pattern is uniform)
- Configuration access (no env vars in source, all config flows through RunnableConfig)
- Null handling (consistent `= None` defaults, consistent `is None` checks)
- register_scripted naming (consistent `_node_` prefix pattern)
