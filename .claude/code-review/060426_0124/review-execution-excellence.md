# Execution Excellence Review

**Scope**: Node.inputs refactor (commits 43f39cd..f58f9b9, diff base 41d910a) — focused on whether established patterns (factory dispatch, sidecar lifecycle, register_scripted contract, effective_producer_type as single source of truth, weakref finalize) are applied consistently in the new code.
**Date**: 2026-04-06

## Pattern Adoption Summary

| Pattern | Fully Adopted | Partially | Not Used | Files Checked |
|---------|--------------|-----------|----------|---------------|
| Sidecar re-registration after `\|` | 4 locations | 0 | 0 | 1 (decorators.py) |
| effective_producer_type SSOT | 2 call sites | 0 | 0 | 2 (_construct_validation.py, decorators.py) |
| register_scripted dispatch | 2 locations | 0 | 0 | 2 (decorators.py, factory.py) |
| weakref.finalize lifecycle | 2 locations | 0 | 0 | 1 (decorators.py) |
| Node.input -> Node.inputs rename | 6 files | 1 docstring | 0 | 8 |
| fan_out_param coordination | 3 locations | 1 vestigial | 0 | 3 (node.py, factory.py, decorators.py) |
| _validate_fan_in_types elimination | 1 location | 0 | 0 | 2 (decorators.py, _construct_validation.py) |

## Findings

### EE-01: Sidecar tuple's fan_out_param element is vestigial

- **Severity**: Low
- **Pattern**: Sidecar tuple contract
- **File**: `src/neograph/decorators.py:130-146`
- **Description**: The sidecar tuple is documented as `(original_fn, param_names_tuple, fan_out_param)` and the `_register_sidecar` function accepts `fan_out_param` as a parameter. However, all 4 call sites (`decorators.py:583`, `588`, `610`, `631`) pass only 3 positional args, so `fan_out_param` is always `None`. The actual fan-out param is now computed at `_build_construct_from_decorated` time (line 856-862) and stored on `node.fan_out_param` (line 973). The third sidecar element is dead weight.
- **Reproduction**: `grep -n "_register_sidecar(" src/neograph/decorators.py` — none of the call sites pass a fourth argument.
- **Recommended fix**: Remove the `fan_out_param` parameter from `_register_sidecar` and change the sidecar tuple to `(fn, param_names)`. Update the comment at line 130 and all destructuring sites (`_, pnames, _`).

### EE-02: Duplicate branches in scripted_shim for fan-out vs upstream params

- **Severity**: Low
- **Pattern**: _register_node_scripted / factory dispatch
- **File**: `src/neograph/decorators.py:1025-1041`
- **Description**: The `scripted_shim` has two branches — `elif pname in _fan_out` (lines 1025-1032) and `else` (lines 1033-1041) — that execute identical code: `input_data.get(pname) if isinstance(input_data, dict) else input_data`. This is because `_extract_input` already routes fan_out_param to `neo_each_item` and places the result in the dict under the param name. Both branches are functionally correct, but the distinction is misleading — it suggests the fan-out path does something special when it doesn't.
- **Reproduction**: Read `src/neograph/decorators.py:1025-1041` — both branches are identical.
- **Recommended fix**: Merge the two branches into one. Keep a comment explaining that fan-out params are already resolved by `_extract_input`.

### EE-03: Stale docstring references `item.input` instead of `item.inputs`

- **Severity**: Low
- **Pattern**: Node.input -> Node.inputs rename
- **File**: `src/neograph/_construct_validation.py:113`
- **Description**: The docstring for `_check_item_input` says "Validate that `item.input` is satisfied by some upstream producer." This should say `item.inputs` to match the renamed field. The code itself correctly uses `getattr(item, "inputs", None)` (line 86).
- **Reproduction**: `grep -n "item.input" src/neograph/_construct_validation.py`
- **Recommended fix**: Update the docstring: `item.input` -> `item.inputs`.

### EE-04: Stale docstring in decorators.py module header references old raw_fn dispatch

- **Severity**: Low
- **Pattern**: register_scripted contract
- **File**: `src/neograph/decorators.py:28-35`
- **Description**: The module docstring (lines 28-35) says "Scripted @node functions are dispatched via the existing `raw_fn` field on `Node` (which the `factory._make_raw_wrapper` branch already supports for `raw_node`)." This was the pre-kqd.8 design. The new design uses `register_scripted` + `_make_scripted_wrapper`. The comment at line 270 also references "raw_adapter" which is the old function name.
- **Reproduction**: `head -36 src/neograph/decorators.py | grep raw_fn`
- **Recommended fix**: Update the module docstring to describe the current dispatch path: `register_scripted()` -> `_make_scripted_wrapper`.

### EE-05: _validate_node_chain fallback from `inputs` to `input` is correct but undocumented in CLAUDE.md

- **Severity**: Low
- **Pattern**: Node.input -> Node.inputs rename
- **File**: `src/neograph/_construct_validation.py:82-88`
- **Description**: The validator tries `getattr(item, "inputs", None)` first, then falls back to `getattr(item, "input", None)`. This is correct — Node has `inputs`, Construct has `input`. The inline comment explains it. However, the duality (Node.inputs vs Construct.input) is a naming inconsistency that's easy to trip over for future code. This is an inherent design choice (Construct represents a boundary port, Node represents multiple upstream dependencies), but it should be called out in CLAUDE.md for awareness.
- **Reproduction**: `grep -n "getattr.*inputs\|getattr.*input" src/neograph/_construct_validation.py`
- **Recommended fix**: Add a brief note to the "Three API surfaces" or "Node.inputs refactor" section of CLAUDE.md explaining the naming split: `Node.inputs` (plural, for fan-in dict or single type) vs `Construct.input` (singular, sub-construct boundary port).

## Patterns Verified Clean (No Findings)

### Sidecar re-registration after modifier application
All 3 modifier paths in `node()` decorator (Each at line 583, Oracle at line 610, Operator at line 631) correctly re-register both `_node_sidecar` and `_param_resolutions` on the new Node returned by `|`. Pattern fully adopted.

### effective_producer_type as single source of truth
After the refactor, `_validate_fan_in_types` was deleted from `decorators.py`. The only producer-type computation is in `effective_producer_type` (`_construct_validation.py:38`), called at line 104. The new `_check_fan_in_inputs` (line 163) reads from the pre-computed `producers` list, which already contains effective types. No inline modifier logic anywhere. Pattern fully adopted.

### register_scripted dispatch path
`_register_node_scripted` (decorators.py:988) correctly:
1. Calls `register_scripted(synthetic_name, scripted_shim)` to register the function
2. Sets `n.scripted_fn = synthetic_name` so factory dispatch finds it
3. Uses `_resolve_di_value` for DI params (shared helper, no inline resolution)
4. `_make_scripted_wrapper` handles Each keying (lines 158-163), so the shim returns raw results

The old `_attach_scripted_raw_fn` set `n.raw_fn = raw_adapter`, which routed through `_make_raw_wrapper` (no state extraction, no Each keying). The new path routes through `_make_scripted_wrapper` which provides both `_extract_input` and Each keying — correct and consistent with the factory's existing contract for scripted nodes.

### weakref.finalize lifecycle
Both `_register_sidecar` (line 147) and `_register_param_resolutions` (line 154) install `weakref.finalize` callbacks. All call sites (4 for sidecar, 5 for param_resolutions) go through these functions. No direct dict mutation outside the registration functions.

### fan_out_param coordination across modules
- `node.py:68`: Field declaration on Node
- `decorators.py:973`: Set during `_build_construct_from_decorated`
- `factory.py:336`: Used in `_extract_input` to route fan_out_param key to `neo_each_item`
- `_construct_validation.py:183`: Used in `_check_fan_in_inputs` to skip fan-out key from upstream name validation

All four touch points are consistent.

### Import cleanup
`decorators.py` no longer imports `_fmt_type`, `_types_compatible`, or `effective_producer_type` from `_construct_validation`. Only `ConstructError` is imported (line 111). Clean separation — decorators no longer does its own type validation.

## Summary

- Critical: 0
- High: 0
- Medium: 0
- Low: 5

The Node.inputs refactor is executed with high consistency against established patterns. All 5 findings are low-severity cleanup items (stale docstrings, vestigial sidecar element, duplicate code branches). No pattern violations, no bypassed safety mechanisms, no inline modifier logic.

Key architectural wins in this refactor:
1. **Two walkers -> one walker**: `_validate_fan_in_types` in decorators.py was eliminated. Fan-in validation now flows through the existing `_validate_node_chain` -> `_check_fan_in_inputs` path, using dict-form `inputs` emitted at decoration time.
2. **raw_fn -> register_scripted**: Scripted @node functions now dispatch through `_make_scripted_wrapper` (the standard factory path) instead of `_make_raw_wrapper`. This gives them proper `_extract_input`, oracle gen ID injection, and Each keying for free.
3. **fan_out_param on Node**: Moved from a sidecar-only concern to a first-class Node field, enabling `_extract_input` and `_check_fan_in_inputs` to coordinate without accessing decorator internals.
