# Architecture Review

**Scope**: Node.inputs refactor (commits 43f39cd..f58f9b9, diff base 41d910a). Reviewed: `node.py`, `factory.py`, `decorators.py`, `_construct_validation.py`, `construct.py`, `state.py`, `compiler.py`.
**Date**: 2026-04-06

## Findings

### CR-01: Stale docstring references deleted `_validate_fan_in_types` walker

- **Severity**: Medium
- **Pattern**: Documentation drift (related to CP-5 — validator architecture)
- **File**: `src/neograph/_construct_validation.py:43-44`
- **Description**: The `effective_producer_type()` docstring still says "Both validator walkers (`_validate_node_chain` here and `_validate_fan_in_types` in `decorators.py`) consult it". But `_validate_fan_in_types` was deleted in neograph-kqd.4 — there is now only one walker. The tombstone at `decorators.py:1049` confirms the deletion. This docstring will mislead anyone looking for the second walker. The AGENTS.md was correctly updated ("One validator walker, not two" at line 81), so only this source-level docstring is stale.
- **Reproduction**: `grep -n "_validate_fan_in_types" src/neograph/_construct_validation.py`
- **Recommended fix**: Rewrite the docstring to say "The sole validator walker (`_validate_node_chain`) consults it" and remove the reference to `decorators.py`.

### CR-02: Stale docstring in `_check_item_input` references `item.input` (singular)

- **Severity**: Low
- **Pattern**: Documentation drift (naming consistency)
- **File**: `src/neograph/_construct_validation.py:113`
- **Description**: The docstring says "Validate that `item.input` is satisfied by some upstream producer." The field on Node is now `item.inputs` (plural). The function itself receives `input_type` as a parameter (not reading the field directly), so there's no functional issue — just a misleading doc reference.
- **Reproduction**: `grep -n 'item\.input' src/neograph/_construct_validation.py`
- **Recommended fix**: Change docstring to "Validate that `item.inputs` is satisfied by some upstream producer."

### CR-03: Sidecar tuple's `fan_out_param` third element is never read

- **Severity**: Medium
- **Pattern**: Dead field / technical debt from the refactor
- **File**: `src/neograph/decorators.py:130-133` (declaration), `src/neograph/decorators.py:583,588,610,631` (all registration calls pass only 3 positional args, `fan_out_param` defaults to `None`)
- **Description**: The `_node_sidecar` type is declared as `dict[int, tuple[Callable, tuple[str, ...], str | None]]` where the third element is `fan_out_param`. However, no registration call ever passes a non-`None` value for this element (all calls are `_register_sidecar(n, f, param_names)` without the optional fourth arg). The actual fan_out_param is set directly on `Node.fan_out_param` at `decorators.py:973`. Every destructure of the sidecar ignores the third element: `_, pnames, _ = sidecar` (line 861), `fn, pnames, _ = sidecar` (line 869, 1009). This dead field creates confusion about where `fan_out_param` is authoritative (answer: `Node.fan_out_param`, not the sidecar).
- **Reproduction**: `grep -n "sidecar\[2\]" src/neograph/decorators.py` (returns zero hits — the field is never read)
- **Recommended fix**: Remove the `fan_out_param` from the sidecar tuple type and from `_register_sidecar`'s signature, making the sidecar a 2-tuple `(Callable, tuple[str, ...])`. The Node field is the sole authority.

### CR-04: `_check_fan_in_inputs` docstring overstates its own role

- **Severity**: Low
- **Pattern**: Documentation accuracy
- **File**: `src/neograph/_construct_validation.py:175-176`
- **Description**: The docstring says "Compute the producer's effective state-bus type via `effective_producer_type`." But `_check_fan_in_inputs` does not call `effective_producer_type` — it receives pre-computed producer types from `_validate_node_chain` (line 104: `producers.append((field_name, effective_producer_type(item), label))`). The docstring should say "use" not "compute", and clarify that the shared helper was already applied upstream. Not a bug, but misleading if someone reads this function in isolation and expects to find the `effective_producer_type` call inside it.
- **Reproduction**: `grep -n "effective_producer_type" src/neograph/_construct_validation.py` (only found at lines 38, 104, 176 — 176 is the docstring, 104 is the actual call, both in `_validate_node_chain`)
- **Recommended fix**: Change docstring step 2 to "The producer's effective state-bus type was already computed via `effective_producer_type` in `_validate_node_chain` and passed in the `producers` list."

### CR-05: Error messages in `_construct_validation.py` still say `input=` not `inputs=`

- **Severity**: Low
- **Pattern**: Naming consistency across the rename
- **Files**: `src/neograph/_construct_validation.py:265`, `src/neograph/_construct_validation.py:391`
- **Description**: Two user-facing `ConstructError` messages use `input=` in their text (e.g., "Node 'X' declares input=Y but..."). Since the field was renamed to `inputs`, these messages create a mismatch: the error says `input=` but the fix the user would apply is `inputs=SomeType`. For dict-form inputs the messages correctly use `inputs['key']=` (lines 194, 204), so the inconsistency is only in the single-type and Each-path error messages.
- **Reproduction**: `grep -n "input=" src/neograph/_construct_validation.py | grep -v "inputs\|input_type\|input_data\|neo_subgraph_input"`
- **Recommended fix**: Change `input=` to `inputs=` in the error message strings at lines 265 and 391.

### CR-06: Unified scripted dispatch via `register_scripted` — architecture validated

- **Severity**: N/A (positive finding)
- **Pattern**: Layer discipline / single-path principle
- **Files**: `src/neograph/decorators.py:977-1046`, `src/neograph/factory.py:129-173`
- **Description**: The refactor correctly routes all `@node(mode='scripted')` functions through `_register_node_scripted` -> `register_scripted` -> `_make_scripted_wrapper`. This eliminates the prior `raw_fn` bypass for scripted `@node` and creates a single dispatch path: the scripted shim receives `(input_data, config)` from `_extract_input`, unpacks dict-form inputs by parameter name, resolves DI params, and calls the user function with positional args. The `_make_scripted_wrapper` handles Each fan-out dict-keying (lines 159-163) consistently. No double-wrapping with `make_each_redirect_fn` (which is correctly limited to Construct subgraphs). Layer discipline is maintained: `node.py` was not given `@node`-specific logic; `factory.py` was not given DI-resolution logic.

### CR-07: `Node.fan_out_param` field correctly lives at the IR layer

- **Severity**: N/A (positive finding)
- **Pattern**: Layer discipline
- **File**: `src/neograph/node.py:64-68`
- **Description**: `fan_out_param` is consumed by both `factory._extract_input` (line 336) and `_construct_validation._check_fan_in_inputs` (line 183) — two IR/compiler-layer modules. It is set by `decorators._build_construct_from_decorated` (line 973), following the same pattern as `raw_fn` and `scripted_fn`. This correctly places the field at the IR layer where its consumers live, with the DX layer as the producer. No layer violation.

### CR-08: `_validate_node_chain` dual-field fallback for Node/Construct compatibility

- **Severity**: N/A (positive finding)
- **Pattern**: Transport parity / naming split
- **File**: `src/neograph/_construct_validation.py:82-90`
- **Description**: The walker correctly handles both `Node.inputs` (plural) and `Construct.input` (singular) through `getattr(item, "inputs", None)` with fallback to `getattr(item, "input", None)`. The naming split (Node=plural, Construct=singular) is intentional and documented in the inline comment. When a sub-Construct appears in a Construct's node list, `item.inputs` is `None` (Construct has no `inputs` field), so it falls back to `item.input` and validates the sub-construct boundary type correctly.

## Summary

- Critical: 0
- High: 0
- Medium: 2 (CR-01, CR-03)
- Low: 3 (CR-02, CR-04, CR-05)
- Positive: 3 (CR-06, CR-07, CR-08)

No critical or high-severity architecture violations found. The refactor correctly:
1. Unified scripted dispatch through `register_scripted` (eliminating the `raw_fn` bypass)
2. Introduced dict-form `inputs` with proper fan-in validation
3. Added `fan_out_param` at the IR layer without leaking `@node`-specific logic downward
4. Consolidated from two validator walkers to one
5. Maintained layer discipline throughout

The medium findings (CR-01, CR-03) are cleanup items from the refactor: stale docstrings referencing the deleted walker, and a dead sidecar field that was superseded by `Node.fan_out_param`. The low findings (CR-02, CR-04, CR-05) are naming/doc inconsistencies that should be fixed to prevent future confusion.
