# Layering Review

**Scope**: Node.inputs refactor (commits 43f39cd..f58f9b9, diff base 41d910a). Focus on `node.py`, `factory.py`, `decorators.py`, `_construct_validation.py`.
**Date**: 2026-04-06

## Layer Definitions Applied

```
DX layer:              decorators.py, forward.py
IR layer:              node.py, construct.py, _construct_validation.py
Compiler layer:        compiler.py, state.py
Runtime dispatch:      factory.py
```

## Findings

### LR-01: `fan_out_param` added to IR layer (Node) for DX-only use

- **Severity**: Medium
- **Violation**: DX layer (`decorators.py`) → IR layer (`node.py`) leak of @node-specific concern
- **File**: `src/neograph/node.py:64-68`
- **Description**: `Node.fan_out_param` was added to the IR-layer `Node` model, but it is only ever set from `decorators.py:973` (`n.fan_out_param = fan_out_name`). The programmatic Node API never sets this field. The field exists solely to communicate @node decoration-time topology information (which parameter receives Each fan-out items) downstream to `factory._extract_input` (line 336) and `_construct_validation._check_fan_in_inputs` (line 183). This is an @node-layer concern embedded in the IR.
- **Reproduction**: `grep -rn "fan_out_param\s*=" src/neograph/ | grep -v "test"` — only `decorators.py:973` assigns it.
- **Mitigating factors**: The field is documented in AGENTS.md as a "deliberate, documented trade-off." The comment on lines 64-67 explicitly states "Set by @node decoration when map_over= is used." The field defaults to `None` and is inert for programmatic Node usage. This is architecturally similar to the pre-existing `raw_fn` field, which is also set by the DX layer.
- **Recommended fix**: If strict layer purity is desired, `fan_out_param` could move to the sidecar dict (it's already a 3-tuple; could become 4). However, this would require factory.py to import from decorators to read the sidecar, creating a worse cross-layer dependency. The current placement is a pragmatic trade-off — the IR field serves as a message passing channel between DX and runtime layers without introducing coupling between them.

### LR-02: `decorators.py` directly mutates `Node.inputs` post-construction

- **Severity**: Low
- **Violation**: DX layer mutating IR-layer model instances
- **File**: `src/neograph/decorators.py:954-975`
- **Description**: `_build_construct_from_decorated` directly mutates `n.inputs` (line 965) and `n.fan_out_param` (line 973) on Node instances after construction. While Node is not frozen (pydantic v2 without `frozen=True`), this pattern of post-hoc mutation makes it harder to reason about Node state at any given point. The mutations strip DI/constant params from the inputs dict and set the fan-out param marker.
- **Reproduction**: `grep -n "n\.inputs\s*=\|n\.fan_out_param\s*=\|n\.scripted_fn\s*=" src/neograph/decorators.py`
- **Mitigating factors**: All mutations happen inside a `try/except (TypeError, ValueError): pass` guard, anticipating potential future model freezing. The mutations occur at a well-defined point (after topo-sort, before Construct assembly) and are documented. The `n.scripted_fn = synthetic_name` mutation at line 1046 follows the same pattern.
- **Recommended fix**: No action needed for this refactor. If Node ever becomes frozen, these would need to use `model_copy()` — the try/except guard already anticipates this.

### LR-03: `_validate_node_chain` now reads `inputs` (plural) with fallback to `input` (singular)

- **Severity**: Low
- **Violation**: None — correctly handles both IR forms
- **File**: `src/neograph/_construct_validation.py:82-90`
- **Description**: The validator now reads `getattr(item, "inputs", None)` first, then falls back to `getattr(item, "input", None)`. This is correct: `Node` has `inputs` (plural) after the rename, while `Construct` still has `input` (singular, the sub-construct boundary port). The fallback preserves backward compatibility for Construct items in the node chain.
- **Reproduction**: `grep -n "getattr.*inputs\|getattr.*input" src/neograph/_construct_validation.py`
- **Recommended fix**: None. The dual lookup is intentional and well-commented.

### LR-04: Elimination of `_validate_fan_in_types` — two walkers collapsed to one

- **Severity**: Low (positive finding)
- **Violation**: None — this is a layering improvement
- **File**: `src/neograph/decorators.py:1050-1055` (deletion comment)
- **Description**: The refactor deleted `_validate_fan_in_types` (~70 lines) from `decorators.py` and moved its responsibility into `_construct_validation._check_fan_in_inputs`. Pre-refactor, `decorators.py` imported `effective_producer_type`, `_types_compatible`, and `_fmt_type` from `_construct_validation.py` to run its own type-checking walker. Post-refactor, `decorators.py` only imports `ConstructError`. Validation now flows through a single path: `Construct.__init__` -> `_validate_node_chain` -> `_check_fan_in_inputs`. This is a net layering improvement — the DX layer no longer duplicates IR-layer validation logic.
- **Reproduction**: `git show 41d910a:src/neograph/decorators.py | grep "from neograph._construct_validation"` shows four imports pre-refactor vs. one post-refactor.
- **Recommended fix**: None. This consolidation is the intended design.

### LR-05: Pre-existing — `factory.py` imports from `decorators.py` (lower imports higher)

- **Severity**: Medium (pre-existing, not introduced by this refactor)
- **Violation**: Runtime dispatch layer (`factory.py`) → DX layer (`decorators.py`)
- **File**: `src/neograph/factory.py:409`
- **Description**: `make_oracle_merge_fn` contains a deferred import `from neograph.decorators import get_merge_fn_metadata, _resolve_di_value`. This is a lower-layer module importing from a higher-layer module. The import is deferred (inside a function body) to avoid circular imports, which itself is a smell indicating the dependency direction is inverted.
- **Reproduction**: `grep -n "from neograph.decorators" src/neograph/factory.py`
- **Mitigating factors**: This was introduced in the `@merge_fn` decorator feature (pre-refactor, commit `a593a98`), not by the Node.inputs refactor. The deferred import keeps the module importable; the runtime dependency is narrow (two specific functions for DI resolution).
- **Recommended fix**: Consider extracting `_resolve_di_value` and the merge_fn registry into a shared module (e.g., `_di_resolution.py`) that both `decorators.py` and `factory.py` can import without a layer violation.

### LR-06: Pre-existing — `node.py` imports from `factory.py` (IR imports runtime)

- **Severity**: Low (pre-existing, not introduced by this refactor)
- **Violation**: IR layer (`node.py`) → Runtime dispatch layer (`factory.py`)
- **File**: `src/neograph/node.py:134`
- **Description**: `Node.run_isolated()` contains a deferred import `from neograph.factory import make_node_fn`. This is the IR layer importing from the runtime dispatch layer. The method is explicitly for testing convenience (`run_isolated`) and the import is deferred.
- **Reproduction**: `grep -n "from neograph.factory" src/neograph/node.py`
- **Recommended fix**: Consider moving `run_isolated` to a testing utility or making it a standalone function rather than a method on Node. Low priority — the deferred import keeps the dependency from affecting module-level imports.

## Pre-existing Cross-Layer Import Map

For context, here are all cross-layer imports in the reviewed files:

| Source (layer) | Target (layer) | Import | Introduced by refactor? |
|---|---|---|---|
| `decorators.py` (DX) | `_construct_validation.py` (IR) | `ConstructError` | No (narrowed from 4 imports to 1) |
| `decorators.py` (DX) | `factory.py` (Runtime) | `register_condition`, `register_scripted` | No (`register_scripted` usage added for kqd.8) |
| `factory.py` (Runtime) | `decorators.py` (DX) | `get_merge_fn_metadata`, `_resolve_di_value` | No |
| `node.py` (IR) | `factory.py` (Runtime) | `make_node_fn` | No |
| `compiler.py` (Compiler) | `factory.py` (Runtime) | 6 factory functions | No |

## Summary

- Critical: 0
- High: 0
- Medium: 2 (LR-01 is new from the refactor; LR-05 is pre-existing)
- Low: 4 (LR-02, LR-03, LR-06 pre-existing; LR-04 is a positive improvement)

**Overall assessment**: The Node.inputs refactor **improves** the layering posture compared to the baseline. The primary improvement is collapsing two type-validation walkers into one, removing the DX layer's dependency on three IR-layer internal functions (`_types_compatible`, `_fmt_type`, `effective_producer_type`). The `fan_out_param` addition to the IR (LR-01) is the main new layering concern, but it follows the same pattern as the pre-existing `raw_fn` field and is documented as a deliberate trade-off. No critical or high-severity violations were found.
