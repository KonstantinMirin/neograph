# Python Practices Review

**Scope**: Node.inputs refactor (commits 43f39cd..f58f9b9, diff base 41d910a). Files: `src/neograph/node.py`, `src/neograph/factory.py`, `src/neograph/decorators.py`, `src/neograph/_construct_validation.py`.
**Date**: 2026-04-06

## Findings

### PP-01: Silent exception swallowing in `_resolve_di_value` hides Pydantic validation errors
- **Severity**: High
- **Category**: Errors
- **File**: `src/neograph/decorators.py:289-292`
- **Description**: When `_resolve_di_value` attempts to construct a Pydantic model via `model_cls(**field_values)`, any `Exception` is caught and `None` is silently returned. This swallows `ValidationError` from Pydantic (required field missing, type coercion failure) and any other error. The user function then receives `None` with no indication that DI resolution failed. This is especially problematic because it can mask configuration errors (wrong field names in `run(input=...)`) that would otherwise surface as clear Pydantic validation errors. The old `raw_adapter` had the same pattern, but the refactor preserved it rather than fixing it.
- **Reproduction**: `grep -n "except Exception:" src/neograph/decorators.py | head -5`
- **Recommended fix**: At minimum, log a warning with the exception details before returning `None`. Better: let the `ValidationError` propagate so misconfigured DI surfaces at runtime rather than silently degrading. If backwards compatibility requires `None`, use `except ValidationError as e: log.warning("di_resolution_failed", model=model_cls.__name__, error=str(e)); return None`.

### PP-02: Namespace-building logic duplicated between `_classify_di_params` and inline in `node()` decorator
- **Severity**: Medium
- **Category**: Types
- **File**: `src/neograph/decorators.py:521-552` (duplicate of lines 194-225)
- **Description**: The `node()` decorator's inputs-inference block (lines 521-552) duplicates the frame-walking + namespace-building logic from `_classify_di_params` (lines 194-225). Both build `extra_locals`/`extra_ns` with the same keys (`FromInput`, `FromConfig`, `Annotated`), walk closure vars, walk caller frames with the same 8-hop limit, and call `get_type_hints()` with the built namespace. The only differences are: (1) variable naming (`extra_locals` vs `extra_ns`), (2) `include_extras=True` vs `False`, and (3) `frame_depth` parameter vs hardcoded `sys._getframe(1)`. This duplication was introduced by the refactor (the old code inferred a single `.input` type from the first annotated param and didn't need full hint resolution). If the frame-walking strategy changes (e.g., the 8-hop limit needs adjustment, or new markers are added to the namespace), both sites must be updated in lockstep.
- **Recommended fix**: Extract a shared helper like `_build_annotation_namespace(f: Callable, frame_depth: int) -> dict[str, Any]` that both `_classify_di_params` and the inputs-inference block call. Each caller then does its own `get_type_hints(f, localns=ns, include_extras=...)`.

### PP-03: Silent `try/except (TypeError, ValueError): pass` when mutating Node fields
- **Severity**: Medium
- **Category**: Errors
- **File**: `src/neograph/decorators.py:964-967` and `src/neograph/decorators.py:972-975`
- **Description**: In `_build_construct_from_decorated`, after computing the filtered `inputs` dict and `fan_out_param`, the code wraps `n.inputs = filtered` and `n.fan_out_param = fan_out_name` in `try/except (TypeError, ValueError): pass`. Node is a Pydantic v2 BaseModel that is NOT frozen (`model_config = {"arbitrary_types_allowed": True}`), so these assignments should always succeed. If they fail for some unexpected reason, the node would silently carry wrong `inputs` (including DI params that should have been stripped) or a missing `fan_out_param`, leading to incorrect runtime behavior. The comment on Construct's similar pattern (line 93-96) mentions "Frozen model" as justification, but Node is not frozen.
- **Reproduction**: `grep -n "except (TypeError, ValueError)" src/neograph/decorators.py`
- **Recommended fix**: Remove the try/except guards for Node mutations. If the intent is defensive coding against a future `frozen=True` on Node, add a comment explaining this. Otherwise, let errors propagate — a TypeError on field assignment is a real bug that should surface.

### PP-04: Sidecar `fan_out_param` (3rd tuple element) is always `None` and unused
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/decorators.py:130-146`
- **Description**: The `_node_sidecar` dict stores `(fn, param_names, fan_out_param)` tuples (line 134), and `_register_sidecar` accepts a `fan_out_param` argument (line 143). However, every call to `_register_sidecar` passes the default `None` for `fan_out_param` — the actual fan-out param is now stored on `Node.fan_out_param` (the new field added by this refactor). The 3rd tuple element is never read for its fan-out semantics. All destructuring sites use `_, pnames, _` (lines 861, 869, 890, 1009). This is dead weight in the sidecar contract.
- **Reproduction**: `grep -n "_register_sidecar" src/neograph/decorators.py`
- **Recommended fix**: Remove the `fan_out_param` from the sidecar tuple (make it a 2-tuple `(fn, param_names)`) or document that it's reserved for future use. The fact that `Node.fan_out_param` now carries this data makes the sidecar slot redundant.

### PP-05: Stale docstring references `item.input` instead of `item.inputs`
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/_construct_validation.py:113`
- **Description**: The docstring for `_check_item_input` still reads "Validate that `item.input` is satisfied by some upstream producer." After the rename, this should be `item.inputs`. Minor, but could confuse someone reading the validation code.
- **Reproduction**: `grep -n "item.input" src/neograph/_construct_validation.py`
- **Recommended fix**: Change to "Validate that `item.inputs` is satisfied by some upstream producer."

### PP-06: `_validate_node_chain` fallback from `.inputs` to `.input` is overly permissive
- **Severity**: Medium
- **Category**: Types
- **File**: `src/neograph/_construct_validation.py:84-88`
- **Description**: The validator tries `getattr(item, "inputs", None)` and falls back to `getattr(item, "input", None)`. This fallback means that any object with an `.input` attribute (including sub-Constructs, which have `input` for their boundary port) would have that value used as an `input_type` and passed to `_check_item_input`. In practice this is safe because Constructs are handled via `construct.input` above the loop (line 75), and by the time the fallback fires on a Construct node, `item.inputs` is `None` (Constructs don't have an `inputs` field) and `item.input` IS the sub-construct boundary type, which is the correct thing to validate. However, the fallback silently accepts any future type that has `.input` but not `.inputs`. The comment says "Construct still has `input`" but doesn't explain that the fallback is intentional for Construct sub-nodes. If the intent is to support both Node (`.inputs`) and Construct (`.input`), making this explicit with an isinstance check would be clearer.
- **Reproduction**: `grep -n "getattr(item" src/neograph/_construct_validation.py`
- **Recommended fix**: Add a comment clarifying that the `.input` fallback is specifically for `Construct` items in the node list. Or use `item.inputs if isinstance(item, Node) else getattr(item, "input", None)` to make the intent explicit.

### PP-07: `_extract_input` imports `get_origin` inline on every dict-input call
- **Severity**: Low
- **Category**: Resources
- **File**: `src/neograph/factory.py:332`
- **Description**: Inside `_extract_input`, `from typing import get_origin as _get_origin` is imported inside the `if isinstance(node.inputs, dict):` branch. This means it runs on every call to `_extract_input` for dict-input nodes (which is every invocation of an `@node`-decorated scripted function). While Python caches module imports after the first load, the import machinery still has overhead per call (lock acquisition, dict lookup in `sys.modules`). The module already imports `from __future__ import annotations` at the top level, and `get_origin` / `get_args` are used in `_is_instance_safe` (line 293) without issue.
- **Reproduction**: `grep -n "from typing import get_origin" src/neograph/factory.py`
- **Recommended fix**: Move `from typing import get_origin` to the module-level imports (it's already used indirectly via `_is_instance_safe` which calls `get_origin`).

### PP-08: `_check_fan_in_inputs` does not use `effective_producer_type` directly -- relies on pre-computed values
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/_construct_validation.py:163-209`
- **Description**: The docstring for `_check_fan_in_inputs` says "Compute the producer's effective state-bus type via `effective_producer_type`" (step 2), but the function never calls `effective_producer_type` itself. It relies on the `producers` list which was already built with `effective_producer_type` applied at line 104 in `_validate_node_chain`. This is functionally correct, but the docstring implies the function does something it doesn't, which could mislead a developer adding a new validation path that calls `_check_fan_in_inputs` with a raw producer list.
- **Reproduction**: `grep -n "effective_producer_type" src/neograph/_construct_validation.py`
- **Recommended fix**: Update the docstring step 2 to say "The producer's effective state-bus type is already computed by the caller via `effective_producer_type`" or similar.

## Summary

- Critical: 0
- High: 1
- Medium: 3
- Low: 4

## Automated Check Results (no findings)

The following automated checks from the Python practices checklist returned clean results for the in-scope files:

- **SQLAlchemy**: No `session.query()` usage (not applicable to this project)
- **Pydantic v1**: No `@validator`, `@root_validator`, `.dict()`, or `.parse_obj()` calls
- **Async/Sync**: No `run_async_in_sync_context` or `asyncio.run()` calls
- **f-string logging**: No `logger.info(f"...")` patterns
- **Mutable defaults**: No `def f(x=[])` or `def f(x={})` patterns
- **FastAPI imports in business logic**: No `from starlette` or `from fastapi` in core modules
- **File handles**: No `open()` calls without context managers
- **Bare `except:`**: No bare except clauses (all use `except Exception` or narrower)
