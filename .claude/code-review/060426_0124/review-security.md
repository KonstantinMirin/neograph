# Security Review

**Scope**: Node.inputs refactor (commits 43f39cd..f58f9b9, diff base 41d910a). Focused on `src/neograph/decorators.py` and `src/neograph/factory.py` — the two files with security-relevant changes in this feature.
**Date**: 2026-04-06

## Context

neograph is a graph compiler library, not a multi-tenant web service. The standard web-app threat model (T1-T11: tenant isolation, auth bypass, SQL injection, SSRF, XSS, CSRF, etc.) does not apply. There are no HTTP endpoints, no database queries, no authentication flows, no user-facing templates, and no secrets management.

The relevant threat surface is **library-level safety**: can the patterns introduced by this refactor be exploited by a caller who controls inputs to the library's public API? The review addresses the four specific concerns raised in the review request.

## Threat Coverage

| Threat | Checked | Findings | Highest Severity |
|--------|---------|----------|-----------------|
| T1 Tenant Isolation | N/A | 0 | N/A (not a multi-tenant system) |
| T2 Principal Isolation | N/A | 0 | N/A |
| T3 Auth Bypass | N/A | 0 | N/A |
| T4 SQL Injection | N/A | 0 | N/A |
| T5 Secret Exposure | N/A | 0 | N/A |
| T6 SSRF | N/A | 0 | N/A |
| T7 Open Redirect | N/A | 0 | N/A |
| T8 XSS | N/A | 0 | N/A |
| T9 CSRF | N/A | 0 | N/A |
| T10 Mass Assignment | N/A | 0 | N/A |
| T11 Timing | N/A | 0 | N/A |
| Frame walking exploitation | Yes | 1 | Low |
| Synthesized register_scripted name collision | Yes | 1 | Low |
| State key injection via _extract_input | Yes | 1 | Low |
| Type injection via get_type_hints + frame locals | Yes | 1 | Low |

## Findings

### SEC-01: Frame walking captures locals from unrelated frames

- **Severity**: Low
- **Threat**: Type injection / information leakage via frame walking
- **File**: `src/neograph/decorators.py:207-214` (in `_classify_di_params`) and `src/neograph/decorators.py:538-545` (in `node()` decorator, inputs inference block)
- **Description**: Both `_classify_di_params` and the inputs-inference block in `node()` walk up to 8 frames via `sys._getframe()`, harvesting `f_locals` from every frame into `extra_locals` / `extra_ns`. These dicts are then passed as `localns` to `typing.get_type_hints()` to resolve string annotations. The walk collects *all* non-underscore-prefixed locals from *every* ancestor frame up to 8 hops — including frames that have nothing to do with the current `@node` decoration.

    In practice, this means that if an unrelated function higher in the call stack has a local variable with the same name as a type annotation in the decorated function, the annotation could resolve to that variable's value rather than the intended type. For example:

    ```python
    def outer():
        Claims = "not a type"  # local in an ancestor frame
        # ... some call chain eventually leads to:
        @node(output=Result)
        def my_node(data: Claims) -> Result: ...  # resolves Claims to "not a type"
    ```

    The `not _k.startswith("_")` filter at line 211/542 prevents capturing private/dunder names but still captures any public-named local.

- **Reproduction**: A caller would need to define a local variable with the same name as a type used in a `@node` annotation, in a frame that's within 8 hops of the decoration call. This is unlikely in normal usage because: (a) `@node` is typically applied at module scope where the frame stack is shallow, and (b) the `k not in extra_locals` guard means the function's own closure/globals take precedence. The realistic scenario is test code that defines types inside test methods — which is exactly what this frame walking is designed to support.

- **Recommended fix**: No code change needed. The risk is theoretical and the pattern is well-established (Pydantic uses the same technique for forward-ref resolution, as noted in the code comments). The 8-hop limit is reasonable. If future hardening is desired, the walk could stop at frames whose `f_globals['__name__']` starts with `neograph.` (skip neograph internals, same pattern as `_source_location` in `_construct_validation.py:457-472`) and stop walking once it finds the first user frame rather than continuing for 8 hops.


### SEC-02: Synthesized register_scripted name uses `id(n)` — address reuse risk

- **Severity**: Low
- **Threat**: Registry collision via CPython id reuse
- **File**: `src/neograph/decorators.py:1015`
- **Description**: `_register_node_scripted` synthesizes a scripted-registry key as `f"_node_{n.name}_{id(n):x}"`. CPython's `id()` returns the memory address of the object, and addresses can be reused after garbage collection. If a Node is created, registered, then garbage-collected, and a *new* Node happens to get the same memory address, the new node's shim could overwrite the old entry in `_scripted_registry`.

    However, this requires:
    1. The old Node to be garbage-collected (which means no references to it anywhere).
    2. The new Node to get the exact same memory address.
    3. The new Node to have the exact same `n.name`.
    4. Both to go through `_register_node_scripted`.

    Conditions (1) + (3) together are contradictory in practice: if the old Node was GC'd, it's not part of any live Construct, so even if the registry entry is overwritten, no live code references the old shim name.

    Additionally, `_scripted_registry` in `factory.py` is a global `dict[str, Callable]` with no cleanup mechanism — entries persist for the process lifetime. The `_node_sidecar` dict has `weakref.finalize` cleanup, but `_scripted_registry` does not. This means old shim entries accumulate over time. This is a minor memory leak, not a security issue per se, but worth noting.

- **Reproduction**: Would require deterministic control of CPython's memory allocator to force address reuse, plus timing the GC cycle. Not practically exploitable.

- **Recommended fix**: No change needed for security. The `id(n)` pattern is safe because the Node is kept alive by the Construct that holds it. If the team wants belt-and-suspenders, an incrementing counter (`itertools.count()`) would eliminate the theoretical id-reuse concern and the stale-registry accumulation issue in one change — but this is a cleanliness improvement, not a security fix.


### SEC-03: `_extract_input` reads state keys by name without allowlisting

- **Severity**: Low
- **Threat**: State field leakage if user controls `node.inputs` dict keys
- **File**: `src/neograph/factory.py:331-348`
- **Description**: `_extract_input` iterates over `node.inputs.items()` and reads the corresponding state field for each key via `_get(state_key)`. If an attacker could control the keys of `node.inputs`, they could read arbitrary fields from the LangGraph state object — including internal fields like `neo_each_item`, `neo_oracle_gen_id`, `neo_subgraph_input`, etc.

    However, `node.inputs` is set in two ways:
    1. **Programmatic API**: the caller explicitly constructs `Node(inputs={'key': Type})` — but this caller already has full access to the state object (they're writing the pipeline code).
    2. **`@node` decorator**: `inputs` is inferred from function parameter annotations at decoration time (`decorators.py:554-563`), then filtered to only include keys that match other `@node` names in the same Construct (`decorators.py:959-961`). This filtering is an effective allowlist.

    The only scenario where a malicious `inputs` dict could be injected is if an LLM-driven runtime pipeline (the documented use case in `website/src/content/docs/runtime/llm-driven.mdx`) accepts untrusted input for Node construction. In that case, the LLM (or its tool output) would supply the `inputs` dict keys.

    The `_get` helper at line 312-315 does `state.get(key)` for dicts or `getattr(state, key, None)` for models. The `getattr` path could theoretically access any attribute on the state object, not just Pydantic fields. However, LangGraph state objects are Pydantic BaseModels with a defined schema, so `getattr` on unknown names returns `None` (Pydantic models don't have arbitrary attributes), and there's no way to trigger code execution via attribute access on a Pydantic model.

- **Reproduction**: An attacker controlling the `inputs` dict of a programmatically-constructed Node can read any state field by name, but they already have access to the state object in that context (they're defining the pipeline). For `@node`, the inputs dict keys are filtered against the declared node set, so injection is not possible.

- **Recommended fix**: No change needed. For the LLM-driven runtime use case, if untrusted input controls Node construction, the recommendation is to validate/allowlist the `inputs` dict keys at the point where the LLM's tool output is deserialized into Node specs — not inside `_extract_input`, which is a low-level internal function that trusts its caller.


### SEC-04: `get_type_hints` with caller frame locals — type injection surface

- **Severity**: Low
- **Threat**: Malicious type resolution via poisoned frame locals
- **File**: `src/neograph/decorators.py:218-223` and `src/neograph/decorators.py:548-550`
- **Description**: `typing.get_type_hints(f, localns=extra_locals, include_extras=True)` resolves string annotations using the `extra_locals` namespace built from frame walking (SEC-01). If a malicious value is in `extra_locals` under a name matching a string annotation, `get_type_hints` will resolve the annotation to that value.

    The consequence of a poisoned type resolution:
    - For `_classify_di_params`: the `inner_type` extracted from `Annotated[T, FromInput]` could be a non-type object. The `issubclass(inner_type, _BaseModel)` check at line 255 would either return `False` (treating it as a per-param lookup, benign) or raise `TypeError` (caught by the outer `except Exception` at line 225, also benign — falls through to empty `param_res`).
    - For the inputs inference block: a poisoned type would be stored in `inputs_dict` and eventually in `Node.inputs`. The validator (`_check_fan_in_inputs`) would then try `_types_compatible(producer_type, expected_type)` with the poisoned value — which would either fail cleanly (raising `ConstructError` at assembly time) or pass if the poisoned type happens to be compatible.

    In the worst case, a poisoned type causes the validator to accept an incompatible pipeline that fails at runtime with a confusing error. It cannot cause code execution, data exfiltration, or privilege escalation — only a misleading validation result.

- **Reproduction**: Requires a malicious frame local with the exact name of a string annotation in a `@node`-decorated function, within 8 hops of the decoration call site. The attacker must control code running in the same process. At that point, they can already do anything — so this is not an escalation of privilege.

- **Recommended fix**: No change needed. Same reasoning as SEC-01. The frame-walking pattern is inherent to supporting `from __future__ import annotations` with locally-defined types. The blast radius of a poisoned type is limited to misleading validation errors.


## Summary

- Critical: 0
- High: 0
- Medium: 0
- Low: 4

All four findings are theoretical risks inherent to the patterns used (frame walking for forward-ref resolution, `id()` for registry keying, dict-key-driven state access). None represent exploitable vulnerabilities in the context of a graph compiler library where the caller is trusted (they're writing pipeline code that runs in-process). The patterns are well-established — Pydantic uses identical frame-walking for its own forward-ref resolution.

The refactor itself (Node.inputs rename, dict-form inputs, single validator walker, register_scripted dispatch) does not introduce new attack surface beyond what existed before. The frame walking and `_extract_input` patterns were present in the pre-refactor code; the refactor extends them consistently without weakening existing boundaries.
