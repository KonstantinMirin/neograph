# Python Practices Review

**Scope**: `src/neograph/` (79 modules, ~21.7k lines) — Python 3.11+, Pydantic v2, LangGraph/LangChain. Focus per the review brief: Pydantic v2 idioms, the "typed end-to-end" claim, frame-stack DI classification, AST dead-body detection, async correctness (sync/async twins, retry), and exception/resource hygiene.
**Date**: 2026-07-08
**Method**: read the flagged hot-spots (`node.py`, `_di_classify.py`, `decorators.py`, `_llm_retry.py`, `di.py`, `_run_cache.py`, `_oracle.py`, `_tool_loop.py`, `_sidecar.py`, `_validation_types.py`), ran `mypy src/neograph/` (clean), and grepped the whole tree for the classic smells (bare except, mutable defaults, blocking calls in async, `asyncio.run` nesting, `cast`/`type: ignore` clustering).

## Headline

A senior Python engineer would call this **elegantly engineered**. The things that usually go wrong in a "clever" framework — fragile frame walking, event-loop hacks, silent exception swallowing, mutable-default traps, `Any` smeared through the core — are, with a couple of small exceptions, either handled correctly or deliberately confined to documented framework boundaries. `mypy` passes clean across all 79 files. The findings below are refinements, not structural defects: the highest-severity item is a silent name-collision in one legacy registry.

## Findings

### PP-01: `@merge_fn` global registry silently overwrites on name collision
- **Severity**: Medium
- **Category**: Types / Correctness
- **File**: `src/neograph/decorators.py:762-779`, `src/neograph/_sidecar.py:72-73`
- **Description**: `@merge_fn` registration does `_merge_fn_registry[fn_name] = (f, param_res)` and `register_scripted(fn_name, legacy_shim)` where `fn_name = name or f.__name__`. Keying by the bare function name with a plain assignment means two `@merge_fn`-decorated functions that share a `__name__` in different modules silently overwrite each other — the second decoration wins, the first Oracle's merge resolves to the wrong function with no error. This is the one place in the codebase that still uses the old global-dict pattern the sidecar refactor deliberately moved away from (the module docstring in `_sidecar.py` and CLAUDE.md both state sidecar metadata "lives on the Node via PrivateAttr, not in global dicts" — `@node` honours that; `@merge_fn` does not). CLAUDE.md lists the name-keyed registry as a "known open DX item," but frames it as a style choice; the silent-overwrite is a genuine correctness footgun, not just style. Additionally `_merge_fn_caller_ns[fn_name] = caller_ns` retains a snapshot of the caller's `f_locals` for the entire process lifetime, pinning whatever objects were in that local scope (harmless for module-scope decoration where `f_locals` is the module dict; a retention leak for merge_fns decorated inside function/test scopes).
- **Reproduction**: `grep -n "_merge_fn_registry\[" src/neograph/decorators.py` — note the unconditional assignment with no collision check. Two `@merge_fn def combine(...)` in separate modules both key on `"combine"`.
- **Recommended fix**: detect collisions at registration (`if fn_name in _merge_fn_registry and _merge_fn_registry[fn_name][0] is not f: raise ConstructError`), matching the fail-loud posture used everywhere else. Longer term, move `@merge_fn` onto the same PrivateAttr-sidecar mechanism as `@node` so the "not in global dicts" invariant is actually uniform, and drop the process-lifetime `_merge_fn_caller_ns` retention (resolve annotations at decoration time and store only the `ParamResolution`, as `@node` does).

### PP-02: `Node._param_res` is typed as bare `dict`
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/node.py:248`
- **Description**: `_param_res: dict | None = PrivateAttr(default=None)`. Every producer and consumer of this attribute uses the concrete `ParamResolution = dict[str, DIBinding]` alias (see `_di_classify.py:17`), but the field itself erases to `dict`. In a codebase that markets "typed end-to-end" and enforces a no-`Any`-in-public-IR guard, the load-bearing DI-binding store on the central IR object is the one place the parameterization is dropped. It typechecks because `dict` is permissive, so mypy won't catch a wrong-shaped assignment here.
- **Reproduction**: `grep -n "_param_res" src/neograph/node.py`
- **Recommended fix**: `from neograph._di_classify import ParamResolution` (guard the import cycle if needed) and annotate `_param_res: ParamResolution | None`. Zero runtime change; restores the type signal on the IR core.

### PP-03: `Node.tools` uses a raw mutable default instead of `default_factory`
- **Severity**: Low
- **Category**: Pydantic
- **File**: `src/neograph/node.py:179`
- **Description**: `tools: list[Tool | BaseTool] = []`. This is safe under Pydantic v2 (it deep-copies field defaults per instance, so there is no shared-mutable-state bug), but it is inconsistent with the two other collection fields on the same model — `llm_config: LlmConfig = Field(default_factory=LlmConfig)` and `modifier_set: ModifierSet = Field(default_factory=ModifierSet)`. A reader has to know Pydantic's copy-default semantics to be sure `= []` is safe; `Field(default_factory=list)` states it. Not a bug — a consistency/clarity nit on an otherwise fastidious model.
- **Reproduction**: `grep -n "= \[\]\|default_factory" src/neograph/node.py`
- **Recommended fix**: `tools: list[Tool | BaseTool] = Field(default_factory=list)`.

### PP-04: tool-arg coercion returns an empty `AIMessage` on secondary failure
- **Severity**: Low
- **Category**: Errors
- **File**: `src/neograph/_tool_loop.py:110-119` (and the async twin `:132-141`)
- **Description**: The string-args tool-error recovery path, when its `_generate`/`_agenerate` re-attempt itself raises, logs via `_log_coercion_generate_failed(inner)` and then returns `AIMessage(content="")`. The empty-content turn is a plausible-but-silent outcome: downstream the ReAct loop sees an empty assistant turn rather than a surfaced error, which can read as "the model chose to say nothing" instead of "coercion failed twice." This is a documented workaround for a real LangChain string-args tool bug, and it is logged, so it is not a swallow in the strict sense — but it is the one place where a failure degrades to empty data rather than failing loud, which is against the grain of the rest of the codebase's fail-loud posture.
- **Reproduction**: `grep -n "AIMessage(content=\"\")" src/neograph/_tool_loop.py`
- **Recommended fix**: keep the fallback but make the empty turn distinguishable — e.g. carry the `inner` error into the message metadata or raise `ExecutionError` when the second attempt fails, so an operator can tell "empty because coercion failed" from "empty because the model was terse."

### PP-05: async single-flight latches keyed by `id(loop)` — narrow id-reuse window + linger
- **Severity**: Low (mostly informational — the hard part is already handled correctly)
- **Category**: Async
- **File**: `src/neograph/_run_cache.py:64,101-111`
- **Description**: `_alatch` keys the per-key `asyncio.Lock` on `(id(asyncio.get_running_loop()), run_id, subkey)` — this is exactly the right fix for the real footgun (awaiting an `asyncio.Lock` bound to a foreign loop raises), and the double-checked single-flight logic in `get_or_build`/`aget_or_build` is textbook-correct. Two residual, minor issues: (1) `id()` of a garbage-collected loop can be recycled, so if a loop is created→used→GC'd and a new loop is allocated at the same address while a stale entry survives, `_alatch` could hand back a Lock bound to the dead loop; this requires `evict_run` to have missed the entry AND an address reuse, so it is narrow, but it is not impossible. (2) Entries for dead loops linger in `_alatches` until `evict_run` fires for their run_id — a slow leak if a run ends without eviction. Long-lived single-`asyncio.run` processes never hit either.
- **Reproduction**: `grep -n "id(asyncio.get_running_loop())" src/neograph/_run_cache.py`
- **Recommended fix**: optional. Keying on a `weakref` to the loop (or storing the loop object and comparing identity on fetch, minting fresh if it differs) closes the id-reuse window; a `WeakValueDictionary`-style sweep or an eviction hook on loop close addresses the linger. Given the constraints, documenting the assumption ("one long-lived loop per process; `evict_run` owns cleanup") may be sufficient.

## Things that are done right (called out because the brief asked "clever-but-fragile or genuinely elegant")

- **DI frame capture is sound, not fragile.** CLAUDE.md still describes "8-hop caller frame inspection," but the live code (`decorators.py:323`, `_di_classify.py:140-178`) captures `sys._getframe(1).f_locals` **once** at decoration time and threads it explicitly as `caller_ns` — no depth arithmetic, no stack walking. Both `@node` and `@node(...)` call `node()` from user code, so `_getframe(1)` is the user frame in either form, and module-level types are additionally recovered via `inspect.getclosurevars(f).globals`. This is the correct, robust version of the pattern Pydantic itself uses. (The CLAUDE.md text is stale and should be updated — reviewers reading it would wrongly believe the fragile approach is live.)
- **AST dead-body detection is conservative and warning-only.** `_is_trivial_body` (`decorators.py:112`) strips docstrings, only flags a genuinely non-trivial body, warns (never errors), and swallows `OSError/TypeError` when source is unavailable (REPL/dynamic). It cannot break a valid pipeline.
- **The sync/async DI split fails loud instead of hacking the event loop.** `DIBinding.resolve` raises a clear `ConfigurationError` for `FROM_RESOURCE` (which must await), and `aresolve` delegates every non-resource kind straight back to `resolve` (`di.py:329-451`). No `asyncio.run` nesting, no `run_until_complete`, no blocking calls anywhere in an async path (grep-verified across all 20 async modules).
- **Exception hygiene is clean.** Zero bare `except:`. All 13 `except Exception` sites either re-raise selectively (`_llm.py`, `_oracle.py`), wrap into a domain error (`_llm_retry.py`), or have an explicitly-documented fallback with a `# noqa: BLE001` rationale (`di.py` fetch-as-candidate-expiry). No swallow-and-continue.
- **Pydantic PrivateAttr sidecar is idiomatic v2 and genuinely tested.** `_sidecar`/`_param_res`/`_scripted_shim` as `PrivateAttr(default=None)` relying on v2's `__pydantic_private__` copy through `model_copy` is the correct way to carry a non-schema `Callable` on a model; `test_node_sidecar_contract.py` pins the preservation across `model_copy`/pipe/deepcopy (56 references).
- **Type-safety claim holds.** `mypy src/neograph/` → "Success: no issues found in 79 source files." The 25 `cast(...)` and 35 `type: ignore` sites are confined to two honest boundaries: LangGraph's loosely-typed `graph.add_node` (`_wiring.py`, all `cast(Any, ...)`) and the PEP 747 TypeForm gap where `TypeSpec` is statically `Any` because Python has no `TypeForm` yet (documented in `node.py:132-145`). None hide a real error.

## Summary

- Critical: 0
- High: 0
- Medium: 1 (PP-01 — silent `@merge_fn` name collision)
- Low: 4 (PP-02 bare-`dict` IR field; PP-03 mutable-default field; PP-04 empty-message degrade; PP-05 loop-id latch window)
