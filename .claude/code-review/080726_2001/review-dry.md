# DRY Review

**Scope**: `src/neograph/` (76 files, ~21.7k lines) — full source tree, with emphasis on the single-source-of-truth (SSOT) claims in AGENTS.md/CLAUDE.md and the prompt-rendering paths.
**Date**: 2026-07-08

## Verdict on the codebase's own SSOT claims

The load-bearing SSOT claims **hold**. This is a genuinely well-factored codebase for
duplication; the violations that exist are small, localized, and all sit in the
prompt-rendering helpers — none contradict a documented monopoly.

| Claim | Verdict | Evidence |
|-------|---------|----------|
| `effective_producer_type` is the ONE place modifier-aware producer types are computed | **UPHELD** | `_validation_types.py:78` + per-key core `effective_producer_type_for:104`. Both producer-registration paths (`_construct_validation.py:212,223`) and the loop-self resolver (`_construct_graph.py:81`) route through it. No re-inlined `Each → dict[str, X]` modifier check found in producer position. |
| `_declared_output` monopolizes Node.outputs vs Construct.output selection | **UPHELD** | Single definition `_normalize.py:128`. All other `.output` reads (`state.py:160,177`, `compiler.py:429`, `_subconstruct.py:160`, `_construct_builder.py:139`) are on values already discriminated as `Construct`/sub — accessing a known field, not hand-rolling the selector. No stray `getattr(item, 'output', None)` outside `_normalize.py`. |
| DI has "one resolver, one classifier" shared by @node and @merge_fn | **UPHELD** | Classifier `_classify_di_params` (`_di_classify.py:181`) called from both the `@node` path (`decorators.py:448`) and `@merge_fn` (`decorators.py:717`). Resolver is `DIBinding.resolve`/`aresolve` (`di.py:329,409`); `_resolve_di_args` (`_di_classify.py:318`) delegates to it. No second resolver. |
| `_inject_di_inputs` is the single injector for think/agent/act | **UPHELD (with justified twin)** | think dispatches through `_dispatch.py:262`; agent/act reuse the *same* module-level function at `_agent_cycle.py:170`. The only split is the async twin `_ainject_di_inputs` (`_dispatch.py:78`) — see DRY-04; it is legitimate (awaits `FROM_RESOURCE`) and reuses the canonical resolver. |

Modifier classification (`classify_modifiers`, `modifiers.py:85`), inline-prompt
detection (`_is_inline_prompt`, `_llm_render.py:40`), and Each dict→list unwrap
(`_unwrap_each_dict`, `di.py:267`, shared by `_input_shape.py:116,127`) are each
single-sourced and reused. The runner's sync/async checkpoint twins deliberately
extract the shared decision into `_decide_checkpoint_schema` and reuse the pure
helpers verbatim — the *correct* way to handle twin duplication, and a useful
contrast to the findings below.

## Duplication Map

| Pattern | Occurrences | Files | Extractable? |
|---------|------------|-------|-------------|
| `render_for_prompt()` str/BaseModel/list dispatch | 2 | `renderers.py` (`_render_with_flattening`, `_render_single`) | Yes → have `_render_with_flattening` delegate to `_render_single` |
| `${path}` root+getattr-walk resolution | 2 | `_llm_render.py` (`_resolve_var`, `_resolve_var_raw`) | Yes → extract `_walk_path`, apply formatting at the tail |
| dict-walk + flatten-merge | 2 | `renderers.py` (`render_input`, `build_rendered_input`) | Yes → `render_input` = `build_rendered_input(...).for_template_ref` |
| di_inputs build loop (sync/async twin) | 2 | `_dispatch.py` (`_inject_di_inputs`, `_ainject_di_inputs`) | Partially (inherent sync/async cost) |

## Findings

### DRY-01: `render_for_prompt()` dispatch re-implemented across two rendering helpers
- **Severity**: Medium
- **Category**: Response (rendering)
- **Occurrences**: 2
- **Files**:
  - `src/neograph/renderers.py:414-439` — `_render_with_flattening` inline-implements the `render_for_prompt()` protocol: `hasattr` check, call, then branch `isinstance(result, str)` / `isinstance(result, BaseModel)` (→ `renderer.render` else `describe_value`) / `list[BaseModel]` (same).
  - `src/neograph/renderers.py:442-461` — `_render_single` implements the identical protocol dispatch with the same three-way branch and the same `renderer.render` vs `describe_value` fallback.
- **Description**: Both functions independently decode the `render_for_prompt()` return-type contract (str verbatim / BaseModel re-render / list[BaseModel] re-render / passthrough). They are not textually identical — `_render_with_flattening` additionally flattens model fields — but the *protocol-decoding core* is semantically duplicated. A change to that contract (e.g. supporting a new return shape, or changing the `renderer is None → describe_value` fallback) must be edited in both, and `_render_with_flattening` calls `_render_single` only in its final `else` branch, so the overlap is avoidable.
- **Proposed extraction**: Extract the protocol decode into one helper `_render_prompt_result(result, renderer) -> str | None` (returns `None` when `value` has no `render_for_prompt`), and have `_render_with_flattening` call it, adding only the field-flattening on top. `_render_single` becomes a thin caller of the same helper plus the framework-container rules (`_is_tool_interaction_list` / `_is_model_dict`).

### DRY-02: `${path}` variable-resolution walk duplicated verbatim
- **Severity**: Medium
- **Category**: Response (rendering)
- **Occurrences**: 2
- **Files**:
  - `src/neograph/_llm_render.py:50-89` — `_resolve_var`: `path.split(".")`, dict-vs-single root resolution with `prompt_var_missing` warning, `getattr`-walk over `rest` with per-segment warning; tail renders BaseModel via `describe_value` else `str`.
  - `src/neograph/_llm_render.py:92-114` — `_resolve_var_raw`: byte-for-byte identical split / root-resolution / getattr-walk (same two warnings), differing ONLY in the tail: returns the raw object instead of rendering.
- **Description**: The entire resolution mechanism — the part with actual logic and the two fail-soft warning sites — is copied. The functions differ by exactly their last three lines. A fix to path resolution (e.g. list-index access `items[0]`, or changing the missing-var warning) silently diverges between inline text rendering and inline image resolution.
- **Proposed extraction**: Extract `_walk_var_path(path, input_data) -> Any` holding the split + root + getattr-walk + warnings. `_resolve_var` wraps it with the BaseModel/`describe_value` tail; `_resolve_var_raw` returns it directly.

### DRY-03: `render_input` re-walks the dict that `build_rendered_input` already walks
- **Severity**: Low
- **Category**: Response (rendering)
- **Occurrences**: 2
- **Files**:
  - `src/neograph/renderers.py:298-325` — `render_input` walks the input dict, calls `_render_with_flattening` per value, and merges flattened extras into the same result dict (`if fname not in result`).
  - `src/neograph/renderers.py:337-360` — `build_rendered_input` walks the same dict, calls the same `_render_with_flattening`, and produces `rendered` + `flattened` which `for_template_ref` (`:57-66`) then merges with the identical precedence rule.
- **Description**: For the dict case, `render_input(x, renderer=r)` is equal to `build_rendered_input(x, r).for_template_ref` — the same walk and the same "existing key wins over flattened" merge, expressed twice. `render_input` remains a live public API (exported in `__init__.py`, exercised by `test_renderers.py`, examples 12/18) and `test_guards_meta.py:321` documents the `render_input`/`render_inputs` split as intentional, so the *entry point* should stay — but its body can delegate.
- **Proposed extraction**: Reimplement `render_input` as a thin wrapper: `return build_rendered_input(input_data, renderer=renderer).for_template_ref`. Verify the single-value path matches (`build_rendered_input` returns `rendered` for non-dicts, which equals `_render_single`).

### DRY-04: sync/async di_inputs injector twins duplicate the build loop
- **Severity**: Low (largely inherent)
- **Category**: Error/dispatch (sync-async twin)
- **Occurrences**: 2
- **Files**:
  - `src/neograph/_dispatch.py:40-75` — `_inject_di_inputs`: param_res guard, `DI_TEMPLATE_KINDS` comprehension, empty guard, `_with_configurable` stash.
  - `src/neograph/_dispatch.py:78-112` — `_ainject_di_inputs`: same guard/gate/stash skeleton, but a `for`-loop that additionally awaits `FROM_RESOURCE` via `aget_or_build`.
- **Description**: The guard/gate/stash envelope is repeated. Unlike DRY-01/02 this is close to the irreducible sync/async cost in Python (the async twin must `await`), and the resolver itself is already shared (`DIBinding.resolve`), so this is noted for completeness rather than as a defect. The runner's `_decide_checkpoint_schema` shows the reduction is *possible* here too: the param_res guard and empty-check envelope could be a shared `_finalize_di_inputs(node, config, di_inputs)` the two twins call after building their maps.
- **Proposed extraction**: Optional — a shared `_finalize_di_inputs(node, di_inputs, config)` for the guard + `_with_configurable` tail. Low value; only pursue if a third injection path appears.

## Summary

- Critical: 0
- High: 0
- Medium: 2 (DRY-01, DRY-02)
- Low: 2 (DRY-03, DRY-04)
- Total duplicated logic blocks: 4
- Estimated lines removable by extraction: ~35-45 (all in `renderers.py` + `_llm_render.py`)

**Overall**: Elegantly engineered with respect to DRY. Every documented single-source-of-truth
monopoly (`effective_producer_type`, `_declared_output`, the DI resolver/classifier,
`_inject_di_inputs`) is verified intact — no re-inlined modifier checks, no hand-rolled
output selectors, no shadow resolvers. The only real duplication is a cluster of four
rendering helpers that re-implement a shared core (protocol decode, path walk, dict merge)
instead of delegating — the classic AI-generated "two functions, same spine, different tail"
pattern. All four are in two files and are mechanically extractable without touching the IR
or the SSOT helpers. Fixing DRY-01 and DRY-02 is worthwhile because both straddle the
inline-vs-template-ref rendering split the maintainer specifically flagged: a divergence there
ships wrong text to the model.
