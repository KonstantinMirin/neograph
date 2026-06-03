# Disease Scan -- layering lens
**Disease pattern**: Oracle merge/redirect algorithm re-derived inline at call sites instead of being routed through the single canonical merge helper (`make_oracle_merge_fn` / `make_oracle_redirect_fn` in `src/neograph/_oracle.py`).
**Lens**: layering = flag instances where the inline merge/redirect logic (or its consolidation fix) sits in or crosses a layer boundary it should not. The merge ALGORITHM (pre-process -> merge-fn / merge-prompt -> post-process -> fallback -> output validation) belongs in the `_oracle.py` helper, NOT embedded in compiler/`_wiring.py` topology code or factory runtime dispatch.
**Scan command**:
```
grep -rn --include='*.py' "invoke_structured\|merge_fallback\|merge_pre_process\|merge_post_process\|get_merge_fn_metadata\|_resolve_merge_args\|variants\[0\]\|scripted_merge\|_merge_one_group" src/neograph/ | grep -vE "_oracle\.py|/tests/"
```
(Supplementary topology grep: `grep -rn "_merge_one_group\|make_oracle_merge_fn\|make_oracle_redirect_fn\|invoke_structured\|merge_prompt" src/neograph/_wiring.py src/neograph/compiler.py src/neograph/factory.py`)
**Total instances (this lens)**: 1

| # | File:line | Form | Disposition | Reason |
|---|-----------|------|-------------|--------|
| 1 | `src/neograph/_wiring.py:303-372` (`_merge_one_group`) | Full re-implementation of the Oracle merge algorithm inside the compiler/wiring layer. Re-derives both branches that `_oracle.py::make_oracle_merge_fn` already owns: (a) `merge_prompt` LLM-judge branch — `merge_pre_process` (322), `invoke_structured` (332), `merge_fallback` (343), `merge_post_process` (349); (b) `merge_fn` branch — `get_merge_fn_metadata` + `_resolve_merge_args` (354-361), scripted-registry lookup + `ConfigurationError` (364-369); (c) empty-fallback `variants[0]` (372). | **MIGRATE** (tracked: ARCH-1 / neograph-s0iz) | Layer violation: the merge ALGORITHM lives in `_wiring.py`, which is compiler-layer topology code (its job is `Send`/barrier/edge wiring, per `_add_each_oracle_fused`). Algorithm ownership belongs in `_oracle.py`. The code's own comment (line 324) admits it is "Symmetric with make_oracle_merge_fn (construct-level)" — an explicit parallel-maintenance hazard. Two sites can drift in pre/post/fallback semantics (the construct-level path at `_oracle.py:251-323` already auto-injects upstream `node_inputs` context and unwraps secondaries via `_unwrap_oracle_results`/`_build_oracle_merge_result`; the fused `_merge_one_group` path does neither and threads no `from_state` DI). Fix per ARCH-1 Phase 2: extract `_run_merge_prompt(...)` / `_run_merge_fn(...)` in `_oracle.py`, have both `make_oracle_merge_fn` (single-group) and `_merge_one_group` (per-group) call them; `_merge_one_group` shrinks to a grouping wrapper. |

## Notes

- **Single instance, not a swarm.** The whole-tree scan surfaced one — and only one — site that re-derives the merge algorithm outside `_oracle.py`: `_merge_one_group` in `_wiring.py`. Every other grep hit is layer-legal and is **not** counted:
  - `src/neograph/_oracle.py` — the canonical home (excluded by scan, by design).
  - `src/neograph/_sidecar.py:73`, `_di_classify.py:228-237` — *definitions* of the shared helpers (`get_merge_fn_metadata`, `_resolve_merge_args`). Both `_oracle.py` and `_wiring.py` import these; the helpers themselves are correctly factored. Not a duplication.
  - `src/neograph/_dispatch.py:143`, `_llm.py:128` — the produce-mode `invoke_structured` entry point and its node-dispatch caller. This is the ordinary node LLM path, not Oracle merge re-derivation. Layer-correct (factory/runtime calling the LLM service).
  - `src/neograph/modifiers.py:403-405,428` — Oracle dataclass field declarations + a config-validation guard (merge_fn vs merge-hooks mutual exclusion). IR-layer schema, not algorithm.
  - `src/neograph/_construct_validation.py:196-284,453-455`, `lint.py:248` — assembly-time *validation* of merge-hook signatures and merge_fn registration. IR/lint layer; inspects the hooks, does not execute the merge. Layer-correct.
  - `src/neograph/decorators.py:171-236,257-612` — DX layer plumbing the `merge_*` kwargs straight into the `Oracle(...)` IR object. Pure pass-through, no algorithm. Layer-correct.
  - `src/neograph/compiler.py:30` — merely *imports* `_merge_one_group` from `_wiring`; the compiler delegates and does not itself re-derive. The redirect/merge construction at `compiler.py:429-535` correctly routes through `make_oracle_redirect_fn` / `make_oracle_merge_fn`. Layer-correct.
  - `src/neograph/di.py:60` — docstring reference only.

- **Redirect side is clean.** `make_oracle_redirect_fn` / `make_eachoracle_redirect_fn` / `make_each_redirect_fn` are the single source for redirect generators; `_wiring.py` and `compiler.py` consume them as callables and never re-derive the redirect algorithm. No redirect-side disease instances.

- **Why this is a layering finding, not just DRY.** `_wiring.py` is compiler-layer code whose responsibility is LangGraph topology (Send fan-out, defer barriers, edges — see `_add_each_oracle_fused`). Embedding the LLM-judge/scripted-merge algorithm there pulls a runtime/algorithm concern up into the wiring layer and forces `_wiring.py` to take function-local imports of `invoke_structured` (line 319) and decorator-layer DI helpers (line 313) that the canonical helper in `_oracle.py` already owns. Routing through `_oracle.py` keeps the algorithm in one layer and leaves `_wiring.py` doing only grouping + topology.

- **Disposition rationale (MIGRATE):** doable now, single site, already designed and tracked under ARCH-1 / neograph-s0iz (OPEN, P1) with a concrete Phase-2 extraction plan. Not DEFER — the fix is well-scoped and the consolidation ticket is unblocked. Not ALLOWLIST — there is no legitimate reason for the algorithm to live in the wiring layer; the only argument for the inline copy (per-group vs single-group) is exactly the parameterization ARCH-1 prescribes.
