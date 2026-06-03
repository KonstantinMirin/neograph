# CRIT-01 Elegance Assessment (opus reviewer) — Oracle merge/redirect inline re-derivation

**Scope read**: `src/neograph/_oracle.py` (full), `src/neograph/_wiring.py` (full), `src/neograph/compiler.py` 420–535, `.claude/code-review/020626_2218/synthesis.md`.

## 1. Is the duplication real, and how many sites?
**Two genuine inline-derivation sites + one ghost (dead) site.**
- **Site 1 — `_wiring.py:303-372` (`_merge_one_group`)**: re-implements the full three-branch algorithm of `make_oracle_merge_fn` (`_oracle.py:209-325`): merge_prompt path with pre_process/fallback/post_process hooks, `@merge_fn` metadata path via `get_merge_fn_metadata`, scripted-lookup fallback. Comment at :324 "Symmetric with make_oracle_merge_fn" is a self-indictment. The site that can diverge behaviorally.
- **Site 2 (ghost/dead) — `_wiring.py:262-266`**: `make_oracle_merge_fn(...)` assigned to `merge_fn_impl`, never referenced again (grep → 1 hit). Closure built then discarded; actual merge on :281 calls `_merge_one_group` directly. Wasted allocation per compile.
- **Behavioral gap**: `_merge_one_group` omits the `node_inputs` upstream-context injection that `make_oracle_merge_fn:263-268` performs. An Each×Oracle node with `inputs={'claim':...}` and a `merge_prompt` referencing `${claim}` sees only `{"variants": [...]}` — upstream context silently dropped on the fused path.

## 2. Deeper design problem or incomplete migration?
**Incomplete migration, rooted in a calling-convention mismatch (separable).**
- Standard Oracle path needs `(state, config) -> dict` (LangGraph node fn).
- Each×Oracle fused path needs `(variants, config) -> value` (group barrier pre-extracts variants).
These shapes genuinely differ, so a second entry point is legitimate. The defect: the second entry point RE-DERIVES the algorithm instead of delegating to an extracted pure core. `make_oracle_merge_fn`'s signature does not force re-derivation — it just packages the result in the wrong shape for the fused caller. Fix is one extraction, not a redesign.

## 3. Cleanest consolidation
Extract a pure `_run_oracle_merge(oracle, variants, config, output_model, *, node_inputs=None, state=None, llm_config=None, runtime=EMPTY_RUNTIME, scripted_lookup=None) -> Any` kernel in `_oracle.py` holding the three-branch algorithm ONCE.
- `make_oracle_merge_fn` becomes a thin `(state, config)->dict` wrapper: unwrap collector → `_run_oracle_merge(...)` → `_build_oracle_merge_result(...)`.
- `_merge_one_group` becomes a one-liner delegating to `_run_oracle_merge(...)` with `node_inputs=node.inputs`.
- Delete the dead `merge_fn_impl` assignment (`_wiring.py:262-266`).
- The upstream-context gap closes for free (kernel accepts `node_inputs`/`state`).
Canonical location: `_oracle.py` (already owns make_oracle_merge_fn, _unwrap_oracle_results, _build_oracle_merge_result). `_wiring.py` is topology, not merge logic.

## Disposition feed (for ARCH-1 / neograph-s0iz)
| # | File:line | Form | Disposition | Reason |
|---|-----------|------|-------------|--------|
| 1 | _wiring.py:303-372 (`_merge_one_group`) | inline re-derivation of the 3-branch merge | MIGRATE → s0iz | delegate to extracted `_run_oracle_merge` kernel |
| 2 | _wiring.py:262-266 (`merge_fn_impl`) | dead `make_oracle_merge_fn` call, result discarded | MIGRATE → s0iz | delete dead allocation |
| 3 | _wiring.py:321-328 | missing node_inputs upstream-context injection | MIGRATE → s0iz | behavioral gap; kernel extraction closes it |

_Note: assessment authored by the opus elegance reviewer (feature-dev:code-reviewer); captured to disk by the executor (the reviewer agent type is read-only/no Write)._
