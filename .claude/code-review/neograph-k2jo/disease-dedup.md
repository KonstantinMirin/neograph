# Disease Scan -- dedup lens
**Disease pattern**: Oracle merge/redirect algorithm re-derived inline at call sites instead of being routed through the single canonical merge helper (`make_oracle_merge_fn` / `make_oracle_redirect_fn` in `src/neograph/_oracle.py`).
**Lens**: dedup (find SEMANTIC duplicates — re-implementations of the oracle merge/redirect logic, not legitimate calls to the canonical helper)
**Scan command**: `grep -rn "make_oracle_merge_fn\|make_oracle_redirect_fn\|_merge_one_group\|merge_prompt\|get_merge_fn_metadata\|merge_pre_process\|merge_post_process\|merge_fallback\|_resolve_merge_args" src/neograph` (run from repo root; whole-tree, not diff-anchored). Cross-checked against `src/neograph/_oracle.py` (canonical `make_oracle_merge_fn` body, lines 209-325).
**Total instances (this lens)**: 1

| # | File:line | Form | Disposition | Reason |
|---|-----------|------|-------------|--------|
| 1 | `src/neograph/_wiring.py:303-372` (`_merge_one_group`) | Full inline re-implementation of the Oracle merge algorithm: merge_prompt LLM path (`merge_pre_process` → `invoke_structured` with `merge_model` → `merge_fallback` on exception → `merge_post_process`) **and** merge_fn path (`get_merge_fn_metadata` + `_resolve_merge_args` user-fn, else scripted_lookup fallback). This is a parallel copy of the body of `make_oracle_merge_fn` (`_oracle.py:244-323`). | MIGRATE (ARCH-1 / neograph-s0iz) | Semantic duplicate of the canonical merge helper. Both encode the same merge-decision algorithm. Drift risk is real: `_merge_one_group` does NOT call `_unwrap_oracle_results` / `_build_oracle_merge_result`, so dict-form output unwrapping and the merge-result type-validation guard (`ExecutionError` on wrong merge_fn return type, `_oracle.py:184-191`) are absent on the Each×Oracle group path. It also threads `state=None` into `_resolve_merge_args` (drops `from_state` params) while the canonical fn passes real `state`. Consolidate: have `group_merge_barrier` (`_wiring.py:268-294`) route per-group merges through the canonical helper (e.g. a `make_oracle_merge_fn`-derived per-group closure or a shared extracted `_merge_variants(oracle, output_model, variants, ...)` core that both `make_oracle_merge_fn` and the group barrier call), then delete `_merge_one_group`. |

## Notes
- **Legitimate callers of the canonical helper (NOT instances)** — left as-is:
  - `compiler.py:429-435` (`ModifierCombo.ORACLE` on sub-construct) and `compiler.py:526-529` (node-level Oracle): both correctly call `make_oracle_redirect_fn` + `make_oracle_merge_fn`. Proper single-source usage.
  - `_wiring.py:262-266`: `group_merge_barrier` already constructs a `make_oracle_merge_fn` instance for the construct-level path — but does not use it for the per-group merge (it calls `_merge_one_group` instead). This is the seam where the duplicate lives; the canonical instance is built right next to the inline re-derivation.
  - `factory.py:44-45`: import-only re-export of the helpers.
- **Non-instances (different concern, not merge execution)**:
  - `_construct_validation.py:196-284, 504-508`: validates merge-hook *signatures* (arity) at assembly time. Schema-level, not algorithm re-derivation.
  - `loader.py:254-258`: forwards `merge_prompt`/`merge_model` from spec into the `Oracle` kwargs. Spec plumbing.
  - `decorators.py:168-225, 571-609`: composes `Oracle(...)` from `@node(ensemble_n=/merge_fn=/merge_prompt=)`. Construction sugar, runs through the same IR.
  - `_oracle.py` itself + `_sidecar.py:73` (`get_merge_fn_metadata` def): the canonical source.
- **Why MIGRATE not DEFER**: the duplication is already causing a behavioral divergence (missing dict-form unwrap + missing merge-result type guard on the Each×Oracle group path), which is exactly the class of bug the AGENTS.md "single source of truth" / `effective_producer_type` discipline exists to prevent. ARCH-1 / neograph-s0iz is the consolidation target.
- One-page, single-instance scan: the canonical helper is well-respected everywhere except the Each×Oracle group barrier.
