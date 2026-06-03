# Codebase Disease Scan — neograph-s0iz (ARCH-1, CRIT-01)

**Disease**: Oracle merge/redirect algorithm re-derived inline at call sites instead of routed through the single canonical merge helper in `src/neograph/_oracle.py`.
**Method**: targeted in-loop multi-agent scan (`code-review --targeted`) was run on this EXACT disease as the PROC-3 probe — full per-lens artifacts live at `.claude/code-review/neograph-k2jo/` (disease-dedup.md, disease-consistency.md, disease-layering.md, elegance.md). Per the task directive, this atom VALIDATES those findings against current HEAD rather than re-running the fan-out.
**Scan command (reproducible)**: `grep -rn "make_oracle_merge_fn\|make_oracle_redirect_fn\|_merge_one_group\|merge_prompt\|get_merge_fn_metadata\|invoke_structured" src/neograph`
**Validation vs HEAD**: all 3 sites still present — `_wiring.py:262` (merge_fn_impl), `:324` ("Symmetric with" comment), `:328` (`{"variants": variants}` with no node_inputs). Table current.

## Disposition table (validated)

| # | File:line | Form | Disposition | Reason |
|---|-----------|------|-------------|--------|
| 1 | `_wiring.py:303-372` (`_merge_one_group`) | full inline re-implementation of the 3-branch merge dispatch | MIGRATE | extract canonical `_merge_variants` kernel in `_oracle.py`; delegate |
| 2 | `_wiring.py:262-266` (`merge_fn_impl`) | dead `make_oracle_merge_fn(...)`, result discarded (grep → 1 hit, no use) | MIGRATE | delete dead allocation |
| 3 | `_wiring.py:328` | merge_prompt path drops `node_inputs` upstream-context injection (vs `_oracle.py:264-268`) | MIGRATE | behavioral gap on Each×Oracle; kernel + barrier upstream_context closes it |

ALLOWLIST: ~100 other grep hits (compiler.py wiring sites, factory/loader/decorator imports, the canonical `group_merge_barrier` → `make_oracle_merge_fn`/`_merge_one_group` calls, the produce-mode `invoke_structured` node path, redirect-side `make_*_redirect_fn` sites) — confirmed legitimate canonical usage by all three probe lenses.
DEFER: none. (Separate documented limitation, NOT in scope: `_merge_one_group` merge_fn path passes `state=None` to `_resolve_merge_args` so `from_state` DI params don't resolve on the fused path — orthogonal to the merge-step consolidation; left for a future ticket if it ever bites.)

## Cross-lens convergence (from probe)
3 lenses + opus all converged on the single `_wiring.py:303-372` site → disease is bounded, not diffuse. Design need only consolidate one site.

_See `.claude/code-review/neograph-k2jo/` for the full per-lens reports and the opus elegance assessment (which recommended the `_run_oracle_merge` kernel extraction this ticket's design adopts as `_merge_variants`)._
