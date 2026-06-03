# Targeted Disease Scan — synthesis (probe neograph-k2jo)

**Disease**: Oracle merge/redirect algorithm re-derived inline at call sites instead of routed through the single canonical merge helper (`make_oracle_merge_fn` / `make_oracle_redirect_fn` in `src/neograph/_oracle.py`).
**Mode**: `code-review --targeted` — 3 disease-scoped lenses (dedup / consistency / layering) + opus elegance reviewer, parallel.
**Elapsed**: 267s (~4.5 min, within the <5 min budget; the full Jun-2026 audit was 7 agents whole-codebase ≈ 3 min/agent serial-ish).
**Scan command (lenses converged on)**: `grep -rn "make_oracle_merge_fn\|make_oracle_redirect_fn\|_merge_one_group\|merge_prompt\|get_merge_fn_metadata\|invoke_structured" src/neograph`

## Cross-lens signal
All three lenses + the opus reviewer independently converged on a SINGLE site. That convergence is the strongest possible result — it means the disease is bounded, not diffuse.

## Disposition table (merged, deduped by file:line + form)

| # | File:line | Form | Disposition | Reason |
|---|-----------|------|-------------|--------|
| 1 | `_wiring.py:303-372` (`_merge_one_group`) | full inline re-implementation of the 3-branch merge-dispatch (merge_prompt + pre/post/fallback hooks, `@merge_fn` metadata, scripted fallback) — comment :324 "Symmetric with make_oracle_merge_fn" | MIGRATE → neograph-s0iz (ARCH-1) | extract a shared `variants→merged` kernel in `_oracle.py`; `_merge_one_group` delegates |
| 2 | `_wiring.py:262-266` (`merge_fn_impl`) | dead `make_oracle_merge_fn(...)` call, result discarded (grep → 1 hit) | MIGRATE → neograph-s0iz | delete dead allocation |
| 3 | `_wiring.py:321-328` | missing `node_inputs` upstream-context injection vs `make_oracle_merge_fn:263-268` | MIGRATE → neograph-s0iz | behavioral gap on the Each×Oracle path; kernel extraction closes it for free |

ALLOWLIST: all other ~40 grep hits (compiler.py wiring sites, factory/loader/decorator imports, the canonical `group_merge_barrier` call, redirect-side sites) are legitimate canonical-helper usages — confirmed across all three lenses.
DEFER: none new — neograph-s0iz (ARCH-1, OPEN, P1) already tracks this exact consolidation with a Phase-2 extraction plan.

## Probe verdict
The in-loop targeted scan works end-to-end: parallel fan-out → per-lens artifacts in `.claude/code-review/<ticket-id>/` → one-page disposition table, under the 5-min budget, producing actionable output (a 3-row disposition table that feeds ARCH-1 directly) rather than a 41-finding catalog. Cross-lens convergence on one site demonstrates the disease-scoped lenses add signal over a single grep (the dead-call and upstream-context-gap rows were surfaced by dedup/elegance, not by the naive grep).

_Caveat: in this session the lenses ran via `general-purpose` agents carrying the `review-disease-scan` contract inline, because the newly-added `agents/review-disease-scan.md` type is not in the running session's agent registry (plugin agents load at session start; the new agent becomes invocable as `subagent_type: review-disease-scan` after a plugin/session reload). The wiring guard + SKILL doc + formula references are correct; only the live agent-type registration awaits reload._
