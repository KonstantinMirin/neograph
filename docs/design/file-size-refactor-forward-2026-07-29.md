# File-size refactor proposal: `src/neograph/forward.py`

Date: 2026-07-29
Scope: `src/neograph/forward.py` (1578 lines). Design-only, no code changes.
Context: repo-wide 500-line-per-file cap; active epic neograph-s7zt3 (Phase 8 /
neograph-s7zt3.11, fusion `ModifierCombo` lowering to the Construct level) is
still touching `_agent_spec.py`, `loader.py`, `_wiring.py`, `modifiers.py`,
`factory.py`, `_agent_cycle.py`. `forward.py` is not on that critical path but
is checked here for real overlap per the task brief.

## 1. Responsibility map (as read, top to bottom)

| Lines | Section | What it is |
|---|---|---|
| 1–78 | Module docstring + imports | Design rationale for symbolic-proxy tracing, branch re-trace strategy, try/except limitation. |
| 83–133 | Free helper functions | `_primary_type`, `_attr_chain_after_prefix`, `_over_path_for_proxy` — proxy-name/attr-chain parsing shared by tracer + condition + each-call code. |
| 135–234 | `ForwardConstruct` | The actual public class: MRO-based Node-attribute discovery, `__init__` validation, `forward()` stub. This is the one thing the module is *named* for. |
| 239–528 | Proxy/tracer core | `_Proxy`, `_ConditionProxy`, `_BranchPoint`, `_BranchTrace`, `_Tracer`, `_NodeCall` — the symbolic-execution engine: records calls, branch decisions, loop-iteration mode. |
| 531–1137 | DX builder classes | `_declared_primary_of_body_item`, `_LoopCall`, `_EachCall`, `_ModifierWrapCall` (+ `_EnsembleCall`, `_InterruptCall`) — the `self.loop(...)`/`self.each(...)`/`self.ensemble(...)`/`self.interrupt(...)` sub-construct materializers. **By far the largest cluster (~610 lines, 39% of the file).** |
| 1140–1352 | `_ForwardSelf` | The shim `self` swapped in during tracing; dispatches attribute lookups to `_NodeCall`, hosts `.loop()/.each()/.ensemble()/.interrupt()/.handoff()` builder-factory methods. |
| 1355–1578 | Trace orchestration + branch merge | `_run_trace`, `_apply_loop_modifiers`, `_trace_forward`, `_merge_branch_traces`, `_build_condition_spec`, `_merge_single_branch`, `_merge_sequential_branches` — runs forward() N times (re-trace strategy) and diffs traces into a flat node list with `_BranchNode` sentinels. ~224 lines, functionally self-contained (only touches `_Tracer`/`_BranchTrace`/`_BranchPoint`/`_ConditionSpec`/`_BranchMeta`/`_BranchNode`). |

Note: `_BranchMeta`/`_BranchNode`/`_ConditionSpec` themselves are NOT defined
here — they're imported from `_ir_branch.py` (136 lines) and merely
re-exported. That's already the right pattern; the branch-merge *logic* here
is the untouched twin that still needs a home.

## 2. Extraction candidates

### (A) Trace-orchestration + branch-merge → new `_forward_trace.py`
**Moves**: `_run_trace`, `_apply_loop_modifiers`, `_trace_forward`,
`_merge_branch_traces`, `_build_condition_spec`, `_merge_single_branch`,
`_merge_sequential_branches` (lines 1355–1578, ~224 lines).
**Depends on**: `_Tracer`, `_BranchTrace`, `_BranchPoint` (stay in
`forward.py` or move together), `_ConditionSpec`/`_BranchMeta`/`_BranchNode`
(already external, from `_ir_branch.py`), `op_module` (stdlib `operator`).
No dependency on the DX builder classes (section 531–1137) or on
`ForwardConstruct` itself — `ForwardConstruct.__init__` just calls
`_trace_forward(self, discovered)`, a single call site.
**Line reduction**: ~230 lines removed from `forward.py`; `forward.py` keeps
a one-line `from neograph._forward_trace import _trace_forward` (plus
re-exports for the test-facing names below).

### (B) Proxy/tracer core → new `_forward_proxy.py`
**Moves**: `_Proxy`, `_ConditionProxy`, `_BranchPoint`, `_BranchTrace`,
`_Tracer`, `_NodeCall` (lines 239–528, ~290 lines), plus the three helper
functions in (83–133) since `_ConditionProxy._build_runtime_condition` and
`_Tracer.record_iteration` are their only callers today (the DX builders in
section 531–1137 also call them, so they'd import from here).
**Line reduction**: ~340 lines removed.

### (C) DX builder classes → new `_forward_builders.py` (largest, riskiest)
**Moves**: `_declared_primary_of_body_item`, `_LoopCall`, `_EachCall`,
`_ModifierWrapCall`, `_EnsembleCall`, `_InterruptCall`, and probably
`_ForwardSelf` (its `.loop()/.each()/.ensemble()/.interrupt()/.handoff()`
methods are thin factories over these classes) — lines 531–1352, ~820 lines.
**Line reduction**: potentially removes over half the file.

## 3. SAFE NOW vs DEFER

### SAFE NOW: (A) trace-orchestration + branch-merge extraction
- Pure functions/dataclasses operating on already-built node lists; zero
  coupling to `Node`/`Construct`/`Modifier` construction logic, zero overlap
  with any file Phase 8 touches (`_agent_spec.py`, `loader.py`, `_wiring.py`,
  `modifiers.py`, `factory.py`, `_agent_cycle.py`).
- Single call site (`ForwardConstruct.__init__` → `_trace_forward`) plus
  test-only imports (`tests/test_forward.py` imports `_Tracer` for a few unit
  tests) — the split is a mechanical cut-and-paste with an import-line update
  on each side, no logic touched, no behavior change.
- **Caveat**: `tests/test_forward.py`, `tests/test_ir_protocols.py`,
  `tests/test_coverage_gaps.py` import `_Tracer`, `_ConditionProxy`,
  `_BranchMeta`, `_BranchNode`, `_ConditionSpec` directly from
  `neograph.forward` (not from wherever they're actually defined already —
  `_BranchMeta`/`_BranchNode`/`_ConditionSpec` live in `_ir_branch.py` today
  and are re-exported through `forward.py`, and tests rely on that). Any
  split MUST keep re-exporting the same names from `forward.py` (`from
  neograph._forward_trace import _run_trace, _merge_branch_traces, ...`) so
  existing test imports keep working unchanged — this is the same pattern
  the file already uses for `_ir_branch` symbols, so it's not a new
  precedent, just repeating it once more.
- **This is the single best SAFE NOW recommendation** — smallest blast
  radius, most self-contained, largest safe line-count win relative to risk.

### SAFE NOW (secondary, slightly larger): (B) proxy/tracer core extraction
- Also mechanical and self-contained — nothing outside `forward.py` builds
  a `_Proxy`/`_Tracer` except through `ForwardConstruct`/`_ForwardSelf`
  machinery, and tests only ever `from neograph.forward import _Proxy` etc.,
  which the same re-export pattern preserves.
- Slightly riskier than (A) only because the DX builder classes (section C)
  are its heaviest consumer, so doing (B) without (C) leaves an import
  edge from the (unmoved, still-huge) builders back into the new proxy
  module — harmless, but means (B) alone doesn't shrink the *builders'*
  contribution to the cap problem. Still worth doing; do (A) first since
  it's strictly self-contained, then (B).

### DEFER: (C) DX builder classes (`_LoopCall`/`_EachCall`/`_ModifierWrapCall`/`_EnsembleCall`/`_InterruptCall`/`_ForwardSelf`)
- This is where the real design work is, and it's also exactly the surface
  Phase 8 (fusion `ModifierCombo` lowering to the Construct level) is likely
  to add to or change semantics around — these classes are the tracer-side
  implementation of "wrap a node or node-list with a modifier, inferring the
  sub-construct's input/output ports and naming it deterministically,"
  which is conceptually the same operation Phase 8 is extending. Moving 820
  lines to a new module right before/during Phase 8 touches the same
  concepts under a different name at the exact moment the epic needs a
  stable target — high conflict/rebase risk, not "mechanical."
- Needs its own design pass: what's the module boundary (one file for all
  four builder kinds, or one per modifier kind mirroring
  `_ModifierWrapCall`'s already-generic subclass split)? Does `_ForwardSelf`
  move with it or stay as a thin dispatcher in `forward.py`? Do this AFTER
  Phase 8 lands, once the fusion lowering shape at the Construct level is
  settled, so the new module boundary can absorb whatever Phase 8 adds
  instead of being invalidated by it.

## 4. Duplication found (real, not superficial)

**`forward.py`'s sub-construct materialization pattern vs `loader.py`'s spec-driven materialization is a genuine, independently-implemented duplicate of the same operation.**

Both files implement, separately:
1. Copy-not-mutate: fill a body node's `inputs=` from the inferred input
   type only when `inputs is None` (`_LoopCall._materialize` /
   `_EachCall._build` / `_ModifierWrapCall._materialize` in `forward.py`
   lines 662–679, 867–871, 1087–1091 — vs. `loader.py`'s equivalent
   spec-to-`Construct|Modifier` builders around lines 486, 530, 547, 1377–1391).
2. Deterministic occurrence-slug naming (`f"{kind}-{body_slug}"`, bumped by
   an occurrence counter on duplicate slugs) — `forward.py`'s
   `_Tracer.next_occurrence` (lines 427–439) has no counterpart surfaced in
   `loader.py`, but the naming *goal* (stable, deterministic sub-construct
   names across repeated builds) is solved twice with different mechanisms.
3. `Construct(input=, output=, nodes=[...]) | Modifier(...)` assembly with
   inferred boundary ports — `forward.py`'s three `_materialize*` methods
   vs. `loader.py`'s `base_node | Oracle(...)` / `inner | Each(...)` /
   `body | Loop(...)` (lines 486, 530, 547) and the mesh-building loop
   around `loader.py:1377–1391`.

This is not "two files that both call `Construct(...)`" — it's two
hand-rolled implementations of the same "build me a modifier-wrapped
sub-construct from a body + inferred port type" operation, one driven by
Python tracing (`forward.py`), one driven by parsed Agent-Spec YAML
(`loader.py`). A shared `_construct_wrap.py` (or similar) helper could
collapse both, but **this is squarely `loader.py` territory, which Phase 8
is actively changing (fusion `ModifierCombo` lowering)** — do not touch
either side until Phase 8 lands and the lowering shape stabilizes. Name it
as a follow-up design item, not a task to start now.

No other real duplication found against `_agent_spec.py`, `_wiring.py`,
`modifiers.py`, `factory.py`, or `_agent_cycle.py` — `forward.py`'s
`_ForwardSelf.handoff()` (Portal mesh recording) is explicitly documented
in its own docstring as deliberately matching `examples/28_portal_swarm.py`'s
declarative shape byte-for-byte, which is intentional twin-IR parity, not
duplication to collapse.

## 5. Summary

| Extraction | Lines removed | Classification |
|---|---|---|
| (A) trace-orchestration + branch-merge → `_forward_trace.py` | ~230 | SAFE NOW |
| (B) proxy/tracer core → `_forward_proxy.py` | ~340 | SAFE NOW (do after A) |
| (C) DX builder classes → `_forward_builders.py` | ~820 | DEFER — until Phase 8 lands |
| loader.py/forward.py sub-construct-wrap duplication | n/a (cross-file) | DEFER — needs dedicated design pass, blocked on Phase 8 |

Doing (A) + (B) takes `forward.py` from 1578 lines to roughly **1578 − 230 −
340 ≈ 1008 lines** — still over the 500-line cap, but removes the two safe,
self-contained clusters without going near Phase 8's working set. Getting
under 500 requires (C), which is explicitly DEFERred.
