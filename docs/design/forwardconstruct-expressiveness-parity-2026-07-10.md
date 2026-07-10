# Design spike: ForwardConstruct expressiveness parity with the declarative form

**Date:** 2026-07-10
**Status:** spike — feature scoping, phased. Recommendation: ship Phase 1.
**Found via:** the 0.6.0 consumer-DX audit (ox cmu.9 — ForwardConstruct fits *none* of the cascade's shape) + maintainer direction ("it's unused because it's very limited; the topologies it can express are a tiny subset — fix that").

## The invariant

**Every topology the declarative `Construct` form can express, `ForwardConstruct.forward()` must also express — tracing to IDENTICAL IR.** The `forward()` module docstring already promises this at the node-list level (`forward.py:15`: "the resulting node list is identical to what a declarative `Construct(nodes=[...])` produces"); this spike extends that promise from straight-line pipelines to the full modifier/composition surface. Parity is *bounded by* declarative capability — ForwardConstruct need not exceed what the declarative form supports (e.g. Each×Oracle fusion on sub-constructs is a documented declarative non-support, `compiler.py:474-476`, and stays out of scope for both).

## Why this matters (the diagnosis)

The imperative "write the graph as Python control flow" DX is compelling — but ForwardConstruct appears in **zero apps** (only demo example 15). Not because it's unwanted: because its expressible topology set is a tiny subset (straight-line + per-node fan-out keyed on `"label"` + node-only loops + constant-comparison branches). Real agent pipelines need fan-out-inside-loop over sub-constructs, ensembles, and HITL — none expressible today. **The fix is parity, not new IR:** the tracer is artificially stricter than the IR it targets.

## How the tracer works (torch.fx symbolic proxy)

`forward()` runs with a shim `self` (`_ForwardSelf`) and a seed `_Proxy`. Node calls record into a `_Tracer` and return output proxies; attribute access returns child proxies; `<`/`>`/`==` return `_ConditionProxy`; `if` triggers `_Proxy.__bool__` → the **re-trace** branch strategy (2^N passes, `_MAX_BRANCHES=8`); `for x in proxy` triggers `__iter__` → loop-mode → `_apply_loop_modifiers` attaches `Each`; `self.loop(body=[...])(x)` builds a bounded sub-construct + `Loop`. The result is a `list[Node | Construct]` (+`_BranchNode` sentinels) handed to `Construct.__init__`.

## The parity gap (capability × coverage × difficulty)

| Capability (declarative) | ForwardConstruct today | Gap | Difficulty | Phase |
|---|---|---|---|---|
| Straight-line `@node` edges | ✅ full | — | — | shipped |
| Fan-in (multiple upstream → one node) | ⚠️ via Node.inputs | edges come from `Node.inputs`, not `forward()` args — **verify** multi-input wiring | verify | 1 |
| Fan-out `Node \| Each(over, key)` | ⚠️ per-node, **key hardcoded `"label"`** (`:652`) | custom key; `on_error='collect'` | small | 1 |
| Fan-out over a **sub-construct** (`verify_construct \| Each(over, key="id")`) | ❌ per-node only | wrap a multi-node body into one Each'd sub-construct | small–med | 1 |
| `Loop` over nodes | ✅ `self.loop(body=[nodes])` | — | — | shipped |
| `Loop` over a **sub-construct body** (loop containing an Each fan-out) | ❌ `self.loop()` rejects non-`_NodeCall` bodies (`:491-498`) | accept sub-construct body items; nest | small–med | 1 |
| Arbitrary **sub-construct composition/nesting** | ⚠️ only implicitly via `self.loop()` | declare/compose a sub-construct as a unit; nest freely | med | 1 |
| Multi-output dict nodes (`outputs={"result":…, "tool_log":…}`) | ⚠️ proxy attr access | **verify** `{node}_{key}` wiring through tracing | verify | 1 |
| `Oracle` (ensemble_n + merge) | ❌ **entirely absent** (no `self.ensemble()`; Oracle not imported) | new tracing surface | med | 2 |
| `Operator` / `interrupt_when` (HITL) | ❌ not exposed in `forward()` | new surface | med | 2 |
| `skip_when` / `skip_value` | ⚠️ node-level kwarg on the attr | **verify** it survives tracing | verify | 2 |
| Branching `if/else` | ⚠️ re-trace, **constant comparisons only**, max 8 (`:33-36`) | richer conditions; scale | med–hard | 3 |
| `try/except` fallback | ❌ dead-code during trace (`:38-47`) | fundamentally hard under symbolic tracing | hard | 3 (or cap) |

## The load-bearing fixes (Phase 1 — ship this)

Both blockers for real topologies are "teach the tracer to emit what the IR already accepts," with `self.loop()` as the working blueprint:

1. **`self.each(body=[...], over=..., key=...)`** — mirror `_LoopCall`/`self.loop()` (`:462-621`): build `Construct(input=, output=, nodes=body) | Each(over, key, on_error=)`, `record_construct`, return a proxy. ~40 lines, a copy-adapt. Keeps the bare `for x in proxy` form as sugar for the trivial single-node/`key="label"` case.
2. **Relax `self.loop()` body validation** (`:491-498`) to accept sub-construct-producing items (`_LoopCall`/`_EachCall` results) alongside `_NodeCall`. `Construct(nodes=...)` already takes `list[Node | Construct]`; reuse the `_declared_output` port inference (`:519`) for mixed bodies. The only fiddly part is input/output type inference on a body whose members are modified sub-constructs.

Neither touches the IR or compiler.

## The hard / orthogonal items (Phase 3 — decide: close or cap)

Branch richness and `try/except` fallback are the genuinely hard parts (2^N re-trace has inherent limits; symbolic tracing can't observe the except arm). **Crucially, they do NOT block the topologies that make ForwardConstruct unused** — the cascade's "branch" is an LLM output field (`Diagnosis.data_status`), not graph control flow. Recommendation: keep them as documented v1 limits with a **loud escape to the declarative form**, rather than force a fragile 2^N expansion. Parity's spirit is met when every *reasonable* declarative topology has a forward() twin; the escape hatch covers the residue.

## Acceptance — one verifiable principle

**Traced IR == declarative IR, per topology class.** For each capability, a `forward()` twin must produce the byte-identical IR (graph-lint diff clean) of the equivalent declarative `Construct`. The **reference integration test is the cascade shape** (intake → triage → `Loop(investigate{ verify | Each(over, key="id") })` → explain): a ForwardConstruct expressing it must trace to the same IR the declarative ox cascade compiles. This is unambiguous — no behavioral judgment call.

## Anti-band-aid

Do NOT special-case the cascade. The deliverable is **parity, enforced by a parity test matrix**: a corpus of declarative topologies, each with a forward() twin, asserted IR-identical. The permanent guard is a property test comparing traced IR to declarative IR across that corpus — so a new declarative capability that lacks a forward() path fails loudly (the parity ratchet). New tracer surfaces (`self.each`, future `self.ensemble`) must build existing IR, never introduce ForwardConstruct-only IR.

## Phasing & effort

- **Phase 1 (ship): fan-out over sub-construct + custom key, loop-over-sub-construct body, free nesting, + verify fan-in/multi-output.** ~2–3 days, entirely in `forward.py`, zero IR/compiler changes. Unlocks the cascade-class topologies — the ones real agents use.
- **Phase 2: Oracle + Operator/interrupt surfaces.** Medium — new `self.ensemble()` / HITL tracing that build the existing modifiers.
- **Phase 3: branch richness / try-except — close or cap** with the declarative escape. Hardest, lowest marginal value.

## Independent-review gate

The DSL-surface design (bare `for` sugar vs explicit `self.each()` builder; how `self.loop()`/`self.each()` compose and nest; whether to keep the `key="label"` default) is an API decision — confirm with a fresh-context reviewer before Phase 1 implementation. The parity test matrix must be reviewed for coverage (does every declarative modifier/composition have a twin?).
