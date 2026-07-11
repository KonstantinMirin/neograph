# Archaeology: how `Loop.history` / `loop_history=` became redundant

**Date:** 2026-07-11
**Trigger:** the d5pvl three-surface-parity review surfaced that `loop_history=`'s `@node` tests were construction-only; runtime investigation found the feature works but appears redundant with the self-loop's main output field. This doc reconstructs *how it came to be* before deciding keep-vs-remove.

## One-sentence answer

`Loop.history` was **born redundant** — a schema-first speculative field superseded on its birth-day (2026-04-07) by the main output field's `_append_loop_result` accumulator — and it **survived by passivity** while its speculative siblings (`reenter`, `loop_to`) were pruned, because every subsequent touch was a mechanical propagation (wire-the-write, forward-the-kwarg) that inherited the field without re-auditing whether it earns its place.

## The commit timeline (all 2026-04-07 — Loop's birth-day)

| Time | Commit | What it did |
|---|---|---|
| 03:53 | `057689b` feat: Loop modifier class + export (ba8a) | Spec'd `Loop(when, max_iterations, reenter, on_exhaust, history)` — **"Schema only — no compiler/factory wiring yet."** `history` is a speculative field in a top-down field set. |
| 03:59 | `345275f` feat: Loop state model, validator, @node kwargs (04ol/8eid/y6wh) | Added `neo_loop_count` + **`neo_loop_history`** state fields; kwargs `loop_when=/max_iterations=/loop_to=/on_exhaust=` — **note `loop_history=` is NOT among them.** The history *state field* exists from day one but is unexposed. |
| 11:32 | `5083d5f` feat: Loop append-list reducer + self-loop working (bd3z) | Wired the self-loop via **`_append_loop_result` on the MAIN output field** — accumulates every iteration as a list. Message: **"Self-loop test passes: 3 iterations … history preserved."** The main field is understood, that day, AS the history mechanism. |

The redundancy is created at 11:32: the working implementation delivers per-iteration history through the main field's append reducer, making the separately-spec'd `neo_loop_history` field a duplicate — while it is still schema-only (not yet written).

## How the speculative field set was pruned — unevenly

The 03:53 set was `{when, max_iterations, reenter, on_exhaust, history}`. As the architecture matured:

- **`reenter` / `loop_to` → REMOVED.** They were the multi-node-cycle mechanism (`revise | Loop(reenter="review")`). When the design pivoted to *sub-constructs-with-Loop* for multi-node bodies, they were deleted — `_construct_validation.py:253`: *"Loop reenter validation removed — Loop.reenter no longer exists."* `loop_to=` has zero hits today.
- **`on_exhaust` → KEPT (earned it).** Live across `_wiring`, `loader`, `_spec_schema`, `forward`, `decorators`, the JSON schema, and `testing/scaffold` — a real behavior (error vs last-value on exhaustion).
- **`history` → NEITHER pruned NOR justified.** It survived by passivity, then got re-activated twice by tasks whose remit was propagation, not audit:
  1. `8bb9f80` (pgso — *"architectural rewrite of factory.py"*) wired the actual write `if loop_mod.history: update[neo_loop_history_{node}] = result` — because the field existed.
  2. `e67fa03` (d5pvl, 2026-07-11, three months later) exposed `loop_history=` as a `@node` kwarg *"for three-surface parity"* — because the modifier field existed and parity demanded symmetry. Construction-only tests, because the parity task's job is "forward the kwarg," not "does this feature earn its place."

## Why it's redundant (mechanically)

`history` is legal ONLY on a **Node self-loop** (banned on Constructs — `modifiers.py:266` *"Loop(history=True) is not supported on Constructs"*). On that exact surface, the main output field already accumulates every iteration:
- Main field: `list[output_type]` via `_append_loop_result` (`state.py:558`) → in the run **output** (`result[node]`).
- History field: `Annotated[list, _concat_reducer]` (`state.py:238`) → `neo_*` **internal**, get_state-only.

Both reducers append each single-value write; for the loop case they collect identical data. The only real difference is *accessibility* (output dict vs `get_state()`), not content. So on its only legal surface, `loop_history` duplicates data the loop already surfaces for free.

## Runtime evidence (now pinned)

`tests/test_loop.py::TestLoopHistoryRuntimeAndRedundancy` (added 2026-07-11) makes the redundancy provable rather than inferred:
- `test_loop_history_collects_every_iteration_at_runtime` — the previously-missing e2e: history=True genuinely collects via `get_state()`.
- `test_loop_history_field_duplicates_the_main_append_field` — main field == history field, per-iteration.
- `test_main_append_field_alone_preserves_history_without_the_flag` — WITHOUT the flag, `result[node]` still has every iteration; the history field isn't even created.

## The decision (keep vs remove) — tracked separately

- **Remove** (leaning): at 0.x with no compat burden, delete `Loop.history`, the `neo_loop_history` field + write, the `_append`-vs-`_concat` duplication, and the `loop_history=` kwarg + docs. The self-loop's `result[node]` list is the single, output-visible history.
- **Keep** (needs a real justification): the only distinct value is an *audit trail readable via `get_state()` without consuming the node's output* — plausible for observability, but currently unexemplified and undocumented as such. If kept, it MUST gain: an example showing why you'd use it over `result[node]`, and the runtime tests above (done).

The removal decision should also re-check whether `result[node]` being a *list of all iterations* (not the last value) is itself the intended terminal-output shape, or a second thing worth revisiting — it is the append reducer's documented behavior (`test_loop.py` TestSelfLoop) but is a sharp edge for consumers expecting the final value.

## The pattern worth remembering

Schema-first speculative field sets (enumerate everything a modifier "should" have, wire later) leave orphans when the implementation picks a different mechanism. Pruning caught `reenter`/`loop_to`; `history` slipped because it was passively harmless and each later touch propagated it mechanically. **A parity/wiring task is the wrong place to catch this** — its remit is symmetry, not justification. Redundancy this shape only surfaces when someone runs the feature end-to-end and sees the same data twice.
