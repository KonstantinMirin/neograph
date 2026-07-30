# File-size refactor design: `src/neograph/_agent_cycle.py`

Date: 2026-07-29
Scope: `src/neograph/_agent_cycle.py` (1054 lines) only, as part of the repo-wide
500-line-per-file cap effort. Read-only research; no code changed.

Related active epic: neograph-s7zt3 (Agent Spec / Portal rebuild), open child
neograph-s7zt3.11 (Phase 8, fusion ModifierCombo → Construct-level export
lowering). **Verified via `bd show neograph-s7zt3.11`: Phase 8's remaining scope
is `_agent_spec.py` export-lowering (`_lower_each`/`_lower_loop`/`_lower_oracle`)
and the Agent-Spec-representation question for Construct-level Oracle — it does
NOT modify `_agent_cycle.py`'s runtime bodies.** The only coupling is read-only:
Phase 8's corrected scope note explicitly cites `_agent_cycle.py`'s existing
`config.get('configurable', {}).get(StateKeys.ORACLE_MODEL_OVERRIDE, node.model)`
read (line 142, inside `_turn_prep_kwargs`) as *already correct and shared*,
consumed by `_wire_oracle`/`_inject_oracle_config`. So the risk profile here is
lower than "actively edited every day" — it's "referenced as a stable contract
by a neighboring file that Phase 8 IS actively editing." That argues FOR doing
safe, mechanical extractions now (fewer lines for Phase 8's own file-size
pressure to spill into), not against it.

## 1. Responsibility map

The file's docstring (lines 1-30) is accurate: it owns the three agent/act
per-superstep node *bodies* (`{node}__agent`, `{node}__tools`, `{node}__parse`)
and the router between them, plus the tool-approval-gate body. Concretely, six
clusters:

| Lines | Cluster | What it is |
|---|---|---|
| 87-102 | `AgentCycleNames` / `cycle_names` | Tiny naming helper — the 3 child-node names for a node. Imported by `_wiring.py` and `factory.py` too. |
| 104-238 | **Turn-prep infra** | `_TurnPrep` dataclass, `_turn_prep_kwargs`, `_build_turn_prep`/`_abuild_turn_prep` (sync/async twins), `_init_budget`, `_maybe_skip`, `_tracker_from_budget`. Rebuilds the tool-loop preamble once per superstep. |
| 241-287 | **Portal tool-triggered-handoff synthesis** | `_HANDOFF_TOOL_PREFIX`, `_handoff_targets`, `_synthesize_handoff_tools`. Builds the ephemeral `transfer_to_<peer>` StructuredTools for a tool-triggered Portal member. |
| 290-420 | **Agent-turn shared skeleton** | `_agent_caller`, `_agent_working_messages`, `_record_turn_usage`, `_total_calls`, `_emit_limit_event`, `_emit_guard_forced_break`, `_obs_type_name`, `_agent_start_log`, `_agent_turn_prelude`, `_agent_turn_finalize`. The sync/async `agent_body` twins' shared pre/postamble. |
| 422-624 | **Per-tool-call handling (pure functions)** | `_raise_sync_tool_async`, `_tool_call_precheck`, `_idempotent_repeat_key`, `_seed_repeat_cache`, `_build_tool_interaction`, `_handoff_ack`, `_record_tool_result`, `_ainvoke_tool_timed`, `_lift_resource_refs`. Explicitly documented as "the DRY-01 extraction" — already recognized internally as a separable cluster. Depends only on `ToolMessage`, `ToolBudgetTracker`, `ToolInteraction`, `ResourceRef`/`ProducingCall`, `_content_blocks` helpers, `_tool_loop._render_tool_result_for_llm`/`_unparseable_args_raw`. No dependency on `Node`, `_TurnPrep`, or closures. |
| 627-969 | **`make_agent_cycle_bodies`** (the factory) | The actual closures: `agent_body`/`aagent_body`, `router`, `tools_body`/`atools_body` (+ nested `_tools_guards`/`_limit_messages`/`_tools_prelude`/`_tools_result`/`_run_tool_calls`/`_arun_tool_calls`), `parse_body`/`aparse_body` (+ `_finish_and_shape`). ~340 lines, all closing over `node`, `field`, `msgs_key`, etc. |
| 972-1054 | **Tool-approval gate** | `_gate_approved`, `make_tool_gate_bodies` (+ nested `_pending_tool_calls`/`_denial_messages`/`gate_body`/`gate_router`). Self-contained: only needs `Node`, `StateKeys`, `field_name_for`, `cycle_names`, `ToolMessage`, `interrupt`. No dependency on any other cluster in this file. |

## 2. Proposed extractions

### (A) `_agent_tool_calls.py` (new module) — the per-tool-call pure-function cluster
**Moves**: lines 422-624 (`_raise_sync_tool_async` through `_lift_resource_refs`),
~200 lines including docstrings. These are pure functions taking explicit args
(`tc: dict`, `tracker`, `result`, `renderer`, etc.) — zero closure capture, zero
dependency on `make_agent_cycle_bodies`'s local state. `_agent_cycle.py` would
import them back (`_tool_call_precheck`, `_idempotent_repeat_key`,
`_seed_repeat_cache`, `_build_tool_interaction`, `_handoff_ack`,
`_record_tool_result`, `_ainvoke_tool_timed`, `_lift_resource_refs`,
`_raise_sync_tool_async`) for use inside `_run_tool_calls`/`_arun_tool_calls`.
**Removes ~200 lines** from `_agent_cycle.py` (1054 → ~854).

Why here and not `_tool_loop.py`: `_tool_loop.py` is already 727 lines (would
become ~930, still over cap) and its actual concern (prompt/turn preparation,
final-parse) is one layer up from per-call tool execution mechanics; a new
sibling module keeps `_tool_loop.py`'s own line count moving in the right
direction instead of the wrong one.

### (B) `_agent_gate.py` (new module) — the tool-approval gate
**Moves**: lines 972-1054 (`_gate_approved`, `make_tool_gate_bodies`), ~83 lines.
Fully self-contained per the map above; its only shared symbol with the rest of
the file is `cycle_names`, which is a one-line import. **Removes ~83 lines.**

### (C) Portal handoff-tool synthesis — DEFER, do not extract alone
Lines 241-287 (`_handoff_targets`/`_synthesize_handoff_tools`) plus `_handoff_ack`
(already counted in bucket A, lines 537-552) are logically one feature
("tool-triggered Portal handoff") but currently split across two of the six
clusters above, and `handoff_target` plumbing threads through
`_run_tool_calls`/`_arun_tool_calls`/`_tools_result` in the factory closures too.
Pulling only the 47-line synthesis piece out leaves the feature more scattered,
not less. If Portal-handoff mechanics get their own module later, do it as one
pass covering the synthesis + `_handoff_ack` + the plumbing through
`make_agent_cycle_bodies`, together with `factory.py`'s
`make_portal_agent_cycle_tool_handoff_fn` which is the same feature's other
half. That is a DEFER-bucket item (see below), not a SAFE-NOW one.

### (D) `make_agent_cycle_bodies` itself — DEFER
This is the 340-line core: three node bodies + router, all closures over
`node`/`field`/`*_key` locals. Splitting it would mean either (a) turning every
closure-captured local into an explicit parameter threaded through free
functions (a real signature-shape change, high blast radius, exactly the kind
of restructuring Phase 8's sibling work in `factory.py`/`_wiring.py` is also
mid-flight on), or (b) introducing a small context/config object to carry the
captured locals — itself a mini-design decision (naming, whether `_TurnPrep`
absorbs it, whether `_wiring.py`'s call sites need to change). Either path is a
genuine design pass, not a mechanical house move. **DEFER.**

### (E) Turn-prep infra + agent-turn skeleton — DEFER (marginal, coupled)
Clusters at lines 104-238 and 290-420 are each "shared by sync/async twins
inside `make_agent_cycle_bodies`" — they're already factored out of the closures
that use them, but they take `node`/`runtime`/`config` as explicit params (not
closures), so they COULD move mechanically. However: (1) `_TurnPrep` the
dataclass is referenced by name in the type signature of nearly every function
in cluster (E)'s sibling clusters (the gate body doesn't need it, but the tools
cluster and the factory closures both do), so splitting these two clusters
into a second new module vs. leaving them here is a closer call requiring the
executor to decide the right home (module boundary: "turn-prep + turn-skeleton"
vs. "everything but per-call and gate") — worth a deliberate pass alongside (D)
rather than bundling into the same mechanical commit as (A)/(B). **DEFER**,
but noted as the natural next 270-line reduction once (A)/(B) land and the
remaining ~770-line file's shape is re-assessed.

## 3. SAFE NOW vs DEFER summary

**SAFE NOW** (do together, one mechanical commit, ~283 lines removed,
1054 → ~770):
- (A) Extract lines 422-624 to new `_agent_tool_calls.py` (~200 lines moved).
- (B) Extract lines 972-1054 to new `_agent_gate.py` (~83 lines moved).

Both are pure "move + re-import" with no behavior change: no closures broken,
no shared mutable state, no signature changes, callers (`_wiring.py`,
`factory.py`) only reference `make_agent_cycle_bodies`/`make_tool_gate_bodies`/
`cycle_names` by name and don't care which file defines the private helpers
those two factories call. Neither touches anything Phase 8 reads (the
`ORACLE_MODEL_OVERRIDE` read stays put in `_turn_prep_kwargs`, untouched).
This does NOT get `_agent_cycle.py` under 500 alone — it's a real dent (27%),
not a full fix; the guard's shrink-only ceiling allowlist entry for this file
should be set expecting a further pass, not close-out.

**DEFER** (needs its own design pass, don't attempt as part of a size-cap sweep):
- (C) Portal tool-triggered-handoff mechanics as one cohesive module spanning
  `_agent_cycle.py` + `factory.py`'s `make_portal_agent_cycle_tool_handoff_fn`.
- (D) Restructuring `make_agent_cycle_bodies` itself (340 lines of closures) —
  requires deciding a closure-capture vs. explicit-context-object shape.
- (E) Whether turn-prep infra (104-238) + agent-turn skeleton (290-420) get a
  third module — depends on the shape chosen for (D).

## 4. Duplication found

No real (behavioral) duplication with other epic-active files was found in
this pass — `factory.py`'s `make_portal_agent_cycle_tool_handoff_fn` CALLS
`make_agent_cycle_bodies` (composition, not duplication) and documents itself
as "Any `Command(` construction stays HERE (factory.py) per guard G1," which is
the intended layering (this file never constructs `Command`). `_wiring.py` only
imports `cycle_names`/`AgentCycleNames`, again composition not duplication.
The one thing worth flagging as *near*-duplication rather than true
duplication: `_build_tool_interaction` (line 514, no tracker touch) and
`_record_tool_result` (line 555, tracker touch + delegates to
`_build_tool_interaction`) are intentionally two functions for the documented
sync-advance-then-build vs. async-pre-reserve-then-build split — this is
already commented as deliberate, not accidental duplication, so no action
needed.

## 5. Bottom line

Best single SAFE NOW move: **extract the per-tool-call pure-function cluster
(lines 422-624) to a new `_agent_tool_calls.py`.** It is the largest
(~200 lines), the most clearly self-contained (already labeled "the DRY-01
extraction" in its own docstring), has zero closure coupling, and directly
un-blocks future growth in the exact area (tool-call handling) most likely to
keep growing as Portal tool-triggered-handoff work continues.
