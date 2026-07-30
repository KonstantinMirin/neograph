# File-split proposal: `src/neograph/_wiring.py` (1461 lines)

Date: 2026-07-29
Scope: design only, no code changed. Read in full before writing this.

## Context

`_wiring.py` lowers every modifier topology (Each, Oracle, Each×Oracle
fusion, Loop, Branch, Portal mesh/dispatch, agent/act ReAct cycle,
Operator) into LangGraph node/edge/`Command` wiring. It is called
exclusively from `compiler.py`'s single walk loop, plus one function-local
import from `factory.py` (`_resolve_condition`).

Active epic constraint: neograph-s7zt3 Phase 8 / neograph-s7zt3.11 extends
Each×Oracle fusion (`ModifierCombo` lowering) from the Node level to the
Construct level. That work will add code to the fusion cluster
(`_add_each_oracle_fused` + its helpers) specifically — not to Loop,
Branch, or Operator wiring.

## 1. Responsibility map

| Lines | Cluster | Approx. lines |
|---|---|---|
| 1–70 | Module docstring, imports, type aliases | 70 |
| 71–140 | Each-navigation shared primitives: `_collect_each_items`, `_empty_each_bypass` | 70 |
| 142–215 | Branch-arm descent primitives: `_add_arm_nodes`, `_wire_arm_edges` | 74 |
| 217–316 | Oracle/Each single-node/sub-construct wiring: `_wire_oracle`, `_wire_each` | 100 |
| 318–539 | Each×Oracle **fusion**: `_add_each_oracle_fused`, `_merge_one_group`, `_amerge_one_group` | 222 |
| 542–701 | Loop: `_make_loop_router`, `_node_loop_unwrap`, `_construct_loop_unwrap`, `_resolve_condition`, `_add_loop_back_edge` | 160 |
| 703–1038 | Portal mesh + dispatch: `_contiguous_portal_mesh`, `_make_portal_subgraph_member_fn`, `_add_portal_mesh`, `_add_portal_dispatch` | 336 |
| 1040–1086 | Loop-on-subconstruct: `_add_subgraph_loop` | 47 |
| 1089–1198 | Branch: `_add_branch_to_graph` | 110 |
| 1201–1436 | ReAct agent-cycle wiring (shared by plain agent nodes and Portal mesh members): `_wire_agent_cycle_body`, `_add_agent_cycle`, `_add_portal_agent_cycle_member` | 236 |
| 1439–1461 | Operator: `_add_operator_check` | 23 |

Nine independently-namable clusters live in one file. `compiler.py`
imports only the top-level entry points (`_add_agent_cycle`,
`_add_branch_to_graph`, `_add_each_oracle_fused`, `_add_loop_back_edge`,
`_add_operator_check`, `_add_portal_dispatch`, `_add_portal_mesh`,
`_add_subgraph_loop`, `_contiguous_portal_mesh`, `_wire_each`,
`_wire_oracle`) — every cluster's internal helpers
(`_make_loop_router`, `_wire_arm_edges`, `_merge_one_group`, etc.) are
private to `_wiring.py` today, confirmed by grep across `src/`.

## 2/3. Proposed extractions, target modules, SAFE NOW vs DEFER

### SAFE NOW #1 (top recommendation): `_wiring_oracle_each.py`

**Move**: `_collect_each_items`, `_empty_each_bypass` (71–140), `_wire_oracle`,
`_wire_each` (217–316), `_add_each_oracle_fused`, `_merge_one_group`,
`_amerge_one_group` (318–539). **~392 lines removed.**

**Why SAFE NOW, and why first**: this is exactly the cluster Phase 8 is
about to extend (Construct-level fusion). It is self-contained — no
other file calls its internal helpers, only `compiler.py` calls the four
public entry points (`_wire_oracle`, `_wire_each`, `_add_each_oracle_fused`,
implicitly `_merge_one_group`/`_amerge_one_group` are called only from
within `_add_each_oracle_fused`). The move is pure cut-paste + one import
block edit in `compiler.py` (swap the `from neograph._wiring import (...)`
partial list for two import lines) — no behavior change, no signature
change. Doing it **before** Phase 8 starts writing the Construct-level
lowering means that new code lands directly in a ~390-line module instead
of adding onto an already-1461-line file, and there is no in-flight diff
for the move to conflict with yet (Phase 8 is still "open work," not
mid-edit). This is the single highest-leverage, lowest-risk change on
this file.

**Caution**: confirm with whoever owns Phase 8 scoping that no branch is
already mid-edit on `_add_each_oracle_fused` before landing this — the
move itself doesn't touch logic, but a concurrent branch editing the same
lines would need a trivial rebase, not a re-plan.

### SAFE NOW #2: `_wiring_branch.py`

**Move**: `_add_arm_nodes`, `_wire_arm_edges` (142–215), `_add_branch_to_graph`
(1089–1198). **~185 lines removed.**

**Why SAFE NOW**: `_add_arm_nodes`/`_wire_arm_edges` are called only by
`_add_branch_to_graph`, which is called only from `compiler.py`. Zero
overlap with Portal, Loop, or fusion work — Branch has no relationship to
the Phase 8 fusion-at-Construct-level effort. Pure mechanical move.

### SAFE NOW #3 (slightly larger touch surface): `_wiring_loop.py`

**Move**: `_make_loop_router`, `_node_loop_unwrap`, `_construct_loop_unwrap`,
`_add_loop_back_edge` (542–701, excluding `_resolve_condition`),
`_add_subgraph_loop` (1040–1086). **~205 lines removed.**

**Why SAFE NOW with a caveat**: Loop wiring has no relationship to the
active epic (Portal / fusion). The one wrinkle is `_resolve_condition`
(624–640), which is also called from `_add_portal_mesh` (approval-node
wiring), `_wire_agent_cycle_body` (tool-gate condition), and
`_add_operator_check`, and is imported function-locally by `factory.py`
(`neograph/factory.py:404`). Recommend leaving `_resolve_condition` in
`_wiring.py` itself (it's 17 lines and is the one genuinely cross-cutting
utility in the file — every other cluster only calls it, none of them
own it), rather than inventing a fifth micro-module for one function.
That keeps this extraction to a single mechanical move plus one `import`
line added to the new module (`from neograph._wiring import _resolve_condition`).

### DEFER #1: Portal mesh + dispatch (`_contiguous_portal_mesh`,
`_make_portal_subgraph_member_fn`, `_add_portal_mesh`, `_add_portal_dispatch`,
703–1038, ~336 lines) together with the agent-cycle wiring cluster
(`_wire_agent_cycle_body`, `_add_agent_cycle`, `_add_portal_agent_cycle_member`,
1201–1436, ~236 lines).

**Why DEFER**: these two clusters are the live surface of the same
"Agent Spec / Portal architecture rebuild" epic that Phase 8 belongs to
(even though Phase 8 itself targets fusion, not mesh wiring) — mixing an
extraction here with concurrent epic churn is exactly the kind of
non-mechanical, needs-its-own-design-pass work the task description
warns against. It is also NOT a clean two-way split:
`_wire_agent_cycle_body` is shared verbatim between a plain agent/act
node (`_add_agent_cycle`) and a Portal mesh member
(`_add_portal_agent_cycle_member`) — deciding whether the shared body
function belongs in a Portal-owned module or an agent-cycle-owned module
(or a third shared module) is a real design call, not a mechanical
cut-paste, since it changes which module the "one dispatch path for all
ReAct wiring" invariant lives in. Recommend a dedicated design pass once
Phase 8 lands and the epic's file-touch pattern stabilizes — likely
resulting in `_wiring_portal.py` (mesh + dispatch) and
`_wiring_agent_cycle.py` (shared ReAct body + both call sites), with an
explicit decision on which one exports `_wire_agent_cycle_body`.

### DEFER #2: `_add_operator_check` (1439–1461, 23 lines)

Too small to justify its own module. Fold it into whichever module ends
up owning Loop or Portal during the DEFER #1 pass (it's already called
right after Loop/agent-cycle/branch wiring in `compiler.py`, so it's a
natural tail addition to one of those future modules) rather than
extracting it alone now.

## 4. Duplication found (within this file)

Five structurally-identical no-op passthrough closures exist, one per
topology's join/exit node, never consolidated:

- `barrier_fn` (`_wire_each`, line 308): `def barrier_fn(state): return {}`
- `loop_exit` (`_add_loop_back_edge`, line 678)
- `handoff_exit` (`_add_portal_mesh`, line 860)
- `dispatch_exit` (`_add_portal_dispatch`, line 1004)
- `loop_exit` (`_add_subgraph_loop`, line 1064)
- `join_fn` (`_add_branch_to_graph`, line 1184)

All six are `def f(state: Any) -> dict: return {}`, differing only in
name (chosen to match the node name registered right after). This is
real, not superficial, duplication — a single `_noop_passthrough_fn()`
factory (mirroring the existing `_empty_each_bypass` factory pattern
already in this file) would collapse all six. Small win (~15 lines), does
not require a file split, and is orthogonal to the SAFE NOW/DEFER
extractions above — worth doing as a one-line-per-callsite cleanup
whenever one of the SAFE NOW moves touches that function anyway, but not
proposed here as its own ticket since it's cosmetic, not a line-count
driver.

No duplication was found between `_wiring.py` and the Each×Oracle merge
logic in `_oracle.py` — `_merge_one_group`/`_amerge_one_group` are
already pure delegations to `_oracle._merge_variants`/`_amerge_variants`
(confirmed by reading both; the docstrings are accurate: "does NOT
re-implement any merge step").

## Summary of reduction potential

| Extraction | Lines removed | Bucket |
|---|---|---|
| `_wiring_oracle_each.py` | ~392 | SAFE NOW (do first — this is what Phase 8 touches) |
| `_wiring_branch.py` | ~185 | SAFE NOW |
| `_wiring_loop.py` | ~205 | SAFE NOW |
| Portal mesh + agent-cycle (2 modules) | ~572 | DEFER — needs its own design pass |
| Operator | 23 | DEFER (fold into a DEFER #1 module later) |

If all three SAFE NOW extractions land, `_wiring.py` drops from 1461 to
roughly **~680 lines** (imports/aliases + Portal mesh + agent-cycle +
Operator + `_resolve_condition`), comfortably clear of any interim
allowlist ceiling while leaving the epic-sensitive Portal/agent-cycle
code untouched and in place for Phase 8 and the later DEFER pass.
