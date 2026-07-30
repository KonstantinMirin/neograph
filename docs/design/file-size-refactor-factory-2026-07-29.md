# File-size refactor proposal: `src/neograph/factory.py` (1007 lines)

Date: 2026-07-29. Read in full (1007/1007 lines). Read-only research — no code changed.

Context: repo-wide 500-line-per-file cap incoming (separate beads ticket owns the
guard). Active epic `neograph-s7zt3` Phase 8 (`neograph-s7zt3.11`, "extend the 5
fusion ModifierCombo values to the Construct level") is still open. Verified via
`bd show neograph-s7zt3.11`: its scope is (a) Agent-Spec-**export** lowering bugs
in `_agent_spec.py` (`_lower_each`/`_lower_loop`/`_lower_oracle` crashing on a
`Construct`), and (b) a residual-uncertainty check on `_wire_oracle`/
`_inject_oracle_config` (`_wiring.py`/`_subconstruct.py`) — **not** `factory.py`
directly. None of Phase 8's named work items touch any function in this file.
The reason factory.py is "hot" is structural, not this-phase-specific: guard G1
(`TestCommandConstructionMonopoly`, `test_guards_assembly.py:84`) confines ALL
`Command(...)` construction in the whole codebase to exactly two files —
`factory.py` and `runner.py` — so *any* future Portal-combo work (this epic or
later ones) that needs new dynamic routing will keep landing new code in
`factory.py` by design. That constraint, not Phase 8's current task list, is
what should shape the SAFE NOW boundary below.

## 1. Responsibility map

| Lines | Symbol(s) | Responsibility |
|---|---|---|
| 1–37 | imports, module docstring, `log` | module setup |
| 39–107 | `make_node_fn` | THE generic node factory — raw/scripted/think/agent/act dispatch. Core, non-Portal. |
| 109–511 | `make_portal_fn`, `_portal_route_to_command`, `_tool_handoff_to_command`, `make_portal_approval_fn`, `make_portal_subgraph_fn` | Portal **atomic mesh-member** routing: the shared `Command(goto=...)` decision (`_portal_route_to_command`/`_tool_handoff_to_command`) plus its three callers (plain node member, approval-gated member, sub-construct member). |
| 513–719 | `make_portal_agent_cycle_fn`, `make_portal_agent_cycle_tool_handoff_fn` | Portal **agent/act ReAct-cycle** mesh members — wraps `_agent_cycle.make_agent_cycle_bodies`' terminal hop through the same `_portal_route_to_command`/`_tool_handoff_to_command` helpers above. |
| 722–943 | `make_portal_dispatch_fn` | Portal **DISPATCH mode** (`route="decide"`) — Agent-Spec-driven runtime flow construction/compile/invoke. Self-contained: does not call `_portal_route_to_command` or any function in the 109–719 block. |
| 946–1007 | `_make_raw_wrapper`, `_make_araw_wrapper` | `mode='raw'` observability wrappers. No `Command(` use at all; independent of everything else in the file. |

So of 1007 lines, only ~170 (imports + `make_node_fn` + raw wrappers) are the
"generic factory" the module docstring describes. The remaining ~835 lines
(109–943) are Portal-specific, added across the still-active epic's earlier
phases (`neograph-kdr1u`, `do0d9`, portal-tool-triggered-handoff, etc.) and
verified by repeated in-file docstring citations of guard G1.

## 2. Candidate extractions

### (a) Raw-mode wrappers → new module `_raw_dispatch.py`
- Moves: `_make_raw_wrapper`, `_make_araw_wrapper` (lines 946–1007, ~62 lines incl. docstrings).
- Zero `Command(` usage, zero coupling to Portal code, only depends on `Node`,
  `ExecutionError`, `log`, `_type_name` — all already public/importable.
- `make_node_fn` imports the two functions instead of defining them; one `from
  neograph._raw_dispatch import _make_raw_wrapper, _make_araw_wrapper` line added.
- **No G1 guard change needed** — neither function constructs a `Command`.

### (b) Portal DISPATCH-mode cluster → new module `_portal_dispatch_factory.py`
- Moves: `make_portal_dispatch_fn` and its nested helpers (lines 722–943, ~222 lines).
- This cluster is already self-contained: it never calls `_portal_route_to_command`
  or `_tool_handoff_to_command`, and its only intra-file dependency is
  `make_node_fn` (import, not inline reuse of private helpers). Its own imports
  (`pyagentspec.serialization.AgentSpecDeserializer`, `neograph.compiler.compile`,
  `neograph.loader.from_agent_spec`, `neograph.spec_types.lookup_type`) are all
  function-local already (cycle-avoidance + optional-dependency gating), so the
  move doesn't disturb any import-cycle discipline.
- Requires: (1) add `_portal_dispatch_factory.py` to G1's allowlist in
  `test_guards_assembly.py` (the two `Command(` constructions inside
  `dispatch_wrapper`/`adispatch_wrapper`, lines 920/925/934/939, move with it);
  (2) update the ~6 in-file docstring mentions of "Confined to factory.py per
  guard G1" that refer to this cluster specifically.
- Confirmed **not** in Phase 8's path (dispatch mode is orthogonal to the
  Node-level fusion combos and the Agent-Spec-export lowering bugs Phase 8 owns).

## 3. SAFE NOW vs DEFER

**SAFE NOW — (a) raw wrappers → `_raw_dispatch.py`.**
Smallest, cleanest cut: no `Command(` interaction (no G1 change), no shared
state with anything Phase 8 touches, no docstring-invariant rewording needed.
Purely mechanical move-and-import. ~62 lines out of factory.py immediately.
**This is the single best SAFE NOW recommendation** — it can land today with
zero coordination cost and zero risk to the epic.

**SAFE NOW (secondary, slightly larger) — (b) Portal DISPATCH cluster →
`_portal_dispatch_factory.py`.** Mechanical move; the one real cost is
updating G1's allowlist (one line in a test file) plus ~6 docstring
"guard G1"/"factory.py" citations inside the moved functions to name the new
module instead. Verified against `bd show neograph-s7zt3.11`: Phase 8 does not
touch dispatch-mode code, so this can land alongside the epic without
conflict. Removes ~222 lines. Combined with (a): 1007 → ~723 lines — still
over the 500 cap but a meaningful first pass with near-zero epic risk.

**DEFER — the atomic + agent-cycle Portal mesh-member cluster (lines 109–719,
~610 lines: `make_portal_fn`, `_portal_route_to_command`,
`_tool_handoff_to_command`, `make_portal_approval_fn`, `make_portal_subgraph_fn`,
`make_portal_agent_cycle_fn`, `make_portal_agent_cycle_tool_handoff_fn`).**
This is where the guard-invariant language is heaviest ("Confined to factory.py
per guard G1" appears 4+ times across these functions specifically) and where
`_portal_route_to_command`/`_tool_handoff_to_command` are shared core routing
decisions reused by three different wrapper shapes (atomic node, sub-construct,
agent/act cycle). A move here is not "extract and import" — it requires a real
design pass to decide: does the shared routing-decision core (`_portal_route_to_command`,
`_tool_handoff_to_command`) live in one module while the three wrapper-builder
families live in another (or the same)? How does G1's stated two-file invariant
get reworded/expanded in `AGENTS.md` itself (which currently hard-codes
"`Command(` may be constructed ONLY in `factory.py` and `runner.py`" as an
architectural rule, not just a test assertion)? This is exactly the surface a
**future** Portal-combo phase (beyond Phase 8's current 5 non-Portal combos) is
most likely to keep extending — moving it now risks the next epic phase having
to re-learn a new file layout mid-flight. Defer to a dedicated design pass once
Phase 8 (and any near-term Portal-combo follow-on) lands.

**DEFER — sync/async wrapper-pair duplication.** `make_portal_fn`,
`make_portal_subgraph_fn`, `make_portal_agent_cycle_fn`, and
`make_portal_agent_cycle_tool_handoff_fn` each repeat the same ~15-line
kwarg-forwarding call into `_portal_route_to_command` twice (once for the sync
wrapper, once for the async twin) — real, not superficial, duplication (4
functions × 2 near-identical call sites = 8 copies of the same argument list).
It's currently justified by "keep the sync/async pair symmetric and easy to
audit independently," but a small helper that builds the `(sync, async)` pair
from one "get the raw update" callable + one shared kwargs dict would remove
~60-80 lines without changing behavior. Left as DEFER because it touches the
exact functions named above as high-risk for the in-flight epic, and because
verifying it needs re-running the Portal example suites (`tests/test_example_portal.py`,
`tests/test_example_portal_dynamic_flow.py`) plus the tool-triggered-handoff and
spike tests to confirm zero behavior drift — a distinct, scoped task, not a
side effect of a line-count-driven split.

## 4. Duplication check against other assigned/epic-active files

No cross-file duplication found with `_agent_cycle.py`, `_wiring.py`, or
`_subconstruct.py` (the files Phase 8 actually touches) beyond the expected,
intentional call-through (`factory.py` imports and wraps their public
functions; it does not reimplement their logic). The duplication that exists
is **internal** to `factory.py` (the sync/async Portal wrapper pairs noted
above), not shared with other modules.

## Summary

- Total reduction potential identified: ~284 lines (62 raw-wrapper + 222
  dispatch-cluster) landable now with near-zero epic risk, bringing
  factory.py from 1007 → ~723.
- A further ~610-line Portal mesh-member cluster is real extraction material
  but needs its own design pass (G1/AGENTS.md wording + shared-helper
  ownership) and should wait — it is precisely the code future Portal-combo
  phases are most likely to keep growing.
- Best SAFE NOW move: extract `_make_raw_wrapper`/`_make_araw_wrapper` into
  `_raw_dispatch.py` — zero `Command(` involvement, zero G1 guard touch, zero
  coupling to anything Phase 8 or later Portal work will extend.
