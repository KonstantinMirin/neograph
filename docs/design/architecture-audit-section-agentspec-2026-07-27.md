# Agent Spec export/import architecture (master-doc section)

Synthesizes five 2026-07-27 audit docs (`architecture-audit-agentspec-export`,
`-agentspec-import`, `-existing-capabilities`, `-false-not-supported-claims`,
`-crosscheck-agentspec`) plus the `agent-spec-rewrite-2026-07-27.md` design
spec and its two adversarial review rounds. All claims below are re-verified
against those documents; nothing here is newly re-derived from source except
where explicitly marked. Companion, not a restatement of,
`docs/design/modifier-combo-single-source-of-truth-2026-07-27.md` (the
systemic 9-module duplication finding this section's Agent Spec instance is
one case of).

---

## 1. Ground truth: the current state, cell by cell

### 1.1 The `ModifierCombo` axis (12 values) — export side

`_lower_construct_item` (`_agent_spec.py:793`) dispatches on
`classify_modifiers(item)`. Handled: `BARE`, `EACH`, `ORACLE`, `LOOP`,
`OPERATOR` (5 of 12). Unhandled, falling to a generic catch-all
`ConfigurationError` ("has modifier combination {combo.name} — no Agent Spec
lowering yet"): `PORTAL`, `EACH_ORACLE`, `EACH_OPERATOR`, `ORACLE_OPERATOR`,
`LOOP_OPERATOR`, `EACH_ORACLE_OPERATOR`, `PORTAL_OPERATOR` (7 of 12).

`compiler.py` — the proven-correct reference, since it is what actually runs
pipelines on LangGraph — has **two** independent `match combo:` statements,
each with all 6 case-arm groups (`_add_node_to_graph`, Node-level,
`compiler.py:571-681`; `_add_subgraph`, Construct-level,
`compiler.py:444-568`), confirmed identical in shape by both review rounds.
**All 5 unhandled non-Portal combos compile to real, running LangGraph
pipelines at the Node level today.** At the Construct level, `compiler.py`
itself unconditionally rejects `EACH_ORACLE`/`EACH_ORACLE_OPERATOR`
(`compiler.py:511-516`, genuine, accepted scope boundary — no Construct-level
analog of `map_over`/`ensemble_n` exists) and `PORTAL`/`PORTAL_OPERATOR`
(`compiler.py:552-560`, see §1.3 for why this message is now confirmed
partly stale).

**Verdict for the 5 non-Portal fusion combos**: this is the textbook
"capability exists, just not wired to export" pattern — no genuinely new
lowering primitive is needed, only glue code composing the same
already-lowered primitives (`_lower_each`, `_lower_oracle`, `_lower_operator`,
`_lower_loop`) `_lower_construct_item` already calls for the un-fused case.

### 1.2 Per-mode × per-combo × per-input-shape — what's actually tested

`tests/test_agent_spec_matrix.py` mechanically generates
`MODES × SUPPORTED_COMBOS × {oracle configs} × {input shapes}`, where
`SUPPORTED_COMBOS = {BARE, EACH, ORACLE, LOOP, OPERATOR}` — the same 5 that
`_lower_construct_item` handles — by explicit docstring design ("UNSUPPORTED
= PORTAL + every fusion combo that raises ConfigurationError"), i.e. the
matrix **ratifies today's reject-everything-fused behavior as ground truth**
rather than testing whether it is a genuine limit. Within that 5-combo world,
`RED_EXPORT` is empty: every generated cell is either GREEN (exports +
round-trips) or UNREPRESENTABLE (neograph's own assembly validator rejects
the construct before export is reached, e.g. agent/act mode Each/Loop over
>1 upstream input — a genuine "can't write it" case, not an export gap).
**So: within the 5-combo world, export is fully green and fully tested.
Outside it — the 7 unhandled combos, plus 4 additional gaps below — there is
zero matrix coverage**, and 3 of those 4 additional gaps are not even
fail-loud; they are silently wrong or misleadingly worded.

`tests/agent_spec_capabilities.py` is an orthogonal completeness axis (does
every *pyagentspec primitive* have export/import code somewhere) — necessary,
but says nothing about `ModifierCombo` coverage.

### 1.3 Beyond the ModifierCombo axis — the four additional confirmed gaps

These were found independently of the combo-dispatch inventory and are **not
covered by the matrix in either direction**:

1. **Portal-mesh export mis-detection (false rejection).**
   `to_agent_spec`'s `mesh_members` filter (`_agent_spec.py:961-967`) is
   `isinstance(item, Node) and item.modifier_set.portal is not None and not
   is_dispatch` — Node-only. `_validation_portal.py`'s do0d9 fix already
   admits a `Construct` as a first-class non-entry Portal mesh member (backed
   by a passing fixture, `tests/check_fixtures/should_pass/portal_construct_member.py`,
   and `test_portal_cross_subconstruct.py`), so a legal, validator-certified
   mesh containing one Construct member gets silently dropped from
   `mesh_members`, `len(mesh_members) != len(all_items)` fires, and export
   rejects with **"mixes a Portal peer mesh with non-mesh nodes"** — a
   diagnosis that is factually false (nothing is mixed); the detector is
   do0d9-blind. Confirmed independently by review round 2 as the false
   justification underlying a design-doc claim (see §2 below).

2. **`PORTAL_OPERATOR` peer-mesh member silently drops the Operator HITL
   gate.** `_lower_portal_mesh_to_swarm` (`_agent_spec.py:880-945`) builds one
   `Agent` per member via `_make_agent` and never inspects
   `member.modifier_set.operator` at all. A mesh member combining Portal +
   Operator (a legitimate, `compiler.py`-supported combo — human-approval
   gate on the dynamic-routing path) exports as a bare `Agent` with **no
   trace of the approval gate**, no error, no marker. This is the
   `neograph-s7zt3.2` bug and the sharpest safety-relevant symptom in the
   whole audit: a control-flow seam vanishes silently. Untested in either
   direction (the matrix classifies `PORTAL_OPERATOR` `UNSUPPORTED`, i.e.
   assumes it never reaches dispatch, but never tests the mesh-export path
   for it either).

3. **Oracle-variant guard-skip.** `_lower_node` (`_agent_spec.py:377`) is the
   only caller of `_reject_unrepresentable_fields` — the guard that
   fail-louds on `raw_fn`, `skip_when`/`skip_value`, `renderer`, Portal
   `handoff_param`/`handoff_channel`, callable `gate_tools_when`. Every combo
   arm routes its primary node through `_lower_node` **except Oracle**:
   `_lower_oracle`'s per-variant loop calls `_lower_generation_step` directly,
   bypassing `_lower_node` and therefore the guard entirely. An Oracle variant
   with `raw_fn` set (a real, framework-forced mode combination) is silently
   exported as a name-only `ToolNode` stub — the actual Python callable
   dropped, no error. Confirmed untested: the one existing raw_fn-rejection
   test (`test_agent_spec_export.py:256`) builds a BARE node, the guarded
   path; no test builds an Oracle-modified node with `raw_fn`/`skip_when`/
   `renderer`.

4. **Dict-form (multi-output) producer → dict-form-input consumer, rejected
   at export.** `to_agent_spec`'s edge-wiring sweep (lines 1093-1102,
   1115-1117) rejects this shape unconditionally. Dict-form `Node.outputs` is
   a first-class, compiler-supported IR feature; `_properties_for` (line 275)
   already implements the identical "`{key}.{field}`-prefixed Property"
   mechanism for the input side (lines 283-290) — it is simply never invoked
   for the producer/output side. High-confidence unwired-capability, not a
   representability wall.

### 1.4 Import side (`loader.py`) — verified faithful inverse, with two confirmed asymmetries and one confirmed-live compiler bug found along the way

`_group_flow_items` (`loader.py:559-638`) is a single forward pass, recognize-
and-emit-in-one-step, no backtracking — verified in detail. Every reconstruct
helper (`_reconstruct_primitive_node`, `_reconstruct_oracle_group`,
`_reconstruct_each_node`, `_reconstruct_agent_node`/`_node_from_spec_agent`,
`_reconstruct_loop_item`/`_reconstruct_operator_item`, top-level
`from_agent_spec` dispatch) is a faithful, fail-loud-on-mismatch inverse of
its `_agent_spec.py` export counterpart — **except**:

- **Asymmetry 1 — Construct as Portal mesh member never reconstructed.**
  `_reconstruct_swarm_mesh` (`loader.py:689-734`) always builds every mesh
  member as an agent-mode `Node` via `_node_from_spec_agent` — there is no
  branch that could produce a `Construct` member. This mirrors export's
  Node-only filter (§1.3.1) from the other direction: neither side implements
  Construct-as-mesh-member end to end.
- **Asymmetry 2 — `neograph/portal_spec` marker written but never read.**
  `_lower_portal_mesh_to_swarm` writes `_MARK_PORTAL_SPEC`
  (`max_hops`/`on_exhaust`/`route`) unconditionally on export;
  `_reconstruct_swarm_mesh` never reads it back (confirmed absent from
  `loader.py` by grep). Silent drop on re-import of a neograph-exported Swarm.
- **Symmetric, by-design non-gaps** (both sides agree, not asymmetries):
  Each/Loop/Operator never wrap a Construct on either side (only
  `LlmNode`/`ToolNode`/`AgentNode` are legal in those positions);
  `Oracle.merge_pre_process`/`merge_post_process`/`merge_fallback` and
  callable `Loop.when` make export fail-loud before the round trip starts, so
  import correctly never needs a case for them.
- **Confirmed-live compiler bug found while chasing Asymmetry 1** (review
  round 2, independently reproduced): a `Construct` CAN be a **non-entry**
  Portal mesh member today (do0d9-admitted, passing fixture, actually
  compiles) but CANNOT be the mesh **entry** — `compiler.py:254`'s mesh-entry
  detection is `isinstance(item, Node) and classify_modifiers(item)[0] in
  (PORTAL, PORTAL_OPERATOR)`, so a Construct-as-entry falls through to
  `_add_subgraph`'s unconditional `CompileError`, whose message ("mesh
  members must be sibling Nodes") is now confirmed **stale** — do0d9 already
  relaxed exactly this for the non-entry case. This is a real, narrow,
  untested `compiler.py` bug (missing `isinstance` relaxation, same family as
  do0d9's other four already-relaxed sites), not a fundamental limit, and it
  blocks Agent Spec work on Construct-as-entry regardless of what the export/
  import layers do.
- **Loader never re-validates the reconstructed IR.** `_check_portal_mesh`
  (`_validation_portal.py:40`) is the single construct-assembly-time gate for
  every Portal mesh rule, called from exactly one site
  (`_construct_validation.py:348`). `loader.py` builds a brand-new `Construct`
  from imported spec data and never routes it through that gate (or an
  equivalent) — a structurally malformed reconstruction could silently
  produce a `Construct` that was never actually checked by the rule that
  governs every other construction path.

### 1.5 Why this happened: the reuse-vs-reimplementation root cause

Cutting across every gap above: `classify_modifiers`/`_COMBO_MAP`
(`modifiers.py:91-165`), `_group_portal_members` (`modifiers.py:727`), and
`_check_portal_mesh` (`_validation_portal.py:40`) are already the proven,
type-agnostic (operate on `.modifier_set`, not `isinstance(Node)`) single
sources of truth used by `compiler.py`/`state.py`/`runner.py`/`_wiring.py`/
`_validation_portal.py`/`_ir_normalize.py`. Agent Spec export/import are the
one place in the codebase that reimplements ad hoc, `isinstance`-gated
mesh-membership detection instead of calling them — directly explaining both
the Construct-as-mesh-member export gap and the unvalidated-reconstruction
import gap. By contrast, `iter_with_arms`/`iter_item_slots`
(`_ir_branch.py`) and `spec_types.py`'s type-conversion functions ARE reused
correctly and symmetrically by both `_agent_spec.py` and `loader.py` — cited
as the template the Portal-mesh handling should have followed.

---

## 2. Every gap, with its exact fix-or-reuse plan

Ordered by the crosscheck doc's dependency-driven priority (not severity),
each tagged with its reuse class.

| # | Gap | Fix site | Reusable capability | Class |
|---|---|---|---|---|
| **A3** | `compiler.py` Construct-as-mesh-**entry** fails (§1.4) | `compiler.py:251-254` — drop the redundant `isinstance(item, Node)` conjunct, leave `classify_modifiers(item)[0] in (...)` standing alone | `classify_modifiers` already type-agnostic | One-line relaxation, 5th site in the do0d9 family. **Must land first** — Agent Spec export of Construct-as-mesh-entry is meaningless if the compiler can't run it. |
| **A1** | Export mesh-detection false rejection (§1.3.1) | `_agent_spec.py:961-967` — replace the `isinstance`-gated filter with `classify_modifiers(item)[0] in (PORTAL, PORTAL_OPERATOR)`, and replace the all-or-nothing check with `_group_portal_members` (disambiguates multiple distinctly-named adjacent meshes — a real bug the fix must not reintroduce) | `classify_modifiers` + `_group_portal_members` | Reuse-and-delete, no new logic |
| **A5** | Loader never re-validates reconstructed IR (§1.4) | `loader.py`'s `from_agent_spec` — route the reconstructed pipeline through the same `Construct(...)` assembly path `_construct_validation.py:348` already calls `_check_portal_mesh` from (preferred), or call it explicitly before returning | `_check_portal_mesh` | Pure reuse, zero new validation code |
| **B2** | Oracle-variant guard-skip (§1.3.3) | `_agent_spec.py`'s Oracle per-variant loop (~521-543) — add a `_reject_unrepresentable_fields(variant_node)` call before `_lower_generation_step` | `_reject_unrepresentable_fields` | Pure reuse, one added call |
| **B3** | `_MARK_PORTAL_SPEC` written, never read (Asymmetry 2) | `loader.py`'s `_reconstruct_swarm_mesh` (~689-734) — read the marker the same way `_reconstruct_agent_node`/`_reconstruct_oracle_group` already read theirs, pass `max_hops=`/`on_exhaust=`/`route=` into the reconstructed `Portal(...)` | Copy-the-pattern from two sibling reconstructors in the same file | Pure reuse |
| **B4** | Dict-form outputs → dict-form inputs rejected (§1.3.4) | `_agent_spec.py`'s edge-wiring sweep (~1093-1117) — apply `_properties_for`'s existing `{key}.{field}`-prefix logic to the producer side instead of raising | `_properties_for` | Pure reuse |
| **A4** | 5 non-Portal fusion combos rejected (§1.1) | `_agent_spec.py:846-877`, `_lower_construct_item`'s match — needs real per-combo glue code composing existing primitives (`_lower_each`/`_lower_oracle`/`_lower_operator`), e.g. `EACH_ORACLE` = a `MapNode` whose inner subflow is what `_lower_oracle` already produces un-fused. Recommended order: `EACH_OPERATOR`/`LOOP_OPERATOR` first (single wrap), `EACH_ORACLE` next (two proven wraps composed), `EACH_ORACLE_OPERATOR` last (three-way) | `_lower_each`, `_lower_oracle`, `_lower_operator`, `_lower_loop` as building blocks | Reuse-of-primitives, new glue code (not one-line) |
| **B1** | `PORTAL_OPERATOR` silently drops the gate (§1.3.2) | `_agent_spec.py`, inside `_lower_portal_mesh_to_swarm`'s per-member loop (~914-924) — branch on `member.modifier_set.operator`, reuse `_lower_operator`'s gate-attachment logic via a mesh-exit pause composite (`AgentNode(agent=swarm)` → `BranchingNode`/`InputMessageNode`), since `Swarm` has no interior per-member pause primitive (verified live against installed `pyagentspec`: `Swarm`'s only fields are `first_agent`/`relationships`/`handoff`) | `_lower_operator`'s existing gate logic as the reference; new call site | Reuse-the-logic, new design for how it attaches to Swarm |
| **PORTAL (dispatch-mode)** | No Flow-node lowering at all for a dispatch-mode Portal node (`is_dispatch=True`) | New design needed — check `tests/agent_spec_capabilities.py`'s primitive registry for a conditional/dynamic-routing pyagentspec node type before assuming infeasibility | None identified yet | Genuinely new, unresolved |
| **C1** | Construct-as-Portal-mesh-member export/import, full (§1.3.1 + Asymmetry 1, beyond A1/A3) | `_lower_portal_mesh_to_swarm` unconditionally accesses `member.prompt`/`.inputs`/`.name`; a `Construct` has none of these — would `AttributeError`, not degrade gracefully, even after A1/A3 land. Needs new `_make_agent`-equivalent lowering for a Construct member (likely: lower its sub-pipeline to a nested `Flow`/`FlowNode`, wrap that as the Swarm member unit). Import needs a new branch in `_reconstruct_swarm_mesh` recognizing a sub-`Flow` shape and reconstructing a `Construct` — the top-level `FlowNode → Construct` recursion `from_agent_spec` already does elsewhere is a reusable pattern to copy, but the Swarm-specific wiring (member ordering, entry detection, handoff routing) is new | Partial reuse of the recursion pattern; real new design on both sides | **The largest single item — own tracked epic**, not a one-line fix |
| **C2** | Mixed Portal-mesh + non-mesh construct (Swarm wrapped in a Flow) | Open question: does pyagentspec even support nesting a `Swarm` inside a `Flow` node? No existing pattern found in either inventory. Check the primitive registry first | Unknown | Genuinely open, not obviously wrong to currently reject |
| **EACH_ORACLE on sub-constructs** | Genuinely infeasible, not a gap | N/A — `EACH_ORACLE` is defined entirely in terms of a single Node's `map_over`/`ensemble_n`; no Construct-level analog exists | None — nothing to reuse | Accepted, permanent scope boundary |

**Permanent, by-design fail-loud list** (confirmed, no cross-check flag —
callable-valued fields have no serialization target in any spec): `raw_fn`,
`skip_when`/`skip_value`, `renderer`, Portal `handoff_param`/
`handoff_channel` (dispatch-mode-specific), callable `gate_tools_when`,
Oracle `merge_pre_process`/`merge_post_process`/`merge_fallback`, callable
`Loop.when`.

---

## 3. Reconciliation with `agent-spec-rewrite-2026-07-27.md` + its two review rounds

The rewrite spec proposes a `COMBO_DECOMPOSITION` total-function table
(`ModifierCombo → (PrimaryShape, has_operator)`) in `modifiers.py`, consulted
read-only by both `compiler.py`'s two match statements and a rebuilt
`_lower_construct_item`, plus a `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` frozenset
for the narrower Construct-level question, with structural guards preventing
a second hand-rolled table from regrowing. **This section confirms the
following still holds, unchanged, after both review rounds:**

- The table's cell values themselves (§1.1 of the spec) — correct.
- Per-combo feasibility for all 12 combos, including the two hardest —
  `PORTAL_OPERATOR`'s mesh-exit pause composite and the `EACH_ORACLE`
  Node-level-feasible/Construct-level-rejected split — **independently
  confirmed by live repro against the installed `pyagentspec` package** in
  review round 1 (`Swarm`/`AgentNode`/`BranchingNode` field-level checks).
  Zero of the 12 combos are genuinely infeasible at the Node level.
- The placeholder-translation (Option F) preservation claim — confirmed
  landed and correctly described, not to be redone.
- CLAUDE.md layer-discipline clearance for adding `COMBO_DECOMPOSITION` to
  `modifiers.py` — confirmed accurate on the literal text (the off-limits
  list is about @node-layer features leaking in, not a blanket freeze).
- The two-match-statement Step-0 refactor sketch — **round 1 flagged this as
  imprecise (spec showed one `match` block, real code has two with divergent
  per-combo bodies); round 2 confirmed the fix landed correctly** (spec now
  shows one sketch per function, matching `compiler.py:444-568` /
  `571-681` exactly).
- The Construct-item-modifier-drop fix (§1.6: `_lower_construct_item`
  currently never calls `classify_modifiers` on a `Construct` item at all,
  silently ignoring modifiers on a `Construct(...) | Each() | Oracle()`
  sub-item) — **independently reproduced fresh in review round 2** (built the
  exact case, grepped the serialized output, confirmed zero trace of the
  `Loop` modifier, no error, no marker). The proposed fix (check
  `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` first, mirroring `_add_subgraph`) is
  confirmed correct and sufficient for this specific gap.

**What the spec got wrong, corrected here using this section's findings:**

1. **The Portal-exclusion justification in `SUB_CONSTRUCT_UNSUPPORTED_COMBOS`
   is false, per review round 2 and independently corroborated by this
   section's §1.3.1/§1.4.** The spec claims Portal-on-Construct is "impossible
   by construction" (mesh membership requires a bare Node). This is
   contradicted directly by `_validation_portal.py`'s own do0d9 admission and
   a passing fixture: a Construct **can** be a non-entry Portal mesh member
   today. The correct framing (this section's §2, row A1/A3/C1) is: Portal is
   excluded from `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` not because it's
   impossible, but because it needs its OWN dedicated handling (a Construct
   mesh member routes through the mesh path, not the generic Construct-item
   modifier-check path) — and that mesh path itself has two live bugs (A1
   export false-rejection, A3 compiler entry-detection) that must be fixed
   first, plus C1's genuinely-new Construct-as-Swarm-member lowering that is
   NOT in today's scope. The rewrite's Construct-item branch must make an
   explicit, accurate decision for `PORTAL`/`PORTAL_OPERATOR` classified
   Construct items (reject with an accurate message reflecting current
   real capability, per this section's ground truth) rather than silently
   relying on the false premise.
2. **The loader.py "consult `COMBO_DECOMPOSITION` symmetrically" instruction
   is underspecified for the 6 newly-real composed combos**, per review round
   2's most significant finding. `_group_flow_items` recognizes and emits in
   one step with fixed, per-modifier lookahead (0/1/2 nodes); it has no
   intermediate `frozenset[str]` of co-occurring modifier names to classify.
   For each of the 6 composed combos (`EACH_OPERATOR`, `ORACLE_OPERATOR`,
   `LOOP_OPERATOR`, `EACH_ORACLE`, `EACH_ORACLE_OPERATOR`, `PORTAL_OPERATOR`),
   the recognition walk itself needs new variable-length lookahead
   composition (fold a trailing Operator check+pause triple onto an
   Each/Oracle/Loop/Portal group; recognize a nested Oracle fan-out+merge
   inside a `MapNode`'s subflow for the EACH_ORACLE family) — this is
   real, structural work on the recognition step, not merely "add a
   classification helper alongside an unchanged walk." Build Plan Step 6
   should be re-sized as one recognition-pattern change per composed combo,
   landed in lockstep with its matching export-side combo, not one generic
   step at the end.
3. **A previously-unknown, independently-confirmed `compiler.py` bug was
   surfaced by chasing the Portal-exclusion question**, which this section's
   §1.4/§2 (row A3) now documents as its own fix, prioritized ahead of any
   Agent Spec Portal-mesh work: Construct-as-mesh-**entry** fails today
   (`compiler.py:254`'s `isinstance(item, Node)` gate), while
   Construct-as-non-entry-member already works — an inconsistency, not a
   deliberate boundary, that must be closed (one-line relaxation) before any
   export/import work on Construct-as-mesh-member is meaningful.
4. `_MARK_REMOTE_AGENT` is confirmed dead code (zero references anywhere
   outside its own definition) — file as a small, separate cleanup ticket,
   not part of the rewrite's scope.

**Net reconciliation verdict**: the rewrite spec's core design (shared
decomposition table, read-only consumption by both `compiler.py` match
statements and Agent Spec export, structural anti-regrowth guards) remains
sound and is the correct direction. It is **not ready for a single
whole-spec implementation pass**: Build Plan Steps 0-2 (shared tables, both
`compiler.py` match statements, the 5 already-shipped combos +
Construct-item-modifier fix) can start now unblocked. Steps 3-5
(`PORTAL_OPERATOR`, the other composed combos) must wait on an accurate,
non-false Portal/Construct-item decision (correction #1 above) and on A3's
compiler fix landing first (it is a prerequisite, not parallelizable).
Step 6 (loader.py) must not start implementation until its recognition-side
design is corrected per correction #2 — otherwise the implementer hits the
exact "recognition doesn't work the way the spec assumes" wall two review
rounds already found and documented.

---

## 4. Priority ordering (dependency-driven synthesis)

1. **A3** — `compiler.py` Construct-as-mesh-entry `isinstance` fix. Must land
   first; every downstream Agent Spec Portal-mesh fix is moot if the compiler
   itself can't run the shape being exported.
2. **A1 + A5** — export mesh-detection filter fix, loader validation gate.
   Both pure reuse, independent of each other and of A3's timing for A5, but
   A1 depends on A3 being correct for its "is this really representable"
   framing. Turns a false rejection into either a correct export or a
   correct, validated rejection.
3. **B2, B3, B4** — pure reuse-the-existing-helper fixes, no design decisions,
   safe to parallelize, no dependency on anything else in this list.
4. **B1 + A4** — bounded new glue code reusing existing lowering primitives;
   needs new round-trip fixtures (none exist today for any of these cells).
5. **C1 (full Construct-as-mesh-member) + C2 (Swarm-in-Flow) + dispatch-mode
   PORTAL lowering** — genuinely new design; gate on a pyagentspec
   primitive-capability check (`tests/agent_spec_capabilities.py`'s registry)
   before committing to an implementation approach for any of the three.

**Cross-cutting root cause** (unchanged from the crosscheck doc, restated
here as the section's closing point): every reuse-class fix (A1, A3, A5, B2,
B3, B4) shares one cause — Agent Spec export/import re-derive type-shape and
marker logic ad hoc instead of calling the same functions
(`classify_modifiers`, `_group_portal_members`, `_check_portal_mesh`,
`_reject_unrepresentable_fields`, `_properties_for`, the existing
marker-read pattern) already proven correct and used by
`compiler.py`/`state.py`/`runner.py`/`_wiring.py`/`_validation_portal.py`.
None of these six fixes invents new semantics; they delete a hand-rolled
reimplementation and call the existing helper instead. Only C1/C2/the
dispatch-mode Portal case and the bounded new-glue-code items (A4, B1)
require actual new design or implementation work.
