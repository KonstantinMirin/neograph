# Agent Spec / Portal master architecture (supersedes the rewrite spec)

Date: 2026-07-27
Status: authoritative design going forward

**This document REPLACES `docs/design/agent-spec-rewrite-2026-07-27.md` (and its
two review rounds) as the authoritative design for Agent Spec export/import and
Portal.** The rewrite spec is not deleted — it remains useful raw material and
is cited throughout — but wherever it conflicts with this document, this
document wins. Its **core design survives intact**: a shared
`COMBO_DECOMPOSITION` total-function table in `modifiers.py`
(`ModifierCombo → (PrimaryShape, has_operator)`), a `SUB_CONSTRUCT_UNSUPPORTED_COMBOS`
frozenset for the Construct-level restriction, and structural anti-regrowth
guards forbidding a second hand-rolled enumeration. What is **superseded**:
(1) its scope — three consumers (`compiler.py`, `_agent_spec.py`, `loader.py`)
widens to **ten** (adding `state.py`, `_state_write.py`, `_subconstruct.py`,
`_input_shape.py`, `runner.py`, `_wiring.py`); (2) its
`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` Portal-exclusion rationale, which was false
("impossible by construction") — corrected in §4 below; (3) its Build Plan,
replaced wholesale by §6; (4) its loader.py recognition-side design for the six
newly-real composed combos, which under-specified the recognition-walk change
as "add a classification helper" when it is real structural lookahead work —
corrected in §6.

Synthesizes, and is verified against real source consistent with:
`docs/design/architecture-audit-section-portal-2026-07-27.md`,
`docs/design/architecture-audit-section-agentspec-2026-07-27.md`, and
`docs/design/modifier-combo-single-source-of-truth-2026-07-27.md`. Citations
below point to those documents' own file:line evidence rather than re-deriving
it from memory.

---

## 1. Executive summary

Three lessons recur across every angle of this investigation, and they are the
same lesson wearing different clothes.

**Duplicated dispatch is the root cause of nearly every bug found.** At least
ten modules — `compiler.py` (two `match combo:` blocks), `_agent_spec.py`,
`state.py` (three blocks), `_state_write.py`, `_subconstruct.py`,
`_input_shape.py`, `runner.py`, `_wiring.py`, `_fan_agent.py`, and `loader.py`
(via a structurally different marker/pattern-matching mechanism) — each
independently re-derive "what does this `ModifierCombo` mean" for their own
purposes. `compiler.py` is the **proven-correct reference**: it is what
actually makes pipelines run on LangGraph, so wherever another module's answer
disagrees with `compiler.py`'s, the other module is wrong, not `compiler.py`.
Portal's own rollout is itself an instance of this anti-pattern, not merely
its latest victim: the IR-field discipline (`Node.handoff_param`/
`handoff_channel`, single-writer, guard G3) was followed correctly, but "which
consumers need to know about PORTAL/PORTAL_OPERATOR" was never centralized,
so each of the ten modules grew its own independent check at Portal's
introduction time.

**Claims of "not supported" were repeatedly wrong.** Every adversarial review
round in this investigation found that a stated infeasibility was actually a
failure to reuse existing, already-correct capability: a `Construct` CAN be a
non-entry Portal mesh member today (do0d9, passing fixture) even though the
design doc that introduced it explicitly called this "out of scope" and even
though `compiler.py` itself still fails on the entry case via a stale
`isinstance(item, Node)` gate whose accompanying comment ("already rejected at
assembly — defense-in-depth") is now confirmed false. The rewrite spec's own
`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` Portal-exclusion rationale repeated this
exact mistake in the opposite direction — claiming impossibility where the
validator already grants the capability.

**The fix is overwhelmingly reuse, not new design.** Of the roughly dozen
distinct gaps catalogued in §5, only three require genuinely new design work
(Construct-as-Portal-mesh-member Agent Spec lowering, dispatch-mode Portal's
Flow-node lowering, and the `PORTAL_OPERATOR` gate-preservation composite).
Everything else is: call the function that already exists
(`classify_modifiers`, `_group_portal_members`, `_check_portal_mesh`,
`_reject_unrepresentable_fields`, `_properties_for`) instead of re-deriving
its answer ad hoc. The maintainer's standing principle — "any neograph
structure that compiles to LangGraph must be representable in Agent Spec" —
survives this investigation intact as an achievable target, not an aspiration
in tension with reality; every closed gap turned out to be "the reference
lowering already knows how; a duplicate, staler decision site hasn't caught
up yet."

---

## 2. Portal architecture

`Portal` (formerly `Keymaker`) is the sanctioned second "genuinely new IR
capability" alongside `_BranchNode` (AGENTS.md's layering doctrine), with two
mutually exclusive modes discriminated solely by `Portal.is_dispatch`:

- **Peer mode** (`to=[...]`) — a mesh of members routing to each other at
  runtime via `Command(goto=...)`, entered through one static edge at the
  mesh's `entry = members[0]`, exited through a synthesized
  `__handoff_exit_{entry}` node.
- **Dispatch mode** (`route="decide"`) — a standalone linear node whose body
  synthesizes/validates an emitted flow spec; never a mesh member.

**IR model**: `Node.handoff_param`/`Node.handoff_channel` are written by a
single writer (`_ir_normalize.py`), pinned by guard G3 — no `Construct`-level
analog exists by deliberate design (a mesh member's channel key threads
through the recursive `compile()` closure instead). `ModifierCombo.PORTAL`/
`PORTAL_OPERATOR` (`_COMBO_MAP`, `modifiers.py`), `HANDOFF_END`,
`DISPATCH_ROUTE`, and `StateKeys.handoff_payload/hops(...)` round out the
model.

**Assembly validation** (`_validation_portal.py::_check_portal_mesh`): groups
PEER members via `_group_portal_members` (itself a genuine, narrow single
source of truth reused by `_ir_normalize.py`, `_wiring.py`, and `state.py`),
then per-group checks contiguity, uniform declared-output payload type,
Operator-gated members must be atomic (agent/act or Construct members with
Operator are rejected — this is why `PORTAL_OPERATOR` on a `Construct` is
genuinely, permanently unrepresentable, not a duplication bug), one connected
component, and a route field typed `str` or `Literal[member names] |
HANDOFF_END`. **Critically, nothing in this validator singles out the entry's
type** — a `Construct` is explicitly admitted as a mesh member at any
position, including implicitly the entry position, by the do0d9 relaxation.
The validator's actual behavior and the do0d9 design doc's own stated intent
("Construct as entry is out of scope for v1") silently diverged.

**Compilation** (`compiler.py::compile()`'s top-level walk): dispatches
`isinstance(item, Node)` + Portal combo → Portal branch (peer mesh via
`_contiguous_portal_mesh`/`_add_portal_mesh`, both in `_wiring.py`, or dispatch
via `_add_portal_dispatch`); else `isinstance(item, _BranchNode)` →
branch-to-graph; else `isinstance(item, Construct)` → `_add_subgraph`. Both
`_contiguous_portal_mesh` and `_add_portal_mesh` are **already fully
Construct-agnostic**, verified line by line (`.name`, `.modifier_set.portal`
via the shared `Modifiable` base; `isinstance(member, Construct)` already
special-cased via `make_portal_subgraph_fn`, proven live by the passing
`portal_construct_member.py` fixture and `test_portal_cross_subconstruct.py`
for non-entry members). Runtime routing is one shared mechanism
(`_portal_route_to_command` in `factory.py`) across atomic/agent-act/Construct
member kinds. The recursion-limit floor (`runner.py`) is likewise already
generic over `Node | Construct` members.

**The Construct-as-mesh-entry bug (two sites, must move together)**:

1. `compiler.py:254`'s `isinstance(item, Node)` gate misroutes a
   Portal-modified `Construct` entry to `_add_subgraph`, whose PORTAL arm
   (`compiler.py:552-560`) unconditionally raises `CompileError` with a false
   "already rejected at assembly" comment.
2. `state.py:261-265`'s Portal state-field builder sources `portal_members`
   from `nodes_only` only (unlike the Oracle/Each block two lines below it,
   which correctly unions `nodes_only + sub_constructs`), so a Construct
   entry's hop-counter/payload state fields are never declared. Fixing site 1
   alone does NOT crash — `StateBus.get_counter`/`_ModelStateBus.get_counter`
   read via `getattr(self._state, key, None)` with a default, not a raising
   access — so the real failure mode is **silent state-key divergence**: the
   hop counter silently resets to 0 and the payload channel silently reads
   empty for any topology that loops back through, or routes via, the
   miskeyed Construct entry. This is WORSE than a crash (harder to detect,
   no stack trace pointing at the cause) and is a regression under the
   project's fail-loud-over-fail-soft north star even though it looks like
   progress. Both sites must land together, with a runtime-invoking test
   asserting the state model's declared Portal field names match
   `_wiring.py`'s actual `entry_field` for a Construct entry — a
   does-not-raise test would NOT catch this bug class.

Dispatch-mode Portal is legitimately Node-only by design (it runs a body to
synthesize a flow; a Construct has no single body) — a bona fide scope
boundary, not a gap.

**Verified capability boundary**: works today — any-size pure-Node mesh (any
mode mix), Construct as non-entry member, dispatch mode, full round-trip
export/import for a pure-Node mesh. Assembly-clean but compile/runtime-broken
— Construct as mesh entry (the two-site bug above). Not yet Agent-Spec
representable, independent of the compiler bug — any mesh containing a
Construct member at all (export filter misclassifies it as "mixed" before
reaching per-member lowering, which itself would then `AttributeError`).

---

## 3. Agent Spec export/import architecture

### Export (`_agent_spec.py`)

`_lower_construct_item` dispatches on `classify_modifiers(item)`. Of 12
`ModifierCombo` values, 5 are handled (`BARE`, `EACH`, `ORACLE`, `LOOP`,
`OPERATOR`); the other 7 (`PORTAL`, `EACH_OPERATOR`, `ORACLE_OPERATOR`,
`LOOP_OPERATOR`, `EACH_ORACLE`, `EACH_ORACLE_OPERATOR`, `PORTAL_OPERATOR`) fall
to a generic `ConfigurationError`. `compiler.py` runs **all 5 non-Portal
fusion combos** today at the Node level in real LangGraph pipelines — this is
the textbook "capability exists, just not wired to export" pattern; no new
lowering primitive is needed, only glue composing `_lower_each`/`_lower_oracle`/
`_lower_operator`/`_lower_loop`, which `_lower_construct_item` already calls
for the un-fused case.

`tests/test_agent_spec_matrix.py` mechanically ratifies today's
reject-everything-fused behavior as ground truth (its `SUPPORTED_COMBOS` is
exactly the 5 handled combos) rather than testing whether it is a genuine
limit — within that world export is fully green and fully tested; outside it,
zero coverage.

**Four additional confirmed gaps, independent of the `ModifierCombo` axis and
uncovered by the matrix in either direction:**

1. **Portal-mesh export false-rejection** — the `mesh_members` filter
   (`_agent_spec.py:961-967`) is `isinstance(item, Node)`-gated, so a legal,
   validator-certified mesh containing a do0d9-admitted Construct member is
   silently dropped from `mesh_members`, and the count mismatch fires a
   factually false **"mixes a Portal peer mesh with non-mesh nodes"** error.
2. **`PORTAL_OPERATOR` silently drops the Operator HITL gate** —
   `_lower_portal_mesh_to_swarm` builds a bare `Agent` per member via
   `_make_agent` and never inspects `member.modifier_set.operator`. A
   legitimate, compiler-supported combo (human-approval gate on a
   dynamic-routing path) vanishes with **no error, no marker** — the sharpest
   safety-relevant symptom in the whole audit (`neograph-s7zt3.2`).
3. **Oracle-variant guard-skip** — every combo arm routes its primary node
   through `_lower_node` (which calls `_reject_unrepresentable_fields`)
   *except* Oracle: `_lower_oracle`'s per-variant loop calls
   `_lower_generation_step` directly, bypassing the guard. An Oracle variant
   with `raw_fn` set silently exports as a name-only `ToolNode` stub — the
   real callable dropped, no error.
4. **Dict-form multi-output → dict-form-input consumer rejected** — the
   edge-wiring sweep rejects this unconditionally even though
   `_properties_for` already implements the identical `{key}.{field}`-prefix
   mechanism for the input side; it is simply never invoked for the producer
   side. High-confidence unwired capability, not a representability wall.

**A fifth gap, orthogonal to the `ModifierCombo` axis entirely**:
`_lower_construct_item` never calls `classify_modifiers` on a `Construct`
item at all — so `Construct(...) | Each(...)` (or Oracle, Loop, Operator)
**silently loses its modifier on export**, independently reproduced by
building the exact case and grepping the serialized output for zero trace of
the modifier, no error. This is a different failure mode than the 7-combo
rejection above (which at least fails loud) — this one is silent data loss.

### Import (`loader.py`)

`_group_flow_items` is a single forward recognize-and-emit pass, no
backtracking. Every reconstruct helper is a faithful, fail-loud-on-mismatch
inverse of its export counterpart, **except**:

- **Construct-as-mesh-member never reconstructed** — `_reconstruct_swarm_mesh`
  always builds agent-mode `Node`s via `_node_from_spec_agent`; mirrors
  export's Node-only filter from the other direction.
- **`_MARK_PORTAL_SPEC` written on export, never read on import** —
  `max_hops`/`on_exhaust`/`route` are silently dropped on re-import, **even
  for the pure-Node mesh case that otherwise round-trips correctly today**.
- **Loader never re-validates the reconstructed IR** — `_check_portal_mesh` is
  the single construct-assembly-time gate for every Portal mesh rule, called
  from exactly one site (`_construct_validation.py:348`); `loader.py` builds a
  brand-new `Construct` from imported spec data and never routes it through
  that gate (or an equivalent), unlike every other construction path.
- A confirmed-live `compiler.py` bug (the Construct-as-mesh-entry bug, §2) was
  found while chasing the mesh-member asymmetry, independently reproduced.

**Root cause, cutting across every export and import gap**:
`classify_modifiers`/`_group_portal_members`/`_check_portal_mesh` are already
type-agnostic (operate on `.modifier_set`, not `isinstance(Node)`) and are the
proven single sources of truth for `compiler.py`/`state.py`/`runner.py`/
`_wiring.py`. Agent Spec export/import is the one place in the codebase that
reimplements ad hoc, `isinstance`-gated detection instead of calling them —
by contrast, `iter_with_arms`/`iter_item_slots` (`_ir_branch.py`) and
`spec_types.py` ARE reused correctly and symmetrically, proving the pattern
is achievable, not a structural impossibility of the export/import layer.

---

## 4. The unified `ModifierCombo` single-source-of-truth design

**The finding**: at least ten modules independently re-derive combo
decomposition, not the three the original rewrite spec named:

| # | Module | What it re-derives |
|---|---|---|
| 1 | `compiler.py` (`_add_node_to_graph`, `_add_subgraph`) | full `(primary, has_operator)` decomposition, two separate `match combo:` blocks |
| 2 | `_agent_spec.py` (`_lower_construct_item`) | same decomposition, flat 5-branch chain, incomplete |
| 3 | `state.py:165,537,596` | same combo-to-bucket grouping, three separate `match combo:` blocks (byte-identical grouping, differing per-arm bodies) |
| 4 | `_state_write.py:72-97` | same primary-with-operator-orthogonal grouping |
| 5 | `_subconstruct.py:89-91` | `sub_combo in (LOOP, LOOP_OPERATOR)` / `(EACH, EACH_OPERATOR)` membership (Operator correctly not consulted — orthogonal wrapper, not a shape) |
| 6 | `_input_shape.py:32-33` | `combo in (LOOP, LOOP_OPERATOR)` |
| 7 | `runner.py:116,123,143,154` | `combo in (PORTAL, PORTAL_OPERATOR)`, four times across two functions |
| 8 | `_wiring.py:718` (+ reads at 713,725,853,865,912,997) | same Portal-shape re-derivation |
| 9 | `_fan_agent.py` | agent/act fan-modifier support checking (a genuine 10th site the original 9-module inventory missed) |
| 10 | `loader.py` | structurally different mechanism (marker/edge pattern-matching on exported pyagentspec shapes, not `classify_modifiers` calls) — belongs on the list, kept distinct so a grep-based consolidation doesn't miss it |

Two narrower precedents already solve adjacent problems correctly and are the
template to copy: `_COMBO_MAP` (raw modifier-set → `ModifierCombo`
classification) and `_group_portal_members` (which named mesh a Portal member
belongs to) — both single-sourced, both consumed correctly by multiple
modules.

**`SUB_CONSTRUCT_UNSUPPORTED_COMBOS`** is not a new restriction and not about
Agent Spec representability — it is neograph's own pre-existing compiler
restriction (`compiler.py:511-516`): `Each`+`Oracle` fusion is defined
entirely in terms of a single `Node`'s `map_over=`/`ensemble_n=` fields, which
a `Construct` structurally lacks. The Agent Spec exporter must mirror this
exact restriction, not silently narrow scope below what the compiler
elsewhere allows.

**Corrected Portal-exclusion rationale** (the rewrite spec got this wrong):
the spec claimed Portal-on-Construct is "impossible by construction" (mesh
membership requires a bare Node) — directly contradicted by do0d9's own
admission and a passing fixture. Portal is excluded from
`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` **not because it's impossible, but because
it needs its own dedicated handling**: a Construct mesh member routes through
the mesh path (`_contiguous_portal_mesh`/`_add_portal_mesh`), not the generic
Construct-item modifier-check path that the frozenset governs. That mesh path
itself has two live bugs (compiler entry-detection, export false-rejection)
that must be fixed first, plus a genuinely-new Construct-as-Swarm-member
lowering that is out of today's scope. The `_lower_construct_item` rewrite
must make an explicit, accurate decision for Portal-classified Construct
items reflecting this — not silently rely on the false "impossible" premise.

**Widened mandate**: all ten consumers must read from the shared
`COMBO_DECOMPOSITION`/`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` table in
`modifiers.py`; the structural anti-regrowth guard must enumerate all ten, not
three, or it will either fail immediately on landing (state.py already
contains the forbidden pattern) or get silently scoped down, leaving identical
duplication in six more places.

---

## 5. Complete matrix: every (ModifierCombo × Node-or-Construct × export/import) cell

All 12 `ModifierCombo` values × {Node, Construct} × {compile, export, import}.
Every cell below has a verified status — none are TODO/unknown.

| Combo | Node: compile | Node: export | Node: import | Construct: compile | Construct: export | Construct: import | Fix / reuse plan |
|---|---|---|---|---|---|---|---|
| **BARE** | WORKS | GREEN, tested | GREEN, tested | WORKS (plain recursion) | GREEN (recursion via `to_agent_spec`) | GREEN | None needed |
| **EACH** | WORKS | GREEN, tested | GREEN, tested | WORKS (`_subconstruct.py:89-91` groups EACH/EACH_OPERATOR at Construct granularity; `state.py` sub-construct-output-shaping block handles it) | **BROKEN — silent drop.** `_lower_construct_item` never calls `classify_modifiers` on a Construct item; modifier vanishes with no error | N/A (nothing emitted to import; needs new reconstruction branch once export fixed) | Gate on `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` (EACH not in it) inside `_lower_construct_item`, then reuse `_lower_each` on the Construct item mirroring the Node path; loader needs a new FlowNode→Construct-under-Map reconstruction branch, reusing the recursion pattern that already exists elsewhere in `loader.py` |
| **ORACLE** | WORKS | GREEN, tested | GREEN, tested | WORKS (`state.py`'s ORACLE/ORACLE_OPERATOR bucket applies to the sub-construct-output-shaping category too) | **BROKEN — same silent drop** | N/A, same as EACH | Same pattern, reuse `_lower_oracle` |
| **LOOP** | WORKS | GREEN, tested | GREEN, tested | WORKS (`test_loop.py`'s "Loop-on-Construct" suite; `_subconstruct.py`'s explicit `(LOOP, LOOP_OPERATOR)` check) | **BROKEN — same silent drop**, independently reproduced by building `Construct(...) \| Loop()` and grepping serialized output for zero trace | N/A, same as EACH | Same pattern, reuse `_lower_loop` |
| **OPERATOR** | WORKS | GREEN, tested | GREEN, tested | WORKS (`state.py`'s BARE/OPERATOR bucket applies uniformly) | **BROKEN — same silent drop** | N/A, same as EACH | Same pattern, reuse `_lower_operator` |
| **EACH_OPERATOR** | WORKS (both `compiler.py` match blocks handle all 6 groups) | **REJECTED** — 1 of 7 unhandled combos, `ConfigurationError` | N/A (nothing to import) | WORKS (`state.py` EACH/EACH_OPERATOR bucket, sub-construct category) | **REJECTED + silent-drop** (double-blocked: 7-combo gap AND Construct-item-modifier-drop) | N/A | Node level: A4, **first priority** among fusion combos ("single wrap"), glue = `_lower_each` + `_lower_operator` composed. Construct level: same glue once the modifier-drop fix (above) lands. Loader: new variable-length lookahead recognizing a trailing Operator pause on an Each group — real structural work, not a classification helper |
| **ORACLE_OPERATOR** | WORKS | **REJECTED**, same 7-combo bucket | N/A | WORKS (`state.py` ORACLE/ORACLE_OPERATOR bucket) | **REJECTED + silent-drop** | N/A | Same A4 track: glue `_lower_oracle` + `_lower_operator`; loader needs matching lookahead for a trailing Operator pause on an Oracle group |
| **LOOP_OPERATOR** | WORKS | **REJECTED**, same 7-combo bucket | N/A | WORKS (`state.py` LOOP/LOOP_OPERATOR bucket) | **REJECTED + silent-drop** | N/A | Same A4 track, **also first priority** ("single wrap"); glue `_lower_loop` + `_lower_operator`; loader lookahead for trailing Operator pause on a Loop group |
| **EACH_ORACLE** | WORKS (both compiler match blocks) | **REJECTED**, 7-combo bucket | N/A | **REJECTED by compiler itself** (`compiler.py:511-516`) — permanent, structural: fusion is defined via a single Node's `map_over`/`ensemble_n` fields, which Construct lacks. Not a duplication gap | Currently **silent-drop** (worse than the compiler's own fail-loud rejection) | N/A | Node level: A4, **third priority** ("two proven wraps composed" — MapNode whose inner subflow is what `_lower_oracle` already produces un-fused); loader needs to recognize a nested Oracle fan-out+merge inside a MapNode's subflow. Construct level: fix must make export **fail loud mirroring `compiler.py`'s exact rejection** — this itself is real, needed work (today it silently drops instead of erroring), but the target state is "no support," not "support" |
| **EACH_ORACLE_OPERATOR** | WORKS (three-way fusion, both match blocks) | **REJECTED**, 7-combo bucket | N/A | **REJECTED by compiler itself**, same permanent boundary as EACH_ORACLE (Operator doesn't change the underlying fusion-has-no-Construct-meaning fact) | Currently silent-drop; target is fail-loud parity with compiler | N/A | Node level: A4, **last priority** ("three-way", composing all three primitives); loader needs 3-level recognition lookahead. Construct level: same fail-loud-mirror fix as EACH_ORACLE |
| **PORTAL** (dispatch mode) | WORKS — standalone linear node, fully supported | **NO lowering exists at all** | N/A | N/A — dispatch mode is Node-only by design (a Construct has no single body to synthesize a flow from); permanent, legitimate scope boundary | N/A | N/A | Genuinely new design: check `tests/agent_spec_capabilities.py`'s pyagentspec primitive registry for a conditional/dynamic-routing node type before assuming infeasibility. Part of the C1/C2 tracked epic, not a one-line fix |
| **PORTAL** (peer, pure-Node mesh, any position) | WORKS, verified fixtures | **GREEN today** for a mesh with zero Construct members | GREEN, **but `_MARK_PORTAL_SPEC` (`max_hops`/`on_exhaust`/`route`) is silently dropped on re-import even in this working case** (B3) | — | — | — | B3: read the marker back in `_reconstruct_swarm_mesh`, copying the pattern two sibling reconstructors in the same file already use. Pure reuse |
| **PORTAL** (peer, Construct as **non-entry** member) | **WORKS** (do0d9, verified: `portal_construct_member.py` fixture, `test_portal_cross_subconstruct.py`) | **BROKEN** — `mesh_members` filter (Node-only `isinstance`) misclassifies the mesh as "mixed," false-rejects before reaching per-member lowering; even if the filter is fixed, `_lower_portal_mesh_to_swarm` would `AttributeError` on `member.prompt` for the Construct member | N/A — `_reconstruct_swarm_mesh` has no Construct-member branch (Asymmetry 1) | WORKS, same as Node column | Same BROKEN status | Same N/A status | A1 (filter fix, reuse `classify_modifiers` + `_group_portal_members`) unblocks correct detection; C1 (new `_make_agent`-equivalent Construct lowering, reusing the existing recursive-Flow-production pattern already in the same file) is required for the export to actually succeed rather than just be correctly detected. Import: new reconstruction branch, reusing the FlowNode→Construct recursion pattern used elsewhere in `loader.py`. **Largest single item — own tracked epic (C1)** |
| **PORTAL** (peer, Construct as **entry**) | **BROKEN — two-site compiler bug** (`compiler.py:254` isinstance gate + `state.py:261-265` field-builder gap); assembly-clean, compile-time CompileError today, **silent state-key divergence** (hop counter silently resets to 0 / payload channel silently empty — NOT a crash, `StateBus.get_counter` reads via `getattr(..., default=None)`) if only site 1 is naively fixed | N/A until compiler fixed | N/A until compiler fixed | same as Node column (a Portal-carrying Construct entry IS the Construct-level case) | N/A until A3 lands | N/A until A3 lands | **A3 must land first**: one-line `isinstance(item, (Node, Construct))` relaxation in `compiler.py:254` (dispatch mode line stays Node-only) + the `state.py` `nodes_only + sub_constructs` fix, landed together with a runtime-invoking test asserting the state model's declared Portal field names match `_wiring.py`'s actual `entry_field` (not a does-not-raise test). Then the same C1 Construct-lowering work as the non-entry case applies |
| **PORTAL_OPERATOR** (mesh member, atomic) | **WORKS**, legitimately (validator requires atomic scripted/think/raw for Operator-gated members) | **BROKEN — silently drops the Operator HITL gate**, no error, no marker (`neograph-s7zt3.2`, the sharpest safety bug in the audit) | N/A — nothing preserves it because nothing is emitted | — (mesh membership is a Node/Construct-item concept; the "atomic" requirement below governs Construct eligibility) | — | — | B1: branch on `member.modifier_set.operator` in the per-member loop, reuse `_lower_operator`'s gate-attachment logic via a mesh-exit pause composite (`AgentNode` → `BranchingNode`/`InputMessageNode`, verified live against installed pyagentspec — `Swarm` has no interior per-member pause primitive). Reuses existing gate logic; new design only for how it attaches to `Swarm`. **Should not be delayed past the reuse-fix phase** given its safety severity |
| **PORTAL_OPERATOR** (Construct as mesh member) | **REJECTED at assembly**, by design (`_validation_portal.py`: "Operator-gated members must be atomic... Construct members with Operator are rejected") — genuinely, permanently unrepresentable, not a duplication bug | N/A — nothing to export, assembly already rejects it | N/A | REJECTED at assembly, same rule | N/A | N/A | No fix needed for support; verify Agent Spec's error path (if any) matches assembly's rejection message for fail-loud parity — a minor verification item, not a design gap |

---

## 6. Build plan (supersedes the rewrite spec's Build Plan; ordered to avoid long red stretches)

**Phase 0 — additive, zero risk.** Land `COMBO_DECOMPOSITION` / `PrimaryShape`
/ `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` in `modifiers.py`, read-only, consumed by
nothing yet. No guard yet (a guard now would immediately fail against
`state.py`/`_state_write.py`'s existing hand-rolled pattern).

**Phase 1 — the two-site Portal compiler bug (A3), must land before any
Agent-Spec Portal-mesh work is meaningful.** `compiler.py:254` isinstance
relaxation + `state.py:261-265` `nodes_only + sub_constructs` fix, landed
together, with a runtime-invoking test (not compile-only) per §2/§5. Update
the two now-stale "already rejected at assembly" comments with corrected
justification text.

**Phase 2 — migrate the reference implementation first.** Rewire
`compiler.py`'s two `match combo:` blocks to consume `COMBO_DECOMPOSITION`
instead of their hand-rolled decomposition. This proves the table against the
one implementation everything else must match, before any other consumer
depends on it.

**Phase 3 — migrate the remaining six non-Agent-Spec consumers.**
`state.py` (three blocks), `_state_write.py`, `_subconstruct.py`,
`_input_shape.py`, `runner.py`, `_wiring.py` — each is a mechanical
"shared classification, different per-arm body" swap, verified individually
(per the combo doc's mandate) rather than assumed to transfer cleanly. Land
the structural anti-regrowth guard now, enumerating all ten consumers
(the eight migrated so far plus `_agent_spec.py`/`loader.py`, still pending).

**Phase 4 — fix the Construct-item-modifier-drop bug.**
`_lower_construct_item` gains a `classify_modifiers` call for `Construct`
items, checked against `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` first (mirroring
`_add_subgraph`'s existing pattern). Turns silent data loss into either
correct export or accurate fail-loud for EACH/ORACLE/LOOP/OPERATOR-on-Construct.

**Phase 5 — pure reuse-and-delete fixes, parallelizable, no design
decisions.** A1 (export mesh-detection filter: `classify_modifiers` +
`_group_portal_members`, replacing the isinstance-gated filter), A5 (loader
re-validation: route reconstructed IR through `_check_portal_mesh`), B3
(`_MARK_PORTAL_SPEC` read-back in `_reconstruct_swarm_mesh`), B2 (Oracle
guard-skip: add `_reject_unrepresentable_fields` call to the per-variant
loop), B4 (dict-form outputs: apply `_properties_for`'s existing prefix logic
to the producer side). None of these five invent new semantics.

**Phase 6 — B1, the highest-severity remaining safety gap.**
`PORTAL_OPERATOR` gate-preservation: branch on `member.modifier_set.operator`
in `_lower_portal_mesh_to_swarm`'s per-member loop, attach via a mesh-exit
pause composite. Bounded new design (how the composite attaches to `Swarm`),
reusing `_lower_operator`'s existing gate logic. Prioritized ahead of the
larger fusion-combo work below because it is a currently-silent control-flow
seam disappearing, not a missing feature.

**Phase 7 — A4, the 5 non-Portal fusion combos, one at a time, each landed
in lockstep with its matching loader.py recognition-walk extension** (per
the corrected understanding in §4/§5 — this is real structural lookahead
work, not a classification-helper add-on). Order: `EACH_OPERATOR`/
`LOOP_OPERATOR` first (single wrap), `EACH_ORACLE` next (two proven wraps
composed), `EACH_ORACLE_OPERATOR` last (three-way), `ORACLE_OPERATOR`
alongside the single-wrap pair. Each combo gets new round-trip fixtures
(none exist today for any of these cells) before moving to the next.

**Phase 8 — extend the same fusion-combo glue to the Construct level**, now
that Phase 4 (Construct-item modifier check) and Phase 7 (Node-level glue)
both exist: `EACH_OPERATOR`/`ORACLE_OPERATOR`/`LOOP_OPERATOR`-on-Construct via
the same primitives; `EACH_ORACLE`/`EACH_ORACLE_OPERATOR`-on-Construct
implemented as an explicit fail-loud mirror of `compiler.py`'s permanent
rejection (target is accurate error, not new support).

**Phase 9 — C1/C2/dispatch-mode Portal, the largest remaining items, own
tracked epic.** Full Construct-as-Portal-mesh-member export/import (entry and
non-entry), Swarm-in-Flow feasibility (C2), and dispatch-mode Portal's
Flow-node lowering. Gate on a `tests/agent_spec_capabilities.py` primitive-
capability check against the installed pyagentspec package before committing
to any implementation approach — do not assume infeasibility without that
check, per the executive summary's core lesson.

**Phase 10 — cleanup.** Remove `_MARK_REMOTE_AGENT` (confirmed dead code,
zero references outside its own definition) as a small, separate ticket.

This ordering keeps every phase either purely additive, a small mechanical
relaxation with an immediate test, or a reuse-and-delete fix with no open
design question, before any phase requiring genuinely new design (Phases 6, 7
partially, and 9) — so no phase leaves the suite red for longer than its own
bounded fix takes.

---

## Consolidated Review Verdict (2026-07-27)

Three independent reviews were run against this document — completeness,
correctness (repro-based, not read-only), and consistency with the
investigation's own stated principles. All three are on disk alongside this
file (`-review-completeness.md`, `-review-correctness.md`,
`-review-consistency.md`). This section consolidates their combined findings
into one verdict and one action list.

### Overall verdict: **NEEDS-REVISION (minor)**

The document's architecture, root-cause analysis, matrix completeness, and
build-plan ordering all hold up under independent scrutiny — no reviewer found
a structural or consistency defect, and the correctness reviewer confirmed 3
of 4 spot-checked claims exactly as written via live repro (not just reading).
It is **not SOUND-as-written** only because one confirmed claim's *causal
mechanism* is factually wrong in a way that would mis-scope its Phase 1
acceptance test, and because one cited bug site's live status may have already
changed under a concurrent in-flight edit. Both are narrow, textual fixes, not
re-analysis — the underlying engineering conclusions they attach to are
correct and do not need to be redone.

### Prioritized action list (deduplicated across all three reviews)

1. **RESOLVED (2026-07-27).** ~~Fix the Claim-2 failure-mechanism
   description~~ — §2 and §5's PORTAL-Construct-as-entry row now correctly
   state the failure mode as **silent state-key divergence** (hop counter
   silently resets to 0, payload channel silently empty), not a crash,
   grounded in `StateBus.get_counter`/`_ModelStateBus.get_counter`'s
   `getattr(self._state, key, None)` default-read semantics. Both locations'
   acceptance-test language now reads "state model's declared Portal field
   names match `_wiring.py`'s actual `entry_field`," not "does not raise."
2. **RESOLVED (2026-07-27).** ~~Re-verify `compiler.py:254`'s live status~~ —
   re-checked directly against HEAD: `compiler.py:254` still reads
   `isinstance(item, Node)` (unchanged, no concurrent edit occurred) and
   `state.py`'s `portal_members` filter (the `nodes_only`-only list
   comprehension) still excludes `sub_constructs`. Both sites are confirmed,
   as of this correction pass, in the exact state this document describes —
   Phase 1 can proceed on the document's site descriptions as written, no
   stale-read risk found.
3. **Add one clarifying sentence to §5's preamble on the agent/act fan-shape
   axis, and one explicit citation for the fan-wrap/export interaction —
   completeness reviewer, both minor.** (a) `_fan_agent.py`'s
   fan-shape-legality gate runs before compile ever sees an
   Each/Oracle/Loop-over-agent node, so §5's "Node: compile WORKS" cells for
   those rows are implicitly conditioned on `_fan_agent.py` admitting the
   shape — not a broken claim, but worth one sentence so a reader doesn't read
   "WORKS" as unconditional. (b) The document never states, with the same
   file:line rigor as its other claims, that `_wrap_fan_over_agents` runs at
   *compile* time (after Agent-Spec export would have already run), so
   `to_agent_spec` sees the pre-wrap IR and dispatches through
   `_lower_generation_step`'s already-confirmed agent/act handling
   (`_agent_spec.py` lines 331+). This is very likely correct by the
   mechanism the document credits elsewhere, but should be stated as
   "verified" or explicitly flagged "inferred, not verified" rather than left
   implicit.
4. **Documentation-hygiene note, no action required for this document but
   worth a follow-up ticket** — correctness reviewer: the header comment in
   `tests/check_fixtures/should_pass/portal_construct_member.py` and the
   module docstring in `tests/test_portal_cross_subconstruct.py` both still
   say the tests are in a "TDD-red"/"FAIL today" state; both suites currently
   pass in full. Not a defect in this document (it correctly describes the
   capability as working), but a stale-comment cleanup item worth its own
   small ticket so a future reader skimming only the fixture file isn't misled
   in the opposite direction from item 2 above.

No other findings from any of the three reviews rise above these four — the
consistency review found zero inconsistencies, and the completeness review's
remaining observations are folded into action item 3.

### Ready to implement vs. needs more design

**Ready to implement now, as designed:**
- Phase 0 (additive `COMBO_DECOMPOSITION` table, zero consumers yet)
- Phase 2/3 (migrate `compiler.py` then the six remaining non-Agent-Spec
  consumers onto the shared table, with the anti-regrowth guard)
- Phase 4 (Construct-item `classify_modifiers` call — fixes the confirmed
  silent-drop bug, correctness-verified via repro)
- Phase 5 (A1, A5, B3, B2, B4 — five pure reuse-and-delete fixes, no open
  design question in any of them)
- Phase 7 (A4, the 5 non-Portal fusion combos — confirmed glue-only, no new
  primitive needed)
- Phase 8 (Construct-level fusion combos, once Phases 4 and 7 land)
- Phase 10 (dead-code cleanup)

**Ready to implement only after the Phase 1 correction above lands:**
- Phase 1 itself (the two-site Portal-entry bug) — the *fix* (land both
  sites together) is correct and does not need re-design, but its written
  acceptance criterion must be corrected per action item 1 before it is
  handed to an implementer, and site 1's current status must be re-checked
  per action item 2.

**Still needs design work before implementation, exactly as the document
itself already flags (confirmed, not weakened, by all three reviews):**
- Phase 6 / B1 (`PORTAL_OPERATOR` HITL-gate preservation on export) — the
  gate-attachment logic is reusable, but how a pause composite attaches to
  `Swarm` is a genuine open design question; correctly prioritized ahead of
  Phase 7 given its safety severity (confirmed CONFIRMED by correctness
  review as a live, silent gap).
- Phase 9 (C1/C2 Construct-as-mesh-member export/import, dispatch-mode
  Portal's Flow-node lowering) — explicitly gated on checking
  `tests/agent_spec_capabilities.py`'s installed-pyagentspec primitive
  registry before committing to an approach; correctly deferred as its own
  tracked epic, not scoped down to a quick fix.
