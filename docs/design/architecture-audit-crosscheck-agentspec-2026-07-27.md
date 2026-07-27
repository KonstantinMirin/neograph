# Crosscheck: false "not supported" claims x existing capabilities x export/import fix sites (2026-07-27)

Synthesis of the four 2026-07-27 survey docs (false-claims, existing-capabilities,
export-inventory, import-inventory). For every FALSE/PARTIALLY-FALSE claim, this
names the exact reusable capability that closes it, the exact code site(s) to
change, and the concrete fix shape. Gaps with no reusable capability are called
out as genuinely new work, separated from the reuse cases. Silent-loss findings
(not fail-loud, but wrong) are folded in since they are worse than the fail-loud
false claims and were found by the same audit pass.

## A. Reuse cases — existing capability closes the gap

### A1. Portal-mesh export mis-detection ("mixes mesh with non-mesh nodes")
- **False claim**: `_agent_spec.py:948-977` `to_agent_spec()` rejects a
  construct containing a Portal-carrying `Construct` member as "mixed",
  even though `_check_portal_mesh` already certified it as a uniform mesh.
- **Root cause**: hand-rolled filter `isinstance(item, Node) and
  item.modifier_set.portal is not None and not is_dispatch` — `isinstance`-gated
  to `Node`, silently drops `Construct` members from `mesh_members`.
- **Reusable capability**: `classify_modifiers` (`modifiers.py:91-165`) +
  `_group_portal_members` (`modifiers.py:727`) — both type-agnostic (operate on
  `.modifier_set`, not `isinstance(Node)`), both already proven correct and used
  by `compiler.py`/`state.py`/`runner.py`/`_wiring.py`/`_validation_portal.py`/
  `_ir_normalize.py`. `_agent_spec.py:844` already calls `classify_modifiers` for
  per-item dispatch elsewhere in the same file — it just isn't used for mesh
  detection.
- **Fix site**: `_agent_spec.py`'s `to_agent_spec`, the `mesh_members`
  computation at line ~961-967. Replace the `isinstance`-gated list
  comprehension with `classify_modifiers(item)[0] in (ModifierCombo.PORTAL,
  ModifierCombo.PORTAL_OPERATOR)`, and replace the ad hoc "is it the whole
  construct" check with `_group_portal_members` so multiple distinctly-named
  adjacent meshes are correctly disambiguated instead of merged (existing-
  capabilities doc item 2's second finding — a real bug this fix must not
  reintroduce). This is a reuse-and-delete fix, not new logic.

### A2. `compiler.py` Portal-on-subconstruct, non-entry case
- **False claim** (the FALSE half of #2, "partially false"): non-entry
  Construct mesh members are claimed unsupported.
- **Reality**: `_contiguous_portal_mesh` + `_add_portal_mesh` (`_wiring.py`)
  already route to a `Construct` member correctly today; `_add_subgraph`'s
  reject-arm is simply never reached for these members because they're marked
  `meshed` first. **No fix needed here** — this is not a gap, it's a
  documentation/messaging correction: the "not supported" comment at
  `compiler.py:~556` should be narrowed to say "mesh entry only," since it
  currently reads as blanket and is misleading (same misleading-message pattern
  as A4 below, but on the compiler side, not the Agent Spec side).

### A3. `compiler.py` Portal-on-subconstruct, entry case — the one TRUE compiler bug in this batch
- **Claim**: TRUE-infeasible today, but for a narrow, mechanical reason: mesh-
  entry detection in `compile()` (`compiler.py:251-254`) is
  `isinstance(item, Node) and classify_modifiers(item)[0] in (PORTAL,
  PORTAL_OPERATOR)` — `Construct` can never satisfy `isinstance(item, Node)`.
- **Reusable capability**: `classify_modifiers` itself is already
  type-agnostic (operates on `.modifier_set`, present on both `Node` and
  `Construct` per `construct.py:137`) — the bug is purely the redundant
  `isinstance` guard sitting in front of it.
- **Fix site**: `compiler.py:251-254`. Drop the `isinstance(item, Node)`
  conjunct, leave the `classify_modifiers(item)[0] in (...)` check standing
  alone. This is the fifth site in the do0d9 family (existing-capabilities doc
  calls the other four sites already relaxed) — same one-line relaxation
  pattern, not new semantics. **This is real, scoped compiler work**, distinct
  from the Agent Spec fixes, and should land first since Agent Spec export of a
  Construct-as-mesh-entry is meaningless if the compiler itself can't run it.

### A4. `_agent_spec.py` composed non-Portal combo rejection (5 combos)
- **False claim**: `EACH_ORACLE`, `EACH_OPERATOR`, `ORACLE_OPERATOR`,
  `LOOP_OPERATOR`, `EACH_ORACLE_OPERATOR` rejected as "no Agent Spec lowering
  yet" at `_lower_construct_item`'s catch-all (`_agent_spec.py:872-877`).
- **Reality**: `compiler.py` has explicit `case` arms for every one of these
  at both its dispatch sites (lines ~505-560, ~590-670) — they compile to real
  running LangGraph pipelines.
- **Reusable capability**: none of these needs a NEW capability — each is a
  composition of the SAME three already-lowered primitives (`_lower_each`,
  `_lower_oracle`, `_lower_operator`, `_lower_loop`) that `_lower_construct_item`
  already calls for the un-fused case. The gap is that the dispatch is a flat
  `match` over 5 named arms with no fallthrough/fusion path, not a missing
  primitive.
- **Fix site**: `_agent_spec.py:846-877`, `_lower_construct_item`'s match.
  This needs genuinely new per-combo lowering code (nested wrapping: e.g.
  `EACH_ORACLE` = an Each-shaped pyagentspec `MapNode` whose inner is what
  `_lower_oracle` already produces for an un-fused Oracle node) — call it
  **reuse-of-primitives, new-glue-code**, not reuse-of-a-ready-made-function.
  Distinguish from A1/A3: those are one-line/one-expression swaps; this is a
  real (if bounded) implementation task per combo, using existing lowering
  functions as building blocks. Recommend implementing in order of the
  compiler's own fusion families: EACH_OPERATOR and LOOP_OPERATOR first (single
  extra wrap around an already-not-fused case), EACH_ORACLE next (two
  independently-proven wraps composed), EACH_ORACLE_OPERATOR last (three-way).
  Each needs its own `should_pass`-style Agent Spec round-trip fixture — none
  exist today (`test_agent_spec_matrix.py`'s `UNSUPPORTED_COMBOS` explicitly
  excludes all 5, so there is zero regression coverage to protect during the
  rewrite).

### A5. Loader's unvalidated Portal-mesh + general reconstruction
- **Gap** (existing-capabilities doc item 3, reinforced by import-inventory
  §8): `loader.py` reconstructs a brand-new `Construct` from imported spec data
  and never re-runs `_check_portal_mesh` (or any equivalent whole-construct
  validation) against it.
- **Reusable capability**: `_check_portal_mesh` (`_validation_portal.py:40`)
  is already the single, construct-assembly-time gate for every Portal mesh
  rule — called today from exactly one site, `_construct_validation.py:348`
  (i.e., it already runs automatically for any construct built via the normal
  assembly path).
- **Fix site**: `loader.py`'s `from_agent_spec` should either (a) route its
  reconstructed pipeline through the SAME `Construct(...)` assembly path that
  `_construct_validation.py:348` already calls `_check_portal_mesh` from
  (preferred — zero new validation code, guaranteed parity with every other
  construction path), or (b) if the reconstruction bypasses normal assembly for
  some structural reason, explicitly call `_check_portal_mesh` before returning.
  This is a pure reuse fix — no new validation logic, just make sure the
  reconstructed IR goes through the existing gate instead of skipping it.

## B. Silent-loss findings (worse than fail-loud; folded in because same audit pass, same fix family)

### B1. `PORTAL_OPERATOR` peer-mesh member silently drops the Operator HITL gate
- `_lower_portal_mesh_to_swarm` (`_agent_spec.py:880-945`) builds one `Agent`
  per member via `_make_agent` and never inspects `member.modifier_set.operator`
  — a `PORTAL_OPERATOR` member (Portal + human-approval gate) exports as a bare
  `Agent`, silently dropping the interrupt-when condition.
- **No existing "reusable function" closes this directly** — `_make_agent`
  needs a new branch that inspects `.modifier_set.operator` and attaches the
  gate condition the SAME way `_lower_operator`'s non-mesh path already does
  (that non-mesh lowering IS the reusable reference — copy its gate-attachment
  logic into `_make_agent`, don't invent new Operator-on-Swarm semantics).
  Classify as: **reuse-the-existing-Operator-lowering-logic, new call site**.
- **Fix site**: `_agent_spec.py`, inside the per-member loop of
  `_lower_portal_mesh_to_swarm` (~914-924), branch on
  `member.modifier_set.operator` and reuse whatever `_lower_operator` does to
  express a gate (its `BranchingNode`/pause-node shape, or an equivalent
  Swarm-compatible representation — needs a design decision on how HITL gates
  attach to Swarm `Agent`s, since Swarm has no `Flow`-level `BranchingNode`
  concept; this sub-piece IS new design work, not pure reuse).

### B2. Oracle-variant guard-skip: `raw_fn`/`skip_when`/`renderer` silently exported as name-only stub
- `_lower_oracle`'s per-variant loop calls `_lower_generation_step` directly,
  bypassing `_lower_node` and therefore bypassing `_reject_unrepresentable_fields`
  entirely — an Oracle variant with `raw_fn` silently exports as a
  name-only `ToolNode`, the real callable dropped with no error.
- **Reusable capability**: `_reject_unrepresentable_fields` itself — it
  already exists, is already correct, and is already called from `_lower_node`.
  This is a pure "call the existing guard from a second call site" fix.
- **Fix site**: `_agent_spec.py`'s Oracle per-variant loop (~521-543, calling
  `_lower_generation_step` directly) — add a
  `_reject_unrepresentable_fields(variant_node)` call before
  `_lower_generation_step`, mirroring what `_lower_node` already does. Zero new
  logic, one added call.

### B3. `neograph/portal_spec` marker (`max_hops`/`on_exhaust`/`route`) written but never read
- `_lower_portal_mesh_to_swarm` writes `_MARK_PORTAL_SPEC` unconditionally;
  `_reconstruct_swarm_mesh` never reads it back (confirmed absent from
  `loader.py` by grep) — silent drop on re-import.
- **Reusable capability**: the marker-read pattern already exists and is
  proven elsewhere in the SAME file — `_reconstruct_agent_node` reads
  `neograph/agent_spec` and `_reconstruct_oracle_group` reads
  `_MARK_ORACLE_SPEC` the identical way. This is copy-the-pattern reuse, not new
  design.
- **Fix site**: `loader.py`'s `_reconstruct_swarm_mesh` (~689-734) — read
  `_MARK_PORTAL_SPEC` off the Swarm/marker the same way the other two
  reconstructors read their markers, and pass `max_hops=`/`on_exhaust=`/`route=`
  into the reconstructed `Portal(to=peers, ...)`.

### B4. Dict-form outputs -> dict-form inputs edge rejected at export
- `to_agent_spec`'s edge-wiring sweep (1093-1102, 1115-1117) rejects a
  dict-form-output producer referenced by a downstream dict-form input.
- **Reusable capability**: `_properties_for` (line 275) already does the
  exact same "one `Property` per dict key with a `{key}.{field}`-style prefix"
  for the INPUT side (lines 283-290) — the identical mechanism, just never
  invoked for the producer/output side.
- **Fix site**: `_agent_spec.py`'s edge-wiring sweep, ~1093-1117 — call the
  same `{key}.{field}`-prefixing logic `_properties_for` already has, applied
  to the upstream producer's dict-form outputs, instead of raising. Pure reuse.

## C. Genuinely new work — no existing capability to reuse

### C1. Construct-as-Portal-mesh-member export/import (the "big" gap)
Even after A1 (fixing the detection filter) and A3 (fixing compiler entry
detection), `_lower_portal_mesh_to_swarm` itself unconditionally accesses
`member.prompt`, `member.inputs`, `member.name`, `entry.modifier_set.portal`
assuming `Node`-shaped attributes — a `Construct` has none of these in the same
shape, so it would `AttributeError`, not degrade gracefully. **This needs new
`_make_agent`-equivalent lowering for a `Construct` member** (likely: lower the
Construct's own sub-pipeline to a nested `Flow`/`FlowNode` and wrap THAT as the
Swarm's per-member unit, rather than assuming an `Agent`). Symmetrically,
`_reconstruct_swarm_mesh` (loader.py:715) always builds an agent-mode `Node`
via `_node_from_spec_agent` for every mesh member — it needs a new branch that
recognizes a member's Agent Spec shape corresponds to a sub-`Flow` and
reconstructs a `Construct` instead, mirroring the existing (but currently only
reachable from the non-mesh path) `FlowNode` -> `Construct` recursion
`from_agent_spec` already does at the top level (loader.py §7, "the only place
loader.py reconstructs a Construct"). That recursion IS a reusable pattern to
copy, but the Swarm-specific wiring around it (member ordering, entry
detection, handoff routing) is new. Net: **partial reuse of the recursion
pattern, but real new design + code on both the export lowering and the import
reconstruction sides** — this is the largest single item in the whole
crosscheck and should be its own tracked epic, not a one-line fix.

### C2. Mixed Portal-mesh + non-mesh construct (Swarm wrapped in a Flow)
False-claims doc flags this as "LOW — plausible genuine Swarm-vs-Flow
type-shape limit, worth a second look but not obviously wrong." Cross-checked:
`Swarm` is pyagentspec's own top-level `AgenticComponent`, distinct from `Flow`
— there is no existing pyagentspec-side "FlowNode wrapping a Swarm" pattern
found anywhere in the export/import inventories to reuse. **Genuinely open
question, not a reuse case** — needs either a pyagentspec-level design decision
(does the SDK even support nesting a Swarm inside a Flow node?) before any
neograph-side code is written. Do not implement against an assumption; check
pyagentspec's own primitive set first (this is exactly what
`tests/agent_spec_capabilities.py`'s registry-completeness axis is for).

### C3. `EACH_ORACLE` fusion on sub-constructs (compiler.py:511-516)
Confirmed TRUE-infeasible by the false-claims doc (#4) and not contradicted
by any capability in the existing-capabilities inventory: `EACH_ORACLE` is
defined entirely in terms of a single `Node`'s `map_over`/`ensemble_n` fields;
a multi-node `Construct` has no structural analog. **No reuse path exists
because there is nothing to reuse — this is a real, accepted scope boundary**,
already documented as intentional in `agent-spec-rewrite-2026-07-27.md:83`.
Leave as-is; do not add speculative Construct-level Each x Oracle fusion
without a concrete driving use case.

### C4. `PORTAL` (dispatch mode) has no Flow-node lowering at all
Distinct from the peer-mesh case (A1/C1): a dispatch-mode Portal node
(`is_dispatch=True`) mixed into an otherwise-plain pipeline hits
`_lower_construct_item`'s same generic catch-all with a misleading "composed
modifier" message, but the real gap is structural — there is no Flow-node
representation of "jump to a named entry port" in pyagentspec at all (Flow's
edges are static; Portal dispatch is runtime `Command(goto=)`). **This is new
design work** (does pyagentspec have a conditional/dynamic-routing node type
to lower into, e.g. a `BranchingNode` keyed on a runtime-computed target?) —
check `tests/agent_spec_capabilities.py`'s primitive registry before assuming
infeasibility; do not copy A4's "just wire up existing primitives" verdict here
without that check, since dispatch-mode Portal has no existing lowered-form
precedent anywhere in `_agent_spec.py` today (unlike A4's 5 combos, which reuse
proven single-modifier lowerings).

## D. Priority ordering (dependency-driven, not severity-driven)

1. **A3** (compiler.py Construct-as-mesh-entry `isinstance` fix) — must land
   first; A1/C1 export fixes are moot if the compiler itself can't run the
   shape being exported.
2. **A1** (export mesh-detection filter) + **A5** (loader validation gate) —
   both pure reuse, independent of each other, unblock correct fail-loud
   behavior immediately (turns a false rejection into either a correct export
   or a correct, validated rejection).
3. **B2, B3, B4** — all pure reuse-the-existing-helper fixes, no design
   decisions needed, safe to parallelize.
4. **B1** (Operator-on-Swarm gate) and **A4** (5 fused combos) — bounded new
   glue code reusing existing lowering primitives; needs new round-trip
   fixtures since none exist today.
5. **C1** (Construct-as-mesh-member, full) and **C2**/**C4** (Swarm-wrapped-
   in-Flow, dispatch-mode Portal lowering) — genuinely new design, gate on a
   pyagentspec primitive-capability check before committing to an approach.

## Key cross-cutting note
Every reuse case (A1, A3, A5, B2, B3, B4) shares one root cause: Agent Spec
export/import re-derive type-shape and marker logic ad hoc instead of calling
the SAME functions (`classify_modifiers`, `_group_portal_members`,
`_check_portal_mesh`, `_reject_unrepresentable_fields`, `_properties_for`,
existing marker-read helpers) that are already proven correct and already used
by `compiler.py`/`state.py`/`runner.py`/`_wiring.py`/`_validation_portal.py`.
None of these six fixes requires inventing new semantics — they require
deleting a hand-rolled reimplementation and calling the existing helper
instead. Only C1/C2/C4 (Construct-as-mesh-member, Swarm-in-Flow, dispatch-mode
Portal lowering) and the bounded new-glue-code items (A4, B1) require actual
new design/implementation work.
