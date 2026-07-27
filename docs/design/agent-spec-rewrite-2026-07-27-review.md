# Review: `agent-spec-rewrite-2026-07-27.md`

Reviewer: independent skeptical pass. Every claim below was checked against the
real source as it exists on `develop` today (not the design doc's own
description of it), plus `bd show` on the epic and every child, `git show` on
the cited commit, and one live empirical repro against the installed
`pyagentspec` package. Per this epic's own history (two prior design docs each
needed a correction round), no claim below was taken on the doc's authority
alone.

---

## 1. The shared-table design (§1)

**PARTIALLY-CONFIRMED — the decomposition table's cell values are correct, but
the "5 match arms" framing in §0/§1.5 is imprecise, and the build-plan sketch
addresses only half of `compiler.py`'s actual dispatch surface.**

- `compiler.py` has **two independent match statements** over `ModifierCombo`,
  not one: `_add_node_to_graph` (Node-level, `compiler.py:595-675`) and
  `_add_subgraph` (Construct/sub-construct-level, `compiler.py:509-562`). The
  spec's §1.5 sketch shows a single `match decomp.primary` block and never
  mentions that there are two separate call sites with **different bodies for
  the same combo** — e.g. `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` dispatches to
  `_add_each_oracle_fused` at the Node level but raises `CompileError` at the
  Construct level (`compiler.py:510-517`); `PORTAL`/`PORTAL_OPERATOR` is
  unreachable-by-construction at the Node level (`compiler.py:665-673`, a
  defense-in-depth arm) but raises `CompileError` unconditionally at the
  Construct level (`compiler.py:552-560`). A single `COMBO_DECOMPOSITION`
  table keyed only by `(primary, has_operator)` cannot express "Node-level:
  real lowering; Construct-level: reject" — the spec's sketch implicitly
  assumes one dispatch surface where there are two, each needing its own
  primary-shape-to-body mapping (or an explicit third axis: "is this item a
  Node or a Construct"). This is a real, non-cosmetic gap in the "zero
  behavior change" claim for Step 0 (§6): refactoring "compiler.py's 5 match
  arms" as if there's one match statement will not compile against the actual
  two-match-statement structure without redesign.
- Each of the two match statements has **6 case arms**, not 5:
  `EACH_ORACLE | EACH_ORACLE_OPERATOR` is its own explicit `case` (e.g.
  `compiler.py:596-607`), separate from the `EACH | EACH_OPERATOR` arm
  (`compiler.py:619-629`). Functionally this is consistent with the table's
  own `primary=EACH` classification for `EACH_ORACLE*` (§1.1, correctly
  flagged "fused, see 2.1"), so the decomposition's *values* are right — but
  §0's prose ("5 match arms... each end with a uniform Operator-wrapping
  postlude") undercounts the real arm count by one and should say 6, one of
  which (`EACH_ORACLE*`) is a special-cased fusion rather than a plain
  primary/operator pair. Minor, but it means §0's own summary of "what
  `compiler.py` proves" is not quite what `compiler.py` contains.
- §1.4's CLAUDE.md clearance is **CONFIRMED accurate on the literal text**:
  the off-limits list is exactly `"node.py, construct.py,
  _construct_validation.py, factory.py, modifiers.py are off-limits for
  @node-layer features"` (CLAUDE.md, Layer discipline section), and
  `compiler.py` is not in that list. The argument that `COMBO_DECOMPOSITION`
  is "IR-level metadata... squarely inside what modifiers.py is for" is
  reasonable by analogy to `_COMBO_MAP` (same file, same shape: a frozen
  dict keyed by the enum), but it is a judgment call, not a mechanically
  verified fact — flag as sound-but-argued, not proven.

## 2. Per-combo feasibility (§2)

**CONFIRMED, including by empirical repro, for the three combos checked in
depth.**

- **§2.7 `PORTAL_OPERATOR`** — verified by reading `pyagentspec/swarm.py` and
  `pyagentspec/flows/nodes/agentnode.py` directly, plus a live repro:
  `Swarm(AgenticComponent)` (`swarm.py:77`) and
  `AgentNode.agent: SerializeAsAny[AgenticComponent]` (`agentnode.py:96`) are
  exactly as claimed. Repro: `AgentNode(name="wrap", agent=swarm)` where
  `swarm = Swarm(name="mesh", first_agent=a1, relationships=[(a1, a2)])`
  constructs cleanly with no validation error, and
  `node._get_inferred_inputs()` / `_get_inferred_outputs()` both return `[]`
  (Swarm has no I/O properties of its own) — so wrapping a `Swarm` in an
  `AgentNode` inside an enclosing `Flow` for a mesh-exit pause is confirmed
  mechanically possible with no data-flow wiring required for the wrapper
  itself. Also confirmed: `Swarm`'s only fields are `first_agent`,
  `relationships`, `handoff: HandoffMode` (`swarm.py:105-124`) — there is
  genuinely no interior pause/branch primitive on `Swarm` itself, matching the
  spec's claim that a per-member interior pause doesn't exist natively.
- **`BranchingNode`** (used by both §2.5's Operator composite and §2.9's Loop
  claim) — confirmed via `branchingnode.py:130-137`: `_get_inferred_inputs`
  derives a single `StringProperty` from `self.inputs[0].title` (or a default
  name) with **zero type constraint on whatever precedes it** — the node
  doesn't reference or care about the predecessor's class at all; wiring is
  ordinary `ControlFlowEdge`/`DataFlowEdge`. The spec's claim that "nothing
  pins the composite to a specific predecessor node type" is CONFIRMED.
- **§2.11 `EACH_ORACLE` fusion / compiler restriction (§2.11, verification
  task 3)** — CONFIRMED, exact and unconditional for the case tested. The
  quote is real: `compiler.py:511-516`, `case ModifierCombo.EACH_ORACLE |
  ModifierCombo.EACH_ORACLE_OPERATOR:` → `raise CompileError.build("Each x
  Oracle fusion is not supported on sub-constructs", ..., hint="Use a Node
  with map_over + ensemble_n instead", ...)`. This arm is reached
  unconditionally for any Construct-level item classified into either combo —
  there is no sub-condition narrowing it further, so the spec's "match
  compiler.py's own restriction" framing is accurate, not overstated. The
  identical unconditional-`CompileError` shape also applies to
  `PORTAL`/`PORTAL_OPERATOR` at the Construct level (`compiler.py:552-560`,
  not discussed by the spec but consistent with Portal being Node-mesh-only
  by design elsewhere).
- **Minor completeness gap in §2's own primitive inventory**: pyagentspec
  ships **two** parallel-flavored primitives —
  `ParallelFlowNode` *and* `ParallelMapNode` (both confirmed present and both
  classified `"structural"` in `tests/agent_spec_capabilities.py`'s
  `NODE_FAMILIES`, `agent_spec_capabilities.py:52-53`). The spec's §2.2/§2.11
  discussion only ever mentions `ParallelFlowNode` as the rejected
  alternative to `MapNode`-wrapping; it never considers `ParallelMapNode` at
  all. Not fatal (the spec's chosen `MapNode`-wraps-Oracle-Flow design is
  still verified sound), but the "considered and rejected" argument in §2.11
  is incomplete against the actual primitive inventory.

## 3. `EACH_ORACLE` compiler restriction narrowing (task 3)

**CONFIRMED, no narrowing needed.** As above — the sub-construct rejection is
unconditional on `combo in (EACH_ORACLE, EACH_ORACLE_OPERATOR)`, with no
further gating. The spec's characterization is accurate as written.

## 4. Test/doc asset disposition (§5)

**Two of the five hedged/claimed dispositions can be resolved definitively
now, rather than left as "verify during implementation":**

- **`tests/test_agent_spec_refactor_snapshot.py`** — the spec hedges
  ("Needs re-verification, likely needs updating... verify exact assertions
  before assuming full survival"). Read in full: this file is a **byte
  snapshot of `to_agent_spec`'s canonicalized, id-free `Flow.to_dict()`**
  output (`_canonicalize`, lines 102-129) for a `REPRESENTATIVE_CELLS` set
  spanning `scripted/think/agent/act × {bare, oracle-merge_fn,
  oracle-merge_prompt} × {single, dict}` (lines 71-99), diffed against a
  committed golden fixture. It asserts **only observable export behavior**
  (the serialized `Flow`), never `_lower_construct_item`'s internal branch
  structure. This means: (a) it is definitively NOT the kind of
  implementation-detail snapshot that would spuriously break under a
  compositional dispatch refactor, and (b) it is exactly the enforcement
  mechanism the "zero behavior change" claims in Build Plan Steps 0-1 need —
  it should be named explicitly as a **required gate** for those steps, not
  left in the "likely needs updating" bucket. One caveat: it covers only
  `BARE` and `ORACLE`, not `EACH`/`LOOP`/`OPERATOR` — so it is necessary but
  not sufficient proof of zero-behavior-change across all 5 currently-shipped
  combos.
- **`tests/test_agent_spec_import.py`** and **`_group_flow_items`** — see §7
  below (the loader.py finding); the spec's claim here needs revision, not
  just re-verification.
- **`_MARK_REMOTE_AGENT`** (§4 marker section) — the spec hedges "possibly
  dead... may be the intended consumer, verify during implementation."
  CONFIRMED DEAD, not merely possibly: `_MARK_REMOTE_AGENT =
  "neograph/remote_agent"` (`_agent_spec.py:74`) has **zero** other references
  anywhere in `_agent_spec.py` or `loader.py`. `_REMOTE_AGENT_ENDPOINT_ATTRS`
  (`loader.py:277`) is an unrelated dict (endpoint-attribute-name lookup for
  reconstructing remote-agent Node kinds) that merely shares a naming pattern
  — grepping both symbols independently shows they never touch each other.
  This should be filed as a definite small cleanup ticket now, not left as an
  open question for the rewrite's implementer to re-discover.

## 5. Build plan zero-behavior-change claim (§6)

**PARTIALLY REFUTED for Step 0.** Per §1 above, `compiler.py`'s dispatch is
two match statements with divergent per-combo bodies (Node vs. Construct), not
one. A literal reading of §1.5's sketch — a single `match decomp.primary`
block driving both `_add_node_to_graph` and `_add_subgraph` — cannot be
"zero behavior change" as sketched, because it would need to either (a)
duplicate the sketch once per match statement (which is fine and probably
what was intended, but isn't what's written), or (b) thread an extra
Node-vs-Construct axis through `COMBO_DECOMPOSITION` itself (a bigger change
than advertised). This is fixable — the underlying claim that the refactor
*can* be zero-behavior-change is plausible — but the spec as written
understates the shape of the change needed to get there, and Step 0's
"refactor `compiler.py`'s existing 5 match arms" instruction is imprecise
about which/how-many match statements are in scope.

## 6. Marker/round-trip convention (§4)

**CONFIRMED sound for the 13-marker inventory** (spot-checked several: all
present and named as described), **except `_MARK_REMOTE_AGENT`** (§4 above,
dead code, not merely uncertain).

## 7. NEW-GAP-FOUND — `loader.py`'s `_group_flow_items` does not use
`ModifierCombo`/`classify_modifiers` at all, so "the same compositional
treatment... consult `COMBO_DECOMPOSITION` symmetrically" (§5, §6 Step 6) is
underspecified, not merely deferred

This is the most significant independent finding of this review, and it sits
squarely in the same failure category the epic's own retrospective (§2.4) and
the Option F review (§8) both diagnosed: a plausible-sounding analogy asserted
without checking the actual mechanism on the other side.

`grep -n "classify_modifiers\|ModifierCombo\." src/neograph/loader.py` returns
**zero hits**. `_group_flow_items` (`loader.py:559-638`) reconstructs pipeline
shape entirely differently from `_lower_construct_item`: it walks
`flow.nodes` in order and pattern-matches on **string values read off each
primitive's `metadata` dict** (`metadata.get(_MARK_MODIFIER)` ==
`"oracle"`/`"each"`/`"loop"`/`"operator"`), then does a lookahead check
against `flow.control_flow_connections` to confirm the marker's claim is
structurally real (e.g. the Loop back-edge actually exists) before grouping.
There is no `ModifierCombo` enum value anywhere in this function — it
**recovers** a combo-shaped grouping from markers plus edge-shape evidence; it
never **dispatches from** a known `ModifierCombo`, which is the opposite
direction from what `COMBO_DECOMPOSITION` (a `ModifierCombo -> shape` map) is
built to answer.

This means the spec's own build-plan instruction — "Rebuild `_group_flow_items`
to consult `COMBO_DECOMPOSITION` the same way" (§6 Step 6) — does not have an
obvious concrete meaning yet: there is no combo value in hand at the point
`_group_flow_items` runs, so "consult the table" cannot mean the same lookup
`_lower_construct_item`/`compiler.py` perform. A real design for the import
side needs its own pass (something like: derive a `PrimaryShape` from the
*pattern* recognized — which the function already does implicitly through its
branching — and centralize *that* mapping, rather than assume it is
symmetrical to the export-side table lookup). The spec's §5 table entry for
`test_agent_spec_import.py` and §6 Step 6 should be revised to say this
explicitly rather than imply the loader-side fix is a mechanical mirror of the
compiler/`_agent_spec.py` side.

## 8. The meta-question — is there a third instance of the anchoring-bias
pattern?

**Weak/partial YES, one hop removed, same shape as §7.** I grepped every
consumer of `classify_modifiers`/`ModifierCombo`/`modifier_set`/
`has_modifier`/`get_modifier` in `_agent_spec.py`, `loader.py`, and
`compiler.py` directly (not relying on the spec's own citations):

- `_agent_spec.py`: exactly 5 call sites, all inside `_lower_construct_item`
  (`:844-868`) plus 3 duck-typed `.modifier_set.portal` reads inside
  `_lower_portal_mesh_to_swarm`/`to_agent_spec`'s Portal-mesh interception
  (`:900, :929, :965-966`) — **all accounted for by the spec's own §2.6/§1.6
  discussion**. No orphan site found here.
- `loader.py`: **zero** hits — see §7. Not a "missed call site of
  `classify_modifiers`" in the narrow sense (there's nothing to miss, since
  the file doesn't use the API at all), but it IS a missed **mechanism
  mismatch**: the spec's enumeration of "what needs the compositional
  treatment" (§5's test-disposition table, §6's build plan) treats the import
  side as symmetrical to the export side without verifying that the import
  side's actual mechanism (marker/structure pattern-matching) can even
  consume `COMBO_DECOMPOSITION` the way described. This is the same *shape*
  of gap as the retrospective's `merge_prompt` site and the Option F review's
  `_emit_input_edges` site — an under-swept **consumer** of the shared
  artifact, found by asking "does this really work the way the spec assumes"
  rather than trusting the spec's own call-site list — but it is weaker than
  those two precedents: it does not (yet) constitute a silently-shipped bug,
  because no code has been written yet. It is a **design-completeness gap in
  the spec**, not a code defect, and it will become a real instance of the
  pattern if the rewrite proceeds on the "consult `COMBO_DECOMPOSITION`
  symmetrically" instruction as literally written without a separate design
  pass for the import side.
- `compiler.py`: all `classify_modifiers` call sites (`:181, :192, :254,
  :506, :584`) are within the two match-statement-adjacent dispatch functions
  already discussed in §1 — no orphan site found.

**Verdict: no clean third repeat of "a construction site was silently
unguarded and would have shipped a safety-relevant bug," but yes to a
narrower, structurally-identical process gap: an unverified symmetry
assumption about a consumer the spec's own call-site sweep didn't actually
open and read.**

---

## Overall verdict: **NEEDS-REVISION**

The core design — a single `COMBO_DECOMPOSITION` table in `modifiers.py`,
consulted read-only by both `compiler.py` and `_agent_spec.py`, with Operator
as an orthogonal wrapper — is directionally sound, and the per-combo
feasibility claims for `PORTAL_OPERATOR` (§2.7, the highest-risk item) and the
`EACH_ORACLE` sub-construct restriction (§2.11) are independently confirmed,
including by a live repro against the real `pyagentspec` package. The
`s7zt3.1` preservation claim (§3) is correctly described and verified landed
(`git show 691895f`, `bd show neograph-s7zt3.1` CLOSED).

**What must change before implementation starts, in priority order:**

1. **Fix the build-plan/§1.5 sketch to account for `compiler.py`'s real
   two-match-statement structure** (Node-level `_add_node_to_graph` vs.
   Construct-level `_add_subgraph`, each with 6 arms, with genuinely
   different bodies for the same combo at the two levels). Step 0 cannot be
   "zero behavior change" as currently sketched until this is resolved —
   either two parallel table-driven refactors are specified, or an explicit
   Node/Construct axis is added to the design.
2. **Give `loader.py`'s import-side rebuild its own concrete design**, not a
   "consult `COMBO_DECOMPOSITION` symmetrically" one-liner (§7 above).
   `_group_flow_items` recovers structure from markers and edge shape; it has
   no combo value to look up. Name what the shared artifact for the import
   side actually is before Build Plan Step 6 is treated as scoped.
3. **File a small cleanup ticket for `_MARK_REMOTE_AGENT`** (confirmed dead,
   not merely possibly dead) rather than deferring the question to the
   rewrite's implementer.
4. Cosmetic/precision fixes: correct §0's "5 match arms" to reflect the real
   6-arm-per-statement, two-statement structure; note `ParallelMapNode`
   alongside `ParallelFlowNode` in §2's primitive inventory for completeness.

Everything else — the table's cell values themselves (§1.1), the marker
convention (§4, apart from the one dead marker), the placeholder-translation
preservation (§3), and the overall "12/12 combos are representable" claim
(§2's summary) — is confirmed sound and does not need to change.
