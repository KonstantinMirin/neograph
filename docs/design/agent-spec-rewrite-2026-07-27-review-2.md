# Review round 2: `agent-spec-rewrite-2026-07-27.md` (post-revision)

Reviewer: independent skeptical pass on the CORRECTED spec (the one already
revised in place after round 1's `agent-spec-rewrite-2026-07-27-review.md`).
Scope, per the task brief: verify the specific fixes made after round 1 and
resolve the loader.py feasibility question that a prior sub-agent raised but
never wrote up. Every claim below was checked against real source on
`develop` (not the spec's own prose) via direct reads and two live repros
against the actual `Construct`/`compile()`/`to_agent_spec()` code paths.

---

## 1. `compiler.py`'s real two-match-statement structure vs. spec §1.5

**CONFIRMED — the round-1 fix landed correctly.**

Read `_add_subgraph` (`compiler.py:444-568`) and `_add_node_to_graph`
(`compiler.py:571-681`) in full. Both are real, independent `match combo:`
statements, each with exactly 6 case arms
(`EACH_ORACLE|EACH_ORACLE_OPERATOR`, `ORACLE|ORACLE_OPERATOR`,
`EACH|EACH_OPERATOR`, `LOOP|LOOP_OPERATOR`, `BARE|OPERATOR`,
`PORTAL|PORTAL_OPERATOR`), followed by an unconditional
`if operator: last_name = _add_operator_check(...)` postlude in both. This
now matches §1.5's two separate sketches exactly (one block per function,
correctly separated in the current spec text — round 1's complaint that the
spec showed only one `match` block no longer applies). The
`EACH_ORACLE`/`EACH_ORACLE_OPERATOR` divergence (`_add_each_oracle_fused` at
Node level vs. unconditional `CompileError` at Construct level,
`compiler.py:510-517`) and the `PORTAL` divergence (unreachable-defense-in-depth
at Node level, `compiler.py:665-673`, vs. unconditional `CompileError` at
Construct level, `compiler.py:552-560`) are both exactly as the spec
describes. §1.1's `COMBO_DECOMPOSITION` table's cell values, applied per
dispatch surface with `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` checked first at the
Construct level, would compile cleanly against the real code as sketched —
verified by direct comparison, not merely re-asserted.

## 2. Construct-item-modifier-drop claim (§1.6) — reproduced independently

**CONFIRMED, by a fresh repro (not reused from any prior pass).**

Built `enrich = Construct("enrich", input=Claims, output=ScoredClaims,
nodes=[...])`, then `enrich_looped = enrich | Loop(when=lambda d: False,
max_iterations=3)`, placed as an item inside a parent `Construct`, and called
`to_agent_spec(parent)`. Result: the exported `FlowNode` for `enrich` has a
single `"next"` branch and no back-edge; grepping the full serialized
`Flow.to_dict()` JSON for `"neograph/loop_spec"` and the substring `"loop"`
both return `False` — the `Loop` modifier is silently absent from the
export, with **no error, no marker, no diagnostic**. This independently
reproduces the exact gap `_agent_spec.py:832-835`'s `isinstance(item,
Construct)` branch describes (`sub_flow = to_agent_spec(item); flow_node =
FlowNode(...); return ...` — `classify_modifiers(item)` is never called).
The spec's §1.6 fix (checking `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` and raising
`ConfigurationError` before wrapping, mirroring `_add_subgraph`'s own check)
is a correct and sufficient fix for this specific gap, and Build Plan Step
1's characterization of this as "a real (if narrow) behavior CHANGE, not
zero-behavior-change" needing its own TDD is accurate.

## 3. `loader.py`'s `_group_flow_items` — the un-synthesized question

**PARTIALLY REFUTED. This is the most important finding of this review.**

Read `_group_flow_items` in full (`loader.py:559-638`). Confirmed
mechanically: it is a **single forward pass over `flow.nodes`** with a
mutable index `i`, where each iteration inspects `node.metadata` for
`_MARK_MODIFIER`, does a **bounded, hand-coded lookahead** (0, 1, or 2 nodes
ahead depending on which marker string was seen), and **immediately emits**
one `(kind, payload)` tuple before advancing `i` by exactly the consumed
span (1 for `each`, N for an oracle group via `_MARK_GROUP_ID`, 2 for
`loop` body+check, 3 for `operator` primary+check+pause). There is no
intermediate representation where "the full set of co-occurring modifier
names for this group" exists as a `frozenset[str]` before a decision is
made — recognition and emission are the same step, one marker pattern at a
time. Sub-agent `verify-agentspec-loader`'s unresolved question is answered:
**yes**, exactly as suspected.

The spec's §6 step 1 claims: *"Recognition (unchanged from today, this logic
is genuinely different work and doesn't need to change in kind)... produces,
per recognized group, the SET of modifier names actually present-and-verified
— a `frozenset[str]` like `{"each", "oracle"}`."*

This claim is **not accurate for the 6 newly-real composed combos**
(`EACH_OPERATOR`, `ORACLE_OPERATOR`, `LOOP_OPERATOR`, `EACH_ORACLE`,
`EACH_ORACLE_OPERATOR`, `PORTAL_OPERATOR`). Concretely, today's walk has
**no path that folds a trailing Operator check+pause triple onto an
Each/Oracle/Loop/Portal group** — it only ever composes a "body" node with
an immediately-following "check" node for the `loop` (2-node) and `operator`
(3-node) cases, and the `each`/`oracle` branches emit their tuple and advance
`i` **without ever peeking further for a subsequent operator group**. To
recognize `EACH_OPERATOR` on import, the walk must, after tentatively
grouping an `each` `MapNode`, look ahead for a following `BranchingNode` +
`InputMessageNode` pause pair, verify the connecting edge, and — only if
found — fold it into the SAME group (consuming a variable-length span:
1+3 nodes instead of 1) before emitting a combined `frozenset[str]` like
`{"each", "operator"}`. The same restructuring is needed for
`ORACLE_OPERATOR` (N-variant-plus-merge group, variable length, then +3 for
the pause), `LOOP_OPERATOR` (2 + 3), `PORTAL_OPERATOR` (whatever the mesh's
node-count is, +3), and `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` (recognizing a
nested Oracle-fan-out-plus-merge *inside* the `MapNode`'s subflow, which today's
walk never inspects — it treats `each` as an opaque single-node match, never
descending into `MapNode.subflow`).

This is a real, structural change to the **recognition** step's control
flow — new variable-length lookahead patterns that compose two currently-
independent branches of the walk — not merely "a small new loader.py helper"
sitting alongside an unchanged walk. The spec's steps 2-4 (classify via
`_COMBO_MAP`, cross-validate via `COMBO_DECOMPOSITION`, Construct-level
restriction) are fine as a small new function once a correctly-composed
`frozenset[str]` is in hand — but step 1's claim that recognition itself
"doesn't need to change in kind" is false for exactly the combos this rewrite
exists to add. Build Plan Step 6 ("Add the `_classify_recognized_group`
helper... land reconstruct-side support in lockstep with each export-side
combo from Steps 3-5") understates the work: for each of the 6 composed
combos, Step 6 is not "add classification," it is "extend `_group_flow_items`'s
lookahead to recognize a new composed shape, THEN classify it." The spec
should say this explicitly and size Step 6 accordingly (one lookahead-pattern
change per composed combo, mirroring the one-PR-per-combo cadence Step 4
already uses for the export side), rather than implying the recognition
walk is a fixed, already-correct piece of infrastructure that a new
classification layer merely sits on top of.

## 4. `SUB_CONSTRUCT_UNSUPPORTED_COMBOS`'s Portal exclusion — REFUTED as justified, real gap found

**REFUTED. The stated reasoning is factually wrong, and a real (pre-existing,
independent of this rewrite) bug was found in the process.**

The spec (§1.5) says: *"Portal combos are excluded from this set deliberately:
a Portal mesh member being a Construct is already impossible by construction
elsewhere (mesh membership requires a bare Node)."*

This is **false**. `_validation_portal.py`'s own docstring says otherwise —
literally: *"do0d9 (§4 Q2): a Construct member is ADMITTED as a first-class
mesh member... The former blanket `isinstance(member, Node)` rejection is
relaxed"* (`_validation_portal.py:114-119`). This is not a hypothetical or
stale comment — it is backed by a real passing fixture,
`tests/check_fixtures/should_pass/portal_construct_member.py`, and by
`tests/test_portal_cross_subconstruct.py::test_construct_as_portal_member_assembles_and_compiles_across_surfaces`.
I ran the fixture directly: **it assembles and compiles successfully today**,
with a `Construct` (`resolver_sub`) as a non-entry Portal mesh member,
routed through via `_add_portal_mesh` (not `_add_subgraph`).

I then built a second repro — a mesh where the **Construct is the entry**
(the first contiguous mesh member) rather than a later member. This **does
NOT compile**: `compiler.py`'s top-level walk gates mesh-detection on
`isinstance(item, Node)` (`compiler.py:254`), so when the first item in a
contiguous Portal run is a Construct, the walk never invokes
`_contiguous_portal_mesh`/`_add_portal_mesh` for it at all — it falls
through to the `elif isinstance(item, Construct): _add_subgraph(...)`
branch, which then hits the unconditional `CompileError` at
`compiler.py:552-560` — a message ("mesh members must be sibling Nodes
(D-MESH-LEVEL)") that is **stale**, since do0d9 already relaxed exactly this
restriction for non-entry members. This is a genuine, previously-undetected
asymmetry: a `Construct` CAN be a Portal mesh member (non-entry, proven
passing), but CANNOT be the mesh entry (compile-time `CompileError`, with
wording that contradicts the validator's own admitted capability) — no test
in `test_portal_validation.py` or `test_portal_cross_subconstruct.py` covers
the Construct-as-entry case, consistent with this being an untested gap
rather than a deliberate restriction.

Separately, and independently: `to_agent_spec`'s own Portal-mesh detection
(`_agent_spec.py:961-967`) is **also** `isinstance(item, Node)`-gated, and is
in fact *more* restrictive than `compiler.py` — it requires the ENTIRE
top-level construct to be mesh members or none (`if len(mesh_members) !=
len(all_items): raise ConfigurationError("mixes a Portal peer mesh with
non-mesh nodes")`), with no notion of a contiguous sub-run at all. So today,
a construct with a `Construct`-as-non-entry Portal member (the do0d9-admitted,
compiler.py-passing case) would hit `to_agent_spec`'s "mixes a mesh with
non-mesh nodes" `ConfigurationError` for the WRONG reason — it isn't actually
mixing mesh/non-mesh, it's a legitimate mesh with a Construct member the
detection code doesn't recognize.

**Net assessment**: this is fail-loud, not silent — so it is not literally
the same *symptom* class as the `EACH_ORACLE` Node-item-modifier-drop gap
(§1.6). But the spec's specific justification for excluding Portal from
`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` is unsound, and the underlying compiler
behavior it leans on is itself an inconsistent, untested, and now-confirmed
bug (do0d9 admits Construct mesh members generally; both `compiler.py`'s
mesh-entry detection and `_agent_spec.py`'s all-or-nothing mesh detection
disagree with that admission in different ways). This is a real
**NEW-GAP-FOUND**, orthogonal to but directly touching this rewrite's scope,
because `_agent_spec.py`'s Construct-item branch (§1.6) is exactly the code
this rewrite is rebuilding, and it needs an explicit, ACCURATE answer for
"what happens when a Construct item classifies to `PORTAL`/`PORTAL_OPERATOR`"
— not silence backed by a false "impossible by construction" premise.

**Recommendation**: (a) fix the spec's §1.5 comment to stop claiming Portal-
on-Construct is impossible by construction; (b) explicitly decide whether the
rewrite's Construct-item branch rejects `PORTAL`/`PORTAL_OPERATOR` with an
accurate message (simplest, matches `compiler.py`'s CURRENT — if
inconsistent — behavior) or whether a proper Construct-mesh-member Agent
Spec design is now in scope (bigger, likely NOT in scope for this rewrite);
(c) file a separate, small bug ticket for the `compiler.py`
Construct-as-mesh-entry inconsistency (do0d9 admits it, `compiler.py:254`'s
`isinstance(item, Node)` gate doesn't honor it) plus `_agent_spec.py`'s
all-or-nothing mesh detection being unaware of do0d9 entirely — both
pre-existing, independent of whether this rewrite proceeds.

## 5. Build plan internal consistency (§5, §7) given fixes 1-4

- **Step 0** ("TWO refactors not one") is now accurate against the real code
  (§1 above) — no further correction needed.
- **Step 1** (Construct-item-modifier-fix, TDD'd) is correctly scoped per §2
  above.
- **Step 6** (loader.py rebuild) needs revision per §3 above — it currently
  under-sizes the recognition-side work per composed combo. It should be
  split the same way Step 4 splits the export side: one recognition-pattern
  change + classification/cross-validation per composed combo, not one
  generic "add the helper" step at the end.
- Step 6 also inherits the §4 gap silently: none of the 12 combos' loader-side
  design discusses what happens when import recognizes a
  `Swarm`/Construct-mesh-member combination — `_reconstruct_swarm_mesh`
  (`loader.py:689-734`) always reconstructs Swarm agents as `Node`s (never
  a `Construct`), so there is no round-trip path for a Construct-as-mesh-member
  export even if the export side eventually supported it. Not a blocker (the
  export side doesn't support it either, per §4), but worth naming so it
  isn't rediscovered mid-implementation.

---

## Overall verdict: **NEEDS-REVISION**

Two of round 1's three fixes hold up cleanly under independent
re-verification (the two-match-statement §1.5 sketch, and the
Construct-item-modifier-drop fix in §1.6 — the latter now confirmed by a
fresh repro, not just re-read). The third fix — §6's loader.py design — is
directionally right (recognize → classify via `_COMBO_MAP` → cross-validate
via `COMBO_DECOMPOSITION` is the correct shape) but its load-bearing claim
that recognition is "unchanged... doesn't need to change in kind" is false
for the 6 newly-real composed combos, and this review additionally found a
new, independently-confirmed gap: the spec's justification for excluding
Portal from `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` rests on a claim
(“impossible by construction”) that is directly contradicted by
`_validation_portal.py`'s own do0d9 admission and a passing fixture, and
chasing that thread surfaced a real, previously untested compiler
inconsistency (Construct-as-Portal-mesh-entry fails, Construct-as-non-entry-
member succeeds).

**Action list, priority order:**

1. **Correct §1.5's Portal-exclusion justification** in
   `SUB_CONSTRUCT_UNSUPPORTED_COMBOS`'s docstring/comment — remove the false
   "impossible by construction" claim — and make an explicit decision for
   the Construct-item branch's `PORTAL`/`PORTAL_OPERATOR` handling (reject
   accurately, or scope in a real design). This is a correctness-of-spec
   issue, not implementation detail — the current text will mislead whoever
   implements it into believing there's nothing to do here.
2. **File the compiler.py Construct-as-Portal-mesh-entry bug** (do0d9 admits
   it; `compiler.py:254`'s `isinstance(item, Node)` gate and
   `_add_subgraph`'s stale rejection message disagree) and the
   `_agent_spec.py` all-or-nothing mesh-detection gap, as their own small
   ticket(s) — independent of whether this rewrite proceeds, but discovered
   by it.
3. **Revise Build Plan Step 6 / spec §6** to state plainly that
   `_group_flow_items`'s recognition walk needs new variable-length lookahead
   composition (body-group + trailing operator triple; nested Oracle inside
   an Each `MapNode`'s subflow) for each of the 6 composed combos — size it
   as one recognition-pattern change per combo, landed in lockstep with the
   matching export-side combo from Step 4, not as a single generic
   "add a helper" step at the end.
4. Everything else — the shared-table design (§1.1-§1.4), the per-combo
   feasibility verdicts for the 5 already-shipped combos and
   `PORTAL_OPERATOR`/`EACH_ORACLE` (§2), the placeholder-translation
   preservation (§3), and the marker convention (§4) — is confirmed sound
   and does not need further revision.

**Is the spec ready for implementation to begin?** Not as a single
whole-spec go. Steps 0-2 of the Build Plan (shared tables, both `compiler.py`
match statements, the 5 already-shipped combos + Construct-item-modifier fix
in `_agent_spec.py`) can start now — nothing in this review's findings
touches them. Steps 3-5 (Portal_Operator, the other composed combos) should
wait for action item 1 above (an accurate Portal/Construct-item decision)
before the Construct-item branch is finalized. Step 6 (loader.py) should not
start implementation until its design is corrected per action item 3 — as
written, an implementer would hit the same "wait, `_group_flow_items` doesn't
work the way the spec says" wall the stalled prior review attempt hit,
without a written resolution to point to.
