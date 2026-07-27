# Review: `modifier-combo-single-source-of-truth-2026-07-27.md`

Reviewer stance: skeptical, first-draft treatment. Every citation below was
re-read from the actual source at the stated (or corrected) line range — none
taken on the document's word.

---

## 1. The 8-module consumer inventory (§1)

| File | Verdict | Evidence |
|---|---|---|
| `compiler.py` | CONFIRMED | Two `match combo:` statements, `_add_subgraph` (`combo, mods = classify_modifiers(sub)` at line 506, `match combo:` at 509, arms for `EACH_ORACLE`/`ORACLE`/`EACH`/`LOOP`/`BARE`/`OPERATOR`/`PORTAL`) and `_add_node_to_graph` (`match combo:` at 595, same arm shape, `PORTAL`/`PORTAL_OPERATOR` arm at 665 raises "unreachable" defensively). Both are genuine "combo → build this LangGraph shape" dispatch — the real thing the doc means by decomposition. |
| `_agent_spec.py` (`_lower_construct_item`) | CONFIRMED | Lines 844-868: five `if combo == ModifierCombo.X:` branches (`ORACLE`, `EACH`, `LOOP`, `OPERATOR`, `BARE`) — no `match`, no `assert_never`, and no arm for `EACH_ORACLE`, `ORACLE_OPERATOR`, `EACH_OPERATOR`, `LOOP_OPERATOR`, `PORTAL`, `PORTAL_OPERATOR`, `EACH_ORACLE_OPERATOR`. "Flat 5-branch, incomplete" is accurate, not overstated. |
| `state.py:164-202,536-635` | CONFIRMED, with a correction to the doc's own hedge | Three separate `match combo:` blocks genuinely exist: line 165 (sub-construct field shaping), line 537 (dict-form node-output dispatch inside `_add_output_field`), line 596 (`_add_single_output_field`). All three groups combos into the identical BARE/OPERATOR, EACH/EACH_OPERATOR, ORACLE/ORACLE_OPERATOR, LOOP/LOOP_OPERATOR, EACH_ORACLE/EACH_ORACLE_OPERATOR, PORTAL/PORTAL_OPERATOR buckets — i.e. the grouping (which combos are "operator-orthogonal" siblings) is byte-identical across all three; only the per-bucket *body* (what field/reducer to build) differs. See §2 below — the doc's own §4 hedge speculates one of the three "may" be schema-fingerprinting; that's factually wrong (see §2), which matters because it means all three ARE the same concern (state-shape derivation) applied to three producer categories, not three unrelated concerns that happen to share an enum. |
| `_state_write.py:72-97` | CONFIRMED | `combo, mods = classify_modifiers(node)` at 72, `match combo:` at 79, grouping `EACH`/`EACH_OPERATOR` vs `EACH_ORACLE`/`EACH_ORACLE_OPERATOR` vs `{BARE,OPERATOR,ORACLE,ORACLE_OPERATOR,LOOP,LOOP_OPERATOR,PORTAL,PORTAL_OPERATOR}` — exactly "primary-with-operator-orthogonal" grouping, same partition as `state.py`. |
| `_subconstruct.py:89-91` | CONFIRMED, but the "(Operator ignored)" parenthetical needs a clarifying edit | `sub_combo, _ = classify_modifiers(sub)`, `has_loop = sub_combo in (LOOP, LOOP_OPERATOR)`, `has_each = sub_combo in (EACH, EACH_OPERATOR)`. This is a genuine, narrower re-derivation of the same "which combos are Loop-shaped / Each-shaped" fact every other site encodes — Operator's presence/absence correctly does not change `has_loop`/`has_each` because Operator is an orthogonal wrapper, not a shape. The doc's parenthetical is technically accurate (Operator is not consulted) but is written in a way a reader could misread as "this is a bug" rather than "this is the correct orthogonality" — it is not a gap, it's the file correctly ignoring an irrelevant axis. Recommend the doc state this explicitly rather than leave it ambiguous. |
| `_input_shape.py:32-33` | CONFIRMED (substance), citation syntax is wrong | Real code: `combo, _ = classify_modifiers(node)` then `if combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR):`. There is no `.primary` attribute anywhere in `modifiers.py`, `state.py`, `_input_shape.py`, or `runner.py` — `ModifierCombo` is a flat 12-value `Enum` with no `.primary` field; `.primary`/`PrimaryShape` only exist in the **not-yet-written** design spec (`agent-spec-rewrite-2026-07-27.md` §1.1), as the planned future API. The doc's own citation notation (`.primary == LOOP`) borrows syntax from the *proposed fix* to describe *today's* code, which doesn't have it. Substance of the claim (this site re-derives "is this Loop-shaped") is correct; the literal quoted syntax is not real and should be corrected to `combo in (LOOP, LOOP_OPERATOR)`. |
| `runner.py:116-154` | CONFIRMED (substance), same citation-syntax issue, and undercounted | Real code: `classify_modifiers(item)[0] in (ModifierCombo.PORTAL, ModifierCombo.PORTAL_OPERATOR)` — appears FOUR times in this file (lines 116, 123, 143, 154: `_mesh_hop_cost`'s walk twice, `_portal_mesh_member_ids`'s walk twice), not once. The cited range 116-154 catches only the first function; `_portal_mesh_member_ids` (131-156) does the identical check again immediately after. Same `.primary ==` notation issue as `_input_shape.py`. |

**Bottom line on §1**: every one of the five newly-claimed files is a real consumer of the same kind of decision the doc says it is — none of the citations turned out to be a false positive (e.g., an unrelated `Node.modifier_set.X is not None` check masquerading as decomposition). The two defects found are (a) a fabricated `.primary ==` syntax that doesn't exist in current code (borrowed from the proposed-fix spec) and (b) an undercount in `runner.py` (4 occurrences, not folded into "one" citation) — neither changes the underlying finding, both should be fixed before this doc is treated as a citation-grade record.

---

## 2. Resolving the doc's own §4 hedge about `state.py`'s three blocks

The doc's §4 speculates the three `state.py` match blocks "may encode different
information at each site (state-shape derivation vs. reducer wiring vs. schema
fingerprinting are plausibly different concerns)".

Checked directly: `state.py` has exactly one fingerprinting pair,
`compute_node_fingerprints` (line 410) and `compute_schema_fingerprint` (line
465). Neither calls `classify_modifiers` or references `ModifierCombo` at all —
they fingerprint off the **already-built** Pydantic field types via
`_type_signature`, downstream of the three match blocks, not a fourth site that
switches on combo. So "schema fingerprinting" is not a live candidate for one of
the three match blocks — that specific guess in the doc's hedge is **REFUTED**.

What the three blocks (165, 537, 596) actually differ on is not "concern" but
"producer category": sub-construct output shaping, dict-form node-output
shaping, and single-type node-output shaping. All three are the same concern —
what Pydantic field(s)/reducer(s) to allocate for a given combo — applied to
three different IR shapes. The case-grouping (which combos are bucketed
together) is identical byte-for-byte across all three; only the field-building
body differs per producer category. This actually **strengthens** rather than
weakens the doc's "8 consumers, same fix" framing for the classification
question (a single `COMBO_DECOMPOSITION`/`PrimaryShape` lookup could
legitimately replace all three `match combo:` headers), while still leaving the
doc's underlying caution correct: the per-arm **bodies** cannot be mechanically
unified (e.g. `_add_single_output_field`'s `EACH_ORACLE` arm does real collector
+ dict-field construction; the sub-construct block's `EACH_ORACLE` arm at line
198 is an explicitly-commented "defensive fallback", already unreachable
today). So: PARTIALLY-CONFIRMED — the doc's caution about "read the real code,
don't assume the pattern transfers" is validated, but its specific example
(schema fingerprinting) is wrong and should be swapped for the real
distinguishing factor (producer category, and real-vs-defensive-fallback arms).

---

## 3. Portal root-cause diagnosis (§2)

CONFIRMED. `tests/test_guards_llm_runtime.py:1018`:
`IR_FIELDS = frozenset({"fan_out_param", "oracle_gen_type", "handoff_param", "handoff_channel"})`,
enforced as single-writer via `_ir_normalize.py` at lines 1061/1065/1087/1096/1101
of the guard test. The fields' single-writer discipline is real and correctly
described (not overstated).

Searched for any pre-existing combo→shape table across the whole tree:
`grep -rln "ModifierCombo\.\|classify_modifiers(\|get_modifier(\|modifier_set\.\|has_modifier(" src/neograph/*.py` —
no `COMBO_DECOMPOSITION`/`PrimaryShape` exists anywhere in `src/neograph/`
(only in the not-yet-implemented design doc and its review). Two **partial**,
narrower shared helpers do already exist and are worth the doc acknowledging
explicitly (it currently implies zero convergence exists anywhere):
- `_COMBO_MAP` (`modifiers.py`) — already centralizes raw-modifier-set →
  `ModifierCombo` *classification* (not decomposition/dispatch). This is a
  different, already-solved problem the doc doesn't confuse it with, good.
- `_group_portal_members` (`modifiers.py:727`) — already the documented single
  source of truth for "which named mesh a Portal member belongs to," consumed
  by `_validation_portal.py`, `_ir_normalize.py`, and `_wiring.py`'s
  `_contiguous_portal_mesh`. This is real prior art for the exact kind of
  anti-duplication fix this doc proposes, scoped narrowly to Portal-grouping
  rather than general combo-decomposition — worth a one-line mention as
  precedent, not a refutation of anything claimed.

Neither of these is a hit against the doc's central claim (no shared
combo-DECOMPOSITION table exists) — that claim holds.

---

## 4. `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` / `compiler.py:511-516` (§3)

CONFIRMED, independently re-verified (not just trusted from the earlier
retrospective). `compiler.py` lines 509-517:
```
match combo:
    case ModifierCombo.EACH_ORACLE | ModifierCombo.EACH_ORACLE_OPERATOR:
        raise CompileError.build(
            "Each x Oracle fusion is not supported on sub-constructs",
            found="both Oracle and Each modifiers on a sub-construct",
            hint="Use a Node with map_over + ensemble_n instead",
            construct=sub.name,
        )
```
Matches the doc's description exactly: a pre-existing compiler restriction
grounded in `Each`/`Oracle`'s fusion being defined only in terms of a single
`Node`'s `map_over=`/`ensemble_n=` fields, not a `Construct`'s. The doc's
framing (not an Agent-Spec-representability restriction, predates this epic)
is accurate.

---

## 5. The `AGENTS.md` edit

Read in full (`AGENTS.md` line 197, the "Lesson from the Portal rollout"
paragraph). Findings:

- **Accurate**: the six files it names (`compiler.py`, `state.py`,
  `_state_write.py`, `_subconstruct.py`, `_input_shape.py`, `runner.py`) are
  all real, verified consumers (§1 above). It correctly scopes itself to the
  IR/runtime layer (leaves out `_agent_spec.py`/`loader.py`, which are a
  separate export-layer concern already covered by
  `agent-spec-rewrite-2026-07-27.md`) — appropriate, not an omission, since the
  AGENTS.md PORTAL section is specifically about the IR/compiler/runtime
  discipline, not the Agent Spec surface.
- **The `.primary ==` phrase inherits the same fabricated-syntax problem
  as the reviewed doc** ("`compiler.py`, `state.py`, ... each independently
  grew their own `case PORTAL` / `.primary == ...` handling"). As shown in §1,
  no `.primary` attribute exists in current code; the real patterns are
  `match combo:` (compiler.py/state.py/_state_write.py) and `combo in (X,
  X_OPERATOR)` membership checks (runner.py/_input_shape.py) and a
  `handoff_channel` parameter check (_subconstruct.py, which doesn't literally
  branch on Portal at all — Portal routing there is handled via a distinct
  parameter path, not a `case`/`.primary` check). This is a minor but real
  inaccuracy in a file meant to be a durable, precisely-quotable project
  invariant — a future engineer grepping for `.primary ==` to find "the
  Portal-handling arms" will find nothing.
- **Appropriately durable/general otherwise**: it states the invariant ("any
  new Modifier/ModifierCombo value... must update ONE shared table") without
  over-anchoring to this epic's specific beads IDs in the body text (only the
  linked doc and one retrospective filename carry dates/IDs, which is fine —
  they're citations, not the rule itself). Reads cleanly, doesn't reference an
  ID that will look stale in a year for the *rule* itself.
- **Recommendation**: replace `.primary == ...` with accurate shorthand, e.g.
  "each independently grew its own combo-membership check (`match combo:` /
  `combo in (X, X_OPERATOR)`) for `PORTAL`/`PORTAL_OPERATOR`" — same fix needed
  in the reviewed doc's §1 table.

---

## 6. Meta-question: is the "8 modules" sweep exhaustive? NEW-GAP-FOUND

Ran the same sweep the doc claims to have run, over the whole tree:
```
grep -rln "ModifierCombo\.\|classify_modifiers(\|get_modifier(\|modifier_set\.\|has_modifier(" src/neograph/*.py
```
Result: 17 files, not 8 — `__main__.py`, `_agent_spec.py`, `_construct_graph.py`,
`_construct_validation.py`, `_fan_agent.py`, `_input_shape.py`,
`_ir_normalize.py`, `_param_classify.py`, `_state_write.py`, `_subconstruct.py`,
`_validation_modifiers.py`, `_validation_portal.py`, `_wiring.py`, `compiler.py`,
`lint.py`, `modifiers.py` (the source), `runner.py`, `state.py`.

Most of the extra hits are legitimately a **different** concern than combo
decomposition — single-modifier presence checks for validation/DI/topology
purposes (`_construct_validation.py`'s `.portal`/`.oracle`/`.loop is not None`
checks feeding assembly-time error messages; `_param_classify.py`'s `.each`/
`.portal` checks for DI param classification; `_construct_graph.py`'s `.loop is
not None` for dependency-graph self-reference resolution; `_fan_agent.py`'s
`classify_modifiers` use feeding a fan-over-agent support predicate, not a
build-dispatch). These are correctly out of scope — conflating "does this node
have modifier X at all" with "what does this ModifierCombo decompose into for
build purposes" would be the exact category error the task asked me to watch
for, and I did not find evidence the doc made it (it simply didn't look at
these files, rather than looking and mischaracterizing them).

**One real miss**: `_wiring.py`. It contains, at minimum:
```
_wiring.py:718:  if classify_modifiers(item)[0] not in (ModifierCombo.PORTAL, ModifierCombo.PORTAL_OPERATOR):
```
inside `_contiguous_portal_mesh` — the SAME "is this combo Portal-shaped"
membership re-derivation the doc already credits to `runner.py`,
`_subconstruct.py`, and `_input_shape.py`. `_wiring.py` additionally has
`entry.modifier_set.portal`/`member.modifier_set.portal`/`.operator` reads at
lines 713, 725, 853, 865, 912, 997 feeding the same class of mesh-membership
and per-hop-cost decisions `runner.py` makes independently. This is a genuine
**ninth consumer**, doing exactly the pattern the doc is trying to eliminate,
that the "eight modules" inventory did not find. It should be added to §1's
table and to the widened epic's consumer list in §4, and the `AGENTS.md`
lesson-paragraph's six-file list should become seven (or the paragraph should
be phrased to not enumerate a closed set, given this review just found an
omission in a list that was itself supposed to be the corrected, exhaustive
one).

---

## Overall verdict: NEEDS-REVISION (not REJECT — the core finding is sound)

The central architectural claim — Portal's rollout centralized IR-field
single-writer discipline but never centralized combo-dispatch, and the gap is
real, systemic, and not confined to the Agent Spec export layer — holds up
under independent re-verification. All five newly-claimed consumers are real;
none is a false-positive category error. The `SUB_CONSTRUCT_UNSUPPORTED_COMBOS`
recap is accurate. The `AGENTS.md` edit is well-scoped and durable in framing.

What must change before this is a citation-grade record:

1. **Add `_wiring.py` as a ninth consumer** (§1/§4 of the doc, and the
   `AGENTS.md` six-file list) — found by literally re-running the sweep the doc
   claims to have run. This is the one substantive gap: the doc anchored on the
   list handed to it and didn't verify its own sweep was exhaustive over the
   whole tree, which is precisely the failure mode task-instructions warned
   about.
2. **Fix the fabricated `.primary ==` citation syntax** in both the reviewed
   doc's §1 table (`_input_shape.py`, `runner.py` rows) and the `AGENTS.md`
   paragraph. No `.primary` attribute exists in current code; it belongs to the
   not-yet-implemented `PrimaryShape` design. Replace with the real syntax
   (`combo in (X, X_OPERATOR)` / `match combo:`).
3. **Correct the §4 hedge's wrong guess** ("schema fingerprinting" is not one
   of `state.py`'s three match-block concerns — fingerprinting doesn't consult
   `ModifierCombo` at all). Replace with the real distinguishing factor:
   producer category (sub-construct / dict-form node-output / single-type
   node-output), plus the real-vs-defensive-fallback-arm distinction (e.g. the
   sub-construct block's `EACH_ORACLE` arm is a documented unreachable
   fallback, unlike `_add_single_output_field`'s real fusion logic for the same
   combo).
4. **Clarify `_subconstruct.py`'s "(Operator ignored)" parenthetical** so it
   reads as "Operator is correctly orthogonal here, not a gap" rather than
   leaving it ambiguous.
5. Optional but strengthens the doc: mention `_group_portal_members` and
   `_COMBO_MAP` as existing narrower precedent for the shared-table pattern,
   so the doc doesn't read as claiming zero prior convergence exists anywhere
   in the codebase.

None of these require reopening the epic's widened scope or its priority —
they are citation-accuracy and completeness fixes to a fundamentally sound
finding.
