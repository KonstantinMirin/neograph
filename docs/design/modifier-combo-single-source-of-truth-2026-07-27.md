# `ModifierCombo` decomposition: codebase-wide single source of truth (widened scope)

Date: 2026-07-27
Status: architecture finding + widened mandate (durable record — this is a permanent
project invariant, not a scoped-to-one-epic fix; also referenced from `AGENTS.md`'s
PORTAL-exception section)
Parent epic: `neograph-s7zt3`

---

## 1. The finding

While reviewing `docs/design/agent-spec-rewrite-2026-07-27.md` (a spec to unify
`_agent_spec.py`'s modifier-composition dispatch around one shared
`COMBO_DECOMPOSITION` table, consumed by `compiler.py` and `_agent_spec.py`), the
mandatory "does this sweep every real consumer, or does it anchor on the ticket's
own framing" check — the same check the `agent-spec-oracle-inputs-2026-07-25-
architecture-retrospective.md` retrospective demanded after `m57mn`'s design pass
missed the `merge_prompt` construction site — found that the shared-source-of-truth
problem is **not confined to the three modules the spec named**. Grepped, with
file:line citations, not assumed:

| File | What it independently re-derives |
|---|---|
| `compiler.py` (`_add_node_to_graph`, `_add_subgraph`) | the full `(primary, has_operator)` decomposition, in two separate `match combo:` statements |
| `_agent_spec.py` (`_lower_construct_item`) | the same decomposition, flat 5-branch `if combo == ModifierCombo.X:` chain, incomplete |
| `state.py:165,537,596` | the same combo-to-bucket grouping (BARE/OPERATOR, EACH/EACH_OPERATOR, ORACLE/ORACLE_OPERATOR, LOOP/LOOP_OPERATOR, EACH_ORACLE/EACH_ORACLE_OPERATOR, PORTAL/PORTAL_OPERATOR), in three separate `match combo:` blocks — one per producer category (sub-construct output shaping, dict-form node-output shaping, single-type node-output shaping); the grouping is byte-identical across all three, only the per-arm field/reducer-building body differs |
| `_state_write.py:72-97` | `combo, mods = classify_modifiers(node)` then `match combo:` — the same primary-with-operator-orthogonal grouping |
| `_subconstruct.py:89-91` | `sub_combo, _ = classify_modifiers(sub)` then `sub_combo in (LOOP, LOOP_OPERATOR)` / `sub_combo in (EACH, EACH_OPERATOR)` membership checks — Operator is correctly not consulted here (it's an orthogonal wrapper, not a shape, so its presence/absence doesn't change whether something is Loop-shaped or Each-shaped); this is correct orthogonality, not a gap |
| `_input_shape.py:32-33` | `combo, _ = classify_modifiers(node)` then `combo in (ModifierCombo.LOOP, ModifierCombo.LOOP_OPERATOR)` |
| `runner.py:116,123,143,154` | `classify_modifiers(item)[0] in (ModifierCombo.PORTAL, ModifierCombo.PORTAL_OPERATOR)`, appearing FOUR times across two functions (`_mesh_hop_cost`, `_portal_mesh_member_ids`) |
| `_wiring.py:718` (plus `.modifier_set.portal`/`.operator` reads at 713, 725, 853, 865, 912, 997) | `classify_modifiers(item)[0] not in (ModifierCombo.PORTAL, ModifierCombo.PORTAL_OPERATOR)` inside `_contiguous_portal_mesh` — the same Portal-shape re-derivation as `runner.py`/`_subconstruct.py`/`_input_shape.py` |

**Nine modules, not three**, each independently answering "what does this
`ModifierCombo` mean" for their own purposes, with no shared table any of them
consult. (`_wiring.py` was missed by the first pass of this sweep and found only
by re-running the grep during this doc's own adversarial review — see the
companion review doc's §6: exactly the "does the sweep actually cover
everything" failure mode this whole finding is about, caught recursively.
Everything else this same grep surfaces — `_construct_validation.py`,
`_param_classify.py`, `_construct_graph.py`, `_fan_agent.py`, `lint.py`,
`__main__.py` — reads a modifier's mere *presence* for validation/DI/topology
purposes, a genuinely different question from "what does this combo decompose
into for build purposes," and is correctly out of scope.)

Two existing, narrower precedents for this exact pattern are worth naming so
this doc doesn't read as claiming zero convergence exists anywhere: `_COMBO_MAP`
(`modifiers.py`) already centralizes raw-modifier-set → `ModifierCombo`
*classification* (a different, already-solved problem, not confused with
decomposition here), and `_group_portal_members` (`modifiers.py:727`) is already
the single source of truth for "which named mesh a Portal member belongs to,"
consumed by `_validation_portal.py`, `_ir_normalize.py`, and `_wiring.py` — real
prior art for the anti-duplication fix this doc proposes, just scoped narrowly
to Portal-grouping rather than general combo-decomposition.

## 2. Root diagnosis: Portal's own introduction is an instance of the anti-pattern

This is not a new gap the Agent Spec rewrite discovered incidentally — it is
retroactive evidence that **Portal's rollout itself repeated the exact
duplicated-dispatch anti-pattern** this whole epic exists to close.

`AGENTS.md`'s "PORTAL dynamic-handoff exception" documents a deliberate,
single-writer discipline for Portal's two new IR *fields* (`Node.handoff_param`,
`Node.handoff_channel`, written only by `_ir_normalize.py`, pinned by guard G3).
That discipline was followed correctly for the fields. It was **not** followed for
Portal's *dispatch logic* — "which consumers need to know about `PORTAL`/
`PORTAL_OPERATOR`, and what do they each do about it" was never centralized. Each
of the nine modules above grew its own independent Portal-handling check (a
`match combo:` arm or a `combo in (PORTAL, PORTAL_OPERATOR)` membership test), at
the time Portal was added, without a shared table forcing them to converge. The
`_agent_spec.py`/Agent-Spec-export flavor of this (found first, in `s7zt3.2`) is
just the most recently-discovered instance — the pattern is systemic, not
export-specific.

**This means the fix cannot be scoped to the Agent Spec export/import layer.**
"No out of scope for core changes" (maintainer directive, 2026-07-27): the shared
`COMBO_DECOMPOSITION` table (and its structural anti-regrowth guard) must cover
all nine consumers, and the guard must enumerate all nine — not the three the
original rewrite spec named, which would either fail immediately on landing
(`state.py`/`_state_write.py` already contain the pattern the guard is meant to
forbid) or get silently scoped down, leaving the identical duplication standing
in six more places.

## 3. What `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` is (recorded here since it's a
related, frequently-misread piece of the same design)

Not a new restriction, and not about Agent Spec representability. It is neograph's
own **pre-existing compiler restriction** (`compiler.py:511-516`,
`CompileError.build("Each x Oracle fusion is not supported on sub-constructs")`):
`Each`+`Oracle`'s fusion is defined entirely in terms of a single `Node`'s own
`map_over=`/`ensemble_n=` fields (a special M×N `Send`-based topology). A
`Construct` (a multi-node sub-pipeline) has no such fields, so the fusion has no
defined compilation semantics when stacked on one — this is true independent of
Agent Spec, and predates this epic. The Agent Spec exporter must mirror this exact
restriction (raise the equivalent error) rather than either exporting a pipeline
shape neograph's own compiler refuses to run, or silently narrowing what "export"
claims to support below what the compiler actually does elsewhere. It does **not**
mean any class of LangGraph/Agent-Spec pipeline is unrepresentable in neograph —
it means one specific modifier stacking has no defined *compilation* meaning in
neograph's own execution model, for a structural reason (missing fields), not a
missing-feature reason.

## 4. Widened mandate for `neograph-s7zt3`

The epic (title updated 2026-07-27) now covers:

1. `COMBO_DECOMPOSITION`/`PrimaryShape`/`SUB_CONSTRUCT_UNSUPPORTED_COMBOS` in
   `modifiers.py` (as designed in `agent-spec-rewrite-2026-07-27.md` §1) — unchanged.
2. **All nine consumers** read from it: `compiler.py` (both match statements),
   `_agent_spec.py` (rewritten dispatch), `loader.py` (§6's recognize→classify
   design), **and now** `state.py`, `_state_write.py`, `_subconstruct.py`,
   `_input_shape.py`, `runner.py`, `_wiring.py`. Each of the six newly-found
   consumers needs the SAME verification rigor already applied to `compiler.py`:
   read the real code, do not assume the pattern transfers cleanly. `state.py`'s
   three separate match blocks, checked directly, group combos identically
   across all three (the same BARE/OPERATOR, EACH/EACH_OPERATOR, etc. buckets) —
   they differ only in which producer category's field/reducer body each arm
   builds (sub-construct output shaping vs. dict-form node-output shaping vs.
   single-type node-output shaping), and at least one arm (the sub-construct
   block's `EACH_ORACLE` case) is a documented, currently-unreachable defensive
   fallback rather than live logic. So the *classification* question is a single
   clean unification across all three blocks; only the per-arm *bodies* need
   individual care — the same "shared decision, different bodies per site"
   shape the `compiler.py` Node-vs-Construct discovery already proved necessary
   to check for, not assume away.
3. The structural anti-regrowth guard (§1.7 of the rewrite spec) is extended to
   assert all nine modules import the shared table and none contains a second,
   independent `ModifierCombo` enumeration for dispatch purposes.

## 5. Why this record exists outside of beads

Per maintainer instruction: a beads epic's notes are operational tracking, not a
durable architectural record — they get archived, condensed, and eventually
stop being anyone's first read. This finding is a **permanent lesson about how
this codebase must add modifiers going forward**, not a one-off implementation
detail of one epic, so it is recorded here (a durable `docs/design/` artifact,
this project's established convention for exactly this purpose) and referenced
from `AGENTS.md`'s PORTAL-exception section itself, so the NEXT engineer or agent
adding a modifier or a `ModifierCombo` value reads the rule before repeating the
Portal rollout's mistake a third time.
