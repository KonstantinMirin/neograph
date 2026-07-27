# Completeness review: agent-spec-portal-master-architecture-2026-07-27.md

Date: 2026-07-27
Lens: is every cell of the §5 per-combo matrix addressed with a verified
status, and does the matrix (plus the surrounding claims it depends on) match
real source? Verified by direct source reads, not re-derivation from the
document's own prose.

## Verdict: PASS, with 3 minor gaps (no hand-waved/TODO/unknown cells found)

## Enum cross-check

`ModifierCombo` in `src/neograph/modifiers.py` has exactly 12 values (confirmed
via `_COMBO_MAP`, lines 91-103): BARE, EACH, ORACLE, LOOP, OPERATOR, PORTAL,
EACH_ORACLE, EACH_OPERATOR, ORACLE_OPERATOR, LOOP_OPERATOR,
EACH_ORACLE_OPERATOR, PORTAL_OPERATOR. The §5 matrix's row set (BARE, EACH,
ORACLE, LOOP, OPERATOR, EACH_OPERATOR, ORACLE_OPERATOR, LOOP_OPERATOR,
EACH_ORACLE, EACH_ORACLE_OPERATOR, PORTAL split into 4 sub-rows, PORTAL_OPERATOR
split into 2 sub-rows) covers all 12 values with none missing and none
duplicated in a way that hides a distinct value. The PORTAL/PORTAL_OPERATOR
sub-row split (by dispatch-vs-peer and by member composition) is not padding —
each sub-row carries a genuinely different verified status, so the split
increases rather than reduces completeness.

One implicit completeness point verified independently: the document does not
give "PORTAL_OPERATOR, dispatch mode" a row. This is correct, not an omission —
`ModifierSet.__post_init__`-equivalent validation in `modifiers.py:892-895`
raises `ConstructError` at ModifierSet-construction time for
`portal.is_dispatch and operator is not None`, so that cell is structurally
unreachable and rightly excluded rather than silently dropped.

## Claims spot-verified directly against source (all confirmed accurate)

- `compiler.py`'s two `match combo:` blocks (`_add_node_to_graph` line ~575,
  `_add_subgraph` line ~509) exhaustively handle all 12 combos (`assert_never`
  on both), including the EACH_ORACLE/EACH_ORACLE_OPERATOR permanent rejection
  on Constructs (lines 509-517) and the Portal-on-subconstruct rejection
  (lines 552-560, still present). Confirms §5's "WORKS" claims for Node rows
  and the "REJECTED by compiler itself" claim for EACH_ORACLE(_OPERATOR) on
  Construct.
- `_agent_spec.py::_lower_construct_item` (lines 793-877): exactly 5 combos
  handled (ORACLE/EACH/LOOP/OPERATOR/BARE, lines 846-870), the other 7 fall to
  a generic `ConfigurationError` (line 872). The `Construct` branch (lines
  832-835) calls `to_agent_spec(item)` directly with **no**
  `classify_modifiers` call at all — confirms the "silent modifier drop on
  Construct export" claim exactly as described.
- `to_agent_spec`'s `mesh_members` filter (lines 957-967) is
  `isinstance(item, Node)`-gated — confirms the Portal-mesh-with-Construct-member
  false-rejection claim.
- `_lower_oracle` (line 497) calls `_lower_generation_step` directly per
  variant, never `_lower_node`, so it never reaches
  `_reject_unrepresentable_fields` (only called from `_lower_node`, line 385) —
  confirms the Oracle guard-skip gap.
- `_lower_portal_mesh_to_swarm` (lines 880-940) never reads
  `member.modifier_set.operator` anywhere in the per-member loop — confirms the
  PORTAL_OPERATOR HITL-gate silent drop.
- `loader.py::_reconstruct_swarm_mesh` (line 689) always builds members via
  `_node_from_spec_agent` (line 715), no Construct-member branch — confirms
  "Construct-as-mesh-member never reconstructed." `_MARK_PORTAL_SPEC` is
  written in `_agent_spec.py` but has zero read-side references in `loader.py`
  — confirms the marker-write-never-read claim.
- `state.py`'s Portal state-field builder (lines 261-265) sources
  `portal_members` from `nodes_only` only, not `nodes_only + sub_constructs`
  (contrast the Oracle/Each bucket immediately above it, which does union both)
  — confirms the state.py half of the Construct-as-mesh-entry bug, unchanged
  in the current working tree.
- `modifiers.py`'s dispatch+Operator rejection (lines 892-895), `_fan_agent.py`'s
  fan-over-agent support gate, and all ten modules' cited line ranges
  (`_subconstruct.py`, `_input_shape.py`, `runner.py`, `_wiring.py`,
  `_state_write.py`) were spot-checked and match the document's characterization.

## Gaps found

1. **Stale claim on one of the two "Construct-as-mesh-entry" bug sites**
   (`docs/design/agent-spec-portal-master-architecture-2026-07-27.md` §2 and
   §5's PORTAL-peer-Construct-entry row). The document states
   `compiler.py:254`'s dispatch gate is `isinstance(item, Node)`, misrouting a
   Portal-modified Construct entry to `_add_subgraph`. The current working
   tree (uncommitted, per `git blame` timestamped today) already reads
   `isinstance(item, (Node, Construct))` at that line — i.e., **the top-level
   dispatch half of this bug appears to already be fixed**, live, by another
   in-flight agent, while `state.py:261-265`'s `nodes_only`-only bug (the
   second site) is still unfixed. This isn't a defect in the document's
   research (it was almost certainly accurate against the committed tree when
   written/reviewed), but a reader picking up the Build Plan's Phase 1 right
   now needs to re-verify site 1's status before treating it as still-broken —
   the document doesn't (and structurally can't) account for a concurrent
   uncommitted edit landing after it was written. Flagging so Phase 1 isn't
   redone or mis-scoped.

2. **Agent/act-mode nodes are an orthogonal axis the matrix doesn't surface,
   though the document is aware of it.** `_fan_agent.py` (cited correctly in
   §4's ten-module table as duplicated-dispatch site #9) governs which
   Each/Oracle/Loop-over-agent/act shapes are legal at all (self-contained or
   single-producer; multi-producer only for Oracle via packer synthesis) via
   an auto-wrap-to-Construct pre-pass (`_wrap_fan_over_agents`) before the
   Node-level `match combo:` in compiler.py ever sees it. The §5 matrix's
   "Node: compile WORKS" cells for EACH/ORACLE/LOOP rows are true only for the
   shapes `_fan_agent.py` admits — an agent/act node with a disallowed fan
   shape fails at assembly, before compile. The document never claims the
   matrix is mode-granular, so this isn't a broken promise, but it also isn't
   flagged as an explicit scope exclusion the way "dispatch-mode PORTAL_OPERATOR
   doesn't get a row" is — a reader could reasonably expect §5's "WORKS" to be
   unconditional. Minor: worth one sentence in §5's preamble noting agent/act
   node-mode + fan-shape legality is a separate, already-covered-elsewhere
   axis, not folded into this matrix.

3. **Export-side interaction between the agent/act fan-wrap pre-pass and
   `_agent_spec.py` is not addressed anywhere in the document.** Because
   `_wrap_fan_over_agents` runs at *compile* time, not at Agent-Spec-export
   time, an Oracle/Each/Loop over an agent/act node presented to
   `to_agent_spec` is still the original un-wrapped `Node` (IR before the
   compiler's rewrite pass), so it goes through `_lower_construct_item`'s
   ORACLE/EACH/LOOP Node arms, which dispatch per-mode via
   `_lower_generation_step` (confirmed to already handle `agent`/`act` modes,
   `_agent_spec.py` lines 331+). This likely works correctly by the same
   mode-dispatch mechanism the document credits elsewhere, but the document
   never states this explicitly or cites it as verified — it's an inference
   from adjacent code, not a claim the document makes and backs with
   file:line evidence the way every other cell is. Worth an explicit
   verified-or-not line in a future revision; not confirmed broken, just
   unaddressed.

None of the three gaps above are hand-waved "TODO"/"unknown" cells in the
literal sense the document's own line 308 claim ("Every cell below has a
verified status — none are TODO/unknown") addresses — that specific claim
holds for the 16 rows as written. Gaps 2 and 3 are about an axis the matrix
doesn't claim to cover; gap 1 is a currency/staleness issue caused by a
concurrent in-flight edit, not a research shortcoming.
