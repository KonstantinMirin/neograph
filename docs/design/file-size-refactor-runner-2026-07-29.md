# File-size refactor proposal: `src/neograph/runner.py`

Status: design only, no code changes. Written against `runner.py` as of
2026-07-29 (1225 lines). Context: repo-wide 500-line-per-file cap (tracked
separately in beads; not re-litigated here). Active epic neograph-s7zt3
(Agent Spec / Portal rebuild) has open Phase 8 work (neograph-s7zt3.11,
extending fusion ModifierCombo lowering to the Construct level) touching
`_agent_spec.py`, `loader.py`, `_wiring.py`, `modifiers.py`, `compiler.py`.
`runner.py` is NOT on that critical path, but one cluster below shares a
concept (Portal-as-Construct-member) with it — flagged explicitly.

## 1. Responsibility map (what's actually in this file)

Reading top to bottom, `runner.py` is currently SEVEN logically distinct
things wearing one filename:

| Lines | Cluster | What it does |
|---|---|---|
| 38–204 | **Portal-mesh recursion-limit budget** | `_member_hop_cost`, `_mesh_hop_cost`, `_portal_mesh_member_ids`, `_ensure_agent_recursion_limit` + 3 module constants. Pure functions that walk the construct tree (`iter_with_arms`) to compute a recursion-limit floor for agent/act cycles and Portal meshes. |
| 207–249 | **Pre-engine input/DI helpers** | `_preflight_di_check`, `_inject_input_to_config`. Small, generic. |
| 251–522 | **Checkpoint schema divergence + auto-rewind (sync)** | `_decide_checkpoint_schema`, `_verify_checkpoint_schema`, `_auto_resume_from_divergence`, `_raise_no_rewind_point`, `_raise_incompatible_schema`, `_compute_invalidated_nodes`, `_build_producer_consumer_adjacency`, `_transitive_closure`. |
| 523–630 | **Checkpointer sync/async driver-mismatch guard** | `_required_checkpointer_driver`, `_assert_checkpointer_matches_driver`. |
| 633–727 | **Checkpoint existence probe + resume/new-input config prep** | `_has_existing_checkpoint`, `_prepare_resume_config`, `_prepare_new_input`, `_mark_stream_custom`, `_mint_run_id`. |
| 736–823 | **`observe=` Langfuse integration** | `_observe_wants_langfuse`, `_langfuse_keys_present`, `_merge_observe_callbacks`, `_flush_observe`, `_evict_run_cache`. Already delimited by its own banner comment (line 736). |
| 826–1225 | **The actual verbs** | `_finalize_prepare_config`, `_prepare`/`_aprepare` (the shared pre-engine brain), `_finalize_by_mode`/`_finalize_chunk` (stream stripping), and the four public verbs `run`/`stream`/`arun`/`astream`, plus the async twins of the checkpoint-rewind cluster (`_ahas_existing_checkpoint`, `_averify_checkpoint_schema`, `_aauto_resume_from_divergence`) interleaved among them (1042–1160). |

The async twins of the checkpoint cluster (1042–1160, ~120 lines) are
physically separated from their sync counterparts by ~500 lines of unrelated
code (driver-guard, observe, verbs) purely because the file groups "sync
stuff" then "async stuff" at the top level rather than by concern. That's
itself a readability cost independent of the line-count cap.

## 2. Proposed extractions

### (A) Portal-mesh recursion-limit budget → new `_recursion_budget.py`

**Moves**: lines 38–204 verbatim (`_member_hop_cost`, `_mesh_hop_cost`,
`_portal_mesh_member_ids`, `_ensure_agent_recursion_limit`, and the three
`_LANGGRAPH_DEFAULT_RECURSION_LIMIT`/`_SUPERSTEPS_PER_AGENT_TURN`/
`_AGENT_CYCLE_OVERHEAD` constants). ~167 lines removed.

**Why it's a clean seam**: these four functions form one closed call graph
(`_ensure_agent_recursion_limit` calls the other three; nothing else in
`runner.py` calls any of them). Only inbound caller is
`_finalize_prepare_config` (line 845), which becomes a one-line import.
Dependencies (`Node`, `Construct`, `iter_with_arms`, `primary_shape`,
`PrimaryShape`, `_coerce_llm_config`) are already imported in `runner.py`
today, so the new module's import list is just a subset copy-paste.

**Epic-overlap flag**: this cluster explicitly special-cases "a Portal-carrying
Construct is a mesh member, costed as one opaque hop, its interior excluded
from the flat per-node budget" (comments cite `do0d9 §3.1 site 6`). That is
conceptually the same territory Phase 8 (fusion ModifierCombo lowering to the
Construct level) is generalizing — a Construct that carries a modifier and
must be treated specially by shape-dispatch code elsewhere. Moving this
cluster does not touch its logic, only its file location, so it does not
block or require re-planning Phase 8. But if Phase 8's branch is mid-flight
against `runner.py` lines in this range, landing this move first will cause a
rebase/merge-conflict cost for that branch. **Recommendation: sequence
this — either land it before Phase 8 touches these lines, or coordinate a
one-time rebase with whoever owns s7zt3.11.**

### (B) Checkpoint schema divergence + auto-rewind (sync + async) → new `_checkpoint_rewind.py`

**Moves**: lines 251–522 (sync: `_decide_checkpoint_schema`,
`_verify_checkpoint_schema`, `_auto_resume_from_divergence`,
`_raise_no_rewind_point`, `_raise_incompatible_schema`,
`_compute_invalidated_nodes`, `_build_producer_consumer_adjacency`,
`_transitive_closure`) **and** their async twins currently stranded at
1064–1113 (`_averify_checkpoint_schema`, `_aauto_resume_from_divergence`).
~340 lines removed combined.

**Why it's a clean seam**: this is the single most self-contained cluster in
the file — it has its own vocabulary (schema fingerprint, node fingerprint,
invalidated-node closure, rewind point), its own bead citations
(neograph-ykun, neograph-v63o, neograph-1gdw) separate from every other
cluster, and a narrow, already-documented shared-decision function
(`_decide_checkpoint_schema`) that both sync and async callers route through.
Only inbound callers from the rest of `runner.py` are `_prepare` (line 887,
897) and `_aprepare` (line 1144, 1152) — both become imports. Moving the async
twins alongside the sync originals also **fixes the readability problem noted
in §1** (they're currently exiled ~500 lines away from the logic they mirror).

**No epic overlap**: nothing here touches Portal/ModifierCombo/primary_shape;
it's purely checkpoint-fingerprint bookkeeping.

### (C) Checkpointer sync/async driver-mismatch guard → new `_checkpoint_driver.py`

**Moves**: lines 523–630 (`_required_checkpointer_driver`,
`_assert_checkpointer_matches_driver`). ~108 lines removed.

**Why it's a clean seam**: single well-documented mechanism (event-loop /
source-sniffing classification), one inbound caller each from `_prepare`
(877) and `_aprepare` (1136). No shared state with the rewind cluster beyond
both reading `graph.checkpointer` — could alternatively be folded into (B) as
one `_checkpoint.py` module (see consolidated option below) since both are
"checkpoint pre-engine gates."

**Consolidated option**: (B) + (C) + the existence-probe half of the
"prepare" cluster (`_has_existing_checkpoint` / `_ahas_existing_checkpoint`,
~50 lines total) could land as ONE `_checkpoint.py` module (~500 lines) rather
than three files, since all of it is "things `_prepare`/`_aprepare` do to
reason about the checkpointer before invoking the engine." This trades one
slightly-large-but-cohesive file for three small ones — either split is
mechanical; pick based on whether 500 lines in one checkpoint module is
preferred over module proliferation.

### (D) `observe=` Langfuse integration → new `_observe.py`

**Moves**: lines 736–823 (`_observe_wants_langfuse`, `_langfuse_keys_present`,
`_merge_observe_callbacks`, `_flush_observe`, `_evict_run_cache`). ~90 lines
removed.

**Why it's a clean seam**: already delimited by its own banner comment (line
736) declaring it "A THIN-VERB concern" with a structural guard
(`TestNoModuleLevelLangfuseImports`) pinning that its `langfuse` imports stay
function-local. Moving it to its own module makes that guard's job *more*
legible (it can assert "no top-level `langfuse` import in `_observe.py`"
instead of scanning all of `runner.py`). `_evict_run_cache` isn't
Langfuse-specific but is the symmetric finalize-tail partner called from every
verb's `finally` right next to `_flush_observe` — keeping them together
matches the existing comment's framing ("the SAME finalize seam"). Four
inbound callers (`run`, `stream`, `arun`, `astream`) each become an import.

**No epic overlap.**

## 3. SAFE NOW vs DEFER

**SAFE NOW — all four extractions (A, B, C, D):**

All four are pure mechanical moves: each cluster is a closed call graph
(callees don't call back into `runner.py` internals), each has at most 1–2
inbound call sites that become one-line imports, and none requires touching
function bodies, signatures, or test files (tests import from `neograph.runner`
today via the package `__init__`/test imports — verify call sites still
resolve, but no behavioral test should need editing). This is exactly the kind
of "move a self-contained cluster, no behavior change" work the cap ticket
wants landed before/alongside the enforcement guard.

Combined reduction: **167 (A) + 340 (B) + 108 (C) + 90 (D) = ~705 lines**,
leaving `runner.py` at roughly **520 lines** (prepare/verb orchestration:
`_preflight_di_check`, `_inject_input_to_config`, `_prepare_resume_config`,
`_prepare_new_input`, `_mark_stream_custom`, `_mint_run_id`,
`_finalize_prepare_config`, `_prepare`/`_aprepare`, `_finalize_by_mode`/
`_finalize_chunk`, and the four verbs). That's close to but not quite under
500 — a fifth micro-extraction (moving `_finalize_by_mode`/`_finalize_chunk`,
~40 lines, into a `_stream_finalize.py`) would close the remaining gap if the
cap is enforced strictly at 500, not "close enough."

**Sequencing note**: do (B), (C), (D) first (zero epic overlap, no
coordination needed). Do (A) either first (before Phase 8 branch touches
these lines) or coordinate a rebase — it's still SAFE NOW in the sense of
"mechanical, no behavior change," just needs a heads-up to whoever is
mid-flight on s7zt3.11.

**DEFER — none proposed for this file.** Every cluster identified above is a
closed, non-overlapping subgraph with a small, stable call interface; there is
no case here where splitting would require re-threading shared mutable state,
changing a public signature, or resolving a genuine layering ambiguity. If a
future pass wants to go further, the one candidate for a *real* design
question (not a mechanical move) is: **should `_prepare`/`_aprepare` be
restructured to reduce the sync/async duplication itself** (both are
~60-line near-mirrors that already share every pure helper) — e.g. a single
async-native implementation with a sync bridge, or a codegen/template
approach. That is a genuine "how much duplication between the two engine
drivers is acceptable vs. worth collapsing" architectural question or the
kind of thing that would touch the checkpointer-driver-mismatch guard's
reason for existing (justifying two drivers at all) — out of scope for a
mechanical file-split pass, name it and defer.

## 4. Duplication check against epic-active files

Checked `_wiring.py`, `modifiers.py`, `factory.py`, `_agent_cycle.py`,
`_agent_spec.py`, `loader.py`, `compiler.py` for overlap with `runner.py`'s
logic (not just superficial name similarity):

- **No literal/copy-paste duplication found.** `runner.py`'s Portal-mesh cost
  functions are the only code in the repo that computes a *superstep-cost
  floor*; nothing in the other files re-implements or reads that same
  computation.
- **Real but non-duplicative shared pattern**: `iter_with_arms(construct)` +
  `switch/match on primary_shape(item)` is used as a construct-tree walking
  idiom in FIVE places — `compiler.py` (graph building), `loader.py` (spec
  export, `case PrimaryShape.*` at ~1105–1139), `_agent_spec.py` (export
  walk, ~1021–1043 and ~1294–1346), `_wiring.py` (line 719, Operator/Portal
  check), and `runner.py`'s `_mesh_hop_cost`/`_portal_mesh_member_ids`
  (38–156). Each does a DIFFERENT thing per shape (compile a subgraph vs.
  serialize a spec vs. sum a cost), so this is not "extract a shared
  function" duplication — it's a shared *traversal idiom* with divergent
  per-shape bodies. Not a SAFE NOW extraction (collapsing five different
  match arms into one dispatcher is a real design question — different
  callback signatures, different Construct-vs-Node handling per site) but
  worth naming as a DEFER candidate for a future "shape-dispatch visitor"
  design pass if the five call sites keep growing in lockstep (they already
  show signs of it: `_agent_spec.py` and `loader.py` both have near-identical
  `case PrimaryShape.ORACLE: ... case PrimaryShape.EACH: ... case
  PrimaryShape.LOOP: ... case PrimaryShape.BARE: ... case
  PrimaryShape.PORTAL:` blocks at ~1021 and ~1105 respectively — closest
  thing to real duplication found, but restructuring it is a cross-file design
  question, not a mechanical runner.py move, so it's out of scope here.**
  Flagging for whoever owns those two files**, not proposing it as part of
  this runner.py extraction.
