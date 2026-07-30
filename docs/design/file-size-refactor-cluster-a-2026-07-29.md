# File-size refactor cluster A — compiler.py / _tool_loop.py / state.py

Date: 2026-07-29
Scope: `src/neograph/compiler.py` (761 lines), `src/neograph/_tool_loop.py` (727 lines),
`src/neograph/state.py` (659 lines). Read in full before proposing anything below.
Read-only research; no production code touched.

Constraint carried through every proposal: **neograph-s7zt3.11 (Phase 8, extending
fusion ModifierCombo lowering to the Construct level) is an open epic that will keep
editing the modifier-dispatch `match COMBO_DECOMPOSITION[combo].primary:` blocks** in
both `compiler.py` (`_add_subgraph`, `_add_node_to_graph`) and `state.py`
(`compile_state_model`'s sub-construct loop, `_add_output_field`,
`_add_single_output_field`). Nothing below touches those blocks or reorders their
control flow. SAFE NOW picks are deliberately the parts of these files Phase 8 will
never need to open.

---

## 1. `src/neograph/compiler.py` (761 lines)

### Responsibility map

| Lines | Cluster | Notes |
|---|---|---|
| 1-70 | Module docstring + imports | |
| 73-364 | `compile()` — the orchestration entry point | validates, builds state model, walks `construct.nodes`, dispatches to `_add_portal_mesh`/`_add_branch_to_graph`/`_add_subgraph`/`_add_node_to_graph`, compiles, wraps `CompiledNeograph` |
| 367-380 | `_collect_scripted_shims` | pure construct-tree walker, self-contained |
| 383-414 | `_collect_required_di` | pure construct-tree walker, self-contained |
| 417-431 | `describe_graph` | public API (re-exported in `__init__.py`), Mermaid diagram string, only touches `compiled.get_graph()` |
| 434-454 | `_print_dag_summary` | dev-mode (`DEV_MODE`) stderr diagnostic, only touches `compiled.get_graph()` |
| 457-588 | `_add_subgraph` | sub-construct dispatch — **Phase 8 territory** (the `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` gate + `match COMBO_DECOMPOSITION[combo].primary` is exactly what Phase 8 extends) |
| 591-707 | `_add_node_to_graph` | node dispatch — same `match` pattern, **Phase 8-adjacent** (Each×Oracle fusion check + primary-shape match) |
| 710-745 | `_add_oracle_nodes` | Oracle fan-out expander, called from `_add_node_to_graph` |
| 747-761 | `_add_each_nodes` | Each fan-out expander, called from `_add_node_to_graph` |

### Proposed extractions

**A1. `_collect_scripted_shims` + `_collect_required_di` + `describe_graph` +
`_print_dag_summary` → new module `_compile_diagnostics.py`**

- What moves: all four functions, verbatim (lines 367-454, ~88 lines including
  docstrings/blank lines).
- Why these four cluster together: none of them touch the `StateGraph` build or the
  modifier-dispatch match statements. `_collect_scripted_shims`/`_collect_required_di`
  are pure `iter_nodes(construct)` walkers producing lookup dicts; `describe_graph`/
  `_print_dag_summary` are pure read-only introspection over an already-compiled
  LangGraph object (`compiled.get_graph()`). They are grouped here as "compile-adjacent
  utilities that read the IR/compiled graph but never build it" — a real seam, not an
  arbitrary line-count split.
- Consumers to update: `compiler.py` imports them back (`from neograph._compile_diagnostics
  import _collect_scripted_shims, _collect_required_di, describe_graph,
  _print_dag_summary`); `__init__.py:31` (`from neograph.compiler import compile,
  describe_graph`) can either keep importing `describe_graph` re-exported through
  `compiler.py`, or point straight at the new module — either is fine per the
  naming-policy rule (public-ness is the `__init__.__all__` re-export, not the module
  the symbol physically lives in).
- Estimated reduction: ~90 lines off `compiler.py` (761 → ~670).
- **SAFE NOW.** Zero behavior change, no touched line falls inside or adjacent to the
  `match COMBO_DECOMPOSITION[...]` blocks Phase 8 owns. Four self-contained functions,
  one new file, two import-line edits. This is the single best low-risk win for this
  file and can land before or during Phase 8 without any coordination — Phase 8's
  diffs are entirely inside `_add_subgraph`/`_add_node_to_graph`, which are untouched
  by this extraction.

**A2. `_add_oracle_nodes` + `_add_each_nodes` → `_add_subgraph`/`_add_node_to_graph` DEFER (see below)**

- These are the modifier-expansion helpers Phase 8 will most likely need to touch
  or duplicate a Construct-level twin of (Each×Oracle fusion currently only exists
  at Node level here; Phase 8's stated goal is extending it to Construct level, i.e.
  `_add_subgraph`). Moving them now would create merge friction against Phase 8's
  in-flight diffs for no line-count win worth the risk (only ~50 lines).
- **DEFER.** Bundle with Restructure R1 below.

### DEFER — real restructuring

**R1. Split `compiler.py` into an orchestration module (`compiler.py`, keeping
`compile()` + the top-level node-walk loop) and a "node/subgraph dispatch" module
(`_compile_dispatch.py` holding `_add_subgraph`, `_add_node_to_graph`,
`_add_oracle_nodes`, `_add_each_nodes`).**

- This is the natural seam (`compile()` orchestrates; the four `_add_*` functions are
  the actual per-item lowering), and would take `compiler.py` from ~670 (post-A1) to
  roughly 290 lines. But every one of those four functions is exactly what Phase 8
  is actively editing (extending Each×Oracle fusion, and likely adding new
  `PrimaryShape` arms/branches at the Construct level in `_add_subgraph`). Moving them
  to a new module NOW would force Phase 8 to rebase against a file move mid-epic —
  exactly what we've been told not to do.
- Needs its own design pass **after** Phase 8 lands: at that point the fusion logic
  will have stabilized in its final shape (Node-level fused in `_add_node_to_graph`,
  Construct-level fused in `_add_subgraph`), and the split can be drawn cleanly along
  "orchestration vs per-item lowering" without fighting an in-flight diff.

---

## 2. `src/neograph/_tool_loop.py` (727 lines)

### Responsibility map

| Lines | Cluster | Notes |
|---|---|---|
| 1-48 | Module docstring + imports | |
| 50-63 | `_render_tool_result_for_llm` | tool-result-to-LLM-text rendering |
| 66-253 | **Provider resilience: string tool_calls.args coercion** — `_CoercingToolWrapper`, `_string_args_tool_errors`, `_to_lc_messages`, `UNPARSEABLE_ARGS_MARKER`, `_unparseable_args_raw`, `_coerce_string_args_result`, `_empty_recovery_message` | fully self-contained; only depends on `structlog`/`langchain_core.messages`/`pydantic.ValidationError`, none of which are neograph-internal |
| 261-450 | Tool-loop prep — `_ToolLoopPrep` dataclass, `_raise_async_factory_error`, `_lookup_factory`, `_factory_tool_config`, `_instantiate_tools`/`_ainstantiate_tools` (sync/async twins), `_build_loop_preamble`, `_assemble_tool_loop_prep` | tool instantiation + LLM/messages preamble |
| 453-542 | `_prepare_tool_loop` / `_aprepare_tool_loop` | sync/async twins gluing the prep pieces together |
| 545-595 | `_finish_tool_loop` | postamble: usage summation, logging, cost callback |
| 597-727 | `_raise_no_structured_output`, `_parse_final_turn` / `_aparse_final_turn` | ReAct final-turn parse + fallback-strategy dispatch, sync/async twins |

### Proposed extraction

**B1. The provider-resilience coercion cluster (lines 66-253, ~188 lines) →
new module `_tool_call_coercion.py`**

- What moves: `_CoercingToolWrapper`, `_string_args_tool_errors`, `_to_lc_messages`,
  `UNPARSEABLE_ARGS_MARKER`, `_unparseable_args_raw`, `_coerce_string_args_result`,
  `_empty_recovery_message`, verbatim.
- Why this is a real seam: this cluster solves ONE well-bounded problem (a provider —
  documented as DeepSeek R1 via OpenRouter — emitting `tool_calls.args` as a JSON
  string instead of a dict) and has no dependency on the tool-loop prep/parse
  machinery below it. The only outside consumer of anything in this cluster is
  `_agent_cycle.py`, which imports `UNPARSEABLE_ARGS_MARKER` directly (confirmed via
  grep — `_agent_cycle.py`'s `_tool_call_precheck` reads the marker). That import
  already crosses a module boundary today (`_tool_loop` → consumed by `_agent_cycle`),
  so extracting it to its own module is a pure rename of the import path, not a new
  coupling.
- Estimated reduction: ~188 lines off `_tool_loop.py` (727 → ~540).
- Update needed: `_tool_loop.py` re-imports the symbols it still uses
  (`_CoercingToolWrapper` in `_assemble_tool_loop_prep`); `_agent_cycle.py`'s import of
  `UNPARSEABLE_ARGS_MARKER` (and presumably `_unparseable_args_raw`) repoints to the
  new module.
- **SAFE NOW.** This cluster has zero relationship to the Portal/fusion epic — it is
  pure LLM-provider-quirk handling, nowhere near `ModifierCombo`/`COMBO_DECOMPOSITION`.
  Mechanical cut-paste + two import-path edits. Does not fully bring the file under
  500 on its own (540 lines) but is the single biggest, safest, most self-contained
  cut available in this file.

### DEFER — real restructuring

**R2. Split the remaining ~540 lines along the prep/parse seam**: a
`_tool_loop_prep.py` (the `_ToolLoopPrep` dataclass + `_instantiate_tools`/
`_ainstantiate_tools`/`_build_loop_preamble`/`_assemble_tool_loop_prep`/
`_prepare_tool_loop`/`_aprepare_tool_loop`, ~230 lines) and a `_tool_loop_parse.py`
(the `_finish_tool_loop`/`_parse_final_turn`/`_aparse_final_turn` postamble+parse
cluster, ~180 lines), leaving `_tool_loop.py` as a thin re-export/coordination shim
(or folding the split fully and retiring `_tool_loop.py`'s name).

- Not SAFE NOW because the sync/async twinning here (`_instantiate_tools`/
  `_ainstantiate_tools`, `_prepare_tool_loop`/`_aprepare_tool_loop`,
  `_parse_final_turn`/`_aparse_final_turn`) is dense and the docstrings explicitly
  call out that `_parse_final_turn` is "the single source of truth for the hard
  cluster — shared by the monolithic tool loop and the inline agent-cycle parse
  node" — i.e. there's a THIRD consumer of this logic (`_agent_cycle.py`'s inline
  parse node) that must be traced carefully before any module boundary is drawn, to
  avoid creating an import cycle or splitting a "single source of truth" comment
  across a module boundary that then silently drifts. This needs a dedicated pass
  to map every caller of `_parse_final_turn`/`_aparse_final_turn` first.
- Also worth deciding, in that same pass: whether the sync/async twin pairs should
  move to a shared "twin" module together (prep twins + parse twins in one place) or
  split prep-vs-parse as sketched above — a real design choice, not mechanical.

---

## 3. `src/neograph/state.py` (659 lines)

### Responsibility map

| Lines | Cluster | Notes |
|---|---|---|
| 1-38 | Module docstring + imports | |
| 40-98 | Reducers — `_last_write_wins`, `_append_loop_result`, `_concat_reducer`, `_merge_dicts` | pure functions, no Construct/Node/modifier awareness |
| 101-363 | `compile_state_model` | the big one — nodes-only fields, branch-arm fields, **sub-construct modifier dispatch (164-230, Phase 8 territory)**, Oracle/Each id fields, loop counters, Portal mesh/dispatch fields, context fields, framework fields |
| 366-399 | `build_output_schema_model` | public-facing output-schema builder, self-contained |
| 402-429 | `_type_signature` | structural type-signature helper, self-contained |
| 432-484 | `compute_node_fingerprints` | per-node fingerprint walker, uses `_type_signature` |
| 487-512 | `compute_schema_fingerprint` | whole-state fingerprint, uses `_type_signature` |
| 515-536 | `_add_agent_channels` | agent/act ReAct-cycle state channels |
| 539-660 | `_add_output_field` / `_add_single_output_field` | per-node output-field builder — **also Phase 8-adjacent** (contains the Each×Oracle fusion collector-field logic that mirrors the Construct-level gate Phase 8 is extending) |

### Proposed extractions

**C1. The checkpoint-fingerprint cluster — `_type_signature`,
`compute_node_fingerprints`, `compute_schema_fingerprint` (lines 402-512, ~111 lines)
→ new module `_schema_fingerprint.py`**

- Why this is a real seam: this is a documented, independently-named subsystem
  ("Checkpoint resume — schema-aware auto-rewind" in `AGENTS.md`) with its own
  invariant ("both fingerprints had to move in lockstep") that is already
  conceptually separate from state-MODEL construction — it consumes a finished
  `state_model`/`construct` and produces hashes for the checkpoint-resume runner. It
  has zero interaction with the modifier-dispatch code Phase 8 touches: it reads
  `_declared_output`/`normalize_outputs` (already-computed declarations), never the
  live `COMBO_DECOMPOSITION` match.
- Consumers: `compiler.py` imports `compute_node_fingerprints`/
  `compute_schema_fingerprint` from `neograph.state` today (see `compiler.py:62-67`)
  — becomes `from neograph._schema_fingerprint import ...` instead, a one-line edit.
  Any test importing `state._type_signature` directly needs the same import-path
  update (grep before landing).
- Estimated reduction: ~112 lines off `state.py` (659 → ~547).
- **SAFE NOW.** Purely additive move, no shared local state with `compile_state_model`,
  no proximity to the Phase 8 match blocks. This is the best win for this file
  specifically because Phase 8 (per the epic description) extends fusion lowering,
  not checkpoint fingerprinting — these two concerns have never touched in this file
  and moving one out doesn't risk merge conflicts with in-flight Phase 8 diffs to
  `compile_state_model`/`_add_output_field`.

**C2. The four reducers (lines 40-98, ~59 lines) → new module `_state_reducers.py`**

- Pure functions, no dependency on `Construct`/`Node`/modifiers — only used as
  `Annotated[..., reducer]` values inside `compile_state_model`/`_add_output_field`.
  Could combine with C1 into one `_state_support.py` module, or ship standalone.
- Estimated reduction: ~59 lines (state.py → ~488 after C1+C2 combined, i.e. under
  the 500 cap on its own).
- **SAFE NOW**, same reasoning as C1 — but slightly lower priority than C1 because
  the reducers are called from inside `compile_state_model`'s Phase-8-adjacent
  branches (sub-construct match, `_add_output_field`), so review should double-check
  each call site still resolves after the import-path change. Mechanically trivial
  either way (rename import, no logic change).

### DEFER — real restructuring

**R3. Split `compile_state_model`'s ~260-line body along its per-construct-item
concerns** (nodes-only fields / branch-arm fields / sub-construct fields / Portal
mesh fields / context fields / framework fields) into named helper functions or a
small builder-object, and correspondingly split `_add_output_field`/
`_add_single_output_field`'s modifier-shape matches out of the main body.

- This is exactly the code Phase 8 will be editing (the sub-construct combo match at
  164-230, and the Each×Oracle dict-form/single-form fusion arms in
  `_add_output_field`/`_add_single_output_field`). Restructuring it now means Phase 8
  rebases against a reshaped function; restructuring it during Phase 8 means the
  restructure has to track a moving target. Do this **after** Phase 8 lands, once the
  Construct-level fusion arms are final, so the split can be drawn along the
  post-Phase-8 shape instead of guessing at it now.
- Also worth deciding in that pass: whether the Portal mesh/dispatch field-building
  block (lines 260-322) — which is unrelated to Each/Oracle/Loop fusion — could be
  extracted independently of the fusion-related restructure, since it's a separate
  concern that doesn't overlap Phase 8's edits. That's a legitimate SAFE-NOW
  candidate in its own right but was left out of this pass's SAFE NOW list to keep
  the recommended diff small and single-purpose; worth a follow-up ticket.

---

## Duplication found

- No true (copy-pasted-logic) duplication found across `compiler.py` / `_tool_loop.py`
  / `state.py`, nor against the named epic-active files. The closest thing —
  `spec_types.py:_property_type_signature` vs. `state.py:_type_signature` — is a
  false positive: same name shape, different domain (Agent Spec `Property` schema
  signatures vs. Python type-annotation signatures for checkpoint fingerprints); not
  worth consolidating.
- The repeated sync/async "twin" pairs inside `_tool_loop.py`
  (`_instantiate_tools`/`_ainstantiate_tools`, `_prepare_tool_loop`/
  `_aprepare_tool_loop`, `_parse_final_turn`/`_aparse_final_turn`) are near-duplicate
  bodies, but this mirrors the project's documented async-native twin convention
  (one graph, four verbs; sync/async twins throughout the runtime layer) — treating
  this as a defect to "de-duplicate" would fight the architecture, not fix it. Not
  flagged as duplication debt; noted only because it's why this file's line count is
  inherently ~2x what a sync-only version would be.

---

## Summary table

| File | Current | SAFE NOW total | Post-SAFE-NOW | DEFER (needs own design pass) |
|---|---|---|---|---|
| `compiler.py` | 761 | A1 (~90) | ~670 | R1: split orchestration vs. per-item dispatch (`_add_subgraph`/`_add_node_to_graph`/`_add_oracle_nodes`/`_add_each_nodes`) — after Phase 8 |
| `_tool_loop.py` | 727 | B1 (~188) | ~540 | R2: prep-vs-parse split, gated on tracing every `_parse_final_turn` consumer (incl. `_agent_cycle.py`) first |
| `state.py` | 659 | C1+C2 (~171) | ~488 | R3: split `compile_state_model`/`_add_output_field` per-concern — after Phase 8; Portal-field extraction flagged as a possible independent SAFE NOW follow-up |
