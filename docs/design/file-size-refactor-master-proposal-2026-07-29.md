# Master file-size-refactor proposal (consolidated)

Date: 2026-07-29. Synthesizes 13 per-file/per-cluster survey docs covering 19
files, all part of the repo-wide 500-line-per-file cap effort (guard test
tracked separately in beads — not re-litigated here). Read-only synthesis; no
code changed by this document.

Source surveys: `file-size-refactor-{agent-spec,loader,wiring,modifiers,
factory,agent-cycle,forward,runner,lint,decorators,cluster-a,cluster-b,
cluster-c}-2026-07-29.md`.

Constraint carried through every item below: **neograph-s7zt3 Phase 8
(neograph-s7zt3.11, extending fusion `ModifierCombo` lowering to the
Construct level) is open and must not be paused or re-planned.** Every SAFE
NOW item was chosen by its surveying agent specifically because Phase 8's
diff does not reach it.

---

## 1. SAFE NOW — prioritized, epic-active files first

### Epic-active files (Phase 8 is currently editing or adjacent to these)

| # | File | Extraction target | Lines removed | Why safe |
|---|---|---|---|---|
| 1 | `_wiring.py` (1461) | `_wiring_oracle_each.py` — `_collect_each_items`, `_empty_each_bypass`, `_wire_oracle`, `_wire_each`, `_add_each_oracle_fused`, `_merge_one_group`, `_amerge_one_group` | ~392 | This IS the cluster Phase 8 extends; self-contained (only `compiler.py` calls the public entry points), so moving it *before* Phase 8 adds Construct-level fusion code means that new code lands in a small module, not a 1461-line one. Highest-leverage single move in the whole survey. |
| 2 | `_wiring.py` | `_wiring_branch.py` — `_add_arm_nodes`, `_wire_arm_edges`, `_add_branch_to_graph` | ~185 | Branch has zero relationship to Portal/fusion. |
| 3 | `_wiring.py` | `_wiring_loop.py` — `_make_loop_router`, `_node_loop_unwrap`, `_construct_loop_unwrap`, `_add_loop_back_edge`, `_add_subgraph_loop` (keep `_resolve_condition` in `_wiring.py`, it's the one genuinely cross-cutting 17-line utility) | ~205 | Loop wiring is orthogonal to fusion/Portal. |
| 4 | `_agent_spec.py` (1561) | `_agent_spec_markers.py` — `_MARK_*` constants + `_import_agent_spec_flow_classes` | ~45 | Also fixes a backwards import (`loader.py` currently imports markers from the exporter module `_agent_spec.py`); both should import from a neutral module. |
| 5 | `_agent_spec.py` | `_agent_spec_placeholders.py` — `_translate_placeholders`, `_node_translation`, `_is_translation_eligible`, `_prompt_spec_marker` | ~122 | Stable call signature; Phase 8 never changes it (`_is_translation_eligible` already returns `False` for `Construct`, which is what Phase 8 needs, unchanged). |
| 6 | `_agent_spec.py` | `_agent_spec_node_lowering.py` — `_reject_unrepresentable_fields`, `_lower_generation_step`, `_lower_node`, `_make_agent`, `_tool_to_server_tool`, `_agent_spec_marker`, `_make_llm_config`, `_make_server_tool` | ~240 | A `Construct` item never reaches this cluster (routed to `FlowNode(subflow=...)` instead), so Phase 8's Construct-fusion work has no reason to touch it. |
| 7 | `_agent_spec.py` | `_agent_spec_portal.py` — `_lower_portal_mesh_to_swarm`, `_is_peer_mesh_member` | ~185 | Structurally separate path (mesh detection happens before the normal Flow-building loop). Land after items 4–6 (depends on their new modules). |
| 8 | `loader.py` (1393) | `_spec_loader.py` — `load_spec`, `MAX_SPEC_SIZE`, `_parse_input`, `_validate_spec`, `_build_construct`, `_resolve_tool`, `_build_node`, `_build_sub_construct`, `_apply_modifiers` | ~240 | Zero call graph into the Agent-Spec importer half of the file — genuinely two unrelated importers sharing a file for historical reasons. |
| 9 | `loader.py` | `_agent_spec_swarm_import.py` — `_swarm_agents_ordered`, `_synthesize_swarm_payload`, `_swarm_trigger`, `_flow_member_to_construct`, `_reconstruct_swarm_mesh`, `_reconstruct_swarm_mesh_with_operator_gates` | ~250 | No reference to `PrimaryShape`/`COMBO_DECOMPOSITION`/`is_each_oracle_fused` — confirmed by reading; Swarm/Portal-mesh reconstruction is orthogonal to Phase 8's fusion dispatch. |
| 10 | `modifiers.py` (1116) | `_portal.py` — `HANDOFF_END`, `DISPATCH_ROUTE`, `Portal` class, `_group_portal_members` | ~175 | Largest self-contained cluster; every external consumer already imports `Portal` by name, so `modifiers.py` just re-exports — zero call-site edits. |
| 11 | `modifiers.py` | `_oracle_protocols.py` — `MergePreProcess`, `MergePostProcess`, `MergeFallback` | ~40 | Typing-only Protocols, single consumer (`Oracle`'s field types). |
| 12 | `modifiers.py` | `_each.py` — `EachFailure`, `Each`, `split_each_path` | ~50 | Self-contained; external callers only use the already-public `split_each_path`. |
| 13 | `factory.py` (1007) | `_raw_dispatch.py` — `_make_raw_wrapper`, `_make_araw_wrapper` | ~62 | Zero `Command(` usage → no guard-G1-allowlist edit needed. Cleanest cut in the file. |
| 14 | `factory.py` | `_portal_dispatch_factory.py` — `make_portal_dispatch_fn` + nested helpers | ~222 | Self-contained (never calls `_portal_route_to_command`/`_tool_handoff_to_command`); confirmed outside Phase 8's scope via `bd show neograph-s7zt3.11`. Requires one G1-allowlist line + ~6 docstring citation updates. |
| 15 | `_agent_cycle.py` (1054) | `_agent_tool_calls.py` — the per-tool-call pure-function cluster (`_raise_sync_tool_async` … `_lift_resource_refs`), already self-labeled "the DRY-01 extraction" | ~200 | Zero closure coupling; largest, cleanest cluster in the file. |
| 16 | `_agent_cycle.py` | `_agent_gate.py` — `_gate_approved`, `make_tool_gate_bodies` | ~83 | Fully self-contained; only shared symbol is `cycle_names` (one-line import). |
| 17 | `compiler.py` (761) | `_compile_diagnostics.py` — `_collect_scripted_shims`, `_collect_required_di`, `describe_graph`, `_print_dag_summary` | ~90 | None of the four touch the `StateGraph` build or the `match COMBO_DECOMPOSITION[...]` blocks Phase 8 owns. |

**Epic-active-file SAFE NOW subtotal: ~2,786 lines**, with each file's
post-extraction size: `_wiring.py` 1461→~680, `_agent_spec.py` 1561→~970,
`loader.py` 1393→~900, `modifiers.py` 1116→~895 (per its survey's stated
combined figure), `factory.py` 1007→~723, `_agent_cycle.py` 1054→~770,
`compiler.py` 761→~670.

### Non-epic-active files (no coordination needed, land whenever convenient)

| # | File | Extraction target | Lines removed |
|---|---|---|---|
| 18 | `runner.py` (1225) | `_recursion_budget.py` (Portal-mesh recursion-limit budget) | ~167 |
| 19 | `runner.py` | `_checkpoint_rewind.py` (schema divergence + auto-rewind, sync **and** async twins reunited) | ~340 |
| 20 | `runner.py` | `_checkpoint_driver.py` (sync/async driver-mismatch guard) | ~108 |
| 21 | `runner.py` | `_observe.py` (Langfuse integration) | ~90 |
| 22 | `lint.py` (1068) | `_lint_predict.py` (pure predictors, zero `LintIssue` emissions) | ~122 |
| 23 | `lint.py` | `_lint_kind_registry.py` (`LintKindMeta` + `LINT_KIND_META`; needs a companion 1-line import edit in `scripts/gen_api_manifest.py`) | ~112 |
| 24 | `lint.py` | `_lint_tool_checks.py` (tool-policy cluster; **needs `gen_api_manifest.py`'s AST-scan list extended** — contains literal `LintIssue(kind=...)` sites) | ~208 |
| 25 | `decorators.py` (975) | `_node_modifier_kwargs.py` (`_is_trivial_body`, `_apply_eager_oracle_gen_type`, `_build_oracle_kwargs`, `_build_each_kwargs`, `_build_portal_kwargs`) | ~149 |
| 26 | `decorators.py` | `_merge_fn_decorator.py` (`_qualname_site`, `_same_def_site`, `merge_fn()`) | ~168 |
| 27 | `forward.py` (1578) | `_forward_trace.py` (trace orchestration + branch merge) | ~230 |
| 28 | `forward.py` | `_forward_proxy.py` (proxy/tracer core, do after #27) | ~340 |
| 29 | `_tool_loop.py` (727) | `_tool_call_coercion.py` (provider string-args coercion cluster) | ~188 |
| 30 | `state.py` (659) | `_schema_fingerprint.py` (`_type_signature`, `compute_node_fingerprints`, `compute_schema_fingerprint`) | ~112 |
| 31 | `state.py` | `_state_reducers.py` (the four reducer functions) | ~59 |
| 32 | `_llm_retry.py` (658) | `_null_defaults.py` (null-default coercion cluster) | ~163 |
| 33 | `_llm_retry.py` | `_json_extract.py` (`_extract_balanced`, `_extract_json`) | ~73 |
| 34 | `_oracle.py` (650) | `_each_redirect.py` (`make_each_redirect_fn` — the one non-Oracle function in the file) | ~65 (up to ~155 if the two Oracle-redirect wrappers are folded in too) |
| 35 | `describe_type.py` (552) | `_describe_value.py` (`describe_value` instance-rendering cluster) | ~108 (alone clears the 500 cap for this file) |
| 36 | `node.py` (522) | `_node_protocols.py` (`SkipPredicate`, `SkipValueFactory`, `RawNodeFn`, `HasName`) | ~55 |
| 37 | `node.py` | `_type_spec.py` (`_validate_type_spec`, `_is_type_like`, `TypeSpec`, `TypeSpecStatic`) | ~55 |
| 38 | `node.py` | `_node_run_isolated.py` (`Node.run_isolated`'s body, delegated from a 5-line wrapper) | ~150 |
| 39 | `di.py` (506) | `_resource_hydration.py` (resource-fetcher + hydration cluster) | ~230 |

**Non-epic-active SAFE NOW subtotal: ~3,288 lines** (using the lower `_oracle.py`
estimate of 65).

### Grand total

**~6,074 lines removable via SAFE NOW extractions alone, across 19 files**,
zero behavior change, zero pause/re-plan of Phase 8. Only 3 of the 19 files
(`describe_type.py`, `node.py`, `di.py`) drop under the 500-line cap from
SAFE NOW alone; every other file still needs its DEFER item(s) (§2) to fully
clear the bar — expect the guard ticket's shrink-only ceiling allowlist to
carry non-zero entries for the rest even after this pass lands.

---

## 2. DEFER — grouped by theme

### Theme 1: Portal mesh-member routing + agent-cycle wiring (spans 2 files)
- `_wiring.py` lines 703–1038 (`_contiguous_portal_mesh`, `_make_portal_subgraph_member_fn`, `_add_portal_mesh`, `_add_portal_dispatch`) + lines 1201–1436 (`_wire_agent_cycle_body`, `_add_agent_cycle`, `_add_portal_agent_cycle_member`) — ~572 lines.
- `factory.py` lines 109–719 (`make_portal_fn`, `_portal_route_to_command`, `_tool_handoff_to_command`, `make_portal_approval_fn`, `make_portal_subgraph_fn`, `make_portal_agent_cycle_fn`, `make_portal_agent_cycle_tool_handoff_fn`) — ~610 lines.
- What planning it needs: a decision on where the shared routing-decision core (`_portal_route_to_command`/`_tool_handoff_to_command`) and the shared `_wire_agent_cycle_body` live relative to their multiple call sites (plain node / sub-construct / agent-cycle member), plus a reworded G1 guard invariant in `AGENTS.md` itself (currently hard-codes "`Command(` only in `factory.py` and `runner.py`"). This is the live surface of the same epic Phase 8 belongs to — do after Phase 8 (and any near-term Portal-combo follow-on) lands.

### Theme 2: The `ModifierCombo`/`PrimaryShape` dispatch core (spans 5+ files) — Phase 8's actual working set
- `_agent_spec.py` lines 290–1096 (`_lower_oracle`/`_lower_each`/`_lower_loop`/`_lower_operator`/`_LoweredItem`/`_lower_construct_item`) and 1282–1562 (`to_agent_spec`'s edge-wiring body).
- `loader.py` lines 348–782 (`_reconstruct_*` walk) + 1036–1152 (`from_agent_spec`) + the shared property/type helpers at 74–345.
- `modifiers.py` lines 65–309 (`ModifierCombo`/`COMBO_DECOMPOSITION`/`classify_modifiers` itself — do not relocate, only revisit once Phase 8 stabilizes it).
- `compiler.py`'s `_add_subgraph`/`_add_node_to_graph`/`_add_oracle_nodes`/`_add_each_nodes` (a clean "orchestration vs per-item dispatch" split, ~380 lines, blocked the same way).
- `state.py`'s `compile_state_model` sub-construct dispatch + `_add_output_field`/`_add_single_output_field` fusion arms (~260+120 lines).
- `_oracle.py` lines 223–582 (merge-algorithm core: `_build_upstream_context`, merge_prompt/merge_fn execution, `_merge_variants`/`_amerge_variants`, `make_oracle_merge_fn`) — `_wiring.py` already imports these as "the canonical merge step."
- What planning it needs: this is literally what Phase 8 is rewriting. Every one of these clusters' *final shape* depends on what Phase 8's Construct-level fusion change does. Pre-designing any boundary now would likely be thrown away. Do a dedicated design pass immediately after Phase 8 (`neograph-s7zt3.11`) closes, informed by the shape it actually lands in.

### Theme 3: `ModifierSet` / cross-modifier exclusion table
- `modifiers.py` lines 916–1116 (`ModifierSet`, `_SlotRule`, `_SLOT_RULES`) → candidate `_modifier_set.py`.
- Needs resolving a genuine two-way dependency (the modifier classes ↔ `ModifierSet`'s `isinstance` dispatch) via `TYPE_CHECKING`/function-local imports — and should also fix the duplication noted in §3 (drive `ModifierSet.model_post_init`'s hard-coded Portal-exclusion arms off `_SLOT_RULES` instead of a second hand-written copy) in the same pass.
- Also entangled: `modifiers.py`'s `Modifiable` mixin (lines 312–528, `__or__`/`.map()`) — the lazy `Construct`/`Node`/`_construct_validation` import inside `__or__` dodges a real cycle; relocating the mixin doesn't remove the cycle, it just relocates the dance. Bundle into the same design pass.

### Theme 4: `node()` decorator's closure decomposition
- `decorators.py` lines 274–789, the `decorator(f)` closure inside `node()` (~515 lines: validation → mode-inference → DI classification → in/out inference → `Node(...)` construction → sequential modifier piping incl. Each×Oracle fusion (642–666) and Portal (746–765) → eager shim registration).
- What it needs: deciding a `NodeBuildState`-style per-stage decomposition. Directly touches the same Each/Oracle/Portal piping sequence Phase 8 is extending — do after Phase 8 lands.

### Theme 5: `forward.py`'s DX builder classes
- Lines 531–1352 (`_LoopCall`/`_EachCall`/`_ModifierWrapCall`/`_EnsembleCall`/`_InterruptCall`/`_ForwardSelf`), ~820 lines — the largest single DEFER item in the whole survey.
- What it needs: deciding the module boundary (one file for all builder kinds vs. one per modifier kind) and whether `_ForwardSelf` moves with it. Conceptually the same operation ("wrap a node/list with a modifier, infer boundary ports, name deterministically") Phase 8 is extending at the Construct level — do after Phase 8 lands, once the fusion lowering shape is settled.

### Theme 6: sync/async duplication (named, not a file-split — cross-cutting)
- `factory.py`'s four Portal wrapper-pair functions each repeat the same ~15-line kwarg-forwarding call twice (sync+async), ~60–80 lines of real duplication.
- `runner.py`'s `_prepare`/`_aprepare` are ~60-line near-mirrors sharing every pure helper already.
- `_tool_loop.py`'s `_instantiate_tools`/`_ainstantiate_tools`, `_prepare_tool_loop`/`_aprepare_tool_loop`, `_parse_final_turn`/`_aparse_final_turn` twins.
- These mirror the project's documented async-native "one graph, four verbs" twin convention — NOT flagged as debt to eliminate outright, but worth one shared design question: how much sync/async duplication is architecturally required vs. collapsible via a shared core + thin sync bridge. Scope this as its own investigation, not bundled into any mechanical split.

### Theme 7: Agent Spec bridge in `spec_types.py`
- Lines 238–521 (`_import_agent_spec_property_classes` … `model_to_agent_spec_properties`), ~284 lines, mechanically clean to extract to `_spec_types_agent_spec.py` but deliberately DEFERred because it's Agent-Spec-surface churn during an Agent-Spec-focused epic. Revisit once Phase 8 closes and `_agent_spec.py`/`loader.py` import paths stabilize.

### Theme 8: `lint.py`'s remaining placeholder-check function + true package split
- `_check_template_placeholders` (~130 lines) — entangled with `_walk`'s `known_vars`/`template_resolver`/`di_inputs_enabled` threading; needs the same `gen_api_manifest.py` AST-scan companion-edit as items 22–24 above, times three literal `LintIssue` sites.
- The real fix (not designed here): convert `lint.py` into a `lint/` package and generalize `gen_api_manifest.py`'s hardcoded single-file AST scan into a glob. Touches a script + its guard test outside `lint.py` — its own dedicated pass.

---

## 3. Cross-file duplication (named once)

1. **The Agent-Spec-import-guard pattern is copy-pasted, not shared.**
   `_agent_spec.py:_import_agent_spec_flow_classes` explicitly documents
   itself as copying `spec_types.py:_import_agent_spec_property_classes`'s
   exact shape (fail-loud `ConfigurationError` on a missing `pyagentspec`
   optional import), just for a different class subset. `loader.py`'s own
   `_import_agent_spec_import_classes` docstring says the same thing
   verbatim. Three independent copies of the same try/except/raise
   import-guard idiom. A shared `import_pyagentspec_submodules(*names)`
   helper would collapse all three to one line each — flagged by both the
   `_agent-spec` survey and the cluster-C survey, independently. DEFER (Theme
   7 territory — touches Agent-Spec-surface files mid-epic).

2. **`loader.py` vs `forward.py`: two independent hand-rolled
   implementations of "build me a modifier-wrapped sub-construct from a body
   + inferred port type."** Both files separately implement (a) copy-not-mutate
   `inputs=` inference, (b) deterministic occurrence-slug naming, (c)
   `Construct(input=,output=,nodes=[...]) | Modifier(...)` assembly with
   inferred boundary ports — one driven by Python tracing, one by parsed
   Agent-Spec YAML. Named by the `forward` survey. DEFER — squarely
   `loader.py` territory, blocked on Phase 8 landing.

3. **`decorators.py` vs `loader.py`: the same "translate a DSL spec into
   Oracle/Each/Portal kwargs" concern solved twice** (`_build_oracle_kwargs`/
   `_build_each_kwargs`/`_build_portal_kwargs` vs. `loader.py`'s inline
   kwargs-dict construction at multiple sites, including re-derived
   merge_fn/merge_prompt exclusivity and `n>=2` validation). Not literally
   duplicated code, but the same shape of problem for two different input
   DSLs. Named by the `decorators` survey. DEFER — `loader.py` side is
   epic-active.

4. **Five files share one construct-tree traversal idiom
   (`iter_with_arms(construct)` + switch/match on `primary_shape(item)`)
   with divergent per-shape bodies**: `compiler.py`, `loader.py`,
   `_agent_spec.py`, `_wiring.py`, `runner.py`. `_agent_spec.py` and
   `loader.py` in particular have near-identical `case PrimaryShape.ORACLE:
   ... EACH: ... LOOP: ... BARE: ... PORTAL:` blocks. Named by the `runner`
   survey. Not a mechanical dedup (different callback signatures per site) —
   flagged as a candidate "shape-dispatch visitor" design question for
   whoever eventually owns Theme 2, not proposed as a standalone task.

5. **`ModifierSet.model_post_init`'s hard-coded Portal-exclusion arms
   duplicate `_SLOT_RULES`'s table-driven exclusions** — two independent
   sources of truth for the same five-modifier exclusion matrix, one
   table-driven (`with_modifier` path) one inline (`ModifierSet(...)`
   direct-construction path). The code comment at `modifiers.py:1037-1041`
   already admits this. Named by the `modifiers` survey; folds into Theme 3.

6. **Reciprocal sync/async wrapper-pair duplication** (see Theme 6) is a
   convention, not a defect — named once here so it isn't repeated as a
   "finding" per file (it was independently flagged in the `factory`,
   `runner`, and `cluster-a`/`_tool_loop.py` surveys).

No other cross-file duplication was found; every survey explicitly checked
its assigned file(s) against the six epic-active files and reported clean
in the absence of an item above.

---

## 4. Sequencing recommendation

**Land before Phase 8 (`neograph-s7zt3.11`) starts touching these lines, if
at all possible:**
- Item 1 (`_wiring_oracle_each.py`) — this is literally the cluster Phase 8
  is about to extend. Landing it first means Phase 8's new Construct-level
  code goes into a ~390-line module instead of piling onto a 1461-line file.
  If Phase 8 is already mid-flight on these lines, coordinate a one-time
  rebase instead of skipping the extraction.
- Items 4–6 (`_agent_spec.py` marker/placeholder/node-lowering clusters) —
  same logic: shrinks the file Phase 8's remaining edit surface lives in,
  with confirmed zero overlap.
- Item 18 (`runner.py`'s `_recursion_budget.py`) — flagged with the same
  caveat: it special-cases Portal-mesh-as-opaque-hop costing, conceptually
  adjacent to Phase 8's territory even though it's a different file; land
  first or coordinate a rebase.

**Can land alongside Phase 8 with zero coordination** (confirmed no
overlap by the surveying agent, via `bd show neograph-s7zt3.11` where
checked): items 2–3, 7–17 (the rest of the epic-active-file list) and all
of items 18–39 (every non-epic-active file). These have no shared lines
with Phase 8's diff and can be executed in any order, in parallel, by
different people.

**Wait until after Phase 8 lands:** everything in §2 (Themes 1–8) — these
are the real restructurings, not mechanical moves, and several sit directly
on Phase 8's working set (Theme 2 especially — do not start it until Phase
8 closes and its final shape is known).
