# Factory split — change-axis analysis (2026-05-20)

Ticket: **neograph-w046**. Audits the `neograph-cgkl` split of `factory.py` from 601 → 290 lines into four modules (`factory.py`, `_state_io.py`, `_modifier_io.py`, `_observability.py`).

The cgkl split was *topical*: it grouped functions by what they read/write (state, modifier shapes, logs). That looks like architecture. It is not. Architecture is about *change axes* — the conditions under which we'd modify one thing without rippling through others. This doc redoes the grouping that way and proposes module-by-module diffs.

§ references are to `docs/design/architecture-decisions.md`.

---

## Section 1 — Per-function change-axis analysis

Eleven non-trivial functions live across the four modules (twelve if `_extract_input` and its five shape helpers count separately; they do). One enum class (`InputShape`). One module-level logger.

| Function | Current home | Callers (file:line) | Change scenarios that force a change here | Change-axis cluster |
|---|---|---|---|---|
| `make_node_fn` | `factory.py:133` | `_wiring.py:200,453,576,585`; `compiler.py:378,406,423`; `node.py:309` (deferred); structural guard at `tests/test_structural_guards.py:149` | New execution mode added; new pre-execute hook (e.g., span open); raw-vs-non-raw branching changes; new node-level decorator that needs early validation | **A. Node-wrapper assembly** |
| `_execute_node` | `factory.py:89` | `factory.py:159` only (closure inside `make_node_fn`) | Pre/post-amble step added or reordered (skip → context → render → dispatch → write); observability span added; new modifier injects state-side-effect before dispatch | **B. Per-call lifecycle orchestration** |
| `_make_raw_wrapper` | `factory.py:165` | `factory.py:145` only | Raw escape-hatch contract changes (e.g., async raw, streaming raw); raw observability changes; raw error wrapping added | **A. Node-wrapper assembly** (raw is a sibling of `make_node_fn`'s non-raw path) |
| `make_subgraph_fn` | `factory.py:191` | `_wiring.py:573,582`; `compiler.py:298` | Sub-construct boundary semantics change (e.g., partial input dicts, multi-output ports); loop re-entry shortcut redesign; sub-construct schema fingerprinting; context-field forwarding rule changes | **C. Sub-construct boundary** |
| `_type_name` | `factory.py:77` | `factory.py:106,178` only (two log lines) | Log shape changes (structured fields vs strings); telemetry adopted; type names rendered into traces/spans | **D. Log/observability rendering** |
| `_build_state_update` | `_state_io.py:33` | `factory.py:126` (post-dispatch); `_modifier_io.py:61` (post-skip-with-skip_value); `tests/test_coverage_gaps.py` direct tests | Each modifier wrap rule changes; Loop counter/history schema changes; Oracle fusion writes to new collector fields; new modifier shape that needs per-key state write; dict-form vs single-type output handling | **E. Modifier→state-write rules** |
| `_inject_oracle_config` | `_state_io.py:15` | `factory.py:111`; `factory.py:252` (subgraph) | Oracle config-channel schema changes (more `neo_oracle_*` fields); model-override propagation rule changes; new per-Send config injection | **F. Oracle generator dispatch** |
| `_apply_skip_when` | `_modifier_io.py:23` | `factory.py:114` only | `skip_when` semantics change (e.g., post-dispatch skip); skip_value type contract changes; Loop+skip interaction redesign; Each+skip interaction redesign | **E. Modifier→state-write rules** (writes state on skip; couples to Loop counter and Each key) |
| `InputShape` enum | `_modifier_io.py:74` | `_modifier_io.py:189,201` (dispatch in `_extract_input`); structural guard `test_extract_input_body_is_short` | New input shape added (e.g., explicit `EACH_MULTI_KEY`, `LOOP_HISTORY`); priority order between shapes changes | **G. Input-shape classification** |
| `_classify_input_shape` | `_modifier_io.py:84` | `_modifier_io.py:189` | Same as `InputShape` enum + new modifier shape introduced | **G. Input-shape classification** |
| `_extract_loop_reentry` | `_modifier_io.py:109` | `_modifier_io.py:193` | Loop re-entry semantics change (dict-form, multi-key self-ref, sort/index changes); reducer change from append-list to something else | **H. Per-shape input extraction** |
| `_extract_each_item` | `_modifier_io.py:149` | `_modifier_io.py:195` | `neo_each_item` key renamed; per-item routing changes (e.g., per-item DI propagation) | **H. Per-shape input extraction** |
| `_extract_fan_in_dict` | `_modifier_io.py:154` | `_modifier_io.py:197` | Fan-in dict shape changes; `list[X]` vs `dict[str, X]` consumer rules change; fan-out param representation changes | **H. Per-shape input extraction** |
| `_extract_single_type` | `_modifier_io.py:176` | `_modifier_io.py:199` | `isinstance` scan ordering changes (neograph-np0y); `_unwrap_loop_value`/`_unwrap_each_dict` rules change; state-bus iteration order changes | **H. Per-shape input extraction** |
| `_extract_input` | `_modifier_io.py:187` | `factory.py:112`; `lint.py:381` references in docstring; tests | New input shape added (forces both classifier and dispatch branch); `_extract_input` signature change (rare) | **G. Input-shape classification** (pure dispatch; `InputShape` is its driver) |
| `_extract_context` | `_observability.py:12` | `factory.py:118` only | Context-field representation changes (verbatim strings → rendered objects); new context source added (e.g., from config rather than state); LLM-vs-scripted context rules change | **D. Log/observability rendering** OR **B. Per-call lifecycle orchestration** (interrogated in §4) |

Counts:
- Functions analyzed: **15** (12 functions + 1 enum + `_extract_input` shape-dispatch siblings counted individually + module-level `log`).
- Distinct callers: 8 source files (`_wiring`, `compiler`, `node`, `lint`, `_construct_builder` for `register_scripted`, plus the four target modules themselves) + tests.

---

## Section 2 — Cluster identification

The per-function table yields **eight** distinct change-axis clusters. Several of them straddle the cgkl topical boundaries; that mismatch is the point.

### Cluster A: Node-wrapper assembly

**Functions belonging to it**: `make_node_fn` (factory.py), `_make_raw_wrapper` (factory.py).

**Mismatch from current layout**: None. Both are in `factory.py` already. This is the cluster that *justifies* `factory.py`'s name.

**Hypothesis**: The cgkl name "factory" actually fits exactly *this* cluster. The reason `factory.py` still felt bloated after the cgkl split is that it also contains `_execute_node` (Cluster B) and `make_subgraph_fn` (Cluster C) which assemble different things on different axes.

### Cluster B: Per-call lifecycle orchestration

**Functions belonging to it**: `_execute_node` (factory.py).

**Mismatch from current layout**: `_execute_node` is in `factory.py` but doesn't *build* anything — it *runs* the pipeline (preamble → dispatch → postamble) once per invocation. Its co-tenants in `factory.py` (`make_node_fn`, `_make_raw_wrapper`) are *one-shot factories called at compile time*; `_execute_node` is *called once per node invocation at runtime*.

**Hypothesis**: cgkl kept `_execute_node` in `factory.py` because it's the closure body called by `make_node_fn`'s returned `node_wrapper`. That's a *call-graph* fact, not a change-axis fact. A change to the lifecycle (new step, reorder, observability span) touches `_execute_node` and nothing in `make_node_fn`; a change to wrapper assembly touches `make_node_fn` and nothing in `_execute_node`.

### Cluster C: Sub-construct boundary

**Functions belonging to it**: `make_subgraph_fn` (factory.py).

**Mismatch from current layout**: Sits in `factory.py` next to `make_node_fn` because both are "callable factories." But sub-construct boundary semantics (input-type scan, output-type scan, loop-reentry shortcut, context-field forwarding, isolated `sub_input` dict) is an entirely different problem space from per-node wrapper assembly. A change to "what flows across the sub-construct boundary" (e.g., partial inputs, multi-port outputs, propagation of typed extras) does not touch `make_node_fn` and vice-versa.

**Hypothesis**: Sub-graphs and nodes both get "made into a Callable that LangGraph adds to the StateGraph," so they live together. That's a *signature similarity*, not a change-axis similarity. The sub-construct boundary is closely related to `Construct.input`/`Construct.output` semantics (described in CLAUDE.md as a deliberate plural-vs-singular split) — those are IR concepts, and `make_subgraph_fn` is their runtime counterpart.

### Cluster D: Log/observability rendering

**Functions belonging to it**: `_type_name` (factory.py), `_extract_context` (_observability.py).

**Mismatch from current layout**: `_type_name` lives in `factory.py` solely because `_execute_node` and `_make_raw_wrapper` log it. `_extract_context` lives alone in `_observability.py`. Both are observability-adjacent helpers — but `_extract_context` is misnamed: it's not "observability," it's "context fields for LLM prompts." The 24-line `_observability.py` is a stub that has no observability content.

**Hypothesis**: The cgkl split named a module after a topic ("observability") and put the only handy candidate (`_extract_context`) inside. The naming masked the function's actual responsibility (prompt-context extraction, which is a sibling of input rendering, not of logging).

### Cluster E: Modifier→state-write rules

**Functions belonging to it**: `_build_state_update` (_state_io.py), `_apply_skip_when` (_modifier_io.py).

**Mismatch from current layout**: These two share a deep coupling — `_apply_skip_when` calls `_build_state_update` when `skip_value` fires (line 61 of `_modifier_io.py`), and both encode the same modifier rules: Each-wrapping by key, Loop counter increment, dict-form fan-out. They're *the same change axis* split across two files because cgkl grouped one as "writing state" and the other as "modifier-aware input." The shared rule set ("when X modifier is present, the state write looks like Y") changes them in lockstep.

**Hypothesis**: cgkl saw "this one extracts input, that one writes output" and split by direction (in vs out). But the modifier rules don't care about direction — they care about *which modifier is on the node*. The Each-key extraction in `_build_state_update` (line 79) and the Each-aware skip dispatch in `_apply_skip_when` are two halves of one rule. A new modifier shape forces both to change.

### Cluster F: Oracle generator dispatch

**Functions belonging to it**: `_inject_oracle_config` (_state_io.py), plus all of `_oracle.py` (out of cgkl scope but in this audit's frame).

**Mismatch from current layout**: `_inject_oracle_config` reads `neo_oracle_gen_id` / `neo_oracle_model` from state and injects into `config['configurable']`. Every reader of those fields lives in `_oracle.py` (or in the Oracle wiring in `_wiring.py`). The injection function is the *opposite half* of Oracle config propagation, and it's stranded in `_state_io.py`.

**Hypothesis**: cgkl saw `_inject_oracle_config` reading from state and grouped it with `_build_state_update` ("anything that touches state goes in `_state_io`"). But `_inject_oracle_config` doesn't *write* state — it *reads* Oracle plumbing fields and transforms config. Its change-axis cluster is "Oracle generator dispatch," which is already mostly in `_oracle.py`.

### Cluster G: Input-shape classification

**Functions belonging to it**: `InputShape` enum, `_classify_input_shape`, `_extract_input` (all in _modifier_io.py).

**Mismatch from current layout**: These three form a tight unit (enum, classifier, dispatcher). Their current home `_modifier_io.py` is correct *for them* — but the file also contains `_apply_skip_when` (Cluster E) and the shape extractors (Cluster H), three change axes under one roof.

**Hypothesis**: cgkl extracted "modifier-aware input" as a coherent topic. The topic *is* coherent at the level of "stuff that knows about modifier shapes on the read side," but it bundles three independent change axes: shape classification, per-shape extraction, and skip-side-effect handling.

### Cluster H: Per-shape input extraction

**Functions belonging to it**: `_extract_loop_reentry`, `_extract_each_item`, `_extract_fan_in_dict`, `_extract_single_type` (all in _modifier_io.py).

**Mismatch from current layout**: These are the four leaves of the `_classify_input_shape` dispatch. Each one encodes its own shape rule. A change to "Loop re-entry with dict-form outputs" touches only `_extract_loop_reentry`. A change to "fan-in dict element-type unwrapping" touches only `_extract_fan_in_dict`. They are independent leaves on the same dispatch.

**Hypothesis**: They cohabit with the classifier because the dispatch is short and reads like a unit. That's fine *if* you also accept that the file groups three change axes (G, H, E). Splitting H from G is possible but probably not a net win — the leaves are <20 lines each and never change without the classifier also changing (a new shape implies both a new enum value and a new extractor). G+H together is a real cluster; G+H+E in one file is the smell.

---

## Section 3 — Proposed module layout (architectural, not topical)

The eight clusters fold down to **five** modules — A, C and F absorb their siblings cleanly, and the (G, H) pair is one module by construction.

### Module: `factory.py` (Cluster A only)

**Single responsibility statement**: Builds the per-node Callable that LangGraph will add to the StateGraph; one path for ordinary modes, one for the raw escape-hatch.

**Functions inside**: `make_node_fn`, `_make_raw_wrapper`.

**Imports it consumes**: `_dispatch._dispatch_for_mode`, `_execute._execute_node` (new module — see below), `_registry.registry`, `naming.field_name_for`, `node.Node`, `errors.ConfigurationError`.

**Imports it exposes**: `make_node_fn`. (Re-export `make_subgraph_fn` from the new sub-construct module if needed for backward-compat; see Cluster C.)

**Future-change scenarios it absorbs**:
- New execution mode that requires a different *wrapper* (e.g., streaming would change `node_wrapper`'s contract).
- New early validation step at compile time (`scripted_fn` registration check; future LLM-config validation).
- Raw escape-hatch becomes async — touches `_make_raw_wrapper` but nothing in `_execute_node`.

### Module: `_execute.py` (new, Cluster B)

**Single responsibility statement**: Runs one node invocation through the standard preamble → dispatch → postamble lifecycle.

**Functions inside**: `_execute_node`.

**Imports it consumes**: `_dispatch.ModeDispatch, NodeInput`, `_modifier_io._apply_skip_when, _extract_input`, `_normalize.normalize_inputs`, `_observability_or_logs._extract_context` (rename pending), `_state_bus.adapt_state`, `_state_io._build_state_update`, `_oracle_config._inject_oracle_config` (renamed), `naming.field_name_for`, `node.Node`.

**Imports it exposes**: `_execute_node` (private; only `factory.make_node_fn` calls it).

**Future-change scenarios it absorbs**:
- New lifecycle step (e.g., open OTel span, increment cost meter, fire pre/post hooks).
- Reordering of preamble/postamble.
- Centralized error wrapping for non-raw modes.

### Module: `_subconstruct.py` (new, Cluster C)

**Single responsibility statement**: Builds the per-sub-construct Callable that LangGraph will add to the parent StateGraph; encodes the sub-construct's input/output boundary semantics.

**Functions inside**: `make_subgraph_fn`, plus (eventually) the input-by-type scan helper and the output-by-type scan helper as named private functions instead of inline closure code (they're currently 50 lines of inline logic — first refactor target if this module is created).

**Imports it consumes**: `_state_bus.adapt_state`, `_oracle_config._inject_oracle_config`, `naming.field_name_for`, `errors.ExecutionError`, `modifiers.classify_modifiers, ModifierCombo`, `construct.Construct`, plus a deferred `runner._strip_internals`.

**Imports it exposes**: `make_subgraph_fn`.

**Future-change scenarios it absorbs**:
- Sub-construct partial input dict (currently single-type by isinstance scan).
- Multi-port sub-construct outputs (currently single `sub.output`).
- Sub-construct schema fingerprinting (CLAUDE.md §checkpoint-resume).
- Loop re-entry shortcut redesign.
- Context-field forwarding from parent → sub.

### Module: `_state_write.py` (renames `_state_io.py`, Cluster E + part of G's writes)

**Single responsibility statement**: All rules for transforming a node's output into a LangGraph state-update dict, including modifier-aware key wrapping (Each, Loop, Oracle-fusion) and dict-form output projection.

**Functions inside**: `_build_state_update` (kept), `_apply_skip_when` (**moved** from `_modifier_io.py`).

**Imports it consumes**: `_normalize.normalize_outputs`, `_state_bus.StateBus`, `modifiers.Each, ModifierCombo, classify_modifiers`, `naming.field_name_for`, `node.Node`, `errors.ExecutionError`, `di._isinstance_safe`.

**Imports it exposes**: `_build_state_update`, `_apply_skip_when`.

**Future-change scenarios it absorbs**:
- New modifier shape that reshapes state writes (the same axis `state.py:_add_output_field` and `_construct_validation.effective_producer_type` already encode for type-side rules).
- Loop history schema change.
- Each-key extraction rule change.
- Dict-form fan-out rule change.

**Note on renaming**: `_state_io.py` is misleading — it implies "I/O," but it only handles writes (and the Oracle-config injection, which is Cluster F and leaves with that). `_state_write.py` mirrors `_state_bus.py` (read interface) and pairs naturally.

### Module: `_input_shape.py` (renames `_modifier_io.py` minus the moves, Clusters G + H)

**Single responsibility statement**: Classifies a node's input shape against state and extracts the typed input via the matching shape extractor; pure read-side, no state writes.

**Functions inside**: `InputShape` enum, `_classify_input_shape`, `_extract_input`, `_extract_loop_reentry`, `_extract_each_item`, `_extract_fan_in_dict`, `_extract_single_type`.

**Imports it consumes**: `_normalize.normalize_inputs, normalize_outputs`, `_state_bus.StateBus`, `di._isinstance_safe, _unwrap_each_dict, _unwrap_loop_value`, `modifiers.ModifierCombo, classify_modifiers`, `naming.field_name_for`, `node.Node`.

**Imports it exposes**: `_extract_input`, `InputShape` (the latter only for the structural guard at `tests/test_structural_guards.py:735`).

**Future-change scenarios it absorbs**:
- New input shape (Loop-history, dict-fan-in-with-default, sparse-fan-in).
- Shape priority reorder.
- Per-shape extraction rule change.

**Note**: keeping G+H together is deliberate — a new shape always implies a new enum value and a new extractor function, and the classifier's match statement enumerates them. Splitting them buys nothing.

### Module: `_oracle_config.py` (new, Cluster F merge target — folds into `_oracle.py`)

**Single responsibility statement**: Oracle generator dispatch — propagates per-Send config (gen_id, model override) through state and config plumbing.

**Functions inside**: `_inject_oracle_config` (**moved** from `_state_io.py`).

**Recommendation**: Fold this function into the existing `_oracle.py` rather than creating a third file for one function. `_oracle.py` already houses `make_oracle_redirect_fn`, `make_oracle_merge_fn`, `_unwrap_oracle_results`, etc. — adding `_inject_oracle_config` makes `_oracle.py` the single source of truth for Oracle plumbing.

**Future-change scenarios it absorbs**:
- New `neo_oracle_*` field (e.g., a `neo_oracle_temperature` per-Send override).
- Oracle generator dispatch redesign (e.g., per-generator tool budget overrides).

### Module: `_observability.py` — verdict pending in §4

The 24-line file holds only `_extract_context`, which is *not* observability — it's prompt-context extraction for LLM nodes. Two real options, decided in §4:
- (a) Rename to `_prompt_context.py` and accept that the file holds one short function.
- (b) Fold `_extract_context` into `_execute.py` (Cluster B) where its sole caller lives, and delete the module.

`_type_name` is also tangentially in Cluster D — but its only role is rendering type names into log lines, called twice. Likely belongs as a private helper *inside* `_execute.py` (one of its callers) or as a method on a future structlog binding.

---

## Section 4 — Diff from current state

The interrogations the maintainer asked for are answered first; the per-function verdict table follows.

### Q1. Does `_execute_node` belong in `_dispatch.py`?

**No.** `_dispatch.py` houses the `ModeDispatch` Protocol and its three concrete implementations (`ScriptedDispatch`, `ThinkDispatch`, `ToolDispatch`). Those are *strategies* injected into a lifecycle. `_execute_node` is the *lifecycle driver* that injects them. They depend on each other but encode different change axes: a new dispatch strategy (e.g., `StreamingDispatch`) touches `_dispatch.py` and not `_execute_node`; a new lifecycle step (e.g., span open) touches `_execute_node` and not any dispatch.

Putting `_execute_node` in `_dispatch.py` would make `_dispatch.py` simultaneously hold "the strategies" and "the place strategies are called from" — the same anti-pattern cgkl was trying to undo. The right home is a new `_execute.py` (Cluster B).

Specifically actionable: **move `_execute_node` from `factory.py:89-130` to `src/neograph/_execute.py` as the only public function. `factory.py:159` becomes `from neograph._execute import _execute_node`.**

### Q2. Is `_observability.py` (24L, one function) a stub or a real namespace?

**Stub.** It does not hold "observability" — it holds prompt-context extraction. `_extract_context` reads `node.context` field names from state and returns them as `dict[str, str]` for LLM prompts. The only caller is `_execute_node` at `factory.py:118`.

Three options, with strong preference for option 2:
1. **Rename** to `_prompt_context.py`. Honest but leaves a 24-line file.
2. **Fold into `_execute.py`** as a private helper. Single caller, single line of import disappears. This is the recommendation.
3. **Promote to a real observability module** by absorbing future OTel/structlog spans, cost callbacks, etc. Defensible only if those spans are scheduled work — they aren't on the current roadmap.

If the maintainer plans OTel ingestion in the near term, option 3 wins and the rename to `_telemetry.py` or `_obs.py` becomes natural; documenting the deferred work is enough.

Specifically actionable for option 2: **move `_extract_context` into `_execute.py` as a private function. Delete `_observability.py`.**

### Q3. Should `_modifier_io.py` split into `_skip.py` + `_input_extract.py`?

**Partially.** The actual split is **`_apply_skip_when` leaves for `_state_write.py`** (Cluster E), and the rest of `_modifier_io.py` becomes `_input_shape.py` (Clusters G + H). There is no need for a standalone `_skip.py` — `_apply_skip_when` is one half of the modifier→state-write rule cluster, paired with `_build_state_update`. Their shared coupling is at `_modifier_io.py:61`, where `_apply_skip_when` calls `_build_state_update` with the `skip_value` result.

Splitting them apart (the current cgkl layout) means the shared modifier rule set (Each-key wrapping, Loop counter increment) is encoded in two files and must change in lockstep. The §4 architecture-decision principle "Each function does one job. Multi-axis dispatch splits before it grows" applies *between* files too — multi-axis split should not splay one axis across files.

Specifically actionable: **move `_apply_skip_when` from `_modifier_io.py:23-71` to `_state_io.py` (or its renamed successor `_state_write.py`). Rename `_modifier_io.py` to `_input_shape.py`.**

### Q4. Is `_state_io.py` correctly named?

**No.** It's "state writes." There is no read in this module — every read goes through `_state_bus.py`. The "I/O" framing makes the name plausible at a glance but obscures the actual responsibility, which is "rules for writing state updates from node outputs." The symmetric pair to `_state_bus.py` (read interface) would be `_state_write.py` (write rules) — that mirroring documents itself.

There's also a free rider: `_inject_oracle_config` is not state I/O at all; it reads two state keys and writes to `config['configurable']`. It belongs in `_oracle.py` (Cluster F).

Specifically actionable: **rename `_state_io.py` → `_state_write.py`. Move `_inject_oracle_config` from `_state_io.py:15-30` to `_oracle.py` as a sibling of the redirect/merge factories.**

### Q5. Where should `_type_name` live?

**Inside `_execute.py`** (its post-move home) as a private function. `_type_name` is called twice (at `factory.py:106` in `_execute_node`'s log line, at `factory.py:178` in `_make_raw_wrapper`'s log line). If `_execute_node` moves to `_execute.py` (Cluster B), one of `_type_name`'s call sites moves with it; the other is in `_make_raw_wrapper`, which stays in `factory.py`.

The lowest-friction layout: **`_type_name` lives in `_execute.py`, and `_make_raw_wrapper` imports it.** This is fine — `_make_raw_wrapper` is the raw escape-hatch, and importing one rendering helper from the execute module is cheaper than a third file. Alternatively, both wrappers' log lines could move to a tiny `_log_fields.py` if more such helpers appear; right now `_type_name` is the only candidate.

Specifically actionable: **move `_type_name` from `factory.py:77-86` to `_execute.py` as a private function. `_make_raw_wrapper` keeps its `from neograph._execute import _type_name` import (one line).**

### Per-function verdict table

| Function | Current home | Verdict | Rationale (cite §1 cluster) |
|---|---|---|---|
| `make_node_fn` | factory.py | **stay-in-current-home** | Cluster A; `factory.py`'s name and contents align exactly. |
| `_make_raw_wrapper` | factory.py | **stay-in-current-home** | Cluster A; raw is a sibling of the non-raw assembly. |
| `_execute_node` | factory.py | **move-to-`_execute.py`** | Cluster B; lifecycle orchestration is a separate axis from wrapper assembly (Q1). |
| `make_subgraph_fn` | factory.py | **move-to-`_subconstruct.py`** | Cluster C; sub-construct boundary semantics are unrelated to per-node wrapper assembly. |
| `_type_name` | factory.py | **move-to-`_execute.py`** | Cluster D; one caller moves with `_execute_node`; the other (`_make_raw_wrapper`) imports back (Q5). |
| `_build_state_update` | _state_io.py | **rename-current-home** to `_state_write.py` | Cluster E; module name doesn't match the function's actual responsibility (Q4). |
| `_inject_oracle_config` | _state_io.py | **move-to-`_oracle.py`** | Cluster F; Oracle plumbing reads state but writes config — it's not "state I/O" (Q4). |
| `_apply_skip_when` | _modifier_io.py | **move-to-`_state_write.py`** | Cluster E; tightly coupled to `_build_state_update` via `skip_value` dispatch (Q3). |
| `InputShape` enum | _modifier_io.py | **rename-current-home** to `_input_shape.py` | Cluster G; module name should reflect the read-side dispatch responsibility. |
| `_classify_input_shape` | _modifier_io.py | **rename-current-home** to `_input_shape.py` | Cluster G. |
| `_extract_loop_reentry` | _modifier_io.py | **rename-current-home** to `_input_shape.py` | Cluster H; stays with classifier. |
| `_extract_each_item` | _modifier_io.py | **rename-current-home** to `_input_shape.py` | Cluster H. |
| `_extract_fan_in_dict` | _modifier_io.py | **rename-current-home** to `_input_shape.py` | Cluster H. |
| `_extract_single_type` | _modifier_io.py | **rename-current-home** to `_input_shape.py` | Cluster H. |
| `_extract_input` | _modifier_io.py | **rename-current-home** to `_input_shape.py` | Cluster G (pure dispatch). |
| `_extract_context` | _observability.py | **move-to-`_execute.py`** | Cluster D's misnamed inhabitant; sole caller is `_execute_node` (Q2). |

Net: **2 stay**, **5 move**, **8 rename-current-home** (the renames are: `_state_io.py` → `_state_write.py`, `_modifier_io.py` → `_input_shape.py`). One module is deleted (`_observability.py`), one module is created (`_execute.py`), one new module is optional (`_subconstruct.py` — see §6).

---

## Section 5 — Architectural guard proposals

The current guard (`TestFactoryResponsibilityDiscipline`) tests *line count* and *name whitelist* in `factory.py`. Both are stand-ins, not architecture. A name whitelist tells you "we promised to keep these names here" — but if the wrong names happen to match the whitelist for the wrong reasons, the guard passes and architecture has drifted. Examples of stronger guards follow, one per proposed module.

### Guard 1 — Import-graph layering for the assembly cluster

**For**: `factory.py`, `_execute.py`, `_subconstruct.py`, `_oracle.py`, `_state_write.py`, `_input_shape.py`.

**Rule**: Define a fixed DAG of allowed imports between these modules. Walk the AST imports of each file at test time and assert no edge violates the DAG.

**The DAG**:
```
factory.py     ─► _execute.py ─► _state_write.py ─► (leaves)
                              ─► _input_shape.py ─► (leaves)
                              ─► _oracle.py
                              ─► _dispatch.py
_subconstruct  ─► _oracle.py  ─► (leaves)
                              ─► _state_write.py
```

**Test sketch**:
```python
ALLOWED_EDGES = {
    "factory.py": {"_execute.py", "_dispatch.py", "_registry.py", ...},
    "_execute.py": {"_state_write.py", "_input_shape.py", "_oracle.py", "_dispatch.py", "_state_bus.py", "_normalize.py", ...},
    "_state_write.py": {"_normalize.py", "_state_bus.py", "di.py", "modifiers.py", "naming.py", "node.py", "errors.py"},
    "_input_shape.py": {"_normalize.py", "_state_bus.py", "di.py", "modifiers.py", "naming.py", "node.py"},
    ...
}
for mod, allowed in ALLOWED_EDGES.items():
    actual = parse_imports(SRC_DIR / mod)
    extras = actual - allowed
    assert not extras, f"{mod} imports from outside its layer: {extras}"
```

**Why this is stronger than the current name-whitelist**: A drift like "I added a helper called `_execute_node_v2` to factory.py" would still pass the whitelist if the test was loosened to allow new names. The import-graph test fires when factory.py starts to depend on `_state_write.py` directly (skipping `_execute.py`), which is the *real* architectural drift.

### Guard 2 — Scenario walkthrough: "add a new execution mode"

**For**: `factory.py` + `_execute.py` + `_dispatch.py`.

**Rule**: Add a parametrized test whose docstring enumerates expected files for each architectural change scenario. The test stays in CI as a *documentation contract*; when the maintainer adds a new mode and the wrong file list shows up in the diff, code review catches it.

**Test sketch**:
```python
ARCHITECTURE_SCENARIOS = {
    "add_new_execution_mode": {
        "must_touch": ["_dispatch.py"],
        "may_touch": ["factory.py"],  # for early validation
        "must_not_touch": ["_state_write.py", "_input_shape.py", "_oracle.py"],
    },
    "add_new_modifier_shape": {
        "must_touch": ["_state_write.py", "_input_shape.py", "modifiers.py"],
        "must_not_touch": ["_dispatch.py", "factory.py"],
    },
    "add_new_oracle_config_field": {
        "must_touch": ["_oracle.py"],
        "must_not_touch": ["_state_write.py", "_input_shape.py"],
    },
}
```

The test itself is documentation; it doesn't assert at CI time. But a stricter version could: parse the last N commits, find ones whose message matches "add mode", and assert the actual file diff was a subset of `must_touch ∪ may_touch`. This is heavier; treat as optional.

### Guard 3 — Cohesion metric: callers per module

**For**: every module in the proposed layout.

**Rule**: For each module, count the number of external modules that import from it. Assert the count is below a threshold. High fan-out means low cohesion (the module is doing too much).

**Test sketch**:
```python
EXPECTED_FAN_OUT = {
    "factory.py":       3,  # compiler, _wiring, __init__ (re-exports)
    "_execute.py":      1,  # only factory imports it
    "_subconstruct.py": 2,  # compiler, _wiring
    "_state_write.py":  2,  # _execute, _state_io tests
    "_input_shape.py":  1,  # _execute
    "_oracle.py":       3,  # _wiring, _execute, compiler
}
for mod, expected_max in EXPECTED_FAN_OUT.items():
    importers = count_modules_importing(mod)
    assert importers <= expected_max, f"{mod} is imported by {importers} modules (max {expected_max})"
```

**Why this is stronger than line count**: it catches the case where a module becomes a "kitchen sink" that everyone imports from, even if it stays under 300 lines. Today's `factory.py` fails this naturally because of all the noqa re-exports — the test would force us to either *commit to those re-exports as the contract* or *prune them*.

### Guard 4 — Co-change witness: `_apply_skip_when` ↔ `_build_state_update`

**For**: `_state_write.py`.

**Rule**: Specifically test that the two state-write functions live in the same file. This guard *encodes* the Cluster E hypothesis: if someone later splits them again, the guard fires and a re-evaluation must happen.

**Test sketch**:
```python
def test_skip_when_and_state_update_co_located():
    """The Each-key/Loop-counter rules are encoded in both functions.
    A change to one almost always changes the other (Cluster E).
    """
    state_write = (SRC_DIR / "_state_write.py").read_text()
    assert "def _build_state_update" in state_write
    assert "def _apply_skip_when" in state_write
```

This is a one-off guard, narrow on purpose. It's the same flavor as `TestDeadCodeRemoval` (assert specific names stay/leave specific files).

### Guard 5 — `InputShape` exhaustiveness

**For**: `_input_shape.py`.

**Rule**: For every variant of the `InputShape` enum, assert a matching extractor function exists in the module. The existing guard at `tests/test_structural_guards.py:705` already targets `_extract_input` body length; extend it to enforce the dispatch surface.

**Test sketch**:
```python
def test_input_shape_extractor_per_variant():
    from neograph._input_shape import InputShape
    src = (SRC_DIR / "_input_shape.py").read_text()
    for variant in InputShape:
        if variant == InputShape.NONE:
            continue
        expected_fn = f"_extract_{variant.value}"
        assert f"def {expected_fn}" in src, f"InputShape.{variant.name} has no extractor function"
```

**Why this is stronger than the current `test_extract_input_body_is_short`**: line count won't catch a missing branch in the match statement; this does. Combined with the existing exhaustive `assert_never`, it makes the dispatch table self-documenting.

### Guard 6 — Sub-construct boundary owns input-by-type scan

**For**: `_subconstruct.py`.

**Rule**: The "scan parent state in reverse for first value matching `sub.input`" logic appears only in this module. If someone re-implements the same scan in `factory.py` or `_execute.py`, fire.

**Test sketch**:
```python
def test_input_type_scan_only_in_subconstruct():
    """The reverse-state isinstance scan is sub-construct boundary logic.
    Re-implementations elsewhere are a code smell.
    """
    needle = "for attr_name in reversed(bus.keys())"
    occurrences = []
    for py in SRC_DIR.glob("*.py"):
        if needle in py.read_text():
            occurrences.append(py.name)
    assert occurrences == ["_subconstruct.py"], (
        f"Reverse state scan found in: {occurrences}. "
        "Sub-construct boundary semantics belong in _subconstruct.py only."
    )
```

This is a witness-string test — fragile if the implementation changes phrasing. The maintainer's preference (per the precedent in `tests/test_structural_guards.py`) is to accept fragile-but-honest tests over no test at all.

---

## Section 6 — Acknowledged limits

Each of the following is a real constraint the proposed layout has to honor.

### Limit 1 — `make_subgraph_fn` lives where `_wiring.py` and `compiler.py` import from

`make_subgraph_fn` is imported by `_wiring.py:573,582` and `compiler.py:298`. Moving it to a new `_subconstruct.py` is mechanically straightforward, but the import sites must update. There is no circularity blocking the move — `_wiring.py` already imports from `factory.py` and would simply add a new import line. **No real blocker; just churn.** The maintainer should weigh "one more module" vs "one large factory.py."

### Limit 2 — `_inject_oracle_config` move to `_oracle.py` does not introduce a cycle

`_oracle.py` currently imports from `_runtime_registry.py`, `_llm_config.py`, `_state_bus.py`, `errors`, `modifiers`, `naming`, `node`. `_inject_oracle_config` needs `_state_bus.StateBus` and `langchain_core.runnables.RunnableConfig` — both already present in `_oracle.py`. **No cycle.** This move is safe.

### Limit 3 — `_apply_skip_when` move to `_state_write.py` requires care around `classify_modifiers`

`_apply_skip_when` calls `classify_modifiers(node)` for the Loop counter increment when no `skip_value` is provided (lines 65-71). `_build_state_update` also calls `classify_modifiers(node)`. Both already import from `modifiers`; co-locating them in `_state_write.py` keeps the import surface simple and consolidates the modifier-rules use site. **No blocker.**

### Limit 4 — `_extract_context` move to `_execute.py` is unconditional

The function has exactly one caller (`_execute_node` at `factory.py:118`). Moving it as a private helper inside `_execute.py` removes one file and one cross-module import. **No real blocker.**

The only reason to *not* move it: if observability/telemetry work (OTel spans, structlog binding, cost meters) is scheduled and would naturally land in an `_observability.py` namespace, keeping the stub avoids a churn-now-churn-later sequence. **If the maintainer's plan has no OTel landing in the next two minor releases, delete the module.** If OTel is imminent, rename the file to `_telemetry.py` and absorb `_extract_context` into it as one of several helpers.

### Limit 5 — Tests import from `factory.py` directly

Several tests import private helpers via `from neograph.factory import _build_state_update` and `_extract_input` (e.g., `tests/test_coverage_gaps.py:797,805,817,842,941`). The current `noqa: F401` re-exports in `factory.py` exist to keep these imports working post-cgkl-split. **Any further move continues to require those re-exports unless tests are updated.** This is the same backward-compat tail the cgkl split incurred — extend it once more, or update the test imports as part of the move.

### Limit 6 — `_execute.py` would import from many siblings

After the proposed moves, `_execute.py` would import from `_dispatch.py`, `_input_shape.py`, `_state_write.py`, `_oracle.py`, `_observability.py` (or wherever `_extract_context` lands), `_normalize.py`, `_state_bus.py`, plus `naming`, `node`, `errors`. That's eight neograph-internal imports for one function — but the function *is* the orchestrator; high fan-in to a single coordinator is the correct shape. **Not a blocker; flag for review if the import count crosses ~12.**

### Limit 7 — `tests/test_structural_guards.py:TestFactoryResponsibilityDiscipline` becomes stale

The current `FACTORY_DEFINITION_WHITELIST` and `FACTORY_LINE_BUDGET` are designed for the cgkl-era layout. Adopting the proposed layout requires rewriting this guard class (the new guards in §5 replace it). **This is intended, not a blocker.** The maintainer's framing rejects the line-budget + name-whitelist guard as "not architecture"; the §5 guards are the replacement.

---

## Summary

- **15 functions / enums analyzed** across `factory.py`, `_state_io.py`, `_modifier_io.py`, `_observability.py`, plus the adjacent `_dispatch.py` and `_oracle.py` for context.
- **8 change-axis clusters identified** (A–H). Five of them straddle the cgkl topical boundaries; three coincide with current files (cluster A = `factory.py`, cluster G+H = `_modifier_io.py`'s read side, cluster F is already mostly in `_oracle.py`).
- **2 stay-in-current-home**, **5 move**, **8 rename-current-home** (the two renames cover all 8 functions in those two files). 1 module deleted (`_observability.py`), 1 created (`_execute.py`), 1 optional (`_subconstruct.py`).
- **Real architectural problems found** (not just naming):
  - `_apply_skip_when` and `_build_state_update` are one change axis split across two files (Cluster E). This is the strongest finding — the modifier rules are duplicated by virtue of being co-located with sibling functions instead of with their co-changing partner.
  - `_inject_oracle_config` is stranded in `_state_io.py` despite being part of Oracle dispatch (Cluster F). It reads state but doesn't write state; the cgkl naming hid the mismatch.
  - `_observability.py` is a misnamed stub. Its contents (`_extract_context`) is not observability; it's prompt-context extraction. Either rename and own the future-OTel space, or absorb into `_execute.py`.
  - `factory.py` still holds two unrelated factories (`make_node_fn` and `make_subgraph_fn`) and one orchestrator (`_execute_node`). The "factory" name is right for *one* of these; the others are squatting.
- **Naming-only issues** (vs real architectural problems):
  - `_state_io.py` should be `_state_write.py` (mirrors `_state_bus.py` for reads).
  - `_modifier_io.py` should be `_input_shape.py` after `_apply_skip_when` leaves.

The cgkl split shrank one file from 601 to 290 lines, which is real. The structural problem it *did not solve* is that the four resulting modules don't correspond to change axes — three of them mix axes, and one (`_observability.py`) is a stub that masks its only function's real responsibility.
