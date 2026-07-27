# Fresh sweep: every consumer of ModifierCombo / classify_modifiers / modifier_set / get_modifier / has_modifier

Method: `grep -rn "ModifierCombo\|classify_modifiers\|modifier_set\|get_modifier\|has_modifier" src/neograph/` (whole tree, no directory excluded), then every hit read in context and classified as:

- **DECOMPOSITION/DISPATCH** — genuinely re-derives "what does this combo mean" to decide what to build/lower/wrap (duplicated-source-of-truth risk).
- **PRESENCE** — narrow single-modifier `.oracle`/`.each`/`.loop`/`.portal`/`.operator is not None` check for an unrelated purpose (validation, DI, topology, logging, test-gen).
- **DEFINITION** — `modifiers.py` itself, or a class declaring the `modifier_set` field / inheriting `has_modifier`/`get_modifier` (not a "consumer").

## Result vs. the existing 9-module list

The prior inventory (compiler.py, _agent_spec.py, loader.py, state.py, _state_write.py, _subconstruct.py, _input_shape.py, runner.py, _wiring.py) is **incomplete**. My independent sweep confirms all 9 are genuine (mostly) but finds **one more genuine DECOMPOSITION/DISPATCH consumer it missed: `_fan_agent.py`**. It also surfaces `loader.py`'s entry as needing a caveat (see below), and one borderline test-generation consumer (`testing/scaffold.py`) worth naming even though it doesn't affect runtime lowering.

## DECOMPOSITION/DISPATCH consumers (genuine duplicated-source-of-truth)

| # | File:line | What it decides |
|---|---|---|
| 1 | `compiler.py:506-552` (`_lower_arm`/similar) — `match combo:` over ORACLE/EACH/LOOP/BARE/PORTAL(+OPERATOR variants) | Which LangGraph sub-wiring to build per combo, inside branch arms |
| 2 | `compiler.py:584-665` — second `match combo:` block, same combo set | Top-level per-node LangGraph wiring dispatch (the reference/ground-truth lowering) |
| 3 | `compiler.py:181-192,254-258` | `classify_modifiers` narrow uses (operator presence, Portal grab) — PRESENCE, not full dispatch (listed here only to disambiguate from #1/#2 in the same file) |
| 4 | `state.py:164-202`, `538-572`, `595-635` — three separate `match combo:` blocks | State-model field construction per combo (Each dict-wrap, Oracle single-field, Portal passthrough, etc.) — independently re-derives the same combo→shape rule 3x in one file |
| 5 | `_state_write.py:72-92` — `match combo:` | Which state key(s) a node's output write targets, per combo |
| 6 | `_subconstruct.py:89-91` | Derives `has_loop`/`has_each` booleans from `classify_modifiers` combo to decide sub-construct wiring shape |
| 7 | `_input_shape.py:32-33` | Combo-based decision of whether the node's *input* shape is Loop-shaped |
| 8 | `runner.py:104-154` | Combo-based decisions (`PORTAL`/`PORTAL_OPERATOR`) for run-time entry/dispatch routing |
| 9 | `_wiring.py:713-997` (7 call sites) | Combo/`.modifier_set.portal`/`.operator` based recursive graph-wiring decisions — heaviest single-file consumer |
| 10 | `_agent_spec.py:844-877` — `if combo == ModifierCombo.X:` chain (BARE/ORACLE/EACH/LOOP/OPERATOR), raises for anything else (e.g. composed combos) | Independently re-derives the SAME combo→shape rule as compiler.py/state.py, but only for Agent Spec lowering, and admits it doesn't cover composed modifiers ("out of scope for i3zsh's primitive-level export") — this is the flagged "wrong non-support" pattern: Portal itself is special-cased OUTSIDE this dispatch (`to_agent_spec`, lines 948-967) via direct `modifier_set.portal` presence, not through `classify_modifiers`, so Portal combos never even reach this raise — an asymmetry with compiler.py, where Portal IS a case inside the unified `match combo:` |
| 11 | **`_fan_agent.py:148-171`** (`is_supported_fan_over_agent` / `raise_if_unsupported_fan_over_agent`) — **NOT in the prior 9-module list** | Uses `classify_modifiers(item)` to decide, for agent/act-mode nodes, whether a fan modifier (Oracle/Each/Loop) is one of the shapes the auto-wrap machinery supports — a genuine "what does this combo mean for building purposes" decision, structurally identical in kind to compiler.py's/state.py's dispatch, just scoped to the agent-auto-wrap sub-problem. This is the concrete miss my from-scratch sweep found. |

`loader.py` **caveat**: it is correctly on the prior list as a member of the "duplicated semantics" problem, but it does **not** call `classify_modifiers`/`ModifierCombo`/`modifier_set` at all (confirmed: zero hits for those exact symbols). Instead it re-derives combo semantics in the reverse (import) direction by pattern-matching exported pyagentspec structural markers (`neograph/group_id`, `neograph/each_spec`, Swarm-vs-Flow shape, etc.) and reconstructing via direct modifier application (`node | Oracle(...)`, `node | Each(...)`, etc., `loader.py:474,518,535,556,724,1028-1042`). It is a duplicated-source-of-truth site for the SAME underlying rule set, but via a structurally different mechanism (marker-driven reconstruction, not `classify_modifiers` dispatch) — worth stating precisely rather than implying it shares the same grep signature as the others.

## PRESENCE-only consumers (narrow, not combo dispatch — do not conflate)

| File:line | Purpose |
|---|---|
| `_validation_inputs.py:55-56,92-95,138-139` | Fan-in validation: checks `.loop`/`.each` presence to pick a validation branch (self-reference-without-Loop check; Each path-rewrite check) |
| `_validation_types.py:101-124` (`effective_producer_type_for`) | Checks `.each` presence only, to decide the state-bus wrapped type (dict[str,X] vs raw) — single-modifier rule, not full combo |
| `_construct_graph.py:194` | Checks `.loop` presence for arm/topology walk |
| `_validation_modifiers.py:38` | Checks `.loop` presence for Loop-specific validation |
| `_param_classify.py:94,132` | Checks `.each`/`.portal` presence to classify a decorator param as a fan-out/handoff receiver (DI/topology classification, not lowering) |
| `_validation_portal.py:35,66-68,146,324` | All Portal-specific presence/`.is_dispatch` checks for mesh validation rules — single-modifier, not combo dispatch |
| `_construct_validation.py:226,248,264,283,339` | `.loop`/`.oracle`/`.portal` presence checks for assembly-time validation warnings/errors (not in the 9-list, correctly so — pure PRESENCE) |
| `_ir_normalize.py:142,169,192,214,265,299-300` | `.each`/`.portal`/`.oracle` presence checks driving IR-field normalizers (`fan_out_param`, `handoff_param`, `oracle_gen_type`, `handoff_channel`) — single-modifier gates, not combo dispatch |
| `_fan_agent_wrap.py:160,166,241,247` | Copies/resets `modifier_set` when building the bare-vs-wrapped Node pair — structural copy, not classification |
| `forward.py:701` | Single call into `effective_producer_type_for` via `.modifier_set` — narrow |
| `verify.py:137-139` | Presence check to route into Loop/Operator condition-registration check (test/verify harness) |
| `lint.py:479-486,654-662` | `.oracle`/`.loop` presence checks for lint rules (merge_fn DI check; Loop condition None-safety check) |
| `__main__.py:167-176` | `"operator" in classify_modifiers(item)[1]` — presence-only, to decide whether `neograph check` auto-supplies a MemorySaver |
| `compiler.py:181-192,254-258,167-172` | `"operator" in mods`, Portal grab, and `.modifier_set.combo.name` used only for logging/checkpointer-requirement/condition-registration gates — PRESENCE, distinct from the genuine dispatch blocks (#1/#2 above) in the same file |
| `state.py:142-143,217-241,256-292` | Additional narrower `.loop`/`.portal`/`.operator` presence gates interleaved with the genuine dispatch blocks — same file, different purpose |
| `node.py:410` | Reads `self.modifier_set` for `__or__`/pipe composition bookkeeping — structural, not classification |

## DEFINITION (source of truth, not a consumer)

- `modifiers.py` — `ModifierCombo` enum, `_COMBO_MAP`, `classify_modifiers()`, `ModifierSet.combo` property, `Modifiable.has_modifier`/`get_modifier` (lines 65-303, 848-849).
- `construct.py:137,196`, `node.py:256,357`, `_ir_branch.py:61,67`, `_ir_protocols.py:6,33` — field declarations / Protocol shape / inheritance, not logic.

## Borderline: worth naming, not part of the runtime/export duplication

- `testing/scaffold.py:47-50,255-275,574-575` — a test-scaffold code generator that branches on `.oracle`/`.each`/`.loop`/`.operator` presence to decide which assertions to emit into generated test files. It does not affect compiled/exported behavior, so it is NOT part of the "must agree with compiler.py" correctness problem, but it is yet another place that encodes "what does this modifier mean" and would silently drift if the combo rules changed without updating this generator too.

## Bottom line

Fresh sweep confirms the prior 9-module list is real but **undercounts by at least one**: `_fan_agent.py` performs genuine combo-based dispatch (deciding fan-over-agent support) via `classify_modifiers` and was not on the list. `loader.py` belongs on the list for re-deriving the same semantics, but via structural-marker pattern matching, not the `classify_modifiers`/`ModifierCombo` API surface — a mechanism distinction worth preserving in any consolidation plan. Total distinct DECOMPOSITION/DISPATCH sites: 11 (across 10 files, `compiler.py` containing 2 independent `match` blocks); everything else in the grep sweep is either narrow PRESENCE checks (16 files) or definitional.
