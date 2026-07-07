# DRY Review

**Scope**: The `cf14801^..HEAD` arc on `develop` (24 commits, 71 files, +8021/−489) — di_inputs injectors, twin remediation + thinness guard (ykun), hgpt v1+v2 (resource_reader/FromResource/manifest/hydration/expiry), fan-over-agent auto-wrap (packer/unpacker), run_id + `_run_cache`, json_mode, `_trace.named`, `_wiring` dedup. Focus: did NEW code added AFTER the ykun twin remediation honor extract-then-thin, or did fresh near-verbatim twins / semantic duplicates appear?
**Date**: 2026-07-07
**Method**: read every new leaf module in full (`_fan_agent*`, `_run_cache`, `_content_blocks`, `_uri_template`, `_trace`, `di`), traced each against its call sites, ran the `TestTwinThinness` guard (29 passed) to confirm the tabled twins are clean.

## Headline

The resource/DI/fan layers are **mostly well-factored** — the hard extractions (resource parse, URI templating, content-block scanning, DI resolution, port synthesis) all landed single-site. The one systemic gap is a **config-carrier injection block re-inlined at 4 sites** that the new twin-thinness guard structurally cannot see, so two of the fresh twins (`_inject_di_inputs`/`_ainject_di_inputs`) duplicate it verbatim with no guard coverage. Two smaller inline re-derivations round it out. No Critical/High business-logic divergence.

## Duplication Map

| Pattern | Occurrences | Files | Extractable? |
|---------|------------|-------|-------------|
| Copy-not-mutate config carrier `{**config, "configurable": {**configurable, K: v}}` | 4 | `_oracle.py:50`, `_dispatch.py:79`, `_dispatch.py:117`, `_execute.py:83` | Yes → `_inject_configurable(config, extra)` leaf |
| Read `RUN_ID` from config['configurable'] | 2 (variant spellings) | `_run_cache.py:54`, `_execute.py:48` | Yes → reuse `_run_cache._run_id` (or a shared `run_id(config)`) |
| Fan-modifier label re-derivation (loop/each/oracle priority) | 2 | `_fan_agent.py:54` (`_fan_modifier_label`) vs `_fan_agent.py:108` (inline) | Yes → call the existing helper |

## Findings

### DRY-01: Config-carrier injection block re-inlined across 4 sites — and it is a twin-guard blind spot
- **Severity**: Medium
- **Category**: Response (config-object construction) / Twin
- **Occurrences**: 4 construction sites; 2 of them are an un-tabled sync/async twin pair
- **Files**:
  - `src/neograph/_oracle.py:43,50` — `_inject_oracle_config`: `configurable = config.get("configurable", {})` … `return {**config, "configurable": {**configurable, **extra}}` (this is effectively the *general* form already)
  - `src/neograph/_dispatch.py:78-79` — `_inject_di_inputs` tail: `configurable = config.get("configurable", {})` / `return {**config, "configurable": {**configurable, StateKeys.DI_INPUTS: di_inputs}}`
  - `src/neograph/_dispatch.py:116-117` — `_ainject_di_inputs` tail: **byte-identical** to the sync twin above
  - `src/neograph/_execute.py:82-86` — `_inject_resource_manifest`: same carrier under `StateKeys.RESOURCE_MANIFEST_INJECT`
- **Description**: Every one of these builds the same copy-not-mutate config carrier — read `configurable`, splice one framework key, spread back into a fresh config dict. The docstrings even cross-reference each other ("copy-not-mutate, mirroring `_inject_oracle_config` / `_inject_di_inputs`"), which is the tell that a shared helper is missing: the intent was consciously copied, not extracted. A change to the merge semantics (handling a `None` config, an attr-form `configurable`, or a deep-merge vs shallow-merge decision) must be applied in 4 places. `_oracle._inject_oracle_config` is already the generic shape (`{**configurable, **extra}`) — the two di_inputs injectors and the manifest injector re-inline the single-key variant of it rather than calling it.
- **Twin-guard interaction (the load-bearing part)**: `_inject_di_inputs`/`_ainject_di_inputs` are a fresh sync/async twin pair introduced by the di_inputs work (`cf14801`) and are **not** in `TestTwinThinness.TWIN_TABLE` (`tests/test_guards_llm_runtime.py:1750`). Even if they were added, the guard would **not** catch this duplicate: `_builder_blocks` (`:1823`) only emits error builders (`X.build`), `CheckpointSchemaError`/`ToolMessage` constructors, structlog event calls, and usage dicts carrying `total_tokens`. A config-carrier dict literal matches none of those, so it is invisible to the guard. This is a genuine closure hole in the ykun invariant: "value-builder blocks must be single-site" was scoped to error/log/usage/ToolMessage content and left the config-carrier — an equally drift-prone framework value builder — uncovered.
- **Proposed extraction**: A neutral leaf `_config_inject.py` (or a function in the existing `_state_keys.py`, which both `_dispatch` and `_execute` already import) exposing `inject_configurable(config, extra: dict) -> config` and `read_configurable(config, key)`. Route all four sites through it; make `_inject_oracle_config` call it too. Then extend the guard's builder set (or add a dedicated one-line check) to treat a dict literal keyed on a `StateKeys.*_INJECT`/`DI_INPUTS` constant as a value-builder block so a future re-inline is caught.

### DRY-02: `RUN_ID` read re-inlined in `_execute` instead of reusing the `_run_cache` accessor
- **Severity**: Low
- **Category**: Config access
- **Occurrences**: 2
- **Files**:
  - `src/neograph/_run_cache.py:49-55` — `_run_id(config)`: `configurable = config.get("configurable") or {}` / `return configurable.get(StateKeys.RUN_ID)`
  - `src/neograph/_execute.py:48` — inline: `run_id = (config or {}).get("configurable", {}).get(StateKeys.RUN_ID)`
- **Description**: `RUN_ID` is a *new* primitive this arc (commit `847e7b7`, "per-run id primitive"). Two readers were born with two different null-handling spellings (`or {}` on the outer config vs on the inner map). Same key, same intent; a change to how the id is located (e.g. attr-form config, a namespaced key) touches both. The `_run_cache._run_id` helper is the natural home; `_execute._run_id_binds` should call it.
- **Proposed extraction**: Make `_run_cache._run_id` public-within-package (`run_id(config)`) and have `_execute._run_id_binds` call it. Both modules are leaves; no cycle.

### DRY-03: `_unsupported_reason` re-derives the fan label inline instead of calling `_fan_modifier_label`
- **Severity**: Low
- **Category**: Logic re-derivation
- **Occurrences**: 2 (same file)
- **Files**:
  - `src/neograph/_fan_agent.py:54-68` — `_fan_modifier_label(mods)`: loop → "Loop", each → "Each", oracle → "Oracle"
  - `src/neograph/_fan_agent.py:108` — inline: `fan = "Each" if "each" in mods else ("Loop" if "loop" in mods else "Oracle")`
- **Description**: Line 78 already calls `_fan_modifier_label(mods)` into `fan`, then line 108 re-derives the same label inline with the *same* priority order but a *different* tie-break spelling (helper is loop→each→oracle; inline is each→loop→oracle). They agree today only because the multi-modifier combos are all fail-loud, but the two orderings are an accident waiting to diverge, and the label logic now lives in two places. A new fan family (or a change to precedence) must be edited twice.
- **Proposed extraction**: Delete the line-108 inline and reuse the `fan` already computed at line 78 (it is guaranteed non-`None` past the `if fan is None: return None` gate at line 79). One label authority.

## What was checked and found CLEAN (answers to the review's specific questions)

These are recorded because the brief asked pointed questions and the answers are "no duplication" — worth pinning so a future reviewer doesn't re-audit:

- **`_content_blocks` vs the lift scanning** — CLEAN. `_iter_content_blocks` / `_block_field` / `_first_resource_link_uri` are single-site in `_content_blocks.py`; both consumers (`_agent_cycle._lift_resource_refs:428` and `di.hydrate_resource_ref` via `_first_resource_link_uri`) import them. No re-implemented block walk anywhere (verified by grep — the only other `isinstance(result, list)` sites are unrelated BaseModel renderers in `renderers.py`/`_tool_loop.py`).
- **`_run_cache` locking vs other registries** — CLEAN. `get_or_build`/`aget_or_build` are a twin pair NOT in TWIN_TABLE, but they route all state mutation through shared `_lookup`/`_store` (`_run_cache.py:58-71`); the twins differ only at the `await build()` seam. No duplicated value-builder block. This is the target shape the ykun guard describes — correctly thinned without needing a table entry.
- **`parse_resource_content` / `_expand_uri` / `_extract_uri_vars`** — CLEAN. `tool.py:34-35` imports all of them; `resource_reader` (`tool.py:369-377`) and `FromResource.aresolve` (`di.py:432-438`) share the exact same parse+interpolate path. The v1/v2 hgpt work honored extract-then-thin. (One behavioral asymmetry, not DRY: the `resource_reader` tool does not call `_enforce_max_bytes` while `FromResource` does — flagging for the consistency reviewer, not here.)
- **DI resolution** — CLEAN. `DIBinding.resolve`/`aresolve` (`di.py:318,399`) is the single resolution path; `aresolve` delegates every non-`FROM_RESOURCE` kind straight to `resolve`. No parallel `_resolve_di_value` survived the consolidation (grep returns zero hits outside `di.py`). `_classify_di_params`/`_classify_constants` all emit `DIBinding` — one binding type, one resolver.
- **Packer/unpacker port synthesis vs `construct_from_functions` port convention** — ONE convention, not two. `_synthesize_port` (`_fan_agent_wrap.py:135`) and `_synthesize_packer_wrap` (`:256`) both build the sub-construct boundary on `StateKeys.SUBGRAPH_INPUT` (`neo_subgraph_input`), the same field `construct_from_functions(input=...)` / `_construct_builder._cleanup_inputs_and_register` uses — and the docstring at `:110-115` explicitly commits to that convention (original prompt-var name is dropped to `neo_subgraph_input` "matching manual `construct_from_functions(input=...)` wrapping"). The unpacker re-exposes original keys inside the isolated sub-graph; no second port protocol was invented.
- **`_trace.named` vs inline span binding** — CLEAN. `named()` is the sole span-labelling helper and is applied uniformly at all 12 `add_node` / wrapper sites (`_wiring.py`, `compiler.py:429`, `factory.py:57,91`). No inline `.with_config(run_name=...)` re-inlining anywhere. The `_wiring` dedup landed as intended.

## Summary

- Critical: 0
- High: 0
- Medium: 1 (config-carrier injection block × 4 sites, incl. an un-tabled twin the thinness guard cannot see)
- Low: 2 (RUN_ID reader ×2; fan-label re-derivation ×2)
- Total duplicated logic blocks: 3 patterns / 8 concrete sites
- Estimated lines removable by extraction: ~15–20 (small — the value is single-siting drift-prone framework builders + closing the guard hole, not line count)

**Recommendation for 0.6.0**: DRY-01 is worth doing before release *specifically to close the guard blind spot* — extract `inject_configurable`/`read_configurable`, route the 4 sites through it, and extend the twin-thinness builder set to cover the config carrier so the ykun invariant actually holds for the di_inputs twins it was written to protect. DRY-02 and DRY-03 are trivial follow-ups (reuse an existing helper each) and can ride along or be filed. The rest of the arc — the resource layer, DI consolidation, fan-over-agent port synthesis — is a clean extract-then-thin execution.
