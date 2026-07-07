# Layering Review

**Scope**: `cf14801^..HEAD` on `develop` (24 commits, 71 files) — di_inputs side-channel,
hgpt v1+v2, fan-over-agent auto-wrap + compile pre-pass, per-run cache, run_id primitive,
json_mode wrapper, `_trace.named`, and the new leaf modules (`_uri_template`, `_content_blocks`,
`_run_cache`, `_fan_agent`, `_fan_agent_wrap`, `_trace`, `prompt`, `di`, `_usage`).
**Governing docs**: `docs/design/three-layer-principle-2026-07-03.md` (incl. §1.2.1 config-carrier
worked example), `docs/design/mcp-session-ownership-review-2026-07-05.md` (no-session-ownership verdict).
**Date**: 2026-07-07

## Verdict

**Clean on the layering dimension. No Critical / High / Medium violations.** Every item on the
brief's checklist holds. The three-layer boundary is not just respected — it is *locked* by an
AST guard (`tests/test_guards_three_layer.py`, 4 GUARD-A tests + import-direction GUARD-B +
no-ReAct-loop guard, all passing on HEAD). Two Low/informational observations below, neither
requiring action before release.

Checklist results (each verified against source):

| Brief item | Result |
|---|---|
| Layer placement of every NEW module | All Layer-2 leaf helpers or Layer-1 compile transforms. Correct. |
| Auto-wrap pre-pass — Layer 1, imports run-layer? | Layer-1 topology emission; imports only IR/compile modules. **No run-layer import.** |
| `_run_cache` — which layer, engine leak? | Layer-2 memoization helper; caches only Layer-2 cognition resources (LLM handle, tool instances, fetched resource). **No engine object cached.** |
| FROM_RESOURCE network I/O at node entry — purity test | Compliant. Read-I/O inside node body is Layer-2 cognition (like `llm.invoke`); async-only, mutation-replay idempotency-gated. |
| Import direction (all new modules) | None import `runner`/`_compiled`. GUARD-B passes. |
| Engine verbs confined to `_compiled.py` + `runner.py` | GUARD-A passes; `EXPECTED_ENGINE_SURFACE` exact-match holds. |
| Config-key carrier Layer-1→Layer-2 | `_neo_di_inputs` / `_neo_run_id` stashed in `config['configurable']`, copy-not-mutate, never enter state. Correct. |

## What I verified (positive confirmations)

### Auto-wrap pre-pass (`_fan_agent_wrap.wrap_fan_over_agents`, `compiler.py:154`)
Called before the state model is built. This is a **compile-time IR rewrite** — squarely Layer-1
topology emission (the doc's sanctioned category, alongside boundary-interrupt node insertion,
§1.3). Imports are `_fan_agent`, `_ir_branch`, `_normalize`, `_state_keys`, `construct`, `errors`,
`modifiers`, `naming`, `node` — all IR/compile-layer; **zero run-layer reach**. The cycle-avoidance
split (classifier `_fan_agent` imports only leaf IR so `_construct_validation` can call it at
assembly; wrapper-builder `_fan_agent_wrap` imports `Construct` and is reached only from
`compiler`) is documented and correct. The old inline fail-loud pass in `compiler.py` was cleanly
deleted and replaced by assembly-time `raise_if_unsupported_fan_over_agent` +
the pre-pass — a net reduction in compile-layer branching.

### `_run_cache` (process-global keyed store)
Layer-2 helper. Imports only `_state_keys`. Keyed on `StateKeys.RUN_ID`, which is minted
`uuid4().hex` per run/attempt in `runner._mint_run_id` (`runner.py:490`, called from
`_prepare`/`_aprepare` at `:638`/`:867`) and never persisted into a checkpoint. I specifically
checked the cross-run isolation risk: because the key is a fresh uuid4 per run, two concurrent
`arun()` calls in one process cannot collide, so an **auth-scoped** FROM_RESOURCE fetch cached
for operator A can never be served into operator B's run — the isolation is structural, not a
heuristic. Bounded LRU (`_MAX_ENTRIES=1024`) + `clear()` test hook wired into
`tests/conftest.py::_clean_registries`. The build callback runs outside the lock; only the map
mutation is locked. No engine concern (StateGraph / checkpointer / scheduling) touches this module.

### FROM_RESOURCE DI (`di.py` — `DIBinding.aresolve`, `hydrate_resource_ref`)
Purity-test compliant by construction. The fetch is **async-only**: `DIBinding.resolve` (sync)
fails loud for FROM_RESOURCE (`di.py:385`), and `_dispatch._inject_di_inputs` (sync driver) raises
a clear "drive with arun()" error rather than silently dropping the template var
(`_dispatch.py:60-72`). A resource *read* inside a node body is Layer-2 cognition, directly
analogous to the `llm.invoke` the doc explicitly permits (§1.2). The *mutation* hazard is gated:
`hydrate_resource_ref` refuses replay of a non-idempotent (`act`-mode) producer with
`NonIdempotentReplayError` (`di.py:207-209`) — side-effectful re-derivation is blocked, only a
read replays. `neograph` also refuses hidden LLM parsing inside resolution
(`parse_resource_content`, fail-loud on unparseable mime). This is a textbook application of the
purity test.

### `_neo_di_inputs` config carrier (three-layer §1.2.1)
The worked example in the governing doc matches the implementation exactly. Resolution runs once
through the canonical `DIBinding.resolve`/`aresolve` at the LLM-dispatch seams
(`_dispatch._inject_di_inputs`/`_ainject_di_inputs`, `_dispatch.py:264/298`; agent/act via
`_agent_cycle._turn_prep_kwargs`, `:170/191`), stashes under `StateKeys.DI_INPUTS` in
`config['configurable']` copy-not-mutate, and `_compile_prompt` (`_llm_render.py:209`) reads it
back behind the introspection gate. Never enters state → never touches the schema fingerprint →
re-derived fresh per superstep. Single resolver / single key / single reader, pinned by
`TestDiInputsInjectedAtLlmDispatchSeams`. Clean Layer-1→Layer-2 flow.

### `_trace.named` (`compiler.py:_add_subgraph`)
Layer-1 static labeling. It is a single `.with_config(run_name=..., tags=..., metadata=...)`
binding applied at compile time — no runtime branching, no invoke interception, no neograph-emitted
spans. Delegates `invoke`/`ainvoke` to the wrapped runnable so the sync/async dual path is
preserved. Imports only `langchain_core.runnables`. This is topology-adjacent config emission, not
runtime wrapping — correctly placed.

### Leaf helpers `_content_blocks`, `_uri_template`, `_usage`, `prompt`
No `langgraph`/`langchain`/`runner`/`compiled`/`StateGraph` imports. Pure Layer-2 data helpers.
Correctly placed.

### MCP session ownership (`di.py` fetcher/replayer contract)
Consistent with the kill-nmb2 verdict. The resource fetcher and replayer are **consumer-supplied**
callables read from `config['configurable']` (`RESOURCE_FETCHER_KEY`, `RESOURCE_REPLAYER_KEY`) —
neograph owns no session, no lifecycle, no disposal. Session/auth stay Layer-3 (adapter/ecosystem).
The module docstrings explicitly cite "no session ownership, exactly like tool factories". Correct.

## Findings

### LR-01: `_run_cache` is a process-global mutable singleton in the Layer-2 helper surface
- **Severity**: Low (informational — no action required for release)
- **Violation**: none — layer-placement observation
- **File**: `src/neograph/_run_cache.py:44-46`
- **Description**: `_run_cache` holds module-level mutable state (`_cache: OrderedDict`, `_lock`)
  shared across all runs in a process. The three-layer doc describes Layer 2 as "a small,
  self-contained helper library" — a process-global memoization store is a slightly heavier
  pattern than a pure function library. Its correctness rests **entirely** on the RUN_ID contract
  (fresh-per-run uuid4, never persisted). That contract is currently satisfied and pinned, and the
  module documents its own soundness argument thoroughly. The residual risk is future-facing: if
  RUN_ID were ever made deterministic/reused, or if the cache were ever keyed on `thread_id`
  (which the docstring explicitly warns against and avoids), auth-scoped fetched resources could
  leak across runs. No change needed now; noted so a future RUN_ID change is understood to have
  cache-isolation blast radius.
- **Reproduction**: `grep -n "uuid4\|_mint_run_id" src/neograph/runner.py` (confirms the
  collision-free mint the cache depends on); `sed -n '1,29p' src/neograph/_run_cache.py`.
- **Recommended fix**: None. Optionally, a one-line structural guard asserting `_mint_run_id` uses
  `uuid4` would lock the invariant the cache silently depends on. (The `_run_cache` docstring is the
  contract today; a guard would make it enforced.)

### LR-02: Compile pre-pass rewrites user IR → `compiled.construct` no longer `is` the passed pipeline when a fan-over-agent is wrapped
- **Severity**: Low (informational)
- **Violation**: none — legitimate Layer-1 transform; introspection-identity note
- **File**: `src/neograph/compiler.py:154`, `_fan_agent_wrap.py:73`
- **Description**: `construct = wrap_fan_over_agents(construct, scripted_lookup)` reassigns the
  local, and `CompiledNeograph(..., construct=construct, ...)` (`compiler.py:273`) stores the
  rewritten construct. `wrap_fan_over_agents` is identity-preserving **only on no-op** (its
  docstring says so explicitly, to keep `compiled.construct is pipeline` for the common case). When
  a supported fan-over-agent IS wrapped, `compiled.construct` is a `model_copy` with the node
  replaced by a sub-construct, so `compiled.construct is user_pipeline` becomes `False`. This is
  arguably *correct* — `compiled.construct` should reflect what was actually compiled — and the IR
  rewrite is exactly the Layer-1 topology emission the doc sanctions. Flagging only because any
  post-compile introspection that assumed object identity with the passed pipeline would silently
  diverge for wrapped pipelines. No layer boundary is crossed.
- **Reproduction**: `sed -n '55,73p' src/neograph/_fan_agent_wrap.py` (identity-preserve-on-no-op
  comment); `grep -n "construct=construct" src/neograph/compiler.py`.
- **Recommended fix**: None. If desired, one sentence in the pre-pass docstring noting that a
  *wrapped* pipeline yields a non-identical `compiled.construct` would remove the only surprise.

## Summary

- Critical: 0
- High: 0
- Medium: 0
- Low: 2 (both informational; no release blocker)

The layering discipline across this arc is exemplary: new engine-adjacent behavior (fan-over-agent
isolation, trace spans, per-run caching, resource hydration) was added **without** any engine verb
escaping `_compiled.py`/`runner.py`, without any compile-layer module reaching into the run layer,
and with the config side-channel used exactly as the Layer-1→Layer-2 carrier the doc prescribes.
The `test_guards_three_layer.py` lock (exact-match `EXPECTED_ENGINE_SURFACE` + synthetic slip
meta-tests) means this is enforced going forward, not just true today.
