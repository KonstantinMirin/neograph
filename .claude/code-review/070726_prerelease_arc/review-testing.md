# Testing Quality Review

**Scope**: `cf14801^..HEAD` on `develop` (24 commits, 71 files, +8021/−489) — the 0.6.0 pre-release arc: di_inputs side-channel + twin extract-then-thin, hgpt v1+v2 (resource_reader, FromResource sync+async DI, ResourceRef manifest, layered expiry + idempotency gate, templated URIs, per-run fetch cache), fan-over-agent auto-wrap (Oracle/Each/Loop + bundle-port synthesis), per-run run_id, json_mode native `response_format`, trace span hygiene.
**Date**: 2026-07-07
**How run**: `uv run --extra dev pytest` (589 arc tests executed across two batches — all green).

## Test Suite Shape

| Category | Count | Notes |
|----------|-------|-------|
| New arc test files | 20 | ~4,300 lines of new test code |
| E2E through real compiled graph + `run()`/`arun()` | high | cache/resume, di_inputs→agent, each/oracle/loop-over-agent, trace hygiene, observability contract all drive `compile()` + `run()` |
| Seam-level unit (`hydrate_resource_ref`, `mint_run_id`, `_substitute`) | moderate | correctly scoped to the unit under test, with real config dicts |
| Assertion-free tests | 0 | none found |
| Weak truthy/`is not None`-only tests | 0 | every `is not None` is a fake-internal guard OR paired with an equality/inequality assertion (e.g. `test_run_id.py:68-69`) |
| High-mock tests (>5 patches) | 0 | the suite uses hand-built stateless fakes + a real callback probe, not `unittest.mock` patch stacks |

Ran: `tests/test_agent_cycle_run_cache.py`, `tests/modes/test_resource_hydration.py`, `tests/modes/test_resource_di_inputs.py`, `tests/test_from_resource.py`, `tests/modifiers/test_each.py`, `tests/modifiers/test_oracle.py`, `tests/test_run_id.py`, `tests/test_trace_span_hygiene.py`, `tests/test_observability_contract.py`, `tests/test_prompt_compiler.py` → **185 passed**; then `tests/test_resource_reader.py`, `tests/modes/test_resource_manifest.py`, `tests/modes/test_output_strategies.py`, `tests/test_di.py`, `tests/test_lint.py`, `tests/test_template_lint.py`, `tests/test_loop.py`, `tests/test_check_fixtures.py`, `tests/test_guards_run_id.py`, `tests/test_guards_llm_runtime.py`, `tests/test_guards_branch_arm_walks.py`, `tests/test_guards_meta.py` → **404 passed**.

## Findings

### TQ-01: Oracle-over-agent "isolation" tests prove no-collapse, not channel-content isolation — docstrings overclaim
- **Severity**: Low
- **Anti-pattern**: Assertion weaker than the claim (borderline Happy-Path/over-claim)
- **File**: `tests/modifiers/test_oracle.py:514-724` (`TestOracleModels.test_oracle_over_*_agent_*`, 7 tests)
- **Test**: e.g. `test_oracle_over_self_contained_agent_node_runs`
- **Description**: Every fan-over-agent Oracle test asserts only `result[...] == Claims(items=["merged-2"])`, where the merge fn returns `f"merged-{len(variants)}"`. The docstrings claim this is "the proof the two ReAct cycles ran with **isolated message channels**, not one merged channel." It is not that proof. `len(variants) == 2` proves two results reached the merge barrier — i.e. the branches did **not collapse**. It does NOT prove the two cycles ran on isolated message channels: a hypothetical shared-channel implementation that still emitted two `Send` results would also produce `merged-2`.
- **Evidence**: If you broke isolation by cross-contaminating the two cycles' message histories but still returned two results, the assertion passes. The test only regresses if the count changes. It DOES catch the specific documented regression (m6d3.6 collapse → the barrier would see 1 → `merged-1`), so it is not false confidence for that bug — but the docstring's "isolated message channels" framing writes a cheque the assertion doesn't cash.
- **Why acceptable in part**: Oracle variants share identical input, so per-branch content isolation is genuinely unobservable through the output (unlike Each). The Each-over-agent suite (`test_each.py:166-230`) is the correct model: its `_EchoReActFake` + `_echo_prompt_compiler` reflect the rendered per-branch value back into the output, and `_assert_per_branch_isolation` proves `alpha` saw "alpha" and NOT "beta" (and vice-versa) — that genuinely catches a channel-merge regression.
- **Recommended fix**: No new test needed. Tighten the Oracle docstrings to say what the assertion proves ("N ISOLATED variants reach the barrier — no fan-in collapse") and drop the "isolated message channels" phrasing, or add one Oracle test that gives each generator a distinct model tier (the `models=[...]` path already exists) and asserts each variant carries its tier, to observe per-branch isolation the way Each does.

### TQ-02: `observability_contract` payload tests assert key-presence, not values
- **Severity**: Low (noted as acceptable-by-design)
- **Anti-pattern**: Type/shape check rather than value check
- **File**: `tests/test_observability_contract.py:84-121, 164-201, 412-475, 522-564`
- **Description**: The payload tests assert `"max_iterations" in evt`, `"loops" in evt`, `"token_budget" in evt`, `evt.get("log_level") == "warning"` etc. — they pin the event **schema** (key set + severity) but not the numeric values (e.g. that `loops` equals the actual loop count driven).
- **Evidence**: A regression that emitted `loops: 0` when the cycle actually ran 3 loops would still pass the payload test. The event-name tests (`test_event_name_emitted_*`) do drive real cycles to the limit and assert the event fires, so the "did the condition trigger" path is covered; only the value fidelity of the payload is loose.
- **Why acceptable**: This file is explicitly a *contract* suite — its stated purpose (file docstring) is that behavioral value-assertions live elsewhere and this pins the stable event name/schema so consumers can index on it. Flagging only so the reviewer knows the value-fidelity of these payloads rides on the separate behavioral tests, not here.
- **Recommended fix**: None required; if desired, one payload test could assert `evt["loops"] == <driven count>` to make the numeric contract load-bearing.

## Positive Examples

These are the strongest tests in the arc — genuinely behavioral, driving real compiled graphs, with assertions that would fail if the production code broke:

- **`tests/test_agent_cycle_run_cache.py` (whole file)** — the per-run handle/resource reuse tests are exemplary. They count real factory invocations (`factory_calls`, `fetch_count`) through a real `compile()` + `run()`/`arun()` with a **real file-backed `SqliteSaver`/`AsyncSqliteSaver`** across an interrupt/resume boundary, asserting `== 1` within a run (cache HIT across supersteps) and `== 2` after resume (fresh `RUN_ID` → cache MISS → rebuild). The fakes are stateless/history-driven (`_MultiTurnFake`, `_InterruptFake` decide from message history), so a cached-vs-rebuilt handle is observable only through the counter — exactly the two-lifetime invariant the feature exists to enforce. Break the RUN_ID re-mint on resume and `test_*_rebuilt_on_resume` fails.

- **`tests/modes/test_resource_hydration.py:162-231` (`TestLayeredExpiry`)** — read → replay → fail-loud with an **idempotency HARD GATE**, tested through `hydrate_resource_ref` with real fetcher/replayer closures. Asserts error *attributes* not just types (`ei.value.tool_name == "list_emails"`, `ei.value.node == "assess"`, `ei.value.ref.uri == ...`), asserts the replay was invoked with the correct args (`calls == [("list_emails", {"deal_id": 42})]`), and — critically — `test_read_parse_error_does_not_trigger_replay` asserts a parse failure is NOT masked by a replay (`assert not isinstance(ei.value, (ResourceExpiredError, NonIdempotentReplayError))`), with the replay closure guarded by `raise AssertionError` if wrongly called. This is textbook negative-path testing.

- **`tests/modifiers/test_each.py:65-230` (`TestEachOverAgent`)** — the `_EchoReActFake` + `_echo_prompt_compiler` pairing is the gold-standard fan-isolation test: it renders the per-branch upstream value into the user message and reflects it back through the model's typed output, so `_assert_per_branch_isolation` can prove each isolated cycle saw ONLY its own item (`"alpha" in alpha_seen and "beta" not in alpha_seen`). This WOULD catch a channel-merge regression. `test_each_over_self_contained_agent_fails_loud` pins the assembly-time `ConstructError` for the unsupportable case.

- **`tests/test_from_resource.py:74-92`** — the sync/async fail-loud pair drives the **real compiled graph**: `arun` asserts the fetcher was awaited with the exact URI AND the body received the parsed `Doc`; sync `run()` asserts a `NeographError` matching `arun`. Not a seam echo — it goes through `run()`/`arun()`. (Companion `test_resource_di_inputs.py:102-112` additionally asserts the sync fail names the offending param, the node, and the async driver.)

- **`tests/test_trace_span_hygiene.py`** — a real `BaseCallbackHandler` probe records every chain-start span across scripted/think/agent/branch/subgraph modes on real `run()`s, asserting BOTH the negative invariant (no internal wrapper name in `LEAKING_WRAPPER_NAMES` ever surfaces) AND the positive (user node names + `neograph_mode`/`neograph_output_type` metadata present, agent cycle emits `explore__agent`/`__tools`/`__parse`). `test_no_wrapper_leak_and_correct_result_when_no_callbacks_attached` proves the result is still correct with tracing off.

- **`tests/test_run_id.py`** — covers the full invariant surface: presence, stability across supersteps, distinctness across runs, re-mint on resume (real Sqlite), no-mutation of the caller's config, distinct ids for parallel `arun` sharing one config dict, and the three exclusion invariants (absent from returned state, stream chunks, AND checkpoint channel_values). The negative "absent from checkpoint" is the load-bearing one and it's asserted directly.

- **`tests/test_prompt_compiler.py:522-616` (`TestDiInputReachesAgentModelEndToEnd`)** — a genuine RED-first TDD E2E: a capturing prompt_compiler proves the resolved `FromInput` value reaches the agent cycle's compiler as `di_inputs == {"domain": "oncology"}` AND is rendered into the user message (`"Analyze the oncology domain." in content` and `"{domain}" not in content`), with `strict=False` deliberately chosen so the RED run surfaces as a behavioral assertion, not a crash.

## Fake fidelity note

The echo/history-driven fakes are high fidelity and reuse-safe:
- `ReActFake` and the local `_MultiTurnFake`/`_InterruptFake`/`_EchoReActFake` derive turn index from **message history** (`sum(isinstance(m, ToolMessage) ...)`), so they behave identically whether the agent runs as one node body or as a cycle of rebuilt supersteps — the precise property that makes the cache-reuse counting tests valid.
- `neograph.testing.fakes.py` uses real `langchain_core.AIMessage` and emits **parseable JSON on the final turn** (`_final_json_content`), matching the f7nt "parse the loop's final turn directly" production path — no AIMessage duck-typing that would drift from real coercion.
- `TextFake`/`StructuredFake` record `bind_calls` on a shared list so the json_mode native-`response_format` tests (15s2) can assert what was bound and the attempt-bind-and-fall-back path (`reject_response_format=True`) offline. Fidelity here is real, not a stub.

## Fixture-suite coherence

The `check_fixtures` moves are correct per the CLAUDE.md workflow: the old `should_fail/oracle_over_agent_node.py` (`CHECK_ERROR: Oracle over an agent/act node is not supported`) became `should_pass/oracle_over_agent_node.py` now that auto-wrap makes it valid, and two new `should_fail` fixtures pin the remaining unsupported cases (`each_over_self_contained_agent`, `oracle_over_agent_multi_output`). New `should_pass` fixtures cover each/loop/oracle-over-agent with single/multiple inputs. The validator's supported/unsupported boundary is fixture-pinned in both directions.

## Summary

- Critical: 0
- High: 0
- Medium: 0
- Low: 2 (TQ-01 docstring over-claim on Oracle isolation; TQ-02 contract payloads pin schema not value — by design)
- Overall test quality assessment: **High.** This arc's tests are behavioral and drive real compiled graphs; the cache/replay/expiry, run_id, fail-loud, and Each-isolation suites have strong, regression-catching assertions (error attributes, real Sqlite resume counting, negative-path guards). Fakes are stateless/history-driven and provider-faithful, so seam-level echoes are not a risk here. The only soft spot is cosmetic: the Oracle-over-agent isolation tests catch the collapse regression they target but their docstrings claim more (message-channel isolation) than `merged-N` counting can prove — tighten the wording or borrow the Each suite's per-branch-value technique.
