# Python Practices Review

**Scope**: The `cf14801^..HEAD` develop arc (24 commits, 71 files) — di_inputs, hgpt v1+v2, fan-over-agent, run_id + `_run_cache`, json_mode `_NativeJsonModeLLM`, span hygiene, dedup — reviewed through the Python-practices lens (async correctness, Pydantic v2 idioms, exception hygiene, resource lifecycle). Emphases per the team-lead brief: async DI resolution, the process-global `_run_cache`, the layered-expiry exception chain, `create_model`/frozen-model usage, the mypy-fix `partial()`, uuid4 placement, and the `_uri_template` RFC-6570 subset.
**Date**: 2026-07-07

## Baseline gates (grounding)

- `uv run --extra dev mypy src/neograph/` → **clean** (78 files). The `bb63ef8` mypy-fix commit's claim holds.
- `uv run --extra dev ruff check` on the five new leaf modules → **clean**.

So this review is entirely about semantics linters can't see.

## Findings

### PP-01: `_run_cache` never releases entries at run completion — event-loop-bound handles (MCP clients, LLM clients) linger until LRU eviction
- **Severity**: Medium
- **Category**: Resources / Async
- **File**: `src/neograph/_run_cache.py:44-71`, callers `src/neograph/_tool_loop.py:318,400,435`
- **Description**: The cache stores per-run **LLM clients** (`llm:{node}:{tier}`) and **tool instances** (`tools:{node}`) keyed on `RUN_ID`. Entries are only ever removed by the 1024-entry LRU eviction (`_store`) or the test-only `clear()`. There is no run-completion hook that drops a finished run's entries. In a long-lived process (the "production-grade agents on LangGraph" target), a completed run's tool instances can be MCP `StructuredTool`s / clients that opened sessions/sockets on that run's event loop; the cache keeps them referenced (so no GC, no `__del__`/`aclose`) for up to 1024 subsequent runs. Two consequences: (a) live connections and the run's event-loop-bound objects outlive the run — a resource leak that is invisible until eviction; (b) nothing ever calls an `aclose()`/`__aexit__` on cached handles, so cleanup is left entirely to GC finalizers of objects that reference a since-closed loop.

  The design docstring is honest that eviction is "a perf miss, never a correctness fault" — that framing is right for *rebuild cost*, but it does not address *holding heavyweight, loop-bound handles past their run*. The key soundness argument (run_id re-minted on resume ⇒ never serve a stale handle into a fresh lifetime) is correct and well-reasoned; the gap is purely lifecycle/cleanup, not correctness.
- **Reproduction**: `grep -n "popitem\|clear\|del _cache\|aclose\|__aexit__" src/neograph/_run_cache.py` — only LRU `popitem` + test `clear()`; no per-run eviction, no handle close.
- **Recommended fix**: Add a `drop_run(run_id)` that removes all `(run_id, *)` entries, and call it from `runner._prepare`/`_aprepare`'s teardown (finally block after `ainvoke`/`invoke` completes) — the same brains that mint the id own its disposal. If cached tool instances can hold async resources, prefer draining them through an `aclose()` protocol on eviction rather than dropping the reference. At minimum, document in the module that cached values MUST be safe to abandon without explicit close (i.e. no owned sockets) so the contract on `build` callbacks is explicit.

### PP-02: `aget_or_build` has no single-flight — concurrent same-key builds double-fetch/double-mint under fan-out
- **Severity**: Medium
- **Category**: Async
- **File**: `src/neograph/_run_cache.py:92-105`; callers `src/neograph/_dispatch.py:109` and `src/neograph/_tool_loop.py:435`
- **Description**: The module docstring justifies its lock discipline with "within one run the cycle's supersteps are sequential, so a single key is never built concurrently." That holds for a single agent cycle. It does **not** hold for **fan-over-agent** (Oracle/Each over an agent), which is a headline feature of this very arc. The auto-wrap (`_fan_agent_wrap._wrap_agent_node`) keeps the bare agent's `node.name`, and every fanned variant shares the same forwarded `config` (hence the same `RUN_ID`). On the async driver LangGraph executes the fanned `Send()`s concurrently, so N variants call `aget_or_build(config, "tools:{node_name}", …)` — and `aresolve` `"resource:{node}:{name}"` — with an **identical key at the same time**. Because `_lookup` and `_store` are separate locked critical sections and the `await build()` runs *between* them (outside the lock), all N miss and all N build. Result: N concurrent MCP-client builds / N concurrent per-run token mints / N concurrent resource fetches for what the design intends as a build-once-per-run value. Not a correctness fault (factories are declared re-invocable/idempotent), but it defeats the cache's stated purpose exactly in the concurrent case, and an idempotent-but-costly per-run token mint gets multiplied by the fan width.
- **Reproduction**: Inspect the read→build→write window: `sed -n '92,105p' src/neograph/_run_cache.py` — no in-flight `Future`/event held under the key between lookup-miss and store.
- **Recommended fix**: Add single-flight for the async path: store an `asyncio.Future` (or an `asyncio.Event` + slot) under the key at first miss so concurrent callers on the same loop await the in-flight build instead of launching their own. Keep the `threading.Lock` for the map mutation (it is correctly non-blocking); the single-flight primitive is per-key and only needed on `aget_or_build`. Alternatively, if double-build is genuinely acceptable, weaken the docstring claim so it doesn't assert sequentiality that fan-out violates.

### PP-03: `threading.Lock` vs `asyncio.Lock` choice is correct — no change needed (positive)
- **Severity**: Info
- **Category**: Async
- **File**: `src/neograph/_run_cache.py:35,45,58-71`
- **Description**: The team-lead brief flagged "asyncio.Lock vs threading.Lock." The `threading.Lock` here is the **right** choice: the cache is shared by both the sync (`get_or_build`) and async (`aget_or_build`) entrypoints, an `asyncio.Lock` would be unusable from the sync path, and — critically — the lock only ever guards a **non-blocking dict mutation** (`get`/`move_to_end`/`__setitem__`/`popitem`), never an `await`. The `build`/`await build()` deliberately runs *outside* the lock. So there is no "hold a threading.Lock across await" hazard and no event-loop blocking. This is correct as written; the only residual issue is the missing single-flight (PP-02), which is orthogonal to the lock type.

### PP-04: `hydrate_resource_ref` read step catches bare `Exception`, misclassifying fetcher programming errors as expiry
- **Severity**: Low
- **Category**: Errors
- **File**: `src/neograph/di.py:198-204` (and the replay catch at `218-227`)
- **Description**: `try: content, mime = await fetcher(ref.uri) except Exception as exc: read_error = exc` treats **any** exception from the consumer fetcher as "candidate expiry" and proceeds to the replay path. A genuine bug in the consumer's fetcher (an `AttributeError`, `KeyError`, a `TypeError` from a signature mismatch) is thus laundered into an expiry signal and triggers a producing-call replay — masking the real defect and doing extra work. The `noqa: BLE001` comment documents the intent ("any fetch failure = candidate expiry"), and cause-chaining is preserved end-to-end (`ResourceExpiredError.of(..., cause=read_error)` and the `raise … from exc` at line 227), so the underlying error is *recoverable in diagnosis* — which is why this is Low, not Medium. But "any fetch failure" is broader than "the resource expired": a fetcher raising a `NotImplementedError` or a programming error is not expiry. Consider narrowing to the transport/protocol errors that actually denote expiry, or at least `log.warning`-ing the read_error before falling through so a non-expiry cause is visible even when the replay ultimately succeeds and swallows it.
- **Reproduction**: `sed -n '196,229p' src/neograph/di.py` — the read `except Exception` unconditionally routes to replay for every exception type.
- **Note (GraphInterrupt/GraphBubbleUp safety)**: The brief asked about interrupt safety in replay. These broad catches wrap **consumer I/O callables** (`fetcher`, `replayer`) — not a graph `.invoke`/`.astream` — so LangGraph's `GraphInterrupt`/`GraphBubbleUp` control-flow exceptions do not originate inside them, and the agent-cycle tool call (`_agent_cycle.py:557-560`) narrows its own catch to `except NotImplementedError`, leaving interrupts to bubble. So replay does not swallow interrupts today. The only theoretical exposure is a fetcher/replayer that *itself* calls `interrupt()` (it shouldn't — it's a plain fetch/replay callable); if that contract were ever relaxed, these bare catches would eat the interrupt. Worth a one-line contract note on `RESOURCE_FETCHER_KEY`/`RESOURCE_REPLAYER_KEY` that the callables must not raise graph control-flow exceptions.

### PP-05: `bb63ef8` `partial()` fix is correct and superior to the lambda it replaced (positive)
- **Severity**: Info
- **Category**: Async / Types
- **File**: `src/neograph/_dispatch.py:109-113`
- **Description**: Replacing `lambda binding=binding: binding.aresolve(config)` with `partial(binding.aresolve, config)` is not just a mypy appeasement — it is the more correct form. `partial` binds the *bound method of the specific `binding`* at creation time inside the `for name, binding in param_res.items()` loop, so per-iteration capture is guaranteed without the `binding=binding` default-arg trick (the classic late-binding-closure footgun). `aresolve(self, config, *, state=None)` means `partial(binding.aresolve, config)()` calls `binding.aresolve(config)` returning the awaitable that `aget_or_build` awaits — exactly right. No change needed.
- **Note on the companion `create_model` fix** (`_fan_agent_wrap.py:218-221`): dropping the explicit `__base__=BaseModel` (which `create_model` defaults to anyway) to use the field-definition-dict overload is a clean, idiomatic Pydantic v2 usage. Note the *other* `create_model` call at `_synthesize_port` (`_fan_agent_wrap.py:127-129`) still passes `__base__=BaseModel` explicitly — that call has **no** `**field_defs`, so it hits a different (valid) overload and mypy accepts it. The inconsistency between the two call sites is cosmetic, not a bug; leaving it is fine, but a one-word comment would prevent a future reader "fixing" the surviving `__base__` and re-breaking mypy.

### PP-06: uuid4 run-id minting placement is sound (positive)
- **Severity**: Info
- **Category**: Async / Correctness
- **File**: `src/neograph/runner.py:477-491`, `636-638`, `865-867`
- **Description**: `_mint_run_id` builds a fresh `{**config, "configurable": {**configurable, RUN_ID: uuid4().hex}}` (copy-not-mutate), is called once inside `_prepare`/`_aprepare` before the engine runs, is never accepted from the caller, and lands in `config['configurable']` (so it never enters state, never touches the schema fingerprint, never persists into a checkpoint). This is precisely what makes the `_run_cache` key sound: fresh-per-attempt, stable-across-supersteps, re-minted on resume. The copy-not-mutate is important — two `arun` calls sharing one config dict each get their own id. `uuid4().hex` (122 bits) is appropriate for a per-process correlation id. Placement and idiom are correct.

### PP-07: `_uri_template` is a hand-rolled RFC-6570 *subset* — correctly scoped and documented, with two minor rough edges
- **Severity**: Low
- **Category**: Types / Correctness
- **File**: `src/neograph/_uri_template.py:22-60`
- **Description**: This is deliberately a hand-rolled subset (simple `{var}`, reserved `{+var}`, form-query `{?a,b}` / continuation `{&a}`), stdlib-only, and the module docstring says so plainly — the right call over pulling in a full `uritemplate` dependency for the handful of shapes the resource layer emits. Two rough edges worth noting, neither a correctness bug for the documented input set:
  1. **The regex `\{([+#./;?&]?)([^{}]+)\}` admits operators (`#`, `.`, `/`, `;`) that `_sub` does not handle.** For `{#var}` (fragment) or `{/var}` (path segment), the `op` falls through to the final `else` branch and is expanded as a *simple* comma-join with no `#`/`/` prefix — silently wrong RFC-6570 semantics. Since the docstring scopes support to `{var}`/`{+var}`/`{?…}`/`{&…}`, an author who writes an unsupported operator gets silent misexpansion rather than a fail-loud. Given neograph's fail-loud ethos elsewhere in this arc, consider either narrowing the operator char class to `[+?&]` (so an unsupported op simply isn't recognized as a var — though then it'd be left literal, also imperfect) or raising on an operator that reaches the `else` with a non-`+` op.
  2. **`{?a,b}` uses `values.get(vn) is not None` to decide inclusion but `values[vn]` to read.** These are consistent, but note a present-but-falsy value like `0` or `""` **is** included (correct RFC-6570 behavior — only *undefined* is omitted), whereas the `_extract_uri_vars`/`_expand` split means a var that is entirely absent from `values` is dropped. That's the intended "missing values are omitted" contract; just confirm callers never rely on `""` being treated as absent. `quote(str(values[vn]), safe='')` correctly percent-encodes reserved chars for the non-`+` case and leaves them for `{+var}`. This part is right.
- **Reproduction**: `python -c "from neograph._uri_template import _expand_uri; print(_expand_uri('a{#x}', {'x':'p/q'}))"` → prints `ap/q` (no `#`, unencoded) rather than the RFC-6570 `a#p/q`.
- **Recommended fix**: Either tighten the operator character class to only the supported set so unsupported operators are not silently mis-expanded, or add an explicit `raise ConfigurationError` in `_sub` when `op` is not in `("", "+", "?", "&")`. Low priority — no current caller emits the unsupported operators.

### PP-08: `_NativeJsonModeLLM._fell_back` is a shared mutable flag toggled from both sync and async paths (benign, note only)
- **Severity**: Low
- **Category**: Async
- **File**: `src/neograph/_llm.py:98-131`
- **Description**: The wrapper's `_fell_back` boolean is set to `True` from both `invoke` and `ainvoke` on the first `response_format` rejection, and the docstring explicitly intends the flag to be shared so "the first rejection (on either surface) switches every subsequent call to the unbound client." If the wrapper instance is reused concurrently (e.g. a cached LLM handle under fan-out, per PP-02), two coroutines could race to set `_fell_back = True` — but the write is idempotent (always `True`) and the pre-check `if self._fell_back` only ever transitions false→true, so a race cannot produce a wrong result. The rejection detection (`_is_response_format_rejection`) is string/type-name-based rather than provider-class-based, which is a deliberate, well-documented choice to avoid provider coupling. No change needed; flagged only because the brief asked about the json_mode wrapper and shared-state safety.

## Positives worth recording

- **Sync/async twin discipline is consistently applied and honest.** Every new async surface has a named twin (`_inject_di_inputs`/`_ainject_di_inputs`, `resolve`/`aresolve`, `_build_turn_prep`/`_abuild_turn_prep`, `get_or_build`/`aget_or_build`), and the sync twins **fail loud** on `FROM_RESOURCE` (`di.py:385-395`, `_dispatch.py:63-70`) rather than silently dropping an awaited value — the correct handling of "can't await on the sync driver." `ScriptedDispatch.execute` even calls `result.close()` to suppress the never-awaited `RuntimeWarning` before failing loud (`_dispatch.py:216-225`). This is exactly right.
- **No blocking I/O inside async DI resolution.** `aresolve`/`hydrate_resource_ref` await the fetcher and otherwise only do CPU-bound `model_validate_json` — no hidden sync socket/file calls on the event loop.
- **Exception hygiene in the layered-expiry chain is good.** `ResourceExpiredError` carries `cause=`, the replay failure re-raises `from exc`, and `NonIdempotentReplayError`/`ResourceExpiredError` are typed subclasses of `NeographError` with structured `.of(...)` builders and actionable hints. A parse failure in the read step is deliberately **not** masked by replay (`di.py:170-186` docstring + code path) — the distinction between "expired" and "malformed" is preserved.
- **Pydantic v2 idioms are clean throughout.** `frozen=True` via class kwarg, `Field(default_factory=dict)`, `PrivateAttr(default=None)`, and the canonical `object.__setattr__` for frozen-model private-attr mutation (`tool.py:93-96`) with a comment explaining why. No v1 `@validator`/`.dict()`/`.parse_obj()` anywhere in the arc.

## Summary

- Critical: 0
- High: 0
- Medium: 2 (PP-01 run-cache lifecycle / no run-completion eviction; PP-02 missing single-flight under fan-out concurrency)
- Low: 4 (PP-04 broad fetcher catch; PP-07 URI-template unsupported-operator silent misexpansion; PP-08 shared `_fell_back` flag — benign)
- Info/positive: PP-03 (lock choice correct), PP-05 (partial fix correct), PP-06 (uuid4 placement correct)

The two Medium findings are the ones that matter and they are related: both stem from the `_run_cache` design's stated assumption of run sequentiality, which the arc's own fan-over-agent feature can violate on the async driver. Neither is a correctness fault today (idempotent factories, bounded LRU), but PP-01 (loop-bound handles held past their run) and PP-02 (fan-width-multiplied builds/mints) both undercut the cache's purpose in exactly the long-lived-server + fan-out scenario neograph targets. Everything else is clean and the async twin discipline is genuinely well-executed.
