# DSML Recovery Fix — Best-Path Analysis

**Ticket:** [neograph-0tid](../../.beads/issues.jsonl) — `BUG: DSML recovery is json_mode/text exclusive — structured strategy has no DSML handling`
**Author:** analysis pass for maintainer review
**Date:** 2026-05-20
**Status:** read-only analysis. No code change in this document.

---

## TL;DR

- **Bug still exists in current code** (post-Wave-E). The strategy gate at `src/neograph/_tool_loop.py:354` still encloses the entire DSML detect-and-retry block inside `if strategy in ("json_mode", "text"):`. The `else` branch (`structured`) jumps straight to `_call_structured` (line 382) with no DSML-awareness.
- The behaviour is currently **pinned by a passing test** (`tests/modes/test_llm_internals.py:2617` — `test_structured_path_has_no_dsml_recovery_when_provider_returns_dsml`) that asserts the parity gap as the documented status quo. Fixing the bug requires updating that test or replacing it.
- A second, more silent failure mode applies when the provider's `with_structured_output(..., include_raw=True)` returns `{"parsed": None, "raw": <AIMessage with DSML>, "parsing_error": ...}` instead of raising `TypeError`. `_call_structured` (line 34) hands the `None` parsed value back to the caller as a valid result. Nothing in this analysis is invalidated by that path — it is a related, slightly worse shape of the same gap and deserves coverage in the fix.
- **Recommended fix: Option B (strategy-agnostic recovery).** Detection is purely a property of the response text; the strategy is purely a property of how the LLM was invoked. Decoupling matches the actual semantics, costs roughly the same as Option A, and accommodates the include_raw-returns-None path naturally.
- **Estimated effort: 4–7 hours** for fix + tests + adjusting the existing parity-gap-pinning tests.

---

## Section 1: Confirm the bug

The bug exists in the current code. Citations are against `develop` HEAD as of 2026-05-20.

### Where the strategy gate lives

`src/neograph/_tool_loop.py:349-382`:

```text
349  # Parse final response as structured output — strategy-aware
350  assert config is not None
351  strategy = cfg.output_strategy
352  max_retries = cfg.max_retries
353
354  if strategy in ("json_mode", "text"):
355      last_msg = messages[-1]
356      raw_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
357      try:
358          parse_result = _parse_json_response(raw_text, output_model)
359      except ExecutionError as parse_exc:
360          # Layer C: DSML/XML tool-call markup in final response — targeted retry
361          import re as _re
362          if _re.search(r"<[^>]*(?:function_call|invoke|DSML)[^>]*>", raw_text, _re.IGNORECASE):
363              log.warning("trailing_tool_call_markup", ...)
364              ...
365              messages.append({"role": "assistant", "content": raw_text})
366              messages.append({"role": "user", "content": budget_msg})
367              try:
368                  retry_response = llm.invoke(messages, config=config)
369                  ...
370              except ExecutionError:
371                  parse_result, _ = _invoke_json_with_retry(...)
372          else:
373              parse_result, _ = _invoke_json_with_retry(...)
374      usage = getattr(last_msg, "usage_metadata", None)
375  else:
376      parse_result, usage = _call_structured(llm, messages, output_model, strategy, config, ...)
```

The DSML detect-and-retry block at lines 360-373 is **lexically nested** inside the `if strategy in ("json_mode", "text")` branch. When `strategy == "structured"` (the framework default — see `_llm_config.py:36`), control jumps to the `else:` on line 375 and the entire DSML rescue logic is bypassed.

### What the structured branch does instead

`src/neograph/_llm_dispatch.py:20-43`:

```text
20  def _call_structured(llm, messages, output_model, strategy, config, max_retries=1):
...
29      if strategy == "structured":
30          usage = None
31          try:
32              structured_llm = llm.with_structured_output(output_model, include_raw=True)
33              raw_result = structured_llm.invoke(messages, config=config)
34              if isinstance(raw_result, dict) and "parsed" in raw_result:
35                  result = raw_result["parsed"]
36                  raw_msg = raw_result.get("raw")
37                  usage = getattr(raw_msg, "usage_metadata", None) if raw_msg else None
38              else:
39                  result = raw_result
40          except TypeError:
41              structured_llm = llm.with_structured_output(output_model)
42              result = structured_llm.invoke(messages, config=config)
43          return result, usage
```

Two paths fail under DSML:

1. **TypeError path.** Some LangChain provider adapters raise `TypeError` when the model response cannot be coerced into `output_model` via JSON-schema-mode tool calling. The `except TypeError` at line 40 retries with the same messages and a stripped (`include_raw=False`) signature. The DSML markup is still in the message history, the same LLM is still used, no directive is appended — a second failure is overwhelmingly likely and the `TypeError` bubbles to the caller.
2. **`parsed=None` path** (not covered by the current pinning test). LangChain's `with_structured_output(..., include_raw=True)` is specified to return `{"raw": AIMessage, "parsed": None, "parsing_error": <Exception>}` on parse failure rather than raising. In that case `raw_result["parsed"]` is `None`, `_call_structured` returns `(None, <usage>)`, and the caller in `_tool_loop.py:382` unpacks `parse_result = None`. The framework then writes `None` into the state bus as if it were a valid `output_model` instance. No DSML detection, no retry, no warning. **This is the silent variant of the same bug.**

### Wave D / Wave E impact assessment

The architectural sprint did **not** fix this bug:

- Commit `63ada61 refactor: extract _tool_loop.py from _llm.py` moved the code from `_llm.py` to `_tool_loop.py` but preserved the strategy gate verbatim.
- Commit `727d1c9 fix: DSML trailing tool-call markup recovered via targeted retry` (the original recovery patch) added the rescue inside the existing `if strategy in ("json_mode", "text")` branch — that branch already existed before the patch.
- Commit `efdca9f Wave E: neograph-8ne2 — _llm.py split` extracted `_llm_dispatch.py` and `_llm_retry.py` but kept `_tool_loop.py`'s gate intact. The new `_call_structured` in `_llm_dispatch.py` was carved out unchanged from the prior inline code; no DSML hook was added.

The bug is **literally the same code shape** filed on 2026-04-23, now living in a different file path.

### Test that pins the gap

`tests/modes/test_llm_internals.py:2600-2691` is a passing characterisation test that documents and locks in the parity gap. Excerpt:

```text
2601  class TestDSMLInStructuredStrategyPath:
2602      """neograph-bxxf (axis 6): structured strategy has NO DSML recovery.
...
2687      assert structured_calls[0] == 2, (
2688          "with_structured_output().invoke() must fire EXACTLY twice — once "
2689          "for the include_raw=True attempt, once for the except-TypeError "
2690          f"compat fallback. Got {structured_calls[0]}."
2691      )
```

This test must be updated (or replaced) as part of the fix — it currently asserts the buggy behaviour as a contract.

---

## Section 2: Reproduce the bug

### Spec: TDD-red repro

The TDD-red repro reuses the same fake-LLM scaffolding as the existing `TestDSMLTrailingToolCallRecovery.test_dsml_markup_retried_with_targeted_directive` (lines 1941-2005) — same tool-call → DSML-final → recovery shape — but with `output_strategy="structured"`. After the fix, the structured path must match the json_mode path.

**Pipeline configuration:**
- `output_model = Answer` (single string field `answer`)
- One registered tool: `search` (read-only string echo)
- `tool_budget = 1`
- `llm_config = {"output_strategy": "structured"}`

**Fake LLM behaviour (`invoke()` driven by a `call_count` counter):**

| Call | Path | Content |
|------|------|---------|
| 1 | `bind_tools(...).invoke()` | `AIMessage(tool_calls=[{"name":"search","args":{"q":"test"},"id":"c1"}])` (forces one ReAct iteration) |
| 2 | `bind_tools(...).invoke()` | `AIMessage(content="<｜DSML｜tool_calls>...<｜DSML｜invoke name=\"search\">...</｜DSML｜tool_calls>")` (final response after budget exhaustion) |
| 3+ | `with_structured_output(Answer).invoke()` or `llm.invoke()` (depends on chosen fix path) | `AIMessage(content='{"answer": "recovered"}')` (or `Answer(answer="recovered")` for `with_structured_output`) — produced under the targeted DSML directive |

The fake's `with_structured_output(model, include_raw=True)` must return one of these on the first call (representing the "parsed = None / TypeError" path) so the test can decide which silent-failure variant it exercises:

- **Variant A — TypeError**: raise `TypeError("Expected Answer but got non-JSON content: ...")` (matches the existing `test_structured_path_has_no_dsml_recovery_when_provider_returns_dsml` shape).
- **Variant B — parsed=None**: return `{"parsed": None, "raw": AIMessage(content=DSML), "parsing_error": Exception("...")}` to exercise the silent path.

Both variants must surface the same behaviour after the fix: detect DSML in `raw_result["raw"].content` (Variant B) or in the message history (Variant A), append the `budget_exhausted_message` directive, retry, parse, return `Answer(answer="recovered")`.

**Assertions for the red test (current behaviour — must fail after fix):**
- Variant A: `pytest.raises(TypeError)`; `structured_calls == 2`; never receives a valid `Answer`.
- Variant B: returned value is `None`; no `trailing_tool_call_markup` log event observed.

**Assertions for the green test (post-fix behaviour):**
- `parsed.answer == "recovered"`
- `len(interactions) == 1` (the one successful tool call before DSML)
- `caplog`-captured log line with `trailing_tool_call_markup` event
- Number of fake `invoke()` calls matches the recovery path's expected count (3 for Variant A: tool call + DSML + retry; 3 for Variant B: tool call + structured failed + targeted retry)
- For Variant B specifically: the post-fix path must not have left `parsed=None` written to the state bus

---

## Section 3: Root cause

### Why the gate exists

The gate at `_tool_loop.py:354` predates the DSML patch. Git history:

```text
63ada61 refactor: extract _tool_loop.py from _llm.py  (creates the strategy-aware split)
727d1c9 fix: DSML trailing tool-call markup recovered via targeted retry
```

Inspecting `727d1c9 -- src/neograph/_tool_loop.py` shows the DSML rescue was bolted onto the *existing* `except ExecutionError:` branch — and that branch existed only inside the json_mode/text arm because `_parse_json_response` only runs there. The structured arm uses `with_structured_output` which doesn't raise `ExecutionError`; it raises `TypeError` (sometimes) or returns silently with `parsed=None` (others).

So the rationale was **not a defensive decision** about whether structured-mode users want DSML recovery. It was a **scope artefact**: the author had a hook (the `except ExecutionError`) available in the json_mode arm and added recovery inline. The structured arm had no equivalent hook because its failure modes are heterogeneous (`TypeError`, `parsed=None`, validation errors hidden inside provider adapters, etc.).

### Commit `727d1c9` rationale

From the commit message:

> When the model emits tool-call XML markup in the final response (after tool budget exhaustion), the generic retry message didn't address the root cause. Now detects DSML/XML markup and retries with a targeted directive telling the model tools are exhausted.

The motivating bug was about DeepSeek R1 via OpenRouter emitting `<｜DSML｜function_calls>...</｜DSML｜function_calls>` in `message.content` after running out of tool budget. **Nothing in that motivation is strategy-specific.** The same model can be configured with `output_strategy="structured"` and exhibit the same DSML-emit behaviour — and would, since the strategy controls how neograph invokes the model, not how the model decides to express its response.

### Verdict

The gate is an oversight, not a defensive choice. There is no documented or implied reason that structured-strategy users should be opted out of DSML recovery.

---

## Section 4: Proposed fix shapes

### Option A — extend the gate

Change line 354 to `if strategy in ("json_mode", "text", "structured")`, push the DSML detection inside `_call_structured` (since the failure shapes are different there), or split the rescue into a shared helper.

**Implementation sketch:**

1. In `_call_structured`, on `TypeError` or `parsed=None`, extract the raw response text (from `messages[-1]` or `raw_result["raw"].content`), run the DSML regex, and on hit, do the targeted-retry directive append + re-invoke via `_invoke_json_with_retry`.
2. Leave the json_mode/text branch in `_tool_loop.py` untouched.

**Complexity:** Low. ~30 lines added to `_call_structured`. Duplicates the regex check.
**Risk:** Two regex sites drift apart over time (one in `_tool_loop.py:362`, one in `_llm_retry.py:163`, and now a third in `_call_structured`).
**Future flexibility:** Low. Adding a fourth strategy means another inline DSML check.

### Option B — strategy-agnostic recovery (recommended)

The DSML detection is about **response content** (markup pattern). The strategy is about **invocation mechanism** (with_structured_output vs raw invoke + parse). Decoupling them makes the boundaries match the actual semantics.

**Implementation sketch:**

1. Extract a free function in `_llm_retry.py`:

   ```text
   def _attempt_dsml_recovery(llm, messages, output_model, cfg, config, raw_text) -> BaseModel | None:
       """If raw_text contains DSML/XML tool-call markup, append directive
       and re-invoke as raw JSON. Returns parsed model or None if no DSML
       was detected. Raises ExecutionError if retry also fails."""
   ```

2. In `_tool_loop.py` after the ReAct loop completes, **before** the strategy dispatch, peek at `messages[-1].content`. If DSML is detected, run `_attempt_dsml_recovery` directly. On success, return that result and skip strategy dispatch entirely. On no-detection, fall through to the normal dispatch.
3. In `_call_structured` (`_llm_dispatch.py`), on `TypeError` or `parsed=None`, also call `_attempt_dsml_recovery` against the raw response — this catches the case where the LLM's final non-tool message in the ReAct loop *passes* DSML detection but the structured adapter still chokes on it for adapter-internal reasons.

**Complexity:** Medium. ~50 lines added across `_llm_retry.py` and `_tool_loop.py`. One regex site, one rescue function.
**Risk:** Pre-strategy DSML peek changes the order of operations. If the structured adapter would have accepted the DSML content (it shouldn't — but provider adapters do surprising things), Option B skips it. Mitigation: only run the pre-dispatch peek when the regex actually matches, which is the same gating that the current json_mode branch uses.
**Future flexibility:** High. Adding a new output_strategy never touches DSML logic. DSML detection lives in one place. The `parsed=None` silent path is naturally caught.

### Option C — strategy-specific opt-in hook

Each strategy declares (via a method, attribute, or function lookup) whether DSML recovery applies. `"structured"` opts in by default; new strategies pick.

**Implementation sketch:**

1. Add a dispatch table or strategy-protocol method:

   ```text
   STRATEGY_DSML_RECOVERY = {"structured": True, "json_mode": True, "text": True}
   ```

2. Wire it through `_tool_loop.py` and `_call_structured`.

**Complexity:** High. Requires designing the opt-in mechanism + threading it through dispatch + per-strategy adapter implementation.
**Risk:** Over-engineering. There is no current evidence that any future strategy would *not* want DSML recovery.
**Future flexibility:** Highest. Pays off only if we add many strategies with divergent failure semantics.

### Recommendation: Option B

- The detection (`re.search(r"<[^>]*(?:function_call|invoke|DSML)[^>]*>", ...)`) is content-based and strategy-orthogonal — that's the simplest semantic carving.
- Option A duplicates the regex and the directive-append code; Option C designs a hook we don't need yet.
- Option B also catches the `parsed=None` variant for free, because the pre-dispatch peek runs against `messages[-1].content` which is set during the ReAct loop regardless of strategy.
- Refactor cost is contained: one helper, one re-shape in `_tool_loop.py`, no public API change.

---

## Section 5: Test plan

TDD red-then-green. Tests live in `tests/modes/test_llm_internals.py` next to the existing `TestDSMLTrailingToolCallRecovery` and `TestDSMLInStructuredStrategyPath` clusters.

### New tests (must pass after fix)

| Test | Strategy | LLM behaviour | Asserts |
|------|----------|---------------|---------|
| `test_dsml_recovery_in_structured_strategy_with_typeerror` | `"structured"` | call 1 = tool call; call 2 = DSML AIMessage; `with_structured_output().invoke()` raises `TypeError`; targeted retry returns valid JSON | `parsed.answer == "recovered"`; `trailing_tool_call_markup` log fired; ≥1 tool interaction |
| `test_dsml_recovery_in_structured_strategy_with_parsed_none` | `"structured"` | same but `with_structured_output(..., include_raw=True).invoke()` returns `{"parsed": None, "raw": AIMessage(content=DSML), "parsing_error": ...}`; targeted retry returns valid model | `parsed.answer == "recovered"`; result is not None; warning fired |
| `test_structured_strategy_does_not_misfire_dsml_when_response_is_valid` | `"structured"` | clean tool call → valid model response (no DSML anywhere) | only one `with_structured_output().invoke()` call; no `trailing_tool_call_markup` log; no extra `llm.invoke()` |

### Existing tests (must still pass)

| Test | Path |
|------|------|
| `TestDSMLTrailingToolCallRecovery.test_dsml_markup_retried_with_targeted_directive` | `tests/modes/test_llm_internals.py:1941` (json_mode + DSML — must remain green) |
| `TestDSMLTrailingToolCallRecovery.test_custom_budget_exhausted_message` | `:2007` (json_mode + custom message) |
| `TestDSMLDoubleFailure.test_double_dsml_falls_through_to_generic_retry` | `:2082` (json_mode double-failure → generic retry) |
| `TestDSMLAllRetriesFail.test_exhausted_retries_raise_execution_error_with_dsml_hint` | `:2149` (json_mode all retries fail) |
| `TestNonDSMLParseFailureTakesGenericRetry` | `:2220` (json_mode non-DSML parse failure → generic retry; must not gain a false positive in structured mode) |
| `TestDSMLInStructuredStrategyPath.test_structured_path_happy_baseline_no_dsml` | `:2694` (structured happy path — must remain green) |

### Existing tests that must be updated or replaced

| Test | Action |
|------|--------|
| `TestDSMLInStructuredStrategyPath.test_structured_path_has_no_dsml_recovery_when_provider_returns_dsml` (`:2617`) | **Delete or invert.** This test currently locks in the bug. After the fix it must either be removed or rewritten to assert the new behaviour (DSML rescue fires once, returns valid model). The class docstring at `:2601-2614` ("structured strategy has NO DSML recovery... documented parity gap vs json_mode/text") must also be removed. |
| `TestDSMLAfterMaxIterationsGuard` (`:2763` — neograph-tdp3) | **Verify still passes.** This is json_mode-only by construction; no change expected. |

### Coverage check after fix

- `_call_structured` (or its replacement / helper): exercised by the two new structured-mode DSML tests + the existing happy-baseline test.
- Shared `_attempt_dsml_recovery` helper (if Option B): exercised by the four scenarios across json_mode and structured.
- Three-surface parity is **not in scope** — this is a runtime LLM-invocation bug, not an IR-shape bug. Same DSML rescue applies to `@node` mode='act'/'agent', declarative, and programmatic surfaces uniformly because they all funnel through `invoke_with_tools`.

---

## Section 6: Risks

### Why "structured" strategy might *not* want DSML recovery

I see no defensible reason. Arguments and rebuttals:

| Argument against | Rebuttal |
|------------------|----------|
| "structured strategy uses provider native tool calling; DSML implies the model isn't using native tool calling, so there's a deeper mismatch" | DSML emission is downstream of the model's own decoding. Provider adapters expose tool-calling via JSON-schema prompting, JSON Mode, or function-call APIs — but the model can still emit DSML in `content` if it's a DeepSeek R1 / similar that does its own markup. The strategy is about how we *ask*, not how the model *replies*. |
| "If `with_structured_output` says it failed, we should trust the adapter rather than scraping markup" | The current `except TypeError` path already doesn't trust the adapter — it retries unblinded. We're already in fallback territory; the question is only whether the fallback is informed (DSML directive) or naive (same prompt, same LLM). |
| "Backward-compat for callers who depend on TypeError bubbling" | The current pinning test was added to *document* a gap, not because callers depend on it. Risk is low; the bug is filed P2 because it bites real users. |

### Tests that might break under the new behaviour

- **`test_structured_path_has_no_dsml_recovery_when_provider_returns_dsml`** (`:2617`) — explicitly. Must be removed or inverted.
- Any test elsewhere that pins `structured_calls[0]` to a small number after returning DSML. Search confirms there is exactly one (`:2687`).
- Tests that count `llm.invoke` calls in the structured branch may need their counts updated if the DSML rescue fires.

### Three-surface parity

Not applicable — this bug lives in the runtime LLM invocation layer (`_tool_loop.py`, `_llm_dispatch.py`), not in any IR construction layer. All three surfaces (declarative / `@node` / `ForwardConstruct`) compile to the same factory dispatch that calls `invoke_with_tools` for agent/act modes. The fix automatically applies uniformly across surfaces.

### Possible regressions worth checking

- Models that emit DSML *and* successfully produce valid JSON in the same response: the regex currently fires only on `_parse_json_response` failure, but Option B's pre-dispatch peek would fire on any DSML match. **Mitigation:** keep the `_parse_json_response`-fails-first ordering — only run DSML rescue when the strategy's native parse path has already raised or returned None. This is the same gating as the current json_mode branch.
- The new tests must use the same `register_tool_factory` / `build_fake_runtime` helpers as the existing DSML cluster to avoid drift in test scaffolding.

---

## Section 7: Estimated effort

| Step | Time |
|------|------|
| Write the two new red repro tests (TypeError variant + parsed=None variant) | 0.5 h |
| Verify red — both tests fail against current code | 0.25 h |
| Implement Option B helper (`_attempt_dsml_recovery` in `_llm_retry.py`) | 1.5 h |
| Wire helper into `_tool_loop.py` (pre-dispatch peek or post-failure rescue) | 1 h |
| Wire helper into `_call_structured` (TypeError + parsed=None branches) | 0.75 h |
| Adjust the existing pinning test (`test_structured_path_has_no_dsml_recovery_when_provider_returns_dsml`) — remove or invert | 0.25 h |
| Add the `test_structured_strategy_does_not_misfire_dsml_when_response_is_valid` safety test | 0.25 h |
| Run full DSML cluster + obligation tests + lint + mypy | 0.5 h |
| Update CLAUDE.md / docs entry if the parity gap is documented anywhere | 0.25 h (only if needed; see Section 8) |

**Total: ~4.5–5.5 hours** for Option B. Option A is roughly 1 hour less but accrues maintenance debt. Option C is roughly 4 hours more for the dispatch-table design.

---

## Section 8: Open questions for maintainer

1. **Was the parity gap an intentional choice or an oversight?** Section 3 concludes oversight based on commit archaeology. If there's a non-archival reason (e.g., a provider where the structured-mode rescue would misfire that didn't make it into commit messages), please surface it before Option B is implemented.
2. **`parsed=None` variant: in scope or follow-up ticket?** The current ticket text describes the `TypeError` variant. The `parsed=None` variant is a related but distinct silent-failure mode. Recommendation: include in this fix because Option B catches both for the same cost. If the maintainer prefers narrower scope, file the parsed=None case as a follow-up under a new ticket.
3. **DSML regex placement.** Current regex `r"<[^>]*(?:function_call|invoke|DSML)[^>]*>"` is defined twice: `_tool_loop.py:362` and `_llm_retry.py:163`. Option B's helper would consolidate into one. Should the helper be public-ish (importable from elsewhere) or stay private with `_` prefix? Recommendation: private, internal helper; not part of the framework API surface.
4. **Logging contract.** The current `trailing_tool_call_markup` log event is structlog-only and json_mode-only. Should the same event fire for the structured-strategy rescue? Recommendation: yes, same key, so downstream observability dashboards work uniformly. Add a `strategy` field to the bind so operators can filter.
5. **Should `weak-point-map.md:92` (the "R1 XML after budget" row) be updated post-fix?** That row currently cites `neograph-irv3` only; the fix should also cite `neograph-0tid` and the closed companion tickets (`bxxf`, `tjhe`, `44eq`, `gxv8`).

---

## Citations summary

| File | Lines | What |
|------|-------|------|
| `src/neograph/_tool_loop.py` | 349-382 | Strategy gate around DSML recovery (the bug) |
| `src/neograph/_tool_loop.py` | 354 | The literal `if strategy in ("json_mode", "text"):` |
| `src/neograph/_tool_loop.py` | 360-373 | DSML detect + targeted retry (json_mode/text only) |
| `src/neograph/_tool_loop.py` | 382 | Else-arm dispatch to `_call_structured` (no DSML hook) |
| `src/neograph/_llm_dispatch.py` | 29-43 | `_call_structured` — no DSML handling; only `except TypeError` |
| `src/neograph/_llm_dispatch.py` | 32-39 | `include_raw=True` path that silently returns `parsed=None` |
| `src/neograph/_llm_retry.py` | 159, 163 | Second DSML regex site (inside `_parse_json_response`) |
| `src/neograph/_llm_config.py` | 36 | `output_strategy` default = `"structured"` (why this bug bites the default) |
| `src/neograph/_llm_config.py` | 49-60 | `resolved_budget_exhausted_message` — the directive Option B would reuse |
| `tests/modes/test_llm_internals.py` | 2600-2691 | Test that pins the parity gap as documented behaviour |
| `tests/modes/test_llm_internals.py` | 1941-2005 | Reference shape for the new structured-mode test |
| `docs/design/weak-point-map.md` | 92 | "R1 XML after budget" entry; update post-fix |
| git `727d1c9` | — | Original DSML patch; nested rescue inside existing except branch |
| git `efdca9f` | — | Wave E split; preserved gate unchanged |
| git `63ada61` | — | `_tool_loop.py` extraction; preserved gate unchanged |
