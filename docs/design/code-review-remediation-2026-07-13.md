# Code Review Remediation — Architecture Decision (2026-07-13)

**Spike**: neograph-d149a. **Input**: `.claude/code-review/130726_1121/synthesis.md` (authoritative) + 5 per-agent reports. **Deliverable**: this document. No source touched; no tickets filed.

## Context

The 130726 review returned zero Critical and zero High findings across five dimensions. Its single load-bearing conclusion is a meta-observation about *how* neograph stays clean, not a defect list: **every convention backed by a `test_guards_*.py` structural guard held perfectly under independent audit; the only drift found anywhere in `src/` is against the one hard rule that has no guard** (the bare `structlog.get_logger()` convention). The remediation is therefore not "fix N lines" — it is "close the guard gaps so the drift class becomes structurally impossible," applied to four patterns of decreasing strength: PAT-01 (documented-only convention drifted, 2 sites + a level bug), PAT-03 (a fail-loud policy re-implemented 3× with one site bypassing the canonical reader), PAT-04 (test assertions weaker than the property their name claims, ~6 sites in an unbounded population), and PAT-02 (a guard heuristic narrower than the rule it enforces — not exploited, the precursor shape of PAT-01).

I verified the mechanism assumptions against the tree before designing. Load-bearing confirmations: `tool.py` already imports `from neograph.di import RESOURCE_FETCHER_KEY, parse_resource_content` (line 35) and `di.py` does **not** import `tool.py`, so relocating the fetcher guard into `di.py` and importing it from `tool.py` is layer-legal and cycle-free (PAT-03). The guard suite uses a consistent AST-scanner-plus-meta-test idiom (`test_guards_helper_monopoly.py`, `test_guards_consistency.py`), and `test_guards_meta.py::TestRegexGuardsHaveSlipMetaTests` mandates that any `re` pattern in a guard module be a **named constant with a `slip` meta-test** — so the new logging guard is specified as **pure-AST** (no `re`), which is exempt by construction and mirrors the majority of existing guards. `_dev_warnings.DEV_MODE` exists (`NEOGRAPH_DEV=1`) for PP-01. Every guard in this doc follows the project's **guard-first discipline**: the guard is written FIRST, must FAIL on today's tree (proving it detects the live drift), then the fix turns it green.

---

## PAT-01 — Bare-logger convention has drifted (unguarded) and carries a level bug

**Invariant.** Every module in `src/neograph/` that logs binds its logger exactly once, at module scope, with the bare zero-argument call `log = structlog.get_logger()`; no module passes an argument to `get_logger` and none resolves a logger inline inside a function body. (Consequence: the only permitted `get_logger` token in `src/neograph/*.py` is the bare module-level binding.)

**Guard.** New file `tests/test_guards_logging.py`, `TestBareModuleLoggerConvention`. **Pure-AST** (deliberately no `re`, to stay exempt from the meta-guard's named-regex discipline and match the `test_guards_consistency.py` template). Detection strategy, per `src/neograph/**/*.py`:
- Walk the AST for every `ast.Call` whose func resolves to `get_logger` (either `ast.Name` `get_logger` or `ast.Attribute` `.get_logger` on a `structlog` receiver).
- **Flag** if the call has any positional or keyword args (bans `get_logger(__name__)`, `get_logger("neograph")`).
- **Flag** if the call is not bound at module level to a `Name` target (bans the inline `structlog.get_logger(...).error(...)` form — detected by: the `get_logger` call's nearest statement ancestor is not a module-scope `Assign` to a simple `Name`). The simplest robust implementation: collect all module-level `Assign` nodes whose value is a bare `get_logger()` call into an allowset, then flag every *other* `get_logger` call node in the tree.
- Anti-vacuity: three meta-tests on synthetic sources — positive (`log = structlog.get_logger(__name__)` flagged), positive (`structlog.get_logger("neograph").warning(...)` inline flagged), negative (`log = structlog.get_logger()` at module scope clean). Because this is AST not text, no regex-slip case is required (mirrors `TestComboMapMonopoly`).

The guard must FAIL first — it will flag `_construct_validation.py:243` and `_llm_retry.py:370` on today's tree.

**Mechanism.** Two sites, mechanical:
- `_construct_validation.py`: add module-level `log = structlog.get_logger()` next to the imports; delete the function-local `import structlog` (line 241); replace the inline call at 243 with `log.warning("loop_skip_when_no_skip_value", ...)`. **Note the level change from `.error` to `.warning`** — this folds MED-03/CON-02 into the same touch: the surrounding comment says "Warn rather than reject," the path does not raise, and the analogous Loop advisory in `lint.py` (`loop_condition_none_unsafe`) is a WARN. An advisory logged at ERROR is a spurious ERROR in a consumer's observability for a pipeline that compiles and runs fine.
- `_llm_retry.py`: add module-level `log = structlog.get_logger()`; delete the function-local `import structlog` (line 368); replace the inline call at 370 with `log.warning("trailing_tool_call_markup", ...)` (already `.warning`, no level change).

**Anti-band-aid audit.** After the fix, `grep -rn "get_logger" src/neograph/` must return only module-level bare bindings (one per logging module), and the new guard must pass. The guard *is* the permanent audit — it makes the population exact. As a one-time sweep confirmation: `grep -rn "import structlog" src/neograph/` should show no function-local imports remain (both were tied to the two drift sites).

**Tickets.** Single ticket, guard-first (the fix is 2 sites and the level change is causally bound to the same lines — splitting would be churn). See T1 in the consolidated list.

---

## PAT-02 — Engine-verb guard heuristic is narrower than the rule (not exploited)

**Invariant.** The three-layer engine-verb guard classifies a Runnable-shared verb (`invoke`/`ainvoke`/`stream`/`astream`) as an engine call whenever its receiver is a compiled graph — *including* a compiled graph bound to a name that does not contain the token `graph` (e.g. `g = compile(...); g.invoke(...)`). No compile-layer module may schedule engine execution and evade detection by receiver naming.

**Guard.** This is a *hardening of an existing guard*, `tests/test_guards_three_layer.py::_receiver_is_graphlike` (line ~198), not a new file. Two-part, and the honest trade-off matters here:
- The **real backstop already exists and already holds**: `EXPECTED_ENGINE_SURFACE` is an exact-match allowlist of every engine call site, so a new engine call forces a conscious edit there regardless of receiver name. The synthesis verified the blind spot is *not currently exploited* (the only non-`graph`-named `.invoke`/`.ainvoke` receivers in the tree — `caller`, `payload` in `_agent_cycle.py` — are bound chat models, correctly Layer-2).
- The heuristic gap is nonetheless a latent PAT-01 (a detector covering a syntactic subset of its semantic rule). Close it by teaching `_receiver_is_graphlike` (or `_scan_engine_verbs`) to also treat a receiver whose binding site is `compile(...)` / a `CompiledStateGraph` as graph-like, **and** — this is the load-bearing part per the project's anti-vacuity discipline — add an **evasion meta-test**: a synthetic source binding a compiled graph to an innocent name (`g = compile(...); g.invoke(state, cfg)`) inside a compile-layer module must be flagged. The meta-test is what converts "we broadened a heuristic" into "the broadening is pinned."

**Mechanism.** Prefer the receiver-provenance approach: within a scanned module, collect names assigned from a `compile(...)` call (`ast.Assign` whose value is a `Call` to `compile`), and treat `.invoke`/`.astream`/etc. on those names as engine calls even without a `graph` token. This is local, per-module AST — no cross-module dataflow needed, because the disease (compile *and* invoke in the same compile-layer module) is intra-module by definition. If provenance tracking proves fiddly, the acceptable floor is: keep the `graph`-token heuristic, but add the evasion meta-test documenting that `EXPECTED_ENGINE_SURFACE` is the true backstop and the heuristic is a fast-path only. Blunt call: do the provenance broadening — it is ~15 lines of AST and removes the latent gap entirely.

**Anti-band-aid audit.** `grep -rnE "= *compile\(" src/neograph/*.py` cross-referenced against `.invoke`/`.ainvoke`/`.stream`/`.astream` receivers confirms no compile-and-invoke pair exists outside the allowlisted engine surface (`runner.py`, `_compiled.py`, `_subconstruct.py`). The evasion meta-test proves the guard would catch a future one.

**Tickets.** Single small ticket, T2. Parallelizable with everything else (touches only `test_guards_three_layer.py`).

---

## PAT-03 — Resource-fetcher fail-loud policy re-implemented 3× (one site bypasses the canonical reader)

**Invariant.** Reading the consumer-owned resource fetcher from `config['configurable'][RESOURCE_FETCHER_KEY]` and failing loud when it is absent is expressed exactly once, in `di.py`, via a single helper and a single hint constant. No `src/neograph/` site hand-rolls the configurable read for the fetcher, and the fetcher-signature hint string (`"an async 'fetch(uri) -> (content, mime)' callable"`) appears in exactly one place.

**Guard.** Add to `tests/test_guards_helper_monopoly.py` (this is precisely its charter — it already pins `di._unwrap_loop_value` and friends). Two checks, mirroring the existing monopoly idiom:
- **Hint monopoly** (`_normalized_idiom_count`): the whitespace/quote-normalized hint literal `an async 'fetch(uri) -> (content, mime)' callable` appears **exactly once** across `src/neograph/**/*.py` (the `_FETCHER_HINT` definition in `di.py`). This catches the copy-paste directly and reuses the exact scanner (`_normalized_idiom_count`) that already resists spacing/quote slips, with the existing regex-slip meta-test pattern.
- **Reader monopoly** (AST): `tool.py` must not hand-roll `(config or {}).get("configurable", {}).get(...)` for the fetcher — i.e. `_resolve_fetcher` must call `_require_fetcher` (assert `_count_calls(tool.py source, "_require_fetcher") >= 1` and that the hand-rolled `.get("configurable", {})` chain no longer appears in `tool.py`). Simplest durable form: assert the `_FETCHER_HINT` literal does not appear in `tool.py` at all (it now lives only in `di.py`), which transitively forces the delegation.
- Anti-vacuity: positive meta-test (two copies of the hint → count 2), negative meta-test (single definition → count 1), and the whitespace-slip meta-test the module already templates.

Guard fails first: today the hint appears 3× and `tool.py` hand-rolls the read.

**Mechanism.** In `di.py`, next to `_get_configurable`:
```
_FETCHER_HINT = (
    "provide config['configurable']['{key}'] = "
    "an async 'fetch(uri) -> (content, mime)' callable"
)  # formatted with key=RESOURCE_FETCHER_KEY at the raise site

def _require_fetcher(config, *, subject: str, required: bool = True) -> Callable | None:
    """Read the consumer-owned resource fetcher from config; fail loud when
    absent (unless required=False, which returns None). subject names the
    consumer for the lead message ('resource ref ...', 'resource tool ...',
    'resource DI parameter ...')."""
    fetcher = _get_configurable(config, RESOURCE_FETCHER_KEY)
    if fetcher is None:
        if not required:
            return None
        raise _ConfigurationError.build(subject, hint=_FETCHER_HINT.format(key=RESOURCE_FETCHER_KEY))
    return fetcher
```
The `required` parameter is what lets **all three** sites collapse onto one helper despite their one behavioral difference:
- `di.py:189` (`hydrate_resource_ref`) → `_require_fetcher(config, subject=f"resource ref '{...}' has no fetcher to read from")` (required=True default).
- `di.py:433` (`aresolve` FROM_RESOURCE) → the optional-param case: `fetcher = _require_fetcher(config, subject=f"resource DI parameter '{self.name}' has no fetcher to resolve from", required=not_… )` — pass `required=self.required`; the `not required → return None` short-circuit now lives inside the helper, removing the duplicated read.
- `tool.py:323` (`_resolve_fetcher`) → `return _require_fetcher(config, subject=f"resource tool '{tool_name}' has no resource fetcher to call")`. **Import legality confirmed**: `tool.py` already imports from `di.py`; add `_require_fetcher` to that import. No new dependency edge, no cycle (`di.py` does not import `tool.py`).

The subject (lead message) stays per-site — it is genuinely different information (which consumer failed). Only the *hint* (the fetcher's signature contract) and the *read semantics* are shared, which is exactly the drift surface DRY-01 identified.

**Anti-band-aid audit.** `grep -rn "fetch(uri) -> (content, mime)" src/neograph/` returns exactly one live site (the `_FETCHER_HINT` constant; the two docstring mentions in `hydrate_resource_ref`'s prose are prose, not the hint and may stay or be left as-is). `grep -rn 'get("configurable"' src/neograph/tool.py` returns nothing. The two guard checks pass.

**Tickets.** Single ticket, T3, guard-first. Not parallelizable with T4-of-DRY-nothing but independent of T1/T2/T5+.

---

## PAT-04 — Test names claim stronger properties than their assertions verify

**Invariant.** A test whose name or docstring asserts a specific behavioral property (runs *end-to-end through the tool*, *interrupts*, *is the same object as its source*) contains at least one assertion that fails if that specific property regresses — never a lone `assert <result> is not None` as the sole assertion, and never an `assert A or B` where one disjunct is trivially/always true.

**Guard.** This population is *unbounded* (any future test can regrow the shape), so the honest answer is a **detector guard plus a bounded sweep**, not just 6 edits. Two parts:

1. **Sweep methodology** (one-time, bounds today's population). Hunt two syntactic signatures across `tests/`:
   - **Lone-non-null**: a test function whose *only* `assert` statement is `assert <name> is not None` (or `is not None` with no other assert in the body). Grep seed: `grep -rn "is not None" tests/` then filter to functions with a single assert. Excludes fake-internal guards (`self._model is not None`) — those are not test assertions. The testing report already enumerated the population as ~42 `is not None` sites, *mostly benign fake guards*; the sweep's job is to separate the ~4 hollow-test sites from the benign guards.
   - **Always-true disjunct**: `assert A or B` where B is `X.get(...) is not None` / `X is not None` guarding a value that is produced whenever the graph completes. Grep seed: `grep -rn "assert .* or .*is not None" tests/`.
   - For each hit, the classifier is human/agent judgment: does the *name* claim a property the assertion cannot detect regressing? If yes → strengthen. If the test is a legitimate crash-only fuzz test (Hypothesis `@given` — the *execution* is the assertion), either strengthen to a real invariant (`isinstance(result, str)`, non-empty) or rename to make crash-only intent explicit and drop the no-op assert.

2. **What a strengthened assertion must pin** (per known site):
   - `tests/test_mcp_tools.py:194` (MED-01, the worst) — assert the *output value* `result["scan"] == Claims(items=["done"])`, and to pin *tool invocation* specifically, use a `FakeTool("run_search")` (records `.calls`) or a `StructuredTool` closing over a call-recording list, then `assert tool.calls == [{"query": "q"}]`. This is the one site that can currently pass with the tool-dispatch turn skipped entirely.
   - `tests/modes/test_execution.py:584` (TQ-02) — drop the disjunct: on first run `assert "__interrupt" in str(result)`, or assert the state snapshot's `.next` is non-empty. The interrupt-fired property is the one thing currently under-verified.
   - `tests/hypothesis/test_rendering_invariants.py:97,114` (TQ-03) — strengthen to `isinstance(result, str)` / non-empty, or rename to `*_never_raises` and drop the no-op.
   - `tests/test_public_fakes.py:120-122` (TQ-04) — assert object *identity* against the internal source (`from neograph.testing.fakes import StructuredFake as _S; assert StructuredFake is _S`) so it guards the "one implementation, two consumers" contract, not mere importability.

**Guard (optional, recommended-floor).** A pure-AST detector in a new `tests/test_guards_assertion_strength.py` that flags any `test_*` function whose sole assertion is `assert <Name/Subscript> is not None`, with an **allowlist** of legitimate crash-only/import-smoke tests (seeded from the sweep's benign classifications). This is a ratcheting guard in the project's existing style — the allowlist may only shrink. Trade-off: this guard has a higher false-positive rate than the monopoly guards (some lone-non-null asserts are legitimately all a test can say), so it ships with an allowlist rather than a zero-tolerance assertion. If the allowlist maintenance cost is judged not worth it for a sole maintainer, the fallback is sweep-only (T4a) and *decline* the standing guard (T4b) with the reason recorded. **Recommendation: do the sweep (T4a) now; ship the guard (T4b) only if the allowlist lands under ~15 entries** — above that it is noise.

**Anti-band-aid audit.** After the sweep: `grep -rn "is not None" tests/ | grep -i "assert"` reviewed to confirm every remaining lone-non-null is either a fake-internal guard or an allowlisted crash-only test; `grep -rnE "assert .+ or .+is not None" tests/` returns only disjunctions where both branches are meaningful.

**Tickets.** T4a (sweep + strengthen the ~6 known sites, TDD-adjacent: each strengthened assert must fail if you mutate the production path it now pins). T4b (standing guard, conditional per the recommendation above). T4a is parallelizable per-file if desired but small enough for one pass.

---

## Low-findings disposition

| ID | Finding | Disposition | Rationale |
|----|---------|-------------|-----------|
| DRY-02 | `normalize_outputs(_declared_output(x)).primary` chain 11× in `forward.py` | **Fold into T3** (or own trivial ticket T6) | Extract `_primary_type(item)` module-private helper in `forward.py`; replace 11 call sites. Not a monopoly *violation* (the primitives are authoritative) but an un-named repeated composition. Worth checking whether `forward.py`'s tracer should share `_dispatch._resolve_primary_output` (the richer form with Oracle override) rather than bare `.primary` — flag for the implementer, don't pre-decide. Low blast radius, do it. |
| DRY-03 | Fingerprint `sha256(f"{name}:{_type_signature(typ)}")[:12]` on 2 adjacent lines in `state.py` | **Fix now (fold into T6)** | One-line local closure `_fp(name, typ)` at top of `compute_node_fingerprints`, called from both branches. The `:12` width + `{name}:{sig}` layout is the load-bearing fingerprint contract (CLAUDE.md: both fingerprints move in lockstep) — writing it twice invites exactly the drift the checkpoint machinery cannot tolerate. Trivial, high-value-per-line. |
| CON-03 | `McpToolCallError(Exception)` outside `NeographError` hierarchy | **Decline (record decision)** | Defensible and likely intentional: `neograph_mcp` deliberately imports nothing from `neograph` core (verified zero `from neograph` refs), keeping core MCP-free. Making it inherit `NeographError` introduces exactly that dependency. **Open question for the user** (below) — the only reason to change is if uniform `except NeographError` catchability across both packages is a product goal. |
| PP-01 | CLI swallows traceback on unexpected failure (`__main__.py:240`) | **Fix now (T6)** | Gate `traceback.print_exc()` on the existing `_dev_warnings.DEV_MODE` (`NEOGRAPH_DEV=1`) before the `print(f"FAIL ...")`. Preserves clean default UX, gives maintainers opt-in triage. `DEV_MODE` confirmed to exist. One line. |
| PP-02 | 355 `: Any` annotations repo-wide | **Decline (informational)** | Not a defect. Already governed by `test_guards_any_audit.py` (no-Any in public IR). No action beyond confirming that guard stays green post-change — which the T1-T4 tickets' gate runs anyway. |
| TQ-02 | Interrupt OR-disjunct assertion | **Fold into T4a** | It is a PAT-04 site; the sweep owns it. |
| TQ-03 | Crash-only fuzz no-op non-null asserts | **Fold into T4a** | PAT-04 site. |
| TQ-04 | Import-smoke trivial asserts | **Fold into T4a** | PAT-04 site. |
| LR-03 | Engine-verb receiver heuristic evadable | **Fix now (T2)** | This IS PAT-02; owned by T2. |

`state.py`, `forward.py`, `__main__.py` cleanups (DRY-02, DRY-03, PP-01) are grouped into one "trivial cleanups" ticket **T6** because they are individually sub-ticket-sized, touch three different files (no conflict), and share no invariant worth stating separately. Each carries its own one-line anti-band-aid check.

---

## Consolidated ticket list (dependency order)

All tickets are self-contained (executable without this conversation). Each leads with its invariant and an anti-band-aid clause per project convention. Guard-first tickets state that the guard must FAIL on the pre-fix tree.

**T1 — Guard + fix the bare-logger convention (PAT-01).** *Invariant*: every `src/neograph/` module binds its logger once at module scope via bare `log = structlog.get_logger()`; no argument, no inline resolution. *Anti-band-aid*: do not fix only the two known sites — write the pure-AST guard `tests/test_guards_logging.py` FIRST (it must fail, flagging `_construct_validation.py:243` + `_llm_retry.py:370`), with positive/positive/negative meta-tests; then fix both sites (add module-level `log`, delete function-local `import structlog`, and change `_construct_validation.py:243` from `.error` to `.warning` — the advisory does not raise). Audit: `grep -rn "get_logger" src/neograph/` shows only bare module-level bindings; guard green. *Parallel*: yes (independent files). *Depends on*: none.

**T2 — Harden the engine-verb receiver heuristic (PAT-02 / LR-03).** *Invariant*: the three-layer guard classifies a `.invoke`/`.ainvoke`/`.stream`/`.astream` on a `compile(...)`-bound receiver as an engine call regardless of variable name. *Anti-band-aid*: add an evasion meta-test (`g = compile(...); g.invoke(state, cfg)` in a compile-layer module must be flagged) BEFORE broadening `_receiver_is_graphlike`/`_scan_engine_verbs` to track `compile(...)` provenance; the meta-test is what pins the broadening. Confirm `EXPECTED_ENGINE_SURFACE` exact-match still passes. Audit: `grep -rnE "= *compile\(" src/neograph/*.py` cross-checked against invoke receivers shows no compile-and-invoke outside the allowlisted engine modules. *Parallel*: yes (only `tests/test_guards_three_layer.py`). *Depends on*: none.

**T3 — Monopolize the resource-fetcher fail-loud policy (PAT-03 / DRY-01).** *Invariant*: the fetcher read + fail-loud + signature hint is expressed once in `di.py`; no site hand-rolls the configurable read for the fetcher and the hint literal appears exactly once. *Anti-band-aid*: add the two guard checks to `tests/test_guards_helper_monopoly.py` FIRST (hint `_normalized_idiom_count == 1`; `tool.py` calls `_require_fetcher` and contains no `_FETCHER_HINT` literal) — they must fail on today's 3-copy tree — then introduce `_require_fetcher(config, *, subject, required=True)` + `_FETCHER_HINT` in `di.py` and route all three sites (`hydrate_resource_ref`, `DIBinding.aresolve` FROM_RESOURCE with `required=self.required`, `tool._resolve_fetcher`) through it. Import `_require_fetcher` into `tool.py` (legal: `tool.py` already imports from `di.py`; no cycle). Audit: `grep -rn "fetch(uri) -> (content, mime)" src/neograph/` → one live constant; `grep -rn 'get("configurable"' src/neograph/tool.py` → nothing. *Parallel*: yes. *Depends on*: none.

**T4a — Sweep and strengthen weak test assertions (PAT-04).** *Invariant*: no `test_*` in `tests/` has a lone `assert <x> is not None` as its only assertion (excluding fake-internal guards and allowlisted crash-only fuzz tests), and no `assert A or B` has an always-true disjunct, where the test name claims a property the assertion cannot detect regressing. *Anti-band-aid*: run the two-signature sweep (`is not None`-only-assert; always-true-disjunct) across ALL of `tests/`, not just the 4 named sites; classify each hit (strengthen vs benign-guard vs crash-only). Strengthen the known sites as specified: `test_mcp_tools.py:194` → assert output value + record tool `.calls`; `test_execution.py:584` → drop disjunct, assert interrupt; `test_rendering_invariants.py:97,114` → real invariant or rename+drop; `test_public_fakes.py:120-122` → object-identity assert. Verify each strengthened assert fails under a mutation of the production path it now pins. *Parallel*: per-file if split; small enough for one pass. *Depends on*: none.

**T4b — (Conditional) Standing assertion-strength guard (PAT-04).** *Invariant*: the lone-non-null-only-assert shape cannot silently regrow. *Anti-band-aid*: pure-AST detector in `tests/test_guards_assertion_strength.py` flagging any `test_*` whose sole assertion is `assert <Name/Subscript> is not None`, with a shrink-only allowlist seeded from T4a's benign classifications. **Ship only if the allowlist lands under ~15 entries** — otherwise DECLINE and record that the sweep (T4a) plus code review is the chosen control, because a high-false-positive standing guard costs a sole maintainer more than it saves. *Depends on*: T4a (needs its allowlist classifications).

**T6 — Trivial cleanups: DRY-02, DRY-03, PP-01.** *Invariant (three, one per file)*: (a) `forward.py` — the "primary type of an item" derivation is named once (`_primary_type`), not inlined 11×; (b) `state.py` — the node-fingerprint `sha256(...)[:12]` idiom is one local closure `_fp`, not two adjacent copies; (c) `__main__.py` — an unexpected CLI failure prints its traceback under `DEV_MODE`. *Anti-band-aid*: for (a), after extraction `grep -c "normalize_outputs(_declared_output" src/neograph/forward.py` returns 1 (inside the helper) — and note for the implementer to evaluate sharing `_dispatch._resolve_primary_output` rather than bare `.primary`; for (b) the `f"{...}:{_type_signature` idiom appears once in `state.py`; for (c) `NEOGRAPH_DEV=1 neograph scaffold` on a forced error shows the traceback, default run does not. *Parallel*: yes (three disjoint files). *Depends on*: none. No standing guard — these are one-shot tidy-ups, not convention drift.

**No inter-ticket dependencies except T4b→T4a.** T1, T2, T3, T4a, T6 are all mutually independent and trivially parallelizable (disjoint files; T4a touches only `tests/`). The whole set is a natural fit for the `/team` workflow with scoped regions: `{T1: _construct_validation.py + _llm_retry.py + new logging guard}`, `{T2: test_guards_three_layer.py}`, `{T3: di.py + tool.py + helper_monopoly guard}`, `{T4a: tests/ assertion sweep}`, `{T6: forward.py + state.py + __main__.py}`.

---

## Decisions (user, 2026-07-13)

1. **CON-03**: keep `McpToolCallError` off `NeographError` — the zero-import isolation of `neograph_mcp` from core is intentional and stands. Declined, no code change.
2. **T4b**: gate on allowlist size as recommended — build the guard from T4a's classifications, ship only if the allowlist lands under ~15 entries, otherwise close as declined with reason.
3. **DRY-02**: investigate first — before extracting a helper in `forward.py`, probe whether the tracer/runtime type-view divergence is a live bug for Oracle-modified nodes (`_resolve_primary_output` vs bare `.primary`), then pick the helper accordingly. Folded into T6.

**Filed tickets**: T1 = neograph-q4p72, T2 = neograph-outsr, T3 = neograph-hnbsq, T4a = neograph-gwncf, T4b = neograph-m5k19 (dep on T4a), T6 = neograph-2yi7q. Spike bead: neograph-d149a.

## Open questions genuinely needing the user's call

1. **CON-03 — `McpToolCallError` base class.** Keep it off bare `Exception` (preserves `neograph_mcp`'s deliberate zero-import isolation from core), or is uniform `except NeographError` catchability across `neograph` + `neograph_mcp` a product goal? My recommendation is **keep as-is** (decline) — the isolation is a real architectural property and a shared base would either re-couple the packages or add a third shared micro-package for one exception. Only you know whether a downstream consumer wants to catch both uniformly.

2. **T4b — standing assertion-strength guard: ship or decline?** The guard has a materially higher false-positive rate than the monopoly guards, so it needs a maintained allowlist. For a sole maintainer, my recommendation is **sweep now (T4a), ship the guard only if its allowlist is small (<~15)**; otherwise rely on T4a + review. This is a judgment about your tolerance for allowlist upkeep vs. recurrence risk — your call.

3. **DRY-02 depth — bare `.primary` vs `_resolve_primary_output`.** The extraction is safe either way, but `_dispatch._resolve_primary_output` already encapsulates a richer form (Oracle gen-type override) for the runtime side. If `forward.py`'s tracer should reflect Oracle overrides, it should share that helper, not a new bare-`.primary` one — a small semantic decision I flagged for the implementer rather than pre-deciding, since it touches whether the tracer's type view must match the runtime's.
