# Architecture decisions — 2026-05-01

**Status**: locked. Authoritative source for ticket scoping derived from `senior-architect-review-2026-04-28.md` (the source diagnosis) and `handoff-2026-04-29.md` (the session continuity record).

**Background**: the senior-architect review identified 7 weaknesses (W1–W7), 5 risks (R1–R5), and 7 recommendations (Rec1–Rec7). Several of the recommendations were strategically ambiguous because they depended on a load-bearing positioning question (Rec7: DX layer vs. backend-neutral graph engine). This document captures the answers locked during the diagnosis sessions on 2026-04-30 and 2026-05-01, walks through their rationale, and binds them to concrete implementation guidance.

**Citation contract**: every ticket filed in response to the senior-architect review MUST cite a section of this document by section number, in addition to citing the W/R/Rec number from the source review. Tickets that introduce design that this document does not cover should add a section here first.

---

## Q1 — LangGraph commitment (replaces "backend-neutral")

### Decision

neograph commits to LangGraph-family runtimes. Python on LangGraph, TypeScript on LangGraphJS. **Drop "backend-neutral" from positioning.** Provider-neutrality exists at the model-invocation layer (user's `llm_factory` returns any `BaseChatModel`); it does NOT exist at the graph-execution layer, and there is no realistic candidate to be neutral over.

### Rationale

The candidate "alternative backends" (Anthropic Agent SDK, Google ADK, OpenAI Agents) are agent-loop SDKs, not graph executors. They do not expose primitives equivalent to LangGraph's `Send` (fan-out), `defer=True` (deferred barriers), conditional back-edges (Loop), or interrupts. Lowering neograph's IR (Each / Oracle / Loop / Operator) onto an agent-loop SDK would require neograph to bring its own graph executor in user-space — at which point the SDK is a model-call adapter, not a backend, and neograph has reimplemented LangGraph in user code.

A `NeoConfig` shimmed over `RunnableConfig` is rename, not abstraction — the modifier semantics commit to LangGraph concepts regardless of import path. Two parallel codebases (Python+LangGraph, TS+LangGraphJS) are not "two backends to abstract over"; they are two implementations of the same product against two implementations of the same runtime family by the same vendor.

### Implications

- The IR may import `RunnableConfig` directly. No quarantine work required.
- `langgraph._internal._serde` private import in `compiler.py:51` MUST be removed. Either use a public msgpack-allowlist API or accept warn-all behavior with a documented workaround.
- LangGraph version range MUST be pinned in `pyproject.toml` to a supported public-API range. Public API is the contract.
- The "if it compiles, it runs" claim is property of the IR + LangGraph-family runtime, not of an abstract backend.
- When considering features, the question is "does this exist in both LangGraph and LangGraphJS?" If a feature is Python-only, ship it on Python and document the asymmetry. Do not shrink the IR to the minimum common subset preemptively; do shrink it when LangGraphJS users encounter a missing feature.
- `CLAUDE.md` (alias `AGENTS.md`) MUST be updated to remove any "backend-neutral" or "abstract away from LangGraph" framing.
- Website positioning MUST be reviewed for the same.

### What this collapses in the senior-architect review

- **Rec3** ("NeoConfig at the IR layer, translate at the boundary"): collapses from ~3 days to ~1 day. The work is now `_serde` removal + version pinning + (optional) LangGraphJS feature-parity audit, not abstract-backend introduction.
- **Rec7** ("decide DX layer vs competing engine"): answered. DX layer for LangGraph family.
- **W5** ("LangGraph coupling not quarantined"): reframes from "leak everywhere" to "embraced commitment with clean public-API discipline."
- **R1** ("LangGraph 1.0 will break the codebase"): reframes from "existential risk of abstraction failure" to "scheduled minor-version maintenance against pinned API range."

---

## Q2 — `LlmRuntime` is closure-captured at compile time

### Decision

`compile(construct, runtime=LlmRuntime(...))`. The factory functions (`make_node_fn`, `make_subgraph_fn`, `make_oracle_redirect_fn`, etc.) gain a `runtime=` parameter and capture it in returned closures. The compiled graph stashes `compiled._neo_runtime` for inspection. The six module-level mutables in `_llm.py:98-109` (`_llm_factory`, `_llm_factory_params`, `_prompt_compiler`, `_prompt_compiler_params`, `_global_renderer`, `_cost_callback`) and the `configure_llm()` global-mutating function are **eliminated, not replaced with a context-manager shim.**

### Rationale

Three options were considered:
- **A — closure-captured at compile time** (chosen). Explicit-is-better-than-implicit. True multi-tenant isolation. `compile()` becomes deterministic with respect to process state by construction. No hidden state survives compile.
- **B — `RunnableConfig['configurable']['_neo_runtime']`** (rejected). Doesn't actually solve the problem — `compile()` would still need to read globals to do early-fail validation. Per-call dispatch works but it's a routing convenience, not an architectural fix.
- **C — contextvar** (rejected). Cleanest API but the failure modes (sync node functions in thread pools, multi-process executors, edge cases of contextvar propagation across LangGraph's Send mechanism) introduce subtle bugs the architecture cannot reason about cheaply.

A is the most disciplined and the largest refactor. The cost (~3 days of mostly-mechanical work touching every test that calls `configure_llm()`) is justified by the elimination of an entire class of bugs (test isolation, multi-tenant, compile-time non-determinism).

### Implications

- `compile(construct, runtime=...)` becomes the canonical compilation API. `compile(construct)` without runtime fails loudly when the construct contains LLM nodes.
- `lint(construct)` accepts an optional `runtime=` parameter; without it, lint validates DI bindings but cannot validate runtime-dependent properties (e.g., `llm_factory` callable signature).
- `Node.run_isolated(input=...)` accepts an optional `runtime=` parameter; required when the node's `mode in ("think", "agent", "act")`.
- **No `configure_llm()` shim retained.** Per Rec6 / no-backcompat policy at 0.x with a single consumer, the global-mutating function is removed entirely; piarch's `_configure_llm()` in `derive_ensemble/integration/neograph_bridge.py` will need to migrate to constructing an `LlmRuntime` and passing it to `compile()`.
- The `compiler.py:130` early-fail check (`from neograph._llm import _llm_factory, _prompt_compiler`) reads from the passed runtime instead of from globals. `compile()` becomes deterministic.
- The `_dispatch.py:177-181` `try/except ImportError` defensive pattern around `_get_global_renderer` becomes unnecessary — the runtime is passed through closures, not imported via potentially-cyclic module reads.
- `tests/conftest.py` registry-cleanup fixture goes away. Tests build runtimes in fixtures (or inline) and pass them to `compile()`.

### Cost

~3 days of refactor. The migration touches essentially every test that exercises an LLM-mode node — primarily `test_llm_internals.py` (4416 lines) plus most files in `tests/modes/`. See Q6 for the bundled test cleanup that reduces this cost via shared editing surface.

---

## Q3 — Runtime is bound at compile, not per-call

### Decision

The runtime lives on the compiled graph. `run(graph, ...)` does NOT take a runtime argument. Multi-tenant servers recompile per tenant (cheap — compiles are fast).

### Rationale

Falls out of Q2=A. A "per-call runtime" (passing runtime to `run()` to allow one compiled graph to serve many tenants) is more flexible but adds complexity to every place that reads runtime — they'd need to check both compile-bound and run-bound. piarch is not multi-tenant and the foreseeable consumers won't be either; YAGNI applies.

### Implications

- `compile(construct, runtime=runtime_a)` and `compile(construct, runtime=runtime_b)` produce two distinct compiled graphs. They can coexist in one process; their state is fully isolated.
- A future per-call override is possible but not required: `run(graph, runtime=override)` could be added later as a non-breaking addition that defaults to `compiled._neo_runtime`. Do not build it speculatively.

---

## Q4 — Per-compile registry, paired with Q2

### Decision

`_registry.py`'s process-global singleton (`registry = Registry()` at line 52) becomes per-compile. `compile()` builds a fresh `Registry()` instance, walks the construct, registers `@node` shims into the fresh registry, and dispatches scripted lookups through that registry via closure capture.

Manual user registrations move from module-level side-effect APIs to `compile()` keyword arguments:

```python
compile(
    construct,
    runtime=runtime,
    scripted={"name": fn},          # replaces register_scripted("name", fn)
    tool_factories={"name": factory},  # replaces register_tool_factory("name", factory)
    conditions={"name": condition_fn}, # replaces register_condition("name", condition_fn)
)
```

The exact API shape is TBD during implementation; the principle is that registrations are compile-time arguments, not module-load-time side effects.

### Rationale

Q2=A makes the runtime per-compile. If the registry stays process-global, isolation is partial — runtime is isolated, registry is not. Two compile() calls in the same process can still collide on tool-factory names or scripted shim names. Pairing Q4=B with Q2=A delivers true isolation.

The `Registry.session()` context manager exists today as a test workaround. Per-compile registries make the workaround unnecessary; tests get isolation for free.

### Implications

- **R4 (id(f) silent shadowing)** closes for free. Each compile gets its own scope; `id`-keyed shim names cannot leak across compiles or across test boundaries. The latent shadowing bug from `decorators.py:151` becomes structurally impossible.
- `@node` decoration no longer registers anywhere at decoration time. The `Node._sidecar` PrivateAttr already carries the original function on the Node instance; `_register_node_scripted` is called at construct-assembly time during `compile()` walk. This is approximately how the design already works on the `@node` path; the user-facing `register_scripted("name", fn)` API is what changes.
- Migration cost: ~2-3 days. Pairs with Q2; ships as one epic.

### Cost

Combined with Q2: ~5-6 days for the runtime + registry migration as a single coherent change.

---

## Q5 — Single-responsibility retry

### Decision

Each retry concern lives in exactly one layer. The two-layer composition that produces multiplicative `outer × (inner+1)` retry behavior is **eliminated**, not merely documented.

| Concern | Layer | Mechanism |
|---|---|---|
| Transient API failures (network errors, 429s, 5xx, timeouts) | User's `llm_factory` | `model.with_retry(...)` from LangChain, or SDK-level retry, or a custom rate-limited wrapper |
| Output-quality failures (malformed JSON, schema violations) | `LlmConfig.max_retries` via `_invoke_json_with_retry` | Existing BAML-rendered feedback retry. **Keep ours, do NOT adopt LangChain's `OutputFixingParser`.** |
| Scripted node failures (transient Python exceptions in deterministic code) | LangGraph `RetryPolicy` | Whole-node retry — safe because no tool side effects |

### Rationale

The accidental complexity argument is load-bearing: neograph exists to fix complexity, not introduce it. The multiplicative retry math is pure accidental complexity created by neograph layering its own retry on top of LangGraph's without coordinating semantics. Documentation alone (option α — explain the math) is insufficient because the user shouldn't have to reason about the math at all to use neograph correctly.

There is also a correctness concern, not just an efficiency one: **LangGraph's `RetryPolicy` on `act`-mode nodes replays tool side effects on retry.** Today `compiler.py:385` applies `retry_policy` to `mode in ("think", "agent", "act")` — including `act`, which is the mutations path. A retry on a node that already executed `send_email` or `write_file` results in a double-write, not a retry. This is a latent correctness bug independent of the multiplicative-spend concern, and the same architectural fix closes both.

LangChain's repair mechanisms (`OutputFixingParser`, `RetryWithErrorOutputParser`) were considered as a replacement for `_invoke_json_with_retry`. Rejected: neograph's retry uses BAML-style schema rendering (`describe_type`) in the feedback message, which is materially better for LLM parsing than LangChain's JSON-Schema-based format instructions. Adopting LangChain's parsers would throw away the BAML advantage in the part of the call where it matters most — the retry, where the LLM has already failed once. neograph's retry layer is coupled to neograph's prompt-input rendering pipeline; that's a feature, not duplication.

### Implementation

Three concrete changes:

1. **`compiler.py:_add_node_to_graph`** — change one line:
   ```python
   # Before
   rp = retry_policy if node.mode in ("think", "agent", "act") else None
   # After
   rp = retry_policy if node.mode == "scripted" else None
   ```

2. **Compile-time warning** — when `compile(construct, retry_policy=...)` is called AND any node has `mode in ("think", "agent", "act")`, emit a clear warning explaining:
   - `retry_policy` applies only to `scripted` nodes
   - For LLM transient errors, configure retry on the model in `llm_factory` (recommend `model.with_retry(...)`)
   - For LLM output quality, set `LlmConfig.max_retries` (default 1)

3. **Documentation rewrite** — neograph.pro retry-semantics page explaining the three-knob model with concrete `model.with_retry(...)` examples, especially for DeepSeek and other non-conforming models.

### Implications

- The user has **one knob per concern**. No compounding. No multiplication math. No documentation table apologizing for surprising behavior.
- The act-mode tool-replay correctness bug closes.
- Existing consumers that rely on `retry_policy` for LLM-node retry MUST migrate transient retry handling to their `llm_factory`.

### Cost

~1 day total: one-line compiler change + warning + docs rewrite.

### Open follow-up (not blocking)

- **Shared retry budget (β)**: a `LlmConfig.total_call_budget` knob bounding total LLM calls per node across both retry layers. Rejected for now because Q5's single-responsibility design probably eliminates the need. Revisit only if pain emerges (e.g., a consumer reports confusing token spend after the migration).

---

## Q6 — Test realism is bundled into Q2 migration

### Decision

`test_llm_internals.py` and the rest of the LLM-touching test suite are NOT a separate epic. As the Q2 migration touches these files (every test that calls `configure_llm()` migrates), audit each test for implementation-coupling and either delete or downgrade to contract assertions in the same pass.

### Rationale

Q6 in isolation is "rewrite the worst third — 1 week of stop-the-world testing work." Q2 already requires touching most of these files mechanically. The marginal cost of triaging while migrating is small; the marginal benefit is identical. Bundling avoids paying twice.

### Triage rule

- Test asserts call-count (`call_n["n"] == 3`) on internal dispatch → suspect; verify it tests user-visible behavior, otherwise simplify or delete.
- Test asserts on edge presence (`"assemble_verify -> summary" in edges`) → almost always wrong; testing LangGraph internals, not neograph contract.
- Test asserts on output value, retry count, error message, structured-output success, schema-fingerprint matching → keep as-is.

### Implications

- The Q2 migration epic includes test-cleanup as part of its acceptance criteria, not a separate ticket.
- If Q2 migration completes without ~30% reduction in implementation-coupled assertions in `test_llm_internals.py`, the bundled cleanup wasn't taken seriously and the work isn't done.

---

## piarch impact summary

Verified during the 2026-05-01 research phase against `/Users/konst/projects/piarch`:

- piarch's `runner.py:152` passes `retry_policy=RetryPolicy(max_attempts=5, initial_interval=5.0, backoff_factor=2.0)` to `compile()` for **every pipeline run.**
- piarch's `get_llm()` in `derive_ensemble/llm/client.py:184` returns a raw `ChatOpenAI` instance. **No `with_retry()` wrapping.**
- piarch's `RateLimitedCaller` in `derive_ensemble/llm/rate_limiter.py:31` is **dead code** — the class is constructed at `runner.py:76` and `workflow/config.py:121` and threaded through config layers, but `.call()` is never invoked anywhere in piarch's source. It exists but does nothing.
- piarch uses `output_strategy="json_mode"` extensively (10+ nodes across `uc_derivation.py` and `rw_ingestion.py`). With the default `LlmConfig.max_retries=1` and piarch's `RetryPolicy(max_attempts=5)`, the worst-case multiplicative behavior is **5 × (1+1) = 10 LLM calls per node on chronic JSON-parse failure.** piarch is paying this cost today.

The piarch impact splits cleanly into two unrelated changes with different sequencing:

### piarch hygiene — independent of neograph (can ship today)

Q5's piarch-side fix does NOT require any neograph change:

- Wrap `ChatOpenAI(...)` in `.with_retry(stop_after_attempt=5, wait_exponential_jitter=True)` inside `get_llm()`. Or properly plumb `RateLimitedCaller` if a custom rate-limit pattern is preferred.
- Drop `RetryPolicy(max_attempts=5)` from `compile()` for LLM-mode nodes (or scope it to scripted nodes only if piarch has any that benefit).
- Audit `RateLimitedCaller`: either properly plumb it into the LLM call path or delete it (currently dead infrastructure).

This works on current neograph because `with_retry()` is a model-level decoration and dropping `retry_policy` simply stops engaging the multiplicative path. After this lands, piarch's worst-case retry count drops from 10 to ~6 (5 transient retries + 1 parse-feedback retry, additive not multiplicative).

**This is a piarch hygiene ticket filed in piarch's beads. It SHOULD ship before neograph's Q5 epic so the Q5 compile-time warning does not trigger on piarch on day one. It MUST ship before piarch upgrades to a neograph version that includes Q5.**

Architectural value of decoupling: piarch's hygiene migration becomes a real-world test of the new retry pattern before Q5 enforces it at the framework level. If the recommendation has flaws, they surface in piarch first, not in the neograph release.

### piarch bridge migration — coupled to neograph Q2

piarch's `derive_ensemble/integration/neograph_bridge.py:226` calls `configure_llm(...)`. After neograph's Q2+Q3+Q4 epic ships, this API is gone. piarch's bridge migrates to:

```python
runtime = LlmRuntime(
    llm_factory=llm_factory,
    prompt_compiler=prompt_compiler,
    cost_callback=cost_callback,
)
# runtime is then passed to compile() inside the runner
```

The `bootstrap_neograph()` pattern shifts from "set globals" to "build runtime, return it for runner to use." This is a piarch ticket filed in piarch's beads, ordered AFTER neograph's Q2+Q3+Q4+Q6 epic lands (because the new API does not exist on current neograph).

---

## Open standalone bug tickets (independent of any architectural decision)

These are bug-fix tickets that should ship on their own timeline, not bundled with the architectural epics:

- **R2** (transitive fingerprinting in `runner.py:_compute_invalidated_nodes`): the comment claims construct unavailable on the compiled graph; it IS available via `compiled._neo_construct` (stashed at `compiler.py:199`). Walk the producer→consumer adjacency, return the transitive closure of changed nodes. TDD with the failing repro: A's fingerprint changes, B reads A's output, B's own fingerprint unchanged because B's return type is the same — currently B is silently stale on resume.
- **R4** (`id(f)` shim naming in `decorators.py:151`): replace with `secrets.token_hex(8)`. Closes the silent-shadowing path. Q4 closes this transitively, but R4 should ship before the Q2+Q4 epic so the bug doesn't survive while the larger refactor is in flight.
- **Rec4** (`neo_*` magic strings centralized): introduce `_state_keys.py` with `class StateKeys` defining all `neo_*` constants. Add a structural-guard test that no `"neo_*"` string literals exist outside `_state_keys.py`. Half a day.
- **Rec5** (`Node.outputs` polymorphism normalizer): one helper that converts `type | dict[str, type] | None` into a tagged result. Replace 26 isinstance sites across the codebase with normalized access. Centralize the discrimination in one place; eliminate `isinstance(node.outputs, dict)` everywhere else.

---

## Open feature tickets surfaced by the diagnosis

These are NEW features (not refactors) that emerged during the decision walkthrough. File separately; do not bundle with the architectural epics:

- **Auto-strategy selection**: `LlmConfig.output_strategy = "auto"` with a model-capability registry. Detects models that can't reliably produce structured output (DeepSeek, etc.) and picks `json_mode` automatically. Best built after Q2 migration so it can live as a property of `LlmRuntime`. Requires a maintained registry of model substring → known-good strategy.

---

## What is NOT in this document

The senior-architect review identified these items that this document does not directly address:

- **W3** (god-modules `factory.py` and `_llm.py`): not a separate epic. Both modules shrink as a side effect of Q2 (LlmRuntime extraction reduces `_llm.py`) and Q4 (registry extraction reduces `factory.py`). If they remain large after Q2+Q4, file follow-up tickets then.
- **W4 (Type-system cargo-culted)**: partially addressed by Rec5 (Node.outputs polymorphism normalizer). The `Any` count in `_wiring.py` (32), `_oracle.py` (14), `factory.py` (28), `_construct_validation.py` (23) is a structural concern that will improve incrementally as the runtime+registry refactor surfaces typed data flows. Not a standalone epic.
- **W6 (test implementation-coupling)**: covered by Q6 bundling.
- **R5 (retry × json_retry mixing)**: solved by Q5.

---

## Slicing strategy

Five epics total — four in neograph's beads, two in piarch's beads, with one explicit cross-repo dependency.

### neograph epics

1. **Pre-cleanup epic** (independent, ship anytime). Children: R2 (transitive fingerprinting bug), R4 (id(f) shim naming bug), Rec4 (state-key constants module), Rec5 (Node.outputs polymorphism normalizer). Each child ~half a day to a day. Total: ~3 days. No dependencies. R4 should ship before the runtime+registry epic so the bug doesn't survive while the larger refactor is in flight.

2. **Q1 epic — LangGraph commitment cleanup** (independent). Drop `langgraph._internal._serde` private import, pin LangGraph version range in `pyproject.toml`, update `CLAUDE.md` / `AGENTS.md` to remove "backend-neutral" framing, audit and update website positioning. ~1 day. No dependencies.

3. **Q5 epic — Single-responsibility retry** (depends softly on piarch hygiene). One-line change in `compiler.py:_add_node_to_graph`, compile-time warning when `retry_policy=` is passed AND any node has `mode in ("think", "agent", "act")`, full documentation rewrite of retry semantics on neograph.pro. ~1 day. **Soft dependency on piarch hygiene epic landing first** — Q5 can technically ship without it, but the new compile-time warning would trigger on piarch immediately, forcing piarch's hand. Cleaner to let piarch migrate at its own pace.

4. **Q2+Q3+Q4+Q6 epic — Runtime + per-compile registry + test cleanup** (independent of piarch hygiene; blocks piarch bridge migration). Define `LlmRuntime` IR type, migrate factory functions to closure-capture, replace process-global `Registry` singleton with per-compile registry, update `compile()` signature with new kwargs (`runtime=`, `scripted=`, `tool_factories=`, `conditions=`), drop `configure_llm()` and module-level register functions entirely, migrate test suite (Q6 bundled — apply triage rule from §Q6). ~5-6 days. Largest single epic.

### piarch epics

5. **piarch hygiene epic — LLM retry pattern cleanup** (filed in piarch's beads, independent of any neograph change). Wrap `ChatOpenAI` in `.with_retry(...)`, drop `RetryPolicy(max_attempts=5)` from `compile()` for LLM nodes, audit/fix/delete dead `RateLimitedCaller` infrastructure. ~1 day. Can ship today on current neograph. SHOULD ship before neograph Q5 epic to prevent compile-time warning churn.

6. **piarch bridge migration — LlmRuntime API adoption** (filed in piarch's beads, depends on neograph Q2+Q3+Q4+Q6 epic). Migrate `_configure_llm()` to construct `LlmRuntime` and pass through to `compile()`, update `neograph_bridge.py`, verify all pipelines work end-to-end with new API. ~1-2 days.

### Total cost

- **neograph**: ~10-11 days across 4 epics
- **piarch**: ~2-3 days across 2 epics
- **Cross-repo coordination points**: 1 (neograph Q2+Q3+Q4+Q6 → piarch bridge migration)

The previous review's framing implied multi-month work. The decisions captured here compress it because they answer the strategic ambiguity (Q1) that was inflating the estimates and identify the genuine independence of piarch's hygiene work from neograph's framework changes.

---

## Tickets filed (2026-05-01)

### neograph beads

| ID | Type | Title |
|---|---|---|
| `neograph-u6u0` | epic | Architectural debt pre-cleanup (R2 + R4 + Rec4 + Rec5) |
| `neograph-j36u` | bug | R2: transitive fingerprinting in runner.py:_compute_invalidated_nodes |
| `neograph-xdvt` | bug | R4: replace id(f) shim naming in decorators.py:151 with secrets.token_hex |
| `neograph-n3f1` | task | Rec4: centralize neo_* magic strings in _state_keys module |
| `neograph-mqr3` | task | Rec5: Node.outputs polymorphism normalizer |
| `neograph-0frr` | epic | Q1: LangGraph commitment cleanup (drop _serde, pin version, update positioning) |
| `neograph-mxnw` | epic | Q5: single-responsibility retry (RetryPolicy → scripted only; transient → llm_factory) |
| `neograph-n2li` | epic | Q2+Q3+Q4+Q6: LlmRuntime + per-compile registry + test cleanup |

Dependency: `neograph-u6u0` (pre-cleanup epic) depends on its 4 children (`-j36u`, `-xdvt`, `-n3f1`, `-mqr3`). Other epics are independent within the neograph repo.

### piarch beads

| ID | Type | Title |
|---|---|---|
| `piarch-zhhr2` | epic | LLM retry hygiene: with_retry, drop neograph RetryPolicy, audit dead RateLimitedCaller |
| `piarch-rwyga` | epic | Migrate neograph_bridge from configure_llm() to LlmRuntime API |

### Cross-repo soft dependencies (NOT enforced by beads, tracked here)

- `piarch-zhhr2` (piarch hygiene) SHOULD ship before piarch upgrades to a neograph version that includes `neograph-mxnw` (Q5). Both work on current neograph, so there is flexibility — but if piarch upgrades to a Q5 neograph without first landing the hygiene epic, the new compile-time warning triggers immediately.
- `piarch-rwyga` (piarch bridge migration) MUST wait until `neograph-n2li` (Q2+Q3+Q4+Q6) ships. The new LlmRuntime API does not exist on current neograph. piarch cannot start this epic before the neograph epic lands.
- `neograph-mxnw` (Q5) is independent of `neograph-n2li` (Q2+Q3+Q4+Q6) and can ship in either order. Recommended: Q5 first because it is smaller, lower-risk, and enables piarch's hygiene migration to be validated against a stricter neograph.

### Recommended ordering

1. **In parallel, anytime**: `neograph-0frr` (Q1 cleanup), the four pre-cleanup children (`neograph-j36u`, `-xdvt`, `-n3f1`, `-mqr3`), and `piarch-zhhr2` (piarch hygiene). All independent of each other and of the larger refactors.
2. **After piarch hygiene lands**: `neograph-mxnw` (Q5). The compile-time warning will not trigger on a clean piarch.
3. **After Q5 lands and is stable**: `neograph-n2li` (Q2+Q3+Q4+Q6 — runtime + registry + test cleanup). Largest epic.
4. **After Q2+Q3+Q4+Q6 lands and is released**: `piarch-rwyga` (piarch bridge migration to LlmRuntime API).

This ordering minimizes coordination friction, lets independent work proceed in parallel, and uses piarch as the real-world validation point for the Q5 retry pattern before the framework enforces it.
