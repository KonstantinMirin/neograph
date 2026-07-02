# Adversarial architecture review — async / streaming / MCP / mid-loop HITL sequencing

**Reviewer:** architect agent (adversarial pass) · **Date:** 2026-07-02
**Under review:** `docs/design/async-streaming-mcp-hitl-2026-07-02.md`
**Grounding:** `docs/design/durable-execution-replay-research-2026-07-02.md`; `ox-troubleshooting-demo/docs/research/mcp-client-landscape.md`; direct code inspection of `factory.py`, `_tool_loop.py`, `_wiring.py`, `runner.py` this session.

---

## Verdict

**Yes, with changes.** This is a strong sequencing doc: the stateless-binding insight (§5) and the node-vs-step granularity analysis for HITL (§6.4) are both correct and genuinely load-bearing, and the dependency graph is basically right. But it is **not yet safe to cook epics from as written**, because four load-bearing specifics are either over-broad, internally contradictory across the two source docs, or entirely absent — and each would get baked into an epic's acceptance criteria. Resolve the HIGH findings first; the MEDIUMs can be resolved inside the epics they touch. The doc is honest about being "sketch, not frozen API," so this review targets the places where the *sketch itself* would misdirect implementation, not API bikeshedding.

**Severity counts:** HIGH 4 · MEDIUM 7 · LOW 4.

---

## HIGH

### H1 — The load-bearing invariant (§4.5/§5) conflates two lifetimes; as written it forbids connection pooling and breaks stateful MCP sessions

**Section:** §5, §6.3, §6.3.1.

**The gap.** The invariant — "run-time bindings are re-mintable functions of run context, never captured live state" — is *directionally correct but over-broad*. It collapses two different lifetimes into one rule:

- **(a) Across-checkpoint survival.** A binding that must outlive a checkpoint→resume (possibly in a fresh process) genuinely cannot be a captured live handle. This is the real insight and it is correct.
- **(b) Within-run reuse.** Between two steps of the *same uninterrupted run on the same loop*, reusing a live handle (a pooled MCP session, an HTTP connection) is not only safe, it is the correct, performant design.

The doc's phrasing ("A live MCP *session* is opened only for the actual call, then disposed — never held across a checkpoint boundary" slides into "opened only for the actual call, then disposed" — §6.3) mandates **open-initialize-call-dispose per `tools/call`**. That is a session-hostile design:

1. **MCP sessions are legitimately stateful.** The MCP `initialize` handshake negotiates capabilities and returns an `Mcp-Session-Id`; servers can hold subscriptions (resource-update notifications), sampling callbacks, and elicitation state on that session. "Dispose after every call" destroys all of it and makes resources/prompts/sampling (the doc's own "later" roadmap, §6.3) impossible without redesign.
2. **Per-call handshake latency.** Re-initializing a session on every tool call is exactly the "asyncio-per-call latency" risk the doc flags as open (§9.4) — but the invariant *causes* it rather than merely risking it. For a ReAct loop that calls the same server 5-10 times, that is 5-10 initialize round-trips.

**Suggested change.** Refine the invariant to name the lifetime explicitly: *bind a provider/factory, not a live handle; **cache** the live handle keyed to (process, event-loop) lifetime; invalidate the cache on resume.* On a fresh process the cache misses and the factory re-mints — so across-checkpoint survival is preserved — while within-run reuse (pooling, session reuse, subscriptions) is allowed. This is strictly more permissive than the current wording and removes the per-call handshake cost without weakening the checkpoint guarantee. Also add the carve-out that **serializable, deterministic effect *results*** may be captured (that is exactly what Level B / durable-execution journaling does, §6.4) — "never captured live state" currently reads as forbidding the very journaling the doc later recommends.

---

### H2 — "@node detects `async def` bodies → async wrapper" (§6.1) is the wrong trigger for the exact modes that motivate the work

**Section:** §6.1 "DX shape," §4.1.

**The gap.** The doc says async-ness is detected from the user function body. But the I/O-bound modes that justify this entire effort — `think`, `agent`, `act` — have **empty bodies** (`...`; they are dead code by design, per CLAUDE.md "Modes" table). There is no `async def` to detect. Their async-ness is a property of the **runtime/driver**, not the user's function: an `agent` node must run its ReAct loop under `await llm.ainvoke()` whenever driven by `arun()`, regardless of what its (dead) body looks like. Confirmed against the code: `make_node_fn` (`factory.py:22`) emits a sync `def node_wrapper(state, config) -> dict` over a fully synchronous vertical — `_execute_node` → `_dispatch_for_mode` → `_tool_loop` (which calls `llm_with_tools.invoke` at `:264` and `tool_fn.invoke` at `:324`). None of that reads the body's `async`-ness, and it shouldn't.

Conflating "async body" with "async execution path" will produce a factory that makes *scripted* nodes async when the user wrote `async def`, but leaves the LLM modes — the ones that actually block on the network 90% of the time — on the sync path. That is exactly backwards from the goal.

**Suggested change.** State the real rule: **the execution surface (`run` vs `arun`) chooses sync-vs-async for LLM/tool/MCP calls; the user body's `async def`-ness only governs how a *scripted* body is invoked** (await it vs threadpool it). The dual-path factory must emit an `async def node_wrapper` whose `_execute_node`/`_tool_loop` twin awaits `llm.ainvoke`/`tool.ainvoke`, selected by the driver, independent of body inspection. Make this the Phase-1 "key decision," because getting it wrong is silent (tests pass; production still blocks the loop).

---

### H3 — The AG-UI integration the whole doc is *for* — streaming across an interrupt — is owned by no phase; B and D are wrongly treated as independent

**Section:** §3 (calls B and D "independent"), §6.2, §6.4, §8.

**The gap.** The stated payoff (§10) is "a first-class AG-UI surface — streamed domain progress **plus** mid-loop HITL … an operator UI consumes neograph directly." That requires **one** event stream that carries progress events, *pauses at an interrupt*, surfaces the interrupt payload to the UI, and *resumes the same stream* on the human's answer. Mechanically that is `astream(...)` → yields custom events → yields an interrupt → generator returns → consumer calls `astream(Command(resume=answer))` → stream continues. This astream⇄interrupt⇄astream handshake is the actual hard integration, and it sits precisely at the **intersection of B (streaming) and D (HITL)** — which §3 declares "independent" and §8 schedules on different tracks (2 vs D). No phase owns "streaming across an interrupt boundary," and §6.2's "AG-UI event mapping" hand-waves it as an event-shape detail.

Compounding this: **cancellation and resource lifecycle are absent entirely.** When the SSE consumer disconnects mid-`astream` (browser closes, operator navigates away), who cancels the run, disposes any live MCP session (H1), and flushes the checkpoint? Async introduces `CancelledError` and task cancellation as first-order concerns; none of §6, §7, or §9 mentions them. For a durable, interruptible, streamed agent this is not an edge case — it is the normal way a UI session ends.

**Suggested change.** (1) Add an explicit deliverable — a phase or a cross-cut between Phase 2 and D — for the **resume-stream protocol**: the typed contract for how an interrupt surfaces *in* the astream event sequence and how a resume re-enters it. Write the AG-UI loop end-to-end as its own E2E, not as two independent features. (2) Add **cancellation/cleanup** as an invariant and an acceptance criterion: a disconnected consumer must cancel the run and dispose live handles deterministically; add an E2E that cancels mid-stream and asserts the MCP session is closed and the checkpoint is consistent.

---

### H4 — Audience binding: the two source docs directly contradict each other on whether neograph enforces RFC 8707

**Section:** §6.3.1 (FR) vs landscape §4.1.

**The gap.** The FR says neograph "**never inspects** the token (opaque; no parsing, no authz decision)" and adds a structural guard that "neograph code paths never read/parse the token's contents." The landscape doc §4.1 annotates the server-config `audience` field as "RFC 8707 binding, **enforced client-side**." These cannot both hold: you cannot enforce that a provider-returned token is audience-bound to the configured server **without reading its `aud` claim** — which the carries-not-decides guard forbids. As written, a buggy or malicious provider that returns a token minted for the *wrong* audience is forwarded verbatim by neograph — which is precisely the "token passthrough" the MCP spec marks a hard MUST-NOT, and neograph's own guard guarantees it cannot detect it.

**Suggested change.** Pick one and make it explicit in the epic:
- **Option A (recommended, consistent with carries-not-decides):** the `audience` config field is **advisory/documentary only** — it is the value the consumer's provider is expected to bind, and *the tool server* enforces `aud` on receipt (spec-correct; the server is the resource server). Change landscape §4.1's "enforced client-side" to "declared client-side, enforced server-side." neograph carries, the server decides — end to end.
- **Option B:** neograph performs a **minimal, non-authz `aud`-claim equality check** (read one claim, compare to configured audience, reject on mismatch — no scopes, no policy). This is defensible as a passthrough-prevention safety check rather than an authorization decision, but it *does* cross the "never parses the token" line, so the guard wording must change accordingly.

Do not ship the contradiction — it will produce a guard test and an FR acceptance criterion that assert opposite things.

---

## MEDIUM

### M1 — The dual-surface parity harness (§7) buys structural confidence where it's cheap and is absent where async is actually hard

**Section:** §7, §9.5.

**The gap.** Running every behavioral test through `run()` and `arun()` with **fakes whose `ainvoke` mirrors `invoke`** proves the async *plumbing* is wired, but proves nothing about the things async actually changes:
- **Concurrency** — the headline benefit ("one event loop serves many in-flight runs," §10) is never exercised; the harness runs one run at a time.
- **Ordering under real await points** — Each/Oracle fan-out via `Send` has a *documented* arrival-order-vs-collection-order caveat (CLAUDE.md, `list[X]` consumers). A mirrored fake collapses the timing that surfaces that bug class.
- **Event-loop blocking** — a sync provider/tool (or a blocking `broker.mint()`, see M2) that stalls the loop fails **no** functional assertion; it silently destroys concurrency. Parity cannot catch it.
- **Cancellation / timeout / partial failure** — never exercised (see H3).

Meanwhile, for pure/scripted nodes the async path is genuinely identical (LangGraph threadpools them), so "every behavioral test through both" spends most of its doubled cost (§9.5) buying near-zero behavioral signal.

**Suggested change.** Keep the parity harness for wiring confidence, but stop treating it as *sufficient*. Add the three tests it structurally cannot be: (1) a **real-concurrency integration test** — N runs in flight on one loop, asserting interleaving/isolation; (2) an **event-loop-blocking watchdog** — a slow sync call under `arun()` trips a loop-lag detector; (3) a **cancellation E2E** (shared with H3). And target the parity doubling at I/O-bearing tests rather than blanket-applying it to pure nodes.

### M2 — Provider invocation cadence is contradictory across the two docs, and the provider signature can't be a blocking sync lambda

**Section:** §6.3.1 vs landscape §4.3.

**The gap.** Landscape §4.3 says the provider is "called **per run** (constant per diagnosis)"; FR §6.3.1 says it must be invoked "**at call time** on the executor's loop." These imply very different behaviors: per-run minting (one token reused across the run's tool calls) vs per-`tools/call` minting (a fresh JWT per call, potentially hammering the broker). Separately, the example provider is a **synchronous** `lambda rc: broker.mint(...)` — but `broker.mint` is network I/O; invoking it "on the executor's loop" at call time blocks the event loop (M1).

**Suggested change.** Define the contract: the provider is invoked **once per run by default, cached for the run's duration, and re-invoked on resume** (fresh process → cache miss → re-mint). The cache lives *outside* checkpoint state (it is a live secret, per the no-token-in-state guard) — so it is exactly an H1-style within-run cache. Allow the provider to be **async** (`async def provider(rc) -> str`) so blocking brokers don't stall the loop; support a sync provider only via threadpool. State plainly: **neograph does not persist minted tokens; any caching beyond the per-run window is the provider's responsibility and must itself be checkpoint-safe.**

### M3 — HITL skips the cheap middle: a lint rule keyed on the existing `agent`/`act` mode distinction eliminates the footgun at compile time

**Section:** §6.4, §7.

**The gap.** The doc jumps from Level A ("document the footgun") straight to Level B ("build a checkpoint-keyed step-memo engine in `_tool_loop.py`") and misses the option in between that plays to neograph's actual differentiator — **the compile-time linter**. The footgun is "a costly/**mutating** tool runs before an `ask_human` in the same fat node, and re-executes on resume." But neograph *already* distinguishes mutation intent: `agent` mode = read-only tools, `act` mode = mutations (CLAUDE.md "Modes"). So:
- `ask_human` inside an **`agent`** node → provably safe (read-only re-execution is idempotent).
- `ask_human` inside an **`act`** node → the footgun.

That maps the "fine for read-only research tools" hand-wave onto an *enforceable* signal. Additionally, note the existing `interrupt()` (verified at `_wiring.py:685`) lives in a **dedicated micro-node** `operator_check` that does nothing but check-and-interrupt — so today's interrupt usage has *no* re-execution problem. `ask_human` is a categorically more dangerous placement (interrupt *deep inside* a fat agent node), and the doc's "all machinery exists" framing (§6.4) obscures that the safety profile is completely different.

**Suggested change.** Ship Level A **plus a lint check** (Level A.5): flag `ask_human` in an `act`-mode node (ERROR or WARN), and/or flag any mutating-tool call preceding `ask_human` in the same node. Cheap, consistent with the "linter is the gate" differentiator, and it converts "document the footgun" into "catch the footgun." Also state the Level-A→B compatibility contract: Level B must preserve results for tools that were idempotent under Level A (it narrows re-execution, an observable change for anyone who wrongly relied on double-execution).

### M4 — `run()` + MCP re-introduces the sync bridge that async supposedly "removes"

**Section:** §3, §6.3, §9.6.

**The gap.** MCP clients are async-only (both source docs). §6.3 says "a sync bridge is a workaround (A) removes." But §4.1/§9.6 also promise `run()` stays first-class and unchanged forever. If a consumer drives an `agent` node with MCP tools via **sync `run()`**, you are back to the background-loop-thread bridge — async did *not* remove it; it removed it only for the `arun()` path. The doc cannot have all three of {`run()` unchanged, MCP works everywhere, no sync bridge}.

**Suggested change.** Decide and document: **MCP tools require `arun()`** (cleanest — the linter can enforce "MCP tool bound in a graph driven by `run()` → error/warn"), *or* the sync bridge stays as the `run()`-path adapter (then stop claiming async removes it). Given the doc's own "astream-first, I/O is async" stance, requiring `arun()` for MCP is the coherent choice.

### M5 — Sequencing rationale is maintainer-preference, not consumer-readiness; Phases 2 and 3 are parallelizable and Phase 0 can't be fully designed before Phase 1

**Section:** §8.

**The gap.** (a) "astream-first because neograph is a key component across multiple systems the maintainer designs" is a preference, not a derivation. There are two *concrete, filed* consumer needs — ox-troubleshooting-demo (MCP + per-user auth, a filed FR) and agent-stark (AG-UI streaming/HITL). The doc doesn't rank them by readiness/deadline, so the "lead train" choice is unjustified rather than wrong. (b) The dependency table shows **both** Phase 2 (streaming) and Phase 3 (MCP) depend only on Phase 1 — they are parallelizable; serializing them (2 then 3) is a resourcing choice presented as a critical path. (c) Phase 0's stated key decision is "shape of the parametrized run/arun fixture," but `arun()` doesn't exist until Phase 1 — you cannot finalize the fixture against semantics you haven't built.

**Suggested change.** (a) Add a one-line consumer-readiness ranking to justify the lead train (or flip it if ox is closer to shipping). (b) State explicitly that 2 and 3 are parallelizable and that serialization is a resourcing decision, so a second implementer can pick up MCP without a false dependency block. (c) Reframe Phase 0 as *scaffolding only* (the parametrization mechanism + async-fake source-of-truth), co-evolving with Phase 1 — do not gate Phase 1 on a "complete" harness.

### M6 — Async observability, mixed-driver checkpoint resume, and a performance gate are unscheduled

**Section:** §6.1, §9.2, §9.4.

**The gap.** Three real async concerns have no phase: (a) **Observability/tracing** — neograph ships a Langfuse example (`observable_pipeline`, neograph-b6hm); async breaks naive callback/`contextvars` propagation across `await`, a well-known LangChain footgun. Not mentioned. (b) **Mixed-driver checkpoint resume** (§9.2) — can a thread started under sync `SqliteSaver` resume under `AsyncSqliteSaver` on the same `thread_id`? Listed as an open question but not assigned to a phase. (c) **Performance** — §9.4 flags asyncio-per-call latency as open, but no phase has a benchmark acceptance gate, and H1's per-call-handshake risk makes this acute.

**Suggested change.** Fold (a) into Phase 1 acceptance (assert trace spans survive an `await` boundary). Make (b) a Phase-1 acceptance criterion, not a lingering open question. Add an explicit **perf gate** to Phase 1/3: a benchmark asserting the async path adds no per-call overhead beyond a set threshold, and that MCP session reuse (H1) is exercised.

### M7 — The test matrix is a 6-cell grid, not a "doubling"

**Section:** §4.3, §7, §9.5.

**The gap.** §4.3 extends three-surface parity (`@node` / declarative / programmatic) to async. That is 3 API surfaces × 2 execution surfaces = **6 combinations** per IR-level behavioral change — and the three-surface rule is *already* the top bug source per CLAUDE.md ("a feature works via `@node` but breaks via the programmatic API"). §7/§9.5 frame the cost as a "doubling," which undercounts and underestimates the parity-maintenance burden.

**Suggested change.** Name the 6-cell reality. Decide where full 6-cell coverage is mandatory (IR-level behavioral changes) vs where a representative subset suffices (pure rendering/validation logic that is execution-surface-agnostic). Bake that decision into the harness so it doesn't become a wall-clock and maintenance sink.

---

## LOW

### L1 — `emit_progress` under sync `run()` silently no-ops
**§6.2, §8 (sync `stream()` dropped).** With sync `stream()` dropped, a node calling `emit_progress` while driven by `run()` has no writer — the event is discarded silently. For a maintainer-only userbase that's acceptable, but it's a latent surprise. Add a documented behavior + a lightweight guard/warning: `emit_progress` under a non-streaming driver warns once (or buffers), rather than vanishing.

### L2 — The carries-not-decides guard should extend to logs, not just state/fingerprint
**§6.3.1 guardrails.** The doc guards "token never enters checkpointed state or the schema fingerprint." Good — but the structlog `tool_call` event and any MCP-adapter request logging are the other leak paths. Extend the guard/assertion to "no `Authorization` header value appears in any log or `ToolInteraction` record."

### L3 — "Async DI resolution" is over-scoped; the real async-sidecar issue is awaiting async scripted bodies
**§6.1 "Hard part."** DI resolution (`_resolve_di_value`) is pure `config` dict lookup — it needs no `await`. Listing "async DI resolution" as a hard part misdirects. The actual async-sidecar work is narrower and specific: the scripted shim must `await` the user function when it is `async def` (vs threadpool a sync body). Re-scope the bullet to that.

### L4 — Enumerate the structured-output/coercion wrapper as an explicit async twin
**§6.1.** The doc lists "`await llm.ainvoke()` / `await tool.ainvoke()`" but the actual sync call sites are wrapped: `_CoercingToolWrapper(llm.bind_tools(...))` (`_tool_loop.py:251`) and the `think`-mode structured-output path. Each custom wrapper needs its own async path (`abind_tools`, async coercion). Name them so they aren't discovered late; they're the kind of custom seam that a "just call ainvoke" framing hides.

---

## What is genuinely right (so it survives the edits)

- **The stateless-binding *direction* (§5)** is the correct organizing insight — H1 refines its lifetime scoping, it does not overturn it.
- **The node-vs-step granularity analysis (§6.4 + research doc §1b/§4)** is correct and well-sourced: an LLM call is non-deterministic, so under deterministic replay it must be a journaled step, and LangGraph's node-granular memoization is genuinely too coarse for a ReAct loop. "Level A is the field's floor, not a cop-out" is a fair and well-supported reading.
- **The single-seam / no-reach-around invariant (§4.4)** and the custom-event-fingerprint guard (§6.2) are the right structural guards for this codebase's culture.
- **Compile-time lint of MCP bindings** is a real, defensible differentiator, and the doc is right that neograph's bounded-step scoping solves tool-context-rot by construction.
