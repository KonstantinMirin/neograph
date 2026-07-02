# Async, streaming, MCP, and mid-loop HITL — a sequencing architecture

**Status:** Design / sequencing (pre-epic) · **Date:** 2026-07-02
**Provenance:** surfaced by real consumer usage — agent-stark ADR-0007 (interaction-protocol-agui) and the ox-troubleshooting-demo MCP-client landscape research (`ox-troubleshooting-demo/docs/research/mcp-client-landscape.md`). Feeds beads features neograph-q8ec (custom stream events), neograph-m6d3 (mid-loop HITL), and the not-yet-filed async foundation + MCP client.

> This is a sequencing document, not a commitment to a specific API. It exists to make the dependency order, the load-bearing invariants, and the testing strategy explicit **before** any atoms are cooked, so we don't build a sync `stream()` we throw away or bolt async on in a way that erodes the correctness we have today.

---

## 1. The strategic shift

Until now neograph optimized for **correctness**: typed boundaries, compile-time validation, "if it compiles it runs", a strong structural-guard suite, and no known bugs. That was the right first phase. Real consumers (agent-stark, ox-troubleshooting-demo) have now surfaced a different axis — **feature richness for production I/O** — that the correctness phase deliberately deferred:

- An agent must **consume MCP tools** (and later resources/prompts), bound **per step**, with **per-user identity**.
- An operator UI (AG-UI / CopilotKit) needs a **streamed progress channel** and **mid-loop human-in-the-loop** decisions.
- All of this is I/O-bound — the system spends ~90% of its time waiting on the network.

None of these are exotic. LangGraph already provides every primitive. The gap is that **neograph forces a synchronous, non-streaming, single-shot execution model over an async-capable substrate**, and offers no sanctioned seam for interrupts-in-tools or custom events. Consumers are reaching around neograph to raw LangGraph — which defeats the single-seam design and is the signal that these are real gaps, not nice-to-haves.

---

## 2. The core realization (verified against the code)

neograph forces sync; LangGraph does not.

- neograph source contains **zero** async constructs — `grep -rn "async def|await|ainvoke|astream" src/neograph/` returns 0.
- LangGraph's `CompiledStateGraph` exposes `ainvoke`, `astream`, `astream_events` (verified present), supports `async def` nodes natively, runs **sync nodes in a threadpool** under async execution, and ships async checkpointers.
- The LLM layer is already async-capable: LangChain chat models expose `.ainvoke()`; neograph just calls `.invoke()` (`_tool_loop.py:266`, `:324`; `_llm.py`).
- `interrupt` / `Command` / `get_stream_writer` are all importable from LangGraph today. neograph already uses `interrupt()` for the boundary Operator path (`_wiring.py:681`) and `Command(resume=...)` in the runner (`runner.py:314`).
- neograph re-exports **nothing** for streaming or agent-callable interrupts (`__init__.py` has no `stream`/`emit`/`ask_human`).

**Conclusion:** the synchronicity is an arbitrary neograph choice (path-of-least-resistance DX), not a LangGraph constraint. Async is the correct foundation for a production I/O-bound system, and it is the enabler the other three features want to stand on.

---

## 3. The four capabilities and how they interlock

```
                ┌─────────────────────────────┐
                │  (A) Async foundation        │  dual sync/async; arun(); await llm/tool
                └──────────────┬──────────────┘
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
   ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐
   │ (B) Streaming │  │ (C) MCP client│  │ (D) Mid-loop HITL │  ← independent; sync-capable
   │  + emit_prog  │  │  per-step bind│  │   ask_human()      │     but shares the loop code
   │  (q8ec)       │  │  stateless auth│  │   (m6d3)           │
   └───────────────┘  └───────────────┘  └───────────────────┘
             ▲                 ▲
             └── AG-UI SSE ────┘  (both want the async event stream a UI consumes)
```

- **(A) Async foundation** is the keystone. (B) and (C) are its first consumers.
- **(B) Streaming (q8ec)** = a sanctioned streaming runner + `emit_progress`. The event helper is small; the *runner* is the missing piece, and the production (AG-UI SSE) version is async → depends on (A). A **sync `stream()` can ship first** and unblock the feature, but is throwaway-ish for the AG-UI path.
- **(C) MCP client** = typed per-step server binding, discovery decoupled from live session, per-run token provider. MCP clients are async-only; a sync bridge (background loop thread) is possible but is a workaround that (A) removes.
- **(D) Mid-loop HITL (m6d3)** is **independent** — interrupt/resume already works under sync `.invoke()`. Its real risk is loop-replay semantics (§6.4), not async. It can proceed in parallel on the sync track, but touches the same `_tool_loop.py` code (A) and (C) touch, so coordinate.

---

## 4. Invariants to preserve (the crown jewels)

The biggest risk of this work is regressing the two things that make neograph trustworthy today: **no known bugs** and **strong, meaningful coverage**. These invariants are non-negotiable and constrain every design choice below.

1. **Dual sync/async, never a flip.** `def node(...)` stays first-class (LangGraph runs sync nodes in a threadpool under `ainvoke`). `async def` nodes are added for the I/O path. We add a dimension; we do not rewrite the surface.
2. **One Construct, many drivers.** The same compiled `Construct` is drivable by `run()` (sync), `arun()` (async), `stream()` (sync events), and `astream()` (async events). Compilation happens once; the driver chooses the execution/observation mode.
3. **Three-surface parity extends to async.** `@node` / declarative / programmatic must all reach the async + streaming behavior identically. This is the existing parity rule (AGENTS.md) with an async axis added.
4. **Single seam.** Consumers never import `langgraph.types.interrupt` / `get_stream_writer` / `ainvoke` directly. neograph exposes `ask_human`, `emit_progress`, `arun`, `stream`, `astream`, `mcp_tool`. A structural guard enforces "no reach-around" in examples and (where detectable) consumer-facing surface.
5. **Run-time bindings obey the two-lifetime rule** — a re-mintable provider/factory across checkpoints; a live handle cached per (process, loop) within a run. (§5 — the load-bearing insight.)
6. **Deterministic cancellation & cleanup (review H3).** A consumer disconnecting mid-`astream` (SSE closed, operator navigates away) must **cancel the run, dispose any live MCP session, and leave the checkpoint consistent** — deterministically. Async makes `CancelledError`/task-cancellation first-order; for a durable, interruptible, streamed agent this is the *normal* end of a UI session, not an edge case. Every I/O path is cancellation-safe; a mid-stream cancel E2E asserts session close + checkpoint consistency.
7. **Exit with today's confidence.** The suite must end this work as trustworthy as it is now: behavioral tests run through both surfaces, streaming/interrupt/MCP/cancellation each get integration + E2E coverage, and structural guards lock the new invariants. No phase lands with the suite red or coverage thinned. (Parity alone is insufficient — see review M1/M7 and §7.)

---

## 5. The load-bearing insight: stateless-by-construction bindings

A neograph graph can be **checkpointed and resurrected** — potentially in a different process, at a later time, after the original in-memory state is gone (this is exactly what mid-loop HITL and durable execution require). The invariant governs **two distinct lifetimes** and must not collapse them (per architecture review H1):

> **(1) Across-checkpoint survival.** Anything that must outlive a checkpoint → resume (possibly a fresh process) is bound as a **pure, re-invocable provider/factory keyed on run context** (`config["configurable"]`) — never captured live state — so the runtime can re-mint/re-open it on resume.
> **(2) Within-run reuse.** Between steps of the *same uninterrupted run on the same loop*, a **live handle may be cached** (pooled MCP session, HTTP connection). The cache is keyed to **(process, event-loop) lifetime and invalidated on resume** — so a fresh process misses the cache and the factory re-opens, preserving (1) while allowing performant reuse.
> **Corollary:** *serializable, deterministic effect **results*** may be captured (that is exactly Level-B journaling, §6.4). "Never captured live state" bans live handles across checkpoints, not recorded results.

Consequences that shape the MCP and auth design:

- **Auth is a token *provider*, not a token.** `config["configurable"]["mcp_auth"]["axiom"] = provider(rc)`. Provider cadence (per review M2): **invoked once per run, cached for the run's duration, re-invoked on resume** (fresh process → cache miss → re-mint fresh, attenuated, audience-bound token). The cache lives *outside* checkpoint state (a live secret, never persisted). The provider may be **`async`** so a blocking broker doesn't stall the loop; a sync provider runs in a threadpool. neograph never persists minted tokens; caching beyond the per-run window is the provider's responsibility.
- **Discovery is decoupled from the live session.** Tool *specs* (names, schemas, budgets) are cacheable/manifest-able and used by the **linter at compile time** (neograph's unique DX: compile-time validation of MCP bindings). The **linter holds no runtime session**; a live session is opened for calls and **reused within the run** (not re-initialized per `tools/call` — review H1), and disposed at run end or on cancellation (review H3).
- **Per-step server binding is fine; live sessions are cached-not-persisted.** A step declares which server(s)/tools it binds (static allowlist = the per-step filter, lintable). The connection is (re)opened from a transport factory (Strands pattern) on the executor's own loop and cached for the run; credentials are the provider above.
- **This generalizes beyond MCP.** Any run-time resource (rate limiter, DB handle, HTTP client) obeys the same two-lifetime rule: bind a factory/provider; cache the live handle for the run; invalidate on resume. A neograph-wide invariant, not an MCP special case.

The elegance payoff: the two-lifetime rule makes async + checkpointing + MCP + HITL *compose* instead of fighting each other, **without** forcing a session-hostile open-call-dispose-per-call design. It is the most important design constraint in this document.

---

## 6. Per-capability design sketch

Each is a sketch to bound the epics, not a frozen API.

### 6.1 (A) Async foundation
- **Mechanism:** LangGraph async execution; async node functions; sync nodes in threadpool.
- **neograph gap:** sync-only node wrappers (`factory.make_node_fn`), sync LLM/tool calls (`_tool_loop.py`, `_llm.py`), sync runner (`runner.py:graph.invoke`).
- **DX shape — the driver chooses async, NOT the body (review H2).** The I/O modes that motivate this work (`think`/`agent`/`act`) have **empty bodies** (`...`) — there is no `async def` to detect. Async-ness is a property of the **execution surface**: under `arun()`, LLM/tool/MCP modes run their vertical (`_execute_node`→`_dispatch`→`_tool_loop`) with `await llm.ainvoke()` / `await tool.ainvoke()`; under `run()` they stay sync. The user body's `async def`-ness governs **only** how a *scripted* body is invoked (await it vs threadpool it). New `arun()`; sync `run()` unchanged. Async checkpointer support. **Getting this wrong is silent** (tests pass; production still blocks the loop) — make it the Phase-1 key decision.
- **Hard part:** dual-path factory emitting an `async def node_wrapper` whose `_execute_node`/`_tool_loop` twin awaits the LLM/tool calls, selected by the driver (not by body inspection); the scripted shim must `await` an `async def` user body (vs threadpool a sync one); the **custom wrapper seams** need async twins — `_CoercingToolWrapper`/`abind_tools` (`_tool_loop.py:251`) and the `think`-mode structured-output/coercion path (review L3/L4). The fakes gain `ainvoke`/`abind_tools`. DI resolution (`_resolve_di_value`) is pure dict lookup — **no `await` needed** (review L3). Fan-out (`Each`/`Oracle`) already uses LangGraph `Send` — works async.
- **Coupling:** the keystone; everything else waits on it (except D's sync path).

### 6.2 (B) Streaming + `emit_progress` (q8ec)
- **Mechanism:** `get_stream_writer()(payload)` inside a node; surfaces via `graph.stream(stream_mode="custom")` / `astream`.
- **neograph gap:** **no streaming at all** — `run()` only calls `graph.invoke()`, discarding custom events. Two pieces: `emit_progress(model)` (small) + a **streaming runner** (missing, bigger).
- **DX shape:** `emit_progress(MyEvent(...))` from any node/tool; `astream(...)` yielding typed neograph events (custom + lifecycle/token, filtered/typed). Documented, stable event shape. **`emit_progress` under a non-streaming driver (`run()`/`arun()` without stream consumption) must not vanish silently (review L1) — warn once (or buffer), so a missing stream consumer is a visible mistake, not a black hole.**
- **Hard part:** the runner + event typing, not the emit. AG-UI/CopilotKit consume async SSE → `astream` only (no sync `stream()`).
- **The actual hard integration is the resume-stream protocol (review H3) — a cross-cut between this phase (B) and HITL (D), NOT two independent features.** The AG-UI payoff is one stream that carries progress events, *pauses at an interrupt*, surfaces the interrupt payload to the UI, and *resumes the same logical stream* on the human's answer: `astream(...)` → custom events → interrupt event → generator returns → consumer calls `astream(Command(resume=answer))` → stream continues. This `astream ⇄ interrupt ⇄ astream` handshake is a **named deliverable** (owned jointly by Phases 2 and D) with a **typed contract** for how an interrupt appears in the event sequence and how a resume re-enters it — written as its own E2E, not hand-waved as "AG-UI event mapping."
- **Checkpoint safety:** custom events (and per-call tokens) are ephemeral, not state → must never touch the schema fingerprint (`state.py`) **or any log/ToolInteraction record** (review L2). Add guards.

### 6.3 (C) MCP client
- **Mechanism:** langchain-mcp-adapters (async-only) + the MCP auth spec.
- **DX shape (from the landscape doc §4):** `compile(mcp_servers={...})` typed server config with a factory transport; `mcp_tool("server","tool",budget=…)` reusing the `Tool` spec (MCP tools are not special); per-run token **provider** via `config["configurable"]` (§5, FR below); mandatory namespacing `server.tool`; **linter validates bindings at compile time** (the differentiator).
- **Hard part:** the two-lifetime binding design (§5); per-step binding + budgets interplay.
- **MCP requires `arun()` (review M4).** MCP clients are async-only. Async does not *remove* the sync bridge — it removes the *need* for it by requiring the async driver. An MCP tool bound in a graph driven by sync `run()` would re-introduce the background-loop-thread bridge, contradicting "async removes it." Resolution: **the linter flags an MCP tool under a `run()`-driven graph** (error/warn); MCP ⇒ `arun()`. `run()` stays first-class for non-MCP graphs. No sync bridge is built.
- **Coupling:** wants (A). Parallelizable with Phase 2, not strictly after it (review M5). Later: resources/prompts/sampling (MVP is tools-only) — enabled by the within-run session reuse of §5 (H1).

#### 6.3.1 FR — per-run MCP identity via a token-provider hook (on-behalf-of, no passthrough)

The direct application of the §5 stateless-binding invariant to auth. This is a filed feature request, folded into Phase 3.

- **What.** A per-run hook: per MCP server, the consumer supplies a **callable that returns the bearer token, bound from run context**, so an MCP-backed tool stamps `Authorization: Bearer <token>` on every `tools/call`. Shape:
  ```python
  config["configurable"]["mcp_auth"][server] = lambda run_ctx: <jwt>
  ```
  **Static `headers=` stays the simple default; the provider callable is the per-run escape hatch.** The provider is exactly the "re-mintable function of run context, never captured live state" of §5 — it survives checkpoint→resume because on resume the runtime re-invokes it to mint a fresh token.
- **Why.** Multi-operator consumers must pass the **end-user's** identity to the tool server, which runs its **own** authorization (on-behalf-of; ADR-0009). The MCP spec makes **token passthrough a hard MUST-NOT** and requires **audience binding (RFC 8707)**. neograph **carries** identity, **never decides** on it. Most frameworks bind headers at construction (fresh-client-per-user); the spec's per-request model wants a token provider — the open gap neograph can win on.
- **Contract / expectations.**
  - The tool obtains its token by calling the provider with **run context** (identity + tenant/deal keys) **at call time** → fresh, attenuated, **audience-bound** tokens.
  - neograph **never inspects** the token (opaque; no parsing, no authz decision).
  - Transport split per spec: **HTTP → bearer header; stdio → env creds (no OAuth).**
  - **Audience binding is *declared* client-side, *enforced* server-side (review H4 — resolves the contradiction with landscape §4.1's "enforced client-side").** The server-config `audience` URI is the value the consumer's provider is expected to bind (RFC 8707); the **tool server** (the OAuth resource server) validates the `aud` claim on receipt. neograph carries the token opaquely and does **not** parse `aud` — consistent with carries-not-decides. (Landscape §4.1 wording to be corrected: "declared client-side, enforced server-side.") *Trade-off accepted:* a buggy/malicious provider returning a wrong-audience token is caught by the server, not by neograph — spec-correct, since passthrough-prevention is the resource server's job. A future opt-in non-authz `aud`-equality safety check is possible but would cross the never-parse line and is out of scope for the MVP.
  - **No verbatim upstream-token forwarding** — the provider mints/attenuates; neograph never forwards a received user token as-is.
- **Verification (integration + E2E).**
  - Integration: register a server with a provider; run under **two** run-context identities; assert each `tools/call` carried the correct per-identity token at a **stub server**, and that the **no-auth default path is unaffected**.
  - E2E: one tool call resolves **different server-side authorization** under two operators.
- **Acceptance.** Per-server token-provider callable accepted from run config; token stamped per call; static `headers=` still the default; stdio→env / HTTP→bearer; neograph makes **no** authz decision (enforcement documented as server-side).
- **Design guardrails this imposes on the broader work:**
  - The provider must be invoked **at call time on the executor's loop**, not captured/frozen at bind time (so resume re-mints).
  - The token must never enter checkpointed state or the schema fingerprint (it's a per-call secret, not durable state) — a guard, analogous to the custom-event fingerprint guard (§6.2).
  - A structural guard: neograph code paths never read/parse the token's contents (carries-not-decides).

### 6.4 (D) Mid-loop HITL — `ask_human` (m6d3)
- **Mechanism:** `interrupt(payload)` from inside a tool; on resume the node re-runs, the resolved `interrupt()` returns the resume value. Requires checkpointer. All machinery exists (`_wiring.py:681`, `runner.py:314`, `human_feedback` state key). Tool execution site (`_tool_loop.py:324`) and dispatch layer have **no broad `except`** → a `GraphInterrupt` propagates correctly today.
- **DX shape:** `ask_human(payload: BaseModel) -> ResumeModel` callable from a tool; typed payload/resume matching the `human_feedback` shape.
- **The real design decision (the ticket underplays it):** LangGraph re-executes the **whole node from the top** on resume. For an `agent` node that means **re-running the ReAct loop** — re-invoking the LLM (cost) and re-calling prior tools — unless made idempotent. `interrupt()` is cached (won't re-pause) but the surrounding loop is not. Two levels:
  - **Level A (ticket V1) — DECIDED 2026-07-02:** document the idempotency footgun; accept re-execution. Fine for read-only research tools; the docs must be explicit that costly/mutating tool calls before an `ask_human` are re-executed on resume. m6d3 ships at Level A.
  - **Level A.5 — a lint check (review M3), shipping WITH Level A.** neograph already distinguishes mutation intent by mode: `agent` = read-only tools, `act` = mutations (CLAUDE.md Modes). So the footgun is *statically detectable*: `ask_human` inside an **`agent`** node is provably safe (read-only re-execution is idempotent); inside an **`act`** node it is the footgun. Ship a lint rule that flags `ask_human` in an `act`-mode node (and/or a mutating-tool call preceding `ask_human` in the same node) — converting "document the footgun" into "catch the footgun," consistent with the linter-is-the-gate differentiator. Note the safety-profile jump the "all machinery exists" framing hides: today's `interrupt()` lives in a **dedicated micro-node** (`operator_check`, `_wiring.py:685`) that only checks-and-interrupts — *no* re-execution problem; `ask_human` deep inside a fat agent node is categorically more dangerous. **Level A→B compatibility contract:** Level B must preserve results for tools idempotent under Level A (it *narrows* re-execution — an observable change for anyone who wrongly relied on double-execution).
  - **Level B (proper) — deferred to a research track:** true replay-safety (fast-forward resume without re-calling LLM/tools) is a hard, general **durable-execution** problem, not a neograph quirk. Before attempting it we researched how the field solves replay/idempotency *in principle*. See `docs/design/durable-execution-replay-research-2026-07-02.md` — key result: the durable-execution field (Temporal, Azure Durable, Inngest) solves this by **step-granular effect memoization** (each non-deterministic effect, explicitly incl. "LLM/AI invocations", is a journaled activity, never replayed), while LangGraph memoizes at **node granularity** — too coarse for a ReAct loop. So Level B ≈ "build a checkpoint-keyed step-memoization layer in the tool loop", deferred until a real consumer needs it and the async foundation has landed. Even Temporal-class engines rest on user idempotency for the crash window, so Level A is the field's floor, not a cop-out.
- **Coupling:** independent of async; ships sync. Shares `_tool_loop.py` with (A)/(C) — sequence to avoid churn conflicts.

---

## 7. Testing & correctness strategy (the primary concern)

The explicit goal: **exit this work as trustworthy as we enter it** — no known bugs, strong coverage. The threat is that async + streaming touch nearly everything and "all tests today are synchronous."

- **Dual-surface parity harness (the central idea).** Introduce a parametrized fixture that runs each behavioral test through **both** `run()` and `arun()` (and, where relevant, `stream()`/`astream()`). This is the async analogue of the three-surface parity rule: the two execution surfaces cannot silently diverge because every behavioral assertion is made against both. This converts "all tests are sync" from a liability into a coverage multiplier.
- **Fakes gain async.** `tests/fakes.py` LLM fakes (`StructuredFake`, `ReActFake`, `TextFake`, `GuardFake`, …) grow `ainvoke`/`abind_tools` mirrors. Keep one source of truth (a shared `_final_json_content`-style helper) so sync and async fakes can't drift.
- **Incremental, gate-preserving rollout.** Async lands as an added dimension behind the dual-surface harness, not a big-bang rewrite. Every phase lands with `make quality` green; no phase leaves the suite red for "a while."
- **Feature-specific coverage (integration + E2E, per the tickets):**
  - Streaming: assert the full typed event sequence, interleaved with lifecycle events; assert a consumer can reconstruct stage progression from the stream alone (q8ec E2E).
  - HITL: run to the interrupt, assert typed pause payload; resume with a structured answer; assert the tool receives it; **assert no duplicate side effects across resume** (idempotency E2E — this is the test that pins the Level A/B decision).
  - MCP: compile-time lint catches an unknown tool / arg-type mismatch (the differentiator); a run mints a fresh token via the provider on resume (stateless-auth E2E); namespacing collision is flagged.
- **New structural guards (lock the invariants):**
  - No reach-around: examples/consumer surface don't import raw `interrupt`/`get_stream_writer`/`ainvoke`.
  - Custom events never enter the schema fingerprint.
  - Run-time bindings are providers/factories, not captured live handles (as far as statically detectable).
  - Dual-surface: LLM modes have both a sync and async dispatch path (no sync-only mode).
- **Parity is necessary but NOT sufficient (review M1).** Running every behavioral test through `run()`/`arun()` with fakes whose `ainvoke` mirrors `invoke` proves the *plumbing* is wired but proves nothing about what async actually changes. Add the three tests parity structurally cannot be: (1) **real-concurrency** — N runs in flight on one loop, asserting interleaving/isolation (the headline benefit, never otherwise exercised); (2) **event-loop-blocking watchdog** — a slow sync call under `arun()` trips a loop-lag detector (a blocking `broker.mint()`/tool fails *no* functional assertion yet destroys concurrency); (3) **cancellation E2E** (shared with §4.6/H3). Target the parity doubling at I/O-bearing tests, not pure/scripted nodes (LangGraph threadpools those — near-zero async signal).
- **It is a 6-cell grid, not a doubling (review M7).** Three-surface parity (`@node`/declarative/programmatic) × two execution surfaces = **6 combinations** per IR-level behavioral change — and three-surface is *already* the top bug source (CLAUDE.md). Decide where full 6-cell coverage is mandatory (IR-level behavioral changes) vs a representative subset (execution-surface-agnostic rendering/validation), and bake that into the harness so it isn't a wall-clock sink.

---

## 8. Sequencing

Phased, each phase a beads epic derived from this doc, each landing green.

**Priority (decided 2026-07-02):** the async streaming path is the lead train — neograph is a key component across multiple systems the maintainer designs, so **async + `astream` come first**, ahead of MCP. Sync `stream()` is **dropped** (we go straight to async streaming; no throwaway). HITL ships small at Level A in parallel. **Caveat (review M5):** this is a maintainer-preference ordering, not a consumer-readiness derivation — there are two concrete filed needs (ox-troubleshooting-demo: MCP+auth; agent-stark: AG-UI streaming/HITL). Rank them by deadline/readiness before locking the lead train; flip if ox is closer to shipping. **Phases 2 and 3 both depend only on Phase 1 — they are parallelizable; serializing 2-then-3 is a resourcing choice, not a critical path** (a second implementer can take MCP without a false block).

| Phase | Epic | Depends on | Ships | Key decision to make first |
|---|---|---|---|---|
| 0 | **Dual-surface test harness + async fakes** | — | internal | shape of the parametrized run/arun fixture |
| 1 | **Async foundation** (arun, async node wrappers, await llm/tool, async checkpointer) | 0 | `arun()` | dual-path factory design; async sidecar/DI |
| 2 | **Async streaming + `emit_progress`** (q8ec) — **LEAD FEATURE** | 1 | `astream()` (no sync `stream()`) | typed neograph event shape; AG-UI event mapping |
| 3 | **MCP client** (per-step bind, stateless auth, compile-time lint) | 1 | `mcp_servers=`, `mcp_tool()` | discovery source of truth (live vs manifest); provider signature |
| D (parallel, small) | **Mid-loop HITL — Level A** (m6d3) | checkpointer (exists); coordinate on `_tool_loop.py` | `ask_human()` sync | none (Level A decided; Level B is a research track) |
| R (research) | **Durable-execution replay research** | — | design doc | scope of the survey |

- Phase 0 is **scaffolding only** (the parametrization mechanism + the async-fake source-of-truth), **co-evolving with Phase 1** — do NOT gate Phase 1 on a "complete" harness, since the `run/arun` fixture can't be finalized against `arun()` semantics that don't exist until Phase 1 (review M5).
- Phases 0→1→2 are the lead train (async foundation → async streaming). MCP (3) is parallelizable with 2 on the same async substrate.
- HITL (D) is small at Level A (+ the A.5 lint) and runs in parallel on the sync track; coordinate `_tool_loop.py` edits with Phases 1/3.
- Research (R) runs independently now and gates any future HITL Level-B / durable-execution work.
- **Phase-1 acceptance must include (review M6):** (a) **observability survives `await`** — trace spans (the Langfuse `observable_pipeline` example) propagate across an `await` boundary (async breaks naive `contextvars` propagation — a known LangChain footgun); (b) **mixed-driver checkpoint resume** — a thread started under sync `SqliteSaver` resumes under `AsyncSqliteSaver` on the same `thread_id` (resolve §9.2, don't leave it open); (c) a **performance gate** — a benchmark asserting the async path adds no per-call overhead beyond a threshold, exercising within-run MCP session reuse (§5/H1).

---

## 9. Risks & open questions

1. **HITL replay-safety** — RESOLVED: Level A (document footgun) ships now; Level B waits on the durable-execution replay research (§6.4, doc R).
2. **Async checkpointer story** — sync uses SqliteSaver; async wants AsyncSqliteSaver/AsyncPostgresSaver. Confirm resume semantics across sync/async drivers of the same thread_id.
3. **Sync `stream()`** — RESOLVED: dropped. We go straight to async `astream` (Phase 2 is the lead feature). No sync streaming runner.
4. **asyncio-per-call latency** (MCP doc §5.2) — validate the async path doesn't add per-call overhead that hurts the common case.
5. **Test-matrix cost** — doubling behavioral runs; mitigated by the harness, but real. Watch suite wall-clock.
6. **Backward compatibility** — `run()`, sync nodes, existing checkpoints must keep working unchanged. No deprecation of the sync surface.
7. **Discovery source of truth for MCP** — live connect at lint time vs checked-in manifest (offline/reproducible CI). Likely live-with-manifest-fallback (doc §5.1).

---

## 10. What this unlocks

This is not four isolated conveniences — together they roughly double neograph's utility surface for production agents:

- **Any agent can bind any MCP server, varied per step, with per-user identity** — and (uniquely) neograph validates those bindings at compile time. Later: MCP resources / prompts / sampling.
- **Production-grade concurrency** — one event loop serves many in-flight, network-blocked runs instead of one-thread-per-run.
- **A first-class AG-UI surface** — streamed domain progress + mid-loop HITL, so an operator UI (CopilotKit) consumes neograph directly through one sanctioned seam.
- **Durable, resurrectable interactive agents** — checkpoint through a human decision or a process restart, resume with fresh-minted credentials, because bindings are stateless by construction.

The correctness phase built a foundation you can trust. This phase makes that foundation *useful in production* without spending the trust — provided we treat async as the keystone, hold the stateless-binding invariant, and let the dual-surface harness keep us as honest as we are today.

---

## 11. Next steps
1. Review this doc (maintainer + optionally the architect agent) and settle the open decisions in §9 — especially HITL Level A/B and the sync-`stream()` question.
2. File Phase 0–3 + D as beads epics that **cite this doc**, each with the anti-band-aid invariant clause and the dual-surface coverage requirement.
3. Start with Phase 0 (the harness) — it is the thing that lets every later phase land without eroding trust.
