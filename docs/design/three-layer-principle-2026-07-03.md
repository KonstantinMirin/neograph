# The Three-Layer Principle: DX from compile-time + node-internals, engine surface untouched

**Date**: 2026-07-03
**Status**: Adopted (this doc) — re-scopes epic neograph-w74k phases 2/3/D
**Companion docs**: `async-streaming-mcp-hitl-2026-07-02.md` (sequencing), `durable-execution-replay-research-2026-07-02.md` (replay research), `architecture-review-async-streaming-mcp-2026-07-02.md` (adversarial review)

---

## 1. The principle

neograph's value must come from exactly two places, and never a third:

1. **neograph-compile** (Layer 1): decorators / IR / validation in, validated
   LangGraph `StateGraph` out. Engine imports are legal here — `StateGraph`,
   `Send`, edges, subgraph compilation, boundary-interrupt node insertion.
   This is compile-time *topology emission*, not runtime wrapping.
2. **neograph-node-runtime** (Layer 2): a small, self-contained helper library
   that compiled **node bodies** call — prompt rendering, LLM invocation,
   schema-aligned parsing, retry-on-validation-error, DI resolution. It needs
   sync/async duals, but it is a contained surface with zero engine
   entanglement. LangGraph never knows it exists.
3. **The engine surface** (Layer 3) belongs to LangGraph, untouched:
   scheduling, streaming, checkpointing, interrupts, `invoke`/`astream`/
   `astream_events`. neograph's entry points are **thin verbs** — `prepare()`
   → engine verb → `finalize()` — and every engine capability must remain
   reachable. neograph must never make an engine capability *harder* to
   access than raw LangGraph.

**The one-sentence version**: *DX comes from in-node improvements; the
execution runtime stays fully transparent.*

### 1.1 Why this is the load-bearing rule

The Phase-1/2/3/D parity debt (async, streaming, MCP tools, custom events) was
never four missing features. It was **one** missing property: the wrapper owned
the execution path instead of only the assembly path. A DX layer that adds a
compile phase and zero runtime wrapping never has parity debt again — `astream`
works because it is LangGraph's `astream`; MCP tools bind because they are
plain LangChain tools by the time a node sees them; custom events work because
`stream_mode="custom"` + `get_stream_writer` already exists.

This is also the property that makes "neograph as a spec over multiple
engines" credible: a compile phase plus a node-runtime library transfers to any
graph-style framework; a runtime wrapper transfers to none.

### 1.2 The purity test (sorts every future feature)

> **Side-effect-free cognition may live inside a node body; anything
> side-effectful or interactive must sit at a superstep or task boundary the
> checkpointer can see.**

Applications:

- The validate-retry loop (failed parse → fix-it call) is pure, cheap, and
  safe to replay → Layer 2, stays in the helper. No journaling needed.
- A tool call that mutates external state fails the test → must be a
  boundary (a tool node in a subgraph, or a `@task`-journaled step).
- An `interrupt()` waiting on a human fails the test → must be a boundary
  node the checkpointer can see (the Operator modifier already does this
  correctly: it compiles to a dedicated check node *after* the producer).
- The schema-aligned parsing / JSON-repair strategy passes the test → Layer 2.
  This is not incidental: that helper is what makes model choice a config
  value rather than a rewrite (output-contract-level agnosticism — the level
  that matters for typed channels — vs. LiteLLM-style transport agnosticism).

The mid-loop-interrupt problem restated: it was painful precisely because
something non-pure (tool calls) was living inside a node body
(`_tool_loop.py`'s `while True`). The fix is not smarter idempotency
documentation; it is moving the boundary — compile agent mode to an
agent-node / tool-node subgraph so every ReAct turn is a superstep.

### 1.3 What is explicitly allowed (justified exceptions)

- **Sub-constructs invoked inside a wrapper function** (`_subconstruct.py`):
  neograph sub-constructs have their own schema (input scanned from parent by
  type, output scanned back). LangGraph's own documented rule: shared state
  keys → add compiled subgraph as node; *different schemas → invoke inside a
  wrapper function that transforms state*. Isolation-by-type forces
  function-form. Verdict: **verify propagation** (streaming `subgraphs=True`,
  interrupt/persistence through the invoked child), not restructure.
- **`compiler.py`'s three-way isinstance match** — an irreducible sum-type
  dispatch, pre-existing sanctioned exception (see AGENTS.md).
- **Boundary-interrupt emission**: Layer 1 inserting a dedicated interrupt
  node is topology emission, not runtime wrapping.

---

## 2. Audit findings (2026-07-03, post-Phase-1 develop)

_Consolidated from four parallel audit agents (engine-touchpoint
classification, runner responsibility inventory, tool-loop/subconstruct/MCP
readiness, docs/examples DX-promise sweep — the last fanned into four child
sweeps covering concepts/getting-started/comparison, all 23 examples,
walkthrough/forward/node-api/runtime docs, and README + design docs)._

### 2.1 Engine touchpoints: ~95% clean

Engine execution verbs are confined to exactly two sanctioned places —
`runner.py` top-level `run`/`arun` and `_compiled.py`'s explicit delegations.
Every LLM/tool `.invoke` hit (`_llm_*`, `_tool_loop`, `_oracle`,
`node.run_isolated`) is Layer-2 node-internal cognition, not scheduling. The
Operator `interrupt()` fires in a compile-emitted boundary node
(`_wiring.py:684`, inserted by `_add_operator_check`), which is the correct
design. Three auditors independently converged on the same conclusion: the
compile layer — the part that is actually neograph — is clean; the rot
concentrates in the facade allowlist, the runner verbs, the tool loop, and
`_subconstruct.py`.

### 2.2 Facade: an accidental allowlist (violation E)

`CompiledNeograph` delegates `invoke`/`ainvoke`, `get_state`/`aget_state`,
`get_state_history`/`aget_state_history`, `astream`, `get_graph`,
`checkpointer`, `builder`. Missing, by priority:

| Missing | Priority | Why |
|---|---|---|
| `stream` (sync) | needed-now | `astream` is delegated but sync `stream` is not — asymmetric; sync consumers blocked by construction |
| `astream_events` | needed-now | primary token/step observability verb; a real gap for an "observable" framework |
| `update_state` / `aupdate_state` | needed-now | HITL resume with state edits cannot patch state; resume today only via `Command(resume=)` |
| `stream_events`/`astream_log`, `bulk_update_state`, `get_subgraphs`, `with_config` | later | secondary |
| LCEL-Runnable surface (`batch`, `as_tool`, `pipe`, …) | never (for now) | deliberately not exposed |

**History is exculpatory and decisive**: the facade was introduced by
`architecture-review-solid-grasp-2026-06-02.md` §R6 to fix a DIP violation
(killing `getattr(graph, "_neo_*")` reads) — a *typing* fix. Its allowlist
nature was a by-product, never a deliberate capability gate. `stream`,
`astream_events`, and `update_state` fell off the surface by accident. The fix
is "complete the allowlist for the claimed capability set", not "rethink the
facade".

**Reconciling two invariants**: the sequencing doc's invariant 4 ("single seam
— consumers never import `langgraph.types` directly") and this doc's "every
engine capability must remain reachable" are compatible only when stated
precisely: *every claimed capability is reachable THROUGH the facade/runner
verbs, never by reaching around them*. The facade stays closed; the allowlist
must be complete. An in-repo example already violates the seam under pressure:
`examples/17_fanout_resilience.py:121` resumes via raw `graph.invoke(None,
config)`, bypassing `run()`'s checkpoint-verify and `_strip_internals` —
exactly the reach-around the sequencing doc predicted consumers would do.

### 2.3 Runner: the extraction is 80% done; the new code is `_finalize_chunk`

Phase 1 already extracted the shared brain further than feared:
`_inject_input_to_config`, `_prepare_new_input`, `_prepare_resume_config`,
`_strip_internals`, and the whole invalidation/adjacency/closure cluster are
shared between `run` and `arun`; the sync/async twins
(`_has_existing_checkpoint`/`_ahas…`, `_verify_checkpoint_schema`/`_averify…`,
`_auto_resume_from_divergence`/`_aauto…`) are deliberate and guard-pinned.
Line-by-line comparison found **no logic drift** between `run()` and `arun()`
(only comment drift: `run()`'s load-bearing inline comments at
`runner.py:351-354` and `:361-368` are absent from `arun()`).

What remains for stream verbs:

1. Collapse the three inline mode branches of `run`/`arun` into
   `_prepare(graph, *, input, resume, config, auto_resume) ->
   (engine_input, config)` where `engine_input` is
   `input_dict | Command(resume=) | None`, plus an `_aprepare` twin.
   All twelve pre-engine responsibilities apply to streaming **as-is**; the
   auto-resume rewind is pure pre-flight config mutation and carries over for
   free — with one caveat: it must run inside `_prepare`, not lazily inside
   the generator, or the first chunk fires against the un-rewound checkpoint.
2. The genuinely new code: `_finalize_chunk(chunk, stream_mode)` —
   stream_mode-aware stripping. `values`: strip `neo_*` (as today).
   `updates`: strip one level down (per-node deltas can carry fingerprints).
   `custom` / `messages` / `debug`: pass through untouched (user payloads and
   token tuples must NOT be stripped). Must handle `stream_mode` as str OR
   list (LangGraph then yields `(mode, chunk)` tuples).
3. Facade delegations per §2.2.
4. Twin-guard tables (`test_guards_async_dispatch.py`, `test_guards_meta.py`)
   need `_prepare`/`_aprepare` added or co-location isn't enforced.

**Weakly pinned responsibilities (pin BEFORE refactoring — guard-first)**:
the defensive input copy is *entirely unpinned* (an in-place mutation would
leak `neo_*` into the caller's dict with zero test failure); CONFIG_INPUT
stash/re-inject, fingerprint-injection ordering, and preflight-DI ordering
are only transitively pinned. Estimated size: ~half a day of mechanical work
plus `_finalize_chunk` and its per-mode tests.

### 2.4 Tool loop: monolith confirmed; the migration hinges on ONE decision

`_tool_loop.py` (788 lines): sync/async twins share pure `_prepare_tool_loop`
/ `_finish_tool_loop`; the `while True` bodies + parse-fallback tails
(~360 lines total) are duplicated because control flow interleaves awaits.
An agent-as-subgraph recompile (agent-node / tool-node cycle, conditional
edges) **deletes those ~360 lines** and keeps the helpers as node bodies and
compile-time preambles.

Concern disposition (full table in audit): preambles/tool-instantiation →
compile-time (trivial); LLM call, tool dispatch, no-tool-calls exit →
node bodies + conditional edge (trivial); tool_log, budget counters,
iteration/token guards → **subgraph state channels** (moderate — today
in-memory locals/`ToolBudgetTracker`, which per-turn checkpoints would reset);
final parse + structured fallback chain + usage summation → **hard, and all
three reduce to one prerequisite**: the fallback parse
(`_call_structured`/`recover_dsml`/`_invoke_json_with_retry`) needs the FULL
message history.

> **The load-bearing design decision**: message history (plus budget/iteration
> counters and tool_log) becomes subgraph state channels — LangGraph
> `add_messages`-style accumulation. Do that and the hard cluster becomes an
> ordinary parse node reading `state["messages"]`; the guard/unbind dance
> becomes conditional edges; per-turn checkpoints, mid-loop interrupt at turn
> boundaries, agent token streaming, and honest budget enforcement all follow
> from the same structural change.

Blast radius: concentrated and predictable — ~50–70 internal-reaching tests
rewrite (49 direct `invoke_with_tools` calls in
`tests/modes/test_llm_internals.py`, 16 observability-contract reaches, the
guard/stubborn fakes become edge-routing tests); the through-`run()` majority
and all examples survive because they assert on final typed output.

### 2.5 Sub-construct: one Phase-1 miss and two smaller defects

1. **`_subconstruct.py` has NO async twin (Phase-1 gap — the audit's biggest
   correctness finding).** One sync `subgraph_node` calling
   `sub_graph.invoke(...)` (`_subconstruct.py:117`); no
   `ainvoke`/afunc anywhere in the module. `arun()` over any pipeline
   containing a sub-construct runs the child **synchronously inside the async
   driver** — blocks the event loop, and any async-only leaf (MCP tool, LLM
   `ainvoke`) inside the child runs through LangGraph's sync path. **Zero
   test coverage** (no test drives `arun()` over a sub-construct; the async
   tool tests are flat pipelines). Violates Phase 1's own H2 invariant
   (driver-selected async) by silently downgrading the child to sync.
2. **Branch arms compile sub-constructs with `checkpointer=None`**
   (`_wiring.py:576,585`) while the main path threads the parent checkpointer
   (`compiler.py:213`). Caveat from the audit: LangGraph subgraphs *may*
   inherit the parent checkpointer via config propagation, so the two paths
   may coincidentally behave identically — which is exactly why this wants a
   failing-test-first investigation (branch arm → sub-construct with Operator
   → interrupt → resume → assert continues-not-restarts), either outcome
   informative.
3. **Layer inversion**: `_subconstruct.py:63` imports `_strip_internals` from
   `runner` (function-local, dodging the circular import that the inversion
   itself creates). `_strip_internals` is a result-shaping utility; it belongs
   in a neutral module (`_state_keys`), imported by both.

The function-form sub-construct itself remains a justified exception (§1.3).

### 2.6 MCP readiness: "not harder through neograph" holds — narrowly

The tool invocation paths are correctly twinned: sync loop calls
`tool_fn.invoke` (`_tool_loop.py:462`), async loop awaits `tool_fn.ainvoke`
(`:691`). Consequences:

- **Zero code changes for the happy path**: load MCP tools once ahead of
  compile (`load_mcp_tools(session)`), wrap each as `Tool(name)` +
  `register_tool_factory(name, lambda c, tc: preloaded_tool)`, use agent
  mode, call `arun()` on a **flat** pipeline — works today. The factory being
  invoked synchronously (`_tool_loop.py:276`) is a non-issue for this
  pattern (the factory closes over a pre-loaded tool).
- **Hard prerequisite for MCP under composition**: finding §2.5(1) — without
  the sub-construct async twin, an MCP-tool agent nested in a sub-construct
  blocks the loop.
- **Ergonomics deltas (optional)**: accept a raw LangChain `BaseTool`
  directly in `tools=` (`node.py:172` union + assembly-time normalization,
  auto-registering the factory); replace the sync loop's bare
  `NotImplementedError` on async-only tools with a clear "MCP/async tools
  require arun()" error at `_tool_loop.py:462`.

This shrinks Phase 3 substantially: the MCP client work is mostly *removing
barriers and adding lint/errors*, not building a client. The token-provider
FR (w74k.3.1) is unchanged.

### 2.7 Docs and examples: the gap is concentrated, and partly self-contradictory

The README and the walkthrough/API docs are honest (batch-only, no streaming
promises — the gap there is *silence*: `arun` and the async facade verbs are
documented nowhere). The breakage concentrates in four places:

| Where | Claim | Status |
|---|---|---|
| `why-neograph.mdx` §What You Keep | "the compiled graph supports `.stream()` and `.astream_events()`" | **BROKEN** — both 404 on the facade |
| `what-is-neograph.mdx` L8 | "full access to checkpointing, streaming, and the LangGraph ecosystem" | **BROKEN** (streaming part) |
| `quick-start.mdx` L186 | "You can stream it" | **BROKEN** (sync `stream` not delegated) |
| `human-in-the-loop.mdx` L6 (+ `llm-driven.mdx` L16) | output "approved **before side effects**; a privileged operation requires confirmation" | **CONTRADICTS-PITCH** — tool-gating HITL that the while-True agent loop structurally cannot deliver; interrupts are node-boundary only, and no doc discloses that agent/act nodes are opaque to interrupts |

Notably, `comparison/overview.mdx` L236 honestly concedes "advanced streaming
modes" aren't wrapped — the site contradicts itself between the comparison
page and the landing pages. Also soft-broken: "deploy with LangGraph
Platform" (facade needs unwrapping via `.builder`; not demonstrated).

Examples: all 23 are batch `run()`-and-print; interrupt/resume appears in 09
and 10 (correct, node-boundary); `17_fanout_resilience.py:121` bypasses the
runner (see §2.2). Best demo hosts: `observable_pipeline.py` for
`arun`/`astream` (real latency makes streaming visible), `13` for MCP
(one-factory swap via `tool_factories`, typed tool results map to MCP
structured outputs), `15` for loop-boundary HITL (Operator on the `review`
node inside a Loop).

---

## 3. Re-scoped plan

The principle doesn't add work — it re-shapes and mostly **shrinks** the
filed epic. Changes relative to the 2026-07-02 sequencing doc:

### 3.1 New defect tickets (independent of the epic, file immediately)

- **B1 (P1): `_subconstruct.py` async twin missing.** `arun()` over a
  sub-construct blocks the event loop (§2.5.1). TDD: failing test proving
  the child runs sync under the async driver (thread-identity /
  GatedAsyncFake discriminators from Phase 0 apply directly). Blocks
  MCP-under-composition; violates the Phase-1 H2 invariant today.
- **B2: branch-arm sub-constructs compile with `checkpointer=None`**
  (§2.5.2). Failing-test-first; either outcome informative.
- **B3: relocate `_strip_internals`** to a neutral module (`_state_keys`),
  killing the `_subconstruct → runner` layer inversion (§2.5.3). Natural to
  fold into the q8ec runner refactor; filed separately so it isn't lost.

### 3.2 q8ec (Phase 2, streaming) — re-scoped DOWN

Was: "async streaming + emit_progress." Is: **finish the Phase-1 extraction**.

1. Guard-first: pin the four weakly-pinned runner responsibilities (§2.3)
   with name-level tests BEFORE refactoring.
2. `_prepare`/`_aprepare` + `_finalize`/`_finalize_chunk`; `run`/`arun`/
   `stream`/`astream` become thin verbs; `astream_events` passes through
   un-finalized.
3. Facade delegations: `stream`, `astream_events`, `update_state`/
   `aupdate_state`.
4. Twin-guard tables updated; structural guard for the three-layer rule
   (§3.5).
5. `emit_progress`/custom-events and the resume-stream protocol stay as
   scoped in the sequencing doc (they ride on `stream_mode="custom"`, which
   the runner verbs now expose).

### 3.3 m6d3 (Phase D, HITL) — re-scoped UP, mechanism replaced

Was: Level A (document idempotency) + A.5 lint. Is: **agent-as-subgraph**.

Compile agent/act mode to an agent-node/tool-node subgraph with message
history, budget counters, and tool_log as state channels (§2.4). Then:
turn-boundary checkpoints make Level-A interrupt correct *by construction*
(idempotency documentation becomes unnecessary at the boundary); tool-gating
HITL ("approve before side effects" — the promise `human-in-the-loop.mdx`
already makes) becomes deliverable as an Operator on the tool node; agent
token streaming and honest budget enforcement fall out of the same change;
`@task`-per-tool-call journaling (Temporal-lite, per the replay research)
becomes reachable later. Blast radius is known and concentrated (§2.4).
The A.5 lint reconciliation turned out to be a no-op: the migration landed
atomically (agent/act nodes ALWAYS compile to the real agent/tools/parse
subgraph), so the transitional "agent/act + Operator without subgraph compile →
warn" lint was never needed and was never built. Tool-gating is delivered
instead by the `gate_tools_when=` node kwarg (neograph-m6d3.4), which inserts a
gate on the `{node}__tools` boundary. The unrelated `ask_human_in_mutating_node`
lint (from p8wz) stays as-is.

### 3.4 w74k.3 (Phase 3, MCP) — re-scoped DOWN

Was: "MCP client." Is: **barrier removal + ergonomics** — B1 as hard
prerequisite; accept raw `BaseTool` in `tools=`; clear sync-loop error for
async-only tools; lint rule "MCP/async-only tool ⇒ arun()" (review M4);
example 13 MCP variant. The token-provider FR (w74k.3.1) is unchanged.

### 3.5 New: structural guard for the three-layer rule

AST guard: engine execution verbs (`.invoke`/`.ainvoke`/`.stream`/`.astream`/
`.astream_events` on a compiled graph) appear only in `_compiled.py`
delegations and `runner.py` verbs; allowlisted exception for
`_subconstruct.py`'s wrapper-invoke (both drivers, post-B1). Locks the
principle the way the H2 dual-path guard locks async.

### 3.6 Docs (can start now, independent of code)

Fix the four broken/contradicting claims (§2.7) truthfully NOW — then update
them again as each phase lands capability. Document `arun()` and the async
facade verbs. Disclose node-boundary-only interrupts on the HITL page until
§3.3 ships. Add the "17 bypasses the runner" example to the seam-discipline
fix list (switch it to `run(config=...)`).

### 3.7 Sequencing (updated)

```
B1 (subconstruct async twin)  ──┐            [P1 bug, do first — small]
B2, B3                          │            [small, parallel]
                                ▼
q8ec' (runner verbs + facade + finalize_chunk + guard)   [shrunk]
                                ▼
m6d3' (agent-as-subgraph)       [grown — the new keystone for HITL,
                                 agent streaming, honest budgets]
                                ▼ (parallel after q8ec')
w74k.3' (MCP barriers + ergonomics + auth FR)            [shrunk]
```

Docs fixes (§3.6) run immediately and update per phase. Net effect vs the
old plan: two phases shrink, one grows into the structural change that was
always the real fix, and three latent defects surface before they bite a
consumer.

