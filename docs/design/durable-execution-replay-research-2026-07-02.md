# Durable-execution replay & idempotency — how the field solves it, and what it means for neograph

**Status:** Research (feeds the m6d3 "Level B" replay-safety decision) · **Date:** 2026-07-02
**Provenance:** the mid-loop HITL feature (neograph-m6d3) ships at **Level A** (document the idempotency footgun); this report exists to understand the general **durable-execution replay** problem *in principle* before anyone attempts **Level B** (true replay-safety). Parent doc: `async-streaming-mcp-hitl-2026-07-02.md` §6.4.

**Method & confidence.** Multi-source deep-research workflow (fan-out web search → fetch → 3-vote adversarial verification → synthesis), 103 agents. The **durable-execution half is well-verified** (unanimous 3-0 votes, primary sources). The **LLM-agent-framework half could NOT be web-verified in this run** — every LangGraph-specific verification agent failed with API rate-limit (infrastructure, not content) errors. Where this report states LangGraph behavior, it is labeled **[first-hand]** — grounded in direct code inspection of the neograph↔LangGraph integration this session, not web-verified in this run. Treat other agent frameworks (OpenAI Agents SDK, CrewAI, etc.) as **OPEN** pending a re-run.

---

## The answer, in one paragraph

The durable-execution field solves pause/resume/idempotency with **one dominant pattern**: separate **deterministic control flow** (the "workflow"/"orchestrator") from **non-deterministic, side-effecting work** (the "activity"/"step"); record every effect's result into an **append-only event history**; and on resume **replay the deterministic code while fast-forwarding over already-completed effects** by injecting their journaled results instead of re-executing them. Temporal, Azure Durable Functions/Durable Task, and Inngest all implement this same event-sourcing + memoization model, and it is process-independent (Temporal can resume on entirely different infrastructure days later). **Crucially, Temporal's own docs name "LLM/AI invocations" as exactly the class of non-deterministic operation that MUST be a recorded activity/step — journaled, never replayed.** And **none** of these systems claim exactly-once for effects: they provide at-least-once and explicitly **punt the crash-window edge case to user-authored idempotency** (upserts, existence checks). So the field's *floor* is "document idempotency" (our Level A); the field's *ceiling* is "make every side-effecting step a separately-memoized durable unit" (our Level B). The distance between them is precisely a granularity question: **durable engines memoize at the step/activity boundary; LangGraph memoizes at the node/superstep boundary; a ReAct loop is many steps inside one node — too coarse.**

---

## 1. The dominant pattern (verified, high confidence)

**1a. Event-sourcing + deterministic replay.** The runtime records each significant step into an append-only durable log (the "Event History"), then on failure reconstructs exact in-memory state by **re-executing the deterministic control-flow code against that history** and fast-forwarding over completed effects — producing the observable effect of "resume exactly where it left off, on different infrastructure, days later." (Nuance graders insisted on: it is *not* a deserialized memory image; it is re-execution against journaled history.)
> Sources: [Temporal — replaces state machines](https://temporal.io/blog/temporal-replaces-state-machines-for-distributed-applications), [Temporal Event History](https://docs.temporal.io/encyclopedia/event-history), [Temporal — idempotency & durable execution](https://temporal.io/blog/idempotency-and-durable-execution)

**1b. The fundamental principle — split deterministic control flow from recorded effects.** Workflow/orchestrator code *must* be deterministic **because it is replayed**, so all non-deterministic operations move into activities/steps that run **outside** the replay path. Verbatim from Temporal's docs: *"Workflow code must be deterministic to support replay. To handle non-deterministic operations like API calls, **LLM/AI invocations**, database queries, and other external interactions, put them in Activities."* This is the single most decision-relevant fact for neograph.
> Sources: [Temporal — workflow definition](https://docs.temporal.io/workflow-definition), [Temporal — idempotency & durable execution](https://temporal.io/blog/idempotency-and-durable-execution)

**1c. Effect memoization is what makes replay safe.** Completed activities/steps are **not re-executed** on replay; their inputs/outputs are retrieved from the event history and injected back into the running code, so a costly/mutating tool call or LLM call **fires once** and is fast-forwarded thereafter.
> Sources: Temporal workflow-definition; idempotency blog (above).

**1d. Replay integrity via command-vs-event matching — and its tradeoff.** On replay, each command the workflow code emits is matched against the recorded history; a match progresses, a **mismatch raises a non-determinism error**. The tradeoff is the constraint: user control-flow code must be deterministic or replay fails (no `now()`, no unseeded random, no unordered map iteration, no direct I/O in the orchestrator).
> Source: Temporal workflow-definition.

**1e. The idempotency floor — even the strongest engines rest on it.** Effects are guaranteed **at-least-once, not exactly-once**. Every system in this family punts the crash-window edge case (effect completed, result not yet journaled) to **user-authored idempotency** — idempotency keys, upserts, existence checks. *This is the "document idempotency" floor that even Temporal-class engines rest on.*
> Sources: Temporal idempotency blog; [Azure Durable Task — programming model](https://learn.microsoft.com/en-us/azure/durable-task/common/programming-model-overview)

---

## 2. Per-system (verified where cited)

| System | Model | Determinism constraint on user code? | Verified |
|---|---|---|---|
| **Temporal** | Event-sourcing + replay; activities memoized; command-vs-event matching | **Yes** — workflow code must be deterministic; effects go in Activities | ✅ high |
| **Azure Durable Functions / Durable Task** | *Identical* model — orchestrator replayed via event sourcing; non-deterministic/side-effecting work in activities (exempt from determinism) | **Yes** — orchestrator; activities exempt | ✅ high |
| **Inngest (step functions)** | Same memoized-step replay at the SDK level: re-invokes the handler from the top, but **completed `step.run` steps are memoized** — code not re-run, persisted result injected. Determinism boundary is the **step edge**. | Boundary at `step.run`; code between steps re-runs | ✅ high |
| **DBOS, Restate, AWS Step Functions, Cadence** | Same family (Postgres-journaled durable steps / journaling / state-machine-with-external-state). Not independently verified this run. | Varies | ⚠️ OPEN |

The convergence is the point: **three independently-verified engines implement the same event-sourcing + step-memoization model.** It is the field's settled answer.

---

## 3. The LLM-agent-framework half — the gap, filled where I can

**Web verification failed here** (rate-limited). What follows is **[first-hand]** from this session's code work, not web-verified in this run:

- **LangGraph memoizes at the NODE (superstep) boundary, not the step-inside-a-node boundary. [first-hand]** State is checkpointed between nodes; a *completed* node is not re-run on resume, but **everything inside the interrupted node re-runs from the top**. `interrupt()` resolved values are cached (a re-run `interrupt()` returns the resume value instead of pausing again), but the surrounding work is not memoized. So a neograph `agent` node — which contains an entire ReAct loop (many LLM + tool calls) inside one node — **re-invokes the LLM and re-calls prior tools on resume.** This is exactly the m6d3 Level-A/B problem, and it maps precisely onto §1: LangGraph gives you activity-memoization *granularity of a whole node*, while the durable-execution engines give it *granularity of each step*. A ReAct loop is many steps in one node → too coarse.
- **LangGraph does have node-level result caching** (`CachePolicy` on `add_node` in recent versions) — but that caches a node's *output keyed by input* to skip recomputation; it is **not** mid-loop step journaling and does **not** help the interrupt-resume-in-loop case (the interrupted node is re-run precisely because it's where the pending `interrupt()` lives). **[first-hand, verify version specifics before relying on it.]**
- **Other frameworks (OpenAI Agents SDK, CrewAI, LlamaIndex, AutoGen, Pydantic AI, Google ADK, AWS Strands): OPEN.** The strong prior — consistent with the whole durable-execution literature — is that general-purpose agent frameworks **punt to idempotency** (Level A) and do not journal mid-loop LLM/tool calls, because doing so requires the step-granular durable-execution machinery above. But this must be re-verified; the run couldn't confirm it. Notably, **Temporal's own AI examples** take the opposite, principled route: they model each LLM/tool call as a Temporal **activity**, i.e. Level B by construction.

---

## 4. The crux: a non-deterministic LLM call under deterministic replay

This is the heart of the matter and it is **settled** by §1b: an LLM call is non-deterministic, therefore under any deterministic-replay model it **must be a recorded activity/step whose result is journaled and never replayed.** You cannot put an LLM call in replayed control flow — its output would differ on replay and either (a) corrupt state or (b) trip the command-vs-event non-determinism check.

The implication for a node that internally **loops** over LLM + tool calls is direct and unavoidable:

> **True replay-safety requires every LLM turn AND every costly/mutating tool call inside the ReAct loop to be a separately-journaled/memoized durable step, keyed by checkpoint** — exactly as Temporal/Azure/Inngest treat activities/steps. Running the whole LLM+tool loop inside a single replayed node (LangGraph's model, hence neograph's) is *fundamentally* the wrong granularity for replay-safety. (Finding 7, medium confidence — it's a synthesized recommendation, but it follows necessarily from the verified §1b + §1c.)

---

## 5. What this means for neograph — Level A now, and what Level B actually is

**Level A (shipping now) is not a cop-out — it is the field's floor.** Even Temporal-class engines rest on user idempotency for the crash window (§1e). "Document the footgun; make effects idempotent" is the *same* contract every durable engine ultimately exposes to the user for the un-memoizable edge. For neograph's m6d3, Level A means: the `ask_human` docs must state plainly that any LLM/tool work *before* the interrupt re-executes on resume, and that costly/mutating tools must be idempotent (idempotency key / existence check / upsert).

**Level B is a real durable-execution subsystem, not a patch.** Because LangGraph memoizes at node granularity, there are only three honest ways to get step-granular replay-safety, in increasing order of cost:

1. **Step-memoization cache inside the ReAct loop, keyed by checkpoint.** The tool loop journals each completed (LLM turn, tool call) result into checkpoint-scoped state; on resume it fast-forwards by replaying journaled results instead of re-invoking. This is "build a mini durable-execution engine inside `_tool_loop.py`." Feasible, but it's real machinery (keying, ordering, cache invalidation, interaction with budgets and the interrupt) and it must not corrupt the existing sync/async paths.
2. **Decompose the loop into graph nodes** so LangGraph's node-boundary memoization applies per step (each LLM turn / tool call becomes its own node/superstep). This makes LangGraph's checkpoint granularity match the effect granularity — but it inverts neograph's "an agent is one node with a loop inside" model and is a large IR/compiler change.
3. **Run agent steps on a real durable-execution engine** (Temporal-style activities under the hood). Maximal correctness, maximal dependency/complexity; out of scope for a LangGraph-based executor unless durability becomes a first-class product pillar.

**Recommendation:** keep **Level A** for m6d3. Treat **Level B as option (1)** — a checkpoint-keyed step-memoization layer in the tool loop — but **do not build it until** (a) a real consumer needs replay-safety for costly/mutating mid-loop tools *and* idempotency is insufficient, and (b) the async foundation (Phase 1) has landed, since the journaling layer should be designed against the async loop, not retrofitted twice. When that time comes, this report's §1–§4 is the design spec: the durable-execution field has already proven the shape.

---

## 6. Gaps & follow-up

- **Re-run the agent-framework verification** (this run was rate-limited): confirm LangGraph's node-vs-step memoization behavior and `CachePolicy` semantics against current docs; confirm whether OpenAI Agents SDK / CrewAI / LlamaIndex / Pydantic AI / ADK / Strands journal mid-loop or punt to idempotency. Strong prior: they punt.
- **Verify DBOS / Restate / AWS Step Functions / Cadence** specifics (asserted same-family, not independently checked this run).
- **Confirm the [first-hand] LangGraph claims** with a small experiment: a two-tool agent node that mutates a counter before an `interrupt()`, resumed, asserting the counter double-increments under Level A (this is also the m6d3 idempotency E2E — it doubles as the empirical proof of the footgun).

---

## 7. Primary sources
- Temporal: [replaces state machines](https://temporal.io/blog/temporal-replaces-state-machines-for-distributed-applications) · [Event History](https://docs.temporal.io/encyclopedia/event-history) · [workflow definition / determinism](https://docs.temporal.io/workflow-definition) · [idempotency & durable execution](https://temporal.io/blog/idempotency-and-durable-execution)
- Azure Durable Task: [programming model](https://learn.microsoft.com/en-us/azure/durable-task/common/programming-model-overview) · [code constraints](https://learn.microsoft.com/en-us/azure/durable-task/durable-functions-code-constraints)
- Inngest: [how functions are executed](https://www.inngest.com/docs/learn/how-functions-are-executed)
- LangGraph behavior: **[first-hand code inspection, this session]** — not web-verified in this run (verification rate-limited).
