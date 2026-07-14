> **RENAMED 2026-07-14 (neograph-1t0zh):** the construct designed here as *Keymaker* shipped as **`Portal`** — a functional name that reads for newcomers and spans both modes (peer routing + dynamic flow). This document is preserved verbatim as the historical design record; its body still uses the original *Keymaker* name. See `website/src/content/docs/concepts/portal.mdx` for the shipped API.

# KEYMAKER — dynamic handoff + runtime flow-definition (research & DX)

Design-research for beads **neograph-07inf** ("KEYMAKER"). This is research + DX
definition feeding the spike, NOT the spike's final ratification. It consolidates
four independent research reports (codebase grounding, LangGraph substrate, Agent
Spec / Swarm, prior-art + DX) after an adversarial review pass, and defines the
centerpiece the maintainer asked for: polished, native-idiom example designs.

KEYMAKER is one construct with two modes:
- **(a) Peer routing** — a node picks its successor at runtime from a *declared*
  peer set. Surface: `node | Keymaker(peers=[...])`. Lowers to LangGraph
  `Command(goto=...)`.
- **(b) Dynamic flow definition** — a node emits *at runtime* the spec of the next
  flow (a neograph Spec or an imported Agent Spec Flow); neograph validates →
  compiles → dispatches it. Surface: `node | Keymaker(route='decide')`. Pitch:
  **"if it dispatches, it was validated."**

Name pinned by the maintainer (2026-07-13): **KEYMAKER**, Matrix-native beside
Oracle (merge-judge) and Operator (HITL) — "makes a key to any door, including
destinations not pre-wired."

Citations are `file:line` against `src/neograph/` and the installed
`langgraph==1.2.4` under `.venv/lib/python3.12/site-packages/langgraph/`, plus the
local `docs/design/agent-spec-*.md` and the four scratchpad research reports.

---

## 1. Executive summary

**Is it buildable? Yes, with no runtime barrier.** neograph targets LangGraph, and
LangGraph already has the substrate: `Command(goto=...)` for runtime routing and
`Send` for runtime fan-out (`langgraph/types.py:758` — verified: `goto: Send |
Sequence[Send | N] | N = ()`). The gap is purely a **missing IR primitive** in
neograph, exactly as the bead states. Today nothing in `src/neograph/` returns a
`Command(goto)` — `Command` is imported only in `runner.py:22` and used solely for
`Command(resume=...)` on Operator resume (`runner.py:763,1022`); every node wrapper
returns a plain state-update `dict` (`factory.py:76,79`). Verified by grep: `goto`
appears nowhere in `src/`.

**What is genuinely novel.** Mode (a) peer routing is **table stakes** — every major
framework ships it (OpenAI Agents SDK, langgraph-swarm, AutoGen, CrewAI, Google ADK,
Semantic Kernel). neograph's differentiators there are *typing across the hop* and
*three-surface parity*, not the capability. Mode (b) is **partially novel**: "an
agent emits a graph the framework runs" is crowded (meta-agents, TaskWeaver, Airflow
dynamic DAGs), but "a machine-authored flow passes the **identical** production
compile+validate gate as a hand-written one — typed edges, fan-in/out, modifier
legality, lint — and imported *standard* Agent Spec Flows pass the same gate" is a
positioning **no shipped framework states as a first-class contract.** Claim it as
*validation-as-safety-rail-for-machine-authored-graphs*, NOT as runtime-graph-
generation. (Novelty verdict is defensibly hedged in the prior-art report and I did
not find it overclaimed — see §8 review log.)

**Recommended shape.** Build KEYMAKER as a new frozen `Keymaker(Modifier)` with a new
`ModifierSet` slot, kept to minimal legal combinations (standalone or +Operator).
Lower **mode (a) to `Command(goto=<peer>, update=...)` with `destinations=<declared
peers>`** declared on the node — this is the adjudicated recommendation over a bare
router+`path_map` (§7). Mode (b) reuses the existing `load_spec`/`Construct(...)`/
`compile()` validation seam at dispatch time and **composes neograph-mrb2y's Tier-2
hot-swap** (recompile + durable resume via `_auto_resume_from_divergence`) rather than
rebuilding durability. Copy the `Loop.max_iterations` counter mechanism verbatim for a
`max_hops` budget. Full per-layer lowering in §6.

---

## 2. What exists today

### 2a. neograph is static-topology (inventory)

Every "dynamic-ish" construct today lowers to `add_conditional_edges` with a
**statically enumerated `path_map`** — the router picks among destinations all known
and added to the graph at compile time (codebase report §1, verified):

| Construct | IR | Lowering | Router returns |
|---|---|---|---|
| Branch | `_BranchNode` (`_ir_branch.py:55`) | `_add_branch_to_graph` (`_wiring.py:734`); `add_conditional_edges(prev, branch_router, path_map)` (`_wiring.py:822`) | a node **name** in a static path_map |
| Loop | `Loop` modifier (`modifiers.py:535`) | `_add_loop_back_edge` (`_wiring.py:625`); `add_conditional_edges(node, loop_router, path_map=[reenter, exit])` (`:676`) | reenter target or exit name |
| Each | `Each` (`modifiers.py:478`) | `_wire_each` (`_wiring.py:250`); `each_router` returns `[Send(fan, {**state, item})...]` (`:282`) | a list of `Send`s |
| Oracle | `Oracle` (`modifiers.py`) | `_wire_oracle` (`_wiring.py:199`); N `Send`s + `defer=True` barrier | a list of `Send`s |
| Operator | `Operator` | `_add_operator_check` (`_wiring.py:939`); inserts `interrupt(payload)` check node | (interrupt, not routing) |

**Loop is the closest structural analogue to KEYMAKER mode (a):** a router that reads
runtime state, picks a next node from a static destination set with a self-edge
(cycle), guarded by a budget counter. **Each/Oracle are the existing `Send` (dynamic
fan-out) precedent** for a KEYMAKER Send-based runtime-N variant.

Validation is **eager at assembly**: `Construct.__init__` calls `_validate_node_chain`
(`construct.py:194`, verified), separate from `compile()`. This is the crux enabler
for mode (b): "validate-at-dispatch" = call `Construct(...)` (raises `ConstructError`
on a bad emitted spec) **then** `compile()` — both pure-Python, no LLM/network cost
(codebase report §4).

The spec format is **linear-only**: `PipelineRef.nodes` is an ordered ref list
(`_spec_schema.py:119`). `NodeSpec` carries `oracle`/`each`/`loop`/**`operator`**
modifiers (`_spec_schema.py:91-94`) — no handoff/peers/goto/edges. Routing is not
expressible; a `KeymakerSpec` + routing field is net-new schema surface.

> **Correction to the codebase report (§8 point 6 / line 194):** it states "Operator
> isn't even in the spec today." **This is wrong** — `NodeSpec` and `ConstructSpec`
> both carry `operator: OperatorSpec | None` (`_spec_schema.py:94,116`, verified).
> This is the same "Operator mischaracterization" the agent-spec ratification already
> flagged (`agent-spec-ratification-2026-07-13.md:132`). The load-bearing point
> stands: **no routing/handoff/edge concept exists in the spec** — Operator is a
> node-local modifier, not a routing construct.

### 2b. LangGraph capabilities (the substrate)

Verified against the installed `langgraph==1.2.4`:

- **`Command`** (`types.py:758`): `graph`, `update`, `resume`, `goto` fields; `goto:
  Send | Sequence[Send | N] | N = ()`; `PARENT: ClassVar = "__parent__"`. `update` and
  `goto` apply in the **same superstep**. Exact match to the LangGraph report.
- **Destinations are optional** and only affect *compile-time* validation. Declare via
  a `Command[Literal["a","b"]]` return annotation or `add_node(..., destinations=...)`
  (`state.py:385,454,...` — `destinations` kwarg present on every add_node overload,
  verified). At `.compile()`, declared ends are validated: a declared bad target →
  `ValueError: Found edge ending at unknown node`.
- **Undeclared runtime goto to a nonexistent node does NOT raise.** It is silently
  swallowed with a WARNING and the write is dropped (`_algo.py:312` — "wrote to unknown
  channel ... ignoring it"; `Send` variant `_algo.py:978` — "Ignoring unknown node
  name"; both verified). The superstep produces no next task and the graph halts at
  END. **This is the #1 constraint: LangGraph will not catch a bad goto — a typo
  becomes a silent early stop, not an error. KEYMAKER must validate goto targets
  itself.**
- **`Command.PARENT`** escapes a subgraph to re-route at the parent — the mechanism
  swarm/supervisor use to route from inside an agent subgraph.
- **Termination**: no native max-hops. Only `recursion_limit` (default 25, counts
  supersteps, `main.py:2976`) → `GraphRecursionError`. Every goto hop = one superstep.

### 2c. The gap

neograph can express *fan-out over a declared collection* and *branch/loop over
declared arms*, but not *"a node returns which peer, from a runtime-decided set, runs
next"* nor *"a node emits the next flow's spec."* Both are LangGraph-expressible;
neither has an IR primitive. `from_agent_spec` (mode b's Agent-Spec input path,
neograph-01i0g) does not exist yet — grep of `src/` is empty; `load_spec`
(`loader.py:44`) is the closest existing runtime-construction seam and the pattern
mode (b) should reuse.

---

## 3. Prior art & the honest novelty verdict

### 3a. Peer handoff — everyone does it (mode a is table stakes)

| Framework | Signal contract | Set | Termination |
|---|---|---|---|
| OpenAI Agents SDK | handoff-as-tool (`transfer_to_X`), full-history transfer | declared `handoffs=[...]` | `max_turns` (default 10) → `MaxTurnsExceededError` |
| langgraph-swarm | `create_handoff_tool` → `Command(goto=agent, graph=PARENT, update={active_agent})` | declared; `active_agent: str` narrowed to `Literal[names]` | `recursion_limit` → `GraphRecursionError` |
| AutoGen / MS Agent Fw | Swarm (LLM-picks) / GraphFlow (declared edges) | mixed | max-rounds |
| CrewAI hierarchical | manager delegates via tool; `allowed_agents` narrows | declared | iteration caps |
| Google ADK | `transfer_to_agent(name)` fn-call | declared sub-agent tree | loop caps |
| Agent Spec / WayFlow Swarm | decentralized: `first_agent` + `relationships` | declared relationship graph | none (WayFlow `while True`) |

Verdict: **well-trodden.** neograph's differentiators are (1) **static typing across
the hop** — raw `Command(goto)` is untyped; neograph can type-check that every
reachable peer's `inputs` are satisfiable and narrow the peer set to a `Literal` union
(langgraph-swarm's proven `_update_state_schema_agent_names` trick); (2) **three-surface
parity** (declarative / `@node` / programmatic) — no competitor offers the construct
across a decorator, an imperative builder, and a runtime-composed pipe; (3) it lowers
to the *same* `Command(goto)` primitive, so no runtime barrier.

### 3b. Validated runtime-emitted flows — rare (mode b is the interesting one)

The distinction that matters (prior-art report §2):
- **Text plans** (re-interpreted each step) — ubiquitous (ReAct, plan-and-execute,
  BabyAGI). Not mode (b).
- **Typed/structured plans** (planner emits a Pydantic/JSON object, parsed
  deterministically) — common and rising (TaskWeaver, DSPy, "use a JSON schema for
  planner output"). Validates the plan's *shape*.
- **A validated, typed GRAPH spec compiled and dispatched as an executable workflow** —
  **rare, mostly research/bespoke, not a shipped framework primitive.** Closest hits:
  hand-rolled LangGraph "meta-agent"/Graph-Builder patterns (validation = "the JSON
  parsed against my schema," not "the graph compiler ran on the emitted graph");
  research systems (GraphFlow, Prompt2DAG, G-Designer) whose validation is *structural*
  (acyclicity, config sanity), not a typed-edge compiler gate; Airflow/Flyte dynamic
  DAGs (scheduler conformance, not agent-graph type safety).

**Novelty verdict: PARTIALLY NOVEL — novel in framing and integration, not in raw
idea.** Genuinely differentiated for neograph:
1. **The validator is the SAME gate for hand-written and machine-authored flows** —
   neograph runs its *production* compile+validate pipeline (typed edges, fan-in/out
   compatibility, modifier legality, lint) on the emitted spec, not a bespoke planner
   schema. "If it dispatches, it was validated."
2. **Typed I/O across the dispatch boundary** — the emitted flow's `input`/`output` are
   typed Pydantic ports validated against the emitting node's state.
3. **`from_agent_spec` as the runtime flow-definition seam** — dispatching an imported
   *standard* Agent Spec Flow under the same gate; found in no shipped framework.

This verdict is **defensible, not overclaimed** — the report explicitly scopes the
claim away from "first to let an agent build a graph" (which would invite "meta-agents
already do that") and toward "first to make machine-authored graphs pass the identical
production validation gate as hand-written ones." I concur.

---

## 4. The DX — three example designs

These are the centerpiece. They follow the existing suite's idiom (examples/01c, 15,
16, 27; `website/.../runtime/llm-driven.mdx`): frozen Pydantic models,
`@node(outputs=...)` with parameter-name edges, `construct_from_module`/`compile`/`run`,
dict `input=` with a `node_id`, scripted nodes so they run keyless, narrated
docstrings, no emojis. **All `Keymaker(...)` kwargs are proposals the spike must
ratify** — they are shown to make the DX concrete and to surface the open API
questions, not to pre-decide them. I refined the Opus candidates to fix internal
inconsistencies (noted per-example). New examples would be `examples/28_*`, `29_*`,
`30_*` (suite currently ends at 27).

### E1 — Peer routing over declared specialists (the typed swarm)

Demonstrates: a router picks a specialist peer at runtime, specialists can **route
back** (a real cycle), a `max_hops` budget guarantees termination, and a **typed**
`HandoffDecision` drives `Command(goto)`. Every reachable peer is type-checked at
compile time.

Refinements over the Opus candidate: (1) the original claimed cycles in the docstring
but never wired a back-edge — here specialists genuinely hand back to `triage` and
carry the routing field, so the cycle + `max_hops` are exercised; (2) the terminal
signal is unified — `HandoffDecision.goto` is a `Literal` whose members are exactly the
declared `peers` **plus** the reserved `END` sentinel, removing the peers-vs-goto
mismatch; (3) all nodes emit `HandoffDecision`, so the routing contract is uniform
across router and peers (a mesh, not a star).

```python
"""Example 28: Keymaker peer routing -- a typed swarm over declared specialists.

A triage node inspects a support ticket and hands off to one of three declared
specialist peers. A specialist either RESOLVES (routes to END with a filled-in
resolution) or hands BACK to triage for re-routing -- a genuine runtime cycle.
Routing is decided at RUNTIME (Command(goto)), not by a static conditional edge,
yet every reachable peer is type-checked at compile time and a max_hops budget
guarantees termination.

All nodes scripted -> keyless.
"""
from __future__ import annotations

import sys
from typing import Literal

from pydantic import BaseModel

from neograph import Keymaker, compile, construct_from_module, node, run

# The reserved terminal sentinel (mirrors LangGraph END; a peer routes here to stop).
END = "__end__"


class Ticket(BaseModel, frozen=True):
    subject: str
    body: str


class HandoffDecision(BaseModel, frozen=True):
    # `goto` names a declared peer OR the END sentinel. The spike may narrow this
    # Literal to the declared peer set automatically (langgraph-swarm's trick),
    # turning a typo into a compile error instead of a silent runtime goto-miss.
    goto: Literal["triage", "billing", "technical", "account", "__end__"]
    ticket: Ticket
    resolution: str | None = None
    hops: int = 0


@node(outputs=HandoffDecision)
def triage(ticket: Ticket) -> HandoffDecision:
    text = (ticket.subject + " " + ticket.body).lower()
    if "refund" in text or "charge" in text:
        return HandoffDecision(goto="billing", ticket=ticket)
    if "error" in text or "crash" in text:
        return HandoffDecision(goto="technical", ticket=ticket)
    return HandoffDecision(goto="account", ticket=ticket)


@node(outputs=HandoffDecision)
def billing(triage: HandoffDecision) -> HandoffDecision:
    # Resolve, or bounce back to triage if it is not actually billing.
    if "charge" in triage.ticket.body.lower():
        return HandoffDecision(goto=END, ticket=triage.ticket,
                               resolution="Refund issued.")
    return HandoffDecision(goto="triage", ticket=triage.ticket)


@node(outputs=HandoffDecision)
def technical(triage: HandoffDecision) -> HandoffDecision:
    return HandoffDecision(goto=END, ticket=triage.ticket, resolution="Patch shipped.")


@node(outputs=HandoffDecision)
def account(triage: HandoffDecision) -> HandoffDecision:
    return HandoffDecision(goto=END, ticket=triage.ticket, resolution="Account updated.")


# Keymaker declares each caller's peer set + the routing field + the cycle budget.
# `route="goto"` = read the `.goto` field of the node's HandoffDecision output and
# map it to Command(goto). `max_hops` copies Loop.max_iterations -> KeymakerBudgetError
# on breach. peers= is per-node and directed (Swarm `relationships` semantics).
triage = triage | Keymaker(peers=["billing", "technical", "account"], route="goto",
                           max_hops=6)
billing = billing | Keymaker(peers=["triage"], route="goto")
technical = technical | Keymaker(peers=["triage"], route="goto")
account = account | Keymaker(peers=["triage"], route="goto")

pipeline = construct_from_module(sys.modules[__name__], name="support-swarm")

if __name__ == "__main__":
    graph = compile(pipeline)
    out = run(graph, input={"node_id": "t-1",
                            "ticket": Ticket(subject="Double charge",
                                             body="I was charged twice")})
    # The last active peer's resolution surfaces on the bus.
    print(out)
```

Why it is cool: it looks like a static pipeline, but the edge out of `triage` is chosen
at runtime — and neograph *still* type-checks that each peer consumes a
`HandoffDecision`. `describe_graph(graph)` can print the reachable-peer edges to show
the validator enumerated every hop. The `max_hops` budget is the "we thought about
termination" flourish.

### E2 — Runtime-emitted new wiring, validated-then-dispatched (with the rejection path)

Demonstrates: a planner emits at runtime the **spec** of a pipeline that did not exist
at compile time; `Keymaker(route='decide')` validates+compiles+dispatches it; and the
**rejection path** — an invalid emitted spec is caught by the *same validator* before
any node runs. This is "if it dispatches, it was validated" made concrete.

Refinements over the Opus candidate: the original mixed `Node("plan", mode="raw", ...)`
with a bare `def plan(request, config)` never wired together. Here the emitter is a
proper `@node`-decorated scripted function returning a typed `EmittedFlow`, and the
Keymaker fields name that type's fields explicitly. The rejection path is shown as a
real second run, not a comment.

```python
"""Example 29: Keymaker dynamic flow definition -- the planner emits the next flow.

A planner node inspects the request and EMITS a neograph Spec for the pipeline
that should handle it. Keymaker(route='decide') runs neograph's real compile+
validate gate on that emitted spec, then dispatches it. If the planner emits a
type-incompatible spec, the SAME validator that guards hand-written pipelines
rejects it BEFORE execution -- the flow never runs malformed.

Scripted planner (deterministic) so the example is keyless; in production the
planner is a `think`/`agent` node emitting the spec via structured output.
"""
from __future__ import annotations

import sys

from pydantic import BaseModel

from neograph import (Keymaker, compile, construct_from_module, node, run,
                      register_type)


class Request(BaseModel, frozen=True):
    kind: str  # "summarize" | "classify"
    payload: str


class Summary(BaseModel, frozen=True):
    text: str


class Label(BaseModel, frozen=True):
    value: str


class EmittedFlow(BaseModel, frozen=True):
    # The planner's typed output: a neograph Spec dict + the input to feed it.
    spec: dict
    dispatch_input: dict


# The dispatched sub-flows reference these types by string name -- register them
# in the same type registry load_spec uses (spec_types.register_type).
register_type("Summary", Summary)
register_type("Label", Label)


@node(outputs=EmittedFlow)
def plan(request: Request) -> EmittedFlow:
    """Emit the spec of the flow that should run next (chosen at runtime)."""
    if request.kind == "summarize":
        spec = {"name": "summarize-flow",
                "nodes": [{"name": "shorten", "scripted_fn": "shorten",
                           "outputs": "Summary"}],
                "pipeline": {"nodes": ["shorten"]}}
    else:
        spec = {"name": "classify-flow",
                "nodes": [{"name": "label", "scripted_fn": "label",
                           "outputs": "Label"}],
                "pipeline": {"nodes": ["label"]}}
    return EmittedFlow(spec=spec, dispatch_input={"text": request.payload})


# route="decide": the node's output carries a spec that Keymaker validates+compiles
# +dispatches at runtime. spec_field/input_field name the EmittedFlow fields.
# on_invalid="raise" makes the dispatch gate the failure boundary; max_depth bounds
# self-extending flows and PROPAGATES into dispatched sub-flows (anti deepagents#1698).
plan = plan | Keymaker(route="decide", spec_field="spec", input_field="dispatch_input",
                       on_invalid="raise", max_depth=3)

pipeline = construct_from_module(sys.modules[__name__], name="dynamic-dispatch")

if __name__ == "__main__":
    graph = compile(pipeline)  # the planner itself compiles statically
    out = run(graph, input={"node_id": "r-1",
                            "request": Request(kind="summarize", payload="...long...")})
    print(out)  # results of the dynamically-dispatched summarize-flow

    # --- Rejection path: an emitted spec with a type-incompatible edge is rejected
    # at DISPATCH by the same ConstructError a hand-written bad pipeline raises,
    # BEFORE any node of the sub-flow runs. In production the planner catches the
    # error and revises (the self-correcting loop from llm-driven.mdx).
```

Why it is cool: the killer demo is the **rejection** — the planner emits a spec whose
edge types do not match, and neograph rejects it *with the same error message* a human
gets from `Construct(...)`, proving machine-authored graphs get human-grade static
safety. This is the `llm-driven.mdx` story, but the *dispatch* is now a first-class
construct rather than app glue calling `compile()` by hand.

### E3 — The fusion: emit an Agent Spec Flow at runtime + self-extending hop budget

Demonstrates: mode (b) coupled to `from_agent_spec` (neograph-01i0g) — a node emits a
**standard** Agent Spec Flow (not a private neograph spec), imported at runtime and
dispatched under the same gate; plus a self-extending flow bounded by a hop budget that
**propagates** into dispatched sub-flows (explicitly dodges the deepagents #1698
budget-reset footgun). Requires the Agent Spec import seam (neograph-01i0g).

```python
"""Example 30: Keymaker + Agent Spec -- dispatch an imported standard flow at runtime.

The orchestrator emits an Agent Spec Flow (the portable, standard representation).
Keymaker imports it via from_agent_spec, runs neograph's validation gate, and
dispatches -- so an externally- or LLM-authored *standard* spec passes the identical
safety rail as a hand-written neograph pipeline. A hop budget bounds a flow that may
emit further flows (self-extension), and the budget propagates into the sub-flow.

Requires the Agent Spec import seam (neograph-01i0g / from_agent_spec).
"""
from __future__ import annotations

import sys

from pydantic import BaseModel

from neograph import Keymaker, compile, construct_from_module, node, run
from neograph import from_agent_spec  # neograph-01i0g


class Goal(BaseModel, frozen=True):
    text: str
    hops_left: int = 3


class EmittedFlow(BaseModel, frozen=True):
    agent_spec: dict  # a standard Agent Spec Flow document
    dispatch_input: dict
    hops_left: int


@node(outputs=EmittedFlow)
def orchestrate(goal: Goal) -> EmittedFlow:
    """Emit a STANDARD Agent Spec Flow for the next leg; decrement the budget."""
    agent_spec = {"component_type": "Flow", "name": "research-leg",
                  "start_node": "gather",
                  "nodes": {"gather": {"type": "LLMNode",
                                       "prompt": "Research: " + goal.text}}}
    return EmittedFlow(agent_spec=agent_spec,
                       dispatch_input={"query": goal.text},
                       hops_left=goal.hops_left - 1)


# loader=from_agent_spec: the emitted document is imported through the Agent Spec seam,
# THEN validated+compiled+dispatched. budget_field ties the self-extension hop count to
# the emitted flow; when hops_left hits 0 Keymaker stops re-dispatching.
orchestrate = orchestrate | Keymaker(route="decide", loader=from_agent_spec,
                                     spec_field="agent_spec",
                                     input_field="dispatch_input",
                                     budget_field="hops_left",
                                     on_invalid="route_to_error")

pipeline = construct_from_module(sys.modules[__name__], name="self-extending-research")

if __name__ == "__main__":
    graph = compile(pipeline)
    out = run(graph, input={"node_id": "g-1",
                            "goal": Goal(text="state of tokamak fusion 2026")})
    print(out)
```

Why it is cool: the thesis at full volume — a **portable, standard** Agent Spec Flow
(LLM- or third-party-authored) is dispatched through neograph's compiler and gets the
same typed-edge validation a hand-written pipeline gets. It makes `from_agent_spec`
load-bearing *at runtime*, not just a file loader.

### Cross-cutting API tensions the spike must resolve (surfaced by all three)

1. **One routing contract or two?** Mode (a) routes to a *name* (`route="goto"` reads a
   typed field); mode (b) routes to a *spec* (`route="decide"`). `peers=` and
   `route="decide"` are near-disjoint kwarg sets, so validation and lowering **fork
   early** even under one construct. Decide whether that fork is one `Keymaker` with a
   discriminator (sketched) or two sibling constructs sharing a base.
2. **Typed field vs registered condition** for the routing signal. neograph-native =
   typed output field (`route="goto"` reads `.goto`, types over strings); the `Loop`
   `when` precedent = a registered string condition. The typed-field form needs the
   output model to carry a routing field the framework strips (like `tool_log` is
   demand-driven, per AGENTS.md).
3. **The validation-vs-runtime boundary** (the whole thesis): mode (a) is *statically*
   checkable (enumerate declared peers, type-check every hop) — keep it in the compile
   gate. Mode (b) is *dispatch-time* checkable only — the emitted spec runs the full
   validator at dispatch. The marketable one-liner: "compile-time for known peers,
   dispatch-time for emitted flows, **same validator either way**."
4. **Budget vocabulary unification**: `max_hops` (a) and `max_depth`/`budget_field` (b)
   should be one concept with propagation semantics (anti-#1698). Fail-loud
   (`KeymakerBudgetError`) matching `Loop.max_iterations`/`on_exhaust` and langgraph's
   `recursion_limit`.
5. **Three-surface parity**: `@node` sugar, declarative `Node(...) | Keymaker(...)`, and
   a `self.keymaker(...)`/`self.handoff(...)` ForwardConstruct builder must produce the
   same IR — the exact `fan_out_param`/`Each` parity lesson (neograph-ts7). E1's
   mesh-of-peers is the hardest to express in ForwardConstruct's imperative `forward()`
   — prototype it there first.

---

## 5. Interop analysis

### 5a. Composition with current wiring

- **Modifiers**: KEYMAKER is a **terminal router** (it owns the outgoing edge), so it
  almost certainly conflicts with `Loop` (both own the back-edge) and with `Each`/
  `Oracle` on the same node (both own fan-out). Recommended `_SLOT_RULES` position:
  KEYMAKER standalone or KEYMAKER+Operator only (a handoff peer that needs approval).
  Keeping the legal set minimal avoids a `ModifierCombo`/`_COMBO_MAP` blow-up — every
  combo forces an `assert_never` arm at the two dispatch sites (`compiler.py:516`
  node-level and the subgraph match at `:536`+, verified). **This exhaustiveness is a
  feature**: a missing arm fails loud at compile, no silent gap.
- **Sub-constructs**: a peer can itself be an `agent`/`think` node or a sub-construct.
  `Command.PARENT` is the mechanism for a peer *inside* a sub-construct to re-route at
  the parent level — this is why the adjudicated lowering is `Command(goto)`, not a
  bare `path_map` (a `path_map` is edges at one graph level and cannot escape a
  subgraph — §6/§7).
- **Validation — the union-frontier honesty (weakened static guarantee).** The current
  validator (`_validate_node_chain`, `_construct_validation.py`) assumes a **linear
  prev→next producer frontier**. Branch arms are the one existing exception, and they
  carry a documented, un-caught **cross-arm-leakage** limitation
  (`_construct_validation.py:139-147`) — a false-arm node reading a true-arm output is
  not flagged. KEYMAKER breaks the linear assumption **harder**: any peer may run after
  the router, so the producer frontier a peer sees is not statically single-valued, and
  since peers hand off to each other the reachable producer set is a **fixpoint over the
  peer graph**, not a walk. The honest static-safety position for mode (a): validate
  **(i) every declared peer exists** and **(ii) the router's output type is compatible
  with every peer's declared input**, and DEFER per-producer resolution to the runtime
  `_extract_input` isinstance scan (the same fallback single-type `inputs` already uses,
  `_construct_validation.py:167-180`). Be explicit in the docs: **this is weaker than
  the linear-DAG "if it compiles, it runs" guarantee** — it is "every hop is type-
  reachable," not "exactly one producer feeds each consumer." Mode (b) has no such
  weakening: the emitted spec is validated in full at dispatch.

### 5b. Composition with checkpointing

- **Mode (a) peer hops are checkpointed supersteps.** Each `Command(goto)` hop writes
  one checkpoint recording state + the planned `.next` (LangGraph report §4, consistent
  with neograph's per-superstep durability). A K-hop chain = K checkpoints. This aligns
  with the existing model — **no new checkpoint machinery for mode (a).** The `max_hops`
  counter is a plain state field, copied from Loop's `neo_loop_count_<field>`
  (`state.py`, `_state_write.py:136`), so it persists in the checkpoint like any Loop
  counter.
- **Mode (b) durable-dispatch options** (the mrb2y Tier-1/Tier-2 split):
  - **Tier 1 (no recompile)** — known path-set compiled as one superset graph; router
    picks via `Command(goto)`/`Send`. Durable + streaming intact. This is mode (a) and
    the "known peers" half of mode (b).
  - **Tier 2 (recompile + resume)** — a genuinely new shape can't mutate a running
    LangGraph in place. Path: emit spec → `from_agent_spec`/`load_spec` → validate +
    recompile → resume on the SAME `thread_id`, **leveraging existing checkpoint auto-
    rewind** (`_auto_resume_from_divergence`, `runner.py:210`): the changed schema
    fingerprint re-runs only invalidated nodes and reuses checkpointed state. mrb2y's
    instruction is explicit and correct: **compose the rewind, do not rebuild it.** Mode
    (b) is the *authoring surface* (a Keymaker node saying "here is the next flow");
    mrb2y is the *runtime mechanism* (validate → recompile → durable resume). Build mode
    (b) ON mrb2y, not beside it.
  - **The trap to avoid** (LangGraph report §4, verified reasoning): a *manual*
    `inner = builder.compile(); inner.invoke(...)` inside a node body is **invisible to
    the parent thread's checkpointer/namespace** — on resume the parent re-executes the
    whole node, rebuilding and re-invoking the inner graph from scratch; the inner run is
    NOT independently resumable. And LangGraph has **no reconciliation for a graph that
    changes shape between resumes** — unknown channels from an old checkpoint are silently
    dropped (`_algo.py:312`). This is precisely the gap neograph's schema-fingerprint
    auto-rewind fills at the neograph layer; mode (b) must fingerprint/validate the
    dispatched spec against the checkpoint, **extending** (not merely reusing) that
    machinery for a runtime-variable topology.

---

## 6. Lowering sketch per layer

| Layer | File(s) | Change |
|---|---|---|
| **DX / public** | `__init__.py` (`__all__` near the Each/Oracle/Loop block, line ~65) | add `Keymaker`; later `from_agent_spec` |
| **DX / decorator** | `decorators.py` | `@node(peers=..., route=...)` → build `Keymaker` (sugar only; must NOT touch `node.py`/`compiler.py` per layer discipline) |
| **DX / forward** | `forward.py` (~`self.interrupt` builder) | new `self.keymaker(peers=, route=)` returning a `_KeymakerCall` (copy `self.interrupt`, the simplest single-modifier builder) |
| **Modifier decl** | `modifiers.py:65` (`ModifierCombo`), `:89` (`_COMBO_MAP`), `:604` (`_SLOT_RULES`), `:617` (`ModifierSet`) | `Keymaker(Modifier)`; new `keymaker` slot + `_SlotRule` row (standalone / +Operator); new combo entries |
| **IR (mode b)** | possibly new `_ir_keymaker.py` | a spec-emitting sentinel if needed (parallel `_ir_branch.py`) |
| **Validation** | `_construct_validation.py:139`, `_validation_types.py:78` | peer-existence + router-output↔peer-input type check across the peer closure; `effective_producer_type` likely **unchanged** (the node writes its own declared output; the routing field is side metadata) |
| **Compiler dispatch** | `compiler.py:516` (node), `:536`+ (subgraph) | new combo arm(s); `assert_never` forces handling |
| **Lowering (mode a)** | `_wiring.py` (new `_add_keymaker_*`, model on `_add_loop_back_edge`) | teach the factory (`factory.py:76`) to return `Command(goto=<peer>, update=...)`; declare `add_node(..., destinations=<peers>)`; hop-budget guard |
| **State** | `state.py`, `_state_write.py:136` | `neo_keymaker_hops` counter (copy the Loop counter verbatim) |
| **Runtime (mode b)** | `loader.py:44`, `spec_types.py`, `_spec_schema.py` | dispatch-time build→`Construct()`(validate)→`compile()`; new `KeymakerSpec`+routing schema; type-registry constraint |
| **Checkpoint (mode b)** | `runner.py:210` | compose `_auto_resume_from_divergence` for recompile-resume (mrb2y Tier-2); do NOT rebuild |

### The adjudicated router-vs-`Command(goto)` recommendation

The reports offer two lowerings and do not truly conflict — they name two valid paths.
The codebase report is correct that mode (a) *could* use a plain router + static
`path_map` with **zero factory change** (identical to branch/loop, and it gets peer-
existence validation for free from the static path_map). The LangGraph report
recommends `Command(goto=peer, update=...)` with `destinations=` declared.

**Adjudication: lower mode (a) to `Command(goto=<peer>, update=...)` AND declare
`add_node(..., destinations=<declared peers>)`.** Reasoning:

1. The bead **pins** `Command(goto)` as the lowering, and there is a real semantic
   reason: a bare `add_conditional_edges` `path_map` routes only among peers at the
   **same graph level** and cannot express the **mesh handoff across sub-construct
   boundaries** (`Command.PARENT`) — the exact swarm case where a peer inside a sub-
   construct re-routes at the parent. The router+`path_map` approach structurally cannot
   do this; `Command(goto)` can.
2. It **unifies mode (a) and mode (b)** on one factory change (`Command`-returning
   wrapper), rather than two divergent lowerings.
3. Declaring `destinations=<peers>` recovers the *only* advantage the router+`path_map`
   had — LangGraph's **compile-time target-existence check** (`state.py` ends
   validation) — closing the silent-drop hole (`_algo.py:312`) for the declared-peer
   case for free. KEYMAKER already knows the declared peer set at assembly time, so emit
   it.

Record the router+`path_map` form as a **viable fallback** for the same-level, no-mesh
case if the factory change proves too invasive in the spike — but the primary
recommendation is `Command(goto)` + `destinations=`. Either way, **KEYMAKER must
validate goto targets itself at dispatch** — LangGraph silently drops a bad undeclared
goto (LangGraph report constraint 1, verified).

---

## 7. Open questions for the spike + Swarm-import re-scope

1. **`path_map` vs `Command(goto)`** — adjudicated above (Command(goto) +
   `destinations=`); the spike should confirm the factory change is acceptable and pin
   the fallback trigger.
2. **Cross-peer type validation is a fixpoint, not a linear walk** (§5a). Pin exactly
   what mode (a) statically guarantees (peer existence + router-output↔peer-input type
   compat) and document the weakened guarantee honestly.
3. **Mode (b) type-registry coupling.** An emitted spec resolves types only via
   `register_type`/`lookup_type` (`spec_types.py:41,61`). An emitted flow referencing an
   unregistered type fails at dispatch (fail-loud, good) but is a real authoring
   constraint. Who populates the registry for a runtime-emitted flow?
4. **Checkpoint semantics of the sub-flow.** compile-inside-node (opaque to parent
   thread) vs superset-graph-goto (Tier-1, durable) vs recompile-resume (Tier-2). Adopt
   the mrb2y split; do not invent a third.
5. **Cycle termination.** Copy the Loop counter (`_state_write.py:136` write,
   `_wiring.py` check). Pin `on_exhaust` behavior (raise `KeymakerBudgetError` vs route-
   to-terminal). Ensure the budget **propagates** into dispatched sub-flows (anti
   deepagents #1698).
6. **Spec schema is linear-only.** Adding routing to `_spec_schema.py` is net-new schema
   (a `KeymakerSpec` + a routing field on `PipelineRef`), the first non-linear construct
   expressible in a spec.
7. **Failure mode on an invalid emitted spec** (bead-flagged): `on_invalid` ∈ {`raise`,
   `route_to_error`, `return`-as-data-for-self-correction}. Likely all three,
   configurable; default should not crash a durable run.
8. **Import ambiguity (Swarm).** A pure Agent Spec Swarm carries only `first_agent` +
   `relationships` — **no `handoff` flag** (that is WayFlow-runtime-only). Import cannot
   tell query-mode (`send_message`) from handoff-mode (`handoff_conversation`) → require
   a `neograph/swarm_mode` metadata marker or document a default (recommend handoff-
   mode). Verified against the agent-spec report and consistent with the local ratifi-
   cation doc.

### Swarm-import re-scope note

The swcy1 ratification (`agent-spec-ratification-2026-07-13.md:80-84`, verified) chose
**REJECT-with-error** for Swarm import — a `NeographError` naming Swarm as unsupported,
**preserving `first_agent` + `relationships` in `metadata` for a possible future
combinator.** That future combinator is KEYMAKER. Once KEYMAKER lands, the decision
flips from "reject forever" to **"faithful import (handoff-mode) via `Command(goto)`"**:

- `first_agent` → KEYMAKER entry node (FAITHFUL).
- `relationships` = `(caller, recipient)` pairs → **per-node, directed** `peers=` (each
  caller's `peers` = the recipients it is paired with) — NOT a single global peer set.
  This per-node directedness is the key structural mapping (agent-spec report §3).
- Each `Agent` → a `Node(mode="agent")` peer that emits a routing decision; whole-history
  transfer → the state bus carried across the goto; `active_agent` → a KEYMAKER routing-
  state field.
- **Lossy:** `send_message` (query-and-return) mode is **NOT KEYMAKER** — it is a peer
  exposed as a callable *tool* on an agent node (subagent-as-tool), a separate primitive.
  Scope KEYMAKER to **handoff/control-transfer semantics only**; mixed-mode Swarms are
  only partially faithful. `swarm_template` and chat scaffolding ride in metadata.
  KEYMAKER also becomes the **first faithful Swarm EXPORT source** (round-trip the
  handoff-vs-query intent via a `neograph/swarm_mode` marker; `max_hops` has no Agent
  Spec field → metadata).
- **Mode (b) has NO Agent Spec representation** — Agent Spec flow topology is fixed at
  serialization time; there is no runtime-emitted-flow component. Mode (b) is neograph-
  native, expressible in Agent Spec only as metadata-tagged opacity (a scripted
  placeholder node), never a faithful round-trip. But the *payload* a mode-(b) node
  emits CAN be an Agent Spec Flow — that is the `from_agent_spec` runtime coupling (E3).

RemoteAgent/A2A remains a **separate axis** (client-initiated call node → best-effort
scripted node) and is explicitly NOT KEYMAKER's concern.

---

## 8. Appendix — review log

### What I verified against source (spot-checks, not exhaustive)

**Codebase report (`research-codebase.md`) — reliability HIGH.**
- ✓ No `Command(goto)` in `src/` (grep empty). `Command` imported `runner.py:22`, used
  only for `resume` (`:763`, `:1022`). `node_wrapper` returns `dict` (`factory.py:76`).
- ✓ Validation eager at `Construct.__init__` → `_validate_node_chain` (`construct.py:194`).
- ✓ Branch lowers via `add_conditional_edges(..., path_map)` (`_wiring.py:822`); Oracle/
  Each/flat variants at `:239,285,367`.
- ✓ Loop budget in `_make_loop_router` — `if count >= loop.max_iterations` raise/exit
  (`_wiring.py:524`+).
- ✓ `_COMBO_MAP` + `ModifierCombo` + `_SLOT_RULES` + `ModifierSet` (`modifiers.py:65,89,
  604,617`); `assert_never` at `compiler.py:516`, two dispatch sites (`:467`/`:536`).
- ✓ `effective_producer_type` (`_validation_types.py:78`).
- ✗ **CORRECTED**: report claims "no Operator in the spec" (§8 point 6, line 194).
  `NodeSpec.operator` and `ConstructSpec.operator` both exist (`_spec_schema.py:94,116`).
  The load-bearing point (no routing/handoff/edges in the spec) is unaffected. Same
  mischaracterization the ratification doc already flagged (`:132`).

**LangGraph report (`research-langgraph.md`) — reliability HIGH.**
- ✓ `Command` dataclass fields and `goto: Send | Sequence[Send | N] | N = ()` exact
  match (`types.py:758`). `PARENT` ClassVar present.
- ✓ Silent-drop of unknown channel/node: `_algo.py:312` ("wrote to unknown channel ...
  ignoring") and `:978` ("Ignoring unknown node name") — the #1 constraint, confirmed.
- ✓ `destinations=` kwarg on every `add_node` overload (`state.py:385,454,...`); compile-
  time ends validation exists. ✓ `recursion_limit` default/enforcement (`main.py:2534,
  2976`).
- Swarm/langgraph-swarm specifics (`create_handoff_tool` → `Command(goto, graph=PARENT,
  update={active_agent})`) are deepwiki/package-doc sourced, not re-verified line-by-line
  — MEDIUM confidence, but internally coherent and consistent with the known
  langgraph-swarm design. No corrections.

**Agent Spec report (`research-agentspec.md`) — reliability HIGH on local docs, MEDIUM
on WayFlow internals.**
- ✓ swcy1 Swarm REJECT-with-error, `first_agent`+`relationships` preserved in metadata
  (`agent-spec-ratification-2026-07-13.md:80-84`).
- ✓ Tier 1 / Tier 2 framing and auto-rewind composition
  (`agent-spec-interop-2026-07-09.md:41-51`).
- The `send_message` vs `handoff_conversation` split and the "`handoff` flag not in
  portable Agent Spec" finding are deepwiki-sourced (oracle/wayflow, pyagentspec) — not
  locally verifiable, but this is the report's single most important and internally
  consistent nuance; I accept it and flagged the import-ambiguity consequence (§7.8). No
  corrections.

**Prior-art report (`research-priorart-dx.md`) — reliability MEDIUM-HIGH.**
- The novelty verdict ("PARTIALLY NOVEL — novel in framing/integration, not raw idea")
  is **appropriately hedged, not overclaimed.** It explicitly scopes the claim away from
  "first to let an agent build a graph" and toward "first to run the identical production
  validation gate on machine-authored graphs." Given what it surveyed (meta-agents
  validate against bespoke schemas; research systems validate structurally; orchestrators
  validate scheduler conformance — none reuse a typed-edge compiler gate verbatim), the
  narrowed claim is defensible. External source URLs not individually re-fetched (MEDIUM
  on citation completeness); the *reasoning* is sound. No corrections; I tightened the
  DX examples' internal consistency (§4 per-example notes).

### Conflicts adjudicated

- **Router+`path_map` (codebase report) vs `Command(goto)`+`destinations=` (LangGraph
  report/bead)** → resolved in favor of `Command(goto)`+`destinations=` for mode (a),
  with router+`path_map` as a documented fallback (§6). The deciding factor: only
  `Command(goto)`/`Command.PARENT` can express the cross-sub-construct mesh handoff, and
  it unifies both modes on one factory change; `destinations=` recovers the free compile-
  time target check.

### Report path

`docs/design/dynamic-handoff-research-2026-07-13.md` (this file).
