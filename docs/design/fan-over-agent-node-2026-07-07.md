# Fan (Each/Oracle/Loop) over an agent/act node — the inline mechanism is impossible; auto-wrap is the fix

Date: 2026-07-07
Source: neograph-m6d3.6 investigation (Wave-1 team, exec-m6d36). No code written; finding recorded before action per the artifact-to-disk rule.

## The prescribed mechanism (neograph-m6d3.6 deliverable 1) is architecturally broken for N>1

The ticket asked to generalize `_wire_each`/`_wire_oracle` from a single `fan_name` to an inline
`(entry_node, exit_node)` pair: `Send` to `{node}__agent` and defer the barrier on `{node}__parse`,
so a fan modifier could wrap the multi-node inline ReAct region an agent/act node expands into.

This cannot work. An agent/act node's ReAct cycle keeps its per-turn state in **shared reducer
channels** (`neo_agent_messages_*` = `add_messages`, `neo_agent_budget_*`, `neo_agent_tool_log_*`).
LangGraph `Send` isolates only the **first** superstep's input payload. Every subsequent superstep
(tools -> agent loopback, then parse) reads the shared channel. So N fanned branches' messages merge
into ONE channel and the loopback/parse collapse into a single tangled continuation.

Minimal repro (exec-m6d36): 3 `Send`s into a 2-superstep A->B->A region with a shared `add`-reducer
channel -> all three branches' messages merged into a single list; only one tangled continuation ran.
This is a fundamental LangGraph property, not a neograph bug: **subgraphs are the isolation mechanism
for multi-superstep parallel state.**

## What actually works (proven today, no code change)

Wrap the agent node in an isolated single-node sub-construct and fan over THAT via the EXISTING
subgraph path (`_add_subgraph` -> `_wire_oracle`/`_wire_each`). exec-m6d36 ran `Oracle(n=2)` over an
agent-in-sub-construct through a history-driven fake: two isolated `subgraph_start/complete`, isolated
message channels per invocation, both variants reached the merge. Here `entry == exit ==` the subgraph
node, so the `(entry, exit)` signature generalization buys nothing — the real fix is **auto-wrapping**.
(The m6d3.2 compile-error's own hint already says "wrap the agent node in a sub-construct"; m6d3.6
should AUTOMATE that, not go inline.)

## Re-scope of neograph-m6d3.6

- Deliverable 1 (inline `(entry, exit)` signature change) -> REPLACED by: when a fan modifier
  (Each/Oracle/Loop) is applied to an agent/act `Node`, auto-synthesize an isolated single-node
  sub-construct wrapping the ReAct cycle and route through the existing subgraph fan path.
- Deliverable 2 (remove the m6d3.2 compile-time error) -> STAYS.
- Deliverable 3 (move `oracle_over_agent_node.py` fixture should_fail -> should_pass) -> STAYS.

## Open design questions the auto-wrap must answer (need a design call)

1. **Input-port synthesis** for an agent node that has upstream `inputs` (dict-form / DI params /
   `fan_out_param`): wrapping shifts reads from upstream state fields to `neo_subgraph_input`.
2. **Each-item delivery across the subgraph boundary**: does `neo_each_item` reach the agent's
   `fan_out_param` inside the sub-construct?
3. **Loop-over-agent** condition reading the subgraph output.

The proven base case is Oracle/Each over a **self-contained** agent node (no upstream inputs, no
fan_out_param). The three cases above are the hard surface.

## Decision (2026-07-07)

Re-scoped to **base case + fail-loud** (user away; taken on best judgment, to be re-surfaced):
implement auto-wrap for Oracle over a self-contained agent node (satisfies deliverable 3's fixture);
for the three hard cases raise a clear assembly-time `ConstructError` naming the exact unsupported
combination + node (never a silent broken graph) and file one follow-up bead per case. Each-over-agent
is implemented only if `neo_each_item` delivery across the subgraph boundary is a clean, unambiguous
change; otherwise it too is fail-loud + a bead.

## Repro 1 — broken inline: Send into a multi-superstep shared-reducer region

Pure LangGraph. `A` = `{node}__agent`, `B` = `{node}__tools`; shared `add`-reducer `msgs` channel
== `neo_agent_messages_*`. Three fan branches. Expected 3 isolated loops -> 3 collector entries;
actual: 1 merged channel, 1 tangled continuation.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

class State(TypedDict):
    items: list
    msgs: Annotated[list, operator.add]        # shared reducer (== neo_agent_messages_*)
    collector: Annotated[list, operator.add]   # barrier collects here

def router(state):   return [Send("A", {"seed": i}) for i in state["items"]]
def A(state):        return {"msgs": [f"A-turn:{state.get('seed')}"]}
def after_A(state):  # per-task loopback: loop once, then parse
    turns = [m for m in state["msgs"] if m.startswith("A-turn")]
    return "B" if len(turns) < 2 else "parse"
def B(state):        return {"msgs": ["B-tools"]}
def parse(state):    return {"collector": [tuple(state["msgs"])]}

g = StateGraph(State)
for n, f in [("A", A), ("B", B)]: g.add_node(n, f)
g.add_node("parse", parse, defer=True)
g.add_conditional_edges(START, router, path_map=["A"])
g.add_conditional_edges("A", after_A, path_map=["B", "parse"])
g.add_edge("B", "A"); g.add_edge(["parse"], END)
out = g.compile().invoke({"items": [1, 2, 3], "msgs": [], "collector": []})
print(out["collector"])
# -> [('A-turn:1', 'A-turn:2', 'A-turn:3', 'B-tools', 'A-turn:None')]
#   ONE tangled entry — the 3 branches merged into one shared channel. NOT 3 isolated loops.
```

## Repro 2 — working auto-wrap: Oracle(n=2) over an agent-in-sub-construct

neograph, runs today, no source change. Two isolated `subgraph_start/complete`, per-invocation
`neo_agent_messages_*`, both variants reach the merge.

```python
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel
from neograph import Construct, Oracle, Tool, compile, construct_from_functions, node, run
from tests.fakes import (build_test_compile_kwargs, build_fake_llm_kwargs,
                         register_tool_factory, register_scripted)

class Port(BaseModel, frozen=True): topic: str
class Res(BaseModel, frozen=True):  items: list[str]

class Fake:   # stateless, history-driven: turn 1 calls the tool, then answers
    def __init__(self): self._model=None; self._structured=False
    def bind_tools(self, t): return self
    def invoke(self, messages, **k):
        if self._structured: return self._model(items=["variant"])
        n = sum(isinstance(m, ToolMessage) for m in messages)
        if n == 0:
            m = AIMessage(content=""); m.tool_calls=[{"name":"search","args":{},"id":"s1"}]; return m
        return AIMessage(content='{"items": ["done"]}')
    async def ainvoke(self,*a,**k): return self.invoke(*a,**k)
    def with_structured_output(self, model, **k):
        c = Fake(); c._model=model; c._structured=True; return c

class SearchTool:
    name="search"
    def invoke(self, a, config=None, **k): return "found"
    async def ainvoke(self,*a,**k): return "found"

register_tool_factory("search", lambda config, tc: SearchTool())
register_scripted("merge_fn", lambda variants, config: Res(
    items=sorted(i for v in variants for i in v.items)))

@node(mode="agent", outputs=Res, model="reason", prompt="test/explore",
      tools=[Tool(name="search", budget=3)])
def agent_fn(port: Port) -> Res: ...

# THE FIX: isolate the agent in a single-node sub-construct, fan over THAT.
sub = construct_from_functions("agent_sub", [agent_fn], input=Port, output=Res) \
      | Oracle(n=2, merge_fn="merge_fn")
graph = compile(Construct("p", nodes=[sub]),
                **build_test_compile_kwargs(), **build_fake_llm_kwargs(lambda tier: Fake()))
print(run(graph, input={"topic": "x"}))
# -> {'agent_sub': Res(items=['done', 'done'])}  — 2 isolated ReAct cycles, both merged.
```

---

## Addendum 2026-07-07: independent adversarial verification

A fresh-context architect re-derived the impossibility with its own repros
(N=2 Send into a 2-node cycle over a shared reducer channel -> ONE merged
entry; a last-value channel -> InvalidUpdateError "Can receive only one value
per step" — so reducer channels merge silently and static channels crash;
neither isolates) and confirmed NO missed engine mechanism: Send seeds only
the first superstep; LangGraph channel keys are static (no per-branch
namespacing); subgraph-per-Send is the only isolation primitive — which is the
shipped auto-wrap fix. Verdict: impossibility CONFIRMED; auto-wrap is the
architecturally right answer (it reuses the framework's one isolation
primitive and automates what the old compile error's hint already told users
to do; the (entry,exit) signature was the wrong abstraction — the real axis is
single-node vs multi-superstep region, which sub-constructs already
discriminate).

Deferred-case estimates and REQUIRED SEQUENCE (front-load the keystone):
qot6 (input-port synthesis, ~2-3d, genuinely hard: dict-form fan-in + DI +
neo_subgraph_input surface) -> 1h8c (each-item across the boundary, ~0.5-1d
once a non-empty port exists) -> gk3e (loop-over-agent, ~1-1.5d, gated on
qot6's port work). Repro scripts (session scratchpad, ephemeral):
repro_merge.py, repro_send_namespace.py, repro_subgraph_isolation.py.

---

## Addendum 2026-07-07: qot6 delivered — input-port synthesis (single-producer)

Open design Q #1 (input-port synthesis) is now IMPLEMENTED for the tractable,
unambiguous cases. `_fan_agent_wrap._synthesize_port` derives the wrapping
sub-construct's `input=` boundary + the bare agent's read side from the agent's
declared upstream inputs. Three shapes are supported; two stay fail-loud.

Supported (Oracle over an agent with **zero or one** upstream producer):
- **Self-contained** (`inputs=None`) — empty synthesized port. The base case.
- **Single-type** (`inputs=T`) — port IS `T`. The parent's upstream `T` is found
  by the subgraph's existing type-based `_scan_subgraph_input` and delivered as
  `neo_subgraph_input`; the bare agent keeps `inputs=T` and single-type
  extraction reads it back. No inner rewrite.
- **Single-key dict-form** (`inputs={k: T}`) — port is `T`; the bare agent's read
  is rewritten to `{neo_subgraph_input: T}`, the same convention the `@node`
  sub-construct port mechanism uses (`_construct_builder._cleanup_inputs_and_register`).
  Fan-in extraction (`_extract_fan_in_dict`, which uses `get_required`) then reads
  the boundary field — so a failed delivery crashes rather than silently passing.
- **DI params ride free**: `FromInput`/`FromConfig` never enter `inputs` (the
  decorator strips them into `_param_res`, preserved across `model_copy`), and the
  subgraph forwards `config`, so DI resolves inside the sub-construct with no port
  work. An agent with BOTH an upstream edge and DI params is supported.

Fail-loud (precise `ConstructError`, follow-up bead **neograph-qzrv**):
- **Multiple distinct dict-form producers** (`inputs={a: A, b: B}`) — the
  single-value `neo_subgraph_input` boundary can't carry N values; a clean fix
  needs a synthesized parent "packer" node (bundle the N upstreams into one port
  model) + an inner unpacker or prompt-var mapping. Genuinely ambiguous (which
  prompt vars the bundle exposes), so deferred rather than punted-with-a-guess.
- **Dict-form (multi-output) agent OUTPUTS** — the sub-construct has one output
  boundary port; also tracked under neograph-qzrv.

Design note on the dict-form rewrite: the original prompt-var name `k` is NOT
preserved — it becomes `neo_subgraph_input`, exactly matching a manual
`construct_from_functions(input=...)` wrapping. Auto-wrap == manual wrap, which is
the consistency we want. There is no existing behavior to preserve (this case was
fail-loud before), so no regression. Fixture `oracle_over_agent_with_inputs.py`
moved should_fail -> should_pass; new should_fail `oracle_over_agent_multiple_inputs.py`
pins the multi-producer guard.

---

## Addendum 2026-07-07: 1h8c delivered — Each-item delivery across the boundary

Open design Q #2 (Each-item delivery) is now IMPLEMENTED. **Design call: option
(2)** from the ticket — map the fanned `neo_each_item` AS the wrapping
sub-construct's single-value input port (`neo_each_item` -> `neo_subgraph_input`),
mirroring the qot6 single-key dict-form rewrite. Option (1) (thread a *separate*
`neo_each_item` channel through the subgraph boundary INTO the isolated sub-state)
was rejected: it would add a second item-delivery mechanism parallel to the port
one, and the port is already the exact single-value carrier the item needs.
Auto-wrap == manual `construct_from_functions(input=ItemType)` + `| Each(...)`.

Mechanism (three touch points, symmetric with qot6):
- `_fan_agent._unsupported_reason`: Oracle and Each now share ONE input-shape
  support gate (self-contained OR single upstream producer; multi-producer and
  multi-output stay fail-loud under qzrv). The ONE asymmetry: Each PERMITS a
  fan-out receiver (`fan_out_param`) — that IS how an Each node consumes its item
  — while Oracle rejects one (Oracle fans no per-item value).
- `_fan_agent_wrap._wrap_agent_node`: the bare inner agent has `fan_out_param`
  CLEARED. Inside the isolated sub-construct the agent is NOT fanned (the Each fan
  runs at the PARENT level over the sub-construct); the item arrives as
  `neo_subgraph_input`, which `_synthesize_port`'s dict-form rewrite already points
  the read at. A stale `fan_out_param` would name a key no longer in `inputs`.
- `_subconstruct.make_subgraph_fn._build_sub_input`: for an Each-carrying sub,
  read `EACH_ITEM` from the (Send-populated) parent state and use it as
  `input_data` -> `neo_subgraph_input`, taking precedence over the blind
  `_scan_subgraph_input` type-scan (the scan could match the wrong instance; the
  item is the SPECIFIC dispatched value). The parent state model already carries
  `neo_each_item` because the wrapped sub classifies as EACH (`state.py`
  `has_any_each` walks `nodes_only + sub_constructs`).

**Self-contained Each-over-agent stays fail-loud** (new guard + should_fail
fixture `each_over_self_contained_agent.py`): an agent with `inputs=None` has no
port for the item, so every isolated cycle would run on empty input and emit
identical results keyed by distinct Each keys — a silent broken fan. Require a
consumed input (single-type or single-key dict-form).

Value-delivery is PINNED, not probe-only: `TestEachOverAgent` uses an
echo prompt-compiler (renders the upstream value into the user turn) + an echo
ReAct fake (reflects the seen prompt into its typed output), and asserts each
per-branch entry saw its OWN item value and NOT the sibling branch's — the
positive proof that the fanned item crossed the boundary into an isolated cycle.
Three surfaces covered: declarative single-type, declarative single-key dict-form,
and `@node` `map_over`. Fixture `each_over_agent_node.py` moved
should_fail -> should_pass (adjusted to declare `inputs=Item`).

---

## Addendum 2026-07-07: gk3e delivered — Loop-over-agent condition reads sub output

Open design Q #3 (Loop-over-agent) is now IMPLEMENTED. **Design call: reuse the
existing subgraph-loop path verbatim — NO new mechanism.** The auto-wrap isolates
the agent's ReAct cycle in a single-node sub-construct carrying the Loop modifier;
`_add_subgraph` classifies it LOOP and routes to `_add_subgraph_loop`, whose
`loop_router` unwraps the loop value via `_construct_loop_unwrap`, which reads
`field_name_for(sub.name)` — i.e. **the sub-construct's OUTPUT boundary surfaced
onto the parent field** (`make_subgraph_fn._build_update` writes
`{field_name: output_val}`). That is precisely the ticket's requirement ("the loop
condition must read the subgraph output, not a node output field"): the isolated
agent's typed output IS the sub output, and the router already reads that field,
never an internal `{node}__parse` field. No code path in `_wiring`/`_subconstruct`
needed changing for the read side.

Re-entry / iteration-count semantics (verified by E2E, not just asserted from
theory): the looped sub-construct's output field is an append-reducer list
(`result["<agent>"]` is `list[Output]`, `[-1]` is the latest), so
`make_subgraph_fn._build_sub_input`'s has_loop branch feeds the PRIOR typed output
back as `neo_subgraph_input` (the refine pattern) — each iteration runs a FRESH
isolated ReAct cycle seeded by the previous cycle's result. `_build_update`
increments `StateKeys.loop_count(field_name)` once per subgraph invocation, and
`loop_router` caps on `loop.max_iterations`. The three families now share ONE
input-shape support gate in `_unsupported_reason` (Oracle/Each/Loop); the only
per-family asymmetry is item-delivery (Each) vs a fan-out-receiver rejection
(Oracle) — Loop needs neither.

Value-delivery + convergence PINNED: `TestLoopOverAgent` drives a Draft-refine
agent whose fake reads the fed-back score from the rendered prompt (echo
prompt-compiler) and advances it +0.3 per cycle; asserts the loop converges
0.3 -> 0.6 -> 0.9 across EXACTLY 3 isolated ReAct sub-construct invocations and
exits when `score >= 0.8`. Two surfaces: declarative `| Loop(...)` and `@node`
`loop_when=`. Fixture `loop_over_agent_node.py` moved should_fail -> should_pass
(callable `when` reading `Draft.score`; declares `inputs=Draft` as the refine
port). Multi-input/multi-output Loop-over-agent stays fail-loud under qzrv, same
as Oracle/Each.

---

## Addendum 2026-07-07: qzrv delivered — packer-port synthesis (multi-input) + value-delivery pins

Open design Q from qot6's fail-loud list is now PARTIALLY lifted with two
deliberate design calls.

**Multi-input (LIFTED for Oracle) — packer-port synthesis.** An Oracle over an
agent with N distinct dict-form producers (`inputs={a: A, b: B, ...}`) compiles and
runs. The single-value subgraph boundary (`neo_subgraph_input`) carries ONE typed
value, so:
- A synthesized **packer** node runs in the PARENT: it fan-ins the N original
  upstreams and emits a synthesized `_NeoAgentPort_*` model bundling them. The
  wrapped sub-construct's `input=` is that model, found by the subgraph type-scan.
- Inside the sub-construct, one **unpacker** node PER key reads the port model
  (single-type) and re-emits its field under the ORIGINAL producer name, so the
  bare agent's dict-form fan-in reads them as peer producers — exactly as before
  wrapping. **This resolves the "ambiguous prompt-var surface" qot6 flagged: the
  answer is to re-expose the original keys (`{a}`/`{b}` still resolve), not rename
  them to a bundle field.** Auto-wrap == manual wrap: what the user would hand-write.

Mechanism: `wrap_fan_over_agents` now takes the per-compile `scripted_lookup` and
registers the packer/unpacker shims into it; `_add_subgraph` already threads that
dict into the recursive sub-construct compile, so the inner unpackers resolve in
the isolated sub-graph. `_expand_agent_node` returns `[packer, wrapped_sub]` for
the multi-producer case and `[wrapped_sub]` otherwise; `wrap_fan_over_agents`
`extend()`s the expansion into the parent node list in order.

**Scoped to Oracle.** Each/Loop multi-input stay fail-loud with a precise error:
Each ALSO delivers a fanned `neo_each_item` and Loop ALSO feeds its output back —
both compete for the same single-value boundary the packer occupies, so a sound
wiring needs more design. Not punted-with-a-guess.

**Multi-output (KEPT fail-loud) — design call.** Dict-form (multi-output) agent
OUTPUTS stay fail-loud. The isolating sub-construct has ONE output boundary port,
and an N-way merge of SECONDARY outputs (e.g. `tool_log`) across fanned variants is
undefined — the Oracle `merge_fn` contract is single-type. Surfacing only the
primary and silently dropping the declared secondaries would violate the user's
declared output contract (the anti-band-aid rule), so the precise `ConstructError`
stays. Pinned by `should_fail/oracle_over_agent_multi_output.py` +
`test_multi_output_agent_stays_fail_loud`.

**Wave-8 value-delivery pins (MANDATORY, committed).** The shipped qot6 agent-fan
tests asserted merged-N wiring but NOT that each isolated cycle SAW its upstream
value. `TestOracleOverAgentValueDelivery` now commits that property for the
single-type, dict-form, and DI shapes via the echo pattern (prompt-compiler renders
the upstream into the message; an echo ReAct fake reflects it into `Claims.items`;
the merge concatenates), asserting the upstream value appears in EVERY variant's
output. `TestOracleOverAgentMultiInputBundle` extends the same to the packer path,
asserting BOTH bundled values (`AVAL` and `BVAL`) reach each isolated cycle.
Fixture `oracle_over_agent_multiple_inputs.py` moved should_fail -> should_pass;
new `should_fail/oracle_over_agent_multi_output.py` pins the multi-output guard.
