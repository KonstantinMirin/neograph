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
