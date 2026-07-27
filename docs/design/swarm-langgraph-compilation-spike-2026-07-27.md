# Empirical spike: how pyagentspec's real LangGraph adapter compiles `Swarm`

Date: 2026-07-27
Status: empirical ground-truth, supersedes the static-analysis-only conclusions
in `foreign-swarm-import-faithfulness-2026-07-27.md` regarding *how hard*
genuine parity would be (the mechanism identification there was directionally
correct; this spike pins the exact LangGraph primitives with running code).

## What was built and run

In an isolated worktree venv: installed `pyagentspec==26.1.2` +
`pyagentspec[langgraph]` (pulls `langgraph-swarm==0.1.0`, `langchain>=1.3`,
`langchain-openai`). Built real `pyagentspec.agent.Agent` +
`pyagentspec.swarm.Swarm` objects and compiled them through pyagentspec's own
`pyagentspec.adapters.langgraph._langgraphconverter.AgentSpecToLangGraphConverter`
(the same converter `AgentSpecLoader` uses internally) — **not** neograph's
compiler. Four configurations:

1. 2-agent Swarm, `HandoffMode.OPTIONAL`, one relationship `(a, b)`.
2. Same topology, `HandoffMode.ALWAYS`.
3. Same topology, `HandoffMode.NEVER`.
4. 3-agent Swarm, partial topology `(a,b), (b,c)` — no `a<->c`, no reverse edges.

Inspected the compiled `CompiledStateGraph.get_graph()` structure for 1-4
(no LLM calls needed for this). Then, for the strongest evidence, monkeypatched
`BaseChatOpenAI._generate` to return a canned tool-call (no network, no key)
and ran a real `compiled.invoke(...)` on the 2-agent OPTIONAL swarm to observe
actual runtime control flow when the model "calls" the handoff tool.

Scripts are throwaway, left in the scratchpad
(`spike_swarm.py`, `spike_invoke.py`); not part of any deliverable.

## Finding 1: the adapter delegates to `langgraph-swarm`; the primitive is `Command(goto=..., graph=Command.PARENT)` returned FROM a tool

`_langgraphconverter.py:_swarm_convert_to_langgraph` (line ~989) does not hand-roll
any handoff mechanism. For every `(from_agent, to_agent)` in `relationships` it
does:

```python
langgraph_swarm.create_handoff_tool(agent_name=to_agent_name)
```
and binds that tool onto `from_agent`'s tool list, then:
```python
return langgraph_swarm.create_swarm(
    agents=langgraph_agents,
    default_active_agent=agentspec_component.first_agent.name,
).compile(name=agentspec_component.name, checkpointer=checkpointer)
```

`langgraph_swarm.handoff.create_handoff_tool` (the actual mechanism) is a
`@tool`-decorated function whose BODY is:

```python
def handoff_to_agent(state: Annotated[Any, InjectedState],
                      tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    tool_message = ToolMessage(content=f"Successfully transferred to {agent_name}", ...)
    return Command(
        goto=agent_name,
        graph=Command.PARENT,
        update={"messages": [*state["messages"], tool_message], "active_agent": agent_name},
    )
```

This is confirmed both by source and by the live run (`spike_invoke.py`
output below): the handoff is a **normal bound tool**. When the LLM emits a
tool call for it, LangGraph's `ToolNode`/react-agent tool-execution step
invokes the tool function like any other tool — except this tool's return
value is a `Command` object instead of a string/dict. LangGraph recognizes a
`Command`-returning tool specially: instead of wrapping the return in a
`ToolMessage` and continuing the current node's loop, it propagates the
`Command` up (`graph=Command.PARENT` explicitly escapes the agent's own
compiled subgraph) and the **parent graph** immediately transfers control to
the node named `goto=agent_name` — a different top-level agent node — within
the *same* `.invoke()` call, no second top-level invocation needed.

Live proof (`spike_invoke.py`, `BaseChatOpenAI._generate` patched to return a
canned tool-call for the handoff tool on the first call, plain text on the
second — no network, no API key beyond a dummy for client construction):

```
=== final state active_agent === agent_b
=== message trace ===
- HumanMessage content='please handle my request'
- AIMessage name='agent_a' tool_calls=[{'name': 'transfer_to_agent_b', ...}]
- ToolMessage name='transfer_to_agent_b' content='Successfully transferred to agent_b'
- AIMessage name='agent_b' content='Done (fake call #2).'
```

All four messages came out of ONE `compiled.invoke()` call. Control genuinely
left `agent_a`'s own ReAct subgraph mid-turn and entered `agent_b`'s subgraph,
which then produced its own model turn and answered — matching the prior
static-analysis conclusion that this is **not** a typed-structured-output
routing decision at all. It is a tool call whose result is a LangGraph
`Command`, handled by LangGraph core's tool-node/Command machinery. Also
confirmed: `Command(` construction here lives entirely inside the
`langgraph-swarm` library, not inside pyagentspec's adapter — pyagentspec
just imports and calls the ready-made `create_handoff_tool`/`create_swarm`.

**Important correction to the "send_message vs handoff_conversation" framing**:
in the *pyagentspec docstrings* for `HandoffMode`, `OPTIONAL` is described as
giving both a `send_message` tool (reply-and-continue) and a
`handoff_conversation` tool (full transfer), while `ALWAYS` gives only
`handoff_conversation`. But the **actual LangGraph adapter code does not
implement this distinction at all**: `AgentSpecHandoffMode` is checked ONLY to
reject `NEVER` (`raise ValueError("Handoff mode NEVER is not supported...")`,
confirmed live — `HandoffMode.NEVER` fails at conversion time, before any
graph is built). For `OPTIONAL` and `ALWAYS` the exact same code path runs:
one `create_handoff_tool` per relationship, unconditionally. There is no
`send_message` tool anywhere in the LangGraph adapter or in `langgraph-swarm`
— grepping both confirms zero hits.

**Empirical confirmation, structural**: compiling the same 2-agent topology
under `OPTIONAL` and `ALWAYS` produces byte-identical top-level graphs:

```
=== 2agent (handoff=HandoffMode.OPTIONAL) ===
Top-level nodes: ['__end__', '__start__', 'agent_a', 'agent_b']
  __start__ -> agent_a  (conditional)
  __start__ -> agent_b  (conditional)
  agent_a -> agent_b    (conditional)
  agent_b -> __end__    (static)

=== 2agent (handoff=HandoffMode.ALWAYS) ===
Top-level nodes: ['__end__', '__start__', 'agent_a', 'agent_b']
  __start__ -> agent_a  (conditional)
  __start__ -> agent_b  (conditional)
  agent_a -> agent_b    (conditional)
  agent_b -> __end__    (static)
```

**Takeaway**: as-shipped, the LangGraph *reference implementation* of
`HandoffMode` is effectively two-valued (works / rejected-outright), not
three-valued. `OPTIONAL` vs `ALWAYS` is currently a no-op distinction in this
adapter (may be a genuine gap/bug in pyagentspec 26.1.2's LangGraph adapter,
or intentionally deferred — either way, it is what actually ships). Anyone
depending on the docstring's semantic difference for the LangGraph backend
specifically would be surprised.

## Finding 2: static `relationships` topology is enforced ONLY via which tools an agent is bound, not by any separate graph-structural gate

Each agent becomes its own top-level `StateGraph` node — a full compiled
ReAct subgraph (built via `langchain.agents.create_agent`). The parent graph
adds one node per agent via:

```python
builder.add_node(agent.name, agent, destinations=tuple(get_handoff_destinations(agent)))
```

`get_handoff_destinations` (`langgraph_swarm/handoff.py`) inspects the agent's
own bound tools and returns the `agent_name` for every tool tagged with
`METADATA_KEY_HANDOFF_DESTINATION` — i.e. it is **derived from the tool set
that was bound**, which was itself derived from `relationships`. The
`destinations=` kwarg on `add_node` is then only used by LangGraph (confirmed
by reading `langgraph/graph/state.py:850`, `if destinations is not None: ends
= destinations`) to build `ends` — the declared possible-`goto` set used for
static edge construction / graph introspection (`.get_graph()` drawing). It is
NOT an independent runtime gate layered on top of tool availability: there is
no code path that would additionally reject a `Command(goto=X)` at runtime
whose `X` wasn't in `destinations` — the enforcement that actually matters is
upstream of that, at the point tools are bound: an agent literally cannot
emit a tool call for a transfer tool it was never given, because LLM
function-calling can only select from the bound tool list.

Empirical confirmation with the asymmetric 3-agent topology
`relationships=[(a,b), (b,c)]` (no `a→c`, no `c→a`, no `b→a`, no `c→b`):

```
Top-level nodes: ['__end__', '__start__', 'agent_a', 'agent_b', 'agent_c']
  __start__ -> agent_a  (conditional)
  __start__ -> agent_b  (conditional)
  __start__ -> agent_c  (conditional)
  agent_a -> agent_b    (conditional)
  agent_b -> agent_c    (conditional)
  agent_c -> __end__    (static)
```

`agent_c` has NO outgoing conditional edge to any peer (it received no
handoff tool, since no relationship has `agent_c` as the "from" side) — the
static topology shows up exactly and only as "which conditional edges exist,"
which is a direct readout of "which tools this agent got," not an
independently-checked graph constraint. So: **topology is a soft (tool-binding
level) constraint, not a hard graph-structural one, in the reference
implementation** — confirming the prior static-analysis conclusion, now with
a running counter-example proving there's no hidden extra enforcement layer.
(It's still a *real* constraint in the sense that the LLM API genuinely cannot
call an unbound tool — but it lives at the tool-catalog layer, not the graph
layer neograph's Portal validates against.)

## Finding 3: recommendation for neograph's IR

The reference implementation is structurally much closer to a flat mesh of
independent LangGraph nodes than to a single Portal region with typed
structured-output routing. Concretely, for genuine `Swarm` parity neograph
would need:

1. **A tool, not a modifier on typed output.** The handoff destination must be
   exposed to the model as an actual bound tool (`transfer_to_<agent>`), whose
   *invocation* (a normal ReAct tool call, mid-turn) is what triggers the jump
   — not a `Node.outputs` type discriminant read after the model returns.
   This is a different node shape than Portal's current "typed output →
   `Command(goto)`" model: it needs a tool whose handler itself is the
   `Command`-returning function, wired into the SAME `agent`/`act` ReAct tool
   loop neograph already runs for `agent`/`act` mode nodes.
2. **The `Command(goto=..., graph=Command.PARENT)` escape hatch is exactly
   the mechanism neograph would need to add** to the agent/act ReAct tool
   execution path (`_agent_cycle.py`'s tool-dispatch), gated so it can ONLY
   fire for a specifically-declared handoff tool, never an arbitrary tool
   — preserving the "no jump into a region's interior" invariant by
   construction: a handoff tool's only legal `goto` targets are the entry
   nodes of OTHER declared peer regions in the same mesh (this is exactly
   what `Node.handoff_channel`/entry-port routing already gives Portal; the
   gap is only that today Portal triggers `goto` from a typed-output
   decision, not from a mid-ReAct-loop tool call).
3. **Topology should be validated at neograph's compile time, not left as a
   tool-availability-only soft constraint** — this is a place neograph can
   be STRICTLY BETTER than the reference implementation, consistent with the
   "restriction is the product" north star: since neograph already knows the
   full declared mesh topology at assembly time (`_validation_portal.py`),
   it can statically verify that every handoff-tool's target is a legal peer
   entry port, and reject unreachable destinations up front — a check the
   real `pyagentspec`/`langgraph-swarm` reference implementation does not
   perform at all (it only "enforces" topology by which tools got bound,
   with no compile-time cross-check).
4. **`HandoffMode` semantics are underspecified in the actual reference
   backend** (Finding 1) — neograph should NOT try to faithfully reproduce a
   3-way `HandoffMode.OPTIONAL` vs `ALWAYS` distinction that doesn't
   materially exist in the LangGraph adapter being imported from. If/when
   neograph implements Agent-Spec `Swarm` import, `HandoffMode.NEVER` should
   map to neograph's existing static/typed routing (its true analogue,
   consistent with pyagentspec's own docstring: "first_agent always remains
   the primary point of contact"), while `OPTIONAL`/`ALWAYS` should both map
   to "bind a handoff tool for every declared relationship" (matching what
   the reference implementation actually does), rather than inventing two
   different neograph-side behaviors for a distinction the source ecosystem
   itself doesn't currently implement.

**Bottom line reversal of the static-analysis-only framing**: the prior
investigation's core claim — that Swarm handoff is fundamentally a
mid-conversation TOOL CALL, not a typed-structured-output routing decision —
is confirmed, not overturned, by running the real code. But the *shape* of
the gap is narrower and more mechanical than it might have sounded in the
abstract: it is "add one more kind of tool whose handler returns
`Command(goto, graph=PARENT)` inside the existing agent/act ReAct loop, with
compile-time topology validation neograph can add that the reference
implementation lacks" — not a wholesale new control-flow paradigm. Genuine
parity is a bounded, addable capability alongside Portal, not a redesign of it.
