# Foreign Swarm import faithfulness: can neograph's tool-loop represent inter-agent handoff?

Date: 2026-07-27
Status: research, no code change

## Question

`_reconstruct_swarm_mesh` (`src/neograph/loader.py:689`) imports ANY pyagentspec
`Swarm` — foreign or neograph-originated — onto a native Portal peer mesh, giving
every member a synthesized uniform `Handoff(goto: str, ...)` payload
(`_synthesize_swarm_payload`, `loader.py:661`) and routing via `Command(goto=...)`.

A real pyagentspec `Swarm` (`.venv/.../pyagentspec/swarm.py`) does NOT route this
way: "each agent determines the next agent to be executed" via `HandoffMode`
(`NEVER`/`OPTIONAL`/`ALWAYS`) using `send_message`/`handoff_conversation` **tools**
called mid-conversation — there is no `goto` field or typed routing output in the
real model. So for a genuinely foreign Swarm, `_synthesize_swarm_payload`'s `goto`
field is neograph's own invention, not data the foreign Swarm ever produced.

Investigated: could a foreign Swarm import MORE faithfully as agent/act-mode
nodes equipped with synthesized "hand off to member X" tools, mirroring
`send_message`/`handoff_conversation`, instead of forcing Portal's typed-goto
model?

## Finding 1 (load-bearing): neograph's tool loop cannot transfer control flow to a peer node

Read in full: `src/neograph/_agent_cycle.py`, `src/neograph/_wiring.py`
(`_add_agent_cycle`, `_add_portal_agent_cycle_member`), `src/neograph/factory.py`
(guard G1 region), `src/neograph/tool.py`.

An agent/act node compiles to exactly three parent nodes — `{node}__agent`,
`{node}__tools`, `{node}__parse` — wired by a 3-way router
(`_agent_cycle.py:589-599`):

- forced-final or skipped → `{node}__parse`
- last message has no tool calls → `{node}__parse`
- otherwise → `{node}__tools`, which always loops back to `{node}__agent`
  (`_wire_agent_cycle_body`; `{node}__tools` is a plain node, never registered
  with `destinations=`)

Tool execution (`tools_body`/`atools_body`, `_agent_cycle.py:738-758`) always
appends `ToolMessage`s to the SAME node's message channel
(`StateKeys.agent_messages(field)`) and returns a plain dict state-update — never
a `Command`. `Command(` construction is a hard monopoly pinned by guard G1
(`TestCommandConstructionMonopoly`): it may only be constructed in `factory.py`/
`runner.py`, and grepping confirms `_agent_cycle.py` contains zero `Command(`
sites. The tools superstep is architecturally incapable of causing the graph to
route anywhere but back to its own node's `__agent` body.

Crucially, this holds even for a Portal mesh member (`_add_portal_agent_cycle_member`,
`_wiring.py:1329-1389`, docstring lines 1350-1353): "the interior
`__tools`/loopback nodes never [return a Command]" — ONLY `{node}__parse`, the
member's reconverging exit port, returns `Command(goto=...)`
(`factory.make_portal_agent_cycle_fn`). So even in neograph's OWN mesh
implementation, a tool call never causes routing to a peer. Peer handoff happens
only after the full ReAct turn completes and the node's *typed final output*
(the `goto` field, produced by `{node}__parse`) is read — structurally identical
to Oracle/Each/Loop/Branch: one IR node -> several parent nodes, with dynamic
routing decided at a single reconverging exit port, never mid-turn.

**Conclusion on Q1**: neograph's tool mechanism is exclusively "call a tool, get
a result back into the same node's continuing turn." There is no existing or
readily-added variant where a tool call itself transfers control flow to a peer
node. `send_message`/`handoff_conversation`-as-a-tool is not representable as
"just call it like any other tool" in the current architecture.

## Finding 2: is a tool-based handoff buildable as a new capability, and would it be better?

Technically, yes, something COULD be built: register a `handoff_conversation`-like
tool whose execution short-circuits `{node}__tools` into a `Command(goto=peer)`
instead of looping back to `{node}__agent`. But this is not "reuse the tool
mechanism" — it is a NEW IR capability on the same footing as Portal itself
(cf. `AGENTS.md`'s "PORTAL dynamic-handoff exception is the second sanctioned
new-IR capability"): it would need `{node}__tools` to sometimes be a
Command-emitting exit port (breaking the current invariant that only `__parse`
exits), a new reserved tool name/marker convention, and a story for what happens
to budget/checkpoint state and the in-flight ReAct turn when a handoff fires
mid-loop instead of at a clean turn boundary. Concretely it would be reinventing
Portal's `Command(goto)` primitive with a different trigger (an LLM tool call
instead of a typed final-output field), while giving up Portal's existing
guarantees: entry-port-only routing, `max_hops`/`on_exhaust` accounting, and
static reachability validated by `_check_portal_mesh`. It would not be a more
faithful representation so much as a parallel, weaker reimplementation of the
same primitive — and it would still be neograph's own invention layered on top
of a foreign Swarm's tool-calling semantics, not a lossless capture of them
(pyagentspec's `send_message`/`handoff_conversation` distinction — sub-task vs.
full-context transfer — has no analog either way).

Given that, **Portal's typed-goto model is not "the wrong target that we
happened to pick" — it is the only control-flow-transfer primitive neograph
has**, and building an alternative would mean building Portal again under a
different name, with strictly worse invariants. This directly matches the
task's third hypothesis: it is the honest, correct "best available" answer, not
a wrong target to fix.

## Finding 3: the static `relationships` list as a partial alternative

`Swarm.relationships` (the directed agent-to-agent edges) IS statically known
even though a foreign Swarm decides at runtime whether a given relationship
fires. `_reconstruct_swarm_mesh` already uses this static structure for Portal's
`to=[peers]` list (`loader.py:711`: `peers = [dst.name for (src, dst) in
swarm.relationships if src is agent]`) — so the "static skeleton, dynamic
choice" split the task asks about is already exactly what's implemented: the
Portal mesh's static topology comes from `relationships`, and the runtime
choice of which edge to actually take is Portal's `goto` field, decided by the
LLM's structured final output. There is no unused static information being
discarded by forcing a goto model — the graph shape already rides the
relationships list; only the *decision mechanism* (typed output vs. tool call)
differs from the source semantics, and that mechanism gap is exactly what
Finding 1 shows cannot be closed without rebuilding Portal.

The one difference worth naming explicitly: `HandoffMode.OPTIONAL` (an agent
handing off after possibly doing local work, e.g. via `send_message` and
continuing) vs `ALWAYS` (immediate transfer) is real pyagentspec semantics the
goto-payload model flattens into "you always emit exactly one goto." A foreign
Swarm agent that legitimately never hands off some turns is forced into a
uniform must-choose-a-goto contract it wasn't designed around. This is real
information loss beyond what the current warning states.

## Recommendation

Do not change `_reconstruct_swarm_mesh`'s target representation. Portal's
typed-goto model is the best available faithful-enough representation given
what neograph can express, precisely because neograph's tool-calling mechanism
structurally cannot transfer control flow to a peer node (Finding 1), and
building a tool-triggered-Command mechanism would only reinvent Portal with
weaker guarantees (Finding 2). The existing warning (`loader.py:726-733`) is
correct in spirit ("structural downgrade... not a lossless import") but should
be sharpened to name the SPECIFIC mismatch precisely, since the current wording
is generic ("route-only synthesis... name-bound live-LLM agents") and doesn't
say why:

Suggested addition to the warning / docstring: *"pyagentspec Swarms route via
mid-conversation tool calls (`send_message`/`handoff_conversation`,
`HandoffMode.NEVER/OPTIONAL/ALWAYS`); neograph's agent/act tool loop cannot
transfer control flow mid-turn (only a node's reconverging exit port can emit
`Command(goto=...)`), so every member is forced to ALWAYS decide a `goto` as
structured final output. A Swarm agent's `HandoffMode.OPTIONAL` turns (do local
work, maybe hand off) collapse to a hard must-route-somewhere contract."*

This is a documentation/precision improvement, not an architecture change — no
source edit was made as part of this research (per instructions, read-only).

## Confidence

High. The load-bearing claim (Finding 1) is grounded in direct code reading of
`_agent_cycle.py`'s router and body functions, `_wiring.py`'s two agent-cycle
wiring functions (including the explicit docstring statement that only
`{node}__parse` ever emits `Command` even for Portal mesh members), and guard G1
(`Command(` construction monopoly) confirmed by grep showing zero `Command(`
occurrences outside `factory.py`/`runner.py`. No throwaway repro was run beyond
this static analysis — the control-flow claim is provable directly from the
wiring code (which node names are registered with `destinations=` and which
bodies return `Command` vs plain dicts) with no ambiguity a live run would
resolve differently.
