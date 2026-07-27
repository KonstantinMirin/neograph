# Phase 9 design-verification: Construct-as-Portal-mesh-member export/import + dispatch-mode Portal lowering

Date: 2026-07-27
Scope: `neograph-s7zt3.12` (master doc `docs/design/agent-spec-portal-master-architecture-2026-07-27.md` §2/§3/§5, rows "PORTAL (peer, Construct as non-entry/entry member)" and "PORTAL (dispatch mode)").

This is a design-verification pass only — nothing implemented. All claims below are grounded in real reads of the installed `pyagentspec==26.1.2` package and neograph's own `_agent_spec.py` / `loader.py` / `_wiring.py` / `factory.py` / `modifiers.py`, plus one throwaway repro (built and deleted, not committed).

---

## 0. Gate check (required before committing to any approach)

`tests/agent_spec_capabilities.py`'s `NODE_FAMILIES` registry was diffed against the installed `pyagentspec` package by walking `pyagentspec.flows.node.Node.__subclasses__()` recursively. **The registry is exhaustive and up to date** — all 14 concrete node classes it lists (`AgentNode`, `ApiNode`, `BranchingNode`, `CatchExceptionNode`, `EndNode`, `FlowNode`, `InputMessageNode`, `LlmNode`, `MapNode`, `OutputMessageNode`, `ParallelFlowNode`, `ParallelMapNode`, `StartNode`, `ToolNode`) are exactly the classes the installed package ships; nothing missing, nothing stale.

The one primitive that looks like a "conditional/dynamic-routing node" is `BranchingNode` (family `echo`). Read its real source (`pyagentspec/flows/nodes/branchingnode.py`): it takes one input value as a lookup key into a **statically pre-declared** `mapping: Dict[str, str]` (branch-name values fixed at spec-authoring time) and falls back to a `default` branch on a miss. This is a *static discriminated router over a fixed, named candidate set* — it is **not** a "synthesize an arbitrary flow at runtime and execute it" primitive. This distinction turns out to matter a great deal for C2 below (§3).

The second capability check — Swarm composability — turned up the actual resolving fact for C1: `pyagentspec.swarm.Swarm.first_agent` and each element of `relationships: List[Tuple[AgenticComponent, AgenticComponent]]` are typed as the abstract base **`AgenticComponent`**, not `Agent`. Concretely verified:

```python
from pyagentspec.agenticcomponent import AgenticComponent
from pyagentspec.flows.flow import Flow
from pyagentspec.agent import Agent
issubclass(Flow, AgenticComponent)   # True
issubclass(Agent, AgenticComponent)  # True
```

**`Flow` is itself an `AgenticComponent`.** A repro built a real `Swarm(first_agent=some_agent, relationships=[(some_agent, some_flow)])` and it constructed successfully (Pydantic validation passed) — a bare `Flow` object can sit directly in a `Swarm`'s member graph, no wrapper, no new primitive. This is the single fact that resolves the "does Agent support being backed by a multi-step Flow" question in the task: **no, `Agent` itself has no Flow-backing field (its only body-bearing field is `system_prompt: str`, confirmed from `Agent.model_fields`) — but the mesh membership question was never actually "can an `Agent` be Flow-backed," it's "can a `Swarm` *member slot* hold something Flow-backed," and the answer is yes, directly, because the member slot's declared type is the wider `AgenticComponent`, of which `Flow` is a first-class citizen.** `FlowNode` (the wrapper `_lower_construct_item`'s ordinary Construct branch uses to embed a subflow *inside* a parent `Flow`'s node list) is confirmed **not** an `AgenticComponent` subclass — so it is the wrong shape for a Swarm member and must not be reused here; the raw `Flow` returned by `to_agent_spec()` is used unwrapped instead.

(One incidental repro finding, not part of this design: `Swarm.model_dump_json()` raised `PydanticSerializationError: Missing proper serialization context` on my ad hoc call. This is irrelevant to neograph's contract — `tests/test_agent_spec_matrix.py` confirms `to_agent_spec`/`from_agent_spec` operate on **in-memory pyagentspec objects directly**, never JSON strings, so this is not a gap neograph needs to solve.)

---

## 1. C1 export: Construct-as-Swarm-member lowering

### What it builds, concretely

Reuse the exact recursion `_lower_construct_item`'s own `isinstance(item, Construct)` branch already uses (`_agent_spec.py:832-835`, `sub_flow = to_agent_spec(item)`), but do **not** wrap the result in a `FlowNode` — use the returned `Flow` directly as the `AgenticComponent` slotted into `Swarm.first_agent` / `Swarm.relationships`.

Verified against the actual `should_pass` fixture that makes the non-entry case WORK at the IR level today (`tests/check_fixtures/should_pass/portal_construct_member.py`): the mesh-member `Construct` declares `input=Handoff, output=Handoff` — i.e. **the author already declares the Construct's boundary port to equal the mesh's uniform payload type**, exactly the same discipline `_check_portal_mesh` enforces for every other member. Because `to_agent_spec()`'s existing (non-Portal) Construct-lowering path already produces a faithful `StartNode`/`EndNode` pair from `Construct.input`/`.output` for every ordinary sub-construct export, the recursive `Flow` for a mesh-member Construct will already expose exactly the mesh payload's fields as its Start/End properties — no new translation glue needed. This is why the master doc correctly calls this "the existing recursive-Flow-production pattern," not new design.

Concretely, `_lower_portal_mesh_to_swarm`'s per-member loop (`_agent_spec.py:914-924`) needs a type branch:

```python
for member in members:
    if isinstance(member, Construct):
        agents_by_name[member.name] = to_agent_spec(member)   # Flow IS-A AgenticComponent, used unwrapped
        # no _translate_placeholders call: Construct has no .prompt / dict-form .inputs
        # no _MARK_PROMPT_SPEC: nothing prompt-shaped to preserve at this level
        # (member's own internal nodes carry their own prompt markers recursively)
    else:
        rewritten, ref_props, flat_to_original = _translate_placeholders(
            member.prompt or "", _properties_for(member.inputs), member.name
        )
        agent = _make_agent(member, tools_mod, ref_props, [], rewritten)
        agent.metadata = {**(agent.metadata or {}), _MARK_PROMPT_SPEC: _prompt_spec_marker(member, flat_to_original)}
        agents_by_name[member.name] = agent
```

This is the *only* code-shape change needed inside `_lower_portal_mesh_to_swarm` itself. Everything downstream of `agents_by_name` (relationship-tuple construction, `entry_portal` marker attachment) already reads `member.modifier_set.portal`/`.name`, which are shared-base (`Modifiable`) fields present on both `Node` and `Construct` — confirmed no change needed there.

**Prerequisite, already tracked separately, not new scope for this phase**: the `mesh_members` filter one function up (`_agent_spec.py:961-967`, `isinstance(item, Node)`-gated) must widen to `isinstance(item, (Node, Construct))` — this is Phase 5 / A1, already filed as its own reuse-only fix and depended on by this phase per the master doc's own ordering. Confirmed still present at HEAD, unchanged.

**Entry-position case**: identical lowering — `entry.modifier_set.portal` is read the same way regardless of Node/Construct (shared base). The *only* reason the entry case isn't ready today is Phase 1 (`neograph-s7zt3.5`, the two-site compiler bug), already an explicit dependency of this bead. Once Phase 1 lands, this same per-member branch handles entry and non-entry identically — there is no separate entry-specific export design needed.

**Verdict: C1 export is now fully resolved, no remaining open design question.** It is glue, not new capability — one `isinstance` branch reusing `to_agent_spec` recursion and an already-`AgenticComponent`-typed slot.

### C1 import (the reverse)

`_reconstruct_swarm_mesh` (`loader.py:689`) calls `_swarm_agents_ordered(swarm)` (generically named "agents" but implementation-agnostic — it only touches `.name`/identity/`swarm.relationships`, verified by reading it: no `Agent`-specific access) and then, per member, unconditionally calls `_node_from_spec_agent(...)`.

The new branch: if a swarm member is a `Flow` (`isinstance(member, Flow)` after `_import_agent_spec_import_classes()` brings the real class in), recurse via the existing `from_agent_spec(member)` entry point (the *same* function this whole call is inside) to get back a `Construct`, then attach `Portal(to=peers)` to it exactly as the Node branch does — `Construct` supports `|` (Modifiable base), so `sub | Portal(to=peers)` is not new mechanism, it's the same operator already used on Node members one line below.

```python
for agent in agents:
    peers = [dst.name for (src, dst) in swarm.relationships if src is agent]
    if isinstance(agent, Flow):
        sub = from_agent_spec(agent)
        sub = sub.model_copy(update={"name": agent.name})
        members.append(sub | Portal(to=peers))
        continue
    member = _node_from_spec_agent(agent.name, agent, None, {"handoff": payload}, payload)
    ...
    members.append(member | Portal(to=peers))
```

**This is where a genuine, only-partially-closed design question remains** (see §1a below) — everything *except* the payload-type reconciliation is pure reuse (the `FlowNode`→`Construct` recursion pattern the "bare" `from_agent_spec` walk already uses at line 776-780 is structurally the same recursion, just triggered from the Swarm-member context instead of the ordinary Flow.nodes walk).

### §1a. The one real open question in C1: import-side payload-type reconciliation

For a **Node** member, `_node_from_spec_agent(agent.name, agent, None, {"handoff": payload}, payload)` *forcibly overrides* the reconstructed node's inputs/outputs to be the synthesized uniform mesh-payload type — always safe, because an agent-mode Node's actual runtime output is whatever an LLM factory is told to structure-output as; declaring it as `payload` doesn't have to match anything already fixed.

For a **Construct** member, this override is not available the same way: `from_agent_spec(agent)` reconstructs a `Construct` whose `.input`/`.output` are *derived from the sub-flow's own `StartNode`/`EndNode` Properties* — they are not a knob the loader can independently rebind without lying about what the sub-pipeline's interior nodes actually produce. `Construct.model_copy(update={"output": payload})` would change the *declared* type without changing what the reconstructed interior nodes actually compute, silently reintroducing exactly the kind of divergence this whole investigation's north star (fail-loud over fail-soft) forbids.

**For the primary target — round-tripping neograph's own export** — this is a non-issue, verified structurally, not just assumed: a Construct only ever reaches `_lower_portal_mesh_to_swarm` as a mesh member because `_check_portal_mesh` already required its `.output` to equal the mesh's uniform payload type *before* export ever ran (per §2's validator description, and directly demonstrated by the fixture declaring `output=Handoff` matching the mesh's `Handoff` payload). Since `to_agent_spec`'s ordinary Construct-lowering path is already required (by every other existing sub-construct export) to preserve `.output`'s field set faithfully into the sub-flow's `EndNode` Properties, the reconstructed `Construct.output` on import will *already* equal the payload type with no forcing needed — `sub.output` should come back exactly as `payload` (or a structurally-identical model) with zero coercion.

**What is NOT fully closed**: for a **foreign** (non-neograph-originated) Agent Spec `Swarm` containing a `Flow` member — the general "best-effort" import case `_reconstruct_swarm_mesh`'s own docstring already commits to (it explicitly says *"best-effort... a structural downgrade of the Swarm's own runtime, not a lossless import"* for the Node case) — there is no guarantee the recursively-reconstructed `Construct.output` will equal `_synthesize_swarm_payload`'s heuristically-folded type. Two honest options, neither implemented here, both requiring an explicit choice at implementation time:

1. **Fail loud** if `sub.output` doesn't structurally match `payload` for a foreign Flow member (consistent with the project's fail-loud posture, and arguably *more* correct than the existing Node path, which silently coerces).
2. **Wrap**, not reconcile: synthesize a thin adapter — but Construct has no native "declare a different output type than what your nodes produce" concept, so this would need a new synthetic terminal node inside the reconstructed Construct that maps its real output onto the payload shape, which is real new machinery, not reuse.

This document does not pick between (1) and (2) — it is a genuine open question, but only for the foreign-Swarm-import branch of the reconstruction; **the round-trip-of-our-own-export branch, which is what "full Construct-as-mesh-member export/import" primarily means for this ticket, has no open question** (verified above, not assumed).

---

## 2. C2: dispatch-mode Portal's Flow-node lowering

### What dispatch mode actually does at runtime (read from `factory.py:542` `make_portal_dispatch_fn` and `_wiring.py:971` `_add_portal_dispatch`)

A dispatch node (`route="decide"`) is a **plain linear node** at the graph level — one static in-edge, one static (or gate-routed) out-edge, never a `Command`. Its body runs like any think/agent/scripted node and its typed OUTPUT carries two special fields (`portal.spec_field`, `portal.input_field`): a *serialized Agent Spec dict* and a *dispatch input dict*. The wrapper then, at every invocation: deserializes that dict via `pyagentspec.serialization.AgentSpecDeserializer` → `Flow` → `from_agent_spec(flow)` → `Construct`, compiles it (`compile(sub, scripted=portal.scripted, conditions=portal.conditions)`), invokes it with the dispatch-input dict, and writes the scanned result to state.

This means dispatch mode is **already built on top of Agent Spec as a runtime planning/execution mechanism** — but the flow it invokes is chosen *and shaped* fresh at every call, from whatever the node's own body computes. There is no fixed, spec-authoring-time-known "candidate flow" for a static Agent Spec graph to point at.

### Why `BranchingNode` does not close this gap (contra a hopeful reading of the gate-check)

`BranchingNode.mapping: Dict[str, str]` is a *closed, named, pre-declared* set of branch destinations chosen at spec-authoring time — a discriminated static router, structurally the same shape as neograph's own `_BranchNode` lowering (`_agent_spec.py:824-830`, which is exactly a `BranchingNode` with a two-value `{"true": "true", "false": "false"}` mapping). Dispatch mode's "next step" is not a member of any fixed named set — it is an arbitrary `Flow` object synthesized at runtime by executing the node's own body, with no upper bound on shape, size, or existence of a natural correspondence to any pre-declared destination. **No primitive in the installed pyagentspec registry (`FlowNode`, `MapNode`, `ParallelMapNode`, `ParallelFlowNode` — the four "structural" family members, all of which declare their subflow as a static field set at construction time, confirmed by reading `flownode.py`) supports "the subflow reference is computed at runtime."** This is a genuine, verified capability boundary in the installed package, not an unresearched assumption — the gate check was performed and it correctly surfaces that pyagentspec HAS a dynamic-routing-*shaped* primitive, but that primitive solves a different problem (static discriminated choice) than dispatch mode's actual runtime behavior (runtime flow synthesis) requires.

### The concrete, closable design

Dispatch mode's *body* (the part that computes the flow spec) is not itself unrepresentable — it is a completely ordinary think/agent/scripted node whose declared output happens to include a spec-string and an input-dict field, both plain string/dict Properties. The part that genuinely has no static counterpart is "then compile and invoke whatever that string deserializes to." Following the exact precedent this codebase already uses everywhere else a modifier can't be statically represented (Oracle/Each/Loop/Portal-peer's own `max_hops`/`on_exhaust`/`route` marker, and the blanket `_reject_unrepresentable_fields` pattern for genuinely callable-valued fields):

**Export**: lower the dispatch node through its own ordinary per-mode lowering (`_lower_node` / `_make_agent` / `_lower_generation_step`, whichever the node's actual `mode` dispatches to today — unchanged, fully reused), producing a normal `LlmNode`/`AgentNode`/`ToolNode`. Attach a new marker, e.g. `_MARK_PORTAL_DISPATCH_SPEC`, on that primitive's `.metadata`, recording every dispatch-mode-only field with a static representation: `route` (always `"decide"`), `spec_field`, `input_field`, `output` (resolved type name via the same `str`-or-type handling `Portal.output` already supports), `on_invalid`, `error_handler`, `max_depth`. **`scripted`/`conditions` (the two `dict[str, Callable]` registries) cannot round-trip** — they are genuinely callable-valued, the same class of field `_reject_unrepresentable_fields` already fails loud on elsewhere (`raw_fn`, `skip_when`, `renderer`, `handoff_param`/`handoff_channel`) — so export must either (a) fail loud if `scripted`/`conditions` is non-empty (consistent precedent), or (b) accept a name-only registry-key list precondition mirroring the existing "only registered-string conditions serialize" rule used for `Loop.when`/`gate_tools_when` elsewhere in this codebase. This document recommends (b) is the better target (it preserves round-trip fidelity for the common case where the emitted sub-flows only ever reference pre-registered building blocks, which the docstring for `make_portal_dispatch_fn` states is already required — "the emitted flow may wire ONLY the pre-registered building blocks"), but does not fully resolve which registry-naming convention the marker should use — that is a small, bounded remaining decision, not a structural unknown.

**Import**: `loader.py` needs one new recognition arm — when a reconstructed primitive (whatever `_reconstruct_primitive_node` or the agent-reconstruction path returns) carries `_MARK_PORTAL_DISPATCH_SPEC` in its metadata, re-attach `Portal(route="decide", spec_field=..., input_field=..., output=..., on_invalid=..., error_handler=..., max_depth=..., scripted=<resolved from registry>, conditions=<resolved from registry>)` instead of leaving it as a bare node. This is structurally the same "read a `neograph/*_spec` marker back and re-attach a modifier" pattern already used for Oracle/Each/Loop and *planned but not yet implemented* for Portal-peer's own `_MARK_PORTAL_SPEC` (Phase 5 / B3) — genuine reuse of an established shape, not a new mechanism.

**What this design explicitly does NOT attempt, and states as a permanent, legitimate limitation, not a gap**: the static Agent Spec `Flow` graph produced by this lowering never shows what the dispatch node's emitted-and-invoked inner flow will look like, because that flow doesn't exist until runtime and isn't fixed across invocations. This mirrors — and should be documented alongside — the existing "raw_fn nodes cannot be exported" and "callable-valued field" family of honest, permanent export boundaries already established in `_reject_unrepresentable_fields`'s docstrings, rather than being framed as an incompleteness to eventually close.

**Verdict: C2 is now concretely designed** (per-mode lowering reused + a new marker, symmetric read-back), with exactly one small remaining implementation-level decision explicitly flagged (`scripted`/`conditions` registry-key naming convention for the marker) rather than a structural unknown.

---

## 3. Summary: what's resolved vs. what remains open

**Fully resolved, ready for an implementer to TDD with no further design decisions:**
- C1 export (both entry and non-entry, modulo Phase 1 landing first for entry): reuse `to_agent_spec` recursion, use the returned `Flow` unwrapped as the `AgenticComponent` Swarm-member slot (verified live: `Flow` IS-A `AgenticComponent`, `FlowNode` is NOT).
- C1 import for the round-trip-of-neograph's-own-export case: reuse the existing `Flow`→`Construct` recursion (`from_agent_spec`) plus the existing `Construct | Portal(...)` operator; no payload-coercion machinery needed because `_check_portal_mesh` already guaranteed the match before export ran.
- C2: reuse each dispatch node's existing per-mode lowering unchanged; add one new `_MARK_PORTAL_DISPATCH_SPEC` marker (symmetric read-back import arm) carrying every dispatch-mode-only *statically representable* field. The BranchingNode-based approach is explicitly ruled out with cited evidence, not assumed infeasible.

**Genuinely still open (flagged, not papered over):**
1. **C1 import, foreign-Swarm case only**: whether a `Flow`-member's reconstructed `Construct.output` should fail loud or be coerced when it doesn't structurally match `_synthesize_swarm_payload`'s heuristic type, for a *non-neograph-originated* Agent Spec Swarm. Two concrete options are laid out (§1a) but not chosen between — this needs a maintainer call, not further investigation (the investigation is done; the fix, whichever direction, is small).
2. **C2's `scripted`/`conditions` marker convention**: whether unregistered (raw dict) callables should hard-fail export or whether a name-only registry-key precondition should be added and enforced. Both are small, bounded implementation choices with a stated recommendation, not open research.

Everything else this ticket named — most importantly "does Agent Spec support a Construct backing a Swarm member at all" and "what should dispatch mode lower to" — is now answered with cited, repro-verified evidence rather than deferred.
