# Agent Spec interop — RATIFICATION gate (neograph-swcy1)

Date: 2026-07-13
Ratifies: `agent-spec-interop-2026-07-09.md` §4/§5/§6, `agent-spec-api/00-master-synthesis.md`, `02-flows-nodes.md`, `06-tools.md`, `09-patterns.md`.
Grounded against: `src/neograph/modifiers.py`, `construct.py`, `compiler.py`, `_wiring.py`, `_agent_cycle.py`, `state.py`, `runner.py`.

**INVARIANT ratified: Agent Spec is added BESIDE the neograph `Spec`, never replacing it.** The neograph `Spec` (`_spec_schema.py`, modifier-aware, one-line `oracle:`/`each:`/`loop:`/`operator:`) stays the authoring format. Agent Spec is a low-level flat-primitive interchange target added via `from_agent_spec()` (import, in `loader.py`) and `to_agent_spec()` (export, new `_agent_spec.py` free function wired through `__init__.__all__`). Confirmed: the two formats sit at different levels (§4 of the interop doc), and retargeting the neograph `Spec` at Agent Spec would discard the modifier DX. **GO on the invariant.**

---

## 1. Ratified mapping table (IR ↔ Agent Spec)

Round-trip codes: **DIRECT** (structure preserved both ways) · **LOWER** (export-only expansion; import cannot re-infer the modifier without a `neograph/` marker) · **GAP** (no representation, must fail) · **BEST-EFFORT** (import lowers to a name-bound node the user must supply a factory for).

### 1a. Structural — RATIFIED as written (no corrections)

| neograph IR | Agent Spec | Round-trip | Grounding |
|---|---|---|---|
| `Construct` | `Flow` | DIRECT | `construct.py` — nodes list + inferred DAG |
| `Construct.input` / `.output` (boundary ports) | `StartNode` inputs / `EndNode` outputs | DIRECT | singular boundary port, one typed entry/exit |
| `Node` mode `think` | `LlmNode` (+ `llm_config`) | DIRECT | `ThinkDispatch` |
| `Node` mode `agent`/`act` | `Agent` / `AgentNode` (+ tools) | DIRECT | ReAct cycle |
| `Node` mode `scripted`/`raw` | `ToolNode` / `ApiNode` | DIRECT (behavior name-bound at runtime) | factory binds impl |
| sub-`Construct` in `nodes` | `FlowNode` (nested) | DIRECT | `iter_nodes` recursion |
| `Node.inputs` dict (inferred) | explicit `DataFlowEdge`s | DIRECT (explicit↔implicit) | neograph infers what the spec declares |
| execution order (inferred) | explicit `ControlFlowEdge`s | DIRECT (explicit↔implicit) | compiler walks `nodes` order |
| typed I/O (Pydantic) | `Property` (JSON Schema) via type registry | DIRECT for primitives/list/dict/object/null/union/BaseModel; NO-REPR for `tuple`/`Literal`/`Enum` | `spec_types.py` |
| `Tool(name, budget, config)` | `Tool` / `MCPTool` (name-only) | DIRECT (name+budget+config); factory not serialized | `ToolSpec` |
| `_BranchNode` (ForwardConstruct if/elif/else) | `BranchingNode` | DIRECT | genuine sum-type routing |

### 1b. Modifiers — the export-only value-add

| neograph modifier (modifiers.py reality) | Agent Spec target | Round-trip | Verdict |
|---|---|---|---|
| `Oracle(n, models, merge_fn/merge_prompt)` — fan-out `Send()`×N + `defer=True` barrier + LLM-judge or scripted merge | `ParallelFlowNode` of N `LlmNode`s + merge `LlmNode`(merge_prompt)/`ToolNode`(merge_fn) + fan-out/fan-in edges | **LOWER** | RATIFIED. Flagship irreversible gap. Per-variant model selection (`models` round-robin, `merge_model`) has NO Agent Spec repr. Needs `neograph/modifier="oracle"` + `group_id` marker to re-infer. |
| `Each(over, key, on_error)` — router iterates collection + `Send()` per item + `defer=True` barrier | `MapNode` / `ParallelMapNode` wrapping an extracted sub-Flow | **LOWER** | RATIFIED. `over` dotted path + `on_error='collect'`/`EachFailure` have no native repr → ride in `EachSpec` metadata. |
| `Loop(when, max_iterations, on_exhaust)` — self-loop on Node / sub-construct re-run | `BranchingNode` (`{continue: back-edge, done: next}`) + cyclic `ControlFlowEdge` + self `DataFlowEdge` | **LOWER** | RATIFIED. `when`-as-callable, `max_iterations`, `on_exhaust`, `history` → `LoopSpec` metadata. A bare BranchingNode+back-edge is ambiguous (loop vs branch) without the `neograph/modifier="loop"` marker. |
| `Operator(when)` — **conditional HITL interrupt** (see §2) | **`BranchingNode`(gate) → `InputMessageNode` (the interrupt) → reconverge** | **LOWER** | **CORRECTED — see §2.** The design docs' "Operator → BranchingNode / gated edge, DIRECT" is WRONG: it omits the `InputMessageNode`, which is the load-bearing HITL-pause semantic. |

### 1c. Agentic patterns / cross-runtime

| Agent Spec | neograph | Round-trip | Verdict |
|---|---|---|---|
| `ManagerWorkers` | router `@node` + worker sub-constructs + `Loop` | reconstructable | NOT A GAP (recipe over existing primitives). RATIFIED. |
| `Swarm` | `Portal` mesh (`Command(goto)`+`destinations=`, shipped 2026-07-14) | reconstructable (v1: sibling peers, one construct level) | **SUPERSEDED 2026-07-22: Import onto Portal mesh (§3a).** Was REJECT-with-error at ratification; superseded once Portal shipped. |
| `RemoteAgent` / `OciAgent` / `A2AAgent` | client-initiated scripted/raw HTTP node + DI + metadata | **BEST-EFFORT** | **Import: best-effort lower (§3b).** |
| `ServerTool` / `BuiltinTool` | orchestrator-side execution — none in neograph | **GAP** | Import: FAIL with clear error (client-only substrate). RATIFIED unchanged. |
| `RemoteTool` | scripted `@node` + full spec in `metadata['neograph/remote_tool']` | BEST-EFFORT (lossy: declarative HTTP shape → function body) | RATIFIED. |

---

## 2. Operator lowering — PINNED (s9.3), and a doc correction

**What `Operator(when=)` actually is (grounded in `_wiring.py:_add_operator_check` + `compiler.py:518-520`):** a **conditional human-in-the-loop interrupt**, NOT a routing/branch primitive. The compiler inserts a check node *after* the modified node:

```python
def operator_check(state):
    should_pause = when(state)          # registered condition, name-only serialization
    if should_pause:
        human_input = interrupt(should_pause)   # graph checkpoints + STOPS
        return {StateKeys.HUMAN_FEEDBACK: human_input}
    return {}                                    # falsy → pass straight through
```

Resume is `run(graph, resume={...}, config)` → LangGraph `Command`. Both truthy/falsy paths reconverge to the same next node — **there is no fork.** Operator requires a checkpointer (`compiler.py:187-195`). The same `interrupt()`→`HUMAN_FEEDBACK` shape backs `gate_tools_when` on agent nodes (`_agent_cycle.py:892`).

**Faithful Agent Spec target:** `BranchingNode`(evaluates `when`) → true-branch **`InputMessageNode`** (Agent Spec's "interrupt flow, retrieve user input" — the actual `interrupt()` semantic) → reconverge; false-branch → next node. The `InputMessageNode` is the load-bearing piece.

**Is the round-trip lossy? YES — LOWER (export-only):**
1. `when` is a name-only registered-condition string (like `scripted_fn`/`merge_fn`); the callable never serializes.
2. Without `metadata["neograph/modifier"]="operator"` (+ the condition name), import sees `BranchingNode`+`InputMessageNode` and reconstructs primitives, not the `Operator` modifier.
3. `HUMAN_FEEDBACK` state plumbing and the checkpointer requirement are neograph runtime concerns with no Agent Spec field.

**Doc corrections required before the exporter is written** (§4 `mapping_corrections`).

---

## 3. The two localized import decisions

### 3a. Swarm — SUPERSEDED 2026-07-22: import onto a native Portal mesh (was: REJECT)

**This section is superseded.** At ratification time (2026-07-13), neograph had no runtime-decentralized peer-routing mechanism, so a Swarm import could only be a lossy "router-in-the-middle" reconstruction — hence the original REJECT decision below. `Portal` (formerly Keymaker) shipped afterward (epic neograph-t1f7z, 2026-07-14) and gives neograph its own `Command(goto)`+`destinations=` peer-routing mesh — a genuine runtime-decentralized primitive, not a static-DAG approximation. Swarm's `relationships`-driven runtime handoff now has a faithful, non-lossy target: a Portal mesh. Confirmed by explicit user decision 2026-07-22 (neograph-01i0g triage) in favor of import over reject.

**Current decision:** `from_agent_spec()` imports a Swarm-shaped Agent Spec Flow onto a Portal mesh — v1 scope: sibling-node peers at one construct level (matches Portal export's own v1 scope); cross-construct-boundary handoff remains deferred (neograph-do0d9, blocked pending a parent-scoped bubble-up design). `first_agent` maps to the mesh's entry member; `relationships` maps to each member's `destinations=`. This is doable losslessly because Portal mesh membership is already a first-class, marker-carrying neograph primitive (`neograph/modifier` markers on mesh members, symmetric to Oracle/Each/Loop/Operator) — there is no "silent mis-reconstruction" risk the original REJECT decision was guarding against, since the reconstruction target is a real primitive, not an approximation.

<details>
<summary>Original 2026-07-13 REJECT decision (superseded, kept for history)</summary>

Swarm is genuinely runtime-decentralized: agents choose the next recipient at runtime from `relationships`. neograph is a static-DAG, edges-inferred-from-types compiler. A best-effort "router-in-the-middle" reconstruction changes runtime semantics (topologically-ordered ≠ peer-to-peer) — exactly the silent-mis-reconstruction the interop doc's "VERIFY, don't trust / fail-loud" discipline (§6a) and the "validation as a safety rail for MACHINE-authored graphs" thesis (§1a) exist to prevent.

**Decision:** `from_agent_spec()` raises a `NeographError` naming `Swarm` as unsupported-on-import, and preserves `first_agent` + `relationships` in `metadata` for a possible future combinator. **No lossy lowering.** (Swarm remains "addable later as a first-class combinator over `Command(goto)`/`Send`" — a separate feature, not part of this import path.)

</details>

### 3b. RemoteAgent / A2AAgent / OciAgent — BEST-EFFORT lower

Unlike Swarm, these are **client-initiated**: neograph calls out to a remote endpoint and receives a result. That maps faithfully to a scripted `@node` doing an HTTP/RPC call, with connection params via `Annotated[..., FromConfig]` DI and the endpoint/protocol stashed in `metadata['neograph/remote_agent']`. No runtime-topology-semantics mismatch — it is an opaque call node.

**Decision:** `from_agent_spec()` lowers RemoteAgent/A2AAgent to a **name-bound scripted/raw Node** (factory supplied at bind time, same contract as any `ToolNode`/`ApiNode` import) and emits a WARNING that the remote-call implementation must be provided. The endpoint/auth-shape rides in metadata for re-export.

**The principled line: client-initiated → lowerable; orchestrator-side execution → fail.** RemoteAgent (neograph initiates the call) is lowerable; `ServerTool`/`BuiltinTool` (orchestrator executes) FAIL. This is the same axis already applied in `06-tools.md`.

---

## 4. Corrections to the design docs (s5 claims WRONG vs modifiers.py reality)

All four are the SAME error — the Operator mischaracterization — plus one internal inconsistency:

1. **`02-flows-nodes.md` line 22:** `BranchingNode → Operator(when=...) ... DIRECT (Operator)` — WRONG. Operator is a conditional HITL interrupt (LOWER), not a DIRECT BranchingNode mapping. BranchingNode's real DIRECT counterpart is `_BranchNode`/`Loop`, not Operator.
2. **`02-flows-nodes.md` lines 132-137 (Operator export lowering):** "Check node + BranchingNode routing + interrupt() on true branch" omits the `InputMessageNode` — the gate is real but the human-pause node is the semantic payload. Replace "exact lowering TBD" with the pinned §2 target.
3. **`00-master-synthesis.md` line 38:** "Routing: BranchingNode↔Operator/Loop/_BranchNode. DIRECT" — remove Operator from the DIRECT-routing row; it is a §3 LOWER modifier (interrupt gate), consistent with line 59's own `Operator → conditional BranchingNode/gated edge → no`. Internal contradiction: line 38 (DIRECT) vs line 59 (LOWER); modifiers.py confirms **LOWER** is correct.
4. **`agent-spec-interop-2026-07-09.md` §5 helper-lowering row:** `Operator(when=…) | condition → BranchingNode / gated edge (exact lowering TBD)` — target must be `BranchingNode(gate) → InputMessageNode → reconverge`.
5. **`00-master-synthesis.md` line 78 (GAP table, InputMessageNode):** "interrupt() not first-class in neograph, no node" is misleading — neograph DOES expose a first-class *conditional* interrupt (the `Operator` modifier). Correct framing: `InputMessageNode` (unconditional interrupt) ≈ `Operator` with an always-true `when`; on import an `InputMessageNode` reconstructs as `Operator` (via marker). It is the Operator import-target, not a pure gap.

(The Oracle / Each / Loop structural claims are all CORRECT against modifiers.py — no correction.)

---

## 5. Dependency — depend on `pyagentspec` behind an optional extra (fold-in of license input)

License-phase recommendation = **depend-optional-extra**. Ratified as §6.3 of the master synthesis:

- Depend on `pyagentspec>=26.1.0,<27` via an optional `[agent-spec]` extra (mirrors the `mcp-examples` pattern). **Do NOT vendor** (the plugin architecture + two-phase disaggregation is non-trivial and version-tracks the spec). **Do NOT depend on `pyagentspec.adapters.langgraph`** (neograph is its own compile target).
- Core `loader.py` stays YAML-only and dependency-light; `from_agent_spec` / `to_agent_spec` live behind the extra and import `pyagentspec.serialization` (core Serializer/Deserializer) only.
- License gate: confirm Apache-2.0 on the **published package**, not just the repos.

**Dep-vs-vendor: DEPEND (optional extra), not vendor.**

---

## 6. Residual blocking item (carried, not resolved here)

**§6.1 metadata round-trip survival test — UNVERIFIED, blocking for LOSSLESS round-trip only.** The entire `neograph/modifier` + `neograph/source` marker/embed strategy assumes `Component.metadata` survives `to_dict → from_dict` **including disaggregation/reaggregation**. This must be proven by a test before the marker strategy is trusted. It does NOT block primitive-level interop (export runs; import produces runnable primitives) — it blocks only the modifier-reconstruction (lossless) tier. First implementation task must assert `metadata["neograph/source"]` survives a full round-trip. If it doesn't, the embed needs a different carrier.

---

## 7. Go / No-Go

**GO — with caveats.**

The mapping is comprehensive and, apart from the single Operator mischaracterization (a doc fix, not a substrate limit), correct against `modifiers.py`. The two real unknowns resolve on a principled client-initiated-vs-orchestrator-side line: Swarm rejects, RemoteAgent best-efforts. The invariant (Agent Spec beside, not replacing, the neograph Spec) is sound.

**Caveats (must clear before/within the exporter task):**
1. Apply the five §4 doc corrections BEFORE writing `to_agent_spec()` — the Operator lowering must target `BranchingNode(gate)→InputMessageNode→reconverge`, not a bare BranchingNode.
2. The §6.1 metadata-round-trip survival test must pass before the marker/embed (lossless) tier is trusted.
3. `from_agent_spec()` must RAISE on `ServerTool` / `BuiltinTool` (no silent lossy lower); best-effort only for client-initiated RemoteAgent/A2AAgent + RemoteTool, each with a warning. `Swarm` no longer raises — see §3a (superseded 2026-07-22): it imports onto a Portal mesh instead, since that reconstruction is now lossless (a real primitive, not an approximation).
4. `pyagentspec` behind the `[agent-spec]` extra; license confirmed Apache-2.0 on the package.
