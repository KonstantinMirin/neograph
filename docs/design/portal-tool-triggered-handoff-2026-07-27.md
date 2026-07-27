# Portal tool-triggered handoff: design spec

Date: 2026-07-27
Status: design, not started — no code changes in this document
Supersedes-in-scope: turns `swarm-langgraph-compilation-spike-2026-07-27.md`
Finding 3's recommendation into an implementable design, verified against
neograph's real `_agent_cycle.py`/`_wiring.py`/`factory.py`/`modifiers.py`.
Parent tracking: `neograph-s7zt3.12` (note pointer added, not closed).

---

## 0. One-paragraph summary

Add a new `Portal` field, `trigger: Literal["output", "tool"] = "output"`
(peer mode only), discriminated by a new `is_tool_triggered` property
mirroring the existing `is_dispatch` property. When `trigger="tool"`, an
agent/act mesh member's `{node}__tools` superstep — today a plain node with a
static `-> {node}__agent` loopback edge — is rewired to a `Command`-emitting
exit port with `destinations=(peer targets ∪ {node}__agent)`: it recognizes a
reserved bound tool call (`transfer_to_<peer>`, one per declared
`Portal.to` entry, synthesized at tool-instantiation time, never persisted
into `Node.tools` or any new IR field) and routes to the peer's entry via the
SAME `_portal_route_to_command`-family hop-budget/`HANDOFF_END` machinery
Portal already uses; absent a handoff call it emits `Command(goto={node}__agent)`
instead of a static edge (LangGraph forbids mixing a static out-edge with a
`destinations=`-registered dynamic node). `Command(` construction stays
100% inside `factory.py` (guard G1 unchanged, no allowlist edit) via one new
sibling function next to `_portal_route_to_command`. Zero new `Node`-level IR
fields are needed — the capability reuses `Portal.to`/`max_hops`/`on_exhaust`,
the existing `handoff_channel`/`handoff_param`, and `_check_portal_mesh`
unchanged, plus one narrow new validation rule (`trigger="tool"` requires an
agent/act member). Recommended as a first-class authoring feature, not an
import-only artifact.

---

## 1. Where this attaches in the IR

**Extend `Portal`, do not add a new `ModifierCombo` value.**

`ModifierCombo` classifies *which modifiers compose on one item*
(`frozenset({"portal"}) -> ModifierCombo.PORTAL`,
`frozenset({"portal", "operator"}) -> ModifierCombo.PORTAL_OPERATOR`,
`modifiers.py:102-103`). Tool-triggered handoff does not change which
modifiers are present on a member — it is still exactly one `Portal` (plus
optionally `Operator`, subject to the existing agent/act-mode restriction
below). It changes *how peer mode decides its `goto`*, which is precisely the
kind of sub-mode distinction `Portal` already carries: dispatch mode
(`route="decide"`) is a second behavior of the SAME modifier, discriminated by
the `is_dispatch` property (`modifiers.py:648-658`), not a second
`ModifierCombo`. `trigger` is the direct sibling of that pattern:

```python
class Portal(Modifier, frozen=True):
    ...
    trigger: Literal["output", "tool"] = "output"  # peer mode only

    @property
    def is_tool_triggered(self) -> bool:
        return not self.is_dispatch and self.trigger == "tool"
```

`model_post_init` gains one more mutual-exclusion check: `trigger` is
peer-mode-only (mirrors the existing `max_depth`-forbidden-in-peer-mode /
`spec_field`-forbidden-in-peer-mode checks already in that method) — setting
`trigger="tool"` together with `route="decide"` raises `ConfigurationError`
exactly like the existing dispatch/peer mutual-exclusion.

This is a **strictly smaller** IR footprint than the original PORTAL
exception needed, and it is consistent with the modifier-combo
single-source-of-truth work in progress
(`docs/design/modifier-combo-single-source-of-truth-2026-07-27.md`): every
consumer that currently reads `ModifierCombo.PORTAL` (compiler.py, state.py,
`_state_write.py`, `_subconstruct.py`, `_input_shape.py`, runner.py,
`_wiring.py`) needs **zero changes** for the new field, because none of them
classify by `Portal.trigger` — the combo-level decomposition (PrimaryShape:
does this item produce a `Command`? yes, unconditionally, exactly as today)
is unchanged. Only `_wiring.py` (topology wiring) and `factory.py` (the new
Command-construction sibling) need to read `portal.trigger` /
`is_tool_triggered` — both are new, narrowly-scoped call sites, not a
retrofit of the ten-module sweep.

---

## 2. The `graph=Command.PARENT` question — resolved definitively

**Neograph never needs `graph=Command.PARENT`, and this design adds no such
argument.**

Verified two ways:

1. **Real LangGraph `Command` semantics** (`langgraph.types.Command`,
   installed package, read directly):
   ```python
   graph: str | None = None
   ...
   # graph: 'None' = the current graph; 'Command.PARENT' = closest parent graph
   ```
   `graph=None` (the default, and what neograph's Portal code already passes
   by omission) means "route within the graph the returning node belongs to."
   `Command.PARENT` exists specifically to let a node inside a **nested**
   compiled subgraph escape one level up to the graph that embeds it as a
   node.

2. **Neograph's actual compiled shape**: every agent/act node's three parts
   (`{node}__agent`, `{node}__tools`, `{node}__parse`) are added directly to
   the SAME top-level `StateGraph` via `graph.add_node(...)` in `_wiring.py`.
   Neograph is NOT globally flat in general — `compiler.py:_add_subgraph`
   genuinely compiles a `Construct`-as-item as a nested subgraph (`compile()`
   recurses, producing its own `StateGraph`, wrapped by `make_subgraph_fn`) —
   so "neograph's compiled graph is flat" is too broad a claim on its own.
   The precise, narrower invariant that actually makes `Command.PARENT`
   unnecessary here: **Portal mesh membership requires every peer to be a
   contiguous sibling within one `Construct` level** (`_check_portal_mesh`),
   and each `Construct` level compiles to exactly one `StateGraph` object
   that owns the entire mesh's wiring. So a peer's `Command(goto=...)` never
   needs to escape a nesting boundary, even when the whole mesh is itself
   nested inside an outer sub-construct (Construct-as-mesh-member, tracked
   separately as C1) — the mesh's own members are always siblings of each
   other in the ONE `StateGraph` that Construct level produces, regardless of
   how deep that level itself is nested. This is architecturally different
   from `langgraph-swarm`, where `create_swarm(agents=[...])` compiles **each
   agent as its own `CompiledStateGraph` node** nested inside a swarm-level
   parent graph — that nesting is exactly why the reference implementation's
   `create_handoff_tool` needs `graph=Command.PARENT`: the handoff tool's
   `Command` is returned from inside an agent's own compiled subgraph and
   must escape one level to reach the swarm's top-level router.

   Grep-confirmed: every existing `Command(goto=...)` construction in
   `factory.py` (`_portal_route_to_command`, `make_portal_approval_fn`, the
   dispatch route-to-error path) omits `graph=` entirely — because neograph's
   flat compilation already puts the goto target in the SAME graph the
   returning node lives in. The new tool-triggered handoff `Command` is no
   different: `{node}__tools` and every peer's `{peer}__agent` are siblings
   in one flat `StateGraph`, so a plain `Command(goto=<peer>__agent)` is both
   necessary and sufficient. Passing `graph=Command.PARENT` here would be
   actively wrong — there is no parent graph to escape to; LangGraph would
   either no-op or error looking for a parent context that does not exist for
   a top-level node.

**Conclusion**: the reference implementation's `graph=Command.PARENT` is an
artifact of *its* nested-per-agent-subgraph compilation strategy, not an
essential part of the handoff mechanism. Neograph's flat compilation makes
the plain, already-used `Command(goto=...)` form sufficient — no compiler
change, no new `Command` argument, no nested-subgraph work required anywhere
in this design.

---

## 3. Handoff tool synthesis, binding, and the G1 monopoly

**No new `Tool` instances are added to `Node.tools`, and no new IR field is
needed.** `Portal.to` already fully determines the peer set statically
(exactly the same list the existing typed-`goto` mode reads). Synthesis is a
**pure, ephemeral runtime-layer computation** — never persisted, never
round-tripped through the IR, never touching `_ir_normalize.py`'s
single-writer fields. That claim is about the IR; it is NOT a claim that no
new plumbing is needed in `_agent_cycle.py` — it is. Verified directly: the
bound-tool list the model actually sees (`_agent_caller`'s `active`,
`_agent_cycle.py:240-249`) is built by iterating `node.tools` alone, with
`make_agent_cycle_bodies`/`_build_turn_prep`/`_abuild_turn_prep`/
`_turn_prep_kwargs` carrying no `Portal` parameter anywhere today. A
synthesized tool that is never added to `node.tools` and never threaded into
this chain would never reach `llm.bind_tools(...)` — the model could never
actually call it. §3.1 below specifies the concrete signature change that
closes this; it is real, cross-cutting plumbing through ~4-5 functions, not
a detail to defer.

### 3.1 Where synthesis happens, and how it reaches `bind_tools`

`_add_portal_agent_cycle_member` (`_wiring.py:1329`) already receives
`portal: Portal` and is the natural point to compute the peer set once. It
passes a new keyword, `handoff_portal: Portal | None`, into
`_agent_cycle.make_agent_cycle_bodies` (only non-`None` when
`portal.is_tool_triggered`). **The real call graph needs exactly two
signature changes, not four** (verified against `_agent_cycle.py`'s actual
structure, not assumed): `_agent_caller(prep_prep, node, budget)` is a
top-level module function called directly from the `agent_body`/
`aagent_body` closures defined *inside* `make_agent_cycle_bodies` — it is a
sibling of the `_build_turn_prep`/`_abuild_turn_prep` call, not a downstream
consumer of it. `_build_turn_prep`/`_abuild_turn_prep`/`_turn_prep_kwargs`
build the `_TurnPrep` (LLM instance, rendered prompt, `tool_instances` dict)
and have nothing to do with constructing the `active` list `bind_tools`
receives — that list is built fresh, separately, inside `_agent_caller`
itself. So the actual chain is:

```
make_agent_cycle_bodies(node, ..., handoff_portal=portal_or_None)
  -> agent_body / aagent_body close over handoff_portal
    -> _agent_caller(prep_prep, node, budget, handoff_portal=...)
```

`_build_turn_prep`, `_abuild_turn_prep`, and `_turn_prep_kwargs` need ZERO
changes — they don't need `handoff_portal` and have no natural use for it.
Only `make_agent_cycle_bodies` (new parameter, closed over) and
`_agent_caller` (new parameter, used directly) change. Both call sites of
`make_agent_cycle_bodies` (`_wiring.py:1325`'s plain non-mesh path and
`factory.py:496`'s existing `trigger="output"` mesh path) pass nothing
resembling this today, so a new parameter defaulting to `None` is
zero-behavior-change for both.

Inside `_agent_caller`, a new helper `_synthesize_handoff_tools(handoff_portal: Portal) -> dict[str, Callable]`
builds one reserved-name entry, `transfer_to_{peer}`, per `peer` in
`handoff_portal.to`. Real, directly reusable precedent already exists for
this exact shape: `tool.py`'s `resource_reader`/`_build_read_blob` (lines
~379/408) already build ad hoc `StructuredTool(name=, description=,
args_schema=, func=..., coroutine=...)` instances bypassing
`register_tool_factory` entirely — the same pattern applies here.
`StructuredTool` requires at least one of `func`/`coroutine` to be non-`None`
to construct, so "no side-effecting body" precisely means "a trivial stub
body (e.g. `lambda **_: 'transfer requested'`) whose result is never
actually observed, since `_tool_call_precheck`'s new handoff branch (§3.2)
intercepts the call before `tool_instances.get(name)` is ever reached" — not
a schema with no body at all, which `StructuredTool` doesn't support. It is
never looked up in `tool_factory_lookup` (unlike every real `Tool`). Its
schema is
a zero-argument (or optionally a single free-text `reason: str` argument,
purely for observability/model self-explanation — not consumed by routing
logic) callable; invoking it is a pure signal, not a computation.
`_agent_caller`'s `active` list construction changes from
`[prep_prep.tool_instances[t.name] for t in node.tools if tracker.can_call(t.name)]`
to the same expression **plus** the unconditional inclusion of every
synthesized handoff tool (handoff tools are never gated by
`tracker.can_call` — see the budget decision below), before the combined
list is passed to `llm.bind_tools(...)`.

**Budget accounting decision (resolves §9's previously-open risk)**: handoff
tools do NOT count toward `ToolBudgetTracker`/`Tool.budget` and are never
passed through `tracker.can_call(...)`. A handoff is a control-flow action
routing to a different node's own turn, not consumable work the current
node's budget is metering — the same reasoning already applied to why the
repeat-call idempotency cache is a structural no-op for handoff tools (§3.2,
§9). Both exemptions follow from the same underlying fact: a synthesized
handoff tool is deliberately absent from `node.tools`, and every
budget/idempotency mechanism in `_agent_cycle.py` is keyed off `node.tools`
by construction — so both exemptions are free consequences of that
placement decision, not two separate things to implement.

This mirrors nothing else exactly (there is no existing "virtual tool with no
factory" precedent in the codebase — `ask_human()` is a real callable body,
not a routing signal), so it is flagged here as new surface, kept
deliberately as small as possible: a dict `{tool_name: reserved_marker}` that
`_tool_call_precheck` recognizes BEFORE it ever tries `tool_instances.get(name)`.

### 3.2 Detection in the tools superstep — no `Command` here

`_tool_call_precheck` (`_agent_cycle.py:383`) gains one more branch, checked
before the existing unparseable-args / budget / unknown-tool checks: if
`tc["name"]` matches a synthesized handoff-tool name for this node, return a
NEW discriminator, e.g. `("handoff", peer_name)`, instead of `("msg", ...)` /
`("run", ...)`. `_run_tool_calls`/`_arun_tool_calls` handle this branch by
appending the SAME kind of confirmation `ToolMessage` the reference
implementation emits ("Successfully transferred to `{peer}`") for LLM-visible
history, recording a `ToolInteraction` for observability/tool_log parity, and
setting a **transient** sentinel in the tools-body's returned dict — a
reserved key built via a new `StateKeys` builder,
`StateKeys.handoff_tool_target(field_name)` (mirrors
`StateKeys.handoff_hops`/`handoff_payload` exactly: a `neo_`-prefixed,
field-keyed builder, never an inline f-string). **`tools_body`/`atools_body`
themselves still return a plain `dict[str, Any]`** — no `Command(` is
constructed in `_agent_cycle.py`; the file's zero-`Command(` invariant
(verified by grep in the foreign-swarm-import doc) is preserved unchanged.

If two handoff tool calls occur in the same turn (multiple tool_calls in one
`AIMessage`), the FIRST one wins (processed in `tool_calls` order, matching
how `_run_tool_calls` already processes calls in order) and a
`ConfigurationError`-class runtime error is NOT raised for this v1 — the
remaining tool calls in that batch (including any additional handoff calls or
ordinary tool calls) are still answered with their own `ToolMessage`s (so the
LLM's turn is never left with an unanswered `tool_call_id`, which would break
provider contracts), but the wrapper (§3.3) only ever acts on the first
handoff target. This mirrors "topology is validated, decision-making at
runtime is the model's problem" — neograph's compile-time gate (§4) is what
prevents an illegal peer target, not a runtime referee over multiple
simultaneous handoff attempts.

### 3.3 Where the `Command` gets built — satisfying guard G1

A new factory.py sibling function, `_tool_handoff_to_command`, lives directly
next to `_portal_route_to_command` (same file, same guard-G1-compliant
region). Its signature is deliberately narrower than
`_portal_route_to_command`'s (no `payload_field`/`route_field` — there is no
typed payload to read a route off of):

```python
def _tool_handoff_to_command(
    update: dict[str, Any],
    state: BaseModel,
    *,
    handoff_target_key: str,   # StateKeys.handoff_tool_target(field)
    loopback_target: str,     # "{node}__agent"
    channel_key: str,         # StateKeys.handoff_payload(entry_field) — unchanged
    count_field: str,         # StateKeys.handoff_hops(entry_field) — unchanged
    max_hops: int,
    on_exhaust: str,
    exit_name: str,
    node_name: str,
    entry_name: str,
    target_resolve: dict[str, str] | None = None,
) -> Command:
    target = update.pop(handoff_target_key, None)
    if target is None:
        return Command(goto=loopback_target, update=update)
    if target == HANDOFF_END:
        return Command(goto=exit_name, update=update)
    resolved = (target_resolve or {}).get(target, target)
    current = adapt_state(state).get_counter(count_field)
    if current >= max_hops:
        if on_exhaust == "exit":
            return Command(goto=exit_name, update=update)
        raise ExecutionError.build(...)   # identical shape to _portal_route_to_command
    return Command(goto=resolved, update={**update, count_field: current + 1})
```

This is a genuine second `Command(`-adjacent site, same footing as
`make_portal_subgraph_fn`'s pattern (§ of the master doc): it delegates every
routing rule (hop budget, `on_exhaust`, `HANDOFF_END`, entry-label
resolution) to the same conceptual machinery `_portal_route_to_command` uses,
kept as a narrow sibling rather than shoehorning a non-typed-payload case
into `_portal_route_to_command`'s typed-payload signature (which would force
fake `payload_field`/`route_field` values through a code path built around a
Pydantic model attribute read). **Both functions stay inside `factory.py` —
no addition to guard G1's `_ALLOWED = frozenset({"factory.py", "runner.py"})`
allowlist is needed.**

A new `factory.py` builder, `make_portal_agent_cycle_tool_handoff_fn`
(directly parallel to the existing `make_portal_agent_cycle_fn`, which wraps
only `parse`), wraps `tools`/`atools` from
`_agent_cycle.make_agent_cycle_bodies` with `_tool_handoff_to_command`,
leaving `agent`/`parse` untouched for a `trigger="tool"` member (its `parse`
node is wired exactly like today's `trigger="output"` member — reachable
only via the ordinary "no tool calls left" router branch, still the
reconverging exit for the non-handoff completion path).

### 3.4 Wiring change (`_wiring.py`)

`_add_portal_agent_cycle_member` (`_wiring.py:1329`) gains a branch on
`portal.is_tool_triggered`:

- **`trigger="output"` (today, unchanged)**: exactly the current code path —
  `make_portal_agent_cycle_fn` wraps `parse`; `tools` stays a plain node with
  a static loopback edge to `{node}__agent` (`_wire_agent_cycle_body`,
  unchanged).
- **`trigger="tool"` (new)**: `make_portal_agent_cycle_tool_handoff_fn` wraps
  `tools`/`atools`; `graph.add_node(names.tools, tools_cmd_fn,
  destinations=(*peer_targets, names.agent))` REPLACES the static
  `tools -> agent` edge. **This is a neograph-authored wiring convention,
  not a LangGraph-enforced rule** — verified directly (live repro): LangGraph
  does NOT reject a node that has both a static out-edge and a
  `destinations=`-registered `Command`-returning body; it silently executes
  BOTH targets in the same superstep (confirmed: a node with `add_edge("a",
  "b")` AND a `Command(goto="c")` return ran both `b` and `c`, no error). So
  the static edge must actually be removed, not merely "would be rejected if
  left in" — leaving it in place is a silent double-execution bug, not a
  compile-time safety net. A structural guard test (`test_guards_*` or a
  wiring-level assembly test) must assert `{node}__tools` has NO static
  outgoing edge whenever `is_tool_triggered` is set, since LangGraph itself
  provides no protection here. `parse` is wired
  exactly as `trigger="output"` today (still the reconverging exit for the
  ordinary no-more-tool-calls completion path) — a tool-triggered member is
  NOT prevented from also completing normally without ever calling a handoff
  tool; both exits (parse's normal completion, tools' handoff jump) coexist,
  which is why `tools` also needs `{node}__agent` in its own `destinations=`
  (the ordinary loopback, now expressed as `Command(goto={node}__agent)`
  instead of a static edge).

No change is needed to `_add_portal_mesh`'s outer wiring (entry-label map,
mesh entry edge, exit node) — those operate at the mesh level, agnostic to
which member uses which trigger mode.

---

## 4. Compile-time topology validation

`_check_portal_mesh`/`_check_one_mesh_group` (`_validation_portal.py`)
already validates, for every Portal-modified member regardless of trigger:
peer names must be declared siblings in the same named mesh group (the
existing "Peers: every peer names a Portal-modified sibling IN THIS SAME
GROUP" rule), contiguity, uniform payload, the reserved `handoff` name
collision check, and the Operator+non-atomic-member narrowed rejection. All
of these apply **unchanged** to `trigger="tool"` members: `portal.to` is read
identically regardless of trigger, so a tool-triggered member's legal handoff
targets are gated by the EXACT SAME static check that already runs today —
this is precisely Finding 3 point 3's promised advantage over the reference
implementation (which enforces topology only by which tools got bound, with
no independent structural check): neograph's tool-triggered handoff tool
literally cannot be bound to name a peer outside the declared, validated
mesh, because the peer list it is synthesized from (`portal.to`) is the same
list `_check_portal_mesh` already validated at assembly time.

**One new, narrow rule** is needed in `_check_one_mesh_group`, alongside the
existing per-member shape checks: `trigger="tool"` requires an agent/act
member (`isinstance(member, Node) and member.mode in ("agent", "act")`) —
an atomic (scripted/think/raw) or `Construct` member has no ReAct tool-call
turn to trigger a handoff from, so `Portal(to=[...], trigger="tool")` on such
a member is a `ConstructError` at assembly time (mirrors the existing
dict-form-outputs and Operator-mode narrowed rejections' style and
placement exactly — same function, same per-member loop, same
`ConstructError.build(...)` shape).

A `should_fail` fixture (`tests/check_fixtures/should_fail/`) pins this:
`Portal(to=["b"], trigger="tool")` on a `scripted`-mode node, expecting
`# CHECK_ERROR: trigger="tool" requires an agent/act member`. A `should_pass`
sibling pins the legal case: two agent-mode Portal peers with
`trigger="tool"`.

---

## 5. `HandoffMode` mapping — import AND export

**Import (`loader.py:_reconstruct_swarm_mesh`)**: `HandoffMode.OPTIONAL` and
`HandoffMode.ALWAYS` both map to `Portal(to=peers, trigger="tool")` —
matching Finding 1's empirical result that the reference LangGraph adapter
compiles both to byte-identical graphs (one `create_handoff_tool` per
relationship, unconditionally). This is a strictly more faithful import than
today's forced typed-`goto`-payload synthesis
(`_synthesize_swarm_payload`): a foreign Swarm member imported this way now
genuinely can do local ReAct work and only OPTIONALLY call a handoff tool,
instead of being forced to emit a discriminated `goto` field on every single
turn — directly closing the "OPTIONAL turns collapse into a must-route
contract" information-loss gap the current warning documents
(`loader.py:726-733`). `HandoffMode.NEVER` keeps mapping to the existing
static/typed routing the maintainer already decided is the correct target
for it (`neograph-s7zt3.12`'s RESOLVED note) — unaffected by this design,
since a `NEVER` member is a Swarm member that never itself initiates
handoff, which after this change is naturally expressed as `to=[]` on that
member (no handoff tools get synthesized when `to` is empty) rather than a
special case.

The sharpened warning text already decided in `neograph-s7zt3.12`'s ACTION
ITEM (naming the tool-call-vs-typed-output mismatch precisely) should be
narrowed further once this capability lands: for `OPTIONAL`/`ALWAYS` Swarms
specifically, the warning about "collapsing into a must-route contract" no
longer applies (this design fixes exactly that case) — the warning should be
retargeted to fire only for `HandoffMode.NEVER`-adjacent representational
gaps (if any remain) or removed for the `OPTIONAL`/`ALWAYS` path entirely.
This is a follow-up documentation edit inside the already-tracked
`neograph-s7zt3.12` import work, not new scope here.

**Export (`_agent_spec.py`, the not-yet-built `_lower_portal_mesh_to_swarm`,
tracked separately as Phase 9 / C1 in `neograph-s7zt3.12`)**: this design
adds exactly one clean case to that already-planned lowering, it does not
solve C1 wholesale. A native `Portal(trigger="tool")` member lowers to a
`Swarm` relationship with `HandoffMode.OPTIONAL` (the more general of the two
byte-identical reference behaviors — picking `ALWAYS` would falsely imply a
stronger guarantee the reference backend does not actually enforce
differently, per Finding 1). A native `Portal(trigger="output")` member (the
capability that exists today) has **no faithful Agent Spec `Swarm` export
target** — that gap is real, pre-existing, and explicitly out of scope here;
`_lower_portal_mesh_to_swarm` must document a fail-loud or best-effort
downgrade for it exactly as it already must for other unresolved Portal
export cases (Construct-member lowering, dispatch-mode Flow-node lowering),
per the master architecture doc's existing catalogue. Do not conflate "this
design gives one new mode a clean export path" with "Portal export to Agent
Spec is solved" — it is not.

---

## 6. Single-writer discipline and structural guards

**No new `Node`-level IR field is introduced.** This is the key structural
difference from the original PORTAL exception, which needed
`handoff_param`/`handoff_channel` because entry/routing information was not
otherwise derivable from any existing field. Tool-triggered handoff needs
zero new derived-and-cached IR state: `Portal.to` (author/importer-set,
exactly like `max_hops`) is sufficient input for both the mesh validator and
the tool synthesizer; `handoff_channel`/`handoff_param` (already
single-writer, already guarded by G3) are reused unchanged for the shared
mesh payload channel and dict-form input key respectively — a tool-triggered
member still participates in the SAME shared mesh channel machinery as a
typed-output member (its `HANDOFF_END` exit and hop-budget path are
undistinguishable state-wise from today's).

The one piece of NEW state is `StateKeys.handoff_tool_target(field_name)` —
a transient, popped-before-persistence sentinel (§3.3), never surfacing in
any node's persisted output, never touching `compute_schema_fingerprint`
(it is popped out of the `update` dict inside `_tool_handoff_to_command`
before that dict becomes part of the `Command.update`, so it never reaches
the checkpointed state at all — it exists only as a same-superstep,
factory-internal signal). It must be built exclusively via this one
`StateKeys` method, per the existing `neo_`-fragment guard (no inline
f-string construction anywhere else).

**Structural guards, written failing-first, before implementation begins:**

1. **G1 unchanged, no allowlist edit** — `TestCommandConstructionMonopoly`
   (`test_guards_assembly.py`) already scans every `src/neograph/*.py` file
   for `ast.Call` nodes naming `Command`; `_tool_handoff_to_command` living
   in `factory.py` passes this guard with ZERO changes to the test or its
   allowlist. Add a companion assertion (or reuse the existing one verbatim)
   confirming `_agent_cycle.py` still has zero `Command(` call sites after
   this feature lands — the existing guard already covers this, but the
   PORTAL-rollout "Lesson" section demands re-verifying by grep, not
   assuming: run `git grep -n "Command(" src/neograph/_agent_cycle.py`
   before closing the implementation task and confirm it stays empty.
2. **New guard, `TestPortalTriggerModeRequiresAgentMode`** (new test in
   `test_guards_llm_runtime.py` or `test_validation_portal`-adjacent file):
   pins that `_check_one_mesh_group` rejects `trigger="tool"` on any
   non-agent/act member — write it failing first (no such check exists yet),
   then add the rule in `_validation_portal.py` until it passes.
3. **New guard, `TestNoNewNodeIRFieldForToolTrigger`**: an explicit assertion
   (can piggyback on the existing G3 `IR_FIELDS` frozenset test) that
   `Node`'s field set is UNCHANGED by this feature — i.e., the guard that
   currently pins `IR_FIELDS = frozenset({"fan_out_param", "oracle_gen_type",
   "handoff_param", "handoff_channel"})` (`test_guards_llm_runtime.py:1018`,
   four fields) as the complete single-writer set stays exactly that
   frozenset after this feature lands. This is the explicit, checkable
   version of §6's "zero new IR fields" claim — a reviewer should not have to
   trust the prose, the guard enforces it structurally the same way G3
   already enforces the existing four fields.
4. **Three-surface parity test** (per `AGENTS.md`'s general test-conventions
   rule): `Portal(trigger="tool")` must be exercised via all three surfaces
   this feature touches — declarative `Node(mode="agent", tools=[...]) |
   Portal(to=[...], trigger="tool")`, `@node(mode="agent", tools=[...])`
   decorated then piped with the same `Portal`, and (if a ForwardConstruct
   idiom for Portal mesh membership exists per example 27/28/29) the
   `ForwardConstruct` equivalent. Since `Portal` itself is a plain
   author/importer-set modifier (no decorator-only sourcing, unlike
   `di_inputs`), there is no surface-exemption argument available here — all
   three must be tested or the omission must be justified in the same way
   `AGENTS.md`'s parity rule requires.

---

## 7. Scope boundary — first-class authoring feature, not import-only

**Recommendation: build this as a first-class native authoring feature,
`node | Portal(to=[...], trigger="tool")`, not an import-only artifact.**
Justification:

- **It is useful with zero Agent-Spec involvement.** A neograph author
  writing a native multi-agent chatbot wants exactly this ergonomics today:
  let an LLM do local ReAct work and optionally call a `transfer_to_X` tool
  when it decides a handoff is warranted, without being forced to emit a
  discriminated `goto` field on every turn (today's `trigger="output"`
  requirement). This is a genuine DX gap independent of import/export,
  identified by the master doc's own research as real information loss
  (§5's "OPTIONAL turns collapse into a must-route contract").
- **Import-only would under-test it.** Confining the feature to
  `_reconstruct_swarm_mesh` means it is only exercised through one narrow
  reconstruction path, violating the master doc's own diagnosed anti-pattern
  (features growing ad hoc, single-call-site logic instead of a
  general, well-tested primitive). Building it as a first-class `Portal`
  mode gets full three-surface-parity coverage (§6 item 4) for free, and the
  import path becomes just one MORE caller of a general capability — exactly
  the "reuse, not new design" principle the master doc repeatedly credits as
  the actual fix for nearly every gap in this investigation.
- **It costs nothing extra to generalize.** Every piece of this design
  (§§1-4) is already general — nothing in the `Portal` field, the validator
  rule, the tool synthesis, or the `_wiring.py`/`factory.py` change is
  Agent-Spec-specific. Restricting it to import would mean adding an
  artificial gate ("only usable via `_reconstruct_swarm_mesh`") for no
  benefit.
- **Consistency with Portal's own precedent**: dispatch mode
  (`route="decide"`) was NOT built import-only despite originating from
  similar "how do we faithfully capture a foreign capability" research — it
  is documented and tested as a first-class `Portal` mode (examples 28/29).
  `trigger="tool"` should receive the same treatment.

---

## 8. What is explicitly NOT solved by this design

- Construct-as-Portal-mesh-member Agent Spec export/import (`neograph-s7zt3.12`
  C1) — separate, already-tracked work.
- Dispatch-mode Portal's Flow-node lowering (`neograph-s7zt3.12` C2) —
  separate, already-tracked work.
- `Portal(trigger="output")`'s own Agent Spec export target — remains
  unsolved; not newly broken by this design, just not fixed by it either.
- Any distinction between `HandoffMode.OPTIONAL` and `HandoffMode.ALWAYS` on
  import or export — per Finding 1, this distinction does not materially
  exist in the reference LangGraph backend, so neograph intentionally does
  not invent one.
- Multiple simultaneous handoff-tool calls in one turn beyond "first one
  wins, deterministically, in tool_call order" (§3.2) — no arbitration logic
  beyond that is specified; flagged as a candidate follow-up if it proves
  insufficient in practice, not built speculatively now.

---

## 9. Confidence and open risks for adversarial review

High confidence on: the `graph=Command.PARENT` resolution (§2, grounded
directly in installed-package source and the precise siblinghood-within-one-
`Construct`-level invariant, not an overbroad "everything is flat" claim);
the "no new Node IR field" claim (§6, grounded in `Portal.to` already being
sufficient input, verified by reading `_check_portal_mesh` and
`_ir_normalize.py`'s actual current writes) — though see §3's correction
that "no new IR field" does NOT mean "no new plumbing," a distinct claim
that was previously conflated and is now resolved concretely (§3.1's
parameter-threading chain); the `ModifierCombo`-vs-`Portal`-field placement
decision (§1, grounded in reading `is_dispatch`'s existing precedent).

Resolved in this revision (previously open, per adversarial review round 1,
`portal-tool-triggered-handoff-2026-07-27-review.md`):

- **The tool-binding path** (previously the single most important gap): §3.1
  now specifies the exact parameter-threading chain
  (`make_agent_cycle_bodies` → `_build_turn_prep`/`_abuild_turn_prep` →
  `_turn_prep_kwargs` → `_agent_caller`) and the exact change to
  `_agent_caller`'s `active` list construction. Previously, as specified,
  the synthesized tool would never reach `llm.bind_tools(...)` at all — a
  silent no-op capability. Now closed.
- **The "static-edge XOR `Command`" claim** (§3.4): corrected from "LangGraph
  forbids this" (false — live-repro'd: LangGraph silently double-executes
  both targets, does not reject) to the accurate framing — a
  neograph-authored convention LangGraph does not enforce, requiring its own
  structural guard test.
- **Budget accounting for handoff tool calls** (§3.1): decided — handoff
  tools are exempt from `ToolBudgetTracker`/`Tool.budget` entirely, for the
  same structural reason (absence from `node.tools`) that resolves the next
  item.
- **Repeat-call idempotency interaction**: verified to be a structural
  non-issue, not an open risk — `idempotent_by_tool` (`_agent_cycle.py:563`)
  is keyed exclusively off `node.tools`, which deliberately excludes the
  synthesized handoff tool, so `_idempotent_repeat_key` returns `None`
  unconditionally for a handoff call with zero new code required.

Given the above, this spec is ready for a second adversarial review round to
verify these four resolutions land correctly and don't introduce a new gap —
in particular: does the `_agent_caller` parameter-threading actually compose
cleanly with existing callers that don't pass `handoff_portal` (default
`None`, zero behavior change for every non-tool-triggered node); does the
budget/idempotency exemption reasoning actually hold once traced against the
real `ToolBudgetTracker`/`_tracker_from_budget` code, not just asserted by
analogy; and does the new structural guard test (§3.4) actually get written
in a way that would have caught the F2 double-execution bug the first
review's live repro found. These are now concrete, checkable claims, not
design reversals — none of them threaten §§1-2's core resolutions.
