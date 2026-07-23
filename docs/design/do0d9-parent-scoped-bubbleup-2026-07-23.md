# do0d9: cross-subconstruct Portal handoff via parent-scoped bubble-up

Date: 2026-07-23
Implements: `neograph-do0d9` (Portal: cross-subconstruct mesh). Design task: `neograph-670qe`.
Builds on: `docs/design/portal-addressability-2026-07-15.md` (the ratified Core
Invariant — this doc is Mechanism 2, mesh-transparent exit, extended across a
sub-construct boundary), `docs/design/keymaker-decision-log-2026-07-13.md`
(D-MESH-LEVEL, now partially falsified — see §1), `docs/design/dynamic-handoff-2026-07-13.md`
+ `docs/design/dynamic-handoff-research-2026-07-13.md` (Command.PARENT semantics,
langgraph-swarm precedent).
Spike: proven runnable, sync + async, for the ROUTING/channel/hop-budget
primitives — see §8. Independently reviewed 2026-07-23; ACCEPT-WITH-CHANGES
(the mechanism is sound; several "unchanged"/"surface-agnostic" claims in the
original draft were factually wrong and are corrected in place below, marked
"Correction (post-review, 2026-07-23)"). The spike does NOT exercise a
`Construct` as an actual Portal mesh member (see §8's revised honest-findings
list) — that integration path is unspiked and is where do0d9's real
implementation work concentrates (§3.1).

This doc establishes the neograph-native mechanism that replaces the DEAD literal
`Command(graph=Command.PARENT)` path (spiked and killed, `neograph-do0d9` notes,
2026-07-15). It writes the mechanism prescriptively; the implementation
(`neograph-do0d9`) is built AGAINST it, not re-invented.

---

## Core Invariant

A routing decision made **inside** an isolated sub-construct that names a
**parent-level Portal peer** must reach that peer as a **BOUNDARY-PORT handoff on
the parent mesh** — never by the parent graph reaching into the sub-construct's
interior, and never by the sub-construct's isolated `invoke()` mutating parent
control flow directly. The escape is carried **out as ordinary typed data** (the
sub-construct's declared-output payload, whose route field names the parent peer),
and the parent mesh routes on it via the **exact same** `Command(goto=…)` /
entry-keyed-channel Portal path a same-level handoff uses. No new lowering, no new
state-key scheme, no second validation site, and `Command(` construction stays
inside the `factory.py`/`runner.py` monopoly (guard G1).

This preserves the two invariants do0d9 must not trade away:

1. **Isolated-invoke state hygiene** — a sub-construct runs as its own top-level
   Pregel invocation (`sub_graph.invoke()` / `ainvoke()`,
   `src/neograph/_subconstruct.py:188,198`); the parent has no structural
   knowledge of the child's interior. (This is why `Command.PARENT` — a native
   subgraph-nesting feature — cannot work here; §1.)
2. **Route-to-entry-port** — the parent mesh routes to a declared peer's entry,
   never a region interior (the ratified port taxonomy,
   `portal-addressability-2026-07-15.md`).

---

## 1. What is dead, and why (do not re-litigate)

The bead title's literal mechanism — an inner node returning
`Command(graph=Command.PARENT, goto=<parent_peer>)` to bubble control up the
Pregel task stack — was spiked (2026-07-15) and **does not propagate** through
neograph's isolated `sub_graph.invoke()` sub-construct architecture:

- **Catch it** in the sub-construct wrapper → the escape is swallowed, the static
  successor edge fires, the parent peer never runs. A **silent seam** — the North
  Star explicitly forbids this.
- **Let it propagate** → an uncaught exception; the outer Pregel does not
  interpret it as routing (it is a native subgraph-nesting bubble keyed on the
  checkpoint namespace, and neograph adds the child via a `RunnableLambda` that
  calls `child.invoke()`, not via `add_node(name, compiled_child_graph)`).

`Command.PARENT` is a **native subgraph-nesting** feature; neograph deliberately
uses isolated invoke for state hygiene, so it is structurally unavailable.
**D-MESH-LEVEL's forward-compat bet** ("lower to `Command(goto)` so v2
cross-boundary needs no re-lowering") was **falsified in its premise**: the
lowering was never the blocker; the *propagation mechanism* was. The `Command(goto)`
lowering is nonetheless still exactly what we reuse — just at the parent level,
not bubbled from the child.

---

## 2. Topology in scope (and what is explicitly NOT)

**In scope (do0d9):** a parent Construct carries a Portal mesh (≥2 peers). One
peer is a **sub-construct** (a region compiled via `_add_subgraph`,
`src/neograph/compiler.py:439`). At runtime some member **inside** that
sub-construct decides to hand off to a **different parent-level peer** — not
itself, not one of the sub-construct's own local peers. do0d9 makes that decision
reach the named parent peer.

**Not in scope (deferred, per the ratified addressability doc):**

- **Shape (ii) — parent peer routing *into* a sub-construct by NAME/entry-label
  addressing.** That is Mechanism 1 (entry-label map) territory
  (`portal-addressability-2026-07-15.md`), a separate mechanism where a peer
  *chooses among a sub-construct's multiple internal entry ports by label*. do0d9
  does not add entry-label addressing.
- **Native LangGraph subgraph-as-node nesting** — rejected (§7).

**Scope clarification (post-review, 2026-07-23) — the boundary-node handoff-IN is
IN scope and is NOT shape (ii).** An architect review flagged that §2 read as
"route-INTO out of scope" while the acceptance topology needs a peer's
`Command(goto=<subconstruct>)` to land on the sub-construct boundary node and
deliver the routed payload. These are two different things: shape (ii)
(deferred) is *entry-label addressing* — picking which interior port a
name resolves to. The boundary-node **handoff-IN** do0d9 DOES cover is the
ordinary Portal delivery of the reserved `handoff` payload to a mesh member's
single entry — identical to how an atomic member receives its `handoff` input,
except the member is a sub-construct whose single entry is its boundary port
(`Construct.input`). A sub-construct has exactly one entry port, so there is no
label-choice to make and Mechanism 1 is not invoked. **This handoff-IN delivery
MUST be deterministic (site 7, §3.1)** — it is the exact locus §8 finding 5 marks
unspiked, and the review's MEDIUM finding: `make_subgraph_fn._build_sub_input`
sources the boundary input via a **blind reverse type-scan**
(`_scan_subgraph_input`, `_subconstruct.py:34-40`), which in a uniform-payload
mesh (every member output field AND the `neo_handoff_<entry>` channel hold a
`Handoff`) can feed the sub-construct the WRONG `Handoff` instance — a
wrong-but-typed input, i.e. a silent mis-route the North Star forbids. Site 7
replaces that blind scan, for a Construct mesh member, with a deterministic read
of the parent handoff channel (mirroring the atomic member's reserved-`handoff`
read, `_input_shape.py:119-123`).

**Collapsed into this design (an honest §8 spike finding):** the re-scope note on
`neograph-do0d9` separated "internal member escapes" (do0d9) from "whole-Construct
same-level mesh member" (shape iii). **The spike shows these are the same boundary
mechanism** — the parent mesh only ever sees the sub-construct's *declared-output
payload*; *where inside the sub-construct that payload's route field was set* (a
top-level node vs a nested Portal member) is a sub-construct-internal concern
invisible to the parent. So do0d9's boundary machinery **is** "a sub-construct is a
parent Portal mesh member whose declared-output payload drives parent routing." The
"internal member decides" framing describes the sub-construct's *internal
composition* and imposes **no additional boundary machinery**. §6 states the one
narrow case that DOES need an extra thread (an internal *Portal* member naming a
parent peer directly).

---

## 3. The mechanism (Mechanism 2 extended across the boundary)

Mechanism 2 (mesh-transparent exit) says: a PORT-BEARING region's **single exit
node**'s returned update is piped through Portal's routing decision
(`Command(goto=…)`), reusing the entry-keyed mesh channel. For the agent/act
region (`neograph-nnds9`) the exit is `{node}__parse`. **For a sub-construct
region the exit is the sub-construct's boundary node** — the single opaque
parent-graph node that `_add_subgraph` produces (`make_subgraph_fn`'s
`subgraph_node`/`asubgraph_node`, `src/neograph/_subconstruct.py:182,191`).

Data flow, end to end (parent entry `dispatcher` keys the mesh channel
`neo_handoff_<dispatcher_field>`):

1. **Inside the sub-construct** an internal member produces the sub-construct's
   **declared output** (`Construct.output`, `src/neograph/_subconstruct.py:160`),
   a payload whose route field (the parent Portal's `route`, e.g. `goto`) names a
   parent peer (`"specialist"`) — or `HANDOFF_END` for a clean local finish.
2. **The sub-construct terminates** its own isolated `invoke()`; the payload comes
   back as an ordinary result value, extracted by the existing reverse type-scan
   (`_scan_subgraph_output`, `src/neograph/_subconstruct.py:43`). **No parent
   state was touched by the child** — state hygiene intact.
3. **At the boundary** the sub-construct node's returned update
   (`{field_name: payload}`) is piped through the **same**
   `_portal_route_to_command` (`src/neograph/factory.py:208`) that atomic and
   agent/act members use. It: (a) reads the payload's route field; (b) validates
   the target ∈ parent peers ∪ `{HANDOFF_END}` — **fail-loud `ExecutionError`
   before the goto**, never a silent drop; (c) writes the payload onto the parent
   channel `StateKeys.handoff_payload(dispatcher_field)`
   (`src/neograph/_state_keys.py:139`); (d) enforces the parent entry's hop budget
   (§5); (e) emits `Command(goto=<parent_peer>, update={…})`.
4. **The parent mesh routes.** LangGraph delivers control to the parent peer,
   registered with `destinations=` that include the parent peers (so the goto is
   compile-time-validated, `_wiring._add_portal_mesh:818`). The peer reads its
   reserved `handoff` input from the parent channel via `node.handoff_channel`
   (`src/neograph/_input_shape.py:119-123`) — the payload bubbled from the child.

Every one of steps 3–4 is **existing, unmodified Portal machinery**. do0d9 adds
only the boundary wiring that makes a sub-construct node a Portal mesh member
whose output is piped through `_portal_route_to_command`.

### 3.1 What do0d9 actually builds (the delta)

**Correction (post-review, 2026-07-23): the mechanism is sound, but the
implementation touches more Node-typed sites than "extend one function"
suggested. Every one of these sites currently ASSUMES a mesh member is a
`Node` — do0d9's real delta is relaxing that assumption in five places, not
one:**

- A **`make_portal_subgraph_fn`** in `factory.py` (sibling of
  `make_portal_fn`/`make_portal_agent_cycle_fn`): wraps the
  `make_subgraph_fn` runnable (both sync/async twins), and pipes its returned
  update dict through the shared `_portal_route_to_command`. **This is the sole
  new `Command(`-adjacent site, and it is in `factory.py` — guard G1 satisfied**
  (§4 Q5).
- **`_check_portal_mesh` (`_validation_portal.py`)**: the blanket
  `isinstance(member, Node)` rejection (`:86-95`) becomes a validated case, AND
  the `member.outputs` access immediately below it (`:104`) must be rewritten to
  `_declared_output(member)` — see §4 Q2's correction. Two sites in one
  function, not one.
- **`_ir_normalize.py`** (guard G3, the single writer for `handoff_channel`):
  `:264` and `:280` both gate on `isinstance(item, Node)` and must be extended to
  also collect/visit a Portal-carrying Construct member — see §4 Q6's
  correction. This is genuinely new code in a guard-pinned single-writer module,
  not something "already handled."
- **`_wiring.py`**: `_contiguous_portal_mesh`'s member-run collection also gates
  on `isinstance(item, Node)` (confirmed by review) — a Construct member
  currently TERMINATES mesh collection rather than being included, so this
  function needs the same relaxation. `_add_portal_mesh`'s `members: list[Node]`
  typing and its `entry_label_map` construction (keyed off `member.mode`, which
  a Construct does not have) both need a Construct-aware branch that dispatches
  to `make_portal_subgraph_fn` instead of `make_portal_fn`/`make_node_fn`. The
  sub-construct's own `_add_subgraph` compile is threaded exactly as today
  (`compiler.py:475`), and the resolved `destinations=` entry-label-map
  convention (`_wiring.py:761`) extends the same way agent/act peers got
  `{peer}__agent` — a sub-construct parent peer maps to its boundary node name.

- **`runner.py` recursion-limit accounting (added 2026-07-23 by the do0d9
  codebase disease scan — the original 5-site list MISSED this).** The
  recursion-limit floor is computed by a SECOND, independent walk over
  `construct.nodes` that also assumes a mesh member is a `Node`:
  `_mesh_hop_cost` (`runner.py:69`, its `iter_with_arms` loop at `:98-107`)
  and `_portal_mesh_member_ids` (`runner.py:111`). When a `Construct` is a
  **mid-mesh** member, `_mesh_hop_cost`'s `isinstance(item, Construct)` branch
  (`:99-101`) **flushes** the parent contiguous run and re-costs the Construct as
  its own nested mesh — so the parent mesh's `max_hops * per_hop` budget EXCLUDES
  the sub-construct boundary hop, and the members after it (`specialist`,
  `closer`) are mis-segmented into a spurious second mesh with a fake entry. An
  under-count raises `GraphRecursionError` before the mesh reaches its graceful
  budget-exhaust edge (a loud wrong-behavior, not a silent seam, but still a
  do0d9 regression on the exact spike/example topology). **The fix is narrow
  (post-review, 2026-07-23 — the earlier "per-hop = interior cost / continuation"
  phrasing was muddled and risked an under-count):** fix ONLY the flush
  mis-segmentation at `:99-101` so a Portal-carrying `Construct` member does NOT
  break the parent contiguous run. A Construct member's per-parent-superstep cost
  is **1** (one opaque boundary node, exactly like an atomic member) — its
  interior runs as a *separate isolated Pregel invocation* that reuses the shared
  config `recursion_limit` (Q4: sub-construct-internal work contributes 0 to the
  *parent* budget), so the parent floor must NOT fold the interior worst-case in.
  The existing `:101` `isinstance(item, Construct)` recursion into `_mesh_hop_cost`
  for a *standalone nested* Construct stays; the only change is that a
  **Portal-carrying** Construct member is kept in the contiguous run (cost 1)
  instead of flushing it. Over-approximation stays safe (only ever raises the
  floor); an under-count is the bug to avoid. Also include the Construct member's
  boundary id in `_portal_mesh_member_ids` so the flat agent-cost loop
  (`runner.py:159`) does not double-count an agent/act node nested inside it.
  `runner.py` is already inside the guard-G1 allowlist, so no new `Command(`
  constraint is touched. **This is site 6.**

- **Deterministic boundary handoff-IN for a Construct mesh member (site 7, added
  2026-07-23 by architect review — the route-INTO determinism the 6-site
  bubble-OUT plan omitted).** When a peer's `Command(goto=<subconstruct>)` lands
  on the boundary node, `make_portal_subgraph_fn` wraps `make_subgraph_fn`, whose
  `_build_sub_input` sources the sub-construct's boundary input via
  `_scan_subgraph_input` — a **blind reverse type-scan over ALL parent keys**
  (`_subconstruct.py:34-40`). In a uniform-payload Portal mesh, *every* member's
  output field AND the `neo_handoff_<entry>` channel hold a `Handoff`, so the
  blind scan picks the first reversed-key-order match — NOT necessarily the routed
  channel payload. That is a wrong-but-typed input ⇒ the sub-construct runs on the
  WRONG `Handoff` ⇒ a **silent mis-route the North Star forbids**. Atomic members
  do NOT have this problem: they read their reserved `handoff` input
  deterministically from `node.handoff_channel` (`_input_shape.py:119-123`). do0d9
  must give the Construct-member boundary the SAME deterministic sourcing: when the
  boundary node is a Portal mesh member, its boundary input MUST be read from the
  parent handoff channel (`StateKeys.handoff_payload(entry_field)`), not the blind
  type-scan. The Construct's `.input` must therefore equal the mesh's uniform
  payload type (validated at assembly by the extended `_check_portal_mesh`, §4 Q2).
  This is the exact locus `§8` finding 5 admits is unspiked — it MUST be built and
  proven end-to-end (an actual `Construct` object in a `Portal` member list,
  routed to, running on the routed payload) before do0d9 closes. `make_subgraph_fn`
  itself stays a general blind-scan for non-mesh sub-constructs; the deterministic
  read is applied by `make_portal_subgraph_fn` at the mesh-member boundary (so the
  change is scoped to the Portal path, not all sub-constructs).

**`make_portal_subgraph_fn` field-name note (LOW, review):** it keys the returned
payload off `make_subgraph_fn`'s update dict — `{field_name_for(sub.name):
payload}` — NOT `node.outputs` (a `Construct` has no `.outputs`; `make_portal_fn`'s
`payload_field = primary_output_field(field_name, node.outputs)` at `factory.py:159`
has no Construct analog). Read the payload back through `_declared_output(member)`
(the same fix as the `_validation_portal.py:104` correction).

**Site 3 (`_ir_normalize`) is TDD-GATED (LOW, review): do NOT edit the G3
single-writer speculatively.** For the in-scope Node-entry topology, `portal_members[0]`
(entry channel) is a Node, so a Construct member's absence from `portal_members`
does not change the entry computation; `peer_field_names` (`:260`) already collects
declared-output fields for ALL items un-gated; and the Construct's own channel
arrives via the `_add_portal_mesh` closure, not a stamped `.handoff_channel`
(Construct has no such field). So it is plausible site 3 needs **no** edit. Build
the end-to-end test first; touch `_ir_normalize` ONLY if the green path demands it.

None of this changes the MECHANISM (§3's data flow is correct and
spike-confirmed for the routing/channel/hop-budget primitives) — it changes the
honest size of the implementation task. An implementer following only the
original one-bullet framing would hit `AttributeError`s in at least two of these
sites on the first attempt, a `GraphRecursionError` on the sixth, and — most
dangerously — a **silent mis-route** on the seventh (the blind-type-scan boundary
input). Seven sites total: (1) `make_portal_subgraph_fn`, (2) `_check_portal_mesh`
×2, (3) `_ir_normalize` (TDD-gated — may be a no-op), (4) `_wiring`, (5) fixtures +
example, (6) `runner.py` hop-cost flush fix, (7) deterministic channel-sourced
boundary handoff-IN. Sites 1–5 are `AttributeError`-prone wiring; 6 is a loud
recursion mis-count; 7 is the only silent-seam-prone one and is the acceptance's
real unspiked locus (§8 finding 5).

---

## 4. Resolved questions (the §670qe checklist)

### Q1 — Exit-node identification

The sub-construct region's **exit node is its single boundary node** — the opaque
parent-graph node `_add_subgraph` produces (`make_subgraph_fn`,
`_subconstruct.py:57`). Its returned update carries the sub-construct's declared
output (`Construct.output`). "Normal completion" vs "escape to a parent peer" is
**not** a second signal — both are the *same* returned payload; the payload's
route field discriminates: `HANDOFF_END` ⇒ clean mesh exit (route to the
pass-through `__handoff_exit_<entry>` node, never budget-gated); a parent-peer
name ⇒ handoff. This is byte-identical to how an atomic member's payload
discriminates (`_portal_route_to_command:240-276`). There is exactly one exit node
(the Core Invariant requires singular exit for a region to be PORT-BEARING), so no
"which node" ambiguity arises for the sub-construct class — unlike the agent/act
class, which had to nominate `__parse` among three nodes.

### Q2 — Escape-target validation at assembly (reuse `_check_portal_mesh`)

Extend the **single** mesh-validation cluster `_check_portal_mesh`
(`src/neograph/_validation_portal.py:40`), never a second site. The current hard
rejection (`_validation_portal.py:86-95`, "Portal mesh member is a Construct")
becomes a **validated case**: a Construct member is admitted iff its declared
`.output` (`_declared_output`) **is** the mesh's uniform payload type — the exact
uniform-payload rule already applied to Node members (`_validation_portal.py:134-143`).

**Correction (post-review, 2026-07-23): "unchanged below" does NOT hold as
written for two of the existing checks — they access Node-only fields and would
`AttributeError` on a Construct the moment the blanket `isinstance(member, Node)`
rejection above them is relaxed:**

- `_validation_portal.py:104` — `normalize_outputs(member.outputs).is_dict_form`
  reads `member.outputs` (plural). `Construct` has no `.outputs` field (only the
  singular `.output`, per AGENTS.md's inputs/outputs vs input/output split) — this
  MUST be rewritten to read `_declared_output(member)` (the monopolized selector,
  `_normalize.py`) so it works for both Node and Construct members.
- `_validation_portal.py:241-252` (the reserved-`handoff`-key check) already
  iterates `node_members` — a list narrowed via `cast("list[Node]", members)`
  **after** the shape-check loop (`:113-114`, "Past the shape checks every member
  is a Portal-modified Node — narrow the list so the rules below read the typed
  surface"). A Construct member therefore does **not** crash here — it is
  currently **excluded** from `node_members` and so silently SKIPS this check.
  do0d9 must decide explicitly: either a Construct member's boundary node also
  needs a `handoff`-key-typing check reachable some other way (its own inputs
  aren't meaningful — the check is about a NODE'S declared `inputs={"handoff":
  T}`, which has no Construct analog since a sub-construct's `.input` is its
  boundary port, not a fan-in dict), or explicitly document that this check is
  Node-only by design and does not apply to a Construct member (the latter is
  correct: a Construct's boundary port is `Construct.input`, already typed and
  validated by `_add_subgraph`'s own boundary check — this specific rule has no
  Construct analog and should stay Node-scoped, not be extended).

Peer existence + peer-is-a-member (`:146-165`), single-mesh connectivity
(`:167-190`), entry-only `max_hops`/`on_exhaust` (`:192-203`), and
route-field-is-str/Literal on the payload (`:206-237`) all operate on
`member.name`/the declared payload type via `_declared_output`, not Node-only
fields, and DO apply to a Construct member unchanged. Because the parent peer
names a **declared parent mesh member**, an escape to a non-existent /
non-member parent target is a **`ConstructError` at `Construct(...)` assembly** —
the "invalid target unrepresentable" guarantee, identical to same-level peers.

The narrow §6 case (an *internal Portal member* naming a parent peer directly)
additionally validates the parent "escapable peer set" the sub-construct is
compiled with — still inside `_check_portal_mesh`, recursing into the sub-construct
as it already does (`_validation_portal.py:52` "recurses into sub-constructs").
That recursive case fails LOUD at assembly today even without do0d9's changes:
naming a parent peer from inside the sub-construct raises `ConstructError`
"names peer '<x>' which does not exist" (peer-existence check, `:146-165`,
scoped to sub-level siblings) — so deferring §6 leaves no silent gap, only a
(slightly misleading, since the peer DOES exist one level up) error message
worth a follow-up polish, not a blocking risk.

### Q3 — Parent mesh re-activation after a bubbled handoff

There is **no separate "re-activation of the entry"** — that framing (from the
bead) is superseded by the spike. The boundary node **is** a parent mesh member,
so its `Command(goto=<parent_peer>)` routes **directly to the target peer**, not
back through the entry. The parent entry re-activates **only** if the target peer
itself routes back to it (a genuine cycle, budget-gated by `max_hops`), exactly as
any same-level mesh. The mechanic that makes the target peer *see* the bubbled
decision is the shared channel: `_portal_route_to_command` writes
`channel_key: payload` into the `Command.update` (`factory.py:252,264,275`), and
the target peer reads its reserved `handoff` input from `node.handoff_channel`
(`_input_shape.py:119-123`). **Spike-confirmed** (§8): `specialist` read
`handoff.subject` successfully — it could only have come from the parent channel
written by the boundary node's `Command`.

### Q4 — Recursion / hop-budget interaction across the boundary

**Each mesh level owns its own entry-keyed hop counter**
(`StateKeys.handoff_hops(entry_field)`, `_state_keys.py:130`). Consequences,
all clean:

- The **parent** counter is `neo_handoff_hops_<parent_entry>`. A boundary-node
  handoff to a parent peer increments it, checked BEFORE emitting the goto
  (`count >= max_hops`, `factory.py:260`), naming the parent entry on exhaust —
  identical to an atomic member. The parent budget bounds **parent-level** hops
  (including the sub-construct→parent-peer hop).
- The **sub-construct's** internal execution is a **separate isolated Pregel
  invocation with a separate state**; the parent counter lives on the *parent*
  state the child never receives. So **sub-construct-internal work contributes 0
  to the parent budget**, and if the sub-construct has its own internal Portal
  mesh, that mesh has its OWN `neo_handoff_hops_<sub_entry>` counter and its OWN
  `max_hops`, bounded independently. Nesting composes: N levels ⇒ N independent
  budgets, each fail-loud on its own exhaust. This is a *feature* — an infinite
  loop at any level is bounded at that level, and no level can silently consume
  another's budget. Spike: parent did 2 hops (dispatcher→worker, worker→specialist)
  under `max_hops=8`; the sub-construct had no mesh, 0 internal hops.

### Q5 — Command construction placement (guard G1)

All `Command(` construction stays in `factory.py`/`runner.py`
(`TestCommandConstructionMonopoly`, `tests/test_guards_assembly.py:84`). The new
site is **`factory.make_portal_subgraph_fn`**, which delegates the actual `Command`
build to the already-existing `_portal_route_to_command` (`factory.py:208`) — so in
practice **no new `Command(` literal is added at all**; do0d9 reuses the one that
already lives in `factory.py`. `_subconstruct.py` and `_wiring.py` **never**
construct a `Command` (they stay as they are: `_subconstruct.py` returns a plain
dual-path `RunnableLambda`; `_wiring.py` calls the factory and does `add_node(...,
destinations=…)`). Extend the guard's allowlist only if a genuinely new literal is
introduced; the design deliberately avoids one.

### Q6 — Three-surface parity

The mechanism operates on the **IR** (`Construct` as a mesh member + `Portal`
modifier + `Construct.output`), and all three surfaces converge on that IR before
the compiler/normalizer runs — but **correction (post-review, 2026-07-23): this is
not "already handled," it is new, G3-guarded work**:

- **Declarative** `Construct(nodes=[Construct(..., output=Handoff) | Portal(to=[...]), ...])`
  — direct.
- **Programmatic** `sub | Portal(to=[...])` — the pipe attaches the modifier to the
  sub-construct; same IR.
- **`@node` / `construct_from_functions`** — a sub-construct built from `@node`
  functions with `output=` (as the spike does) carries the same `Construct.output`;
  the parent attaches `Portal`.

**The `handoff_channel`/`handoff_param` single-writer (`_ir_normalize.py`,
guard G3) does NOT currently visit a Construct member — it explicitly gates on
`isinstance(item, Node)` in two places**:

- `_ir_normalize.py:264` — `if isinstance(item, Node) and item.modifier_set.portal
  is not None: portal_members.append(item)`. A Portal-carrying Construct member is
  **excluded** from `portal_members`, so it never contributes to the
  entry-detection / `handoff_channel` computation.
- `_ir_normalize.py:280` — `if not isinstance(item, Node): continue` in the
  write-back loop, so a Construct member never gets any normalizer applied.

do0d9 must **deliberately extend this single-writer** (not bypass it) so a
Construct member: (a) is included in the mesh-membership scan the entry
computation walks, and (b) has SOME field to carry the resolved
`handoff_channel` key at runtime, since `Construct` has no `handoff_channel`
attribute the way `Node` does — the boundary node's runtime wrapper
(`make_portal_subgraph_fn`, §3.1) needs this key threaded to it some other way
(the same `_add_subgraph` recursive-`compile()`-kwarg channel used for
`checkpointer`/`runtime`/`scripted_lookup`, §6, is the natural candidate — NOT a
new field on `Construct` itself, which would violate the single-writer
discipline by adding a second place `handoff_channel`-like state lives).

**Scope note**: in the topology this doc's spike and the do0d9 acceptance
criteria describe, the mesh **entry is always a Node** (`dispatcher`), so the
entry-detection computation (`portal_members[0].name`) is unaffected by a
Construct member elsewhere in the SAME mesh. A Construct AS THE ENTRY is out of
scope for v1 — flag it explicitly as unsupported (fail loud, not silently wrong)
until a real need arises.

**One principled exemption — `ForwardConstruct` (D-FORWARD-EXEMPT precedent).**
ForwardConstruct is already exempt from Portal mesh membership (no static dataflow
for a runtime mesh; `portal-addressability-2026-07-15.md`, decision log
D-FORWARD-EXEMPT); a `self.handoff(...)` builder is the fast-follow
(`neograph-a37vk`). do0d9 inherits that exemption unchanged: a ForwardConstruct
sub-construct is not a bubble-up source until a37vk gives it a builder. This is a
*stated structural reason*, not parity-by-omission.

### Q7 — Sync/async parity

`make_subgraph_fn` is already a driver-selected dual path
(`RunnableLambda(subgraph_node, afunc=asubgraph_node)`, `_subconstruct.py:207`);
the two twins share `_build_sub_input`/`_build_update` so they cannot drift. The
new `make_portal_subgraph_fn` wraps BOTH twins and pipes each through the sync/async
pair already present in `_portal_route_to_command`'s callers
(`portal_wrapper`/`aportal_wrapper`, `factory.py:170,187`). **Spike-confirmed for
the isolated-invoke-and-lift primitive** (§8): `run` and `arun` produce identical
trails, and the async worker genuinely awaits the child's `arun` (verified in
the spike source, not just claimed) so async selection propagates into the
child (the Phase-1 H2 invariant, `_subconstruct.py:191-199`). `make_portal_subgraph_fn`'s
own async twin does not exist yet (§8 finding 5) — the claim is scoped to the
primitive it wraps, not to code not yet written.

---

## 5. Worked control/state trace (the spike topology)

Parent mesh entry `dispatcher` (`max_hops=8`) → peers `worker` (wraps the
sub-construct), `specialist`, `closer`. Sub-construct `resolver_sub`
(`output=Handoff`): `sub_intake` → `sub_decide` (the internal decider).

Escalation ticket:

```
dispatcher            emits Handoff(goto="worker")            hop 1  → goto worker
worker  (boundary)    run(resolver_sub) in isolation:
                        sub_intake  → SubDecision(kind="escalate")
                        sub_decide  → Handoff(goto="specialist")   [decision made INSIDE the child]
                      returns that Handoff; _portal_route_to_command:
                        writes neo_handoff_<dispatcher> = payload
                        hop 2  → Command(goto="specialist")
specialist            reads `handoff` from neo_handoff_<dispatcher>  [the bubbled payload]
                      → Handoff(goto=HANDOFF_END, resolution=...)  → __handoff_exit_dispatcher
```

Final trail: `dispatcher → worker → sub_intake → sub_decide → specialist`. The
decision made by `sub_decide` **inside** the isolated child reached the parent peer
`specialist`, routed by ordinary parent Portal machinery. Locally-resolvable
ticket: `sub_decide` returns `goto=HANDOFF_END`; `worker` re-emits it; the mesh
exits cleanly with no escape.

---

## 6. The one case that needs an extra thread (documented, not required for v1)

If a user wants an **internal Portal member** of the sub-construct to name a
parent peer **directly as its route target** (rather than the sub-construct's
top-level declared-output payload carrying it), that internal member's
`make_portal_fn` would reject the parent-peer name (not a local peer,
`factory.py:242-249`). Supporting it requires threading a parent **"escapable peer
set"** down through `_add_subgraph`'s recursive `compile()`
(`compiler.py:475`) — the SAME threading channel as `checkpointer`/`runtime`/
`scripted_lookup`, not a new mechanism — so the internal member's `valid_targets`
becomes `local_peers ∪ escapable_parent_peers ∪ {HANDOFF_END}`, and the escaping
member routes to the sub-local exit while carrying the parent target out on the
declared output. **This is an additive enhancement, not the core mechanism** — the
spike proves the core works with the internal decider as a plain node. Recommend
filing it as a do0d9 fast-follow only if a real consumer (piarch) needs
inner-Portal-to-parent-peer directness; otherwise the top-level-output-payload form
(spike §8) covers the requirement.

---

## 7. Rejected alternatives

### R1 — Literal `Command(graph=Command.PARENT, goto=<parent_peer>)` (DEAD)

Spiked 2026-07-15, does not propagate through isolated `sub_graph.invoke()`;
catching it is a silent seam, letting it propagate crashes. §1. This is the
mechanism the bead title named; it is off the table.

### R2 — Switch sub-construct compilation to native LangGraph subgraph-as-node nesting
(`parent_builder.add_node(name, compiled_child_graph)`)

This WOULD make `Command.PARENT` work natively. **Rejected**: it trades away the
**state-isolated boundary-port invariant** — the single most load-bearing property
of neograph's sub-construct model — for one feature. Native nesting shares the
parent's state channels into the child (that is *how* the bubble finds its way
out), which re-admits exactly the cross-level state bleed the isolated-invoke
architecture exists to prevent (`_subconstruct.py`'s typed `input`/`output` port is
the whole point). A change that buys one feature by re-admitting a silent
state-hygiene seam is a North-Star regression **even if all tests pass**. The
bubble-up design gets the same capability with **zero** loss of isolation, because
the escape crosses the boundary as ordinary typed data on the declared-output port.

### R3 — A second, bespoke cross-boundary lowering or a second validation site

Rejected by the single-sited-cluster discipline: the lowering is the existing
`Command(goto)`/`_portal_route_to_command`; the validation is the existing
`_check_portal_mesh`. do0d9 EXTENDS both, invents neither. (D-MESH-LEVEL's "no
re-lowering" intent survives even though its `Command.PARENT` premise did not.)

### R4 — Make the parent ENTRY forward the bubbled decision (route child → entry → peer)

An earlier reading of the bead ("how does the parent mesh entry re-activate")
implied the boundary writes the channel and hands control to the parent *entry*,
which then forwards to the target. **Rejected as unnecessary indirection**: it puts
routing logic on the entry (which member forwards which handoff?) and adds a hop.
The boundary node is itself a mesh member, so it routes to the target peer
**directly** (§4 Q3). The entry only participates if the target cycles back to it.

---

## 8. Spike — runnable proof (sync + async)

A real two-level neograph pipeline (public `Construct`/`Portal`/`compile`/`run`/
`arun` APIs) proving the mechanism. Runnable source (committed with this doc):
`docs/design/spikes/do0d9_bubbleup_spike.py`
(`uv run --extra dev python docs/design/spikes/do0d9_bubbleup_spike.py`).

**Topology.** Parent mesh (real neograph Portal mesh, entry `dispatcher`,
`max_hops=8`, peers `worker`/`specialist`/`closer`). `worker` is a real
`@node(portal=["specialist","closer"])` whose body invokes the compiled
sub-construct `resolver_sub` (`sub_intake → sub_decide`, `output=Handoff`) in
isolation and returns its `Handoff` payload. Because `worker` is a genuine Portal
member, **100% of the routing is neograph's own machinery** — `make_portal_fn` →
`_portal_route_to_command` → the entry-keyed channel → the reserved `handoff` input.

**Which parts are prototype vs production.** The ONLY hand-written code is
`_run_sub` (isolated `run(_sub_graph, …)` + declared-output extraction), living
inside `worker`'s body. In production do0d9 that exact logic is
`factory.make_portal_subgraph_fn` (so `Command(` stays in `factory.py`, G1), and
`resolver_sub` is a first-class parent mesh member wired by `_add_portal_mesh`
with `destinations=` + validated by the extended `_check_portal_mesh`. The routing
half — the load-bearing, silent-seam-prone half — is already real, unmodified
neograph in the spike.

**Result (all four cases pass):**

| case | trail | outcome |
|------|-------|---------|
| sync / escalate | dispatcher → worker → sub_intake → sub_decide → **specialist** | escape crossed the boundary to the parent peer |
| sync / local | dispatcher → worker → sub_intake → sub_decide | sub-construct finished locally (HANDOFF_END), no escape |
| async / escalate | dispatcher → worker → sub_intake → sub_decide → **specialist** | identical under `arun` |
| async / local | dispatcher → worker → sub_intake → sub_decide | identical under `arun` |

Asserted: the inner member `sub_decide` ran; control reached `specialist` (the
named parent peer); the resolver was the parent specialist (not a swallowed
escape); `closer` was never hit (not mis-routed); and `specialist` successfully
read `handoff.subject` from the parent channel (proving the channel write/read, not
a silent None). No uncaught crash on any path.

**Surprises / honest findings.**

1. **do0d9 and shape (iii) collapse to one boundary mechanism** (§2). The
   re-scope's separation of "internal member escapes" from "whole-Construct mesh
   member" does not survive contact with the boundary: the parent only sees the
   declared-output payload. This *simplifies* do0d9 — it is "a Construct can be a
   parent Portal mesh member," full stop, and the internal-member story is a
   sub-construct-internal detail. The implementer should build the Construct-member
   path and NOT a separate bubble-up-specific code path.
2. **"Parent entry re-activation" is a red herring** (§4 Q3 / R4). The boundary
   routes directly to the target peer.
3. **The one genuinely-extra feature** (inner *Portal* member → parent peer
   directly) needs the escapable-peer-set thread (§6) and is cleanly separable — do
   NOT let it block the core.
4. **`neo_*` channels are stripped from the returned top-level state**
   (`_strip_internals`), so a test proving the channel write must assert on the
   *consumer* (the target peer having read its `handoff`), not on the raw channel
   in the final result — the spike does exactly this.
5. **(Post-review correction) The spike proves the routing primitives, not the
   Construct-as-mesh-member integration.** The spike's `worker` is a plain
   `@node(portal=[...])` whose BODY hand-invokes the sub-construct
   (`_run_sub`/`_arun_sub`) — a shape that already works today without any do0d9
   change. It never puts an actual `Construct` object into `Portal`'s member
   list, so it does not exercise `_check_portal_mesh`'s relaxed rejection,
   `_ir_normalize`'s extended single-writer, or `_wiring`'s Construct-aware
   dispatch (§3.1) — the five sites an independent review (2026-07-23) found are
   Node-typed today. The spike is still valuable and honestly scoped (§8 already
   said "the ONLY hand-written code is `_run_sub`"): it proves the *hard, silent-
   seam-prone* half (does the channel write/read/hop-budget/goto actually work
   across an isolated-invoke boundary) is real neograph machinery. It does not
   yet prove the *wiring* half (can a bare `Construct` legally sit in a
   `Portal`'s member list end to end) — that remains to be spiked or covered by
   the do0d9 implementation's own tests before closing the task.

---

## 9. Bounding the claim

do0d9 makes an inside-the-sub-construct escape to a parent peer **routable and
compile-time-validated** (invalid parent target ⇒ `ConstructError` at assembly;
invalid at runtime ⇒ fail-loud `ExecutionError` before the goto). It does **not**
make the mesh terminate — that is `max_hops`, per-level, bounding the
*nonterminating-but-legally-routed* class (§4 Q4). And it does **not** cover a
semantically-wrong-but-valid escape target (routing to `closer` when `specialist`
was meant) — that is LLM/logic correctness, outside neograph's guarantee, exactly
as the port taxonomy bounds it (`portal-addressability-2026-07-15.md`,
"Bounding the claim"). Addressability across the boundary is preserved; termination
and semantic correctness remain the caller's, bounded where they were before.
