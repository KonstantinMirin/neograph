# Architecture Audit: Portal compilation to LangGraph, end to end (ground truth)

Date: 2026-07-27
Scope: how `Portal` (both peer-mesh and dispatch modes) actually lowers to a
LangGraph `StateGraph`, verified against real source in `compiler.py`,
`_wiring.py`, `factory.py`, `runner.py`, `state.py`. Includes a full, verified
characterization of the known Construct-as-mesh-entry bug and everything else
`compiler.py` knows about combo→LangGraph lowering generally, as ground truth
for the rest of the architecture investigation.

All line numbers below are as of this audit (2026-07-27, `develop`).

---

## 1. Top-level compile walk (`compiler.py::compile`, lines 244–314)

`compile()` walks `construct.nodes` once, left to right, threading `prev_node`
(the name of the last-added LangGraph node) so each iteration knows what to
wire its static in-edge from. A `meshed: set[int]` (identity-keyed) is
maintained so a contiguous Portal mesh, once lowered at its entry, has its
remaining members skipped by the walk (`if id(item) in meshed: continue`,
lines 252–253).

Dispatch order per item (lines 254–314), **in this exact priority**:

1. **`isinstance(item, Node)` AND `classify_modifiers(item)[0] in (PORTAL, PORTAL_OPERATOR)`** (line 254) — Portal branch. Splits further:
   - `portal.is_dispatch` → `_add_portal_dispatch` (standalone linear node, no mesh).
   - else → `_contiguous_portal_mesh(construct.nodes, item)` collects the run, `_add_portal_mesh` lowers it once, `meshed` is updated with every member's `id()`.
2. **`isinstance(item, _BranchNode)`** → `_add_branch_to_graph` (ForwardConstruct `branch()` compiles to `_BranchNode` IR, a third sum-type case alongside Node/Construct — this is the "irreducible sum-type" `compiler.py` three-way match the CLAUDE.md layering doc calls out).
3. **`isinstance(item, Construct)`** → `_add_subgraph` (recursive `compile()` call, wrapped as one opaque LangGraph node).
4. **else** → `assert isinstance(item, Node)`; `_add_node_to_graph` (per-node modifier dispatch: Oracle/Each/EachOracle/Loop/bare/agent-cycle).

**This is the exact mechanism of the known bug**: branch 1's `isinstance(item, Node)` guard is checked *before* branch 3's `isinstance(item, Construct)` — but only for a `Node`. A Portal-modified `Construct` used as a **mesh entry** (first item in a contiguous Portal-tagged run) never matches branch 1 at all; it falls through to branch 3, `_add_subgraph`, whose own `PORTAL` match arm (lines 552–560) unconditionally raises `CompileError("Portal on a sub-construct is not supported", expected="mesh members must be sibling Nodes (D-MESH-LEVEL)")`. That message is now **stale** — see §5.

A Portal-modified `Construct` as a **non-entry** member never hits this gate at all: it is only ever reached via `meshed` (already lowered as part of the entry's contiguous run) or, if a bug caused it not to be meshed, would hit the same `_add_subgraph` fallback. Today it reliably takes the `meshed`-skip path because `_contiguous_portal_mesh` (called from a *Node* entry) already walks forward through Construct members (§2).

---

## 2. `_contiguous_portal_mesh` (`_wiring.py:702–734`)

```python
def _contiguous_portal_mesh(nodes, entry) -> list[ConstructItem]:
```

- Locates `entry` by identity in `nodes`, then walks forward from that index.
- Stops the run the moment an item's `classify_modifiers(item)[0]` is not `PORTAL`/`PORTAL_OPERATOR` (line 718), or the moment it hits a **dispatch**-mode Portal item (line 726 — a dispatch node is never absorbed into a peer mesh, matching the assembly-side collector in `_validation_portal.py`).
- Critically: **this loop's per-item type check is `classify_modifiers(item)[0]`, which is type-agnostic — it already accepts a `Construct` mid-run** (do0d9 relaxation, comment at lines 706–711). It was never restricted to `Node`.
- Final line groups `candidates` via `_group_portal_members` (the same shared grouping helper the validator and state.py use) and returns only the entry's own named group, in case a differently-named mesh sits immediately adjacent with no gap (neograph-fefar).

**Conclusion**: `_contiguous_portal_mesh` itself has zero bias toward `Node` — it is already fully Construct-aware for collecting mid-run members. The only place `Node`-only bias exists is the *caller's* gate at `compiler.py:254`, which decides whether to call this function **at all**.

---

## 3. `_add_portal_mesh` — does it already know how to handle a Construct ENTRY? (`_wiring.py:799–968`)

**Yes — fully, with zero changes needed.** Verified line by line:

- `entry = members[0]` (line 834) — untyped, no `isinstance` check. Works identically whether `members[0]` is a `Node` or a `Construct`.
- `entry_field = field_name_for(entry.name)` (835) — `.name` exists on both.
- `entry_portal = entry.modifier_set.portal` (853) — `Construct` has `.modifier_set: ModifierSet` (via the shared `Modifiable` base class, `construct.py:88,137`), so this is generic too.
- `entry_label_map` (843–848): `f"{member.name}__agent" if getattr(member, "mode", None) in ("agent", "act") else member.name` — for a `Construct` (which has no `.mode` attribute) `getattr(..., None)` safely falls through to `else member.name`, i.e. a Construct entry maps to itself. This is exactly the same fallback already exercised today for atomic Node peers.
- The per-member loop (864–956) already special-cases `isinstance(member, Construct)` at line 867 and calls `_make_portal_subgraph_member_fn` → `make_portal_subgraph_fn` — **this code path already runs today** for non-entry Construct members (proven by the passing `portal_construct_member.py` fixture and `test_portal_cross_subconstruct.py`). Nothing in this per-member loop distinguishes "member is the entry" from "member is not" — the entry is simply `members[0]`, processed by the identical per-member loop as every other member.
- The single static in-edge (958–966): `entry_target = entry_label_map[entry.name]`; `graph.add_edge(prev_node or START, entry_target)` — again generic, no Node-only assumption.

**Verdict on `_add_portal_mesh`: the fix is purely a matter of widening the caller's gate.** `_add_portal_mesh`'s internals were already written generically over `ConstructItem` (Node | Construct) from the do0d9 change — they do not assume the entry is specifically a bare Node. The function was *already* prepared to accept a Construct-led `members` list; it has simply never been invoked that way because `compiler.py:254`'s `isinstance(item, Node)` gate is the sole gatekeeper deciding whether an item enters `_contiguous_portal_mesh`/`_add_portal_mesh` at all.

---

## 4. The bug is NOT confined to compiler.py — a second, independent gap in `state.py`

While tracing `make_portal_subgraph_fn` (factory.py:363–449, the function that
would wrap a Construct-entry's boundary), its wrapper reads, at **runtime**,
from the compiled state:

```python
channel_key = StateKeys.handoff_payload(entry_field)   # factory.py:402
count_field = StateKeys.handoff_hops(entry_field)      # factory.py:403
```

These two state fields are allocated by `state.py`'s Portal state-field
builder — **but only from `nodes_only`**:

```python
# state.py:108-109
nodes_only = [n for n in construct.nodes if isinstance(n, Node)]
sub_constructs = [n for n in construct.nodes if isinstance(n, Construct)]
...
# state.py:261-265
portal_members = [
    n for n in nodes_only
    if classify_modifiers(n)[0] in (PORTAL, PORTAL_OPERATOR) and not _is_dispatch(n)
]
for _group_name, group_members in _group_portal_members(portal_members).items():
    entry = group_members[0]
    entry_field = field_name_for(entry.name)
    fields[StateKeys.handoff_hops(entry_field)] = (int, 0)
    fields[StateKeys.handoff_payload(entry_field)] = (payload | None, None)
```

`portal_members` is built by filtering **only `nodes_only`** — `sub_constructs`
is never consulted here (contrast with the Oracle/Each detection two blocks
below at state.py:213, which correctly uses `all_items = nodes_only +
sub_constructs`). Two distinct failure modes follow, both **verified from
source, not assumed**:

1. **If the mesh entry is a Construct**: `portal_members` won't contain it at
   all (it's filtered out by the `isinstance` implicit in `nodes_only`), so
   `_group_portal_members(portal_members)` groups only the *remaining* Node
   members of that mesh under the entry's group name. `group_members[0]` in
   that call would then be the **wrong** node (the first Node member after the
   real Construct entry, not the actual entry) — `entry_field` would be
   derived from the wrong node's name, and the mesh's `handoff_hops`/
   `handoff_payload` fields would be seeded under the WRONG key entirely (or,
   if no Node members remain in that named group, the group disappears from
   `portal_members` filtering entirely and **no hop-counter/channel fields are
   allocated for that mesh at all**).
2. At runtime, `make_portal_subgraph_fn`'s wrapper (factory.py:402-403) would
   then read `StateKeys.handoff_payload(entry_field)` / `handoff_hops(...)`
   computed from `entry.name` where `entry` is the *actual* Construct entry —
   a state field that was never declared on the compiled Pydantic state model.
   This is not a soft gap; it fails hard (`AttributeError`/`KeyError`-class
   failure via Pydantic model attribute access on an undeclared field) the
   first time that code path executes.

**This means the Construct-as-mesh-entry bug is a *two-module* problem, not a
one-line gate fix.** `compiler.py:254`'s gate must widen, **and** `state.py`'s
Portal field-builder (lines 246–265) must switch its member source from
`nodes_only` to `nodes_only + sub_constructs` (mirroring the `all_items`
pattern already used for Oracle/Each two blocks below it) — otherwise widening
only the compiler gate produces a *worse* failure: a Construct-entry mesh that
now reaches `_add_portal_mesh` successfully, wires all its LangGraph nodes and
edges, and then crashes at first invocation because its shared mesh-channel
state field was never declared.

`_check_one_mesh_group` in `_validation_portal.py` (§5 below) already treats
entry-as-Construct as fully legal — the assembly-time validator has **no**
Node-only restriction on the entry. So the systemic pattern the wider
investigation is tracking (9 modules independently re-deriving Portal
combo/mesh logic) shows up here concretely: `_validation_portal.py`,
`_contiguous_portal_mesh`, and `_add_portal_mesh` are all already
Construct-entry-safe (do0d9 landed correctly in three of the four sites that
needed it); `compiler.py`'s top-level gate and `state.py`'s field builder are
the two sites where the do0d9 relaxation was never propagated.

**By contrast, `runner.py` got it right independently.** `_member_hop_cost`,
`_mesh_hop_cost`, and `_portal_mesh_member_ids` (runner.py:52–156) are all
written generically over `Node | Construct` with no entry-position bias —
`_mesh_hop_cost`'s `_flush()` reads `current_run[0].modifier_set.portal`
unconditionally (works whether `current_run[0]` is a Node or Construct), and
`_portal_mesh_member_ids` explicitly special-cases a Portal-carrying Construct
member as a first-class case (lines 143–151, referencing do0d9 site 6). This
is useful corroborating evidence for the wider investigation: the
Construct-as-entry gap is not "Portal support for Constructs is fundamentally
hard" — three of five relevant modules (validator, `_contiguous_portal_mesh`,
`_add_portal_mesh`, `runner.py`) already handle it correctly; only two
(`compiler.py`'s top-level gate, `state.py`'s field-builder) lag.

---

## 5. Concrete fix-size assessment

**Small, mechanical, two-site fix** — not a redesign:

1. **`compiler.py:254`**: widen `isinstance(item, Node)` to `isinstance(item, (Node, Construct))` for the Portal-peer-mode branch specifically (the dispatch sub-branch at line 259 stays Node-only — see caveat below). Reorder is not needed since this branch already runs before the generic `elif isinstance(item, Construct)` at line 292.
2. **`state.py:261-265`**: change `portal_members` to source from `nodes_only + sub_constructs` (i.e. `all_items`, already computed at line 213 for Oracle/Each) instead of `nodes_only` alone, keeping the same `classify_modifiers(...)[0] in (PORTAL, PORTAL_OPERATOR) and not _is_dispatch(n)` filter (note `_is_dispatch` reads `n.modifier_set.portal`, which is generic and needs no change).
3. **`compiler.py:552-560`** (`_add_subgraph`'s `PORTAL` match arm) and **`state.py:202-208`** (the mirror match arm in the sub-construct state-field builder) both become genuinely unreachable once (1) is fixed for *entries*, but must stay reachable-in-principle for the narrower "Portal peer-mode as a non-contiguous/malformed Construct" defensive case — recommend leaving them as `CompileError`/fallback but updating the stale comment ("mesh members must be sibling Nodes" is no longer accurate; the real invariant is "unreachable because the top-level walk now meshes a Construct entry before this arm is reached").
4. **Dispatch-mode caveat**: Portal **dispatch** mode (`route="decide"`) is not part of this fix and should stay Node-only — it is a *bona fide* scope restriction, not a duplicated-logic gap: `_add_portal_dispatch`/`make_portal_dispatch_fn` run the node's own function body (scripted/LLM call) to *synthesize* the emitted flow spec; a `Construct` has no single body to run in that sense (it's a whole subgraph). This is unlike the peer-mesh-entry case and should not be conflated with it in the fix.
5. No changes needed to `_wiring.py` (`_contiguous_portal_mesh`, `_add_portal_mesh`), `factory.py` (`make_portal_subgraph_fn` already reads the right keys once state.py allocates them), `runner.py` (already correct), or `_validation_portal.py` (already correct).

**Estimated fix size: ~10-20 lines across 2 files** (compiler.py gate widening + state.py member-source widening), plus updating 2 stale comments/error messages. The two changes are independent of each other but both required — shipping only the compiler.py widening without the state.py widening converts a clean fail-loud `CompileError` at assembly-adjacent compile time into a runtime `AttributeError` deep inside the mesh wrapper on first invocation, which is strictly worse (silently worse, in the sense the wider investigation's north star cares about: a currently-loud failure would become either a passing-compile/crashing-at-runtime failure).

A recommended TDD order: (a) write/un-skip a `should_pass` fixture with a Construct as the mesh ENTRY (mirroring `portal_construct_member.py` but swapping which member is first), watch it fail at `_add_subgraph`'s stale CompileError; (b) widen `compiler.py:254`, watch it fail differently (state field missing) at runtime; (c) widen `state.py`'s portal_members source, watch it pass; (d) add a `tests/test_portal_cross_subconstruct.py`-style runtime test that actually invokes the compiled graph (not just compiles it) to catch the class of bug in (4) that a compile-only fixture would miss.

---

## 6. Full compile-time execution model, both Portal combos

### 6a. Peer mode (mesh) — `to=[...]`

- **IR**: `Portal(to=[...], name=?, max_hops=N (entry-only), on_exhaust=..., route="goto" (default))` set via `Node(...) | Portal(...)` or `Construct(...) | Portal(...)`.
- **Assembly validation** (`_validation_portal.py::_check_portal_mesh`): per named group — contiguity, uniform declared-output payload type across all members, peer names exist and are in-group, single connected component (D-SINGLE-MESH), entry-only `max_hops`/`on_exhaust`, route field type (`str` or `Literal[...]` subset of member-names ∪ `HANDOFF_END`), reserved `handoff` inputs key typing (Node-only — no Construct analog), Operator+Portal narrowed to atomic (scripted/think/raw) members only (agent/act and Construct members with Operator are rejected).
- **State model** (`state.py`): one shared `neo_`-prefixed hop counter (`StateKeys.handoff_hops(entry_field)`) and shared payload channel (`StateKeys.handoff_payload(entry_field)`) PER NAMED GROUP, keyed off the entry's field name — both excluded from the schema fingerprint. Each Operator-guarded member additionally gets its own `portal_proposed_target` field.
- **Compile** (`compiler.py` + `_wiring.py::_add_portal_mesh`): ONE static edge `prev → entry` (resolved through the entry-label map: agent/act members' real LangGraph entry node is `{name}__agent`, everything else maps to itself); every member is `add_node(..., destinations=(peers ∪ {exit}))` with **no static inter-member edges** — each member's wrapper returns `Command(goto=peer_or_exit, update={...})`. A single pass-through `__handoff_exit_{entry}` node is the sole re-join point the linear chain resumes from. Atomic members go through `make_portal_fn`; agent/act members through `make_portal_agent_cycle_fn`/`_add_portal_agent_cycle_member`; Construct members through `make_portal_subgraph_fn` (recursive `compile()` of the sub-construct, then its boundary wrapped as a `Command`-returning node). Operator-guarded members detour through a synthesized `{member}__approve` node.
- **Runtime routing decision**: one shared helper, `_portal_route_to_command` (factory.py:219+), used by every member kind — reads the route field off the member's declared output, validates it's a declared peer or `HANDOFF_END`, writes the shared payload channel, increments the shared hop counter (except on `HANDOFF_END`, which is free/unbudgeted), raises/redirects on `max_hops` exhaustion per `on_exhaust`. This is the SINGLE dispatch mechanism — no second, divergent routing implementation for agent/act or Construct members.
- **Recursion-limit floor**: `runner.py::_ensure_agent_recursion_limit` sums `entry.max_hops * worst_member_hop_cost` per mesh (`_mesh_hop_cost`/`_member_hop_cost`), where an agent/act member's hop cost is its own full ReAct-cycle supersteps, a Construct member's hop cost is a flat 1 (opaque boundary; interior cost is NOT folded into the parent per Q4), and everything else is 1 (2 if Operator-guarded, for the approval detour).

### 6b. Dispatch mode — `route="decide"`

- **IR**: `Portal(route="decide", spec_field=..., input_field=..., output=..., max_depth=...)` — mutually exclusive with `to=[...]` (`Portal.is_dispatch` is the single source of truth for the mode discriminator, never an inline string check).
- Never a mesh member — `_contiguous_portal_mesh` explicitly stops a contiguous run the moment it hits a dispatch node (`_wiring.py:726`), and `_validation_portal.py::_check_portal_mesh` filters dispatch nodes out of `member_positions` entirely.
- **Compile** (`_add_portal_dispatch`, `_wiring.py:971-1036`): a **plain linear node** — static `prev → node` edge in. The node's function (`make_portal_dispatch_fn`) runs the node's own body (validates + compiles + invokes the emitted flow spec) and returns either a plain state-update dict (`on_invalid='raise'`, default — no `Command` at all) or, under `on_invalid='route_to_error'`, a `Command(goto=exit_or_error_handler)` via a synthetic `__dispatch_exit_{node}` pass-through, mirroring the mesh's exit-node pattern.
- **State**: writes a separate, NON-`neo_`-prefixed `{field}_dispatch` output field (fingerprinted normally — an output-contract change on the dispatched flow's result type correctly invalidates checkpoints), distinct from the node's own plain output field (which holds the emitted spec/input model, written by the ordinary `PORTAL` arm of `_add_single_output_field`).
- Dispatch mode is legitimately Node-only by design (it runs a body to synthesize a flow), not a duplicated-logic gap — see §5 caveat.

---

## 7. Everything else `compiler.py` knows about combo→LangGraph lowering (general ground truth)

For both Node (`_add_node_to_graph`, lines 571–681) and Construct
(`_add_subgraph`, lines 444–568) items, dispatch is a single `match combo:`
over `ModifierCombo` (from `classify_modifiers`), same case set in both
functions (`assert_never` enforces exhaustiveness on both):

| Combo | Node lowering | Construct lowering |
|---|---|---|
| `EACH_ORACLE` / `_OPERATOR` | `_add_each_oracle_fused` — flat M×N `Send` topology | **rejected** — `CompileError`, "not supported on sub-constructs" |
| `ORACLE` / `_OPERATOR` | `_add_oracle_nodes` — fan-out generators + `_wire_oracle` merge barrier | same shape, wraps the compiled subgraph fn in `make_oracle_redirect_fn`/`make_oracle_merge_fn` |
| `EACH` / `_OPERATOR` | `_add_each_nodes` — fan-out + `_wire_each` barrier | same shape via `make_each_redirect_fn` |
| `LOOP` / `_OPERATOR` | `_add_loop_back_edge` — conditional back-edge router | `_add_subgraph_loop` — same shape, `_construct_loop_unwrap` instead of `_node_loop_unwrap` |
| `BARE` / `OPERATOR` | agent/act → `_add_agent_cycle` (inline ReAct multi-node cycle); else plain `add_node` + static edge | plain `add_node` + static edge |
| `PORTAL` / `PORTAL_OPERATOR` | **unreachable** per-node (mesh walk always intercepts first, line 665-673 raises defensively if the invariant ever regresses) | **rejected today** (the bug this doc characterizes) |

`Operator` is not its own combo lowering step — it "stacks": after the primary
modifier's dispatch produces `last_name`, both `_add_node_to_graph` and
`_add_subgraph` call `_add_operator_check(graph, last_name, operator, ...)`
uniformly (lines 678-679, 565-566) to splice the interrupt-check node after
whatever the primary modifier produced. This is the general
pattern for any *stackable* modifier: dispatch on the primary combo, then
apply Operator as a post-processing step over the resulting last-node-name —
Portal's Operator-approval splice (`{member}__approve`) is a special case of
the same idea, implemented inside `_add_portal_mesh` per-member rather than as
a generic post-step, because Portal's own dispatch already needs `Command`
plumbing that a generic post-step edge can't express.

Three-way top-level sum type (`Node | Construct | _BranchNode`) is the one
place `compiler.py` is allowed to `isinstance`-dispatch directly per CLAUDE.md
(the documented exception to the "no hand-rolled selector" rule) — everything
inside each arm then delegates to `classify_modifiers`/`effective_producer_type`
as the single source of truth for modifier semantics, never re-deriving them.
