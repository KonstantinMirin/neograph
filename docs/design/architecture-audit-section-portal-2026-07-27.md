# Portal architecture (dynamic-handoff mesh + dispatch)

*Section prepared for the master architecture-audit document. Reconciled from
`architecture-audit-portal-ir-2026-07-27.md`, `architecture-audit-portal-compilation-2026-07-27.md`,
`architecture-audit-modifiercombo-consumers-2026-07-27.md`, and
`architecture-audit-crosscheck-portal-2026-07-27.md` — all four verified
against source at the time of writing (2026-07-27, `develop`). All findings
below are traced to file:line, not re-derived from memory.*

## 1. What Portal is

`Portal` (formerly `Keymaker`) is neograph's runtime peer-routing mechanism —
the sanctioned second "genuinely new IR capability" alongside `_BranchNode`
(CLAUDE.md's layering doctrine). It has two mutually exclusive modes,
discriminated solely by `Portal.is_dispatch` (`modifiers.py`, no inline
`route == "decide"` string check anywhere else — pinned by
`TestPortalDispatchDiscriminationMonopoly`, `tests/test_guards_assembly.py:142-187`):

- **Peer mode** (`to=[...]`) — a mesh of members that route control to each
  other at runtime via `Command(goto=...)`, entered through one static edge at
  the mesh's `entry` (`members[0]`), exited through a synthesized
  `__handoff_exit_{entry}` pass-through node.
- **Dispatch mode** (`route="decide"`) — a standalone linear node whose body
  synthesizes and validates an emitted flow spec; never a mesh member, never
  absorbed into a contiguous Portal run.

## 2. IR model

| Field/marker | Type | Sole writer | Purpose |
|---|---|---|---|
| `Node.handoff_param` | `str \| None` | `_ir_normalize.py`'s `_HandoffParamNormalizer` — writes literal `"handoff"` when the node has dict-form `inputs` containing a `"handoff"` key and `modifier_set.portal is not None` | Reserved fan-in input key sibling to `fan_out_param` |
| `Node.handoff_channel` | `str \| None` | `_ir_normalize.py`'s `normalize_ir` main loop (inlined, not a per-node normalizer, because it needs construct-level visibility across the whole mesh) | Names the shared `neo_`-prefixed state channel a mesh group writes to, keyed off the entry's field name via `StateKeys.handoff_payload(...)` |
| `ModifierCombo.PORTAL` / `PORTAL_OPERATOR` | enum | `modifiers.py`'s `_COMBO_MAP` | The dispatch key every consumer (should) switch on |
| `HANDOFF_END = "__end__"` | constant | `modifiers.py` | Valid route-Literal target meaning "leave the mesh" |
| `DISPATCH_ROUTE = "decide"` | constant | `modifiers.py` | Backs `Portal.is_dispatch`, the sole mode discriminator |
| `StateKeys.handoff_payload(...)` / `handoff_hops(...)` | `neo_*` state keys | `_ir_normalize.py` (payload key), `_wiring.py` (hop-counter key, Operator-approval splice) | Shared per-group runtime channel; excluded from checkpoint schema fingerprinting |

**No `Construct`-level `handoff_param`/`handoff_channel` field exists, by
deliberate design** (do0d9 doc, Q6): a Construct mesh member's channel key is
threaded through the recursive `compile()` kwarg closure, not stamped as a
field — adding one would create a second place the same state lives, violating
the single-writer discipline. Consequently, two `isinstance(item, Node)` gates
in `_ir_normalize.py` (member collection at one site, write-back loop at
another) deliberately skip non-`Node` items — correct today because entry is
assumed to always be a Node (see §5).

## 3. Assembly-time validation (`_validation_portal.py`)

Entry point: `_check_portal_mesh` then `_check_portal_dispatch_error_handler`,
both called once per construct level inside the same recursive walk
`_construct_validation.py` already uses for everything else.

`_check_portal_mesh`:
1. Collects PEER-mode members only (dispatch nodes excluded — they're
   standalone).
2. Rejects a sibling literally named `"handoff"` (reserved-key collision),
   construct-wide.
3. Groups PEER members by `Portal.name` via `_group_portal_members`
   (`modifiers.py`) — **the same grouping helper used by `_ir_normalize.py`,
   `_wiring.py`'s `_contiguous_portal_mesh`, and `state.py`**, a genuine single
   source of truth for "which members form which named mesh."
4. Per group (`_check_one_mesh_group`), independently: dict-form-outputs
   rejected (Node-only, a Construct has no `.outputs`); **a `Construct` member
   is explicitly admitted** as a first-class mesh member (the do0d9 relaxation,
   quoted verbatim in the source: its `_declared_output` must equal the uniform
   mesh payload type, exactly like a Node); Operator-gated members must be
   atomic (scripted/think/raw) — agent/act or Construct members with Operator
   are rejected; contiguity within the group; uniform declared-output payload
   type (`is`-identity) across all members; every `to` target must name a
   sibling in the same group; the group must be one connected component
   (`D-SINGLE-MESH`); `max_hops`/`on_exhaust` are entry-only; the payload's
   route field must be `str` or a `Literal[...]` subset of member names ∪
   `HANDOFF_END`; the reserved `handoff` input's type is checked **Node-only**
   (a Construct's boundary is a single `.input`, no reserved-key analog — an
   explicit, considered-and-rejected extension per the do0d9 doc, not an
   oversight).

**Critical fact, verified by reading the whole function body**: nothing in
`_check_portal_mesh`/`_check_one_mesh_group` singles out the entry's type.
Every rule applies identically to `members[0]` as to any other member. **The
validator does not, and never did, reject a Construct as a mesh entry** — this
directly contradicts the do0d9 design doc's own stated intent ("Construct as
entry is out of scope for v1 — flag it explicitly as unsupported... until a
real need arises") and the stale compiler.py comment claiming this case is
"already rejected at assembly." Design intent and shipped validator behavior
silently diverged.

## 4. Compile-to-LangGraph model

`compiler.py::compile()`'s top-level walk (lines 244–314) dispatches each
construct item in this priority order, maintaining a `meshed: set[int]`
(identity-keyed) so a contiguous mesh's later members are skipped once its
entry has been lowered:

1. `isinstance(item, Node)` AND `classify_modifiers(item)[0] in (PORTAL,
   PORTAL_OPERATOR)` → Portal branch. `is_dispatch` → `_add_portal_dispatch`
   (linear node). Else → `_contiguous_portal_mesh` collects the run,
   `_add_portal_mesh` lowers it once.
2. `isinstance(item, _BranchNode)` → `_add_branch_to_graph`.
3. `isinstance(item, Construct)` → `_add_subgraph` (recursive `compile()`).
4. else → `_add_node_to_graph` (Oracle/Each/Loop/bare/agent-cycle dispatch).

**`_contiguous_portal_mesh` (`_wiring.py:702-734`) is already fully
Construct-agnostic**: its per-item type check is `classify_modifiers(item)[0]`,
which accepts a `Construct` mid-run with zero bias (the do0d9 relaxation). It
stops a run only when an item's combo isn't PORTAL/PORTAL_OPERATOR, or when it
hits a dispatch-mode item.

**`_add_portal_mesh` (`_wiring.py:799-968`) is already fully generic over
`entry = members[0]` being a `Node` or a `Construct`**, verified line by line:
`.name`, `.modifier_set.portal` work identically (via the shared `Modifiable`
base); the entry-label map's `getattr(member, "mode", None)` safely falls
through to `member.name` for a Construct (no `.mode` attribute); the per-member
loop already special-cases `isinstance(member, Construct)` and wraps it via
`make_portal_subgraph_fn` — proven live today by the passing
`portal_construct_member.py` fixture and `test_portal_cross_subconstruct.py`
for **non-entry** Construct members. Nothing distinguishes entry-position from
any other position in this function.

**Runtime routing is one shared mechanism for every member kind**:
`_portal_route_to_command` (`factory.py`) reads the route field off the
member's declared output, validates it against declared peers ∪
`HANDOFF_END`, writes the shared payload channel, increments the shared hop
counter (except on `HANDOFF_END`, unbudgeted), and enforces `max_hops`/
`on_exhaust`. Atomic members route through `make_portal_fn`, agent/act members
through `make_portal_agent_cycle_fn`, Construct members through
`make_portal_subgraph_fn` — no second, divergent routing implementation per
member kind.

**Recursion-limit floor** (`runner.py::_ensure_agent_recursion_limit`) sums
`entry.max_hops * worst_member_hop_cost` per mesh; a Construct member's hop
cost is a flat 1 (opaque boundary, interior cost not folded into the parent);
this code (`_member_hop_cost`/`_mesh_hop_cost`/`_portal_mesh_member_ids`) is
**already fully generic over Node | Construct**, with an explicit Construct
special-case for member-id collection.

**Dispatch mode** (`_add_portal_dispatch`, `_wiring.py:971-1036`) compiles to a
plain linear node with one static in-edge; its function runs the node's own
body to synthesize/validate the emitted flow, writing a separate
non-`neo_`-prefixed `{field}_dispatch` output (correctly fingerprinted — an
output-contract change on the dispatched flow invalidates checkpoints). This
mode is legitimately Node-only by design: it runs a body to synthesize a flow,
and a Construct has no single body to run in that sense. This is a bona fide
scope boundary, not a duplicated-logic gap, and requires no change.

## 5. The Construct-as-mesh-entry bug — exact mechanism and fix

**Root cause is two independent sites, not one**, both required to fix
together:

**Site 1 — `compiler.py:254`.** The Portal branch's gate is
`isinstance(item, Node)`. A Portal-modified `Construct` used as a mesh entry
fails this test and falls through to branch 3 (`isinstance(item, Construct)`
→ `_add_subgraph`). Inside `_add_subgraph`'s own `match combo:` block, the
`PORTAL`/`PORTAL_OPERATOR` arm (`compiler.py:552-560`) unconditionally raises
`CompileError("Portal on a sub-construct is not supported", ...)` with a
comment claiming this is "already rejected at assembly — defense-in-depth."
**That comment is false as the code stands**: nothing in
`_validation_portal.py` rejects a Construct-as-entry mesh (§3). The comment's
"defense-in-depth" framing is only accurate for a Construct carrying Portal
that is *not* reachable via the `isinstance(item, Node)` gate at all in any
other way — it is not accurate for a Construct meant to be the mesh entry,
which has no assembly-time rejection whatsoever.

Because `_contiguous_portal_mesh` and `_add_portal_mesh` are already fully
generic (§4), **the only thing standing between "works" and "CompileError" is
this one `isinstance(item, Node)` gate** at the top-level walk. A Construct
entry never reaches `_add_portal_mesh`'s already-correct Construct-handling
branch because the dispatch gate misroutes it one step earlier.

**Site 2 — `state.py:261-265`, an independent bug the compiler-focused trace
alone would miss.** The Portal state-field builder computes:

```python
nodes_only = [n for n in construct.nodes if isinstance(n, Node)]
sub_constructs = [n for n in construct.nodes if isinstance(n, Construct)]
...
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

`portal_members` sources from `nodes_only` **only** — `sub_constructs` is
never consulted, unlike the Oracle/Each detection two blocks below in the same
file, which correctly uses `nodes_only + sub_constructs`. If the mesh entry is
a Construct, it's silently absent from `portal_members`; `group_members[0]`
becomes the wrong node (the first *Node* member after the real entry, or — if
no Node members remain in that group — the group vanishes from
`portal_members` entirely and no hop-counter/payload fields are allocated for
that mesh at all). At runtime, `make_portal_subgraph_fn` (`factory.py:402-403`)
reads `StateKeys.handoff_payload/hops(entry_field)` computed from the **real**
entry's name — a field that was never declared on the compiled Pydantic state
model. This fails hard on first invocation (attribute-access error on an
undeclared field), not at compile time.

**Consequence for fix ordering**: widening only `compiler.py:254` makes things
*worse*, not better — it converts a loud, compile-time-adjacent `CompileError`
into a silent-until-runtime crash deep inside the mesh wrapper. Per the
project's fail-loud-over-fail-soft north star, that is a regression even
though it looks like "progress" (the construct now compiles). Both sites must
move together.

**Verified module scorecard**: `_validation_portal.py`, `_contiguous_portal_mesh`,
`_add_portal_mesh`, and `runner.py`'s hop-cost helpers are *already*
Construct-entry-safe — do0d9 landed correctly in four of six relevant sites.
Only `compiler.py`'s top-level dispatch gate and `state.py`'s Portal
field-builder lag behind the relaxation the validator already grants.

**Concrete fix** (mechanical, ~10-20 lines across 2 files, TDD order per the
compilation-angle doc):
1. `compiler.py:254` — widen to `isinstance(item, (Node, Construct))` for the
   peer-mode Portal branch only; dispatch mode (line 259) stays Node-only
   (legitimate scope restriction — see §4).
2. `state.py:261-265` — source `portal_members` from `nodes_only +
   sub_constructs` (mirroring the pattern the Oracle/Each block two lines
   below it already uses), keeping the same combo/`_is_dispatch` filter.
3. Update the two now-stale "already rejected at assembly" comments
   (`compiler.py:552-560` and its `state.py` mirror arm) — they become
   genuinely unreachable for a *contiguous, entry-led* Construct mesh once (1)
   is fixed, but should stay in place as defense-in-depth for a malformed
   non-contiguous Portal-Construct shape, with corrected justification text.
4. Add a runtime-invoking test (not just a compile-only fixture) — a
   compile-only check would pass after step 1 alone and miss the state.py bug
   class entirely; only executing the compiled graph exposes the undeclared
   state field.

## 6. Downstream Agent Spec consequences (export/import — owned by other angles, cited here for completeness)

- **Export filter gap** (`_agent_spec.py::to_agent_spec`, mesh_members filter):
  `isinstance(item, Node)`-gated, so a legal Node-entry + Construct-non-entry
  mesh (the exact do0d9-spiked shape, already compiler-wireable independent of
  the C1 fix) is misclassified as "mixes a Portal peer mesh with non-mesh
  nodes" and rejected. Fix is reuse, not new design: swap the filter for
  `classify_modifiers(item)[0] in (PORTAL, PORTAL_OPERATOR)` +
  `_group_portal_members` — the same primitives already used everywhere else.
- **`_lower_portal_mesh_to_swarm` would also break on an actual Construct
  member** even with the filter fixed — it unconditionally calls
  `_make_agent(member, ...)`, which reads `node.prompt`/`node.tools`,
  attributes that don't exist on `Construct`. The fix again already exists in
  the same file: `_lower_construct_item` already handles a `Construct` item by
  recursively calling `to_agent_spec(item)` to produce a `Flow`; pyagentspec's
  `Swarm.first_agent`/`relationships` are typed `AgenticComponent` (the shared
  base of both `Agent` and `Flow`, confirmed via pyagentspec's own type
  system), so a `Flow` can be a direct Swarm participant with no wrapper
  needed. The fix is a per-member-type branch mirroring an existing branch,
  not new capability.
- **`handoff_param`/`handoff_channel` export guard** (`_agent_spec.py`, fires
  when either is set) is correctly Node-scoped and consistent with the fields'
  design — not a gap, just a per-node check that never gets reached for a
  misclassified "mixed" mesh (a separate symptom of the filter bug above, not
  a second defect).
- **Import direction (`loader.py`)** reconstructs Portal meshes only as `Node`
  members (`Node(inputs={'handoff': Payload}, ...) | Portal(...)`) — there is
  no reverse path to reconstruct a `Construct` mesh member, consistent with
  export never producing one today. More generally, `loader.py` never re-runs
  `_check_portal_mesh`-equivalent validation against a reconstructed
  `Construct`, unlike every other construction path (declarative, `@node`,
  programmatic), which all pass through the one assembly-time gate.

## 7. Where Portal sits in the wider ModifierCombo duplication problem

`classify_modifiers`/`ModifierCombo` dispatch (as opposed to narrow
single-modifier presence checks) is independently re-derived in at least 10
files today — `compiler.py` (two separate `match combo:` blocks),
`_agent_spec.py`, `state.py` (three separate blocks), `_state_write.py`,
`_subconstruct.py`, `_input_shape.py`, `runner.py`, `_wiring.py`, and
`_fan_agent.py` (a genuine 10th site the original 9-module inventory missed —
agent/act fan-modifier support checking). `loader.py` belongs on the list too,
but via a structurally different mechanism (marker/edge pattern-matching on
exported pyagentspec shapes, not `classify_modifiers` calls) — a distinction
worth preserving so a future grep-based consolidation doesn't silently miss
it. Within this picture, Portal is not an outlier: it is the specific combo
where the duplication has actually produced a live, verified bug (C1) rather
than merely a maintenance-burden risk, because the validator, `_wiring.py`,
and `runner.py` independently reached the *correct* generalized answer while
`compiler.py` and `state.py` independently reached the *stale* one — proof
that the "single walker, single source of truth" principle this project
otherwise enforces (`_validate_node_chain`, `effective_producer_type`,
`_declared_output`) has a real, demonstrated gap specifically at the
combo-dispatch layer, not a hypothetical one.

## 8. Portal's true current capability boundary (verified, not assumed)

**Works today, verified by passing fixtures/tests:**
- A mesh of any number of `Node` peer members, any mix of scripted/think/
  raw/agent/act modes, Operator-gated atomic members, named/unnamed groups,
  `Literal` or `str` route fields, `max_hops`/`on_exhaust` on the entry.
- A `Construct` as a **non-entry** peer-mesh member (do0d9) — assembly,
  compile, and runtime all verified correct.
- Dispatch-mode Portal (`route="decide"`) — Node-only by design, fully
  supported, unrelated to the entry-bug class.
- Full round-trip export/import for a pure-Node mesh (no Construct members).

**Assembly-clean but compile/runtime-broken today (the C1 bug):**
- A `Construct` as the mesh **entry** — passes `_check_portal_mesh` cleanly,
  then either raises a stale/mis-justified `CompileError` (if only the
  validator is consulted) or, once the naive compiler-only fix is applied,
  compiles successfully and crashes at first invocation on an undeclared state
  field. Neither is currently a working, intended path; the concrete fix
  (§5) is small, mechanical, and independently designed by two of the four
  source docs converging on the same two-site diagnosis.

**Not yet supported for Agent Spec (export/import), independent of the
compiler-level bug:**
- Any mesh containing a `Construct` member (entry or non-entry) cannot
  currently round-trip through `to_agent_spec`/loader — the export filter
  misclassifies it as "mixed" before even reaching the per-member Swarm
  lowering, which would itself need a small per-type branch to succeed. This
  is a genuine, currently-real gap in the "anything that compiles to
  LangGraph must be representable in Agent Spec" standing principle — but,
  per the cross-check doc, the fix requires no new pyagentspec capability:
  `Flow`/`Agent` already share `AgenticComponent`, so a Construct-as-Swarm-
  participant is representable today; the code simply hasn't been written to
  do it.
- Imported/reconstructed IR (via `loader.py`) never re-validates against
  `_check_portal_mesh`, so a malformed round-tripped mesh spec can silently
  produce a `Construct` that skips the one integrity gate every other
  construction path receives.

**Bottom line**: Portal's IR and its "reference" LangGraph compilation model
(the validator, `_wiring.py`'s mesh-building, and `runner.py`'s hop-cost
accounting) already treat `Construct` as a first-class member uniformly,
including as entry — the do0d9 relaxation was implemented correctly in the
majority of the sites that needed it. The gaps that remain are narrow,
mechanical, and localized: two compiler-adjacent sites for the Construct-as-
entry case, and two Agent-Spec-layer sites (a filter + a per-member-type
branch) for Construct-containing meshes generally. None of the four source
docs found a case where the underlying capability is genuinely infeasible —
every gap found is "the reference lowering already knows how; a duplicate,
staler decision site hasn't caught up yet," which is exactly the systemic
pattern the wider single-source-of-truth investigation is tracking.
