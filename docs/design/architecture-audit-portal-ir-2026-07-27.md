# Architecture audit — Portal's IR-level implementation, fully mapped

**Angle**: Portal (formerly Keymaker) — every IR field/marker it introduces, the
full mesh validation ruleset as it exists today, and where validator-admitted
shapes are NOT actually handled by downstream consumers (compiler, Agent Spec
export/import). All claims below are verified against the source at the paths
and line numbers cited; nothing is re-derived from memory of the design docs
without independent source verification.

Sources read in full: `src/neograph/modifiers.py` (Portal class + grouping
helper + ModifierCombo), `src/neograph/_ir_normalize.py` (single-writer
normalizers), `src/neograph/_validation_portal.py` (`_check_portal_mesh`,
`_check_one_mesh_group`, `_check_portal_dispatch_error_handler`),
`src/neograph/_construct_validation.py` (integration point),
`src/neograph/_wiring.py` (`_add_portal_mesh`, `_contiguous_portal_mesh`,
`_add_portal_dispatch`), `src/neograph/compiler.py` (main dispatch loop +
sub-construct combo match), `src/neograph/_agent_spec.py` (`to_agent_spec`
mesh filter + `_lower_portal_mesh_to_swarm`), `src/neograph/loader.py` (Swarm
import, grep-level), plus `docs/design/dynamic-handoff-2026-07-13.md` (v1
design, original Keymaker name) and
`docs/design/do0d9-parent-scoped-bubbleup-2026-07-23.md` (the v2 extension
that admitted Construct mesh members — closed 2026-07-23, commit `481ee34`).

---

## (a) Every IR field/marker Portal introduces — writer/reader map

| Field/marker | Type | Sole writer | Readers |
|---|---|---|---|
| `Node.handoff_param` | `str \| None` | `_ir_normalize.py`'s `_HandoffParamNormalizer.apply` (`_ir_normalize.py:174-175`) — returns literal `"handoff"`, no inference. Gated by `_HandoffParamNormalizer.applies_to` (`:166-172`): fires only when `node.handoff_param is None`, `node.modifier_set.portal is not None`, and the node's `inputs` is dict-form with a `"handoff"` key. Docstring at `:153-164` explicitly names this "the SOLE writer... review H2 / neograph-k7bg". | `_input_shape.py` (per do0d9 doc `:380-381`, reads `node.handoff_channel` to source the reserved `handoff` input at runtime — I did not re-open `_input_shape.py`, cited from the do0d9 doc's own verified quote); `_agent_spec.py:138` — export-side fail-loud guard (see below). |
| `Node.handoff_channel` | `str \| None` | `_ir_normalize.py`'s `normalize_ir` main loop (`:292-302`), NOT a registered `_NORMALIZERS` entry — inlined directly in `normalize_ir` because it needs the *construct-level* view (all Portal members at a level) that a per-node normalizer's `applies_to`/`apply` signature can't see. Computed via `handoff_channels` dict (`:276-279`) keyed by `Portal.name` (mesh group), built with `StateKeys.handoff_payload(field_name_for(members[0].name))` — i.e. keyed off each named group's own entry. Idempotent (`if item.handoff_channel is None`). | Runtime handoff-channel resolution (same `_input_shape.py` site); `_agent_spec.py:138-144` fail-loud export guard. |
| `Node.fan_out_param` | (pre-existing, not Portal-specific) | n/a | n/a — listed here only to note it is the sibling field `handoff_param` mirrors; not itself a Portal field. |
| `ModifierCombo.PORTAL` / `ModifierCombo.PORTAL_OPERATOR` | enum values | `modifiers.py:78,84`; the combo→enum table at `modifiers.py:102-103` maps `frozenset({"portal"})` → `PORTAL` and `frozenset({"portal","operator"})` → `PORTAL_OPERATOR`. | `compiler.py:254-257` (main dispatch gate), `compiler.py:552-560` (sub-construct combo match, raises), `_agent_spec.py` (implicitly, via `modifier_set.portal is not None` checks rather than combo directly). |
| `Portal` modifier itself | `ModifierSet.portal: Portal \| None` slot | `ModifierSet.with_modifier` (`modifiers.py` `_SlotRule` table, `:762-789`) — Portal excludes every other modifier reciprocally (D-NO-OPERATOR-COMBO, checked both directions per review MEDIUM-2). | Every mesh/dispatch site below. |
| `HANDOFF_END = "__end__"` | module constant | `modifiers.py:593` | Route-Literal validation (`_validation_portal.py:272`), runtime `Command(goto=exit_name)` translation in `factory.py` (`_portal_route_to_command`, cited from grep, not re-opened in full). |
| `DISPATCH_ROUTE = "decide"` | module constant | `modifiers.py:598` | `Portal.is_dispatch` property (`modifiers.py:648,658`) — THE single mode discriminator; a structural guard bans the inline `route == "decide"` string check elsewhere (per the docstring at `:651-656`, not independently re-verified as a live guard test in this pass — **NEEDS CROSS-CHECK** if you need the guard test name). |
| `StateKeys.handoff_payload(...)` / `StateKeys.handoff_hops(...)` | `neo_`-prefixed state keys | Called from `_ir_normalize.py:277` (payload channel key) and `_wiring.py:937` (hop counter key, inside the Operator-approval splice) | Runtime state bus, checkpoint schema fingerprinting (excluded as `neo_*` framework fields per `state.py`'s fingerprint rule — not re-verified here, per AGENTS.md's general checkpoint section). |

**No `Construct`-level field exists for `handoff_param`/`handoff_channel`.**
Verified: `grep` for these names in `node.py` (defines both, `:212,221`) vs
`construct.py` (defines neither). This is deliberate per the do0d9 doc's Q6
answer (`do0d9-parent-scoped-bubbleup-2026-07-23.md:446-455`): a Construct
mesh member's channel key is threaded through the recursive `compile()`
kwarg chain to `make_portal_subgraph_fn`, not stamped as a field on
`Construct` — adding such a field "would violate the single-writer discipline
by adding a second place `handoff_channel`-like state lives."

**Consequence — two `isinstance(item, Node)` gates exclude Construct members from
the field-writing walk, by design, not oversight:**
- `_ir_normalize.py:265` — `portal_members` collection (feeds `handoff_channels`
  entry-detection) only appends `Node` items.
- `_ir_normalize.py:286` — the write-back loop (`iter_item_slots`) skips
  non-`Node` items entirely (`if not isinstance(item, Node): continue`).

Per the do0d9 doc (`:457-462`, its own explicit scope note, verified verbatim
below in §(c)), this was known and accepted for the topology in scope: **entry
is always a Node**, so `portal_members[0]` (the entry-detection anchor) is
never a Construct, and Construct-as-non-entry doesn't need its own
`handoff_channel` field because the channel key reaches it via the compile-time
closure, not a stamped attribute.

---

## (b) Full mesh validation rules as they exist TODAY

Entry point: `_construct_validation.py:348-349` — `_check_portal_mesh(construct)`
then `_check_portal_dispatch_error_handler(construct)`, called once per
construct level inside the same walk that recurses into sub-constructs (so a
mesh at any nesting depth is checked; per `_validation_portal.py:54-55`'s own
docstring, not independently traced further in this pass).

### `_check_portal_mesh` (`_validation_portal.py:40-90`)

1. **Member collection**: PEER-mode only (`portal.is_dispatch` is `False`) —
   a dispatch-mode Portal (`route="decide"`) is excluded; it is a standalone
   linear node, not a mesh member (review M1, comment at `:58-62`).
2. **Reserved name collision** (construct-wide, not per-group): a sibling
   literally named `"handoff"` collides with the reserved mesh-channel input
   key → `ConstructError` (`:79-86`).
3. **Grouping**: `_group_portal_members` (`modifiers.py:727-752`) partitions
   PEER-mode members by `Portal.name` (`None` = implicit default group). This
   is the SAME helper the IR normalizer (`_ir_normalize.py:278`) and the
   compiler's mesh collector (`_wiring.py:734`, inside
   `_contiguous_portal_mesh`) both call — a single grouping source of truth,
   verified by grep cross-reference across all three call sites.
4. Every rule below is applied **per group**, independently.

### `_check_one_mesh_group` (`_validation_portal.py:93-311`) — per-group rules

- **Member shape**:
  - Dict-form outputs (`outputs={...}`) on ANY member → rejected
    (D-DICT-OUTPUTS), Node-only check (a Construct has no `.outputs`).
  - **do0d9 admission (quoted verbatim, `:114-119`)**:
    > "do0d9 (§4 Q2): a Construct member is ADMITTED as a first-class mesh
    > member — its declared boundary output (`_declared_output`) must be the
    > uniform mesh payload, checked by the uniform-payload rule below exactly
    > as for a Node member. The former blanket `isinstance(member, Node)`
    > rejection is relaxed. The dict-form check stays Node-only: a Construct's
    > boundary is a single `.output` type (no dict-form analog)."
    This is the single, exact textual admission the task asked me to locate
    and quote.
  - Operator+non-atomic-member narrowed-rejection (`:140-157`): an
    Operator-gated member must be atomic (scripted/think/raw) — an agent/act
    member OR a Construct member carrying Operator is rejected
    (neograph-kdr1u).
- **Contiguity WITHIN the group** (`:166-178`): this group's members occupy
  consecutive positions among themselves (another group's members
  interleaved would split it).
- **Uniform payload** (`:180-193`): every member's `_declared_output(member)`
  is the same type as the entry's (`is` identity, not structural equality).
  This is the rule the do0d9 admission piggybacks on — a Construct member
  passes here iff its `.output` boundary type matches.
- **Peer existence + peer-is-same-group-member** (`:195-215`): every `to`
  target names a sibling that exists AND belongs to the same named group.
- **Single connected mesh per group** (`:217-242`, D-SINGLE-MESH): the `to`
  relation, treated undirected, must connect every member to the entry —
  two disjoint closures sharing a group name is rejected.
- **Entry-only knobs** (`:244-255`): `max_hops`/`on_exhaust` set on a
  non-entry member (i.e. present in `model_fields_set`) is rejected.
- **Route field typing** (`:257-289`): the payload model's `route`-named
  field (default `"goto"`) must be `str` or `Literal[...]`; Literal targets
  must be ⊆ member names ∪ `{HANDOFF_END}`.
- **Reserved `handoff` input typing** (`:291-310`) — **Node-only by explicit
  design**, per its own comment: "a Construct member's boundary port is its
  singular `.input`... there is no Construct analog to a reserved `handoff`
  inputs key... so a Construct member is skipped here." This matches the
  do0d9 doc's own Q2 resolution (`do0d9-...md:340-348`), which considered
  extending this check to Construct members and explicitly decided NOT to
  ("this specific rule has no Construct analog and should stay Node-scoped").

### `_check_portal_dispatch_error_handler` (`_validation_portal.py:313-338`)

Separate check, dispatch-mode only: `on_invalid="route_to_error"` requires
`error_handler` to name a real sibling in the construct.

### What is NOT restricted at validation time (verified by absence, not by a passing test)

- **`_check_portal_mesh` never singles out the ENTRY's type.** I read the
  full function body; every shape/uniform-payload/route check applies
  identically to `node_members[0]` (the entry) as to any other member. There
  is no `isinstance(entry, Node)` gate anywhere in `_validation_portal.py`.
  **This means the validator, as it stands today, does NOT reject a
  Construct as a mesh ENTRY** — contradicting the do0d9 design doc's own
  stated intent (see §(c) below) and the stale comment in `compiler.py`
  claiming this is "already rejected at assembly."

---

## (c) Validator-admitted shapes downstream consumers do NOT actually handle

### C1 — Construct-as-mesh-ENTRY: validator admits it, compiler crashes with a stale "already rejected at assembly" comment — CONFIRMED, not a guess

The do0d9 design doc states the intended scope explicitly (verbatim,
`do0d9-parent-scoped-bubbleup-2026-07-23.md:457-462`):

> "**Scope note**: in the topology this doc's spike and the do0d9 acceptance
> criteria describe, the mesh **entry is always a Node** (`dispatcher`), so
> the entry-detection computation (`portal_members[0].name`) is unaffected by
> a Construct member elsewhere in the SAME mesh. **A Construct AS THE ENTRY
> is out of scope for v1 — flag it explicitly as unsupported (fail loud, not
> silently wrong) until a real need arises.**"

I traced the actual fail-loud path and it does NOT match this framing:

1. `_check_portal_mesh` (per above) has no entry-type check at all — a
   Construct-as-entry construct **passes assembly-time validation cleanly**.
   The do0d9 doc's own intent ("flag it explicitly as unsupported... at
   assembly") was never implemented as an assembly-time `ConstructError`.
2. The rejection instead happens at **compile time**, and only as a side
   effect of a dispatch gate, not a dedicated check:
   - `compiler.py:254-257` — the main per-construct-level walk only routes an
     item into `_add_portal_mesh` when `isinstance(item, Node) and
     classify_modifiers(item)[0] in (PORTAL, PORTAL_OPERATOR)`. A
     Portal-modified **Construct** entry fails the `isinstance(item, Node)`
     test, so it falls through.
   - It then hits `compiler.py:292-303` — `elif isinstance(item, Construct):
     prev_node = _add_subgraph(...)`.
   - Inside `_add_subgraph`'s own `match combo:` block, the
     `ModifierCombo.PORTAL | ModifierCombo.PORTAL_OPERATOR` arm
     (`compiler.py:552-560`) raises:
     > `"Portal on a sub-construct is not supported"` with comment "already
     > rejected at assembly — this arm is defense-in-depth + exhaustiveness."
   - **That comment is false as the code stands**: nothing in
     `_validation_portal.py` rejects a Construct-as-entry mesh at assembly.
     The "defense-in-depth" framing implies redundancy with an assembly
     check that does not exist for this specific shape (it DOES exist and is
     genuinely redundant for a Construct with Portal that is NOT part of any
     mesh dispatch path reachable via `isinstance(item, Node)` — but a
     Construct that IS meant to be the mesh entry has no assembly-time
     rejection at all).
3. **The wiring layer (`_wiring.py:_add_portal_mesh`) has no actual technical
   obstacle to a Construct entry.** I read the full function body
   (`_wiring.py:799-968`): `entry = members[0]`; `entry.modifier_set.portal`
   works identically for `Node` and `Construct` (both have `modifier_set`);
   the `entry_label_map` construction (`:843-848`) already uses
   `getattr(member, "mode", None)` specifically so a `Construct` (no `.mode`
   attribute) falls through to `member.name` cleanly; the per-member loop
   (`:864-891`) already has an `isinstance(member, Construct)` branch that
   compiles the sub-construct and wraps it via
   `_make_portal_subgraph_member_fn`/`make_portal_subgraph_fn` — this branch
   runs identically whether the Construct is `members[0]` (entry) or a later
   member. **The only thing standing between "works" and "CompileError" is
   the `isinstance(item, Node)` gate in the main `compiler.py` walk
   (`:254`)** — a Construct-entry mesh never reaches `_add_portal_mesh` at
   all, so `_add_portal_mesh`'s own Construct-handling code is verified
   correct for entry-position but structurally unreachable for it.

**Net finding**: this is real, verified, and matches the task background's
framing exactly ("compiler.py itself has a bug where Construct-as-mesh-ENTRY
fails despite this admission") — except the precise failure is not a runtime
crash in `_add_portal_mesh`, it's a **dispatch-gate omission in
`compiler.py`'s top-level walk** that routes a Construct-Portal entry to the
wrong compile arm, which then raises with a stale/inaccurate justification
comment. Fixing this is plausibly a small, well-scoped change (widen the
`compiler.py:254` gate to also match a Portal-modified `Construct`, and add
the entry-detection wiring `_contiguous_portal_mesh` already handles
type-agnostically per its own docstring at `_wiring.py:707,713` — "Portal-
modified member — a Node OR a sub-`Construct` (do0d9, §3.1 site 4)"). **This
is a design/implementation judgment for whoever owns the fix, not something I
implemented — I only mapped and verified the gap exists.**

### C2 — Agent Spec export (`_agent_spec.py:to_agent_spec`): Node-only mesh-member filter silently misclassifies a legal Node-entry + Construct-non-entry mesh as "mixed" — CONFIRMED

`to_agent_spec` (`_agent_spec.py:948-977`) computes:

```python
mesh_members = [
    item for item in all_items
    if isinstance(item, Node)
    and item.modifier_set.portal is not None
    and not item.modifier_set.portal.is_dispatch
]
if mesh_members:
    if len(mesh_members) != len(all_items):
        raise ConfigurationError.build(
            f"construct {construct.name!r} mixes a Portal peer mesh with non-mesh nodes", ...)
```

This filter is **Node-only** (`isinstance(item, Node)`). Consider the exact
topology the do0d9 doc's own acceptance criteria targets and that
`_check_portal_mesh`/`_wiring.py` both actively support today: a mesh whose
entry is a `Node` (`dispatcher`) and whose OTHER member is a `Construct` with
`Portal` attached (`worker` sub-construct, do0d9's own spiked shape). In that
construct, `all_items` (via `iter_with_arms`) includes BOTH the Node and the
Construct, but `mesh_members` only counts the Node (the Construct fails
`isinstance(item, Node)`). So `len(mesh_members) != len(all_items)` is
**always true** for this legal, validator-passing, compiler-wireable (once C1
is separately fixed for entry, this shape doesn't even need C1 — non-entry
Construct membership already compiles per `_wiring.py:867-891`) topology —
`to_agent_spec` will raise "mixes a Portal peer mesh with non-mesh nodes" on a
construct that is, in fact, entirely one mesh. This directly falsifies the
export path for a shape the validator and the compiler's `_add_portal_mesh`
both already accept.

I did not exhaustively trace `_lower_portal_mesh_to_swarm`'s per-member
`_make_agent` call to confirm whether it would ALSO choke on a `Construct`
member if the filter above were fixed (it calls `_make_agent(member, ...)`
which likely assumes Node-shaped fields like `.prompt`/`.inputs` —
`_agent_spec.py:915-919` reads `member.prompt`/`member.inputs` unconditionally
for every mesh member). **NEEDS CROSS-CHECK**: whether `_lower_portal_mesh_to_swarm`
has ANY Construct-member handling at all, or whether fixing the `to_agent_spec`
filter alone would just move the crash one level deeper (an `AttributeError`
on `Construct.prompt`, which doesn't exist) — I recommend the Agent-Spec-angle
research agents (research-agentspec / verify-agentspec-current-state /
verify-agentspec-loader) confirm this specific sub-question since it's their
assigned surface, not mine.

### C3 — `handoff_param`/`handoff_channel` export fail-loud guard (`_agent_spec.py:138-144`) — consistent, not a gap

```python
if node.handoff_param is not None or node.handoff_channel is not None:
    raise ... "node {node.name!r} is a Portal mesh member (handoff_param/handoff_channel set) —
    ... Portal mesh members cannot be exported to Agent Spec ..."
```

This guard is Node-scoped by construction (the fields only exist on `Node`),
consistent with the fields' own design. It does NOT contradict C2 — C2 is
about the mesh-membership FILTER in `to_agent_spec` mis-firing before this
per-node guard is ever reached for a mixed Node/Construct mesh; this guard is
a separate, correctly-scoped check that fires for individual Node members
that ARE correctly identified as mesh members (e.g. a pure Node-only mesh
exported outside `_lower_portal_mesh_to_swarm`'s intended Swarm path, or a
regression elsewhere). Flagging as consistent, not as a second finding.

### C4 — `Portal.is_dispatch` string-literal guard — NEEDS CROSS-CHECK

The `is_dispatch` property docstring (`modifiers.py:651-656`) claims "pinned
by a structural guard that bans the inline literal outside this module." I
did not locate and confirm this guard test by name in this pass (out of
scope for the Portal-IR angle — it's a test-suite claim, not an IR-shape
claim). **NEEDS CROSS-CHECK** by whichever angle covers guard-test inventory.

### C5 — Loader.py (import direction) — NOT independently deep-verified this pass

I grepped `loader.py` for Portal/Swarm handling (`_reconstruct_swarm_mesh` at
`:690-727`, `:759-766`) and confirmed by line-count that it builds
`Node(inputs={'handoff': Payload}, outputs=Payload) | Portal(to=[peers])`
members from a foreign `Swarm`'s agents — i.e. it can only ever reconstruct
**Node** mesh members, never a `Construct` member (there is no reverse
direction here since a Swarm's `Agent` has no sub-construct concept). This is
consistent with C2's finding (export can't correctly round-trip a
Construct-containing mesh) rather than a new independent gap. **NEEDS
CROSS-CHECK**: whether `research-agentspec`/`verify-agentspec-loader` found
anything additional in the full loader.py body I did not re-read line by line
(I did not open the full file, only grep-matched Portal-related lines).

---

## Summary of confirmed vs unresolved

**Confirmed by direct source reading (not inference from docs alone):**
- Full field/writer/reader map in §(a).
- The do0d9 "Construct admitted as mesh member" quote, verbatim, with exact
  line numbers, in §(b).
- C1: Construct-as-mesh-entry is validator-clean but compiler-fatal via a
  dispatch-gate omission, with a demonstrably stale justification comment.
- C2: Agent Spec export's mesh-membership filter is Node-only and will
  misclassify a legal Node-entry + Construct-member mesh as "mixed."
- C3: the handoff_param/handoff_channel export guard is correctly scoped
  (not itself a gap).

**Flagged NEEDS CROSS-CHECK (deliberately not guessed):**
- Whether `_lower_portal_mesh_to_swarm`'s `_make_agent` call would also break
  on an actual Construct member if C2's filter were fixed (C2 detail).
- The named structural guard test for `Portal.is_dispatch`'s "no inline
  literal" claim (C4).
- Full loader.py body beyond the Portal-related grep hits (C5) — owned by
  the Agent-Spec-angle agents.
