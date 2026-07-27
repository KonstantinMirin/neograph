# Phase 1 design-gap probe: Portal Construct-as-mesh-entry (neograph-s7zt3.5)

**Verdict: NOT fully scoped as filed.** The bead describes a two-site fix
(`compiler.py:254` + `state.py`'s `nodes_only`-only `portal_members` filter).
Verified against real source and a live end-to-end repro: there is a genuine
**third site** (`_ir_normalize.py:normalize_ir`) that must change in lockstep,
and the state.py fix as literally described ("union `nodes_only +
sub_constructs`, same as the Oracle/Each block") is **itself wrong** — it
silently reintroduces the exact bug class the bead is trying to close. Both
gaps are closed concretely below, verified by an in-memory-patched repro that
compiled and ran a real Construct-as-Portal-entry mesh end-to-end (repro
deleted after verification, per instructions).

## Method

Read `compiler.py` (the `compile()` walk, `_add_subgraph`'s PORTAL arm),
`_wiring.py` (`_contiguous_portal_mesh`, `_add_portal_mesh`,
`_make_portal_subgraph_member_fn`), `state.py` (`compile_state_model`'s Portal
block), `_validation_portal.py` (`_check_portal_mesh`, `_check_one_mesh_group`),
`_ir_normalize.py` (`normalize_ir`), `runner.py` (`_mesh_hop_cost`,
`_portal_mesh_member_ids`), and `node.py` (confirmed `Construct` has no
`handoff_channel`/`handoff_param` fields — only `Node` does).

Then built a throwaway script
(`entry_sub | Portal(to=["closer"]) ` as mesh entry, `closer` as the only
peer, `origin` as an ordinary upstream feeding `entry_sub`'s boundary port).
It used `inspect.getsource` on the three real functions, applied the textual
fix at each site, `exec`'d the patched source into a copy of the module's own
`__dict__` under new names, and called those as drop-in replacements — every
other function invoked (`_contiguous_portal_mesh`, `_add_portal_mesh`,
`_check_portal_mesh`, `_group_portal_members`, `StateKeys.*`, LangGraph itself)
was the real, unmodified, on-disk implementation. No repo file was ever
written to; the script was deleted after the run recorded below.

## Finding 1 (confirms the bead): sites 1 and 2 are real and as described

- `compiler.py:254`'s `isinstance(item, Node)` gate does misroute a
  Portal-Construct entry into `_add_subgraph`, which raises `CompileError`:
  `"Portal on a sub-construct is not supported"` — reproduced live.
- `_validation_portal.py` already admits a Construct at ANY mesh position,
  including entry (`entry = node_members[0]` has no isinstance restriction) —
  confirmed by reading `_check_one_mesh_group`; assembly is genuinely clean
  today for this shape, matching the bead's "assembly-clean" claim.
- `_wiring.py`'s `_contiguous_portal_mesh` / `_add_portal_mesh` /
  `_make_portal_subgraph_member_fn` are already fully Construct-agnostic for
  ANY position including entry (`entry_label_map` maps a Construct to itself,
  `_make_portal_subgraph_member_fn` compiles it exactly like `_add_subgraph`
  does) — confirmed by direct reading, no changes needed there. Relaxing
  `compiler.py:254` to `isinstance(item, (Node, Construct))` is sufficient at
  that one site.
- `runner.py`'s `_mesh_hop_cost` / `_portal_mesh_member_ids` already handle a
  Portal-carrying Construct generically (`current_run[0].modifier_set.portal`,
  `_member_hop_cost`'s `getattr(member, "modifier_set", None)`) regardless of
  entry-vs-non-entry position — no third finding needed there; these were
  written generically as part of the already-shipped do0d9 (non-entry) work
  and already cover the entry case too.

## Finding 2 (NOT in the bead): the state.py fix as literally worded is wrong

The bead says: fix `state.py`'s `portal_members` filter "the same way" as the
Oracle/Each block two lines above it, i.e. `nodes_only + sub_constructs`.

**This is a real bug if implemented literally.** `nodes_only + sub_constructs`
is list *concatenation*: every plain `Node` first, then every `Construct`,
regardless of their actual interleaved position in `construct.nodes`. The
Oracle/Each usage of this exact pattern two lines above (`all_items =
nodes_only + sub_constructs`) is safe there only because it's consumed as an
unordered boolean OR (`has_any_oracle`/`has_any_each`). Portal's
`portal_members` list is consumed by `_group_portal_members`, whose own
docstring states the invariant explicitly: *"Order is preserved: each group's
member list is in construct order... `items` must already be filtered to
PEER-mode Portal members [in construct order]"* — and `_group_portal_members`
treats `group_members[0]` as **the entry**.

Verified live: for `nodes=[entry_sub(Portal, Construct), closer(Portal,
Node)]`, `nodes_only + sub_constructs` yields `[closer, entry_sub]` (Node
before Construct), so `_group_portal_members` reports `closer` as the entry —
identical wrong-entry-field bug the two-site fix was meant to close, just via
list-concatenation reordering instead of exclusion. This is not a corner
case — it will misfire on essentially every Construct-as-entry mesh, because
the entry (by definition) must precede its Node peers in `construct.nodes`,
and the concatenation always puts Node peers first.

**Concrete fix**: filter over `construct.nodes` directly (not
`nodes_only`/`sub_constructs` at all), mirroring what `_check_portal_mesh` and
`_contiguous_portal_mesh` already do:

```python
portal_members = [
    n
    for n in construct.nodes
    if classify_modifiers(n)[0] in (ModifierCombo.PORTAL, ModifierCombo.PORTAL_OPERATOR) and not _is_dispatch(n)
]
```

Verified live: this produces `[entry_sub, closer]` (real order), `entry_sub`
correctly identified as entry, and the state model declares
`neo_handoff_entry_sub` / `neo_handoff_hops_entry_sub` (matching
`_wiring.py`'s real `entry_field`), not `neo_handoff_closer`.

## Finding 3 (a genuine third site, not in the bead at all): `_ir_normalize.py`

`_ir_normalize.py:normalize_ir` is documented as **the sole writer** of
`Node.handoff_channel` (CLAUDE.md: "Two IR fields on Node... written by a
single writer, `_ir_normalize.py`, and nowhere else"). It independently
computes its own `portal_members` list to derive the per-group channel key:

```python
portal_members: list[Node] = []
for item in construct.nodes:
    ...
    if isinstance(item, Node) and item.modifier_set.portal is not None:
        portal_members.append(item)
...
handoff_channels = {
    group_name: StateKeys.handoff_payload(field_name_for(members[0].name))
    for group_name, members in _group_portal_members(portal_members).items()
}
```

This is the exact same `isinstance(item, Node)` gate as `compiler.py:254`,
independently re-derived in a third module — and it feeds the SAME
order/entry-sensitive `_group_portal_members` call. When the entry is a
Construct, this walk excludes it, so `_group_portal_members` picks the first
*remaining Node* as "entry" and stamps `handoff_channel` on every Node peer
keyed off that wrong member. `Construct` itself has no `handoff_channel`
field (confirmed in `node.py` — only `Node` declares it), so this is not a
question of "does a Construct need the field too" — it's that **the group
lookup used to compute the correct key for the Node peers' field is corrupted
whenever the entry is a Construct**, even after sites 1 and 2 above are fixed.

**Live confirmation**: with sites 1+2 fixed but `_ir_normalize.py` untouched,
`closer.handoff_channel` was stamped `neo_handoff_closer` (wrong — the payload
channel `entry_sub`'s exit actually writes to, per the fixed `state.py`, is
`neo_handoff_entry_sub`). Running the compiled graph end-to-end with only
sites 1+2 fixed produced a real runtime failure: `closer`'s `handoff` input
resolved to `None` (reading from the WRONG, empty channel) — a live instance
of exactly the "silent state-key divergence" failure mode the bead's own
acceptance-test language is designed to catch, just occurring downstream of a
different site than the one the bead names.

**Concrete fix**: the same one-line relaxation, applied at this third site:

```python
if isinstance(item, (Node, Construct)) and item.modifier_set.portal is not None:
    portal_members.append(item)
```

The subsequent write-back loop (`for container, idx in iter_item_slots(...):
item = container[idx]; if not isinstance(item, Node): continue`) correctly
stays Node-only as-is — a `Construct` has no `handoff_channel` field to stamp,
and its own channel key is instead derived fresh, in real `construct.nodes`
order, by `_wiring.py:_add_portal_mesh` directly from `members[0].name` at
compile time (already correct, per Finding 1). Only the *detection* walk
(which peers reference which entry) needs the Construct-inclusive relaxation;
the *write* walk does not.

## Full verified fix (three sites, must land together)

1. `compiler.py:254` — `isinstance(item, Node)` → `isinstance(item, (Node,
   Construct))` in the mesh-detection branch of the `compile()` walk
   (dispatch-mode line, a few lines below, stays Node-only — unaffected, and
   verified unaffected: dispatch-mode Portal is excluded upstream by
   `is_dispatch`, checked only after the (now-relaxed) combo match, and a
   Construct can never itself be `is_dispatch` since that's a `Portal`
   instance attribute reached through a `Node`-only code path one line later
   in the same branch — no interaction).
2. `state.py`'s `portal_members` list comprehension — build it by filtering
   `construct.nodes` directly (preserving real order), **not** by
   concatenating `nodes_only + sub_constructs` (that reorders and
   reintroduces the wrong-entry bug).
3. `_ir_normalize.py:normalize_ir`'s own separate `portal_members` collector
   (used only to compute `handoff_channels`) — same `isinstance(item, (Node,
   Construct))` relaxation. This is the one genuinely new site the bead does
   not mention.

## Acceptance-test implication

The bead's own acceptance-test correction ("assert the state model's declared
Portal field names match `_wiring.py`'s actual `entry_field`, not just
does-not-raise") is the right shape of test, but as filed it would only catch
Finding 2 (the state.py field-naming bug), not Finding 3 (the
`handoff_channel` mis-stamp on the Node peer, which is invisible from the
parent's own `compile_state_model` output — it lives on `Node.handoff_channel`,
consumed at runtime, not in the declared state model's field *names*). The
test suite for this bead should add a second, runtime-invoking assertion:
compile and **run** a Construct-entry + Node-peer mesh end-to-end and assert
the routed payload actually reaches the peer (not just that the expected
field names exist) — exactly the gap a fields-only test would miss, and
exactly what the live repro above caught.
