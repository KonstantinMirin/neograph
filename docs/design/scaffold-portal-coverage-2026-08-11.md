# Design: Portal coverage in the neograph.testing scaffold (neograph-wvp7j)

Design pass, 2026-08-11. Scope: `src/neograph/testing/scaffold.py` only (the YAML
spec surface with the same disease is neograph-2j208's scope).

## The defect, restated precisely

`scaffold.py` hand-enumerates the modifier universe in FOUR places, and all four
stop at oracle/each/loop/operator:

1. `_node_info` (scaffold.py:67-70) — the per-node capture dict has no `portal` key.
2. `_gen_modifiers` (scaffold.py:236) — `modified = [n for n in nodes if n["oracle"] or n["each"] or n["loop"] or n["operator"]]`.
3. `_gen_sync` (scaffold.py:519-521) — the same 4-way disjunct builds `EXPECTED_MODIFIED`.
4. **Inside the GENERATED test file** — `_gen_sync` emits `has_mod = bool(item.modifier_set.oracle or ... or item.modifier_set.operator)` (scaffold.py:574-575). This one is worse than the other three: a generated suite is frozen at scaffold time, so even after scaffold.py is fixed, this emitted enumeration would rot again the day a sixth modifier lands.

Consequence: a Portal-modified node produces no modifier assertions, no
EXPECTED_MODIFIED entry, and — because the emitted `has_mod` is also partial —
the generated drift test can never fire "gained a modifier" for Portal. The
scaffold reports covering the construct while silently not covering it.

## Q1 — What a generated Portal assertion should assert

Portal is one modifier with two structurally different modes, discriminated
ONLY via `Portal.is_dispatch` (`_portal.py:80` — the declared single source of
truth; a structural guard already bans the inline `route == "decide"` literal
outside `_portal.py`). Generated code and scaffold code alike must read
`is_dispatch` / `is_tool_triggered`, never string-compare fields.

### Dispatch mode (`route="decide"`) — per-node assertions in test_modifiers.py

A dispatch Portal is a standalone linear node, never a mesh member
(`_portal_member.py` docstring). The per-node pattern the other four modifiers
use fits it exactly. Emit, following the oracle precedent (unconditional
asserts for required knobs, conditional for optional ones):

```python
portal = planner.modifier_set.portal
assert portal is not None
assert portal.is_dispatch
assert portal.spec_field == "spec"
assert portal.input_field == "dispatch_input"
assert portal.output is Summary            # __name__-rendered if a class, string literal if str
assert portal.max_depth == 3
assert portal.on_invalid == "raise"
# only if set:
assert portal.error_handler == "handle-bad-spec"
assert sorted(portal.scripted) == ["fn_a", "fn_b"]      # keys only — values are callables
assert sorted(portal.conditions) == ["done"]
```

All four required dispatch knobs (`spec_field`, `input_field`, `output`,
`max_depth` — enforced by `model_post_init`, `_portal.py:144-153`) are asserted
unconditionally. `scripted`/`conditions` registries are asserted by sorted key
list only; the callables cannot be value-asserted, and pretending to (repr of a
lambda) would be a fake assertion.

Capture caveat: `output` may be a class or a string. Render with the same
`__name__`-collapsing rule `_node_info` already applies to `outputs_name`
(scaffold.py:43-46, the neograph-wpzg/CON-01 fix) — a raw class repr is not
valid Python in generated source.

### Peer mode (`to=[...]`) — assertions live at MESH level (see Q2)

Per-member knobs: `to` (the declared successor set), `trigger`
(`output`/`tool`), `name` (mesh group, if set), and the member's
`PortalMemberClass`. Entry-only knobs: `max_hops`, `on_exhaust` — meaningful
only on `members[0]` (`_portal.py:65`), so asserting them per node on non-entry
members would emit default-value noise that LOOKS like coverage. They are
asserted once, on the entry, in the mesh test.

The `PortalMemberClass` assertion is the highest-value one: `portal_member_class(item)`
(`_portal_member.py:73`) is the ONE authority for "what kind of participant is
this", and it is exactly the axis on which silent lowering changes happen — a
member whose mode flips scripted→agent moves from ATOMIC to AGENT_CYCLE_OUTPUT
and compiles to a different runtime shape. A generated
`assert portal_member_class(member) is PortalMemberClass.ATOMIC` pins that with
one line and zero re-derived taxonomy.

## Q2 — Per-node dict vs mesh-level notion (the crux)

**The per-node dict is necessary but not sufficient. The scaffold needs a
mesh-level summary, collected through `_group_portal_members` — never a
re-derived grouping.**

Why per-node cannot express the mesh:

- **Entry identity is positional.** `members[0]` of a contiguous sibling run is
  the entry; the entry carries the budget knobs and is the only legal jump
  target (the north-star entry-port rule). No single node's dict knows whether
  it is first. Reordering two siblings changes the entry — a silent routing +
  budget change with zero per-node field difference. That is precisely the
  drift class test_sync exists to name.
- **Membership is a set property.** "Which nodes form mesh `name=X`, in what
  order" is the thing `_group_portal_members` (`_portal.py:175`) computes, and
  its docstring declares it the SINGLE grouping authority (validator, IR
  normalizer, and wiring collector all route through it). The scaffold
  re-deriving grouping inline would replant the exact duplicated-source-of-truth
  disease this ticket is about.
- **Mesh members need not be Nodes.** A SUB_CONSTRUCT member is a `Construct`
  with a portal; `_collect_items` routes Constructs into `subs` dicts that
  carry no modifier info at all. A node-keyed portal capture would silently
  drop sub-construct members — a second silent seam inside the fix.

Concrete design:

- `_node_info` gains a `"portal"` key (per-node capture: `is_dispatch`, mode
  knobs, `member_class` name) — parity with the other four slots, and it feeds
  the totality-fixed `modified` filter and `EXPECTED_MODIFIED`.
- New collector `_collect_meshes(construct) -> list[dict]`: walk
  `construct.nodes` (top-level items, Node OR Construct), keep items whose
  `portal_member_class(item)` is a mesh-member class (not `None`, not
  `DISPATCH`), group via `_group_portal_members`. Each mesh dict:
  `{"name": group_or_None, "members": [ordered names], "entry": members[0],
  "max_hops": entry.portal.max_hops, "on_exhaust": entry.portal.on_exhaust,
  "per_member": [{name, to, trigger, member_class}, ...]}`.
- `_gen_modifiers` emits one `TestPortalMesh...` class per mesh asserting:
  ordered membership against `_group_portal_members(...)` on the live construct
  (pins membership, order, and therefore entry identity and contiguity),
  entry budget knobs, and each member's `to`/`trigger`/`PortalMemberClass`.
- Split of responsibility: dispatch Portal → per-node test in `TestModifiers`;
  peer Portal → everything in the mesh test. This resolves the Node/Construct
  asymmetry (the mesh test looks members up by NAME in `construct.nodes`, so a
  sub-construct member gets identical treatment) and avoids duplicating peer
  assertions in two places.

Generated-code imports: the mesh test imports `_group_portal_members` and
`portal_member_class`/`PortalMemberClass`. Generated suites already import the
private `neograph._sidecar._get_sidecar` (scaffold.py:322), so private imports
are the established scaffold convention. (Optional follow-up, not in scope:
re-export the two symbols through `neograph.testing` so generated suites stop
depending on private paths; file as its own ticket if wanted.)

## Q3 — _gen_sync drift shape

Three changes, each with a named drift class it catches:

1. **`EXPECTED_MODIFIED` becomes total** (via the Q4 registry). Catches: Portal
   added to / removed from a node. This is the tier-drift analog for modifiers
   and today is silently blind to Portal.
2. **The EMITTED `has_mod` expression is replaced with
   `has_mod = bool(item.modifier_set.to_list())`** (`to_list` walks
   `_SLOT_RULES`, `modifiers.py:804` — total over the roster by construction).
   This makes generated suites total FOREVER, including against modifiers that
   do not exist yet: the frozen-at-scaffold-time enumeration is the one site
   the scaffold cannot patch retroactively, so it must not enumerate at all.
3. **Two new expected-blocks + drift tests:**
   - `EXPECTED_PORTAL_MESHES = {mesh_name_or_None: ("entry", "m2", ...)}` —
     compared against `_group_portal_members` output on the live construct.
     Catches: member added/removed, members reordered (entry changed),
     peer member flipped to dispatch (drops out of the grouping). These are
     routing-topology changes — exactly the "shape drift" tier test_sync
     owns (its existing checks are scripted↔LLM tier and modifier
     presence, never knob values).
   - `EXPECTED_DISPATCH = {...}` names — drift test asserts each still has
     `modifier_set.portal is not None and portal.is_dispatch`, and that no
     node OUTSIDE the set became dispatch. Catches: the peer↔dispatch mode
     flip, which rewrites the node's entire lowering (mesh member vs
     standalone Command-returning runtime).

   Knob-value drift (`max_hops` 10→6, `spec_field` renamed) deliberately stays
   in test_modifiers/mesh assertions, not sync — sync's contract is
   shape/tier-level, mirroring how it treats `mode` but not `prompt`.

## Q4 — Yes, the enumeration is the real defect. Recommendation: capture registry + totality guard

Adding a 5th disjunct fixes Portal and re-plants the disease for modifier six.
The root cause is that scaffold.py's modifier vocabulary has no structural tie
to the roster in `modifiers._SLOT_RULES` (the declared "adding a new modifier
means adding ONE row here" table, `modifiers.py:646-655`).

**Recommended:** a scaffold-local capture registry

```python
_MODIFIER_CAPTURE: dict[str, Callable[[ModifierSet, Node], dict | None]] = {
    "each": ..., "oracle": ..., "loop": ..., "operator": ..., "portal": ...,
}
```

- `_node_info` builds its modifier keys by iterating `_MODIFIER_CAPTURE`.
- Every "is this node modified" check derives from the registry keys
  (`any(n[slot] for slot in _MODIFIER_CAPTURE)`), so sites 2 and 3 stop being
  independent enumerations.
- A structural guard (failing-first, per project practice) pins totality:
  `set(_MODIFIER_CAPTURE) == {r.slot for r in _SLOT_RULES}`, plus an AST/grep
  ban on the inline `n["oracle"] or n["each"]`-style disjunct in scaffold.py.
  A sixth modifier without a capture entry now fails CI loudly instead of
  falling through silently. Same shape as the
  `test_guards_modifier_composition_completeness.py` precedent for
  `MODIFIER_KWARGS`.

**Rejected alternative — fully generic capture** (dump each modifier's
`model_fields` and assert everything): truly automatic, but emits meaningless
assertions — lambda reprs for `Operator.when`/`Portal.scripted`, entry-only
budget knobs asserted on non-entry members, mode-forbidden fields asserted at
defaults. That is the "reports covering without covering" failure this ticket
exists to refuse. Per-modifier emission knowledge (which knobs are assertable,
in which mode, at which level) is irreducible; the registry's job is to make
its ABSENCE loud, not to eliminate it.

**Tradeoff stated:** the guard is test-time, not import-time — a sixth
modifier's author sees a red guard, not a type error. Acceptable: it matches
every other totality ratchet in this codebase (`_COMBO_MAP` exhaustiveness,
`MODIFIER_KWARGS` completeness), and an import-time assert in scaffold.py would
punish every user import for a maintainer-only invariant.

Note: `PortalMemberClass` (dgbqv.3) does NOT replace this registry — it answers
"what kind of mesh participant", not "which slots exist". The registry keys on
`_SLOT_RULES`; the member-class taxonomy is the vocabulary INSIDE the portal
capture/emission, as Q1/Q2 use it.

## Q5 — Implementation plan

Constraint discovered during this pass: `testing/scaffold.py` sits at its EXACT
file-size ceiling (`ALLOWLIST["testing/scaffold.py"] == 664`,
`tests/test_guards_file_size.py:86`), and allowlist growth is blocked, not
deferrable. The Portal emitters + mesh collector will not fit. **Plan
accordingly: new sibling module `src/neograph/testing/_scaffold_portal.py`**
(mesh collector + portal emit helpers + `_MODIFIER_CAPTURE`'s portal entry),
kept under 500 lines so it needs no allowlist entry; scaffold.py imports from
it and should end BELOW 664, lowering (or deleting) its ceiling in the same
commit. This is a fresh-file addition, not a split of entangled clusters, so
the file-split procedure's heavier machinery is not required — but the two
refusals (no monopoly widening, no behavior change smuggled in) still apply.

Ordered steps for the executor:

1. **Guards first, failing** (guard-first discipline):
   a. Totality guard: `set(_MODIFIER_CAPTURE) == {r.slot for r in _SLOT_RULES}`
      (fails until the registry exists with all five entries).
   b. Enumeration ban: no inline multi-slot `or`-disjunct over modifier keys in
      `testing/scaffold.py` / `_scaffold_portal.py` (AST or targeted grep).
   Home: `tests/test_scaffold.py` for (a) is acceptable; if placed in a
   `test_guards_*` file, follow the existing guard-file naming.
2. **Behavioral tests, failing** (in `tests/test_scaffold.py`, following
   `TestScaffoldGeneratesCompilableCode`'s tmp_path pattern):
   - Peer fixture construct: 3-member mesh with heterogeneous member classes
     (one ATOMIC, one AGENT_CYCLE_TOOL via `trigger="tool"` + agent mode with a
     FakeTool, one ATOMIC_OPERATOR or SUB_CONSTRUCT), entry with non-default
     `max_hops`/`on_exhaust`.
   - Dispatch fixture construct: one `route="decide"` node with all four
     required knobs, plus `on_invalid="route_to_error"` + `error_handler`.
   - Assert: generated `test_modifiers.py` contains the mesh class + dispatch
     assertions; generated `test_sync.py` contains `EXPECTED_MODIFIED` with the
     portal nodes, `EXPECTED_PORTAL_MESHES`, `EXPECTED_DISPATCH`, and a
     `to_list()`-based `has_mod` (assert the old 4-way emitted disjunct is GONE).
3. **Implement** `_scaffold_portal.py` + scaffold.py wiring per Q1-Q4:
   `_MODIFIER_CAPTURE` registry; portal capture in `_node_info`;
   `_collect_meshes` via `_group_portal_members` + `portal_member_class`;
   `_gen_modifiers` total filter + dispatch per-node emit + mesh test emit;
   `_gen_sync` total sets + emitted-`to_list()` + mesh/dispatch drift blocks.
4. **should_pass-style verification** (the acceptance test that the scaffold
   COVERS rather than REPORTS covering):
   - **Green half:** scaffold both fixture constructs into tmp_path, `compile()`
     every generated file (existing pattern), then EXECUTE the auto-verified
     files (`test_modifiers.py`, `test_sync.py`) against the source construct
     (exec with the construct + imports injected into the namespace, or a
     `pytest.main` sub-run) — they must PASS as generated, honoring the module
     docstring's "auto-verified, always green" contract.
   - **Red half (mutation):** run the SAME generated `test_sync.py` against a
     mutated construct — (i) mesh members reordered (entry changed), (ii) a
     peer member flipped to dispatch, (iii) Portal removed from a member — and
     assert it FAILS each time. Without the red half, the drift tests could be
     vacuous and nobody would know; this is the direct analog of a should_fail
     check fixture. Mutate by building a second construct variant, never by
     `git checkout` tricks.
5. **Sweep:** re-grep the whole of scaffold.py/_scaffold_portal.py for any
   remaining modifier-name enumeration (the Portal-rollout lesson: treat any
   hand-picked list as provisional until re-verified by grep). Update the
   file-size allowlist entries in the same commit (lower/delete scaffold.py's;
   none needed for the new file if <500). Run `uv run pytest` + `make quality`.

Non-goals / explicitly out of scope: the YAML spec surface (2j208), re-exporting
the two private symbols through `neograph.testing` (optional follow-up ticket),
and any change to `_portal.py` / `_portal_member.py` / `modifiers.py` — this
design only CONSUMES their declared authorities.
