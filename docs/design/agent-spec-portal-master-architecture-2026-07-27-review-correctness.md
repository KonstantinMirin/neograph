# Correctness review: `agent-spec-portal-master-architecture-2026-07-27.md`

Date: 2026-07-27
Lens: CORRECTNESS (independent re-verification against real source + throwaway
repros; reading-only verification has previously missed real bugs in this
investigation, so every claim below was exercised with actual code, not just
inspected).

Reviewer note on method: all repro scripts were written to the session
scratchpad, run via `uv run --extra dev python`, and deleted afterward. One
throwaway edit to `src/neograph/compiler.py` (relaxing the `isinstance(item,
Node)` gate to `isinstance(item, (Node, Construct))` at the line the document
calls out) was made to test a specific behavioral prediction, then reverted
with `git checkout --` before this document was finalized — `git status`
confirms `src/neograph/compiler.py` is clean.

---

## Claim 1 — Construct as non-entry Portal mesh member WORKS today (do0d9)

**Document text** (§1, §2, §5): "a `Construct` CAN be a non-entry Portal mesh
member today (do0d9, passing fixture)"; verified fixtures =
`tests/check_fixtures/should_pass/portal_construct_member.py` +
`tests/test_portal_cross_subconstruct.py`.

**Verification**: ran both suites directly.

```
uv run --extra dev pytest tests/test_portal_cross_subconstruct.py -q
→ 7 passed in 0.34s

uv run --extra dev pytest tests/test_check_fixtures.py -q -k portal_construct_member
→ 2 passed, 99 deselected in 0.23s
```

Read `portal_construct_member.py`: the mesh is `entry (Node) → resolver_sub
(Construct, non-entry member) → closer (Node)`, i.e. genuinely exercises a
Construct as a non-entry peer. `test_portal_cross_subconstruct.py`'s mesh
(`dispatcher (Node, entry) → resolver_sub (Construct) → specialist (Node)`) is
the same shape and asserts on the actual routed values, not just "it compiles."

**One discrepancy worth flagging** (not a correctness defect in the document,
but a documentation-hygiene one the document doesn't mention): the fixture's
own header comment and the test file's own module docstring are **stale** —
both say "This ... REJECTED at assembly ... the TDD-red state" / "Why these
tests FAIL today". They currently **pass**. This doesn't affect the
document's claim (which correctly states the capability works), but a future
reader skimming only the fixture/test file's own comments (as opposed to
running them) would be actively misled in the opposite direction — a small
irony given this whole investigation's thesis that comments drift from code.

**Verdict: CONFIRMED.**

---

## Claim 2 — The Construct-as-mesh-entry two-site bug, and the "site-1-alone →
silent runtime crash" prediction

**Document text** (§2, §5, Phase 1): two sites must move together —
(1) `compiler.py:254`'s `isinstance(item, Node)` gate misroutes a
Portal-modified Construct entry to `_add_subgraph`, whose PORTAL arm
unconditionally raises `CompileError` with a false "already rejected at
assembly" comment; (2) `state.py:261-265`'s Portal state-field builder sources
`portal_members` from `nodes_only` only (unlike Oracle/Each two lines below,
which correctly use `nodes_only + sub_constructs`). The document further
claims: "Fixing site 1 alone converts a loud `CompileError` into a **silent
runtime crash** on first invocation (undeclared-field attribute error)."

**Verification, site 1 (unpatched)**: built a Construct-as-mesh-entry repro
(`entry_sub | Portal(to=["closer"]), closer | Portal(to=[])`) and called
`compile()`. Result:

```
COMPILE FAILED: CompileError Portal on a sub-construct is not supported
  expected: mesh members must be sibling Nodes (D-MESH-LEVEL)
  found: Portal modifier on sub-construct 'entry_sub'
```

Confirmed exact match to `compiler.py:552-560`'s PORTAL arm inside
`_add_subgraph`, whose comment reads "already rejected at assembly — this arm
is defense-in-depth". Since the repro's `Construct(...)` call itself
**succeeded** (assembly did not reject it — the error only fired at
`compile()`), the comment is demonstrably false, exactly as the document says.

**Verification, site 2 (read only)**: `state.py:239-265` — confirmed
byte-for-byte. Line 213's Oracle/Each block builds `all_items = nodes_only +
sub_constructs`; line 261-263's `portal_members` list comprehension iterates
`nodes_only` only, with no `sub_constructs` union anywhere in that block. This
is exactly the asymmetry the document describes.

**Verification, the "silent runtime crash" prediction — REFUTED as stated**:
patched `compiler.py:254` alone (`isinstance(item, Node)` →
`isinstance(item, (Node, Construct))`, the exact minimal site-1 fix), left
`state.py` untouched, and ran the same repro through `compile()` **and**
`run()`. Result: **it compiled successfully and ran successfully end-to-end,
with no crash of any kind** —

```
RUN OK: {'closer': Handoff(goto='__end__'), 'entry_sub': Handoff(goto='closer')}
```

Traced why: state.py's `portal_members` (nodes_only-only) reduces to `[closer]`
in a 2-member mesh where the Construct entry is excluded — so the state model
declares hop/payload fields keyed off `closer` (the wrong entry) instead of
`entry_sub` (the real entry per `_wiring.py`'s `_contiguous_portal_mesh`,
which correctly computes `entry = members[0] = entry_sub`). At runtime,
`factory.py`'s `_portal_route_to_command` reads/writes these fields via
`StateBus.get_counter`/`_ModelStateBus.get_counter`
(`src/neograph/_state_bus.py:91`), which is `getattr(self._state, key, None)`
with a default — **not** a raising attribute access. So a field-name mismatch
between what state.py declares and what `_wiring.py`/`factory.py` actually key
off does not raise `AttributeError`; it silently falls back to `0`/`None` for
the hop counter and payload channel. In my two-hop, no-loopback topology this
happened not to be observably wrong (the mesh exits after one hop each way,
so the miskeyed hop counter/payload channel were never actually needed), which
is why it "just worked" — but the mechanism the document names
("undeclared-field attribute error") does not exist in the code path I
exercised, and the actual failure mode for a topology that *does* need the
counter (e.g., a mesh that loops back through the Construct entry, or relies
on the shared payload channel to route between non-adjacent members) would be
**silent wrong behavior** (hop budget silently reset to 0 every call, payload
channel silently empty) rather than a crash. This is arguably a **worse**
symptom than what the document claims (a silent correctness bug beats a loud
crash, but is much harder to detect), so the underlying "these two sites must
land together" conclusion is directionally right and, if anything,
understates the risk — but the specific causal mechanism described
("undeclared-field attribute error", "crash") is not what the code does and
should be corrected before this becomes an implementation ticket's stated
acceptance criterion (a ticket written to catch an `AttributeError` would not
catch the actual bug class, which is silent divergence).

**Verdict: PARTIALLY CONFIRMED.** The bug-pairing (both sites must land
together) and the stale-comment finding are both confirmed exactly as stated.
The specific claimed failure mechanism ("silent runtime crash",
"undeclared-field attribute error") is refuted by direct repro — the actual
mechanism is silent state-key divergence tolerated by `StateBus`'s
default-on-missing getattr, which is silent wrong behavior, not a crash. Any
downstream ticket citing this claim should describe the acceptance test as "no
divergent/miskeyed hop-counter or payload-channel field for a Construct
entry" (e.g., assert the state model's declared field names match
`_wiring.py`'s actual `entry_field`), not "does not raise."

---

## Claim 3 — `PORTAL_OPERATOR` silently drops the Operator HITL gate on export

**Document text** (§3, item 2; §5; Phase 6/B1): `_lower_portal_mesh_to_swarm`
builds a bare `Agent` per member via `_make_agent` and never inspects
`member.modifier_set.operator`; a compiler-supported combo (human-approval
gate on a dynamic-routing path) vanishes with no error, no marker
(`neograph-s7zt3.2`).

**Verification**: read `_agent_spec.py:880-941` (`_lower_portal_mesh_to_swarm`)
in full. The per-member loop is:

```python
for member in members:
    rewritten, ref_props, flat_to_original = _translate_placeholders(...)
    agent = _make_agent(member, tools_mod, ref_props, [], rewritten)
    agent.metadata = {**(agent.metadata or {}), _MARK_PROMPT_SPEC: ...}
    agents_by_name[member.name] = agent
```

Grepped the whole function body for `operator` / `modifier_set.operator` —
zero occurrences. There is no branch, no metadata marker, no error path that
references the member's Operator modifier anywhere in this function or in
`to_agent_spec`'s mesh-detection block above it. A `Node | Portal(...) |
Operator(...)` mesh member — which the document states, and I independently
confirmed by reading `_validation_portal.py`, is legitimately compiler-
supported (Operator-gated members must be atomic; this constraint is
enforced, not a rejection of the combo itself) — exports as an
indistinguishable plain `Agent`. No throwaway repro was needed beyond this
read since the claim is a pure absence-of-code claim (there is no operator
handling to trigger); the absence is directly checkable by exhaustive grep of
the function body, which I did.

**Verdict: CONFIRMED.**

---

## Claim 4 — `_lower_construct_item` never calls `classify_modifiers` on a
`Construct` item; a Construct's modifier silently vanishes on export

**Document text** (§3, "a fifth gap"; §5, EACH/ORACLE/LOOP/OPERATOR rows,
Construct-export column: "BROKEN — silent drop"): `_lower_construct_item`
never calls `classify_modifiers` on a `Construct` item, so
`Construct(...) | Each(...)` (or Oracle/Loop/Operator) silently loses its
modifier on export, independently reproduced by building the exact case and
grepping the serialized output for zero trace of the modifier.

**Verification, read**: `_agent_spec.py:793-870` (`_lower_construct_item`).
The `isinstance(item, Construct)` branch is:

```python
if isinstance(item, Construct):
    sub_flow = to_agent_spec(item)
    flow_node = nodes_mod.FlowNode(name=item.name, subflow=sub_flow)
    return [flow_node], [], [], flow_node, flow_node, [(flow_node, False)]
```

`classify_modifiers(item)` is called later in the function, but only after
the `Construct` branch has already `return`ed — i.e., structurally
unreachable for a `Construct` item. Confirmed by direct source read: the
`combo, mods = classify_modifiers(item)` line is textually below the
Construct-branch's `return`, so it is dead code for that item type.

**Verification, repro**: built `parent = Construct("parent", nodes=[seed_node,
sub | Each(over="seed.items", key="item_id")])` where `sub` is a
`Construct(input=Item, output=Out, nodes=[...])`, and ran
`to_agent_spec(parent)`. Inspected the resulting `Flow.nodes` directly (its
`model_dump()` hit an unrelated pyagentspec serialization-context error
unrelated to this claim, so I iterated `flow.nodes` instead):

```
StartNode parent__start
ToolNode seed
FlowNode worker_sub
EndNode parent__end
```

No `MapNode` (the primitive `_lower_each` produces for a `Node`-level Each,
confirmed by reading `_lower_each`'s Node-path). `worker_sub` exports as a
bare `FlowNode` — the `Each` modifier is completely absent from the exported
graph, with **no error and no metadata marker of any kind**. This is a live,
independently-reproduced silent data-loss bug exactly as described.

**Verdict: CONFIRMED.**

---

## Summary

| # | Claim | Verdict |
|---|---|---|
| 1 | Construct as non-entry Portal mesh member works today (do0d9) | CONFIRMED |
| 2 | Two-site Construct-as-mesh-entry bug, sites must land together | PARTIALLY CONFIRMED (bug pairing and stale comment confirmed; the specific "silent runtime crash / undeclared-field attribute error" mechanism is refuted by repro — actual mechanism is silent state-key divergence tolerated by `StateBus`'s default-on-missing getattr, not a raising crash) |
| 3 | `PORTAL_OPERATOR` silently drops the Operator HITL gate on export | CONFIRMED |
| 4 | `_lower_construct_item` never calls `classify_modifiers` on a Construct item → silent modifier drop | CONFIRMED |

3 of 4 claims held up exactly as written; claim 2's headline conclusion (both
sites must land together) is correct, but its specific failure-mode
description is wrong in a way that matters for how the fix is tested — this
should be corrected in the master document and in any ticket derived from it
before implementation starts, since a test written to catch an
`AttributeError` would not catch the actual (silent, worse) bug.
