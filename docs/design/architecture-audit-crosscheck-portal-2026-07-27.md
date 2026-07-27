# Architecture audit — Portal end to end, reconciled ground truth

Reconciles `architecture-audit-portal-ir-2026-07-27.md` (IR),
`architecture-audit-portal-compilation-2026-07-27.md` (compilation),
`architecture-audit-modifiercombo-consumers-2026-07-27.md` (consumer sweep),
`architecture-audit-existing-capabilities-2026-07-27.md` (reuse gaps). All four
agree on substance; below resolves the outstanding NEEDS CROSS-CHECK flags
against real source and states one concrete fix plan. No disagreements found
between the four docs — they are complementary angles on the same ground
truth, and the compilation doc's line-level tracing is strictly more precise
than the IR doc's on C1 (both reach the same verdict; compilation doc actually
executes the trace).

## IR shape (uncontested)

`Node.handoff_param`/`handoff_channel` — sole writer `_ir_normalize.py`, no
`Construct`-level analog by deliberate do0d9 decision (channel key threaded
through the compile-kwarg closure instead). `ModifierCombo.PORTAL`/
`PORTAL_OPERATOR` from `modifiers.py`'s `_COMBO_MAP`. `Portal.is_dispatch` is
the sole mode discriminator — **guard confirmed**: `TestPortalDispatchDiscriminationMonopoly`
in `tests/test_guards_assembly.py:142-187` (AST-level, bans inline
`route == "decide"` outside `modifiers.py`; resolves portal-ir doc's C4).

## The do0d9 admission, exact scope

`_check_portal_mesh`/`_check_one_mesh_group` (`_validation_portal.py`) admit a
`Construct` as a mesh member with NO entry/non-entry distinction anywhere in
the validator body — every rule (uniform payload, contiguity, route typing)
applies identically regardless of position. The do0d9 *design doc* said entry
should stay Node-only ("out of scope for v1, flag as unsupported"), but that
intent was **never implemented as an assembly-time check** — validator
behavior and design-doc intent diverged silently. This is the single most
important reconciliation point across all four docs: the bug is not "the
validator over-admits," it's "the validator was already correct/permissive,
downstream never caught up, and a stale compiler comment asserts a rejection
that doesn't exist."

## C1 — Construct-as-mesh-entry: two-site bug, not one

The compilation doc's trace is authoritative here (it reads further than the
IR doc, which flagged the same finding less precisely):

1. `compiler.py:254`'s `isinstance(item, Node)` gate is the only thing
   preventing a Portal-Construct entry from reaching `_add_portal_mesh`.
   `_contiguous_portal_mesh` and `_add_portal_mesh` (`_wiring.py`) are already
   fully generic over `Node | Construct` — verified line-by-line, zero changes
   needed there. `runner.py`'s hop-cost/member-id helpers are also already
   generic. So 3 of 5 relevant modules are correct today.
2. **A second, independent bug the IR doc did not surface**: `state.py:261-265`
   builds `portal_members` by filtering `nodes_only` only (excluding
   `sub_constructs`), unlike the Oracle/Each detection two blocks below it
   which correctly uses `nodes_only + sub_constructs`. If only the
   compiler.py gate is widened, a Construct-entry mesh compiles and wires
   successfully, then crashes at first invocation with an attribute error
   reading an undeclared state field (`make_portal_subgraph_fn` in
   `factory.py:402-403` reads `StateKeys.handoff_payload/hops(entry_field)`,
   never allocated for this mesh). **Fixing compiler.py alone is worse than
   not fixing it** — it converts a loud compile-time `CompileError` into a
   silent-until-runtime crash, which the project's own north star (fail-loud
   over fail-soft) would treat as a regression, not progress.

**Concrete fix** (both required, in this order per the compilation doc's TDD
plan): widen `compiler.py:254` to `isinstance(item, (Node, Construct))` for
the peer-mode Portal branch only (dispatch mode at line 259 stays Node-only —
legitimate scope restriction, a dispatch node runs a body to synthesize a flow,
a Construct has no single body); widen `state.py:261-265`'s member source to
`nodes_only + sub_constructs`; update the now-stale "already rejected at
assembly" comments at `compiler.py:552-560` and the state.py mirror arm.
Estimated ~10-20 lines across 2 files. Add a runtime-invoking test (not just a
compile-only fixture) — a compile-only fixture would miss the state.py bug
class entirely.

## C2 — Agent Spec export: filter gap resolved, AND its "NEEDS CROSS-CHECK" is now answered

`to_agent_spec`'s `mesh_members` filter (`_agent_spec.py:948-967`) is
`isinstance(item, Node)`-gated, so a legal Node-entry + Construct-non-entry
mesh (the exact do0d9 spiked shape) is misclassified as "mixed" and rejected.
The existing-capabilities doc adds the correct diagnosis: the fix isn't a new
mechanism, it's reuse — `classify_modifiers(item)[0] in (PORTAL,
PORTAL_OPERATOR)` (already the source of truth everywhere else) plus
`_group_portal_members` (already used by the validator, `_ir_normalize.py`,
and `_wiring.py`) instead of the hand-rolled Node-only + ungrouped filter.

**The portal-ir doc's NEEDS CROSS-CHECK — "would `_lower_portal_mesh_to_swarm`
also choke on an actual Construct member?" — is answered here, verified
against source: yes, it would, but the fix already exists elsewhere in the
same file.** `_lower_portal_mesh_to_swarm` (`_agent_spec.py:880-919`)
unconditionally calls `_make_agent(member, ...)` for every mesh member, and
`_make_agent` reads `node.prompt`/`node.tools` (`_agent_spec.py:389-410`) —
attributes that exist on `Node` but not `Construct`. Fixing the filter alone
would move the crash from a clean `ConfigurationError` to an `AttributeError`
one level deeper, exactly as the portal-ir doc suspected.

But the fix is small and the capability is not missing: `_lower_construct_item`
(`_agent_spec.py:826-829`, the generic per-item lowering used for every
non-mesh construct) ALREADY handles a `Construct` item by recursively calling
`to_agent_spec(item)` to produce a `Flow`, wrapped as `FlowNode` for use
inside a parent `Flow`'s node list. Independently verified via pyagentspec's
own type system (introspected directly, not assumed): `Flow` and `Agent`
share the same base, `AgenticComponent`
(`pyagentspec.flows.flow.Flow.__mro__` / `pyagentspec.agent.Agent.__mro__`
both include `AgenticComponent`), and `Swarm.first_agent`/`relationships` are
typed `AgenticComponent` (confirmed via `Swarm.model_fields`) — not
`Agent`-only. So a `Swarm` can structurally hold a `Flow` produced by
`to_agent_spec(construct_member)` directly as a mesh participant, with no
`FlowNode` wrapper needed (that wrapper is only for embedding inside another
`Flow`'s own node list, not for Swarm membership). **The fix for
`_lower_portal_mesh_to_swarm` is a per-member type branch mirroring
`_lower_construct_item`'s existing branch**: `Node` → `_make_agent` (current
behavior, unchanged); `Construct` → `to_agent_spec(member)` used directly as
the `AgenticComponent` participant. This is a capability-reuse fix, not new
design — the exact pattern the existing-capabilities doc's "template" items
(`iter_with_arms`, `spec_types.py`) already demonstrate working.

Second, independently found gap (existing-capabilities doc): the ungrouped
filter also silently merges two distinct, differently-named adjacent Portal
meshes into one `Swarm` — fixed by the same `_group_portal_members` reuse.

## C3 — handoff_param/handoff_channel export guard: consistent, not a gap (uncontested across docs)

## C5 / loader.py (import direction) — reconciled

Portal-ir doc's C5 flagged this as unverified beyond grep. The consumer-sweep
and existing-capabilities docs fill the gap precisely: `loader.py` reconstructs
Portal meshes via its own marker/edge-lookahead walk (`_group_flow_items`,
`loader.py:559`), building only `Node`-shaped mesh members (`Node(inputs=
{'handoff': Payload}, ...) | Portal(...)`) — there is no reverse path to
reconstruct a `Construct` mesh member because a Swarm `Agent` has no
sub-construct concept in pyagentspec's Swarm shape as currently emitted
(export never produces one either, until C2 is fixed). Additionally, and more
generally: **loader.py never re-runs `_check_portal_mesh` (or any equivalent)
against the `Construct` it reconstructs** — every other construction path
(declarative, `@node`, programmatic) is validated by the one assembly-time
gate; imported IR is not. This is a distinct, real gap the existing-capabilities
doc surfaced that neither Portal-focused doc had flagged: a structurally
malformed round-tripped spec (bad contiguity, mixed groups) can produce a
`Construct` that silently skips the one check every other path gets.

## Consumer-count reconciliation

The modifiercombo-consumers doc's fresh sweep is a superset, not a
contradiction, of the "9-module" prior inventory: it independently confirms
all 9 and adds `_fan_agent.py` (agent/act fan-modifier support check) as a
10th genuine DECOMPOSITION/DISPATCH site, plus documents that `loader.py`
belongs on the list via a structurally different mechanism (marker
pattern-matching, not `classify_modifiers`/`ModifierCombo` calls) — worth
preserving as a distinction in any consolidation, since a future grep-based
audit for `classify_modifiers` usage would miss `loader.py` entirely.

## Net actionable plan (Portal-specific; independent of the wider single-source-of-truth remediation)

1. `compiler.py:254` — widen Node gate to `(Node, Construct)` for peer-mode Portal.
2. `state.py:261-265` — widen `portal_members` source to `nodes_only + sub_constructs`.
3. Update 2 stale "already rejected at assembly" comments (`compiler.py:552-560` and its state.py mirror).
4. `_agent_spec.py::to_agent_spec` mesh filter — replace ad hoc `isinstance(item, Node) and item.modifier_set.portal...` with `classify_modifiers` + `_group_portal_members` (handles both the Construct-drop and the multi-mesh-merge bugs).
5. `_agent_spec.py::_lower_portal_mesh_to_swarm` — branch per member type: `Node` → existing `_make_agent`; `Construct` → `to_agent_spec(member)` used directly as the `AgenticComponent` Swarm participant (no `FlowNode` wrapper).
6. `loader.py` — after reconstructing any `Construct` (Portal or otherwise) from imported spec data, run `_check_portal_mesh`-equivalent validation before accepting it, so imported IR gets the same guarantee every other construction path gets. (Scope check: whether this generalizes beyond Portal is for the broader single-source-of-truth remediation, not this angle.)
7. Dispatch mode (`route="decide"`) is correctly Node-only everywhere and needs no change — it is a bona fide scope boundary (runs a body to synthesize a flow), not a duplicated-logic gap.

Items 1-3 are purely mechanical (~10-20 lines, 2 files). Items 4-5 are also
small once the reusable primitives are identified (this doc identifies them
precisely). Item 6 is the one open design question (what "equivalent
validation" means for a freshly-reconstructed IR generally) — flagged for the
broader investigation, not resolved here.
