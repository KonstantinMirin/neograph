# Architecture audit: existing reusable capabilities vs. actual reuse (2026-07-27)

Angle: hunt for already-correct traversal/classification/grouping utilities in
`src/neograph/` that Agent Spec export (`_agent_spec.py`) / import (`loader.py`)
or compiler.py's own Portal-mesh handling SHOULD reuse but reimplement or skip.
All claims below verified against source at the given line numbers, not assumed
from the prior design docs.

## Inventory

1. **`classify_modifiers` + `_COMBO_MAP`** (`modifiers.py:91-165`) — single
   source of truth for "which ModifierCombo does this item have". **Well
   reused**: `compiler.py`, `state.py`, `runner.py`, `_wiring.py`,
   `_input_shape.py`, `_subconstruct.py`, `_state_write.py`, `_fan_agent.py`,
   and `_agent_spec.py:844` (per-item classification) all call it.
   **Gap**: `_agent_spec.py:to_agent_spec` (line ~961-967) does NOT call it to
   detect Portal mesh membership. It hand-rolls
   `isinstance(item, Node) and item.modifier_set.portal is not None and not
   item.modifier_set.portal.is_dispatch` instead of
   `classify_modifiers(item)[0] in (ModifierCombo.PORTAL, PORTAL_OPERATOR)`.
   The `isinstance(item, Node)` gate silently **drops Construct mesh members
   from export** — matching the maintainer's cited "Construct CAN be a
   non-entry Portal mesh member today, but Agent Spec doesn't know it" gap,
   verified live in code, not just in the prior docs.

2. **`_group_portal_members`** (`modifiers.py:727`) — the shared "isolate one
   named Portal mesh's contiguous members" grouping helper. **Well reused** by
   `_ir_normalize.py:278`, `_validation_portal.py:88`, `_wiring.py:734`
   (`_contiguous_portal_mesh`, the compiler's own mesh-entry walk), and
   `state.py:266`. **Gap**: `_agent_spec.py:to_agent_spec` does NOT call it.
   Its ad hoc filter collects `mesh_members` as *every* portal-tagged Node
   anywhere in `all_items`, with no grouping by mesh name. Consequence: two
   distinct, differently-named adjacent Portal meshes in one construct (a
   shape `_group_portal_members` exists specifically to disambiguate — see
   `_wiring.py:729-733`'s own comment on this) would pass export's
   `len(mesh_members) == len(all_items)` check and get merged into a single
   `Swarm`, silently wrong. **Second gap**: `loader.py` never calls
   `_group_portal_members` (or `_check_portal_mesh`) either — it reconstructs
   Portal meshes via its own marker/edge-lookahead walk (`_group_flow_items`,
   `loader.py:559`) with no cross-check against the shared grouping/validation
   rule the native (`@node`/programmatic) path enforces.

3. **`_check_portal_mesh`** (`_validation_portal.py:40`) — the one
   construct-assembly-time gate for every Portal mesh rule. Called from
   exactly one site, `_construct_validation.py:348`. `_agent_spec.py`'s
   docstring (line 894) explicitly says it *trusts* `_check_portal_mesh`
   already ran at assembly time for a construct that exists in memory — true
   for export (the construct was already validated), but **loader.py builds a
   brand-new `Construct` from imported spec data and never re-runs
   `_check_portal_mesh`** (or any equivalent) against the reconstructed IR. A
   structurally malformed reconstruction (wrong contiguity, mixed groups) can
   silently produce a `Construct` that was never actually checked by the rule
   that governs every other construction path.

4. **`iter_with_arms` / `iter_item_slots`** (`_ir_branch.py:71,109`) — the
   arm-aware walk (expands `_BranchNode` sentinels) and its write-back
   counterpart. **This is the exemplary case** — reused correctly by
   `compiler.py`, `runner.py`, `lint.py`, `verify.py`, `tool.py`,
   `_subconstruct.py`, `_fan_agent_wrap.py`, `_construct_validation.py`,
   `__main__.py`, AND `_agent_spec.py` (module docstring at line 5-6
   explicitly cites reusing it). No gap found here — cite as the pattern the
   other three items above should copy.

5. **`iter_nodes`** (`construct.py:58`) — plain (non-arm-aware) node walker.
   Used by `_llm_runtime.py`, `compiler.py`, `runner.py`. Not used by
   `_agent_spec.py`/`loader.py`, but that's correct: they need arm-awareness,
   so `iter_with_arms` (a superset) is the right choice — not a gap.

6. **`spec_types.py`** (`model_to_agent_spec_properties`,
   `agent_spec_properties_to_types`, `_structural_type_name`,
   `_property_to_field_type`) — the type/schema conversion layer. **Well
   reused symmetrically**: `_agent_spec.py:50,278` (export) and
   `loader.py:50,86-88` (import) both call the same functions. This is the
   second good-practice example, alongside `iter_with_arms` — cite both as
   the template for fixing items 1-3.

7. **`_classify_input_shape`/`InputShape`** (`_input_shape.py`) — a
   runtime/state-bus-extraction concern (Each-item vs. fan-in-dict vs.
   loop-reentry shape at actual execution). Correctly out of scope for
   compile-time export/import; not a gap.

## Bottom line

The codebase already has exactly the right shared primitives
(`classify_modifiers`, `_group_portal_members`, `_check_portal_mesh`) proven
correct by the compiler/validator/normalizer path, and two other subsystems
(`iter_with_arms`, `spec_types.py`) show the reuse pattern works well when
applied. Agent Spec's Portal-mesh handling in both directions (`_agent_spec.py`
export, `loader.py` import) is the one place that reimplements ad hoc
mesh-membership detection instead of calling the three proven helpers —
directly explaining the Construct-as-mesh-member export gap and the
unvalidated-reconstruction import gap the maintainer flagged.
