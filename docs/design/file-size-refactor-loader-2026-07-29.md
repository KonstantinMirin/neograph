# File-split proposal: `src/neograph/loader.py` (1393 lines)

Read in full (all 1394 lines). Read-only research; no code changed.

## 1. Responsibility map

The module docstring claims one job ("YAML/JSON pipeline spec -> Construct
IR") but the file actually holds **two unrelated importers** plus their
shared low-level helpers:

| Lines | Section | What it is |
|---|---|---|
| 1-72 | header/imports | shared imports for both importers |
| 74-345 | Agent-Spec node/agent helpers | `_import_agent_spec_import_classes`, `_agent_spec_props_to_type`, `_inputs_from_data_edges`, `_dict_form_inputs_from_props`, `_augment_inputs_from_prompt_marker`, `_tools_from_marker`, `_tools_from_foreign_agent`, `_node_from_spec_agent`, `_REMOTE_AGENT_ENDPOINT_ATTRS`, `_reconstruct_agent_node` — property/type reconstruction + agent/act node rebuilding |
| 348-782 | Primitive + modifier-group reconstruction ("the walk") | `_reconstruct_primitive_node`, `_reconstruct_oracle_group`, `_reconstruct_each_node`, `_reconstruct_loop_item`, `_reconstruct_operator_primary`, `_reconstruct_fused_each_oracle_node`, `_subflow_inner_nodes`, `_subflow_oracle_group`, `_trailing_operator`, `_group_flow_items` — this IS the `PrimaryShape`/`COMBO_DECOMPOSITION` dispatch that Phase 8 (neograph-s7zt3.11) is extending |
| 782-1035 | Swarm / Portal mesh reconstruction | `_swarm_agents_ordered`, `_synthesize_swarm_payload`, `_swarm_trigger`, `_flow_member_to_construct`, `_reconstruct_swarm_mesh`, `_reconstruct_swarm_mesh_with_operator_gates` — entirely Swarm-specific, only reachable from the two Swarm dispatch points inside `from_agent_spec` |
| 1036-1152 | `from_agent_spec` | top-level Agent-Spec-Flow -> Construct entry point; dispatches into all three clusters above |
| 1155-1394 | **Declarative spec loader** | `load_spec`, `_parse_input`, `_validate_spec`, `_build_construct`, `_resolve_tool`, `_build_node`, `_build_sub_construct`, `_apply_modifiers` — the YAML/JSON path the module docstring describes. Zero call edges into the Agent-Spec importer above it (only shared symbol: `Construct`, `Oracle`/`Each`/`Loop`/`Operator`, `lookup_type`). |

So really: **lines 74-1152 (~1080 lines) are the pyagentspec `Flow`
import bridge** (the inverse of `_agent_spec.py`'s `to_agent_spec`, itself
1561 lines), and **lines 1155-1394 (~240 lines) are the actual "spec
loader"** the file is named and documented for. They coexist here for
historical reasons (both "produce a Construct from something external"),
not because they share logic.

## 2. Candidate extractions

### (a) Declarative YAML/JSON spec builder -> new module `_spec_loader.py` (or `spec_loader_build.py`)
- **Moves**: `load_spec`, `MAX_SPEC_SIZE`, `_parse_input`, `_validate_spec`, `_build_construct`, `_resolve_tool`, `_build_node`, `_build_sub_construct`, `_apply_modifiers` (lines ~1155-1394).
- **Removes**: ~240 lines from `loader.py`.
- **Imports needed in new module**: `json`, `yaml`, `pydantic.ValidationError`, `Path`, `_spec_schema.{ConstructSpec,NodeSpec,Spec,ToolSpec}`, `_state_keys.StateKeys`, `conditions.parse_condition`, `construct.Construct`, `errors.ConfigurationError`, `modifiers.{Each,Loop,Operator,Oracle}`, `naming.field_name_for`, `node.Node`, `spec_types.{load_project_types,lookup_type}`, `tool.Tool`, `_normalize.{normalize_outputs,primary_output_field}` — all already top-level imports in the current file, just need re-homing. `loader.py` would re-export `load_spec` (or the package `__init__.py` import path is updated directly — check `neograph/__init__.py`'s `from .loader import load_spec` and repoint it).
- **Why this is the right seam**: no call graph between this cluster and `from_agent_spec`'s cluster. Moving it is a pure cut-and-paste + import fixup, zero behavior change.

### (b) Swarm/Portal mesh reconstruction -> new module `_agent_spec_swarm_import.py` (sibling of `_agent_spec.py`'s own `_lower_portal_mesh_to_swarm`)
- **Moves**: `_swarm_agents_ordered`, `_synthesize_swarm_payload`, `_swarm_trigger`, `_flow_member_to_construct`, `_reconstruct_swarm_mesh`, `_reconstruct_swarm_mesh_with_operator_gates` (lines ~782-1035).
- **Removes**: ~250 lines from `loader.py`.
- **Call surface into the rest of the file**: only `_flow_member_to_construct` calls back into `from_agent_spec` (recursion for a Flow-typed Swarm member) — a normal one-line import of `from_agent_spec` from the new module resolves this (no cycle back the other way, since `from_agent_spec` calls into this cluster via plain function calls dispatched at the top, which become imports in the other direction). Needs `_import_agent_spec_property_classes`/`_agent_spec_props_to_type` from cluster (a) below — cross-import, still one-directional.
- **Why safe**: Phase 8 (fusion ModifierCombo lowering) is about `_group_flow_items`/`_reconstruct_each_node`/`_reconstruct_fused_each_oracle_node`/the `PrimaryShape` dispatch in `from_agent_spec` — it has no documented touch point on Swarm/Portal-mesh reconstruction. Confirmed by reading `_reconstruct_swarm_mesh*`: none of it references `COMBO_DECOMPOSITION`, `is_each_oracle_fused`, or `PrimaryShape`.

### (c) Primitive/modifier-group reconstruction (lines 348-782) and `from_agent_spec` itself (1036-1152)
- This is the ~830-line core Phase 8 is actively extending (`_group_flow_items`, the `COMBO_DECOMPOSITION` match in `from_agent_spec`, `_reconstruct_each_node`/`_reconstruct_fused_each_oracle_node`). **DEFER** — do not touch until Phase 8 lands. When it does land, this is the natural next split target (e.g. `_agent_spec_import_walk.py` for the recognize/classify/dispatch, keeping `from_agent_spec` itself as the public entry in whatever file remains named `loader.py`/`_agent_spec_loader.py`).

### (d) Property/type + agent-node helpers (lines 74-345)
- Feeds both cluster (c) and cluster (b) (`_agent_spec_props_to_type`, `_dict_form_inputs_from_props`, `_augment_inputs_from_prompt_marker` are called from primitive/oracle/each/loop reconstruction; `_tools_from_marker`/`_node_from_spec_agent`/`_reconstruct_agent_node` are called from both the primitive walk and the Swarm builder). A real split here needs to decide which module owns the shared low-level helpers so both (b)'s new module and (c) can import them without a cycle. **DEFER** — this is exactly the kind of "where do shared internals live" design question the epic's own Phase 8 work will also need answered (fusion logic touches some of these same helpers), so doing it now risks a rebase conflict with s7zt3.11. Bundle this decision into whatever follow-up design pass handles (c).

## 3. SAFE NOW vs DEFER

**SAFE NOW** (mechanical, no behavior change, does not touch anything Phase 8 will edit):
1. Extract (a) the declarative spec-loader tail -> `_spec_loader.py`. ~240 lines out, zero coupling to the agent-spec importer.
2. Extract (b) the Swarm/Portal mesh reconstruction -> `_agent_spec_swarm_import.py`. ~250 lines out, one clean one-directional import back to `from_agent_spec` for the Flow-member recursion case; no reference to the `PrimaryShape`/`COMBO_DECOMPOSITION` machinery Phase 8 touches.

Combined: **~490 lines removed**, `loader.py` drops from 1393 -> ~900. Both land independently of Phase 8 and, if anything, make its diff smaller (fewer unrelated lines in the file it has to scroll past).

**DEFER** (needs its own design pass, do not start until Phase 8 / s7zt3.11 lands):
3. Splitting the primitive/modifier-group reconstruction walk (348-782) and `from_agent_spec`'s dispatcher (1036-1152) into their own module(s) — this is the exact code Phase 8 is extending; touching it now would force a rebase/merge-conflict onto active epic work.
4. Deciding the home for the shared low-level helpers (74-345: property<->type reconstruction, prompt-marker/tool-marker restoration, agent-node builder) once (3) has a shape, since both the deferred walk and the now-extracted Swarm module need them and a wrong choice today could require re-plumbing after Phase 8 lands.

## 4. Duplication notes

- No literal duplication *within* `loader.py` itself.
- `loader.py`'s entire Agent-Spec-import half (lines 74-1152, ~1080 lines) is the **structural mirror** of `src/neograph/_agent_spec.py`'s export half (1561 lines, `_lower_node`/`_lower_oracle`/`_lower_each`/`_lower_loop`/`_lower_operator`/`_lower_construct_item`/`_lower_portal_mesh_to_swarm`/`to_agent_spec`). This is intentional per-function "the inverse of X" pairing (documented in nearly every docstring here), not copy-paste duplication to DRY up — but it IS a standing coupling risk: `COMBO_DECOMPOSITION`/`PrimaryShape`/`is_each_oracle_fused` (in `modifiers.py`) are the single shared source of truth both sides already correctly defer to, so the two files stay in sync only as long as every future modifier/combo addition updates both `_lower_*` (export) and `_reconstruct_*` (import) in lockstep. Not an extraction target, but worth flagging for whoever designs (3)/(4): a possible DEFER-tier idea is a shared `_agent_spec_shared.py` for the property<->type helpers actually used by BOTH `_agent_spec.py` and `loader.py` (need to verify overlap — `_agent_spec_props_to_type`-equivalent logic likely has a sibling in `spec_types.py` already, since both files import `spec_types.agent_spec_properties_to_types`/`load_project_types`/`lookup_type`).
- No duplication found against `modifiers.py` (spec_types.py: 521 lines, not overlapping in content, just imported from).
