# Architecture audit: `loader.py` import-side inventory (2026-07-27)

Angle: complete inventory of `src/neograph/loader.py`'s current Agent-Spec ->
neograph reconstruction behavior, verified against real source (both
`loader.py` and its export-side counterpart `_agent_spec.py`), with every
export/import asymmetry flagged. This does not re-derive the systemic
9-module ModifierCombo duplication already recorded in
`docs/design/modifier-combo-single-source-of-truth-2026-07-27.md`; it
verifies the loader-specific claims against current code and adds new
findings (the Swarm/Construct-mesh-member gap in particular).

## 1. `_group_flow_items` — verified: single forward pass, fixed lookahead, recognize-and-emit in one step

Confirmed by reading `loader.py:559-638`. One index `i`, one `while i < n`
loop, no backtracking:

- Skips `StartNode`/`EndNode`.
- `metadata[_MARK_MODIFIER] == "oracle"`: greedily consumes every
  contiguous node sharing the SAME `_MARK_GROUP_ID` (unbounded lookahead,
  but still a single forward scan, no re-visit).
- `== "each"`: single MapNode, no lookahead needed (Each is one Agent Spec
  node).
- `in ("loop", "operator")`: a floating check node with no preceding body
  falls back to `"bare"` — this branch exists only for the "marker present
  but structure doesn't match" defensive case (a body/check pair is always
  supposed to be consumed together when the loop hits the BODY node, one
  step earlier).
- Otherwise, a bare node peeks at `nodes[i+1]` (and, for Operator,
  `nodes[i+2]`) and cross-checks it against `flow.control_flow_connections`
  before committing to "loop" or "operator" — it does not trust the
  metadata marker alone, it verifies the specific edges
  (`from_branch="continue"` back-edge for Loop; `from_branch="pause"`
  edge to a third node for Operator) actually exist. If the edge check
  fails, it falls through to `"bare"`.

This is a **recognize-and-commit** design: each position is classified
exactly once, using only a fixed amount of lookahead (1 node for Loop, 2
for Operator, unbounded-but-forward for Oracle group members via shared
group id). It never re-groups previously-emitted items. This matches the
existing characterization; independently verified.

## 2. `_reconstruct_primitive_node` — bare Node from `LlmNode`/`ToolNode`/`AgentNode`

- `outputs` via `_agent_spec_props_to_type`.
- `inputs`: prefers `_inputs_from_data_edges` (DataFlowEdges targeting this
  node, keyed by producer name — this is how dict-form `Node.inputs` gets
  reconstructed), falls back to `_agent_spec_props_to_type(spec_node.inputs)`
  (single-type) for a self-contained node with no external edges (e.g.
  Each's inner node).
- Dispatches by `type(spec_node).__name__`:
  - `AgentNode` -> `_reconstruct_agent_node` (see §5), plus an extra step:
    if a `neograph/prompt_spec` marker is present, `inputs` is augmented
    with the ORIGINAL dict-form inputs the Option-F translation flattened
    away (`_augment_inputs_from_prompt_marker`).
  - `LlmNode` -> `mode="think"`, prompt from the `neograph/prompt_spec`
    marker if present (untranslated `${var}` form) else the translated
    `prompt_template`; same input-augmentation as above.
  - `ToolNode` -> `mode="scripted"`, `scripted_fn=spec_node.tool.name`.
  - anything else -> fail-loud `ConfigurationError` ("unsupported type ...
    for primitive import").
- **Never handles `FlowNode` itself** — that dispatch lives one level up,
  only in `from_agent_spec`'s `"bare"` branch (§7), NOT inside
  `_reconstruct_primitive_node`. This matters: every OTHER
  `_reconstruct_*` helper that calls `_reconstruct_primitive_node` on an
  inner/body/variant/primary spec node (Each's inner, Loop's body,
  Operator's primary, Oracle's base variant) therefore **cannot** accept a
  `FlowNode` (i.e., a Construct) in that position — only `LlmNode`/
  `ToolNode`/`AgentNode` are legal there. Confirmed this is symmetric with
  export: `_lower_each`/`_lower_loop`/`_lower_operator` in `_agent_spec.py`
  all call `_lower_node(node)` on a bare `Node`, never route a `Construct`
  through Each/Loop/Operator lowering (`classify_modifiers` in
  `_lower_construct_item` is only reached for `Node` instances; a
  `Construct` item is lowered to a bare `FlowNode` upstream of that
  dispatch, before any modifier is considered). So there is **no**
  Each/Loop/Operator-wrapping-a-Construct case on export either — this
  particular asymmetry does not exist; both sides agree Each/Loop/Operator
  only ever wrap atomic Nodes.

## 3. `_reconstruct_oracle_group` — Oracle variant+merge group

- Validates `len(variant_nodes) == spec["n"]`; on mismatch, WARNs and
  returns `None` (caller falls back to importing every node in the group
  as bare primitives — never silently trusts a stale marker).
- Dispatches the BASE variant's Agent-Spec type (`AgentNode`/`LlmNode`/
  `ToolNode`) exactly mirroring `_lower_oracle`'s neograph-m57mn
  per-`node.mode` dispatch (verified: `_lower_oracle` at
  `_agent_spec.py:497` calls `_lower_generation_step` for every mode, so a
  scripted-mode Oracle variant lowers to whatever `_lower_generation_step`
  produces for scripted, and the import's `base_cls` branch covers
  `AgentNode`/`LlmNode`/`ToolNode` symmetrically). This 3-way dispatch is
  in agreement on both sides.
- `merge_prompt` vs `merge_fn` reconstructed via the `_MARK_ORACLE_SPEC`
  marker's `n`/`models`/`merge_prompt`/`merge_model`/`merge_fn` keys — a
  faithful inverse of what `_lower_oracle` stamps.
- **Known, intentional non-round-trip**: `Oracle.merge_pre_process` /
  `merge_post_process` / `merge_fallback` (Python callables) make export
  itself fail loud (`_lower_oracle` raises `ConfigurationError` if any is
  set) — so these fields can never reach a Flow in the first place, and
  the importer correctly never needs to reconstruct them. Not an
  asymmetry (export refuses before the round trip even starts), but worth
  recording as a hard capability boundary, not a bug.

## 4. `_reconstruct_each_node` — MapNode -> Each-modified Node

- Requires the sub-flow to contain EXACTLY 1 inner node (excluding
  Start/End); anything else is `ConfigurationError` (fail-loud, matches
  `_lower_each`'s single-inner-node invariant on export).
- Reconstructs the inner node via `_reconstruct_primitive_node` (so, per
  §2, the inner must be `LlmNode`/`ToolNode`/`AgentNode` — never a nested
  `FlowNode`/Construct; symmetric with export as established above).
- Un-groups dict-form (`map_over`) inputs from dotted Property titles via
  `_dict_form_inputs_from_props` to restore type-identity with the
  producer's list-element type (neograph-3lk2l) — but explicitly does
  **not** overwrite inputs with the MapNode's own external DataFlowEdges,
  since those name the COLLECTION producer, not the per-item shape. This
  is a correct, deliberate divergence from the "use the group's own edges"
  rule other reconstructors follow, and the docstring explains why.
- Output type is taken from the wrapped INNER node's outputs (the MapNode
  itself never declares `outputs=`) — matches `_lower_each` which never
  sets `MapNode.outputs`.

## 5. `_reconstruct_agent_node` / `_node_from_spec_agent` — AgentNode -> agent/act Node

Three-way dispatch, verified against `_agent_spec.py`'s agent/act +
Swarm-agent lowering:

1. `neograph/agent_spec` marker present (or `agent_type == "Agent"` with no
   marker but still constructible via `_node_from_spec_agent`'s
   marker-present branch) -> lossless inversion of mode/prompt/model/
   tools(+budget/config/idempotent)/gate_tools_when/context.
2. Marker absent, plain `Agent` -> best-effort `mode="agent"` (read-only,
   conservative) from `system_prompt`/`llm_config.model_id`/`tools`
   (`_tools_from_foreign_agent`, itself marker-aware per-`ServerTool`).
3. Marker absent, `RemoteAgent`/`A2AAgent`/`OciAgent` -> WARN + name-bound
   `mode="scripted"` stand-in (`scripted_fn=name`), with a private
   `_remote_agent_endpoint` attribute stashing the endpoint fields per
   `_REMOTE_AGENT_ENDPOINT_ATTRS` (verified against the installed SDK,
   per-family attribute names, not a blind two-field getattr).
4. Anything else (e.g. a ServerTool-as-agent, orchestrator-side surface)
   -> fail-loud `ConfigurationError`.

This matches the ratified `agent-spec-ratification-2026-07-13.md` §3b
policy and is internally consistent with what `_agent_spec.py` actually
emits for agent/act nodes (a marker-carrying `AgentNode` wrapping a real
`Agent`). No asymmetry found here — export always stamps the marker for
neograph-originated agent/act nodes, so gap-1 lossless reconstruction is
the common case; gaps 2-4 exist purely to consume FOREIGN (non-neograph)
Agent Specs, which by definition carry no marker.

## 6. `_reconstruct_loop_item` / `_reconstruct_operator_item`

Straightforward inverses of `_lower_loop`/`_lower_operator`:

- Loop: body reconstructed via `_reconstruct_primitive_node`, condition via
  `parse_condition` (string form only — matches export's fail-loud on
  callable `Loop.when`, so import never has to handle a callable-valued
  `when` either; symmetric).
- Operator: primary reconstructed via `_reconstruct_primitive_node`;
  inputs land on the primary (mirrors `to_agent_spec`'s
  `input_targets` routing, which sends Operator's external inputs to the
  PRIMARY node, never the property-less check `BranchingNode`).
- Both correctly ignore the extra structural nodes Agent Spec requires
  (Loop's `BranchingNode`, Operator's `BranchingNode` + `InputMessageNode`
  pause node) — those exist only to carry control-flow shape + the
  `neograph/*_spec` marker, and are discarded on reconstruction (the
  neograph `Loop`/`Operator` modifier IR has no analog for them).

## 7. `from_agent_spec` top-level dispatch

- Import-guards via `_import_agent_spec_import_classes()` (same
  guard-copy pattern as `_agent_spec._import_agent_spec_flow_classes()`,
  keeping `src/neograph` core Agent-Spec-free by default).
- `type(flow).__name__ == "Swarm"` -> dispatches to `_reconstruct_swarm_mesh`
  BEFORE the `Flow.nodes` walk (a Swarm is a top-level `AgenticComponent`,
  never itself a Flow node) — verified this exactly mirrors export's
  `to_agent_spec`, which likewise special-cases the Portal-mesh path
  before the normal per-item walk (`_agent_spec.py:960-977`).
- Otherwise walks `_group_flow_items(flow)` and for `"bare"` items with
  `type(spec_node).__name__ == "FlowNode"`, RECURSES: `from_agent_spec(spec_node.subflow)`,
  renames the resulting `Construct` to the node's name, and registers its
  `.output` in `output_types` — this is the **only** place `loader.py`
  reconstructs a `Construct` (sub-construct) as a pipeline item. All other
  `"bare"`/`"oracle"`/`"each"`/`"loop"`/`"operator"` branches produce a
  bare `Node` (see §2-6). No modifier-wrapped `FlowNode` is ever produced
  or expected — consistent with §2's finding.
- No whole-pipeline `neograph/source` blob is read; fidelity rides
  entirely on the per-group markers `to_agent_spec` actually emits, so a
  Flow authored by a third party (no markers at all) still imports as
  plain primitives + sub-flows, never errors purely for lacking markers.

## 8. `_reconstruct_swarm_mesh` — the confirmed Swarm/Construct-mesh-member asymmetry

`_reconstruct_swarm_mesh` (`loader.py:689-734`) is the import-side inverse
of `_lower_portal_mesh_to_swarm` (`_agent_spec.py:880-945`). Verified in
detail against both:

- Import always builds **every** mesh member as an agent-mode `Node` via
  `_node_from_spec_agent(agent.name, agent, None, {"handoff": payload}, payload)`
  (`loader.py:715`) — there is no branch anywhere in `_reconstruct_swarm_mesh`
  that could produce a `Construct` mesh member. This is a hard, structural
  limitation of the current importer: **a Swarm always reconstructs to an
  all-`Node` Portal mesh**, never one with a `Construct` (sub-pipeline)
  member.
- On the export side, `_lower_portal_mesh_to_swarm` independently makes the
  SAME assumption but for a different, confirmed-buggy reason: `to_agent_spec`'s
  mesh-detection filter (`_agent_spec.py:961-967`) is
  ```python
  mesh_members = [item for item in all_items
                  if isinstance(item, Node) and item.modifier_set.portal is not None
                  and not item.modifier_set.portal.is_dispatch]
  ```
  This is **Node-only** (`isinstance(item, Node)`). A `Construct` carrying
  a `Portal` modifier — which `_validation_portal.py`'s `_check_portal_mesh`
  explicitly ADMITS as a first-class non-entry mesh member per the do0d9
  fix (`_validation_portal.py:114-119`: "a Construct member is ADMITTED as
  a first-class mesh member") — is silently EXCLUDED from `mesh_members`
  by this filter. The immediate consequence: `len(mesh_members) !=
  len(all_items)` fires, and export raises `"construct {name} mixes a
  Portal peer mesh with non-mesh nodes"` — a **false-positive rejection**
  of a construct that `_check_portal_mesh` itself already certified as a
  valid, uniform Portal mesh. This is exactly the maintainer's standing
  complaint pattern: a capability admitted by the validator is not reached
  by the export dispatch that is supposed to serialize anything the
  validator accepts.
- Even if that filter were fixed to admit Construct members structurally,
  `_lower_portal_mesh_to_swarm` itself would still fail on a Construct
  member: it unconditionally calls `member.prompt`, `member.inputs`,
  `member.name` inside `_make_agent(member, ...)` and reads
  `entry.modifier_set.portal` off `members[0]` — `Construct` has no
  `.prompt`/`.modifier_set` attributes with the same shape as `Node`, so a
  Construct-as-entry or Construct-as-peer would raise an `AttributeError`
  (or silently wrong behavior if `Construct` happens to define
  differently-typed same-named attributes), not a clean `ConfigurationError`.
  This corroborates the audit's separately-recorded claim that
  "Construct-as-mesh-ENTRY fails despite validator admission" — the failure
  mode is broader: ANY Construct mesh member (entry or peer) breaks
  `_lower_portal_mesh_to_swarm`, only the detection filter's early exit
  currently masks the deeper `_make_agent` incompatibility for the
  peer case.
- **Net asymmetry**: neither the export dispatcher (filters Construct
  members out before reaching the lowering function, then raises a
  mixed-mesh error) nor the lowering function itself (assumes Node
  attributes unconditionally) nor the importer (`_reconstruct_swarm_mesh`
  only ever emits Node members) actually implements the
  Construct-as-mesh-member capability that `_check_portal_mesh` already
  certifies as valid IR. This is a genuine, verified gap — not a
  restatement of the referenced design doc's claim, but an independent
  confirmation reached by reading the current filter/lowering/import code
  together.
- Swarm import is explicitly best-effort/warn (never silent): a
  `warnings.warn` documents the payload-synthesis + name-bound-live-LLM-
  agent downgrade. `max_hops`/`on_exhaust`/`route` ride a
  `neograph/portal_spec` marker on export (`_MARK_PORTAL_SPEC`,
  `_agent_spec.py:939-943`) but **`_reconstruct_swarm_mesh` never reads
  `_MARK_PORTAL_SPEC` back** — confirmed by grep: `_MARK_PORTAL_SPEC` does
  not appear anywhere in `loader.py`. So a neograph-exported Swarm's
  `max_hops`/`on_exhaust`/`route` are silently dropped on re-import (the
  reconstructed `Portal(to=peers)` never passes `max_hops=`/`on_exhaust=`/
  `route=`), even though the marker exists on the wire specifically to
  prevent that loss. This is a second, independently-confirmed round-trip
  gap in the Swarm path (distinct from the Construct-member gap above):
  the marker is written but never read.

## Summary of flagged asymmetries

| # | Asymmetry | Export side | Import side | Status |
|---|---|---|---|---|
| 1 | Construct as Portal mesh member (entry or peer) | `to_agent_spec`'s Node-only mesh filter false-positive-rejects a validator-admitted Construct member; `_lower_portal_mesh_to_swarm` would additionally `AttributeError` on `member.prompt`/`.modifier_set` | `_reconstruct_swarm_mesh` only ever builds `Node` members, never a `Construct` | Confirmed gap on both sides; validator (`_check_portal_mesh`) already admits this shape, neither export nor import implements it |
| 2 | `neograph/portal_spec` marker (`max_hops`/`on_exhaust`/`route`) | written unconditionally by `_lower_portal_mesh_to_swarm` | never read by `_reconstruct_swarm_mesh` (`_MARK_PORTAL_SPEC` absent from `loader.py` entirely) | Confirmed silent-drop-on-reimport gap |
| 3 | Each/Loop/Operator wrapping a Construct (sub-pipeline) instead of a bare Node | never produced (`classify_modifiers` only reached for `Node` items) | never consumed (`_reconstruct_primitive_node` only handles `LlmNode`/`ToolNode`/`AgentNode`) | Symmetric non-capability, not a bug — both sides agree it doesn't exist |
| 4 | `Oracle.merge_pre_process`/`merge_post_process`/`merge_fallback` | export fail-loud refuses before serializing | import never needs a case for it | Symmetric, by design |
| 5 | `Loop.when` as a callable | export fail-loud refuses | import only parses string conditions | Symmetric, by design |
| 6 | Stale/mismatched Oracle `n` marker | N/A (export always emits consistent `n`) | import WARNs + falls back to primitive-level import per group | Defensive, not an asymmetry — protects against hand-edited Flows |

Everything else audited (`_reconstruct_primitive_node`, `_reconstruct_oracle_group`,
`_reconstruct_each_node`, `_reconstruct_agent_node`/`_node_from_spec_agent`,
`_reconstruct_loop_item`, `_reconstruct_operator_item`, `_group_flow_items`,
`from_agent_spec`'s top-level dispatch) is a faithful, verified inverse of
its `_agent_spec.py` export counterpart, with fail-loud (never silent)
behavior on any structural mismatch between a marker and the actual Flow
shape around it.
