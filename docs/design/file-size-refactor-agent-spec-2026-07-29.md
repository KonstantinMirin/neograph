# File-size refactor proposal: `src/neograph/_agent_spec.py`

Date: 2026-07-29
Scope: this file only (one of several files assigned across a repo-wide 500-line-cap effort).
Current size: **1561 lines**.

Active-epic caveat: neograph-s7zt3 Phase 8 (neograph-s7zt3.11, "extending fusion
ModifierCombo lowering to the Construct level") is still open and will keep
editing the Oracle/Each/Loop/Operator lowering + dispatch cluster in this file
(roughly lines 309–1096: `_item_inputs`/`_item_outputs`/`_lower_item_body`,
`_lower_oracle`, `_lower_each`, `_lower_loop`, `_lower_operator`,
`_LoweredItem`, `_lower_construct_item`, and the edge-wiring body of
`to_agent_spec`). Nothing below proposes touching that cluster now — every
SAFE NOW item is chosen specifically because Phase 8's diff does not reach it,
so an extraction lands with zero merge risk and doesn't require re-planning
the epic.

## 1. Responsibility map

| Lines | Cluster | What it is |
|---|---|---|
| 1–95 | Module docstring + imports + `_MARK_*` constants | The Core-Invariant doc comment + the single source of truth for every `neograph/*` metadata marker key (also imported by `loader.py`) |
| 97–118 | `_import_agent_spec_flow_classes` | Function-local pyagentspec import guard (keeps core `src/neograph` free of the `[agent-spec]` extra) |
| 121–163 | `_reject_unrepresentable_fields` | Fail-loud check for callable-valued Node fields with no Agent Spec representation |
| 166–287 | Placeholder translation: `_translate_placeholders`, `_node_translation`, `_is_translation_eligible`, `_prompt_spec_marker` | Option-F `${var}` → `{{ flat }}` rewrite subsystem + its round-trip marker builder |
| 290–323 | `_properties_for`, `_item_inputs`, `_item_outputs` | TypeSpec → pyagentspec `Property` list conversion + Node/Construct I/O-field uniformity helpers |
| 325–339 | `_lower_item_body` | Shared "lower one item's wrapped body" seam (Node → per-mode dispatch; Construct → recursive `FlowNode`) |
| 342–543 | Single-node lowering: `_lower_generation_step`, `_lower_node`, `_make_agent`, `_tool_to_server_tool`, `_agent_spec_marker`, `_make_llm_config`, `_make_server_tool` | The per-`node.mode` (think/agent-act/scripted) dispatch to `LlmNode`/`AgentNode`/`ToolNode`, and the `Agent`/`ServerTool`/`LlmConfig` builders it uses |
| 545–705 | `_lower_oracle` | Oracle → N-variant fan-out + merge lowering |
| 708–787 | `_lower_each` | Each → `MapNode` lowering (plain and Each×Oracle fused) |
| 790–878 | `_lower_loop` | Loop → `BranchingNode` + back-edge lowering |
| 881–911 | `_lower_operator` | Operator → HITL pause composite (check/InputMessageNode) |
| 914–1096 | `_LoweredItem` + `_lower_construct_item` | The big per-item dispatcher: classifies modifiers, routes to the lowering above, applies the Operator postlude |
| 1099–1280 | Portal/Swarm export: `_lower_portal_mesh_to_swarm`, `_is_peer_mesh_member` | Portal mode-(a) peer mesh → top-level `Swarm` export (entirely orthogonal to Oracle/Each/Loop) |
| 1282–1562 | `to_agent_spec` | Top-level orchestrator: mesh-vs-Flow branch, per-item lowering loop, control/data-edge wiring (incl. the `_emit_input_edges` closure), Start/End synthesis |

## 2 & 3. Extraction candidates, target module, size, SAFE NOW vs DEFER

### SAFE NOW

1. **Marker constants + import guard → `_agent_spec_markers.py`** (new module).
   Move `_MARK_*` (lines 81–94) and `_import_agent_spec_flow_classes`
   (97–118), ~45 lines. `loader.py` currently imports the 11 `_MARK_*`
   constants **from `_agent_spec.py`** (`loader.py:25-37`) — an importer
   reaching into an exporter module for shared constants is backwards
   layering. Moving both to a neutral module fixes that direction too:
   both `_agent_spec.py` and `loader.py` import from `_agent_spec_markers.py`,
   neither from the other. Zero behavior change, zero Phase-8 overlap (Phase
   8 never edits marker key strings).

2. **Placeholder translation cluster → `_agent_spec_placeholders.py`** (new
   module). Move `_translate_placeholders`, `_node_translation`,
   `_is_translation_eligible`, `_prompt_spec_marker` (lines 166–287, ~122
   lines). Self-contained: reads only `Node`, `Property`, the `DOLLAR_RE`/
   `apply_scanner` scanner, and `_properties_for` (which stays put or moves
   alongside, see below). Called *from* `_lower_generation_step`,
   `_lower_oracle`, `_lower_each`, `_lower_loop`, `_lower_portal_mesh_to_swarm`
   — but only as an already-stable call signature (`_translate_placeholders(prompt, props, name)`,
   `_node_translation(node)`, `_is_translation_eligible(item)`); Phase 8's
   Construct-level fusion work does not change this signature or its
   internals (`_is_translation_eligible` already returns `False` for a
   `Construct` item, which is exactly what Phase 8 needs, unchanged). Pure
   mechanical move + import fix-up in every caller.

3. **Single-node lowering cluster → `_agent_spec_node_lowering.py`** (new
   module). Move `_reject_unrepresentable_fields` (121–163),
   `_lower_generation_step`, `_lower_node`, `_make_agent`,
   `_tool_to_server_tool`, `_agent_spec_marker`, `_make_llm_config`,
   `_make_server_tool` (342–543) — ~240 lines total. This is the "how does
   one `Node` become an `LlmNode`/`AgentNode`/`ToolNode`" subsystem; a
   `Construct` item never reaches it (`_lower_item_body` routes a `Construct`
   to `FlowNode(subflow=to_agent_spec(item))` instead), so Phase 8's
   Construct-level fusion work has no reason to touch these functions'
   bodies — only `_lower_oracle`'s per-variant loop *calls*
   `_lower_generation_step`, and a call site one line long survives a module
   move untouched.

4. **Portal/Swarm export → `_agent_spec_portal.py`** (new module). Move
   `_lower_portal_mesh_to_swarm` and `_is_peer_mesh_member` (1099–1280, ~185
   lines). Portal peer-mesh export is a structurally separate code path
   (mesh detection happens *before* `to_agent_spec`'s normal Flow-building
   loop even starts, `to_agent_spec:1295-1309`) with no dependency on the
   Oracle/Each/Loop/Operator dispatcher Phase 8 is extending. Depends only
   on `_make_agent` (item 3), `_translate_placeholders`/`_prompt_spec_marker`
   (item 2), and the marker constants (item 1) — all themselves being
   extracted, so this one is best landed *after* 1–3, or independently with
   three import-path updates.

   **Combined SAFE NOW total: ~590 lines removed** (1561 → ~970). Doesn't
   reach the 500 cap alone, but removes every cluster Phase 8 doesn't touch,
   so Phase 8 lands its remaining edits against a materially smaller file
   without any wait.

### DEFER (own design pass later)

5. **Oracle/Each/Loop/Operator lowering + dispatcher** (lines 290–323,
   325–339, 545–1096 minus the pieces already carved out above — the
   `_item_inputs`/`_item_outputs`/`_lower_item_body` trio, `_lower_oracle`,
   `_lower_each`, `_lower_loop`, `_lower_operator`, `_LoweredItem` +
   `_lower_construct_item`, ~350 lines once items 1–3 are gone). This is
   exactly the surface Phase 8 is actively rewriting (extending Each×Oracle
   fusion to Construct items touches `_lower_oracle`'s `isinstance(node,
   Construct)` branches, `_lower_each`'s fused-body wiring, and
   `_lower_construct_item`'s `SUB_CONSTRUCT_UNSUPPORTED_COMBOS` gate). Splitting
   it now would hand Phase 8 a moving target — every in-flight edit would need
   re-targeting to a new module path mid-epic. Defer to a dedicated design
   pass once Phase 8 lands: candidate shape is a `_agent_spec_modifiers.py`
   housing all four `_lower_<modifier>` functions + the dispatcher, but the
   *right* split depends on what Phase 8's Construct-fusion change actually
   does to `_lower_construct_item`'s shape (e.g. whether the postlude/decomp
   structure changes), so pre-designing the boundary now would likely be
   thrown away.

6. **`to_agent_spec`'s edge-wiring body** (lines 1282–1562, ~280 lines,
   dominated by the `_emit_input_edges` closure and the dict-form/single-type
   input-edge loops at 1407–1517). This is the most tangled section in the
   file — a single function mixing mesh-branch dispatch, per-item lowering,
   control-edge stitching, and two different data-edge resolution strategies
   (dict-form vs single-type fan-in, each with its own B4 multi-output-producer
   sub-case). It's a real "extract a module" candidate (e.g. an
   `_agent_spec_edges.py` for the `_emit_input_edges` + producer-resolution
   logic), but doing it well requires deciding whether `_emit_input_edges`
   keeps closing over `item_by_name`/`input_targets_by_item_name` (i.e.
   becomes a small class/dataclass with that state as fields) or keeps taking
   them as explicit params — a real design choice, not a mechanical cut-paste,
   and it also reads from the Phase-8-owned `input_targets_by_item_name` shape
   (item 5), so it should follow item 5's split, not precede it.

## 4. Duplication found

- **Marker-constant import direction** (not duplication, but a related
  layering smell caught while reading): `loader.py` imports all 11 `_MARK_*`
  constants from `_agent_spec.py` (`loader.py:25-37`). This is the exporter
  being imported by the importer purely for shared constants — extraction
  item 1 above fixes it as a side effect, at no extra cost.
- **`_import_agent_spec_flow_classes` pattern duplicated in `loader.py`**:
  `loader.py`'s own `_import_agent_spec_import_classes` docstring says
  verbatim "Copies `_agent_spec._import_agent_spec_flow_classes()`'s exact
  import-guard shape" (`loader.py:74-77`) — this is a *documented*,
  deliberate copy (two different pyagentspec class subsets: Flow/edges/nodes
  vs. the importer's own set), not an accidental one, and it lives in
  `loader.py`, outside this file's scope. Flagging it here only because
  moving `_import_agent_spec_flow_classes` to `_agent_spec_markers.py` (item
  1) gives a natural place for `loader.py`'s copy to eventually converge on
  if the two class subsets are ever unified — not proposing that now.
- **No real duplication found *within* `_agent_spec.py` itself** — the file
  is long because of breadth (5 modifiers × Node/Construct × 3 LLM modes ×
  Option-F translation), not copy-pasted logic; every shared shape already
  goes through one seam (`_lower_item_body`, `_lower_generation_step`,
  `_properties_for`, `_node_translation`), consistent with the file's own
  "neograph-2s2o6 / one dispatch, not two" comments.
