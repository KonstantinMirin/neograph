# File-size refactor design: `src/neograph/modifiers.py`

Date: 2026-07-29
Scope: read-only research, no code changes. Input to the repo-wide 500-line
guard ticket; not a replacement for it.

Current size: **1116 lines** (single file, `src/neograph/modifiers.py`).

Context: Phase 8 of the active epic (`neograph-s7zt3.11`, extending fusion
`ModifierCombo` lowering to the Construct level) will keep editing the
`ModifierCombo` / `COMBO_DECOMPOSITION` / `classify_modifiers` cluster in this
file (lines ~65-309). Nothing proposed here touches that cluster's location or
shape — only clusters the epic does NOT touch are proposed for extraction, so
Phase 8 can land without a merge-conflict or re-plan risk.

## 1. Responsibility map (in file order)

| Lines | Section | Coupled to |
|---|---|---|
| 23-63 | Oracle merge-hook Protocols (`MergePreProcess`, `MergePostProcess`, `MergeFallback`) | Only `Oracle`'s field annotations (line ~577-579); `__init__.py` re-exports them but no other module imports them |
| 65-309 | **ModifierCombo classification subsystem**: `ModifierCombo` enum, `_COMBO_MAP`, `PrimaryShape`, `ComboDecomposition`, `COMBO_DECOMPOSITION`, `SUB_CONSTRUCT_UNSUPPORTED_COMBOS`, `classify_modifiers`, `combo_for_modifier_names`, `modifier_names_for_combo`, `primary_shape`, `is_each_oracle_fused` | Consumed by `compiler.py`, `_wiring.py`, `_agent_spec.py`, `state.py`, `_state_write.py`, `_input_shape.py`, `_subconstruct.py`, `_fan_agent.py`, `loader.py`, `runner.py`, `__main__.py` — **this is the Phase 8 touch surface** |
| 312-346 | `Modifier` base class + `_PathRecorder` (lambda-introspection proxy for `.map()`) | Only used by `Modifiable.map()` below |
| 349-528 | `Modifiable` mixin: `has_modifier`, `get_modifier`, `__or__`, `.map()` | Depends on `Each`/`Oracle`/`Loop`/`Portal`/`ModifierSet` all being in scope; lazy-imports `Construct`/`Node`/`_construct_validation` inside `__or__` to dodge a cycle |
| 531-619 | `Oracle` modifier class | Depends on merge-hook Protocols (23-63) |
| 622-673 | `EachFailure` + `Each` modifier + `split_each_path` | Self-contained; `split_each_path` also consumed by `construct.py` and `compiler.py` |
| 676-690 | `Operator` modifier | Self-contained (15 lines) |
| 692-734 | `Loop` modifier | Self-contained |
| 737-886 | **Portal subsystem**: `HANDOFF_END`, `DISPATCH_ROUTE`, `Portal` class (incl. `is_dispatch`/`is_tool_triggered` properties + `model_post_init`) | Imported directly (`Portal`) by `_agent_cycle.py`, `factory.py`, `loader.py`, `forward.py`, `_agent_spec.py`, `decorators.py`; `HANDOFF_END`/`DISPATCH_ROUTE` imported by `_wiring.py`, `factory.py`, `_validation_portal.py`, `__init__.py` |
| 888-913 | `_group_portal_members` (Portal mesh grouping) | Used by `_wiring.py`, `_validation_portal.py`, `_ir_normalize.py`, `state.py` |
| 916-1116 | `ModifierSet` composition/validation subsystem: `_SlotRule`, `_EACH_LOOP_CONFLICT`/`_ORACLE_LOOP_CONFLICT`/`_PORTAL_HINT`, `_km_conflict`, `_SLOT_RULES`, `ModifierSet` class (`combo`, `model_post_init`, `with_modifier`, `to_list`) | Needs `Each`/`Oracle`/`Loop`/`Operator`/`Portal` all in scope for `isinstance` dispatch in `_SLOT_RULES`; `ModifierSet` itself is imported directly by `node.py`, `construct.py`, `_ir_branch.py`, `_ir_protocols.py`, `_agent_spec.py`, `_fan_agent_wrap.py` |

## 2. Extraction candidates, target modules, and size impact

### (A) Oracle merge-hook Protocols → `neograph/_oracle_protocols.py` (new)
- Moves: lines 23-63 (~42 lines, all three `Protocol` classes + the three
  `TypeVar` defaults `_Variant`/`_FallbackResult`/`_PostResult` they use).
- `modifiers.py` keeps `from neograph._oracle_protocols import MergePreProcess, MergePostProcess, MergeFallback` for the `Oracle` field annotations; `__init__.py`'s existing re-export line changes its source module only.
- Net removal from `modifiers.py`: **~40 lines**.

### (B) Portal subsystem → `neograph/_portal.py` (new)
- Moves: lines 737-913 — `HANDOFF_END`, `DISPATCH_ROUTE`, the `Portal` class (its docstring alone is ~25 lines), and `_group_portal_members`.
- `Portal` only depends on `Modifier` (base class), `ConfigurationError`, and the `TYPE_CHECKING`-only `ConstructItem` — all already importable without a cycle back into the rest of `modifiers.py`.
- `modifiers.py` re-imports `Portal, HANDOFF_END, DISPATCH_ROUTE, _group_portal_members` from `_portal.py` so every one of the 7+ external call sites (`_agent_cycle.py`, `factory.py`, `loader.py`, `forward.py`, `_agent_spec.py`, `decorators.py`, `_wiring.py`, `_validation_portal.py`, `_ir_normalize.py`, `state.py`, `__init__.py`) keeps importing from `neograph.modifiers` / `neograph` unchanged — a pure move-plus-re-export, zero call-site edits.
- Net removal from `modifiers.py`: **~175 lines** (re-export line adds back ~1-2).

### (C) `EachFailure` + `Each` + `split_each_path` → `neograph/_each.py` (new)
- Moves: lines 622-673 (~52 lines). Fully self-contained (only needs `Modifier`, `BaseModel`, `field_validator`).
- Smaller win; only worth doing alongside (A)/(B) in the same pass, not on its own.
- Net removal: **~50 lines**.

### (D) `ModifierSet` + `_SlotRule` table → `neograph/_modifier_set.py` (new) — **larger, deferred**
- Would move lines 916-1116 (~200 lines) plus require `Each`, `Oracle`, `Loop`, `Operator`, `Portal` to already be defined/importable at that point (the `_SLOT_RULES` table does `isinstance` dispatch against all five). If (B) has already landed, `Portal` becomes a cross-module import; the rest stay in `modifiers.py`, creating a two-way dependency (`_modifier_set.py` imports the modifier classes from `modifiers.py`, but `modifiers.py`'s `Modifiable.modifier_set: ModifierSet` type annotation and `classify_modifiers`'s `isinstance(ms, ModifierSet)` check need `ModifierSet` back) — solvable with `TYPE_CHECKING`-only imports and a runtime import inside `classify_modifiers`, but that's exactly the kind of "make it not actually cyclic" work that deserves its own design pass rather than a mechanical move.

### (E) `Modifiable` mixin (`has_modifier`/`get_modifier`/`__or__`/`.map()` + `_PathRecorder`) → deferred
- ~215 lines (312-528). Not extractable as cleanly: `__or__` lazy-imports `Construct`/`Node`/`_construct_validation` specifically to dodge a real import cycle (`node.py`/`construct.py` both import `modifiers.py` for the `Modifiable` base). Relocating this mixin to its own module doesn't remove the cycle, it just relocates which file has the lazy-import dance, and `.map()`'s `_PathRecorder` + dev-warning logic in `__or__` (Oracle n==1, Loop max_iterations==1) reaches into `Oracle`/`Loop` fields directly. Worth a dedicated pass, not a mechanical extraction.

## 3. SAFE NOW vs DEFER

**SAFE NOW** (mechanical, move + re-export, zero behavior change, does not touch the epic's active editing surface):

1. **(B) Portal subsystem → `_portal.py`** — the single best move. It is the largest self-contained cluster (~175 lines), has no dependency on the `ModifierCombo`/`COMBO_DECOMPOSITION` cluster Phase 8 is editing (Portal has its own `PORTAL`/`PORTAL_OPERATOR` combo entries already in `_COMBO_MAP`/`COMBO_DECOMPOSITION`, but the class body itself doesn't reference those tables), and every external consumer already imports `Portal`/`HANDOFF_END`/`DISPATCH_ROUTE` by name — a re-export line in `modifiers.py` keeps all of them working untouched. This alone takes `modifiers.py` from 1116 to ~940 lines.
2. **(A) Oracle merge-hook Protocols → `_oracle_protocols.py`** — smaller (~40 lines) but purely typing-only Protocols with a single consumer (`Oracle`'s field types) and one re-export site (`__init__.py`). Zero risk.
3. **(C) `Each`/`EachFailure`/`split_each_path` → `_each.py`** — self-contained (~50 lines), touched by `construct.py`/`compiler.py` only via the already-public `split_each_path` function, not internals. Bundle with (A)/(B) in the same PR since it's a small enough diff not to warrant its own review pass.

Combined, (A)+(B)+(C) take `modifiers.py` from **1116 → ~895 lines** in one low-risk, reviewable pass, without touching a single line Phase 8 needs to edit (lines 65-309 stay put, untouched, in the same file).

**DEFER** (needs its own design pass):

4. **(D) `ModifierSet`/`_SLOT_RULES` → `_modifier_set.py`** — real value (~200 lines) but requires resolving a genuine two-way dependency between the modifier classes and `ModifierSet` (whichever module is defined "second" needs a `TYPE_CHECKING`-only or function-local import of the other). Decide during that pass whether `_modifier_set.py` should also absorb the Portal-specific reciprocal-exclude arms in `ModifierSet.model_post_init` (lines 1037-1057) that currently hard-code pairwise checks separately from `_SLOT_RULES` — that duplication (noted in section 4 below) is arguably worth fixing in the same pass rather than mechanically relocating it as-is.
5. **(E) `Modifiable` mixin → its own module** — ~215 lines, but entangled in the `node.py`/`construct.py`/`modifiers.py` import cycle via lazy imports inside `__or__`. Any move needs to first decide whether the lazy-import dodge moves with it or gets restructured (e.g., a small `_loop_validation` seam) — a design question, not a mechanical cut.
6. **General**: once (A)-(C) land and Phase 8 completes, revisit whether the `ModifierCombo` classification cluster (65-309, ~245 lines) itself should move to its own module (e.g. `_combo.py`) now that Phase 8's Construct-level fusion work is done and the cluster has stabilized. Not proposed now specifically because it is the epic's active edit surface.

## 4. Duplication found

- **Reciprocal Portal-exclusion logic exists in two places that must stay in sync by hand**: `_SLOT_RULES` (lines 962-989, the `with_modifier` pipe-composition path) encodes Portal's mutual exclusion with Each/Oracle/Loop, and `ModifierSet.model_post_init` (lines 1024-1057, the direct-`ModifierSet(...)`-construction path) encodes the **same** three exclusions again as hard-coded `if` arms — the code comment at line 1037-1041 admits this explicitly ("this direct-construct path uses hard-coded pairwise arms — it does NOT read `_SLOT_RULES`... the exact parity hazard"). This is real, acknowledged duplication, not superficial similarity: two independent sources of truth for the same five-modifier exclusion matrix, one table-driven and one inline, that a future modifier addition must remember to update in both places. Not proposed for a SAFE NOW fix here (it's a behavior-bearing consolidation, not a file-split), but worth flagging as a candidate for the DEFER-(D) `_modifier_set.py` pass — that's the natural place to also fix the duplication (drive `model_post_init` off `_SLOT_RULES` too) while the file is already being touched.
- **No duplication found between `modifiers.py` and the other epic-active files** (`compiler.py`, `_agent_spec.py`, `_wiring.py`, `state.py`) beyond the expected "consumer reads a table defined once in `modifiers.py`" pattern (`_COMBO_MAP`, `COMBO_DECOMPOSITION`), which is already correctly centralized per the file's own docstrings and is the pattern to preserve, not flatten.
