# File-size refactor design: `src/neograph/decorators.py`

Date: 2026-07-29
Scope: `src/neograph/decorators.py` only (975 lines as of this read), read in full.
Context: repo-wide 500-line-per-file cap being enforced by a separate guard-test
ticket (not this doc's concern). Active epic neograph-s7zt3 (Phase 8 /
neograph-s7zt3.11, fusion ModifierCombo lowering at the Construct level) is
still touching Portal/Oracle/Each lowering paths — nothing proposed here may
require pausing or re-planning that epic.

## 1. Responsibility map (as read, top to bottom)

| Lines | Section | Notes |
|---|---|---|
| 1-51 | Module docstring | Design notes for the whole `@node` surface. Not code. |
| 53-112 | Imports | Re-exports `FromConfig`/`FromInput`/etc. from `_di_classify`, `_sidecar`, `_runtime_registry` for backward compat (already-extracted modules — decorators.py is a facade over them for these symbols). |
| 115-136 | `_is_trivial_body` | AST helper: is a function body a placeholder (`...`/`pass`/bare return)? Used only by the dead-body warning inside `node()`. |
| 139-152 | `_apply_eager_oracle_gen_type` | Thin wrapper around `_ir_normalize.oracle_gen_type_for`, used twice inside `node()`. |
| 155-237 | `_build_oracle_kwargs` | Builds/validates `Oracle(...)` kwargs from decorator args (body-as-merge detection + shim registration, `merge_fn`/`merge_prompt` exclusivity, `ensemble_n>=2`). Pure kwargs-construction, no Node/Construct dependency beyond `register_scripted` + `ConstructError`. |
| 240-249 | `_build_each_kwargs` | Builds `Each(...)` kwargs. Trivial. |
| 252-271 | `_build_portal_kwargs` | Builds `Portal(...)` kwargs, with a load-bearing comment about `model_fields_set` parity with programmatic `Portal(...)`. |
| 274-789 | `node()` | The decorator itself — ~515 lines including its ~110-line docstring (309-382) and the `decorator(f)` closure (389-784). This is the actual size driver: map_over/loop_when/portal validation, mode inference, dead-body warning, raw-mode signature checks, DI classification call-out, output/input inference, `Node(...)` construction, then sequential modifier application (Each×Oracle fusion, Oracle-only, Operator, Loop, Portal) each re-registering the sidecar, plus the eager scripted-shim registration block at the end (767-783). |
| 792-960 | `@merge_fn` decorator cluster | `_qualname_site` (799-811), `_same_def_site` (814-834), `merge_fn()` (837-960). Self-contained: only touches `_merge_fn_registry`/`_merge_fn_caller_ns` (already in `_sidecar.py`), `_classify_di_params`, `_resolve_di_args`, `DIBinding`/`DIKind`, `register_scripted`. Zero interaction with `node()` or the modifier-kwargs helpers above. |
| 963-975 | Bottom re-exports | `_construct_builder`, `_construct_graph`, `_scripted_registry` symbols re-exported for back-compat. Already-extracted; nothing to do. |

Prior extractions are visible in the docstring/comments: `_sidecar.py`,
`_di_classify.py`, `_runtime_registry.py`, `_construct_builder.py`,
`_construct_graph.py`, `_scripted_registry.py` were already split out of what
used to be one file. This file is the residue after those splits, so what's
left is more concentrated (one very large function) rather than diffuse.

## 2. Extraction candidates

### A. Modifier-kwargs helper cluster -> new module `_node_modifier_kwargs.py`
**Move**: `_is_trivial_body` (115-136), `_apply_eager_oracle_gen_type` (139-152),
`_build_oracle_kwargs` (155-237), `_build_each_kwargs` (240-249),
`_build_portal_kwargs` (252-271). ~149 lines total.
**Dependencies**: `ConstructError`, `register_scripted` (already imported from
`_runtime_registry`), `oracle_gen_type_for` (from `_ir_normalize`), `Node` (only
for the type hint on `_apply_eager_oracle_gen_type`). All five functions are
pure kwargs-builders / AST-inspectors called exclusively from inside `node()`'s
`decorator(f)` closure — no other module in the repo calls them (verified: only
call sites are within this file). A straight cut-and-paste + `from
neograph._node_modifier_kwargs import ...` at the top of decorators.py.
**Effect**: -149 lines from decorators.py (975 -> ~826).
**SAFE NOW.** Zero behavior change, no test changes needed beyond import-path
guard tests (if any structural guard enumerates decorators.py's own function
list — check `test_guards_*` for an exact-symbol pin before landing). These
five functions do not touch Portal/Oracle/Each *lowering* logic (that lives in
compiler.py/modifiers.py/_wiring.py) — they only assemble the kwargs dict handed
to the modifier constructors — so Phase 8's ModifierCombo lowering work at the
Construct level never has a reason to open this new module. This is exactly
the kind of extraction that stops decorators.py from being the file Phase 8
has to keep adding lines to, without touching anything Phase 8 touches.

### B. `@merge_fn` decorator cluster -> new module `_merge_fn_decorator.py`
**Move**: `_qualname_site` (799-811), `_same_def_site` (814-834), `merge_fn()`
(837-960). ~162 lines.
**Dependencies**: `_merge_fn_registry`, `_merge_fn_caller_ns` (from `_sidecar.py`,
already a separate module), `_classify_di_params`, `_build_annotation_namespace`
(from `_di_classify.py`), `resolve_hints`, `DIBinding`, `DIKind`,
`register_scripted`, `ConstructError`. No dependency on `node()` or on section A's
helpers. The only inbound reference from the rest of decorators.py is the
"Registry and inference functions live in _sidecar.py" comment block at
792-796, which documents this cluster and can move with it.
**Effect**: -168 lines (162 code + adjacent comment block) (975 -> ~807; combined
with A: ~658).
**SAFE NOW.** This is arguably the cleanest single extraction in the file: it is
a complete, independently-documented decorator with its own registry
(elsewhere), its own collision-detection helpers, and no shared state with
`node()`. `@merge_fn` is orthogonal to Portal/Each/Oracle *lowering* — Phase 8
never needs to touch it.

### C. `node()` itself (274-789, ~515 lines including its own docstring)
Not proposed as a mechanical extraction here. Splitting the decorator would mean
either (a) pulling the ~110-line docstring into a lower-line-count
`node.__doc__` assignment (cosmetic, doesn't reduce logical complexity) or (b)
actually decomposing the `decorator(f)` closure's stages (validation ->
mode-inference -> DI classification -> in/out inference -> Node construction ->
sequential modifier piping -> eager shim registration) into named helper
functions that each take/return the accumulating `Node` and closed-over
decorator kwargs. That's a real refactor of the core `@node` control flow --
it changes the shape of the function that Phase 8 (fusion ModifierCombo
lowering) is a natural next editor of, since the Each×Oracle fusion branch
(642-666) and the Portal branch (746-765) inside this closure are precisely the
kind of code Phase 8's Construct-level lowering work is reasoning about.
**DEFER.** Name it for a dedicated pass: "decompose `node()`'s `decorator(f)`
closure into named per-stage functions (validate_edge_shape_kwargs,
infer_mode, infer_io, apply_modifiers) that thread a single mutable-by-copy
`NodeBuildState`", done only after Phase 8 lands so the two efforts don't
collide on the same ~500 lines.

## 3. SAFE NOW vs DEFER summary

| Extraction | Lines removed | Class | Reason |
|---|---|---|---|
| A: modifier-kwargs helpers -> `_node_modifier_kwargs.py` | ~149 | SAFE NOW | Pure kwargs builders, single call site (inside `node()`), no overlap with Phase 8's lowering surfaces |
| B: `@merge_fn` cluster -> `_merge_fn_decorator.py` | ~168 | SAFE NOW | Fully self-contained decorator + collision-detection, orthogonal to Portal/Each/Oracle lowering |
| C: decompose `node()`'s closure | ~0 net (restructure, not move) | DEFER | Touches the exact Each/Oracle/Portal piping sequence Phase 8 is actively extending; do after Phase 8 lands |

Combined SAFE NOW effect: 975 -> ~658 lines. Still above the 500-line cap, so
even after both safe extractions this file will need the DEFER work (C) --
plus possibly re-splitting the `node()` docstring out to keep the guard green
long-term -- to get under 500. That's expected and consistent with the
maintainer's framing: the safe cut buys headroom now without blocking Phase 8;
getting fully under 500 needs the deferred `node()` decomposition.

## 4. Duplication check against epic-active files

Grepped `_agent_spec.py`, `loader.py`, `_wiring.py`, `modifiers.py`,
`factory.py`, `_agent_cycle.py` for Oracle/Each/Portal kwargs construction
patterns analogous to this file's `_build_oracle_kwargs`/`_build_each_kwargs`/
`_build_portal_kwargs`.

**Real overlap found, not superficial**: `loader.py` independently builds
Oracle/Each/Portal kwargs dicts from the YAML/Agent-Spec DSL at multiple sites
(`loader.py:486` `base_node | Oracle(**oracle_kwargs)`, `:530`/`:614`
`inner | Each(over=..., key=...)`, `:939-964` `Portal(to=peers, trigger=...)`
branching on pyagentspec `HandoffMode`, `:1377-1380` another
`Oracle(**kwargs)` / `Each(...)` pairing). This is the same *shape* of problem
`decorators.py`'s three `_build_*_kwargs` helpers solve (translate a
DSL-specific spec into the modifier's kwargs, with the same
merge_fn/merge_prompt exclusivity and n>=2 validation logic partially
re-derived), but for a structurally different input (parsed spec objects vs.
decorator kwargs) and it is NOT literally duplicated code -- no shared
function bodies, just a shared *concern*.
**Classification: DEFER, and out of scope for this doc.** A shared
"build Oracle/Each/Portal kwargs from a validated spec" abstraction spanning
`decorators.py` and `loader.py` would be a genuine design question (what's the
common intermediate representation?) and `loader.py` is squarely inside Phase
8's active editing surface (Portal/HandoffMode lowering, s7zt3.14 references at
loader.py:838-972). Do not touch this now. Flag it as a candidate topic for
whatever design pass eventually addresses `loader.py`'s own line-count
problem (1393 lines), after Phase 8 lands.

No duplication found between `decorators.py` and `_agent_spec.py`,
`modifiers.py`, `factory.py`, or `_agent_cycle.py` beyond the expected
call-through (all consume `Oracle`/`Each`/`Portal`/`Loop`/`Operator` from
`modifiers.py`, as everything in the codebase does).
