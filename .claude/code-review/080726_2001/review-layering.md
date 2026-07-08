# Layering Review

**Scope**: Full neograph source (`src/neograph/`, 76 files) against the layering
discipline documented in AGENTS.md/CLAUDE.md: User code → DX layer
(`decorators.py`, `forward.py`) → IR (`node.py`, `construct.py`,
`_construct_validation.py`) → Compiler (`compiler.py`, `state.py`) → Runtime
dispatch (`factory.py`, `_dispatch.py`, `_execute.py`, `_llm*.py`) → LangGraph.
**Date**: 2026-07-08
**Verdict**: The documented layering is honored in the source. No active
leakage found. Two findings, both about the *locking* of the invariant (a guard
asymmetry) and doc-rot — not the code itself.

## What was verified clean (high-signal, given how carefully this is documented)

- **Import direction DX → down holds.** No lower-layer module imports
  `neograph.decorators` or `neograph.forward`. Only the package facade
  `__init__.py` imports the DX layer (to re-export `@node`/`ForwardConstruct`).
  Confirmed by AST/grep across all `src/neograph/*.py`.
  Repro: `for f in src/neograph/*.py; do grep -l "from neograph.decorators\|from neograph.forward" "$f"; done` → only `__init__.py`.
- **DX concepts do not leak into IR/compiler as logic.** `decorators.py` owns
  mode inference (kwarg sniffing → `"think"`/`"scripted"` at `decorators.py:350-352`),
  DI classification (`_classify_di_params`, extracted to `_di_classify.py`), and
  sidecar population. None of that decision logic appears in `node.py`,
  `construct.py`, `_construct_validation.py`, `factory.py`, or `modifiers.py`.
- **The sidecar consumption by IR/compiler/runtime is the DOCUMENTED design,
  not a leak.** `compiler.py:340` (`_collect_required_di`) and
  `_dispatch.py`/`_execute.py` (`_get_param_res`) read `node._param_res`, and
  `compiler.py:326` reads `_scripted_shim`. All go through the neutral leaf
  `_sidecar.py` accessor and consume *generic* PrivateAttr data — the other two
  API surfaces leave `_param_res` empty, so the code is surface-agnostic (returns
  unchanged), exactly per the "produce IR instances those modules already accept"
  rule.
- **DI vocabulary lives in a neutral module.** `DIKind`/`DIBinding` are defined in
  `di.py` (not `decorators.py`). `_construct_validation.py:54` imports
  `DIKind` from `di.py`, so the validator's `merge_fn` `from_state` check
  (`_construct_validation.py:251-303`) references DI *types* from the shared
  vocabulary, not the DX layer. Clean.
- **`fan_out_param`** is a genuine `Node` IR field (`node.py:204`), consumed by
  `_input_shape.py:108` — the single documented "IR-level concession to the @node
  layer," and it applies to programmatic `Each` nodes too. Not a leak.
- **`_normalize.py` is genuinely neutral and reachable from `forward.py`** (DX):
  `forward.py:58` imports `_declared_output`/`normalize_outputs` from it; its own
  imports are all IR-level. The `_declared_output`/`effective_producer_type`
  monopolies are locked by `test_guards_helper_monopoly.py`
  (`TestDeclaredOutputSelectorMonopoly`, `TestIrWalkHelperMonopoly`).
- **Engine-verb / run-layer confinement is locked** by
  `test_guards_three_layer.py` (engine verbs only in `_compiled.py`/`runner.py`
  /`_subconstruct.py`; compile layer never imports `runner.py`) — with anti-vacuity
  exact-surface + synthetic slip meta-tests. This is exemplary.
- **Assembly cluster DAG** (`factory → _execute → {_state_write,_input_shape,
  _oracle,_dispatch}`, leaves have no upward edges) is locked by
  `TestAssemblyClusterImportDAG` with a mutation meta-test.

## Findings

### LR-01: DX-import guard is asymmetric — `forward.py` is locked, `decorators.py` is not
- **Severity**: Low (missing lock, not an active violation)
- **Violation**: IR/Compiler/Runtime → DX (`decorators.py`) import edge is
  unguarded except at one narrow seam.
- **File**: `tests/test_guards_assembly.py:1501` (`TestLowerLayersDoNotImportForwardDX`)
  vs. `tests/test_guards_sidecar_imports.py:37`
  (`test_construct_builder_does_not_import_from_decorators`)
- **Description**: AGENTS.md defines the DX layer as **two** modules —
  "`@node` / `ForwardConstruct` ← DX layer (`decorators.py`, `forward.py`)."
  There is a broad structural guard, `TestLowerLayersDoNotImportForwardDX`
  (neograph-9epk), asserting that *no* lower-layer module imports
  `neograph.forward`. There is **no symmetric guard for `neograph.decorators`.**
  The `decorators.py` import boundary is protected only at a single file seam:
  `_construct_builder.py` must not import from `decorators.py`
  (`test_guards_sidecar_imports.py:37`). So a regression where, e.g., `factory.py`
  or `compiler.py` grew `from neograph.decorators import _classify_di_params`
  would be caught for the `forward.py` half of the DX layer but silently pass for
  the `decorators.py` half. This matters more than usual because `decorators.py`
  **re-exports** DI symbols for backward compat
  (`decorators.py:68` re-exports `_di_classify`, `decorators.py:93` re-exports
  `_sidecar`), so `from neograph.decorators import FromInput/get_merge_fn_metadata`
  is an importable backdoor into DX-layer surface. The invariant holds today
  (verified: zero lower-layer decorators imports), but it rests on convention for
  the `decorators.py` half where `forward.py` rests on a guard.
- **Reproduction**:
  `grep -rn "neograph.decorators\|neograph.forward" tests/test_guards_*.py` shows
  a whole-tree guard only for `forward` (assembly:1515-1526); the only
  `decorators` import-guard is the single-file `_construct_builder` check.
  Mutation: adding `from neograph.decorators import _classify_di_params` to
  `src/neograph/factory.py` leaves the full guard suite green.
- **Recommended fix**: Generalize `TestLowerLayersDoNotImportForwardDX` to the
  full DX layer — scan for both `forward` **and** `decorators` in
  `_parse_neograph_imports(path)`, with the same `{__init__}` allowlist. One
  guard, two DX modules; drop the now-subsumed narrow `_construct_builder` check
  or keep it as a named specialization.

### LR-02: `_normalize.py` docstring cites a test file that no longer exists
- **Severity**: Low (doc-rot; no behavioral impact)
- **Violation**: N/A (stale reference)
- **File**: `src/neograph/_normalize.py:11`
- **Description**: The module docstring says "A structural guard in
  `tests/test_structural_guards.py` enforces that no other `src/neograph/*.py`
  file does `isinstance(<expr>.outputs, dict)`…". Per AGENTS.md, `test_validation.py`
  and `test_structural_guards.py` were split by concern (neograph-e8jg). The guard
  now lives in `tests/test_guards_sidecar_imports.py`
  (`TestNodeIOPolymorphismNormalized`, line 817). A maintainer following the
  pointer lands on a nonexistent file.
- **Reproduction**: `ls tests/test_structural_guards.py` → not found;
  `grep -rn "test_structural_guards.py" src/` → `_normalize.py:11`.
- **Recommended fix**: Update the pointer to
  `tests/test_guards_sidecar_imports.py::TestNodeIOPolymorphismNormalized`.

## Summary

- Critical: 0
- High: 0
- Medium: 0
- Low: 2

The layering discipline is exceptionally well-honored and, for the run-layer /
engine boundary, exceptionally well-locked (exact-surface + slip meta-tests). The
single structural gap is an **asymmetry**: the "lower layers must not import the
DX layer" guard was written for `forward.py` but never extended to its twin
`decorators.py`, even though AGENTS.md names both as the DX layer. It is a missing
lock over already-correct source, not an active violation — which is exactly the
class of finding that matters in a codebase that documents its layering this
carefully.
