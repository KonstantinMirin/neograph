# How to perform a file split in this codebase

Canonical, current instruction set. **Rewritten in place when something is learned; it
never grows a changelog or an addendum.** Derived from the nineteen splits of
`neograph-3ffdg`, all landed.

**WHY, before HOW.** The 500-line ceiling is a proxy. What it proxies for is a module
holding parts that are conceptually separate but implementationally entangled — clusters
that no longer belong together yet share helpers, constants and imports densely enough
that the seam is invisible until someone tries to move something. The surgical split is
what surfaces the seam; the shorter file is a side effect. Every extraction in the wave
found something the per-file surveys had not.

So when the number and the architecture disagree, **the architecture wins**. A split
landing at 509 with clean boundaries beats one reaching 499 by putting a function in the
wrong module. The governing statement of this principle, and the **two refusals that
outrank the ceiling**, live in `AGENTS.md` under "The file-size ratchet" — read that
first. This document is the procedure.

---

## Core invariant

**A split moves code and changes NOTHING else.** The full test suite must end at exactly
the same pass count it started at — no test added, removed, skipped, deselected, or
xfailed. Test files may change ONLY to follow a symbol's new location. If a behavioural
assertion needs editing, the split is wrong.

---

## Step 1 — Slice by AST, never by line range

Clusters are routinely non-contiguous: 5 of 7 interleaved with code that had to stay, and
in one case a single-range cut would have swallowed another child's target. Build spans
with `ast.parse`, slice per symbol, delete high-to-low.

A span **starts at the decorators**, not at the `def`:

```python
start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
end   = n.end_lineno
```

`ast` reports `lineno` at the class/def line, so a naive `(lineno, end_lineno)` leaves
decorator lines behind in the parent, where they silently re-decorate whatever definition
follows. In `.12` two stray `@dataclasses.dataclass` lines landed on a function and the
package failed to import with an `AttributeError` on `__mro__`.

---

## Step 2 — Re-derive dependencies from the AST, never from the survey

The per-file surveys in `docs/design/` are reliable on WHICH code forms a cluster and
UNRELIABLE on what it depends on. The `forward.py` survey called its trace cluster
"functionally self-contained" and listed its dependencies; the AST found two more, one a
runtime instantiation forcing a signature change.

For each cluster compute: which STAYING symbols it references, and which other clusters.
Then resolve, strictly in this order:

1. **Annotation-only reference** → move it under `if TYPE_CHECKING`. No cycle.
2. **Only this cluster uses the helper** → move the helper too, even if the ticket does
   not name it (`.3` moved `_properties_for` for this reason).
3. Otherwise **inject it as a parameter**, threaded from the call site (`.3` `export_flow`,
   `.2` `resolve_condition`, `.12` `shim_factory`).
4. Otherwise take a **smaller extraction**.

**NEVER a function-local/deferred import**: that requires growing
`FUNCTION_LOCAL_IMPORT_ALLOWLIST`, and ratchet allowlists only shrink.

Before choosing injection, **check the ceiling of every call site's file**. In `.2`,
injection meant editing `compiler.py`, which would have pushed `compiler.py` past its own
761 ceiling; the answer was a smaller extraction. In `.1` a mere import-block reformat grew
`compiler.py` by 2 lines and the guard refused, correctly.

---

## Step 3 — Sweep the declared inventories

**This is the actual work, not the code move.** Every split so far tripped 2–5 tests that
key on a filename, a symbol location, or a line number. Run every sweep below; each has
caught something the others missed.

**(a) Filename as a path or string**

```bash
grep -rn --include='*.py' -e '"TARGET.py"' -e "'TARGET.py'" tests/
```

**(b) Every moving top-level symbol by name**

```bash
grep -rlE 'sym1|sym2|...' tests/*.py tests/*/*.py
```

**(c) Every moving inner / nested / closure name** — (b) does not find these. `.1` shipped
a red gate because a guard keyed on `group_merge_barrier`, a closure inside a moving
function. Enumerate them with `ast.walk` over each moving function.

**(d) Line-number and file-scoped allowlist keys**

```bash
grep -rn --include='*.py' 'TARGET\.py:' tests/
```

Two distinct forms, both live: `"file.py:fn:param"` (no-Any parameter keys) and bare
`"file.py:233"` (line-number keys in `test_guards_any_audit.py`). The line-number form is
the most fragile thing in the suite:

- entries for moved code need the new module AND new line number;
- entries for code that did NOT move still break, because deleting a block above them
  shifts their line numbers;
- line numbers shift AGAIN under `ruff format`.

So these keys are recomputed LAST, after formatting — see Step 6.

**(e) Monkeypatch targets in behavioural tests** — a test patching
`neograph.runner.evict_run` must patch where the name is now looked up. The assertion does
not change; the target does.

**(f) Generated artifacts that record a symbol's defining module** — the website API
manifest (`website/src/data/api-manifest*.json`) and the generated-reference region of
`website/src/content/docs/reference/api.mdx` both record where each public symbol is
defined, so a split moves entries in them. Regenerate with:

```bash
uv run python scripts/gen_api_manifest.py
```

and rewrite the `api.mdx` region between the `GEN:reference-sections` sentinels from
`render_reference_sections()`, preserving the exact leading newline the extractor expects.
Then run `npm run build` in `website/`.

**(g) Layer-classification lists that name modules by role**

```bash
grep -rn 'DX_MODULES\|ALLOWED_GRAPH_ONLY_MODULES\|_EXEMPT_FILES' tests/
```

A module extracted from a classified layer **inherits that layer**. In `.12`
`_forward_trace.py` tripped the DX-layer guard because it was read as a lower layer
importing DX. Adding it to `DX_MODULES` is a CLASSIFICATION fix, not an exemption, and it
*strengthens* the guard — the extracted module gets the peer edge its parent already had,
and lower layers are now barred from importing it too.

### EXTEND vs REPOINT

If a guard scans a LIST of files, **EXTEND** it. Repointing a list drops the moved-from
file and the guard then passes vacuously over a shrinking surface — nearly shipped on the
markers guard in `.3`.

### RE-KEY vs REDISTRIBUTE vs WIDEN — three different things

- **RE-KEY**: the thing being permitted MOVED. Point the entry at the new module and
  REMOVE the old one. Leaving the old key grants a permission to a file that no longer
  does the thing (`ALLOWED_PREPOP`, `MONOPOLIES`, the `split_output_field` exemption).
- **REDISTRIBUTE**: the parent held N justified uses and the split scattered them across N
  files. Every file is listed and the TOTAL count is unchanged. Not a widening — no new
  use was permitted. State the count in the comment so the next reader can check
  (`ARBITRARY_TYPES_ALLOWLIST` went 1 file / 2 uses → 2 files / 2 uses in `.5`).
- **WIDEN**: a capability becomes available somewhere it was not. **Refuse it** and take a
  smaller extraction (guard G1 in `.6`, `ALLOWED_GRAPH_ONLY_MODULES` in `.9`).

---

## Step 4 — Preserve the parent's import surface, and prove it

Re-export every moved name from the parent with `# noqa: E402,F401`. **Without `F401`,
`ruff --fix` silently strips them**; in `.3` that broke collection and five names had to be
restored.

If a moved symbol's name also appears as a PARAMETER of a function that stays, a plain
re-export in the import header trips `F811` — a module-level import of that name ahead of
the staying function reads as a redefinition of its parameter, where the pre-split `def` in
the same position did not. Import it aliased and rebind it at the position the `def` used
to occupy, usually the file bottom:

```python
from neograph._new_module import merge_fn as _merge_fn_impl  # noqa: E402
...
merge_fn = _merge_fn_impl   # bottom of file
```

Public surface unchanged, the staying signature untouched, no `noqa` inside user-facing
API. (`.11`, `decorators.node` has a `merge_fn` parameter.)

Then **PROVE the surface — do not eyeball it**. The surface is BOTH the names the module
defined AND the names it imported and re-exported:

```
git show HEAD:src/neograph/TARGET.py -> ast.parse ->
   {defs, classes, module-level assigns}
   UNION
   {asname or name for every Import / ImportFrom alias}
-> assert hasattr(module, n) for every one.
```

Counting only definitions is not enough. In `.12` `forward.py` defined 26 names and
re-exported 30 more; the moved code was their only local consumer, so `ruff --fix` stripped
nine as unused and test collection broke on `_BranchMeta`. Re-add such names in their own
block with a comment saying they are re-exports, and mark them `noqa: E402,F401`.

---

## Step 5 — Expect to land short of the ticket estimate

Treat the ticket's line counts as **upper bounds**. A cycle or a caller ceiling forces a
smaller cut about half the time. Planned vs landed across the wave:

| Target | Planned | Landed | Note |
|---|---|---|---|
| `_wiring` | 392+390 | 382+262 | |
| `runner` | 705 | 540 | |
| `_agent_spec` | 592 | 580 | |
| `decorators` | 317 | 304 | no cycle — landed at plan |
| `forward` | 570 | 556 | injection, not a smaller cut |
| `_agent_cycle` | 283 | 278 | |
| `compiler` | 90 | 72 | |
| `modifiers` | 265 | 235 | |
| `factory` | 284 | 58 | G1 blocked the portal half |
| `loader` | 490 | 212 | 3 injections blocked the swarm half |

---

## Step 6 — Sequence, exactly

1. Build the new module(s), leaf-first
2. Remove the spans from the parent, add the re-exports
3. Prove the import surface
4. Repoint the inventories from Step 3
5. `ruff check --fix`, then `ruff format` — **explicit file list only**
6. **Recompute** any line-number allowlist keys (they moved again in step 5)
7. Re-derive the ceiling with `wc -l` and set it
8. Full gate: `uv run pytest` (bare), then `make quality`
9. Commit immediately

Ceiling and line-number keys are LAST because both `--fix` and `format` change line counts;
in `.1` the count moved by 10 between the move and the format.

---

## Step 7 — Process rules

- **NEVER run `ruff check --fix` or `ruff format` over a DIRECTORY.** List this task's
  files explicitly. A blanket run reformatted 78 unrelated files, shifted ceilings
  repo-wide, and tripped an unrelated guard.
- **NEVER pass a file list through an unquoted shell variable**: zsh does not word-split,
  so the filter matches nothing. This turned a scoped cleanup into `git checkout --` over
  78 files and destroyed all uncommitted work on a task that had to be redone from scratch.
- **NEVER put backticks in a `bd update` string**: zsh substitutes inside double quotes and
  silently deletes the word.
- **COMMIT the moment the gate is green.** Budget one split at a time; do not start one you
  cannot finish and commit.

---

## Known inventory families

**Re-sweep; do not trust this list.** It is a starting point for Step 3, not a closed set —
it was already incomplete once (`_wiring.py` was missing from the combo-consumer inventory
and was found only by a second, independent pass).

| File | What it keys on |
|---|---|
| `test_guards_file_size.py` | the ceiling (always) |
| `test_guards_any_audit.py` | `"file.py:LINENO"` and `"file.py:fn:param"` keys |
| `test_guards_ir_compiler.py` | `WIRING_FUNCTION_HOMES` symbol → module map |
| `test_guards_async_dispatch.py` | sync/async twin pairs, keyed by module |
| `test_guards_llm_runtime.py` | twin pairs, `LINE_BUDGETS`, `ALLOWED_PREPOP` |
| `test_guards_sidecar_imports.py` | `FUNCTION_LOCAL_IMPORT_ALLOWLIST`, langfuse scan |
| `test_guards_helper_monopoly.py` | `MONOPOLIES` helper → (owner file, min calls) |
| `test_guards_combo_decomposition_consumers.py` | combo-vocabulary consumer set |
| `test_guards_three_layer.py` | `ALLOWED_GRAPH_ONLY_MODULES`, `EXPECTED_ENGINE_SURFACE` |
| `test_guards_agent_spec_*.py` | single-file AST scans over the export surface |
| behavioural tests | monkeypatch targets |
