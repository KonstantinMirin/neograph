# File-size refactor — Cluster B: `_llm_retry.py`, `_oracle.py`, `describe_type.py`

Design-only. No code changed. Grounded in a full read of all three files on
2026-07-29 (develop @ `926b041`). Cross-checked for duplication against the
Agent Spec / Portal epic's active files: `_agent_spec.py`, `loader.py`,
`_wiring.py`, `modifiers.py`, `factory.py`, `_agent_cycle.py`.

Current sizes: `_llm_retry.py` 658, `_oracle.py` 650, `describe_type.py` 552.

---

## 1. `src/neograph/_llm_retry.py` (658 lines)

### Responsibility map

| Lines | Cluster | What it does |
|---|---|---|
| 30-102 | JSON extraction | `_extract_balanced`, `_extract_json` — find the first balanced `{...}`/`[...]` in raw LLM text, no Pydantic/retry knowledge |
| 105-267 | Null-default coercion | `_is_list_annotation`, `_STRINGLY_NULL`, `_optional_inner_types`, `_unwrap_optional`, `_is_stringly_null`, `_descend_null_defaults`, `_apply_null_defaults` — pure type-introspection + dict-mutation helpers that normalize LLM-emitted `null`/`"null"` sentinels against a Pydantic model's field defaults, recursively through list/dict/Optional containers |
| 270-361 | Parse orchestration | `_parse_json_response` (calls into both clusters above), `_validation_error_details` |
| 364-473 | Retry-message building | `_repair_hint`, `_build_retry_msg`, `_is_truncated`, `_build_continuation_msg`, `_retry_msg_for_failure`, `build_structured_repair_message`, `structured_retry_messages` — pure string assembly, no network |
| 476-510, 589-623 | Invoke-with-retry loops | `_invoke_json_with_retry` (sync), `_ainvoke_json_with_retry` (async) — own the network round-trip + retry budget |
| 513-586, 626-658 | DSML recovery | `_dsml_recovery_messages`, `recover_dsml` (sync), `arecover_dsml` (async) |

### Proposed extraction

**Target module: `src/neograph/_null_defaults.py`** (new).
Move the null-default coercion cluster (105-267, ~163 lines): `_is_list_annotation`,
`_STRINGLY_NULL`, `_optional_inner_types`, `_unwrap_optional`, `_is_stringly_null`,
`_descend_null_defaults`, `_apply_null_defaults`. This cluster:
- has zero dependency on the retry/network/DSML machinery in the rest of the file —
  its only imports are `pydantic.BaseModel` and `pydantic_core.PydanticUndefined`;
- is called from exactly one site (`_parse_json_response`, line ~312/320);
- is independently documented already (each function's docstring references
  `neograph-zhwgh` / `neograph-s1u4` as its own bug-fix lineage, distinct from the
  retry-loop's `neograph-p3c7`), i.e. it already reads as a separate concern that
  happens to live in the same file.

`_llm_retry.py` would `from neograph._null_defaults import _apply_null_defaults`
and drop to **~495 lines**.

**Target module: `src/neograph/_json_extract.py`** (new).
Move the JSON-extraction cluster (30-102, ~73 lines): `_extract_balanced`,
`_extract_json`. Pure string-scanning, no Pydantic import at all, called only
from `_parse_json_response`. Combined with the null-defaults extraction this
drops `_llm_retry.py` to **~420 lines**.

### SAFE NOW vs DEFER

- **SAFE NOW**: both extractions above. Mechanical cut-paste + import-line swap.
  No behavior change (verify via existing `test_*retry*`/`test_fakes.py`-adjacent
  suites — grep shows `_llm_retry` is imported only by `_llm_dispatch.py` and
  `_tool_loop.py`, both outside the epic's active-file list, so there is zero
  overlap risk with Phase 8). Recommended single best action: extract the
  null-default cluster first — it's the larger win (163 lines) and the cleanest
  separation (no shared helpers with anything else in the file).
- **DEFER**: the retry-message-building cluster (364-473) and the two
  invoke-with-retry loops overlap conceptually (both are "what do we say to the
  LLM when parsing/validation fails") but are entangled with `ExecutionError`
  construction, `_is_truncated` provider-specific metadata sniffing, and the
  sync/async twinning pattern used throughout the LLM layer. A real split here
  (e.g., separating "message templates" from "retry-loop control flow") is a
  legitimate future design pass, but doesn't reduce line count as cleanly and
  isn't needed to get this file under 500 — the two SAFE NOW extractions already
  do that.

### Duplication check

None found with epic-active files. `_llm_retry.py` is not imported by
`_agent_spec.py`, `loader.py`, `_wiring.py`, `modifiers.py`, or `_agent_cycle.py`.
`factory.py` does not import it either (dispatch goes through `_llm_dispatch.py`
/ `_tool_loop.py`).

---

## 2. `src/neograph/_oracle.py` (650 lines)

### Responsibility map

| Lines | Cluster | What it does |
|---|---|---|
| 32-50 | Config injection | `_inject_oracle_config` |
| 53-140 | Redirect wrappers | `make_oracle_redirect_fn`, `make_eachoracle_redirect_fn` — wrap a generator node's `Runnable` so its output lands in a collector field instead of the consumer-facing field |
| 143-220 | Merge-result shaping | `_unwrap_oracle_results`, `_build_oracle_merge_result` |
| 223-242 | Upstream context | `_build_upstream_context` — shared by both the single-group Oracle merge (this file) and the Each×Oracle fused merge (`_wiring.group_merge_barrier`) |
| 245-378 | merge_prompt execution | `_run_merge_prompt`, `_merge_prompt_input`, `_merge_prompt_post`, `_merge_fallback_or_reraise`, `_arun_merge_prompt` |
| 381-429 | merge_fn execution | `_run_merge_fn`, `_assert_merge_fn_registered` |
| 432-507 | Canonical merge dispatch | `_merge_variants`, `_amerge_variants` — the ONE site selecting merge_prompt-vs-merge_fn, called by both this file's `make_oracle_merge_fn` AND by `_wiring._merge_one_group` (Each×Oracle fusion) |
| 510-582 | Barrier orchestrator | `make_oracle_merge_fn` — single-group merge barrier factory |
| 585-650 | Each (non-Oracle) redirect | `make_each_redirect_fn` — plain `Each` modifier key-by-item wrapper, **not** an Oracle concern at all |

### Proposed extraction

**Target module: `src/neograph/_each_redirect.py`** (new), OR fold into an
existing Each-owning module if one exists (none currently does — Each's other
runtime logic lives inline in `_wiring.py`/`compiler.py`).
Move `make_each_redirect_fn` (585-650, ~65 lines). This function has nothing to
do with Oracle: it keys a single generator's result by the `Each` item's key
field, with `on_error='collect'` → `EachFailure` handling. It sits in
`_oracle.py` only because the module's docstring says it was "extracted from
factory.py" alongside the Oracle wiring, not because it's Oracle logic — the
module's own header ("Oracle and Each×Oracle modifier wiring") already signals
this is the odd one out (plain Each, no Oracle in the loop). Re-export it from
`_oracle.py` (`from neograph._each_redirect import make_each_redirect_fn`) so
`compiler.py`'s existing `from neograph._oracle import (..., make_each_redirect_fn, ...)`
needs no edit.

This is a modest cut (~65 lines, to ~585) but it's the only cluster in this
file that is BOTH self-contained AND outside the merge-algorithm surface that
Phase 8 is actively extending — see DEFER below.

**Considered and rejected for SAFE NOW**: pulling `make_oracle_redirect_fn` /
`make_eachoracle_redirect_fn` (53-140, ~90 lines) into a sibling
`_oracle_redirect.py`. Mechanically this is just as safe (re-export keeps
`_wiring.py`'s `from neograph._oracle import make_eachoracle_redirect_fn`
working). Held back only because 65+90=155 lines already gets `_oracle.py`
to ~495, and touching more of this file than strictly needed increases the
diff Phase 8 has to rebase past for no additional size-cap benefit. If the
guard ticket needs `_oracle.py` under 500 in one shot, do this one too — it is
equally SAFE NOW, just not required to hit the number.

### SAFE NOW vs DEFER

- **SAFE NOW**: `make_each_redirect_fn` extraction (and, if needed, the two
  redirect-wrapper functions too) — pure cut/paste + re-export, zero behavior
  change, does not touch anything `_merge_variants`/`_amerge_variants`/
  `_build_upstream_context` related.
- **DEFER**: the merge-algorithm core (223-582: upstream-context, merge_prompt
  execution, merge_fn execution, canonical dispatch, barrier orchestrator) is
  **exactly** the surface neograph-s7zt3.11 (Phase 8, fusion ModifierCombo
  lowering to Construct level) is extending — `_wiring.py` already imports
  `_merge_variants`, `_amerge_variants`, and `_build_upstream_context` directly
  from this file as the "canonical merge step" it delegates to (its own
  docstrings say so verbatim: "Pure delegation to the canonical merge step in
  `_oracle._merge_variants`"). Splitting this cluster now — even mechanically —
  changes the exact import surface Phase 8 is adding new callers onto, and any
  module boundary drawn here is a bet on where Construct-level fusion will next
  need to plug in. This needs its own design pass **after** Phase 8 lands, not
  before or alongside it.

### Duplication check

No duplication with epic files — the relationship is intentional, single-sited
delegation (`_wiring.py` calls into `_oracle.py`'s canonical merge functions;
this is the sanctioned "ONE site for the merge algorithm" pattern per the
in-file docstrings), not copy-paste. Nothing to consolidate.

---

## 3. `src/neograph/describe_type.py` (552 lines)

### Responsibility map

| Lines | Cluster | What it does |
|---|---|---|
| 21-67 | Shared markers/helpers | `type_display_name`, `ExcludeFromOutput`, `_is_output_excluded`, `_PRIMITIVE_MAP` |
| 70-165 | `describe_type` entry point | Orchestrates the two-pass hoist-then-render algorithm |
| 173-246 | Pass 1 (counting) | `_count_classes`, `_count_annotation` |
| 253-417 | Pass 2 (type-schema render) | `_render_model_body`, `_render_type` |
| 425-440 | Render helpers | `_field_comment`, `_render_enum_declaration`, `_stable_sort` |
| 448-552 | `describe_value` + instance renderer | `describe_value`, `_render_instance`, `_render_value`, `_render_list_value`, `_render_dict_value` — BAML-style **value** (not type) rendering, a distinct feature from the schema emitter above |

### Proposed extraction

**Target module: `src/neograph/_describe_value.py`** (new).
Move the entire `describe_value` cluster (448-552, ~108 lines):
`describe_value`, `_render_instance`, `_render_value`, `_render_list_value`,
`_render_dict_value`. This is a clean seam:
- it renders instances (values), the type-schema pass renders classes (types) —
  different inputs, different purpose (few-shot examples vs. structured-output
  instructions), already documented as two separate public functions in
  `describe_type.py`'s own module docstring and in `neograph-dev-rendering`'s
  skill description;
- its only shared helper with the rest of the file is `_field_comment` (5
  lines) — trivial to either duplicate or import back (`from
  neograph.describe_type import _field_comment`, one-way, no cycle, since
  `_describe_value.py` would sit below `describe_type.py` in the import graph);
- every external caller (`__init__.py`, `_llm_render.py`, `_tool_loop.py`,
  `renderers.py`) imports `describe_value` via `from neograph.describe_type
  import describe_value` — per the repo's naming policy (`__all__` is the
  public contract, module prefix is advisory), `describe_type.py` just needs to
  re-export it (`from neograph._describe_value import describe_value`) and
  every call site is untouched.

This drops `describe_type.py` to **~444 lines** — under the 500 cap in one
extraction.

### SAFE NOW vs DEFER

- **SAFE NOW**: the `describe_value` extraction. It is the single best
  recommendation for this file — one mechanical cut, no behavior change, no
  call-site edits needed (re-export), and it alone clears the 500-line bar.
- **DEFER**: nothing else needs to move. Pass-1/pass-2 of the type-schema
  renderer (173-417, ~245 lines) are mutually recursive and share `hoisted`/
  `visited` bookkeeping threaded through every call — splitting counting from
  rendering would require passing that shared state across a module boundary
  for no line-count benefit (the file is already under the cap after the
  `describe_value` extraction). Not worth doing even later unless the file
  grows again.

### Duplication check

None found. No epic-active file (`_agent_spec.py`, `loader.py`, `_wiring.py`,
`modifiers.py`, `factory.py`, `_agent_cycle.py`) defines its own type-schema or
value renderer — `_agent_cycle.py` imports `type_display_name` from this module
for structlog field rendering (single shared renderer, not a duplicate), and
neither `_agent_spec.py` nor `loader.py` reference `describe_type`/
`describe_value` at all (Agent Spec import/export apparently does not currently
route through the BAML-style renderer — worth flagging to the epic owner as a
possible future integration point, but out of scope for this line-count pass).

---

## Summary table

| File | Current | SAFE NOW extraction(s) | Lines removed | Resulting size | DEFER (needs own design pass) |
|---|---|---|---|---|---|
| `_llm_retry.py` | 658 | null-default cluster → `_null_defaults.py`; JSON-extraction cluster → `_json_extract.py` | ~236 | ~420 | retry-message-building vs. invoke-loop separation |
| `_oracle.py` | 650 | `make_each_redirect_fn` → `_each_redirect.py` (+ optionally the two Oracle redirect wrappers → `_oracle_redirect.py`) | ~65 (up to ~155) | ~585 (up to ~495) | merge-algorithm core (223-582) — blocked on Phase 8 (neograph-s7zt3.11) landing first |
| `describe_type.py` | 552 | `describe_value` cluster → `_describe_value.py` | ~108 | ~444 | none needed |

All SAFE NOW items are re-export-preserving (zero call-site edits) and
independently verifiable by running the existing test suite unchanged.
