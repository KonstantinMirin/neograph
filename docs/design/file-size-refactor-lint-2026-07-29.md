# File-split proposal: `src/neograph/lint.py` (1068 lines)

Read in full 2026-07-29. Read-only research; no code changed. Scope per
mandate: propose WHERE the real seams are and WHAT moves WHERE, split into
SAFE NOW (mechanical, no epic disruption) vs DEFER (needs its own design
pass). Epic `neograph-s7zt3` (Phase 8 / `.11`, fusion ModifierCombo lowering
to Construct level) is not on this file's critical path — no references to
`lint.py` internals found in `_agent_spec.py`, `loader.py`, `_wiring.py`,
`modifiers.py`, `factory.py`, or `_agent_cycle.py`. No duplication with those
files was found either.

## 1. Responsibility map

| Lines | Section | What it is |
|---|---|---|
| 1-52 | imports, module docstring, `_KNOWN_EXTRAS`, `_PLACEHOLDER_RE` alias | plumbing |
| 55-64 | `LintIssue` dataclass | the public result type |
| 66-177 | `LintKindMeta` dataclass + `LINT_KIND_META` registry | static severity/meaning metadata table, consumed externally by `scripts/gen_api_manifest.py` |
| 180-253 | `_check_binding` | DI binding (`FromInput`/`FromConfig`/model forms) vs config check |
| 255-321 | `lint()` | public entry point — orchestrates everything below |
| 324-341 | `_has_resource_link_producer` | static approximation for resource-hydration check |
| 344-377 | `_check_resource_hydration` | `FromResource(ref=...)` vs no producer check |
| 380-393 | `_compiler_accepts_di_inputs` | introspects `prompt_compiler` signature |
| 396-428 | `_emit_missing_llm_kwargs_issue` | fail-loud LLM kwargs surfacing |
| 431-503 | `_walk` | the recursive construct walker — dispatches to every check below |
| 506-635 | `_check_template_placeholders` | inline `${var}` + template-ref `{var}` placeholder resolution (the biggest single function, ~130 lines) |
| 638-720 | `_check_loop_condition` | Loop `when` condition checks (string-registered / None-safety) |
| 723-930 | tool-policy cluster: `_check_async_only_tools`, `_TOOL_BODY_ATTRS`, `_tool_references_ask_human`, `_check_ask_human_in_mutating_node`, `_check_act_mode_all_idempotent`, `_spec_factory`, `_resolve_tool_object` | everything about tool objects: async-only driver requirement, ask_human-in-mutating-node safety, act/agent misclassification — all consume `Tool`/tool_factories, contiguous block, ~208 lines |
| 933-944 | `_extract_format_placeholders` | `str.format`-style `{}` name extraction, used only by template-placeholder check |
| 947-969 | `_di_template_var_names`, `_di_resource_template_var_names` | DI-binding-name-as-template-var predictors |
| 972-1068 | `_predict_input_keys`, `_get_flattened_field_names`, `_resolve_return_type` | predicts what keys `_extract_input`/`render_for_prompt` will produce at runtime — pure prediction, **zero `LintIssue` emissions** |

## 2. A critical coupling that constrains every extraction

`scripts/gen_api_manifest.py:_literal_kind_required_sites()` does:

```python
lint_path = REPO_ROOT / "src" / "neograph" / "lint.py"
tree = ast.parse(lint_path.read_text())
```

It AST-walks **only that one hardcoded file** looking for `LintIssue(kind=<literal>, required=<literal>)` call sites, to co-derive the `lint_issue_kinds` manifest section and fail loud if `LINT_KIND_META` drifts from what the code actually emits (`test_guards_api_manifest.py` pins this).

**Consequence**: any extraction that moves a function containing a literal `LintIssue(kind="...", ...)` call OUT of `lint.py` silently breaks this completeness check (the kind vanishes from `code_kinds`, so `code_kinds != set(LINT_KIND_META)` fires) unless `_literal_kind_required_sites()` is updated in the same change to also scan the new file. This is the single biggest constraint on how much can move mechanically without a companion edit outside `lint.py`.

Functions with **zero** `LintIssue(...)` emissions (pure predictors) are exempt from this coupling entirely — they are the cleanest extractions.

## 3. Proposed extractions

### SAFE NOW — no `LintIssue` emissions, no manifest-generator coupling
**Move**: `_predict_input_keys`, `_get_flattened_field_names`, `_resolve_return_type`, `_di_template_var_names`, `_di_resource_template_var_names` (lines 947-1068, contiguous with lines 933-944 optionally included) → new module `src/neograph/_lint_predict.py`.

- **Why safe**: none of these five functions constructs a `LintIssue`. They are pure prediction helpers consumed only by `_check_template_placeholders` and `_compiler_accepts_di_inputs`, which stay in `lint.py` and simply gain an import. Zero behavior change, zero interaction with `gen_api_manifest.py`'s AST scan.
- **Removes**: ~122 lines.
- **This is the single best SAFE NOW pick** — cleanest boundary in the file, no companion edits required anywhere else.

### SAFE NOW — pure data, one companion import-line edit
**Move**: `LintKindMeta` dataclass + `LINT_KIND_META` registry (lines 66-177) → new module `src/neograph/_lint_kind_registry.py`; `lint.py` imports `LINT_KIND_META` from it (or re-exports for back-compat).

- **Why safe**: pure data table, no logic. The AST-scanned `LintIssue(...)` **call sites themselves** (the thing `gen_api_manifest.py` actually scans for) all stay physically in `lint.py` — only the *registry table* moves, so the completeness check is unaffected. The one required companion edit is `scripts/gen_api_manifest.py:62`'s import line (`from neograph.lint import LINT_KIND_META` → new path), which is mechanical and covered by `test_guards_api_manifest.py` if broken.
- **Removes**: ~112 lines.

### SAFE NOW, but flag the manifest-generator companion edit explicitly
**Move**: the tool-policy cluster — `_check_async_only_tools`, `_TOOL_BODY_ATTRS`, `_tool_references_ask_human`, `_check_ask_human_in_mutating_node`, `_check_act_mode_all_idempotent`, `_spec_factory`, `_resolve_tool_object` (lines 723-930, one contiguous block) → new module `src/neograph/_lint_tool_checks.py`.

- **Why it's a clean cluster**: all seven functions revolve around one concept (introspecting `Tool`/tool-factory objects for policy violations), share the same two resolver helpers (`_spec_factory`, `_resolve_tool_object`), and take only `Node`/`Tool`/`is_async_only_tool`/`log`/`LintIssue` as dependencies — no entanglement with `_walk`'s `known_vars`/`di_inputs_enabled` threading that the template-placeholder cluster has.
- **Required companion edit**: this cluster DOES contain literal `LintIssue(kind="tool_requires_async_driver"/"ask_human_in_mutating_node"/"act_mode_all_idempotent_tools", ...)` sites, so `gen_api_manifest.py:_literal_kind_required_sites()` must be extended to also scan the new file (a one-line addition to a list of paths), or the manifest completeness guard breaks. Still mechanical, just not zero-touch outside `lint.py` — call this out explicitly to whoever executes so the guard failure isn't a surprise.
- **Removes**: ~208 lines.

Combined SAFE NOW total: **~440 lines removed** (1068 → ~630). Still above the 500 cap — see DEFER below for the rest.

### DEFER — needs its own design pass
1. **`_check_template_placeholders` (~130 lines) + `_extract_format_placeholders`**: the largest remaining function. Entangled with `_walk`'s `known_vars`/`template_resolver`/`di_inputs_enabled` parameter threading and the inline-vs-template-ref key-asymmetry logic documented at length in the module docstring and `CLAUDE.md`. Also contains 3 literal `LintIssue(kind=...)` sites (same manifest-generator coupling as above). Extracting this cleanly wants a decision on whether the `known_vars`/`resource_vars` computation is teased apart from the placeholder-scanning loop first — a real design question, not a mechanical cut.
2. **`_check_loop_condition` (~83 lines)**: same manifest-generator coupling (2 literal `LintIssue` sites, one of them the sanctioned dual-severity `WARN/ERROR` case) plus it is invoked from two different call sites in `_walk` (once for `Construct`, once for `Node`) — needs to keep both.
3. **The real fix, named but not designed here**: convert `lint.py` into a `lint/` package (`lint/__init__.py` re-exporting `lint`, `LintIssue`, `LINT_KIND_META` for `neograph/__init__.py`'s existing import to keep working; `_checks_di.py`, `_checks_template.py`, `_checks_loop.py`, `_checks_tools.py`, `_predict.py`, `_kind_registry.py`) **and** generalize `gen_api_manifest.py`'s hardcoded single-file AST scan into a glob over `src/neograph/lint/*.py` (or an explicit file list). This is the change that actually gets the file under 500 lines sustainably; it touches a script + its guard test outside `lint.py` itself and deserves a dedicated pass rather than being folded into a mechanical extraction PR.

## 4. Duplication check against epic-active files

Grepped `_agent_spec.py`, `loader.py`, `_wiring.py`, `modifiers.py`, `factory.py`, `_agent_cycle.py` for any of `lint.py`'s internals (`LintIssue`, `_predict_input_keys`, `_check_template_placeholders`, `_check_async_only_tools`, `_check_ask_human_in_mutating_node`, `_check_act_mode_all_idempotent`, `_di_template_var_names`, `_spec_factory`, `_resolve_tool_object`, `def lint`). **No hits.** No real duplication found; `lint.py` is not on Phase 8's touch path and none of the proposed extractions interact with the fusion ModifierCombo work.
