# Code Review Synthesis — 2026-04-06

**Scope**: Node.inputs refactor epic (neograph-kqd) — commits 43f39cd..f58f9b9, diff base 41d910a. 6 source files, 8 example files, ~1200 LoC changed.
**Agents**: 8 ran, 8 produced findings
**Date**: 2026-04-06

## Validation Summary

| Agent | Raw Findings | Verified | False Positives | Deduped |
|-------|-------------|----------|-----------------|---------|
| architecture | 5 | 5 | 0 | 3 (CR-02→PP-05, CR-04→PP-08, CR-05→CON-01) |
| python-practices | 8 | 8 | 0 | 2 (PP-02→DRY-01, PP-04→CR-03, PP-05→CR-02, PP-08→CR-04) |
| layering | 6 | 4 | 0 | 1 pre-existing (LR-05), 1 positive |
| execution-excellence | 5 | 5 | 0 | 3 (EE-01→CR-03, EE-02 unique, EE-03→CR-02) |
| consistency | 6 | 6 | 0 | 1 (CON-01→CR-05) |
| dry | 6 | 6 | 0 | 0 |
| security | 4 | 4 | 0 | 0 |
| testing | 6 | 6 | 0 | 0 |
| **Total** | **46** | **44** | **0** | **7 deduped away** |

After deduplication: **0 Critical, 3 High, 7 Medium, 13 Low** unique findings.

## High Findings (verified)

### HIGH-01: Silent exception swallowing in `_resolve_di_value`
- **Source agents**: python-practices (PP-01)
- **File**: `src/neograph/decorators.py:289-292`
- **Verification**: Confirmed via grep — `except Exception: return None` swallows Pydantic `ValidationError` when constructing bundled DI models.
- **Impact**: Misconfigured `run(input={...})` or `config={"configurable": {...}}` silently degrades — user function receives `None` with no error message. Configuration bugs become runtime data bugs.
- **Pre-existing**: Yes — the old `raw_adapter` had the same pattern. Refactor preserved it rather than fixing it.
- **Recommended action**: At minimum `log.warning("di_resolution_failed", ...)`. Better: let `ValidationError` propagate.

### HIGH-02: Frame-walking namespace builder duplicated
- **Source agents**: dry (DRY-01), python-practices (PP-02)
- **File**: `src/neograph/decorators.py:194-225` and `521-552`
- **Verification**: Confirmed — ~25 lines of identical logic with cosmetic variable name differences. Both build a namespace dict with `FromInput`/`FromConfig`/`Annotated`, walk closure vars, walk 8 caller frames, call `get_type_hints`.
- **Impact**: If the frame-walking strategy changes (e.g., hop limit, marker set), both sites must be updated in lockstep. Introduced by this refactor.
- **Recommended action**: Extract `_build_annotation_namespace(f, frame_depth) -> dict` shared helper. Each caller does its own `get_type_hints(f, localns=ns, include_extras=...)`.

### HIGH-03: `_make_gather_fn` and `_make_execute_fn` are near-identical
- **Source agents**: dry (DRY-02)
- **File**: `src/neograph/factory.py:210-248` and `251-289`
- **Verification**: Confirmed — ~35 lines identical except `mode="gather"` vs `mode="execute"` in the log bind.
- **Pre-existing**: Yes — predates this refactor entirely.
- **Impact**: Any bug fix or feature in one must be replicated in the other.
- **Recommended action**: Merge into `_make_tool_fn(node)` that reads `node.mode` for the log line. Out of scope for the kqd epic but worth filing.

## Medium Findings (verified)

### MED-01: Dead sidecar `fan_out_param` third tuple element
- **Source agents**: architecture (CR-03), python-practices (PP-04), execution-excellence (EE-01)
- **File**: `src/neograph/decorators.py:130-146`
- **Verification**: Confirmed — all 4 `_register_sidecar` calls pass only `(n, f, param_names)`, `fan_out_param` always defaults to `None`. All destructure sites ignore position 2: `fn, pnames, _ = sidecar`.
- **Recommended action**: Remove the third element. Make the sidecar a 2-tuple `(Callable, tuple[str, ...])`.

### MED-02: Error messages say `input=` not `inputs=`
- **Source agents**: architecture (CR-05), consistency (CON-01)
- **File**: `src/neograph/_construct_validation.py:265, 391`
- **Verification**: Confirmed — `f"input={_fmt_type(input_type)}"` in two error message strings. The new fan-in errors correctly use `inputs['{key}']` making this an intra-file inconsistency.
- **Recommended action**: Replace `input=` with `inputs=` in both message strings.

### MED-03: Stale docstring references deleted `_validate_fan_in_types`
- **Source agents**: architecture (CR-01)
- **File**: `src/neograph/_construct_validation.py:43-44`
- **Verification**: Confirmed — docstring says "Both validator walkers" but the second was deleted in kqd.4. AGENTS.md was updated; source docstring was not.
- **Recommended action**: Rewrite to "The sole validator walker (`_validate_node_chain`) consults it".

### MED-04: Silent `try/except (TypeError, ValueError): pass` on Node field mutation
- **Source agents**: python-practices (PP-03)
- **File**: `src/neograph/decorators.py:964-967, 972-975`
- **Verification**: Confirmed — Node is not frozen, so `n.inputs = filtered` and `n.fan_out_param = name` should always succeed. Silent failure would leave wrong data.
- **Recommended action**: Remove the try/except guards. If a future `frozen=True` is planned, add a comment.

### MED-05: `_validate_node_chain` `.inputs`/`.input` fallback not explicit
- **Source agents**: python-practices (PP-06)
- **File**: `src/neograph/_construct_validation.py:84-88`
- **Verification**: Confirmed — works correctly (Construct has `.input`, Node has `.inputs`) but the intent is implicit. `getattr` chain without a type check.
- **Recommended action**: Add `isinstance` check or a clarifying comment.

### MED-06: `_type_name()` noisy output for dict-form inputs in logs
- **Source agents**: consistency (CON-03)
- **File**: `src/neograph/factory.py:71-75`
- **Verification**: Confirmed — `_type_name(node.inputs)` calls `str(dict)` for dict-form inputs, producing unreadable log entries like `input_type="{'verify': list[...]}"`.
- **Recommended action**: Make `_type_name` handle dict by summarizing keys, e.g. `"{verify: MatchResult, ...}"`.

### MED-07: Missing test for `inputs={}` (empty dict)
- **Source agents**: testing (TQ-04)
- **Verification**: Confirmed — `grep -rn 'inputs={}' tests/` returns 0 hits. `_check_fan_in_inputs` iterates zero times and returns without error.
- **Recommended action**: Add a test documenting expected behavior.

## Low Findings (summary only)

| ID | Agent(s) | File | Description |
|----|----------|------|-------------|
| LOW-01 | arch (CR-02), python (PP-05), exec (EE-03) | `_construct_validation.py:113` | Stale docstring `item.input` → `item.inputs` |
| LOW-02 | arch (CR-04), python (PP-08) | `_construct_validation.py:175` | Docstring says "compute" but function receives pre-computed types |
| LOW-03 | exec (EE-02) | `decorators.py:1028-1037` | Duplicate branches in `scripted_shim` (fan-out vs upstream do identical `input_data.get(pname)`) |
| LOW-04 | exec (EE-04) | `decorators.py:1-51` | Module header docstring still describes old raw_fn dispatch |
| LOW-05 | exec (EE-05) | — | Node.inputs vs Construct.input naming duality worth documenting |
| LOW-06 | consistency (CON-02) | `_construct_validation.py:113` | Same as LOW-01 |
| LOW-07 | consistency (CON-04) | `decorators.py` | Synthesized name patterns differ slightly across decorators |
| LOW-08 | dry (DRY-03) | `decorators.py:720`, `factory.py:415` | DI resolution loop duplicated in legacy_shim and make_oracle_merge_fn (pre-existing) |
| LOW-09 | dry (DRY-04) | `factory.py:141,312` | State access polymorphism (dict.get/getattr dispatch) duplicated (pre-existing) |
| LOW-10 | dry (DRY-05) | `decorators.py` | Sidecar re-registration repeated 4x after modifier applications (documented convention) |
| LOW-11 | dry (DRY-06) | `decorators.py` | Name collision check duplicated between construct_from_module and construct_from_functions (pre-existing) |
| LOW-12 | python (PP-07) | `factory.py:332` | `from typing import get_origin` imported inline instead of module-level |
| LOW-13 | testing (TQ-01-03) | tests | Trivial assertions on happy-path tests, unused caplog fixture, global registry pollution without cleanup |

## Patterns Observed

**Cross-agent convergence (3+ agents flagged independently)**:
- **Dead sidecar field** (CR-03, PP-04, EE-01): The `fan_out_param` third element in the sidecar tuple is dead weight — `Node.fan_out_param` is the authority. Three agents independently identified this.
- **Stale docstrings** (CR-01, CR-02, PP-05, PP-08, EE-03, EE-04, CON-02): Six findings across four agents about documentation not catching up with the rename/deletion. The code works; the docs mislead.
- **Frame-walking duplication** (DRY-01, PP-02): Two agents independently flagged the same ~25-line duplication.

**Pre-existing vs introduced by this refactor**:
- **Introduced**: HIGH-02 (frame-walking dup), MED-01 (dead sidecar field), MED-02 (error msg), MED-03 (stale docstring), MED-04 (silent try/except), MED-06 (noisy log), MED-07 (missing test)
- **Pre-existing (preserved, not introduced)**: HIGH-01 (silent exception), HIGH-03 (gather/execute dup), LOW-08 (DI loop dup), LOW-09 (state access dup), LOW-11 (name collision dup), LR-05 (factory→decorators import)

## False Positives Discarded

None — all 46 raw findings were verified as accurate, though 7 were deduplicated across agents.

## Metrics

- **Architecture compliance**: Layer discipline maintained. One deliberate IR-layer concession (`fan_out_param`) documented and justified.
- **Pattern adoption**: `effective_producer_type` single source of truth: fully adopted (0 inline copies). `register_scripted` contract: respected (167 existing callers unbroken). Sidecar lifecycle: `weakref.finalize` used on both sidecars.
- **Test coverage shape**: unit=15, integration/factory=3, e2e=12. Healthy pyramid.
- **Security posture**: 0 critical, 0 high. 4 low theoretical risks, none exploitable in library context.
- **DRY**: 1 new duplication introduced (frame-walking). 4 pre-existing duplications documented but not addressed (out of scope).
