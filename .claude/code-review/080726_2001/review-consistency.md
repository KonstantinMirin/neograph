# Consistency Review

**Scope**: Full neograph source tree (`src/neograph/`, 76 modules). Cross-module
consistency of naming, error handling, the plural/singular API split, import
discipline, and warning/log emission. Read-only review.
**Date**: 2026-07-08
**Reviewer framing**: would a senior Python engineer call this "elegantly engineered"?

## Verdict

Mostly yes. The disciplines that are *guard-enforced* are genuinely uniform and
would impress: absolute-only imports, a single error root with rule-based
parentage, `.build()`-only error construction, and centralized state keys are all
consistent to the letter. The inconsistencies that remain are the ones **no guard
watches**: the public/private module-naming convention is unreliable in both
directions, and the error-formatting body is duplicated in a way the `.build()`
guard structurally cannot catch. None are correctness bugs; several are exactly
the "pockets of convention" smell that undercuts a type-safety/discipline pitch.

## Convention Inventory

| Convention | Canonical form | Following | Diverging |
|------------|----------------|-----------|-----------|
| Intra-package imports | `from neograph.x import ...` (absolute) | 422 | 0 |
| Diagnostics channel | `warnings.warn(...)` + `print(file=sys.stderr)`; **no `logging`** | all | 0 |
| State-bus key strings | `StateKeys.*` in `_state_keys.py` | all real state keys | 0 (18 residual literals are object slots / fn-registry names, not state keys) |
| Error root | `NeographError` + THE RULE parentage (guarded) | all public errors | 0 |
| Error construction | `.build()` / `.of()` (guarded for 5 core classes) | 161 raises | 0 direct string raises of core classes |
| Error message shape | `expected:` / `found:` / `hint:` labels via builder | all builder sites | 0 ad-hoc "expected X got Y" in src |
| Runtime-value param | `input=` (singular) on `run`/`arun`/`stream`/`astream`/`run_isolated` | all | 0 |
| Module public/private | `_foo.py` = private, `foo.py` = public | ~50 | **6 (CON-01)** |
| Error format body | one `NeographError.build` implementation | 1 | **2 duplicates (CON-02)** |
| Assembly-time validation error type | `ConstructError.build` | `_construct_validation.py` | **5 sites raise bare `ValueError`/`TypeError` (CON-03)** |
| Node-label in diagnostics | `[Node 'X' in construct 'Y']` | error builder | 2 warning forms (CON-04) |

## Findings

### CON-01: `_`-prefix does not reliably signal module visibility — in either direction
- **Severity**: Medium (High for a codebase that markets on discipline)
- **Convention**: `_foo.py` = internal, `foo.py` = public API. Applied to ~50 modules.
- **Files**:
  - Public-named but **internal-only** (not imported by `__init__.py`, not in `__all__`):
    `factory.py`, `state.py`, `naming.py`, `di.py`. These sit un-prefixed next to
    genuinely public `construct.py`, `node.py`, `compiler.py`.
  - Underscore-named but **exporting public API** (imported by `__init__.py`, symbols
    in `__all__`): `_llm.py` (`render_prompt`, `compile_prompt`, `LlmFactory`,
    `PromptCompiler`, `CostCallback` — `__init__.py:22`), `_image.py`
    (`configure_image`, `resolve_image` — `__init__.py:21`).
- **Description**: The prefix is not a load-bearing signal. `di.py` is pure internal
  machinery (`DIKind`/`DIBinding`, all-`_`-prefixed helpers, 9 internal importers) yet
  reads as public; `factory.py`/`state.py`/`naming.py` are equally internal to
  `_execute.py`/`_dispatch.py` yet only the latter carry the `_`. Conversely `_llm.py`
  reads as private but is a documented public surface. A reader (or tooling) that infers
  visibility from the filename is wrong both ways.
- **Reproduction**:
  ```bash
  cd src/neograph
  # public-named, internal-only:
  for m in factory state naming di; do echo -n "$m: "; grep -c "neograph.$m" __init__.py; done   # all 0
  # underscore-named, public API:
  sed -n '21,28p' __init__.py                                                                     # imports from _image, _llm
  grep -nE '"render_prompt"|"compile_prompt"|"LlmFactory"' __init__.py                            # in __all__
  ```
- **Recommended fix**: Pick one and align. Either rename the four internal modules
  (`factory.py`→`_factory.py`, `state.py`→`_state.py`, `naming.py`→`_naming.py`,
  `di.py`→`_di.py`) and move `_llm.py`/`_image.py`'s public symbols to a public module
  (or accept them as public and drop the `_`), **or** add a structural guard that pins
  the public-module set to `__all__`'s source modules so the convention is enforced
  rather than aspirational. The un-guarded state is why it drifted.

### CON-02: `NeographError.build` message-assembly body is duplicated verbatim
- **Severity**: Medium
- **Convention**: One error-format implementation (`NeographError.build`, `errors.py:49-90`).
- **Files**: `errors.py:129-162` (`ExecutionError.build`) reproduces the entire
  prefix-building + `expected`/`found`/`hint`/`location` assembly **verbatim**, only to
  thread a `validation_errors` kwarg into `__init__`. `errors.py:213-217`
  (`StateMissingError.build`) hand-rolls a third, partial copy of the `[Node 'X']` prefix.
- **Description**: The `.build()` guard (`test_guards_function_local_imports.py`,
  `ERROR_CLASSES`) forces call sites to *use* `build`, guaranteeing uniform wording —
  but it cannot see that the two `build` **bodies** are independent. If the format ever
  changes in one (say, `found:` → `got:`), `ExecutionError`-category messages silently
  diverge from every other error, defeating the very uniformity the guard exists to
  protect. This is the highest-leverage fix: it is a latent inconsistency, not yet a
  visible one.
- **Reproduction**: `diff <(sed -n '71,90p' errors.py) <(sed -n '143,162p' errors.py)`
  — identical but for the trailing `return` line.
- **Recommended fix**: Extract the body into a module-level `_format_message(...) -> str`
  and have both `build` classmethods call it; `ExecutionError.build` then only adds the
  `validation_errors=` pass-through. Fold `StateMissingError`'s prefix into the same helper.

### CON-03: assembly-time validation raises three different exception types
- **Severity**: Low–Medium
- **Convention**: "a failure during Construct assembly / validation → `ConstructError`"
  (errors.py THE RULE; `_construct_validation.py` obeys it with `ConstructError.build`
  at lines 181/271/282/296/328).
- **Files**: The Pydantic `@field_validator` bodies raise bare exceptions instead —
  `node.py:109/111/117` (`TypeError`), `construct.py:46/49` (`TypeError`),
  `modifiers.py:422` (`ValueError: Oracle n must be >= 1`), `modifiers.py:497`
  (`Each.over must not be empty`). `conditions.py:66/110/121` raises bare `ValueError`
  from `parse_condition` (outside any validator).
- **Description**: Raising `ValueError`/`TypeError` *inside* a Pydantic field validator
  is idiomatic (Pydantic re-wraps into `ValidationError`), so the field-validator cases
  are defensible. But the net effect is two assembly-time validation surfaces that
  produce two different exception types and two message shapes: a fan-in type mismatch
  surfaces as a structured `ConstructError` (`[Node 'X' in construct 'Y'] ... expected/found`),
  while "Oracle n must be >= 1" surfaces as a Pydantic `ValidationError` wrapping a bare
  string. A consumer writing `except ConstructError` (or `except NeographError`) around
  assembly catches the first and misses the second. `conditions.py` is the clearer
  divergence — a bare `ValueError` not wrapped by Pydantic, straight past THE RULE.
- **Reproduction**:
  ```bash
  grep -nE 'raise (ValueError|TypeError)' node.py construct.py modifiers.py conditions.py
  grep -n 'ConstructError.build' _construct_validation.py
  ```
- **Recommended fix**: Leave field-validator raises as-is (idiomatic) but document the
  seam; convert `conditions.py`'s parse failures to `ConstructError.build` (or a
  documented `ConfigurationError`) so every user-reachable assembly/parse failure is a
  `NeographError` subclass with the structured format.

### CON-04: node-label prefix has three cosmetic forms across diagnostics
- **Severity**: Low
- **Convention**: `[Node 'X' in construct 'Y']` (error builder).
- **Files**: `decorators.py:176,394` warnings use `@node 'X':`; `_construct_validation.py:161`
  warning uses `Node 'X':`; `errors.py` uses `[Node 'X' in construct 'Y']`.
- **Description**: Three ways to name the same node in a diagnostic. Harmless to
  behavior, but a grep for a node's messages won't catch all three, and it reads as
  three authors rather than one system.
- **Reproduction**: `grep -rnE "@node '|Node '|\[Node '" src/neograph/*.py`
- **Recommended fix**: Standardize the warning prefix on the error-builder form (or a
  single shared `_node_label(name)` helper).

### CON-05: the word "input" carries three meanings
- **Severity**: Low (partly documented-intentional)
- **Convention**: AGENTS.md documents `Node.inputs` (plural, fan-in type map) vs
  `Construct.input` (singular, boundary-port type) as an intentional structural split.
- **Files**: `node.py:167` (`inputs:` field, types), `construct.py:122` (`input:` field,
  boundary type), `node.py:339` + `runner.py:712/766/859/907/937` (`input=` param, a
  runtime **value**).
- **Description**: On top of the intentional plural/singular type-declaration split, the
  runtime-seed value is *also* called `input` across `run`/`arun`/`stream`/`astream`/
  `run_isolated`. The runtime-value meaning is a standard convention and internally
  consistent, so this is not a defect — but `Node` simultaneously exposing an `inputs`
  field (a type map) and a `run_isolated(input=...)` param (a value) is a real
  first-read confusion surface for the "typed end-to-end" audience the project targets.
- **Reproduction**: `grep -nE '\b(inputs?|input)\b\s*[:=]' node.py construct.py runner.py`
- **Recommended fix**: No rename needed; a one-line class docstring on `Node`
  distinguishing "`inputs` = declared consumer types" from "`input=` = a runtime seed
  value (see `run`/`run_isolated`)" closes the gap cheaply.

### CON-06: the `_neo_` framework prefix is hand-built once outside `_state_keys.py`
- **Severity**: Low
- **Convention**: "No code outside this module should reference `neo_*` as a string
  literal" (`_state_keys.py` docstring; `TestNeoStateKeysCentralized` guard).
- **Files**: `_fan_agent_wrap.py:216` (`f"_neo_pack_{field}"`), `:229`
  (`f"_neo_unpack_{field}_{...}"`) synthesize **scripted-function registry names** with
  the framework prefix as raw f-strings.
- **Description**: Not a state-bus-key leak (these are function-registry names, a
  different namespace), so the guard correctly ignores them — but it is the one place
  the `_neo_` prefix convention is reproduced by hand rather than sourced from a named
  constant. If the framework prefix ever changes, these two names won't follow. Worth a
  `StateKeys.FRAMEWORK_PREFIX`-derived helper or at minimum a comment noting the
  parallel namespace.
- **Reproduction**: `grep -n '_neo_pack_\|_neo_unpack_' _fan_agent_wrap.py`
- **Recommended fix**: Derive from `StateKeys.FRAMEWORK_PREFIX`, or add a sibling
  `StateKeys.fan_pack_fn(field)` static method so all `_neo_`-prefixed name-building
  lives in one module.

## Non-findings (verified consistent — worth stating for the "elegant?" question)

- **Imports**: 422 absolute intra-package imports, **0 relative**. Uniform.
- **No logging framework**: zero `logging` usage; all diagnostics go through
  `warnings.warn` (5 sites, sensible `UserWarning`/`DeprecationWarning` split) and
  `print(file=sys.stderr)` (CLI + `describe_graph`). No mixed channels.
- **Error construction**: 0 direct string-arg raises of the 5 core error classes; all
  route through `.build()`/`.of()`. Error wording (`expected:`/`found:`/`hint:`) is
  uniform — no ad-hoc "got X, saw Y" phrasings in `src/`.
- **State keys**: every real state-bus key is a `StateKeys.*` symbol; the 18 residual
  `neo_`/`_neo_` literals are Python object `__slots__` (`forward.py`, `modifiers.py`)
  and fn-registry names, not state keys (CON-06 aside).
- **Boolean flags**: `auto_resume`, `is_async`, `include_extras`, `idempotent` — mild
  prefixed/bare mix but no same-concept-two-names collision; not worth a finding.

## Summary

- Critical: 0
- High: 0
- Medium: 2 (CON-01 module-visibility naming; CON-02 duplicated error-format body)
- Low: 4 (CON-03 assembly error types; CON-04 node-label prefix; CON-05 "input"
  overload; CON-06 hand-built `_neo_` prefix)

The guard-enforced conventions are airtight; the divergences all live in the gaps
between guards. CON-01 and CON-02 are the two a senior reviewer would flag first, and
both are cheaply closable — CON-02 by a shared `_format_message` helper, CON-01 by
either renaming the six mis-signalled modules or adding a public-surface guard so the
convention stops relying on manual vigilance.
