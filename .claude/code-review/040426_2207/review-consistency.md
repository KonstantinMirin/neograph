# Consistency Review

**Scope**: New code in `src/neograph/modifiers.py` (`_PathRecorder`, `Modifiable.map`), `src/neograph/construct.py` (`ConstructError` + 12 private validation helpers + `_MISSING`), `src/neograph/__init__.py` (`ConstructError` export), and `tests/test_e2e_piarch_ready.py` (`TestNodeMap`, `TestConstructValidation`).
**Date**: 2026-04-04

## Convention Inventory

| Convention | Canonical Form | Files Following | Files Diverging |
|------------|---------------|-----------------|-----------------|
| Raise idiom | `msg = "..."; raise X(msg)` (EM101-style) | `state.py` (3), `compiler.py` (2), `factory.py` (3), `_llm.py` (3), `modifiers.py` (6), `runner.py` (1) — 18 sites | `construct.py` new code — 4 sites at lines 163, 198, 208, 218 raise with the f-string/helper passed inline |
| Exception classes | Raise stdlib `ValueError` / `TypeError` directly; no custom class hierarchy | 6 modules, 18 raise sites | `construct.py` introduces `ConstructError(ValueError)` — the first custom exception in the package |
| Error message shape | One-line, `Subject 'name' problem-statement. Remediation-sentence.` | `state.py:40,63,112`, `compiler.py:55,92`, `factory.py:57,66,91`, `_llm.py:107,197,280`, `runner.py:70` (13 sites, all end in period) | `construct.py:199-223,308-314` — multi-line messages with embedded newlines, `hint:` line, and `at file:line` suffix; `modifiers.py:97-122` — four `.map()` TypeError messages ending in `{exc}` / `{type.__name__}`, no terminal period |
| Identifier quoting in errors | Single quotes around names: `Node 'x'`, `Sub-construct 'x'`, `Construct 'x'`, `Tool 'x'`, `Condition 'x'`, `Scripted function 'x'` | All 18 existing sites | New code matches (`Node 'x' in construct 'x'`) — consistent |
| Private helper prefixing | `_verb_noun` (`_add_`, `_make_`, `_wire_`, `_extract_`, `_compile_`, `_parse_`, `_call_`, `_get_`, `_is_`, `_strip_`, `_inject_`, `_collect_`, `_merge_`, `_last_`, `_accept_`) | `state.py`, `factory.py`, `compiler.py`, `_llm.py`, `runner.py` | `construct.py` new helpers use `_validate_`, `_check_`, `_resolve_`, `_extract_`, `_format_`, `_suggest_`, `_type_name`, `_location_suffix`, `_source_location` — same verb-prefix style, consistent |
| `_type_name` helper | Defined once in `factory.py:71` (`(t: Any) -> str \| None`, returns `None` when `t is None`) | `factory.py` uses it across 5 log sites | `construct.py:282` defines a second `_type_name(tp: Any) -> str` that returns the literal string `"None"` — duplicate symbol with subtly different semantics |
| `typing` imports | Absolute at top of module: `from typing import ...` | `node.py`, `state.py`, `factory.py`, `modifiers.py`, `tool.py`, `_llm.py`, `compiler.py`, `runner.py` | `construct.py` matches top-level import (`from typing import Any, Union, get_args, get_origin, get_type_hints`) — consistent. (One existing in-function import exists at `factory.py:289`, which is itself the outlier.) |
| Union type syntax | PEP 604 `X \| Y` / `X \| None` in annotations | Every annotation across the package | `construct.py:26` imports `Union` only for runtime `origin is Union` comparison (a legitimate exception — `get_origin(X \| Y)` returns `types.UnionType`, not `typing.Union`, and the new code correctly handles both) — consistent |
| Sentinel pattern | `_ACCEPT_ALL = frozenset({"__all__"})` in `_llm.py:36` is the only pre-existing module-level sentinel; "not found" is expressed via `None` with `is None` checks (15 sites across 5 modules) | `state.py`, `construct.py`, `_llm.py`, `factory.py`, `compiler.py` | `construct.py:226` introduces `_MISSING = object()` — new sentinel style not used elsewhere in the package |
| Docstring section headers | `Usage:` then optional `Args:` / `Returns:` (PEP 257, not Google-strict) | `tool.py`, `node.py` (`Node.run_isolated` in particular), `modifiers.py:Oracle/Each/Operator`, `compiler.py:compile`, `runner.py:run`, `_llm.py:configure_llm` — 13 sites | `modifiers.py:68-89` (`Modifiable.map`) uses inline RST `::` numbered examples and an inline "Returns a new instance..." paragraph, no `Usage:` / `Args:` / `Returns:` headers; sibling newly-added method `Node.run_isolated` (same commit generation) uses the canonical `Usage:` + `Args:` headers |
| Test class docstrings | One-line class-level docstring summarizing the area | 29 of 37 test classes (every class from `TestExecuteMode` at line 576 onward) | `TestNodeMap` at line 331 has no class-level docstring. Its direct peer `TestConstructValidation` at line 1738 does (`"""Input/output compatibility is checked at Construct assembly time."""`). (The pre-line-576 test classes also omit it, so `TestNodeMap` matches its immediate neighborhood but diverges from the dominant convention.) |

## Findings

### CON-01: New raises bypass the `msg = ...; raise X(msg)` idiom
- **Severity**: Medium
- **Convention**: Every existing raise site in the package uses a two-line pattern where the message is first bound to a local `msg` variable, then raised. This is the standard flake8-errmsg (`EM101`/`EM102`) style. The ruff config at `pyproject.toml:52-54` does not enable `EM` rules, so the convention is cultural, not enforced — but it is perfectly uniform across the 18 pre-existing raise sites.
- **Files**:
  - Canonical: `src/neograph/state.py:40-41`, `src/neograph/state.py:63-64`, `src/neograph/state.py:112-113`, `src/neograph/compiler.py:54-58`, `src/neograph/compiler.py:92-93`, `src/neograph/factory.py:57-58`, `src/neograph/factory.py:66-67`, `src/neograph/factory.py:91-92`, `src/neograph/_llm.py:107-108`, `src/neograph/_llm.py:197-198`, `src/neograph/_llm.py:280-281`, `src/neograph/modifiers.py:152-153`, `src/neograph/modifiers.py:155-156`, `src/neograph/runner.py:70-71`
  - Diverging: `src/neograph/construct.py:163-165`, `src/neograph/construct.py:198-203`, `src/neograph/construct.py:208-215`, `src/neograph/construct.py:218-223`
- **Description**: The four `raise ConstructError(...)` sites in `construct.py` pass the message (either a helper-function call or an f-string) directly to the constructor without first binding a `msg` local.
- **Reproduction**: `rg -n "msg = " src/neograph/state.py src/neograph/compiler.py src/neograph/_llm.py` vs `rg -n "raise ConstructError" src/neograph/construct.py`.
- **Recommended fix**: Introduce `msg = _format_no_producer_error(...)` / `msg = f"..."` locals before each `raise ConstructError(msg)` to match the dominant pattern.

### CON-02: Duplicate `_type_name` helper with different semantics
- **Severity**: High
- **Convention**: Private helper names are unique across the package; when the same concept reappears, it's imported, not redefined.
- **Files**:
  - `src/neograph/factory.py:71-75` — `def _type_name(t: Any) -> str | None: ... return None` when `t is None`, else `getattr(t, '__name__', str(t))`.
  - `src/neograph/construct.py:282-287` — `def _type_name(tp: Any) -> str: ... return "None"` when `tp is None`, else `tp.__name__` or `repr(tp)`.
- **Description**: Two helpers with the same name but different signatures (`str | None` vs `str`) and different behavior for `None` input (returns `None` vs returns the literal string `"None"`). Grepping for `_type_name` no longer gives a single definition. A reader paging in `construct.py` might reasonably expect `_type_name(None)` to short-circuit to `None` as it does in `factory.py`.
- **Reproduction**: `rg -n "^def _type_name" src/neograph/`.
- **Recommended fix**: Either rename the construct.py version (e.g. `_fmt_type`, `_render_type`) to make the semantic difference visible, or promote a single canonical `_type_name` to a shared utility module (or to `factory.py` and import it). Given that `construct.py` needs the literal `"None"` string for error rendering and `factory.py` needs short-circuiting for log field omission, renaming is the lower-friction fix.

### CON-03: ConstructError messages diverge from package-wide error shape
- **Severity**: Medium
- **Convention**: Every other error message in the package is a single sentence (or two sentences joined with a period) of the form `Subject 'name' problem-statement. Remediation-sentence.`, terminated with a period. See `state.py:63` (`"Sub-construct '{sub.name}' has no output type. Declare output=SomeModel."`), `compiler.py:92` (`"Sub-construct '{sub.name}' has no input type. Declare input=SomeModel."`), `factory.py:57` (`"Condition '{name}' not registered. Use register_condition()."`). 13 existing messages, all periodic, all one-line.
- **Files**:
  - Canonical: `src/neograph/state.py:40,63,112`, `src/neograph/compiler.py:55,92`, `src/neograph/factory.py:57,66,91`, `src/neograph/_llm.py:107,197,280`, `src/neograph/modifiers.py:152,155`, `src/neograph/runner.py:70`.
  - Diverging: `src/neograph/construct.py:198-203` (multi-line, embedded newlines, terminal `\n{_location_suffix()}`), `src/neograph/construct.py:208-215` (three-line error with `hint:` line), `src/neograph/construct.py:218-223`, `src/neograph/construct.py:307-314` via `_format_no_producer_error` (multi-line with `upstream producers:` list, optional `hint:` line, `at file:line` suffix).
- **Description**: `ConstructError` messages are structurally richer — multi-line with a producer inventory, a `.map()` hint, and a source-location pointer. This is arguably a UX improvement, but it's novel: no other error in the package carries hints or source pointers. The divergence is defensible (assembly-time type errors benefit from structured help), and `ConstructError` does subclass `ValueError` so existing callers still catch it, but it breaks reader expectations set by every other error message.
- **Reproduction**: `rg -n 'msg = f?"' src/neograph/ --type py` and compare to `rg -n 'ConstructError' src/neograph/construct.py`.
- **Recommended fix**: Keep the richer format (it's genuinely more useful at assembly time) but document the divergence in a short comment near `_format_no_producer_error`, so future additions to `state.py` / `compiler.py` either adopt the richer format deliberately or stay on the one-line pattern deliberately. Alternatively, promote the `_format_no_producer_error` + `_location_suffix` pattern into a small helper shared by `state.py` and `compiler.py` error sites to bring the rest of the package up to this quality bar.

### CON-04: `.map()` TypeError messages lack terminal period
- **Severity**: Low
- **Convention**: 13 existing `msg = ...` strings in the package all end in `.`. See inventory table for the full list.
- **Files**:
  - Canonical: `src/neograph/_llm.py:107` (`"LLM not configured. Call neograph.configure_llm() first."`), `src/neograph/factory.py:57` (`"Condition '{name}' not registered. Use register_condition()."`), plus 11 others.
  - Diverging: `src/neograph/modifiers.py:97-101` (`"...got error when introspecting: {exc}"`), `src/neograph/modifiers.py:104-107` (`"...got {type(result).__name__}"`), `src/neograph/modifiers.py:111-114` (`"...e.g. \`lambda s: s.make_clusters.groups\`"`), `src/neograph/modifiers.py:118-121` (`"...got {type(source).__name__}"`).
- **Description**: All four TypeError messages emitted by `Modifiable.map()` end with an interpolated expression and no terminal period. Compare `modifiers.py:152` (pre-existing, ends in period) within the same file.
- **Reproduction**: `rg -n 'msg = ' src/neograph/modifiers.py`.
- **Recommended fix**: Append a `.` to each message so they terminate like the rest of the package.

### CON-05: `_MISSING = object()` sentinel is novel to the package
- **Severity**: Low
- **Convention**: The package has one pre-existing module-level sentinel — `_ACCEPT_ALL = frozenset({"__all__"})` in `_llm.py:36`, used for parameter-introspection matching. "Not found" / "absent" is otherwise expressed via `None` plus `is None` checks (15 sites across `state.py`, `_llm.py`, `factory.py`, `compiler.py`, `construct.py`).
- **Files**:
  - Prior art: `src/neograph/_llm.py:36` (single sentinel, used within one file).
  - Novel: `src/neograph/construct.py:226` — `_MISSING = object()` with identity comparison (`is _MISSING`) at lines 197, 239, 326.
- **Description**: `_resolve_field_annotation` returns `_MISSING` to distinguish "field not present on model" from "field present but its annotation happens to be `None`". In practice a Pydantic field annotation is never `None` (it'd be typed, even if `type(None)`), so a `None` return could serve the same role as the sentinel. That said, the sentinel is defensively correct and the pattern is idiomatic Python — the flag is that the codebase didn't previously use `object()` sentinels, and a reader looking for prior art will find only `_ACCEPT_ALL` (a `frozenset` sentinel for a very different purpose).
- **Reproduction**: `rg -n 'MISSING|sentinel|object\(\)' src/neograph/`.
- **Recommended fix**: Leave as-is — the sentinel is justified by the function's contract. Optionally add a one-line comment at `construct.py:226` noting why the sentinel is needed (to distinguish absent from `None`-valued annotation), matching the self-documenting comment style at `_llm.py:36`.

### CON-06: `_PathRecorder` and `Modifiable.map` docstrings skip the canonical `Usage:` / `Args:` headers
- **Severity**: Low
- **Convention**: Public methods and classes in the package consistently use `Usage:` blocks and (when applicable) `Args:` blocks. Clearest example: `Node.run_isolated` at `src/neograph/node.py:104-127` is a brand-new method added in the same generation as `Modifiable.map` and uses both `Usage:` and `Args:`. Other examples: `compile` at `compiler.py:36`, `run` at `runner.py:53`, `configure_llm` at `_llm.py:55-97`, `Oracle` / `Each` / `Operator` at `modifiers.py:137,167,184`, `Tool` at `tool.py:29`, `tool` at `tool.py:101`.
- **Files**:
  - Canonical: `src/neograph/node.py:104-127`, `src/neograph/modifiers.py:127-156` (Oracle), `src/neograph/_llm.py:55-97`.
  - Diverging: `src/neograph/modifiers.py:19-31` (`_PathRecorder` — private class, inline code block without `Usage:` header), `src/neograph/modifiers.py:68-89` (`Modifiable.map` — uses RST `::` numbered examples and an inline "Returns a new instance..." paragraph; no `Usage:` / `Args:` / `Returns:` headers).
- **Description**: `Modifiable.map` is a public method (exposed on `Node` and `Construct`) whose docstring uses a different shape from its sibling `Node.run_isolated` added at the same time. `_PathRecorder` is private so less critical, but the inconsistency with `Oracle` / `Each` / `Operator` (which also have inline code in `Usage:` blocks) is still visible.
- **Reproduction**: `rg -n 'Usage:|Args:|Returns:' src/neograph/` and compare to `src/neograph/modifiers.py:68`.
- **Recommended fix**: Restructure `Modifiable.map` docstring to use `Usage:` (with the two numbered examples) and `Args:` (for `source` and `key`) headers, matching `Node.run_isolated`. `_PathRecorder` is fine as-is because it's private.

### CON-07: `TestNodeMap` lacks the class-level docstring its peers use
- **Severity**: Low
- **Convention**: 29 of 37 test classes in `tests/test_e2e_piarch_ready.py` have a one-line class-level docstring (see the `TestExecuteMode` onward block, lines 576-2459). The `TestConstructValidation` sibling in the same change set (line 1738) follows the convention. The eight classes without a docstring are all in the "old half" of the file (lines 58-517).
- **Files**:
  - Canonical: `tests/test_e2e_piarch_ready.py:2084-2085` — `class TestRunIsolated: """Node.run_isolated() — direct invocation for unit testing."""` (another recently-added class), `tests/test_e2e_piarch_ready.py:1738-1739` — `class TestConstructValidation: """Input/output compatibility is checked at Construct assembly time."""`.
  - Diverging: `tests/test_e2e_piarch_ready.py:331-332` — `class TestNodeMap:` followed directly by the first `def test_...`.
- **Description**: `TestNodeMap` sits in the older half of the file near other doc-less classes (`TestEach`, `TestOperator`) so it matches its immediate neighbors, but new test classes added since `TestExecuteMode` have uniformly picked up the one-line class docstring convention. `TestRunIsolated` (a sibling addition from the same generation of changes) follows the convention; `TestNodeMap` doesn't.
- **Reproduction**: `rg -nU '^class Test\w+:\n    """' tests/test_e2e_piarch_ready.py` vs `rg -n '^class Test' tests/test_e2e_piarch_ready.py`.
- **Recommended fix**: Add a one-liner such as `"""Node.map() — lambda- and string-path fan-out sugar over `| Each(...)`."""` to `TestNodeMap`.

## Summary

- Critical: 0
- High: 1 (CON-02)
- Medium: 2 (CON-01, CON-03)
- Low: 4 (CON-04, CON-05, CON-06, CON-07)
