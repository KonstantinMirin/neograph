# DRY Review

**Scope**: `src/neograph/modifiers.py` (new `_PathRecorder` + `Modifiable.map`), `src/neograph/construct.py` (new `ConstructError` + validation helpers), `tests/test_e2e_piarch_ready.py` (new `TestNodeMap`, `TestConstructValidation`).
**Date**: 2026-04-04

## Duplication Map

| Pattern | Occurrences | Files | Extractable? |
|---------|------------|-------|--------------|
| `_type_name(t)` — `getattr(t, '__name__', ...)` helper | 2 | `src/neograph/factory.py:71-75`, `src/neograph/construct.py:282-287` | Yes — promote one copy to a shared util (or import the existing `factory._type_name`) |
| Walking `Each.over` dotted path | 2 | `src/neograph/construct.py:168-223` (`_check_each_path`, static), `src/neograph/compiler.py:262-275` (`each_router`, runtime) | Partially — see finding DRY-02 |
| Walking `model_fields` via `_resolve_field_annotation` | 2 (same file) | `src/neograph/construct.py:194-204` (explicit walk), `src/neograph/construct.py:322-329` (hint scan) | No — different purposes, see DRY-03 (not a finding) |
| `TestConstructValidation` node-setup boilerplate (`a = Node.scripted("a", fn="f", output=...)`, `b = Node.scripted("b", fn="f", input=..., output=...)`) | ~12 | `tests/test_e2e_piarch_ready.py:1741-1857` | Yes — tiny factory helpers, see DRY-04 |

## Findings

### DRY-01: `_type_name` helper duplicated in `construct.py` and `factory.py`

- **Severity**: Medium
- **Category**: Response construction / shared util
- **Occurrences**: 2
- **Files**:
  - `/Users/konst/projects/neograph/src/neograph/factory.py:71-75` — `_type_name(t) -> str | None`, returns `getattr(t, '__name__', str(t))` and returns `None` for `None` input.
  - `/Users/konst/projects/neograph/src/neograph/construct.py:282-287` — `_type_name(tp) -> str`, `hasattr(tp, '__name__')` branch and `repr(tp)` fallback; returns `"None"` for `None` input.
- **Description**: Both functions solve the same problem: produce a human-readable name for a type/class object for logging or error messages. The only semantic differences are (a) how they represent `None` (`None` vs the string `"None"`), and (b) the fallback (`str(tp)` vs `repr(tp)`). A bug fix or extension (e.g. unwrap `list[X]` into `"list[X]"`, handle `Union`, pretty-print `ForwardRef`) would now need to be made in two places — and the `construct.py` version's error messages would especially benefit from unwrapping generics like `list[str]` since users will see those in `ConstructError` output.
- **Proposed extraction**: Move the helper to a new `src/neograph/_types.py` (or add it to an existing shared module like `neograph.node` / `neograph._llm`) as `type_name(tp) -> str`, settling on a single `None` convention. Import from both `factory.py` and `construct.py`. Low-risk: both call sites use it only for building strings.

### DRY-02: `Each.over` dotted-path traversal is implemented twice — once statically, once at runtime

- **Severity**: Medium
- **Category**: Query / path resolution
- **Occurrences**: 2
- **Files**:
  - `/Users/konst/projects/neograph/src/neograph/construct.py:168-223` — `_check_each_path` splits `each.over` on `.`, takes `parts[0]` as root, then walks remaining segments through `model_fields` (`_resolve_field_annotation`) until it hits a terminal it expects to be `list[X]`.
  - `/Users/konst/projects/neograph/src/neograph/compiler.py:262-275` — `each_router` inside `_wire_each` splits `each.over` on `.`, walks via `getattr`/`__getitem__` on the *instance* at runtime to get the collection.
- **Description**: These are not exact duplicates — the first walks *types* via `model_fields`, the second walks *values* via `getattr`. But they are two halves of the same conceptual operation (resolve a dotted path on a Pydantic state tree), and they share the same splitting/iteration structure and the same invariants (root must be a state field, each segment must exist on the parent). A concrete hazard: if `.over` ever grows to support indexing, filters, or escaped dots, that change has to land in both walkers in lockstep or the static checker and the runtime will disagree. The new assembly-time check was added specifically to prevent runtime surprises — so these two walkers diverging would defeat the feature's purpose.
- **Proposed extraction**: Add a small helper in `modifiers.py` (since `Each` lives there) or a new `_paths.py`:

  ```python
  def split_each_path(over: str) -> tuple[str, tuple[str, ...]]:
      """Return (root_field, remaining_segments) for an Each.over path."""
      parts = over.split(".")
      return parts[0], tuple(parts[1:])
  ```

  Minor, but it creates a single place to evolve path syntax and documents the "root + remainder" structure that both walkers currently rediscover. The static walker (`_check_each_path`) and runtime walker (`each_router`) then each keep their own *type-vs-value* stepping loops but share the parsing/contract.

  A stronger alternative: promote a `walk_state_path(root_obj, segments)` helper that both `each_router` and a type-walking sibling can use, but that's a bigger refactor and may not pay off given there are only two call sites.

### DRY-04: `TestConstructValidation` node boilerplate repeats across 12+ tests

- **Severity**: Low
- **Category**: Test setup
- **Occurrences**: ~12 tests, all in one class
- **Files**:
  - `/Users/konst/projects/neograph/tests/test_e2e_piarch_ready.py:1741-1857` — every test in `TestConstructValidation` opens with one or two of these lines:
    - `a = Node.scripted("a", fn="f", output=<Type>)`
    - `b = Node.scripted("b", fn="f", input=<Type>, output=<Type>)`
  - Specific repeats: 1743-1745, 1751-1752, 1760-1761, 1767-1768, 1774-1777, 1782-1785, 1792-1795, 1802-1805, 1812, 1842-1843, 1850-1856.
- **Description**: Each test manually constructs 2–3 nearly identical `Node.scripted(...)` calls differing only in name, input type, and output type. The scripted function name (`"f"`) is a placeholder that never gets registered — validation runs purely on metadata, so the `fn=` argument is noise in the test body. Consolidating setup will shrink the class by ~20 lines and make the actual assertion (which types mismatch) the most visible part of each test.
- **Proposed extraction**: Add module-local or class-local helpers near `TestConstructValidation` at `tests/test_e2e_piarch_ready.py:1738`:

  ```python
  def _producer(name: str, out: type) -> Node:
      return Node.scripted(name, fn="f", output=out)

  def _consumer(name: str, in_: type, out: type) -> Node:
      return Node.scripted(name, fn="f", input=in_, output=out)
  ```

  Each test body then becomes, e.g.:

  ```python
  def test_plain_input_mismatch_raises(self):
      a = _producer("a", RawText)
      b = _consumer("b", Claims, ClassifiedClaims)
      with pytest.raises(ConstructError, match="declares input=Claims"):
          Construct("bad", nodes=[a, b])
  ```

  No fixtures needed — simple factory functions are enough given the per-test variation. Parametrization via `@pytest.mark.parametrize` is *not* recommended here because each test asserts on a distinct error-message fragment; keeping them as separate methods preserves readability.

## Non-findings (explicitly considered and dismissed)

- **`_PathRecorder` vs any existing proxy pattern**: Searched `src/neograph/` — no prior proxy/recorder class exists. `_PathRecorder` is genuinely new and is the minimal implementation (`__slots__` + `__getattr__` returning a fresh instance). Not duplication.
- **`Modifiable.map` vs `Modifiable.__or__`**: `.map()` resolves a `source` argument to an `over` string, then delegates to `self | Each(over=..., key=key)` at `modifiers.py:124`. It does *not* re-implement the `model_copy` logic — the delegation is already the minimal form. The string-vs-callable branch in `.map()` (`modifiers.py:90-122`) is irreducible: each branch emits a differently-shaped `TypeError` on failure, which is the whole point of the sugar. No simplification available without losing the diagnostic messages.
- **`_suggest_hint` vs `_check_each_path` field walk**: Both touch `model_fields`, but `_suggest_hint` (`construct.py:322-334`) iterates *all* fields of *all* producers looking for any `list[input_type]`, while `_check_each_path` (`construct.py:194-204`) walks a *specific* dotted path. They already share `_resolve_field_annotation` and `_extract_list_element`, which is the correct level of sharing. Not duplication.
- **`_source_location` stack walk**: Unique to `construct.py`. No parallel frame-inspection logic elsewhere in `src/neograph/`. Not duplication.
- **`TestNodeMap` test setup**: The 8 tests in `TestNodeMap` mostly use a single-line `Node.scripted("verify", ...)` setup and each exercises a distinct branch of `.map()`. Not enough repetition to warrant extraction.

## Summary

- Critical: 0
- High: 0
- Medium: 2
- Low: 1
- Total duplicated logic blocks: 3
- Estimated lines removable by extraction: ~25 (DRY-01: ~5; DRY-02: ~3 parsing + clearer invariants; DRY-04: ~15–20 in tests)

## Recommended action ordering

1. **DRY-01** first — trivial (delete one copy, import the other), eliminates a latent drift risk in user-visible error messages.
2. **DRY-04** next — purely a test-ergonomics improvement, no source impact, makes the validation tests easier to scan.
3. **DRY-02** only if you plan to extend `Each.over` syntax. For today's `"a.b.c"` dotted-path shape, the two walkers operating on different domains (types vs values) is acceptable separation.
