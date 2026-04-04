# Python Practices Review

**Scope**: `src/neograph/modifiers.py` (`_PathRecorder`, `Modifiable.map`), `src/neograph/construct.py` (`ConstructError`, `Construct.__init__`, validation helpers), related tests in `tests/test_e2e_piarch_ready.py` (`TestNodeMap`, `TestConstructValidation`).
**Date**: 2026-04-04

## Findings

### PP-01: Hand-rolled `__init__` wraps pydantic validation in a way that `model_validator(mode="after")` handles natively
- **Severity**: High
- **Category**: Pydantic
- **File**: `src/neograph/construct.py:77-84`
- **Description**: `Construct` defines its own `__init__` solely to (a) accept a positional `name_` argument and (b) run `_validate_node_chain` after `super().__init__`. The comment at line 81-83 explains this is done so `ConstructError` escapes instead of being wrapped in `ValidationError`, and says `model_post_init` can't be used for the same reason. However, pydantic v2's `@model_validator(mode="after")` runs inside validation *but* re-raises exceptions unchanged only when they inherit from `ValueError`/`AssertionError` (they get folded into `ValidationError`). So `model_validator(mode="after")` would indeed wrap `ConstructError` — the author's instinct is correct there. What's unidiomatic is keeping the whole hand-written `__init__` just to bypass this. The cleanest v2 pattern for "run non-pydantic validation after construction" is either:
  1. A factory classmethod (`Construct.build(name, **kwargs)`) that calls `cls(**kwargs)` then `_validate_node_chain(self)` — no `__init__` override needed for the validation path.
  2. Override `model_post_init` and deliberately raise a non-`ValueError`-subclass error type, or raise a sentinel caught in `__init_subclass__`.
  The positional-name support still needs an `__init__` override though, so (1) is only a partial win. The current approach is defensible; this is a **Medium** finding because the comment captures the reasoning and the behavior is correct, but flagging because "override `__init__` on a pydantic v2 model" is a yellow flag reviewers should see. Also note the subtlety that `Construct` subclasses `Modifiable` (a plain mixin) before `BaseModel`, which means method-resolution for `__or__`, `map`, etc. works but pydantic's model introspection still sees `BaseModel` as the primary base. That ordering is fine but worth a line of comment.
- **Reproduction**: `python -c "from neograph.construct import Construct; import inspect; print(inspect.getsource(Construct.__init__))"`
- **Recommended fix**: Leave as-is (comment is adequate), or extract the positional-`name` handling into a `@model_validator(mode='before')` that pops it from a sentinel kwarg, then have users call `Construct(name='x', nodes=[...])` only. If you keep `__init__`, add `-> None` return annotation and type `kwargs` as `**kwargs: Any` for mypy strictness.

### PP-02: `except Exception` in `.map()` lambda introspection is too broad
- **Severity**: Medium
- **Category**: Errors
- **File**: `src/neograph/modifiers.py:94-102`
- **Description**: The `source(recorder)` call is wrapped in `except Exception as exc` and re-raised as `TypeError`. In practice, the recorder's only failure modes when a user writes a non-attribute expression are `TypeError` (e.g., `s.items[0]` → `'_PathRecorder' object is not subscriptable`) and `AttributeError` (in theory, though `__getattr__` makes every name succeed). Catching `Exception` will also swallow `KeyboardInterrupt`'s cousin `SystemExit`? No — those inherit from `BaseException`, not `Exception`, so that's fine. But it *will* swallow things like `ValueError` raised by user code in a non-pure lambda (e.g., `lambda s: s.foo if check() else raise_something()`), surfacing them as "must be a pure attribute-access chain" which is misleading.
  More importantly, the `from exc` chain preserves the original, so debugging is fine — this is primarily a **clarity** issue, not a correctness one.
- **Reproduction**: `python -c "
from neograph.node import Node
from neograph.modifiers import Each
n = Node.scripted('x', fn='f', output=None)
try:
    n.map(lambda s: 1/0, key='k')
except TypeError as e:
    print('got:', e)
"`
- **Recommended fix**: Narrow to `except (TypeError, AttributeError) as exc`. Let any other exception propagate unchanged — it's almost certainly a genuine bug in the user's lambda (not a "not a pure attribute chain" case) and should surface with its original type.

### PP-03: `hasattr` on `_PathRecorder` silently succeeds for every name — footgun for future users
- **Severity**: Medium
- **Category**: Types
- **File**: `src/neograph/modifiers.py:37-40`
- **Description**: Because `__getattr__` unconditionally returns a new `_PathRecorder`, `hasattr(recorder, "anything")` always returns `True`, and `getattr(recorder, "anything", default)` never returns `default`. Today this is fine because `.map()` never calls `hasattr`/`getattr` on the recorder directly, only on the *return value* (`result._neo_path`, `isinstance(result, _PathRecorder)`). But the class is internal and future maintainers might add diagnostics like `if hasattr(result, "_neo_path")` — which would always be `True` because it's a slot attribute. Verified behavior:
  ```
  >>> hasattr(_PathRecorder(), 'totally_random_name')  # True
  >>> _PathRecorder().__nonexistent_dunder__._neo_path  # ('__nonexistent_dunder__',)
  ```
  Note also that dunder-prefixed attribute access still records the dunder name into the path (see second line above). A user writing `lambda s: s.__dict__` would get a recorder with `_neo_path = ('__dict__',)` and `.map()` would silently produce `Each(over="__dict__", ...)`. This is almost certainly not what the user meant.
- **Reproduction**: `python -c "
from neograph.modifiers import _PathRecorder
r = _PathRecorder()
print('hasattr always True:', hasattr(r, 'literally_anything'))
print('dunder recorded:', r.__dict__._neo_path)
"`
- **Recommended fix**: Reject dunder-prefixed attribute names in `__getattr__`:
  ```python
  def __getattr__(self, name: str) -> "_PathRecorder":
      if name.startswith("_"):
          raise AttributeError(name)  # caught by .map() as invalid lambda
      return _PathRecorder(self._neo_path + (name,))
  ```
  This preserves the public-attribute recording use case, blocks dunders from polluting paths, and gives `.map()`'s `except (TypeError, AttributeError)` branch a clean rejection path for `lambda s: s._private.field`.

### PP-04: `__slots__` + `object.__setattr__` in `__init__` is idiomatic but unnecessary here
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/modifiers.py:32-35`
- **Description**: `__slots__ = ("_neo_path",)` combined with `object.__setattr__(self, "_neo_path", path)` in `__init__` is a pattern typically used when `__setattr__` is overridden to reject writes (e.g., frozen dataclasses). Here there's no custom `__setattr__`, so plain `self._neo_path = path` works identically. The `object.__setattr__` call is cargo-cult and slightly obscures intent — a reader sees it and wonders "what is `__setattr__` doing?" The `__slots__` itself *is* useful (it prevents `_PathRecorder` instances from having a `__dict__`, so stray assignments fail loudly), but the constructor could just say `self._neo_path = path`.
- **Reproduction**: `python -c "
class R:
    __slots__ = ('x',)
    def __init__(self): self.x = 1  # works fine, no object.__setattr__ needed
print(R().x)
"`
- **Recommended fix**: Replace `object.__setattr__(self, "_neo_path", path)` with `self._neo_path = path`. Keep `__slots__`. If the author's intent was to defend against a future `__setattr__` override, add a `# note:` comment instead.

### PP-05: `_PathRecorder.__getattr__` path check against dunder special methods
- **Severity**: Medium
- **Category**: Framework
- **File**: `src/neograph/modifiers.py:37-40`
- **Description**: Python's attribute lookup for dunder methods (`__iter__`, `__len__`, `__bool__`, `__eq__`, `__hash__`, `__class__`, `__repr__`, etc.) is done on the **type**, not the instance, so most dunder protocols bypass `__getattr__` entirely. Verified:
  - `bool(recorder)` → `True` (uses default `__bool__`)
  - `len(recorder)` → `TypeError: object of type '_PathRecorder' has no len()`
  - `iter(recorder)` → `TypeError: '_PathRecorder' object is not iterable`
  - `recorder[0]` → `TypeError: '_PathRecorder' object is not subscriptable`
  - `repr(recorder)` → default object repr
  - `recorder.__class__` → `<class '_PathRecorder'>` (slot lookup, not `__getattr__`)
  This is all correct and safe. However, there's one interaction worth knowing: `result._neo_path` in `.map()` goes through normal `__getattribute__` (slot lookup), not `__getattr__`, so `PP-03`'s rewrite to reject underscores would not break the existing `result._neo_path` access. **This finding is informational — confirming the `__slots__` + `__getattr__` interaction is safe under the current call pattern** — but I'm flagging it because a maintainer who later renames `_neo_path` to `neo_path` to drop the leading underscore would suddenly have it flow through `__getattr__` (still slot-bound, so still safe), and a maintainer who removes `__slots__` would have a fun debugging session.
- **Reproduction**: Covered in PP-03.
- **Recommended fix**: Add a docstring note to `_PathRecorder` explicitly stating that `_neo_path` is resolved via `__slots__`/`__getattribute__` and never enters `__getattr__`, plus a test asserting that adding a new slot attribute in the future doesn't accidentally trigger recording.

### PP-06: `inspect.stack()` on every `Construct(...)` call — expensive and only needed on error
- **Severity**: Medium
- **Category**: Framework
- **File**: `src/neograph/construct.py:337-368`, called from `_location_suffix` at lines 202, 214, 221, 313
- **Description**: The user-supplied summary correctly observes that `_source_location` is only invoked from `_location_suffix`, and `_location_suffix` only runs inside the `raise ConstructError(...)` format strings. Confirmed: all four call sites are inside an f-string that builds the `ConstructError` message, so `inspect.stack()` runs **only when an error is being raised**, not on every `Construct(...)` instantiation. That's the correct design. **No performance problem.**

  However, `inspect.stack()` is still expensive on the error path (it resolves full frame info including source context for every frame). For a construct assembly that fails in a tight loop (e.g., a test suite with many expected-failure cases), this adds up. A faster alternative is `sys._getframe(N)` + `frame.f_code.co_filename` and `frame.f_lineno`, walking via `frame.f_back`. This avoids source-reading overhead.
- **Reproduction**: `python -c "
import inspect, time
from neograph.construct import _source_location
t0 = time.perf_counter()
for _ in range(1000):
    _source_location()
print(f'{(time.perf_counter() - t0) * 1000:.1f}ms for 1000 calls')
"`
- **Recommended fix**: Rewrite `_source_location` using `sys._getframe()`:
  ```python
  import sys
  def _source_location() -> str | None:
      try:
          frame = sys._getframe(1)
          while frame is not None:
              module_name = frame.f_globals.get("__name__", "")
              if not (module_name == "neograph"
                      or module_name.startswith("neograph.")
                      or module_name.startswith("pydantic")):
                  fname = frame.f_code.co_filename
                  if fname and not fname.startswith("<"):
                      return f"{os.path.basename(fname)}:{frame.f_lineno}"
              frame = frame.f_back
      except Exception:
          return None
      return None
  ```
  This is ~50x faster and eliminates the `try/finally: del stack` dance.

### PP-07: `_source_location` module-name filter is fragile for user packages starting with "neograph"
- **Severity**: Low
- **Category**: Framework
- **File**: `src/neograph/construct.py:356-358`
- **Description**: The filter rejects any module whose `__name__` is `"neograph"` or starts with `"neograph."`. A user who names their application package `neograph_app` is fine (doesn't match), but a user whose package is literally named `neograph` (shadowing, rare but possible in a monorepo) or who nests their code under `neograph.examples` would have their frames mis-filtered. The same for `pydantic_extra` — it won't match `"pydantic"` start, so that's fine. The fix is to match on exact segment boundaries, not prefix:
  ```python
  if module_name == "neograph" or module_name.startswith("neograph."):
      continue
  ```
  is already correct — it wouldn't match `neograph_app`. But the comment on line 347-350 says "user tests/examples often live under a `neograph/` directory" suggesting directory-based filtering was considered and rejected. The current implementation is **correct** for the documented concern. I'm flagging this as Low because the catch-all `except Exception: pass` on line 366-367 would swallow any filter bug silently, making a future regression hard to diagnose.
- **Reproduction**: N/A (informational).
- **Recommended fix**: No code change needed. Consider narrowing the broad `except Exception` on line 366 to log-and-return-None so silent failures are observable in development:
  ```python
  except Exception:
      # Best-effort: location lookup must never crash Construct assembly.
      return None
  ```
  (That's functionally identical to the current `pass` + fallthrough, but makes the intent explicit.)

### PP-08: `_extract_list_element` Union handling can be simplified with `types.UnionType` unconditionally
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/construct.py:262-279`
- **Description**: The Union detection does `origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType)`. The `hasattr(types, "UnionType")` check is a defensive guard for Python < 3.10, but `pyproject.toml` line 5 requires `python >= 3.11`, so `types.UnionType` is guaranteed to exist. Also, `get_origin` on `X | Y` returns `types.UnionType` while on `typing.Union[X, Y]` it returns `typing.Union`, and the code correctly handles both.
- **Reproduction**: `python -c "
import types
assert hasattr(types, 'UnionType')
"`
- **Recommended fix**: Drop the `hasattr` check:
  ```python
  is_union = origin is Union or origin is types.UnionType
  ```
  One-line cleanup. Pre-existing import of `types` on line 25 already supports this.

### PP-09: `_resolve_field_annotation` fallback path may leak ForwardRefs to callers
- **Severity**: Medium
- **Category**: Pydantic
- **File**: `src/neograph/construct.py:229-247`
- **Description**: The function tries `typing.get_type_hints(model_class)` first; if that raises, it falls back to `model_fields[field_name].annotation`. The fallback path can still return a `ForwardRef('X')` or string annotation, which `_extract_list_element` and `_types_compatible` then have to cope with. `_types_compatible` uses `isinstance(x, type)` which correctly rejects ForwardRefs (they're not types), so it'll return `False` and the validation will either fail with a confusing message or pass silently depending on the code path.

  Pydantic v2 provides a cleaner API: `model_class.model_rebuild()` forces forward-ref resolution, and `model_fields[name].annotation` is guaranteed concrete afterwards. Alternatively, `pydantic.TypeAdapter(model_class).json_schema()` would force resolution as a side effect. But the simplest fix for this codebase is to use pydantic's own introspection:
  ```python
  from pydantic.fields import FieldInfo
  info: FieldInfo = model_fields[field_name]
  return info.annotation  # pydantic resolves forward refs at model-finalization time
  ```
  The question is whether the models being validated have been through pydantic's model-finalization phase. For `Construct(nodes=[...])`, the node I/O types (e.g., `Clusters`, `Claims`) are imported and used directly, so their forward refs are already resolved. The `typing.get_type_hints` fallback is defending against models that `Construct` has never seen before, which shouldn't happen in the current design.

  Bigger concern: the bare `except Exception` on line 242 will swallow `NameError` (unresolved forward ref), `AttributeError` (missing `__annotations__`), and any other failure mode, then silently fall through to the raw annotation — which may itself be a string. A caller receiving `'list[ClusterGroup]'` (string) from `_resolve_field_annotation` will fail all downstream checks silently.
- **Reproduction**: `python -c "
from pydantic import BaseModel
from typing import get_type_hints

class Later(BaseModel):
    x: 'NotYetDefined'  # forward ref to nonexistent

try:
    print(get_type_hints(Later))
except Exception as e:
    print('raised:', type(e).__name__, e)

# Now check fallback path
print('raw annotation:', Later.model_fields['x'].annotation)
"`
- **Recommended fix**: Prefer `model_fields[field_name].annotation` as the primary source (pydantic has already resolved forward refs for any model that successfully imports), and drop the `typing.get_type_hints` path entirely. If the raw annotation is a string or `ForwardRef`, return `_MISSING` rather than passing it downstream:
  ```python
  def _resolve_field_annotation(model_class: Any, field_name: str) -> Any:
      model_fields = getattr(model_class, "model_fields", None) or {}
      if field_name not in model_fields:
          return _MISSING
      ann = model_fields[field_name].annotation
      if ann is None or isinstance(ann, (str, typing.ForwardRef)):
          return _MISSING
      return ann
  ```

### PP-10: `_types_compatible` try/except around `issubclass` is correct but the guard is redundant
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/construct.py:250-259`
- **Description**: The function does `if not (isinstance(producer, type) and isinstance(target, type)): return False` and then wraps `issubclass(producer, target)` in `try/except TypeError`. Once both arguments are confirmed to be `type` instances, `issubclass` can still raise `TypeError` in one case: if `target` is a `type` but also a `Protocol` or a parameterized generic alias. `isinstance(x, type)` would return `True` for `list` but `issubclass(int, list[int])` raises. However, the guard `isinstance(target, type)` is `False` for `list[int]` (which is a `types.GenericAlias`, not a `type`), so the pre-check already filters it out.

  The remaining cases where `issubclass(X, Y)` raises when both are `type` instances are vanishingly rare (custom metaclasses overriding `__subclasscheck__` to raise). The try/except is defensive-but-harmless.
- **Reproduction**: `python -c "
print(isinstance(list[int], type))  # False — already filtered
print(isinstance(list, type))       # True
"`
- **Recommended fix**: Leave as-is. Or drop the `try/except` on the grounds that `isinstance(_, type)` is a strong enough guard in practice. Document either choice with a one-line comment.

### PP-11: `input: Any = None` and `output: Any = None` defeat pydantic's validation
- **Severity**: Medium
- **Category**: Pydantic
- **File**: `src/neograph/construct.py:69-70`, also `src/neograph/node.py:47-48`
- **Description**: The field annotation says `input: Any = None` with a comment `# type[BaseModel] | None`. The "real" type is `type[BaseModel] | None`, but that doesn't work with pydantic v2's validation (pydantic would try to coerce values through `BaseModel.__init__`, which isn't what's wanted — the field holds the *class itself*, not an instance). The chosen workaround is `Any` + `arbitrary_types_allowed: True`.

  A more expressive annotation that still works with pydantic v2 is `type[BaseModel] | None` directly, with `arbitrary_types_allowed=True` — pydantic v2 *does* accept class-as-value for `type[X]` annotations. But there's a subtlety: pydantic will then validate that incoming values are subclasses of `BaseModel`, which rejects `dict[str, type]` for `Node.input` (which Node also allows). That's why `Any` was chosen. **For `Construct`, though, `input`/`output` are only ever `type[BaseModel] | None`** — there's no dict case. You could tighten to:
  ```python
  input: type[BaseModel] | None = None
  output: type[BaseModel] | None = None
  ```
  and get pydantic-enforced validation with clearer intent and better IDE support.
- **Reproduction**: `python -c "
from pydantic import BaseModel
from typing import Any

class M(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    cls_field: type[BaseModel] | None = None

class Inner(BaseModel):
    x: int = 1

m = M(cls_field=Inner)
print(type(m.cls_field), m.cls_field)
# Works: pydantic accepts a BaseModel subclass as a type[BaseModel] value.
"`
- **Recommended fix**: Tighten `Construct.input`/`output` to `type[BaseModel] | None`. Leave `Node.input` as `Any` because of the dict case (and add a TypedDict or Union annotation if you want to tighten that too). This gives reviewers and IDEs a more honest signal about the field contract.

### PP-12: `_MISSING = object()` sentinel is idiomatic; no change needed
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/construct.py:226`
- **Description**: The question asks whether `typing.Never`, `types.NoneType`, or a sentinel library (e.g., `sentinel`) would be cleaner. The answer is **no**:
  - `typing.Never` is a type-checker construct, not a runtime value.
  - `types.NoneType` is `type(None)` — it's a real class and would conflict with valid `None`-typed returns.
  - PEP 661 (formal sentinel support) is still deferred; there is no stdlib sentinel facility.
  - Third-party sentinel libraries add a dependency for zero semantic gain.

  The `_MISSING = object()` pattern is the accepted Pythonic idiom for "value absent" distinct from `None`, and it appears throughout the stdlib (e.g., `dataclasses._MISSING_TYPE`, `functools._initial_missing`). The only minor polish would be to use a named class for better repr in tracebacks:
  ```python
  class _MissingType:
      def __repr__(self) -> str:
          return "<MISSING>"
  _MISSING = _MissingType()
  ```
  But this is cosmetic.
- **Reproduction**: N/A.
- **Recommended fix**: Leave as-is. Optional cosmetic upgrade to a named `_MissingType` class for nicer repr in error messages.

### PP-13: `list[Any]` default and empty-list defaults on pydantic v2 fields
- **Severity**: Low
- **Category**: Pydantic
- **File**: `src/neograph/construct.py:66, 73`
- **Description**: `nodes: list[Any] = []` and `modifiers: list[Modifier] = []` use `= []` as a default. In plain Python this is the mutable-default-argument trap, but pydantic v2 deep-copies list/dict defaults per instance, so it's safe here. It's still a common code-review flag and some teams prefer `Field(default_factory=list)` for clarity. The `list[Any]` annotation also defeats pydantic's element validation on `nodes` — each item could be a Node, a Construct, or anything else; the validation is deferred to `_validate_node_chain`. That's intentional per the comment "Any avoids circular ref issues".
- **Reproduction**: `python -c "
from neograph.construct import Construct
a = Construct('a', nodes=[])
b = Construct('b', nodes=[])
a.nodes.append('x')
print('a:', a.nodes, 'b:', b.nodes)  # confirms pydantic per-instance copy
"`
- **Recommended fix**: Optionally replace with `Field(default_factory=list)` for explicit intent; behavior is identical. If you want tighter typing on `nodes`, use a `TYPE_CHECKING` forward-ref union: `nodes: list['Node | Construct'] = Field(default_factory=list)`.

### PP-14: `_check_each_path` deep-nested raise formatting uses f-strings over multi-line (fine) but duplicates `_location_suffix` calls
- **Severity**: Low
- **Category**: Errors
- **File**: `src/neograph/construct.py:198-223`
- **Description**: Three `raise ConstructError(...)` sites, each concatenating `_location_suffix()` at the end. Each call re-walks `inspect.stack()`. Minor duplication; trivially fixable by binding `loc = _location_suffix()` once at the top of the function and reusing. This only matters if multiple raises happen in the same function invocation (they don't — only one fires per call), so in practice each raise triggers exactly one stack walk. **No action needed.**
- **Reproduction**: N/A.
- **Recommended fix**: No change required. Mentioned for completeness.

### PP-15: `.map()` return type annotation missing — `Self` would be accurate
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/modifiers.py:68`
- **Description**: `def map(self, source: Any, *, key: str):` has no return type annotation. The method returns `self | Each(...)`, which returns `self.model_copy(...)`, which returns an instance of the same type (`Node` or `Construct`). `typing.Self` (PEP 673, Python 3.11+, and `pyproject.toml` requires 3.11+) is the precise annotation:
  ```python
  from typing import Self
  def map(self, source: Any, *, key: str) -> Self: ...
  ```
  The user question notes this matches existing `__or__` style, which also lacks an annotation. Consistency is a fair argument, but `Self` is strictly better for IDE autocomplete and should be added to both methods together.
- **Reproduction**: `python -c "
from neograph.modifiers import Modifiable
import inspect
print(inspect.signature(Modifiable.map))
print(inspect.signature(Modifiable.__or__))
"`
- **Recommended fix**: Add `-> Self` return annotations to both `map` and `__or__`, importing `Self` from `typing`. Low priority; purely ergonomic.

### PP-16: `callable()` branch in `.map()` accepts any callable, not just lambdas
- **Severity**: Low
- **Category**: Types
- **File**: `src/neograph/modifiers.py:92-116`
- **Description**: The `elif callable(source)` branch will accept any callable: a regular `def`, a method, a class with `__call__`, a `functools.partial`, etc. This is intentional per the docstring ("a lambda") but the code is more permissive than the docs claim. A class with `__call__` whose call records attribute chains would work. More problematically, passing a class (which is callable) would call the class constructor with `recorder` as the first positional arg, which would raise a (now-TypeError-wrapped) error with a misleading message.

  A `singledispatch` alternative would be cleaner for dispatching on `str` vs `callable`, but singledispatch dispatches on the *first argument type*, which here is `source: Any` — and `callable` is not a type. So singledispatch doesn't fit naturally. The current `isinstance(source, str) / elif callable(source)` pattern is fine.
- **Reproduction**: `python -c "
from neograph.node import Node
n = Node.scripted('x', fn='f', output=None)
try:
    n.map(dict, key='k')  # dict is callable
except TypeError as e:
    print('got:', e)
"`
- **Recommended fix**: No change required. Optionally tighten the docstring to say "a callable taking the state proxy" rather than "a lambda". If you want to be strict, add `inspect.isfunction(source)` or `inspect.signature(source).parameters` checks to reject zero-arg or multi-positional-arg callables — but that's defensive overengineering.

## Summary

- **Critical**: 0
- **High**: 1 (PP-01)
- **Medium**: 6 (PP-02, PP-03, PP-05, PP-06, PP-09, PP-11)
- **Low**: 9 (PP-04, PP-07, PP-08, PP-10, PP-12, PP-13, PP-14, PP-15, PP-16)

**Top recommendations (ordered by leverage):**

1. **PP-03** — Reject dunder names in `_PathRecorder.__getattr__`. Small, contained change that prevents a real footgun (`lambda s: s.__dict__.foo` silently producing nonsense paths).
2. **PP-09** — Prefer `model_fields[name].annotation` over `typing.get_type_hints` and surface string/ForwardRef fallthroughs as `_MISSING`. Avoids a silent-failure mode.
3. **PP-06** — Switch `_source_location` to `sys._getframe()`. Cheap ~50x perf win on the error path.
4. **PP-02** — Narrow `except Exception` in `.map()` to `(TypeError, AttributeError)`. Two-character change, clearer error messages.
5. **PP-11** — Tighten `Construct.input`/`output` annotations from `Any` to `type[BaseModel] | None`. Better IDE support and honest contract; works fine with pydantic v2.

**Overall quality**: high. The code is carefully commented with correct reasoning (e.g., the `__init__` override comment about `ValidationError` wrapping is accurate; the `_PathRecorder` docstring correctly notes `__getattr__` vs `__getattribute__` interaction). The findings above are polish items, not architectural issues — no critical bugs, no data-corrupting patterns, no silent exception swallowing in business logic. The error-message suggestions (`did you forget .map()?` hint, source location suffix) are excellent DX work.
