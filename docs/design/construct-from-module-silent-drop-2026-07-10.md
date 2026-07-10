# Design spike: `construct_from_module` silently drops module-level sub-constructs

**Date:** 2026-07-10
**Status:** spike — recommendation pending independent review
**Found via:** the 0.6.0 consumer-DX audit (ox-troubleshooting-demo migration ticket `cmu.8` proposed "replace `construct_from_functions` with `construct_from_module` to shrink `cascade.py` 2–3x" — which would silently drop the cascade's nested `investigate` sub-construct and its `Each`/`Loop` modifiers).

## The invariant that is violated

**`construct_from_module` must never silently omit a module-level pipeline component.** neograph's whole pitch is "if it compiles, it runs" and fail-loud over silent degradation. A builder that quietly discards a declared sub-construct — producing a *different, valid-looking* compiled graph than the author wrote — is the single most damaging failure mode for that pitch.

## Confirmed behavior (empirical repro)

```python
import types
from typing import Annotated
from pydantic import BaseModel
from neograph import node, construct_from_module, construct_from_functions, FromInput

class Mid(BaseModel): text: str

@node(outputs=Mid)
def enrich(seed: Mid) -> Mid: ...
sub = construct_from_functions("enrich_sub", [enrich], input=Mid, output=Mid)

@node(outputs=Mid)
def seed(topic: Annotated[str, FromInput]) -> Mid: ...

mod = types.ModuleType("demo_mod"); mod.seed = seed; mod.sub = sub
result = construct_from_module(mod)
# -> result.nodes == ['seed']   # 'enrich_sub' silently gone. No error, no warning.
```

`construct_from_module` returns a Construct containing only `seed`. The sub-construct is discarded with no diagnostic.

## Root cause: divergent collection contracts between the two builders

`src/neograph/_construct_builder.py`:

| | `construct_from_module` (walk `vars(mod)`) | `construct_from_functions` (explicit list) |
|---|---|---|
| `Node` w/ sidecar | collect | collect |
| plain `Node` | collect | (n/a) |
| `Construct` | **silently skipped** (`isinstance(attr, Node)` is false) | collect into `sub_constructs`, validate `output` port (`:138`) |
| anything else | **silently skipped** | **raise `ConstructError`** (`:163`) |
| passes `sub_constructs=` to builder? | **no** | yes |

The underlying `_build_construct_from_decorated(..., sub_constructs=...)` **already supports sub-constructs** — `construct_from_functions` proves the wiring works. `construct_from_module` simply never collects or forwards them, and — unlike its sibling — never rejects unrecognized module members.

Example 10 (`examples/10_full_pipeline.py:18-19`) documents the limitation verbatim ("construct_from_module cannot inline subgraphs"), so the *limitation* is known — but it is implemented as a **silent drop** instead of a loud error, which is the actual defect.

## Options

### Option A — fail loud (the guaranteed floor)
When the module walk encounters a `Construct` (or any non-Node it is about to discard), raise `ConstructError` naming it and pointing to the idiomatic form:

> `module 'X' defines a Construct 'enrich_sub' that construct_from_module cannot wire. Sub-constructs have a typed input/output boundary; assemble them explicitly: construct_from_functions('name', [...nodes, enrich_sub]) or Construct('name', nodes=[...]).`

- **Pro:** minimal, philosophy-consistent, preserves the documented limitation, zero ambiguity. Mirrors `construct_from_functions`'s existing else-branch.
- **Con:** leaves the two builders asymmetric — a module mixing @nodes and a sub-construct still can't be walked; the author must switch to the list form.

### Option B — symmetric collection (make it work)
Collect `isinstance(attr, Construct)` in the module walk exactly as `construct_from_functions` does (same output-port validation), and thread `sub_constructs=` into `_build_construct_from_decorated` (which already accepts it). This **eliminates** the "cannot inline subgraphs" limitation and makes the two builders consistent.

- **Pro:** removes the divergence that caused the bug; matches author expectation (a module with a sub-construct just works); no new builder machinery needed.
- **Con — the one real design question:** `vars(mod)` includes **imported** names. A shared sub-construct imported for use elsewhere (`from shared import verify_sub`) would be auto-collected and wired — possibly unintended. Note this risk *already exists latently for imported @node Nodes*, so Option B is consistent with current namespace semantics — but Constructs are shared/imported far more often than individual nodes, so the practical exposure is higher. Construct **instances** carry no reliable "defined in this module" provenance (unlike a function's `__module__`), so a clean imported-vs-defined filter is not obviously available.

## Recommendation

**Target Option B; fall back to Option A if the imported-construct ambiguity cannot be cleanly resolved. Never ship the silent drop under either.** The invariant ("no silent omission") is satisfied by either; B is preferred because it also closes the capability gap and de-duplicates the two builders' contracts. The imported-collection behavior must be **explicitly decided and pinned by a test** during implementation (either: accept namespace semantics + document, or add a provenance filter). This A/B choice is an architectural decision and should be confirmed by an independent reviewer before implementation.

## Docs impact (either option)
- `examples/10_full_pipeline.py:18-19` — the "cannot inline subgraphs" comment becomes false under B; under A it stays true but the behavior is now loud. Update accordingly.
- `website/src/content/docs/walkthrough/scripted-pipeline.mdx` and `runtime/programmatic.mdx` mention `construct_from_module` — check for any "walks all @node functions" phrasing that should acknowledge sub-constructs.

## Audit (sibling silent-drop surfaces)
- Only one module-walk collector exists (`construct_from_module`, `_construct_builder.py:78`). No other `vars(mod)` walker in `src/neograph`.
- **Open question for the implementer:** `ForwardConstruct` compiles to a Construct; a module-level `pipeline = MyForwardConstruct()` instance would also fall through the same non-Node skip. Decide whether it is in-scope (collect/reject) or explicitly out (ForwardConstruct is compiled directly, not walked).

## Cross-project note (not neograph)
ox-troubleshooting-demo ticket `cmu.8`'s AC ("replace `construct_from_functions` with `construct_from_module`, shrink `cascade.py` 2–3x") is unsafe **today** (it would trigger this drop on the `investigate` sub-construct). Under Option B it could become viable; under Option A it stays wrong. Correct that AC on the ox side regardless — `construct_from_functions` is the idiomatic form for the cascade.
