# IR-item dispatch: polymorphism vs. centralized helpers

**Date:** 2026-06-04
**Status:** Decision required (architect-level). Do NOT spawn further per-operation
dedup tickets against this disease until this is decided.
**Origin:** Emerged while closing the Jun-2026 disease-scan dedup epic
(`neograph-97c6`: etxo, ovx1, e27b, x3f0, 7s2n, hbhi, my84). All seven were
symptoms; this doc names the root cause.

---

## 1. The observation

A reviewer asked, looking at `forward.py`'s loop-boundary inference:

```python
input_type = (
    normalize_outputs(source_node.outputs).primary
    if isinstance(source_node, Node)
    else getattr(source_node, 'output', None)
)
```

> "Would it be correct to say that all `isinstance` and `getattr` are signs of smells?"

No — that is too strong, and acting on it would damage correct code. `isinstance`
/ `getattr` are legitimate when:

- parsing untyped/external input at a boundary (Pydantic `BeforeValidator`,
  `config['configurable'][name]` lookups);
- expressing a runtime fact the type system cannot (`_isinstance_safe` over
  `Union`/`Optional`; the `_is_construct_like` `TypeGuard`);
- duck-typing types that **cannot** share a base (see §3);
- attribute access that is legitimately optional (`getattr(t, '__name__', str(t))`
  — generic aliases have no `__name__`).

They are a **smell only when they are recurring *structural* type-dispatch** — the
same closed-set discrimination scattered across call sites. The smell is the
*missing single abstraction* and the *duplication*, not the keyword. The signal
is **count and dispersion**.

By that test, neograph has the smell.

---

## 2. Census (2026-06-04)

Reproducible:

```bash
grep -rn "isinstance([a-z_]*, *Node)\|isinstance([a-z_]*, *Construct)\|isinstance([a-z_]*, *_BranchNode)" src/neograph/*.py
grep -rn "getattr([a-z_]*, *['\"]\(output\|outputs\|input\|inputs\|nodes\)['\"]" src/neograph/*.py
```

- **52** `isinstance(_, Node | Construct | _BranchNode)` dispatch sites across **15**
  files: `__main__`, `_construct_graph`, `_construct_validation`,
  `_construct_builder`, `_ir_normalize`, `_wiring`, `_validation_inputs`,
  `compiler`, `forward`, `verify`, `_validation_types`, `construct`, `state`,
  `lint`, `testing`.
- **~9** `getattr(item, 'output'|'outputs'|'input'|'inputs'|'nodes', ...)` IR-attr
  accesses papering over the same union (`_construct_validation:134/137`,
  `_validation_types:71/72/110`, `forward:562`, `_llm_render:260`,
  `runner:213/219`).

These are not random. They cluster into a handful of *operations* that each
re-discriminate the same three item types:

| Operation | "Centralized" helper (already extracted) | Ticket |
|---|---|---|
| walk leaf nodes | `iter_nodes(construct)` | x3f0 |
| declared output of an item | `_declared_output(item)` | x3f0 |
| modifier-adjusted producer type | `effective_producer_type[_for]` | etxo |
| dict-vs-single outputs/inputs | `normalize_outputs` / `normalize_inputs` (~10 files) | pre-existing |
| primary output state-field | `primary_output_field(base, outputs)` | my84 |

Each ticket moved one discrimination into one helper. **That is centralized
*procedural dispatch*, not polymorphism** — the helper still `isinstance`-es
internally. We traded "isinstance scattered 18×" for "one isinstance per helper
+ a growing pile of helpers." The forward-boundary site above is the same
`_declared_output` selector hand-rolled a *third* time, and it can't cleanly
reuse `_declared_output` because that helper is **cluster-private to the
validation package** (not in `_construct_validation.__all__`) and `forward.py`
is the DX layer — reusing it would punch a layer boundary.

---

## 3. Root cause

There are three sibling IR item types that share a `Construct.nodes` list but
**no behavioral interface**:

- `Node` — `class Node(BaseModel)`; declares `outputs` (plural: `type | dict | None`).
- `Construct` — `class Construct(Modifiable, BaseModel)`; declares `output`
  (singular: `type[BaseModel] | None`, a boundary port) and `nodes`.
- `_BranchNode` — `class _BranchNode(Modifiable)` (`forward.py:350`). **Not a
  Pydantic model** (`Modifiable` is a plain mixin).

The existing protocols in `_ir_protocols.py` only expose *data shape*:
`ConstructItem` = `{name, modifier_set}`; `ConstructLike` =
`{name, input, output, nodes}`. Neither exposes the *behavior* callers actually
need — `declared_output()`, walkability, `effective_producer_type()`,
`primary_output_field()`. So every operation over the list re-discriminates the
types by hand or routes through a per-operation free function.

That is the bigger problem: **no unified behavioral surface for an IR item.**

---

## 4. Why we can't naively "just add methods"

Two real constraints, both deliberate:

1. **`_BranchNode` is non-Pydantic.** Behavior added to `Node`/`Construct` as
   methods will not uniformly cover `_BranchNode`; it would need parallel
   hand-written methods, re-introducing drift by another door.
2. **Layer discipline (AGENTS.md).** The IR types are intentionally *dumb data*;
   behavior lives in the compiler/validation layer. This is load-bearing — it is
   *why* the planned TypeScript port can reuse the same IR. Pushing
   `effective_producer_type`/`declared_output` onto `Node` as methods would bake
   Python-runtime behavior into the IR and break that separation.

So the choice is genuinely a trade, not an obvious win.

---

## 5. Options

**A. Status quo — per-operation free helpers.** Keep extracting one helper per
discrimination (the seven-ticket approach). Each helper `isinstance`-es once,
centrally.
- Pro: respects dumb-data IR + layering; TS port stays clean; small, safe diffs.
- Con: helper count grows; nothing stops a *new* operation from hand-rolling the
  dispatch again (only a per-helper structural guard catches each one); cross-layer
  reuse blocked (the `forward` ↔ `_declared_output` case); the "52 isinstance"
  number trends down slowly but never structurally to ~0.

**B. Widen the `ConstructItem` protocol + a single adapter.** Add the needed
operations (`declared_output`, `is_container`, `child_nodes`, `effective_producer_type`)
to a richer protocol in a **neutral low-level module**, implemented once per type
via a small adapter/visitor — `Node`/`Construct`/`_BranchNode` stay dumb data;
one module owns the dispatch table; callers depend on the protocol, never `isinstance`.
- Pro: one home for *all* item dispatch (not one-per-operation); callers branch-free;
  cross-layer reuse works (neutral module); a single guard ("no `isinstance(_, Node)`
  outside the adapter") replaces N per-helper guards.
- Con: bigger up-front change; must define the protocol surface carefully; the
  adapter still `isinstance`-es internally (but in exactly one place, by design).

**C. Methods on the types.** Rejected up front — violates §4 (both constraints).
Listed only to record that it was considered.

---

## 6. Recommendation

Pursue **B as a design spike, measured against A**:

1. Inventory the *operations* (not call sites) that discriminate IR items — the
   §2 table is the seed; confirm it is complete (`describe`, `state`-builder,
   `verify`, `lint` walkers included).
2. Draft the widened protocol surface and a single adapter module (neutral layer,
   importable by both validation and DX/`forward`).
3. Migrate 2–3 representative call sites (incl. the `forward` boundary that
   started this) as a proof; measure the isinstance-site delta and whether the
   layering/`_BranchNode` constraints actually hold in practice.
4. **Decision gate:** if the spike collapses ≥~40 of the 52 sites and keeps
   `_BranchNode`/layering clean → adopt B and retire the per-op helpers behind the
   adapter. If it fights the constraints → keep A, but add **one** guard banning
   `isinstance(_, Node|Construct|_BranchNode)` outside an allowlisted set of
   dispatch homes, so new operations are forced through an existing helper.

Either outcome ends with a single structural guard, not N.

---

## 7. Folded-in symptom (do NOT fix standalone)

The `forward.py` loop-boundary consolidation (collapse the inline
`isinstance/​getattr` selector to one expression; drop the `getattr(..., 'output',
None)` default) is **blocked on this decision** — it is census row "declared output
of an item" surfacing in the DX layer. Fixing it ad-hoc now would hard-code the
very duplication this doc is about. It becomes one of the proof-of-concept
migrations in §6.3.

## 8. Guardrail for the interim

Until decided: **stop spawning per-operation dedup tickets** against IR-item
dispatch. They each win a skirmish (real fixes) while the root — no unified
IR-item interface — persists and re-spawns the pattern.
