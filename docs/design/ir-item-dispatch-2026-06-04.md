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

---

## 9. Three-lens review synthesis (2026-06-04)

Three independent Opus analyses were run against §1–8 — DDD, SOLID/GRASP, and
pragmatic engineering. Companion docs:
`ir-item-dispatch-ddd-2026-06-04.md`, `-solid-grasp-2026-06-04.md`,
`-pragmatic-2026-06-04.md`. Where they converge is now high-confidence; where
they split is the actual decision.

### 9.1 Strong convergence (do this regardless of A/B)

All three independently land on the **same concrete first move**, at three
ambition levels but one mechanism:

- **The only *live* duplication is `_declared_output` being cluster-private.**
  `forward.py:~561` hand-rolls the `Node.outputs` / `Construct.output` selector
  *because* `_declared_output` (`_validation_types.py:110`) is not in
  `_construct_validation.__all__` and `forward` is the DX layer. DDD: "imprisoned
  domain logic"; SOLID: "relocate to a neutral module + fix the DIP"; pragmatic:
  "promote to `_ir_normalize.py`, point forward at it." Same move.
- **New finding the seed missed — a live DIP inversion / context leak.**
  `compiler.py:44`, `state.py:16`, `_wiring.py` import `_BranchNode` *upward*
  from the DX-layer `forward.py`. `_BranchNode` is a core-IR concept living in a
  high layer. Both DDD (anti-corruption) and SOLID (DIP) flag it independently.
  Relocate `_BranchNode` to a neutral low-level IR module. This is worth doing on
  its own merits.

### 9.2 The sizing was wrong (pragmatic lens, uncontested)

The "52 isinstance sites" overstates the disease by ~25×. Segmented: ~7 are
non-code (docstrings, the `testing.py` codegen template); ~4 are an *irreducible*
sum-type dispatch (`compiler.py:209-215` routes to three different graph builders
— no protocol removes this); ~30 are benign, non-duplicated **local Node-only
guards** (legitimate, not the disease). The true recurring-dispatch problem is
**~2 selector idioms with exactly one live duplication**. Neither the DDD nor
SOLID lens contradicts this — they argue the *shape* of B is right, not that the
52 are all real disease.

Consequences:
- The **"collapse ≥~40 of 52" gate in §6.4 is a vanity metric** — it would reward
  converting *correct* local guards into protocol calls. Strike it.
- The cited drift bugs (`8k3`/`ayq`) were a duplicated **modifier rule**, which
  is **already centralized** in `effective_producer_type` (second walker deleted).
  They argue "centralize the rule" (done) — **not** for a general dispatch
  framework.
- The **"load-bearing for the TS port" justification is aspirational**: there is
  no `.ts` source yet. It cannot carry Option B's weight today.

### 9.3 The shape, when it's time (DDD + SOLID, converged)

If/when B is triggered, its principled form — both lenses agree — is **not** the
seed's literal "widen `ConstructItem`":
- **One neutral IR-ops module** owning all item-level dispatch (DDD calls it a
  domain service; SOLID calls it GRASP **Pure Fabrication**). The
  Information-Expert (the type itself) is overridden by the two constraints —
  which is *exactly* the textbook condition for Pure Fabrication, so the adapter
  is canonically correct, not a workaround. The seed undersold it as "a trade."
- **Capability-segregated protocols** (e.g. `HasDeclaredOutput`, `Walkable`,
  `HasBoundaryPort`), **not** one fat `ConstructItem` (ISP; avoids re-coupling
  `forward` to validation vocabulary).
- **Structural `match` over the closed union, not the Visitor pattern** — Visitor
  needs `accept()` on the dumb IR types, which the layering forbids.
- The real defect under LSP: singular `output` vs plural `outputs` + `getattr(…,
  None)` defaults is a **heterogeneous closed union papered as homogeneous**.

### 9.4 Revised recommendation (supersedes §6)

Two-step, decision-gated:

1. **NOW (this is "A done properly" *and* B's first migration *and* the domain-
   service seed — all three at once):**
   a. Create a neutral low-level IR module (working name `_ir_ops.py`) and move
      `_declared_output` there; relocate `_BranchNode` out of `forward.py` into a
      neutral module (fixes the DIP inversion). Point `forward.py` at the shared
      selector; drop the `getattr(source_node, 'output', None)` default.
   b. Add **one** backstop guard: generalize the existing
      `test_declared_output_ternary_appears_once` to also ban the
      `getattr(item,'output',None)` form and `isinstance(_, Node|Construct)`
      selector idioms outside the allowlisted IR-ops home.
   Half-to-one day, reversible, layer-clean. Resolves the only live duplication
   and the context leak.

2. **DEFER full Option B** (capability protocols + match-based dispatch module,
   retiring the per-op helpers) behind a **real trigger**, not a site count:
   *a fourth IR item type is introduced* **OR** *the TypeScript port starts and
   forces a shared item contract*. Until then it is YAGNI; §9.3 records the shape
   so the decision is pre-made when the trigger fires.

Net: the seed's instinct (no unified interface is the root) is correct; its
*urgency and scope* were inflated. The cheap convergent move closes the real
wound now; the framework waits for a real trigger.
