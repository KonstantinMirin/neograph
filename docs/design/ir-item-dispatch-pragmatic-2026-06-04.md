# IR-item dispatch — pragmatic-lens companion

**Date:** 2026-06-04
**Status:** Pragmatic-lens companion to `ir-item-dispatch-2026-06-04.md`. Feeds `neograph-8cqd`.
**Lens:** cost/benefit, reversibility, YAGNI, actual failure modes. Not principle purity.

The seed doc is a clean piece of root-cause analysis and its diagnosis ("no unified
behavioral interface for an IR item") is *correct as a description*. My disagreement is
about **magnitude and remedy**: the doc sizes the disease by an undifferentiated census
(52 sites), then proposes a generalized polymorphism framework (Option B) to cure it. When
you segment the census and look at what actually caused escaped bugs, the true problem is
roughly an order of magnitude smaller than 52, the root cause that produced the *real* bugs
is **already fixed**, and the cheapest high-value move is one guard + one helper promotion —
not a protocol/adapter rewrite of a TS-load-bearing IR. Option B is YAGNI today.

---

## 1. True problem sizing — segment the 52

The seed doc's own census command returns ~50 narrow hits today. I read every one and
binned them by whether they are *recurring structural dispatch* (the disease) or
*legitimate / irreducible / non-code*. Evidence is `file:line`.

### Bin A — NOT code dispatch (drop entirely): 7 sites
- `testing.py:531,542` — strings inside a **code-generator template** (emitted test source). Not dispatch.
- `construct.py:61` — **docstring** illustrating the IR shape.
- `forward.py:159` — **docstring** ("attribute that isinstance(v, Node)").
- `_validation_types.py:7` (and the `_ir_protocols.py` header) — prose.
- `forward.py:167` — node *discovery* on a class body (`isinstance(attr_val, Node)` walking class attributes), a boundary parse, not item dispatch over `nodes`.
- `_construct_builder.py:79` — same: discovering `@node` class attributes.

### Bin B — irreducible sum-type match (a protocol does NOT remove these): ~4 sites
- `compiler.py:209/211/214` — the three-way `_BranchNode` / `Construct` / `Node`
  match that calls **three different graph-builder functions with different signatures**
  (`_add_branch_to_graph` / `_add_subgraph` / `_add_node_to_graph`). This is a genuine sum
  type. Option B's adapter can't collapse it — you still need per-variant behavior with
  per-variant arguments. At best you rename the `isinstance` to a `match`. No value.
- `verify.py:90/164/167`, `testing.py:73/75/91` — same shape: "render Node one way, recurse
  Construct another way" tree walks. Each is local, single-purpose, and the two arms do
  structurally different things. A `child_nodes()`/`is_container()` protocol shaves the
  *recursion guard* but not the per-arm rendering. Marginal.

### Bin C — already monopolized or trivially monopolizable local guards: ~30 sites
The bulk. These are 1-line `if isinstance(item, Node):` filters/guards that are *local,
non-duplicated, and correct*:
- `_construct_validation.py:133/141/164/195/212/226/245/304` — eight guards, each gating a
  *Node-only* validation concern (inputs vs input port, context refs, loop+skip_when,
  dict-form outputs). These are not "the same discrimination scattered" — they are
  **different predicates that happen to start with the same type test**. Folding them behind
  a protocol method would mean inventing 8 protocol methods, one per concern. That is more
  surface, not less.
- `state.py:100-102` — the canonical `[n for n in nodes if isinstance(n, Node)]` /
  `Construct` / `_BranchNode` partition. One place, clear, correct.
- `compiler.py:140,147,163,184`, `lint.py:214/223/359`, `_wiring.py:572/581`,
  `_construct_graph.py:246`, `_validation_inputs.py:286`, `_ir_normalize.py:87/229`,
  `__main__.py:54`, `state.py:128/242` — local guards or error-message branches.
- `_construct_builder.py:134/142/238` — assembly-time Node/Construct handling.

### Bin D — the actual disease (recurring structural selectors): ~2 distinct idioms
- **declared-output selector**: `item.outputs if isinstance(item, Node) else getattr(item, "output", None)`.
  Canonical home: `_declared_output` (`_validation_types.py:110`). Hand-rolled a *third*
  time at `forward.py:559-562` because that helper is cluster-private and `forward` is the
  DX layer. This is the one the seed doc rightly flags.
- **effective-producer-type selector** — already monopolized in `effective_producer_type`
  (`_validation_types.py:76`); the `_construct_graph.py:81` caller correctly imports it. No
  drift here anymore.

**Defensible true problem size: ~2 idioms, with exactly ONE live duplication
(`forward.py` re-rolling `_declared_output`).** Not 52. The "52" conflates docstrings, a
code-gen template, an irreducible compiler sum-type match, and ~30 benign local guards with
the ~2 selectors that are the real smell. Sizing the decision off 52 inflates the case for B
by ~25×.

---

## 2. Real vs hypothetical failure cost

The doc cites `neograph-8k3` and `neograph-ayq` as the drift bugs that justify acting. I
read both tickets and the fix commit (`41d910a`). The story is the opposite of what the "52
isinstance sites" framing implies:

- **Both bugs were the same duplicated *semantic rule*** — "Each modifier means the producer
  writes `dict[str, X]`" — inlined in **two different walkers** (`_validate_node_chain` in
  `_construct_validation.py` and `_validate_fan_in_types` in `decorators.py`). 8k3 fixed one
  walker; ayq was the identical bug in the other walker a day later.
- **They were NOT caused by `isinstance(item, Node)` dispatch.** They were caused by a
  *modifier→state-shape rule* being copy-pasted. The `isinstance` was incidental.
- **That root cause is already eliminated.** `effective_producer_type` is now the single
  source of truth (`_validation_types.py:76`), the second walker
  (`_validate_fan_in_types`) **no longer exists** (grep returns zero hits — the `@node` path
  was unified onto `_validate_node_chain`), and `test_declared_output_ternary_appears_once`
  + the `iter_nodes`/`collect_llm_nodes` monopolies in `test_guards_helper_monopoly.py`
  permanently pin the selectors to one home.

So: **the failure mode that actually escaped to production has already been closed by the
seven-ticket Option-A campaign.** What remains is hypothetical: "some *future* new operation
hand-rolls the dispatch a *fourth* time." The cost of the next such bug is bounded and low —
it's an assembly-time validator gap, caught by the compiler safety-net fixtures
(`tests/check_fixtures/`) and `known_gaps/`, not silent runtime data loss (8k3-class silent
failures were specifically about the *modifier rule*, which is now centralized). The
`forward.py:561` hand-roll is the proof: it has existed for the whole `forward()` epic and
caused **zero filed bugs** — it's a latent duplication, not an active wound.

**Verdict: the cited failures are real but already remediated by A; the residual risk B
would buy down is hypothetical and cheap.**

---

## 3. Options ranked by value / effort

| Rank | Move | Value | Effort | Reversible? |
|---|---|---|---|---|
| 1 | **(a) One backstop guard** banning the declared-output / producer-type *idioms* outside their helper homes, with an explicit allowlist for the irreducible compiler match. | High — closes the one open door at the source. | ~1hr (extend `test_guards_helper_monopoly.py`; the pattern is already there). | Trivially. |
| 2 | **(b) Promote `_declared_output` to a neutral module** so `forward.py` (DX layer) can reuse it instead of re-rolling `getattr(..., 'output', None)`. Kills the ONE live duplication and unblocks the folded-in `forward.py:561` symptom. | High — removes the only real Bin-D duplication; ~5-line move. | ~1-2hr incl. import-DAG guard update. | Trivially. |
| 3 | **(d) Do nothing** beyond what the dedup epic shipped. | Medium — A already closed the real bug class. | Zero. | n/a |
| 4 | **(c) Full Option B** — widen `ConstructItem`, build adapter/visitor module, migrate sites. | Low *today* — collapses Bin C (benign) and can't touch Bin B (irreducible); the real Bin-D win is achievable by (a)+(b) at 5% of the effort. | Days, on a TS-load-bearing IR. | Hard (see §4). |

(a) and (b) are complementary and together *are* the seed doc's "Option A + one guard"
fallback — done properly. They should ship together this sprint. (c) is dominated.

---

## 4. Reversibility / risk of Option B

Option B touches `_ir_protocols.py` (the structural typing seam the validator and `forward`
both depend on) and introduces an adapter module every walker routes through. Blast radius:

- **`_BranchNode` is non-Pydantic and the protocol can't be `runtime_checkable` for the
  narrowing path** (see `_is_construct_like`'s deliberate "NOT runtime_checkable" comment,
  `_ir_protocols.py:44`). So the adapter still `isinstance`-es internally — B's own doc
  concedes this. You don't remove the dispatch; you relocate it and add an indirection layer
  in front of it. Every future reader now has to learn the adapter.
- **TS port is aspirational, not real.** `find -name '*.ts'` for neograph source: zero. The
  "load-bearing for the TS port" constraint is honored *today* by keeping IR dumb — which
  (a)+(b) preserve perfectly (helpers in a neutral free-function module are exactly as
  portable as a visitor; arguably more, since a visitor encodes Python dispatch mechanics).
  B's claimed cross-language advantage over A is **nil** until TS source exists, and
  `typescript-timeline.md` only commits to "TS follows" with no date.
- **Rollback story for B is poor**: once N callers depend on `item_adapter.declared_output(x)`,
  backing out means re-inlining N sites. (a)+(b) back out by deleting one guard and one
  re-export.

**The "≥40 of 52" success gate is a vanity metric.** ~30 of those 52 are Bin C benign local
guards and Bin A non-code. Driving that number to 40 means *converting correct, clear,
single-purpose local guards into protocol-method calls* — i.e. manufacturing churn to hit a
target. Hitting 40 would be evidence of over-application, not success. The honest metric is
"how many *duplicated selectors* remain," and that number is **1** today (forward.py),
fixable by (b) alone.

---

## 5. YAGNI verdict

Option B builds a generalized item-polymorphism framework for a problem that **3 helpers + 2
monopoly guards have already 80% solved** (and the seventh ticket closed the last *real* bug
class). The remaining 20% is one duplication and one open door — addressable by (a)+(b).

B pays off only when at least one of these becomes true, and **none is true yet**:
1. **A 4th IR item type lands** (e.g. a parallel/group node) needing the same N operations —
   then a shared contract amortizes. Today it's 3 types, stable since the `forward()` epic.
2. **The TS port starts and forces a shared, language-neutral item contract** — at which
   point you'd design the protocol *against the actual TS Zod/JSON-Schema constraints*, not
   speculatively in Python now. Designing it now risks baking Python-isms into a contract you
   intended to be portable — the exact failure mode the layer discipline guards against.
3. **The selector count climbs back toward double digits** despite the guards — i.e. the
   guards prove insufficient. They haven't been tried as a *backstop* yet (only per-helper);
   give them one sprint first.

Until one of those fires, B is speculative generality. **YAGNI: defer B.**

---

## 6. Concrete sequenced plan with STOP/GO gate

**This sprint (do now, ~half a day total):**

1. **Promote `_declared_output`** from `_validation_types.py` (cluster-private) to the
   existing neutral low-level module that both validation and `forward` may import (the
   `normalize_outputs`/`primary_output_field` home in `_ir_normalize.py` is the natural seam
   — it's already neutral and already imported by `forward.py:557,560`). Re-export from the
   validation cluster for back-compat. Update the import-DAG guard.
2. **Fix the folded-in `forward.py:559-562` symptom** by calling the promoted
   `_declared_output(source_node)` instead of the inline `getattr(..., 'output', None)`
   ternary. This is `neograph-8cqd`'s blocked symptom — it unblocks the moment (1) lands.
3. **Add ONE backstop guard** in `test_guards_helper_monopoly.py`: the declared-output
   idiom (`isinstance(_, Node) ... else getattr(_, 'output'...)` AND the bare
   `getattr(item, 'output', None)` variant) may appear only inside `_declared_output`'s home.
   Allowlist the irreducible `compiler.py:209-215` sum-type match explicitly (it is *correct*
   dispatch, not a selector). This generalizes the existing
   `test_declared_output_ternary_appears_once` to also catch the `getattr` form that
   currently slips it.
4. **Update `AGENTS.md`** "effective_producer_type" section to note the declared-output
   selector is also monopolized + guarded, and that `compiler.py` sum-type dispatch is the
   sanctioned exception.

**STOP/GO gate for Option B (revisit, do NOT build now):**

GO to a B spike *only if* one of these triggers fires:
- a 4th `Construct.nodes` item type is proposed, OR
- TS port work is scheduled (then design the contract against TS constraints), OR
- after the backstop guard ships, ≥3 *new* duplicated selectors appear within two quarters
  (guard proven insufficient).

If none fires, B stays unbuilt. Close `neograph-8cqd` as "resolved via Option A done
properly (helper promotion + backstop guard); B deferred behind the gate above."

---

## 7. This-sprint vs defer

**This sprint:** §6 steps 1–4. Net effect: the *only* live duplication dies, the DX-layer
symptom is fixed, a single backstop guard closes the open door, and the decision ticket
closes. Small, reversible, layer-clean, TS-neutral.

**Defer (gated):** Option B in full. It is a solution to a 1-duplication problem dressed as a
52-site problem. Build it when a 4th type or the TS port makes a shared contract pay for
itself — not before.

---

## 8. Where I disagree with the seed doc

1. **Sizing.** "52 isinstance sites" is not the problem size; ~2 idioms with 1 live
   duplication is. The doc treats Bin C benign local guards and Bin A docstrings/codegen as
   if they were the disease.
2. **The "≥40 of 52" gate is a vanity metric** that would *reward over-application*. The
   right metric is duplicated-selector count (=1).
3. **The root cause it names ("no behavioral interface") is real but is NOT what caused the
   cited bugs.** 8k3/ayq were a duplicated *modifier rule*, already centralized — the
   `isinstance` was incidental. The doc lets those bugs argue for B when they actually argue
   for "centralize the rule," which A already did.
4. **B's cross-language/TS advantage over A is asserted, not real today** — no TS source
   exists, and free-function helpers are at least as portable as a visitor.
5. **Agreement:** the doc is right that `forward.py` re-rolling `_declared_output` is the one
   genuine open wound, and right that per-helper guards should converge to a backstop. I just
   reach those via (a)+(b) and stop there, rather than gating them behind a B spike.
