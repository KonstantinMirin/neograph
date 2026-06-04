# IR-item dispatch — the SOLID/GRASP lens

**Date:** 2026-06-04
**Status:** SOLID/GRASP-lens companion to `ir-item-dispatch-2026-06-04.md`. Feeds
`neograph-8cqd`. Read the seed doc first; this one deepens and, in two places,
disputes its framing.
**Scope:** read-only architectural analysis. No production code touched.

---

## 0. TL;DR

The seed doc is right about the disease (no unified behavioral surface; 52
isinstance sites; per-op helpers are procedural dispatch, not polymorphism) and
right to prefer B. But it under-specifies B and slightly mis-frames the trade:

- B as written ("widen `ConstructItem` + a single adapter") is **ISP-violating**
  if the protocol is one fat surface. The principled form is **capability-
  segregated protocols + one Pure Fabrication dispatch module** that implements
  them per concrete type.
- The seed names "adapter/visitor" interchangeably. They are not interchangeable
  under LSP: the heterogeneity here is real (singular `output` vs plural
  `outputs`; non-Pydantic `_BranchNode`), which is the textbook signal for a
  **sum-type / match dispatch**, not a row of `getattr` defaults pretending the
  union is homogeneous.
- The sharpest finding is a GRASP one the seed never states: **Information
  Expert points at the type, but the two constraints (non-Pydantic `_BranchNode`,
  dumb-data IR for the TS port) forbid the natural expert — which is the canonical
  trigger for Pure Fabrication.** That is not a compromise of GRASP; it is GRASP
  working as designed. The adapter is not a workaround for the constraints; it is
  what GRASP prescribes *because of* them.

Recommendation: **B, in its segregated-protocol + Pure-Fabrication form**, with
internal dispatch via structural `match` over the three concrete types. Concrete
sketch and migration in §7–§9.

---

## 1. The union, precisely (LSP grounding)

Three siblings share `Construct.nodes` (`construct.py:73`, list element type
`Node | Construct | _BranchNode`):

| Type | Base | Output attr | Containerness |
|---|---|---|---|
| `Node` | `BaseModel` | `.outputs` (plural; `type \| dict \| None`) | leaf |
| `Construct` | `Modifiable, BaseModel` | `.output` (singular; boundary port) | container (`.nodes`) |
| `_BranchNode` | `Modifiable` (non-Pydantic; `forward.py:350`) | `.output` (always `None`, set in `__init__`, `forward.py:363`) | neither |

The only shared surface today is `ConstructItem = {name, modifier_set}`
(`_ir_protocols.py:28-33`). Everything else callers need —
declared output, walkability, modifier-adjusted producer type, primary state
field — is reconstructed at each site by hand.

---

## 2. OCP — Open/Closed: quantified blast radius

The census (`grep`, 2026-06-04): **52** `isinstance(_, Node|Construct|_BranchNode)`
sites across 15 files, plus ~9 `getattr(item, 'output'|...)` defaults.

OCP says: adding a behavior variant should *add* code, not *edit* existing code.
Two concrete change scenarios show the violation:

**Scenario 1 — add a 4th item type** (e.g. a `_MapNode` sentinel, or making
`Loop` a first-class item). Every `isinstance(item, Node)` that means "is this a
leaf producer?" and every `else`/`assert isinstance(item, Node)` fallthrough
must be revisited. Grep the fallthroughs that would silently misclassify a new
type:
- `compiler.py:213-214` — `else: assert isinstance(item, Node)`. A 4th type trips
  the assert at *runtime*, not compile.
- `_validation_types.py:110` — `item.outputs if isinstance(item, Node) else
  getattr(item, "output", None)`. A 4th type with neither attr silently yields
  `None` (a wrong producer type, no error).
- `state.py:100-102` — three parallel list comprehensions partition `nodes` into
  `nodes_only` / `sub_constructs` / `branch_nodes`. A 4th type is silently dropped
  from all three.
- `construct.py:66-70` (`iter_nodes`) — `Construct → recurse; Node → yield; else
  → skip`. A 4th *container* type is silently not walked.

That is at least 4 mandatory edit sites for one new type, several of them
*silent* failures (the worst OCP outcome: the open-ended edit you forget).

**Scenario 2 — add a modifier rule that reshapes state** (the recurring one). The
codebase *already* solved this correctly in one place: `effective_producer_type`
(`_validation_types.py:76-99`) is the single home, and AGENTS.md codifies "teach
this one function, the walker picks it up." That is OCP-closed *for that
operation*. The disease is that this discipline exists for exactly one operation;
the other ~5 operations (walk, declared-output, partition, primary-field,
boundary-port) each have their own closed-or-not home, and nothing structurally
forces a *new* operation to follow suit. The `forward.py:559-562` loop boundary
is the proof: it hand-rolled `_declared_output` a third time because the real
helper is cluster-private and cross-layer reuse is blocked.

**Verdict:** OCP is violated not because isinstance exists, but because the
*number of independent dispatch homes is unbounded and growing*. Each new
operation opens a new closed-or-not decision. The seed's "52 trends down slowly
but never structurally to ~0" is exactly the OCP smell: the asymptote is wrong.

---

## 3. DIP — Dependency Inversion: what to depend on, and where it must live

Today high-level policy modules (`compiler.py`, `state.py`, `lint.py`,
`verify.py`, `_construct_validation.py`) depend on the *concrete* `Node` /
`Construct` / `_BranchNode` classes. DIP says they should depend on an
abstraction, and the abstraction should be owned by neither side's concretion.

There is a live DIP inversion that the seed doc misses and that is decisive for
*where the abstraction lives*:

> `_BranchNode` is defined in `forward.py` (the DX layer, top of the stack), but
> `compiler.py:44`, `state.py:16`, and `_wiring.py:35` all
> `from neograph.forward import _BranchNode`.

Low-level layers import a concrete sentinel *up* from the DX layer. That is a
textbook DIP violation already shipped. Any "just centralize dispatch in the
compiler" answer makes it worse by concentrating the upward concrete dependency.

The correct abstraction-owner is a **neutral low-level module that every layer
may import downward**. Two existing modules prove such a layer exists and is
import-clean:
- `_ir_protocols.py` imports only `modifiers` (`_ir_protocols.py:25`).
- `_normalize.py` imports only `naming` (`_normalize.py:21`).
- `_validation_types.py` does **not** import `forward` (verified: grep for
  `forward` in its imports returns nothing) — yet it must reason about
  `_BranchNode`'s `.output`. It only gets away with this today via
  `getattr(item, "output", None)` (`_validation_types.py:110`) — i.e. it pays the
  LSP tax (§4) precisely to dodge the DIP-illegal import.

So the abstraction belongs in (or beside) `_ir_protocols.py`. The protocol is the
inverted dependency; a small dispatch module implements it. The `_BranchNode`
*type identity* must become importable from a neutral home (move the class down,
or register an adapter), or the dispatch module re-inherits the upward import. I
treat this relocation as in-scope for B — it is the same DIP fix, not a side
quest.

---

## 4. LSP — are they substitutable? No, and that is the real diagnosis

The seed treats the union as "duck-typeable with a few getattr defaults." That is
the LSP violation in disguise.

Liskov: if callers hold `ConstructItem` and the three types were truly
substitutable, no caller would need to know which one it holds. But they
constantly do, and the tells are:

1. **`outputs` (plural) vs `output` (singular).** Not a naming quirk — a genuine
   semantic split (AGENTS.md is explicit: `Node.outputs` is a multi-producer map
   to the state bus; `Construct.output` is a single boundary port). They are
   *different concepts that happen to both answer "what type comes out."* A caller
   asking "what does this item produce for downstream type-checking?" must
   translate both into one answer (`_declared_output`, `_validation_types.py:102`).
   That translation existing at all is the LSP failure: the supertype does not
   expose the behavior, so the client downcasts.

2. **`getattr(item, "output", None)` defaults** (`_validation_types.py:110`,
   `forward.py:562`, `_construct_validation.py:134/137`). A `getattr`-with-default
   over a *closed* union is "I know this attribute is absent on one member but I'm
   pretending the union is uniform." That is a behavioral substitution that *isn't
   safe* — it converts "this type has no such concept" into "the value is None,"
   which then flows into type compatibility as a wrong-but-silent producer type.

3. **`_BranchNode` is not even Pydantic.** It cannot stand in for `Node` in any
   context that calls `model_copy`, `model_fields`, validation, or serialization.
   It is substitutable *only* for the `{name, modifier_set}` slice — which is
   exactly what `ConstructItem` already (correctly) narrows to.

**Diagnosis:** this is **not** a fixable LSP violation (you cannot make the three
honestly substitutable without erasing the singular/plural distinction the design
deliberately keeps). It is a **legitimately heterogeneous closed union**. The
correct mechanism for a heterogeneous closed union is **sum-type dispatch**
(`match`/visitor double-dispatch), *not* a row of `getattr` defaults and *not* a
fake-uniform protocol. The seed's instinct to "widen the protocol" is half right;
the other half is that the *implementation behind* the protocol must be explicit
per-case dispatch, because the cases genuinely differ.

This is the single most important reframing: **the problem is not "missing
polymorphism," it is "heterogeneous union pretending to be homogeneous via
getattr."** The fix is to make the heterogeneity explicit and centralized, not to
paper it flatter.

---

## 5. ISP — critique the fat `ConstructItem`

The seed's B says "widen `ConstructItem`." If taken literally — one protocol
exposing `declared_output`, `is_container`, `child_nodes`,
`effective_producer_type`, `primary_output_field`, `boundary_port` — it violates
ISP. No caller needs all of that:

| Caller | Needs |
|---|---|
| `iter_nodes` / `state.py` partition | walkability (`child_nodes`, `is_container`) |
| validation producer registration | `declared_output`, `effective_producer_type` |
| `forward.py` loop boundary | `declared_output` only (one attr) |
| compiler dispatch | the *kind* (branch/container/leaf) for lowering |
| `_is_construct_like` recursion | boundary port (`input`/`output`/`nodes`) |

Forcing `forward.py` (which wants one thing: "the single output type of my loop
source") to depend on a protocol that also promises `effective_producer_type`
couples the DX layer to validation vocabulary. That is the coupling that *caused*
the cross-layer reuse block in the first place.

**Segregate into capability protocols** (all `runtime_checkable=False`; narrowing
is done in the dispatch module, never by callers):

- `HasDeclaredOutput` — `declared_output: ...` (the one `forward.py` needs)
- `Walkable` — `is_container: bool`, `child_items: Sequence[ConstructItem]`
- `HasBoundaryPort` — `input`, `output`, `nodes` (today's `ConstructLike`,
  already exists at `_ir_protocols.py:36` — rename/keep)
- `ProducesToBus` — `effective_producer_type()`, `primary_output_field()`
  (validation/state-builder only)

Callers depend on the *narrowest* capability. `ConstructItem` stays the minimal
`{name, modifier_set}` root. This directly satisfies ISP and *unblocks* DIP §3:
`forward.py` imports only `HasDeclaredOutput` from the neutral module, never the
validation cluster's `ProducesToBus`.

---

## 6. SRP — who is doing too much

Two SRP observations:

1. **`_validation_types.py` is doing two jobs.** It owns type-compatibility rules
   (`_types_compatible`) *and* IR-item discrimination (`_declared_output`,
   `effective_producer_type`, `_is_construct_like`). The discrimination half is
   needed by `forward.py`, `state.py`, `lint.py` — but it lives in a validation
   leaf, so it cannot be reused without a layer break. Extracting the
   discrimination into the neutral dispatch module gives each module one reason to
   change: `_validation_types` changes when *type rules* change; the dispatch
   module changes when *the item union* changes.

2. **The per-op helper sprawl is an SRP win turned into an OCP loss.** Each helper
   (`iter_nodes`, `_declared_output`, `primary_output_field`,
   `declared_output_fields`) is individually single-responsibility and clean. The
   problem is they share a *hidden* responsibility — "know the three-member union"
   — that is duplicated across all of them (each re-`isinstance`es). The union's
   shape is a single responsibility that has been smeared across ~6 helpers in 3
   modules. Consolidating it is an SRP improvement, not just an OCP one.

---

## 7. GRASP synthesis — the heart

### 7.1 Information Expert says: put it on the type. The constraints forbid it.

GRASP Information Expert: assign a responsibility to the class that has the
information to fulfill it. `declared_output` for a `Node` is `self.outputs`; for a
`Construct` it is `self.output`. The expert is unambiguously the type itself. The
*natural* design is methods on the types — i.e. the seed's rejected Option C.

The two constraints are exactly what overrides Information Expert here:
- **Constraint 1 (`_BranchNode` non-Pydantic):** the union has no common base that
  can carry a method. You would write the method three times (twice on Pydantic
  models, once on a hand-rolled sentinel) — reintroducing the duplication by
  another door (the seed says this; it is correct).
- **Constraint 2 (dumb-data IR for the TS port):** putting *behavior* on the IR
  bakes Python-runtime semantics into the data model that the TypeScript port must
  re-implement. The IR is a *data contract* across two language runtimes; behavior
  is per-runtime.

### 7.2 This is the textbook Pure Fabrication trigger

GRASP **Pure Fabrication**: when assigning a responsibility to the
Information-Expert class would violate High Cohesion, Low Coupling, or
reusability, invent a class that does not represent a domain concept and put the
responsibility there. That is *precisely* this situation: the expert is the type,
but assigning to it violates the cross-runtime-data-contract cohesion and couples
the IR to the Python runtime. The dispatch module is the fabricated class.

**Key insight (the one the seed never states):** the adapter in Option B is not a
pragmatic dodge of the constraints. It is the *canonically correct* GRASP answer
*given* those constraints. "Information Expert wants it on the type, but cohesion/
coupling/reuse say no" is the literal definition of when to reach for Pure
Fabrication. The constraints don't make B a compromise; they make B *the
prescribed pattern*. Anyone arguing B is "more indirection than necessary" is
arguing against GRASP without naming it.

### 7.3 Pure Fabrication + Indirection + Protected Variations are one move here

The fabricated dispatch module simultaneously realizes three GRASP principles:
- **Pure Fabrication** — it is not a domain object; it exists to hold dispatch.
- **Indirection** — callers go through it instead of touching concrete types,
  decoupling policy modules from the union's membership.
- **Protected Variations** — the union's membership (which types, which output
  attr) is the variation point; the module is the stable interface that wraps it.
  Add a 4th type → edit one `match` → every caller is protected (this is the OCP
  §2 fix expressed in GRASP terms).

### 7.4 Polymorphism (GRASP) — yes, but *parametric/ad-hoc*, not inheritance

GRASP Polymorphism is the canonical cure for type-code switches. The subtlety:
classic OO polymorphism (virtual method per subclass) is *blocked* by the two
constraints. So we use **ad-hoc polymorphism via structural `match`** inside the
fabricated module — the dispatch is polymorphic in behavior, but realized as a
single closed `match` rather than scattered methods. Python 3.11 `match` over
`Node()` / `Construct()` / `_BranchNode()` is a sealed-union dispatch: one place,
exhaustive, and a 4th member is a one-line addition flagged by the lone guard.

### 7.5 Visitor vs match vs adapter — pick match

- **Visitor (double dispatch):** requires an `accept(visitor)` method on each
  type → reintroduces a method on `_BranchNode` and on the dumb-data IR. Violates
  both constraints. **Reject.** (The seed lists "visitor" as a realization of B;
  under the constraints it is not viable. Minor mis-framing.)
- **Adapter-per-type registry** (dict `{type: impl}`): works, no method on the
  types, but the registry indirection buys nothing over `match` for a *closed*
  3-member union and hides exhaustiveness. Use only if the union becomes open
  (it is not).
- **Structural `match` in one fabricated module:** simplest realization that is
  exhaustive, importable downward, method-free on the IR. **Adopt.**

---

## 8. Reconciliation with the two constraints

| Constraint | Does the recommendation hold it? |
|---|---|
| `_BranchNode` non-Pydantic | Yes — `match _BranchNode()` needs no method, no Pydantic. The module reads `.output`/`.name`/`.modifier_set`, all set in `__init__` (`forward.py:359-363`). |
| Dumb-data IR / TS port | Yes — zero behavior added to `Node`/`Construct`/`_BranchNode`. The TS port re-implements the *one* dispatch module against its own three types; the IR data contract is unchanged. This is *better* for the port than today's 52 scattered sites, which the port would otherwise have to replicate 52 times. |

One genuine cost: to keep the dispatch module in a neutral layer it must reference
`_BranchNode`'s *type identity*. Either (a) move `_BranchNode` (and `_BranchMeta`)
into a neutral `_ir_branch.py` that `forward.py` re-exports, or (b) have the
dispatch module treat "not Node, not Construct, has `.output`" as the branch case
(structural, no import). (b) is what `_is_construct_like` already does
(`_validation_types.py:69-73`) and avoids the relocation, at the cost of a
structural rather than nominal branch test. I lean (a) — it also fixes the
pre-existing DIP inversion in §3 (compiler/state/_wiring importing up from
forward). Do (a) as part of the spike; fall back to (b) if relocation churns too
much.

---

## 9. Recommendation

**Adopt B, in the segregated-protocol + Pure-Fabrication-dispatch form.** Not A.
Not the seed's literal "one fat protocol." A is a managed decline (the OCP
asymptote never reaches 0; every new operation re-opens the dispatch decision; the
cross-layer reuse block is permanent). The seed's "spike then decide" is sound
process, but the principles already tell you the answer; the spike is to size the
diff, not to re-litigate the verdict.

### 9.1 Interface sketch (module: `_ir_dispatch.py`, neutral low-level)

Imports only `_ir_protocols`, `_normalize`, `node`/`construct`/`_ir_branch` type
identities (all downward). Re-exports nothing behavioral onto the IR.

```text
# _ir_protocols.py  (segregated capabilities — data shape only)
class ConstructItem(Protocol):        # unchanged: name, modifier_set
class HasBoundaryPort(Protocol):      # = today's ConstructLike: input, output, nodes
# (the rest are realized as functions, not protocols callers implement)

# _ir_dispatch.py  (Pure Fabrication — the ONE dispatch home)
def declared_output(item) -> TypeSpecStatic:        # match: Node->outputs, else->output
def is_container(item) -> bool:                      # match: Construct->True, else False
def child_items(item) -> Sequence[ConstructItem]:    # match: Construct->nodes, else ()
def item_kind(item) -> Literal["leaf","container","branch"]:  # for compiler lowering
def boundary_port(item) -> HasBoundaryPort | None:   # narrow or None
# effective_producer_type / primary_output_field MOVE here from _validation_types,
# implemented over declared_output(item) so the modifier rule stays single-source.
```

Each function is a single `match`/exhaustive dispatch. `_validation_types` keeps
`_types_compatible` (pure type-rule SRP) and imports `declared_output` /
`effective_producer_type` from `_ir_dispatch`. `forward.py:559-562` collapses to
`declared_output(source_node)` and drops the `getattr(..., 'output', None)`
default. `iter_nodes`, `state.py:100-102`, `compiler.py:208-215`,
`verify.py`, `lint.py`, `testing.py` all route through `item_kind` / `child_items`.

### 9.2 The single guard that replaces the per-op guards

One structural-guard test: **`isinstance(_, Node | Construct | _BranchNode)` (and
`getattr(item, 'output'|'outputs'|'input'|'inputs'|'nodes', ...)`) is banned
outside `_ir_dispatch.py` and the IR types' own modules
(`node.py`/`construct.py`/`forward.py` self-reference is fine).** This is one
allowlist, enforced once, replacing the implicit "add a guard per helper" the
seed's Option A would require. It is also the mechanical enforcement of OCP §2:
new dispatch *cannot* be hand-rolled; it must land in `_ir_dispatch`.

### 9.3 Migration sketch (matches the seed's spike, but commits to the outcome)

1. Add `_ir_dispatch.py` with `declared_output`, `is_container`, `child_items`,
   `item_kind`. Move `_BranchNode`/`_BranchMeta` to `_ir_branch.py`; re-export
   from `forward.py` (fixes DIP §3).
2. Migrate the 3 proof sites: `forward.py:559-562` (the trigger), `compiler.py:208-215`,
   `state.py:100-102`. Measure isinstance-site delta.
3. Move `effective_producer_type` / `primary_output_field` into `_ir_dispatch`;
   re-point `_validation_types` and `_ir_normalize`.
4. Sweep remaining sites; add the §9.2 guard last (guard-first is impossible here
   since the guard *bans* the pattern the migration removes — write the guard, let
   it fail, migrate until green, lock it).
5. Retire the per-op helpers' internal isinstance (they delegate to dispatch).

Expected collapse: the §2 fallthroughs (`compiler.py:213`, `state.py:100-102`,
`construct.py:66-70`, `_validation_types.py:110`, all `verify.py`/`lint.py`/
`testing.py` walkers) plus the `forward.py` boundary — comfortably ≥40 of 52. The
residual legitimate isinstance (Pydantic boundary validators, `_isinstance_safe`
over Union, the dispatch module's own internal match) stays and is allowlisted.

---

## 10. Risks

- **Relocating `_BranchNode`** touches `forward.py`, `compiler.py`, `state.py`,
  `_wiring.py` imports. Mechanical but wide. Mitigate with re-export shim; or take
  §8(b) structural-branch fallback.
- **`match` exhaustiveness is not statically enforced in Python** the way a sealed
  type would be. Mitigate: the §9.2 guard + an explicit `case _: raise` arm that
  fails loudly on an unmodeled type (better than `compiler.py:214`'s silent
  `assert`).
- **Three-surface parity (AGENTS.md):** any change to producer-type computation
  must be tested via `@node`, declarative, and programmatic surfaces. The move of
  `effective_producer_type` is behavior-preserving but must be parity-tested.
- **Over-segmentation:** four capability protocols could be one too many. If a
  protocol has exactly one function and one caller, fold it into a plain function
  in `_ir_dispatch` — don't ship ceremony. ISP is satisfied by *callers depending
  on narrow surfaces*, which functions achieve as well as protocols.

---

## 11. Where I disagree with the seed doc

1. **"Adapter/visitor" conflation (seed §5B).** Visitor is constraint-incompatible
   (needs `accept` on the IR). Say "match-based sum-type dispatch," not visitor.
2. **"Widen `ConstructItem`" (seed §5B).** One fat protocol violates ISP and would
   re-couple `forward.py` to validation vocabulary — the very thing that blocked
   reuse. Segregate.
3. **"B as a spike measured against A" (seed §6).** The principles already decide
   it; A's OCP asymptote is structurally wrong. Run the spike to *size* the change,
   not to keep A on the table.
4. **The seed never names the decisive insight:** Information Expert → constraints
   override → Pure Fabrication is the *sanctioned* answer. Framing B as a "trade"
   (seed §4 "genuinely a trade, not an obvious win") undersells it. Given the two
   constraints, B is not a trade — it is the pattern the constraints select for.

---

*Companion to `ir-item-dispatch-2026-06-04.md`. Feeds `neograph-8cqd`.*
